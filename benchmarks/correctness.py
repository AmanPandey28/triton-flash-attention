"""Run a deterministic CUDA differential-correctness matrix."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import triton

from triton_flash_attention import scaled_dot_product_attention

FORWARD_ATOL = 2e-2
BACKWARD_ATOL = 3e-2


@dataclass(frozen=True)
class AttentionCase:
    batch: int
    query_heads: int
    kv_heads: int
    sequence: int
    head_dim: int
    causal: bool

    @property
    def name(self) -> str:
        mask = "causal" if self.causal else "noncausal"
        return (
            f"b{self.batch}-hq{self.query_heads}-hkv{self.kv_heads}-"
            f"n{self.sequence}-d{self.head_dim}-{mask}"
        )


# Edges around the 16/32/64/128 tile boundaries plus MHA and 2:1/4:1/8:1 GQA.
CORRECTNESS_CASES = (
    AttentionCase(1, 2, 2, 1, 32, False),
    AttentionCase(2, 2, 2, 2, 64, True),
    AttentionCase(4, 2, 2, 15, 128, False),
    AttentionCase(1, 2, 2, 16, 32, True),
    AttentionCase(1, 2, 2, 17, 64, False),
    AttentionCase(1, 2, 2, 31, 128, True),
    AttentionCase(1, 2, 2, 32, 32, False),
    AttentionCase(1, 2, 2, 33, 64, True),
    AttentionCase(1, 2, 2, 63, 128, False),
    AttentionCase(1, 2, 2, 64, 32, True),
    AttentionCase(1, 2, 2, 65, 64, False),
    AttentionCase(1, 2, 2, 127, 128, True),
    AttentionCase(1, 2, 2, 128, 32, False),
    AttentionCase(1, 2, 2, 129, 64, True),
    AttentionCase(1, 2, 2, 511, 128, False),
    AttentionCase(1, 2, 2, 512, 64, True),
    AttentionCase(1, 2, 2, 1025, 64, True),
    AttentionCase(1, 4, 2, 65, 64, True),
    AttentionCase(1, 8, 2, 129, 32, False),
    AttentionCase(2, 8, 1, 33, 128, True),
)


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual - expected).abs().float()
    relative = difference / expected.abs().float().clamp_min(1e-3)
    return {
        "max_abs_error": float(difference.max().item()),
        "max_rel_error_floor_1e-3": float(relative.max().item()),
    }


def run_case(case: AttentionCase, seed: int) -> dict[str, Any]:
    """Compare output and all input gradients for one shape."""
    torch.manual_seed(seed)
    shape_q = (case.batch, case.query_heads, case.sequence, case.head_dim)
    shape_kv = (case.batch, case.kv_heads, case.sequence, case.head_dim)
    query = (
        torch.randn(shape_q, device="cuda", dtype=torch.float16) * 0.5
    ).requires_grad_()
    key = (
        torch.randn(shape_kv, device="cuda", dtype=torch.float16) * 0.5
    ).requires_grad_()
    value = (
        torch.randn(shape_kv, device="cuda", dtype=torch.float16) * 0.5
    ).requires_grad_()
    grad_output = torch.randn_like(query) * 0.1
    enable_gqa = case.query_heads != case.kv_heads

    start = time.perf_counter()
    actual = scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=case.causal,
        enable_gqa=enable_gqa,
        backend="triton",
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=case.causal,
        enable_gqa=enable_gqa,
    )
    actual_grads = torch.autograd.grad(
        actual, (query, key, value), grad_output, retain_graph=True
    )
    expected_grads = torch.autograd.grad(
        expected, (query, key, value), grad_output
    )
    torch.cuda.synchronize()

    tensors = {
        "output": (actual, expected, FORWARD_ATOL),
        "dQ": (actual_grads[0], expected_grads[0], BACKWARD_ATOL),
        "dK": (actual_grads[1], expected_grads[1], BACKWARD_ATOL),
        "dV": (actual_grads[2], expected_grads[2], BACKWARD_ATOL),
    }
    checks = {}
    for name, (actual_tensor, expected_tensor, tolerance) in tensors.items():
        checks[name] = _errors(actual_tensor, expected_tensor)
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            atol=tolerance,
            rtol=0,
        )

    return {
        **asdict(case),
        "name": case.name,
        "seed": seed,
        "status": "pass",
        "elapsed_ms_including_first_call": round(
            (time.perf_counter() - start) * 1e3, 3
        ),
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/correctness.json"),
    )
    parser.add_argument("--seed", type=int, default=1729)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run the correctness matrix")

    rows = []
    for index, case in enumerate(CORRECTNESS_CASES):
        print(f"checking {case.name}")
        rows.append(run_case(case, args.seed + index))

    artifact = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": ".".join(
            str(value) for value in torch.cuda.get_device_capability()
        ),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "cuda_runtime": torch.version.cuda,
        "dtype": "float16",
        "forward_atol": FORWARD_ATOL,
        "backward_atol": BACKWARD_ATOL,
        "case_count": len(rows),
        "passed": len(rows),
        "failed": 0,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)}/{len(rows)} cases passed; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
