"""Benchmark the custom kernel against PyTorch SDPA over a shape sweep."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import triton
from triton_flash_attention import scaled_dot_product_attention


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _int_list(value: str) -> list[int]:
    try:
        values = [_positive_int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return values


def attention_flops(
    *,
    batch: int,
    query_heads: int,
    sequence: int,
    head_dim: int,
    causal: bool,
    mode: str,
) -> int:
    """Count algorithmic FLOPs using the convention from Triton's tutorial."""
    pairs = sequence * (sequence + 1) // 2 if causal else sequence * sequence
    forward = 4 * batch * query_heads * pairs * head_dim
    if mode == "forward":
        return forward
    if mode == "backward":
        return int(2.5 * forward)
    if mode == "forward-backward":
        return int(3.5 * forward)
    raise ValueError(f"unknown mode: {mode}")


def memory_model(
    *,
    batch: int,
    query_heads: int,
    sequence: int,
    element_size: int = 2,
) -> dict[str, float]:
    """Compare one materialized score matrix with the saved LSE vector."""
    mib = 1024**2
    return {
        "naive_score_mib": (
            batch * query_heads * sequence * sequence * element_size / mib
        ),
        "saved_lse_mib": batch * query_heads * sequence * 4 / mib,
    }


def _make_runner(
    *,
    provider: str,
    mode: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output: torch.Tensor,
    causal: bool,
    enable_gqa: bool,
):
    backend = "triton" if provider == "triton" else "torch"

    def attention():
        return scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=causal,
            enable_gqa=enable_gqa,
            backend=backend,
        )

    if mode == "forward":
        return attention

    if mode == "backward":
        output = attention()

        def backward():
            query.grad = key.grad = value.grad = None
            output.backward(grad_output, retain_graph=True)

        return backward

    def forward_backward():
        query.grad = key.grad = value.grad = None
        attention().backward(grad_output)

    return forward_backward


def _benchmark_case(
    *,
    provider: str,
    mode: str,
    batch: int,
    query_heads: int,
    kv_heads: int,
    sequence: int,
    head_dim: int,
    causal: bool,
    warmup_ms: int,
    rep_ms: int,
) -> dict[str, Any]:
    requires_grad = mode != "forward"
    shape_q = (batch, query_heads, sequence, head_dim)
    shape_kv = (batch, kv_heads, sequence, head_dim)
    query = torch.randn(
        shape_q, device="cuda", dtype=torch.float16, requires_grad=requires_grad
    )
    key = torch.randn(
        shape_kv, device="cuda", dtype=torch.float16, requires_grad=requires_grad
    )
    value = torch.randn(
        shape_kv, device="cuda", dtype=torch.float16, requires_grad=requires_grad
    )
    grad_output = torch.randn_like(query)
    runner = _make_runner(
        provider=provider,
        mode=mode,
        query=query,
        key=key,
        value=value,
        grad_output=grad_output,
        causal=causal,
        enable_gqa=query_heads != kv_heads,
    )

    median, low, high = triton.testing.do_bench(
        runner,
        warmup=warmup_ms,
        rep=rep_ms,
        quantiles=[0.5, 0.2, 0.8],
    )
    flops = attention_flops(
        batch=batch,
        query_heads=query_heads,
        sequence=sequence,
        head_dim=head_dim,
        causal=causal,
        mode=mode,
    )
    memory = memory_model(
        batch=batch,
        query_heads=query_heads,
        sequence=sequence,
    )
    return {
        "provider": provider,
        "mode": mode,
        "causal": causal,
        "batch": batch,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "sequence": sequence,
        "head_dim": head_dim,
        "median_ms": round(float(median), 5),
        "p20_ms": round(float(low), 5),
        "p80_ms": round(float(high), 5),
        "tflops": round(flops / (float(median) * 1e-3) / 1e12, 3),
        **{name: round(value, 3) for name, value in memory.items()},
    }


def _metadata() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": ".".join(
            str(value) for value in torch.cuda.get_device_capability()
        ),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "cuda_runtime": torch.version.cuda,
    }


def _write_results(
    rows: list[dict[str, Any]], metadata: dict[str, Any], output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def _print_rows(rows: list[dict[str, Any]]) -> None:
    columns = (
        "provider",
        "mode",
        "causal",
        "sequence",
        "head_dim",
        "median_ms",
        "tflops",
    )
    print("  ".join(f"{column:>12}" for column in columns))
    for row in rows:
        print("  ".join(f"{str(row[column]):>12}" for column in columns))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-lengths",
        type=_int_list,
        default=[512, 1024, 2048, 4096],
        help="comma-separated sequence lengths",
    )
    parser.add_argument(
        "--head-dims",
        type=_int_list,
        default=[64, 128],
        help="comma-separated head dimensions",
    )
    parser.add_argument("--batch", type=_positive_int, default=4)
    parser.add_argument("--query-heads", type=_positive_int, default=16)
    parser.add_argument(
        "--kv-heads",
        type=_positive_int,
        default=None,
        help="set below query-heads to benchmark GQA",
    )
    parser.add_argument(
        "--mode",
        choices=("forward", "backward", "forward-backward"),
        default="forward-backward",
    )
    parser.add_argument(
        "--mask",
        choices=("causal", "noncausal", "both"),
        default="causal",
    )
    parser.add_argument("--warmup-ms", type=_positive_int, default=100)
    parser.add_argument("--rep-ms", type=_positive_int, default=300)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/latest.csv"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run the benchmark")
    kv_heads = args.kv_heads or args.query_heads
    if args.query_heads % kv_heads:
        raise SystemExit("--query-heads must be divisible by --kv-heads")
    unsupported = set(args.head_dims) - {32, 64, 128}
    if unsupported:
        raise SystemExit("custom Triton path supports head dimensions 32, 64, and 128")

    causal_values = {
        "causal": [True],
        "noncausal": [False],
        "both": [False, True],
    }[args.mask]
    rows = []
    for causal in causal_values:
        for head_dim in args.head_dims:
            for sequence in args.sequence_lengths:
                for provider in ("torch", "triton"):
                    print(
                        f"benchmarking {provider}: N={sequence}, D={head_dim}, "
                        f"causal={causal}"
                    )
                    rows.append(
                        _benchmark_case(
                            provider=provider,
                            mode=args.mode,
                            batch=args.batch,
                            query_heads=args.query_heads,
                            kv_heads=kv_heads,
                            sequence=sequence,
                            head_dim=head_dim,
                            causal=causal,
                            warmup_ms=args.warmup_ms,
                            rep_ms=args.rep_ms,
                        )
                    )

    metadata = _metadata()
    _write_results(rows, metadata, args.output)
    print(json.dumps(metadata, indent=2))
    _print_rows(rows)
    print(f"wrote {args.output} and {args.output.with_suffix('.metadata.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
