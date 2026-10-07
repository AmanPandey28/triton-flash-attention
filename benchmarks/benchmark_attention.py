"""Benchmark the custom kernel against PyTorch SDPA over a shape sweep."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import subprocess
import sys
import time
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
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
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
    torch.cuda.synchronize()
    setup_start = time.perf_counter()
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
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - setup_start) * 1e3

    torch.cuda.synchronize()
    first_call_start = time.perf_counter()
    runner()
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - first_call_start) * 1e3

    median, low, high = triton.testing.do_bench(
        runner,
        warmup=warmup_ms,
        rep=rep_ms,
        quantiles=[0.5, 0.1, 0.9],
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
        "setup_ms": round(setup_ms, 3),
        "first_call_ms": round(first_call_ms, 3),
        "median_ms": round(float(median), 5),
        "p10_ms": round(float(low), 5),
        "p90_ms": round(float(high), 5),
        "tflops": round(flops / (float(median) * 1e-3) / 1e12, 3),
        **{name: round(value, 3) for name, value in memory.items()},
    }


def _gpu_runtime_state() -> dict[str, str]:
    fields = (
        "driver_version",
        "pstate",
        "power_draw_w",
        "power_limit_w",
        "sm_clock_mhz",
        "memory_clock_mhz",
    )
    command = [
        "nvidia-smi",
        "--query-gpu=driver_version,pstate,power.draw,power.limit,"
        "clocks.current.sm,clocks.current.memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.splitlines()[0]
    except (FileNotFoundError, IndexError, subprocess.SubprocessError):
        return {}
    return dict(zip(fields, (value.strip() for value in output.split(","))))


def _identify_torch_sdpa_backend(
    *,
    mode: str,
    batch: int,
    query_heads: int,
    kv_heads: int,
    sequence: int,
    head_dim: int,
    causal: bool,
) -> str:
    """Identify the selected SDPA implementation from one CUDA profile."""
    from torch.profiler import ProfilerActivity, profile

    requires_grad = mode != "forward"
    query = torch.randn(
        batch,
        query_heads,
        sequence,
        head_dim,
        device="cuda",
        dtype=torch.float16,
        requires_grad=requires_grad,
    )
    key = torch.randn(
        batch,
        kv_heads,
        sequence,
        head_dim,
        device="cuda",
        dtype=torch.float16,
        requires_grad=requires_grad,
    )
    value = torch.randn_like(key, requires_grad=requires_grad)
    grad_output = torch.randn_like(query)
    runner = _make_runner(
        provider="torch",
        mode=mode,
        query=query,
        key=key,
        value=value,
        grad_output=grad_output,
        causal=causal,
        enable_gqa=query_heads != kv_heads,
    )
    runner()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as torch_profile:
        runner()
    torch.cuda.synchronize()
    event_names = "\n".join(event.key.lower() for event in torch_profile.key_averages())
    if "pytorch_flash" in event_names or "flash_attention" in event_names:
        return "FLASH_ATTENTION"
    if "cudnn" in event_names and "attention" in event_names:
        return "CUDNN_ATTENTION"
    if "efficient_attention" in event_names or "memory_efficient" in event_names:
        return "EFFICIENT_ATTENTION"
    if "scaled_dot_product" in event_names:
        return "MATH_OR_UNCLASSIFIED"
    return "UNCLASSIFIED"


def _metadata(args: argparse.Namespace, torch_sdpa_backend: str) -> dict[str, Any]:
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": ".".join(
            str(value) for value in torch.cuda.get_device_capability()
        ),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "cuda_runtime": torch.version.cuda,
        "torch_sdpa_backend": torch_sdpa_backend,
        "dtype": "float16",
        "seed": args.seed,
        "warmup_ms": args.warmup_ms,
        "measurement_ms": args.rep_ms,
        "command": shlex.join(
            ["python", "-m", "benchmarks.benchmark_attention", *sys.argv[1:]]
        ),
    }
    metadata.update(_gpu_runtime_state())
    return metadata


def _add_speedups(rows: list[dict[str, Any]]) -> None:
    baseline_by_shape = {
        (
            row["mode"],
            row["causal"],
            row["batch"],
            row["query_heads"],
            row["kv_heads"],
            row["sequence"],
            row["head_dim"],
        ): row["median_ms"]
        for row in rows
        if row["provider"] == "torch"
    }
    for row in rows:
        key = (
            row["mode"],
            row["causal"],
            row["batch"],
            row["query_heads"],
            row["kv_heads"],
            row["sequence"],
            row["head_dim"],
        )
        baseline = baseline_by_shape[key]
        row["speedup_vs_torch"] = round(baseline / row["median_ms"], 3)
        row["latency_reduction_percent"] = round(
            100 * (baseline - row["median_ms"]) / baseline,
            1,
        )


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
    triton_rows = [row for row in rows if row["provider"] == "triton"]
    wins = [row for row in triton_rows if row["speedup_vs_torch"] > 1]
    best = max(triton_rows, key=lambda row: row["speedup_vs_torch"])
    summary = [
        f"# {metadata['gpu']} benchmark summary",
        "",
        (
            f"The custom kernel was faster on **{len(wins)}/{len(triton_rows)}** "
            "measured shapes. Speedup is PyTorch SDPA median divided by the "
            "custom-kernel median; values above 1.0 are faster."
        ),
        "",
        (
            f"Best result: **{best['speedup_vs_torch']:.2f}x** at "
            f"N={best['sequence']}, D={best['head_dim']}, "
            f"causal={best['causal']}, Hq/Hkv="
            f"{best['query_heads']}/{best['kv_heads']}."
        ),
        "",
        "| Mask | N | D | Hq/Hkv | PyTorch ms | Triton ms | Speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    baselines = {
        (
            row["causal"],
            row["sequence"],
            row["head_dim"],
        ): row["median_ms"]
        for row in rows
        if row["provider"] == "torch"
    }
    for row in triton_rows:
        baseline = baselines[(row["causal"], row["sequence"], row["head_dim"])]
        summary.append(
            f"| {'causal' if row['causal'] else 'noncausal'} | "
            f"{row['sequence']} | {row['head_dim']} | "
            f"{row['query_heads']}/{row['kv_heads']} | {baseline:.5f} | "
            f"{row['median_ms']:.5f} | {row['speedup_vs_torch']:.2f}x |"
        )
    summary.extend(
        [
            "",
            "This is a shape sweep, not a claim of universal superiority. See the "
            "CSV for p10/p50/p90 latency, first-call cost, TFLOP/s, and memory model; "
            "see the metadata JSON for the exact environment and command.",
            "",
        ]
    )
    output.with_suffix(".summary.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )


def _print_rows(rows: list[dict[str, Any]]) -> None:
    columns = (
        "provider",
        "mode",
        "causal",
        "sequence",
        "head_dim",
        "first_call_ms",
        "median_ms",
        "speedup_vs_torch",
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
    parser.add_argument("--seed", type=int, default=1729)
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
    torch_sdpa_backend = _identify_torch_sdpa_backend(
        mode=args.mode,
        batch=args.batch,
        query_heads=args.query_heads,
        kv_heads=kv_heads,
        sequence=args.sequence_lengths[0],
        head_dim=args.head_dims[0],
        causal=causal_values[0],
    )
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
                            seed=args.seed,
                        )
                    )

    _add_speedups(rows)
    metadata = _metadata(args, torch_sdpa_backend)
    _write_results(rows, metadata, args.output)
    print(json.dumps(metadata, indent=2))
    _print_rows(rows)
    print(
        f"wrote {args.output}, {args.output.with_suffix('.metadata.json')}, "
        f"and {args.output.with_suffix('.summary.md')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
