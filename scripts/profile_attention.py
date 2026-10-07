"""Emit one warmed attention step inside a CUDA profiler capture range."""

from __future__ import annotations

import argparse

import torch

from triton_flash_attention import scaled_dot_product_attention


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("triton", "torch"), default="triton")
    parser.add_argument("--batch", type=_positive_int, default=1)
    parser.add_argument("--query-heads", type=_positive_int, default=16)
    parser.add_argument("--kv-heads", type=_positive_int, default=None)
    parser.add_argument("--sequence", type=_positive_int, default=512)
    parser.add_argument("--head-dim", type=_positive_int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--seed", type=int, default=1729)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to profile attention")
    kv_heads = args.kv_heads or args.query_heads
    if args.query_heads % kv_heads:
        raise SystemExit("--query-heads must be divisible by --kv-heads")

    torch.manual_seed(args.seed)
    requires_grad = not args.forward_only
    query = torch.randn(
        args.batch,
        args.query_heads,
        args.sequence,
        args.head_dim,
        device="cuda",
        dtype=torch.float16,
        requires_grad=requires_grad,
    )
    key = torch.randn(
        args.batch,
        kv_heads,
        args.sequence,
        args.head_dim,
        device="cuda",
        dtype=torch.float16,
        requires_grad=requires_grad,
    )
    value = torch.randn_like(key, requires_grad=requires_grad)
    grad_output = torch.randn_like(query)

    def step() -> None:
        query.grad = key.grad = value.grad = None
        output = scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=args.causal,
            enable_gqa=args.query_heads != kv_heads,
            backend=args.provider,
        )
        if not args.forward_only:
            output.backward(grad_output)

    for _ in range(2):
        step()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    step()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
