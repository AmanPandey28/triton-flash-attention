"""AOT-compile representative kernels without requiring a visible GPU driver."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "TRITON_CACHE_DIR", str(Path("/tmp/triton-flash-attention-aot-cache"))
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource, make_backend  # noqa: E402

import triton_flash_attention.attention as attention  # noqa: E402


def _compile(
    *,
    function,
    signature: dict[str, str],
    constants: dict[str, int | bool],
    target: GPUTarget,
    warps: int = 4,
    stages: int = 3,
) -> None:
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": warps, "num_stages": stages})
    source = ASTSource(
        fn=function,
        signature=signature,
        constexprs=constants,
    )
    triton.compile(
        source,
        target=target,
        options=options.__dict__,
    )


def _forward_signature(function) -> dict[str, str]:
    signature = {name: "i32" for name in function.arg_names}
    for name in ("Q", "K", "V", "Out"):
        signature[name] = "*fp16"
    signature["LSE"] = "*fp32"
    signature["sm_scale"] = "fp32"
    return signature


def _backward_signature(function) -> dict[str, str]:
    signature = {name: "i32" for name in function.arg_names}
    for name in (
        "Q_ptr",
        "K_ptr",
        "V_ptr",
        "dO_ptr",
        "dQ_ptr",
        "dK_ptr",
        "dV_ptr",
    ):
        signature[name] = "*fp16"
    signature["LSE_ptr"] = "*fp32"
    signature["Delta_ptr"] = "*fp32"
    signature["sm_scale"] = "fp32"
    return signature


def _mark_constants(
    signature: dict[str, str], constants: dict[str, int | bool]
) -> None:
    for name in constants:
        signature[name] = "constexpr"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch",
        type=int,
        default=80,
        help="CUDA SM architecture without the decimal point (default: 80)",
    )
    args = parser.parse_args(argv)
    target = GPUTarget("cuda", args.arch, 32)

    forward = attention._attn_fwd.fn
    for causal in (False, True):
        constants = {
            "NUM_Q_HEADS": 4,
            "NUM_KV_HEADS": 2,
            "N_CTX": 129,
            "HEAD_DIM": 64,
            "BLOCK_M": 64,
            "BLOCK_N": 32,
            "CAUSAL": causal,
        }
        signature = _forward_signature(forward)
        _mark_constants(signature, constants)
        _compile(
            function=forward,
            signature=signature,
            constants=constants,
            target=target,
        )

    preprocess = attention._attn_bwd_preprocess
    preprocess_constants = {
        "N_CTX": 129,
        "HEAD_DIM": 64,
        "BLOCK_M": 128,
    }
    _compile(
        function=preprocess,
        signature={
            "Out": "*fp16",
            "dO": "*fp16",
            "Delta": "*fp32",
            "N_CTX": "constexpr",
            "HEAD_DIM": "constexpr",
            "BLOCK_M": "constexpr",
        },
        constants=preprocess_constants,
        target=target,
    )

    backward = attention._attn_bwd.fn
    for causal in (False, True):
        constants = {
            "NUM_Q_HEADS": 4,
            "NUM_KV_HEADS": 2,
            "N_CTX": 129,
            "HEAD_DIM": 64,
            "CAUSAL": causal,
            "BLOCK_MACRO": 64,
            "BLOCK_MICRO": 16,
        }
        signature = _backward_signature(backward)
        _mark_constants(signature, constants)
        _compile(
            function=backward,
            signature=signature,
            constants=constants,
            target=target,
        )

    print(
        f"AOT compile passed for sm_{args.arch}: forward, preprocess, "
        "backward, causal/noncausal, GQA"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
