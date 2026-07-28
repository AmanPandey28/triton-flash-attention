# Triton FlashAttention-2

[![CI](https://github.com/AmanPandey28/triton-flash-attention/actions/workflows/ci.yml/badge.svg)](https://github.com/AmanPandey28/triton-flash-attention/actions/workflows/ci.yml)

A fused FlashAttention-2 implementation in Triton with custom forward and
backward kernels, causal masking, grouped-query attention, and automatic
PyTorch fallback.

## Features

- Tiled online softmax without materializing the quadratic attention matrix
- Custom autograd-compatible forward and backward kernels
- Causal backward traversal that skips the masked upper triangle
- Forward and backward autotuning
- Arbitrary positive sequence lengths with boundary-safe loads and stores
- Multi-head attention and grouped-query attention (GQA)
- Explicit backend selection with safe PyTorch SDPA fallback
- Differential correctness tests for outputs and all input gradients
- Reproducible benchmark output with environment metadata

## Installation

Requirements:

- Python 3.10 or newer
- PyTorch 2.4 or newer
- Triton 3.0 or newer
- NVIDIA Ampere-or-newer GPU for the custom kernel

```bash
git clone https://github.com/AmanPandey28/triton-flash-attention.git
cd triton-flash-attention
python -m pip install -e .
```

Install development dependencies with:

```bash
python -m pip install -e ".[dev]"
```

## Usage

```python
import torch

from triton_flash_attention import scaled_dot_product_attention

q = torch.randn(2, 16, 2049, 64, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

output = scaled_dot_product_attention(q, k, v, is_causal=True)
```

GQA uses fewer key/value heads than query heads:

```python
q = torch.randn(2, 16, 2049, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 4, 2049, 64, device="cuda", dtype=torch.float16)
v = torch.randn_like(k)

output = scaled_dot_product_attention(
    q,
    k,
    v,
    is_causal=True,
    enable_gqa=True,
)
```

The default `backend="auto"` selects the custom kernel for supported inputs and
uses PyTorch SDPA otherwise. Set `backend="triton"` to require the custom
implementation. `explain_dispatch(q, k, v)` reports the selected backend and
reason.

## Custom kernel support

| Capability | Supported |
| --- | --- |
| Device | NVIDIA CUDA, Ampere or newer |
| Dtype | FP16 |
| Head dimension | 32, 64, or 128 |
| Attention | Self-attention |
| Masking | Causal or bidirectional |
| Sequence length | Any positive length |
| Heads | MHA and GQA |
| Autograd | Forward and backward |

CPU tensors, unsupported dtypes and head dimensions, cross-attention, and
unequal value dimensions are handled by the PyTorch fallback in automatic mode.
Dropout and arbitrary attention masks are not currently implemented.

## Algorithm

Each Triton program keeps a query tile resident while streaming key and value
tiles through on-chip memory. A running maximum and normalization sum implement
a numerically stable online softmax. The kernel stores one FP32 log-sum-exp
value per query row for backward, rather than the full attention matrix.

For `B=4`, `H=16`, and `N=4096`, one materialized FP16 score matrix requires
2,048 MiB. The saved FP32 log-sum-exp tensor requires 1 MiB.

Backward reconstructs probability tiles from the saved log-sum-exp values.
Causal execution processes diagonal tiles with an element mask and visits only
the valid lower-triangular off-diagonal tiles. See
[docs/architecture.md](docs/architecture.md) for the derivation and kernel
layout.

## Correctness

```bash
pytest
ruff check .
python scripts/aot_compile_check.py --arch 80
```

The test suite contains 13 API and performance-model tests plus 9 CUDA
differential tests. All 22 tests pass on an NVIDIA GeForce RTX 5050 Laptop GPU.
The CUDA cases compare the output, `dQ`, `dK`, and `dV` with PyTorch across
causal and bidirectional modes, non-aligned sequence lengths, multiple head
dimensions, and GQA.

## Benchmarking

```bash
python -m benchmarks.benchmark_attention \
  --sequence-lengths 512,1024,2048,4096 \
  --head-dims 64,128 \
  --batch 4 \
  --query-heads 16 \
  --mode forward-backward \
  --mask causal \
  --output benchmarks/results/rtx5050.csv
```

The benchmark reports median, p20, and p80 latency plus algorithmic TFLOP/s. It
writes a CSV and a metadata JSON containing the GPU, compute capability, CUDA,
PyTorch, Triton, and Python versions.

### RTX 5050 Laptop GPU

`B=4`, `H=16`, `N=4096`, `D=64`, causal forward and backward, 1-second warmup,
3-second measurement window:

| Provider | Median | p20–p80 | TFLOP/s |
| --- | ---: | ---: | ---: |
| PyTorch SDPA | 30.50 ms | 28.54–33.86 ms | 15.776 |
| Triton FlashAttention-2 | 36.22 ms | 34.75–37.87 ms | 13.283 |

Raw measurements and environment details are available in
[benchmarks/results/rtx5050.csv](benchmarks/results/rtx5050.csv) and
[benchmarks/results/rtx5050.metadata.json](benchmarks/results/rtx5050.metadata.json).

## Repository structure

```text
triton_flash_attention/             Python package and Triton kernels
benchmarks/benchmark_attention.py   Reproducible benchmark CLI
benchmarks/results/                 Recorded benchmark data
tests/                              API, model, and CUDA differential tests
scripts/aot_compile_check.py        Driver-independent CUDA compilation check
docs/architecture.md                Algorithm and implementation details
```

## Acknowledgements

This implementation builds on Triton's official fused-attention tutorial and
the projects listed in [NOTICE.md](NOTICE.md).
