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

Inputs outside the CUDA kernel's support envelope—including non-CUDA tensors,
unsupported dtypes and head dimensions, cross-attention, and unequal value
dimensions—are handled by the PyTorch fallback in automatic mode.
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
python -m pytest tests/test_cuda_correctness.py
python -m benchmarks.correctness \
  --output benchmarks/results/rtx5050_correctness.json
python -m pytest tests/test_api.py tests/test_benchmark_model.py
python scripts/aot_compile_check.py --arch 80
python -m ruff check .
```

Validation is split by responsibility, following the same pattern used for
production accelerator libraries:

- CUDA differential tests compare the output, `dQ`, `dK`, and `dV` with
  PyTorch across causal and bidirectional modes, non-aligned sequence lengths,
  multiple head dimensions, and GQA.
- Hardware-independent contract tests cover dispatch, input validation,
  fallback semantics, autograd integration, and analytical performance models.
- Driver-independent AOT checks compile representative forward and backward
  kernels for a declared CUDA architecture.

The matrix contains 20 CUDA differential cases spanning batch sizes 1/2/4,
head dimensions 32/64/128, causal and bidirectional MHA, tile-boundary tails,
and 2:1/4:1/8:1 GQA. It records maximum absolute and relative error for the
output, `dQ`, `dK`, and `dV`; all 20 cases pass on an NVIDIA GeForce RTX 5050
Laptop GPU. The repository also has 13 API and analytical tests.

## Benchmarking

```bash
python -m benchmarks.benchmark_attention \
  --sequence-lengths 128,511,512,1025,2048,4096 \
  --head-dims 64,128 \
  --batch 1 \
  --query-heads 16 \
  --mode forward-backward \
  --mask both \
  --warmup-ms 500 \
  --rep-ms 1000 \
  --output benchmarks/results/rtx5050_mha.csv
```

The benchmark reports p10/p50/p90 latency, cold first-call cost, algorithmic
TFLOP/s, and speedup relative to PyTorch. It identifies the selected PyTorch
SDPA backend with a CUDA profiler trace and writes raw CSV, a concise Markdown
summary, and environment metadata including the seed, command, driver, clocks,
and power state.

### RTX 5050 Laptop GPU

Against the profiler-confirmed PyTorch `FLASH_ATTENTION` backend, the custom
forward+backward path is faster on 11/24 MHA shapes and 9/16 4:1 GQA shapes.
The best MHA result is **1.38x** at `B=1, H=16, N=512, D=64`, causal. The best
GQA result is **2.09x** at `B=1, Hq/Hkv=16/4, N=128, D=64`, causal. Long
sequences and several `D=128` shapes remain slower; the complete crossover is
reported instead of only the winning cases.

See [the performance study](docs/performance.md), the
[MHA sweep](benchmarks/results/rtx5050_mha.summary.md), and the
[GQA sweep](benchmarks/results/rtx5050_gqa.summary.md) for raw-data links,
methodology, Nsight Compute evidence, and optimization decisions.

## Profiling

`scripts/profile_attention.py` emits one warmed iteration inside a CUDA
profiler capture range. For example:

```bash
ncu --profile-from-start off \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
  python -m scripts.profile_attention \
  --sequence 512 --head-dim 64 --causal --forward-only
```

Recorded profiles show the `D=128` forward specialization using 255
registers/thread and 64–96 KiB shared memory, reducing achieved occupancy to
8.3%. This explains the measured long-sequence crossover and identifies
resource pressure—not DRAM bandwidth—as the next tuning target.

## Repository structure

```text
triton_flash_attention/             Python package and Triton kernels
benchmarks/benchmark_attention.py   Reproducible benchmark CLI
benchmarks/correctness.py           Deterministic CUDA correctness matrix
benchmarks/results/                 Recorded benchmark data
tests/                              API, model, and CUDA differential tests
scripts/aot_compile_check.py        Driver-independent CUDA compilation check
scripts/profile_attention.py        Nsight Compute capture workload
docs/architecture.md                Algorithm and implementation details
docs/performance.md                 Benchmark and profiler findings
```

## Acknowledgements

This implementation builds on Triton's official fused-attention tutorial and
the projects listed in [NOTICE.md](NOTICE.md). The boundary between upstream
ideas and repository-specific work is documented in
[PROVENANCE.md](PROVENANCE.md).
