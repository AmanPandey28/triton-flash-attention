# Implementation provenance

This repository is an independently maintained Triton implementation for GPU
kernel development and performance analysis. It is not the PyTorch or Triton
project and does not claim to be a drop-in replacement for their production
kernels.

## Upstream foundation

The forward algorithm and its terminology follow Triton's
[fused-attention tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py):
tiled Q/K/V traversal, online softmax, base-2 exponentiation, saved log-sum-exp,
and PyTorch autograd integration. That tutorial is the primary technical
reference and is acknowledged in `NOTICE.md`.

The following repositories were reviewed as technical comparison material:

- `hkproj/triton-flash-attention`
- `priyammaz/MyTorch`
- `evintunador/triton_docs_tutorials`

No dependency imports code from those repositories and no third-party source
tree is vendored here. Their license notices are retained in `NOTICE.md`.

## Project extensions

The repository's implementation and engineering work includes:

- a public dispatcher with explicit strict/automatic backend selection and
  PyTorch SDPA fallback;
- boundary-safe arbitrary positive sequence lengths, including partial query,
  key, value, output, log-sum-exp, and gradient tiles;
- grouped-query attention head mapping and atomic dK/dV accumulation;
- a custom backward path that recomputes probability tiles from FP32
  log-sum-exp and skips the masked causal tile region;
- forward/backward autotuning with invalid-configuration pruning and persistent
  tuning-result caching;
- deterministic output/dQ/dK/dV differential testing across tile boundaries,
  all supported head dimensions, three batch sizes, and multiple GQA ratios;
- reproducible benchmark artifacts that record quantiles, first-call cost,
  baseline backend, environment, clocks, power state, and exact command;
- Nsight Compute capture workloads and raw counter tables; and
- driver-independent AOT compilation checks for representative forward and
  backward specializations.

## Generated artifacts

Files under `benchmarks/results/` were generated locally on the GPU and
software stack recorded in each metadata file. They are measurements, not
upstream data or universal performance claims.
