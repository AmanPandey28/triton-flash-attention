# Performance study

This document records the measurement method, results, profiler evidence, and
optimization decisions for the custom kernel. It intentionally reports both
winning and losing shapes.

## Environment and method

Measurements were collected on an NVIDIA GeForce RTX 5050 Laptop GPU with
PyTorch 2.12.0+cu130, Triton 3.7.0, CUDA 13.0, and driver 595.91.07. The exact
command, timestamp, clocks, power state, seed, and package versions are stored
next to each CSV.

The benchmark uses FP16 tensors, one deterministic seed, a 500 ms warmup, and a
1,000 ms measurement window. It reports p10, median, and p90 latency. A CUDA
profiler trace verifies that the PyTorch comparison dispatches to its
`FLASH_ATTENTION` backend. The timed workload is forward plus backward; cold
first-call cost is recorded separately from steady-state latency.

```bash
python -m benchmarks.benchmark_attention \
  --sequence-lengths 128,511,512,1025,2048,4096 \
  --head-dims 64,128 --batch 1 --query-heads 16 \
  --mode forward-backward --mask both --warmup-ms 500 --rep-ms 1000 \
  --output benchmarks/results/rtx5050_mha.csv
```

## End-to-end results

The MHA sweep covers 24 shapes. The custom implementation is faster on 11 and
slower on 13. Its best result is **1.38x** PyTorch FlashAttention at
`B=1, H=16, N=512, D=64`, causal, with median latency reduced from 0.318 ms to
0.230 ms. It wins most consistently for causal sequences through 512 tokens,
then crosses below PyTorch at longer sequences.

The 4:1 GQA sweep covers 16 shapes. The custom implementation is faster on 9
and slower on 7. Its best result is **2.09x** at
`B=1, Hq/Hkv=16/4, N=128, D=64`, causal, with median latency reduced from
0.096 ms to 0.046 ms. At `N=512, D=64`, it remains 1.21x faster. The advantage
again disappears for several long-sequence and `D=128` shapes.

Complete tables are in:

- [MHA summary](../benchmarks/results/rtx5050_mha.summary.md)
- [GQA summary](../benchmarks/results/rtx5050_gqa.summary.md)
- [MHA raw data](../benchmarks/results/rtx5050_mha.csv)
- [GQA raw data](../benchmarks/results/rtx5050_gqa.csv)

## Nsight Compute findings

The forward kernel was profiled on a fast shape, the same sequence with twice
the head dimension, and a long shape where the full operation loses to
PyTorch. Durations below are profiler-replay durations and should not be mixed
with the benchmark latency.

| Causal shape | Duration | SM throughput | DRAM throughput | Tensor pipe | Registers/thread | Shared memory | Occupancy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N=512, D=64 | 98.4 us | 62.0% | 13.2% | 37.9% | 147 | 40 KiB | 15.4% |
| N=512, D=128 | 236.4 us | 55.7% | 11.5% | 34.9% | 255 | 64 KiB | 8.3% |
| N=4096, D=128 | 8.77 ms | 77.7% | 2.8% | 40.2% | 255 | 96 KiB | 8.3% |

The `D=128` specialization reaches the architectural register limit and uses
up to 96 KiB of dynamic shared memory, allowing only one block and four active
warps per SM. Math-pipe throttling is the largest measured warp-stall category:
42.7% at `N=512, D=64`, 43.0% at `N=512, D=128`, and 50.9% at
`N=4096, D=128`. Fixed-latency dependency stalls rise from 17.9% to 27.1%.
The low DRAM utilization and high compute pressure show that the long `D=128`
case is resource/compute bound, not bandwidth bound.

The normalized Nsight Compute counter table is stored in
[`benchmarks/results/rtx5050_ncu.csv`](../benchmarks/results/rtx5050_ncu.csv),
with capture metadata in the adjacent JSON. It contains the high-signal launch,
occupancy, memory, compute, tensor-pipe, and warp-stall counters without
machine-local process data.

## Optimization decisions

- Forward autotuning now searches both 32- and 64-column K/V tiles, prunes
  invalid tile/head-dimension combinations, and persists tuning results.
- Backward autotuning also persists results. Cold specialization remains
  visible in the CSV rather than being hidden inside steady-state latency.
- A prototype split dK/dV and dQ into separate kernels to reduce per-kernel
  live state. It passed gradient checks but regressed the primary
  `N=512, D=64` causal workload from 0.230 ms to 0.250 ms, so the combined
  kernel was retained.
- The next high-value optimization target is `D=128`: reduce live accumulator
  state or specialize the backward tiles before expanding the supported API.

These results are specific to the recorded GPU and software stack. They are a
performance characterization, not a universal speedup claim.
