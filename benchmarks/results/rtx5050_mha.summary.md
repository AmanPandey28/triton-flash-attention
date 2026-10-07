# NVIDIA GeForce RTX 5050 Laptop GPU benchmark summary

The custom kernel was faster on **11/24** measured shapes. Speedup is PyTorch SDPA median divided by the custom-kernel median; values above 1.0 are faster.

Best result: **1.38x** at N=512, D=64, causal=True, Hq/Hkv=16/16.

| Mask | N | D | Hq/Hkv | PyTorch ms | Triton ms | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| noncausal | 128 | 64 | 16/16 | 0.04506 | 0.04301 | 1.05x |
| noncausal | 511 | 64 | 16/16 | 0.40502 | 0.39018 | 1.04x |
| noncausal | 512 | 64 | 16/16 | 0.37378 | 0.35734 | 1.05x |
| noncausal | 1025 | 64 | 16/16 | 1.25365 | 1.30157 | 0.96x |
| noncausal | 2048 | 64 | 16/16 | 3.68026 | 4.11338 | 0.90x |
| noncausal | 4096 | 64 | 16/16 | 14.19322 | 16.59656 | 0.85x |
| noncausal | 128 | 128 | 16/16 | 0.11878 | 0.09933 | 1.20x |
| noncausal | 511 | 128 | 16/16 | 0.70150 | 0.77720 | 0.90x |
| noncausal | 512 | 128 | 16/16 | 0.69014 | 0.74448 | 0.93x |
| noncausal | 1025 | 128 | 16/16 | 2.47304 | 2.84176 | 0.87x |
| noncausal | 2048 | 128 | 16/16 | 7.71466 | 10.15421 | 0.76x |
| noncausal | 4096 | 128 | 16/16 | 28.85078 | 37.42466 | 0.77x |
| causal | 128 | 64 | 16/16 | 0.05834 | 0.04608 | 1.27x |
| causal | 511 | 64 | 16/16 | 0.30515 | 0.22835 | 1.34x |
| causal | 512 | 64 | 16/16 | 0.31752 | 0.23035 | 1.38x |
| causal | 1025 | 64 | 16/16 | 0.82184 | 0.76379 | 1.08x |
| causal | 2048 | 64 | 16/16 | 2.30502 | 2.39715 | 0.96x |
| causal | 4096 | 64 | 16/16 | 7.85306 | 9.17363 | 0.86x |
| causal | 128 | 128 | 16/16 | 0.12077 | 0.08810 | 1.37x |
| causal | 511 | 128 | 16/16 | 0.57958 | 0.50483 | 1.15x |
| causal | 512 | 128 | 16/16 | 0.57251 | 0.49786 | 1.15x |
| causal | 1025 | 128 | 16/16 | 1.66387 | 1.78278 | 0.93x |
| causal | 2048 | 128 | 16/16 | 4.36826 | 5.60838 | 0.78x |
| causal | 4096 | 128 | 16/16 | 15.97683 | 20.55117 | 0.78x |

This is a shape sweep, not a claim of universal superiority. See the CSV for p10/p50/p90 latency, first-call cost, TFLOP/s, and memory model; see the metadata JSON for the exact environment and command.
