# NVIDIA GeForce RTX 5050 Laptop GPU benchmark summary

The custom kernel was faster on **9/16** measured shapes. Speedup is PyTorch SDPA median divided by the custom-kernel median; values above 1.0 are faster.

Best result: **2.09x** at N=128, D=64, causal=True, Hq/Hkv=16/4.

| Mask | N | D | Hq/Hkv | PyTorch ms | Triton ms | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| noncausal | 128 | 64 | 16/4 | 0.05656 | 0.04813 | 1.18x |
| noncausal | 512 | 64 | 16/4 | 0.35434 | 0.35021 | 1.01x |
| noncausal | 1025 | 64 | 16/4 | 1.32048 | 1.27640 | 1.03x |
| noncausal | 2048 | 64 | 16/4 | 3.73166 | 4.02538 | 0.93x |
| noncausal | 128 | 128 | 16/4 | 0.14442 | 0.10346 | 1.40x |
| noncausal | 512 | 128 | 16/4 | 0.73203 | 0.76794 | 0.95x |
| noncausal | 1025 | 128 | 16/4 | 2.51286 | 2.81757 | 0.89x |
| noncausal | 2048 | 128 | 16/4 | 7.52749 | 9.25731 | 0.81x |
| causal | 128 | 64 | 16/4 | 0.09638 | 0.04602 | 2.09x |
| causal | 512 | 64 | 16/4 | 0.28413 | 0.23450 | 1.21x |
| causal | 1025 | 64 | 16/4 | 0.81526 | 0.78347 | 1.04x |
| causal | 2048 | 64 | 16/4 | 2.32806 | 2.34906 | 0.99x |
| causal | 128 | 128 | 16/4 | 0.14229 | 0.10038 | 1.42x |
| causal | 512 | 128 | 16/4 | 0.56728 | 0.53046 | 1.07x |
| causal | 1025 | 128 | 16/4 | 1.53722 | 1.71622 | 0.90x |
| causal | 2048 | 128 | 16/4 | 4.41562 | 5.27861 | 0.84x |

This is a shape sweep, not a claim of universal superiority. See the CSV for p10/p50/p90 latency, first-call cost, TFLOP/s, and memory model; see the metadata JSON for the exact environment and command.
