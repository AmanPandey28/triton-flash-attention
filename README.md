# Triton FlashAttention-2

A custom, fused implementation of FlashAttention-2 written in OpenAI Triton.

The kernel implements causal masking plus end-to-end forward and backward passes. It avoids materializing the O(N^2) attention matrix in high-bandwidth memory by tiling QK^T and V into SRAM and applying a numerically stable online softmax. The backward pass recomputes attention weights with the log-sum-exp values instead of materializing sparse Jacobians.

This implementation closely follows Umar Jamil's Triton FlashAttention walkthrough: https://www.youtube.com/watch?v=zy8ChVd_oTM

## Performance Profiling

Hardware: Lenovo LOQ with NVIDIA RTX 5050 Mobile  
Context window: 4096 tokens  
Precision: FP16

The benchmark compares this Triton kernel against PyTorch's native `scaled_dot_product_attention` backend using the same tensor shapes and backward pass. With autotuned block sizing, pipeline stages, and warp counts, the custom kernel sustains **16.2 TFLOP/s**, which is roughly 40% of the mobile GPU's theoretical FP16 Tensor Core peak throughput.

## Reproducing the Benchmark

```bash
cd triton
pip install -r requirements.txt
python benchmark_fa2.py
```

## Project Layout

```text
triton/flash_attention.py   # Triton FlashAttention-2 forward and backward kernels
triton/benchmark_fa2.py     # RTX 5050 benchmark against PyTorch SDPA
triton/requirements.txt     # Python dependencies
```
