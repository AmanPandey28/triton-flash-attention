import torch
import triton
from flash_attention import TritonAttention

# Hardware and sequence configurations tailored for RTX 5050 VRAM
BATCH_SIZE = 4
NUM_HEADS = 16
SEQ_LEN = 4096
HEAD_DIM = 64
CAUSAL = True

# Initialize tensors
q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device="cuda", dtype=torch.float16, requires_grad=True)
v = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device="cuda", dtype=torch.float16, requires_grad=True)
dout = torch.randn_like(q)
sm_scale = 1.0 / (HEAD_DIM ** 0.5)

def run_sdpa():
    # Clear gradients to prevent accumulation OOMs during benchmarking
    q.grad, k.grad, v.grad = None, None, None
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=CAUSAL)
    out.backward(dout, retain_graph=True)
    return out

def run_triton():
    q.grad, k.grad, v.grad = None, None, None
    out = TritonAttention.apply(q, k, v, CAUSAL, sm_scale)
    out.backward(dout, retain_graph=True)
    return out

print(f"Warming up GPU for B={BATCH_SIZE}, H={NUM_HEADS}, S={SEQ_LEN}, D={HEAD_DIM}...")
for _ in range(5):
    run_sdpa()
    run_triton()

print("Running triton.testing.do_bench (this takes a few seconds)...")
ms_sdpa = triton.testing.do_bench(run_sdpa, quantiles=[0.5, 0.2, 0.8])[0]
ms_triton = triton.testing.do_bench(run_triton, quantiles=[0.5, 0.2, 0.8])[0]

# Calculate [X] Speedup
speedup = (ms_sdpa / ms_triton - 1) * 100

# Calculate [Y] TFLOP/s
# Total ~ 14 * B * H * S^2 * D for Fwd + Bwd
flops = 14 * BATCH_SIZE * NUM_HEADS * (SEQ_LEN ** 2) * HEAD_DIM
tflops_per_sec = (flops / (ms_triton * 1e-3)) / 1e12

print("\n" + "="*50)
print(f"--- Benchmark Results (Context: {SEQ_LEN}) ---")
print("="*50)
print(f"PyTorch SDPA Time : {ms_sdpa:.3f} ms")
print(f"Triton FA2 Time   : {ms_triton:.3f} ms")
print(f"--> [X] Speedup   : {speedup:.1f}% faster than SDPA")
print(f"--> [Y] Throughput: {tflops_per_sec:.2f} TFLOP/s")
print("="*50)

