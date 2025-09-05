import os
import torch
import torch.nn.functional as F
import time
from torch.nn.attention import SDPBackend, sdpa_kernel

from torch.backends.cuda import (
    flash_sdp_enabled,
    mem_efficient_sdp_enabled,
    math_sdp_enabled,
    cudnn_sdp_enabled
)

print("FLASH:", flash_sdp_enabled())
print("EFFICIENT:", mem_efficient_sdp_enabled())
print("MATH:", math_sdp_enabled())
print("CUDNN:", cudnn_sdp_enabled())

device = "cuda"
dtype = torch.float16

batch = 2
heads = 8
seq_len = 1024
head_dim = 64

q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)


def benchmark(backend, warmup=10, iters=50):
    
    with sdpa_kernel(backend):
        for _ in range(warmup):
            F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(iters):
            F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        end = time.time()

    return (end - start) * 1000 / iters  # ms


print("Benchmarking attention backends...")
for backend in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]:
    try:
        t = benchmark(backend)
        print(f"{backend.name:<20}: {t:.3f} ms")
    except Exception as e:
        print(f"{backend.name:<20}: Not supported ({e})")
