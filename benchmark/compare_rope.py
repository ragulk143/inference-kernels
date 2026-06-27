import torch
import time
import sys
sys.path.insert(0, '../python')
from inference_kernels import naive_rope
from inference_kernels.triton_rope import triton_rope


def benchmark(fn, *args, warmup=10, iters=100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / iters * 1000


def main():
    torch.manual_seed(0)
    batch, seq_len, dim = 32, 4096, 128  # realistic LLM-scale shape

    x = torch.randn(batch, seq_len, dim, device='cuda')

    print(f"Benchmarking RoPE: shape=({batch}, {seq_len}, {dim})")
    print("-" * 50)

    naive_ms = benchmark(naive_rope, x, seq_len, dim)
    print(f"Naive PyTorch RoPE: {naive_ms:.4f} ms")

    triton_ms = benchmark(triton_rope, x, seq_len, dim)
    print(f"Triton Fused RoPE:  {triton_ms:.4f} ms")

    speedup = naive_ms / triton_ms
    print("-" * 50)
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
