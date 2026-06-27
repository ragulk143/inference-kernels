import torch
import time
import sys
sys.path.insert(0, '../python')
from inference_kernels import NaiveRMSNorm
from inference_kernels.triton_rmsnorm import triton_rmsnorm


def benchmark(fn, *args, warmup=10, iters=100):
    # Warmup — GPU needs a few runs to reach steady state (JIT compile, cache warm)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) / iters * 1000
    return avg_ms


def main():
    torch.manual_seed(0)
    dim = 4096  # realistic LLM hidden dimension size
    batch_seq = 8192  # realistic batch*seq_len for a forward pass

    x = torch.randn(batch_seq, dim, device='cuda')
    naive = NaiveRMSNorm(dim).to('cuda')

    print(f"Benchmarking RMSNorm: shape=({batch_seq}, {dim})")
    print("-" * 50)

    naive_ms = benchmark(naive, x)
    print(f"Naive PyTorch RMSNorm: {naive_ms:.4f} ms")

    triton_ms = benchmark(triton_rmsnorm, x, naive.weight, naive.eps)
    print(f"Triton Fused RMSNorm:  {triton_ms:.4f} ms")

    speedup = naive_ms / triton_ms
    print("-" * 50)
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
