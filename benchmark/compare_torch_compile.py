import torch
import time
import sys
sys.path.insert(0, '../python')
from inference_kernels import NaiveRMSNorm
from inference_kernels.triton_rmsnorm import triton_rmsnorm


def benchmark(fn, *args, warmup=10, iters=100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def main():
    torch.manual_seed(0)
    dim = 4096
    x = torch.randn(8192, dim, device='cuda')

    naive = NaiveRMSNorm(dim).to('cuda')
    compiled_naive = torch.compile(naive)

    print("Three-way RMSNorm comparison")
    print("-" * 50)

    naive_ms = benchmark(naive, x)
    print(f"Naive PyTorch (uncompiled): {naive_ms:.4f} ms")

    compiled_ms = benchmark(compiled_naive, x)
    print(f"Naive PyTorch + torch.compile: {compiled_ms:.4f} ms")

    triton_ms = benchmark(triton_rmsnorm, x, naive.weight, naive.eps)
    print(f"Hand-written Triton kernel: {triton_ms:.4f} ms")

    print("-" * 50)
    print(f"torch.compile speedup over naive: {naive_ms/compiled_ms:.2f}x")
    print(f"Hand-written Triton speedup over naive: {naive_ms/triton_ms:.2f}x")
    print(f"Hand-written Triton vs torch.compile: {compiled_ms/triton_ms:.2f}x")


if __name__ == "__main__":
    main()
