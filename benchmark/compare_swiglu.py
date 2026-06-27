import torch
import time
import sys
sys.path.insert(0, '../python')
from inference_kernels import naive_swiglu
from inference_kernels.triton_swiglu import triton_swiglu


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
    shape = (8192, 4096)

    gate = torch.randn(*shape, device='cuda')
    up = torch.randn(*shape, device='cuda')

    print(f"Benchmarking SwiGLU: shape={shape}")
    print("-" * 50)

    naive_ms = benchmark(naive_swiglu, gate, up)
    print(f"Naive PyTorch SwiGLU: {naive_ms:.4f} ms")

    triton_ms = benchmark(triton_swiglu, gate, up)
    print(f"Triton Fused SwiGLU:  {triton_ms:.4f} ms")

    speedup = naive_ms / triton_ms
    print("-" * 50)
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
