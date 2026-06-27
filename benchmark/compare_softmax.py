import torch
import time
import sys
sys.path.insert(0, '../python')
from inference_kernels import naive_softmax
from inference_kernels.triton_softmax import triton_softmax


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
    x = torch.randn(*shape, device='cuda')

    print(f"Benchmarking Softmax: shape={shape}")
    print("-" * 50)

    naive_ms = benchmark(naive_softmax, x)
    print(f"PyTorch built-in Softmax: {naive_ms:.4f} ms")

    triton_ms = benchmark(triton_softmax, x)
    print(f"Triton Fused Softmax:     {triton_ms:.4f} ms")

    speedup = naive_ms / triton_ms
    print("-" * 50)
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
