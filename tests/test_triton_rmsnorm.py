import torch
import sys
sys.path.insert(0, '../python')
from inference_kernels import NaiveRMSNorm
from inference_kernels.triton_rmsnorm import triton_rmsnorm

def test_triton_matches_naive():
    torch.manual_seed(0)
    dim = 64
    x = torch.randn(4, 10, dim, device='cuda')

    naive = NaiveRMSNorm(dim).to('cuda')
    naive_out = naive(x)

    triton_out = triton_rmsnorm(x, naive.weight, eps=naive.eps)

    max_diff = (naive_out - triton_out).abs().max().item()
    print(f"Max difference: {max_diff}")

    assert torch.allclose(naive_out, triton_out, atol=1e-4), f"Mismatch! Max diff: {max_diff}"
    print("Triton RMSNorm matches Naive RMSNorm!")

if __name__ == "__main__":
    test_triton_matches_naive()
