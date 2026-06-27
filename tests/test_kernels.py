import torch
import sys
sys.path.insert(0, '../python')
from inference_kernels import NaiveRMSNorm

def test_naive_rmsnorm_shape():
    dim = 64
    x = torch.randn(4, 10, dim)  # batch=4, seq=10, dim=64
    norm = NaiveRMSNorm(dim)
    out = norm(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print("Shape test passed:", out.shape)

def test_naive_rmsnorm_values():
    # Manually verify the math on a tiny known input
    x = torch.tensor([[3.0, 4.0]])  # mean(x^2) = (9+16)/2 = 12.5
    norm = NaiveRMSNorm(dim=2, eps=0.0)
    norm.weight.data = torch.ones(2)  # weight = 1, isolates the normalization
    out = norm(x)
    expected_rms = (12.5) ** 0.5  # sqrt(12.5)
    expected = x / expected_rms
    assert torch.allclose(out, expected, atol=1e-5), f"Got {out}, expected {expected}"
    print("Value test passed:", out)

if __name__ == "__main__":
    test_naive_rmsnorm_shape()
    test_naive_rmsnorm_values()
    print("\nAll tests passed!")
