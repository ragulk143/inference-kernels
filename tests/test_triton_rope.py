import torch
import sys
sys.path.insert(0, '../python')
from inference_kernels import naive_rope
from inference_kernels.triton_rope import triton_rope

def test_triton_matches_naive():
    torch.manual_seed(0)
    batch, seq_len, dim = 4, 128, 64
    x = torch.randn(batch, seq_len, dim, device='cuda')

    naive_out = naive_rope(x, seq_len, dim)
    triton_out = triton_rope(x, seq_len, dim)

    max_diff = (naive_out - triton_out).abs().max().item()
    print(f"Max difference: {max_diff}")

    assert torch.allclose(naive_out, triton_out, atol=1e-4), f"Mismatch! Max diff: {max_diff}"
    print("Triton RoPE matches Naive RoPE!")

if __name__ == "__main__":
    test_triton_matches_naive()
