import torch
import sys
sys.path.insert(0, '../python')
from inference_kernels import naive_swiglu
from inference_kernels.triton_swiglu import triton_swiglu

def test_triton_matches_naive():
    torch.manual_seed(0)
    gate = torch.randn(8192, 4096, device='cuda')
    up = torch.randn(8192, 4096, device='cuda')

    naive_out = naive_swiglu(gate, up)
    triton_out = triton_swiglu(gate, up)

    max_diff = (naive_out - triton_out).abs().max().item()
    print(f"Max difference: {max_diff}")

    assert torch.allclose(naive_out, triton_out, atol=1e-4), f"Mismatch! Max diff: {max_diff}"
    print("Triton SwiGLU matches Naive SwiGLU!")

if __name__ == "__main__":
    test_triton_matches_naive()
