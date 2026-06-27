import torch
import sys
sys.path.insert(0, '../python')
from inference_kernels import naive_rope

def test_rope_shape():
    batch, seq_len, dim = 2, 16, 64
    x = torch.randn(batch, seq_len, dim, device='cuda')
    out = naive_rope(x, seq_len, dim)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    print("Shape test passed:", out.shape)

def test_rope_preserves_norm():
    # Rotation should NOT change the magnitude of each (x1,x2) pair
    # This is a key property of RoPE -- it's a pure rotation, so norm is preserved
    batch, seq_len, dim = 1, 4, 8
    x = torch.randn(batch, seq_len, dim, device='cuda')
    out = naive_rope(x, seq_len, dim)

    x1, x2 = x[..., 0::2], x[..., 1::2]
    out1, out2 = out[..., 0::2], out[..., 1::2]

    orig_norm = (x1**2 + x2**2).sqrt()
    new_norm = (out1**2 + out2**2).sqrt()

    assert torch.allclose(orig_norm, new_norm, atol=1e-4), "Norm not preserved -- rotation math is wrong!"
    print("Norm preservation test passed -- rotation math is correct")

if __name__ == "__main__":
    test_rope_shape()
    test_rope_preserves_norm()
    print("\nAll tests passed!")
