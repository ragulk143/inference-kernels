import torch
import sys
sys.path.insert(0, '../python')
from inference_kernels import naive_softmax
from inference_kernels.triton_softmax import triton_softmax

def test_triton_matches_naive():
    torch.manual_seed(0)
    x = torch.randn(8192, 4096, device='cuda')

    naive_out = naive_softmax(x)
    triton_out = triton_softmax(x)

    max_diff = (naive_out - triton_out).abs().max().item()
    print(f"Max difference: {max_diff}")

    assert torch.allclose(naive_out, triton_out, atol=1e-4), f"Mismatch! Max diff: {max_diff}"
    print("Triton softmax matches Naive softmax!")

def test_rows_sum_to_one():
    # Sanity check -- every row of a valid softmax output must sum to 1.0
    x = torch.randn(100, 50, device='cuda')
    out = triton_softmax(x)
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
    print("Row-sum-to-1 test passed!")

if __name__ == "__main__":
    test_triton_matches_naive()
    test_rows_sum_to_one()
