import torch
import sys
sys.path.insert(0, '../python')
from inference_kernels.triton_rope import triton_rope

torch.manual_seed(0)
batch, seq_len, dim = 32, 4096, 128
x = torch.randn(batch, seq_len, dim, device='cuda')

out = triton_rope(x, seq_len, dim)
torch.cuda.synchronize()
