import torch
import sys
sys.path.insert(0, '../python')
from inference_kernels.triton_rmsnorm import triton_rmsnorm

torch.manual_seed(0)
dim = 4096
x = torch.randn(8192, dim, device='cuda')
weight = torch.ones(dim, device='cuda')

out = triton_rmsnorm(x, weight, eps=1e-6)
torch.cuda.synchronize()
