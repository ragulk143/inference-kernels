import torch
import sys
sys.path.insert(0, '../python')
from inference_kernels.triton_swiglu import triton_swiglu

torch.manual_seed(0)
shape = (8192, 4096)
gate = torch.randn(*shape, device='cuda')
up = torch.randn(*shape, device='cuda')

out = triton_swiglu(gate, up)
torch.cuda.synchronize()
