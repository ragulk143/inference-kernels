import torch
import sys
sys.path.insert(0, '../python')
from inference_kernels.triton_softmax import triton_softmax

torch.manual_seed(0)
shape = (8192, 4096)
x = torch.randn(*shape, device='cuda')

out = triton_softmax(x)
torch.cuda.synchronize()
