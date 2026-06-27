import torch
import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(
    gate_ptr, up_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance handles one BLOCK_SIZE chunk of the flattened tensor
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0)

    # SiLU(gate) = gate * sigmoid(gate)
    sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate))
    silu_gate = gate * sigmoid_gate

    result = silu_gate * up

    tl.store(out_ptr + offsets, result, mask=mask)


def triton_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    assert gate.shape == up.shape, "gate and up must have the same shape"
    orig_shape = gate.shape

    gate_flat = gate.reshape(-1)
    up_flat = up.reshape(-1)
    n_elements = gate_flat.numel()

    out = torch.empty_like(gate_flat)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    swiglu_kernel[grid](
        gate_flat, up_flat, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.reshape(orig_shape)
