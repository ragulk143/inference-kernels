import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    x_ptr, out_ptr,
    cos_ptr, sin_ptr,
    seq_len, half_dim,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    pos = row_idx % seq_len

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < half_dim

    row_start = x_ptr + row_idx * (half_dim * 2)
    x1 = tl.load(row_start + col_offsets * 2, mask=mask, other=0.0)
    x2 = tl.load(row_start + col_offsets * 2 + 1, mask=mask, other=0.0)

    cos_row = cos_ptr + pos * half_dim
    sin_row = sin_ptr + pos * half_dim
    cos = tl.load(cos_row + col_offsets, mask=mask, other=0.0)
    sin = tl.load(sin_row + col_offsets, mask=mask, other=0.0)

    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos

    out_row_start = out_ptr + row_idx * (half_dim * 2)
    tl.store(out_row_start + col_offsets * 2, x1_rot, mask=mask)
    tl.store(out_row_start + col_offsets * 2 + 1, x2_rot, mask=mask)


def triton_rope(x: torch.Tensor, seq_len: int, dim: int, base: float = 10000.0) -> torch.Tensor:
    device = x.device
    half_dim = dim // 2

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(seq_len, device=device).float()
    angles = torch.outer(positions, freqs)
    cos = torch.cos(angles).contiguous()
    sin = torch.sin(angles).contiguous()

    orig_shape = x.shape
    x_flat = x.reshape(-1, dim)
    n_rows = x_flat.shape[0]

    out = torch.empty_like(x_flat)

    BLOCK_SIZE = triton.next_power_of_2(half_dim)
    grid = (n_rows,)

    rope_kernel[grid](
        x_flat, out,
        cos, sin,
        seq_len, half_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.reshape(orig_shape)
