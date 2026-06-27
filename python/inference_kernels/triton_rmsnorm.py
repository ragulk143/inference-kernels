import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, out_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance handles ONE row of the input
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Pointer to the start of this row
    row_start_ptr = x_ptr + row_idx * n_cols
    x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)

    # Compute mean of squares (RMS normalization denominator)
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / n_cols

    rms = tl.sqrt(mean_sq + eps)
    x_normed = x / rms

    # Load weight and apply scale
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    out = x_normed * weight

    # Write result back
    out_row_start_ptr = out_ptr + row_idx * n_cols
    tl.store(out_row_start_ptr + col_offsets, out, mask=mask)


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: shape (n_rows, n_cols) - we'll flatten leading dims into n_rows
    weight: shape (n_cols,)
    """
    orig_shape = x.shape
    n_cols = orig_shape[-1]
    x_flat = x.reshape(-1, n_cols)
    n_rows = x_flat.shape[0]

    out = torch.empty_like(x_flat)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grid = (n_rows,)  # one program instance per row
    rmsnorm_kernel[grid](
        x_flat, weight, out,
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8
    )

    return out.reshape(orig_shape)
