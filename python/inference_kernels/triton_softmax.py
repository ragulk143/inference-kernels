import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr, out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # One program instance handles one full row
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_start_ptr = x_ptr + row_idx * n_cols
    x = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))

    # Numerical stability: subtract the row's max before exponentiating
    row_max = tl.max(x, axis=0)
    x_shifted = x - row_max

    numerator = tl.exp(x_shifted)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    out_row_start_ptr = out_ptr + row_idx * n_cols
    tl.store(out_row_start_ptr + col_offsets, result, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    n_cols = orig_shape[-1]
    x_flat = x.reshape(-1, n_cols)
    n_rows = x_flat.shape[0]

    out = torch.empty_like(x_flat)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)

    softmax_kernel[grid](
        x_flat, out,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.reshape(orig_shape)
