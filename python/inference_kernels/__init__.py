import torch
import torch.nn as nn


class NaiveRMSNorm(nn.Module):
    """Reference implementation — plain PyTorch ops, no fusion."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(mean_sq + self.eps)
        return x_normed * self.weight


def naive_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference implementation — plain PyTorch ops."""
    silu_gate = gate * torch.sigmoid(gate)
    return silu_gate * up


def naive_rope(x: torch.Tensor, seq_len: int, dim: int, base: float = 10000.0) -> torch.Tensor:
    """
    Reference implementation — plain PyTorch ops.
    x shape: (batch, seq_len, dim) — dim must be even (pairs of elements rotated together)
    """
    device = x.device

    # Compute rotation frequencies for each pair of dimensions
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))  # shape (dim/2,)

    # Position indices: 0, 1, 2, ..., seq_len-1
    positions = torch.arange(seq_len, device=device).float()  # shape (seq_len,)

    # Outer product: angle for each (position, frequency) pair
    angles = torch.outer(positions, freqs)  # shape (seq_len, dim/2)

    cos = torch.cos(angles)  # (seq_len, dim/2)
    sin = torch.sin(angles)  # (seq_len, dim/2)

    # Split x into even/odd indexed elements (the pairs we rotate)
    x1 = x[..., 0::2]  # (batch, seq_len, dim/2)
    x2 = x[..., 1::2]  # (batch, seq_len, dim/2)

    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos

    # Interleave back together
    out = torch.empty_like(x)
    out[..., 0::2] = x1_rot
    out[..., 1::2] = x2_rot

    return out


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Reference implementation -- PyTorch's built-in softmax."""
    return torch.softmax(x, dim=-1)
