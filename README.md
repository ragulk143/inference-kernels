# inference-kernels

Fused GPU kernels for LLM inference, written in [Triton](https://triton-lang.org/).
Each kernel is explained from first principles, implemented, verified for
correctness against a naive PyTorch baseline, benchmarked, and analyzed using
the GPU roofline model.

This README is written to be readable by someone with no prior CUDA/Triton
background — every kernel section explains *what the operation does*, *why
it matters for LLM inference*, and *why it got the speedup it got*.

---

## Why this exists

Every time a large language model generates a response, it runs the same set
of operations, over and over, for every single token: normalize activations
(RMSNorm), apply position information (RoPE), pass through a gated activation
(SwiGLU), and convert raw scores into probabilities (Softmax). These four
operations sit inside every transformer layer, in every LLM — GPT, LLaMA,
Mistral, all of them.

In plain PyTorch, each of these operations is written as several separate
steps, and each step is its own trip to the GPU's memory. **Fusion** means
rewriting the same math as a single custom kernel that loads data once, does
all the math while it's sitting in fast on-chip memory, and writes the result
once. Fewer memory trips = faster, which matters enormously when you're
running this billions of times a day in production.

This repo builds and benchmarks fused versions of all four operations, and
is honest about which ones actually benefit from fusion and why.

---

## 1. RMSNorm

**What it does:** normalizes a vector by its root-mean-square value, then
scales it by a learned weight. Used instead of LayerNorm in models like
LLaMA, because it skips computing the mean — fewer operations, same
stabilizing effect on training and inference.
**Where it's used:** applied before (or after) every attention block and
every feed-forward block in a transformer — it runs many times per token,
per layer.

**Naive PyTorch version:** 4 separate operations — square, mean, rsqrt,
multiply — each one a full read-and-write trip to GPU memory (HBM).

**Fused Triton version:** one kernel. Loads the row once, computes the whole
formula while the data sits in fast on-chip memory, writes the result once.

**Result:**

| Version | Time (shape 8192×4096) |
|---|---|
| Naive PyTorch | ~7.1 ms |
| Fused Triton | ~2.0 ms |
| **Speedup** | **~3.5x** |

**Why this speedup, specifically:** I calculated the theoretical minimum time
using the roofline model — the GPU has to read the input and write the output
at least once, and that movement alone takes time based on memory bandwidth:
The kernel runs at 2.0ms, which is **~68% of the theoretical best possible
time**. This tells us the kernel is **memory-bound** — its speed is limited
by how fast data can move in and out of GPU memory, not by how much math the
GPU has to do. This matters because it tells you *where to spend further
optimization effort* (or where not to): no amount of extra compute power
would make this kernel faster, only faster memory access would.

**Experiment — does increasing parallelism help?** I tested `num_warps`
values of 4 (default), 8, 16, and 32. All four landed within 3.46x-3.51x of
each other — essentially no difference. This confirms the memory-bound
diagnosis: warp count controls how work is scheduled across compute units,
but it can't make data arrive from memory any faster. Tuning scheduling
parameters only helps *compute-bound* kernels, not memory-bound ones like
this.

---

## 2. SwiGLU

**What it does:** a gated activation function used in the feed-forward block
of LLaMA and similar models, instead of the plain ReLU used in older
transformers.
The model computes two separate linear projections of its input (`gate` and
`up` — these are just matrix multiplies, not part of this kernel), then this
kernel combines them with the gating activation.

**Where it's used:** inside every feed-forward block in LLaMA-style models —
again, once per layer, per token.

**Naive PyTorch version:** 3 separate operations — sigmoid, multiply (for
SiLU), then a second multiply (for the gate).

**Fused Triton version:** one kernel, single pass.

**Result:**

| Version | Time (shape 8192×4096) |
|---|---|
| Naive PyTorch | ~8.1 ms |
| Fused Triton | ~3.1 ms |
| **Speedup** | **~2.65x** |

**Why a smaller speedup than RMSNorm:** roofline math says the theoretical
minimum is ~2.1ms (more data moved than RMSNorm, since there are two input
tensors), and the kernel achieves ~3.1ms — about 68% of peak, the *same
efficiency ratio* as RMSNorm. The smaller speedup number isn't because this
kernel is less optimized — it's because the naive PyTorch baseline had less
*wasted* memory traffic to begin with (no separate reduction step like
RMSNorm's `.mean()`), so there was simply less inefficiency available to
eliminate through fusion.

**Experiment — block size tuning:** tested `BLOCK_SIZE` values of 1024, 2048,
4096. All landed within noise of each other (~3.0-3.2ms). Same conclusion as
the warp experiment above: this is a memory-bound kernel, and scheduling
parameters don't move a memory-bound kernel's ceiling.

---

## 3. RoPE (Rotary Position Embedding)

**What it does:** injects position information into a sequence by rotating
pairs of elements in the input vector. The rotation angle depends on the
token's position in the sequence and which pair of dimensions is being
rotated (different pairs rotate at different "speeds").
**Where it's used:** applied to the Query and Key vectors in attention,
*before* the attention score computation — every layer, every token, in
LLaMA, Mistral, GPT-NeoX, and most modern open LLMs. Without this, the model
has no way of knowing token order.

**Naive PyTorch version:** computing the cos/sin angle tables, splitting the
input into even/odd-indexed elements (strided slicing), four multiplies for
the rotation math, and writing the result back into an interleaved layout —
many separate operations, including some that don't read/write memory
efficiently (strided slicing).

**Fused Triton version:** one kernel computes the full rotation per row in a
single pass.

**Result:**

| Version | Time (shape 32×4096×128) |
|---|---|
| Naive PyTorch | ~8.0 ms |
| Fused Triton | ~1.6 ms |
| **Speedup** | **~5.0x** — the best result in this repo |

**Why the biggest speedup of all four kernels:** the naive version has the
*most* separate operations and includes inefficient strided memory access
(PyTorch's `x[..., 0::2]` slicing) in the baseline itself — every one of
those gets eliminated by fusion. More inefficiency in the baseline means more
room for fusion to win.

**But — lower efficiency than RMSNorm/SwiGLU:** roofline math gives a
theoretical minimum of ~0.7ms; the kernel achieves ~1.6ms, only **~44% of
peak bandwidth** (versus ~68% for the other two). This is because the kernel
itself still uses a strided access pattern internally (`col_offsets * 2` and
`col_offsets * 2 + 1`) to read interleaved pairs — it's far better than the
naive baseline's strided access, but not as efficient as a fully contiguous
read.

**Failed optimization attempt (documented honestly):** I tried fixing the
strided access by pre-splitting the input into two contiguous tensors
(`x[:, 0::2].contiguous()` and `x[:, 1::2].contiguous()`) before calling the
kernel, so the kernel itself would only need simple contiguous loads. This
made performance *worse* — 5.75ms instead of 1.6ms. The reason: the
pre-split step and the final recombine step each cost their own full
read/write pass over the data, adding two extra HBM round-trips. I had moved
the strided cost earlier in the pipeline and added more total memory traffic,
rather than removing it. I reverted to the original version. This taught me
that "remove a strided access" only helps if the replacement doesn't
introduce new full-tensor passes elsewhere — a fix has to reduce total bytes
moved, not just relocate where the inefficiency happens.

---

## 4. Fused Softmax

**What it does:** converts a row of raw scores into probabilities that sum
to 1, used at the end of the attention score computation (`softmax(QK^T)`)
and anywhere else a probability distribution is needed.The `- max(x)` step prevents numerical overflow: without it, `exp()` of large
input values produces infinity. Subtracting a constant from every element in
a row before exponentiating doesn't change the final result, because the
constant cancels out in the numerator/denominator ratio — so this safety
trick is mathematically free.

**Where it's used:** the final step of every attention computation, in every
layer, every token, every model.

**Baseline used:** PyTorch's built-in `torch.softmax()` — unlike the other
three kernels, this baseline is *not* naive. It's already a hand-optimized,
fused CUDA kernel written by PyTorch's own engineers.

**Result:**

| Version | Time (shape 8192×4096) |
|---|---|
| PyTorch built-in | ~2.0 ms |
| My Triton kernel | ~2.0 ms |
| **Speedup** | **~1.0x** — no meaningful difference |

**Why this result matters:** this is the most important result in the repo,
specifically *because* it's not an improvement. Fusion only helps when
there's real inefficiency in the baseline to eliminate. PyTorch's softmax has
no such inefficiency — there was nothing left to fuse away. Landing within
1% of an expert-tuned kernel on a first implementation attempt is actually a
strong outcome on its own, but the headline number ("1.0x speedup") looks
unimpressive unless you understand *why*: it's the boundary condition that
proves the other three results aren't a fluke or measurement error. If
fusion always showed a 3-5x win regardless of baseline, that would actually
be suspicious. This result is what makes the other three credible.

---

## Summary table

| Kernel | Speedup | % of theoretical peak bandwidth | What this tells you |
|---|---|---|---|
| RMSNorm | ~3.5x | ~68% | Memory-bound; fusion eliminates real redundant HBM traffic |
| SwiGLU | ~2.65x | ~68% | Memory-bound; smaller speedup because less redundancy existed to begin with |
| RoPE | ~5.0x | ~44% | Biggest win, but strided access pattern leaves room for further optimization |
| Softmax | ~1.0x | N/A | Confirms fusion only helps against genuinely inefficient baselines |

All benchmarks on NVIDIA RTX 3050 Laptop GPU (6GB VRAM, ~192 GB/s memory
bandwidth, Ampere architecture, compute capability 8.6). CUDA 13.0, PyTorch
2.5.1, Triton 3.1.0.

---

## Usage

These kernels are meant to be dropped directly into a transformer model's
forward pass, replacing the equivalent PyTorch operations. Here's how each
one fits into real model code — not isolated examples, but the actual
context where they're used.

### RMSNorm — inside a transformer block

```python
import torch
import torch.nn as nn
from inference_kernels.triton_rmsnorm import triton_rmsnorm

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(dim))
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.eps = 1e-6

    def forward(self, x):
        # Normalize before attention -- this replaces nn.LayerNorm(dim)
        normed = triton_rmsnorm(x, self.norm_weight, self.eps)
        attn_out, _ = self.attn(normed, normed, normed)
        return x + attn_out  # residual connection
```

### SwiGLU — inside a feed-forward block

```python
from inference_kernels.triton_swiglu import triton_swiglu

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # This replaces: F.silu(gate) * up
        activated = triton_swiglu(gate, up)
        return self.down_proj(activated)
```

### RoPE — applied to Q/K before attention scores

```python
from inference_kernels.triton_rope import triton_rope

class Attention(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        batch, seq_len, dim = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)

        # Apply position information before computing attention scores
        q = triton_rope(q, seq_len, self.head_dim)
        k = triton_rope(k, seq_len, self.head_dim)

        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        # ... attention score computation continues with q, k, v
        return q, k, v
```

### Softmax — converting attention scores to probabilities

```python
from inference_kernels.triton_softmax import triton_softmax

def attention_scores(q, k, v, scale):
    # q, k shape: (batch, heads, seq_len, head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = triton_softmax(scores)  # replaces F.softmax(scores, dim=-1)
    return torch.matmul(probs, v)
```

### Installation

```bash
git clone https://github.com/ragulk143/inference-kernels.git
cd inference-kernels
pip install -e .
```

Requires a CUDA-capable GPU, PyTorch >= 2.0, Triton >= 2.0.

## Running tests and benchmarks

```bash
cd tests && python3 test_triton_rmsnorm.py       # correctness
cd benchmark && python3 compare_fused_vs_unfused.py   # performance
```

## License

MIT
