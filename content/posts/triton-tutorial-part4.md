---
title: "Learning Triton Part 4: Custom Attention Kernel"
date: 2023-08-22
draft: false
tags: ["triton", "cuda", "gpu", "kernel", "deep-learning", "attention", "flash-attention", "transformers"]
math: true
summary: "Implementing scaled dot-product attention with causal masking in Triton — applying everything from Parts 1–3 to build a tiled, memory-efficient attention kernel from scratch."
---

This is the payoff. In Parts [1](/posts/triton-tutorial-part1), [2](/posts/triton-tutorial-part2), and [3](/posts/triton-tutorial-part3) we built up the Triton programming model, autotuning, and kernel fusion. Now we apply all of it to implement **scaled dot-product attention** — the core operation of every transformer.

---

## The Problem with Naive Attention

Standard attention computes:

$$O = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

where $M$ is the causal mask. The naive implementation materializes the full $N \times N$ attention matrix in GPU global memory. For `N=8192` and `float32`, that's `8192² × 4 bytes = 256 MB` — per head, per layer, per batch item. It adds up fast.

**FlashAttention** (Dao et al., 2022) showed you can compute attention without ever materializing the full matrix, by tiling the computation and keeping intermediate results in fast on-chip SRAM. We'll implement a simplified version of this idea.

---

## The Tiling Strategy

Instead of computing all of `QK^T` at once, we process it in **tiles**:

- Split `Q` into row tiles of size `BLOCK_M`
- Split `K, V` into column tiles of size `BLOCK_N`
- For each row tile of `Q`, iterate over all column tiles of `K, V`
- Accumulate a running softmax using the online softmax trick

The key insight: for each output row, we only need to keep in SRAM:
- The current tile of `K` and `V`
- A running max `m` and running sum `l` for the online softmax
- The accumulating output `acc`

This is $O(N \cdot d)$ memory instead of $O(N^2)$.

---

## Online Softmax

The standard softmax requires two passes: find max, then normalize. For tiled attention we need to update these incrementally as we see new tiles.

Given previous state $(m_{\text{prev}}, l_{\text{prev}}, \text{acc}_{\text{prev}})$ and a new tile of scores $s$:

$$m_{\text{new}} = \max(m_{\text{prev}}, \max(s))$$
$$l_{\text{new}} = e^{m_{\text{prev}} - m_{\text{new}}} \cdot l_{\text{prev}} + \sum e^{s - m_{\text{new}}}$$
$$\text{acc}_{\text{new}} = e^{m_{\text{prev}} - m_{\text{new}}} \cdot \text{acc}_{\text{prev}} + e^{s - m_{\text{new}}} \cdot V_{\text{tile}}$$

At the end, normalize: $O = \text{acc} / l$.

---

## The Kernel

```python
import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    N,           # sequence length
    D,           # head dimension
    scale,       # 1 / sqrt(D)
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,   # rows of Q per program
    BLOCK_N: tl.constexpr,   # cols of K/V per tile
    BLOCK_D: tl.constexpr,   # head dim (must cover full D)
):
    # program handles one (batch, head, row_block) triple
    batch_head = tl.program_id(0)
    start_m    = tl.program_id(1)  # which Q row block

    # Decode batch and head from flat index
    num_heads = stride_qb // stride_qh
    b = batch_head // num_heads
    h = batch_head  % num_heads

    # Base pointers for this (batch, head)
    Q_bh = Q_ptr + b * stride_qb + h * stride_qh
    K_bh = K_ptr + b * stride_kb + h * stride_kh
    V_bh = V_ptr + b * stride_vb + h * stride_vh
    O_bh = O_ptr + b * stride_ob + h * stride_oh

    # Row indices this program is responsible for
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q tile: shape [BLOCK_M, BLOCK_D]
    Q = tl.load(
        Q_bh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=(offs_m[:, None] < N) & (offs_d[None, :] < D),
        other=0.0,
    )

    # Running state for online softmax
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Determine how many K/V tiles to process (causal: only up to current row)
    end_n = (start_m + 1) * BLOCK_M if IS_CAUSAL else N

    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K tile: shape [BLOCK_N, BLOCK_D]
        K = tl.load(
            K_bh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=(offs_n[:, None] < N) & (offs_d[None, :] < D),
            other=0.0,
        )

        # QK^T: shape [BLOCK_M, BLOCK_N]
        scores = tl.dot(Q, tl.trans(K)) * scale

        # Causal mask: mask out positions where key > query
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            scores = tl.where(causal_mask, scores, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)          # rescale factor for old acc
        p    = tl.exp(scores - m_new[:, None])  # exp of new scores

        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = alpha[:, None] * acc

        # Load V tile and accumulate
        V = tl.load(
            V_bh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=(offs_n[:, None] < N) & (offs_d[None, :] < D),
            other=0.0,
        )
        acc += tl.dot(p.to(V.dtype), V)

        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Write output
    tl.store(
        O_bh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc,
        mask=(offs_m[:, None] < N) & (offs_d[None, :] < D),
    )


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              causal: bool = True) -> torch.Tensor:
    """
    q, k, v: shape [B, H, N, D]
    """
    B, H, N, D = q.shape
    assert D in (32, 64, 128), "BLOCK_D must match D exactly"

    o = torch.empty_like(q)
    scale = D ** -0.5

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D  # must cover full head dim

    # Grid: (B*H, ceil(N/BLOCK_M))
    grid = (B * H, triton.cdiv(N, BLOCK_M))

    attention_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        N, D, scale,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return o
```

---

## Verify Against PyTorch

```python
def ref_attention(q, k, v, causal=True):
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("bhmd,bhnd->bhmn", q, k) * scale
    if causal:
        N = q.shape[2]
        mask = torch.tril(torch.ones(N, N, device=q.device)).bool()
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.einsum("bhmn,bhnd->bhmd", torch.softmax(scores, dim=-1), v)


if __name__ == "__main__":
    B, H, N, D = 2, 4, 512, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    out_triton = attention(q, k, v, causal=True)
    out_ref    = ref_attention(q.float(), k.float(), v.float()).half()

    print(f"Max diff: {(out_triton - out_ref).abs().max().item():.6f}")
    # Max diff: ~0.002 (expected for float16 accumulation)
```

Small numerical differences are expected due to float16 accumulation order — the values should be close, not identical.

---

## What We Built

| Feature | Our kernel |
|---|---|
| Tiled computation | ✅ BLOCK_M × BLOCK_N tiles |
| Online softmax | ✅ No N×N matrix materialized |
| Causal masking | ✅ `IS_CAUSAL` constexpr flag |
| Multi-head | ✅ Flat batch×head grid |
| Memory complexity | O(N·D) instead of O(N²) |

This is the core idea behind FlashAttention. The real FlashAttention adds backward pass, fp16/bf16 optimizations, bank conflict avoidance in shared memory, and highly tuned block sizes — but the fundamental algorithm is what we've implemented here.

---

## Limitations and Next Steps

This kernel is a learning implementation. Production-grade improvements:

- **Backward pass** — requires storing the softmax normalization factors (`m` and `l`) for the backward kernel
- **Flash Attention v2/v3** — better work partitioning across warps, reduced non-matmul FLOPs
- **MQA / GQA** — grouped-query attention where K and V have fewer heads than Q
- **Variable sequence lengths** — for batches with padding

For production use, reach for [flash-attn](https://github.com/Dao-AILab/flash-attention) directly. For learning how it works — you just built it.

---

## Series Recap

| Part | Topic |
|---|---|
| [Part 1](/posts/triton-tutorial-part1) | Vector addition, core programming model |
| [Part 2](/posts/triton-tutorial-part2) | Autotuning and benchmarking |
| [Part 3](/posts/triton-tutorial-part3) | Fused kernels and row-wise softmax |
| **Part 4** | **Custom attention kernel (this post)** |

From `C[i] = A[i] + B[i]` to a full tiled attention kernel in four posts. The Triton programming model scales surprisingly far.
