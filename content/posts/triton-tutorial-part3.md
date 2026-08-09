---
title: "Learning Triton Part 3: Fused Kernels"
date: 2023-08-15
draft: false
tags: ["triton", "cuda", "gpu", "kernel", "deep-learning", "fused-kernels", "softmax"]
summary: "Why kernel fusion matters for GPU performance, and how to implement fused operations in Triton — covering a fused bias+ReLU and a full row-wise softmax kernel."
math: true
---

In [Part 2](/posts/triton-tutorial-part2) we saw that Triton matches PyTorch on simple vector addition. So why bother writing custom kernels at all?

The answer is **fusion**. This is where Triton earns its keep.

---

## The Memory Bandwidth Problem

Modern GPUs are fast at compute but relatively slow at memory access. Consider `y = relu(x + bias)` in PyTorch:

```python
y = torch.relu(x + bias)
```

Under the hood this is **two kernel launches**:
1. `x + bias` — reads `x` and `bias`, writes a temp tensor
2. `relu(...)` — reads the temp tensor, writes `y`

That's **3 reads + 2 writes** of the full tensor. With a fused kernel, it's **2 reads + 1 write** — the intermediate result never touches global memory.

For memory-bandwidth-bound ops (most elementwise work), this is a direct speedup proportional to the bandwidth saved.

---

## Fused Bias + ReLU

```python
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def bias_relu_kernel(
    x_ptr, bias_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x    = tl.load(x_ptr    + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)

    # Fused: add then relu — no intermediate write to global memory
    out = tl.maximum(x + bias, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bias_relu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and bias.is_cuda
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    bias_relu_kernel[grid](x, bias, out, N)
    return out
```

`tl.maximum(val, 0.0)` is ReLU — Triton provides it as an intrinsic. The key point: the addition and the clamp happen **in registers**, never spilling to global memory.

---

## Row-wise Softmax

Softmax is a more interesting fusion challenge. For a 2D tensor of shape `(M, N)`, we compute for each row:


<div>
$$
\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
$$
</div>


The numerically stable version requires **three passes** over the row: find max, compute exp, normalize. In Triton, a single program handles one full row — all three passes happen in registers (or shared memory for large rows), with just one read and one write of global memory.

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128},  num_warps=2),
        triton.Config({"BLOCK_SIZE": 256},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def softmax_kernel(
    out_ptr, x_ptr,
    x_row_stride, out_row_stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)

    x_row_start   = x_ptr   + row * x_row_stride
    out_row_start = out_ptr + row * out_row_stride

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load the row (pad out-of-bounds with -inf so they don't affect max/sum)
    x = tl.load(x_row_start + offsets, mask=mask, other=-float("inf"))

    # Pass 1: numerically stable shift
    x = x - tl.max(x, axis=0)

    # Pass 2: exp
    x = tl.exp(x)

    # Pass 3: normalize
    x = x / tl.sum(x, axis=0)

    tl.store(out_row_start + offsets, x, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2
    M, N = x.shape
    out = torch.empty_like(x)

    # One program per row
    grid = (M,)

    softmax_kernel[grid](
        out, x,
        x.stride(0), out.stride(0),
        N,
    )
    return out
```

### Key ideas:

**`x.stride(0)`** — the number of elements to skip to move to the next row. Using strides instead of hardcoded shapes makes the kernel work correctly for non-contiguous tensors.

**`other=-float("inf")`** — out-of-bounds elements are loaded as `-inf`, so they don't corrupt the max or the sum.

**`tl.max` and `tl.sum` with `axis=0`** — these are reduction operations across the block. Triton compiles them to efficient warp-level reductions.

---

## Benchmark: Triton Softmax vs PyTorch

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="Softmax Throughput (M=4096 rows)",
        args={"M": 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1), quantiles=quantiles)

    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True)
```

For smaller row sizes (< 512), Triton is competitive or faster. For very large rows, PyTorch's optimized CUDA kernels pull ahead — but our implementation is readable and correct, and a solid starting point.

---

## The Fusion Mindset

When writing a Triton kernel, ask:

> What data do I load from global memory? Can I do more work before writing it back?

Each round-trip to global memory costs ~100 cycles of latency. Every operation you can fold in — activation, normalization, masking, scaling — eliminates a kernel launch and a full tensor read/write.

This is why FlashAttention is fast: it fuses the entire attention computation (QK^T, softmax, ×V) into a single kernel, using on-chip SRAM as a scratchpad instead of materializing the N×N attention matrix.

---

## What's Next

Part 4 is the payoff: a **custom attention kernel**. We'll apply everything from Parts 1–3 — the Triton programming model, autotuning, and fusion — to implement scaled dot-product attention with causal masking in Triton.
