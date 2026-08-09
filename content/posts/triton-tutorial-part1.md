---
title: "Learning Triton Part 1: Vector Addition"
date: 2023-08-01
draft: false
tags: ["triton", "cuda", "gpu", "kernel", "deep-learning"]
summary: "An introduction to writing GPU kernels in Triton using vector addition — the Hello World of GPU programming. Covers the core programming model, how it compares to CUDA, and the pattern every Triton kernel follows."
math: true
---

If you've ever wanted to write custom GPU kernels without diving deep into CUDA C++, [Triton](https://github.com/openai/triton) is the answer. Developed by OpenAI, Triton lets you write GPU kernels in Python — and it's what powers many of the custom attention and matmul kernels in production ML systems today.

This is Part 1 of a series. We'll start with the simplest possible kernel: **vector addition**.

---

## Why Triton?

Standard PyTorch is fast, but it can't fuse arbitrary operations together. Every `.relu()`, `.add()`, `.norm()` is a separate kernel launch — each one reads from and writes to GPU memory. For memory-bandwidth-bound workloads, this is wasteful.

Triton lets you write **fused kernels**: combine multiple ops into one pass over the data, cutting memory traffic dramatically. This is how FlashAttention, fast RoPE, and fused softmax kernels are built.

---

## CUDA vs. Triton Mental Model

Before writing code, the key conceptual shift:

| Concept | CUDA | Triton |
|---|---|---|
| Unit of work | Single thread | Block of threads (a *program*) |
| Indexing | `threadIdx`, `blockIdx` | `tl.program_id` + `tl.arange` |
| Memory access | Per-element | Vectorized load/store with masks |
| Thread management | You manage warps | Triton handles it |
| Language | C++ | Python |

In CUDA, you think about individual threads. In Triton, you think about **chunks of data** — each program instance owns a contiguous block and operates on it as a vector.

---

## The Kernel

```python
import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    a_ptr,   # pointer to input vector A
    b_ptr,   # pointer to input vector B
    c_ptr,   # pointer to output vector C
    N,       # total number of elements
    BLOCK_SIZE: tl.constexpr,  # elements per program instance
):
    # Which block am I?
    pid = tl.program_id(axis=0)

    # What indices do I own?
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask out-of-bounds accesses (N may not be divisible by BLOCK_SIZE)
    mask = offsets < N

    # Load from global memory
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # Compute and store
    tl.store(c_ptr + offsets, a + b, mask=mask)
```

---

## The Launcher

The kernel needs a Python wrapper to handle grid sizing and tensor validation:

```python
def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape
    assert a.is_cuda and b.is_cuda

    c = torch.empty_like(a)
    N = a.numel()
    BLOCK_SIZE = 1024

    # How many program instances to launch
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)
    return c
```

`triton.cdiv(N, BLOCK_SIZE)` is ceiling division — it ensures we launch enough programs to cover all elements even when N isn't a clean multiple of BLOCK_SIZE.

---

## Test It

```python
if __name__ == "__main__":
    N = 1 << 20  # 1M elements

    a = torch.ones(N, device="cuda", dtype=torch.float32)
    b = torch.full((N,), 2.0, device="cuda", dtype=torch.float32)

    c = vector_add(a, b)

    expected = a + b
    assert torch.allclose(c, expected)
    print(f"Correct! Max diff: {(c - expected).abs().max().item()}")
```

```bash
pip install triton torch
python vector_add_triton.py
# Correct! Max diff: 0.0
```

---

## The Universal Triton Pattern

Every Triton kernel — from vector add to FlashAttention — follows this same skeleton:

```python
pid = tl.program_id(0)                                 # which block am I?
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # what indices do I own?
mask = offsets < N                                     # bounds check
x = tl.load(ptr + offsets, mask=mask)                 # load my chunk
y = compute(x)                                         # do work
tl.store(out_ptr + offsets, y, mask=mask)             # write result
```

Internalize this and the more complex kernels become variations on a theme.

---

## What's Next

In Part 2, we'll add **autotuning** — letting Triton automatically search for the best `BLOCK_SIZE` — and benchmark our kernel against PyTorch's native addition to see how close we get.

After that: fused kernels (add + activation in one pass), and eventually the real goal — a custom attention kernel.
