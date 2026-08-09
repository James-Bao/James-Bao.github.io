---
title: "Learning Triton Part 2: Autotuning and Benchmarking"
date: 2023-09-01
draft: false
tags: ["triton", "cuda", "gpu", "kernel", "deep-learning", "autotuning", "benchmarking"]
summary: "How to use Triton's built-in autotuner to find optimal kernel configurations, and how to benchmark Triton kernels against PyTorch using triton.testing.Benchmark."
---

In [Part 1](/posts/triton-tutorial-part1) we wrote a correct vector addition kernel with a hardcoded `BLOCK_SIZE=1024`. But how do we know 1024 is optimal? Different GPUs, different data sizes, and different memory access patterns all affect the best choice.

This is Part 2: **autotuning and benchmarking** — letting Triton find the best configuration automatically, and measuring how we compare to native PyTorch.

---

## The Problem with Hardcoding

GPU performance is sensitive to:

- **Block size** — affects occupancy and memory coalescing
- **Number of warps** — how many warps Triton uses per program
- **Pipeline stages** — how aggressively to prefetch memory

The right values depend on your GPU architecture and workload. Instead of guessing, Triton can search for you.

---

## Adding `@triton.autotune`

Wrap your kernel with `@triton.autotune` and provide a list of configs to try:

```python
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["N"],  # re-tune when N changes
)
@triton.jit
def vector_add_kernel(
    a_ptr, b_ptr, c_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    tl.store(c_ptr + offsets, a + b, mask=mask)


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty_like(a)
    N = a.numel()
    # Grid is a lambda now — BLOCK_SIZE comes from the autotuner
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    vector_add_kernel[grid](a, b, c, N)
    return c
```

The key changes:
- `@triton.autotune` wraps the kernel and runs each config, keeping the fastest
- `key=["N"]` tells Triton to re-run the search when input size changes
- `grid` becomes a **lambda** that reads `BLOCK_SIZE` from the winning config at runtime

The first call is slow (benchmarking all configs), but every subsequent call with the same `N` uses the cached best config.

---

## Benchmarking with `triton.testing.Benchmark`

Now let's measure how our kernel compares to PyTorch's native `a + b`:

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],                          # x-axis: sequence length
        x_vals=[2**i for i in range(12, 28)],   # 4K to 128M elements
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="Vector Addition Throughput",
        args={},
    )
)
def benchmark(N, provider):
    a = torch.rand(N, device="cuda", dtype=torch.float32)
    b = torch.rand(N, device="cuda", dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vector_add(a, b), quantiles=quantiles
        )
    elif provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: a + b, quantiles=quantiles
        )

    # Convert ms → GB/s: we read 2 vectors and write 1 (3 * N * 4 bytes)
    gbps = lambda ms: 3 * N * a.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True)
```

`triton.testing.do_bench` handles warmup runs and timing automatically. The result is a table + plot of **memory bandwidth (GB/s)** across input sizes.

---

## What to Expect

Vector addition is purely memory-bandwidth-bound — the compute (one add) is trivial compared to the time spent reading and writing memory. A well-tuned kernel should approach the GPU's **peak memory bandwidth**.

On an A100 (2 TB/s peak):
- PyTorch: ~1.8–1.9 TB/s (it's already well optimized)
- Triton: ~1.8–1.9 TB/s (matches PyTorch after autotuning)

For simple ops like this, Triton reaches parity. The real wins come with **fused kernels** — which we'll cover in Part 3.

---

## Understanding `num_warps`

A warp is 32 threads executing in lockstep. `num_warps` controls how many warps Triton schedules per program block:

- More warps → better latency hiding when waiting on memory
- Too many warps → register pressure, reduced occupancy

Triton handles the low-level warp scheduling, but `num_warps` is a hint you can tune. The autotuner will find what works best for your GPU.

---

## Summary

| Feature | What it does |
|---|---|
| `@triton.autotune` | Tries all configs, caches the winner per input shape |
| `key=["N"]` | Invalidates cache when input size changes |
| `grid = lambda meta: ...` | Reads winning config at launch time |
| `triton.testing.do_bench` | Warmup + timed benchmark |
| `triton.testing.perf_report` | Plots throughput across input sizes |

---

## What's Next

Part 3 covers **fused kernels** — combining multiple operations (e.g., add + ReLU, or add + LayerNorm) into a single kernel pass. This is where Triton starts beating PyTorch meaningfully, by eliminating redundant memory reads and writes.
