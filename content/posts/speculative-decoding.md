---
title: "Speculative Decoding: A Deep Dive into Every Major Algorithm"
date: 2026-07-31
draft: false
tags: ["LLM", "inference", "speculative-decoding", "optimization", "DeepSeek", "EAGLE", "Medusa"]
summary: "A comprehensive guide to speculative decoding algorithms — from the foundational rejection-sampling approach to DeepSeek's production-deployed DSpark. Covers Vanilla SpecDec, Medusa, EAGLE, Lookahead, DFlash, DSpark, and more."
math: true
---

Speculative decoding is a family of techniques that accelerate autoregressive LLM inference without changing output quality. The core insight: **draft cheap tokens speculatively, then verify them in parallel with the expensive target model.**

Autoregressive generation is memory-bandwidth-bound at small batch sizes — the GPU spends most of its time loading model weights, not computing. Speculative decoding exploits this by batching multiple verification steps into a single forward pass, converting wasted memory bandwidth into useful token generation.

## 1. Vanilla Speculative Decoding

*[Leviathan et al., 2023](https://arxiv.org/abs/2211.17192); [Chen et al., 2023](https://arxiv.org/abs/2302.01318)*

The foundational algorithm. Two papers independently proposed the same idea.

**Setup:** A small "draft" model $M_q$ and a large "target" model $M_p$.

**Algorithm:**
1. Draft model generates $\gamma$ tokens autoregressively (cheap)
2. Target model scores all $\gamma$ drafted tokens in a single forward pass (parallel)
3. Accept/reject each token left-to-right using a rejection sampling scheme:
   - Accept token $x_i$ with probability $\min\left(1, \frac{p(x_i)}{q(x_i)}\right)$
   - On rejection, resample from an adjusted distribution $\text{norm}(\max(0, p(x) - q(x)))$
4. Always generate at least 1 token (the resampled one if the first draft is rejected), up to $\gamma + 1$

**Key property:** The output distribution is *identical* to sampling from $M_p$ alone — lossless acceleration.

**Speedup:** Typically 2-3x for well-matched draft/target pairs. Governed by the *acceptance rate* $\alpha$ and the ratio of draft/target latency.

**Expected tokens per step:** $\frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$

## 2. Medusa

*[Cai et al., 2024](https://arxiv.org/abs/2401.10774)*

Eliminates the separate draft model entirely by adding **multiple decoding heads** to the target model itself.

**Architecture:**
- Add $K$ extra prediction heads on top of the target model's hidden states
- Head $k$ predicts the token at position $t + k$ (looking ahead)
- Each head is a lightweight MLP (1-2 layers)

**Decoding:**
1. Run one forward pass through the base model
2. Each head proposes top-$s$ candidates for its position
3. Form a **tree** of candidate continuations (Cartesian product, pruned)
4. Verify the entire tree in one forward pass using a tree attention mask
5. Accept the longest prefix that matches the target distribution

**Medusa-1:** Heads are trained with frozen backbone (fast to train, slightly lower acceptance).

**Medusa-2:** Heads + backbone are jointly fine-tuned. Uses a combined acceptance scheme that preserves the target distribution.

**Advantages:** No separate draft model to serve; heads add <1% parameters. Works well for batch-size-1, memory-bound regimes.

## 3. EAGLE & EAGLE-2

*Li et al., 2024 ([EAGLE](https://arxiv.org/abs/2401.15077); [EAGLE-2](https://arxiv.org/abs/2406.16858))*

**Insight:** Predicting at the *feature level* (hidden states) is easier than at the token level.

**EAGLE-1:**
- Train a lightweight autoregressive head that predicts the **next hidden state** given the current hidden state + token embedding
- Use these predicted hidden states to draft tokens through the original LM head
- Verify with tree attention (like Medusa)

**EAGLE-2:**
- Adds **context-aware dynamic draft trees** — the tree structure adapts based on confidence scores from the draft model
- Expands high-confidence branches more aggressively, prunes low-confidence ones
- Achieves higher acceptance rates with fewer total candidates

**Performance:** 3-4x speedup on code/math tasks. One of the first methods to significantly outperform vanilla speculative decoding.

## 4. EAGLE-3

*[Li et al., 2025](https://arxiv.org/abs/2503.01840)*

**Key departure:** Abandons feature prediction in favor of **direct token prediction** with multi-layer feature fusion.

EAGLE-1/2 reused top-layer features for autoregression, but this created a scaling bottleneck — more training data gave diminishing returns due to feature prediction constraints. EAGLE-3 resolves this with a technique called "training-time test" that fuses features from multiple target model layers, enabling the draft model to fully benefit from scaled training data.

**Performance:** Up to 6.5x speedup (approximately 1.4x improvement over EAGLE-2). At batch size 64 in SGLang, achieves 1.38x throughput improvement.

## 5. Multi-Token Prediction (MTP)

*Meta ([Gloeckle et al., 2024](https://arxiv.org/abs/2404.19737)); [DeepSeek-V3 variant (Dec 2024)](https://arxiv.org/abs/2412.19437)*

MTP was originally introduced by **Meta** as a training objective: train $D$ independent parallel heads, each predicting a different future token position. The primary motivation was improved representations and training efficiency — speculative decoding was a bonus.

**DeepSeek-V3's sequential variant:**
- Depth $D=1$ (one extra token predicted beyond next-token)
- Unlike Meta's parallel independent heads, DeepSeek **sequentially predicts** the additional token, maintaining a **complete causal chain** at each depth
- Each MTP module contains: a shared embedding layer, a shared output head (both shared with the main model), a dedicated Transformer block, and a projection matrix

**Inference usage:**
- MTP modules can be discarded entirely — the main model functions independently
- Or repurposed as a built-in draft head for speculative decoding at zero additional memory cost

**Who uses MTP:** DeepSeek-V3/R1 and Qwen3/Qwen3.6 ship with MTP heads. Models trained without MTP (Gemma, Llama) require an external drafter.

**Significance:** Demonstrated that training objectives designed for capability improvement can double as speculative decoding infrastructure — a "two birds, one stone" approach.

## 6. Lookahead Decoding

*[Fu et al., 2024](https://arxiv.org/abs/2402.02057)*

A **draft-model-free** approach using Jacobi iteration.

**Core idea:** Treat $n$-token generation as solving a system of equations via Jacobi fixed-point iteration. All positions are updated simultaneously until convergence.

**Two parallel branches per step:**
1. **Lookahead branch:** Maintains $W$ parallel n-gram windows, each being iteratively refined via Jacobi steps
2. **Verification branch:** Previously "converged" n-grams are collected into a pool. At each step, verify if any cached n-gram matches the actual continuation.

**Properties:**
- No draft model, no extra parameters, no fine-tuning
- Uses only the target model
- Leverages the observation that Jacobi iteration on LLMs often converges in few steps for "easy" tokens
- Speedup is task-dependent — works best when there are predictable patterns (code, structured text)

**Typical speedup:** 1.5-2.5x.

## 7. DistillSpec

*[Zhou et al., 2024](https://arxiv.org/abs/2310.08461)*

**Idea:** Align the draft model's distribution to the target model's via knowledge distillation, directly maximizing acceptance rate.

**Method:**
- On-policy distillation: generate sequences from the draft, score with target, train draft to minimize divergence
- Specifically optimizes the acceptance probability $\mathbb{E}\left[\min\left(1, \frac{p}{q}\right)\right]$

**Result:** A draft model that is better aligned produces higher acceptance rates, directly translating to more tokens per verification step. Typically 2.5-3.5x speedup.

## 8. SpecInfer / Tree-Based Speculative Inference

*[Miao et al., 2024](https://arxiv.org/abs/2305.09781)*

**Key contribution:** Instead of a single linear draft, maintain a **tree of speculative candidates**.

**Method:**
- Multiple small models (or one model with diverse sampling) each propose different continuations
- Merge proposals into a token tree
- Verify the entire tree in one pass using **tree attention** (a masked attention pattern)
- Accept the longest valid path through the tree

**Token tree attention:** A generalized attention mask where each position attends to its ancestors in the tree, not just the linear prefix. This allows verifying exponentially many sequences in one forward pass.

## 9. Sequoia

*[Chen et al., 2024](https://arxiv.org/abs/2402.12374)*

**Focus:** Optimal tree structure for speculative decoding.

**Contribution:**
- Derives the theoretically optimal tree topology given a compute budget and token acceptance probabilities
- Uses dynamic programming to construct trees that maximize expected accepted tokens
- Adapts tree shape based on observed acceptance statistics at runtime
- Handles hardware-aware constraints (batch size limits, memory)

**Performance:** 4-5x speedup on A100 for Llama-2-70B with a 7B draft, outperforming fixed tree structures.

## 10. REST — Retrieval-Based Speculative Decoding

*[He et al., 2023](https://arxiv.org/abs/2311.08252)*

**Draft source:** Instead of a neural draft model, retrieve draft tokens from a **datastore** (corpus, code repository, etc.).

**Method:**
1. Maintain a suffix array or n-gram index over a reference corpus
2. Given the current context, retrieve matching suffixes as draft candidates
3. Verify retrieved continuations with the target model

**Sweet spot:** Highly repetitive domains — code completion, templated text, documentation — where retrieval often produces exact matches. 2-4x speedup depending on domain repetitiveness.

## 11. CLLMs — Consistency Large Language Models

*[Kou et al., 2024](https://arxiv.org/abs/2403.00835)*

**Inspired by:** Consistency models in diffusion.

**Method:**
- Fine-tune a model to map any point along a Jacobi trajectory directly to the fixed point
- In one forward pass, the model predicts what the converged sequence would look like (multiple tokens at once)
- The fine-tuning objective enforces that the model's single-step output matches the multi-step Jacobi result

**Effectively:** Teaches the model to "skip ahead" in the Jacobi iteration, producing 2-3 correct tokens in a single pass without separate verification. Approximate (not perfectly lossless).

## 12. Self-Speculative Decoding

Several methods use the **target model itself** to draft, avoiding a separate model entirely:

- **Self-Speculative Decoding ([Zhang et al., 2024](https://arxiv.org/abs/2309.08168)):** Skip certain layers during drafting (early exit), then verify with full depth.
- **LayerSkip ([Elhoushi et al., 2024](https://arxiv.org/abs/2404.16710)):** Train with early-exit loss at intermediate layers; at inference, use early layers for drafting, full model for verification.
- **SPEED ([Hooper et al., 2024](https://arxiv.org/abs/2310.12072)):** Use a subset of attention heads as a "thin" draft model.

**Advantage:** Zero additional memory for a draft model — crucial when the target model already saturates GPU memory. Typical speedup: 1.5-2.5x.

## 13. Online Speculative Decoding

*[Liu et al., 2024](https://arxiv.org/abs/2310.07177)*

Addresses **distribution shift** — the draft model was trained on a general corpus but inference happens on a specific domain.

**Approach:**
- Continuously fine-tune the draft model on tokens accepted by the target during inference
- Uses a replay buffer of recently verified sequences
- Adapts the draft model on-the-fly to the current query distribution

**Benefit:** Maintains high acceptance rates even on out-of-distribution inputs where a static draft model would degrade.

## 14. DFlash

*DeepSeek, 2025 ([arXiv:2602.06036](https://arxiv.org/abs/2602.06036))*

**Core idea:** Replace autoregressive drafting with a **lightweight block diffusion model** that generates an entire block of draft tokens in a single parallel forward pass.

**Architecture:**
- **Draft model:** 5 Transformer layers (8 for Qwen3-Coder)
- **Block size:** 16 tokens generated simultaneously (10 for LLaMA-3.1)
- **Conditioning:** Extracts hidden representations from 5 uniformly-sampled layers of the frozen target model, injects them into the KV cache of every draft layer via a lightweight projection
- **Shared components:** Embedding layer and LM head frozen and shared with target model

**How it works:**
1. Given a context (including the last verified token as "anchor"), mask the next $\gamma$ positions
2. In a single forward pass, the block diffusion model denoises all masked positions simultaneously — no iterative multi-step denoising
3. Target model verifies all $\gamma$ drafted tokens in parallel
4. On mismatch, stop and use the target's correction as the next anchor

**Training innovations:**
- Random anchor sampling with block masking (matching inference behavior)
- Sparse bidirectional attention within blocks, causal attention to context
- Exponential loss weighting: $w_k = \exp(-(k-1)/\gamma)$ — early positions receive higher weight since errors cascade
- ~800K training samples

**Performance (Qwen3-8B):**

| Setting | Speedup | Avg. Accepted Tokens |
|---------|---------|---------------------|
| Offline (T=0) | 4.86x | 6.49 |
| SGLang (Math500) | 5.1x | 8.01 |
| vs. EAGLE-3 | **2.5x faster** | ~2x more tokens accepted |

**Key insight:** Parallel drafting fundamentally changes the latency equation — $t_{\text{parallel}} \ll \gamma \cdot t_{\text{step}}$ — because draft cost is $O(1)$ in block size rather than $O(\gamma)$.

## 15. DSpark

*DeepSeek, 2025 ([arXiv:2607.05147](https://arxiv.org/abs/2607.05147)) — deployed in production on DeepSeek-V4*

**Core idea:** A **semi-autoregressive (SAR)** drafter that combines DFlash's parallel backbone with a lightweight sequential module, plus a **confidence-aware adaptive scheduler** for high-concurrency serving.

### The Problem with Pure Parallel Drafting

Pure parallel drafters like DFlash suffer from **suffix decay**: each position marginalizes over all possible predecessor tokens, causing acceptance rates to degrade at later positions (e.g., 0.87→0.78 across a block). Given context "of [\_]", a parallel drafter might average over "of course" and "no problem," producing incoherent combinations.

### Semi-Autoregressive Architecture

DSpark solves this with a two-stage design:

**Stage 1 — Parallel Backbone (from DFlash):**
- 5 Transformer layers with 3 MoE layers, 128-token sliding window
- Generates hidden states and base logits $U_1, \ldots, U_\gamma$ for all positions in one pass

**Stage 2 — Sequential Module (Markov Head or RNN Head):**
- *Markov Head (default):* First-order transition bias with low-rank factorization ($r=256$), conditioning position $k$ only on the sampled token at $k-1$
- *RNN Head:* Gated recurrent state accumulating full prefix history

Combined factorized distribution: $\prod_{k=1}^{\gamma} p_k(x_k \mid x_0, x_{\lt k})$

This preserves the parallel backbone's speed while injecting just enough sequential coherence to eliminate suffix decay.

### Confidence-Scheduled Verification

The second innovation targets **production serving under load**. In high-concurrency environments, verifying all speculated tokens wastes batch capacity when many will be rejected.

**Mechanism:**
- A **confidence head** outputs $c_k \in (0,1)$ estimating per-position acceptance probability
- **Prefix survival probability:** $a_{r,j} = \prod_{i \leq j} c_{r,i}$
- **Hardware-aware scheduler** maximizes system throughput $\Theta = \tau \cdot \text{SPS}(B)$ by dynamically truncating verification length based on survival estimates and current batch size $B$
- Under heavy load: verification budget shrinks from 4-6 tokens to 1-2
- Under light load: full verification window is used

### Training

Three-term loss function:
- **Cross-entropy** on ground truth tokens
- **Total variation distance:** $\|p_k^d - p_k^t\|_1$ (aligns draft to target distribution)
- **Confidence calibration** via binary cross-entropy on soft labels
- Position weighting: $w_k = \exp(-(k-1)/\gamma)$

Data: 1.3M samples (chat 17.6%, math 39.4%, code 38.9%, instruction 4.1%), 10 epochs.

### Performance

| Setting | Result |
|---------|--------|
| vs. EAGLE-3 (offline) | +30.9% accepted length |
| vs. DFlash (offline) | +16.3% accepted length |
| Estimated offline speedup | ~5.7-7.1x |
| DeepSeek-V4-Flash (live, matched throughput) | 60-85% faster per-user |
| DeepSeek-V4-Pro (live, matched throughput) | 57-78% faster per-user |

DSpark is currently deployed in production on DeepSeek-V4, shifting the Pareto frontier of latency vs. throughput.

## Comparison

| Method | Draft Source | Extra Params | Lossless | Typical Speedup |
|--------|-------------|-------------|----------|-----------------|
| Vanilla SpecDec | Separate small model | Full draft model | Yes | 2-3x |
| Medusa | Multi-head on target | ~0.5-1% | Medusa-2: Yes | 2-3x |
| EAGLE-2 | Feature-level AR head | ~1-2% | Yes | 3-4x |
| EAGLE-3 | Multi-layer fusion | ~1-2% | Yes | up to 6.5x |
| MTP | Built-in training head | 0% | Yes | 1.5-2x |
| Lookahead | Jacobi iteration (self) | None | Yes | 1.5-2.5x |
| DistillSpec | Distilled draft model | Full draft model | Yes | 2.5-3.5x |
| REST | Retrieval datastore | None (just index) | Yes | 2-4x |
| Sequoia | Optimal tree + draft | Full draft model | Yes | 4-5x |
| Self-Speculation | Target early layers | None | Yes | 1.5-2.5x |
| CLLMs | Consistency fine-tune | None (retrained) | Approx. | 2-3x |
| **DFlash** | Block diffusion | 5-layer draft | Yes | **4.9-6.1x** |
| **DSpark** | Semi-AR + confidence sched. | 5-layer draft + head | Yes | **~5.7-7.1x** |

All speedup figures are offline, single-request measurements unless otherwise noted. DSpark's additional advantage manifests under production concurrency where its confidence scheduler avoids wasting batch capacity.

## When to Use What

**Memory-constrained, single model only:** Medusa, EAGLE, Lookahead, Self-Speculation, or MTP (if trained with it).

**Have budget for a draft model:** Vanilla + DistillSpec for simplicity, Sequoia for optimal tree structure.

**Repetitive/structured domain:** REST (retrieval-based) excels on code completion and templated text.

**Maximum offline speedup:** DFlash or DSpark with a trained draft model.

**Production serving under load:** DSpark — its confidence scheduler is specifically designed to maintain speedup as concurrency increases, where all other methods degrade.

**No fine-tuning, drop-in:** Vanilla SpecDec with an off-the-shelf smaller model, or Lookahead (zero setup).

**Models with built-in MTP heads (Qwen3, DeepSeek):** Start with the free MTP draft head. Upgrade to DFlash/DSpark for higher speedup.

## The Evolution

The field has progressed through several eras:

1. **Rejection sampling** (2023): Vanilla speculative decoding proves the concept — lossless acceleration via draft-then-verify
2. **Self-drafting** (2023-2024): Medusa, EAGLE, Self-Speculation eliminate the separate draft model
3. **Training-integrated** (2024): MTP bakes draft capability into the training objective itself
4. **Parallel drafting** (2025): DFlash breaks the autoregressive drafting bottleneck with block diffusion
5. **Production-aware** (2025): DSpark adds sequential coherence and adaptive scheduling for real-world serving

The current frontier combines parallel generation (for speed), sequential corrections (for coherence), and system-aware scheduling (for throughput) — DFlash and DSpark represent the state of the art as of mid-2025.
