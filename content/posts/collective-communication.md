---
title: "Collective Communication Primitives for Distributed Computing"
date: 2026-07-29
draft: false
tags: ["distributed-computing", "HPC", "collective-communication", "distributed-training", "distributed-inference"]
summary: "A visual guide to collective communication operations — AllGather, ReduceScatter, AllReduce, All-to-All, and Permute — the building blocks of distributed training and inference on modern accelerator clusters."
---

Collectives are a set of distributed computing primitives with simple semantics, originally developed in High-Performance Computing (HPC). They coordinate data exchange among multiple processes (called **ranks**) participating in a group operation, as opposed to point-to-point communication which involves only a sender and a receiver.

These primitives form the backbone of distributed workloads — from gradient aggregation in training to request distribution in inference.

## Applications

### Distributed Training

Collective communication aggregates and synchronizes gradients across workers to maintain model consistency. During data-parallel training, each worker computes gradients on its local shard of data, and collectives like AllReduce ensure every worker ends up with the same averaged gradient before the optimizer step.

### Distributed Inference

Collective communication distributes requests across multiple accelerators in serving nodes, optimizing resource utilization and maintaining low latency under high loads. For tensor-parallel inference, operations like AllGather and ReduceScatter split model layers across devices and reassemble activations at each layer boundary.

## Collective Operations

### AllGather

Each rank shares its tensor and receives the aggregated tensors from all ranks, ordered by rank index. The result is that every rank ends up with the full concatenated data from all participants.

![AllGather operation showing each rank contributing its piece and all ranks receiving the full concatenated result](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_images/all-gather.gif)

**Use case:** In FSDP, each device only stores a shard of the model parameters. Before the forward pass, AllGather reconstructs the full parameter tensors from all shards so computation can proceed.

### ReduceScatter

Performs reductions on input data (for example, sum, min, max) across ranks, with each rank receiving an equal-sized block of the result based on its rank index. Unlike AllReduce, the reduced result is *distributed* — each rank gets only its portion.

![ReduceScatter operation showing data being reduced across ranks and scattered into equal portions](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_images/reduce-scatter.gif)

**Use case:** In FSDP, after the backward pass produces full gradients, ReduceScatter simultaneously reduces (sums) gradients across ranks and shards the result — each rank receives only the gradient slice corresponding to its owned parameter shard.

### AllReduce

Performs reductions on data (e.g., sum, max, min) across ranks and stores the result in the output buffer of every rank. This is equivalent to a ReduceScatter followed by an AllGather.

![AllReduce operation showing data being reduced and the full result stored on every rank](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_images/all-reduce.gif)

**Use case:** The classic gradient synchronization primitive in standard Data Parallelism (DDP) — every worker sums its local gradients and ends up with the identical result, so all replicas stay in sync without sharding.

### All-to-All

Each rank sends different data to and receives different data from every other rank, resembling a distributed transpose. Unlike AllGather where every rank contributes the same data to all, All-to-All sends *distinct* payloads to each destination.

![All-to-All operation showing each rank exchanging unique data with every other rank](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_images/all-to-all.gif)

**Use case:** Expert-parallel MoE (Mixture of Experts) models use All-to-All to route tokens to their assigned experts and collect results back, since different tokens go to different experts.

### Permute

Each rank sends its data to a designated destination rank and receives data from a designated source rank, according to a set of source-target pairs. These pairs typically form ring topologies that map to direct physical connectivity for minimum latency.

![Permute operation showing directed data movement along source-target pairs in a ring topology](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_images/permute.gif)

**Use case:** Pipeline-parallel training uses permute-style communication to pass activations forward and gradients backward along a fixed sequence of pipeline stages.

## Communication Scope

### Intra-node

Communication within a single node or across directly interconnected devices. This leverages high-bandwidth, low-latency links (such as NVLink, proprietary device interconnects, or PCIe) and delivers the best performance for small-world communication patterns.

### Inter-node

Communication that traverses the network fabric (e.g., Ethernet, InfiniBand, EFA) across physically separate machines. Higher latency than intra-node, but essential for scaling beyond the device count of a single server. Modern clusters use high-bandwidth network interfaces (100–400 Gbps per link) and topology-aware collective algorithms to minimize this overhead.

## FSDP: Collectives in Action

Fully Sharded Data Parallelism (FSDP) is a memory-efficient training strategy that shards model parameters, gradients, and optimizer states across all ranks. Unlike standard DDP which replicates the full model on every device, FSDP only materializes full parameters on-demand — and collectives are the mechanism that makes this work.

### The FSDP Lifecycle

```
┌─────────────────────────────────────────────────────────┐
│  Each rank stores only 1/N of parameters at rest        │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Forward pass: AllGather reconstructs full parameters    │
│  (each rank contributes its shard → all receive whole)  │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Compute forward on full parameters, then discard them  │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Backward pass: AllGather again (need full params for   │
│  gradient computation), compute gradients               │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  ReduceScatter: sum gradients across ranks AND shard    │
│  the result — each rank gets only its gradient slice    │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Each rank updates only its 1/N parameter shard         │
│  using its 1/N gradient shard + 1/N optimizer state     │
└─────────────────────────────────────────────────────────┘
```

### Why ReduceScatter + AllGather instead of AllReduce?

In DDP, AllReduce is natural: every rank holds the full model, so every rank needs the full gradient. But in FSDP, each rank only *owns* 1/N of the parameters, so it only *needs* 1/N of the gradient. ReduceScatter gives each rank exactly its slice — no wasted memory storing gradients it won't use.

The AllGather in the forward pass is the complementary half: parameters are sharded at rest, but computation requires the full layer, so AllGather temporarily reconstructs it. Together they form a memory-communication tradeoff:

| Strategy | Memory per rank | Communication |
|----------|----------------|---------------|
| DDP (AllReduce) | Full model + full gradients + full optimizer | AllReduce on gradients |
| FSDP (AllGather + ReduceScatter) | 1/N params + 1/N gradients + 1/N optimizer | AllGather params (2×) + ReduceScatter gradients |

FSDP trades more communication volume for dramatically lower memory, enabling models that wouldn't fit on a single device even with gradient checkpointing.

## Choosing the Right Collective

| Scenario | Collective | Why |
|----------|-----------|-----|
| Gradient sync (DDP) | AllReduce | Every rank holds the full model and needs the full reduced gradient |
| Reconstruct sharded parameters (FSDP forward/backward) | AllGather | Each rank contributes its shard; all receive the full tensor |
| Shard gradients after backward (FSDP) | ReduceScatter | Sum gradients across ranks and distribute — each rank gets only its slice |
| MoE token routing | All-to-All | Each token goes to a different expert; distinct payloads per destination |
| Pipeline stage handoff | Permute | Fixed source→destination along the pipeline |

## Summary

Collective communication primitives are the lingua franca of distributed deep learning. Understanding when and why to use each operation — and how they map to your parallelism strategy and hardware topology — is key to building efficient training and inference systems at scale.
