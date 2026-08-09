---
title: "How PyTorch Works: The Core Framework"
date: 2023-04-02
draft: false
tags: ["pytorch", "deep-learning", "python", "neural-networks"]
summary: "A mental model for PyTorch's core abstractions — tensors, autograd, nn.Module, optimizers, and DataLoader — and how they compose into a training loop."
math: false
---

PyTorch is built around a few core abstractions that compose cleanly. Once you internalize each piece, the whole framework clicks into place.

---

## Tensors — The Foundation

Everything is a `torch.Tensor` — an n-dimensional array that lives on CPU or GPU and optionally tracks gradients.

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
y.sum().backward()
print(x.grad)  # tensor([2., 4., 6.])
```

Every operation on a tensor builds a **computation graph** behind the scenes. `.backward()` walks that graph in reverse to compute gradients via the chain rule — this is called **autograd**.

---

## Autograd — Automatic Differentiation

PyTorch uses **define-by-run** (dynamic) autograd. The graph is built as you execute code, not ahead of time. This means you can use normal Python control flow — `if`, `for`, `while` — and PyTorch will differentiate through it correctly.

```python
x = torch.randn(3, requires_grad=True)

# Real Python control flow — autograd handles it fine
if x.sum() > 0:
    y = x * 2
else:
    y = x * -1

y.sum().backward()
print(x.grad)
```

Each operation records itself as a node in the graph. `.backward()` traverses from output back to inputs, multiplying Jacobians (chain rule) and accumulating `.grad` on each leaf tensor.

**Key rule:** only leaf tensors with `requires_grad=True` accumulate gradients. Model parameters are leaf tensors.

---

## `nn.Module` — The Building Block

`nn.Module` is the base class for every layer and model in PyTorch. It does three things:

1. **Holds parameters** — `nn.Parameter` tensors that get updated during training
2. **Holds submodules** — nested `nn.Module`s, registered automatically
3. **Defines the forward pass** — you override `forward()`

```python
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # nn.Parameter: tracked by the module, included in .parameters()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.T + self.bias
```

You call a module like a function — `model(x)` invokes `model.forward(x)` via `__call__`, which also fires any registered hooks.

### Submodules are auto-registered

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)   # registered automatically
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = MLP()
print(list(model.parameters()))  # all params from fc1 AND fc2
```

Because submodules are registered, methods like `.parameters()`, `.to("cuda")`, and `.state_dict()` recurse through the entire tree automatically.

---

## Built-in Layers and `nn.Sequential`

PyTorch ships dozens of ready-to-use modules:

| Module | What it does |
|---|---|
| `nn.Linear(in, out)` | Fully connected layer |
| `nn.Conv2d(in, out, k)` | 2D convolution |
| `nn.MultiheadAttention` | Transformer attention |
| `nn.LayerNorm` | Layer normalization |
| `nn.Embedding(vocab, dim)` | Token embedding lookup |
| `nn.Dropout(p)` | Dropout regularization |

For simple sequential architectures, skip the custom class:

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10),
)
```

---

## Optimizer — Updating Parameters

The optimizer takes `model.parameters()` and applies an update rule after each backward pass:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

Common optimizers: `SGD`, `Adam`, `AdamW` (Adam + weight decay, preferred for transformers).

PyTorch **accumulates** gradients by default, so you must zero them manually each step:

```python
optimizer.zero_grad()   # clear .grad from last step
loss.backward()         # fill .grad on all params
optimizer.step()        # apply update: w ← w - lr * grad
```

This default accumulation is intentional — it enables **gradient accumulation** (simulating larger batch sizes) by simply skipping `zero_grad()` for N steps before calling `optimizer.step()`.

---

## DataLoader — Batching and Loading

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data   = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = MyDataset(data, labels)
loader  = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

`DataLoader` handles batching, shuffling, parallel data loading, and collation. `num_workers` spawns background processes to load data while the GPU is busy training — essential for avoiding I/O bottlenecks.

---

## The Training Loop

Everything comes together in the training loop:

```python
model     = MLP().to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()  # enable dropout, batchnorm training behavior

    for x, y in loader:
        x, y = x.to("cuda"), y.to("cuda")

        optimizer.zero_grad()        # 1. clear old gradients
        pred = model(x)              # 2. forward pass
        loss = criterion(pred, y)    # 3. compute loss
        loss.backward()              # 4. backward pass → fills .grad
        optimizer.step()             # 5. update parameters

    # Validation
    model.eval()  # disable dropout, use running stats for batchnorm
    with torch.no_grad():  # don't build graph, saves memory
        for x, y in val_loader:
            ...
```

`model.train()` and `model.eval()` toggle behavior of layers like `Dropout` and `BatchNorm` that act differently during training vs. inference. `torch.no_grad()` skips building the autograd graph during inference — faster and uses less memory.

---

## Saving and Loading

```python
# Save
torch.save(model.state_dict(), "model.pt")

# Load
model = MLP()
model.load_state_dict(torch.load("model.pt"))
model.eval()
```

`state_dict()` is a plain dict of parameter names → tensors. It's the standard way to checkpoint models.

---

## The Big Picture

```
DataLoader
    │  batches
    ▼
nn.Module.forward()  ──── nn.Parameter (weights)
    │  predictions
    ▼
Loss Function
    │
    ▼
.backward()  ──── fills .grad on all parameters
    │
    ▼
Optimizer.step()  ──── w = w - lr * grad
    │
    └──── repeat
```

The entire framework is one recursive abstraction: `nn.Module` contains `nn.Module`s. `.parameters()` collects leaves. `.to()` moves everything. `.state_dict()` snapshots everything. Once you understand one module, you understand the whole model.
