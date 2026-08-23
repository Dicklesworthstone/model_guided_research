# Activation Checkpointing Strategy

**Bead:** `model_guided_research-1ra`
**Date:** 2026-06-14
**Scope:** doc only — when/how to add activation checkpointing to nanochat,
memory/compute tradeoffs, and the interactions that make it subtle here
(reversible blocks, symplectic double-backward, fixed-FLOPs budgeting,
goldens).

## Current state

- **Not implemented.** No `torch.utils.checkpoint` / `checkpoint_sequential`
  in either `nanochat/` or `bio_inspired_nanochat/` (grep-confirmed). Every
  block keeps its full forward activations for backward.
- **Reversible's memory win is ACTIVE (bead a6k3).** `ReversibleBlock.forward`
  routes grad-enabled, no-KV-cache forwards through `ReversibleFunction`
  (`nanochat/reversible_block_torch.py`), whose backward reconstructs
  `x2 = y2 - G(y1)` and re-runs G/F WITH grad — no F/G intermediate survives
  the step (only the block output is persisted; pinned by saved-tensor
  accounting in `tests/test_mathematical_properties.py::
  TestAdditiveReversibleWiring`). Decode/generation and no-grad eval stay
  eager; symplectic stays eager (its kicks are double-backward and do not
  compose with detach-and-recompute). `estimate_flops` charges the recompute
  (9 forward-equivalent units per block pair, 16 H Q T attention matmuls) —
  the same discipline as the 7lba symplectic correction.

Consequence: standard arms remain O(L) in activation memory; reversible
additive arms now persist only their block outputs (a6k3), so generic
activation checkpointing mainly helps standard/symplectic arms.

## The tradeoff in one line

Activation checkpointing trades **compute for memory**: drop a block's
intermediate activations on the forward and **recompute** them during backward.

| Strategy | Activation memory | Extra compute |
|---|---|---|
| none (today) | O(L) | 0 |
| **full** (checkpoint every block) | O(1) blocks in flight | +1 forward per block (~+33 % step time: fwd+bwd ≈ 1+2 → 2+2) |
| **√L** (checkpoint every √L-th block) | O(√L) | +1 forward total (~+33 % once, amortized) |
| reversible recompute (wire up `ReversibleFunction`) | O(1) | ~+1 forward, but **only valid for invertible blocks** |

For the research-scale models here (E1 rung L4/D128) memory is not the binding
constraint — this matters for the **scaling-sweep** rungs (large L / D / T,
beads `w94.1`/`lab.*`) and any GPU-memory-bound config.

## Recommended implementation (low-risk, opt-in)

Add a flag `--activation-checkpointing {none,full,every-k}` (default `none`,
behavior-preserving) and wrap each block call in the `GPT.forward` block loop:

```python
from torch.utils.checkpoint import checkpoint
# in the per-block loop (gpt.py forward):
if self.training and self.ckpt_mode == "full":
    x = checkpoint(block, x, cos_sin, kv_cache, use_reentrant=False)
else:
    x = block(x, cos_sin, kv_cache)
```

Why these choices:

- **`use_reentrant=False`** is the modern (PyTorch ≥2.x) path: composes with
  `torch.compile`/DDP, supports kwargs and no-grad inputs, and does not require
  inputs to be leaf tensors.
- **Training-only / no KV-cache.** Checkpointing is a *training* memory
  optimization; inference uses the KV-cache path (`engine.py`) and is left
  untouched. In training `kv_cache=None`, so the recompute is clean.
- **RoPE `cos_sin` is passed in**, not stored per block, so it is recompute-free
  and adds no memory.

## Interactions (the subtle part)

1. **Reversible blocks.** Generic checkpointing *works* on reversible blocks and
   would give them the memory benefit they currently lack — but the *right* fix
   is to wire `ReversibleFunction.apply` into `ReversibleBlock.forward` for true
   O(1) memory without a generic-checkpoint wrapper. Until that lands, `--
   activation-checkpointing full` is the pragmatic memory lever for reversible
   too. Filed as a follow-up (below).
2. **Symplectic reversible is already double-backward.** Each symplectic kick
   takes a first backward to form `grad(phi)` during the forward
   (`create_graph=True`, `reversible_block_torch.py:98`) and a second at
   training time — a 3× block-compute multiplier (bead `7lba`). Checkpointing a
   symplectic block recomputes that whole double-backward forward, compounding
   cost. Prefer `every-k` (not `full`) for symplectic, or skip checkpointing it.
3. **Fixed-FLOPs budgeting (must-read).** Checkpointing adds a recompute forward
   that `estimate_flops()` (`gpt.py:665`) does **not** model (it assumes one
   forward + one backward = 6N). If you enable checkpointing under
   `--target-flops`, the budget will train **too many steps** for the true
   compute — exactly the confound the symplectic 3× correction fixed. Two safe
   options: (a) extend `estimate_flops` to add ~`2·N_block` per checkpointed
   block, or (b) budget by `--max-steps` instead of `--target-flops` for any
   arm with checkpointing on. Keep checkpointing **off** for the canonical
   fixed-FLOPs A/B unless the accounting is updated.
4. **torch.compile.** `checkpoint(use_reentrant=False)` composes with
   `torch.compile`; expect graph breaks at the checkpoint boundary. Compile
   both arms or neither (cf. `docs/config_parity.md`).
5. **Goldens / numerics.** With `use_reentrant=False` and **no dropout** (none
   in `GPTConfig`), recompute is numerically identical to the eager forward, so
   loss trajectories match. The autograd *graph* differs, so re-capture the
   attention goldens if checkpointing ever becomes a default (it should not —
   default `none`). See `docs/new_mechanism_checklist.md` (goldens recapture).
6. **Profiling hooks (`b1l`).** Checkpointing changes the backward timeline
   (recompute shows up as extra forward kernels in backward). Any NVTX/
   `torch.profiler` ranges should tag the recompute region so it is not
   mistaken for the primary forward.

## Recommended defaults

| Regime | Setting |
|---|---|
| E1 rung / small models | `none` (memory is not binding) |
| Large-L/D/T scaling rungs, GPU memory-bound | `full` (standard) or `every-2` (symplectic) **with FLOPs accounting fixed or `--max-steps`** |
| Reversible memory recovery (interim) | `full` until `ReversibleFunction` is wired in |
| Canonical fixed-FLOPs A/B | `none` (keep `estimate_flops` honest) |

## Follow-ups

- **Implement `--activation-checkpointing`** (`gpt.py` block loop + `train.py`
  flag), default `none`. Low-risk, mechanism-agnostic. → filed.
- ~~**Wire `ReversibleFunction` into `ReversibleBlock.forward`**~~ **DONE
  (bead a6k3)** — additive training recomputes by default now; see Current
  state above.
- If either is enabled under `--target-flops`, update `estimate_flops` to count
  the recompute forward (same discipline as the 7lba symplectic correction).
