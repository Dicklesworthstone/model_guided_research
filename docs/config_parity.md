# Optimizer / Scheduler / Compile Config Parity vs `bio_inspired_nanochat`

**Bead:** `model_guided_research-wyf`
**Audit date:** 2026-06-14
**Compared trees:**
- `model_guided_research` @ `main` — driver `nanochat/train.py`
- `bio_inspired_nanochat` @ `e09ad46` — driver `scripts/base_train.py`

**Bottom line:** the *optimizer math* is at parity (same AdamW+Muon split,
same LR defaults, same betas/eps, same ∝1/√d scaling, same tf32/bf16 policy).
The benchmark-relevant divergences are in **training-loop policy**, not the
model: ours disables grad-clipping by default while bio clips at 1.0, and ours
runs a **flat constant LR** under the default `--scheduler-type none` while bio
applies a **WSD (warmup→stable→warmdown) schedule** with a 20 % linear
warmdown tail and a Muon-momentum warmup. These don't break a *within-repo*
A/B (all 11 mechanisms share the same loop), but they (a) need to be matched
explicitly for any comparison to bio's numbers and (b) likely leave final loss
on the table for *every* mechanism here. Concrete low-risk flags are in
[§4](#4-action-items--recommended-benchmark-flags).

---

## 1. Optimizer — at parity (math identical)

`nanochat/gpt.py:711 setup_optimizers` vs `bio_inspired_nanochat/gpt.py` `setup_optimizers`:

| Knob | This repo | bio_inspired | Verdict |
|---|---|---|---|
| Param split | AdamW for embedding+lm_head, **Muon** for ≥2-D matrices | identical | ✅ |
| `unembedding_lr / embedding_lr / matrix_lr` defaults | `0.004 / 0.2 / 0.02` (`gpt.py:711`) | `0.004 / 0.2 / 0.02` | ✅ |
| AdamW `betas / eps` | `(0.8, 0.95) / 1e-10` (`gpt.py:756`) | `(0.8, 0.95) / 1e-10` | ✅ |
| AdamW LR scaling | `∝1/√(dmodel/768)` (`gpt.py:746–749`) | identical | ✅ |
| Muon momentum | fixed `0.95` (`gpt.py` `muon_kwargs`) | **ramped 0.85→0.95 over 300 steps** (`base_train.py:447`) | ⚠️ bio-only warmup |
| `weight_decay` default | `0.0` (`train.py:925`) | `0.0` (`base_train.py:100`) | ✅ |
| Fused AdamW | enabled when available & non-DDP (`gpt.py:760–763`) | enabled when non-DDP & CUDA (`gpt.py`) | ✅ equivalent |
| Non-matrix/1-D params | routed to AdamW at matrix LR instead of crashing Muon (`gpt.py:719–726`) | same intent | ✅ |

**Ours-only optimizer:** `--optimizer-type {adamw,muon,hoss}`
(`train.py:56`). `adamw`/`muon` build the same AdamW+Muon dual that bio uses;
`hoss` (`gpt.py:712–714`) replaces the whole optimizer with the HOSS
hyperreal second-order method — a research mechanism with **no bio
counterpart**. Preserve it; just don't use it when reproducing a bio-style
baseline.

The only optimizer-side parity gap is bio's **Muon momentum warmup**
(`get_muon_momentum`: `frac=min(it/300,1); m=(1-frac)*0.85+frac*0.95`). On
short fixed-FLOPs runs (hundreds of steps) this measurably changes early
dynamics. See [§4](#4-action-items--recommended-benchmark-flags).

---

## 2. LR scheduler — the main divergence

| | This repo | bio_inspired |
|---|---|---|
| Mechanism | `--scheduler-type {none, ordinal}` (`train.py:57`) | hardcoded `get_lr_multiplier` (`base_train.py:435`) |
| **Default LR shape** | **flat / constant** (no scheduler attached when `none`; LR never stepped — `train.py:941,1491–1494`) | **WSD / trapezoidal**: `warmup_ratio=0.0`, `warmdown_ratio=0.2` ⇒ stable then linear decay to 0 over last 20 % (`base_train.py:103–104,435–445`) |
| Non-trivial option | **`ordinal`** — transfinite well-founded LR scheduler driven by loss (`OrdinalLRScheduler`, `train.py:941`); research mechanism | none (WSD only) |
| `--warmup-steps` here | **not an LR warmup** — it only excludes the first N steps from *throughput measurement* (`train.py:2238,1349,1684`) | n/a |

This is the most important finding. With the default `--scheduler-type none`,
**LR is held constant for the entire run**. bio's default decays the LR to ~0
over the final 20 % of steps, which typically lowers final train/val loss.
Consequences:

- **Within-repo A/B is still fair** — every attention mechanism trains under
  the same flat LR at matched FLOPs, so relative rankings are valid (this is
  exactly what `hypotheses/registry.yaml` adjudicates).
- **Absolute numbers are not comparable to bio** unless you match the
  schedule, and a flat LR likely leaves a small uniform loss improvement
  unrealized across *all* mechanisms.

> Recommendation (code change → defer to a bead): add an optional WSD/warmdown
> LR option to `train.py` (e.g. `--lr-warmdown-ratio`) so the *baseline*
> matches bio and so non-ordinal runs can use the standard tail. This is a
> training-loop policy change, not a mechanism change, so it preserves all
> mathematical features. Filed as discovered-work, see [§5](#5-followups).

---

## 3. Mixed precision / compile / clipping / horizon

| Knob | This repo | bio_inspired | Verdict |
|---|---|---|---|
| tf32 matmul | `set_float32_matmul_precision("high")` (`common.py:188`) | identical (`common.py:194`) | ✅ |
| Logits dtype | `logits.float()` (`gpt.py:809`) | identical (`gpt.py:512`) | ✅ |
| Autocast | bf16 on CUDA, `nullcontext()` on CPU (`train.py:1026–1029`) | bf16 via `torch.amp.autocast(device_type=…)` (`base_train.py:143`) | ✅ (ours is explicitly CPU-safe) |
| **torch.compile** | **opt-in** `--compile` w/ `backend/mode/fullgraph/dynamic` flags (`train.py:896–912,2293–2316`) | **always on**, `dynamic=False` (`base_train.py:298`) | ⚠️ match per-arm |
| **grad clip** | **off by default** (`--grad-clip-norm`, default `None`; `train.py:926,1984`) | **on, `grad_clip=1.0`** (`base_train.py:102`) | ⚠️ **match this** |
| Dropout | none (no `dropout` in `GPTConfig`) | none | ✅ |
| Horizon | `--target-flops` ∨ `--max-steps` (default 20) (`train.py:2217`) | `num_iterations` ∨ `target_flops` ∨ Chinchilla `target_param_data_ratio=20` (`base_train.py:92–94`) | ✅ for fixed-FLOPs; bio adds Chinchilla ratio |

Two of these need conscious matching for a fair comparison:

- **grad clip.** bio clips global grad-norm to 1.0 every step; we clip only
  when `--grad-clip-norm` is passed. Note HOSS skips clipping by design
  (`train.py:1393`). For a like-for-like baseline, pass `--grad-clip-norm 1.0`.
- **torch.compile.** Identical math, different throughput and warmup cost.
  For an A/B, either compile both arms or neither.

---

## 4. Action items / recommended benchmark flags

For a baseline that mirrors bio's training policy while preserving our
mechanisms, run with:

```bash
python -m nanochat.train \
    --attention-type <mech> \
    --optimizer-type adamw \         # AdamW(embed/head)+Muon(matrices), == bio
    --grad-clip-norm 1.0 \           # match bio default (we default OFF)
    --lr-warmdown-ratio 0.2 \        # match bio's WSD tail (kj8s; we default flat)
    --target-flops <F> \             # shared fixed-FLOPs horizon
    --compile                        # match bio (always-compiled); or drop on both arms
```

- **Always-do for cross-repo parity:** `--grad-clip-norm 1.0`,
  `--lr-warmdown-ratio 0.2` (bio decays the last 20 %; our default stays flat),
  match `--compile`, match `(B, T, world_size)` and `--target-flops`.
- **Still cannot match via flags:** Muon-momentum warmup has no flag here.
  Accept the small uniform gap vs bio on very short runs, or port it (below).
- **Within-repo A/B (the project's actual protocol):** no flags required for
  fairness — all mechanisms share the loop. Keep scheduler settings identical
  across arms when reading absolute `val_ce`.

---

## 5. Follow-ups

Discovered-work worth a bead (`--deps discovered-from:model_guided_research-wyf`):

1. ~~**Optional WSD/warmdown LR for non-ordinal runs**~~ **DONE (bead kj8s)**:
   `--lr-warmdown-ratio` (default 0.0 = flat) implements the linear tail;
   conflicts are rejected for `--scheduler-type ordinal`; stateless in
   (step, max_steps) so resume reproduces trajectories without new checkpoint
   state.
2. **Optional Muon momentum warmup** (`gpt.py`/`train.py`): port
   `get_muon_momentum` (0.85→0.95 over N steps) behind a flag. Lower priority;
   only matters on very short runs.
3. **Chinchilla horizon** (`train.py`): add `--target-param-data-ratio` as a
   third horizon mode alongside `--target-flops`/`--max-steps`. Convenience,
   not a fairness issue.

No code changes are made by this audit (per the bead's "defer code changes"
scope). The fixed-FLOPs benchmark harness is already fair *within* this repo;
the recommendations above are for matching bio and for squeezing a uniform
loss improvement out of the warmdown tail.

---

## Appendix — files compared

| Role | This repo | bio_inspired |
|---|---|---|
| Training driver | `nanochat/train.py` | `scripts/base_train.py` |
| Optimizer setup | `nanochat/gpt.py:711` | `bio_inspired_nanochat/gpt.py` `setup_optimizers` |
| AdamW / Muon | `nanochat/adamw.py`, `nanochat/muon.py` | `bio_inspired_nanochat/adamw.py`, `muon.py` |
| HOSS (ours-only) | `nanochat/hoss_opt_torch.py` | — |
| Ordinal LR (ours-only) | `nanochat/ordinal_scheduler.py` | — |
| tf32/bf16 policy | `nanochat/common.py:188`, `gpt.py:809` | `common.py:194`, `gpt.py:512` |
