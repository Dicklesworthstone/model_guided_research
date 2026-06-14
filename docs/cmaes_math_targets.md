# CMA-ES Targets for the Math Mechanisms

Bead: `model_guided_research-ybp`

Phase 1/2 tune the **synaptic** model's biology knobs. This doc identifies the
small sets of **continuous** knobs in the *mathematical* attention mechanisms
that are the best CMA-ES candidates — where a 2–5-dimensional sweep at a fixed
FLOPs budget could plausibly move the loss. No runs yet; this scopes them.

Each target lists the params, bounds, whether the knob is already a `train.py`
CLI flag (so the existing harness can drive it) or still needs one, a rough
cost note, and a go/no-go gate.

> **Shared gate (read first).** Every target inherits the Phase-2 entry rule
> (`docs/cmaes_phase2_plan.md`): at the chosen budget the objective must carry
> signal — `scripts/cmaes_analyze.py` `has_signal=true` (score std >
> `--signal-threshold`). A mechanism whose loss is flat across the knob range
> at the test budget is a No-Go regardless of how interesting the knob is.

> **Harness note.** `scripts/cmaes_phase1.py` currently hard-codes
> `--model-type synaptic --synaptic-config <json>`. Driving these targets needs
> a small generalization: an objective mode that instead emits
> `--model-type gpt --attention-type <mech>` plus the per-knob flags below.
> That is the one piece of code Phase-2-for-math requires; everything else
> (budget guard, resume, multi-seed, analysis) is mechanism-agnostic already.

---

## 1. Tropical attention + FFN — Maslov smoothing β  *(highest priority)*

Continuous, well-understood, and already certified (the finite-β → exact
tropical endpoint reduction, beads `uvjq`/`y2h9`). β interpolates softmax
(small β) ↔ exact max-plus (β→∞), so it is a genuine accuracy/robustness dial.

| param | flag | kind | bounds | default |
|---|---|---|---|---|
| `semiring_beta` (attention) | `--semiring-beta` ✅ | log10 | 1 – 64 | none (exact) |
| `ffn_beta` (tropical FFN) | `--ffn-beta` ✅ | log10 | 1 – 64 | none (exact) |

- **CLI-ready:** yes (both flags exist).
- **Cost:** same as a standard run; 2-D so a small population (8) suffices.
- **Go/no-go:** go if val-loss or route-coverage varies monotonically with β at
  the test budget; no-go if the β-sweep is flat (then β is irrelevant at this
  scale). Pair with `--tropical-record-margins` to read route coverage.

## 2. Reversible attention — coupling floor λ_min

| param | flag | kind | bounds | default |
|---|---|---|---|---|
| `reversible_lambda_min` | `--reversible-lambda-min` ✅ | linear | 0.01 – 0.30 | 0.05 |

- A 1-D sweep is below CMA-ES's `mu>=2` floor; pair it with `semiring_beta` or
  search it as a **grid** (5–7 points) instead — cheaper and clearer for 1-D.
- **Go/no-go:** go if λ_min trades off symplectic-energy drift
  (`--reversible-record-energy`) against loss; no-go if loss is λ_min-flat.

## 3. Braid attention — temperature τ

| param | flag | kind | bounds | default |
|---|---|---|---|---|
| `braid_tau` | `--braid-tau` ✅ | linear | 0.0 – 2.0 | 0.0 |

- Soft-mode only (`--braid-mode soft`); τ controls crossing sharpness.
- 1-D → prefer a grid. **Go/no-go:** braid–Dyck length-generalization is the
  natural metric, but recall the floored-win trap (`docs/` methodology): test
  at an **off-floor** rung or the result is not structural.

## 4. Ultrametric attention — LCP sharpness & decay  *(needs CLI flags first)*

The richest continuous set, but currently only `--ultrametric-mode` /
`--ultrametric-hard-digits` are CLI-exposed; the dials below live in
`GPTConfig` (`nanochat/gpt.py`) and need flags before the harness can drive
them.

| param (GPTConfig) | flag | kind | bounds | default |
|---|---|---|---|---|
| `ultrametric_lcp_beta` | needs `--ultrametric-lcp-beta` | log10 | 4 – 128 | 32 |
| `ultrametric_alpha` | needs `--ultrametric-alpha` | linear | 0.5 – 4.0 | 2.0 |
| `ultrametric_K` | needs `--ultrametric-k` | int | 4 – 16 | 8 |

- **Prereq:** add the three CLI flags (small, mechanical) → then a 2–3-D sweep.
- **Go/no-go:** go if held-out hierarchical-depth accuracy responds to
  `lcp_beta`/`alpha`; this is the mechanism whose hypothesis
  (`hyp-ultrametric-hier-heldout-depth`) most directly motivates a sweep.

---

## JAX-demo route (separate harness)

The root-level demos (`mgr run tropical`, etc.) have their own env-gated knobs
(`TROP_SPARSE_TRAIN`, `ULTRA_SCALE_COMPARE`, demo CLI flags). Optimizing those
would need a **JAX objective** (demo metric → scalar), not the nanochat
subprocess. That is a distinct, larger effort (`docs/cmaes_plan_mgr.md`
"Integration points"); the nanochat-attention targets above are the
near-term, low-effort wins because they reuse the existing FLOPs harness.

## Recommended order

1. **Tropical β (target 1)** — CLI-ready, 2-D, certified, clearest dial.
2. **Ultrametric (target 4)** — add 3 flags, then sweep; best hypothesis tie-in.
3. Reversible / braid (targets 2–3) — 1-D, do as grids, not CMA-ES.
