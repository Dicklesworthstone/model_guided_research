# Model-state visualizations (`nanochat.viz`)

Beads **hi3** (insightful model-state visualizations) and **7ow** (per-head
entropy / route diversity). This is a *seeing* tool: render what a mathematical
attention mechanism is actually doing on a small, seeded sample batch, rather
than only reading scalar metrics.

It reuses the introspection buffers the mechanisms already populate — so it adds
**no** new state to the hot path and never changes a forward result:

| signal | source buffer | enabled by |
| --- | --- | --- |
| per-head attention entropy | `attn_entropy_head_mean` | `standard_record_attn_entropy` |
| tropical runner-up margins | `tropical_gamma_head_mean` / `_min` | `tropical_record_margins` |
| tropical route coverage | `tropical_route_coverage` | `tropical_record_margins` (+ `semiring_beta`) |
| softmax attention maps | captured live (reversible `attend` patch, `no_grad`) | always (standard heads) |

`viz.py` flips the relevant record flag on automatically for the chosen
mechanism, so you never have to remember the buffer plumbing.

## Commands

All runs are seeded and reproducible. Output lands under `artifacts/vis/`
(git-ignored); a `summary.json` manifest + an `index.html` gallery + matplotlib
PNGs are written, and a rich table is printed.

```bash
# (hi3) >=3 model-state visualizations.
#   standard -> attention-entropy heatmap + per-head softmax attention maps
python -m nanochat.viz state --attention standard --out artifacts/vis/state_standard
#   tropical -> per-head route-margin heatmap (+ coverage when semiring_beta set)
python -m nanochat.viz state --attention tropical --out artifacts/vis/state_tropical

# Visualize a TRAINED checkpoint instead of a fresh seeded probe:
python -m nanochat.viz state \
    --checkpoint artifacts/baseline/nanochat/quicktour/checkpoints \
    --out artifacts/vis/state_quicktour

# (7ow) per-head entropy + cross-head route diversity, baseline vs math feature.
python -m nanochat.viz entropy --baseline standard --feature tropical \
    --out artifacts/vis/entropy/standard_vs_tropical
```

Useful knobs (both subcommands): `--seed`, `--n-layer`, `--n-head`,
`--n-kv-head`, `--n-embd`, `--seq-len`, `--batch-size`, `--device cuda`,
`--text "..."` (encoded for the sample batch) or `--random-input` (seeded random
ids, for offline boxes without the tokenizer).

## The three visualizations and how to read them

### 1. Per-head attention-entropy heatmap (`attention_entropy_heatmap.png`)

A `layer × head` grid of the mean softmax entropy (in nats) of each head.

- **High entropy** (bright) → the head spreads attention broadly (a *mixing* /
  averaging head; near `log(context_len)` ≈ uniform).
- **Low entropy** (dark) → the head is *selective*, concentrating on a few keys
  (an induction / copy / positional head).
- Reading down a column shows how one head's selectivity changes with depth;
  fresh random-init models sit near-uniform — train first to see structure.

### 2. Per-head softmax attention maps (`attention_maps.png`)

For one example and the first layer, a grid of `query × key` attention matrices,
one per head. Because attention is causal, every map is lower-triangular.

- A bright **diagonal** = local/self attention; a bright **first column** =
  attention to the BOS/anchor token; bright **off-diagonal stripes** = induction
  (attend to the token after a previous match).
- Compare heads side-by-side: visually distinct maps ⇒ specialized heads;
  near-identical maps ⇒ redundant heads (quantified by route diversity below).

### 3. Tropical route-margin heatmap (`tropical_route_margins.png`)

A `layer × head` grid of the tropical **runner-up margin** γ — the gap between
the winning (max-plus) route and the second-best, per token, averaged per head.

- **Large margin** → a confident, certifiable route (the min-gap/2 robustness
  certificate is wide; the discrete argmax is stable under perturbation).
- **Small / zero margin** → ties between routes; the piecewise-linear decision
  boundary runs right through these tokens.
- `route coverage` (printed) = fraction of routes above the β route-stability
  threshold (finite only when `--semiring-beta` is set; exact tropical reports
  margins but leaves coverage NaN by design).

## Per-head entropy & route diversity (7ow)

`viz entropy` runs a **baseline** and a **math-feature** mechanism on the same
seeded batch and tabulates, per config:

- **head entropy μ** — mean per-head signal (attention entropy for standard /
  softmax heads; per-head margin for tropical, where entropy is undefined).
- **head entropy σ** — spread across heads = *head specialization* (all heads
  alike ⇒ σ≈0; a mix of broad and selective heads ⇒ large σ).
- **route diversity (JS)** — mean pairwise Jensen-Shannon divergence between
  heads' attention maps, in `[0, 1]`. **0** = heads route identically
  (redundant; candidates for pruning); **→1** = heads send the same tokens to
  different places (a rich, specialized head population). Computed from the
  captured softmax maps, so it is reported for softmax-head mechanisms.
- **coverage** — tropical route coverage (see above).

Outputs: a grouped bar chart of the per-head signal (`per_head_entropy_diversity.png`),
a `summary.json` with every number, and an `index.html`. Saved under
`artifacts/vis/entropy/`.

## Training-dashboard per-head heatmap (92m)

The live training dashboard (`--dashboard`) grows a **per-head route
diversity / entropy heatmap** panel, fed by the per-head metric vectors the
trainer already streams (e.g. `tropical_gamma_head_mean`). Each per-head metric
renders as a self-scaled colored strip (one cell per head, red→green = low→high)
with a cross-head spread `σ` (a cheap route-diversity scalar) and mean `μ`. The
panel shows inline in the terminal **and** in the recorded HTML export
(`<artifacts-dir>/dashboard/<run_id>.html`).

It is **data-gated** (invisible until a per-head field flows, so standard runs
pay nothing) and **toggleable** via `TrainingDashboard(show_head_heatmaps=...)`.

```bash
# Dashboard with the per-head heatmap (tropical route margins per head).
python -m nanochat.train \
    --attention tropical --tropical-record-margins \
    --dashboard \
    --device cpu --target-flops 2e8 --batch-size 8 \
    --artifacts-kind bench --artifacts-topic dashboard/nanochat \
    --run-id dash_trop_demo
# -> live panel "per-head route diversity / entropy · heatmap (bead 92m)"
#    + artifacts/.../dashboard/dash_trop_demo.html
```

Any `*head*` step-record field that is a length-2..64 numeric vector is picked up
automatically, so when standard-attention per-head entropy is streamed
(`attn_entropy_head_mean`) it appears in the same panel with no further wiring.

## Notes

- **Reproducibility** — `--seed` seeds model init and the sample batch; the
  manifest records the exact config so any figure can be regenerated.
- **No hot-path cost** — the visualizer only runs when invoked; the capture
  patch is reversible and `no_grad`, and the record buffers are the same ones
  the trainer already uses (default-off elsewhere).
- **Fresh vs trained** — a fresh probe validates the *plumbing* and shows the
  initialization geometry; point `--checkpoint` at a trained run to see learned
  structure (induction heads, confident tropical routes, specialized heads).
</content>
