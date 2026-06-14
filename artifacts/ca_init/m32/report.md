# CA-init early-phase benchmark — `m32`

Bead model_guided_research-m32. Compares CA weight init (`--ca-init-rule`) vs the standard init vs an alpha-blended mix on small attention-heavy and MLP-heavy models, at matched steps/seeds.

- Device: `cpu`  ·  steps: `120`  ·  warmup: `5`  ·  lr: `0.0006`  ·  optimizer: `adamw`  ·  seeds: `[0, 1]`

## Architectures

- **attn_heavy** — L4 H8 d128 T256 B4: 8 heads / 256-token window: attention + qkv projections dominate
- **mlp_heavy** — L6 H2 d128 T32 B8: 6 layers / 32-token window: FFN per-block work dominates, attention negligible

## Results (mean over seeds)

Loss/grad columns aggregate trained cells; activation/weight columns come from the in-process init-time probe (runs for every cell).

| config | variant | ok | final loss | loss drop | grad‖·‖ mean | grad‖·‖ max | in-proj act RMS | resid RMS | depth ratio | init w-std |
|---|---|---|---|---|---|---|---|---|---|---|
| attn_heavy | standard | 2/2 | 7.421 | 3.405 | 5.799 | 9.399 | 1.001 | 1 | 1 | 0.1421 |
| attn_heavy | ca_rule30 | 2/2 | 7.375 | 3.451 | 5.755 | 9.313 | 1.522 | 1 | 1 | 0.142 |
| attn_heavy | ca_mix0.5 | 2/2 | 7.841 | 2.985 | 3.693 | 5.753 | 0.8154 | 1 | 1 | 0.1004 |
| mlp_heavy | standard | 2/2 | 7.38 | 3.446 | 4.929 | 9.26 | 0.9984 | 1 | 1 | 0.1248 |
| mlp_heavy | ca_rule30 | 2/2 | 7.408 | 3.418 | 5.718 | 9.952 | 1.529 | 1 | 1 | 0.1249 |
| mlp_heavy | ca_mix0.5 | 2/2 | 7.825 | 3.001 | 3.677 | 5.549 | 0.8148 | 1 | 1 | 0.08829 |

## Interpretation

### attn_heavy

- **ca_rule30** vs standard: final-loss Δ = `-0.0465` (better); act-RMS depth ratio `1.0` (standard `1.0`).
- **ca_mix0.5** vs standard: final-loss Δ = `+0.4198` (worse); act-RMS depth ratio `1.0` (standard `1.0`).

### mlp_heavy

- **ca_rule30** vs standard: final-loss Δ = `+0.0283` (worse); act-RMS depth ratio `1.0` (standard `1.0`).
- **ca_mix0.5** vs standard: final-loss Δ = `+0.4450` (worse); act-RMS depth ratio `1.0` (standard `1.0`).

## Reproduction

Each cell's exact command is in `summary.json` (`cells[].command`). Top-level command:

```
uv run python scripts/ca_init_bench.py --run-id m32 --max-steps 120 --warmup-steps 5 --seeds 0 1
```
