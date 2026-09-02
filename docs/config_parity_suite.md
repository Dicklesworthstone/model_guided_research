# Config Parity Suite (drift detection)

**Bead:** `model_guided_research-s53`
**Date:** 2026-06-14
**Builds on:** `configs/fair_comparison.json` (2bx), `docs/config_parity.md`
(wyf), `docs/data_parity.md` (6fl).

A small battery of **matched** training runs with fixed seeds and recorded
**expected metric bands**, so config/optimizer/data drift (here or vs
`bio_inspired_nanochat`) is caught by re-running and comparing. These are
**short smoke runs (20 steps)** — the band is a *drift sentinel*, not a quality
comparison (20 steps barely trains; `CE≈10.5` is near the `ln(50304)=10.83`
random-init floor). Do **not** read the tropical band as a quality win.

## The suite

All runs: E1 rung `L4/H4/KV2/D128/seq256`, `B=8`, `--grad-clip-norm 1.0`
(the wyf fairness flag), `--max-steps 20`, `--seed 42`, `--device cpu`,
`--checkpoint-interval 100000`. Requires ≥2 dataset shards (`train.py:865`).

| # | Arm | Final Train CE band | tok/s (this box) | notes |
|---|---|---|---|---|
| 1 | **baseline** standard | **10.502 ± 0.10** | ~1140 | reference |
| 2 | **math** tropical attn + tropical FFN | **9.129 ± 0.10** | ~170 | slow (see `docs/microbench_ops.md`); CE≠quality at 20 steps |
| 3 | **math** reversible additive | **10.521 ± 0.10** | ~910 | ≈ baseline CE, as expected |
| 4 | **flex** standard + FlexAttention | = arm 1 (GPU) | — | **GPU-only**; on CPU the flag falls back to the dense path ⇒ identical to arm 1 |

Bands were measured on a 64-core CPU box at `commit 56b06a4`. Tolerance `±0.10`
absorbs BLAS thread-count / cross-machine float-reduction-order variation;
**same machine + same `OMP_NUM_THREADS` + same seed reproduces to ~1e-3.**

## Commands

```bash
# 1. baseline standard
python -m nanochat.train --device cpu --attention-type standard \
  --n-layer 4 --n-head 4 --n-kv-head 2 --n-embd 128 --sequence-len 256 \
  --batch-size 8 --grad-clip-norm 1.0 --max-steps 20 --seed 42 \
  --checkpoint-interval 100000 --run-id parity_standard --artifacts-dir /tmp/parity

# 2. math: tropical attn + tropical FFN
python -m nanochat.train --device cpu --attention-type tropical --ffn-type tropical \
  --n-layer 4 --n-head 4 --n-kv-head 2 --n-embd 128 --sequence-len 256 \
  --batch-size 8 --grad-clip-norm 1.0 --max-steps 20 --seed 42 \
  --checkpoint-interval 100000 --run-id parity_tropical --artifacts-dir /tmp/parity

# 3. math: reversible additive
python -m nanochat.train --device cpu --attention-type reversible --reversible-mode additive \
  --n-layer 4 --n-head 4 --n-kv-head 2 --n-embd 128 --sequence-len 256 \
  --batch-size 8 --grad-clip-norm 1.0 --max-steps 20 --seed 42 \
  --checkpoint-interval 100000 --run-id parity_reversible --artifacts-dir /tmp/parity

# 4. flex (GPU box only; on CPU this equals arm 1)
python -m nanochat.train --device cuda --attention-type standard --use-flex-attention \
  --n-layer 4 --n-head 4 --n-kv-head 2 --n-embd 128 --sequence-len 256 \
  --batch-size 8 --grad-clip-norm 1.0 --max-steps 20 --seed 42 \
  --checkpoint-interval 100000 --run-id parity_flex --artifacts-dir /tmp/parity
```

Read `Final Train CE` from each run's summary (or `summary.json`).

## Detecting drift

1. **Manual:** re-run an arm; if its `Final Train CE` leaves the band above,
   something changed (init, optimizer defaults, data, dtype, tokenizer). Bisect
   against `docs/config_parity.md` (the fairness flags) and `docs/data_parity.md`.
2. **Tooling:** the runs write `summary.json`; compare two runs with the
   regression gate (`mgr regressions --baseline … --candidate … --fail-on-regression`,
   defaults `--loss-abs 0.01`). For cross-machine bands use `±0.10`.
3. **Flex parity (GPU):** the real flex check is `flex_on` vs `flex_off`
   producing the **same** loss (identical math, `configs/fair_comparison.json`);
   run both on a CUDA box and assert equality — not testable on CPU.

## Mirror in `bio_inspired_nanochat`

Run the same arms in that repo (its trainer is `scripts/base_train.py`; flag
names differ — see `docs/config_parity.md`). Two cross-repo caveats from the
audits, both of which shift absolute CE so **match them before comparing
numbers**:

- bio defaults `grad_clip=1.0` (we now pass `--grad-clip-norm 1.0` to match);
- bio applies a **WSD warmdown**; ours holds LR flat unless
  `--lr-warmdown-ratio` is passed (bead `kj8s` landed). Pass
  `--lr-warmdown-ratio 0.2` to match bio's tail at 20 steps; without it expect
  a small offset in the final CE.

## Scope / caveats

- **Drift sentinel, not a benchmark.** For real mechanism comparison use the
  fixed-FLOPs harness (`mgr bench-fixed-flops`, `configs/fair_comparison.json`)
  at a real budget, not 20-step CE.
- Bands are CPU/fp32. A GPU/bf16 box will have different absolute CE (re-measure
  bands there once before using them).
- Update the bands here whenever a deliberate config change moves them, with the
  commit that caused the move (so the band always reflects intended state).
</content>
