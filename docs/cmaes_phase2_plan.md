# CMA-ES Phase 2 — Expanded Parameter Groups

Bead: `model_guided_research-2co`

Phase 1 (`docs/cmaes_phase1.md`, bead `68v`) searches **10** SynapticConfig
knobs at a fixed FLOPs budget. Phase 2 either (a) makes the Phase-1 result
**robust** (multi-seed), or (b) **expands** the search to additional parameter
groups. This doc specifies the groups, bounds, sequencing, and stopping rules.
It is the companion to the budget envelope in `docs/cmaes_budget_schedule.md`
and the infra plan in `docs/cmaes_plan_mgr.md`.

The harness (`scripts/cmaes_phase1.py`) already supports everything Phase 2
needs: `--eval-seeds`/`--seed-agg` (robust averaging, bead `q8f`),
`--max-evals`/`--max-wall-seconds`/`--patience`/`--max-crash-rate` (budget
guard, bead `2mj`), `--resume` (preemption, bead `a3u`), and `--data-dir`
fingerprinting (bead `wiz`). No code change is required to run Phase 2; the
only new code would be widening `PARAM_SPECS` for the expanded-space variant.

---

## Gate: do not start Phase 2 until the objective carries signal

The single most important precondition — and the one most likely to be skipped.

`scripts/cmaes_analyze.py` (bead `0wn`) reports the **score std** across
candidates. The CPU-scale pilot (`68v_pilot_cpu`) has score std ≈ `2e-6` over a
loss of ~10.85 — i.e. **flat**: candidates are indistinguishable at that
budget, so any parameter ranking is just fitting numerical noise. CMA-ES cannot
optimize an objective it cannot resolve.

**Phase-2 entry criterion:** on the Phase-1 winner's budget, the score std
across a generation must exceed `--signal-threshold` (default `1e-3` loss
units), and ideally the best-vs-baseline gap should be ≥1–2% (the budget
doc's "Go" bar). If the objective is flat, the fix is **more budget**
(`--target-flops` up, more steps) or a **harder/longer task**, *not* a bigger
search space. Run `cmaes_analyze` after Phase 1 and read the `has_signal` flag
before committing Phase-2 compute.

---

## Variant 1 (do this first): robust averaging, same 10-D space

Confirm the Phase-1 winner is not a single-seed fluke before spending on a
larger space.

```bash
uv run python scripts/cmaes_phase1.py --run-id phase2_robust \
  --device auto --generations 15 --population-size 16 \
  --eval-seeds 0 1 2 --seed-agg mean_std --seed-agg-lambda 1.0 \
  --target-flops <phase1_budget> \
  --patience 4 --max-wall-seconds <budget> --max-crash-rate 0.10
```

- `--seed-agg mean_std` penalizes high across-seed variance, so the search
  prefers configs that are *reliably* good, not lucky on one seed.
- Budget: see "Phase 2 (robust averaging)" tiers in
  `docs/cmaes_budget_schedule.md` (Min 180 / Target 720 / Max 2160 evals).

---

## Variant 2: expanded parameter space

All fields below are real `SynapticConfig` knobs (`nanochat/synaptic.py`). Add
**one group at a time** — joint search over 25+ dims needs far more population
and evals (CMA-ES population scales ~`4 + 3·ln(d)`; effective sample need grows
faster). Encode learning-rate-like knobs as `log10`, everything else `linear`.

### Group A — post-synaptic plasticity (9 knobs)
The learning-rule timescales; most likely to move the loss after the release
knobs are tuned.

| param | kind | suggested bounds | default |
|---|---|---|---|
| `post_fast_decay` | linear | 0.80 – 0.99 | 0.95 |
| `post_trace_decay` | linear | 0.80 – 0.99 | 0.96 |
| `camkii_up` | linear | 0.01 – 0.20 | 0.05 |
| `camkii_down` | linear | 0.005 – 0.10 | 0.02 |
| `pp1_tau` | linear | 0.95 – 0.999 | 0.985 |
| `camkii_thr` | linear | 0.5 – 2.0 | 1.0 |
| `pp1_thr` | linear | 0.3 – 1.0 | 0.7 |
| `bdnf_tau` | linear | 0.95 – 0.999 | 0.985 |
| `bdnf_scale` | linear | 0.5 – 2.0 | 1.0 |

### Group B — calcium / SNARE release sensors (5 knobs)
| param | kind | suggested bounds | default |
|---|---|---|---|
| `syt1_slope` | linear | 4.0 – 16.0 | 8.0 |
| `syt7_slope` | linear | 1.0 – 6.0 | 3.0 |
| `cpx_thresh` | linear | 0.3 – 0.8 | 0.55 |
| `complexin_bias` | linear | -0.5 – 0.5 | 0.0 |
| `doc2_gain` | linear | 0.0 – 0.30 | 0.08 |

### Group C — vesicle pools / recycling (8 knobs)
| param | kind | suggested bounds | default |
|---|---|---|---|
| `init_reserve` | linear | 6.0 – 36.0 | 18.0 |
| `init_snare` | linear | 0.3 – 1.0 | 0.7 |
| `init_clamp` | linear | 0.3 – 1.0 | 0.6 |
| `unprime_per_release` | linear | 0.01 – 0.20 | 0.05 |
| `nsf_recover` | linear | 0.02 – 0.20 | 0.08 |
| `amp_load` | linear | 0.005 – 0.10 | 0.02 |
| `amp_leak` | linear | 0.001 – 0.03 | 0.006 |
| `endo_delay` | int | 1 – 8 | 3 |

### Group D — energy / metabolism (3 knobs)
| param | kind | suggested bounds | default |
|---|---|---|---|
| `init_energy` | linear | 0.5 – 1.0 | 0.85 |
| `energy_fill` | linear | 0.005 – 0.08 | 0.02 |
| `energy_use` | linear | 0.005 – 0.08 | 0.02 |

> `endo_delay` is an integer; until `ParamSpec` grows an `int` kind, search it
> as `linear` and round in the decoder, or hold it fixed and sweep separately.

---

## Sequencing

1. **Robust-averaging pass** (Variant 1) — validate the Phase-1 winner.
2. **One expanded group at a time** — start from the robust winner as the CMA
   mean (seed `PARAM_SPECS` with the winner's decoded values), add Group A
   first (largest expected effect), then B/C/D as budget allows.
3. **Sensitivity-pruned joint search** — after per-group passes, take the
   union of params whose `|Spearman|` from `cmaes_analyze` exceeds ~0.3 and run
   one joint search over just those, holding the rest at their best values.

## Stopping rules (concrete)

- `--patience 4`: stop after 4 generations with no ≥`--min-improve` gain.
- `--max-crash-rate 0.10`: stop if >10% of evals fail (the budget doc's
  "pause immediately" threshold).
- `--max-wall-seconds`: hard wall, set from calibration × tier eval count.
- A run that `cmaes_analyze` reports as still-flat after the **Target** tier is
  a **No-Go** — revise budget or task, do not throw more search at it.

## Artifacts

Reuse the standard layout: per-run `run.json` (records the param space,
budget, and dataset fingerprint), `progress.csv`, `best.json`, `state/`
(resume checkpoints), `summary.md`. Analyze every run with
`scripts/cmaes_analyze.py --run-id <id>` → `artifacts/cmaes/analysis/<id>/`.
