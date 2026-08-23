# Scaling-law comparative report

- Inputs: `['artifacts/scaling/tropical/smoketest_141830']`
- Config: tail_fraction=0.1, bootstrap=2000, seed=1729, confidence=0.95
- Loss definition: mean of the last tail_fraction of each run's loss stream (D2 metrics.jsonl preferred, summary.json fallback); multi-seed rungs averaged.

> **HONESTY CAVEATS (fixed):** these fits describe ONLY the measured compute range — 
> extrapolation beyond it is unsupported by construction. Thin ladders carry wide CIs; 
> treat wide-CI exponent comparisons as undecided, not as evidence of equality.

**Headline:** fewer than two mechanisms produced saturating fits with valid bootstraps — no cross-mechanism comparison is possible yet.

## tropical

| rung | C (FLOPs) | tail loss | seeds | loss source |
|---|---|---|---|---|
| smoke_6M | 1.28031e+11 | 8.4297 | 1 | metrics |
| smoke_14M | 4.47952e+11 | 4.26475 | 1 | metrics |

- Saturating fit REFUSED (2 rungs < 3): a 2-parameter-plus-floor fit on two points is underdetermined.
- Plain power law: k=9.30611e+06, b=0.544052, R²(log space)=1

## G2-compatible result block (`mgr.scaling.v1`)

```json
{
  "config": {
    "bootstrap_draws": 2000,
    "bootstrap_seed": 1729,
    "confidence": 0.95,
    "tail_fraction": 0.1
  },
  "fits": {
    "tropical": {
      "amplitude_a": null,
      "exponent_b": null,
      "exponent_b_ci95": null,
      "floor_c": null,
      "n_rungs": 2,
      "n_seeds_total": 2,
      "plain_power_law_b": 0.5440519125,
      "r2_original_scale": null
    }
  },
  "generated_from": [
    "artifacts/scaling/tropical/smoketest_141830"
  ],
  "pairwise_exponent_tests": [],
  "schema": "mgr.scaling.v1"
}
```

Registry note: no hypothesis prediction currently targets `mgr.scaling.v1`; when one is registered (G1) `mgr adjudicate` can consume this block verbatim.
