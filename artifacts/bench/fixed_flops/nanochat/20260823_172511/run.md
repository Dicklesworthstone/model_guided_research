# Fixed-FLOPs nanochat benchmark

- Run ID: `20260823_172511`
- Baseline: `clifford`
- Device: `cpu`
- Target FLOPs/run (est): `3.000e+07`
- Seeds: `[7]`
- Score metric: `train-loss tail` (lower is better)

## A/B vs baseline (mean ± std over seeds, Welch t-test)

| attention_type (train-loss tail) | n_ok | mean±std | Δ vs base | 95% CI | Welch p | tokens/s | mem GB | NaN/Inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clifford | 1 | 10.8258±0.0000 | — | — | — | 94 | n/a | 0 |
| quaternion | 1 | 10.8258±0.0000 | n/a | n/a | n/a | 1,827 | n/a | 0 |

`\*` = Welch two-sample p < 0.05 vs baseline. This is a descriptive A/B benchmark, deliberately distinct from the preregistered `mgr adjudicate` engine (which tests a registered threshold with floor/power gates). Per-feature deltas in `feature_ablate.csv`.

## Conclusions

- Best (lowest train-loss tail): `clifford` mean=`10.825837`
- No arm differs significantly from baseline at p<0.05 (n=1 seed(s); widen seeds to gain power).

## Command

```bash
/data/projects/model_guided_research/.venv/bin/mgr bench-fixed-flops -a clifford -a quaternion --target-flops 3e7 --seed 7
```
