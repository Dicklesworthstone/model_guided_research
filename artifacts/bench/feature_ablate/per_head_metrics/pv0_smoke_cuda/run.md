# Per-head metrics suite

- Run ID: `pv0_smoke_cuda`
- Device: `cuda`
- Seeds: `0, 1, 2`
- Target FLOPs/run (est): `2.000e+08`

## Variants

### standard

- label: `standard (SDPA)`
- attention_type: `standard`
- expected_use_flex_attention: `False`
- final_loss: mean=`10.82583999633789` std=`0.0` ci95=`0.0` n=`3`

#### attention_entropy (per head)

| head | entropy_mean | std | ci95 |
| --- | --- | --- | --- |
| 0 | 2.814069 | 0.006252 | 0.015533 |
| 1 | 2.803332 | 0.002226 | 0.005529 |
| 2 | 2.803445 | 0.014917 | 0.037058 |
| 3 | 2.809621 | 0.003357 | 0.008339 |

#### Runs

| seed | status | final_loss | summary |
| --- | --- | --- | --- |
| 0 | ok | 10.825840 | bench/feature_ablate/per_head_metrics/pv0_smoke_cuda/standard/seed_0/summary.json |
| 1 | ok | 10.825840 | bench/feature_ablate/per_head_metrics/pv0_smoke_cuda/standard/seed_1/summary.json |
| 2 | ok | 10.825840 | bench/feature_ablate/per_head_metrics/pv0_smoke_cuda/standard/seed_2/summary.json |

### standard_flex

- label: `standard (FlexAttention)`
- attention_type: `standard`
- expected_use_flex_attention: `True`
- final_loss: mean=`10.82583999633789` std=`0.0` ci95=`0.0` n=`3`

#### attention_entropy (per head)

| head | entropy_mean | std | ci95 |
| --- | --- | --- | --- |
| 0 | 2.814069 | 0.006252 | 0.015533 |
| 1 | 2.803332 | 0.002226 | 0.005529 |
| 2 | 2.803445 | 0.014917 | 0.037058 |
| 3 | 2.809621 | 0.003357 | 0.008339 |

#### Runs

| seed | status | final_loss | summary |
| --- | --- | --- | --- |
| 0 | ok | 10.825840 | bench/feature_ablate/per_head_metrics/pv0_smoke_cuda/standard_flex/seed_0/summary.json |
| 1 | ok | 10.825840 | bench/feature_ablate/per_head_metrics/pv0_smoke_cuda/standard_flex/seed_1/summary.json |
| 2 | ok | 10.825840 | bench/feature_ablate/per_head_metrics/pv0_smoke_cuda/standard_flex/seed_2/summary.json |

### tropical

- label: `tropical (margins)`
- attention_type: `tropical`
- expected_use_flex_attention: `False`
- final_loss: mean=`10.82583999633789` std=`0.0` ci95=`0.0` n=`3`

#### tropical margin gamma (per head)

| head | gamma_mean | std | ci95 |
| --- | --- | --- | --- |
| 0 | 0.019875 | 0.000558 | 0.001386 |
| 1 | 0.022208 | 0.001141 | 0.002836 |
| 2 | 0.021820 | 0.004598 | 0.011423 |
| 3 | 0.020319 | 0.004736 | 0.011767 |

#### Runs

| seed | status | final_loss | summary |
| --- | --- | --- | --- |
| 0 | ok | 10.825840 | bench/feature_ablate/per_head_metrics/pv0_smoke_cuda/tropical/seed_0/summary.json |
| 1 | ok | 10.825840 | bench/feature_ablate/per_head_metrics/pv0_smoke_cuda/tropical/seed_1/summary.json |
| 2 | ok | 10.825840 | bench/feature_ablate/per_head_metrics/pv0_smoke_cuda/tropical/seed_2/summary.json |

## Command

```bash
/data/projects/model_guided_research/cli.py per-head-metrics --device cuda --seed 0 --seed 1 --seed 2 --target-flops 2e8 --batch-size 4 --sequence-len 64 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64 --run-id pv0_smoke_cuda --timeout-s 600
```
