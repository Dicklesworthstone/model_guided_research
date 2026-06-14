# CMA-ES analysis — `68v_pilot_cpu`

Bead model_guided_research-0wn. Parameter sensitivity and score distribution over `4` evaluated candidates.

## Score distribution

- min `10.85406` · max `10.85407` · mean `10.85407` · std `1.95e-06` · range `4.77e-06`
- **FLAT — no usable signal** (std ≤ threshold 0.001)

> The objective is flat across candidates: the per-config budget is too small to separate them. Sensitivity below is therefore not meaningful — increase steps/FLOPs or widen the search before trusting parameter rankings.

## Best candidate

- gen `0` cand `3` · score `10.85406`

| param | value |
|---|---|
| `tau_c` | `0.794923` |
| `alpha_c` | `0.64756` |
| `init_rrp` | `5.22198` |
| `prime_rate` | `0.104175` |
| `rec_rate` | `0.184174` |
| `lambda_loge` | `0.940216` |
| `barrier_strength` | `0.300818` |
| `stochastic_train_frac` | `0.341582` |
| `post_fast_lr` | `0.0036928` |
| `post_slow_lr` | `0.000537543` |

## Parameter sensitivity (ranked by |Spearman|)

| param | kind | n | Spearman | Pearson |
|---|---|---|---|---|
| `tau_c` | linear | 4 | — | — |
| `alpha_c` | linear | 4 | — | — |
| `init_rrp` | linear | 4 | — | — |
| `prime_rate` | linear | 4 | — | — |
| `rec_rate` | linear | 4 | — | — |
| `lambda_loge` | linear | 4 | — | — |
| `barrier_strength` | linear | 4 | — | — |
| `stochastic_train_frac` | linear | 4 | — | — |
| `post_fast_lr` | log10 | 4 | — | — |
| `post_slow_lr` | log10 | 4 | — | — |

_Positive Spearman ⇒ larger parameter correlates with **higher** (worse) loss._

## Plots
- `sensitivity.png` — per-param Spearman bar chart
- `param_corr.png` — param×param correlation heatmap
