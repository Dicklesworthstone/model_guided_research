# Certificate Telemetry Instability-Detection Study (`model_guided_research-5ki.5`)

- **Hypothesis**: `hyp-cert-telemetry-leads-divergence`
- **Mechanism**: `gauge` (`n_layer=2, n_head=2, n_embd=32, sequence_len=32`)
- **Task**: `hier` (`--certify-interval 1`, `--max-steps 12`)
- **Detection Rule**: `certs.*` watched invariant (`invariant_drift`, `lipschitz_ratio`, `orthogonality_residual`, `gauge_angle_rms`, `round_trip_error`) breaches `band_k = 3.0` relative to initial baseline or trailing 20-point band.
- **Loss Divergence Definition**: Non-finite (`NaN`/`Inf`) or `loss > 2.0 * running_min_loss`.
- **Adjudication Result (`ci-v6`)**: `inconclusive` (`effect = +2.40` steps, `ci95 = [1.72, 3.08]` straddling the `2.0`-step threshold because 3 of 5 seeds land on the integer boundary `2` and 2 of 5 land on `3`; `power = 100%`, `n = 5/5`, `q = 0.0889`).

## Lead-Time Table (5 Seeds per Arm)

| Run ID | Arm | Seed | `learning_rate` | `matrix_lr` | Step-0 Loss | Step-1 Loss | `first_warn_step` | `loss_divergence_step` | `lead_steps` | `drift_ratio` (`max / init`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gauge-stable-s101` | `stable` | 101 | 0.001 | 0.001 | 5.9506 | 5.8585 | `null` | `null` | 0 | 1.0002x |
| `gauge-stable-s102` | `stable` | 102 | 0.001 | 0.001 | 5.9506 | 5.8410 | `null` | `null` | 0 | 1.0071x |
| `gauge-stable-s103` | `stable` | 103 | 0.001 | 0.001 | 5.9506 | 5.8492 | `null` | `null` | 0 | 1.0001x |
| `gauge-stable-s104` | `stable` | 104 | 0.001 | 0.001 | 5.9506 | 5.8502 | `null` | `null` | 0 | 1.0100x |
| `gauge-stable-s105` | `stable` | 105 | 0.001 | 0.001 | 5.9506 | 5.8297 | `null` | `null` | 0 | 1.0093x |
| `gauge-unstable-s101` | `unstable` | 101 | 0.050 | 0.600 | 5.9506 | 3.5334 | 1 | 4 | 3 | 1,810,510x |
| `gauge-unstable-s102` | `unstable` | 102 | 0.050 | 0.600 | 5.9506 | 2.7021 | 1 | 3 | 2 | 1,389,157x |
| `gauge-unstable-s103` | `unstable` | 103 | 0.050 | 0.600 | 5.9506 | 2.8408 | 1 | 4 | 3 | 900,771x |
| `gauge-unstable-s104` | `unstable` | 104 | 0.050 | 0.600 | 5.9506 | 3.0750 | 1 | 3 | 2 | 4,664,159x |
| `gauge-unstable-s105` | `unstable` | 105 | 0.050 | 0.600 | 5.9506 | 2.3411 | 1 | 3 | 2 | 4,669,199x |

## ROC Table Across `band_k` Multipliers

| `band_k` Threshold | TPR | FPR | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|
| 1.05 | 1.00 | 0.00 | 5 | 0 | 5 | 0 |
| 1.15 | 1.00 | 0.00 | 5 | 0 | 5 | 0 |
| 1.50 | 1.00 | 0.00 | 5 | 0 | 5 | 0 |
| 2.00 | 1.00 | 0.00 | 5 | 0 | 5 | 0 |
| 3.00 (default) | 1.00 | 0.00 | 5 | 0 | 5 | 0 |
| 5.00 | 1.00 | 0.00 | 5 | 0 | 5 | 0 |
| 10.00 | 1.00 | 0.00 | 5 | 0 | 5 | 0 |

## Mechanism Explanation

At step 1 of the unstable runs, the training loss drops sharply (from `5.95` to `2.34`–`3.53`), masking internal instability because final `RMSNorm` projects the residual stream onto the unit sphere and logit softcapping (`30 * tanh(logits / 30)`) bounds the output logits. However, `certs.invariant_drift`, `certs.lipschitz_ratio`, and `certs.orthogonality_residual` probe the internal block transformations directly and breach the `3.0x` band at step 1 — `2` to `3` steps before cross-entropy loss rebounds past `2.0x` best-loss at step 3–4.
