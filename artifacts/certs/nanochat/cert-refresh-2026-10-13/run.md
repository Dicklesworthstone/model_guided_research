# mgr certify

- run_id: `cert-refresh-2026-10-13` · seed: 42 · device: cpu · dtype: fp32
- git: `aeb4c36`
- result: **8 passed / 0 failed / 0 errored** in 2.3s

| Mechanism | Check | Family | Status | Measured | Tolerance | ms |
|---|---|---|---|---:|---:|---:|
| standard | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 17.5 |
| ultrametric | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 10.7 |
| standard | rope_pairwise_norm_preservation | classical | pass | 2.384e-07 | <= 1.000e-05 | 1.5 |
| standard | causal_mask_structure | classical | pass | 0.000e+00 | <= 0.000e+00 | 0.3 |
| standard | rmsnorm_unit_rms | classical | pass | 1.788e-07 | <= 1.000e-03 | 1.0 |
| standard | softmax_row_stochastic | classical | pass | 1.192e-07 | <= 1.000e-06 | 1.1 |
| ultrametric | strong_triangle_inequality_lcp | classical | pass | 0.000e+00 | <= 0.000e+00 | 27.0 |
| ultrametric | trie_decode_matches_hard_kernel | reduction | pass | 2.384e-07 | <= 1.000e-03 | 62.9 |
