# mgr certify

- run_id: `cert-gauge-reduction-2026-09-03` · seed: 42 · device: cpu · dtype: fp32
- git: `53e66e3`
- result: **6 passed / 0 failed / 0 errored** in 5.6s

| Mechanism | Check | Family | Status | Measured | Tolerance | ms |
|---|---|---|---|---:|---:|---:|
| gauge | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 768.6 |
| gauge | rotation_inverse_roundtrip | classical | pass | 2.384e-07 | <= 1.000e-05 | 5.1 |
| gauge | rotation_pairwise_norm_preservation | classical | pass | 4.768e-07 | <= 1.000e-05 | 10.9 |
| gauge | rotation_additivity_cumsum_law | classical | pass | 2.608e-07 | <= 1.000e-05 | 17.4 |
| gauge | zero_transport_reduces_to_standard_attention | reduction | pass | 0.000e+00 | <= 2.000e-05 | 342.3 |
| gauge | kv_decode_matches_full_forward | classical | pass | 5.960e-07 | <= 1.000e-04 | 537.1 |
