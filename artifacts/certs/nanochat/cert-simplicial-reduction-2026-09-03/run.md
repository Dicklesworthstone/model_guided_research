# mgr certify

- run_id: `cert-simplicial-reduction-2026-09-03` · seed: 42 · device: cpu · dtype: fp32
- git: `c4cf6cb`
- result: **3 passed / 0 failed / 0 errored** in 4.4s

| Mechanism | Check | Family | Status | Measured | Tolerance | ms |
|---|---|---|---|---:|---:|---:|
| simplicial | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 1801.2 |
| simplicial | mass_conservation_two_hop | classical | pass | 1.192e-07 | <= 1.000e-05 | 83.7 |
| simplicial | zero_triangle_reduces_to_standard_attention | reduction | pass | 2.384e-07 | <= 2.000e-05 | 152.9 |
