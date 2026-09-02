# mgr certify

- run_id: `cert-refresh-2026-09-02b` · seed: 42 · device: cpu · dtype: fp32
- git: `dbd2d50`
- result: **15 passed / 0 failed / 0 errored** in 1.7s

| Mechanism | Check | Family | Status | Measured | Tolerance | ms |
|---|---|---|---|---:|---:|---:|
| braid | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 29.5 |
| standard | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 85.3 |
| standard | rope_pairwise_norm_preservation | classical | pass | 2.384e-07 | <= 1.000e-05 | 1.3 |
| standard | causal_mask_structure | classical | pass | 0.000e+00 | <= 0.000e+00 | 10.2 |
| standard | rmsnorm_unit_rms | classical | pass | 1.788e-07 | <= 1.000e-03 | 0.8 |
| standard | softmax_row_stochastic | classical | pass | 5.960e-08 | <= 1.000e-06 | 7.1 |
| braid | ybe_law_holds | classical | pass | 8.882e-16 | <= 1.000e-10 | 0.7 |
| braid | restricted_law_violates_ybe | classical | pass | 4.267e+00 | >= 1.000e-03 | 0.7 |
| braid | payload_multiset_invariance | classical | pass | 0.000e+00 | <= 0.000e+00 | 4.8 |
| braid | rmatrix_braid_relation_holds | classical | pass | 1.421e-14 | <= 1.000e-10 | 1.5 |
| braid | rmatrix_inversion_relation_holds | classical | pass | 1.377e-14 | <= 1.000e-10 | 0.1 |
| braid | rmatrix_transfer_matrices_commute | classical | pass | 4.331e-17 | <= 1.000e-10 | 6.8 |
| braid | rmatrix_perturbed_transfer_separates | classical | pass | 3.356e-05 | >= 1.000e-06 | 1.5 |
| braid | rmatrix_mass_partition_charge_conserved | classical | pass | 1.776e-15 | <= 1.000e-05 | 80.1 |
| braid | heuristic_mass_partition_violated | classical | pass | 2.173e+00 | >= 1.000e-03 | 18.9 |
