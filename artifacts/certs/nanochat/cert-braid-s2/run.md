# mgr certify

- run_id: `cert-braid-s2` · seed: 2 · device: cpu · dtype: fp32
- git: `3ffda85`
- result: **10 passed / 0 failed / 0 errored** in 2.4s

| Mechanism | Check | Family | Status | Measured | Tolerance | ms |
|---|---|---|---|---:|---:|---:|
| braid | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 488.9 |
| braid | ybe_law_holds | classical | pass | 8.882e-16 | <= 1.000e-10 | 2.5 |
| braid | restricted_law_violates_ybe | classical | pass | 4.820e+00 | >= 1.000e-03 | 2.1 |
| braid | payload_multiset_invariance | classical | pass | 0.000e+00 | <= 0.000e+00 | 2.2 |
| braid | rmatrix_braid_relation_holds | classical | pass | 2.842e-14 | <= 1.000e-10 | 2.0 |
| braid | rmatrix_inversion_relation_holds | classical | pass | 7.283e-14 | <= 1.000e-10 | 0.1 |
| braid | rmatrix_transfer_matrices_commute | classical | pass | 4.981e-17 | <= 1.000e-10 | 4.1 |
| braid | rmatrix_perturbed_transfer_separates | classical | pass | 3.735e-05 | >= 1.000e-06 | 1.1 |
| braid | rmatrix_mass_partition_charge_conserved | classical | pass | 5.960e-07 | <= 1.000e-05 | 121.6 |
| braid | heuristic_mass_partition_violated | classical | pass | 1.870e+00 | >= 1.000e-03 | 98.0 |
