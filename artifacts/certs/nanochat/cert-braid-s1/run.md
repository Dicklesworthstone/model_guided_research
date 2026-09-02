# mgr certify

- run_id: `cert-braid-s1` · seed: 1 · device: cpu · dtype: fp32
- git: `3ffda85`
- result: **10 passed / 0 failed / 0 errored** in 2.7s

| Mechanism | Check | Family | Status | Measured | Tolerance | ms |
|---|---|---|---|---:|---:|---:|
| braid | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 398.3 |
| braid | ybe_law_holds | classical | pass | 8.882e-16 | <= 1.000e-10 | 1.2 |
| braid | restricted_law_violates_ybe | classical | pass | 4.476e+00 | >= 1.000e-03 | 1.1 |
| braid | payload_multiset_invariance | classical | pass | 0.000e+00 | <= 0.000e+00 | 4.2 |
| braid | rmatrix_braid_relation_holds | classical | pass | 7.105e-15 | <= 1.000e-10 | 0.9 |
| braid | rmatrix_inversion_relation_holds | classical | pass | 5.329e-15 | <= 1.000e-10 | 0.1 |
| braid | rmatrix_transfer_matrices_commute | classical | pass | 5.407e-17 | <= 1.000e-10 | 4.4 |
| braid | rmatrix_perturbed_transfer_separates | classical | pass | 2.498e-05 | >= 1.000e-06 | 1.3 |
| braid | rmatrix_mass_partition_charge_conserved | classical | pass | 5.960e-07 | <= 1.000e-05 | 140.3 |
| braid | heuristic_mass_partition_violated | classical | pass | 2.599e+00 | >= 1.000e-03 | 105.9 |
