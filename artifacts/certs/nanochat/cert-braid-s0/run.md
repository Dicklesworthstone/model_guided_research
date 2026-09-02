# mgr certify

- run_id: `cert-braid-s0` · seed: 0 · device: cpu · dtype: fp32
- git: `3ffda85`
- result: **10 passed / 0 failed / 0 errored** in 2.7s

| Mechanism | Check | Family | Status | Measured | Tolerance | ms |
|---|---|---|---|---:|---:|---:|
| braid | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 584.6 |
| braid | ybe_law_holds | classical | pass | 8.882e-16 | <= 1.000e-10 | 1.1 |
| braid | restricted_law_violates_ybe | classical | pass | 4.681e+00 | >= 1.000e-03 | 0.7 |
| braid | payload_multiset_invariance | classical | pass | 0.000e+00 | <= 0.000e+00 | 3.8 |
| braid | rmatrix_braid_relation_holds | classical | pass | 5.329e-15 | <= 1.000e-10 | 0.6 |
| braid | rmatrix_inversion_relation_holds | classical | pass | 9.104e-15 | <= 1.000e-10 | 0.1 |
| braid | rmatrix_transfer_matrices_commute | classical | pass | 5.602e-17 | <= 1.000e-10 | 1.6 |
| braid | rmatrix_perturbed_transfer_separates | classical | pass | 2.682e-05 | >= 1.000e-06 | 4.1 |
| braid | rmatrix_mass_partition_charge_conserved | classical | pass | 5.960e-07 | <= 1.000e-05 | 118.6 |
| braid | heuristic_mass_partition_violated | classical | pass | 2.662e+00 | >= 1.000e-03 | 117.3 |
