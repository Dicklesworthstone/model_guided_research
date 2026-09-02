# mgr certify

- run_id: `cert-refresh-2026-09-02` · seed: 0 · device: cpu · dtype: fp32
- git: `0eeb168`
- result: **64 passed / 0 failed / 0 errored** in 21.4s

| Mechanism | Check | Family | Status | Measured | Tolerance | ms |
|---|---|---|---|---:|---:|---:|
| standard | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 10.0 |
| tropical | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 12.1 |
| ultrametric | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 56.9 |
| simplicial | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 589.9 |
| quaternion | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 453.7 |
| braid | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 7.5 |
| fractal | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 7.2 |
| octonion | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 17.4 |
| surreal | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 6.0 |
| reversible | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 847.7 |
| gauge | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 50.0 |
| clifford | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 221.5 |
| hyperbolic | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 38.2 |
| standard | rope_pairwise_norm_preservation | classical | pass | 4.768e-07 | <= 1.000e-05 | 4.0 |
| standard | causal_mask_structure | classical | pass | 0.000e+00 | <= 0.000e+00 | 23.1 |
| standard | rmsnorm_unit_rms | classical | pass | 1.788e-07 | <= 1.000e-03 | 0.8 |
| standard | softmax_row_stochastic | classical | pass | 1.192e-07 | <= 1.000e-06 | 5.2 |
| tropical | lipschitz_1_sup_norm_q | classical | pass | 9.954e-01 | <= 1.000e+00 | 5.9 |
| tropical | lipschitz_1_sup_norm_v | classical | pass | 9.917e-01 | <= 1.000e+00 | 0.8 |
| tropical | score_center_pure_gauge_shift | classical | pass | 9.537e-07 | <= 1.000e-05 | 5.3 |
| tropical | margin_matches_bruteforce | classical | pass | 0.000e+00 | <= 1.000e-06 | 1.0 |
| tropical | maslov_endpoint_within_sandwich | reduction | pass | 1.148e-01 | <= 1.000e+00 | 1.6 |
| tropical | ffn_lipschitz_1_sup_norm | classical | pass | 1.000e+00 | <= 1.000e+00 | 4.1 |
| tropical | ffn_collapse_single_layer | classical | pass | 0.000e+00 | <= 1.000e-09 | 5.3 |
| quaternion | qmul_associativity | classical | pass | 7.105e-15 | <= 1.000e-10 | 1.1 |
| quaternion | qmul_norm_multiplicative | classical | pass | 1.776e-15 | <= 1.000e-10 | 0.8 |
| quaternion | qconj_antihomomorphism | classical | pass | 8.882e-16 | <= 1.000e-10 | 4.0 |
| quaternion | rotor_norm_preservation | classical | pass | 8.882e-16 | <= 1.000e-10 | 1.0 |
| clifford | cgp_associativity | classical | pass | 2.132e-14 | <= 1.000e-10 | 26.9 |
| clifford | reversion_antihomomorphism | classical | pass | 3.553e-15 | <= 1.000e-10 | 5.4 |
| clifford | rotor_norm_preservation | classical | pass | 2.137e-16 | <= 1.000e-09 | 14.2 |
| clifford | quaternion_subalgebra_reduction | reduction | pass | 1.776e-15 | <= 1.000e-10 | 12.6 |
| hyperbolic | lorentz_constraint_residual | classical | pass | 8.762e-06 | <= 2.000e-05 | 0.9 |
| hyperbolic | exp_log_origin_roundtrip | classical | pass | 2.980e-08 | <= 2.000e-05 | 0.9 |
| hyperbolic | energy_gromov_reduces_to_standard | reduction | pass | 1.621e-05 | <= 2.000e-05 | 25.6 |
| octonion | omul_norm_multiplicative | classical | pass | 5.329e-15 | <= 1.000e-09 | 4.5 |
| octonion | omul_alternativity | classical | pass | 1.421e-14 | <= 1.000e-09 | 7.1 |
| octonion | omul_nonassociativity_witness | classical | pass | 6.635e+01 | >= 1.000e-02 | 4.6 |
| octonion | o_times_conj_is_norm_squared | classical | pass | 7.105e-15 | <= 1.000e-09 | 4.3 |
| octonion | reduces_to_quaternion_on_subalgebra | reduction | pass | 0.000e+00 | <= 1.000e-09 | 1.3 |
| reversible | forward_inverse_roundtrip | classical | pass | 2.384e-07 | <= 1.000e-05 | 52.9 |
| reversible | custom_autograd_grad_parity | classical | pass | 3.298e-07 | <= 1.000e-04 | 18.9 |
| reversible | symplectic_jacobian | classical | pass | 6.661e-16 | <= 1.000e-09 | 7548.7 |
| reversible | energy_drift_separation | classical | pass | 7.161e-08 | <= 2.000e-01 | 7412.9 |
| gauge | rotation_inverse_roundtrip | classical | pass | 2.384e-07 | <= 1.000e-05 | 6.9 |
| gauge | rotation_pairwise_norm_preservation | classical | pass | 2.384e-07 | <= 1.000e-05 | 5.2 |
| gauge | rotation_additivity_cumsum_law | classical | pass | 3.576e-07 | <= 1.000e-05 | 6.2 |
| gauge | kv_decode_matches_full_forward | classical | pass | 6.557e-07 | <= 1.000e-04 | 323.0 |
| ultrametric | strong_triangle_inequality_lcp | classical | pass | 0.000e+00 | <= 0.000e+00 | 41.7 |
| ultrametric | trie_decode_matches_hard_kernel | reduction | pass | 2.384e-07 | <= 1.000e-03 | 737.5 |
| braid | ybe_law_holds | classical | pass | 8.882e-16 | <= 1.000e-10 | 1.1 |
| braid | restricted_law_violates_ybe | classical | pass | 4.681e+00 | >= 1.000e-03 | 0.8 |
| braid | payload_multiset_invariance | classical | pass | 0.000e+00 | <= 0.000e+00 | 0.8 |
| braid | rmatrix_braid_relation_holds | classical | pass | 5.329e-15 | <= 1.000e-10 | 0.6 |
| braid | rmatrix_inversion_relation_holds | classical | pass | 9.104e-15 | <= 1.000e-10 | 0.1 |
| braid | rmatrix_transfer_matrices_commute | classical | pass | 5.602e-17 | <= 1.000e-10 | 1.3 |
| braid | rmatrix_perturbed_transfer_separates | classical | pass | 2.682e-05 | >= 1.000e-06 | 1.2 |
| braid | rmatrix_mass_partition_charge_conserved | classical | pass | 5.960e-07 | <= 1.000e-05 | 60.5 |
| braid | heuristic_mass_partition_violated | classical | pass | 2.662e+00 | >= 1.000e-03 | 152.1 |
| simplicial | mass_conservation_two_hop | classical | pass | 1.788e-07 | <= 1.000e-05 | 26.9 |
| fractal | router_branch_simplex | classical | pass | 1.192e-07 | <= 1.000e-06 | 18.1 |
| surreal | row_norm_equals_exp_scale | classical | pass | 2.384e-07 | <= 1.000e-05 | 0.9 |
| surreal | layer_linearity | classical | pass | 2.861e-06 | <= 1.000e-04 | 5.9 |
| surreal | scale_shift_equivariance | classical | pass | 1.907e-06 | <= 1.000e-04 | 1.4 |
