# mgr certify

- run_id: `20261122_220336` · seed: 42 · device: cpu · dtype: fp32
- git: `8ce6cde`
- result: **72 passed / 0 failed / 0 errored** in 3.1s

| Mechanism | Check | Family | Status | Measured | Tolerance | ms |
|---|---|---|---|---:|---:|---:|
| standard | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 9.7 |
| tropical | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 8.2 |
| ultrametric | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 8.0 |
| simplicial | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 7.3 |
| quaternion | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 10.0 |
| braid | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 7.0 |
| fractal | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 7.2 |
| octonion | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 19.6 |
| surreal | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 7.1 |
| reversible | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 316.0 |
| gauge | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 7.4 |
| clifford | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 11.8 |
| hyperbolic | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 10.9 |
| standard | rope_pairwise_norm_preservation | classical | pass | 2.384e-07 | <= 1.000e-05 | 1.2 |
| standard | causal_mask_structure | classical | pass | 0.000e+00 | <= 0.000e+00 | 0.2 |
| standard | rmsnorm_unit_rms | classical | pass | 1.788e-07 | <= 1.000e-03 | 0.9 |
| standard | softmax_row_stochastic | classical | pass | 1.192e-07 | <= 1.000e-06 | 0.8 |
| tropical | lipschitz_1_sup_norm_q | classical | pass | 9.886e-01 | <= 1.000e+00 | 1.1 |
| tropical | lipschitz_1_sup_norm_v | classical | pass | 9.886e-01 | <= 1.000e+00 | 1.0 |
| tropical | score_center_pure_gauge_shift | classical | pass | 4.768e-07 | <= 1.000e-05 | 1.0 |
| tropical | margin_matches_bruteforce | classical | pass | 0.000e+00 | <= 1.000e-06 | 1.4 |
| tropical | maslov_endpoint_within_sandwich | reduction | pass | 1.398e-01 | <= 1.000e+00 | 1.3 |
| tropical | ffn_lipschitz_1_sup_norm | classical | pass | 9.528e-01 | <= 1.000e+00 | 1.6 |
| tropical | ffn_collapse_single_layer | classical | pass | 2.220e-16 | <= 1.000e-09 | 1.5 |
| quaternion | qmul_associativity | classical | pass | 7.105e-15 | <= 1.000e-10 | 1.4 |
| quaternion | qmul_norm_multiplicative | classical | pass | 1.776e-15 | <= 1.000e-10 | 1.1 |
| quaternion | qconj_antihomomorphism | classical | pass | 8.882e-16 | <= 1.000e-10 | 1.2 |
| quaternion | rotor_norm_preservation | classical | pass | 1.332e-15 | <= 1.000e-10 | 1.1 |
| clifford | cgp_associativity | classical | pass | 1.066e-14 | <= 1.000e-10 | 2.5 |
| clifford | reversion_antihomomorphism | classical | pass | 3.553e-15 | <= 1.000e-10 | 1.7 |
| clifford | rotor_norm_preservation | classical | pass | 3.847e-16 | <= 1.000e-09 | 5.4 |
| clifford | quaternion_subalgebra_reduction | reduction | pass | 8.882e-16 | <= 1.000e-10 | 1.3 |
| hyperbolic | lorentz_constraint_residual | classical | pass | 8.762e-06 | <= 2.000e-05 | 0.9 |
| hyperbolic | exp_log_origin_roundtrip | classical | pass | 2.980e-08 | <= 2.000e-05 | 1.1 |
| hyperbolic | energy_gromov_reduces_to_standard | reduction | pass | 1.621e-05 | <= 2.000e-05 | 5.0 |
| octonion | omul_norm_multiplicative | classical | pass | 7.105e-15 | <= 1.000e-09 | 1.7 |
| octonion | omul_alternativity | classical | pass | 1.421e-14 | <= 1.000e-09 | 5.4 |
| octonion | omul_nonassociativity_witness | classical | pass | 5.714e+01 | >= 1.000e-02 | 3.1 |
| octonion | o_times_conj_is_norm_squared | classical | pass | 5.329e-15 | <= 1.000e-09 | 1.8 |
| octonion | reduces_to_quaternion_on_subalgebra | reduction | pass | 0.000e+00 | <= 1.000e-09 | 1.8 |
| reversible | forward_inverse_roundtrip | classical | pass | 2.384e-07 | <= 1.000e-05 | 2.5 |
| reversible | custom_autograd_grad_parity | classical | pass | 2.660e-07 | <= 1.000e-04 | 8.1 |
| reversible | symplectic_jacobian | classical | pass | 6.661e-16 | <= 1.000e-09 | 413.5 |
| reversible | energy_drift_separation | classical | pass | 1.010e-07 | <= 2.000e-01 | 395.2 |
| gauge | rotation_inverse_roundtrip | classical | pass | 2.384e-07 | <= 1.000e-05 | 3.4 |
| gauge | rotation_pairwise_norm_preservation | classical | pass | 4.768e-07 | <= 1.000e-05 | 2.4 |
| gauge | rotation_additivity_cumsum_law | classical | pass | 2.608e-07 | <= 1.000e-05 | 3.1 |
| gauge | zero_transport_reduces_to_standard_attention | reduction | pass | 0.000e+00 | <= 2.000e-05 | 3.2 |
| gauge | kv_decode_matches_full_forward | classical | pass | 4.768e-07 | <= 1.000e-04 | 19.9 |
| ultrametric | strong_triangle_inequality_lcp | classical | pass | 0.000e+00 | <= 0.000e+00 | 25.4 |
| ultrametric | trie_decode_matches_hard_kernel | reduction | pass | 2.682e-07 | <= 1.000e-03 | 33.7 |
| braid | ybe_law_holds | classical | pass | 8.882e-16 | <= 1.000e-10 | 1.0 |
| braid | restricted_law_violates_ybe | classical | pass | 4.267e+00 | >= 1.000e-03 | 0.9 |
| braid | payload_multiset_invariance | classical | pass | 0.000e+00 | <= 0.000e+00 | 1.0 |
| braid | rmatrix_braid_relation_holds | classical | pass | 1.421e-14 | <= 1.000e-10 | 0.9 |
| braid | rmatrix_inversion_relation_holds | classical | pass | 1.377e-14 | <= 1.000e-10 | 0.2 |
| braid | rmatrix_transfer_matrices_commute | classical | pass | 4.997e-17 | <= 1.000e-10 | 1.7 |
| braid | rmatrix_perturbed_transfer_separates | classical | pass | 3.356e-05 | >= 1.000e-06 | 1.5 |
| braid | rmatrix_mass_partition_charge_conserved | classical | pass | 1.776e-15 | <= 1.000e-05 | 3.8 |
| braid | heuristic_mass_partition_violated | classical | pass | 2.173e+00 | >= 1.000e-03 | 2.8 |
| simplicial | mass_conservation_two_hop | classical | pass | 1.192e-07 | <= 1.000e-05 | 1.0 |
| simplicial | zero_triangle_reduces_to_standard_attention | reduction | pass | 2.384e-07 | <= 2.000e-05 | 5.8 |
| simplicial | triplet_mass_conservation | classical | pass | 1.788e-07 | <= 1.000e-05 | 2.4 |
| simplicial | triplet_wedge_antisymmetry | classical | pass | 0.000e+00 | <= 1.000e-05 | 1.7 |
| fractal | router_branch_simplex | classical | pass | 1.192e-07 | <= 1.000e-06 | 2.4 |
| fractal | zero_ifs_reduces_to_prototype_attention | reduction | pass | 0.000e+00 | <= 1.000e-05 | 6.4 |
| fractal | ifs_moran_contractivity | classical | pass | 4.768e-07 | <= 1.000e-05 | 4.7 |
| surreal | row_norm_equals_exp_scale | classical | pass | 2.384e-07 | <= 1.000e-05 | 1.0 |
| surreal | layer_linearity | classical | pass | 3.338e-06 | <= 1.000e-04 | 1.7 |
| surreal | scale_shift_equivariance | classical | pass | 1.907e-06 | <= 1.000e-04 | 1.7 |
| surreal | zero_omega_reduces_to_standard_attention | reduction | pass | 2.384e-07 | <= 2.000e-05 | 2.2 |
| surreal | valuation_dominance_monotone | classical | pass | 0.000e+00 | <= 1.000e-05 | 2.4 |
