# mgr certify

- run_id: `cert-ci-v7-7115611` · seed: 42 · device: cpu · dtype: fp32
- git: `7115611`
- result: **72 passed / 0 failed / 0 errored** in 1.4s

| Mechanism | Check | Family | Status | Measured | Tolerance | ms |
|---|---|---|---|---:|---:|---:|
| standard | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 18.3 |
| tropical | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 5.9 |
| ultrametric | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 4.6 |
| simplicial | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 3.9 |
| quaternion | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 4.3 |
| braid | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 2.8 |
| fractal | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 4.8 |
| octonion | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 7.9 |
| surreal | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 6.2 |
| reversible | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 172.7 |
| gauge | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 3.4 |
| clifford | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 6.4 |
| hyperbolic | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 6.5 |
| standard | rope_pairwise_norm_preservation | classical | pass | 2.384e-07 | <= 1.000e-05 | 0.8 |
| standard | causal_mask_structure | classical | pass | 0.000e+00 | <= 0.000e+00 | 0.5 |
| standard | rmsnorm_unit_rms | classical | pass | 1.788e-07 | <= 1.000e-03 | 0.6 |
| standard | softmax_row_stochastic | classical | pass | 1.192e-07 | <= 1.000e-06 | 0.6 |
| tropical | lipschitz_1_sup_norm_q | classical | pass | 9.886e-01 | <= 1.000e+00 | 1.2 |
| tropical | lipschitz_1_sup_norm_v | classical | pass | 9.886e-01 | <= 1.000e+00 | 1.3 |
| tropical | score_center_pure_gauge_shift | classical | pass | 9.537e-07 | <= 1.000e-05 | 1.2 |
| tropical | margin_matches_bruteforce | classical | pass | 0.000e+00 | <= 1.000e-06 | 1.9 |
| tropical | maslov_endpoint_within_sandwich | reduction | pass | 1.398e-01 | <= 1.000e+00 | 1.3 |
| tropical | ffn_lipschitz_1_sup_norm | classical | pass | 9.528e-01 | <= 1.000e+00 | 1.3 |
| tropical | ffn_collapse_single_layer | classical | pass | 2.220e-16 | <= 1.000e-09 | 1.3 |
| quaternion | qmul_associativity | classical | pass | 7.105e-15 | <= 1.000e-10 | 1.0 |
| quaternion | qmul_norm_multiplicative | classical | pass | 2.665e-15 | <= 1.000e-10 | 0.6 |
| quaternion | qconj_antihomomorphism | classical | pass | 8.882e-16 | <= 1.000e-10 | 0.9 |
| quaternion | rotor_norm_preservation | classical | pass | 8.882e-16 | <= 1.000e-10 | 0.7 |
| clifford | cgp_associativity | classical | pass | 1.421e-14 | <= 1.000e-10 | 1.3 |
| clifford | reversion_antihomomorphism | classical | pass | 3.553e-15 | <= 1.000e-10 | 0.9 |
| clifford | rotor_norm_preservation | classical | pass | 3.847e-16 | <= 1.000e-09 | 2.5 |
| clifford | quaternion_subalgebra_reduction | reduction | pass | 8.882e-16 | <= 1.000e-10 | 0.7 |
| hyperbolic | lorentz_constraint_residual | classical | pass | 8.762e-06 | <= 2.000e-05 | 0.5 |
| hyperbolic | exp_log_origin_roundtrip | classical | pass | 2.980e-08 | <= 2.000e-05 | 0.6 |
| hyperbolic | energy_gromov_reduces_to_standard | reduction | pass | 1.621e-05 | <= 2.000e-05 | 3.3 |
| octonion | omul_norm_multiplicative | classical | pass | 7.105e-15 | <= 1.000e-09 | 1.1 |
| octonion | omul_alternativity | classical | pass | 1.421e-14 | <= 1.000e-09 | 3.1 |
| octonion | omul_nonassociativity_witness | classical | pass | 5.714e+01 | >= 1.000e-02 | 1.1 |
| octonion | o_times_conj_is_norm_squared | classical | pass | 7.105e-15 | <= 1.000e-09 | 0.6 |
| octonion | reduces_to_quaternion_on_subalgebra | reduction | pass | 0.000e+00 | <= 1.000e-09 | 0.5 |
| reversible | forward_inverse_roundtrip | classical | pass | 2.384e-07 | <= 1.000e-05 | 0.9 |
| reversible | custom_autograd_grad_parity | classical | pass | 4.275e-07 | <= 1.000e-04 | 4.2 |
| reversible | symplectic_jacobian | classical | pass | 8.882e-16 | <= 1.000e-09 | 192.5 |
| reversible | energy_drift_separation | classical | pass | 1.010e-07 | <= 2.000e-01 | 190.9 |
| gauge | rotation_inverse_roundtrip | classical | pass | 2.384e-07 | <= 1.000e-05 | 1.6 |
| gauge | rotation_pairwise_norm_preservation | classical | pass | 2.384e-07 | <= 1.000e-05 | 1.5 |
| gauge | rotation_additivity_cumsum_law | classical | pass | 4.768e-07 | <= 1.000e-05 | 2.4 |
| gauge | zero_transport_reduces_to_standard_attention | reduction | pass | 0.000e+00 | <= 2.000e-05 | 3.1 |
| gauge | kv_decode_matches_full_forward | classical | pass | 5.960e-07 | <= 1.000e-04 | 23.7 |
| ultrametric | strong_triangle_inequality_lcp | classical | pass | 0.000e+00 | <= 0.000e+00 | 7.2 |
| ultrametric | trie_decode_matches_hard_kernel | reduction | pass | 1.788e-07 | <= 1.000e-03 | 12.4 |
| braid | ybe_law_holds | classical | pass | 8.882e-16 | <= 1.000e-10 | 0.7 |
| braid | restricted_law_violates_ybe | classical | pass | 4.267e+00 | >= 1.000e-03 | 0.4 |
| braid | payload_multiset_invariance | classical | pass | 0.000e+00 | <= 0.000e+00 | 0.5 |
| braid | rmatrix_braid_relation_holds | classical | pass | 1.421e-14 | <= 1.000e-10 | 0.9 |
| braid | rmatrix_inversion_relation_holds | classical | pass | 1.377e-14 | <= 1.000e-10 | 0.1 |
| braid | rmatrix_transfer_matrices_commute | classical | pass | 4.775e-17 | <= 1.000e-10 | 2.4 |
| braid | rmatrix_perturbed_transfer_separates | classical | pass | 3.356e-05 | >= 1.000e-06 | 0.7 |
| braid | rmatrix_mass_partition_charge_conserved | classical | pass | 1.887e-15 | <= 1.000e-05 | 1.2 |
| braid | heuristic_mass_partition_violated | classical | pass | 2.173e+00 | >= 1.000e-03 | 1.3 |
| simplicial | mass_conservation_two_hop | classical | pass | 1.192e-07 | <= 1.000e-05 | 1.0 |
| simplicial | zero_triangle_reduces_to_standard_attention | reduction | pass | 2.384e-07 | <= 2.000e-05 | 5.9 |
| simplicial | triplet_mass_conservation | classical | pass | 1.788e-07 | <= 1.000e-05 | 2.5 |
| simplicial | triplet_wedge_antisymmetry | classical | pass | 0.000e+00 | <= 1.000e-05 | 1.5 |
| fractal | router_branch_simplex | classical | pass | 1.192e-07 | <= 1.000e-06 | 1.7 |
| fractal | zero_ifs_reduces_to_prototype_attention | reduction | pass | 0.000e+00 | <= 1.000e-05 | 5.6 |
| fractal | ifs_moran_contractivity | classical | pass | 4.768e-07 | <= 1.000e-05 | 3.1 |
| surreal | row_norm_equals_exp_scale | classical | pass | 1.192e-07 | <= 1.000e-05 | 0.6 |
| surreal | layer_linearity | classical | pass | 1.669e-06 | <= 1.000e-04 | 1.4 |
| surreal | scale_shift_equivariance | classical | pass | 9.537e-07 | <= 1.000e-04 | 1.0 |
| surreal | zero_omega_reduces_to_standard_attention | reduction | pass | 2.384e-07 | <= 2.000e-05 | 1.6 |
| surreal | valuation_dominance_monotone | classical | pass | 0.000e+00 | <= 1.000e-05 | 2.3 |
