# mgr certify

- run_id: `cert-ci-v7-a5abe77` · seed: 42 · device: cpu · dtype: fp32
- git: `a5abe77`
- result: **72 passed / 0 failed / 0 errored** in 1.7s

| Mechanism | Check | Family | Status | Measured | Tolerance | ms |
|---|---|---|---|---:|---:|---:|
| standard | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 19.6 |
| tropical | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 3.9 |
| ultrametric | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 3.8 |
| simplicial | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 3.2 |
| quaternion | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 4.5 |
| braid | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 2.3 |
| fractal | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 4.3 |
| octonion | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 7.5 |
| surreal | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 4.0 |
| reversible | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 280.7 |
| gauge | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 3.8 |
| clifford | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 9.5 |
| hyperbolic | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 5.6 |
| standard | rope_pairwise_norm_preservation | classical | pass | 2.384e-07 | <= 1.000e-05 | 0.8 |
| standard | causal_mask_structure | classical | pass | 0.000e+00 | <= 0.000e+00 | 1.1 |
| standard | rmsnorm_unit_rms | classical | pass | 1.788e-07 | <= 1.000e-03 | 0.6 |
| standard | softmax_row_stochastic | classical | pass | 1.192e-07 | <= 1.000e-06 | 1.0 |
| tropical | lipschitz_1_sup_norm_q | classical | pass | 9.886e-01 | <= 1.000e+00 | 1.5 |
| tropical | lipschitz_1_sup_norm_v | classical | pass | 9.886e-01 | <= 1.000e+00 | 1.4 |
| tropical | score_center_pure_gauge_shift | classical | pass | 9.537e-07 | <= 1.000e-05 | 1.4 |
| tropical | margin_matches_bruteforce | classical | pass | 0.000e+00 | <= 1.000e-06 | 1.4 |
| tropical | maslov_endpoint_within_sandwich | reduction | pass | 1.398e-01 | <= 1.000e+00 | 1.3 |
| tropical | ffn_lipschitz_1_sup_norm | classical | pass | 9.528e-01 | <= 1.000e+00 | 0.9 |
| tropical | ffn_collapse_single_layer | classical | pass | 2.220e-16 | <= 1.000e-09 | 0.7 |
| quaternion | qmul_associativity | classical | pass | 7.105e-15 | <= 1.000e-10 | 0.8 |
| quaternion | qmul_norm_multiplicative | classical | pass | 2.665e-15 | <= 1.000e-10 | 0.8 |
| quaternion | qconj_antihomomorphism | classical | pass | 8.882e-16 | <= 1.000e-10 | 0.7 |
| quaternion | rotor_norm_preservation | classical | pass | 8.882e-16 | <= 1.000e-10 | 0.5 |
| clifford | cgp_associativity | classical | pass | 1.421e-14 | <= 1.000e-10 | 1.0 |
| clifford | reversion_antihomomorphism | classical | pass | 3.553e-15 | <= 1.000e-10 | 0.7 |
| clifford | rotor_norm_preservation | classical | pass | 3.847e-16 | <= 1.000e-09 | 1.9 |
| clifford | quaternion_subalgebra_reduction | reduction | pass | 8.882e-16 | <= 1.000e-10 | 0.7 |
| hyperbolic | lorentz_constraint_residual | classical | pass | 8.762e-06 | <= 2.000e-05 | 0.5 |
| hyperbolic | exp_log_origin_roundtrip | classical | pass | 2.980e-08 | <= 2.000e-05 | 0.6 |
| hyperbolic | energy_gromov_reduces_to_standard | reduction | pass | 1.621e-05 | <= 2.000e-05 | 2.7 |
| octonion | omul_norm_multiplicative | classical | pass | 7.105e-15 | <= 1.000e-09 | 0.9 |
| octonion | omul_alternativity | classical | pass | 1.421e-14 | <= 1.000e-09 | 2.1 |
| octonion | omul_nonassociativity_witness | classical | pass | 5.714e+01 | >= 1.000e-02 | 1.3 |
| octonion | o_times_conj_is_norm_squared | classical | pass | 7.105e-15 | <= 1.000e-09 | 0.8 |
| octonion | reduces_to_quaternion_on_subalgebra | reduction | pass | 0.000e+00 | <= 1.000e-09 | 0.8 |
| reversible | forward_inverse_roundtrip | classical | pass | 2.384e-07 | <= 1.000e-05 | 1.2 |
| reversible | custom_autograd_grad_parity | classical | pass | 4.275e-07 | <= 1.000e-04 | 3.2 |
| reversible | symplectic_jacobian | classical | pass | 8.882e-16 | <= 1.000e-09 | 196.0 |
| reversible | energy_drift_separation | classical | pass | 1.010e-07 | <= 2.000e-01 | 204.6 |
| gauge | rotation_inverse_roundtrip | classical | pass | 2.384e-07 | <= 1.000e-05 | 1.6 |
| gauge | rotation_pairwise_norm_preservation | classical | pass | 2.384e-07 | <= 1.000e-05 | 1.2 |
| gauge | rotation_additivity_cumsum_law | classical | pass | 4.768e-07 | <= 1.000e-05 | 1.5 |
| gauge | zero_transport_reduces_to_standard_attention | reduction | pass | 0.000e+00 | <= 2.000e-05 | 1.8 |
| gauge | kv_decode_matches_full_forward | classical | pass | 5.960e-07 | <= 1.000e-04 | 22.8 |
| ultrametric | strong_triangle_inequality_lcp | classical | pass | 0.000e+00 | <= 0.000e+00 | 5.7 |
| ultrametric | trie_decode_matches_hard_kernel | reduction | pass | 1.788e-07 | <= 1.000e-03 | 12.5 |
| braid | ybe_law_holds | classical | pass | 8.882e-16 | <= 1.000e-10 | 0.8 |
| braid | restricted_law_violates_ybe | classical | pass | 4.267e+00 | >= 1.000e-03 | 0.5 |
| braid | payload_multiset_invariance | classical | pass | 0.000e+00 | <= 0.000e+00 | 0.5 |
| braid | rmatrix_braid_relation_holds | classical | pass | 1.421e-14 | <= 1.000e-10 | 0.4 |
| braid | rmatrix_inversion_relation_holds | classical | pass | 1.377e-14 | <= 1.000e-10 | 0.1 |
| braid | rmatrix_transfer_matrices_commute | classical | pass | 4.775e-17 | <= 1.000e-10 | 0.8 |
| braid | rmatrix_perturbed_transfer_separates | classical | pass | 3.356e-05 | >= 1.000e-06 | 0.6 |
| braid | rmatrix_mass_partition_charge_conserved | classical | pass | 1.887e-15 | <= 1.000e-05 | 1.8 |
| braid | heuristic_mass_partition_violated | classical | pass | 2.173e+00 | >= 1.000e-03 | 1.4 |
| simplicial | mass_conservation_two_hop | classical | pass | 1.192e-07 | <= 1.000e-05 | 0.8 |
| simplicial | zero_triangle_reduces_to_standard_attention | reduction | pass | 2.384e-07 | <= 2.000e-05 | 3.7 |
| simplicial | triplet_mass_conservation | classical | pass | 1.788e-07 | <= 1.000e-05 | 1.9 |
| simplicial | triplet_wedge_antisymmetry | classical | pass | 0.000e+00 | <= 1.000e-05 | 0.8 |
| fractal | router_branch_simplex | classical | pass | 1.192e-07 | <= 1.000e-06 | 1.7 |
| fractal | zero_ifs_reduces_to_prototype_attention | reduction | pass | 0.000e+00 | <= 1.000e-05 | 4.8 |
| fractal | ifs_moran_contractivity | classical | pass | 4.768e-07 | <= 1.000e-05 | 3.3 |
| surreal | row_norm_equals_exp_scale | classical | pass | 1.192e-07 | <= 1.000e-05 | 0.5 |
| surreal | layer_linearity | classical | pass | 1.669e-06 | <= 1.000e-04 | 1.5 |
| surreal | scale_shift_equivariance | classical | pass | 9.537e-07 | <= 1.000e-04 | 1.2 |
| surreal | zero_omega_reduces_to_standard_attention | reduction | pass | 2.384e-07 | <= 2.000e-05 | 1.8 |
| surreal | valuation_dominance_monotone | classical | pass | 0.000e+00 | <= 1.000e-05 | 2.5 |
