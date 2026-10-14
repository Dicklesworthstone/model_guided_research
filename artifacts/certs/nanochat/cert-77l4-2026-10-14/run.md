# mgr certify

- run_id: `cert-77l4-2026-10-14` · seed: 0 · device: cpu · dtype: fp32
- git: `2ab2f85`
- result: **72 passed / 0 failed / 0 errored** in 3.4s

| Mechanism | Check | Family | Status | Measured | Tolerance | ms |
|---|---|---|---|---:|---:|---:|
| standard | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 14.7 |
| tropical | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 9.3 |
| ultrametric | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 8.6 |
| simplicial | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 7.9 |
| quaternion | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 10.6 |
| braid | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 7.5 |
| fractal | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 8.1 |
| octonion | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 19.9 |
| surreal | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 7.2 |
| reversible | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 264.8 |
| gauge | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 7.5 |
| clifford | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 15.0 |
| hyperbolic | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 10.8 |
| standard | rope_pairwise_norm_preservation | classical | pass | 4.768e-07 | <= 1.000e-05 | 1.0 |
| standard | causal_mask_structure | classical | pass | 0.000e+00 | <= 0.000e+00 | 0.2 |
| standard | rmsnorm_unit_rms | classical | pass | 1.788e-07 | <= 1.000e-03 | 0.9 |
| standard | softmax_row_stochastic | classical | pass | 5.960e-08 | <= 1.000e-06 | 0.9 |
| tropical | lipschitz_1_sup_norm_q | classical | pass | 9.954e-01 | <= 1.000e+00 | 1.2 |
| tropical | lipschitz_1_sup_norm_v | classical | pass | 9.917e-01 | <= 1.000e+00 | 1.2 |
| tropical | score_center_pure_gauge_shift | classical | pass | 9.537e-07 | <= 1.000e-05 | 1.2 |
| tropical | margin_matches_bruteforce | classical | pass | 0.000e+00 | <= 1.000e-06 | 1.4 |
| tropical | maslov_endpoint_within_sandwich | reduction | pass | 1.148e-01 | <= 1.000e+00 | 1.5 |
| tropical | ffn_lipschitz_1_sup_norm | classical | pass | 1.000e+00 | <= 1.000e+00 | 1.7 |
| tropical | ffn_collapse_single_layer | classical | pass | 0.000e+00 | <= 1.000e-09 | 3.3 |
| quaternion | qmul_associativity | classical | pass | 7.105e-15 | <= 1.000e-10 | 1.5 |
| quaternion | qmul_norm_multiplicative | classical | pass | 1.776e-15 | <= 1.000e-10 | 1.4 |
| quaternion | qconj_antihomomorphism | classical | pass | 8.882e-16 | <= 1.000e-10 | 1.3 |
| quaternion | rotor_norm_preservation | classical | pass | 8.882e-16 | <= 1.000e-10 | 1.2 |
| clifford | cgp_associativity | classical | pass | 2.132e-14 | <= 1.000e-10 | 3.0 |
| clifford | reversion_antihomomorphism | classical | pass | 3.553e-15 | <= 1.000e-10 | 2.1 |
| clifford | rotor_norm_preservation | classical | pass | 2.137e-16 | <= 1.000e-09 | 5.9 |
| clifford | quaternion_subalgebra_reduction | reduction | pass | 1.776e-15 | <= 1.000e-10 | 1.4 |
| hyperbolic | lorentz_constraint_residual | classical | pass | 8.762e-06 | <= 2.000e-05 | 1.0 |
| hyperbolic | exp_log_origin_roundtrip | classical | pass | 2.980e-08 | <= 2.000e-05 | 1.1 |
| hyperbolic | energy_gromov_reduces_to_standard | reduction | pass | 1.621e-05 | <= 2.000e-05 | 5.1 |
| octonion | omul_norm_multiplicative | classical | pass | 5.329e-15 | <= 1.000e-09 | 1.9 |
| octonion | omul_alternativity | classical | pass | 1.421e-14 | <= 1.000e-09 | 5.7 |
| octonion | omul_nonassociativity_witness | classical | pass | 6.635e+01 | >= 1.000e-02 | 3.1 |
| octonion | o_times_conj_is_norm_squared | classical | pass | 7.105e-15 | <= 1.000e-09 | 1.6 |
| octonion | reduces_to_quaternion_on_subalgebra | reduction | pass | 0.000e+00 | <= 1.000e-09 | 1.7 |
| reversible | forward_inverse_roundtrip | classical | pass | 2.384e-07 | <= 1.000e-05 | 2.7 |
| reversible | custom_autograd_grad_parity | classical | pass | 3.821e-07 | <= 1.000e-04 | 9.4 |
| reversible | symplectic_jacobian | classical | pass | 6.661e-16 | <= 1.000e-09 | 492.9 |
| reversible | energy_drift_separation | classical | pass | 7.161e-08 | <= 2.000e-01 | 449.1 |
| gauge | rotation_inverse_roundtrip | classical | pass | 2.384e-07 | <= 1.000e-05 | 3.8 |
| gauge | rotation_pairwise_norm_preservation | classical | pass | 2.384e-07 | <= 1.000e-05 | 2.7 |
| gauge | rotation_additivity_cumsum_law | classical | pass | 3.576e-07 | <= 1.000e-05 | 3.0 |
| gauge | zero_transport_reduces_to_standard_attention | reduction | pass | 0.000e+00 | <= 2.000e-05 | 3.4 |
| gauge | kv_decode_matches_full_forward | classical | pass | 6.557e-07 | <= 1.000e-04 | 32.0 |
| ultrametric | strong_triangle_inequality_lcp | classical | pass | 0.000e+00 | <= 0.000e+00 | 26.2 |
| ultrametric | trie_decode_matches_hard_kernel | reduction | pass | 2.980e-07 | <= 1.000e-03 | 36.6 |
| braid | ybe_law_holds | classical | pass | 8.882e-16 | <= 1.000e-10 | 1.2 |
| braid | restricted_law_violates_ybe | classical | pass | 4.681e+00 | >= 1.000e-03 | 0.9 |
| braid | payload_multiset_invariance | classical | pass | 0.000e+00 | <= 0.000e+00 | 1.0 |
| braid | rmatrix_braid_relation_holds | classical | pass | 5.329e-15 | <= 1.000e-10 | 0.9 |
| braid | rmatrix_inversion_relation_holds | classical | pass | 9.104e-15 | <= 1.000e-10 | 0.2 |
| braid | rmatrix_transfer_matrices_commute | classical | pass | 5.602e-17 | <= 1.000e-10 | 1.9 |
| braid | rmatrix_perturbed_transfer_separates | classical | pass | 2.682e-05 | >= 1.000e-06 | 1.4 |
| braid | rmatrix_mass_partition_charge_conserved | classical | pass | 1.776e-15 | <= 1.000e-05 | 4.1 |
| braid | heuristic_mass_partition_violated | classical | pass | 2.662e+00 | >= 1.000e-03 | 5.5 |
| simplicial | mass_conservation_two_hop | classical | pass | 1.192e-07 | <= 1.000e-05 | 3.2 |
| simplicial | zero_triangle_reduces_to_standard_attention | reduction | pass | 1.192e-07 | <= 2.000e-05 | 7.3 |
| simplicial | triplet_mass_conservation | classical | pass | 1.788e-07 | <= 1.000e-05 | 2.9 |
| simplicial | triplet_wedge_antisymmetry | classical | pass | 0.000e+00 | <= 1.000e-05 | 2.1 |
| fractal | router_branch_simplex | classical | pass | 1.192e-07 | <= 1.000e-06 | 2.7 |
| fractal | zero_ifs_reduces_to_prototype_attention | reduction | pass | 0.000e+00 | <= 1.000e-05 | 6.9 |
| fractal | ifs_moran_contractivity | classical | pass | 4.768e-07 | <= 1.000e-05 | 5.9 |
| surreal | row_norm_equals_exp_scale | classical | pass | 2.384e-07 | <= 1.000e-05 | 1.1 |
| surreal | layer_linearity | classical | pass | 2.861e-06 | <= 1.000e-04 | 1.9 |
| surreal | scale_shift_equivariance | classical | pass | 1.907e-06 | <= 1.000e-04 | 1.9 |
| surreal | zero_omega_reduces_to_standard_attention | reduction | pass | 2.384e-07 | <= 2.000e-05 | 2.5 |
| surreal | valuation_dominance_monotone | classical | pass | 0.000e+00 | <= 1.000e-05 | 2.7 |
