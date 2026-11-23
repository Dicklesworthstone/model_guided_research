# mgr certify

- run_id: `20261123_020139` · seed: 42 · device: cpu · dtype: fp32
- git: `95e8d47`
- result: **72 passed / 0 failed / 0 errored** in 44.4s

| Mechanism | Check | Family | Status | Measured | Tolerance | ms |
|---|---|---|---|---:|---:|---:|
| standard | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 805.4 |
| tropical | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 415.3 |
| ultrametric | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 92.6 |
| simplicial | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 315.4 |
| quaternion | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 524.7 |
| braid | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 1171.6 |
| fractal | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 682.9 |
| octonion | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 505.3 |
| surreal | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 527.9 |
| reversible | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 3402.7 |
| gauge | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 258.1 |
| clifford | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 362.4 |
| hyperbolic | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 444.5 |
| standard | rope_pairwise_norm_preservation | classical | pass | 2.384e-07 | <= 1.000e-05 | 13.2 |
| standard | causal_mask_structure | classical | pass | 0.000e+00 | <= 0.000e+00 | 54.3 |
| standard | rmsnorm_unit_rms | classical | pass | 1.788e-07 | <= 1.000e-03 | 15.0 |
| standard | softmax_row_stochastic | classical | pass | 1.192e-07 | <= 1.000e-06 | 182.5 |
| tropical | lipschitz_1_sup_norm_q | classical | pass | 9.886e-01 | <= 1.000e+00 | 24.3 |
| tropical | lipschitz_1_sup_norm_v | classical | pass | 9.886e-01 | <= 1.000e+00 | 76.4 |
| tropical | score_center_pure_gauge_shift | classical | pass | 4.768e-07 | <= 1.000e-05 | 66.4 |
| tropical | margin_matches_bruteforce | classical | pass | 0.000e+00 | <= 1.000e-06 | 86.5 |
| tropical | maslov_endpoint_within_sandwich | reduction | pass | 1.398e-01 | <= 1.000e+00 | 24.1 |
| tropical | ffn_lipschitz_1_sup_norm | classical | pass | 9.528e-01 | <= 1.000e+00 | 13.5 |
| tropical | ffn_collapse_single_layer | classical | pass | 2.220e-16 | <= 1.000e-09 | 13.6 |
| quaternion | qmul_associativity | classical | pass | 7.105e-15 | <= 1.000e-10 | 5.9 |
| quaternion | qmul_norm_multiplicative | classical | pass | 1.776e-15 | <= 1.000e-10 | 2.6 |
| quaternion | qconj_antihomomorphism | classical | pass | 8.882e-16 | <= 1.000e-10 | 2.0 |
| quaternion | rotor_norm_preservation | classical | pass | 1.332e-15 | <= 1.000e-10 | 2.1 |
| clifford | cgp_associativity | classical | pass | 1.066e-14 | <= 1.000e-10 | 66.6 |
| clifford | reversion_antihomomorphism | classical | pass | 3.553e-15 | <= 1.000e-10 | 48.8 |
| clifford | rotor_norm_preservation | classical | pass | 3.847e-16 | <= 1.000e-09 | 22.8 |
| clifford | quaternion_subalgebra_reduction | reduction | pass | 8.882e-16 | <= 1.000e-10 | 9.5 |
| hyperbolic | lorentz_constraint_residual | classical | pass | 8.762e-06 | <= 2.000e-05 | 5.7 |
| hyperbolic | exp_log_origin_roundtrip | classical | pass | 2.980e-08 | <= 2.000e-05 | 5.9 |
| hyperbolic | energy_gromov_reduces_to_standard | reduction | pass | 1.621e-05 | <= 2.000e-05 | 143.5 |
| octonion | omul_norm_multiplicative | classical | pass | 7.105e-15 | <= 1.000e-09 | 11.0 |
| octonion | omul_alternativity | classical | pass | 1.421e-14 | <= 1.000e-09 | 9.1 |
| octonion | omul_nonassociativity_witness | classical | pass | 5.714e+01 | >= 1.000e-02 | 4.7 |
| octonion | o_times_conj_is_norm_squared | classical | pass | 5.329e-15 | <= 1.000e-09 | 3.2 |
| octonion | reduces_to_quaternion_on_subalgebra | reduction | pass | 0.000e+00 | <= 1.000e-09 | 4.2 |
| reversible | forward_inverse_roundtrip | classical | pass | 2.384e-07 | <= 1.000e-05 | 86.0 |
| reversible | custom_autograd_grad_parity | classical | pass | 2.660e-07 | <= 1.000e-04 | 222.2 |
| reversible | symplectic_jacobian | classical | pass | 6.661e-16 | <= 1.000e-09 | 13290.5 |
| reversible | energy_drift_separation | classical | pass | 1.010e-07 | <= 2.000e-01 | 2394.9 |
| gauge | rotation_inverse_roundtrip | classical | pass | 2.384e-07 | <= 1.000e-05 | 8.8 |
| gauge | rotation_pairwise_norm_preservation | classical | pass | 4.768e-07 | <= 1.000e-05 | 3.5 |
| gauge | rotation_additivity_cumsum_law | classical | pass | 2.608e-07 | <= 1.000e-05 | 7.0 |
| gauge | zero_transport_reduces_to_standard_attention | reduction | pass | 0.000e+00 | <= 2.000e-05 | 11.7 |
| gauge | kv_decode_matches_full_forward | classical | pass | 4.768e-07 | <= 1.000e-04 | 749.5 |
| ultrametric | strong_triangle_inequality_lcp | classical | pass | 0.000e+00 | <= 0.000e+00 | 51.1 |
| ultrametric | trie_decode_matches_hard_kernel | reduction | pass | 2.682e-07 | <= 1.000e-03 | 286.9 |
| braid | ybe_law_holds | classical | pass | 8.882e-16 | <= 1.000e-10 | 5.5 |
| braid | restricted_law_violates_ybe | classical | pass | 4.267e+00 | >= 1.000e-03 | 2.9 |
| braid | payload_multiset_invariance | classical | pass | 0.000e+00 | <= 0.000e+00 | 2.6 |
| braid | rmatrix_braid_relation_holds | classical | pass | 1.421e-14 | <= 1.000e-10 | 2.9 |
| braid | rmatrix_inversion_relation_holds | classical | pass | 1.377e-14 | <= 1.000e-10 | 0.5 |
| braid | rmatrix_transfer_matrices_commute | classical | pass | 4.997e-17 | <= 1.000e-10 | 5.5 |
| braid | rmatrix_perturbed_transfer_separates | classical | pass | 3.356e-05 | >= 1.000e-06 | 2.9 |
| braid | rmatrix_mass_partition_charge_conserved | classical | pass | 1.776e-15 | <= 1.000e-05 | 13.9 |
| braid | heuristic_mass_partition_violated | classical | pass | 2.173e+00 | >= 1.000e-03 | 6.3 |
| simplicial | mass_conservation_two_hop | classical | pass | 1.192e-07 | <= 1.000e-05 | 1.8 |
| simplicial | zero_triangle_reduces_to_standard_attention | reduction | pass | 2.384e-07 | <= 2.000e-05 | 14.6 |
| simplicial | triplet_mass_conservation | classical | pass | 1.788e-07 | <= 1.000e-05 | 6.8 |
| simplicial | triplet_wedge_antisymmetry | classical | pass | 0.000e+00 | <= 1.000e-05 | 92.7 |
| fractal | router_branch_simplex | classical | pass | 1.192e-07 | <= 1.000e-06 | 55.4 |
| fractal | zero_ifs_reduces_to_prototype_attention | reduction | pass | 0.000e+00 | <= 1.000e-05 | 307.0 |
| fractal | ifs_moran_contractivity | classical | pass | 4.768e-07 | <= 1.000e-05 | 563.6 |
| surreal | row_norm_equals_exp_scale | classical | pass | 2.384e-07 | <= 1.000e-05 | 4.9 |
| surreal | layer_linearity | classical | pass | 3.338e-06 | <= 1.000e-04 | 18.0 |
| surreal | scale_shift_equivariance | classical | pass | 1.907e-06 | <= 1.000e-04 | 16.0 |
| surreal | zero_omega_reduces_to_standard_attention | reduction | pass | 2.384e-07 | <= 2.000e-05 | 56.2 |
| surreal | valuation_dominance_monotone | classical | pass | 0.000e+00 | <= 1.000e-05 | 61.3 |
