# mgr certify

- run_id: `20261123_020005` · seed: 42 · device: cuda · dtype: fp32
- git: `95e8d47`
- result: **72 passed / 0 failed / 0 errored** in 76.2s

| Mechanism | Check | Family | Status | Measured | Tolerance | ms |
|---|---|---|---|---:|---:|---:|
| standard | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 7774.8 |
| tropical | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 2988.4 |
| ultrametric | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 1431.5 |
| simplicial | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 359.8 |
| quaternion | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 724.8 |
| braid | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 281.4 |
| fractal | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 43.1 |
| octonion | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 247.4 |
| surreal | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 268.4 |
| reversible | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 3500.6 |
| gauge | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 77.3 |
| clifford | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 138.3 |
| hyperbolic | causality_no_future_grad | causality | pass | 0.000e+00 | <= 1.000e-12 | 26950.9 |
| standard | rope_pairwise_norm_preservation | classical | pass | 2.384e-07 | <= 1.000e-05 | 22.1 |
| standard | causal_mask_structure | classical | pass | 0.000e+00 | <= 0.000e+00 | 4.6 |
| standard | rmsnorm_unit_rms | classical | pass | 1.788e-07 | <= 1.000e-03 | 6.4 |
| standard | softmax_row_stochastic | classical | pass | 1.192e-07 | <= 1.000e-06 | 3.7 |
| tropical | lipschitz_1_sup_norm_q | classical | pass | 9.903e-01 | <= 1.000e+00 | 56.0 |
| tropical | lipschitz_1_sup_norm_v | classical | pass | 9.719e-01 | <= 1.000e+00 | 6.2 |
| tropical | score_center_pure_gauge_shift | classical | pass | 9.537e-07 | <= 1.000e-05 | 3.9 |
| tropical | margin_matches_bruteforce | classical | pass | 0.000e+00 | <= 1.000e-06 | 207.4 |
| tropical | maslov_endpoint_within_sandwich | reduction | pass | 1.666e-01 | <= 1.000e+00 | 151.5 |
| tropical | ffn_lipschitz_1_sup_norm | classical | pass | 9.528e-01 | <= 1.000e+00 | 26.6 |
| tropical | ffn_collapse_single_layer | classical | pass | 2.220e-16 | <= 1.000e-09 | 28.2 |
| quaternion | qmul_associativity | classical | pass | 3.553e-15 | <= 1.000e-10 | 15.9 |
| quaternion | qmul_norm_multiplicative | classical | pass | 1.776e-15 | <= 1.000e-10 | 9.8 |
| quaternion | qconj_antihomomorphism | classical | pass | 8.882e-16 | <= 1.000e-10 | 5.4 |
| quaternion | rotor_norm_preservation | classical | pass | 8.882e-16 | <= 1.000e-10 | 8.1 |
| clifford | cgp_associativity | classical | pass | 1.066e-14 | <= 1.000e-10 | 326.8 |
| clifford | reversion_antihomomorphism | classical | pass | 3.553e-15 | <= 1.000e-10 | 6.2 |
| clifford | rotor_norm_preservation | classical | pass | 2.157e-16 | <= 1.000e-09 | 204.0 |
| clifford | quaternion_subalgebra_reduction | reduction | pass | 8.882e-16 | <= 1.000e-10 | 15.2 |
| hyperbolic | lorentz_constraint_residual | classical | pass | 4.172e-06 | <= 2.000e-05 | 11.8 |
| hyperbolic | exp_log_origin_roundtrip | classical | pass | 2.980e-08 | <= 2.000e-05 | 60.2 |
| hyperbolic | energy_gromov_reduces_to_standard | reduction | pass | 1.788e-05 | <= 2.000e-05 | 33.0 |
| octonion | omul_norm_multiplicative | classical | pass | 5.329e-15 | <= 1.000e-09 | 6.8 |
| octonion | omul_alternativity | classical | pass | 1.421e-14 | <= 1.000e-09 | 411.6 |
| octonion | omul_nonassociativity_witness | classical | pass | 5.920e+01 | >= 1.000e-02 | 28.2 |
| octonion | o_times_conj_is_norm_squared | classical | pass | 7.105e-15 | <= 1.000e-09 | 12.7 |
| octonion | reduces_to_quaternion_on_subalgebra | reduction | pass | 0.000e+00 | <= 1.000e-09 | 13.8 |
| reversible | forward_inverse_roundtrip | classical | pass | 1.192e-07 | <= 1.000e-05 | 19.0 |
| reversible | custom_autograd_grad_parity | classical | pass | 4.064e-07 | <= 1.000e-04 | 108.1 |
| reversible | symplectic_jacobian | classical | pass | 6.661e-16 | <= 1.000e-09 | 4105.0 |
| reversible | energy_drift_separation | classical | pass | 1.054e-07 | <= 2.000e-01 | 4404.3 |
| gauge | rotation_inverse_roundtrip | classical | pass | 2.384e-07 | <= 1.000e-05 | 47.2 |
| gauge | rotation_pairwise_norm_preservation | classical | pass | 2.384e-07 | <= 1.000e-05 | 22.8 |
| gauge | rotation_additivity_cumsum_law | classical | pass | 3.576e-07 | <= 1.000e-05 | 44.9 |
| gauge | zero_transport_reduces_to_standard_attention | reduction | pass | 0.000e+00 | <= 2.000e-05 | 30.0 |
| gauge | kv_decode_matches_full_forward | classical | pass | 3.576e-07 | <= 1.000e-04 | 934.9 |
| ultrametric | strong_triangle_inequality_lcp | classical | pass | 0.000e+00 | <= 0.000e+00 | 250.9 |
| ultrametric | trie_decode_matches_hard_kernel | reduction | pass | 2.682e-07 | <= 1.000e-03 | 671.5 |
| braid | ybe_law_holds | classical | pass | 8.882e-16 | <= 1.000e-10 | 12.9 |
| braid | restricted_law_violates_ybe | classical | pass | 4.432e+00 | >= 1.000e-03 | 4.0 |
| braid | payload_multiset_invariance | classical | pass | 0.000e+00 | <= 0.000e+00 | 6.3 |
| braid | rmatrix_braid_relation_holds | classical | pass | 1.421e-14 | <= 1.000e-10 | 7.0 |
| braid | rmatrix_inversion_relation_holds | classical | pass | 1.377e-14 | <= 1.000e-10 | 0.6 |
| braid | rmatrix_transfer_matrices_commute | classical | pass | 4.997e-17 | <= 1.000e-10 | 49.4 |
| braid | rmatrix_perturbed_transfer_separates | classical | pass | 3.356e-05 | >= 1.000e-06 | 8.6 |
| braid | rmatrix_mass_partition_charge_conserved | classical | pass | 1.776e-15 | <= 1.000e-05 | 144.2 |
| braid | heuristic_mass_partition_violated | classical | pass | 2.173e+00 | >= 1.000e-03 | 171.2 |
| simplicial | mass_conservation_two_hop | classical | pass | 5.960e-08 | <= 1.000e-05 | 15.1 |
| simplicial | zero_triangle_reduces_to_standard_attention | reduction | pass | 5.364e-07 | <= 2.000e-05 | 229.3 |
| simplicial | triplet_mass_conservation | classical | pass | 1.192e-07 | <= 1.000e-05 | 39.6 |
| simplicial | triplet_wedge_antisymmetry | classical | pass | 0.000e+00 | <= 1.000e-05 | 11.7 |
| fractal | router_branch_simplex | classical | pass | 1.192e-07 | <= 1.000e-06 | 6.9 |
| fractal | zero_ifs_reduces_to_prototype_attention | reduction | pass | 0.000e+00 | <= 1.000e-05 | 585.0 |
| fractal | ifs_moran_contractivity | classical | pass | 4.768e-07 | <= 1.000e-05 | 50.7 |
| surreal | row_norm_equals_exp_scale | classical | pass | 1.192e-07 | <= 1.000e-05 | 9.9 |
| surreal | layer_linearity | classical | pass | 1.550e-06 | <= 1.000e-04 | 9.4 |
| surreal | scale_shift_equivariance | classical | pass | 1.431e-06 | <= 1.000e-04 | 4.5 |
| surreal | zero_omega_reduces_to_standard_attention | reduction | pass | 7.153e-07 | <= 2.000e-05 | 22.2 |
| surreal | valuation_dominance_monotone | classical | pass | 0.000e+00 | <= 1.000e-05 | 52.8 |
