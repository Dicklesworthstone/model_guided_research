/-
Axiom hygiene gate (beads vnl.2 + vnl.3): `lake build` succeeding does NOT
prove the absence of `sorry` (sorry only warns). This file forces an axiom
audit of every headline lemma; CI greps its output for `sorryAx` and fails
on a match.
-/
import MGRProofs

-- Tranche 1 (vnl.2)
#print axioms MGRProofs.lse_max_sandwich
#print axioms MGRProofs.route_stability_pointwise
#print axioms MGRProofs.route_stability_unique_max
#print axioms MGRProofs.flat_error_sum
#print axioms MGRProofs.flat_error_prod

-- Tranche 2 (vnl.3)
#print axioms MGRProofs.no_infinite_descent
#print axioms MGRProofs.ordinal_termination
#print axioms MGRProofs.kick_jacobian_symplectic
#print axioms MGRProofs.symplectic_composition
#print axioms MGRProofs.gromov_root_eq_lcp
