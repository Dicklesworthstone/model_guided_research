/-
Axiom hygiene gate (bead vnl.2): `lake build` succeeding does NOT prove the
absence of `sorry` (sorry only warns). This file forces an axiom audit of
every headline lemma; CI greps its output for `sorryAx` and fails on a match.
-/
import MGRProofs

#print axioms MGRProofs.lse_max_sandwich
#print axioms MGRProofs.route_stability_pointwise
#print axioms MGRProofs.route_stability_unique_max
#print axioms MGRProofs.flat_error_sum
#print axioms MGRProofs.flat_error_prod
