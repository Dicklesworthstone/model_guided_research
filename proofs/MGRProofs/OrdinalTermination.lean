/-
Copyright (c) 2026 model_guided_research. All rights reserved.
Formalization tranche 2a (ABSTRACT CORE) of bead model_guided_research-vnl.3.

DELIVERED HERE (machine-checked, zero sorry):
- no_infinite_descent / ordinal_termination: the ABSTRACT core of
  thm-ordinal-termination (lab.3's guarantee) - any rank-decreasing step
  dynamics admits no infinite orbit. This is the orchestrator SPEC form.
- Event alphabet docstring below: reference contract for lab.3's Python
  implementation, sync-checked by tests/test_lean_spec_sync.py.

DEFERRED (tracked on the bead): the Cantor-normal-form DECREASE lemmas
(cnfRank_dec_A/B/C over rho = omega^2*A + omega*B + C). Blocking mathlib-API
gap: ordinals provide AddLeftStrictMono but not AddRightStrictMono, and
Nat-cast sums do not split under mul_add without a manual cast lemma -
both documented on the bead with candidate proof skeletons.

EVENT ALPHABET (reference; sync-checked):
- evt: PHASE_ADVANCE
- evt: WITHIN_PHASE_RETRY
- evt: PHASE_ESCALATE

CONVENTIONS (proofs/README): zero `sorry` policy.
-/
import Mathlib

open Ordinal

namespace MGRProofs

/-! ### Abstract termination -/

/-- **No infinite descent in the ordinals** (abstract core of
thm-ordinal-termination): there is no `f : Nat -> Ordinal` that strictly
decreases at every step. Ordinals are well-founded, so any rank-decreasing
dynamics must terminate. -/
theorem no_infinite_descent {f : ℕ → Ordinal} (h : ∀ n, f (n + 1) < f n) :
    False := by
  have key : ∀ o : Ordinal, ∀ g : ℕ → Ordinal, g 0 = o →
      (∀ n, g (n + 1) < g n) → False := by
    intro o
    induction o using WellFoundedLT.induction with
    | ind o IH =>
        intro g hg0 hstep
        have h1 : g 1 < o := by rw [← hg0]; exact hstep 0
        exact IH (g 1) h1 (fun n => g (n + 1)) rfl fun n => hstep (n + 1)
  exact key (f 0) f rfl h

/-- **Ordinal termination, event-system form** (thm-ordinal-termination,
abstract form; the SPEC for lab.3's orchestrator). If every transition of a
step dynamics strictly decreases a rank in the ordinals, then no infinite
orbit starts anywhere: any claimed orbit `f` yields a strictly-descending
rank sequence, contradicting `no_infinite_descent`. -/
theorem ordinal_termination {S : Type} (step : S → S) (rho : S → Ordinal)
    (hdec : ∀ s, rho (step s) < rho s) (s0 : S) :
    ¬ ∃ f : ℕ → S, f 0 = s0 ∧ ∀ n, f (n + 1) = step (f n) := by
  rintro ⟨f, hf0, hf⟩
  refine no_infinite_descent (f := fun n => rho (f n)) ?_
  intro n
  rw [hf n]
  exact hdec _

end MGRProofs
