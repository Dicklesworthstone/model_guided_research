/-
Copyright (c) 2026 model_guided_research. All rights reserved.
Formalization tranche 2a of bead model_guided_research-vnl.3 + bead model_guided_research-mxo3.

DELIVERED HERE (machine-checked, zero sorry):
- no_infinite_descent / ordinal_termination: the ABSTRACT core of
  thm-ordinal-termination (lab.3's guarantee) - any rank-decreasing step
  dynamics admits no infinite orbit. This is the orchestrator SPEC form.
- cnfRank / cnfRank_dec_C / cnfRank_dec_B / cnfRank_dec_A: the Cantor-normal-form
  rank `ρ = ω^2 * A + ω * B + C` and its three strict-decrease lemmas across
  patience ticks (`C + 1 → C`), anneal steps (`B + 1 → B` with arbitrary `C`
  reset), and macro restarts (`A + 1 → A` with arbitrary `B, C` resets).
- Event alphabet docstring below: reference contract for lab.3's Python
  implementation, sync-checked by tests/test_lean_spec_sync.py.

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

/-! ### Cantor Normal Form Rank Decrease Lemmas (mxo3 / vnl.3) -/

/-- Cantor Normal Form rank `ρ(A, B, C) = ω^2 * A + ω * B + C` for a 3-counter
anneal/restart schedule (`A` macro restarts, `B` anneal reductions, `C` patience steps). -/
noncomputable def cnfRank (A B C : ℕ) : Ordinal :=
  ω ^ (2 : ℕ) * (A : Ordinal) + ω * (B : Ordinal) + (C : Ordinal)

/-- Helper: casting `n + 1 : ℕ` to `Ordinal` splits as `(n : Ordinal) + 1`. -/
theorem natCast_succ_eq_add_one (n : ℕ) : ((n + 1 : ℕ) : Ordinal) = (n : Ordinal) + 1 :=
  Nat.cast_succ n

/-- **Patience-step decrease** (`WITHIN_PHASE_RETRY` / patience tick):
decrementing `C + 1 → C` while holding `A, B` fixed strictly decreases `cnfRank`. -/
theorem cnfRank_dec_C (A B C : ℕ) :
    cnfRank A B C < cnfRank A B (C + 1) := by
  unfold cnfRank
  have hC : (C : Ordinal) < ((C + 1 : ℕ) : Ordinal) :=
    Nat.cast_lt.mpr (Nat.lt_succ_self C)
  exact add_lt_add_right hC (ω ^ (2 : ℕ) * (A : Ordinal) + ω * (B : Ordinal))

/-- **Anneal-step decrease** (`PHASE_ADVANCE` / LR reduction):
decrementing `B + 1 → B` while holding `A` fixed strictly decreases `cnfRank`
for ANY reset patience counter `C : ℕ`, because `(C : Ordinal) < ω`. -/
theorem cnfRank_dec_B (A B C C' : ℕ) :
    cnfRank A B C < cnfRank A (B + 1) C' := by
  unfold cnfRank
  rw [add_assoc, add_assoc]
  refine add_lt_add_right ?_ (ω ^ (2 : ℕ) * (A : Ordinal))
  rw [Nat.cast_succ, mul_add, mul_one, add_assoc]
  refine add_lt_add_right ?_ (ω * (B : Ordinal))
  exact lt_of_lt_of_le (natCast_lt_omega0 C) le_self_add

/-- **Restart-step decrease** (`PHASE_ESCALATE` / macro restart):
decrementing `A + 1 → A` strictly decreases `cnfRank` for ANY reset counters
`B, C : ℕ`, because `ω * B + C < ω^2`. -/
theorem cnfRank_dec_A (A B B' C C' : ℕ) :
    cnfRank A B C < cnfRank (A + 1) B' C' := by
  unfold cnfRank
  rw [add_assoc, add_assoc]
  rw [Nat.cast_succ, mul_add, mul_one, add_assoc]
  refine add_lt_add_right ?_ (ω ^ (2 : ℕ) * (A : Ordinal))
  refine lt_of_lt_of_le ?_ le_self_add
  have h1 : ω * (B : Ordinal) + (C : Ordinal) < ω * ((B + 1 : ℕ) : Ordinal) := by
    rw [Nat.cast_succ, mul_add, mul_one]
    exact add_lt_add_right (natCast_lt_omega0 C) (ω * (B : Ordinal))
  have h2 : ω * ((B + 1 : ℕ) : Ordinal) < ω ^ (2 : ℕ) := by
    rw [sq ω]
    exact mul_lt_mul_of_pos_left (natCast_lt_omega0 (B + 1)) omega0_pos
  exact lt_trans h1 h2

end MGRProofs



