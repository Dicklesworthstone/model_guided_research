/-
Copyright (c) 2026 model_guided_research. All rights reserved.
Formalization tranche 1b of bead model_guided_research-vnl.2.

THEOREM ADDRESSED (hypotheses/theorems.yaml): thm-flat-error.
In a field with non-archimedean valuation v, for k >= 0: if v(e_i) >= k for
all i in a finite family, then v(sum_i e_i) >= k and
v(prod_i (1 + e_i) - 1) >= k. The k >= 0 hypothesis is REQUIRED for the
product part (each subset-product term has valuation >= |S|*k, which is >= k
only when k >= 0); the sum part holds for any k.

ENCODING NOTE. Mathlib's `Valuation`/`AbsoluteValue` are MULTIPLICATIVE
(|x+y| <= max(|x|, |y|)); an additive depth-k statement `v(e) >= k` becomes
"every error has multiplicative size at most threshold t" with t = eps^k.
The k >= 0 hypothesis appears here as `t <= 1`, used exactly once, in the
product step. We prove both theorems over a minimal structure
`UltrametricValuation` carrying only the four valuation axioms, then bridge
FROM mathlib absolute values via a plain ultrametric-inequality hypothesis
(mathlib's p-adic norm on Q supplies it), so every non-archimedean absolute
value instantiates the results.

CONVENTIONS (proofs/README): zero `sorry` policy; see RouteStability.lean.
-/
import Mathlib

open scoped NNReal

namespace MGRProofs

/-- A multiplicative non-archimedean valuation on a commutative ring `K`:
exactly the four axioms (zero, one, multiplicativity, strong triangle)
everything below needs. `Γ` is the value monoid, linearly ordered with
zero. -/
structure UltrametricValuation (K Γ : Type*) [CommRing K]
    [LinearOrderedCommMonoidWithZero Γ] where
  /-- The underlying function. -/
  val : K → Γ
  /-- Zero maps to zero. -/
  val_zero : val 0 = 0
  /-- One maps to one. -/
  val_one : val 1 = 1
  /-- Multiplicativity. -/
  val_mul : ∀ a b : K, val (a * b) = val a * val b
  /-- Strong triangle inequality. -/
  val_add : ∀ a b : K, val (a + b) ≤ max (val a) (val b)

/-! ### thm-flat-error, sum part (holds for ANY threshold) -/

/-- **Flat-error lemma, sum part.** If every error in a finite family has
valuation at most the threshold `t`, so does their sum. No constraint on
`t`. This is additive form `v(sum e_i) >= k` under `t = eps^k`. -/
theorem flat_error_sum {K Γ ι : Type*} [CommRing K]
    [LinearOrderedCommMonoidWithZero Γ] (w : UltrametricValuation K Γ)
    (s : Finset ι) (e : ι → K) (t : Γ) (h : ∀ i ∈ s, w.val (e i) ≤ t) :
    w.val (∑ i ∈ s, e i) ≤ t := by
  classical
  induction s using Finset.induction_on with
  | empty =>
      simp only [Finset.sum_empty, w.val_zero]
      exact zero_le
  | insert i s hi ih =>
      have h1 : w.val (∑ j ∈ s, e j) ≤ t :=
        ih fun j hj => h j (Finset.mem_insert_of_mem hj)
      have h2 : w.val (e i) ≤ t := h i (Finset.mem_insert_self _ _)
      have hadd := w.val_add (e i) (∑ j ∈ s, e j)
      rw [Finset.sum_insert hi]
      exact le_trans hadd (max_le h2 h1)

/-! ### thm-flat-error, product part (needs the threshold on the correct side) -/

/-- **Flat-error lemma, product part.** If every error has valuation at most
`t ≤ 1` (the encoding of digit depth `k ≥ 0`), then the relative error of
the product of `(1 + e i)` also has valuation at most `t`. The side
condition `t ≤ 1` is REQUIRED: it is what keeps each subset-product term at
or above depth in the induction step. -/
theorem flat_error_prod {K Γ ι : Type*} [CommRing K]
    [LinearOrderedCommMonoidWithZero Γ] (w : UltrametricValuation K Γ)
    (s : Finset ι) (e : ι → K) (t : Γ) (hu1 : t ≤ 1)
    (h : ∀ i ∈ s, w.val (e i) ≤ t) :
    w.val ((∏ i ∈ s, (1 + e i)) - 1) ≤ t := by
  classical
  induction s using Finset.induction_on with
  | empty =>
      simp only [Finset.prod_empty, sub_self, w.val_zero]
      exact zero_le
  | insert i s hi ih =>
      -- algebra of the induction step:
      --   prod(insert) - 1 = (prod(s) - 1) * (1 + e i) + e i
      have hstep : (∏ j ∈ insert i s, (1 + e j)) - 1 =
          (((∏ j ∈ s, (1 + e j)) - 1) * (1 + e i)) + e i := by
        rw [Finset.prod_insert hi]
        ring
      -- carried facts from the smaller family
      have h1 := ih fun j hj => h j (Finset.mem_insert_of_mem hj)
      have h2 : w.val (e i) ≤ t := h i (Finset.mem_insert_self _ _)
      -- v(1 + e i) <= max(1, v(e i)) <= 1   (this is where t <= 1 enters)
      have hone : w.val (1 + e i) ≤ 1 := by
        have hadd1 := w.val_add 1 (e i)
        have hv1 : w.val 1 ≤ 1 := by rw [w.val_one]
        have hve1 : w.val (e i) ≤ 1 := le_trans h2 hu1
        exact le_trans hadd1 (max_le hv1 hve1)
      -- v((prod-1)*(1+ei)) <= t * 1 = t
      have hp' : w.val (((∏ j ∈ s, (1 + e j)) - 1) * (1 + e i)) ≤ t := by
        rw [w.val_mul]
        calc w.val ((∏ j ∈ s, (1 + e j)) - 1) *
              w.val (1 + e i) ≤ t * 1 :=
              mul_le_mul' h1 hone
          _ = t := mul_one _
      rw [hstep]
      have hadd := w.val_add
        (((∏ j ∈ s, (1 + e j)) - 1) * (1 + e i)) (e i)
      exact le_trans hadd (max_le hp' h2)

/-! ### Bridge from mathlib absolute values

Every non-archimedean absolute value in mathlib (e.g. the p-adic norm on Q,
file `Mathlib.NumberTheory.Padics.PadicNorm`, whose instance carries
mathlib's `AbsoluteValue.IsUltrametric`) satisfies the hypothesis below, so
the two flat-error theorems apply verbatim there. -/

section Bridge

variable {K : Type*} [Field K]

/-- Any absolute value obeying the strong triangle inequality yields an
`UltrametricValuation` (values in `NNReal`). -/
noncomputable def ultrametricValuationOfAbsoluteValue
    (abv : AbsoluteValue K NNReal)
    (hultra : ∀ a b : K, abv (a + b) ≤ max (abv a) (abv b)) :
    UltrametricValuation K NNReal where
  val x := abv x
  val_zero := map_zero _
  val_one := map_one _
  val_mul := fun a b => map_mul abv a b
  val_add := hultra

end Bridge

/-! ### Concrete instance + executable checks (bead requirement)

The trivial valuation (`0 ↦ 0`, everything else `↦ 1`) on Q: degenerate but
fully computable, exercising both statements end-to-end on concrete numbers.
Genuine p-adic use goes through `ultrametricValuationOfAbsoluteValue` applied
to mathlib's padic norm (see README). -/
section Examples

def trivialUlamQ : UltrametricValuation ℚ NNReal where
  val x := if x = 0 then 0 else 1
  val_zero := by simp
  val_one := by simp
  val_mul := by
    intro a b
    by_cases ha : a = 0 <;> by_cases hb : b = 0 <;>
      simp [ha, hb, mul_eq_zero]
  val_add := by
    intro a b
    by_cases hab : a + b = 0
    · simp [hab]
    by_cases ha : a = 0
    · subst ha; simp
    by_cases hb : b = 0
    · subst hb; simp [hab]
    simp [ha, hb, hab]

-- three flat errors: the sum stays flat...
example : trivialUlamQ.val ((0:ℚ) + 0 + 0) ≤ 1 := by
  simp [trivialUlamQ]

-- ...and the relative product error stays flat too.
example : trivialUlamQ.val ((1 + (0:ℚ)) * (1 + 0) * (1 + 0) - 1) ≤ 1 := by
  simp [trivialUlamQ]

-- both general theorems fire through the machinery at threshold t = 1;
-- hypotheses discharged inline, conclusions by computation.
example : trivialUlamQ.val (∑ _i ∈ ({0, 1, 2} : Finset ℕ), (0:ℚ)) ≤ 1 := by
  refine flat_error_sum trivialUlamQ _ _ 1 ?_
  intro i _
  simp [trivialUlamQ]

example : trivialUlamQ.val ((∏ _i ∈ ({0, 1, 2} : Finset ℕ), (1 + (0:ℚ))) - 1) ≤ 1 := by
  refine flat_error_prod trivialUlamQ _ _ 1 le_rfl ?_
  intro i _
  simp [trivialUlamQ]

end Examples
