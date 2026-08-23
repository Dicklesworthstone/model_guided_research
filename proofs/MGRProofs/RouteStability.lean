/-
Copyright (c) 2026 model_guided_research. All rights reserved.
Formalization tranche 1a of bead model_guided_research-vnl.2.

THEOREMS ADDRESSED (hypotheses/theorems.yaml):
- thm-lse-max-sandwich : max_i x_i <= LSE_beta(x) <= max_i x_i + log(n)/beta
  (the inflation is ONE-SIDED: LSE never underestimates the max)
- thm-route-stability  : two-family form. Let x be tropical route scores whose
  chosen route i* dominates every other route j by a runner-up margin m, and
  let y be smoothed scores satisfying x <= y <= x + c pointwise (beta-attention
  dequantization gives c = log(m_arity)/beta for INNER aggregation arity m).
  If c < m, the smoothed scores preserve the argmax: y_j < y_{i*} for all
  j != i*, so the maximizer of y exists and equals i* uniquely.

CONVENTIONS (proofs/README):
- one file per theory pillar; lemma names mirror theorem-registry ids;
- zero `sorry` policy enforced by CI grep plus an explicit axiom check.
-/
import Mathlib

namespace MGRProofs

/-! ### LSE-max sandwich (registry id: thm-lse-max-sandwich) -/

/-- **LSE-max sandwich.** For a finite nonempty family `x : Fin n → ℝ` and
`β > 0`, the log-sum-exp aggregate with inverse temperature `β` sandwiches the
family maximum:
`max x ≤ (1/β) · log (∑ i, exp (β * x i)) ≤ max x + log(n)/β`.
Note the inflation is ONE-SIDED: LSE never underestimates the maximum. -/
theorem lse_max_sandwich {n : ℕ} [NeZero n] (x : Fin n → ℝ) {β : ℝ} (hβ : 0 < β) :
    Finset.univ.sup' Finset.univ_nonempty x ≤
      β⁻¹ * Real.log (∑ i ∈ Finset.univ, Real.exp (β * x i)) ∧
      β⁻¹ * Real.log (∑ i ∈ Finset.univ, Real.exp (β * x i)) ≤
      Finset.univ.sup' Finset.univ_nonempty x + Real.log n / β := by
  classical
  set S : ℝ := ∑ i ∈ Finset.univ, Real.exp (β * x i) with hS_def
  set M : ℝ := Finset.univ.sup' Finset.univ_nonempty x with hM_def
  have hS_pos : 0 < S := by
    rw [hS_def]
    refine Finset.sum_pos' (fun i _ => le_of_lt (Real.exp_pos _))
      ⟨0, Finset.mem_univ _, Real.exp_pos _⟩
  have hx_le_M : ∀ i : Fin n, x i ≤ M := fun i =>
    Finset.le_sup' _ (Finset.mem_univ i)
  -- lower bound: each single exponential term already forces exp(β x i) ≤ S
  have hterm_le_S : ∀ i : Fin n, Real.exp (β * x i) ≤ S := by
    intro i
    rw [hS_def]
    exact
      Finset.single_le_sum (f := fun j : Fin n => Real.exp (β * x j))
        (fun j _ => le_of_lt (Real.exp_pos _)) (Finset.mem_univ i)
  have hlog_term : ∀ i : Fin n, β * x i ≤ Real.log S := by
    intro i
    have h := Real.log_le_log (Real.exp_pos _) (hterm_le_S i)
    rwa [Real.log_exp] at h
  have hinv : 0 < β⁻¹ := inv_pos.mpr hβ
  have hlower : M ≤ β⁻¹ * Real.log S :=
    Finset.sup'_le _ _ fun i _ => by
      have h := hlog_term i
      calc x i = β⁻¹ * (β * x i) := by field_simp
        _ ≤ β⁻¹ * Real.log S := mul_le_mul_of_nonneg_left h hinv.le
  -- upper bound: every term is dominated by exp(β M), hence S ≤ n * exp(β M)
  have hcard : ((Finset.univ : Finset (Fin n)).card : ℝ) = n := by simp
  have hsum_ub : S ≤ (n:ℝ) * Real.exp (β * M) := by
    rw [hS_def]
    have hstep := Finset.sum_le_card_nsmul Finset.univ
      (fun i => Real.exp (β * x i)) (Real.exp (β * M))
      (fun i _ =>
        Real.exp_le_exp.mpr (mul_le_mul_of_nonneg_left (hx_le_M i) hβ.le))
    simpa [nsmul_eq_mul] using hstep
  have hupper : β⁻¹ * Real.log S ≤ M + Real.log n / β := by
    have hnpos : 0 < (n:ℝ) := by
      have := ‹NeZero n›.out
      exact_mod_cast Nat.pos_of_ne_zero this
    have hlog_ub : Real.log S ≤ Real.log ((n:ℝ) * Real.exp (β * M)) :=
      Real.log_le_log hS_pos hsum_ub
    have hsplit : Real.log ((n:ℝ) * Real.exp (β * M)) =
        Real.log n + β * M := by
      rw [Real.log_mul hnpos.ne' (Real.exp_ne_zero _), Real.log_exp]
    calc β⁻¹ * Real.log S ≤ β⁻¹ * (Real.log n + β * M) :=
          mul_le_mul_of_nonneg_left (le_trans hlog_ub (le_of_eq hsplit)) hinv.le
      _ = M + Real.log n / β := by
          field_simp
          ring
  exact ⟨hlower, hupper⟩

/-! ### Route stability (registry id: thm-route-stability)

TWO-FAMILY form, matching what beta-attention actually instantiates: a
single-family phrasing would be vacuous, since smoothing an aggregation does
not re-rank the aggregated values themselves. Smoothing budget per route is
`c`; the tropical winner `istar` beats every rival `j` by runner-up margin `m`.
If `c < m` the ranking cannot flip: every rival stays strictly below `istar`
after smoothing, so the maximizer of `y` exists and equals `istar` uniquely.
-/

/-- **Route-stability corollary (pointwise strict form).** If the tropical
scores `x` elect `istar` with margin `m` over every rival and the smoothed
scores `y` stay within budget `c` of `x` pointwise with `c < m`, then every
rival remains strictly below `istar` after smoothing. -/
theorem route_stability_pointwise {n : ℕ} {x y : Fin n → ℝ} {c m : ℝ}
    (hcm : c < m) (hxy : ∀ i, x i ≤ y i ∧ y i ≤ x i + c) (istar : Fin n)
    (hdom : ∀ j, j ≠ istar → x j + m ≤ x istar) :
    ∀ j : Fin n, j ≠ istar → y j < y istar := by
  intro j hj
  have hyub : y j ≤ x j + c := (hxy j).2
  have hxj : x j + c < x istar := by
    have hd := hdom j hj
    linarith
  have hylb : x istar ≤ y istar := (hxy istar).1
  linarith

/-- **Route-stability corollary (unique maximizer form).** Under the same
hypotheses, the smoothed scores `y` attain their maximum exactly at the
tropical winner `istar` (existence and uniqueness). -/
theorem route_stability_unique_max {n : ℕ} {x y : Fin n → ℝ} {c m : ℝ}
    (hcm : c < m) (hxy : ∀ i, x i ≤ y i ∧ y i ≤ x i + c) (istar : Fin n)
    (hdom : ∀ j, j ≠ istar → x j + m ≤ x istar) :
    ∃! i : Fin n, ∀ j, y j ≤ y i := by
  refine ⟨istar, fun j => ?_, ?_⟩
  · by_cases hj : j = istar
    · subst hj; exact le_refl _
    · exact le_of_lt (route_stability_pointwise hcm hxy istar hdom j hj)
  · intro w hw
    by_cases hw' : w = istar
    · subst hw'; exact rfl
    · have hlt := route_stability_pointwise hcm hxy istar hdom w hw'
      linarith [hw istar]

/-! ### Executable documentation (concrete numbers, n = 3, beta = 2)

Take `x = [log 2, 0, 0]`, `beta = 2`.  Then `max x = log 2`,
`S = exp(2 log 2) + exp(0) + exp(0) = 6`, so `LSE = log 6 / 2`, and the
sandwich reduces to `log 2 <= log 6 / 2 <= log 2 + log 3 / 2`, i.e. exactly
the two nontrivial facts `log 2 <= log 3` and `log 3 <= log 4 = 2 log 2`.
These examples instantiate `lse_max_sandwich`'s conclusion at those numbers.
-/
private theorem exp_two_log_two_eq_four : (Real.exp (2 * Real.log 2) : ℝ) = 4 := by
  have hsplit : (2:ℝ) * Real.log 2 = Real.log 2 + Real.log 2 := by ring
  have hadd : Real.exp (Real.log 2 + Real.log 2) =
      Real.exp (Real.log 2) * Real.exp (Real.log 2) := Real.exp_add _ _
  rw [hsplit, hadd, Real.exp_log (by norm_num)]
  norm_num

example :
    (Real.log 2 : ℝ) ≤
      1 / 2 * Real.log (Real.exp (2 * Real.log 2) + Real.exp 0 + Real.exp 0) ∧
      1 / 2 * Real.log (Real.exp (2 * Real.log 2) + Real.exp 0 + Real.exp 0) ≤
      Real.log 2 + Real.log 3 / 2 := by
  rw [exp_two_log_two_eq_four, Real.exp_zero]
  norm_num
  have h6 : (Real.log 6 : ℝ) = Real.log 2 + Real.log 3 := by
    rw [show (6:ℝ) = 2 * 3 from by norm_num, Real.log_mul (by norm_num) (by norm_num)]
  have h23 : (Real.log 2 : ℝ) ≤ Real.log 3 :=
    Real.log_le_log (by norm_num) (by norm_num)
  have h34 : (Real.log 3 : ℝ) ≤ Real.log 4 :=
    Real.log_le_log (by norm_num) (by norm_num)
  have h42 : (Real.log 4 : ℝ) = 2 * Real.log 2 := by
    have hsplit : Real.log 4 = Real.log 2 + Real.log 2 := by
      rw [show (4:ℝ) = 2 * 2 from by norm_num, Real.log_mul (by norm_num) (by norm_num)]
    linarith
  have h2nn : (0:ℝ) ≤ Real.log 2 :=
    le_of_lt (Real.log_pos (by norm_num))
  rw [h6]
  constructor
  · nlinarith
  · nlinarith
