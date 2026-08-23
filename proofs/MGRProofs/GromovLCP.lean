/-
Copyright (c) 2026 model_guided_research. All rights reserved.
Formalization tranche 2c of bead model_guided_research-vnl.3.

THEOREM ADDRESSED (hypotheses/theorems.yaml): thm-gromov-product-equals-lcp.
On a rooted metric tree with unit edge lengths, the Gromov product of two
leaves at the root equals the depth of their deepest common ancestor -
exactly the LCP length of their address strings. This integer anchor drives
the ultrametric correspondence exp(-(x|y)): tree boundaries carry an
ultrametric precisely because this quantity behaves like a valuation.

MODEL. Leaves are binary address strings (`List Bool`); the root is `[]`;
unit edges mean depth = length. The root-to-leaf distance is |address|.
The tree distance between two addresses is defined INDEPENDENTLY of any LCP
notion, by walking both strings from the root simultaneously (`treeDist`),
so the main theorem's proof is not circular:

    treeDist a b + 2 * lcpLen a b = |a| + |b|,
whence the Gromov product at the root
    (a|b)_root = (|a| + |b| - treeDist a b) / 2 = lcpLen a b.

CONVENTIONS (proofs/README): zero `sorry` policy.
-/
import Mathlib

namespace MGRProofs

/-- Length of the longest common prefix of two address strings, by
simultaneous recursion. -/
def lcpLen : List Bool → List Bool → ℕ
  | [], _ => 0
  | _, [] => 0
  | x :: xs, y :: ys => if x = y then lcpLen xs ys + 1 else 0

/-- Root-metric distance on the unit-edge rooted binary tree, defined by a
root-down simultaneous walk WITHOUT reference to LCP (this non-circularity
is what makes the proof of thm-gromov-product-equals-lcp honest). -/
def treeDist : List Bool → List Bool → ℕ
  | [], b => b.length
  | a, [] => a.length
  | x :: xs, y :: ys => if x = y then treeDist xs ys else xs.length + ys.length + 2

/-! ### Helper bounds -/

theorem lcpLen_le_length_left (a b : List Bool) : lcpLen a b ≤ a.length := by
  induction a generalizing b with
  | nil => simp [lcpLen]
  | cons x xs ih =>
      cases b with
      | nil => simp [lcpLen]
      | cons y ys =>
          unfold lcpLen
          split
          · next h =>
              have hih := ih ys
              simp only [List.length_cons]
              omega
          · exact Nat.zero_le _

theorem lcpLen_le_length_right (a b : List Bool) : lcpLen a b ≤ b.length := by
  cases a with
  | nil => simp [lcpLen]
  | cons x xs =>
      cases b with
      | nil => simp [lcpLen]
      | cons y ys =>
          unfold lcpLen
          split
          · next h =>
              have hih := lcpLen_le_length_right xs ys
              simp only [List.length_cons]
              omega
          · exact Nat.zero_le _

/-! ### Core identity -/

/-- **Core identity.** The walk-distance between two addresses equals the sum
of their depths minus twice their common-prefix depth. -/
theorem treeDist_eq (a b : List Bool) :
    treeDist a b + 2 * lcpLen a b = a.length + b.length := by
  induction a generalizing b with
  | nil =>
      cases b with
      | nil => simp [treeDist, lcpLen]
      | cons y ys =>
          have h1 : treeDist [] (y :: ys) = List.length (y :: ys) := rfl
          have h2 : lcpLen [] (y :: ys) = 0 := rfl
          simp [h1, h2]
  | cons x xs ih =>
      cases b with
      | nil =>
          have h1 : treeDist (x :: xs) [] = List.length (x :: xs) := rfl
          have h2 : lcpLen (x :: xs) [] = 0 := rfl
          simp [h1, h2]
      | cons y ys =>
          by_cases h : x = y
          · have e1 : treeDist (x :: xs) (y :: ys) = treeDist xs ys := by
              simp [treeDist, h]
            have e2 : lcpLen (x :: xs) (y :: ys) = lcpLen xs ys + 1 := by
              simp [lcpLen, h]
            rw [e1, e2]
            have hih := ih ys
            simp only [List.length_cons]
            omega
          · have e1 : treeDist (x :: xs) (y :: ys) = xs.length + ys.length + 2 := by
              simp [treeDist, h]
            have e2 : lcpLen (x :: xs) (y :: ys) = 0 := by
              simp [lcpLen, h]
            rw [e1, e2]
            have h1 := lcpLen_le_length_left xs ys
            have h2 := lcpLen_le_length_right xs ys
            simp only [List.length_cons]
            omega

/-! ### thm-gromov-product-equals-lcp -/

/-- **Gromov product at the root equals LCP length** (thm-gromov-product-
equals-lcp; unit-edge rooted tree, binary address strings):
`(a|b)_root = (|a| + |b| - d(a,b)) / 2 = lcpLen a b`, the depth of the
deepest common ancestor. -/
theorem gromov_root_eq_lcp (a b : List Bool) :
    (a.length + b.length - treeDist a b) / 2 = lcpLen a b := by
  have h := treeDist_eq a b
  have hle : 2 * lcpLen a b ≤ a.length + b.length := by omega
  omega

/-! ### Executable documentation (depth-4 binary tree)

- identical addresses: LCP = full depth (4);
- sibling leaves differing in the last bit: LCP = 3;
- opposite branches at the root: LCP = 0;
- sibling-pair walk distance = 2 (down-and-up through the shared parent).
-/

example : lcpLen [true, false, true, true] [true, false, true, true] = 4 := by
  decide

example : lcpLen [true, false, true, false] [true, false, true, true] = 3 := by
  decide

example : lcpLen [false, false, false, false] [true, true, true, true] = 0 := by
  decide

example : treeDist [true, false, true, false] [true, false, true, true] = 2 := by
  decide

example : (List.length [true, false, true, false] +
    List.length [true, false, true, true] -
    treeDist [true, false, true, false] [true, false, true, true]) / 2 =
    lcpLen [true, false, true, false] [true, false, true, true] := by
  decide

end MGRProofs
