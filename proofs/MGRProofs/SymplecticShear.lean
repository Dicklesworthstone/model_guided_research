/-
Copyright (c) 2026 model_guided_research. All rights reserved.
Formalization tranche 2b of bead model_guided_research-vnl.3.

THEOREM ADDRESSED (hypotheses/theorems.yaml): thm-kick-kick-symplectic
(u55.5). With F = grad(phi_F) and G = grad(phi_G), the coupled reversible
block composes two gradient shears, y1 = x1 + F(x2) and y2 = x2 + G(y1);
each shear is a symplectic map (its Jacobian is an I/H upper block-triangular
matrix with symmetric H), and symplectic maps are closed under composition -
hence the full kick-kick block is symplectic. This is the J^T Omega J = Omega
finite-dimensional core. Exactness: each kick is the time-eps flow of the
separable Hamiltonian H(x1,x2) = phi_F(x2) + phi_G(x1), which is what makes
the discrete integrator track that Hamiltonian; recorded here as commentary,
with the formal content being the preservation identity and composition
closure.

Index convention: phase space is `Fin n ⊕ Fin n` (momenta in `inl`, positions
in `inr`), so mathlib's `Matrix.fromBlocks` applies directly.

CONVENTIONS (proofs/README): zero `sorry` policy.
-/
import Mathlib

namespace MGRProofs

open Matrix

variable {n : ℕ}

/-- The canonical symplectic matrix on R^{n+n} in the (p, q)-splitting:
Omega = [[0, I], [-I, 0]]. -/
def OmegaBlock (n : ℕ) : Matrix (Fin n ⊕ Fin n) (Fin n ⊕ Fin n) ℝ :=
  fromBlocks 0 (1 : Matrix (Fin n) (Fin n) ℝ)
    (-(1 : Matrix (Fin n) (Fin n) ℝ)) 0

/-- Jacobian of the momentum-kick shear (p, q) ↦ (p + H q, q), where `H` is
the Hessian of the potential (symmetric for smooth scalar potentials):
J = [[I, H], [0, I]]. -/
def kickJacobian (H : Matrix (Fin n) (Fin n) ℝ) :
    Matrix (Fin n ⊕ Fin n) (Fin n ⊕ Fin n) ℝ :=
  fromBlocks 1 H 0 1

/-! ### Shear symplecticity -/

/-- **Kick-kick symplecticity, single shear.** The Jacobian of the shear
`(p, q) ↦ (p + H q, q)` with SYMMETRIC block `H` (Hessian of a smooth
scalar potential) satisfies `J^T Omega J = Omega`. Symmetry of `H` is
exactly where the gradient hypothesis (`F = grad phi`) enters; for a
non-gradient force with non-symmetric H the identity fails on the
antisymmetric part. -/
theorem kick_jacobian_symplectic (H : Matrix (Fin n) (Fin n) ℝ)
    (hsym : Hᵀ = H) :
    (kickJacobian H)ᵀ * OmegaBlock n * kickJacobian H = OmegaBlock n := by
  have hJT : (kickJacobian H)ᵀ = fromBlocks 1 0 (Hᵀ) 1 := by
    simp [kickJacobian, Matrix.fromBlocks_transpose]
  have hstep1 : (fromBlocks 1 0 (Hᵀ) 1) * OmegaBlock n =
      fromBlocks 0 1 (-(1 : Matrix (Fin n) (Fin n) ℝ)) Hᵀ := by
    simp [OmegaBlock, Matrix.fromBlocks_multiply]
  have hstep2 : (fromBlocks 0 1 (-(1 : Matrix (Fin n) (Fin n) ℝ)) Hᵀ) *
      kickJacobian H = OmegaBlock n := by
    simp [OmegaBlock, Matrix.fromBlocks_multiply, kickJacobian, hsym,
      Matrix.neg_mul, one_mul, neg_add_cancel]
  rw [hJT, hstep1, hstep2]

/-! ### Composition closure -/

/-- **Symplectic maps are closed under composition**: if A and B both
preserve Omega, so does their product AB. This upgrades per-kick
symplecticity to whole-block symplecticity for the kick-kick integrator. -/
theorem symplectic_composition (A B : Matrix (Fin n ⊕ Fin n) (Fin n ⊕ Fin n) ℝ)
    (hA : Aᵀ * OmegaBlock n * A = OmegaBlock n)
    (hB : Bᵀ * OmegaBlock n * B = OmegaBlock n) :
    (A * B)ᵀ * OmegaBlock n * (A * B) = OmegaBlock n := by
  calc (A * B)ᵀ * OmegaBlock n * (A * B)
      = Bᵀ * (Aᵀ * OmegaBlock n * A) * B := by
        rw [Matrix.transpose_mul]
        simp [Matrix.mul_assoc]
    _ = Bᵀ * OmegaBlock n * B := by rw [hA]
    _ = OmegaBlock n := hB

/-! ### Executable documentation: 1D quadratic potential

phi(q) = q^2 on R^1: Hessian H = [[2]] (symmetric). The general theorem then
gives the 4x4 symplectic identity concretely. -/

example (H : Matrix (Fin 1) (Fin 1) ℝ) (hH : H 0 0 = 2) :
    (kickJacobian (n := 1) H)ᵀ * OmegaBlock 1 * kickJacobian (n := 1) H =
      OmegaBlock 1 := by
  have hsym : Hᵀ = H := by
    ext i j
    fin_cases i <;> fin_cases j <;> simp [hH]
  exact kick_jacobian_symplectic H hsym

end MGRProofs
