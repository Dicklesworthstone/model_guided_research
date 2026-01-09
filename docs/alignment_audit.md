# Docs ↔ Demos Alignment Audit (model_guided_research-3fi)

Audit date: 2025-12-18

Scope:
- Root-level runnable JAX demos (`*.py`)
- Their paired theory notes (`markdown_documentation/*.md`)

Goal:
- Verify the demo contains the features the doc claims are implemented.
- Identify gaps (theory-only / not yet implemented).
- Restore any missing explanatory commentary in the demos (no logic changes).

Legend:
- ✅ = present in demo (implementation exists)
- ⚠️ = partially present / simplified / differs materially
- ❌ = missing (documented but not implemented)
- 📝 = comment/doc gap (implementation exists but demo lacks explanation)

---

## Matrix / Gauge (`matrix_exponential_gauge_learning.py`)

Doc: `markdown_documentation/matrix_exponential_gauge_learning.md`

- ✅ Gauge-covariant token processing via local bases + parallel transport (Givens/angles)
- ✅ Banded continuous-time Markov generator + exact expmv via uniformization (no dense exp)
- ✅ Pullback to native frames (transport-out)
- ✅ SPD channel gating via exponential parameterization
- ✅ Nilpotent/shear channel exponential component
- ✅ Diagnostics for stability + curvature-like metrics
- ✅ Optional structured channel blocks (SO via Cayley, SPD via exp, Sp via symplectic Cayley) behind `cfg.use_structured_blocks`
- ⚠️ Doc’s broader roadmap items (Magnus/BCH fusion, heat-kernel variants, etc.) are discussed but not implemented end-to-end in this demo
- 📝 Commenting: add explicit mapping from doc “Axioms/Steps” → code sections

---

## Ultrametric (`ultrametric_worlds_and_p_adic_computation.py`)

Doc: `markdown_documentation/ultrametric_worlds_and_p_adic_computation.md`

- ✅ LCP-trie / p-adic digit representation
- ✅ Retrieval = deepest occupied ancestor (LCP depth)
- ✅ VOLF update rule (shallowest compatible write; saturating counters)
- ✅ Falsifiers: Task A (exact LCP retrieval) + Task B (leaf exceptions)
- ⚠️ Doc’s cache-opt layout (bitsets + rank/select + fully contiguous arrays) is described as the production path; demo uses a clear reference structure (Python dict/list) rather than the bitset implementation
- 📝 Commenting: clarify which parts are “reference” vs “production layout” and point to doc sections

---

## Simplicial (`simplicial_complexes_and_higher_order_attention.py`)

Doc: `markdown_documentation/simplicial_complexes_and_higher_order_attention.md`

- ✅ Cochains on oriented simplices; boundary matrices `D_k`
- ✅ Down/up lifts using `D_k` and `D_{k+1}^T`
- ✅ Exact mass conservation invariant for the dedicated scalar channel
- ✅ Boundary/Stokes-style consistency loss
- ⚠️ If `jax.experimental.sparse.BCOO` is unavailable, the demo falls back to dense ops (still correct but less scalable)
- 📝 Commenting: ensure the invariant + D∘D=0 reasoning is explicitly tied to the implementation

---

## Nonstandard / Hyperreal (HOSS) (`nonstandard_analysis_and_hyperreal_training.py`)

Doc: `markdown_documentation/nonstandard_analysis_and_hyperreal_training.md`

- ✅ Macro-step derived from infinitesimal micro-process: `Φ_δ(H) g = H^{-1}(I - e^{-δH}) g`
- ✅ Curvature-damped noise via Lyapunov integral / shaped covariance (low-rank approx)
- ✅ Krylov/Lanczos matvec-only approximation path
- ✅ Demos: stiff quadratic and small MLP
- 📝 Commenting: add “where in code” pointers for Φ, exp decay, Lyapunov integral, and Lanczos pieces

---

## Octonion / Quaternion (`octonionic_quaternionic_signal_flow.py`)

Doc: `markdown_documentation/octonionic_quaternionic_signal_flow.md`

- ✅ Quaternion algebra (mul/conj/norm) and expmap to unit rotors
- ✅ Rotor-gate layer ideas: norm-preserving mixing via unit quaternions, separate scalar gate
- ✅ Relative-rotor “attention” style coupling (q * conj(k)) for score/rotate
- ✅ Correctness tests (norm preservation, associativity, etc.)
- ✅ Minimal octonion ops included for tests/illustrations (Cayley–Dickson; non-associativity shows up here)
- ⚠️ The main “rotor-gate” mechanism is quaternionic; octonions are not used as the primary feature representation
- 📝 Commenting: clarify quaternion-vs-octonion scope and map doc sections to code

---

## Ordinal (`ordinal_schedules_and_well_founded_optimization.py`)

Doc: `markdown_documentation/ordinal_schedules_and_well_founded_optimization.md`

- ✅ Well-founded ordinal ranking ρ = ω²·A + ω·B + C with successor/limit transitions
- ✅ Restart/anneal logic consistent with non-increasing rank
- ✅ Baselines (cosine/linear) + streaming regression benchmark
- 📝 Commenting: highlight exact invariants checked (rank monotonicity) and where limit-steps trigger

---

## Reversible (`reversible_computation_and_measure_preserving_learning.py`)

Doc: `markdown_documentation/reversible_computation_and_measure_preserving_learning.md`

- ✅ Additive coupling reversible core + explicit inverse
- ✅ Metered irreversibility “valve” with explicit bit accounting (tape/reservoir)
- ✅ Audit mode for bit-exact forward→inverse cycle checks
- ✅ Diagnostics: irreversibility budget / ledger
- 📝 Commenting: explicitly tie “what makes it bijective” to the concrete tape/reservoir operations

---

## IFS / Fractal (`iterated_function_systems_and_fractal_memory.py`)

Doc: `markdown_documentation/iterated_function_systems_and_fractal_memory.md`

- ✅ Fractal KV store with contraction-based write/read dynamics
- ✅ Separation margin γ = 1 − 2s and contractivity diagnostics
- ✅ Capacity/overlap/interference diagnostics (as described)
- ✅ Learned router (k independent m-way classifiers) mapping queries → paths; inference composes exactly k decisions (O(k)=O(log_m N))
- ✅ Controlled re-indexing hooks (e.g., adjust contractivity / deepen) + microbenchmark for catastrophic forgetting
- 📝 Commenting: map “move toward/away” derivation to the code paths for write/read

---

## Knot / Braid (`knot_theoretic_programs_and_braid_based_attention.py`)

Doc: `markdown_documentation/knot_theoretic_programs_and_braid_based_attention.md`

- ✅ “Program = (π, w)” model with a deliberately restricted braid word family (only σ₁^k; no inverses)
- ✅ Invertible local crossing map + a conserved “payload multiset” invariant used by the task/objective
- ✅ Local verification helpers for the restricted decoder (R2/R3 are vacuous when only σ₁^k is allowed)
- ⚠️ Doc explicitly notes the crossing map used in code is **not** Yang–Baxter / 3‑strand coherent; that is acceptable for σ₁^k but does not support general braid equivalence claims
- 📝 Commenting: keep the scope restriction and the YBE caveat visible near the crossing map

---

## Surreal / Transseries (`surreal_numbers_transseries_and_scaling.py`)

Doc: `markdown_documentation/surreal_numbers_transseries_and_scaling.md`

- ✅ Valuation/order-based decomposition of error terms
- ✅ Projection back to a “balanced frontier” / regime selection procedure (as implemented)
- ⚠️ Doc includes broader transseries/phase-diagram reasoning; demo likely implements a stress-testable subset
- 📝 Commenting: explicitly map implemented decision procedure/projections to doc sections

---

## Tropical (`tropical_geometry_and_idempotent_algebra.py`)

Doc: `markdown_documentation/tropical_geometry_and_idempotent_algebra.md`

- ✅ Max-plus semiring operations and tropical GEMM (`tmm`)
- ✅ Associativity re-bracketing to avoid explicit L×L attention materialization
- ✅ Margin hinge loss and route-wise updates with “safe step” derived from runner-up gaps
- ✅ Route extraction + robustness certificate via per-node gaps
- ✅ Length-generalization toy dataset
- 📝 Commenting: add “route / margin / certificate” mapping from doc to the helper fns/classes

---

## Follow-ups (new beads recommended)

Created during this audit:
- `model_guided_research-a1o` — Ultrametric packed trie layout (bitsets + rank/select)
- `model_guided_research-k2y` — Braid attention YBE-satisfying crossing law option
- `model_guided_research-2l8` — Gauge BCH/Magnus fusion mini-experiment

If any other ❌/⚠️ items above are in-scope for implementation (vs theory discussion), create new beads with `--deps discovered-from:model_guided_research-3fi`.
