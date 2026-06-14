# Newton-Polytope Capacity Theory for Tropical Attention

**Bead:** 8gk.3 (EPIC THEORY-I, 8gk — non-archimedean unification) ·
**Anchor:** `tropical_geometry_and_idempotent_algebra.md` (the broad tropical
note); this is the focused capacity/expressivity companion · **Status of
claims:** every region-count claim is validated computationally before it is
stated (receipts in Appendix A; scratch `/data/tmp/newton_regions_scratch.py`,
pure numpy, no repo dependencies).

A tropical polynomial `f(x) = max_a (c_a + ⟨a, x⟩)` is piecewise linear, and
its linear regions are the cells of the regular subdivision of its Newton
polytope `New(f) = conv{a}` induced by the lifting `c_a`. Zhang–Naitzat–Lim
(2018) showed ReLU networks are tropical rational maps and recovered
expressivity bounds (Montúfar et al.) by counting these cells. **What is new
here:** nobody has computed the polytope geometry of tropical *attention* — a
composition of two max-plus bilinear stages whose inner index set is the
architecture's feature dimension but whose outer index set is the
**data-dependent context of length T**. That single structural fact makes
tropical attention's region count grow with context length, unlike a
feedforward tropical layer whose monomial set is fixed by the architecture.

## 0. The mechanism, and the framing that matters

One head (`nanochat/tropical_attention_torch.py`): with `q_i = W_q x_i`,
`k_j = W_k x_j`, `v_j = W_v x_j` the per-token projections of the input
`x = (x_1, …, x_T) ∈ ℝ^{T×d_model}`,

    score(q_i, k_j) = max_e (q_{i,e} + k_{j,e}),     (inner max over d_head features)
    y_{i,d}(x)      = max_{j ≤ i} ( score(q_i, k_j) + v_{j,d} ).   (outer max over keys)

Each pair **(key j, feature e)** is a **route**; the value it contributes to
`y_{i,d}` is the affine function of `x`

    f_{j,e}(x) = (W_q x_i)_e + (W_k x_j)_e + (W_v x_j)_d,

linear in `x` with slope `row_e(W_q)` on the `x_i` block and
`row_e(W_k) + row_d(W_v)` on the `x_j` block, zero elsewhere. `y_{i,d}` is the
max of these `T·d_head` affine functions; its linear regions are the cells
where one route is the argmax.

**The framing pitfall (a Proposition-0 cautionary lemma — do not skip it).**
If you analyse regions in *query space* with `K, V` held FIXED, the count
**collapses** and shows no T-dependence at all. With fixed `K, V` every route
`(j,e)` has slope exactly the basis vector `ê_e` in `q` (coefficient 1 on
`q_e`, 0 elsewhere), so for each feature `e` only the single key
`k*(e) = argmax_j (K_{j,e} + V_{j,d})` can ever win — and `y_d` collapses to
`max_e (q_e + M_{e,d})` with `M_{e,d} = max_j (K_{j,e}+V_{j,d})`, **at most
`d_head` regions, independent of T** (Appendix A.2 measures a flat 6 for
`d_head = 6` across T = 2…64). The combinatorial-in-T richness is real, but it
lives in **input space**, because `q_i`, `k_j`, `v_j` are all affine functions
of the same `x`: routes with different keys `j ≠ j'` have slopes supported on
*different* input blocks (`x_j` vs `x_{j'}`), hence are generically distinct,
and the max does not collapse. Every capacity statement below is therefore an
**input-space** statement.

## 1. Proposition 1 — per-output region count grows as T·d_head

**Statement.** For a fixed query `i` and output coordinate `d`, the number of
linear regions of `x ↦ y_{i,d}(x)` is bounded by the number of routes,
`(i+1)·d_head`, and this bound is **realized** for generic projections: every
route `(j, e)` with `j ≤ i` is the argmax on a full-dimensional cell.

**Why (sketch; the upper-hull picture).** The `T·d_head` route functions
`f_{j,e}` lift to points `(a_{j,e}, c_{j,e})` whose slopes `a_{j,e}` are
pairwise distinct (the `j ≠ j'` blocks have disjoint support; within a key the
`d_head` features differ by `row_e(W_q)` on the `x_i` block). The regions of a
max of affine functions are the projections of the **upper-hull facets** of
this lifted point set; with points in general position every point is an upper
vertex, so every route owns a region. The count is therefore the route count
`(i+1)·d_head`, linear in the visible context.

**Validation (A.1).** Realized regions of `y_{last,0}` over 4000 random inputs,
`d_model=12, d_head=6`: T = 2,4,8,16,32,64 → **12, 24, 48, 96, 192, 368**
against the `T·d_head` bound 12, 24, 48, 96, 192, 384 — exact through T=32 and
96 % of the bound at T=64 (the last shortfall is finite-sample, not
structural). Linear in T, as claimed; contrast the flat q-space control (A.2).

**Consequence.** A feedforward tropical layer of width `m` has a region count
fixed by `m` (its monomial set does not move with the input length). Tropical
attention's per-output region budget instead **scales with the context** —
the first algebraic statement of why a max-plus *attention* layer is more
expressive than a max-plus *feedforward* layer of matched width on long
sequences.

## 2. Proposition 2 — the multi-head mixed-volume law

Heads write into the residual stream **additively**: the layer output coord is
`Y_{i,d} = Σ_{h=1}^{H} y^h_{i,d}`. The regions of a sum of piecewise-linear
maps are the **common refinement (overlay)** of the summands' region
partitions — the **mixed subdivision** of the Minkowski sum of the H Newton
polytopes — so by the Bernstein–Khovanskii–Kushnirenko mixed-volume principle
the joint region count grows **multiplicatively** in H, not additively.

**Statement.** The number of linear regions of `Σ_h y^h_{i,d}` is bounded by
`∏_h (regions of y^h) = (T·d_head)^H`, and grows super-linearly toward it —
adding a head **multiplies** capacity rather than adding a fixed increment.

**Validation (A.3).** Realized joint regions of `Σ_h y^h_{last,0}` over 6000
random inputs, T=8, d_head=6 (single-head budget 48): H = 1,2,3,4 → **48,
1660, 5321, 5961**. The linear prediction `H·48` is 48, 96, 144, 192; the
product bound `(48)^H` is 48, 2304, 110592, … . The H=1→2 step (48 → 1660, a
35× jump for *doubling* the heads, reaching 72 % of the product bound) is the
decisive evidence of multiplicative growth; H=3,4 are finite-sample-saturated
(6000 samples cannot realize 110k distinct joint routes) but lie far above the
additive line. **Head superposition is mixed-volume accumulation** — the first
algebraic account of what adding heads buys, in capacity terms.

## 3. The conjecture (falsifiable, countable)

**Per-parameter region efficiency.** The per-parameter realized-region count
of tropical attention exceeds that of a parameter-matched ReLU MLP by a factor
that grows with T. Counting protocol (the metric definition): fix a parameter
budget P; build (a) a single tropical-attention head and (b) a ReLU MLP with P
params; over a fixed seeded sample of `N` inputs, count distinct realized
activation patterns (argmax routes for tropical via the route-extraction the
mechanism already exposes; ReLU sign patterns for the MLP); report
`regions / P` for each as a growth curve vs T. If the tropical curve's
T-slope is positive and the MLP's is ~0 (its monomial set is T-independent),
the conjecture holds: tropical attention is **provably more region-efficient
per parameter in long contexts** — a constructive expressivity *separation*,
not a circuit-class upper bound. Realized counts must ALWAYS respect the
Proposition-1/2 upper bounds (a violation is an implementation bug).

## 4. Design corollary — polyhedral architecture design

By the Upper Bound Theorem, region capacity for a fixed number of monomials is
maximized by point configurations whose upper hulls have many vertices —
**cyclic-polytope-type** supports are extremal. Translation: choose the
*support pattern* of the score/value maps (which index sets appear in each max)
as a designed combinatorial structure rather than the dense default.
**Prototype:** structured-support tropical attention (a cyclic-type support
mask on the per-head feature/key index sets) vs dense, at equal parameter
count, on ≥ 1 diagnostic task + an LM smoke. Preregister the directional
prediction (structured ≥ dense on a region-sensitive task) before evidence.

## 5. Registry predictions (registry-ready)

Floor/rung discipline per the braid–Dyck lesson (a floored win is not a
structural win): rung from a quarantined sizing probe; EM claims carry floors.

1. `hyp-tropical-region-count-grows-with-T` — single-arm structural claim on
   the *mechanism* (no training): over a fixed seeded input sample, the
   realized per-output region count of one tropical head fits `T·d_head` to
   within sampling tolerance and its T-slope is strictly positive, while a
   parameter-matched ReLU MLP's realized region count is flat in T.
   `train`/bench-style observable (region-count protocol §3); this is a
   property of the architecture, checkable without a trained model — the
   cheapest, most direct test of the capacity thesis.
2. `hyp-tropical-structured-support-beats-dense` — structured (cyclic-type)
   support beats dense at equal params on a region-sensitive diagnostic by a
   preregistered EM margin at an off-floor rung; baseline = dense tropical,
   floor = the task answer prior.
3. The mixed-volume multi-head law (§2) is a **structural theorem checked
   numerically** (Appendix A.3), not a statistical hypothesis — it gates the
   capacity narrative rather than entering the ledger.

## 6. Relation to the rest of the program

This is the *expressivity* face of the tropical mechanism, complementary to the
*dequantization* face (`maslov_dequantization_annealing.md`, the β-homotopy)
and the *valuation* face (`the_valuation_dictionary.md`). The route-extraction
machinery the count needs is the same the tropical mechanism already exposes
for its certificate margins (the `8gk.9` route-observatory bead is its
first-class-artifact home). The per-mechanism width-scaling table (`lab.1`)
carries the tropical FFN's EVT/Gumbel row; this note adds the *attention*
region-count row, which `lab.1`'s coordinate check should respect.

## Appendix A: verification receipts (8gk.3 scratch, 2026-06-14)

Pure-numpy region counting from first principles; reproduced in the F-series
region-count harness when 8gk.9 lands.

1. **Prop 1 (per-output region count = T·d_head, input space)**: realized
   regions of `y_{last,0}` over 4000 random inputs, `d_model=12, d_head=6`:
   T=2,4,8,16,32,64 → 12, 24, 48, 96, 192, 368 vs the `T·d_head` bound 12, 24,
   48, 96, 192, 384 (exact through T=32; 96 % at T=64, finite-sample). PASS —
   linear growth in T.
2. **Prop 0 (q-space collapse control, FIXED K,V)**: realized regions of
   `y_d` over 4000 random queries → a flat **6 = d_head** across T=2…64,
   no T-dependence. PASS — confirms the analysis must be in input space.
3. **Prop 2 (multi-head mixed-volume growth)**: realized joint regions of
   `Σ_h y^h_{last,0}` over 6000 random inputs, T=8, d_head=6: H=1,2,3,4 → 48,
   1660, 5321, 5961, vs the additive line 48, 96, 144, 192 and the product
   bound 48, 2304, 110592, … . PASS — multiplicative (H=1→2 reaches 72 % of
   the product bound; the linear line is left far behind). H≥3 are
   sampling-saturated, not contradictory.
