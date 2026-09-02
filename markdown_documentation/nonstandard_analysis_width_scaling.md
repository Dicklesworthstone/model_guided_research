# Nonstandard Analysis and Per-Mechanism Width Scaling

**Bead:** lab.1 (EPIC THEORY-III) · **Module:** `nanochat/parameterization.py` · **Tests:** `tests/test_parameterization.py` · **Capstone client:** cbm (position paper). Every numerical claim below was validated before writing (receipts: `tests/test_parameterization.py` + the validation in §8).

---

## 0. The one-sentence stakes

muP / abc-parameterization (Yang–Hu, *Tensor Programs V*) tells you the
init-scale and learning-rate exponents that keep activations and updates
Θ(1) as width N grows — **assuming the CLT universality class** (each
pre-activation is a sum of many iid-ish terms, so it concentrates like
√N·Gaussian). Several mechanisms in this repo are **not iid-sum machines**,
so they live in a different concentration class with a *different* correct
scaling. Parameterize them by the CLT rule and every width-varying
comparison (w94.1, EPIC-E) is silently confounded — "tropical scales better
than standard across widths" stops being a meaningful sentence. This note
derives the right rule per class and gives the coordinate-check that
verifies it.

## 1. The nonstandard-analysis framework (self-contained)

Standard derivations of muP track variances through the forward/backward
pass and demand they stay O(1). Nonstandard analysis (NSA) gives the same
equations by a cleaner route that does not assume Gaussianity.

Work in the hyperreal extension *ℝ. Take the width **N hyperfinite** — an
actual infinite integer (a member of *ℕ ∖ ℕ). Run the forward and backward
pass symbolically at this infinite width. Each activation and each per-step
parameter update is now a hyperreal number. **The parameterization is
correct iff every activation and every update has a finite, non-infinitesimal
standard part** st(·): finite rules out blow-up (an infinite activation), and
non-infinitesimal rules out collapse (an activation that vanishes relative to
the scale that carries signal). The exponent equations are exactly the
*existence conditions* for st(·) of these quantities.

The transfer principle does the descent: any first-order statement true of
the hyperfinite-width network is true of all sufficiently large finite-width
networks. So a parameterization that makes st(·) exist at hyperfinite N is
the correct large-finite-N parameterization. No CLT is invoked — only the
order of magnitude of the relevant concentration, which is what differs by
class. (Anderson's hyperfinite construction of Brownian motion reappears in
§6 for the optimizer's shadow dynamics.)

## 2. Concentration classes and the width-scaling table

The repo's mechanisms fall into four classes by *how their characteristic
aggregation concentrates*. The table (canonical form in
`parameterization.py::WIDTH_SCALING_TABLE`) records, per mechanism, the
class and the exponents **relative to the CLT-muP baseline** (init std
∝ N^−1/2, forward multiplier 1/√fan-in, Adam-class hidden LR exponent 0): an
exponent of 0.0 means "CLT-muP is already correct for this layer."

| mechanism | class | init exp | LR exp | forward multiplier | correction |
|---|---|---|---|---|---|
| standard | CLT (Gaussian sum) | 0 | 0 | 1/√d | baseline |
| reversible | CLT (volume-preserving) | 0 | 0 | 1/√d | coupling preserves the sub-layers' class |
| **tropical** | **EVT (Gumbel max)** | 0 | 0 | **subtract aₙ (per-stage)** | **the score scale grows √(2 ln N) per max-axis** |
| ultrametric | branching / geometric | 0 | 0 | 1 (LCP-weighted) | α/β temperature tracks log-depth |
| quaternion | isometry (normed algebra) | 0 | **−1** | 1 (norm-preserving) | rotor-param LR splits from magnitude |
| octonion | isometry (normed algebra) | 0 | **−1** | 1 (norm-preserving) | same as quaternion (non-associative) |

### 2(a) Tropical / max-plus — the EVT class (the novel contribution)

A tropical score `s_i = max_j (q_i·k_j)/√d` is a **maximum** over N
competitors, not a sum. The max of N iid N(0,1) variables does not
concentrate at O(1) like a CLT sum — it concentrates at the **Gumbel
location** aₙ ≈ √(2 ln N), with fluctuations of order 1/√(2 ln N) (Gumbel
class). So a max-plus layer fed unit-scale scores produces an output whose
*location grows with √(2 ln N)*. CLT-muP, which expects an O(1) pre-activation,
treats this growth as model signal and mis-scales the downstream init/LR.

Validated (`measure_activation_scale`-style check, §8): the mean max-logit
over a unit-variance score field rises 1.73 → 3.59 as context N goes
16 → 4096; the CLT class is flat over the same range.

**The correct (a)-class rule is an *additive per-stage location shift***,
not a power of N: subtract aₙ from the post-max stage so its standard part is
Θ(1). Two refinements that matter in practice:

- **Per-stage, not per-network.** The shift applies where the stage *input*
  is unit-scale (the normed residual stream). A stage whose input is the
  *post-max* distribution (Gumbel fluctuations ~1/√(2 ln N), not unit scale)
  needs only a second-order correction; applying the unit-scale offset there
  overshoots by ~√(2 ln N). The table gives the rule per stage position.
- **Finite-N constant ≠ asymptote.** The bare √(2 ln N) is up to **33% high**
  at N=16 (and the standard second-order asymptote is ~9% *low*). The exact
  finite-N location is the order-statistic mean E[max] = ∫ x·N·φ(x)·Φ(x)^{N−1}
  dx, which matches Monte Carlo within 0.2% at every N. **The table carries
  the exact E[max] for constants and the asymptotic √(2 ln N) for the scaling
  law** — `exact_expected_max(n)` vs `gumbel_asymptotic_location(n)`. A
  smoke-scale validation against the bare asymptote would falsely refute
  correct theory; this is the single most important implementation footgun
  in the whole bead, caught numerics-first.

This row also covers the **tropical FFN** (8gk.8): a max over `d_ff` additive
terms is the same Gumbel class with `d_ff` as the width variable; 8gk.8 cites
this row rather than re-deriving.

### 2(b) Ultrametric / LCP — branching / geometric class

Digit-match indicators multiply along depth-K prefixes; the longest-common-
prefix depth between a query and a key population of size M grows like
log_p M (the height of the trie that separates M keys). The digit
projections themselves are CLT-class (ordinary linear maps, init ∝ N^−1/2),
but the **α/β temperature** that converts LCP depth to a weight must track
log-depth: if the key population grows but β is fixed, the effective
weighting either saturates (all mass on the deepest match) or washes out.
The Θ(1)-weight condition is β ∝ 1/log(M) for the LCP-exponential kernel —
a logarithmic, not power-law, correction, consistent with the geometric
concentration of LCP depth.

### 2(c) Quaternion / octonion / Clifford — isometry class

Unit-rotor products are **exact isometries**: ‖r ⊗ v‖ = ‖v‖ to machine
precision (validated: max error 8.9e−16). So the forward scale is
width-independent and benign — **no forward multiplier and no init-exponent
correction**. The subtlety is on the backward side: rotor *parameters* take
their gradient *through the normalization / exponential map* that keeps r on
the unit sphere, so the gradient of a rotor parameter scales differently from
a plain magnitude parameter. The Θ(1)-update condition gives the rotor
parameters an **LR exponent of −1 relative to magnitude parameters** — which
is exactly the per-group learning-rate split already present in the
optimizer setup (Muon/AdamW groups). This bead supplies its derivation: the
split is not a heuristic, it is the isometry class's abc-rule.

### 2(d) Reversible — CLT, preserved

Additive and symplectic (u55.5) coupling are measure-preserving maps of
CLT-class sub-blocks (the F/G attention and MLP). Volume/symplectic
preservation does not change the concentration class of the sub-layers, so
CLT-muP applies to each half-stream unchanged.

## 3. The N-axes enumeration (scope, narrowed and extended)

**Narrowed (precise confound statement).** "Every width-varying comparison is
confounded" overclaims. The precise statement: a comparison is confounded for
a **max-like** mechanism exactly when the varied quantity changes (i) the
number of competitors in any max, or (ii) the score variance/correlation
entering the max. CLT-class mechanisms at matched parameterization are fine;
so are max-like mechanisms when the swept axis touches neither (i) nor (ii).

**Extended (the new, testable consequence).** Tropical attention has *several
simultaneous max-axes*: head_dim `d` (max over the feature dimension in some
score forms), **context length `T`** (max over keys in aggregation), and head
count via score variance under width scaling. The `T`-axis consequence is new
and is **not a width effect at all** — it is a *length* effect that CLT
reasoning misses entirely:

> **EVT length-extrapolation prediction.** Tropical attention's effective
> routing temperature drifts with √(2 ln T) as context length T grows. Its
> margin (γ) statistics — which the mechanism already logs — shift with ln T
> in a shape predictable *before* measurement from the exact E[max] curve.

This cross-links C5 (length extrapolation) and E3 (length scaling): their
reports should test the logged γ statistics for the √(2 ln T) signature. It
is registered as its own hypothesis (§7).

## 4. The coordinate-check harness (the muP acceptance test)

`parameterization.py::coordinate_check(mechanism, widths)` builds a one-layer
GPT of the given mechanism at each width (head_dim fixed at 16, n_head grown
with N — the muP convention), runs one forward pass on random tokens at init,
measures the residual-stream activation RMS, and fits the **log-log slope of
RMS vs width**. The acceptance criterion is the muP standard:

- **Correctly parameterized ⟹ flat line** (|slope| ≈ 0). The CLT control
  (standard) gives slope +0.003 (validated, the test that gates the harness:
  if *this* drifts, the apparatus is broken, not the theory).
- **A parameterization test that cannot fail proves nothing** — so the
  tropical check must also be run under the *deliberately wrong* CLT scaling
  and show drift, and flatten only under the EVT (per-stage aₙ) correction.
  The harness exposes both arms; §7 registers the separation as a both-ways
  prediction (flat under derived, drifting under CLT-wrong).

The harness is generic over mechanisms via a single config table and is
CPU-light (one forward per width, tiny model) — it runs in-process, no
training subprocesses.

## 5. Wiring (deferred to the campaign follow-on)

The derived rules wire into nanochat as a per-mechanism multiplier table
behind `--parameterization {current, nsa}`, default `current` until E1 adopts
`nsa` deliberately. The table is `WIDTH_SCALING_TABLE`; the only mechanisms
whose `nsa` rule differs from `current` today are tropical (the per-stage aₙ
shift) and quaternion/octonion (the rotor-parameter LR split, which the
optimizer already approximates). This wiring plus the **transfer test** (the
muP acceptance standard: an LR tuned at width 128 transfers within 2× to
width 1024 under the derived rule, and *fails* to transfer under the wrong
one for tropical) needs a width-ladder training campaign and is filed as the
box-gated follow-on bead. The theory + coordinate-check apparatus above is
self-contained and adjudicable without it.

## 6. HOSS shadow-SDE section

NSA frames SGD as a hyperfinite random walk whose standard part **is** an
SDE (Anderson's construction: a hyperfinite random walk with infinitesimal
steps has a standard part that is Brownian motion). The HOSS optimizer's
OU macro-step couples the step size δ to a curvature scale; under the
hyperfinite lens its trajectory's standard part is the Ornstein–Uhlenbeck
SDE the demo motivates heuristically. This puts the hoss_opt "shadow
dynamics" framing on rigorous footing: the shadow SDE is st(·) of the
hyperfinite walk, and the coupling δ ∝ 1/√curvature is the condition for
that standard part to be a non-degenerate diffusion (neither frozen nor
exploding) — the same standard-part criterion as §1, applied in time rather
than width.

## 7. Registered predictions (registry-ready)

Register with the session's preregistration discipline (`candidate_variant`
key; coordinate-check artifacts, not trained-model evals, are the evidence
for 1–2; the γ-signature 3 uses the tropical margin telemetry):

1. **hyp-coordcheck-clt-flat** — standard, reversible coordinate-check
   log-log slope |slope| ≤ 0.05 over widths {64…2048} (the CLT class is
   correctly parameterized as-is). Single-arm, deterministic-seeded.
2. **hyp-tropical-evt-miscoupling** — the both-ways separation: tropical's
   coordinate-check slope is ≥ 0.10 under CLT-wrong scaling (score scale
   drifts) and ≤ 0.05 under the derived per-stage aₙ correction. The
   falsifiable content is the *separation* — a flat line under both would
   refute the EVT account.
3. **hyp-tropical-length-evt-signature** — tropical margin (γ) statistics
   shift with √(2 ln T) as eval context length T grows (C5/E3 telemetry);
   the fitted ln T coefficient is positive and within a CI of the exact
   E[max] slope. Registered before the C5/E3 length sweeps.

(Exact thresholds for 2–3 to be pinned from a sizing probe under the
`artifacts/probes/sizing/` quarantine before the campaign, per the dzor
policy and the `mgr hypotheses power` instrument.)

## 8. Positioning and validation receipts

**Positioning.** *Tensor Programs V* (Yang et al.) is the CLT-class baseline
being extended; the classical extreme-value / Gumbel literature supplies the
max-asymptotics. The contribution absent from the current literature is
**abc-parameterization for non-CLT concentration classes** — specifically the
EVT (max-plus) class, where the correct rule is an additive per-stage Gumbel
location shift rather than a power-law multiplier, and the length-axis EVT
correction to extrapolation.

**Receipts** (all in `tests/test_parameterization.py`, run before this note
was written): exact E[max] within 1.5% of Monte Carlo at N ∈ {16…1024}; the
second-order asymptote >5% off at N=16 and converging (the finite-N lesson);
the √(2 ln N) scaling law confirmed via the exact/asymptote ratio rising
toward 1; the unit-rotor isometry at 8.9e−16; and the coordinate-check CLT
control flat at slope +0.003. The standalone validation script that seeded
these lives at the validation-scratch path.

## 9. Coordinate-check results (2026-09-02) — what the ladder actually showed

Measured with `mgr coord-check` from a clean tree (init-time forwards, widths
64…2048, seq 32, batch 8; artifacts under `artifacts/bench/coord_curves/`),
adjudicated by `mgr adjudicate` under ci-v6:

| arm | |slope| per seed | verdict |
|---|---|---|
| standard (CLT control) | 0.0015 / 0.0011 / 0.0009 | `hyp-coordcheck-clt-flat` SUPPORTED |
| reversible (CLT, volume-preserving) | 0.0005 / 0.0009 / 0.0003 | (same entry, for-all) |
| tropical, tropical FFN, current rule | 0.10 / 0.12 / 0.06 / 0.11 / 0.11 / 0.11 | drifts, as predicted |
| tropical, tropical FFN, nsa rule | 0.12 / 0.14 / 0.08 / 0.13 / 0.13 / 0.13 | drifts MORE: `hyp-tropical-evt-miscoupling` REFUTED |

The separation prediction of section 2(a) fails: nsa minus current |slope| is
+0.019 (CI [−0.009, 0.047], six seeds per arm, 98% power) against the
registered −0.05. The refuted entry stays in the registry as written.

**Where the drift enters (bead 1xov, per-stage hooks at seed 0).** The
attention path alone is the source: tropical attention with the *standard*
FFN already drifts (|slope| 0.06/0.08/0.03, identical under both rules, since
the score-stage shift commutes with per-query centering and cancels), while
standard attention with the *tropical* FFN is flat (−0.02/−0.01/−0.01). Inside
the block, standard attention's output RMS is 0.15 at every width with a
per-channel mean of 0.02–0.03; tropical attention's output RMS grows
0.97 → 1.20 → 1.33 → 1.51 across widths 64 → 2048 and its per-channel mean
grows 0.85 → 1.09 → 1.23 → 1.40, so almost the whole attention output is a
common-mode location offset that grows with width. That offset comes from the
post-max value aggregation y_d = max_k (s_k + v_{k,d}), which section 2(a)
declared second-order and left unshifted; the tropical MLP's output then grows
(1.6 → 2.5) only because its input carries the offset. The exact-E[max] shift
at the score stage — the rule the registered nsa arm implements — cannot touch
this, which is why the arm was refuted rather than merely underpowered.

**Growth law of the offset** (seeds 0–2, tropical attention, standard FFN):
the per-channel offset is a single-signed, negative constant across channels
(signed mean −1.40 / −1.36 / −1.34 at width 2048) whose magnitude grows as
width^0.12 / width^0.15 / width^0.06 over 64…2048, while the *centered* RMS of
the attention output is flat (log-log slope 0.03 / 0.05 / 0.00). The drift is
therefore purely a location effect of the max-plus value stage under score
centering (s_k ≤ 0 pulls every y_d = max_k (s_k + v_{k,d}) below the value
maximum by an amount that grows slowly with the head count); the spread is
already correctly parameterized, and no scale rule can fix a location term.

**What a successor would have to claim** (not registered here; a claim needs
its own pre-evidence entry): a location correction at the value-aggregation
stage — subtracting the expected value-stage location as a function of Tk and
the head count, the same way the score stage subtracts E[max] — flattens the
tropical curve on a fresh seed set. Bead 1xov holds the design.
