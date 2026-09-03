# New-Mechanism Merge Gate (Checklist)

**Every new attention mechanism (an F-series implementation, e.g. mnn.2/3/5/6)
passes this gate before it is believed.** The gate is the cheap correctness
machinery that catches what benchmarks miss — and it has already paid off:
the `mnn.4` hyperbolic design's c→0 reduction was *wrong as written* (raw
distances reduce to a Laplacian kernel, not dot-product attention), and the
reduction-test reasoning is exactly what exposed it before a line of the
mechanism was implemented (see `gromov_product_and_the_ultrametric_hyperbolic_bridge.md` §3).

A mechanism that ships without these is a mechanism whose benchmark numbers you
cannot trust, because you have no independent check that it computes what it
claims. Order matters: items 1–4 are *design-time* (do them before writing the
torch module), 5–8 are *implementation-time*, 9 is *campaign-time* (and lives
in `campaign_preregistration_template.md`).

## The gate

### 1. Exact reduction to a known mechanism — as a `mgr certify` check (the keystone)

If the mechanism generalizes an existing one, there is a parameter regime where
it must reproduce that one **exactly** (to fp32 tolerance). Derive it, validate
it numerically *before writing the design*, and ship it as a certify check
under the new mechanism's name.

- Clifford restricted to the even subalgebra of Cl(3,0) == quaternion
  (`mnn.1` §3; the basis convention i=e₂₃, j=−e₁₃, k=−e₁₂ verified against the
  *implemented* `qmul` — a wrong guess fails silently, which is the point).
- Hyperbolic at c→0 == standard attention — **but only the energy-form score**;
  the raw-distance score reduces to a distance kernel (`8gk.6` §3, the caught
  bug). The reduction test *is* the spec: it tells you which score form is
  correct.
- Hyperbolic rescaled c→∞ == ultrametric LCP attention (`8gk.6` §4).

If the mechanism is genuinely novel (no sub-mechanism), substitute an **exact
hand-computable special case** (e.g. tropical's β→∞ == hard-max with the
LSE–max sandwich; ultrametric hard-digit == exact trie decode). The principle
is the same: one input regime where the answer is known by other means.

### 2. Structure-free placebo control (design the prediction now)

If the mechanism claims to exploit a structure (hierarchy, brackets,
composition, scale), its registry prediction must come with a placebo arm on
structure-free data at equal budget. "Wins on placebo too" ⇒ the win is not
about that structure. The mechanism's interpretability observable (item 6)
should *collapse to trivial* on placebo (e.g. hyperbolic curvature → 0); that
collapse is the honest null and is welcomed. (Campaign wiring: §6 of the
pre-registration template.)

### 3. Parameterization coordinate-check (`lab.1`)

Before any matched-FLOPs comparison across widths, verify the mechanism's
init/LR scaling with a coordinate check (activation- and update-scale flat in
log-width). Non-CLT mechanisms (tropical/max-plus are extreme-value/Gumbel
class, not Gaussian-sum) need different scalings than muP assumes; "X beats Y
at matched FLOPs" can otherwise just mean "Y is mis-scaled at this width."
This is the standing `lab.1` harness; a new mechanism registers its
concentration class and gets one coordinate-check row.

### 4. Validate every theory claim numerically before writing the design doc

Build the algebra/geometry from first principles in a scratch script, assert
the load-bearing identities, and put the receipts in the doc's appendix
(the `mnn.1` Appendix A / `8gk.6` Appendix A pattern). Pins conventions and
catches sign/limit errors that survive prose review.

### 5. Numerics policy, explicit and certified

State the saturation/overflow budget, the clamp thresholds, and the bf16
policy (which products run in fp32 islands under autocast — the same class as
tropical-LSE and the Lorentz-Minkowski products). Each policy gets a certify
assertion (constraint residual, round-trip, etc.). A mechanism that "lives or
dies on numerics" (hyperbolic, tropical) puts this section first.

### 6. A first-class interpretability observable into the metrics stream

The mechanism's learned-structure readout (tropical route coverage, hyperbolic
per-head curvature, Clifford per-grade norms, braid conserved charges) streams
to `metrics.jsonl` and into the run `summary.json` `results` block, so it is
*adjudicable from the train artifact alone* (registry predictions can target
`train:results.<observable>`). Not an afterthought — it is often the cleanest
falsifiable claim the mechanism has.

### 7. Goldens recapture in the SAME commit as any GPTConfig field

Adding a `GPTConfig` field trips the attention-goldens config-drift guard.
Recapture with `MGR_CAPTURE_ATTENTION_GOLDENS=1`, then **verify the diff is
config-only** (one inserted line per fixture, zero trajectory change) in the
same commit. The `eval_weight_quant_bits` field broke the gate for a day
unnoticed (fixed in fdxb) — this line is the lesson.

### 8. Standard invariant checks

Causality (no future-token gradient — every mechanism already has
`causality_no_future_grad`), plus the mechanism's algebra/conservation laws
(norm preservation for rotor mechanisms, Lipschitz bounds for tropical, mass
conservation for simplicial, charge conservation for braid). These are the
existing certify family; a new mechanism adds its own.

### 9. Off-floor campaign with pre-registered rung & stopping rule

The mechanism's headline comparison follows `campaign_preregistration_template.md`:
Phase-0 rung-finding, sample-efficiency vs asymptotic split, power-derived
seeds, one pre-registered stopping rule, quarantined probes, single `-H`
adjudication. **A floored win is not a structural win** (braid–Dyck).

## Copy-paste block for an F-series mechanism bead

```
NEW-MECHANISM GATE (docs/new_mechanism_checklist.md) — mechanism <X>
[ ] 1. Reduction to <known mechanism> in regime <...> == <known>, fp32, as a certify check  (validated in scratch: <receipt>)
[ ] 2. Placebo control designed into the registry prediction; observable collapses to trivial on placebo
[ ] 3. Concentration class declared; lab.1 coordinate-check row green
[ ] 4. Theory claims validated-before-written; receipts in design-doc appendix
[ ] 5. Numerics policy explicit (saturation/clamp/bf16 fp32-islands) + certify assertions
[ ] 6. Interpretability observable in metrics.jsonl + summary.results (train:results.<obs> adjudicable)
[ ] 7. Goldens recaptured in the GPTConfig-field commit; diff verified config-only
[ ] 8. Causality + algebra/conservation certify checks
[ ] 9. Off-floor campaign per the pre-registration template (rung found, claims split, one -H append)
```

## Audit: where the existing mechanisms stand (2026-06-14)

Honest snapshot from `mgr certify` (55/55 pass) and the registry.
✓ present · ✗ gap · — N/A.

| mechanism | causality | algebra/invariant laws | **reduction-to-known certify** | coordinate-check (lab.1) | goldens |
|---|---|---|---|---|---|
| standard | ✓ | ✓ (rope/rmsnorm/softmax) | — (is the baseline) | ✓ flat, \|slope\| 0.0015 (hyp-coordcheck-clt-flat SUPPORTED 2026-09-02) | ✓ |
| tropical | ✓ | ✓ (Lipschitz, ffn-collapse, margin) | ✓ (maslov_endpoint_within_sandwich) | ✗ drifts, \|slope\| ~0.10 under both rules (hyp-tropical-evt-miscoupling REFUTED; bead 1xov) | ✓ |
| ultrametric | ✓ | ✓ (strong-triangle LCP) | ✓ (trie_decode_matches_hard_kernel) | ✓ flat, 0.0014-0.0024 (2026-09-03) | ✓ |
| quaternion | ✓ | ✓ (assoc, norm, rotor) | — (is a reduction *target*) | ✓ flat, 0.0008-0.0019; nsa identical at init (rotor rule changes the LR only) | ✓ |
| octonion | ✓ | ✓ (alternativity, norm, non-assoc) | ✓ (reduces_to_quaternion_on_subalgebra) | ✓ flat, 0.0006-0.0015; nsa identical at init | ✓ |
| braid | ✓ | ✓ (YBE, charges, r-matrix) | — analyzed 2026-09-03: sigmoid additive accumulation (not a softmax) has no standard-attention limit; the algebraic laws are the known-answer checks | ✓ flat, 0.0006-0.0016 | ✓ |
| gauge | ✓ | ✓ (rotation roundtrip/additivity) | ✓ (zero_transport_reduces_to_standard_attention: zero connection = plain SDPA on the same projections, no QK-norm; kill witness in tests) | ✓ flat, 0.0001-0.0019 | ✓ |
| reversible | ✓ | ✓ (inverse roundtrip, autograd parity) | — analyzed 2026-09-03: the RevNet coupling y1 = x1 + F(x2), y2 = x2 + G(y1) has no plain-block limit; forward_inverse_roundtrip and symplectic_jacobian are the known-answer checks | ✓ flat, 0.0003-0.0009 (hyp-coordcheck-clt-flat SUPPORTED) | ✓ |
| hyperbolic | ✓ | ✓ (Lorentz constraint, exp/log roundtrip) | ✓ (energy_gromov_reduces_to_standard) | ✓ flat, 0.0009-0.0013 | ✓ |
| simplicial | ✓ | ✓ (mass conservation) | ✓ (zero_triangle_reduces_to_standard_attention: two-hop weight zero = causal SDPA; proxy tier, the faithful 2-simplex build must keep it) | ✓ flat, 0.0008-0.0016 | ✓ |
| surreal | ✓ | ✓ (row-norm, linearity, equivariance) | — (no simpler sub-mechanism) | ✓ flat, 0.0001-0.0026 | ✓ |
| fractal | ✓ | ✓ (router simplex) | — (no simpler sub-mechanism) | ✓ flat, 0.0012-0.0022 | ✓ |

Coordinate-check readings are `mgr coord-check` log-log slopes of activation
RMS vs width (64..2048, three seeds at init; artifacts under
`artifacts/bench/coord_curves/`). Seeds 0-2 measured the table; the class-level
flatness hypotheses registered on 2026-09-03 are adjudicated on seeds 3-5.

**The reduction-to-known column is now filled wherever a clean reduction
exists** (bead uvjq, 2026-06-14): octonion ⊃ quaternion, tropical → its exact
β→∞ endpoint, ultrametric kernel == trie. `gauge → standard` was analyzed and
deliberately left out — the GaugeBlock omits the QK-norm that standard
attention applies, so it reduces only to QK-norm-stripped dot-product
attention, not the standard *mechanism* (its transport machinery is already
certified by the `rotation_*` checks). braid/fractal/simplicial/surreal are
not generalizations of a simpler attention mechanism, so have no
reduction-to-known to certify. The coordinate-check column is the remaining
gap, empty because `lab.1` (the muP parameterization harness) is open — that is
where a new mechanism's init/LR scaling gets verified before any matched-FLOPs
width comparison. New mechanisms enter the gate green by construction; the
reduction retrofits for existing mechanisms are complete (bead uvjq closed
2026-06-14).
