# Demos ↔ Nanochat Integration Matrix

**Bead:** `model_guided_research-clq`
**Audit date:** 2026-06-14

This maps each root-level JAX **demo** (the "can this work?" exploration) to its
**nanochat** PyTorch counterpart (the production A/B testbed), with the exact
activation flag, config knobs, tensor-shape constraints, dispatch site, and any
remaining gap. **Bottom line: all 11 mathematical frameworks are already
integrated into nanochat** as drop-in attention/blocks, plus the HOSS optimizer,
the ordinal scheduler, the CA initializer, and the surreal/NSA width
parameterization. "Integration" is largely *done*; this doc is the reference
for *how to turn each on* and *what's still demo-only*.

## How dispatch works

`nanochat.gpt.resolve_attention_schedule(config)` expands
`GPTConfig.attention_type` into one mechanism per layer. A single mechanism
keeps the historical homogeneous model; a comma-separated pattern or `list[str]`
must either have length exactly `n_layer` or evenly divide `n_layer` and repeat.
`Block.__init__` then routes each layer's resolved mechanism:

- **Special blocks** that replace the whole Attention+MLP skeleton:
  - `gauge` → `GaugeBlock` (owns its residual + MLP slot)
  - `reversible` → `ReversibleBlock` over a half-width sub-config
    (`n_embd//2`, `n_head//2`, `n_kv_head` unchanged)
- **Attention-slot modules** (normal `x + attn(norm(x)); x + mlp(norm(x))`):
  tropical, ultrametric, simplicial, quaternion, braid, fractal, octonion,
  surreal
- **standard** (default `else`) → `CausalSelfAttention`

All non-special blocks share the FFN built by `_build_ffn(config)`, so
`ffn_type ∈ {standard, tropical, tropical-rational}` composes with *any*
attention (`gpt.py:366`).

## Attention frameworks (11)

| # | Framework | Demo (JAX, root) | Nanochat module | Activate | Status |
|---|---|---|---|---|---|
| 1 | Matrix/Gauge | `matrix_exponential_gauge_learning.py` | `gauge_block_torch.py` (special block) | `--attention-type gauge` | ✅ integrated |
| 2 | Ultrametric / p-adic | `ultrametric_worlds_and_p_adic_computation.py` | `ultrametric_attention_torch.py` | `--attention-type ultrametric` | ✅ kernel mode; ⚠️ trie CPU-only |
| 3 | Tropical / max-plus | `tropical_geometry_and_idempotent_algebra.py` | `tropical_attention_torch.py` | `--attention-type tropical` | ✅ integrated |
| 4 | Simplicial | `simplicial_complexes_and_higher_order_attention.py` | `simplicial_attention_torch.py` | `--attention-type simplicial` | ✅ integrated (KV-cache 2-hop) |
| 5 | Quaternion | `octonionic_quaternionic_signal_flow.py` | `quaternion_attention_torch.py` | `--attention-type quaternion` | ✅ integrated |
| 5b | Octonion | (same demo) | `octonion_attention_torch.py` | `--attention-type octonion` | ✅ integrated |
| 6 | Ordinal schedule | `ordinal_schedules_and_well_founded_optimization.py` | `ordinal_scheduler.py` | `--scheduler-type ordinal` | ✅ integrated (scheduler, not attn) |
| 7 | Reversible | `reversible_computation_and_measure_preserving_learning.py` | `reversible_block_torch.py` (special block) | `--attention-type reversible` | ✅ additive + symplectic |
| 8 | IFS / Fractal | `iterated_function_systems_and_fractal_memory.py` | `fractal_attention_torch.py` | `--attention-type fractal` | ✅ integrated |
| 9 | Knot / Braid | `knot_theoretic_programs_and_braid_based_attention.py` | `braid_attention_torch.py` | `--attention-type braid` | ✅ soft; ⚠️ discrete/YBE partial |
| 10 | Surreal / Transseries | `surreal_numbers_transseries_and_scaling.py` | `surreal_torch.py` (+ `parameterization.py`) | `--attention-type surreal` | ✅ attn; ⏳ NSA width param (lab.1) |
| 11 | Nonstandard / Hyperreal | `nonstandard_analysis_and_hyperreal_training.py` | `hoss_opt_torch.py` | `--optimizer-type hoss` | ✅ integrated (optimizer, not attn) |
| 12 | Clifford Cl(3,0) | `clifford_algebra_and_geometric_attention.py` | `clifford_attention_torch.py` | `--attention-type clifford` | ✅ integrated (`head_dim % 8 == 0`) |
| 13 | Hyperbolic / Lorentz | `hyperbolic_geometry_and_negative_curvature_attention.py` | `hyperbolic_attention_torch.py` | `--attention-type hyperbolic` | ✅ integrated (per-head learnable curvature) |

Knob caveat: not every mechanism hyperparameter has a CLI flag. Ultrametric
exposes `--ultrametric-mode`, `--ultrametric-hard-digits`, and
`--ultrametric-K`; `ultrametric_alpha`, `ultrametric_p`, and
`ultrametric_digits_k` are GPTConfig fields only (set them from Python or a
config file).

(Plus the CA initializer — `--ca-init-rule rule30|rule116` — an init-time
experiment that applies to any mechanism; `GPTConfig.ca_init_*`, `gpt.py:163`.)

## Tensor-shape constraints (the real "API gaps" for drop-in)

Universal (`gpt.py:419 _validate_config`): `n_embd % n_head == 0`;
`n_kv_head | n_head` and `≤ n_head` (GQA); `head_dim = n_embd//n_head` **even**
(RoPE). Per-layer schedules are validated before construction, with errors that
name the offending schedule entry or layer index. Per-mechanism additions:

| Mechanism | Extra constraint | Enforced at |
|---|---|---|
| quaternion | `n_embd % 4 == 0` **and** `head_dim % 4 == 0` | `quaternion_attention_torch.py:67,69` |
| octonion | `n_embd % 8 == 0` **and** `head_dim % 8 == 0` | `octonion_attention_torch.py:74,76` |
| reversible | `n_head` **even**; `n_kv_head | (n_head//2)` | `GPT._validate_config` |
| reversible | `reversible_tied` ⇒ `reversible_mode=symplectic` and every layer reversible | `GPT._validate_config` |
| ultrametric (trie) | CPU only | `ultrametric_attention_torch.py:55` |
| `disable_block_norms` | only when every scheduled layer is `standard` (z4xx control) | `GPT._validate_config` |

So e.g. the E1 rung `D128/H4` ⇒ `head_dim=32`, which satisfies %4 and %8 —
quaternion and octonion both run there. A `D128/H8` rung ⇒ `head_dim=16` (still
%8 ok); `D128/H16` ⇒ `head_dim=8` (octonion ok, quaternion ok). Pick head
counts so `head_dim` stays a multiple of 8 to keep every mechanism eligible.

## Config knobs per mechanism (all on `GPTConfig`, settable via `--…` flags)

- **standard:** `use_flex_attention`, `standard_record_attn_entropy` (`gpt.py:154,161`)
- **tropical:** `tropical_gauge_fix`, `tropical_score_center`,
  `tropical_record_margins`, `semiring_beta` (Maslov smoothing; tropical-only)
  (`gpt.py:167–174`)
- **FFN (any attn):** `ffn_type {standard,tropical,tropical-rational}`,
  `ffn_beta` (`gpt.py:179,183`)
- **ultrametric:** `ultrametric_mode {kernel,trie,balltree}`,
  `ultrametric_K/p/alpha`, `ultrametric_digits_k`, `ultrametric_hard_digits`,
  `eval_weight_quant_bits` (`gpt.py:185–198`)
- **reversible:** `reversible_mode {additive,symplectic}`, `reversible_tied`,
  `reversible_lambda_min`, `reversible_record_energy` (`gpt.py:205–218`)
- **braid:** `braid_mode {soft,discrete}`, `braid_tau`,
  `braid_crossing_law {restricted,ybe,rmatrix}`, `braid_record_schedule`,
  `braid_verify`, `braid_rmatrix_probes` (`gpt.py:220–225`)
- **quaternion/octonion/simplicial/fractal/surreal/gauge:** no extra knobs
  beyond dims (rotor normalization / routing are internal).

## Remaining demo→nanochat gaps (open beads)

These are demo features not (yet) fully realized in nanochat — the genuine
integration follow-ups, already tracked:

1. **Ultrametric packed-trie / sublinear retrieval** — nanochat kernel mode is
   O(T²); the demo's bitset trie isn't the production path here.
   → `model_guided_research-a1o`.
2. **Braid YBE-coherent discrete decoder** — `braid_mode=discrete` +
   `braid_crossing_law=ybe/rmatrix` exist but the full braid-word decoder is
   partial vs the demo's verified invariants. → `model_guided_research-k2y`.
3. **Tropical gauge-fixing certificates/margins** — emitted in the demo, not
   tracked in the nanochat attention path (only `tropical_record_margins`
   telemetry). → see `docs/nanochat_alignment_audit.md` §"Remaining Gaps".
4. **Gauge BCH/Magnus fusion** — demo roadmap item, mini-experiment bead
   `model_guided_research-2l8`.
5. **Surreal/NSA width parameterization** — `parameterization.py` per-mechanism
   width scaling from the standard-part criterion is **in progress**
   (`model_guided_research-lab.1`).
6. **Synaptic reconciliation** — `nanochat/synaptic.py` has a simplified
   consolidation path needing doc/spec reconciliation (alignment-audit item).

## Practical drop-in checklist (new mechanism or demo port)

1. Implement a `*CausalSelfAttention(config, layer_idx)` (attention slot) or a
   special block; honor the `(B, n_head, T, head_dim)` layout and the
   `forward(x, cos_sin, kv_cache)` contract used by `Block.forward`
   (`gpt.py:368`).
2. Add the dispatch branch in `Block.__init__` and any constraint in
   `_validate_config`.
3. Add config knobs to `GPTConfig` + the matching `--…` argparse flag in
   `train.py`; default to the behavior-preserving value.
4. Follow `docs/new_mechanism_checklist.md` (exact reduction-to-known certify,
   placebo control, validate-before-write, goldens recapture) before trusting
   its numbers.
5. Annotate FLOPs: confirm `estimate_flops()` (`gpt.py:665`) accounts for any
   non-standard backward (cf. the reversible-symplectic 3× correction, 7lba).

No code changes are made by this audit — the integration is in place; this is
the map plus the open-gap list.
</content>
