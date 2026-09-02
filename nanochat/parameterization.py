"""Per-mechanism width-scaling parameterization (bead lab.1).

muP / abc-parameterization (Yang-Hu, Tensor Programs V) fixes init-scale and
learning-rate exponents so activations and updates stay Theta(1) as width N
grows -- but it assumes the CLT universality class (iid Gaussian sums). This
project's mechanisms are NOT all iid-sum machines, so several live in a
DIFFERENT concentration class with different correct scalings. Getting this
wrong silently confounds every width-varying A/B (w94.1, EPIC-E): "tropical
scales better than standard" is only a meaningful sentence once both arms are
correctly parameterized.

This module is the canonical home for:
  * the extreme-value (Gumbel) primitives the max-plus class needs,
  * the per-mechanism width-scaling table (concentration class + exponents),
  * a coordinate-check harness that measures init-time activation scale vs
    width and fits the log-log slope (the forward half of the muP acceptance
    test; the multi-step update-scale check and the LR-transfer test are the
    box-gated follow-on bp08).

Theory note: markdown_documentation/nonstandard_analysis_width_scaling.md.
All numerical claims here are validated in tests/test_parameterization.py.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Extreme-value (Gumbel) primitives -- the (a)-class ingredient.
# A max over N iid N(0,1) scores concentrates at sqrt(2 ln N), NOT at the
# O(1) the CLT class assumes. The asymptotic location captures the SCALING
# LAW (the coordinate-check slope) but is ~10% off at practical small N; the
# exact order-statistic mean is the finite-N CONSTANT the table must carry
# (validated: exact within 0.2% of Monte Carlo at N in {16..4096}, asymptote
# up to ~33% off at N=16).
# ---------------------------------------------------------------------------


def gumbel_asymptotic_location(n: int) -> float:
    """Second-order asymptotic location of max of n iid N(0,1):
    a_n = sqrt(2 ln n) - (ln ln n + ln 4pi) / (2 sqrt(2 ln n)).
    Use for the SCALING LAW (slopes); see exact_expected_max for constants."""
    if n < 2:
        return 0.0
    ln_n = math.log(n)
    a = math.sqrt(2.0 * ln_n)
    return a - (math.log(ln_n) + math.log(4.0 * math.pi)) / (2.0 * a)


@functools.cache
def exact_expected_max(n: int) -> float:
    """E[max of n iid N(0,1)] via quadrature of the order-statistic mean
    integral ∫ x · n · phi(x) · Phi(x)^(n-1) dx. The exact finite-n location
    the width-scaling table carries (the asymptotic form is up to ~10% off at
    small n -- a smoke-scale validation against the asymptote would falsely
    refute correct theory)."""
    if n < 2:
        return 0.0
    from scipy import integrate, stats

    def integrand(x: float) -> float:
        return float(x * n * stats.norm.pdf(x) * stats.norm.cdf(x) ** (n - 1))

    val, _ = integrate.quad(integrand, -8.0, 12.0, limit=200)
    return float(val)


# ---------------------------------------------------------------------------
# The per-mechanism width-scaling table.
# init_exponent / lr_exponent are the powers of width N in the init std and
# the per-step learning-rate multiplier (relative to the CLT-muP baseline of
# init ~ N^-0.5, hidden LR ~ N^-1 for Adam-class). multiplier is the forward
# output multiplier. notes records the class-specific correction.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalingRule:
    mechanism: str
    concentration_class: str
    init_exponent: float  # power of N in init std, relative to unit-fan-in
    lr_exponent: float  # power of N in the per-param LR multiplier
    forward_multiplier: str  # symbolic forward output multiplier
    notes: str


# Baseline CLT-muP: init std ~ N^-1/2, output multiplier 1/sqrt(fan_in), and
# (Adam-class) hidden LR ~ N^0 per Tensor Programs V's "Adam muP" (the LR
# exponent that keeps updates Theta(1) for Adam is 0 for hidden layers, -1
# for SGD). We record exponents RELATIVE to this baseline, so 0.0 == "the
# CLT-muP rule is correct for this layer".
WIDTH_SCALING_TABLE: dict[str, ScalingRule] = {
    "standard": ScalingRule(
        "standard",
        "CLT (Gaussian sum)",
        0.0,
        0.0,
        "1/sqrt(d_head)",
        "the baseline; softmax logits are an iid-sum, CLT-muP is correct as-is",
    ),
    "tropical": ScalingRule(
        "tropical",
        "EVT (Gumbel max)",
        0.0,
        0.0,
        "subtract a_N (per-stage)",
        "max over scores is Gumbel, not Gaussian: the score scale GROWS as "
        "sqrt(2 ln N) over each max-axis (head_dim, context T). The correction "
        "is an ADDITIVE per-stage location shift (subtract the exact E[max]), "
        "not a power of N. Applied where the stage input is unit-scale (normed "
        "residual stream); a post-max stage needs only a 2nd-order correction.",
    ),
    "ultrametric": ScalingRule(
        "ultrametric",
        "branching / geometric (LCP depth)",
        0.0,
        0.0,
        "1 (LCP-weighted)",
        "digit-match indicators multiply along depth-K prefixes; expected LCP "
        "depth grows ~log(number of distinguishable keys). Digit projections "
        "scale CLT-normally; the alpha/beta temperature must track the log-depth "
        "so the weighting stays Theta(1) as the key population grows.",
    ),
    "hyperbolic": ScalingRule(
        "hyperbolic",
        "radial / curvature-gated Lorentz energy",
        0.0,
        0.0,
        "1/sqrt(d_head)",
        "the energy-Gromov score has the dot-product CLT limit at c -> 0; "
        "finite curvature adds a bounded radial correction after QK norm. "
        "Curvature is a per-head scalar coordinate and keeps the baseline "
        "projection/LR scaling until the coordinate-check evidence says otherwise.",
    ),
    "quaternion": ScalingRule(
        "quaternion",
        "isometry (normed algebra)",
        0.0,
        -1.0,
        "1 (norm-preserving)",
        "unit-rotor products are exact isometries (forward scale is "
        "width-independent, benign). But rotor PARAMETERS take gradients "
        "through the normalization/exp map, so their LR exponent splits from "
        "magnitude params -- this is the per-group LR split already in the "
        "optimizer setup, here given its derivation.",
    ),
    "octonion": ScalingRule(
        "octonion",
        "isometry (normed algebra)",
        0.0,
        -1.0,
        "1 (norm-preserving)",
        "same isometry class as quaternion (non-associative but still "
        "norm-multiplicative); the rotor-parameter LR split applies identically.",
    ),
    "reversible": ScalingRule(
        "reversible",
        "CLT (volume-preserving)",
        0.0,
        0.0,
        "1/sqrt(d_head)",
        "additive/symplectic coupling of CLT-class sub-blocks; the coupling is "
        "measure-preserving so it does not change the concentration class of "
        "its F/G sub-layers -- CLT-muP applies to each half-stream.",
    ),
}


def scaling_rule(mechanism: str) -> ScalingRule:
    """Lookup with a CLT fallback for mechanisms not yet classified (the
    conservative default: treat as CLT and let the coordinate check flag drift)."""
    return WIDTH_SCALING_TABLE.get(
        mechanism,
        ScalingRule(
            mechanism,
            "CLT (assumed)",
            0.0,
            0.0,
            "1/sqrt(d_head)",
            "unclassified mechanism; CLT-muP assumed -- run the coordinate check",
        ),
    )


# ---------------------------------------------------------------------------
# Coordinate-check harness (the muP acceptance test).
# For a width ladder, measure the residual-stream activation RMS after one
# forward pass of a tiny one-layer GPT of the given mechanism, and fit the
# log-log slope of RMS vs width. A correctly parameterized layer gives a FLAT
# line (|slope| ~ 0); a mis-parameterized one drifts.
# ---------------------------------------------------------------------------


def measure_activation_scale(
    attention_type: str,
    width: int,
    *,
    n_head: int = 4,
    seq_len: int = 32,
    batch_size: int = 8,
    seed: int = 0,
    device: str = "cpu",
    parameterization: str = "current",
    ffn_type: str = "standard",
) -> float:
    """Mean RMS of the post-block residual-stream activation for a one-layer
    GPT of `attention_type` at embedding width `width`, on random tokens at
    init. The coordinate-check observable: flat in width == correctly scaled.

    ffn_type defaults to "standard" so the historical CLT-control numbers are
    unchanged; EVT-class arms pass ffn_type="tropical" - the tropical MLP
    stage-1 bias init is WHERE the exact-E[max] correction lives, so without
    it the current-vs-nsa arms are provably identical (the attention-path
    score shift cancels under default empirical centering)."""
    import torch

    from nanochat.gpt import GPT, GPTConfig

    torch.manual_seed(seed)
    # keep head_dim fixed as width grows (the muP convention: vary n_head with N)
    head_dim = 16
    heads = max(1, width // head_dim)
    # reversible halves query heads and needs even head count; round to even
    if attention_type == "reversible" and heads % 2 == 1:
        heads += 1
    # n_kv_head must divide n_head (GPTConfig invariant). Aim for ~half the
    # heads to exercise GQA, then walk down to the nearest divisor of n_head -
    # heads//2 does NOT divide odd head counts >= 5 (e.g. width=80 -> heads=5
    # -> kv=2 -> 5 % 2 != 0), which would otherwise crash the coordinate check.
    kv = max(1, heads // 2)
    while heads % kv != 0:
        kv -= 1
    if attention_type == "reversible":
        # reversible's inner attention runs on heads//2 query heads, so
        # n_kv_head must additionally divide heads//2.
        half = heads // 2
        kv = max(1, half)
        while half % kv != 0:
            kv -= 1
    cfg = GPTConfig(
        sequence_len=max(seq_len, 16),
        vocab_size=256,
        n_layer=1,
        n_head=heads,
        n_kv_head=kv,
        n_embd=heads * head_dim,
        attention_type=attention_type,
        parameterization=parameterization,
        ffn_type=ffn_type,
    )
    model = GPT(cfg).to(device)
    model.eval()
    idx = torch.randint(0, cfg.vocab_size, (batch_size, min(seq_len, cfg.sequence_len)), device=device)
    acts: dict[str, Any] = {}
    # hook the first block's output (post-residual activation)
    blocks = model.transformer["h"]
    if not isinstance(blocks, torch.nn.ModuleList):
        raise TypeError(f"GPT transformer 'h' must be a ModuleList, got {type(blocks).__name__}")
    if len(blocks) == 0:
        raise RuntimeError("GPT transformer must contain at least one block")
    handle = blocks[0].register_forward_hook(lambda mod, inp, out: acts.__setitem__("y", out.detach()))
    try:
        with torch.no_grad():
            model(idx)
    finally:
        handle.remove()
    y = acts.get("y")
    if y is None:
        raise RuntimeError(f"no block output captured for {attention_type!r}")
    return float(y.float().pow(2).mean().sqrt())


def coordinate_check(
    attention_type: str,
    widths: list[int],
    *,
    seed: int = 0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run measure_activation_scale across a width ladder and fit the log-log
    slope of activation RMS vs width. Returns the per-width scales, the fitted
    slope, and its R^2. |slope| ~ 0 == flat == correctly parameterized."""
    import numpy as np

    scales = {w: measure_activation_scale(attention_type, w, seed=seed, **kwargs) for w in sorted(widths)}
    ws = np.array(sorted(scales), dtype=float)
    ys = np.array([scales[int(w)] for w in ws], dtype=float)
    # guard against nonpositive (shouldn't happen post-init, but be safe)
    mask = ys > 0
    if mask.sum() < 2:
        slope, r2 = float("nan"), float("nan")
    else:
        lx, ly = np.log(ws[mask]), np.log(ys[mask])
        slope, intercept = np.polyfit(lx, ly, 1)
        pred = slope * lx + intercept
        ss_res = float(((ly - pred) ** 2).sum())
        ss_tot = float(((ly - ly.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "attention_type": attention_type,
        "widths": [int(w) for w in ws],
        "activation_rms": {int(w): scales[int(w)] for w in ws},
        "loglog_slope": float(slope),
        "r_squared": float(r2),
        "concentration_class": scaling_rule(attention_type).concentration_class,
    }


# ---------------------------------------------------------------------------
# Artifact emission (bead bp08 part 2): coordinate-check curves as versioned
# bench artifacts, one JSON per (mechanism, parameterization arm). The schema
# name follows the mgr.bench.* convention; engine ingestion (so
# hyp-coordcheck-clt-flat / hyp-tropical-evt-miscoupling become adjudicable)
# lands in the cli collector.
# ---------------------------------------------------------------------------

COORD_CURVES_SCHEMA = "mgr.bench.coord_curves.v1"


def coord_check_artifact(
    result: dict[str, Any],
    *,
    parameterization: str = "current",
    seed: int = 0,
    device: str = "cpu",
    run_id: str | None = None,
    git: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap a coordinate_check() result as a versioned bench artifact payload.

    Shape matches the house mgr.bench.* convention the G2 verdict engine
    already reads (see cli.py bench dispatch + _adj_artifact_matches_arm):
    top-level `mechanism` for arm membership, `meta` carrying seed/device/
    generated_at for observation dedupe, and `results` holding the observables
    so `bench:results.loglog_slope` metric paths resolve. The arm is recorded
    in results.parameterization so registry variant selectors can separate
    current-vs-nsa evidence sharing one metric path (rgyl).

    provenance defaults to a real build_provenance() block — a dirty tree
    yields tainted=true and the engine REFUSES the artifact as evidence (by
    design). Tests may pass an explicit clean block to exercise the engine
    path deterministically.
    """
    import time as _time

    if provenance is None:
        from nanochat.report import build_provenance

        provenance = build_provenance(
            {
                "coord_curves": {
                    "mechanism": result.get("attention_type"),
                    "parameterization": parameterization,
                    "widths": result.get("widths", []),
                    "seed": seed,
                }
            }
        )

    meta: dict[str, Any] = {
        "generated_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "seed": seed,
        "device": device,
    }
    if run_id:
        meta["run_id"] = run_id
    if git:
        meta["git"] = git
    slope = result.get("loglog_slope")
    return {
        "schema_version": COORD_CURVES_SCHEMA,
        "kind": "coord-curve",
        "bead": "model_guided_research-bp08",
        "mechanism": result.get("attention_type"),
        "meta": meta,
        "provenance": provenance,
        "results": {
            "widths": result.get("widths", []),
            "activation_rms": result.get("activation_rms", {}),
            "loglog_slope": slope,
            # flatness is a two-sided claim (|slope| <= 0.05): the registry's
            # comparators are one-sided, so the magnitude is its own observable
            "abs_loglog_slope": abs(float(slope)) if slope is not None else None,
            "r_squared": result.get("r_squared"),
            "concentration_class": result.get("concentration_class"),
            "parameterization": parameterization,
        },
    }


def write_coord_check_artifact(
    result: dict[str, Any],
    out_path: str | Path,
    *,
    parameterization: str = "current",
    seed: int = 0,
    device: str = "cpu",
    run_id: str | None = None,
    git: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Write a coordinate-check artifact to disk (JSON) and return its path.

    If ``out_path`` is a directory (no suffix), writes the collector's expected
    ``summary.json`` inside it; otherwise writes exactly that file.
    """
    import json

    path = Path(out_path)
    if not path.suffix:
        path = path / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = coord_check_artifact(
        result,
        parameterization=parameterization,
        seed=seed,
        device=device,
        run_id=run_id,
        git=git,
        provenance=provenance,
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
