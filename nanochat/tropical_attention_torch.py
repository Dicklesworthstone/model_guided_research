"""
Tropical Attention Module (PyTorch)
Implements attention using Max-Plus algebra for similarity.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from nanochat.model_utils import AttentionCore, causal_attn_mask
from nanochat.parameterization import exact_expected_max


def _tropical_center(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Gauge-fix by subtracting the max along `dim` so the max becomes 0."""
    return x - x.amax(dim=dim, keepdim=True)


def tropical_inner(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Max-plus inner product: out[..., i, j] = max_d (q[..., i, d] + k[..., j, d]).

    This is the score stage of tropical attention, exposed as a pure function.
    It is the torch twin of the JAX demo's tropical matrix product
    (tropical_geometry_and_idempotent_algebra.tmm): tropical_inner(Q, K)
    equals tmm(Q, K^T) entrywise, exactly (max and + are exact float ops).
    Kept separable so the cross-framework parity suite
    (tests/test_cross_framework_parity.py, bead model_guided_research-5ki.6)
    can compare the two implementations directly.
    """
    return torch.max(q.unsqueeze(-2) + k.unsqueeze(-3), dim=-1).values


def tropical_maxplus_layer(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, beta: float | None = None
) -> torch.Tensor:
    """One max-plus affine layer: out_j = max_d (W_jd + x_d) (+ b_j).

    `weight` has shape (out_features, in_features); the broadcast intermediate
    is (..., out_features, in_features) - fine at research scale, a Triton
    tile kernel is the production path (see bead 7b0.7).

    With finite `beta`, max is replaced by the Maslov-smoothed
    (1/beta) * logsumexp(beta * .) - the SAME (+)_beta semiring family as
    dequantization annealing (bead 8gk.1); beta=None is the exact tropical
    endpoint. The LSE-max sandwich guarantees
    max <= lse_beta <= max + log(in_features)/beta (thm-lse-max-sandwich).
    """
    z = x.unsqueeze(-2) + weight  # (..., out, in)
    if beta is None:
        out = z.amax(dim=-1)
    else:
        out = torch.logsumexp(beta * z, dim=-1) / beta
    if bias is not None:
        out = out + bias
    return out


class TropicalMLP(nn.Module):
    """Max-plus FFN: the semiring design axis extended past attention.

    Bead: model_guided_research-8gk.8 (theory: markdown_documentation/
    tropical_geometry_and_idempotent_algebra.md, section "The tropical FFN").

    MODES (config.ffn_type):
      "tropical"          - PURE two-stage max-plus stack. 1-LIPSCHITZ in
          sup-norm by construction (max of unit-slope affines is
          nonexpansive; composition stays nonexpansive) - this is the layer
          that closes the certified chain's MLP hole (bead 8gk.7). The two
          stages COLLAPSE mathematically to one tropical-affine map of
          Barvinok rank <= d_ff (thm-maxplus-ffn-collapse): hidden width is a
          tropical RANK budget, not depth expressivity.
      "tropical-rational" - difference of two pure stacks (param-matched at
          d_ff/2 hidden each): reaches all piecewise-linear maps but is
          2-LIPSCHITZ per layer (difference of two 1-Lipschitz maps); the
          certificate chain must compose the declared constant 2.

    FINITE-BETA SMOOTHING (config.ffn_beta): replaces max by the (+)_beta
    semiring addition, enabling NETWORK-WIDE dequantization annealing with
    the attention path (8gk.1). beta=None is the exact tropical endpoint.

    EVT-AWARE INIT (lab.1 owns the full derivation; the rule used here):
    max over m iid terms of scale s concentrates at ~s*sqrt(2 ln m) (Gumbel
    location), NOT at 0 - so an additive max-plus stage drifts its output
    upward at init. The correction is baked into BIAS init (constant shift:
    Lipschitz constants untouched, no runtime renormalization), and applies
    to STAGE 1 ONLY: its input is the unit-RMS residual stream (norm(x)
    precedes the FFN in Block), so the asymptotic location -sqrt(2 ln d) is
    the right offset there. Stage 2's input is the POST-MAX distribution -
    centered and concentrated (Gumbel fluctuations ~1/sqrt(2 ln d), not unit
    scale) - so its drift is second-order and its bias inits to ZERO; using
    the unit-scale correction there overshoots by ~sqrt(2 ln d_ff) (caught
    by the init-centering test). The exact finite-n constants belong to
    lab.1's width-scaling table. Weights init N(0, 0.02^2) per repo
    convention.

    MARGINS: with config.tropical_record_margins (the existing flag, reused
    by design - one switch for all tropical diagnostics), registers
    ffn_gamma_min / ffn_gamma_mean buffers with the runner-up margin of the
    OUTPUT stage maxes (the certificate-bearing quantity, top1 - top2).
    Hard-max mode only (margins of a smoothed max are not route margins).
    """

    # buffer annotations (registered conditionally in __init__): these give
    # mypy the Tensor type that nn.Module.__getattr__'s union would otherwise hide
    ffn_gamma_min: torch.Tensor
    ffn_gamma_mean: torch.Tensor

    def __init__(self, config) -> None:
        super().__init__()
        d = int(config.n_embd)
        d_ff = 4 * d
        self.mode = str(getattr(config, "ffn_type", "tropical"))
        beta = getattr(config, "ffn_beta", None)
        self.beta: float | None = None if beta is None else float(beta)
        if self.beta is not None and not (self.beta > 0):
            raise ValueError(f"ffn_beta must be None or > 0, got {self.beta}")
        self.record_margins = bool(getattr(config, "tropical_record_margins", False))
        # Width-scaling parameterization arm (lab.1/bp08): "current" keeps the
        # asymptotic bias init; "nsa" uses the exact finite-N E[max] constants.
        self.parameterization = str(getattr(config, "parameterization", "current"))

        def w(out_f: int, in_f: int) -> nn.Parameter:
            return nn.Parameter(torch.randn(out_f, in_f) * 0.02)

        def evt_bias(out_f: int, in_f: int) -> nn.Parameter:
            # Gumbel location correction: center the max over in_f terms.
            # "current": second-order asymptote sqrt(2 ln m) (the original
            # lab.1-prior rule). "nsa": the EXACT order-statistic mean
            # E[max] from the width-scaling table (validated within 0.2% of
            # Monte Carlo; the asymptote is ~9-33% off at small m - the note's
            # single most important implementation footgun).
            if self.parameterization == "nsa":
                center = -exact_expected_max(max(in_f, 1))
            else:
                center = -math.sqrt(2.0 * math.log(max(in_f, 1)))
            return nn.Parameter(torch.full((out_f,), float(center)))

        def zero_bias(out_f: int) -> nn.Parameter:
            # stage-2 bias: post-max inputs are concentrated, drift is
            # second-order - start at zero (see EVT-AWARE INIT above)
            return nn.Parameter(torch.zeros(out_f))

        if self.mode == "tropical":
            self.w1, self.b1 = w(d_ff, d), evt_bias(d_ff, d)
            self.w2, self.b2 = w(d, d_ff), zero_bias(d)
        elif self.mode == "tropical-rational":
            h = d_ff // 2
            self.w1a, self.b1a = w(h, d), evt_bias(h, d)
            self.w2a, self.b2a = w(d, h), zero_bias(d)
            self.w1b, self.b1b = w(h, d), evt_bias(h, d)
            self.w2b, self.b2b = w(d, h), zero_bias(d)
        else:
            raise ValueError(f"TropicalMLP got unknown ffn_type {self.mode!r} (tropical | tropical-rational)")

        if self.record_margins:
            self.register_buffer("ffn_gamma_min", torch.zeros(()), persistent=False)
            self.register_buffer("ffn_gamma_mean", torch.zeros(()), persistent=False)

    def _stage_pair(self, x: torch.Tensor, w1, b1, w2, b2, *, record: bool) -> torch.Tensor:
        h = tropical_maxplus_layer(x, w1, b1, beta=self.beta)
        if record and self.beta is None:
            z = h.unsqueeze(-2) + w2  # (..., out, in): margins of the output stage
            top2 = torch.topk(z, k=2, dim=-1).values
            gamma = top2[..., 0] - top2[..., 1]
            self.ffn_gamma_min.copy_(gamma.detach().min())
            self.ffn_gamma_mean.copy_(gamma.detach().mean())
            out = top2[..., 0]
            return out + b2 if b2 is not None else out
        return tropical_maxplus_layer(h, w2, b2, beta=self.beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        record = self.record_margins and not torch.jit.is_scripting()
        if self.mode == "tropical":
            return self._stage_pair(x, self.w1, self.b1, self.w2, self.b2, record=record)
        pos = self._stage_pair(x, self.w1a, self.b1a, self.w2a, self.b2a, record=record)
        neg = self._stage_pair(x, self.w1b, self.b1b, self.w2b, self.b2b, record=False)
        return pos - neg

    @torch.no_grad()
    def collapsed_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The single tropical-affine map the PURE stack collapses to:
        M[j, d] = max_h (W2[j, h] + b1[h] + W1[h, d]), bias b2 (exact algebra;
        floating-point regrouping costs ulps - thm-maxplus-ffn-collapse).
        Raises for the rational mode (the difference does not collapse)."""
        if self.mode != "tropical":
            raise ValueError("collapsed_weight() is defined for the pure max-plus mode only")
        m = (self.w2.unsqueeze(-1) + self.b1.unsqueeze(0).unsqueeze(-1) + self.w1.unsqueeze(0)).amax(dim=1)
        return m, self.b2.clone()


def tropical_max_plus_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    gauge_fix: bool,
    score_center: bool,
    return_margins: bool,
    beta: float | None = None,
    evt_shift: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Pure max-plus (tropical) attention with optional gauge-fixing and margin
    certificates - and, with finite `beta`, its Maslov dequantization (bead
    8gk.1): every max is replaced by the (+)_beta semiring addition
    x (+)_beta y = (1/beta) log(e^(beta x) + e^(beta y)), and the value
    aggregation is NORMALIZED by semiring division (an LSE subtraction):

        y_d = (1/beta)[LSE_k beta(s_k + v_kd)] - (1/beta)[LSE_k beta s_k]

    beta=None is the EXACT tropical endpoint (this code path is bitwise
    identical to the pre-8gk.1 implementation); beta -> infinity converges to
    it uniformly with rate log(arity)/beta (the LSE-max sandwich,
    thm-lse-max-sandwich). Tropical attention WITH score centering is the
    beta=infinity shadow of softmax normalization: the LSE normalizer above
    degenerates to subtracting max_k s_k, which is exactly what score_center
    does. Theory: markdown_documentation/maslov_dequantization_annealing.md.

    Shapes:
      q: (B, H, Tq, D)
      k: (B, H, Tk, D)
      v: (B, H, Tk, D)

    Returns:
      y: (B, H, Tq, D)
      gamma: optional per-token, per-head runner-up margin (B, H, Tq) (>=0),
        or None if disabled. With finite beta the margins are computed from
        the TROPICAL (hard) scores - the two-family route-stability lemma
        compares smoothed routing against exactly these margins.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("tropical_max_plus_attention expects q/k/v of shape (B, H, T, D)")
    if k.shape != v.shape:
        raise ValueError(
            f"tropical_max_plus_attention expects k and v to have the same shape, got k={k.shape}, v={v.shape}"
        )
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[3] != k.shape[3]:
        raise ValueError(f"tropical_max_plus_attention head shapes mismatch: q={q.shape}, k={k.shape}, v={v.shape}")
    if beta is not None and not (beta > 0):
        raise ValueError(f"beta must be None (exact tropical) or > 0, got {beta}")

    B, H, Tq, D = q.shape
    Tk = k.size(2)

    if gauge_fix:
        q = _tropical_center(q, dim=-1)
        k = _tropical_center(k, dim=-1)
        v = _tropical_center(v, dim=-1)

    mask = causal_attn_mask(Tq, Tk, device=q.device)

    if beta is None:
        # Similarity/logits: score(q,k) = max_d (q_d + k_d)  (max-plus dot product)
        attn_scores = tropical_inner(q, k)  # (B, H, Tq, Tk)
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        if evt_shift is not None:
            # nsa (lab.1/bp08): subtract the exact E[max] of head_dim
            # unit-scale terms at the FIRST max-stage so the score location
            # is Theta(1) by derivation rather than only empirically. The
            # constant commutes with the per-query empirical centering below;
            # it stands alone when gauge_fix is off.
            attn_scores = attn_scores - evt_shift

        # Score-centering is a pure gauge change (per-query constant shift): it preserves argmax structure.
        if score_center:
            attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True)

        # Value aggregation: y_d = max_k (score(q,k) + v_{k,d})  (tropical matmul)
        logits = attn_scores.unsqueeze(-1) + v.unsqueeze(2)  # (B, H, Tq, Tk, D)
        gamma: torch.Tensor | None
        if return_margins:
            if Tk >= 2:
                top2 = torch.topk(logits, k=2, dim=3).values  # (B, H, Tq, 2, D)
                y = top2[..., 0, :]
                margin_d = top2[..., 0, :] - top2[..., 1, :]  # (B, H, Tq, D)
                gamma = margin_d.amin(dim=-1)  # (B, H, Tq)
            else:
                y = torch.max(logits, dim=3).values
                gamma = torch.full((B, H, Tq), float("inf"), device=y.device, dtype=y.dtype)
        else:
            y = torch.max(logits, dim=3).values
            gamma = None

        if gauge_fix:
            y = _tropical_center(y, dim=-1)
        return y, gamma

    # ---- finite beta: the Maslov-smoothed family (8gk.1) ----
    # Score stage: (+)_beta inner product over the head dim (inner arity D);
    # -inf masking is exact in both semirings (e^{beta * -inf} = 0).
    attn_scores = torch.logsumexp(beta * (q.unsqueeze(-2) + k.unsqueeze(-3)), dim=-1) / beta
    attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
    if evt_shift is not None:
        # nsa (lab.1/bp08): exact-E[max] location shift at the unit-scale
        # first max-stage. Mathematically the normalized finite-beta output
        # cancels any global score shift; keeping the scores Theta(1) buys
        # LSE numerical headroom and makes the arm's semantics explicit.
        attn_scores = attn_scores - evt_shift
    # At finite beta the per-query shift cancels EXACTLY in the normalized
    # difference below; applying it anyway buys floating-point headroom and
    # keeps the beta=inf shadow statement aligned.
    if score_center:
        attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True)

    z = beta * attn_scores  # (B, H, Tq, Tk)
    norm = torch.logsumexp(z, dim=-1, keepdim=True)  # semiring mass of the scores
    y = (torch.logsumexp(z.unsqueeze(-1) + beta * v.unsqueeze(2), dim=3) - norm) / beta

    gamma = None
    if return_margins:
        # Certificate margins come from the TROPICAL scores of the same
        # q/k/v (the two-family lemma: smoothed routing is argmax-identical
        # to tropical routing wherever gamma exceeds the smoothing budget).
        hard_scores = tropical_inner(q, k).masked_fill(~mask, float("-inf"))
        if score_center:
            hard_scores = hard_scores - hard_scores.amax(dim=-1, keepdim=True)
        hard_logits = hard_scores.unsqueeze(-1) + v.unsqueeze(2)
        if Tk >= 2:
            top2 = torch.topk(hard_logits, k=2, dim=3).values
            gamma = (top2[..., 0, :] - top2[..., 1, :]).amin(dim=-1)
        else:
            gamma = torch.full((B, H, Tq), float("inf"), device=y.device, dtype=y.dtype)

    if gauge_fix:
        y = _tropical_center(y, dim=-1)
    return y, gamma


def route_stability_threshold(beta: float, head_dim: int, visible: torch.Tensor) -> torch.Tensor:
    """The conservative compositional certificate threshold (8gk.1): a
    tropical route with margin gamma survives Maslov smoothing at level beta
    when gamma > (log D + log m)/beta - log(D)/beta bounds the one-sided
    score-stage inflation (inner arity D = head_dim), log(m)/beta the value
    aggregation sandwich (m = keys visible to the query). `visible` is the
    per-query visible-key count tensor; returns the matching threshold tensor.
    """
    return (math.log(max(head_dim, 1)) + torch.log(visible.clamp_min(1).to(torch.float32))) / beta


def set_semiring_beta(model: nn.Module, beta: float | None) -> int:
    """Set the Maslov smoothing level on every tropical attention module (the
    dequantization-annealing schedule's per-step hook; train.py owns the
    schedule). Returns the number of modules updated."""
    n = 0
    for module in model.modules():
        if isinstance(module, TropicalCausalSelfAttention):
            module.semiring_beta = None if beta is None else float(beta)
            n += 1
    return n


def cyclic_polytope_support_lift(
    n_head: int,
    head_dim: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic cyclic-polytope support weights and concave upper-hull lift (bead oeno / 8gk.3 §4).

    Returns ``(scale, lift)`` each of shape ``(1, n_head, 1, head_dim)``:
      * ``scale`` is unit radial equalization ``1.0`` across all ``head_dim``
        coordinates so that after weight-row norm equalization every vertex lies
        on the exact sphere of radius ``mean_norm`` and is strictly exposed on
        the upper hull of the Newton polytope.
      * ``lift`` is a zero-mean concave moment-curve upper-hull tie-breaker
        ``-c * (t_{h,e}^2 - mean(t_{h,e}^2))`` staggered across heads ``h``.
    """
    h_idx = torch.arange(max(n_head, 1), device=device, dtype=torch.float32).unsqueeze(1)
    e_idx = torch.arange(max(head_dim, 1), device=device, dtype=torch.float32).unsqueeze(0)
    d_f = float(max(head_dim, 1))
    h_f = float(max(n_head, 1))
    scale = torch.ones((1, n_head, 1, head_dim), device=device, dtype=dtype)

    # Moment-curve nodes in [-0.5, 0.5] with cyclic head offset
    u = ((e_idx + h_idx / h_f) % d_f + 0.5) / d_f - 0.5
    quad = -(u.pow(2) - u.pow(2).mean(dim=-1, keepdim=True))
    lift = (1e-4 * quad).view(1, n_head, 1, head_dim).to(dtype=dtype)
    return scale, lift


def _cyclic_moment_orthogonal_basis(
    n_head: int,
    head_dim: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Construct head-staggered orthogonal matrices ``C_h in O(head_dim)`` from
    the cyclic moment-curve Vandermonde/Chebyshev basis (shape ``(n_head, head_dim, head_dim)``)."""
    D = max(head_dim, 1)
    H = max(n_head, 1)
    mats: list[torch.Tensor] = []
    e_idx = torch.arange(D, device=device, dtype=torch.float32)
    deg = torch.arange(D, device=device, dtype=torch.float32)
    for h in range(H):
        # Moment curve nodes in (-1, 1) staggered by head index h
        theta = math.pi * ((e_idx + 0.5 + (h / float(H))) % float(D)) / float(D)
        # Chebyshev-Vandermonde matrix V_{d, k} = cos(k * theta_d)
        v_mat = torch.cos(theta.unsqueeze(1) * deg.unsqueeze(0))
        q_orth, r_mat = torch.linalg.qr(v_mat)
        diag_sign = torch.sgn(torch.diagonal(r_mat))
        diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
        q_orth = q_orth * diag_sign.unsqueeze(0)
        mats.append(q_orth)
    return torch.stack(mats, dim=0).to(dtype=dtype)


@torch.no_grad()
def extract_tropical_routes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    gauge_fix: bool = True,
    score_center: bool = True,
    evt_shift: float | None = None,
    verify_bruteforce: bool = True,
) -> dict[str, Any]:
    """Extract exact discrete argmax routes for tropical max-plus attention (beads 8gk.9 & oeno).

    For query position ``t`` and output coordinate ``d'``,
        ``y_{t,d'} = max_{j <= t} ( max_e (q_{t,e} + k_{j,e}) + v_{j,d'} )
                   = max_{j <= t, 0 <= e < D} (q_{t,e} + k_{j,e} + v_{j,d'})``.

    Returns a dictionary containing:
      * ``score_key_route``: ``(B, H, Tq)`` argmax key index ``j*`` of the score stage ``max_e (q_{t,e} + k_{j,e})``
      * ``score_dim_route``: ``(B, H, Tq)`` argmax feature index ``e*`` at the winning score key ``j*``
      * ``score_margin``: ``(B, H, Tq)`` runner-up margin of the score stage over visible keys
      * ``key_route``: ``(B, H, Tq, D)`` per-output-coordinate winning key ``j*(t, d')``
      * ``dim_route``: ``(B, H, Tq, D)`` per-output-coordinate winning feature ``e*(t, d')``
      * ``compound_route``: ``(B, H, Tq, D)`` flattened route index ``j*(t, d') * D + e*(t, d')``
      * ``gamma``: ``(B, H, Tq)`` per-query certificate margin (min over output coordinates of top1 - top2)
      * ``bruteforce_verified``: ``bool`` confirming two-stage argmax equals single-stage flattened brute-force argmax
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("extract_tropical_routes expects q/k/v of shape (B, H, T, D)")
    B, H, Tq, D = q.shape
    Tk = k.size(2)

    q_work = _tropical_center(q, dim=-1) if gauge_fix else q
    k_work = _tropical_center(k, dim=-1) if gauge_fix else k
    v_work = _tropical_center(v, dim=-1) if gauge_fix else v

    mask = causal_attn_mask(Tq, Tk, device=q.device)

    # Inner stage: pairwise (q_{t,e} + k_{j,e}) of shape (B, H, Tq, Tk, D)
    pairwise_qk = q_work.unsqueeze(-2) + k_work.unsqueeze(-3)
    attn_scores, inner_dim_idx = torch.max(pairwise_qk, dim=-1)  # (B, H, Tq, Tk)
    attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
    if evt_shift is not None:
        attn_scores = attn_scores - evt_shift
    score_shift: torch.Tensor | None = None
    if score_center:
        score_shift = attn_scores.amax(dim=-1, keepdim=True)
        attn_scores = attn_scores - score_shift

    # Score-stage route and score-stage margin over keys
    score_key_route = torch.argmax(attn_scores, dim=-1)  # (B, H, Tq)
    score_dim_route = torch.gather(inner_dim_idx, dim=-1, index=score_key_route.unsqueeze(-1)).squeeze(-1)
    if Tk >= 2:
        score_top2 = torch.topk(attn_scores, k=2, dim=-1).values
        score_margin = score_top2[..., 0] - score_top2[..., 1]
        score_margin = torch.where(torch.isfinite(score_margin), score_margin, torch.zeros_like(score_margin))
    else:
        score_margin = torch.full((B, H, Tq), float("inf"), device=q.device, dtype=q.dtype)

    # Outer stage: logits (B, H, Tq, Tk, D)
    logits = attn_scores.unsqueeze(-1) + v_work.unsqueeze(2)
    y_two_stage, key_route = torch.max(logits, dim=3)  # (B, H, Tq, D)
    # Gather the winning inner feature e* for each output coordinate's winning key j*
    dim_route = torch.gather(inner_dim_idx, dim=3, index=key_route)  # (B, H, Tq, D)
    compound_route = key_route * D + dim_route  # (B, H, Tq, D) in [0, Tk * D)

    if Tk >= 2:
        top2 = torch.topk(logits, k=2, dim=3).values
        margin_d = top2[..., 0, :] - top2[..., 1, :]
        margin_d = torch.where(torch.isfinite(margin_d), margin_d, torch.zeros_like(margin_d))
        gamma = margin_d.amin(dim=-1)
    else:
        gamma = torch.full((B, H, Tq), float("inf"), device=q.device, dtype=q.dtype)

    bruteforce_ok = True
    if verify_bruteforce:
        # Single-stage flattened brute-force max over all (j, e) in {0..Tk-1} x {0..D-1}
        # Tensor of shape (B, H, Tq, Tk, D_e, D_out)
        bf_scores = pairwise_qk.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        if evt_shift is not None:
            bf_scores = bf_scores - evt_shift
        if score_shift is not None:
            bf_scores = bf_scores - score_shift.unsqueeze(-1)
        bf_total = bf_scores.unsqueeze(-1) + v_work.unsqueeze(2).unsqueeze(-2)  # (B, H, Tq, Tk, D, D)
        bf_flat = bf_total.reshape(B, H, Tq, Tk * D, D)
        bf_max_vals, bf_compound = torch.max(bf_flat, dim=3)  # (B, H, Tq, D)
        val_match = bool(torch.allclose(y_two_stage, bf_max_vals, atol=1e-5, rtol=1e-5))
        # Where the winning route has a strict positive margin (no exact float tie),
        # the compound route index must match bitwise; even on ties, the achieved max value must match.
        gathered_bf_val = torch.gather(bf_flat, dim=3, index=compound_route.unsqueeze(3)).squeeze(3)
        route_val_match = bool(torch.allclose(gathered_bf_val, bf_max_vals, atol=1e-5, rtol=1e-5))
        if Tk * D >= 2:
            bf_top2 = torch.topk(bf_flat, k=2, dim=3).values
            strict_mask = (bf_top2[..., 0, :] - bf_top2[..., 1, :]) > 1e-5
            idx_match = bool(torch.equal(compound_route[strict_mask], bf_compound[strict_mask]))
        else:
            idx_match = True
        bruteforce_ok = val_match and route_val_match and idx_match

    return {
        "score_key_route": score_key_route,
        "score_dim_route": score_dim_route,
        "score_margin": score_margin,
        "key_route": key_route,
        "dim_route": dim_route,
        "compound_route": compound_route,
        "gamma": gamma,
        "bruteforce_verified": bruteforce_ok,
    }


class TropicalCausalSelfAttention(AttentionCore):
    tropical_gamma_head_mean: torch.Tensor
    tropical_gamma_head_min: torch.Tensor
    tropical_gamma_min: torch.Tensor
    tropical_route_coverage: torch.Tensor

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.tropical_gauge_fix = bool(getattr(config, "tropical_gauge_fix", True))
        self.tropical_score_center = bool(getattr(config, "tropical_score_center", True))
        self.tropical_record_margins = bool(getattr(config, "tropical_record_margins", False))
        self.tropical_structured_support = bool(getattr(config, "tropical_structured_support", False))
        # Maslov smoothing level (8gk.1): None = exact tropical endpoint;
        # finite values are set per-step by the dequantization-annealing
        # schedule via set_semiring_beta (a plain attribute, never state)
        beta = getattr(config, "semiring_beta", None)
        self.semiring_beta: float | None = None if beta is None else float(beta)
        if self.semiring_beta is not None and not (self.semiring_beta > 0):
            raise ValueError(f"semiring_beta must be None or > 0, got {self.semiring_beta}")
        # Width-scaling parameterization (lab.1/bp08): under "nsa" the FIRST
        # max-stage (scores over head_dim unit-scale terms) gets the exact
        # E[max] location shift — theoretical instead of purely empirical
        # centering. Post-max stages (value aggregation over Tk) are
        # second-order and get no shift, per the theory note.
        self.parameterization = str(getattr(config, "parameterization", "current"))
        self.register_buffer(
            "tropical_gamma_head_mean",
            torch.full((self.n_head,), float("nan"), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "tropical_gamma_head_min",
            torch.full((self.n_head,), float("nan"), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "tropical_gamma_min",
            torch.tensor(float("nan"), dtype=torch.float32),
            persistent=False,
        )
        # fraction of (token, head) routes whose tropical margin exceeds the
        # route-stability threshold at the CURRENT beta (nan when beta=None
        # or margins are off) - the certificate-coverage telemetry (8gk.1)
        self.register_buffer(
            "tropical_route_coverage",
            torch.tensor(float("nan"), dtype=torch.float32),
            persistent=False,
        )

    def _apply_structured_support(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply weight-row spherical equalization + cyclic-polytope monomial support (bead oeno).

        Because row-norm equalization, head-staggered cyclic coordinate pairing
        ``e -> (e + h) mod D``, and the concave moment-curve lift depend only on
        the projection weights ``(c_q.weight, c_k.weight)`` and indices ``(h, e)``
        — never nonlinearly on the input ``x`` — each route ``(j, e)`` remains an
        exact affine function of ``x``. Equalizing row norms places all ``D``
        facets of each head onto the sphere (eliminating shadowed vertices), while
        cyclic head-staggered pairing puts the ``H`` head arrangements into
        transversal general position.
        """
        cq_w = self.c_q.weight
        ck_w = self.c_k.weight
        assert isinstance(cq_w, torch.Tensor) and isinstance(ck_w, torch.Tensor)

        q_w = cq_w.view(self.n_head, self.head_dim, -1)
        q_norms = q_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        q_eq = (q_norms.mean(dim=1, keepdim=True) / q_norms).view(
            1, self.n_head, 1, self.head_dim
        ).to(dtype=q.dtype)
        q_unit = q * q_eq

        k_w = ck_w.view(self.n_kv_head, self.head_dim, -1)
        k_norms = k_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        k_eq = (k_norms.mean(dim=1, keepdim=True) / k_norms).view(
            1, self.n_kv_head, 1, self.head_dim
        ).to(dtype=k.dtype)
        if k.size(1) != self.n_kv_head:
            rep = k.size(1) // self.n_kv_head
            k_eq = k_eq.repeat_interleave(rep, dim=1)
        k_unit = k * k_eq

        # Head-staggered cyclic coordinate pairing on k: head h pairs q[..., e] with k[..., (e + h) % D]
        H_k = int(k_unit.size(1))
        D = int(self.head_dim)
        h_idx = torch.arange(H_k, device=k.device).unsqueeze(1)
        e_idx = torch.arange(D, device=k.device).unsqueeze(0)
        perm_idx = ((e_idx + h_idx) % D).view(1, H_k, 1, D).expand_as(k_unit)
        k_paired = torch.gather(k_unit, dim=-1, index=perm_idx)

        scale_q, lift_q = cyclic_polytope_support_lift(
            q.size(1), self.head_dim, device=q.device, dtype=q.dtype
        )
        scale_k, lift_k = cyclic_polytope_support_lift(
            k.size(1), self.head_dim, device=k.device, dtype=k.dtype
        )
        q_out = q_unit * scale_q + 0.5 * lift_q
        k_out = k_paired * scale_k + 0.5 * lift_k
        return q_out, k_out

    def attend(self, q, k, v, *, kv_cache, pos0):
        # Causal masking lives inside tropical_max_plus_attention (-inf is
        # the additive identity of the max-plus semiring), so the default
        # mask/softmax pipeline does not apply here.
        if self.tropical_structured_support:
            q, k = self._apply_structured_support(q, k)
        # nsa arm (lab.1/bp08): the exact E[max] over the head_dim unit-scale
        # score terms - the theoretical first-stage location shift.
        evt_shift = exact_expected_max(q.size(-1)) if self.parameterization == "nsa" else None
        y, gamma = tropical_max_plus_attention(
            q,
            k,
            v,
            gauge_fix=self.tropical_gauge_fix,
            score_center=self.tropical_score_center,
            return_margins=self.tropical_record_margins,
            beta=self.semiring_beta,
            evt_shift=evt_shift,
        )
        if gamma is not None and self.semiring_beta is not None:
            with torch.no_grad():
                # visible keys per query under the causal mask: rows pos0+1..
                Tq, Tk = q.size(2), k.size(2)
                visible = torch.arange(Tk - Tq + 1, Tk + 1, device=gamma.device)  # (Tq,)
                threshold = route_stability_threshold(self.semiring_beta, q.size(-1), visible)
                gamma_f = gamma.to(torch.float32)
                finite = torch.isfinite(gamma_f)
                covered = (gamma_f > threshold.view(1, 1, -1)) & finite
                total = finite.sum()
                coverage = covered.sum().to(torch.float32) / total.clamp_min(1).to(torch.float32)
                self.tropical_route_coverage.copy_(coverage if int(total) > 0 else torch.tensor(float("nan")))
        else:
            # exact endpoint or margins off: no live reading - never leave a
            # stale coverage value behind for the telemetry collector
            with torch.no_grad():
                self.tropical_route_coverage.fill_(float("nan"))
        if gamma is not None:
            gamma_f = gamma.to(dtype=torch.float32)
            with torch.no_grad():
                finite = torch.isfinite(gamma_f)
                summed = gamma_f.masked_fill(~finite, 0.0).sum(dim=(0, 2))
                count = finite.sum(dim=(0, 2)).to(dtype=summed.dtype)
                mean = summed / count.clamp_min(1.0)
                mean = mean.masked_fill(count == 0, float("nan"))
                self.tropical_gamma_head_mean.copy_(mean)
                self.tropical_gamma_head_min.copy_(gamma_f.amin(dim=(0, 2)))
                self.tropical_gamma_min.copy_(gamma_f.amin())
        return y

    def finalize(self, y: torch.Tensor) -> torch.Tensor:
        # Output gauge-fix: re-center so the max coordinate sits at 0 (a pure
        # tropical gauge choice; argmax structure is preserved).
        # Under "nsa" (1xov), per-head tropical centering already acted inside
        # tropical_max_plus_attention (at constant head_dim before c_proj);
        # applying _tropical_center a second time AFTER the Euclidean c_proj
        # subtracts max_{1<=c<=n_embd} (c_proj(y))_c from a zero-mean Gaussian
        # residual update, injecting an uncompensated -sigma * E[max_N Z]
        # Gumbel drift into the Euclidean residual stream.
        if self.tropical_gauge_fix and self.parameterization != "nsa":
            y = _tropical_center(y, dim=-1)
        return y


@torch.no_grad()
def count_tropical_realized_regions(
    attn: TropicalCausalSelfAttention,
    x_samples: torch.Tensor,
    cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    *,
    output_coord: int = 0,
    head_idx: int = 0,
    relu_mlp: nn.Module | None = None,
) -> dict[str, Any]:
    """Count realized Newton-polytope linear regions of ``attn`` over ``x_samples`` (bead oeno / 8gk.3).

    Evaluates the last query position ``i = T - 1`` and output coordinate ``output_coord``:
      * Proposition 1 (per-output single-head bound): ``realized_single_head <= T * d_head``
      * Proposition 2 (multi-head mixed-volume bound): ``realized_multi_head <= (T * d_head) ** H``
      * Proposition 0 (fixed-K,V query-space control): ``realized_q_space_fixed_kv <= d_head``
      * Optional ReLU MLP comparison: counts distinct hidden-layer sign patterns on the last token
        and reports ``regions_per_param`` for both modules.
    """
    if x_samples.ndim != 3:
        raise ValueError(f"x_samples must have shape (N, T, n_embd), got {x_samples.shape}")
    N, T, _ = x_samples.shape
    H = int(attn.n_head)
    D = int(attn.head_dim)
    prop1_bound = int(T * D)
    prop2_bound = int((T * D) ** H)

    q, k, v = attn.project_qkv(
        x_samples, cos_sin=cos_sin, kv_cache=None, apply_qk_norm=(cos_sin is not None)
    )
    if attn.tropical_structured_support:
        q, k = attn._apply_structured_support(q, k)
    evt_shift = exact_expected_max(D) if attn.parameterization == "nsa" else None
    routes = extract_tropical_routes(
        q,
        k,
        v,
        gauge_fix=attn.tropical_gauge_fix,
        score_center=attn.tropical_score_center,
        evt_shift=evt_shift,
        verify_bruteforce=True,
    )
    # Compound routes at the final query position T - 1 and output_coord: shape (N, H)
    last_compound = routes["compound_route"][:, :, T - 1, output_coord]  # (N, H)
    single_head_routes = last_compound[:, head_idx].detach().cpu().tolist()
    realized_single = len(set(int(r) for r in single_head_routes))

    per_head_realized = [
        len(set(int(r) for r in last_compound[:, h].detach().cpu().tolist())) for h in range(H)
    ]
    joint_tuples = [tuple(int(x) for x in row) for row in last_compound.detach().cpu().tolist()]
    realized_multi = len(set(joint_tuples))

    # Cumulative multi-head curve H = 1..n_head
    multi_head_curve: dict[str, int] = {}
    for h_sub in range(1, H + 1):
        sub_tuples = [t[:h_sub] for t in joint_tuples]
        multi_head_curve[str(h_sub)] = len(set(sub_tuples))

    # Proposition 0 control: hold K, V fixed at sample 0 and vary Q across all N samples
    k_fixed = k[:1].expand_as(k)
    v_fixed = v[:1].expand_as(v)
    routes_q_only = extract_tropical_routes(
        q,
        k_fixed,
        v_fixed,
        gauge_fix=attn.tropical_gauge_fix,
        score_center=attn.tropical_score_center,
        evt_shift=evt_shift,
        verify_bruteforce=False,
    )
    q_only_routes = routes_q_only["compound_route"][:, head_idx, T - 1, output_coord].detach().cpu().tolist()
    realized_q_space = len(set(int(r) for r in q_only_routes))

    bound_violations = int(
        (realized_single > prop1_bound)
        or any(r > prop1_bound for r in per_head_realized)
        or (realized_multi > prop2_bound)
        or (realized_q_space > D)
    )

    # Single-head q,k,v parameter count (3 * d_model * d_head)
    d_model = int(x_samples.size(-1))
    single_head_params = max(1, 3 * d_model * D)
    qkv_weights = [
        attn.c_q.weight,
        attn.c_k.weight,
        attn.c_v.weight,
    ]
    total_qkv_params = max(
        1, sum(int(p.numel()) for p in qkv_weights if isinstance(p, torch.Tensor))
    )

    out: dict[str, Any] = {
        "n_samples": N,
        "seq_len": T,
        "n_head": H,
        "head_dim": D,
        "realized_single_head": realized_single,
        "realized_single_head_mean": float(sum(per_head_realized) / max(len(per_head_realized), 1)),
        "per_head_realized": per_head_realized,
        "prop1_bound": prop1_bound,
        "prop1_saturation": float(realized_single / max(prop1_bound, 1)),
        "realized_multi_head": realized_multi,
        "multi_head_curve": multi_head_curve,
        "prop2_bound": prop2_bound,
        "prop2_additive_line": int(H * prop1_bound),
        "realized_q_space_fixed_kv": realized_q_space,
        "prop0_bound": D,
        "bound_violations": bound_violations,
        "bruteforce_verified": bool(routes["bruteforce_verified"]),
        "single_head_params": single_head_params,
        "total_qkv_params": total_qkv_params,
        "regions_per_param_single_head": float(realized_single / single_head_params),
        "regions_per_param_multi_head": float(realized_multi / total_qkv_params),
    }

    if relu_mlp is not None:
        # Feedforward ReLU MLP acts token-wise on x_{T-1}; its sign pattern is independent of context x_{0..T-2}.
        x_last = x_samples[:, T - 1, :]
        hidden = relu_mlp(x_last)
        signs = (hidden > 0).to(torch.int8).detach().cpu().tolist()
        realized_mlp = len(set(tuple(row) for row in signs))
        mlp_params = max(1, sum(p.numel() for p in relu_mlp.parameters()))
        out["relu_mlp_realized_regions"] = realized_mlp
        out["relu_mlp_params"] = mlp_params
        out["regions_per_param_relu_mlp"] = float(realized_mlp / mlp_params)

    return out

