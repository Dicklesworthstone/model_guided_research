"""
Tropical Attention Module (PyTorch)
Implements attention using Max-Plus algebra for similarity.
"""

from __future__ import annotations

import math

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

    def attend(self, q, k, v, *, kv_cache, pos0):
        # Causal masking lives inside tropical_max_plus_attention (-inf is
        # the additive identity of the max-plus semiring), so the default
        # mask/softmax pipeline does not apply here.
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
                self.tropical_route_coverage.copy_(
                    coverage if int(total) > 0 else torch.tensor(float("nan"))
                )
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
        if self.tropical_gauge_fix:
            y = _tropical_center(y, dim=-1)
        return y
