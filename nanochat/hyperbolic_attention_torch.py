"""
Hyperbolic Attention Module (PyTorch) — Lorentz Model
=====================================================

Framework #13 production implementation (bead model_guided_research-mnn.6),
on the A1 AttentionCore scaffold. Geometry validated by
hyperbolic_geometry_and_negative_curvature_attention.py (mnn.5).

Design:
- Each head's D-vector is lifted without dropping a feature coordinate onto
  H^D_c in the Lorentz model (curvature -c).
- Scores use the stable, arcosh-free energy Gromov product from the corrected
  8gk.6 derivation. It converges to the ordinary dot product as c -> 0; raw
  hyperbolic distance does not.
- Zero-initialized per-head radial projections turn normalized Q/K directions
  into learned chart radii. This is necessary because QK norm otherwise makes
  every radius equal and reduces curvature to a softmax-constant correction.
- Values aggregate in the origin tangent chart, which has the original D
  dimensions and reduces to the ordinary weighted mean as c -> 0.
- Lorentz math runs in explicit fp32 islands under autocast. Stable asinh
  formulas avoid the acosh(1 + epsilon) cancellation in the forward path.
- Per-head learnable curvature c = softplus(raw) + 1e-6, initialized at 1.
  Non-persistent buffers expose curvature and radius telemetry to training.
"""

from __future__ import annotations

import math

import torch

from nanochat.model_utils import AttentionCore, causal_attn_mask

CURVATURE_MIN = 1e-6
CURVATURE_INIT = 1.0
HIERARCHY_CURVATURE_THRESHOLD = 1.25
EUCLIDEAN_CURVATURE_THRESHOLD = 0.05


def minkowski_pairwise(q_pts: torch.Tensor, k_pts: torch.Tensor) -> torch.Tensor:
    """<q_i, k_j>_L for (..., A, N), (..., B, N) -> (..., A, B)."""
    qt, qs = q_pts[..., :1], q_pts[..., 1:]
    kt, ks = k_pts[..., :1], k_pts[..., 1:]

    return -(qt @ kt.transpose(-1, -2)) + qs @ ks.transpose(-1, -2)


def lorentz_project(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Lift (..., D) chart vectors to (..., D+1) Lorentz points.

    ``c`` is broadcastable to ``x[..., :1]``. Keeping every input coordinate
    as a spatial coordinate is essential: treating ``x[..., 0]`` as the time
    coordinate silently discards one learned feature and breaks the standard
    attention reduction.
    """
    x_s = x.float()
    c_f = c.float()
    x_t = torch.sqrt(c_f.reciprocal() + x_s.square().sum(-1, keepdim=True))
    return torch.cat([x_t, x_s], dim=-1)


def lorentz_exp_o(v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """exp_o(v) for origin-tangent (space-only) vectors."""
    v_f = v.float()
    sqrt_c = torch.sqrt(c.float())
    lam = sqrt_c * v_f.norm(dim=-1, keepdim=True)
    lam2 = lam.square()
    sinhc = torch.where(
        lam < 1e-4,
        1.0 + lam2 / 6.0 + lam2.square() / 120.0,
        torch.sinh(lam) / lam.clamp_min(1e-12),
    )
    x_t = torch.cosh(lam) / sqrt_c
    x_s = sinhc * v_f
    return torch.cat([x_t, x_s], dim=-1)


def lorentz_log_o(y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """log_o(y): hyperboloid points -> origin tangents, via stable asinh."""
    y_f = y.float()
    sqrt_c = torch.sqrt(c.float())
    y_s = y_f[..., 1:]
    z = sqrt_c * y_s.norm(dim=-1, keepdim=True)
    z2 = z.square()
    asinhc = torch.where(
        z < 1e-4,
        1.0 - z2 / 6.0 + 3.0 * z2.square() / 40.0,
        torch.asinh(z) / z.clamp_min(1e-12),
    )
    return asinhc * y_s


def lorentz_radius(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Distance from the Lorentz origin for points made by ``lorentz_project``."""
    x_s = x.float()[..., 1:]
    sqrt_c = torch.sqrt(c.float())
    return torch.asinh(sqrt_c * x_s.norm(dim=-1, keepdim=True)) / sqrt_c


def energy_gromov_scores(q: torch.Tensor, k: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Return the arcosh-free score ``2 G_c(q, k)``.

    For the projective lift ``x=(sqrt(1/c + ||u||^2), u)``, expanding the
    chordal/radial energy definition and rationalizing ``sqrt(1+c||u||^2)-1``
    gives the cancellation-free identity

        2 G_c = q.k - c ||q||^2 ||k||^2 / ((r_q + 1)(r_k + 1)).

    The second term vanishes as c -> 0, so this has the required standard
    dot-product limit without evaluating acosh or subtracting two 1/c terms.
    """
    q_f = q.float()
    k_f = k.float()
    c_f = c.float()
    q_norm2 = q_f.square().sum(dim=-1, keepdim=True)
    k_norm2 = k_f.square().sum(dim=-1, keepdim=True)
    r_q = torch.sqrt(1.0 + c_f * q_norm2)
    r_k = torch.sqrt(1.0 + c_f * k_norm2)
    radial_correction = c_f * q_norm2 * k_norm2.transpose(-1, -2) / ((r_q + 1.0) * (r_k + 1.0).transpose(-1, -2))
    return q_f @ k_f.transpose(-1, -2) - radial_correction


class HyperbolicCausalSelfAttention(AttentionCore):
    """Negative-curvature attention: Lorentz scoring, chart-space values."""

    hyperbolic_curvature_head: torch.Tensor
    hyperbolic_radius_head_mean: torch.Tensor

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        raw_init = math.log(math.expm1(CURVATURE_INIT - CURVATURE_MIN))
        self.raw_curvature = torch.nn.Parameter(torch.full((self.n_head,), raw_init))
        self.radial_q = torch.nn.Parameter(torch.zeros(self.n_head, self.head_dim))
        self.radial_k = torch.nn.Parameter(torch.zeros(self.n_head, self.head_dim))
        self.register_buffer(
            "hyperbolic_curvature_head",
            torch.full((self.n_head,), float("nan"), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "hyperbolic_radius_head_mean",
            torch.full((self.n_head,), float("nan"), dtype=torch.float32),
            persistent=False,
        )

    def _curv(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_curvature.float()) + CURVATURE_MIN

    def attend(self, q, k, v, *, kv_cache, pos0):
        src_dtype = q.dtype
        _, _, Tq, D = q.shape
        Tk = k.shape[2]
        c_head = self._curv()
        c = c_head.view(1, self.n_head, 1, 1)

        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        radial_scale = c / (1.0 + c)
        q_signal = torch.tanh(torch.einsum("bhtd,hd->bht", q_f, self.radial_q.float()).unsqueeze(-1) / math.sqrt(D))
        k_signal = torch.tanh(torch.einsum("bhtd,hd->bht", k_f, self.radial_k.float()).unsqueeze(-1) / math.sqrt(D))
        q_chart = q_f * torch.exp(radial_scale * q_signal)
        k_chart = k_f * torch.exp(radial_scale * k_signal)
        scores = energy_gromov_scores(q_chart, k_chart, c) * (1.0 / math.sqrt(D))
        need_mask = kv_cache is None or Tq > 1
        if need_mask:
            mask = causal_attn_mask(Tq, Tk, device=q.device)
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)

        v_points = lorentz_project(v_f, c)
        v_tangents = lorentz_log_o(v_points, c)
        y = weights @ v_tangents

        with torch.no_grad():
            q_radius = lorentz_radius(lorentz_project(q_chart, c), c).mean(dim=(0, 2, 3))
            k_radius = lorentz_radius(lorentz_project(k_chart, c), c).mean(dim=(0, 2, 3))
            v_radius = lorentz_radius(v_points, c).mean(dim=(0, 2, 3))
            self.hyperbolic_curvature_head.copy_(c_head.detach())
            self.hyperbolic_radius_head_mean.copy_((q_radius + k_radius + v_radius) / 3.0)

        return y.to(src_dtype)
