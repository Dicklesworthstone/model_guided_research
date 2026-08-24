"""
Hyperbolic Attention Module (PyTorch) — Lorentz Model
=====================================================

Framework #13 production implementation (bead model_guided_research-mnn.6),
on the A1 AttentionCore scaffold. Geometry validated by
hyperbolic_geometry_and_negative_curvature_attention.py (mnn.5).

Design:
- Each head's D-vector is PROJECTED onto that head's hyperboloid
  H^{D-1}_c (Lorentz model, curvature -1/c) after the standard Q/K/V
  projections.
- Scores are Lorentz inner products <q_pt, k_pt>_L / sqrt(D). REDUCTION
  THEOREM (certified): with RMS-normalized inputs the per-key norms are
  equal across keys, so softmax(<q,k>_L/sqrt(D)) -> softmax(q.k/sqrt(D))
  exactly as c -> 0 - i.e. standard dot-product attention.
- Values aggregate in the origin chart: log_o(v_j) tangents weight-averaged;
  read out re-padded with a ZERO time slot (constant channel, carries no
  information, avoids the 1/sqrt(c) blow-up of the time coordinate).
- Numerics discipline: fp32 islands for all Lorentz math under any autocast;
  arccosh argument clamped >= 1 + 1e-6; constraint projection after every
  lift.
- Per-head learnable curvature c = softplus(raw) + 0.01, init ~= 1.0.
  ``last_mean_curvature`` feeds D2 telemetry: c collapsing toward 0 means
  the head learned to be Euclidean - an honest, interpretable outcome.
"""

from __future__ import annotations

import math

import torch

from nanochat.model_utils import AttentionCore, causal_attn_mask


def minkowski_pairwise(q_pts: torch.Tensor, k_pts: torch.Tensor) -> torch.Tensor:
    """<q_i, k_j>_L for (..., A, N), (..., B, N) -> (..., A, B)."""
    qt, qs = q_pts[..., :1], q_pts[..., 1:]
    kt, ks = k_pts[..., :1], k_pts[..., 1:]

    return -(qt @ kt.transpose(-1, -2)) + qs @ ks.transpose(-1, -2)


def lorentz_project(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Project raw (..., N) vectors onto <x,x>_L = -1/c (c: (..., 1))."""
    x_s = x[..., 1:]

    return torch.cat([torch.sqrt(1.0 / c + (x_s**2).sum(-1, keepdim=True)), x_s], -1)


def lorentz_exp_o(v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """exp_o(v) for origin-tangent (space-only) vectors."""
    lam = torch.sqrt(c) * v.norm(dim=-1, keepdim=True)
    x_t = torch.cosh(lam) / torch.sqrt(c)
    x_s = torch.where(
        lam < 1e-9,
        v / math.sqrt(c),
        torch.sinh(lam) / lam.clamp_min(1e-9) * v,
    )

    return torch.cat([x_t, x_s], -1)


def lorentz_log_o(y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """log_o(y): hyperboloid points -> origin tangents."""
    alpha = torch.acosh((math.sqrt(c) * y[..., :1]).clamp_min(1.0 + 1e-6))
    norm_s = y[..., 1:].norm(dim=-1, keepdim=True)

    return (alpha / (math.sqrt(c) * norm_s.clamp_min(1e-12))) * y[..., 1:]


class HyperbolicCausalSelfAttention(AttentionCore):
    """Negative-curvature attention: Lorentz scoring, chart-space values."""

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        if self.head_dim < 3:
            raise ValueError("head_dim must be >= 3 for Hyperbolic attention")
        # softplus(0.5413) + 0.01 ~= 1.0: init at unit curvature.
        self.raw_curvature = torch.nn.Parameter(torch.full((self.n_head, 1), 0.5413))
        self.last_mean_curvature = float("nan")

    def _curv(self) -> torch.Tensor:
        # fp32 island: strictly positive per-head curvature, (H, 1).
        return torch.nn.functional.softplus(self.raw_curvature.float()) + 0.01

    def attend(self, q, k, v, *, kv_cache, pos0):
        src_dtype = q.dtype
        B, H, Tq, D = q.shape
        Tk = k.shape[3]
        c_all = self._curv()

        # fp32 islands under any autocast; head-major layout for broadcasting.
        q_f = q.float().permute(0, 2, 1, 3)  # (B, H, Tq, D)
        k_f = k.float().permute(0, 2, 1, 3)  # (B, H, Tk, D)
        v_f = v.float().permute(0, 2, 1, 3)

        outs = []
        curv_acc = 0.0
        need_mask = kv_cache is None or Tq > 1
        if need_mask:
            mask = causal_attn_mask(Tq, Tk, device=q.device)

        for h in range(H):
            c_h = c_all[h : h + 1]
            q_pt = lorentz_project(q_f[:, h], c_h)
            k_pt = lorentz_project(k_f[:, h], c_h)
            v_pt = lorentz_project(v_f[:, h], c_h)

            scores = minkowski_pairwise(q_pt, k_pt) * (1.0 / math.sqrt(D))
            if need_mask:
                scores = scores.masked_fill(~mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)

            tangents = lorentz_log_o(v_pt, c_h)  # (B, Tk, D-1)
            chart = weights @ tangents  # (B, Tq, D-1)
            outs.append(
                torch.nn.functional.pad(chart, (1, 0))  # zero time slot
            )
            curv_acc += float(c_h.item())

        self.last_mean_curvature = curv_acc / H

        y = torch.cat(outs, dim=-1)  # (B, Tq, H*D)
        y = y.reshape(B, Tq, H, D).transpose(1, 2)

        return y.to(src_dtype)
