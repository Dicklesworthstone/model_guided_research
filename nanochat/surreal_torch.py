"""
Surreal Parameterization & Two-Scale Levi-Civita Valuation Attention (PyTorch)

Supports two tiers selectable via ``GPTConfig.surreal_scaling_axis``:
1. Proxy tier (``surreal_scaling_axis=False``, default for backward golden/ledger parity):
   Standard causal SDPA whose four projections use ``SurrealLayer``
   (``w = exp(s) * normalize(v)``, separating transseries magnitude from direction).
2. Faithful scaling-axis tier (``surreal_scaling_axis=True``, bead jida.15):
   Combines ``SurrealLayer`` projections with a two-scale Levi-Civita / Hahn
   valuation score decomposition:
       S_{ij} = <q_i, k_j> / sqrt(d_h) - beta_omega * |nu(q_i) - nu(k_j)|
   where ``nu(x) = <x, u_h> / sqrt(d_h)`` is the per-head archimedean valuation
   exponent and ``beta_omega >= 0`` weights the non-Archimedean valuation distance.
   - Exact reduction (checklist item 1): at ``beta_omega == 0`` (or under the
     frozen-scale placebo ``surreal_freeze_scale=True``), ``S_{ij}`` reduces
     exactly to standard causal SDPA on the same ``q, k, v``.
   - Lexicographic valuation dominance: as ``beta_omega -> inf``, attention mass
     concentrates strictly on keys in the closest valuation class ``|nu(q_i) - nu(k_j)|``.
   - Observables (checklist item 6): records ``scale_std``, ``scale_histogram``
     (5-point quantile summary of ``weight_s``), and ``valuation_gap`` on
     ``last_surreal_stats``.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.model_utils import AttentionCore, causal_attn_mask, sdpa_causal_attend


class SurrealLayer(nn.Module):
    """
    A Linear layer with "Surreal" weight parameterization.
    Weights are represented as `w = exp(s) * normalize(v)` where `s` is a learnable
    log-scale exponent and `v` is the unit direction.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Direction v
        self.weight_v = nn.Parameter(torch.randn(out_features, in_features))
        # Scale s (log-magnitude)
        self.weight_s = nn.Parameter(torch.zeros(out_features, 1))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        # w = exp(s) * normalize(v)
        w = torch.exp(self.weight_s) * F.normalize(self.weight_v, dim=1)
        return F.linear(input, w, self.bias)


class SurrealCausalSelfAttention(AttentionCore):
    # GQA is handled inside SDPA (or inside scaling-axis attend); no materialized repeat in AttentionCore.forward.
    gqa_via_repeat = False

    def __init__(self, config, layer_idx):
        # Surreal Linear Layers (w = exp(s) * normalize(v)) for all projections;
        # attribute names and state-dict keys match the canonical scaffold.
        super().__init__(config, layer_idx, linear_cls=SurrealLayer)
        self.surreal_scaling_axis = bool(getattr(config, "surreal_scaling_axis", False))
        self.surreal_freeze_scale = bool(getattr(config, "surreal_freeze_scale", False))
        self.last_surreal_stats: dict[str, Any] | None = None

        if self.surreal_scaling_axis:
            self.beta_omega = nn.Parameter(torch.tensor(1.0))
            self.scale_axis_vec = nn.Parameter(
                torch.ones(self.n_head, self.head_dim) * (1.0 / (self.head_dim**0.5))
            )
        else:
            self.beta_omega = None
            self.scale_axis_vec = None

    def _ensure_scaling_axis_params(self, device: torch.device, dtype: torch.dtype) -> None:
        if self.beta_omega is None or self.scale_axis_vec is None:
            self.beta_omega = nn.Parameter(torch.tensor(1.0, device=device, dtype=torch.float32))
            self.scale_axis_vec = nn.Parameter(
                torch.ones(self.n_head, self.head_dim, device=device, dtype=dtype)
                * (1.0 / (self.head_dim**0.5))
            )

    def collect_scale_summary(self) -> dict[str, Any]:
        """Return summary statistics of ``weight_s`` across all ``SurrealLayer``
        projections in this attention module."""
        scales: list[torch.Tensor] = []
        for proj in (self.c_q, self.c_k, self.c_v, self.c_proj):
            ws = getattr(proj, "weight_s", None)
            if isinstance(ws, torch.Tensor):
                scales.append(ws.detach().float().reshape(-1))
        if not scales:
            return {"scale_std": 0.0, "scale_histogram": [0.0, 0.0, 0.0, 0.0, 0.0]}
        all_s = torch.cat(scales, dim=0)
        q_levels = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=all_s.device, dtype=torch.float32)
        quantiles = torch.quantile(all_s, q_levels)
        return {
            "scale_std": float(all_s.std(unbiased=False).item()),
            "scale_histogram": [float(x) for x in quantiles.tolist()],
        }

    def compute_valuations(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-token, per-head archimedean valuation exponents ``nu(x)``
        of shape ``(B, H, T)`` for ``x`` of shape ``(B, H, T, D)``."""
        if self.surreal_freeze_scale:
            return torch.zeros(x.shape[:-1], device=x.device, dtype=torch.float32)
        self._ensure_scaling_axis_params(x.device, x.dtype)
        assert self.scale_axis_vec is not None
        u = self.scale_axis_vec.to(device=x.device, dtype=torch.float32).view(1, self.n_head, 1, self.head_dim)
        return (x.float() * u).sum(dim=-1) * (1.0 / (self.head_dim**0.5))

    def attend(self, q, k, v, *, kv_cache, pos0):
        enable_gqa = self.n_head != self.n_kv_head
        if not self.surreal_scaling_axis:
            out = sdpa_causal_attend(q, k, v, kv_cache=kv_cache, enable_gqa=enable_gqa)
            with torch.no_grad():
                stats = self.collect_scale_summary()
                stats["valuation_gap"] = 0.0
                self.last_surreal_stats = stats
            return out

        self._ensure_scaling_axis_params(q.device, q.dtype)
        assert self.beta_omega is not None

        if enable_gqa:
            repeat_factor = self.n_head // self.n_kv_head
            k_rep = k.repeat_interleave(repeat_factor, dim=1)
            v_rep = v.repeat_interleave(repeat_factor, dim=1)
        else:
            k_rep = k
            v_rep = v

        Tq, Tk = q.size(-2), k_rep.size(-2)
        # Archimedean directional inner-product score (fp32 island for numerical stability)
        s0 = torch.matmul(q.float(), k_rep.float().transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))

        # Non-Archimedean Levi-Civita valuation distance |nu(q_i) - nu(k_j)|
        nu_q = self.compute_valuations(q)  # (B, H, Tq)
        nu_k = self.compute_valuations(k_rep)  # (B, H, Tk)
        val_dist = (nu_q.unsqueeze(-1) - nu_k.unsqueeze(-2)).abs()  # (B, H, Tq, Tk)

        beta = self.beta_omega.float().clamp_min(0.0)
        scores = s0 - beta * val_dist

        mask = causal_attn_mask(Tq, Tk, device=q.device)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).to(v_rep.dtype)
        out = torch.matmul(weights, v_rep)

        with torch.no_grad():
            stats = self.collect_scale_summary()
            causal_dist = val_dist.masked_fill(~mask, 0.0)
            stats["valuation_gap"] = float(causal_dist.mean().item())
            stats["beta_omega"] = float(beta.item())
            self.last_surreal_stats = stats

        return out

