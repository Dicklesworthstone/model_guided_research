"""
Tropical Attention Module (PyTorch)
Implements attention using Max-Plus algebra for similarity.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from nanochat.model_utils import norm, apply_rotary_emb, causal_attn_mask, repeat_kv_heads


def _tropical_center(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Gauge-fix by subtracting the max along `dim` so the max becomes 0."""
    return x - x.amax(dim=dim, keepdim=True)


def tropical_max_plus_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    gauge_fix: bool,
    score_center: bool,
    return_margins: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Pure max-plus (tropical) attention with optional gauge-fixing and margin certificates.

    Shapes:
      q: (B, H, Tq, D)
      k: (B, H, Tk, D)
      v: (B, H, Tk, D)

    Returns:
      y: (B, H, Tq, D)
      gamma: optional per-token, per-head runner-up margin (B, H, Tq) (>=0), or None if disabled.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("tropical_max_plus_attention expects q/k/v of shape (B, H, T, D)")
    if k.shape != v.shape:
        raise ValueError(f"tropical_max_plus_attention expects k and v to have the same shape, got k={k.shape}, v={v.shape}")
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[3] != k.shape[3]:
        raise ValueError(f"tropical_max_plus_attention head shapes mismatch: q={q.shape}, k={k.shape}, v={v.shape}")

    B, H, Tq, D = q.shape
    Tk = k.size(2)

    if gauge_fix:
        q = _tropical_center(q, dim=-1)
        k = _tropical_center(k, dim=-1)
        v = _tropical_center(v, dim=-1)

    # Similarity/logits: score(q,k) = max_d (q_d + k_d)  (max-plus dot product)
    attn_scores = torch.max(q.unsqueeze(3) + k.unsqueeze(2), dim=-1).values  # (B, H, Tq, Tk)

    mask = causal_attn_mask(Tq, Tk, device=q.device)
    attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

    # Score-centering is a pure gauge change (per-query constant shift): it preserves argmax structure.
    if score_center:
        attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True)

    # Value aggregation: y_d = max_k (score(q,k) + v_{k,d})  (tropical matmul)
    logits = attn_scores.unsqueeze(-1) + v.unsqueeze(2)  # (B, H, Tq, Tk, D)
    gamma: Optional[torch.Tensor]
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


class TropicalCausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.tropical_gauge_fix = bool(getattr(config, "tropical_gauge_fix", True))
        self.tropical_score_center = bool(getattr(config, "tropical_score_center", True))
        self.tropical_record_margins = bool(getattr(config, "tropical_record_margins", False))
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
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

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        if self.n_kv_head != self.n_head:
            k, v = repeat_kv_heads(k, v, n_head=self.n_head)

        y, gamma = tropical_max_plus_attention(
            q,
            k,
            v,
            gauge_fix=self.tropical_gauge_fix,
            score_center=self.tropical_score_center,
            return_margins=self.tropical_record_margins,
        )
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

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        if self.tropical_gauge_fix:
            y = _tropical_center(y, dim=-1)
        return y
