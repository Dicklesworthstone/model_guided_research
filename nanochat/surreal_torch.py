"""
Surreal Regularization / Probe (PyTorch)
Implements a dynamic scaling probe based on "Surreal Numbers and Transseries".
This acts as a "Meta-Optimizer" hook that logs dominance metrics and uses "Surreal Layers"
where weights are parameterized as `w = exp(s) * v` (Scale * Direction) to separate
magnitude (exponent) from direction (coefficient), mimicking Transseries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.model_utils import apply_rotary_emb, causal_attn_mask, norm


class SurrealProbe:
    def __init__(self, model, enabled=False):
        self.model = model
        self.enabled = enabled

    def step(self, loss, inputs, targets):
        """
        Compute dominance metrics:
        T_D: Data scaling benefit (simulated via split?)
        T_H: Depth scaling benefit (simulated via skipping layers)
        T_W: Width scaling benefit (simulated via masking channels)

        Returns: extra_loss, metrics
        """
        if not self.enabled:
            return 0.0, {}

        # Placeholder for dominance check
        # In a full implementation, this would run the forward pass with:
        # 1. Half depth (skip layers)
        # 2. Half width (mask channels)
        # 3. Log the ratios E_half / E_full

        return 0.0, {"surreal_balance": 1.0}


class SurrealLayer(nn.Module):
    """
    A Linear layer with "Surreal" weight parameterization.
    Weights are represented as `w = s * v` where s is a learnable scale (exponent)
    and v is the direction.
    This mimics "transseries" where we separate magnitude (scale) from direction.

    w = exp(s) * normalize(v)
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


class SurrealCausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        # Use Surreal Linear Layers for projections
        self.c_q = SurrealLayer(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = SurrealLayer(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = SurrealLayer(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = SurrealLayer(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # RoPE
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        Tq = q.size(2)
        Tk = k.size(2)
        enable_gqa = self.n_head != self.n_kv_head
        if kv_cache is None or Tq == Tk:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            attn_mask = causal_attn_mask(Tq, Tk, device=q.device)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
