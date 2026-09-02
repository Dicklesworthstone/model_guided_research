"""
Surreal parameterization (PyTorch)

"Surreal Layers" parameterize weights as `w = exp(s) * normalize(v)`
(scale * direction), separating magnitude (exponent) from direction
(coefficient) in the spirit of transseries. `SurrealCausalSelfAttention` is
standard scaled-dot-product attention whose four projections use this
parameterization; it changes the optimization geometry, not the attention
math. (An earlier dominance "probe" stub lived here; it always returned
constants and was imported nowhere, so it was removed rather than left
looking like a feature.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.model_utils import AttentionCore, sdpa_causal_attend


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


class SurrealCausalSelfAttention(AttentionCore):
    # GQA is handled inside SDPA via enable_gqa; no materialized repeat.
    gqa_via_repeat = False

    def __init__(self, config, layer_idx):
        # Surreal Linear Layers (w = exp(s) * normalize(v)) for all projections;
        # attribute names and state-dict keys match the canonical scaffold.
        super().__init__(config, layer_idx, linear_cls=SurrealLayer)

    def attend(self, q, k, v, *, kv_cache, pos0):
        enable_gqa = self.n_head != self.n_kv_head
        return sdpa_causal_attend(q, k, v, kv_cache=kv_cache, enable_gqa=enable_gqa)
