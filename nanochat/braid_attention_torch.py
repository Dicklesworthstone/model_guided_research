"""
Braid Attention Module (PyTorch)
Implements "Braid Attention": permutation + limited crossings with invariant-aware aggregation.

Program model (from JAX reference):
- Score each token (and an 'aggregator' token).
- Output a permutation pi.
- Execute 'braid words' w = sigma_1^k (aggregator swaps with neighbors k times).
- Crossing algebra: (x_a, y_a), (x_b, y_b) -> (x_a + y_b, y_a), (x_b + y_a, y_b).
- This restricted crossing accumulates payloads {y} onto the aggregator {x}.

PyTorch Implementation:
- "Prioritized Accumulation": We simulate the permutation and accumulation process
  using a soft scoring mechanism that respects the additive invariant.
- Instead of hard sorting and scanning (hard to autograd/GPU-optimize in PyTorch without custom kernels),
  we use a "Soft Braid" approximation:
  - Learn priority scores s_i.
  - Probability of crossing (swap) P(i crosses j) ~ Sigmoid(s_i - s_j).
  - Accumulate: x_i += sum_j P(j crosses i) * y_j.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nanochat.model_utils import norm, apply_rotary_emb

class BraidCausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        
        # Braid Scoring Network
        # JAX: "sigmoid(w*[tag, value] + b)"
        # Here we learn a scalar score from the head dimension.
        self.braid_score = nn.Linear(self.head_dim, 1, bias=False)

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
            
        # Braid Attention Logic
        # 1. Compute priority scores for Q (Aggregator) and K (Tokens).
        # In Braid model, aggregator is just a designated strand.
        # Here, every query i acts as a potential aggregator for its past.
        
        # Score(q): (B, H, Tq, 1)
        s_q = self.braid_score(q)
        # Score(k): (B, H, Tk, 1)
        s_k = self.braid_score(k)
        
        # Crossing Condition: Aggregator i interacts with Token j if i "sorts" past j?
        # Or if they are "compatible".
        # JAX: "Allowed set A = { j : p_j > tau }".
        # This implies interaction depends purely on the token j's score, relative to a threshold.
        # But in Attention, we need Q-dependence.
        # Let's interpret: Q sets the threshold/context?
        # "Score = s_q + s_k" (additive interaction) or "s_q * s_k" (multiplicative).
        # The previous "s_q + s_k.T" gave a pairwise matrix.
        
        scores = s_q + s_k.transpose(-2, -1) # (B, H, Tq, Tk)
        
        # Masking (Causal)
        Tq = q.size(2)
        Tk = k.size(2)
        is_causal = (kv_cache is None) or (Tq == Tk)
        if is_causal and Tq > 1:
             mask = torch.tril(torch.ones(Tq, Tk, device=q.device, dtype=torch.bool))
             if Tk > Tq:
                 mask = torch.cat([torch.ones(Tq, Tk-Tq, device=q.device, dtype=torch.bool), mask], dim=1)
             scores.masked_fill_(~mask, float('-inf'))
        
        # Interaction Strength
        # Sigmoid(Score) -> [0, 1].
        # This represents "how much of y_j crosses into x_i".
        # Since Braid crossing is x += y, we don't normalize to sum=1.
        # We sum raw values weighted by crossing probability.
        attn_weights = torch.sigmoid(scores)
        
        # Accumulation: x += sum(p * y)
        # Note: Standard attention is convex combination (sum p = 1).
        # Braid is additive accumulation (sum p can be anything).
        # This can lead to explosion.
        # We add a scaling factor 1/sqrt(T) or similar, or rely on LayerNorm.
        # JAX code uses "MSE(pred_soft - gt)" where GT is sum.
        # So it expects to learn the scale.
        # We'll divide by sqrt(Tk) to keep variance stable at init.
        
        y = attn_weights @ v # [B, H, Tq, D]
        y = y / (Tk ** 0.5 + 1e-6)
        
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y