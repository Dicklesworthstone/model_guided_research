"""
Braid Attention Module (PyTorch)
Implements "Braid Attention" where query-key interactions are modeled as braid crossings.
Based on the concept that a sequence of tokens can be permuted and mixed via local crossings.

Simplified PyTorch implementation:
- Uses a learnable "crossing" network that takes pairs of (query, key) and outputs a mixed state.
- Attention scores are derived from the "braid" sorting process or a simplified proxy.
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
        
        # Braid Crossing Network
        # Takes concatenated (q, k) or (v_left, v_right) and outputs updated values.
        # Ideally, this should be a unitary/invertible operation.
        # Simplification: We assume "sorting" attention.
        # We learn a "sorting score" for each token.
        self.braid_score = nn.Linear(self.head_dim, 1, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        
        # Make head batch dim
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
            
        # Braid Attention Logic
        # 1. Compute "sorting scores" for Q and K using the learned braid_score network.
        # This implements a "Priority-based" attention where interaction is determined by the sum of priorities.
        # Braid Crossing: (x, y) -> (x+y, y). Aggregation order matters.
        # Here we approximate the "selection probability" p_ij ~ Sigmoid(Score(q_i) + Score(k_j)).
        
        # q: (B, H, Tq, D)
        # k: (B, H, Tk, D)
        
        # Score(q): (B, H, Tq, 1)
        s_q = self.braid_score(q)
        # Score(k): (B, H, Tk, 1)
        s_k = self.braid_score(k)
        
        # Pairwise score: s_q + s_k.T
        # Shape: (B, H, Tq, Tk)
        scores = s_q + s_k.transpose(-2, -1)
        
        # Masking
        Tq = q.size(2)
        Tk = k.size(2)
        is_causal = (kv_cache is None) or (Tq == Tk)
        if is_causal and Tq > 1:
             mask = torch.tril(torch.ones(Tq, Tk, device=q.device, dtype=torch.bool))
             if Tk > Tq:
                 mask = torch.cat([torch.ones(Tq, Tk-Tq, device=q.device, dtype=torch.bool), mask], dim=1)
             scores.masked_fill_(~mask, float('-inf'))
        
        # Sigmoid to get "independent crossing probabilities"
        attn_weights = torch.sigmoid(scores)
        
        # Normalize? The paper says "accumulate sum(y)", so maybe no normalization?
        # But standard transformers expect scale preservation.
        # Let's add a learnable scale or normalize by expected number of active tokens?
        # For now: raw sum.
        # To prevent blowup: divide by sqrt(Tk)? Or just let LayerNorm handle it.
        # Let's normalize by a constant factor for stability.
        
        y = attn_weights @ v # [B, H, Tq, D]
        
        y = y / (Tk ** 0.5 + 1e-6) # Heuristic scaling
        
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
