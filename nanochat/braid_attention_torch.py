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
        # 1. Compute "sorting scores" for Q and K.
        # 2. Use sorting permutation to reorder V?
        # The JAX code describes a process: "Score each token... Output permutation placing aggregator first... Output w = sigma^k... execute left->right."
        
        # Interpretation for Standard Attention Replacement:
        # We want a "braid-like" mixing.
        # A simple version:
        # Learn a permutation matrix P based on Q and K compatibility.
        # Y = P @ V.
        # This is standard attention if P is softmax(Q @ K.T).
        # Braid attention usually implies local swaps.
        # Let's implement a "Braid Sort" attention:
        # Use Q to query the "sortedness" of K?
        
        # Let's stick to the JAX paper's "Aggregator" model.
        # "Score each non-aggregator token j with p_j... Output pi placing aggregator first... Accumulate."
        # This sounds like Attention where the Query is the "Aggregator" and Keys/Values are tokens.
        # The "braid" part is the specific accumulation function via crossings.
        # Crossing (x, y) -> (x+y, y).
        # If we chain this: x_agg becomes x_agg + sum(y_j for j in Selected).
        # This is EXACTLY sum-attention (without softmax normalization, maybe gated).
        
        # So Braid Attention (in this specific restricted form) is:
        # 1. Score each key k_j against query q_i -> probability p_ij.
        # 2. Hard/Soft selection of keys.
        # 3. Sum values v_j weighted by p_ij.
        # Difference from Softmax:
        # - Probabilities p_ij are independent (sigmoid), not competing (softmax).
        # - The aggregation is a direct sum, preserving "invariants" (payload sum).
        
        # Impl:
        # Scores = Sigmoid( (Q @ K.T) / sqrt(d) + Bias )
        # Y = Scores @ V
        
        # Q: (B, H, Tq, D)
        # K: (B, H, Tk, D)
        scores = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        
        # Masking
        Tq = q.size(2)
        Tk = k.size(2)
        is_causal = (kv_cache is None) or (Tq == Tk)
        if is_causal and Tq > 1:
             mask = torch.tril(torch.ones(Tq, Tk, device=q.device, dtype=torch.bool))
             if Tk > Tq:
                 mask = torch.cat([torch.ones(Tq, Tk-Tq, device=q.device, dtype=torch.bool), mask], dim=1)
             scores.masked_fill_(~mask, float('-inf'))
        
        # Sigmoid instead of Softmax
        # Note: scores might be very negative due to masking, sigmoid(-inf) = 0. Correct.
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
