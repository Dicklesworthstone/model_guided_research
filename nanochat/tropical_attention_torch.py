"""
Tropical Attention Module (PyTorch)
Implements attention using Max-Plus algebra for similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nanochat.model_utils import norm, apply_rotary_emb

class TropicalCausalSelfAttention(nn.Module):
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
        
        Tq = q.size(2)
        Tk = k.size(2)

        # Tropical Attention
        # Score = max(Q + K)
        # q: [B, H, Tq, D]
        # k: [B, H, Tk, D]
        # Q+K => [B, H, Tq, Tk, D] (Broadcast) -> max over D => [B, H, Tq, Tk]
        
        # Memory optimization: chunking if needed. For now, direct implementation.
        # Expand dims:
        # q: [B, H, Tq, 1, D]
        # k: [B, H, 1, Tk, D]
        attn_scores = torch.max(q.unsqueeze(3) + k.unsqueeze(2), dim=-1).values
        
        # Masking
        if Tq > 1: # Causal mask
             mask = torch.tril(torch.ones(Tq, Tk, device=q.device, dtype=torch.bool))
             if Tk > Tq: # If caching, we can attend to all past
                 mask = torch.cat([torch.ones(Tq, Tk-Tq, device=q.device, dtype=torch.bool), mask], dim=1)
             
             attn_scores.masked_fill_(~mask, float('-inf'))

        # Softmax (Soft Tropical)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        y = attn_weights @ v # [B, H, Tq, D]
        
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
