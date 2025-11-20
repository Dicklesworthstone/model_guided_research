"""
Ultrametric Attention Module (PyTorch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nanochat.model_utils import norm, apply_rotary_emb

class UltrametricCausalSelfAttention(nn.Module):
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
        
        self.K = 8
        self.p = 2
        self.to_digits_q = nn.Linear(self.n_embd, self.K, bias=False)
        self.to_digits_k = nn.Linear(self.n_embd, self.K, bias=False)

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
            
        # Ultrametric Masking
        # Project input x to digits (simplification: project q/k inputs? No, use raw x)
        # We need digits for q and k.
        # Since K needs x from past, we should cache digits too?
        # For simplicity, just recompute from k? No, k is projected.
        # Let's project q and k vectors themselves? No, x.
        # But we don't cache x.
        # Let's assume to_digits takes the head vector.
        
        q_dig_raw = self.to_digits_q(q) # [B, H, Tq, K]
        k_dig_raw = self.to_digits_k(k) # [B, H, Tk, K]
        
        # Quantize to p-adic digits
        q_dig = (torch.sigmoid(q_dig_raw) * (self.p - 0.01)).floor().long()
        k_dig = (torch.sigmoid(k_dig_raw) * (self.p - 0.01)).floor().long()
        
        # Compare prefixes
        # q: [B, H, Tq, 1, K]
        # k: [B, H, 1, Tk, K]
        matches = (q_dig.unsqueeze(3) == k_dig.unsqueeze(2)) # [B, H, Tq, Tk, K]
        
        # Cumprod for prefix
        prefix_matches = torch.cumprod(matches, dim=-1)
        lcp = prefix_matches.sum(dim=-1) # [B, H, Tq, Tk]
        
        threshold = self.K - 1
        lcp_mask = lcp >= threshold
        
        # Standard attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Causal mask
        Tq = q.size(2)
        Tk = k.size(2)
        causal_mask = torch.tril(torch.ones(Tq, Tk, device=q.device, dtype=torch.bool))
        if Tk > Tq:
             causal_mask = torch.cat([torch.ones(Tq, Tk-Tq, device=q.device, dtype=torch.bool), causal_mask], dim=1)
             
        combined_mask = causal_mask & lcp_mask
        
        attn_scores.masked_fill_(~combined_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        y = attn_weights @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
import math
