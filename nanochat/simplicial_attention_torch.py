"""
Simplicial Attention Module (PyTorch)
Implements Higher-Order Attention via multi-hop diffusion, mimicking random walks on the simplicial complex 1-skeleton.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nanochat.model_utils import norm, apply_rotary_emb, causal_attn_mask, repeat_kv_heads

class SimplicialCausalSelfAttention(nn.Module):
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
        
        # Mixing weights for 1-hop (Edge) and 2-hop (Triangle/Path) attention
        self.mix_1 = nn.Parameter(torch.tensor(1.0))
        self.mix_2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        t0 = None
        if kv_cache is not None:
            t0 = kv_cache.get_pos()
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        if self.n_kv_head != self.n_head:
            k, v = repeat_kv_heads(k, v, n_head=self.n_head)
        
        # Standard Attention Weights
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        
        # Masking
        Tq = q.size(2)
        Tk = k.size(2)
        if kv_cache is None or Tq > 1:
            mask = causal_attn_mask(Tq, Tk, device=q.device)
            att.masked_fill_(~mask, float("-inf"))
             
        att = F.softmax(att, dim=-1) # (B, H, Tq, Tk)
        
        # 1-hop Aggregation (Edges)
        y1 = att @ v
        
        # 2-hop Aggregation (Simplicial/Paths)
        #
        # Cache-backed 2-hop: y2 = A @ y1_all, where y1_all stores the 1-hop outputs
        # for every past token (and is updated for the current chunk).
        if kv_cache is None:
            y2 = att @ y1  # A @ (A @ v)
        else:
            if t0 is None:
                raise RuntimeError("Expected t0 to be set when kv_cache is provided")
            kv_cache.ensure_simplicial_y1_cache(
                num_heads=self.n_head,
                head_dim=self.head_dim,
                dtype=y1.dtype,
                device=y1.device,
            )
            y1_all = kv_cache.insert_simplicial_y1(self.layer_idx, t0, y1)  # (B, H, Tk, D)
            y2 = att @ y1_all
            
        y = self.mix_1 * y1 + self.mix_2 * y2
        
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
