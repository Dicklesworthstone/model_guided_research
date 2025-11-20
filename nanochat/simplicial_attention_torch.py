"""
Simplicial Attention Module (PyTorch)
Implements Higher-Order Attention via multi-hop diffusion, mimicking random walks on the simplicial complex 1-skeleton.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nanochat.model_utils import norm, apply_rotary_emb

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

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        
        # Standard Attention Weights
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        
        # Masking
        Tq = q.size(2)
        Tk = k.size(2)
        is_causal = (kv_cache is None) or (Tq == Tk)
        if is_causal and Tq > 1:
             mask = torch.tril(torch.ones(Tq, Tk, device=q.device, dtype=torch.bool))
             if Tk > Tq:
                 mask = torch.cat([torch.ones(Tq, Tk-Tq, device=q.device, dtype=torch.bool), mask], dim=1)
             att.masked_fill_(~mask, float('-inf'))
             
        att = F.softmax(att, dim=-1) # (B, H, Tq, Tk)
        
        # 1-hop Aggregation (Edges)
        y1 = att @ v
        
        # 2-hop Aggregation (Simplicial/Paths)
        # Ideally A @ A @ v, but A is (Tq, Tk) and might be rectangular if cached.
        # If cached, Tq=1, Tk=large. A @ A is not well defined unless A is square (T, T).
        # In inference (Tq=1), we can't easily do 2-hop without full history of A.
        # Simplification: During training (Tq=Tk), compute A @ y1.
        # During inference, ignore 2-hop or approximate?
        # For this "modular" demo, we'll support it fully during training/prefill, and skip/approx during generation.
        
        if Tq == Tk: # Square case (Training or Prefill)
            y2 = att @ y1 # A @ (A @ v)
        else:
            # Inference step.
            # We only have the current row of A. We don't have previous rows to do the second hop.
            # So y2 is just y1 (identity fallback) or 0.
            # Let's reuse y1 to keep scale.
            y2 = y1 
            
        y = self.mix_1 * y1 + self.mix_2 * y2
        
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
