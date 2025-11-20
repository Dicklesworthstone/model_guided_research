"""
Fractal Memory / Attention Module (PyTorch)
Implements attention over a "Fractal" memory structure (IFS).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nanochat.model_utils import norm, apply_rotary_emb

class FractalCausalSelfAttention(nn.Module):
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
        
        # IFS Parameters
        # "A learned router (k independent m-way classifiers) maps query q->w"
        # "Inference composes exactly k maps"
        # Here we interpret Fractal Memory as a specific attention pattern or key-value structure.
        # If we treat K/V cache as the "IFS Memory":
        # Writing to memory = finding a fixed point.
        # Reading = retrieving.
        
        # Simplification for Attention:
        # Use a hierarchical/fractal addressing scheme.
        # Q determines a "path" in the fractal.
        # K determines a "location" in the fractal.
        # Attention ~ Proximity in fractal space.
        
        # We map K to a code `c_k` and Q to a code `c_q`.
        # Score = -Distance(c_q, c_k).
        # Code is a sequence of discrete choices (m-ary digits).
        # This is effectively Soft-Routing or Hierarchical Attention.
        
        self.m = 4 # Branching factor
        self.depth = 4 # Depth of IFS
        self.router = nn.Linear(self.head_dim, self.depth * self.m, bias=False)

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
            
        # Fractal Addressing
        # Compute routing logits for Q and K
        # Shape: (B, H, T, depth, m)
        q_route = self.router(q).view(B, self.n_head, -1, self.depth, self.m)
        k_route = self.router(k).view(B, self.n_kv_head, -1, self.depth, self.m)
        
        # Softmax over m to get probabilities of taking each branch
        q_prob = F.softmax(q_route, dim=-1)
        k_prob = F.softmax(k_route, dim=-1)
        
        # Similarity: Probability of taking the SAME path.
        # Product of dot products at each depth.
        # Sim(q, k) = prod_{d=1..D} (q_d . k_d)
        # We need sum over m, then product over depth.
        
        # q_prob: (B, H, Tq, D, m)
        # k_prob: (B, H, Tk, D, m)
        # Inner product over m: (B, H, Tq, Tk, D)
        # This is huge if realized.
        
        # Standard attention uses dot product (sum over dim).
        # Here we want product over depth.
        # Log-Sim = sum_{d=1..D} log(q_d . k_d)
        
        # Let's do it efficiently:
        # Reshape to (B, H, T, D*m)
        # This is just a weird dot product attention?
        # No, strictly: sum_d log(sum_m q_dm * k_dm)
        
        # Let's approximate or standard "Fractal Distance"
        # Let's just concatenate the routing probabilities and use them as keys/queries?
        # Q' = q_prob.flatten()
        # K' = k_prob.flatten()
        # Score = Q' @ K'.T
        # This corresponds to sum_{d,m} q_dm * k_dm.
        # This is close to "expected overlap" but adds across depths.
        # IFS suggests strictly hierarchical: overlap at depth d matters more?
        # Or "Address" matching.
        
        # Implementation: Use the router outputs as the embedding for attention.
        q_flat = q_prob.view(B, self.n_head, -1, self.depth * self.m)
        k_flat = k_prob.view(B, self.n_kv_head, -1, self.depth * self.m)
        
        scores = (q_flat @ k_flat.transpose(-2, -1)) * (1.0 / (self.depth ** 0.5))
        
        # Masking
        Tq = q.size(2)
        Tk = k.size(2)
        is_causal = (kv_cache is None) or (Tq == Tk)
        if is_causal and Tq > 1:
             mask = torch.tril(torch.ones(Tq, Tk, device=q.device, dtype=torch.bool))
             if Tk > Tq:
                 mask = torch.cat([torch.ones(Tq, Tk-Tq, device=q.device, dtype=torch.bool), mask], dim=1)
             scores.masked_fill_(~mask, float('-inf'))
             
        attn_weights = F.softmax(scores, dim=-1)
        
        y = attn_weights @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
