"""
Octonion Attention Module (PyTorch)
Implements Octonion-based attention using the Cayley-Dickson construction over Quaternions.
Octonions are 8D hypercomplex numbers. Multiplication is non-associative.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nanochat.model_utils import norm, apply_rotary_emb
from nanochat.quaternion_attention_torch import qmul, qconj, qnormalize

# Octonion Multiplication via Cayley-Dickson
# O1 = (a, b), O2 = (c, d) where a,b,c,d are Quaternions.
# O1 * O2 = (a*c - d_conj*b, d*a + b*c_conj)
# Note: Order matters! Octonions are non-associative.

def omul(o1, o2):
    """
    Multiply octonion tensors o1 and o2.
    Shape: (..., 8)
    Splits into two quaternions (..., 4).
    """
    a, b = torch.split(o1, 4, dim=-1)
    c, d = torch.split(o2, 4, dim=-1)
    
    # a*c
    ac = qmul(a, c)
    # d_conj * b
    db = qmul(qconj(d), b)
    # d*a
    da = qmul(d, a)
    # b*c_conj
    bc = qmul(b, qconj(c))
    
    first = ac - db
    second = da + bc
    
    return torch.cat([first, second], dim=-1)

def oconj(o):
    """
    Conjugate of octonion o = (a, b) is (a_conj, -b).
    """
    a, b = torch.split(o, 4, dim=-1)
    return torch.cat([qconj(a), -b], dim=-1)

def onorm(o):
    return torch.norm(o, dim=-1, keepdim=True)

def onormalize(o):
    return F.normalize(o, p=2, dim=-1)

class OctonionCausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        
        if self.n_embd % 8 != 0:
            raise ValueError("n_embd must be divisible by 8 for Octonion attention")
        
        self.head_dim = self.n_embd // self.n_head
        
        if self.head_dim % 8 != 0:
             raise ValueError("head_dim must be divisible by 8 for Octonion attention")

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Rotary Embeddings (Standard)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
            
        # Interpret as Octonions
        # (..., D) -> (..., D/8, 8)
        q_o = q.view(B, self.n_head, -1, self.head_dim // 8, 8)
        k_o = k.view(B, self.n_kv_head, -1, self.head_dim // 8, 8)
        v_o = v.view(B, self.n_kv_head, -1, self.head_dim // 8, 8)
        
        # Normalize Keys (Rotor logic)
        k_o = onormalize(k_o)
        
        # Attention Score
        # Standard Dot Product (Real part of Octonion product?)
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
             
        probs = F.softmax(scores, dim=-1) # (B, H, Tq, Tk)
        
        # Value Aggregation
        # "Octonionic Signal Flow": Y = sum ( Q * K_conj * V )
        # But Octonions are non-associative! (Q * K_conj) * V != Q * (K_conj * V)
        # Which order?
        # "Signal Flow" implies transformation along the path.
        # K->Q is the path.
        # R = Q * K_conj.
        # Y = R * V.
        # So (Q * K_conj) * V.
        
        # Implementation:
        # 1. R_ij = Q_i * K_j_conj  (Octonion Mul)
        # 2. V'_ij = R_ij * V_j     (Octonion Mul)
        # 3. Y_i = sum_j probs_ij * V'_ij
        
        # This is O(T^2).
        # Is there a trick?
        # Associativity holds for real scalars.
        # But not for Octonions.
        # So we cannot factor out Q_i easily like in Quaternion/Real case?
        # Actually, Quaternion IS associative. Octonion is NOT.
        # So for Octonion attention, we MUST compute the pairwise product if we want full non-associativity.
        # This makes it O(T^2 * D). Expensive.
        
        # Simplification for "Efficient" Octonion Attention:
        # Maybe "Associative Octonion" approximation?
        # Or just accept O(T^2) for the demo (Sequence length is short, 256).
        # Let's do O(T^2) for correctness of the "Idea".
        
        # We need to broadcast Q, K, V to (B, H, Tq, Tk, N, 8)
        # This will OOM if not careful.
        # Loop over Tq?
        
        y_list = []
        for i in range(Tq):
            # q_i: (B, H, 1, N, 8)
            q_i = q_o[:, :, i:i+1, :, :]
            
            # k_all: (B, H, Tk, N, 8)
            # v_all: (B, H, Tk, N, 8)
            
            # R_i = q_i * k_all_conj
            # We need to broadcast q_i to Tk
            k_conj = oconj(k_o)
            r_i = omul(q_i, k_conj) # (B, H, Tk, N, 8)
            
            # term = r_i * v_all
            term = omul(r_i, v_o) # (B, H, Tk, N, 8)
            
            # weighted sum
            # probs_i: (B, H, 1, Tk)
            p_i = probs[:, :, i:i+1, :]
            p_i = p_i.unsqueeze(-1).unsqueeze(-1) # (B, H, 1, Tk, 1, 1)
            
            # sum over Tk (dim 2)
            y_i = (term * p_i).sum(dim=2) # (B, H, 1, N, 8)
            y_list.append(y_i)
            
        y_o = torch.cat(y_list, dim=2) # (B, H, Tq, N, 8)
        
        # Flatten
        y = y_o.view(B, self.n_head, Tq, self.head_dim)
        
        y = y.transpose(1, 2).contiguous().view(B, Tq, -1)
        y = self.c_proj(y)
        return y
