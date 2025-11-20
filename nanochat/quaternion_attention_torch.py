"""
Quaternion Attention Module (PyTorch)
Implements "Rotor-Gate" style attention where features are treated as quaternions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nanochat.model_utils import norm, apply_rotary_emb

def qmul(a, b):
    """
    Multiply quaternion tensors a and b.
    Shape: (..., 4)
    """
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    
    return torch.stack((ow, ox, oy, oz), dim=-1)

def qconj(q):
    """
    Conjugate of quaternion q.
    Shape: (..., 4)
    """
    w, x, y, z = q.unbind(-1)
    return torch.stack((w, -x, -y, -z), dim=-1)

def qnorm(q):
    """
    Norm of quaternion q.
    """
    return torch.norm(q, dim=-1, keepdim=True)

def qnormalize(q):
    return F.normalize(q, p=2, dim=-1)

class QuaternionCausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        
        if self.n_embd % 4 != 0:
            raise ValueError("n_embd must be divisible by 4 for Quaternion attention")
        
        self.head_dim = self.n_embd // self.n_head
        
        if self.head_dim % 4 != 0:
             raise ValueError("head_dim must be divisible by 4 for Quaternion attention")

        # We use standard linear layers for projection, but interpret output as quaternions
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # Project
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Rotary Embeddings (Standard)
        # Note: Rotary embeddings are complex rotations. Quaternions are 4D.
        # We can apply rotary embeddings to the pairs inside the quaternion or just apply it before quaternion interpretation.
        # Let's apply standard RoPE for now as position encoding.
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        
        # Transpose to (B, H, T, D)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
            
        # Interpret as Quaternions
        # Reshape (..., D) -> (..., D/4, 4)
        q_q = q.view(B, self.n_head, -1, self.head_dim // 4, 4)
        k_q = k.view(B, self.n_kv_head, -1, self.head_dim // 4, 4)
        v_q = v.view(B, self.n_kv_head, -1, self.head_dim // 4, 4)
        
        # Normalize to unit quaternions for "Rotor" interpretation?
        # The paper/code suggests unit rotors. Let's normalize K.
        k_q = qnormalize(k_q)
        # q_q = qnormalize(q_q) # Maybe?

        # Attention Score = ScalarPart(Q * K_conj)
        # Q: (B, H, Tq, N, 4)
        # K: (B, H, Tk, N, 4)
        # We need to sum over N (the quaternion vector dimension) to get a single score per token pair?
        # Standard attention: dot product.
        # Dot product of two quaternions q1, q2 is ScalarPart(q1 * q2_conj).
        # So we can just compute Q * K_conj and sum the scalar parts over the N dimension?
        # Or do we want vector-valued attention weights?
        # Standard Transformer reduces to scalar attention weights.
        
        # Compute Q * K_conj
        # We need broadcasting for Tq and Tk.
        # q_q: (B, H, Tq, 1, N, 4)
        # k_q: (B, H, 1, Tk, N, 4)
        
        # Memory optimization: This 5D/6D tensor might be huge.
        # Let's compute the dot product directly.
        # Dot product in R^4 is same as ScalarPart(q1 * q2_conj).
        # So we can just treat them as real vectors for the score calculation!
        # (B, H, Tq, D) @ (B, H, Tk, D)^T -> (B, H, Tq, Tk)
        
        # Wait, the "Rotor-Gate" idea is that we use the FULL quaternion product for the value update.
        # But for the *score*, we usually need a scalar.
        # Let's stick to: Score = Real Dot Product (standard).
        # BUT, the value aggregation is: Y = sum(Attn * (Q * K_conj * V)).
        # "Attention via relative rotors: Query-key relations use q * conj(k) both to generate scalar scores and to rotate values"
        
        # 1. Compute Relative Rotors R_ij = Q_i * K_j_conj
        # This is (Tq, Tk, N, 4). Heavy!
        # If we simply compute standard attention scores:
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Masking
        Tq = q.size(2)
        Tk = k.size(2)
        is_causal = (kv_cache is None) or (Tq == Tk) # rough check
        
        if is_causal and Tq > 1:
             mask = torch.tril(torch.ones(Tq, Tk, device=q.device, dtype=torch.bool))
             if Tk > Tq:
                 mask = torch.cat([torch.ones(Tq, Tk-Tq, device=q.device, dtype=torch.bool), mask], dim=1)
             scores.masked_fill_(~mask, float('-inf'))
             
        probs = F.softmax(scores, dim=-1) # (B, H, Tq, Tk)
        
        # Value mixing
        # Standard: y = probs @ v
        # Rotor: y_i = sum_j probs_ij * (R_ij * v_j)
        # R_ij = q_i * k_j_conj
        # y_i = sum_j probs_ij * (q_i * k_j_conj * v_j)
        # This is associative: q_i * sum_j (probs_ij * k_j_conj * v_j)
        # So we can:
        # 1. Compute T_j = k_j_conj * v_j  (Elementwise quaternion mul)
        # 2. Aggregate: A_i = sum_j probs_ij * T_j (Standard attention aggregation)
        # 3. Rotate: y_i = q_i * A_i
        
        # Step 1: T = K_conj * V
        # k_q: (B, H, Tk, N, 4)
        # v_q: (B, H, Tk, N, 4)
        t_q = qmul(qconj(k_q), v_q) # (B, H, Tk, N, 4)
        
        # Flatten T back to (B, H, Tk, D) for aggregation
        t_flat = t_q.view(B, self.n_head, Tk, self.head_dim)
        
        # Step 2: Aggregate
        # probs: (B, H, Tq, Tk)
        # t_flat: (B, H, Tk, D)
        # agg = probs @ t_flat -> (B, H, Tq, D)
        agg = probs @ t_flat
        
        # Step 3: Rotate
        # agg reshaped to quaternion
        agg_q = agg.view(B, self.n_head, -1, self.head_dim // 4, 4)
        
        # y = q * agg
        y_q = qmul(q_q, agg_q)
        
        # Flatten
        y = y_q.view(B, self.n_head, -1, self.head_dim)
        
        # Output projection
        y = y.transpose(1, 2).contiguous().view(B, Tq, -1)
        y = self.c_proj(y)
        return y
