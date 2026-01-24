"""
Matrix Exponential Gauge Block (PyTorch)
Implements Gauge Equivariant Layers using Lie Groups.
Simulates "Matrix Exponential Gauge Learning" via cumulative orthogonal transports.

Mathematical core (from JAX reference):
  - Transport: T_j = prod_{l<j} exp(A_l), with A_l skew-symmetric.
  - Implemented as cumulative product of rotations (or sum of angles for U(1)).
  - Gauge invariance: compare q_i with k_j transported by R_{i<-j} = T_i T_j^{-1}.
  - R_{i<-j} acts on features to align frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaugeBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dim = config.n_embd
        if self.dim % 2 != 0:
            raise ValueError("GaugeBlock requires an even n_embd (pairwise Givens rotations).")
        if config.n_kv_head != config.n_head:
            raise ValueError("GaugeBlock does not support GQA; require n_kv_head == n_head.")

        # Lie Algebra Generators
        # We learn a skew-symmetric matrix A (generator of SO(D)) per token.
        # In the JAX code, this is efficiently handled via "pairs" of indices (Givens rotations).
        # pairs: npairs = floor(D/2).
        # angles: (B, T, npairs).
        # Transport T_j is the cumulative rotation up to j.

        self.n_pairs = self.dim // 2

        # Network to predict local connection A_l (angles) from state x_l
        self.to_angles = nn.Linear(self.dim, self.n_pairs, bias=False)

        # Standard Attention mechanism (inner block)
        # We apply gauge transformation before and after this block.
        self.c_attn = nn.Linear(self.dim, 3 * self.dim, bias=False)
        self.c_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.n_head = config.n_head
        self.head_dim = self.dim // self.n_head

    def _apply_rotations(self, x, thetas, inverse=False):
        """
        Apply 2x2 Givens rotations defined by thetas to x.
        x: (B, T, D)
        thetas: (B, T, D/2)

        Splits D into even/odd pairs (0,1), (2,3), ...
        Rotates each pair by corresponding theta.
        """
        # De-interleave to get pairs
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        c = torch.cos(thetas)
        s = torch.sin(thetas)

        if inverse:
            s = -s

        # Rotation matrix for pair (x_e, x_o):
        # [c -s] [x_e]   [c*xe - s*xo]
        # [s  c] [x_o] = [s*xe + c*xo]

        new_even = c * x_even - s * x_odd
        new_odd = s * x_even + c * x_odd

        # Interleave back
        x_new = torch.zeros_like(x)
        x_new[..., 0::2] = new_even
        x_new[..., 1::2] = new_odd
        return x_new

    def forward(self, x, cos_sin, kv_cache):
        if kv_cache is not None:
            raise NotImplementedError(
                "GaugeBlock does not yet support KV-cache incremental decoding; "
                "use non-cached generation or standard attention for inference."
            )
        B, T, C = x.size()

        # 1. Compute Local Connections A_l (Angles)
        # These represent the "infinitesimal" parallel transport between step t and t+1.
        # angles_local: (B, T, D/2)
        angles_local = self.to_angles(x)

        # 2. Compute Cumulative Transport (Gauge Field T_j)
        # T_j = prod_{l<j} exp(A_l)
        # Since 2x2 rotations in disjoint planes commute, exp(Sum A) = Prod exp(A).
        # So we can just sum the angles cumulatively.
        # angles_global[t] = sum_{k=0}^{t} angles_local[k]
        # (Note: JAX code says T_j = prod_{l<j}, i.e., exclusive prefix sum?
        #  "T_j can be applied ... via prefix-summed angles".
        #  Let's use inclusive sum for T_j, meaning frame j includes rotation A_j.)

        angles_global = torch.cumsum(angles_local, dim=1)

        # 3. Transform to "Global Frame" (Gauge Fixing)
        # v_global_j = T_j @ v_local_j
        # This aligns all vectors to the frame at t=0 (or global identity).
        # Then standard attention (dot product) computes v_global_i . v_global_j
        # which is equivalent to v_local_i . (T_i^{-1} T_j) . v_local_j
        # where T_i^{-1} T_j is the parallel transport R_{i<-j}.
        x_global = self._apply_rotations(x, angles_global)

        # 4. Apply Standard Operation in Global Frame
        # In the global frame, vectors are aligned, so we can mix them with standard ops.

        qkv = self.c_attn(x_global)
        q, k, v = torch.split(qkv, self.dim, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE (optional additional gauge field)
        cos, sin = cos_sin
        # ... RoPE implementation usually provided in model_utils
        from nanochat.model_utils import apply_rotary_emb

        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        # Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        # 5. Transform back to Local Frame (Pullback)
        # y_local_i = T_i^{-1} @ y_global_i
        y_local = self._apply_rotations(y, angles_global, inverse=True)

        return y_local
