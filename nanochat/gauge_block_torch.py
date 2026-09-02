"""
Matrix Exponential Gauge Block (PyTorch)
Implements Gauge Equivariant Layers using Lie Groups.
Simulates "Matrix Exponential Gauge Learning" via cumulative orthogonal transports.

Mathematical core (from JAX reference):
  - Transport: T_j = prod_{l<j} exp(A_l), with A_l skew-symmetric.
  - Implemented as cumulative product of rotations (or sum of angles for U(1)).
  - Gauge invariance: compare q_i with k_j transported by R_{i<-j} = T_i T_j^{-1}.
  - R_{i<-j} acts on features to align frames.

Block structure (bead model_guided_research-1fr6): the gauge block REPLACES the
whole transformer block (the A1 boundary decision from 7b0.1), but it carries
the canonical pre-norm residual skeleton inside itself:

    x = x + gauge_attention(norm(x))      # transport-wrapped attention
    x = x + mlp(norm(x))                  # standard MLP slot (passed in by Block)

Without the residuals, the repo's zero-init convention (c_proj and lm_head
start at zero) made the whole network output exactly zero and the gradients
mutually annihilate (total grad = 0.0) - gauge training was frozen at ln(V).
The residual restores the each-block-starts-as-identity property the zero
init is designed around. The MLP slot is passed in (mirroring the reversible
block's F/G modules) so _build_ffn dispatch - including the tropical FFN -
applies to gauge like every other block.

KV-cache decode (bead model_guided_research-7b0.5): the obstacle is the
cumulative gauge field - token t's global frame is the cumsum of all previous
local angles. The KVCache carries a per-layer fp32 running-cumsum lane
(engine.py, following the simplicial y1 precedent). Correctness crux, the
frame-invariance argument: keys and values are computed from the GLOBALLY
transported representation x_global_t = R(cumsum_t) @ norm(x_t) and inserted
into the cache already in the global frame. The global frame is fixed once
(frame of token 0), so a cached key never needs re-transporting when later
queries arrive: at decode step t we reconstruct exactly the same cumsum_t
(tracked in fp32 across steps so bf16 drift cannot accumulate), transport the
new token into the same global frame, and attend against cached keys as-is.
The output is pulled back to the local frame with the CURRENT token's angles,
which are available at decode time without any history.
"""

import torch
import torch.nn as nn

from nanochat.model_utils import apply_rotary_emb, norm, sdpa_causal_attend


class GaugeBlock(nn.Module):
    def __init__(self, config, layer_idx, mlp: nn.Module):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dim = config.n_embd
        if self.dim % 2 != 0:
            raise ValueError("GaugeBlock requires an even n_embd (pairwise Givens rotations).")

        # Lie Algebra Generators
        # We learn a skew-symmetric matrix A (generator of SO(D)) per token.
        # In the JAX code, this is efficiently handled via "pairs" of indices (Givens rotations).
        # pairs: npairs = floor(D/2).
        # angles: (B, T, npairs).
        # Transport T_j is the cumulative rotation up to j.

        self.n_pairs = self.dim // 2

        # Network to predict local connection A_l (angles) from state x_l
        self.to_angles = nn.Linear(self.dim, self.n_pairs, bias=False)

        # Attention projections in the global frame. Separate q/k/v (rather
        # than a fused c_attn) so k/v can carry n_kv_head heads - GQA matches
        # the standard path, with SDPA's enable_gqa broadcasting KV heads.
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        if not (self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0):
            # GPT._validate_config checks this too, but GaugeBlock is also
            # constructed standalone (certify's make_block builds Block directly).
            raise ValueError("n_kv_head must divide n_head and be <= n_head")
        self.head_dim = self.dim // self.n_head
        self.c_q = nn.Linear(self.dim, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.dim, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.dim, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.dim, self.dim, bias=False)

        # Standard MLP slot, constructed by Block via _build_ffn (so ffn_type
        # dispatch - standard / tropical / tropical-rational - applies here).
        self.mlp = mlp

    @staticmethod
    def _apply_rotations(x: torch.Tensor, thetas: torch.Tensor, inverse: bool = False) -> torch.Tensor:
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

        # Interleave back: stack the pair members on a trailing axis and
        # flatten, which is a single allocation and keeps autograd out of
        # in-place strided writes (the previous zeros_like + two index_put
        # writes were three passes over x).
        return torch.stack((new_even, new_odd), dim=-1).flatten(-2)

    def _gauge_attention(self, x, cos_sin, kv_cache) -> torch.Tensor:
        B, T, C = x.size()

        # 1. Compute Local Connections A_l (Angles)
        # These represent the "infinitesimal" parallel transport between step t and t+1.
        # angles_local: (B, T, D/2)
        angles_local = self.to_angles(x)

        # 2. Compute Cumulative Transport (Gauge Field T_j)
        # T_j = prod_{l<=j} exp(A_l)
        # Since 2x2 rotations in disjoint planes commute, exp(Sum A) = Prod exp(A),
        # so the cumulative rotation is just the prefix sum of angles.
        # Accumulation runs in fp32 REGARDLESS of autocast/input dtype: under a
        # cached long decode, bf16 angle accumulation drifts over thousands of
        # steps; the fp32 lane keeps the cached and full-forward frames aligned.
        if kv_cache is not None:
            kv_cache.ensure_gauge_angle_cache(n_pairs=self.n_pairs, device=x.device)
            # Desync guard: the lane ACCUMULATES, so a re-run of the same
            # forward (or a cache rewound without reset()) would silently
            # double-count angles - and the per-token history needed to
            # rebuild the lane is not stored. Fail loudly instead. The lane's
            # token count must equal the cache write position, which is the
            # same for every layer within one forward (pos advances only
            # after the last layer inserts).
            seen = kv_cache.gauge_angle_pos[self.layer_idx]
            pos0 = kv_cache.get_pos()
            if seen != pos0:
                raise RuntimeError(
                    f"gauge angle lane desync at layer {self.layer_idx}: lane has accumulated {seen} "
                    f"tokens but the cache position is {pos0}. A forward was re-run against the same "
                    "cache, or the cache was rewound without reset(); the cumulative gauge field "
                    "cannot be reconstructed - call kv_cache.reset() and re-prefill."
                )
            prev = kv_cache.gauge_cum_angles[self.layer_idx]  # (B, n_pairs) fp32
            angles_global = prev.unsqueeze(1) + torch.cumsum(angles_local.float(), dim=1)
            kv_cache.gauge_cum_angles[self.layer_idx] = angles_global[:, -1].detach()
            kv_cache.gauge_angle_pos[self.layer_idx] = pos0 + T
        else:
            angles_global = torch.cumsum(angles_local.float(), dim=1)

        # 3. Transform to "Global Frame" (Gauge Fixing)
        # v_global_j = T_j @ v_local_j aligns every token to the frame of t=0,
        # so standard dot-product attention computes
        # v_local_i . (T_i^{-1} T_j) . v_local_j - the parallel-transported
        # comparison R_{i<-j}. Keys/values enter the KV cache ALREADY in this
        # global frame (see the module docstring for why cached keys stay
        # valid for all future queries).
        x_global = self._apply_rotations(x, angles_global).to(x.dtype)

        q = self.c_q(x_global).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x_global).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x_global).view(B, T, self.n_kv_head, self.head_dim)

        # Apply RoPE (optional additional gauge field) in the (B, T, H, D) layout
        # expected by apply_rotary_emb (time at dim 1), BEFORE moving heads to the
        # batch dim — matching the canonical order in gpt.CausalSelfAttention.
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        # Attention in the global frame (training / single-token / chunked decode)
        enable_gqa = self.n_head != self.n_kv_head
        y = sdpa_causal_attend(q, k, v, kv_cache=kv_cache, enable_gqa=enable_gqa)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        # 5. Transform back to Local Frame (Pullback)
        # y_local_i = T_i^{-1} @ y_global_i — uses only the CURRENT tokens'
        # cumulative angles, available at decode time without history.
        y_local = self._apply_rotations(y, angles_global, inverse=True).to(y.dtype)

        return y_local

    def forward(self, x: torch.Tensor, cos_sin, kv_cache) -> torch.Tensor:
        # Canonical pre-norm residual skeleton (see module docstring / 1fr6):
        # zero-initialized c_proj weights make both branches vanish at init,
        # so the block starts as the identity instead of annihilating the
        # residual stream.
        x = x + self._gauge_attention(norm(x), cos_sin, kv_cache)
        mlp_out: torch.Tensor = self.mlp(norm(x))
        x = x + mlp_out
        return x
