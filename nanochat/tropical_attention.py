"""
Tropical Attention Module (JAX/Flax)
Implements attention using Max-Plus algebra for similarity.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional
import math

from nanochat.gpt_jax import GPTConfig, apply_rotary_emb, rms_norm

def tropical_dot_product(q, k):
    """
    Computes tropical dot product: max_d (q_d + k_d)
    q: [B, H, T, D]
    k: [B, H, S, D]
    Returns: [B, H, T, S]
    """
    # Naive implementation: broadcast and max
    # q: [B, H, T, 1, D]
    # k: [B, H, 1, S, D]
    # sum: [B, H, T, S, D]
    # max: [B, H, T, S]
    
    # To save memory, we might want to chunk this if T is large.
    # But for now, let's trust JAX's compiler or run on small seq len.
    return jnp.max(q[..., :, None, :] + k[..., None, :, :], axis=-1)

class TropicalCausalSelfAttention(nn.Module):
    config: GPTConfig
    layer_idx: int

    def setup(self):
        self.n_head = self.config.n_head
        self.n_kv_head = self.config.n_kv_head
        self.n_embd = self.config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        # No bias in linear layers
        self.c_q = nn.Dense(self.n_head * self.head_dim, use_bias=False, kernel_init=nn.initializers.normal(stddev=0.02))
        self.c_k = nn.Dense(self.n_kv_head * self.head_dim, use_bias=False, kernel_init=nn.initializers.normal(stddev=0.02))
        self.c_v = nn.Dense(self.n_kv_head * self.head_dim, use_bias=False, kernel_init=nn.initializers.normal(stddev=0.02))
        self.c_proj = nn.Dense(self.n_embd, use_bias=False, kernel_init=nn.initializers.normal(stddev=0.02))

    def __call__(self, x, cos, sin, mask=None, init_cache=False):
        B, T, C = x.shape
        
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # QK Norm (Important for Tropical? Maybe less so since magnitude matters in max-plus)
        # But let's keep it for stability consistency with the backbone
        q = rms_norm(q)
        k = rms_norm(k)

        # Transpose to [B, H, T, D]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # KV Cache handling (Placeholder for Tropical)
        if self.has_variable('cache', 'cached_key'):
             # TODO: Implement Tropical KV Cache
             pass
        elif init_cache:
             # TODO: Init Tropical KV Cache
             pass

        # GQA
        if self.n_kv_head != self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = jnp.repeat(k, n_rep, axis=1)
            v = jnp.repeat(v, n_rep, axis=1)

        # Tropical Attention Scores
        # Instead of dot product (sum of products), we use tropical dot product (max of sums)
        # This measures "similarity" in the max-plus semiring.
        attn_scores = tropical_dot_product(q, k)
        
        # Scale? 
        # In standard attention we scale by 1/sqrt(D).
        # In tropical, scaling q+k by alpha is just alpha*(q+k).
        # Maybe we don't need scaling or we scale the inputs?
        # Let's try without scaling first, or scale by 1.0.
        
        if mask is not None:
            attn_scores = attn_scores + mask
        else:
            causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
            causal_mask = jnp.where(causal_mask, 0, -jnp.inf)
            attn_scores = attn_scores + causal_mask

        # Softmax or Hardmax?
        # To be differentiable and usable in standard training, we use Softmax.
        # This makes it "Soft Tropical Attention".
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        
        # Value aggregation
        # Standard: sum(weights * v)
        # Tropical: max(weights + v)? 
        # If we want to go Full Tropical, we should use max-plus for values too.
        # But that would require "Hard" selection or a soft-max approximation.
        # Let's stick to "Tropical Similarity, Standard Aggregation" for the first hybrid.
        # This preserves the "1-Lipschitz" property of the scoring function at least.
        
        y = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)

        # Re-assemble
        y = jnp.transpose(y, (0, 2, 1, 3)).reshape(B, T, C)
        y = self.c_proj(y)
        return y

