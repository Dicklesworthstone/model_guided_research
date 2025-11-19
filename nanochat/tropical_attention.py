"""
Tropical Attention Module (JAX/Flax)
Implements attention using Max-Plus algebra for similarity.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax

from nanochat.common_jax import GPTConfig, apply_rotary_emb, rms_norm

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

        # KV Cache handling
        if self.has_variable('cache', 'cached_key'):
            cached_key = self.variable('cache', 'cached_key', jnp.zeros, (B, self.config.sequence_len, self.n_kv_head, self.head_dim), k.dtype)
            cached_val = self.variable('cache', 'cached_val', jnp.zeros, (B, self.config.sequence_len, self.n_kv_head, self.head_dim), v.dtype)
            cache_index = self.variable('cache', 'cache_index', lambda: jnp.array(0, dtype=jnp.int32))
            
            idx = cache_index.value
            # Update cache
            k_cache = cached_key.value
            v_cache = cached_val.value
            
            k_cache = lax.dynamic_update_slice(k_cache, k, (0, idx, 0, 0))
            v_cache = lax.dynamic_update_slice(v_cache, v, (0, idx, 0, 0))
            
            cached_key.value = k_cache
            cached_val.value = v_cache
            cache_index.value = idx + T
            
            k = k_cache
            v = v_cache
            
            if mask is None:
                total_len = self.config.sequence_len
                query_idx = jnp.arange(T) + idx
                key_idx = jnp.arange(total_len)
                
                # [1, 1, T, MaxLen]
                mask = key_idx[None, None, None, :] <= query_idx[None, None, :, None]
                mask = jnp.where(mask, 0, -jnp.inf)

        elif init_cache:
            self.variable('cache', 'cached_key', jnp.zeros, (B, self.config.sequence_len, self.n_kv_head, self.head_dim), k.dtype)
            self.variable('cache', 'cached_val', jnp.zeros, (B, self.config.sequence_len, self.n_kv_head, self.head_dim), v.dtype)
            self.variable('cache', 'cache_index', lambda: jnp.array(0, dtype=jnp.int32))
            
            if T > 0:
                 cached_key = self.variable('cache', 'cached_key')
                 cached_val = self.variable('cache', 'cached_val')
                 cache_index = self.variable('cache', 'cache_index')
                 
                 k_cache = cached_key.value
                 v_cache = cached_val.value
                 
                 k_cache = lax.dynamic_update_slice(k_cache, k, (0, 0, 0, 0))
                 v_cache = lax.dynamic_update_slice(v_cache, v, (0, 0, 0, 0))
                 
                 cached_key.value = k_cache
                 cached_val.value = v_cache
                 cache_index.value = T

        # GQA
        if self.n_kv_head != self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = jnp.repeat(k, n_rep, axis=2) # axis 2 is n_kv_head in [B, T, H, D]
            v = jnp.repeat(v, n_rep, axis=2)

        # Transpose to [B, H, T, D]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Tropical Attention Scores
        # Instead of dot product (sum of products), we use tropical dot product (max of sums)
        # This measures "similarity" in the max-plus semiring.
        attn_scores = tropical_dot_product(q, k)
        
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
