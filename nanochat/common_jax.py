"""
Common JAX utilities for nanochat.
"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    use_tropical: bool = False

def rms_norm(x):
    # Purely functional rmsnorm with no learnable params
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-6)

def apply_rotary_emb(x, cos, sin):
    # x: [B, T, H, D]
    # cos, sin: [1, T, 1, D/2]
    # We need to broadcast cos/sin to [B, T, H, D/2]
    # x is split into x1, x2 of shape [B, T, H, D/2]
    
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    
    # Ensure cos/sin have compatible shapes for broadcasting
    # If cos is [1, T, 1, D/2], it should broadcast fine against [B, T, H, D/2]
    # But let's be explicit if needed.
    
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concatenate([y1, y2], axis=-1)

