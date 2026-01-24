"""
Ultrametric Attention Module (JAX/Flax)

Implements an LCP-kernel ultrametric attention mechanism:
- Map queries/keys to K learned base-p "digits" (continuous relaxation).
- Approximate LCP(q,k) via prefix products of per-digit match probabilities.
- Use the ultrametric similarity kernel w(q,k) ∝ alpha^{LCP(q,k)} to compute a
  causal weighted average of values (no Euclidean dot-product, no softmax).
"""

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax

from nanochat.common_jax import GPTConfig, apply_rotary_emb, rms_norm

# --- Ultrametric core config ---


@dataclass
class UltrametricConfig:
    p: int = 2  # base for p-adic numbers (e.g., 2 for binary)
    K: int = 8  # number of digits / LCP depth
    alpha: float = 2.0  # kernel base (>1): similarity grows with LCP depth
    lcp_beta: float = 32.0  # sharpness for per-digit matching

    def __post_init__(self) -> None:
        if self.p < 2:
            raise ValueError(f"UltrametricConfig.p must be >= 2, got {self.p}")
        if self.K <= 0:
            raise ValueError(f"UltrametricConfig.K must be positive, got {self.K}")
        if not (self.alpha > 1.0 and math.isfinite(self.alpha)):
            raise ValueError(f"UltrametricConfig.alpha must be finite and > 1, got {self.alpha}")
        if not (self.lcp_beta > 0.0 and math.isfinite(self.lcp_beta)):
            raise ValueError(f"UltrametricConfig.lcp_beta must be finite and > 0, got {self.lcp_beta}")


def _soft_lcp_depth(q_digits: jnp.ndarray, k_digits: jnp.ndarray, *, lcp_beta: float) -> jnp.ndarray:
    """
    Continuous proxy for LCP(q,k) computed from per-depth match probabilities.

    q_digits: (B, H, Tq, K)
    k_digits: (B, H, Tk, K)
    returns:  (B, H, Tq, Tk) in [0, K]
    """
    diff = jnp.abs(q_digits[:, :, :, None, :] - k_digits[:, :, None, :, :])  # (B, H, Tq, Tk, K)
    match_prob = jnp.exp(-lcp_beta * jnp.square(diff))
    prefix_prob = jnp.cumprod(match_prob, axis=-1)
    return jnp.sum(prefix_prob, axis=-1)


class UltrametricCausalSelfAttention(nn.Module):
    config: GPTConfig
    layer_idx: int
    ult_config: UltrametricConfig = field(default_factory=UltrametricConfig)  # Default ultrametric settings

    def setup(self):
        self.n_head = self.config.n_head
        self.n_kv_head = self.config.n_kv_head
        self.n_embd = self.config.n_embd
        self.head_dim = self.n_embd // self.n_head

        # Projections for Q, K, V
        self.c_q = nn.Dense(
            self.n_head * self.head_dim, use_bias=False, kernel_init=nn.initializers.normal(stddev=0.02)
        )
        self.c_k = nn.Dense(
            self.n_kv_head * self.head_dim, use_bias=False, kernel_init=nn.initializers.normal(stddev=0.02)
        )
        self.c_v = nn.Dense(
            self.n_kv_head * self.head_dim, use_bias=False, kernel_init=nn.initializers.normal(stddev=0.02)
        )
        self.c_proj = nn.Dense(self.n_embd, use_bias=False, kernel_init=nn.initializers.normal(stddev=0.02))

        # Learned projections into K "digits" per head used to compute an ultrametric LCP-kernel.
        self.to_digits_q = nn.Dense(self.ult_config.K, use_bias=False)
        self.to_digits_k = nn.Dense(self.ult_config.K, use_bias=False)

    def __call__(self, x, cos, sin, mask=None):
        B, T, C = x.shape

        # Project to Q, K, V
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # QK Norm
        q = rms_norm(q)
        k = rms_norm(k)

        init_cache = self.config.init_cache

        # KV Cache handling (mirrors nanochat.gpt_jax.CausalSelfAttention / tropical_attention)
        if self.has_variable("cache", "cached_key"):
            cached_key = self.variable(
                "cache",
                "cached_key",
                jnp.zeros,
                (B, self.config.sequence_len, self.n_kv_head, self.head_dim),
                k.dtype,
            )
            cached_val = self.variable(
                "cache",
                "cached_val",
                jnp.zeros,
                (B, self.config.sequence_len, self.n_kv_head, self.head_dim),
                v.dtype,
            )
            cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

            idx = cache_index.value
            k_cache = lax.dynamic_update_slice(cached_key.value, k, (0, idx, 0, 0))
            v_cache = lax.dynamic_update_slice(cached_val.value, v, (0, idx, 0, 0))

            cached_key.value = k_cache
            cached_val.value = v_cache
            cache_index.value = idx + T

            k = k_cache
            v = v_cache

            if mask is None:
                total_len = self.config.sequence_len
                query_idx = jnp.arange(T) + idx
                key_idx = jnp.arange(total_len)
                # (1, 1, T, total_len) boolean keep-mask
                mask = key_idx[None, None, None, :] <= query_idx[None, None, :, None]

        elif init_cache:
            self.variable(
                "cache", "cached_key", jnp.zeros, (B, self.config.sequence_len, self.n_kv_head, self.head_dim), k.dtype
            )
            self.variable(
                "cache", "cached_val", jnp.zeros, (B, self.config.sequence_len, self.n_kv_head, self.head_dim), v.dtype
            )
            self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

            if T > 0:
                cached_key = self.variable("cache", "cached_key")
                cached_val = self.variable("cache", "cached_val")
                cache_index = self.variable("cache", "cache_index")

                cached_key.value = lax.dynamic_update_slice(cached_key.value, k, (0, 0, 0, 0))
                cached_val.value = lax.dynamic_update_slice(cached_val.value, v, (0, 0, 0, 0))
                cache_index.value = T

        # GQA: Repeat KV heads if needed
        if self.n_kv_head != self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = jnp.repeat(k, n_rep, axis=2)  # [B, Tk, H, D]
            v = jnp.repeat(v, n_rep, axis=2)

        # Transpose to [B, H, T, D]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        if mask is None:
            mask = nn.make_causal_mask(jnp.ones((B, T), dtype=jnp.int32))

        # Support both boolean masks (Flax) and additive masks (0 / -inf).
        if mask.dtype != jnp.bool_:
            keep = mask > -1e8
        else:
            keep = mask

        # Digits in [0, p-1] (continuous relaxation).
        p = float(self.ult_config.p)
        q_digits = jax.nn.sigmoid(self.to_digits_q(q).astype(jnp.float32)) * (p - 1.0)
        k_digits = jax.nn.sigmoid(self.to_digits_k(k).astype(jnp.float32)) * (p - 1.0)

        lcp = _soft_lcp_depth(q_digits, k_digits, lcp_beta=float(self.ult_config.lcp_beta))

        log_alpha = jnp.log(jnp.asarray(self.ult_config.alpha, dtype=jnp.float32))
        weights = jnp.exp(lcp.astype(jnp.float32) * log_alpha)
        weights = jnp.where(keep, weights, 0.0)

        denom = jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-9)
        attn = weights / denom

        y = jnp.einsum("bhqk,bhkd->bhqd", attn.astype(v.dtype), v)

        # Re-assemble
        y = jnp.transpose(y, (0, 2, 1, 3)).reshape(B, T, C)
        y = self.c_proj(y)
        return y
