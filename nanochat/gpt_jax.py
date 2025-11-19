"""
GPT model (JAX/Flax port of nanochat) - Optimized
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random, lax

from nanochat.common_jax import GPTConfig, rms_norm, apply_rotary_emb

class CausalSelfAttention(nn.Module):
    config: GPTConfig
    layer_idx: int

    def setup(self):
        self.n_head = self.config.n_head
        self.n_kv_head = self.config.n_kv_head
        self.n_embd = self.config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
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

        # QK Norm
        q = rms_norm(q)
        k = rms_norm(k)
        
        # Scale query
        q = q * (1.0 / jnp.sqrt(self.head_dim))

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
            
            # Use cached k, v
            k = k_cache
            v = v_cache
            
            if mask is None:
                # Create mask for [B, 1, T, Max_Len]
                total_len = self.config.sequence_len
                query_idx = jnp.arange(T) + idx
                key_idx = jnp.arange(total_len)
                
                # [1, 1, T, MaxLen]
                mask = key_idx[None, None, None, :] <= query_idx[None, None, :, None]
                mask = jnp.where(mask, 0, -jnp.inf)

        elif init_cache:
            # Initialize cache variables
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

        # GQA: Repeat KV heads if needed
        if self.n_kv_head != self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = jnp.repeat(k, n_rep, axis=2)
            v = jnp.repeat(v, n_rep, axis=2)

        # Transpose to [B, H, T, D] for attention
        # Note: if using cache, k/v are [B, Max_Len, H, D]. Transpose -> [B, H, Max_Len, D]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Attention
        if mask is None:
             # Default causal mask for training (T x T) or non-cached inference
             mask = nn.make_causal_mask(jnp.ones((B, T), dtype=jnp.int32))

        y = nn.dot_product_attention(q, k, v, mask=mask)

        # Re-assemble
        y = jnp.transpose(y, (0, 2, 1, 3)).reshape(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    config: GPTConfig

    def setup(self):
        self.c_fc = nn.Dense(4 * self.config.n_embd, use_bias=False, kernel_init=nn.initializers.normal(stddev=0.02))
        self.c_proj = nn.Dense(self.config.n_embd, use_bias=False, kernel_init=nn.initializers.normal(stddev=0.02))

    def __call__(self, x):
        x = self.c_fc(x)
        x = jnp.square(jax.nn.relu(x)) # relu^2
        x = self.c_proj(x)
        return x

from nanochat.tropical_attention import TropicalCausalSelfAttention

class Block(nn.Module):
    config: GPTConfig
    layer_idx: int

    def setup(self):
        if self.config.use_tropical:
            self.attn = TropicalCausalSelfAttention(self.config, self.layer_idx)
        else:
            self.attn = CausalSelfAttention(self.config, self.layer_idx)
        self.mlp = MLP(self.config)

    def __call__(self, x, cos, sin, mask=None, init_cache=False):
        x_norm = rms_norm(x)
        x = x + self.attn(x_norm, cos, sin, mask, init_cache=init_cache)
        x_norm = rms_norm(x)
        x = x + self.mlp(x_norm)
        return x

class GPT(nn.Module):
    config: GPTConfig

    def setup(self):
        self.wte = nn.Embed(self.config.vocab_size, self.config.n_embd, embedding_init=nn.initializers.normal(stddev=0.02))
        self.blocks = [Block(self.config, i) for i in range(self.config.n_layer)]
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False, kernel_init=nn.initializers.normal(stddev=0.02))

    def __call__(self, idx, targets=None, train=True, init_cache=False):
        B, T = idx.shape
        
        # Determine start position for rotary embeddings
        start_pos = 0
        if self.has_variable('cache', 'cache_index'):
            start_pos = self.variable('cache', 'cache_index').value
            # If init_cache is True, we are resetting/starting, so start_pos is 0 (or we ignore the variable)
            if init_cache:
                start_pos = 0
        
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self.precompute_rotary_embeddings(T, head_dim, start_pos=start_pos)
        
        x = self.wte(idx)
        x = rms_norm(x)
        
        mask = None
        if train:
            mask = nn.make_causal_mask(jnp.ones((B, T), dtype=jnp.int32))
        
        for block in self.blocks:
            x = block(x, cos, sin, mask, init_cache=init_cache)
            
        x = rms_norm(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            softcap = 15.0
            logits = softcap * jnp.tanh(logits / softcap)
            return logits
        else:
            logits = self.lm_head(x)
            softcap = 15.0
            logits = softcap * jnp.tanh(logits / softcap)
            return logits

    def precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, start_pos=0):
        channel_range = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = jnp.arange(start_pos, start_pos + seq_len, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        return cos, sin

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, rng=None):
        """
        Autoregressive generation with KV caching.
        Must be called via apply(..., method=GPT.generate, mutable=['cache'])
        
        Uses a Python loop for flexibility and correctness with Flax variables.
        For high performance, compile the step function externally.
        """
        if rng is None:
            rng = random.PRNGKey(0)
            
        B, T = idx.shape
        assert B == 1, "Batch size 1 for now"
        
        # 1. Prefill
        logits = self.__call__(idx, init_cache=True, train=False)
        
        # Sample first token
        last_logit = logits[:, -1, :]
        rng, key = random.split(rng)
        
        if temperature > 0:
            scaled_logits = last_logit / temperature
            # Top-k
            if top_k is not None:
                vals, _ = lax.top_k(scaled_logits, top_k)
                min_val = vals[:, -1]
                scaled_logits = jnp.where(scaled_logits < min_val[:, None], -jnp.inf, scaled_logits)
            
            next_token = random.categorical(key, scaled_logits)
        else:
            next_token = jnp.argmax(last_logit, axis=-1)
            
        next_token = next_token[:, None] # [B, 1]
        
        # Initialize output sequence
        out_seq = jnp.concatenate([idx, next_token], axis=1)
        
        # 2. Decode loop (Python loop)
        curr_token = next_token
        
        for _ in range(max_new_tokens - 1):
            # Run model
            logits = self.__call__(curr_token, train=False)
            last_logit = logits[:, -1, :]
            
            rng, key = random.split(rng)
            
            if temperature > 0:
                scaled_logits = last_logit / temperature
                if top_k is not None:
                    vals, _ = lax.top_k(scaled_logits, top_k)
                    min_val = vals[:, -1]
                    scaled_logits = jnp.where(scaled_logits < min_val[:, None], -jnp.inf, scaled_logits)
                next_token = random.categorical(key, scaled_logits)
            else:
                next_token = jnp.argmax(last_logit, axis=-1)
                
            next_token = next_token[:, None]
            
            # Update sequence
            out_seq = jnp.concatenate([out_seq, next_token], axis=1)
            curr_token = next_token
            
        return out_seq
