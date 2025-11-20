"""
Test script for Tropical Attention integration in JAX GPT.
"""

import jax
import jax.numpy as jnp
import numpy as np
from nanochat.gpt_jax import GPT, GPTConfig


def require(condition, message: str):
    if not bool(condition):
        raise AssertionError(message)

def test_tropical_forward():
    print("Testing Tropical Attention Forward Pass...")
    
    # Configure with Tropical Attention
    config = GPTConfig(
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        sequence_len=32,
        use_tropical=True
    )
    
    model = GPT(config)
    rng = jax.random.PRNGKey(0)
    
    # Dummy input
    x = jnp.ones((1, 32), dtype=jnp.int32)
    
    # Init
    print("Initializing model...")
    variables = model.init(rng, x, train=False)
    params = variables['params']
    
    # Forward
    print("Running forward pass...")
    logits = model.apply({'params': params}, x, train=False)
    
    print(f"Logits shape: {logits.shape}")
    require(logits.shape == (1, 32, config.vocab_size), "Unexpected logits shape")
    require(not jnp.isnan(logits).any(), "NaNs found in logits")
    
    print("Tropical Attention Forward Pass Successful!")

def test_standard_forward():
    print("\nTesting Standard Attention Forward Pass (Baseline)...")
    
    # Configure with Standard Attention
    config = GPTConfig(
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        sequence_len=32,
        use_tropical=False
    )
    
    model = GPT(config)
    rng = jax.random.PRNGKey(0)
    
    # Dummy input
    x = jnp.ones((1, 32), dtype=jnp.int32)
    
    # Init
    variables = model.init(rng, x, train=False)
    params = variables['params']
    
    # Forward
    logits = model.apply({'params': params}, x, train=False)
    
    print(f"Logits shape: {logits.shape}")
    require(logits.shape == (1, 32, config.vocab_size), "Unexpected logits shape")
    
    print("Standard Attention Forward Pass Successful!")

if __name__ == "__main__":
    test_tropical_forward()
    test_standard_forward()
