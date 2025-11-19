"""
Muon optimizer (JAX/Optax port)
"""

import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple, Optional

class MuonState(NamedTuple):
    momentum: optax.Updates

def newton_schulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    # G shape: [rows, cols]
    # We work in bfloat16 or float32. JAX handles types automatically mostly.
    
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G
    
    # Ensure spectral norm is at most 1
    # In PyTorch: X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # We use frobenius norm as proxy or just spectral norm?
    # The PyTorch code uses .norm() which is Frobenius by default for matrices?
    # Wait, PyTorch .norm(dim=(-2,-1)) on a 2D tensor is Frobenius.
    # But spectral norm is what's usually required for NS convergence.
    # However, the PyTorch code says "Ensure spectral norm is at most 1" but uses .norm().
    # Let's stick to the PyTorch implementation: Frobenius norm.
    
    norm = jnp.linalg.norm(X, axis=(-2, -1), keepdims=True)
    X = X / (norm + 1e-7)
    
    if G.shape[-2] > G.shape[-1]:
        X = X.T
        
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    if G.shape[-2] > G.shape[-1]:
        X = X.T
        
    return X

def muon(learning_rate: float, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5):
    """
    Muon optimizer transformation.
    """
    
    def init_fn(params):
        return MuonState(momentum=jax.tree_util.tree_map(jnp.zeros_like, params))

    def update_fn(updates, state, params=None):
        # updates are gradients (g)
        
        # Update momentum
        mu = jax.tree_util.tree_map(
            lambda m, g: m * momentum + g * (1 - momentum),
            state.momentum, updates
        )
        
        # Nesterov
        if nesterov:
            v = jax.tree_util.tree_map(
                lambda m, g: g * momentum + m * (1 - momentum), # Wait, this is not standard Nesterov?
                # PyTorch code:
                # buf.lerp_(g, 1 - momentum)  => buf = buf*momentum + g*(1-momentum)
                # g = g.lerp_(buf, momentum)  => g = g*(1-momentum) + buf*momentum
                # Yes, that matches.
                mu, updates
            )
        else:
            v = mu
            
        # Orthogonalization via Newton-Schulz
        def orthogonalize(g, p):
            if g.ndim != 2:
                return g # Should not happen if we filter correctly
            
            # NS5
            g_orth = newton_schulz5(g, steps=ns_steps)
            
            # Scale
            # scale = (max(1.0, p.size(-2) / p.size(-1)) ** 0.5)
            rows, cols = p.shape[-2], p.shape[-1]
            scale = max(1.0, rows / cols) ** 0.5
            
            return g_orth * scale

        # Apply orthogonalization
        # We need params to get shapes for scaling
        if params is None:
            raise ValueError("Muon requires params to be passed to update_fn")
            
        updates = jax.tree_util.tree_map(orthogonalize, v, params)
        
        # Apply learning rate
        updates = jax.tree_util.tree_map(lambda u: -learning_rate * u, updates)
        
        return updates, MuonState(momentum=mu)

    return optax.GradientTransformation(init_fn, update_fn)

