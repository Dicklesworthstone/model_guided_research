"""
HOSS (Hyperreal OU Shadow Step) optimizer for Optax.

This module implements a custom Optax GradientTransformation for the HOSS optimizer,
which provides a macro-time δ update by analytically solving a linearized SDE
with curvature-damped noise. It leverages Krylov-Lanczos approximations for
Hessian-vector products to remain scalable.
"""

import functools
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from jax import grad, jit, lax, random, vmap
from jax.flatten_util import ravel_pytree


# --- Utility functions adapted from nonstandard_analysis_and_hyperreal_training.py ---

def _symmetrize(M):
    return 0.5 * (M + M.T)


def _phi_delta_fraction(lam, delta):
    return jnp.where(jnp.abs(lam) > 1e-12, (1.0 - jnp.exp(-delta * lam)) / lam, delta * jnp.ones_like(lam))


def _exp_delta_fraction(lam, delta): # new helper for exp(-delta * lam)
    return jnp.exp(-delta * lam)


def _phi_delta_from_eigh(T, delta):
    lam, V = jnp.linalg.eigh(_symmetrize(T))
    phi_vals = _phi_delta_fraction(lam, delta)
    return (V * phi_vals) @ V.T


def _exp_decay_from_eigh(T, delta):
    lam, V = jnp.linalg.eigh(_symmetrize(T))
    e = jnp.exp(-delta * lam)
    return (V * e) @ V.T


def _lyapunov_integral_from_eigh(T, S, delta):
    lam, V = jnp.linalg.eigh(_symmetrize(T))
    S_hat = V.T @ S @ V
    lam_i = lam[:, None]
    lam_j = lam[None, :]
    D = lam_i + lam_j
    num = 1.0 - jnp.exp(-delta * D)
    frac = jnp.where(jnp.abs(D) > 1e-12, num / D, delta * jnp.ones_like(D))
    C_hat = S_hat * frac
    C = V @ C_hat @ V.T
    return _symmetrize(C)


def lanczos_sym(hvp, vec_g, r):
    d = vec_g.shape[0]
    Q = jnp.zeros((d, r))
    alpha = jnp.zeros((r,))
    beta = jnp.zeros((r,))

    q_im1 = jnp.zeros_like(vec_g)
    g_norm = jnp.linalg.norm(vec_g)
    q_i = jnp.where(g_norm > 1e-30, vec_g / g_norm, jnp.zeros_like(vec_g))
    beta_im1 = 0.0

    # Ensure loop iterations are static for JIT compilation
    def body(i, carry):
        q_im1, q_i, beta_im1, Q_inner, alpha_inner, beta_inner = carry
        
        v = hvp(q_i) # hvp now takes only one argument
        a_i = jnp.dot(q_i, v)
        v = v - a_i * q_i - beta_im1 * q_im1
        b_i = jnp.linalg.norm(v)
        q_ip1 = jnp.where(b_i > 1e-30, v / b_i, jnp.zeros_like(v))
        
        Q_inner = Q_inner.at[:, i].set(q_i)
        alpha_inner = alpha_inner.at[i].set(a_i)
        beta_inner = beta_inner.at[i].set(b_i)
        return (q_i, q_ip1, b_i, Q_inner, alpha_inner, beta_inner)

    # Initial value for carry
    q_im1_init = jnp.zeros_like(vec_g)
    q_i_init = q_i
    beta_im1_init = 0.0
    Q_init = jnp.zeros((d, r))
    alpha_init = jnp.zeros((r,))
    beta_init = jnp.zeros((r,))

    # Perform the loop
    q_im1_final, q_i_final, beta_im1_final, Q_final, alpha_final, beta_final = lax.fori_loop(
        0, r, body, (q_im1_init, q_i_init, beta_im1_init, Q_init, alpha_init, beta_init)
    )

    # The last beta needs to be adjusted as the loop runs r times for alpha and beta up to r-1
    # For a square T matrix (r x r), we need beta up to r-1. So beta_final is correct.
    
    # Construct tridiagonal T matrix
    T = jnp.diag(alpha_final) + jnp.diag(beta_final[:-1], k=1) + jnp.diag(beta_final[:-1], k=-1)
    
    return Q_final, T, g_norm # Return Q_final which has all q_i vectors, and T, and the initial g_norm


# --- HOSS Optax Optimizer ---

def hoss(
    learning_rate: float,
    delta: float = 1.0,  # Macro time step
    lanczos_rank: int = 10,
    noise_scale: float = 1.0, # scale for the injected noise
    # We will approximate Sigma with an isotropic noise for now for simplicity
    # A more advanced version would estimate Sigma from minibatches
    isotropic_noise_var: float = 1e-4, 
    min_curvature: float = 1e-6, # min eigenvalue for numerical stability
    gradient_norm_clip: float = 1.0 # Clip gradient norm for stability
) -> optax.GradientTransformation:
    """
    HOSS (Hyperreal OU Shadow Step) optimizer.

    Args:
        learning_rate: Global scaling factor for the mean step (corresponds to delta).
                       Here, `delta` is the actual macro time.
        delta: Macro time step (δ). Corresponds to the physical time duration of the
               micro-process.
        lanczos_rank: Rank of the Krylov subspace for approximating Hessian-vector products.
        noise_scale: Scaling factor for the injected noise.
        isotropic_noise_var: Variance for the isotropic noise approximation (Sigma).
        min_curvature: Minimum curvature for numerical stability when inverting/dividing by eigenvalues.
        gradient_norm_clip: Clips the gradient norm before processing.
    """

    def init_fn(params):
        # HOSS state needs a PRNGKey for sampling noise
        return HossState(rng_key=random.PRNGKey(0))

    def update_fn(grads, state, params, **kwargs):
        # Check if the grads are empty (e.g. some layers are frozen)
        if not grads:
            return grads, state

        # Unflatten params and grads for HVP calculation
        params_flat, unravel_params = ravel_pytree(params)
        grads_flat, _ = ravel_pytree(grads)

        # Apply gradient clipping manually
        if gradient_norm_clip is not None:
            g_norm = jnp.linalg.norm(grads_flat)
            scale = jnp.where(g_norm > gradient_norm_clip, gradient_norm_clip / g_norm, 1.0)
            grads_flat = grads_flat * scale

        # Function to compute loss for HVP
        # Assumes the `apply_fn` for the model is passed via `kwargs`
        # and takes `params` and `batch` (from `kwargs['batch']`) to return logits,
        # and loss is computed externally and then passed to HOSS update_fn.
        # This setup is tricky with Optax's standard API which expects `grads`.
        # A more direct integration might require `train_step` to pass more info.

        # For HOSS, we need the Hessian-vector product of the LOSS with respect to params.
        # Let's assume the `value_and_grad_fn` of the loss is available.
        # Here we need access to the original loss function that Optax does not provide directly.
        # We will assume that `kwargs` provides a `loss_fn(params)` that returns a scalar loss.
        # This is a common pattern for custom optimizers that need second-order info.

        if 'loss_fn' not in kwargs:
            raise ValueError(
                "HOSS optimizer requires `loss_fn(params)` to be passed in `kwargs` "
                "to compute Hessian-vector products."
            )
        loss_fn = kwargs['loss_fn']
        
        # Define HVP function
        def hvp_fn(v):
            # loss_fn expects pytree params, but we are differentiating w.r.t. flat params
            # So we define a wrapper that takes flat params
            def loss_flat_fn(p_flat):
                p_tree = unravel_params(p_flat)
                return loss_fn(p_tree)
            
            return jax.jvp(grad(loss_flat_fn), (params_flat,), (v,))[1] # HVP of the scalar loss w.r.t. params_flat

        # Use Lanczos to get approximate Hessian (Q, T)
        Q, T, g_norm = lanczos_sym(hvp_fn, grads_flat, lanczos_rank) # grads_flat is -g for minimize

        # The input gradient is -g (negative gradient for minimization).
        # HOSS uses g for its mean step.
        grad_true_flat = -grads_flat
        
        # Project gradient into Krylov subspace
        projected_grad_g = Q.T @ grad_true_flat
        
        # Calculate matrix exponential functions using T's eigen decomposition
        # T is (r x r) tridiagonal, its eigh is cheap.
        lam_T, V_T = jnp.linalg.eigh(T)
        
        # Ensure eigenvalues are not too small (for stability if T is not well-conditioned)
        lam_T = jnp.maximum(lam_T, min_curvature)

        phi_delta_T = (V_T * _phi_delta_fraction(lam_T, delta)) @ V_T.T
        exp_delta_T = (V_T * _exp_delta_fraction(lam_T, delta)) @ V_T.T # for C_delta

        # Mean step: -Phi_delta(H_k) g_k, approximated as -Q @ Phi_delta(T) @ Q.T @ g_k
        mean_update_projected = -phi_delta_T @ projected_grad_g
        mean_update_flat = Q @ mean_update_projected

        # Noise component (C_delta is the covariance of the noise)
        # Approximate Sigma with an isotropic matrix scaled by isotropic_noise_var
        # Sigma is assumed to be in the original parameter space.
        # We need projected Sigma into Krylov subspace.
        # For isotropic noise, Q.T @ (sigma^2 * I) @ Q = sigma^2 * I_r
        
        # So S_hat in _lyapunov_integral_from_eigh becomes (isotropic_noise_var * jnp.eye(lanczos_rank))
        S_hat = isotropic_noise_var * jnp.eye(lanczos_rank, dtype=lam_T.dtype) # Use lam_T.dtype for precision

        C_delta_T = _lyapunov_integral_from_eigh(T, S_hat, delta)

        # Sample noise from N(0, C_delta_T) and project back
        rng_key, noise_key = random.split(state.rng_key)
        
        # Sample from N(0, C_delta_T) directly
        # Use multivariate_normal for sampling.
        noise_projected = random.multivariate_normal(noise_key, jnp.zeros(lanczos_rank), C_delta_T)
        
        # Scale noise and project back to original space
        noise_flat = noise_scale * (Q @ noise_projected)
        
        # Total update
        updates_flat = mean_update_flat + noise_flat
        
        # Unflatten updates
        updates = unravel_params(updates_flat)

        # Update the RNG key in the state
        new_state = HossState(rng_key=rng_key)

        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)

# Define HossState for Optax
@dataclass
class HossState:
    rng_key: jax.Array

# Register HossState as a PyTree
jax.tree_util.register_pytree_node(
    HossState,
    lambda node: ((node.rng_key,), None),
    lambda _, children: HossState(rng_key=children[0])
)
