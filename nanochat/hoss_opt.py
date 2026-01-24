"""
HOSS (Hyperreal OU Shadow Step) optimizer for Optax.

Implements a custom Optax `GradientTransformation` that performs a macro-time δ update by
analytically solving a linearized OU SDE with curvature-damped noise. Curvature is
approximated via a symmetric Lanczos iteration using Hessian-vector products.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from jax import grad, lax, random
from jax.flatten_util import ravel_pytree


def _symmetrize(M: jax.Array) -> jax.Array:
    return jnp.float32(0.5) * (M + M.T)


def _phi_delta_fraction(lam: jax.Array, delta: jax.Array) -> jax.Array:
    # (1 - exp(-delta * lam)) / lam with a stable small-lambda fallback.
    mask = jnp.abs(lam) > jnp.float32(1e-12)
    safe = (jnp.float32(1.0) - jnp.exp(-delta * lam)) / lam
    return jnp.where(mask, safe, delta * jnp.ones_like(lam))


def _lyapunov_integral_from_eigh(T: jax.Array, S: jax.Array, delta: jax.Array) -> jax.Array:
    # Computes ∫_0^δ exp(-sT) S exp(-sT) ds via eigendecomposition of T.
    lam, V = jnp.linalg.eigh(_symmetrize(T))
    S_hat = V.T @ S @ V
    D = lam[:, None] + lam[None, :]

    mask = jnp.abs(D) > jnp.float32(1e-12)
    num = jnp.float32(1.0) - jnp.exp(-delta * D)
    frac = jnp.where(mask, num / D, delta * jnp.ones_like(D))
    C_hat = S_hat * frac
    C = V @ C_hat @ V.T
    return _symmetrize(C)


def lanczos_sym(
    hvp: Callable[[jax.Array], jax.Array],
    vec_g: jax.Array,
    r: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Symmetric Lanczos iteration to approximate the Hessian in a Krylov basis."""
    vec_g_f32 = vec_g.astype(jnp.float32)
    g_norm = jnp.linalg.norm(vec_g_f32)
    q0 = jnp.where(g_norm > jnp.float32(1e-30), vec_g_f32 / g_norm, jnp.zeros_like(vec_g_f32))

    d = vec_g_f32.shape[0]
    Q0 = jnp.zeros((d, r), dtype=jnp.float32)
    alpha0 = jnp.zeros((r,), dtype=jnp.float32)
    beta0 = jnp.zeros((r,), dtype=jnp.float32)
    q_prev0 = jnp.zeros_like(q0)
    beta_prev0 = jnp.float32(0.0)

    def body(i, carry):
        q_prev, q, beta_prev, Q, alpha, beta = carry
        v = hvp(q).astype(jnp.float32)
        a_i = jnp.dot(q, v)
        v = v - a_i * q - beta_prev * q_prev
        b_i = jnp.linalg.norm(v)
        q_next = jnp.where(b_i > jnp.float32(1e-30), v / b_i, jnp.zeros_like(v))

        Q = Q.at[:, i].set(q)
        alpha = alpha.at[i].set(a_i)
        beta = beta.at[i].set(b_i)
        return (q, q_next, b_i, Q, alpha, beta)

    q_prev, _q_last, _beta_last, Q, alpha, beta = lax.fori_loop(
        0, r, body, (q_prev0, q0, beta_prev0, Q0, alpha0, beta0)
    )
    del q_prev, _q_last, _beta_last

    T = jnp.diag(alpha) + jnp.diag(beta[:-1], k=1) + jnp.diag(beta[:-1], k=-1)
    return Q, T, g_norm


@dataclass(frozen=True)
class HossState:
    rng_key: jax.Array


jax.tree_util.register_pytree_node(
    HossState,
    lambda node: ((node.rng_key,), None),
    lambda _, children: HossState(rng_key=children[0]),
)


def hoss(
    learning_rate: float,
    *,
    lanczos_rank: int = 10,
    noise_scale: float = 1.0,
    isotropic_noise_var: float = 1e-4,
    min_curvature: float = 1e-6,
    gradient_norm_clip: float | None = 1.0,
    jitter: float = 1e-6,
) -> optax.GradientTransformation:
    """HOSS optimizer.

    Notes:
    - `learning_rate` is interpreted as the macro time step δ.
    - Callers must pass `loss_fn(params)` to `tx.update(..., loss_fn=loss_fn)` so HVPs can be computed.
    """
    delta = jnp.float32(learning_rate)
    r = int(lanczos_rank)

    def init_fn(params):
        del params
        return HossState(rng_key=random.PRNGKey(0))

    def update_fn(grads, state: HossState, params=None, **kwargs):
        if params is None:
            raise ValueError("HOSS optimizer requires `params` for HVP computation.")
        loss_fn = kwargs.get("loss_fn")
        if loss_fn is None:
            raise ValueError("HOSS optimizer requires `loss_fn(params)` passed via `tx.update(..., loss_fn=...)`.")

        params_flat, unravel_params = ravel_pytree(params)
        grads_flat, _ = ravel_pytree(grads)
        params_flat_f32 = params_flat.astype(jnp.float32)
        grads_flat_f32 = grads_flat.astype(jnp.float32)

        if gradient_norm_clip is not None:
            g_norm = jnp.linalg.norm(grads_flat_f32)
            scale = jnp.minimum(jnp.float32(1.0), jnp.float32(gradient_norm_clip) / (g_norm + jnp.float32(1e-12)))
            grads_flat_f32 = grads_flat_f32 * scale

        def loss_flat_fn(p_flat):
            return loss_fn(unravel_params(p_flat))

        grad_loss_flat_fn = grad(loss_flat_fn)

        def hvp_fn(v):
            v = v.astype(jnp.float32)
            return jax.jvp(grad_loss_flat_fn, (params_flat_f32,), (v,))[1].astype(jnp.float32)

        Q, T, g_norm_lanczos = lanczos_sym(hvp_fn, grads_flat_f32, r)

        lam_T, V_T = jnp.linalg.eigh(_symmetrize(T))
        lam_T = jnp.maximum(lam_T, jnp.float32(min_curvature))
        phi_delta_T = (V_T * _phi_delta_fraction(lam_T, delta)) @ V_T.T

        projected_grad = jnp.zeros((r,), dtype=jnp.float32).at[0].set(g_norm_lanczos)
        mean_update_projected = -phi_delta_T @ projected_grad
        mean_update_flat = Q @ mean_update_projected

        S_hat = jnp.float32(isotropic_noise_var) * jnp.eye(r, dtype=jnp.float32)
        C_delta_T = _lyapunov_integral_from_eigh(T.astype(jnp.float32), S_hat, delta)
        C_delta_T = C_delta_T + jnp.float32(jitter) * jnp.eye(r, dtype=jnp.float32)

        rng_key, noise_key = random.split(state.rng_key)
        noise_projected = random.multivariate_normal(
            noise_key,
            jnp.zeros((r,), dtype=jnp.float32),
            C_delta_T,
            method="svd",
        )
        noise_flat = jnp.float32(noise_scale) * (Q @ noise_projected)

        updates_flat = mean_update_flat + noise_flat
        updates_tree_f32 = unravel_params(updates_flat)
        updates = jax.tree_util.tree_map(lambda p, u: u.astype(p.dtype), params, updates_tree_f32)
        return updates, HossState(rng_key=rng_key)

    return optax.GradientTransformation(init_fn, update_fn)
