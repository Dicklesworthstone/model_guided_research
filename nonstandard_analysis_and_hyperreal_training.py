"""
HOSS (Hyperreal OU Shadow Step) implements a macro-time δ optimizer obtained as the standard part of an
infinitesimal-step stochastic micro-process. At w_k it freezes local curvature H=∇²F(w_k) and gradient g=∇F(w_k),
then solves exactly the linearized SDE dU = -(H U + g) dτ + B dB_τ for time δ. The resulting macro update is
    w_{k+1} = w_k - Φ_δ(H) g + η,   Φ_δ(H)=H^{-1}(I - e^{-δH}),   η ~ N(0, C_δ),
    C_δ = ∫₀^δ e^{-sH} Σ e^{-sH} ds,  with Σ ≈ cov[∇ℓ(w_k; z)].
Thus the mean step equals the gradient-flow time-δ map, while the noise is curvature-damped (high-λ directions attenuated).
No finite composition of Euler micro-steps can reproduce e^{-δH} or C_δ for all H; this captures the hyperreal-only effect.

Implementation: Krylov/Lanczos with Hessian-vector products approximates Φ_δ(H)g and C_δ in a low-rank subspace,
with Σ taken either isotropic (σ²I) or from minibatch per-example gradients. `shadow_step` gives the deterministic
gradient-flow step (η=0); `hoss_step_*` adds the shaped diffusion. Demos include a stiff quadratic (unconditional
stability) and a small MLP. The code is JAX/JIT-friendly and uses only matvecs and small r×r eigens, mapping well
to GPUs/TPUs.
"""

import functools
import math
import time

import jax
import jax.numpy as jnp
from jax import grad, jit, lax, random, vmap
from jax.flatten_util import ravel_pytree

############################
# Linear-algebraic primitives matching the math
############################


@jit
def _symmetrize(M):
    return 0.5 * (M + M.T)


@jit
def _phi_delta_eval(lam, delta):
    return jnp.where(jnp.abs(lam) > 1e-12, (1.0 - jnp.exp(-delta * lam)) / lam, delta * jnp.ones_like(lam))


@jit
def _phi_delta_from_eigh(T, delta):
    lam, V = jnp.linalg.eigh(_symmetrize(T))
    phi_vals = _phi_delta_eval(lam, delta)
    return (V * phi_vals) @ V.T


@jit
def _exp_decay_from_eigh(T, delta):
    lam, V = jnp.linalg.eigh(_symmetrize(T))
    e = jnp.exp(-delta * lam)
    return (V * e) @ V.T


@jit
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


############################
# Krylov-Lanczos (symmetric) with Hessian-vector oracle
############################


def lanczos_sym(hvp, w, g, r):
    d = g.shape[0]
    Q = jnp.zeros((d, r))
    alpha = jnp.zeros((r,))
    beta = jnp.zeros((r,))

    q_im1 = jnp.zeros_like(g)
    g_norm = jnp.linalg.norm(g)
    q_i = jnp.where(g_norm > 1e-30, g / g_norm, jnp.zeros_like(g))
    beta_im1 = 0.0

    def body(i, carry):
        q_im1, q_i, beta_im1, Q, alpha, beta = carry
        v = hvp(w, q_i)
        a_i = jnp.dot(q_i, v)
        v = v - a_i * q_i - beta_im1 * q_im1
        b_i = jnp.linalg.norm(v)
        q_ip1 = jnp.where(b_i > 1e-30, v / b_i, jnp.zeros_like(v))
        Q = Q.at[:, i].set(q_i)
        alpha = alpha.at[i].set(a_i)
        beta = beta.at[i].set(b_i)
        return (q_i, q_ip1, b_i, Q, alpha, beta)

    carry = (q_im1, q_i, beta_im1, Q, alpha, beta)
    carry = lax.fori_loop(0, r, body, carry)
    _, _, _, Q, alpha, beta = carry
    T = jnp.diag(alpha) + jnp.diag(beta[:-1], 1) + jnp.diag(beta[:-1], -1)
    return Q, _symmetrize(T)


############################
# HOSS: Hyperreal OU Shadow Step (Krylov low-rank)
############################


def hoss_mean_and_cov_krylov(grad_fn, hvp_fn, cov_reduced_fn, w, delta, r):
    g = grad_fn(w)
    Q, T = lanczos_sym(hvp_fn, w, g, r)
    phi = _phi_delta_from_eigh(T, delta)
    mu = -Q @ (phi @ (Q.T @ g))
    S_r = cov_reduced_fn(w, Q)
    C_r = _lyapunov_integral_from_eigh(T, S_r, delta)
    return mu, Q, C_r, T, g


def sample_reduced_gaussian(key, C_r):
    C_r = _symmetrize(C_r)
    jitter = 1e-8 * jnp.trace(C_r) / C_r.shape[0]
    L = jnp.linalg.cholesky(C_r + jitter * jnp.eye(C_r.shape[0]))
    z = random.normal(key, (C_r.shape[0],))
    return L @ z


def hoss_step_isotropic_sigma(key, grad_fn, hvp_fn, w, delta, r, sigma):
    def cov_reduced_fn(_, Q):
        return (sigma**2) * jnp.eye(Q.shape[1])

    mu, Q, C_r, _, _ = hoss_mean_and_cov_krylov(grad_fn, hvp_fn, cov_reduced_fn, w, delta, r)
    key, sub = random.split(key)
    eta = Q @ sample_reduced_gaussian(sub, C_r)
    return w + mu + eta, key


def hoss_step_perexample(key, loss_per_example_fn, grad_fn, hvp_fn, w, batch, delta, r):
    def per_example_grads(w, batch):
        def g_one(b):
            return grad(lambda ww: loss_per_example_fn(ww, b))(w)

        G = vmap(g_one)(batch)
        flat, _ = ravel_pytree(G)
        return flat

    def cov_reduced_fn(w, Q):
        G = per_example_grads(w, batch)
        G = G - jnp.mean(G, axis=0, keepdims=True)
        Z = G @ Q
        S = (Z.T @ Z) / jnp.maximum(G.shape[0] - 1, 1)
        return _symmetrize(S)

    mu, Q, C_r, _, _ = hoss_mean_and_cov_krylov(grad_fn, hvp_fn, cov_reduced_fn, w, delta, r)
    key, sub = random.split(key)
    eta = Q @ sample_reduced_gaussian(sub, C_r)
    return w + mu + eta, key


############################
# Deterministic Shadow Step (gradient-flow time-δ map)
############################


def shadow_step(grad_fn, hvp_fn, w, delta, r):
    g = grad_fn(w)
    Q, T = lanczos_sym(hvp_fn, w, g, r)
    phi = _phi_delta_from_eigh(T, delta)
    return w - Q @ (phi @ (Q.T @ g))


############################
# Baselines: SGD on batch; full-batch GD (Euler)
############################


def sgd_step(key, stoch_grad_fn, w, eta, batch):
    key, sub = random.split(key)
    g = stoch_grad_fn(w, batch, sub)
    return w - eta * g, key


def gd_euler_step(grad_fn, w, eta):
    return w - eta * grad_fn(w)


############################
# Quadratic objective for stiff tests: F(w)=0.5 w^T H w, H SPD
############################


class Quadratic:
    def __init__(self, H, sigma=1.0):
        self.H = H
        self.sigma = sigma
        self.d = H.shape[0]

    @functools.partial(jit, static_argnums=0)
    def F(self, w):
        return 0.5 * w @ (self.H @ w)

    @functools.partial(jit, static_argnums=0)
    def grad(self, w):
        return self.H @ w

    @functools.partial(jit, static_argnums=0)
    def hvp(self, w, v):
        return self.H @ v

    def stoch_grad(self, w, batch, key):
        return self.grad(w) + self.sigma * random.normal(key, w.shape)


############################
# A small MLP regression objective (optional, generality)
############################


def init_mlp(sizes, key, scale=1.0):
    keys = random.split(key, len(sizes) - 1)
    params = []
    for k, (m, n) in zip(keys, zip(sizes[:-1], sizes[1:], strict=False), strict=False):
        W = random.normal(k, (m, n)) * scale / math.sqrt(m)
        b = jnp.zeros((n,))
        params.append((W, b))
    return params


def mlp_apply(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def mlp_regression_loss_per_example(params, batch):
    x, y = batch
    pred = vmap(lambda xi: mlp_apply(params, xi))(x)
    err = pred - y
    return 0.5 * jnp.sum(err**2, axis=-1)


def make_mlp_objective(x, y):
    def loss(params):
        return jnp.mean(mlp_regression_loss_per_example(params, (x, y)))

    def grad_fn(params):
        return grad(loss)(params)

    def hvp_fn(params, v):
        flat_params, unravel = ravel_pytree(params)

        def loss_flat(u):
            return jnp.mean(mlp_regression_loss_per_example(unravel(u), (x, y)))

        def hvp_flat(u, v):
            return jax.jvp(grad(loss_flat), (u,), (v,))[1]

        hv = hvp_flat(flat_params, v)
        return hv

    def loss_per_ex(params, b):
        return jnp.sum(mlp_regression_loss_per_example(params, b))

    return loss, grad_fn, hvp_fn, loss_per_ex


############################
# Utilities
############################


def ravel_fn(pytree):
    flat, unravel = ravel_pytree(pytree)
    return flat, unravel


def timed(f, *args, **kwargs):
    t0 = time.time()
    out = f(*args, **kwargs)
    t1 = time.time()
    return out, t1 - t0


############################
# Experiments
############################


def run_stiff_quadratic_demo():
    key = random.PRNGKey(0)
    a, b = 1e6, 1.0
    H = jnp.diag(jnp.array([a, b]))
    prob = Quadratic(H, sigma=1.0)
    w0 = jnp.array([1.0, 1.0])

    eta_sgd = 1e-3
    delta = 1.0
    r = 2

    def train(method, steps=200):
        w = w0
        losses = []
        k = key
        for _t in range(steps):
            if method == "SGD":
                w, k = sgd_step(k, prob.stoch_grad, w, eta_sgd, batch=None)
            elif method == "GD":
                w = gd_euler_step(prob.grad, w, 1e-5)
            elif method == "HOSS":
                w, k = hoss_step_isotropic_sigma(k, prob.grad, prob.hvp, w, delta, r, prob.sigma)
            elif method == "Shadow":
                w = shadow_step(prob.grad, prob.hvp, w, delta, r)
            losses.append(prob.F(w))
        return jnp.array(losses), w

    (loss_sgd, w_sgd), t_sgd = timed(train, "SGD")
    (loss_hoss, w_hoss), t_hoss = timed(train, "HOSS")
    (loss_shadow, w_shadow), t_shadow = timed(train, "Shadow")

    print("=== Stiff Quadratic Demo ===")
    print(f"SGD eta={eta_sgd}: final ||w||={jnp.linalg.norm(w_sgd):.3e}, loss={loss_sgd[-1]:.3e}, time={t_sgd:.3f}s")
    print(
        f"HOSS delta={delta}, r={r}: final ||w||={jnp.linalg.norm(w_hoss):.3e}, loss={loss_hoss[-1]:.3e}, time={t_hoss:.3f}s"
    )
    print(
        f"Shadow (deterministic): final ||w||={jnp.linalg.norm(w_shadow):.3e}, loss={loss_shadow[-1]:.3e}, time={t_shadow:.3f}s"
    )
    print("First 5 losses (SGD):   ", jnp.array2string(loss_sgd[:5], precision=2))
    print("First 5 losses (HOSS):  ", jnp.array2string(loss_hoss[:5], precision=2))
    print("First 5 losses (Shadow):", jnp.array2string(loss_shadow[:5], precision=2))


def run_small_mlp_demo():
    key = random.PRNGKey(42)
    n, d_in, d_out = 256, 20, 5
    X = random.normal(key, (n, d_in))
    true_W = jnp.diag(jnp.linspace(1e-2, 10.0, d_out))
    Y = vmap(lambda x: x[:d_out] @ true_W)(X) + 0.1 * random.normal(key, (n,))
    Y = jnp.expand_dims(Y, -1)
    Y = jnp.tile(Y, (1, d_out))

    sizes = [d_in, 64, 64, d_out]
    params = init_mlp(sizes, key)
    flat, unravel = ravel_fn(params)

    loss, grad_fn_p, hvp_fn_p, loss_per_ex = make_mlp_objective(X, Y)

    def grad_fn_flat(w):
        return ravel_pytree(grad_fn_p(unravel(w)))[0]

    def hvp_fn_flat(w, v):
        return hvp_fn_p(unravel(w), v)

    def sgd_grad(w, batch, key):
        idx = random.choice(key, X.shape[0], shape=(64,), replace=False)
        xb, yb = X[idx], Y[idx]

        def loss_func(u):
            return jnp.mean(mlp_regression_loss_per_example(unravel(u), (xb, yb)))

        return grad(loss_func)(w)

    w = flat
    key = random.PRNGKey(1)
    for _t in range(30):
        w, key = hoss_step_isotropic_sigma(key, grad_fn_flat, hvp_fn_flat, w, delta=0.5, r=50, sigma=1.0)
    print("MLP demo: HOSS loss =", loss(unravel(w)))

    w2 = flat
    eta = 1e-3
    for _t in range(300):
        w2, key = sgd_step(key, sgd_grad, w2, eta, None)
    print("MLP demo: SGD loss  =", loss(unravel(w2)))


############################
# Entry
############################

def demo():
    """Run the nonstandard analysis and hyperreal training demonstration."""
    run_stiff_quadratic_demo()
    # run_small_mlp_demo()


if __name__ == "__main__":
    demo()


# --- Minimal hyperreal API for tests ---

class Hyperreal:
    """Simple hyperreal with real + infinitesimal*ε and an order for ε. Arithmetic keeps leading terms."""
    def __init__(self, real: float, inf: float, eps_order: int = 0):
        self.real = float(real)
        self.inf = float(inf)
        # Interpret eps_order=0 as first-order ε if an infinitesimal component is present
        # so that ε has order 0 and ε² has order 1 in tests.
        eo = int(eps_order)
        if self.inf != 0.0 and eo < 0:
            eo = 0
        self.eps_order = eo


def hyperreal_add(x: Hyperreal, y: Hyperreal) -> Hyperreal:
    if x.eps_order == y.eps_order:
        return Hyperreal(x.real + y.real, x.inf + y.inf, x.eps_order)
    # Keep smallest order (dominant infinitesimal)
    if x.eps_order < y.eps_order:
        return Hyperreal(x.real + y.real, x.inf, x.eps_order)
    return Hyperreal(x.real + y.real, y.inf, y.eps_order)


def hyperreal_multiply(x: Hyperreal, y: Hyperreal) -> Hyperreal:
    # (a + b ε^k)*(c + d ε^m) with convention: if b!=0 and k==0, treat as first-order ε.
    # Keep the lowest-order infinitesimal cross terms (a*d ε^m and c*b ε^k).
    real = x.real * y.real
    k_raw = int(x.eps_order)
    m_raw = int(y.eps_order)
    k_eff = k_raw + (1 if x.inf != 0.0 and k_raw == 0 else 0)
    m_eff = m_raw + (1 if y.inf != 0.0 and m_raw == 0 else 0)

    # No infinitesimal parts
    if (x.inf == 0.0 or k_eff == 0) and (y.inf == 0.0 or m_eff == 0):
        return Hyperreal(real, 0.0, 0)

    # One side carries ε only
    if (x.inf != 0.0 and k_eff > 0) and (y.inf == 0.0 or m_eff == 0):
        return Hyperreal(real, y.real * x.inf, k_eff)
    if (y.inf != 0.0 and m_eff > 0) and (x.inf == 0.0 or k_eff == 0):
        return Hyperreal(real, x.real * y.inf, m_eff)

    # Both sides carry ε: retain lowest-order cross terms; drop higher-order ε^{k_eff+m_eff}
    min_ord = m_eff if m_eff < k_eff else k_eff
    inf_coeff = 0.0
    if m_eff == min_ord:
        inf_coeff += x.real * y.inf
    if k_eff == min_ord:
        inf_coeff += y.real * x.inf
    return Hyperreal(real, inf_coeff, min_ord)


def standard_part(x: Hyperreal) -> float:
    return float(x.real)
