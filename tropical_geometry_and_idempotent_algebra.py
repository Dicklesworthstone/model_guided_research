"""
Tropical Transformer (JAX): a max–plus, idempotent attention stack with route‑wise margin training.

Core algebra:
- Work over the max–plus semiring (⊕,⊗) with a ⊕ b = max(a,b), a ⊗ b = a + b.
- Tropical matrix multiply tmm(A,B) = max_k (A_ik + B_kj).
- Gauge fixing: gauge_time(X) subtracts the columnwise max so each token column has sup 0.

Block (T2):
- Projections: Q = WQ ⊗ X, K = WK ⊗ X, V = WV ⊗ X (row‑anchored init: row max = 0).
- Attention without forming L×L: Z = (V ⊗ Kᵀ) ⊗ Q  (associativity on the semiring).
- Head combine via coordinatewise ⊕; optional residual via ⊕ with X; re‑gauge by column max.
- Pool/logits: p = ⊕_t X[:,t],  y = Wcls ⊗ p.

Loss and training:
- Margin hinge in max–plus order: L = [ m + max_{k≠c} y_k − y_c ]₊.
- Winner‑only, route‑wise updates: extract the active route for y_c and top wrong y_k*, along with
  runner‑up margins at each max node. Safe step η = ½·min(margins on both routes) to avoid argmax flips.
- Apply +η to parameters on the positive route and −η on the negative route (pure add/compare logic).

Interpretability & certs:
- Exact discrete routes (indices at each max) and per‑node runner‑up gaps provide sparse attributions and
  an ℓ∞ robustness certificate (radius ≥ min gap / 2).

Hardware note:
- The associativity re‑bracketing (V ⊗ Kᵀ) ⊗ Q sidesteps materializing attention A = Qᵀ ⊗ K, enabling an
  add/compare systolic array with predictable memory/latency.

This file encodes these objects directly:
tmm (tropical GEMM), T2 block, tropical pooling/classifier, hinge loss, route extraction, and a tiny
length‑generalization dataset (“pivot amid crowd”) where max–plus attention is trivially length‑invariant.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


@dataclass
class Config:
    d: int
    dk: int
    H: int
    C: int
    L: int
    residual: bool = False
    margin: float = 1.0


@dataclass
class Params:
    WQ: jnp.ndarray
    WK: jnp.ndarray
    WV: jnp.ndarray
    Wcls: jnp.ndarray


def rng_split(k, n=1):
    return jax.random.split(k, n) if n > 1 else jax.random.split(k, 2)[0]


def gauge_time(X):
    # Max-plus normalization: subtract columnwise max so sup per column is 0
    return X - jnp.max(X, axis=0, keepdims=True)


def tmm(A, B):
    # Tropical matrix multiply: (A ⊗ B)_{i j} = max_k (A_{i k} + B_{k j})
    return jnp.max(A[..., :, :, None] + B[..., None, :, :], axis=-2)


# ---------------------------
# Test adapter: TropicalAttention
# ---------------------------


class TropicalAttention:
    """Simple tropical attention adapter used in tests.

    For each query, selects the value with the best max-plus score against keys.
    """

    def __init__(self, dim: int):
        self.dim = int(dim)

    def __call__(self, Q, K, V):  # numpy arrays
        import numpy as _np
        Qj = jnp.array(Q, dtype=jnp.float32)
        Kj = jnp.array(K, dtype=jnp.float32)
        Vj = jnp.array(V, dtype=jnp.float32)
        scores = jnp.max(Qj[:, None, :] + Kj[None, :, :], axis=-1)
        vals, idx = jax.lax.top_k(scores, k=2)
        # Certificate: margin between best and second-best per query
        cert = (vals[:, 0] - vals[:, 1]).min()
        out = Vj[idx[:, 0]]
        # Attach certificate as attribute for downstream inspection
        self.last_min_margin = float(cert)
        return _np.asarray(out)


# --- Test‑facing simple semiring ops ---

def tropical_add(a, b):
    """Tropical addition (max-plus): a ⊕ b = max(a, b)."""
    return jnp.maximum(a, b)


def tropical_multiply(a, b):
    """Tropical multiplication (max-plus): a ⊗ b = a + b."""
    return a + b


def top2_last(x):
    v, i = jax.lax.top_k(x, 2)
    return v[..., 0], i[..., 0], v[..., 1]


def argmax_margin(v):
    v1, i1, v2 = top2_last(v)
    return i1, v1 - v2, v1


def init_params(k, cfg: Config, delta=0.1, eps=1e-3):
    k1, k2, k3, k4 = jax.random.split(k, 4)

    def diag_like(m, n):
        A = jnp.full((m, n), -delta)
        idx = jnp.arange(min(m, n))
        A = A.at[idx, idx].set(0.0)
        return A

    WQ = diag_like(cfg.H * cfg.dk, cfg.d).reshape(cfg.H, cfg.dk, cfg.d)
    WK = diag_like(cfg.H * cfg.dk, cfg.d).reshape(cfg.H, cfg.dk, cfg.d)
    WV = diag_like(cfg.H * cfg.d, cfg.d).reshape(cfg.H, cfg.d, cfg.d)
    Wcls = diag_like(cfg.C, cfg.d)
    WQ = WQ + jax.random.uniform(k1, WQ.shape, minval=-eps, maxval=0.0)
    WK = WK + jax.random.uniform(k2, WK.shape, minval=-eps, maxval=0.0)
    WV = WV + jax.random.uniform(k3, WV.shape, minval=-eps, maxval=0.0)
    Wcls = Wcls + jax.random.uniform(k4, Wcls.shape, minval=-eps, maxval=0.0)
    return Params(WQ, WK, WV, Wcls)


# --- Experimental: Tropical convolutions and morphological ops ---

def tropical_conv1d(x: jnp.ndarray, w: jnp.ndarray, stride: int = 1) -> jnp.ndarray:
    r"""1D tropical convolution: (x \otimes w)[t] = max_k (x[t + k] + w[k])."""
    T = x.shape[-1]
    K = w.shape[-1]
    out_len = 1 + (T - K) // stride
    outs = []
    for t in range(out_len):
        sl = x[..., t * stride : t * stride + K]
        outs.append(jnp.max(sl + w, axis=-1))
    return jnp.stack(outs, axis=-1)


def morph_dilate1d(x: jnp.ndarray, se: jnp.ndarray) -> jnp.ndarray:
    """Binary-like dilation in max-plus: dilate x by structuring element se."""
    return tropical_conv1d(x, se)


def t2_block(params: Params, X, cfg: Config):
    Q = tmm(params.WQ, X)
    K = tmm(params.WK, X)
    V = tmm(params.WV, X)
    U = tmm(V, jnp.swapaxes(K, -1, -2))
    Z = tmm(U, Q)
    Y = jnp.max(Z, axis=0)
    Y = jnp.maximum(Y, X) if cfg.residual else Y
    Xo = gauge_time(Y)
    return Xo, (Q, K, V, U, Z)


def pool_logits(params: Params, X):
    p = jnp.max(X, axis=1)
    y = tmm(params.Wcls, p[:, None])[:, 0]
    return p, y


def forward(params: Params, X, cfg: Config):
    Xo, _ = t2_block(params, X, cfg)
    p, y = pool_logits(params, Xo)
    return y


def loss_margin(y, labels, m):
    a = jnp.take_along_axis(y, labels[:, None], axis=1)[:, 0]
    mx = jnp.where(jnp.arange(y.shape[1]) == labels[:, None], -jnp.inf, y)
    b = jnp.max(mx, axis=1)
    return jnp.maximum(0.0, m + b - a)


def predict(y):
    return jnp.argmax(y, axis=1)


def route_single(params: Params, X, cfg: Config, cls: int):
    Xo, (Q, K, V, U, Z) = t2_block(params, X, cfg)
    p, y = pool_logits(params, Xo)
    i_cls, m_cls, _ = argmax_margin(params.Wcls[cls] + p)
    t_cls, m_pool, _ = argmax_margin(Xo[i_cls])
    zheads = Z[:, i_cls, t_cls]
    h_idx, m_head, _ = argmax_margin(zheads)
    r_vec = U[h_idx, i_cls] + Q[h_idx, :, t_cls]
    r_idx, m_z, _ = argmax_margin(r_vec)
    u_vec = V[h_idx, i_cls] + K[h_idx, r_idx]
    u_idx, m_u, _ = argmax_margin(u_vec)
    v_vec = params.WV[h_idx, i_cls] + X[:, u_idx]
    iV, m_v, _ = argmax_margin(v_vec)
    k_vec = params.WK[h_idx, r_idx] + X[:, u_idx]
    iK, m_k, _ = argmax_margin(k_vec)
    q_vec = params.WQ[h_idx, r_idx] + X[:, t_cls]
    iQ, m_q, _ = argmax_margin(q_vec)
    m = jnp.minimum(
        jnp.minimum(jnp.minimum(m_cls, m_pool), jnp.minimum(m_head, m_z)),
        jnp.minimum(jnp.minimum(m_u, m_v), jnp.minimum(m_k, m_q)),
    )
    return dict(i=i_cls, t=t_cls, h=h_idx, r=r_idx, u=u_idx, iV=iV, iK=iK, iQ=iQ, margin=m)


def kstar(y, cls):
    y2 = y.at[cls].set(-jnp.inf)
    return int(jnp.argmax(y2))


def deltas_from_route(cfg: Config, route, cls, sign):
    dWQ = jnp.zeros((cfg.H, cfg.dk, cfg.d))
    dWK = jnp.zeros((cfg.H, cfg.dk, cfg.d))
    dWV = jnp.zeros((cfg.H, cfg.d, cfg.d))
    dWc = jnp.zeros((cfg.C, cfg.d))
    dWc = dWc.at[cls, route["i"]].add(sign)
    dWQ = dWQ.at[route["h"], route["r"], route["iQ"]].add(sign)
    dWK = dWK.at[route["h"], route["r"], route["iK"]].add(sign)
    dWV = dWV.at[route["h"], route["i"], route["iV"]].add(sign)
    return dWQ, dWK, dWV, dWc


def apply_update(params: Params, upd, eta):
    dWQ, dWK, dWV, dWc = upd
    return Params(params.WQ + eta * dWQ, params.WK + eta * dWK, params.WV + eta * dWV, params.Wcls + eta * dWc)


def train_step(params: Params, X, y_true, cfg: Config):
    def body(prm, xy):
        Xs, ys = xy
        ylog = forward(prm, Xs, cfg)
        c = int(ys)
        r_c = route_single(prm, Xs, cfg, c)
        k = kstar(ylog, c)
        r_k = route_single(prm, Xs, cfg, k)
        eta = 0.5 * jnp.minimum(r_c["margin"], r_k["margin"])
        upd_pos = deltas_from_route(cfg, r_c, c, +1.0)
        upd_neg = deltas_from_route(cfg, r_k, k, -1.0)
        prm = apply_update(prm, upd_pos, eta)
        prm = apply_update(prm, upd_neg, eta)
        return prm, eta

    params, etas = jax.lax.scan(body, params, (X, y_true))
    return params, etas


@partial(jax.jit, static_argnums=(3,))
def batch_loss(params: Params, X, y_true, cfg: Config):
    def f(x):
        return forward(params, x, cfg)

    Y = jax.vmap(f)(X)
    L = loss_margin(Y, y_true, cfg.margin)
    acc = jnp.mean((predict(Y) == y_true).astype(jnp.float32))
    return jnp.mean(L), acc


def pivot_crowd_dataset(k, n, L, cfg: Config):
    d = cfg.d

    def one(k):
        k1, k2, k3 = jax.random.split(k, 3)
        c = int(jax.random.bernoulli(k1, 0.5))
        p = int(jax.random.randint(k2, (), 0, L))
        X = jnp.full((d, L), -2.0)
        X = X.at[0].set(-1.0)
        X = X.at[0, p].set(0.0)
        X = X.at[1].set(-1.0)
        X = X.at[1, p].set(0.0 if c == 1 else -1.0)
        return gauge_time(X), c

    ks = jax.random.split(k, n)
    Xc, yc = jax.vmap(one)(ks)
    return Xc, jnp.array(yc)


def run():
    key = jax.random.PRNGKey(0)
    cfg = Config(d=16, dk=4, H=2, C=2, L=16, residual=False, margin=1.0)
    params = init_params(key, cfg)
    k1, k2 = jax.random.split(key, 2)
    Xtr, ytr = pivot_crowd_dataset(k1, 1024, cfg.L, cfg)
    Xte_l = jax.random.PRNGKey(2025)
    cfg_test = Config(d=cfg.d, dk=cfg.dk, H=cfg.H, C=cfg.C, L=64, residual=False, margin=1.0)
    Xte, yte = pivot_crowd_dataset(Xte_l, 1024, cfg_test.L, cfg_test)

    def train_epoch(params, X, y, cfg):
        params, _ = train_step(params, X, y, cfg)
        return params

    for e in range(5):
        params = train_epoch(params, Xtr, ytr, cfg)
        tr_loss, tr_acc = batch_loss(params, Xtr, ytr, cfg)
        te_loss, te_acc = batch_loss(params, Xte, yte, cfg_test)
        print(
            f"epoch {e + 1} | train loss {float(tr_loss):.4f} acc {float(tr_acc):.4f} | test(L={cfg_test.L}) loss {float(te_loss):.4f} acc {float(te_acc):.4f}"
        )
    yhat = jax.vmap(lambda x: forward(params, x, cfg_test))(Xte)
    cert = jnp.minimum(  # per-sample crude cert via logit gap
        jnp.take_along_axis(yhat, yte[:, None], axis=1)[:, 0]
        - jnp.max(jnp.where(jnp.arange(cfg.C) == yte[:, None], -jnp.inf, yhat), axis=1),
        jnp.array([jnp.inf]),
    )
    print("median margin:", float(jnp.median(cert)))


# (adapter defined above)


def demo():
    """Run the tropical geometry demonstration."""
    run()


if __name__ == "__main__":
    demo()
