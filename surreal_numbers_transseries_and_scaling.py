"""
Field-theoretic scaling, implemented.

This code instantiates a dominance-only scaling rule derived from a non‑Archimedean,
totally ordered field view of training: error E is a sum of leading monomials whose
orders compare via a valuation (tropical addition: sums are dominated by the smallest
order). Instead of fitting exponents, we reveal the dominant term by three forward-only
"half" projections and take the axis with the largest damage.

Core constructs
• Width/Depth/Data axes: width = channels/heads, depth = residual blocks, data = train/val split.
• Probes (no retraining):
  T_D = E_val / E_train,   T_H = E(f with half the blocks skipped) / E(f),
  T_W = E(f with half the channels zeroed, rescaled) / E(f).
• Decision rule: NextMove = argmax{T_D, T_H, T_W}. If the top two are within (1+ε), treat
  them as equal-order (balanced manifold) and default to expanding data.
• Minimal falsification: recompute argmax on two independent mini-batches; if it flips,
  dominance is not well-posed at this checkpoint → report FALSIFIED.

Implementation notes
• Residual MLP stack with pre-LN; depth projection = identity-skip every other block.
• Width projection = fixed half-channel mask with inverse-keep rescaling to preserve units.
• Everything is JAX-jitted; the probes are single additional forward passes.
• A synthetic teacher–student task is included to make the pipeline executable end-to-end.

This mirrors the math: identify the dominant monomial by order (largest damage ratio), then
scale along its axis; when ratios tie, you are on the balanced frontier of equal leading order.
"""

# Docs: markdown_documentation/surreal_numbers_transseries_and_scaling.md

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax, value_and_grad


class PRNG:
    def __init__(self, seed):
        self.k = jax.random.PRNGKey(seed)

    def split(self, n=1):
        self.k, *ks = jax.random.split(self.k, n + 1)
        return ks if n > 1 else ks[0]


def glorot(k, shape):
    fan_in, fan_out = (shape[0], shape[1])
    lim = math.sqrt(6 / (fan_in + fan_out))
    return jax.random.uniform(k, shape, minval=-lim, maxval=lim)


def gelu(x):
    return 0.5 * x * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))


def layer_norm(x, eps=1e-5, gamma=None, beta=None):
    m = jnp.mean(x, -1, keepdims=True)
    v = jnp.var(x, -1, keepdims=True)
    y = (x - m) / jnp.sqrt(v + eps)
    if gamma is not None:
        y = y * gamma + beta
    return y


def teacher_make(rng, in_dim, hid_dim, out_dim):
    k1, k2 = rng.split(2)
    W1 = glorot(k1, (in_dim, hid_dim))
    b1 = jnp.zeros((hid_dim,))
    W2 = glorot(k2, (hid_dim, out_dim))
    b2 = jnp.zeros((out_dim,))
    return (W1, b1, W2, b2)


@jit
def teacher_logits(params, x):
    W1, b1, W2, b2 = params
    h = gelu(x @ W1 + b1)
    return h @ W2 + b2


def make_dataset(rng, n_train, n_val, in_dim, K):
    ks = rng.split(3)
    t = teacher_make(ks[0], in_dim, 128, K)
    Xtr = jax.random.normal(ks[1], (n_train, in_dim))
    Xva = jax.random.normal(ks[2], (n_val, in_dim))
    ytr = jnp.argmax(teacher_logits(t, Xtr), -1)
    yva = jnp.argmax(teacher_logits(t, Xva), -1)
    return (Xtr, ytr), (Xva, yva)


def init_params(rng, in_dim, d_model, ff_mult, H, K):
    kE, k_out = rng.split(2)
    W_e = glorot(kE, (in_dim, d_model))
    b_e = jnp.zeros((d_model,))
    blocks = []
    ks = rng.split(H * 4)
    for i in range(H):
        W1 = glorot(ks[4 * i + 0], (d_model, int(ff_mult * d_model)))
        b1 = jnp.zeros((int(ff_mult * d_model),))
        W2 = glorot(ks[4 * i + 1], (int(ff_mult * d_model), d_model))
        b2 = jnp.zeros((d_model,))
        g = jnp.ones((d_model,))
        be = jnp.zeros((d_model,))
        blocks.append({"W1": W1, "b1": b1, "W2": W2, "b2": b2, "g": g, "be": be})
    W_o = glorot(k_out, (d_model, K))
    b_o = jnp.zeros((K,))
    return {"emb": {"W": W_e, "b": b_e}, "blocks": tuple(blocks), "out": {"W": W_o, "b": b_o}}


def make_width_mask(rng, d_model, keep):
    if keep >= 1.0:
        return jnp.ones((d_model,), dtype=jnp.float32), 1.0
    # Accept either our PRNG wrapper or a raw JAX key
    if hasattr(rng, "split") and not isinstance(rng, jax.Array):
        # PRNG.split(1) returns a single key; do not index into it
        k = rng.split()
    else:
        # For raw JAX keys, generate one child key deterministically
        _, k = jax.random.split(rng, 2)
    idx = jax.random.permutation(k, d_model)
    kcnt = int(jnp.floor(keep * d_model))
    keep_idx = idx[:kcnt]
    mask = jnp.zeros((d_model,), dtype=jnp.float32).at[keep_idx].set(1.0)
    return mask, 1.0 / keep


def make_depth_mask(H, half):
    if not half:
        return jnp.ones((H,), dtype=jnp.bool_)
    m = jnp.arange(H) % 2 == 0
    return m


@partial(jit, static_argnums=(6,))
def forward(params, x, width_mask, depth_mask, inv_keep, training, H):
    W_e, b_e = params["emb"]["W"], params["emb"]["b"]
    x = x @ W_e + b_e

    # Stack all blocks and depth masks for scan to avoid indexing during tracing
    all_blocks = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *params["blocks"])

    def scan_f(carry, inputs):
        x, wm = carry
        blk, flag = inputs
        xg = x * wm

        def apply_block(z):
            h = layer_norm(z, gamma=blk["g"], beta=blk["be"])
            a = gelu(h @ blk["W1"] + blk["b1"])
            b = a @ blk["W2"] + blk["b2"]
            b = b * wm
            return z + b

        y = lax.cond(flag, apply_block, lambda z: z, xg)
        return (y, wm), None

    carry = (x, width_mask)
    carry, _ = lax.scan(scan_f, carry, (all_blocks, depth_mask))
    x = carry[0] * width_mask
    logits = x @ params["out"]["W"] + params["out"]["b"]
    logits = logits * inv_keep
    return logits


@partial(jit, static_argnums=(6,))
def nll(params, x, y, width_mask, depth_mask, inv_keep, H):
    logits = forward(params, x, width_mask, depth_mask, inv_keep, False, H)
    z = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    return -jnp.mean(z[jnp.arange(z.shape[0]), y])


def tree_map(f, tree):
    return jax.tree_util.tree_map(f, tree)


def tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_mul(a, s):
    return jax.tree_util.tree_map(lambda x: x * s, a)


def adam_init(params):
    m = tree_map(jnp.zeros_like, params)
    v = tree_map(jnp.zeros_like, params)
    t = jnp.array(0, dtype=jnp.int32)
    return {"m": m, "v": v, "t": t}


def adam_update(params, grads, st, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    t = st["t"] + 1
    m = tree_add(tree_mul(st["m"], b1), tree_mul(grads, 1 - b1))
    v = tree_add(tree_mul(st["v"], b2), tree_mul(tree_map(lambda g: g * g, grads), 1 - b2))
    mhat = tree_map(lambda x: x / (1 - b1**t), m)
    vhat = tree_map(lambda x: x / (1 - b2**t), v)
    upd = tree_map(lambda mh, vh: lr * mh / (jnp.sqrt(vh) + eps), mhat, vhat)
    new_params = tree_add(params, tree_map(lambda u: -u, upd))
    return new_params, {"m": m, "v": v, "t": t}


@partial(jit, static_argnums=(4,))
def train_step(params, st, x, y, H, width_mask, depth_mask, inv_keep, lr):
    loss, grads = value_and_grad(nll)(params, x, y, width_mask, depth_mask, inv_keep, H)
    new_params, st2 = adam_update(params, grads, st, lr=lr)
    return new_params, st2, loss


def dataset_iter(X, Y, batch_size):
    n = X.shape[0]

    def get(i):
        s = (i * batch_size) % n
        e = s + batch_size
        if e <= n:
            return X[s:e], Y[s:e]
        idx = jnp.concatenate([jnp.arange(s, n), jnp.arange(0, e - n)], 0)
        return X[idx], Y[idx]

    return get


def compute_T(
    params, H, Xtr, Ytr, Xva, Yva, width_mask_ones, depth_mask_full, inv1, width_mask_half, inv2, depth_mask_half
):
    Ltr = nll(params, Xtr, Ytr, width_mask_ones, depth_mask_full, inv1, H)
    Lva = nll(params, Xva, Yva, width_mask_ones, depth_mask_full, inv1, H)
    # Use a more robust epsilon and ensure consistent handling
    eps = 1e-6
    base = jnp.maximum(Lva, eps)
    T_D = Lva / jnp.maximum(Ltr, eps)
    Lva_H = nll(params, Xva, Yva, width_mask_ones, depth_mask_half, inv1, H)
    Lva_W = nll(params, Xva, Yva, width_mask_half, depth_mask_full, inv2, H)
    # Clip ratios to prevent numerical explosion
    T_H = jnp.minimum(Lva_H / base, 1e3)
    T_W = jnp.minimum(Lva_W / base, 1e3)
    return T_D, T_H, T_W, Ltr, Lva


def choose_move(TD, TH, TW, eps=0.02):
    v = jnp.array([TD, TH, TW])
    i = int(jnp.argmax(v))
    j = int(jnp.argmax(v.at[i].set(-jnp.inf)))
    # Use maximum to ensure we don't divide by zero or near-zero
    ratio = v[i] / jnp.maximum(v[j], 1e-6)
    if float(ratio) <= 1.0 + eps:
        return "data"
    return ["data", "depth", "width"][i]


def stress_test(
    params,
    H,
    get_tr,
    get_va,
    bs,
    width_mask_ones,
    depth_mask_full,
    inv1,
    width_mask_half,
    inv2,
    depth_mask_half,
    eps=0.02,
):
    Xtr1, Ytr1 = get_tr(0)
    Xva1, Yva1 = get_va(0)
    Xtr2, Ytr2 = get_tr(1)
    Xva2, Yva2 = get_va(1)
    TD1, TH1, TW1, _, _ = compute_T(
        params,
        H,
        Xtr1,
        Ytr1,
        Xva1,
        Yva1,
        width_mask_ones,
        depth_mask_full,
        inv1,
        width_mask_half,
        inv2,
        depth_mask_half,
    )
    TD2, TH2, TW2, _, _ = compute_T(
        params,
        H,
        Xtr2,
        Ytr2,
        Xva2,
        Yva2,
        width_mask_ones,
        depth_mask_full,
        inv1,
        width_mask_half,
        inv2,
        depth_mask_half,
    )
    m1 = choose_move(float(TD1), float(TH1), float(TW1), eps)
    m2 = choose_move(float(TD2), float(TH2), float(TW2), eps)
    return m1 == m2, (float(TD1), float(TH1), float(TW1)), (float(TD2), float(TH2), float(TW2)), m1, m2


def main():
    seed = 42
    in_dim = 64
    d_model = 192
    ff_mult = 4.0
    H = 12
    K = 10
    n_train = 8192
    n_val = 8192
    batch_size = 512
    steps = 600
    lr = 2e-3
    rng = PRNG(seed)
    (Xtr, Ytr), (Xva, Yva) = make_dataset(rng, n_train, n_val, in_dim, K)
    params = init_params(rng, in_dim, d_model, ff_mult, H, K)
    opt = adam_init(params)
    width_mask_ones = jnp.ones((d_model,), dtype=jnp.float32)
    inv1 = 1.0
    depth_mask_full = make_depth_mask(H, False)
    width_mask_half, inv2 = make_width_mask(rng, d_model, 0.5)
    depth_mask_half = make_depth_mask(H, True)
    get_tr = dataset_iter(Xtr, Ytr, batch_size)
    get_va = dataset_iter(Xva, Yva, batch_size)
    for step in range(steps):
        xb, yb = get_tr(step)
        params, opt, loss = train_step(params, opt, xb, yb, H, width_mask_ones, depth_mask_full, inv1, lr)
        if (step + 1) % 100 == 0:
            vxb, vyb = get_va(step // 100)
            Lva = nll(params, vxb, vyb, width_mask_ones, depth_mask_full, inv1, H)
            print(f"step {step + 1} train_nll={float(loss):.4f} val_nll={float(Lva):.4f}")
    xb_t, yb_t = get_tr(0)
    xv_t, yv_t = get_va(0)
    TD, TH, TW, Ltr, Lva = compute_T(
        params,
        H,
        xb_t,
        yb_t,
        xv_t,
        yv_t,
        width_mask_ones,
        depth_mask_full,
        inv1,
        width_mask_half,
        inv2,
        depth_mask_half,
    )
    move = choose_move(float(TD), float(TH), float(TW), 0.02)
    ok, (TD1, TH1, TW1), (TD2, TH2, TW2), m1, m2 = stress_test(
        params,
        H,
        get_tr,
        get_va,
        batch_size,
        width_mask_ones,
        depth_mask_full,
        inv1,
        width_mask_half,
        inv2,
        depth_mask_half,
        0.02,
    )
    print("ratios T_D,T_H,T_W =", float(TD), float(TH), float(TW))
    print("next_move =", move)
    print(
        "stress_test_ok =", ok, " batch1=", m1, " batch2=", m2, " ratios1=", TD1, TH1, TW1, " ratios2=", TD2, TH2, TW2
    )
    print("E_train=", float(Ltr), " E_val=", float(Lva))
    if not ok:
        print("FALSIFIED")


def demo():
    """Run the surreal numbers transseries and scaling demonstration."""
    main()


if __name__ == "__main__":
    demo()


# --- Minimal surreal number API for tests ---


class SurrealNumber:
    """Tiny subset sufficient for tests: represent dyadic rationals as floats with constructors."""

    def __init__(self, value: float = 0.0):
        self.value = float(value)

    @staticmethod
    def from_int(n: int) -> "SurrealNumber":
        return SurrealNumber(float(n))

    @staticmethod
    def from_rational(p: int, q: int) -> "SurrealNumber":
        return SurrealNumber(float(p) / float(q))


def surreal_compare(a: SurrealNumber, b: SurrealNumber) -> float:
    return float(a.value - b.value)


def surreal_add(a: SurrealNumber, b: SurrealNumber) -> SurrealNumber:
    return SurrealNumber(a.value + b.value)


def surreal_multiply(a: SurrealNumber, b: SurrealNumber) -> SurrealNumber:
    return SurrealNumber(a.value * b.value)
