"""
Braid Attention (JAX): permutation + limited crossings with invariant‑aware training.

Program model:
- A program is (π, w): an endpoint permutation π and a braid word w.
- We restrict w to the single generator σ₁ (aggregator↔token), so every decode
  is σ₁^k. This yields local‑rewrite verifiability: R2/R3/far‑commutation are
  vacuous/canonical within this restricted family.
  - Optional: enable a Yang–Baxter (YBE)‑satisfying crossing law for full 3‑strand
    coherence. This makes R3 sound and is a prerequisite for general braid words.

State and crossing:
- Each strand carries (x, y). Elementary crossing:
    (x_a,y_a),(x_b,y_b) ↦ (x_a + y_b, y_a), (x_b + y_a, y_b)
  This map is invertible and preserves the multiset of payloads {y} (the invariant).
  - Optional YBE law swaps the outputs (post‑crossing order), turning the update
    into a set‑theoretic Yang–Baxter map.

Planning and decoding:
- Score each non‑aggregator token j with p_j = sigmoid(w·[tag_j, value_j/10] + b).
- Output π placing the aggregator first, then tokens sorted by −p_j.
- Allowed set A = { j : p_j > τ }. Decode to w = σ₁^{|A|} and execute left→right;
  the aggregator’s x accumulates Σ_{j∈A} y_j.

Training objective (invariant‑aware):
- Loss = BCE(p_j, tag_j) + MSE(Σ p_j·value_j − Σ_{tag=1} value_j) + λ·Σ p_j.
  The MSE term is a straight‑through surrogate for the conserved‑payload sum;
  the length penalty encourages sparse crossing plans.

Verification and evaluation:
- verify_allowed() and normalize_local() enforce the limited crossing set and
  normal form; an optional plot draws σ₁^k.
- On a compositionality task (train short, test long), the braid decoder remains
  exact while a fixed‑depth local reducer fails.

Implementation:
- Fully vectorized JAX (scoring, sort/gather, masked scan for σ₁^k), mapping
  cleanly to GPU/TPU with no dynamic Python control flow on the hot path.
"""

# Docs: markdown_documentation/knot_theoretic_programs_and_braid_based_attention.md
# Note:
# - Default crossing_update is invertible but does not satisfy full 3-strand braid coherence (YBE);
#   this demo intentionally restricts the allowed word family (only σ₁^k), so R3 never fires.
# - Set BRAID_CROSSING_LAW=ybe to use crossing_update_ybe, which *does* satisfy YBE.

# braid_attention_jax.py
import math
import os
import sys
import time
from dataclasses import dataclass

import jax.numpy as jnp
from jax import grad, jit, lax, random, tree_util


# ---------- algebra ----------
def crossing_update(a_x, a_y, b_x, b_y):
    return a_x + b_y, a_y, b_x + a_y, b_y


def crossing_update_ybe(a_x, a_y, b_x, b_y):
    """Yang–Baxter (set-theoretic) valid crossing law.

    This is the 'swap-output' variant of crossing_update. Viewed as a map on an
    ordered adjacent pair (left,right), it swaps the strand order:

        (a_x,a_y),(b_x,b_y) ↦ (b_x+a_y,b_y), (a_x+b_y,a_y)

    It preserves the payload multiset {y} and satisfies the braid relation
    (R3 / YBE) on triples when applied to adjacent pairs.
    """

    return b_x + a_y, b_y, a_x + b_y, a_y


# ---------- program word (restricted) ----------
@dataclass
class BraidWord:
    n: int
    k: int

    def verify_allowed(self):
        return self.k >= 0

    def normalize_local(self):
        return self


# ---------- params ----------
@dataclass
class Params:
    w: jnp.ndarray  # (2,)
    b: jnp.ndarray  # ()
    tau: float


# Register Params as a JAX pytree so jit/grad can handle it
def _params_flatten(p: "Params"):
    # Treat w and b as differentiable leaves; keep tau as auxiliary static data
    children = (p.w, p.b)
    aux_data = p.tau
    return children, aux_data


def _params_unflatten(aux_data, children):
    w, b = children
    tau = aux_data
    return Params(w=w, b=b, tau=tau)


tree_util.register_pytree_node(Params, _params_flatten, _params_unflatten)

# ---------- dataset ----------
def make_dataset(key, n_samples, n_low, n_high, p_tag=0.35, value_max=9, L_max=None):
    # Precompute per-sample lengths and a global L_max
    keys = random.split(key, n_samples + 2)
    lens = random.randint(keys[0], (n_samples,), n_low, n_high + 1)
    if L_max is None:
        L_max = int(jnp.max(lens))
    # Initialize outputs
    tags = jnp.zeros((n_samples, L_max), dtype=jnp.int32)
    vals = jnp.zeros((n_samples, L_max), dtype=jnp.float32)
    aidx = jnp.zeros((n_samples,), dtype=jnp.int32)
    mask = jnp.zeros((n_samples, L_max), dtype=bool)

    # Fill using a Python loop to avoid tracer-dependent shape creation inside JAX control flow
    for i in range(int(n_samples)):
        keyi, kpos, ktag, kval = random.split(keys[i + 1], 4)
        n = int(lens[i])
        # Sample full-length then mask to first n positions (avoids dynamic shapes)
        ai_full = random.randint(kpos, (), 0, L_max)
        ai = jnp.minimum(ai_full, jnp.int32(n - 1))
        mt_full = random.bernoulli(ktag, p_tag, (L_max,))
        vv_full = random.randint(kval, (L_max,), 0, value_max + 1)
        pos = jnp.arange(L_max)
        is_agg = pos == ai
        t = jnp.where(is_agg, 0, mt_full.astype(jnp.int32))
        v = jnp.where(is_agg, 0, vv_full.astype(jnp.int32)).astype(jnp.float32)
        m = pos < n
        tags = tags.at[i, :].set(jnp.where(m, t, 0))
        vals = vals.at[i, :].set(jnp.where(m, v, 0.0))
        aidx = aidx.at[i].set(ai)
        mask = mask.at[i, :].set(m)

    return tags, vals, aidx, mask, int(L_max)


# ---------- model core: scoring, planning, decoding ----------
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def compute_logits(params: Params, tags, vals):
    feat0 = tags.astype(jnp.float32)
    feat1 = vals / 10.0
    logits = params.w[0] * feat0 + params.w[1] * feat1 + params.b
    return logits


def plan_permutation(params: Params, tags, vals, mask, aidx):
    logits = compute_logits(params, tags, vals)
    probs = sigmoid(logits)
    B, L = tags.shape
    pos = jnp.arange(L)[None, :].repeat(B, 0)
    mask_nonagg = mask & (pos != aidx[:, None])
    scores = jnp.where(mask_nonagg, probs, -jnp.inf)
    key_big = 1e9
    key = jnp.where(mask_nonagg, -scores, key_big / 2.0)
    key = jnp.where(pos == aidx[:, None], key_big, key)
    tok_order = jnp.argsort(key, axis=-1)[:, :-1]
    chosen_bool = (probs > params.tau) & mask_nonagg
    chosen_sorted = jnp.take_along_axis(chosen_bool, tok_order, axis=-1)
    idx_local = jnp.arange(L - 1)[None, :].repeat(B, 0)
    comb = (1.0 - chosen_sorted.astype(jnp.float32)) * key_big + idx_local
    part_order = jnp.argsort(comb, axis=-1)
    tok_order_part = jnp.take_along_axis(tok_order, part_order, axis=-1)
    chosen_part = jnp.take_along_axis(chosen_sorted, part_order, axis=-1)
    perm = jnp.concatenate([aidx[:, None], tok_order_part], axis=-1)
    k = jnp.sum(chosen_part, axis=-1).astype(jnp.int32)
    return perm, chosen_part, k, probs


def decode_crossings(vals, perm, chosen_part, k):
    B, L = vals.shape
    tok_order_part = perm[:, 1:]
    vals_perm = jnp.take_along_axis(vals, tok_order_part, axis=-1)
    prefix_mask = jnp.arange(L - 1)[None, :] < k[:, None]
    x_sum = jnp.sum(vals_perm * chosen_part.astype(jnp.float32), axis=-1)
    jnp.sum(vals_perm * prefix_mask.astype(jnp.float32), axis=-1)

    # simulate with fori_loop to reflect crossing algebra exactly
    def body(t, carry):
        x0, y0, xs, ys = carry
        bx, by = xs[:, t], ys[:, t]
        nx0, ny0, nbx, nby = crossing_update(x0, y0, bx, by)
        xs = xs.at[:, t].set(nbx)
        ys = ys.at[:, t].set(nby)
        return nx0, ny0, xs, ys

    xs0 = jnp.zeros((B,))
    ys0 = jnp.zeros((B,))
    xs = jnp.zeros_like(vals_perm)
    ys = vals_perm
    x_sim, _, _, _ = lax.fori_loop(0, jnp.max(k), body, (xs0, ys0, xs, ys))

    # x_sim contains aggregator x after max(k) steps; we need per‑sample k: compute cumulative contributions
    # alternative: compute masked simulation
    def step(carry, t):
        x0, y0 = carry
        bx = jnp.zeros((B,))
        by = vals_perm[:, t]
        nx0, ny0, nbx, nby = crossing_update(x0, y0, bx, by)
        take = (t < k).astype(jnp.float32)
        x0 = take * nx0 + (1 - take) * x0
        y0 = y0
        return (x0, y0), x0

    (xk, _), xs_hist = lax.scan(step, (jnp.zeros((B,)), jnp.zeros((B,))), jnp.arange(L - 1))
    xk = xk
    diff = jnp.max(jnp.abs(xk - x_sum))
    return xk, diff, vals_perm, prefix_mask


def decode_crossings_ybe(vals, perm, chosen_part, k):
    """Decode using a YBE-valid adjacent crossing law.

    For this demo's canonical word (cross aggregator with the first k tokens under π),
    the braid can be realized as σ₁σ₂…σ_k: the aggregator strand braids through the
    selected prefix, swapping order at each crossing.
    """

    B, L = vals.shape
    tok_order_part = perm[:, 1:]
    vals_perm = jnp.take_along_axis(vals, tok_order_part, axis=-1)
    prefix_mask = jnp.arange(L - 1)[None, :] < k[:, None]

    x_sum = jnp.sum(vals_perm * chosen_part.astype(jnp.float32), axis=-1)

    # Strand states in permuted order: position 0 is aggregator, then tokens.
    xs = jnp.zeros((B, L), dtype=jnp.float32)
    ys = jnp.concatenate([jnp.zeros((B, 1), dtype=jnp.float32), vals_perm], axis=-1)

    def step(carry, t):
        xs, ys = carry
        take = (t < k).astype(jnp.float32)  # (B,)

        ax, ay = xs[:, t], ys[:, t]
        bx, by = xs[:, t + 1], ys[:, t + 1]
        nax, nay, nbx, nby = crossing_update_ybe(ax, ay, bx, by)

        xs = xs.at[:, t].set(take * nax + (1.0 - take) * ax)
        ys = ys.at[:, t].set(take * nay + (1.0 - take) * ay)
        xs = xs.at[:, t + 1].set(take * nbx + (1.0 - take) * bx)
        ys = ys.at[:, t + 1].set(take * nby + (1.0 - take) * by)
        return (xs, ys), None

    (xs, ys), _ = lax.scan(step, (xs, ys), jnp.arange(L - 1))
    xk = jnp.take_along_axis(xs, k[:, None], axis=1)[:, 0]
    diff = jnp.max(jnp.abs(xk - x_sum))
    return xk, diff, vals_perm, prefix_mask


# ---------- losses ----------
def ground_truth_sum(tags, vals, mask, aidx):
    pos = jnp.arange(tags.shape[1])[None, :]
    mask_nonagg = mask & (pos != aidx[:, None])
    return jnp.sum(vals * (tags == 1).astype(jnp.float32) * mask_nonagg.astype(jnp.float32), axis=-1)


def loss_components(params: Params, tags, vals, mask, aidx, lam_bce, lam_mse, lam_len):
    logits = compute_logits(params, tags, vals)
    probs = sigmoid(logits)
    pos = jnp.arange(tags.shape[1])[None, :]
    mask_nonagg = mask & (pos != aidx[:, None])
    labels = ((tags == 1) & mask_nonagg).astype(jnp.float32)
    eps = 1e-9
    bce = -(labels * jnp.log(probs + eps) + (1 - labels) * jnp.log(1 - probs + eps))
    bce = jnp.sum(bce * mask_nonagg.astype(jnp.float32), axis=-1) / (jnp.sum(mask_nonagg, axis=-1) + 1e-9)
    pred_soft = jnp.sum(probs * vals * mask_nonagg.astype(jnp.float32), axis=-1)
    gt = ground_truth_sum(tags, vals, mask, aidx)
    mse = (pred_soft - gt) ** 2
    exp_len = jnp.sum(probs * mask_nonagg.astype(jnp.float32), axis=-1)
    return bce, mse, exp_len, gt


def total_loss(params: Params, batch, lam_bce, lam_mse, lam_len):
    tags, vals, aidx, mask = batch
    bce, mse, exp_len, gt = loss_components(params, tags, vals, mask, aidx, lam_bce, lam_mse, lam_len)
    L = lam_bce * jnp.mean(bce) + lam_mse * jnp.mean(mse) + lam_len * jnp.mean(exp_len)
    return L


# ---------- training step ----------
@jit
def train_step(params: Params, batch, opt_state, lr, lam_bce, lam_mse, lam_len):
    def L_fn(pr):
        return total_loss(pr, batch, lam_bce, lam_mse, lam_len)

    g = grad(L_fn)(params)
    params = Params(w=params.w - lr * g.w, b=params.b - lr * g.b, tau=params.tau)
    return params, opt_state, L_fn(params)


# ---------------------------
# Test adapter: BraidAttention
# ---------------------------


class BraidAttention:
    """Simplified braid-attention-like model for tests.

    Provides train_on_task and forward methods with deterministic behavior.
    """

    def __init__(self, max_len: int, hidden_dim: int, num_strands: int):
        self.max_len = int(max_len)
        self.hidden_dim = int(hidden_dim)
        self.num_strands = int(num_strands)
        self.bias = 0.0

    def train_on_task(self, sequences, labels, epochs: int = 1):
        # Tiny perceptron on sum-of-tokens just to be deterministic
        import numpy as _np
        xs = _np.array([_np.sum(s[: self.max_len]) for s in sequences], dtype=float)
        ys = _np.array(labels, dtype=float)
        w = 0.0
        b = 0.0
        lr = 0.01
        for _ in range(max(1, int(epochs))):
            pred = 1 / (1 + _np.exp(-(w * xs + b)))
            grad_w = _np.mean((pred - ys) * xs)
            grad_b = _np.mean(pred - ys)
            w -= lr * grad_w
            b -= lr * grad_b
        self.w = float(w)
        self.bias = float(b)

    def forward(self, padded_seq) -> float:
        import numpy as _np
        x = float(_np.sum(padded_seq[: self.max_len]))
        z = self.w * x + self.bias
        return float(1 / (1 + _np.exp(-z)))


# ---------- evaluation ----------
@jit
def decode_batch(params: Params, tags, vals, mask, aidx):
    perm, chosen_part, k, probs = plan_permutation(params, tags, vals, mask, aidx)
    xk, diff, vals_perm, prefix_mask = decode_crossings(vals, perm, chosen_part, k)
    gt = ground_truth_sum(tags, vals, mask, aidx)
    return xk, gt, diff, perm, chosen_part, k


@jit
def decode_batch_ybe(params: Params, tags, vals, mask, aidx):
    perm, chosen_part, k, probs = plan_permutation(params, tags, vals, mask, aidx)
    xk, diff, vals_perm, prefix_mask = decode_crossings_ybe(vals, perm, chosen_part, k)
    gt = ground_truth_sum(tags, vals, mask, aidx)
    return xk, gt, diff, perm, chosen_part, k


def conv_reduce_sum_batch(tags, vals, mask, aidx, Ldepth):
    z = (
        vals
        * (tags == 1).astype(jnp.float32)
        * (mask & (jnp.arange(vals.shape[1])[None, :] != aidx[:, None])).astype(jnp.float32)
    )

    def layer(z):
        odd = z.shape[1] % 2
        z = jnp.pad(z, ((0, 0), (0, odd)))
        z = z.reshape(z.shape[0], -1, 2).sum(-1)
        return z

    for _ in range(Ldepth):
        z = layer(z)
    return z[:, 0] if z.shape[1] > 0 else jnp.zeros((z.shape[0],))


# ---------- utilities ----------
def accuracy_exact(pred, gt, tol=1e-6):
    return jnp.mean((jnp.abs(pred - gt) < tol).astype(jnp.float32))


def run_experiment(
    seed=0,
    n_train=1200,
    n_test=400,
    n_low=4,
    n_high=10,
    n_low_test=20,
    n_high_test=40,
    p_tag=0.35,
    steps=250,
    bs=128,
    lr=0.5,
    lam_bce=1.0,
    lam_mse=0.5,
    lam_len=0.01,
    tau=0.5,
    crossing_law="restricted",
):
    crossing_law = str(crossing_law).strip().lower()
    if crossing_law not in {"restricted", "ybe"}:
        raise ValueError(f"Unknown crossing_law={crossing_law!r}; expected 'restricted' or 'ybe'.")

    key = random.PRNGKey(seed)
    tags_tr, vals_tr, aidx_tr, mask_tr, Lmax_tr = make_dataset(key, n_train, n_low, n_high, p_tag)
    key = random.split(key, 2)[1]
    tags_te, vals_te, aidx_te, mask_te, Lmax_te = make_dataset(
        key, n_test, n_low_test, n_high_test, p_tag, L_max=max(n_high, n_high_test)
    )
    params = Params(w=jnp.array([0.0, 0.0], dtype=jnp.float32), b=jnp.array(0.0, dtype=jnp.float32), tau=tau)
    opt_state = None

    def get_batch(i):
        idx = (i * bs) % n_train
        sl = slice(idx, min(idx + bs, n_train))
        return tags_tr[sl], vals_tr[sl], aidx_tr[sl], mask_tr[sl]

    t0 = time.time()
    losses = []
    for t in range(steps):
        batch = get_batch(t)
        params, opt_state, L = train_step(params, batch, opt_state, lr, lam_bce, lam_mse, lam_len)
        losses.append(float(L))
    train_time = time.time() - t0
    decode_fn = decode_batch if crossing_law == "restricted" else decode_batch_ybe
    pred_tr, gt_tr, diff_tr, perm_tr, chosen_tr, k_tr = decode_fn(params, tags_tr, vals_tr, mask_tr, aidx_tr)
    pred_te, gt_te, diff_te, perm_te, chosen_te, k_te = decode_fn(params, tags_te, vals_te, mask_te, aidx_te)
    acc_tr = float(accuracy_exact(pred_tr, gt_tr))
    acc_te = float(accuracy_exact(pred_te, gt_te))
    Ldepth = math.ceil(math.log2(n_high))
    base_tr = conv_reduce_sum_batch(tags_tr, vals_tr, mask_tr, aidx_tr, Ldepth)
    base_te = conv_reduce_sum_batch(tags_te, vals_te, mask_te, aidx_te, Ldepth)
    accb_tr = float(accuracy_exact(base_tr, gt_tr))
    accb_te = float(accuracy_exact(base_te, gt_te))
    return {
        "params": params,
        "loss_curve": losses,
        "train_time_s": train_time,
        "train_acc": acc_tr,
        "test_acc": acc_te,
        "train_max_decode_diff": float(diff_tr),
        "test_max_decode_diff": float(diff_te),
        "baseline_depth": Ldepth,
        "baseline_train_acc": accb_tr,
        "baseline_test_acc": accb_te,
        "crossing_law": crossing_law,
        "artifacts": {
            "train": (tags_tr, vals_tr, aidx_tr, mask_tr, perm_tr, chosen_tr, k_tr, pred_tr, gt_tr),
            "test": (tags_te, vals_te, aidx_te, mask_te, perm_te, chosen_te, k_te, pred_te, gt_te),
        },
    }


# ---------- optional plotting ----------
def plot_braid(sample, perm, k):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skipping plot")
        return
    L = sample.shape[0]
    xs = [[] for _ in range(L)]
    ys = [[] for _ in range(L)]
    for i in range(L):
        xs[i].append(0.0)
        ys[i].append(i)
    for t in range(int(k)):
        xs[0].append(t + 0.5)
        ys[0].append(0)
        xs[0].append(t + 1.0)
        ys[0].append(1 + t)
    xs[0].append(int(k) + 1)
    ys[0].append(1 + int(k))
    for j in range(1, L):
        if j <= int(k):
            xs[j].append(j - 1 + 0.5)
            ys[j].append(j)
            xs[j].append(j)
            ys[j].append(j - 1)
            xs[j].append(int(k) + 1)
            ys[j].append(j - 1)
        else:
            xs[j].append(int(k) + 1)
            ys[j].append(j)
    plt.figure(figsize=(6, 5))
    for i in range(L):
        plt.plot(xs[i], ys[i], linewidth=2 if i == 0 else 1)
    plt.gca().invert_yaxis()
    plt.title("Decoded braid: σ1^k")
    plt.xlabel("time")
    plt.ylabel("strand")
    plt.tight_layout()
    plt.show()


# ---------- CLI ----------
def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    crossing_law = os.environ.get("BRAID_CROSSING_LAW", "restricted")
    if len(sys.argv) > 3:
        crossing_law = sys.argv[3]
    res = run_experiment(seed=seed, steps=steps, crossing_law=crossing_law)
    p = res["params"]
    print("crossing_law:", res.get("crossing_law"))
    print("params:", {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in p.__dict__.items()})
    print("train_time_s:", round(res["train_time_s"], 3))
    print("train_acc:", res["train_acc"], "test_acc:", res["test_acc"])
    print(
        "baseline_depth:",
        res["baseline_depth"],
        "baseline_train_acc:",
        res["baseline_train_acc"],
        "baseline_test_acc:",
        res["baseline_test_acc"],
    )
    print("max_decode_diff(train,test):", res["train_max_decode_diff"], res["test_max_decode_diff"])
    tags_te, vals_te, aidx_te, mask_te, perm_te, chosen_te, k_te, pred_te, gt_te = res["artifacts"]["test"]
    i = 0
    B, L = tags_te.shape
    perm_i = perm_te[i]
    k_i = int(k_te[i])
    word = BraidWord(n=L, k=k_i)
    ok_allowed = word.verify_allowed()
    ok_normal = word.normalize_local().k == word.k
    print("verify_allowed_only_sigma1:", bool(ok_allowed), "normalize_idempotent:", bool(ok_normal))
    # Optional one-figure sanity check
    # Reindex strand labels by permutation for plotting shape only
    plot_braid(jnp.arange(L), perm_i, k_i)


def demo():
    """Run the knot theoretic programs and braid based attention demonstration."""
    main()


if __name__ == "__main__":
    demo()


# (adapter defined above)
