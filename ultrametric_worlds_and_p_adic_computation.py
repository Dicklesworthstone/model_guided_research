"""
Ultrametric LCP‑Tree Attention (LTA) with Valuation‑Ordered Local Fix (VOLF).

Representations are base‑p digit strings (p‑adic integers). Distance is longest‑common‑prefix
(LCP) depth; balls are nested/disjoint, so the unique “nearest” item is the deepest occupied
ancestor in a p‑ary trie. Each node stores an aggregate S ∈ (Z_p)^m and tiny sign counters R.
Attention = read the deepest occupied ancestor and apply a per‑depth unit upper‑triangular map
U_d (1‑Lipschitz in the p‑adic norm), so coarse output digits depend only on equal‑or‑coarser
inputs; no dot products or softmax.

Learning replaces gradients with VOLF: compute residual e in (Z_p)^m and write it at the
shallowest ancestor whose counters do not oppose the change; else specialize deeper. Updates
touch only O(K) nodes along the path, cannot spill into disjoint balls, and are natively
quantized (mod p). Pruning is lossless: remove unused subtrees.

Complexity: per query/write O(HK) (H heads, K digits) ⇒ O(n log n) total. The file includes
two falsifiers: Task A (exact LCP retrieval) and Task B (leaf exceptions). Arithmetic uses JAX
arrays; the reference trie is explicit for clarity, while a production variant can use
contiguous per‑depth arrays and bitsets for cache‑optimal rank/test/lookup.
"""

import random
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


def p_pow(p, K):
    a = np.ones(K, dtype=np.int64)
    for i in range(1, K):
        a[i] = a[i - 1] * p
    return a


def mod_add(x, y, p):
    return (x + y) % p


def mod_sub(x, y, p):
    return (x - y) % p


def mod_balance(x, p):
    t = x % p
    half = p // 2
    return jnp.where(t > half, t - p, t)


def sign_int(x):
    return jnp.where(x > 0, 1, jnp.where(x < 0, -1, 0)).astype(jnp.int8)


@dataclass
class DepthArrays:
    res2idx: dict[int, int]
    residues: list[int]
    S: jnp.ndarray
    R: jnp.ndarray

    @staticmethod
    def empty(m):
        return DepthArrays({}, [], jnp.zeros((0, m), jnp.int32), jnp.zeros((0, m), jnp.int8))


class HeadTrie:
    def __init__(self, p, K, m, r, U_seed=None, superdiag=False):
        self.p, self.K, self.m, self.r = p, K, m, r
        self.pow = np.array(p_pow(p, K), dtype=np.int64)
        self.levels = [DepthArrays.empty(m) for _ in range(K)]
        key = jax.random.PRNGKey(0 if U_seed is None else int(U_seed))
        mats = []
        for _d in range(K):
            M = jnp.eye(m, dtype=jnp.int32)
            if superdiag and m > 1:
                k = jax.random.split(key, 1)[0]
                idxs = jax.random.randint(k, (m - 1,), 0, 2)
                M = M.at[jnp.arange(m - 1), jnp.arange(1, m)].add(idxs.astype(jnp.int32))
                key = jax.random.split(key, 1)[0]
            mats.append(M)
        self.U = mats

    def ensure(self, d, res):
        L = self.levels[d]
        if res in L.res2idx:
            return L.res2idx[res]
        idx = len(L.residues)
        L.res2idx[res] = idx
        L.residues.append(res)
        L.S = jnp.vstack([L.S, jnp.zeros((1, self.m), jnp.int32)])
        L.R = jnp.vstack([L.R, jnp.zeros((1, self.m), jnp.int8)])
        self.levels[d] = L
        return idx

    def deepest_occupied(self, digits):
        last = (-1, -1)
        r = 0
        for d, a in enumerate(digits):
            r += int(a) * int(self.pow[d])
            L = self.levels[d]
            if r in L.res2idx:
                last = (d, L.res2idx[r])
            else:
                break
        return last

    def path_residues(self, digits, upto=None):
        upto = self.K if upto is None else upto
        res = []
        r = 0
        for d in range(min(self.K, upto)):
            r += int(digits[d]) * int(self.pow[d])
            res.append(r)
        return res

    def read_contrib(self, digits):
        d, idx = self.deepest_occupied(digits)
        if d < 0:
            return jnp.zeros((self.m,), jnp.int32)
        L = self.levels[d]
        return (self.U[d] @ L.S[idx]) % self.p

    def compatible(self, R_vec, e_sign, maj_thresh):
        opp = (R_vec.astype(jnp.int32) != 0) & (jnp.sign(R_vec).astype(jnp.int32) == (-e_sign.astype(jnp.int32)))
        bad = opp & (jnp.abs(R_vec) >= maj_thresh)
        return not bool(jnp.any(bad))

    def volf_update(self, digits, y_star):
        y = self.read_contrib(digits)
        e = mod_sub(y_star, y, self.p)
        eb = mod_balance(e, self.p)
        if int(jnp.all(e == 0)):
            return 0
        e_sign = sign_int(eb)
        d_star, idx_star = self.deepest_occupied(digits)
        path_res = self.path_residues(digits, upto=d_star + 1 if d_star >= 0 else 0)
        chosen = None
        for d, res in enumerate(path_res):
            L = self.levels[d]
            idx = L.res2idx[res]
            if self.compatible(L.R[idx], e_sign, self.r):
                chosen = (d, idx)
                break
        created = 0
        if chosen is None:
            if d_star + 1 < self.K:
                res_next = self.path_residues(digits, upto=d_star + 2)[-1]
                idx_new = self.ensure(d_star + 1, res_next)
                chosen = (d_star + 1, idx_new)
                created = 1
            elif d_star >= 0:
                chosen = (d_star, idx_star)
            else:
                idx0 = self.ensure(0, self.path_residues(digits, upto=1)[-1])
                chosen = (0, idx0)
                created = 1
        d, idx = chosen
        L = self.levels[d]
        L.S = L.S.at[idx].set(mod_add(L.S[idx], e, self.p))
        L.R = L.R.at[idx].set(
            jnp.clip(L.R[idx].astype(jnp.int16) + e_sign.astype(jnp.int16), -self.r, self.r).astype(jnp.int8)
        )
        self.levels[d] = L
        return created


class LCPTreeAttention:
    def __init__(self, p=16, K=8, H=2, m=8, r=3, superdiag=False, seeds=None):
        self.p, self.K, self.H, self.m = p, K, H, m
        self.heads = [
            HeadTrie(p, K, m, r, U_seed=(None if seeds is None else seeds[h]), superdiag=superdiag) for h in range(H)
        ]

    def lookup(self, digits_batch):
        Y = jnp.zeros((len(digits_batch), self.m), jnp.int32)
        for h in range(self.H):
            ys = []
            for q in digits_batch:
                ys.append(self.heads[h].read_contrib(q))
            Y = mod_add(Y, jnp.stack(ys, 0), self.p)
        return Y % self.p

    def volf_step(self, q, y_star):
        created = 0
        y = self.lookup([q])[0]
        e = mod_sub(y_star, y, self.p)
        if int(jnp.all(e == 0)):
            return 0
        for h in range(self.H):
            created += self.heads[h].volf_update(q, y_star)
        return created

    def train_epoch(self, qs, ys, shuffle=True):
        idx = list(range(len(qs)))
        if shuffle:
            random.shuffle(idx)
        acc_cnt = 0
        created = 0
        for i in idx:
            y = self.lookup([qs[i]])[0]
            if int(jnp.all(y == ys[i])):
                acc_cnt += 1
            created += self.volf_step(qs[i], ys[i])
        return acc_cnt / len(qs), created

    def eval_acc(self, qs, ys):
        y = self.lookup(qs)
        eq = (y == ys).all(axis=1)
        return float(jnp.mean(eq.astype(jnp.float32)))


def sample_digits(N, K, p, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, p, size=(N, K), dtype=np.int32)


def lcp_residue(digits, d, pow_p):
    r = 0
    for i in range(d + 1):
        r += int(digits[i]) * int(pow_p[i])
    return r


def taskA_dataset(N_train, N_test, p, K, m, depth_probs=None, seed=0):
    rng = np.random.default_rng(seed)
    pow_p = np.array(p_pow(p, K), dtype=np.int64)
    if depth_probs is None:
        w = np.ones(K)
        w /= w.sum()
    else:
        w = np.array(depth_probs)
        w = w / w.sum()
    keys = sample_digits(N_train + N_test, K, p, seed + 1)
    Ymap = {}

    def rand_label():
        return jnp.array(rng.integers(0, p, size=(m,), dtype=np.int32))

    qs = []
    ys = []
    for i in range(N_train):
        k = keys[i]
        D = int(rng.choice(K, p=w))
        r = lcp_residue(k, D, pow_p)
        if (D, r) not in Ymap:
            Ymap[(D, r)] = rand_label()
        q = k.copy()
        if D + 1 < K:
            q[D + 1 :] = rng.integers(0, p, size=(K - (D + 1),), dtype=np.int32)
        qs.append(jnp.array(q))
        ys.append(Ymap[(D, r)])
    qst = []
    yst = []
    for i in range(N_test):
        k = keys[N_train + i]
        D = int(rng.choice(K, p=w))
        r = lcp_residue(k, D, pow_p)
        if (D, r) not in Ymap:
            Ymap[(D, r)] = rand_label()
        q = k.copy()
        if D + 1 < K:
            q[D + 1 :] = rng.integers(0, p, size=(K - (D + 1),), dtype=np.int32)
        qst.append(jnp.array(q))
        yst.append(Ymap[(D, r)])
    return qs, ys, qst, yst


def taskB_dataset(N_train, N_test, p, K, m, epsilon=0.01, seed=0):
    rng = np.random.default_rng(seed)
    pow_p = np.array(p_pow(p, K), dtype=np.int64)
    keys = sample_digits(N_train + N_test, K, p, seed + 2)
    Ynode = {}

    def rand_label():
        return jnp.array(rng.integers(0, p, size=(m,), dtype=np.int32))

    for i in range(N_train + N_test):
        k = keys[i]
        for d in range(K - 1):
            r = lcp_residue(k, d, pow_p)
            if (d, r) not in Ynode:
                Ynode[(d, r)] = rand_label()
    leaf_overrides = set()
    n_leaves = min(N_train + N_test, max(1, int(epsilon * (N_train + N_test))))
    idxs = rng.choice(N_train + N_test, size=n_leaves, replace=False)
    for i in idxs:
        k = keys[i]
        r = lcp_residue(k, K - 1, pow_p)
        leaf_overrides.add(r)
    qs, ys, qst, yst = [], [], [], []
    for i in range(N_train):
        k = keys[i]
        D = rng.integers(0, K - 1)
        r = lcp_residue(k, D, pow_p)
        y = Ynode[(D, r)]
        if r in leaf_overrides and D == K - 1:
            y = (y + 1) % p
        q = k.copy()
        if D + 1 < K:
            q[D + 1 :] = rng.integers(0, p, size=(K - (D + 1),), dtype=np.int32)
        qs.append(jnp.array(q))
        ys.append(y)
    for i in range(N_test):
        k = keys[N_train + i]
        D = rng.integers(0, K - 1)
        r = lcp_residue(k, D, pow_p)
        y = Ynode[(D, r)]
        if r in leaf_overrides and D == K - 1:
            y = (y + 1) % p
        q = k.copy()
        if D + 1 < K:
            q[D + 1 :] = rng.integers(0, p, size=(K - (D + 1),), dtype=np.int32)
        qst.append(jnp.array(q))
        yst.append(y)
    return qs, ys, qst, yst


def run_task_A():
    p, K, H, m = 16, 8, 2, 8
    qs, ys, qst, yst = taskA_dataset(20000, 4000, p, K, m, seed=1)
    model = LCPTreeAttention(p=p, K=K, H=H, m=m, r=5, superdiag=True, seeds=list(range(H)))
    t0 = time.time()
    acc0 = model.eval_acc(qs, ys)
    t1 = time.time()
    acc_train, created = model.train_epoch(qs, ys, shuffle=False)
    t2 = time.time()
    acc_test = model.eval_acc(qst, yst)
    t3 = time.time()
    print(
        f"Task A: train_acc_pre={acc0:.4f} train_acc_post={acc_train:.4f} test_acc={acc_test:.4f} created_nodes={created}"
    )
    print(f"Timing(s): eval_pre={t1 - t0:.3f} train={t2 - t1:.3f} eval_test={t3 - t2:.3f}")


def run_task_B():
    p, K, H, m = 16, 8, 2, 8
    qs, ys, qst, yst = taskB_dataset(20000, 4000, p, K, m, epsilon=0.02, seed=3)
    model = LCPTreeAttention(p=p, K=K, H=H, m=m, r=5, superdiag=True, seeds=[7, 11])
    t0 = time.time()
    acc0 = model.eval_acc(qs, ys)
    t1 = time.time()
    acc_train, created = model.train_epoch(qs, ys, shuffle=False)
    t2 = time.time()
    acc_test = model.eval_acc(qst, yst)
    t3 = time.time()
    node_count = sum(len(h.levels[d].residues) for h in model.heads for d in range(K))
    print(
        f"Task B: train_acc_pre={acc0:.4f} train_acc_post={acc_train:.4f} test_acc={acc_test:.4f} created_nodes={created} total_nodes={node_count}"
    )
    print(f"Timing(s): eval_pre={t1 - t0:.3f} train={t2 - t1:.3f} eval_test={t3 - t2:.3f}")


def smoke_demo_small():
    p, K, H, m = 8, 6, 2, 6
    qs, ys, qst, yst = taskA_dataset(4000, 1000, p, K, m, seed=9)
    model = LCPTreeAttention(p=p, K=K, H=H, m=m, r=4, superdiag=True, seeds=[3, 5])
    acc0 = model.eval_acc(qs, ys)
    acc_train, created = model.train_epoch(qs, ys, shuffle=True)
    acc_test = model.eval_acc(qst, yst)
    print(f"Small A: pre={acc0:.3f} post={acc_train:.3f} test={acc_test:.3f} created={created}")


def demo():
    """Run all ultrametric demonstrations."""
    random.seed(0)
    np.random.seed(0)
    smoke_demo_small()
    run_task_A()
    run_task_B()


if __name__ == "__main__":
    demo()


# --- Minimal p‑adic helpers for tests ---

def p_adic_encode(n: int, p: int, precision: int) -> np.ndarray:
    """Encode integer n modulo p^precision as base‑p digits least significant first."""
    n_mod = n % (p ** precision)
    digits = []
    for _ in range(precision):
        digits.append(n_mod % p)
        n_mod //= p
    return np.array(digits, dtype=np.int32)


def p_adic_decode(digits: np.ndarray, p: int) -> int:
    """Decode base‑p digits (LSB first) back to integer modulo p^k."""
    val = 0
    mul = 1
    for d in digits.astype(int):
        val += int(d) * mul
        mul *= p
    return int(val)


def p_adic_add(a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    carry = 0
    out = np.zeros_like(a)
    for i in range(len(a)):
        s = int(a[i]) + int(b[i]) + carry
        out[i] = s % p
        carry = s // p
    return out


def p_adic_multiply(a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    # Compute via integer multiply modulo p^k and re-encode to handle carries correctly
    k = len(a)
    n1 = p_adic_decode(a, p)
    n2 = p_adic_decode(b, p)
    mod = p ** k
    prod = (n1 * n2) % mod
    return p_adic_encode(prod, p, k)
