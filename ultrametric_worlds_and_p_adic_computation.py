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

# Docs: markdown_documentation/ultrametric_worlds_and_p_adic_computation.md (this file is the reference trie; the doc
# also sketches a cache-opt production layout with per-depth bitsets + rank/select).

import os
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


def _np_sign_int(x: np.ndarray) -> np.ndarray:
    """Elementwise sign for integer arrays: returns -1, 0, 1 as int8."""
    return np.where(x > 0, 1, np.where(x < 0, -1, 0)).astype(np.int8)


def _np_mod_balance(x: np.ndarray, p: int) -> np.ndarray:
    """Map residues mod p into a roughly symmetric representative set around 0."""
    t = np.asarray(x, dtype=np.int32) % int(p)
    half = int(p) // 2
    return np.where(t > half, t - int(p), t).astype(np.int32)


def _make_unit_upper_mats(p: int, K: int, m: int, *, U_seed: int | None, superdiag: bool) -> list[np.ndarray]:
    """Match HeadTrie U init, but return NumPy arrays (for packed mode)."""
    key = jax.random.PRNGKey(0 if U_seed is None else int(U_seed))
    mats: list[np.ndarray] = []
    for _d in range(int(K)):
        M = np.eye(int(m), dtype=np.int32)
        if superdiag and m > 1:
            subkey = jax.random.split(key, 1)[0]
            idxs = np.asarray(jax.random.randint(subkey, (m - 1,), 0, 2), dtype=np.int32)
            M[np.arange(m - 1), np.arange(1, m)] += idxs
            key = jax.random.split(key, 1)[0]
        mats.append(M % int(p))
    return mats


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


class PackedHeadTrie:
    """Packed, cache-friendlier p-ary trie (no Python dicts).

    Practical stand-in for the doc's "bitset + rank/select" layout:
    - Per-depth contiguous arrays for S and R.
    - Per-node p-bit occupancy mask + dense child table (uint32 + int32).
    - Lookup/update touch O(K) contiguous rows without dict/list pointer chasing.
    """

    def __init__(self, p: int, K: int, m: int, r: int, U_seed: int | None = None, superdiag: bool = False):
        self.p, self.K, self.m, self.r = int(p), int(K), int(m), int(r)
        self.U = _make_unit_upper_mats(self.p, self.K, self.m, U_seed=U_seed, superdiag=superdiag)

        self._root = np.full((self.p,), -1, dtype=np.int32)
        self._root_mask = np.uint32(0)

        self._size = [0 for _ in range(self.K)]
        self._cap = [max(4, self.p) for _ in range(self.K)]
        self._S = [np.zeros((cap, self.m), dtype=np.int32) for cap in self._cap]
        self._R = [np.zeros((cap, self.m), dtype=np.int8) for cap in self._cap]
        self._child = [np.full((cap, self.p), -1, dtype=np.int32) for cap in self._cap[:-1]]
        self._child_mask = [np.zeros((cap,), dtype=np.uint32) for cap in self._cap[:-1]]

    def _grow_depth(self, d: int) -> None:
        new_cap = int(self._cap[d]) * 2
        size = int(self._size[d])

        S_new = np.zeros((new_cap, self.m), dtype=np.int32)
        R_new = np.zeros((new_cap, self.m), dtype=np.int8)
        if size:
            S_new[:size] = self._S[d][:size]
            R_new[:size] = self._R[d][:size]
        self._S[d] = S_new
        self._R[d] = R_new
        self._cap[d] = new_cap

        if d < self.K - 1:
            child_new = np.full((new_cap, self.p), -1, dtype=np.int32)
            mask_new = np.zeros((new_cap,), dtype=np.uint32)
            if size:
                child_new[:size] = self._child[d][:size]
                mask_new[:size] = self._child_mask[d][:size]
            self._child[d] = child_new
            self._child_mask[d] = mask_new

    def _alloc_node(self, d: int) -> int:
        if self._size[d] >= self._cap[d]:
            self._grow_depth(d)
        idx = int(self._size[d])
        self._size[d] += 1
        return idx

    def _ensure_root_child(self, digit0: int) -> int:
        digit0 = int(digit0)
        idx = int(self._root[digit0])
        if idx >= 0:
            return idx
        idx = self._alloc_node(0)
        self._root[digit0] = np.int32(idx)
        self._root_mask |= np.uint32(1 << digit0)
        return idx

    def _ensure_child(self, d: int, parent_idx: int, digit: int) -> int:
        digit = int(digit)
        parent_idx = int(parent_idx)
        nxt = int(self._child[d][parent_idx, digit])
        if nxt >= 0:
            return nxt
        nxt = self._alloc_node(d + 1)
        self._child[d][parent_idx, digit] = np.int32(nxt)
        self._child_mask[d][parent_idx] |= np.uint32(1 << digit)
        return nxt

    def _path_existing(self, digits: np.ndarray) -> list[tuple[int, int]]:
        if self.K <= 0:
            return []
        digits = np.asarray(digits, dtype=np.int32)
        if digits.shape[0] < self.K:
            raise ValueError(f"Expected K={self.K} digits, got shape {digits.shape}")

        digit0 = int(digits[0])
        idx0 = int(self._root[digit0])
        if idx0 < 0:
            return []

        path: list[tuple[int, int]] = [(0, idx0)]
        idx = idx0
        for d in range(0, self.K - 1):
            nxt = int(self._child[d][idx, int(digits[d + 1])])
            if nxt < 0:
                break
            path.append((d + 1, nxt))
            idx = nxt
        return path

    def deepest_occupied(self, digits: np.ndarray) -> tuple[int, int]:
        path = self._path_existing(digits)
        return (-1, -1) if not path else path[-1]

    def read_contrib(self, digits: np.ndarray) -> jnp.ndarray:
        d, idx = self.deepest_occupied(digits)
        if d < 0:
            return jnp.zeros((self.m,), jnp.int32)
        s = self._S[d][idx]
        y = (self.U[d] @ s) % self.p
        return jnp.asarray(y, dtype=jnp.int32)

    def compatible(self, R_vec: np.ndarray, e_sign: np.ndarray, maj_thresh: int) -> bool:
        R_vec = np.asarray(R_vec, dtype=np.int8)
        e_sign = np.asarray(e_sign, dtype=np.int8)
        opp = (R_vec != 0) & (np.sign(R_vec).astype(np.int8) == (-e_sign).astype(np.int8))
        bad = opp & (np.abs(R_vec).astype(np.int32) >= int(maj_thresh))
        return not bool(np.any(bad))

    def volf_update(self, digits: np.ndarray, y_star: jnp.ndarray) -> int:
        digits_np = np.asarray(digits, dtype=np.int32)
        y_star_np = np.asarray(y_star, dtype=np.int32)

        path = self._path_existing(digits_np)
        if path:
            d_star, idx_star = path[-1]
            s = self._S[d_star][idx_star]
            y = (self.U[d_star] @ s) % self.p
        else:
            d_star, idx_star = -1, -1
            y = np.zeros((self.m,), dtype=np.int32)

        e = (y_star_np - y) % self.p
        if bool(np.all(e == 0)):
            return 0
        e_sign = _np_sign_int(_np_mod_balance(e, self.p))

        chosen: tuple[int, int] | None = None
        for d, idx in path:
            if self.compatible(self._R[d][idx], e_sign, self.r):
                chosen = (d, idx)
                break

        created = 0
        if chosen is None:
            if d_star < 0:
                idx0 = self._ensure_root_child(int(digits_np[0]))
                chosen = (0, idx0)
                created = 1
            elif d_star + 1 < self.K:
                nxt = self._ensure_child(d_star, idx_star, int(digits_np[d_star + 1]))
                chosen = (d_star + 1, nxt)
                created = 1
            else:
                chosen = (d_star, idx_star)

        d, idx = chosen
        self._S[d][idx] = (self._S[d][idx] + e) % self.p
        self._R[d][idx] = np.clip(self._R[d][idx].astype(np.int16) + e_sign.astype(np.int16), -self.r, self.r).astype(
            np.int8
        )
        return created


class LCPTreeAttention:
    def __init__(self, p=16, K=8, H=2, m=8, r=3, superdiag=False, seeds=None, packed: bool = False):
        self.p, self.K, self.H, self.m = p, K, H, m
        self.packed = bool(packed)
        head_cls = PackedHeadTrie if self.packed else HeadTrie
        self.heads = [head_cls(p, K, m, r, U_seed=(None if seeds is None else seeds[h]), superdiag=superdiag) for h in range(H)]

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
        ys_arr = ys if isinstance(ys, jnp.ndarray) else jnp.stack(list(ys), axis=0)
        eq = (y == ys_arr).all(axis=1)
        return float(jnp.mean(eq.astype(jnp.float32)))


# --- Minimal adapter expected by tests ---
# (adapter defined below)


# ---------------------------
# Test adapter: UltrametricAttention
# ---------------------------


from typing import cast


class UltrametricAttention:
    """Approximate ultrametric attention via LSH-based LCP trie.

    - Builds a binary prefix tree over random hyperplane signatures of keys.
    - Insert: O(D) where D = `max_depth` bits.
    - Attend: descend to deepest non-empty prefix; select the best candidate
      in that bucket by cosine similarity. Expected O(D) with tiny buckets.

    This keeps the core idea (prefix-tree lookups) while remaining NumPy-only
    and CPU-friendly for tests.
    """

    def __init__(self, dim: int, p: int = 5, max_depth: int = 10, packed: bool = False, heads: int = 1):
        import numpy as _np
        self.dim = int(dim)
        self.max_depth = int(max_depth)
        self._packed = bool(packed)
        self._heads = max(1, int(heads))
        # Random hyperplanes define a binary signature per depth
        rng = _np.random.default_rng(0)
        self._planes = [
            rng.standard_normal((self.max_depth, self.dim)).astype(_np.float64)
            for _ in range(self._heads)
        ]
        # Buckets per head: dict or array-backed by depth depending on mode
        # Buckets type: packed -> list[ list[ dict[int, list[int]] ] ] or array mode -> list[ list[ list[list[int]] ] ]
        # Unpacked -> list[ dict[tuple[int,...], list[int]] ]
        BuckPacked = list[list[dict[int, list[int]]]]
        BuckPackedArrays = list[list[list[list[int]]]]
        BuckUnpacked = list[dict[tuple[int, ...], list[int]]]
        self._buckets: BuckPackedArrays | BuckPacked | BuckUnpacked
        self._packed_arrays = False
        if self._packed and bool(int(os.environ.get("ULTRA_PACKED_ARRAYS", "0"))):
            # Array-of-lists per level, indexable by code with O(1) access
            packed_arr: BuckPackedArrays = []
            for _ in range(self._heads):
                levels_ll: list[list[list[int]]] = []
                for d in range(self.max_depth + 1):
                    size = 1 << d
                    levels_ll.append([[] for _ in range(size)])
                packed_arr.append(levels_ll)
            self._buckets = packed_arr
            self._packed_arrays = True
            # Occupancy bitsets to summarize
            self._occ = [[np.zeros((1 << d,), dtype=np.uint8) for d in range(self.max_depth + 1)] for _ in range(self._heads)]
        elif self._packed:
            # dict-based packed
            packed_buckets: BuckPacked = []
            for _ in range(self._heads):
                levels: list[dict[int, list[int]]] = []
                for __ in range(self.max_depth + 1):
                    levels.append(cast(dict[int, list[int]], {}))
                packed_buckets.append(levels)
            self._buckets = packed_buckets
        else:
            unpacked: BuckUnpacked = []
            for _ in range(self._heads):
                unpacked.append(cast(dict[tuple[int, ...], list[int]], {}))
            self._buckets = unpacked
        # Store keys by index for quick similarity checks
        self._key_vec: dict[int, _np.ndarray] = {}

    @staticmethod
    def _signature(planes, vec):
        import numpy as _np
        proj = planes @ vec  # [D]
        return (proj > 0.0).astype(_np.int8)

    def _prefixes(self, bits):
        for d in range(1, len(bits) + 1):
            yield tuple(int(b) for b in bits[:d])

    def insert(self, idx: int, key_vec):
        import numpy as _np
        v = _np.asarray(key_vec, dtype=_np.float64)
        self._key_vec[int(idx)] = v
        for h in range(self._heads):
            sig = self._signature(self._planes[h], v)
            if self._packed:
                code = 0
                for d in range(1, self.max_depth + 1):
                    code = (code << 1) | int(sig[d - 1])
                    if self._packed_arrays:
                        buckets_pa = cast(list[list[list[list[int]]]], self._buckets)
                        buckets_pa[h][d][code].append(int(idx))
                        self._occ[h][d][code] = 1
                    else:
                        buckets_p = cast(list[list[dict[int, list[int]]]], self._buckets)
                        level = buckets_p[h][d]
                        level.setdefault(code, []).append(int(idx))
            else:
                for pref in self._prefixes(sig):
                    buckets_u = cast(list[dict[tuple[int, ...], list[int]]], self._buckets)
                    bucket = buckets_u[h].setdefault(pref, [])
                    bucket.append(int(idx))

    def attend(self, q, V):
        import numpy as _np
        if not self._key_vec:
            return _np.zeros_like(V[0])
        Q = _np.asarray(q, dtype=_np.float64)
        picks = []
        sims = []
        qn = _np.linalg.norm(Q) + 1e-12
        ULTRA_FUSE = bool(int(os.environ.get("ULTRA_FUSE", "0")))
        for h in range(self._heads):
            sig = self._signature(self._planes[h], Q)
            # Find deepest non-empty bucket
            candidate_idxs: list[int] = []
            if self._packed:
                code = 0
                for d in range(self.max_depth, 0, -1):
                    code = (code << 1) | int(sig[d - 1])
                    if self._packed_arrays:
                        buckets_pa = cast(list[list[list[list[int]]]], self._buckets)
                        lst = buckets_pa[h][d][code] if (d < len(buckets_pa[h]) and code < len(buckets_pa[h][d])) else []
                    else:
                        buckets_p = cast(list[list[dict[int, list[int]]]], self._buckets)
                        level = buckets_p[h][d] if d < len(buckets_p[h]) else {}
                        lst = level.get(code, [])
                    if lst:
                        candidate_idxs = lst
                        break
            else:
                for d in range(self.max_depth, 0, -1):
                    pref = tuple(int(b) for b in sig[:d])
                    buckets_u = cast(list[dict[tuple[int, ...], list[int]]], self._buckets)
                    if pref in buckets_u[h] and buckets_u[h][pref]:
                        candidate_idxs = buckets_u[h][pref]
                        break
            if not candidate_idxs:
                candidate_idxs = list(self._key_vec.keys())
            best_j = None
            best_sim = -_np.inf
            for j in candidate_idxs:
                kv = self._key_vec[j]
                sim = float((kv @ Q) / ((_np.linalg.norm(kv) + 1e-12) * qn))
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            picks.append(int(best_j))
            sims.append(best_sim)
        # Aggregate across heads
        if ULTRA_FUSE:
            # Fuse by selecting value whose index maximizes sum of sims (ultrametric sum proxy)
            # Build candidate set from picked indices and choose argmax over summed sims
            idxs = list(set(picks))
            sim_sum = []
            for j in idxs:
                total = 0.0
                for h in range(self._heads):
                    # reuse head sims approximatively: if pick equals j, use best_sim else penalize
                    total += sims[h] if picks[h] == j else (sims[h] - 1e-3)
                sim_sum.append((total, j))
            j_best = max(sim_sum, key=lambda t: t[0])[1]
            out = V[int(j_best)]
        else:
            out = np.mean([V[int(j)] for j in picks], axis=0)
        # Store head sims for variance reporting
        try:
            self.last_head_sims = sims  # type: ignore[attr-defined]
        except Exception as err:
            print(f"[ultrametric] Could not cache head sims: {err}")
        return _np.asarray(out)

    # --- Packed arrays helpers: finalize + rank/test ---
    def finalize(self):
        """Build per-level prefix sums for O(1) rank/test in array-packed mode."""
        if not getattr(self, "_packed_arrays", False):
            return
        # Build prefix sums of occupancy for each head/level
        self._occ_psum = []  # list[ list[np.ndarray] ]
        for h in range(self._heads):
            levels_ps = []
            for d in range(self.max_depth + 1):
                occ = self._occ[h][d]
                ps = np.cumsum(occ, axis=0)
                levels_ps.append(ps)
            self._occ_psum.append(levels_ps)

    def has_prefix(self, head: int, depth: int, code: int) -> bool:
        if getattr(self, "_packed_arrays", False):
            if head < 0 or head >= self._heads or depth < 0 or depth > self.max_depth:
                return False
            size = len(self._occ[head][depth])
            if code < 0 or code >= size:
                return False
            return bool(self._occ[head][depth][code])
        # Fallback: dict-based packed
        try:
            buckets_p = cast(list[list[dict[int, list[int]]]], self._buckets)
            return code in buckets_p[head][depth]
        except Exception as err:
            print(f"[ultrametric] has_prefix fallback failed: {err}")
            return False

    def rank_prefix(self, head: int, depth: int, code: int) -> int:
        """Return number of occupied codes <= code at given (head, depth)."""
        if getattr(self, "_packed_arrays", False) and hasattr(self, "_occ_psum"):
            if head < 0 or head >= self._heads or depth < 0 or depth > self.max_depth:
                return 0
            size = len(self._occ_psum[head][depth])
            if code < 0:
                return 0
            if code >= size:
                return int(self._occ_psum[head][depth][-1])
            return int(self._occ_psum[head][depth][code])
        # Fallback for dict-based packed (O(K))
        try:
            buckets_p = cast(list[list[dict[int, list[int]]]], self._buckets)
            cnt = 0
            for k in buckets_p[head][depth].keys():
                if int(k) <= int(code):
                    cnt += 1
            return int(cnt)
        except Exception as err:
            print(f"[ultrametric] rank_prefix fallback failed: {err}")
            return 0


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
        # Include leaves so exceptions are actually exercised.
        D = int(rng.integers(0, K))
        if D < K - 1:
            r = lcp_residue(k, D, pow_p)
            y = Ynode[(D, r)]
        else:
            r_leaf = lcp_residue(k, K - 1, pow_p)
            r_parent = lcp_residue(k, K - 2, pow_p)
            y = Ynode[(K - 2, r_parent)]
            if r_leaf in leaf_overrides:
                y = (y + 1) % p
        q = k.copy()
        if D + 1 < K:
            q[D + 1 :] = rng.integers(0, p, size=(K - (D + 1),), dtype=np.int32)
        qs.append(jnp.array(q))
        ys.append(y)
    for i in range(N_test):
        k = keys[N_train + i]
        D = int(rng.integers(0, K))
        if D < K - 1:
            r = lcp_residue(k, D, pow_p)
            y = Ynode[(D, r)]
        else:
            r_leaf = lcp_residue(k, K - 1, pow_p)
            r_parent = lcp_residue(k, K - 2, pow_p)
            y = Ynode[(K - 2, r_parent)]
            if r_leaf in leaf_overrides:
                y = (y + 1) % p
        q = k.copy()
        if D + 1 < K:
            q[D + 1 :] = rng.integers(0, p, size=(K - (D + 1),), dtype=np.int32)
        qst.append(jnp.array(q))
        yst.append(y)
    return qs, ys, qst, yst


def run_task_A(*, packed: bool = False):
    p, K, H, m = 16, 8, 1, 8
    qs, ys, qst, yst = taskA_dataset(20000, 4000, p, K, m, seed=1)
    model = LCPTreeAttention(p=p, K=K, H=H, m=m, r=5, superdiag=False, seeds=list(range(H)), packed=packed)
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


def run_task_B(*, packed: bool = False):
    p, K, H, m = 16, 8, 1, 8
    qs, ys, qst, yst = taskB_dataset(20000, 4000, p, K, m, epsilon=0.02, seed=3)
    model = LCPTreeAttention(p=p, K=K, H=H, m=m, r=5, superdiag=False, seeds=[7], packed=packed)
    t0 = time.time()
    acc0 = model.eval_acc(qs, ys)
    t1 = time.time()
    acc_train, created = model.train_epoch(qs, ys, shuffle=False)
    t2 = time.time()
    acc_test = model.eval_acc(qst, yst)
    t3 = time.time()
    if packed:
        node_count = sum(int(h._size[d]) for h in model.heads for d in range(K))  # type: ignore[attr-defined]
    else:
        node_count = sum(len(h.levels[d].residues) for h in model.heads for d in range(K))  # type: ignore[attr-defined]
    print(
        f"Task B: train_acc_pre={acc0:.4f} train_acc_post={acc_train:.4f} test_acc={acc_test:.4f} created_nodes={created} total_nodes={node_count}"
    )
    print(f"Timing(s): eval_pre={t1 - t0:.3f} train={t2 - t1:.3f} eval_test={t3 - t2:.3f}")


def compare_packed_vs_reference(
    *, task: str = "A", seed: int = 0, n_train: int = 3000, n_test: int = 800
) -> dict[str, float]:
    """Small parity + timing sanity check (packed vs reference) for Tasks A/B."""
    p, K, H, m = 16, 8, 1, 8
    if task.upper() == "A":
        qs, ys, qst, yst = taskA_dataset(n_train, n_test, p, K, m, seed=1 + seed)
    elif task.upper() == "B":
        qs, ys, qst, yst = taskB_dataset(n_train, n_test, p, K, m, epsilon=0.02, seed=3 + seed)
    else:
        raise ValueError("task must be 'A' or 'B'")

    # Keep U seeds identical across modes.
    head_seeds = [7]
    ref = LCPTreeAttention(p=p, K=K, H=H, m=m, r=5, superdiag=False, seeds=head_seeds, packed=False)
    packed_model = LCPTreeAttention(p=p, K=K, H=H, m=m, r=5, superdiag=False, seeds=head_seeds, packed=True)

    def _run(model: LCPTreeAttention) -> tuple[float, float, float]:
        t0 = time.perf_counter()
        acc0 = model.eval_acc(qs, ys)
        _acc_train, _ = model.train_epoch(qs, ys, shuffle=False)
        acc_test = model.eval_acc(qst, yst)
        t1 = time.perf_counter()
        return float(acc0), float(acc_test), float(t1 - t0)

    acc0_ref, acc_test_ref, t_ref = _run(ref)
    acc0_p, acc_test_p, t_packed = _run(packed_model)

    y_ref = ref.lookup(qst)
    y_packed = packed_model.lookup(qst)
    if not bool(jnp.all(y_ref == y_packed)):
        raise AssertionError("Packed/reference lookup mismatch on test split")

    return {
        "acc0_ref": acc0_ref,
        "acc_test_ref": acc_test_ref,
        "time_ref_s": t_ref,
        "acc0_packed": acc0_p,
        "acc_test_packed": acc_test_p,
        "time_packed_s": t_packed,
    }


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
    use_packed = bool(int(os.environ.get("ULTRA_PACKED", "0")))
    smoke_demo_small()
    run_task_A(packed=use_packed)
    run_task_B(packed=use_packed)
    try:
        outA = compare_packed_vs_reference(task="A", seed=0)
        outB = compare_packed_vs_reference(task="B", seed=0)
        try:
            from rich.console import Console as _Console
            from rich.table import Table as _Table

            tab = _Table(title="Packed vs Reference (Tasks A/B, small)", show_header=True, header_style="bold magenta")
            tab.add_column("Task")
            tab.add_column("acc0 ref/packed", justify="right")
            tab.add_column("acc_test ref/packed", justify="right")
            tab.add_column("time(s) ref/packed", justify="right")
            tab.add_row(
                "A",
                f"{outA['acc0_ref']:.3f}/{outA['acc0_packed']:.3f}",
                f"{outA['acc_test_ref']:.3f}/{outA['acc_test_packed']:.3f}",
                f"{outA['time_ref_s']:.3f}/{outA['time_packed_s']:.3f}",
            )
            tab.add_row(
                "B",
                f"{outB['acc0_ref']:.3f}/{outB['acc0_packed']:.3f}",
                f"{outB['acc_test_ref']:.3f}/{outB['acc_test_packed']:.3f}",
                f"{outB['time_ref_s']:.3f}/{outB['time_packed_s']:.3f}",
            )
            _Console().print(tab)
        except Exception as err:
            print(f"[ultrametric] Packed/reference table skipped: {err}")
    except Exception as err:
        print(f"[ultrametric] Packed/reference sanity check failed: {err}")
    # Optional packed timing benchmark to n=4096
    try:
        import time as _time
        print("\n[Packed LCP Timing]")
        Ns = [64, 256, 1024, 4096]
        insert_ms, query_ms, head_vars = [], [], []
        # Optional compare packed vs dict mode
        compare = bool(int(os.environ.get("ULTRA_SCALE_COMPARE", "0")))
        for N in Ns:
            dim = 32
            U = UltrametricAttention(dim=dim, p=5, max_depth=16, packed=True, heads=2)
            keys = np.random.randn(N, dim)
            vals = np.random.randn(N, dim)
            t0 = _time.perf_counter()
            for i in range(N):
                U.insert(i, keys[i])
            t1 = _time.perf_counter()
            q = np.random.randn(dim)
            _ = U.attend(q, vals)
            t2 = _time.perf_counter()
            var_heads = np.var(np.array(getattr(U, 'last_head_sims', [0.0])), ddof=1) if hasattr(U, 'last_head_sims') else 0.0
            print(f"N={N:4d} | insert {1000*(t1-t0):6.1f} ms | query {1000*(t2-t1):6.1f} ms | head var {var_heads:.3e}")
            insert_ms.append(1000 * (t1 - t0))
            query_ms.append(1000 * (t2 - t1))
            head_vars.append(float(var_heads))
        # Tiny scaling sparkline for query times
        def _spark(vals):
            bars = "▁▂▃▄▅▆▇█"
            if not vals:
                return ""
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-12:
                return bars[0] * len(vals)
            idxs = [int((v - lo) / (hi - lo) * (len(bars) - 1)) for v in vals]
            return "".join(bars[i] for i in idxs)
        print("query(ms) spark:", _spark(query_ms))
        if compare:
            # Dict-backed timing for comparison (packed=False)
            ins2, qry2 = [], []
            for N in Ns:
                dim = 32
                U = UltrametricAttention(dim=dim, p=5, max_depth=16, packed=False, heads=2)
                keys = np.random.randn(N, dim)
                vals = np.random.randn(N, dim)
                t0 = _time.perf_counter()
                for i in range(N):
                    U.insert(i, keys[i])
                t1 = _time.perf_counter()
                q = np.random.randn(dim)
                _ = U.attend(q, vals)
                t2 = _time.perf_counter()
                ins2.append(1000 * (t1 - t0))
                qry2.append(1000 * (t2 - t1))
            try:
                from rich.console import Console as _Console
                from rich.table import Table as _Table
                ct = _Table(title="Packed vs Dict Scaling (query ms)", show_header=True, header_style="bold magenta")
                ct.add_column("N")
                ct.add_column("packed")
                ct.add_column("dict")
                for i, N in enumerate(Ns):
                    ct.add_row(str(N), f"{query_ms[i]:.1f}", f"{qry2[i]:.1f}")
                _Console().print(ct)
            except Exception as err:
                print(f"[ultrametric] Skipping scaling table: {err}")
        # Occupancy summary per level when array-packed
        if bool(int(os.environ.get("ULTRA_PACKED_ARRAYS", "0"))):
            p, K, H = 5, 12, 2
            os.environ.setdefault("ULTRA_PACKED_ARRAYS", "1")
            U2 = UltrametricAttention(dim=32, p=p, max_depth=K, packed=True, heads=H)
            for i in range(512):
                U2.insert(i, np.random.randn(32))
            for h in range(H):
                occ = [float(np.mean(U2._occ[h][d])) for d in range(1, K + 1)]
                print(f"head {h} occupancy per level:", [round(x, 3) for x in occ])
            # Build prefix sums and demonstrate O(1) rank/test
            U2.finalize()
            d_demo = K // 2
            code_demo = (1 << d_demo) // 2
            print("rank_prefix demo:", U2.rank_prefix(0, d_demo, code_demo), "has_prefix:", U2.has_prefix(0, d_demo, code_demo))
            # Rank/Test summary table
            try:
                from rich.console import Console as _Console
                from rich.table import Table as _Table
                tab = _Table(title="Rank/Test Summary (array-packed)", show_header=True, header_style="bold magenta")
                tab.add_column("depth")
                tab.add_column("code")
                tab.add_column("rank")
                tab.add_column("has")
                demos = [(d_demo-1, (1<<(d_demo-1))//2), (d_demo, code_demo), (d_demo+1, min((1<<(d_demo+1))//2, (1<<(K))-1))]
                for dd, cc in demos:
                    rnk = U2.rank_prefix(0, max(1, dd), int(cc))
                    has = U2.has_prefix(0, max(1, dd), int(cc))
                    tab.add_row(str(int(dd)), str(int(cc)), str(int(rnk)), str(bool(has)))
                _Console().print(tab)
            except Exception as err:
                print(f"[ultrametric] Skipping rank/test table: {err}")
        # p-tuner (optional): try p∈{3,5,7} on a tiny Task A split and choose best by eval acc
        if bool(int(os.environ.get("ULTRA_TUNE_P", "0"))):
            best_p = None
            best_acc = -1.0
            for p_try in [3, 5, 7]:
                qs, ys, qst, yst = taskA_dataset(2000, 400, p_try, 8, 8, seed=13)
                modelT = LCPTreeAttention(p=p_try, K=8, H=2, m=8, r=4, superdiag=True, seeds=[1, 2])
                _ = modelT.train_epoch(qs, ys, shuffle=True)
                acc = modelT.eval_acc(qst, yst)
                print(f"tuner: p={p_try} acc={acc:.3f}")
                if acc > best_acc:
                    best_acc, best_p = acc, p_try
            print(f"chosen p={best_p} (acc={best_acc:.3f})")
        # Variance reduction (ULTRA_FUSE vs average) across several probes
        try:
            import os as _os
            dim = 32
            Uv = UltrametricAttention(dim=dim, p=5, max_depth=10, packed=True, heads=3)
            for i in range(256):
                Uv.insert(i, np.random.randn(dim))
            valsv = np.random.randn(256, dim)
            deltas = []
            for _pi in range(5):
                qv = np.random.randn(dim)
                _os.environ["ULTRA_FUSE"] = "0"
                _ = Uv.attend(qv, valsv)
                var_avg = float(np.var(np.array(getattr(Uv, 'last_head_sims', [0.0])), ddof=1))
                _os.environ["ULTRA_FUSE"] = "1"
                _ = Uv.attend(qv, valsv)
                var_fuse = float(np.var(np.array(getattr(Uv, 'last_head_sims', [0.0])), ddof=1))
                deltas.append(var_avg - var_fuse)
            var_delta = float(np.mean(deltas))
            # Print a small table for the deltas
            try:
                from rich.console import Console as _Console
                from rich.table import Table as _Table
                vt = _Table(title="Variance Reduction Deltas", show_header=True, header_style="bold magenta")
                vt.add_column("probe")
                vt.add_column("delta(var)")
                for i, dv in enumerate(deltas):
                    vt.add_row(str(i), f"{float(dv):.3e}")
                _Console().print(vt)
            except Exception as err:
                print(f"[ultrametric] Skipping variance table: {err}")
        except Exception as err:
            var_avg = var_fuse = None
            var_delta = None
            print(f"[ultrametric] Variance analysis failed: {err}")

        # Export diagnostics
        global last_diagnostics
        last_diagnostics = {
            "last_head_variance": float(var_heads) if 'var_heads' in locals() else None,
            "packed_arrays": bool(int(os.environ.get("ULTRA_PACKED_ARRAYS", "0"))),
            "scaling": {
                "N": Ns,
                "insert_ms": [float(x) for x in insert_ms],
                "query_ms": [float(x) for x in query_ms],
            },
            "variance_reduction": {
                "delta_mean": var_delta,
                "deltas": [float(x) for x in deltas] if 'deltas' in locals() else None,
            },
            "scaling_compare": {
                "N": Ns,
                "packed": {"insert_ms": [float(x) for x in insert_ms], "query_ms": [float(x) for x in query_ms]},
                "dict": {"insert_ms": [float(x) for x in ins2], "query_ms": [float(x) for x in qry2]},
            } if compare else None,
            "rank_demo": {
                "depth": int(d_demo) if 'd_demo' in locals() else None,
                "code": int(code_demo) if 'code_demo' in locals() else None,
                "rank": int(U2.rank_prefix(0, d_demo, code_demo)) if ('U2' in locals()) else None,
                "has": bool(U2.has_prefix(0, d_demo, code_demo)) if ('U2' in locals()) else None,
            },
            "tuner": {"p": int(best_p), "acc": float(best_acc)} if ('best_p' in locals()) else None,
            "rank_samples": [
                {"depth": int(max(1, d_demo-1)), "code": int((1<<max(1, d_demo-1))//2),
                 "rank": int(U2.rank_prefix(0, max(1, d_demo-1), (1<<max(1, d_demo-1))//2)),
                 "has": bool(U2.has_prefix(0, max(1, d_demo-1), (1<<max(1, d_demo-1))//2))},
                {"depth": int(d_demo), "code": int(code_demo),
                 "rank": int(U2.rank_prefix(0, d_demo, code_demo)),
                 "has": bool(U2.has_prefix(0, d_demo, code_demo))},
            ] if ('U2' in locals()) else None,
        }
    except Exception as err:
        print(f"[ultrametric] Demo failed: {err}")


if __name__ == "__main__":
    demo()


# --- Minimal p‑adic helpers for tests ---

def p_adic_encode(n: int, p: int, precision: int) -> np.ndarray:
    """Encode integer n modulo p^precision as base‑p digits least significant first."""
    p_pow = p ** precision
    n_mod = n % p_pow
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
