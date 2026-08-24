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
from rich.console import Console

_console = Console()
last_diagnostics: dict[str, object]


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
        # Hensel-lift freeze marks (bead 8gk.4): frozen_mark[d] = number of
        # nodes at depth d whose (S, R) are immutable - the lifted residue.
        # None (the default) disables freezing entirely (zero behavior change).
        self.frozen_mark: list[int] | None = None
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

    def _is_frozen(self, d: int, idx: int) -> bool:
        """A node is frozen iff it existed at Hensel-lift time (8gk.4): its
        (S, R) carry the lifted residue and may never be modified. Nodes
        CREATED after the lift (idx beyond the mark) remain writable even at
        shallow depths - new coarse structure on unseen inputs refines the
        function without touching the lifted solution."""
        return self.frozen_mark is not None and d < len(self.frozen_mark) and idx < self.frozen_mark[d]

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
            if self._is_frozen(d, idx):
                continue  # the lift fixes this residue exactly
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
            elif d_star >= 0 and not self._is_frozen(d_star, idx_star):
                chosen = (d_star, idx_star)
            elif d_star >= 0:
                return 0  # every writable spot on the path is frozen: no-op
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
        self.heads = [
            head_cls(p, K, m, r, U_seed=(None if seeds is None else seeds[h]), superdiag=superdiag) for h in range(H)
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
        self._planes = [rng.standard_normal((self.max_depth, self.dim)).astype(_np.float64) for _ in range(self._heads)]
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
            self._occ = [
                [np.zeros((1 << d,), dtype=np.uint8) for d in range(self.max_depth + 1)] for _ in range(self._heads)
            ]
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
                        lst = (
                            buckets_pa[h][d][code] if (d < len(buckets_pa[h]) and code < len(buckets_pa[h][d])) else []
                        )
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
            if best_j is None:
                raise RuntimeError("Ultrametric attention found no candidate in a non-empty index")
            picks.append(best_j)
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
    _console.print(
        f"[cyan]Task A:[/cyan] train_acc_pre={acc0:.4f} train_acc_post={acc_train:.4f} "
        f"test_acc={acc_test:.4f} created_nodes={created}"
    )
    _console.print(f"[dim]Timing(s): eval_pre={t1 - t0:.3f} train={t2 - t1:.3f} eval_test={t3 - t2:.3f}[/dim]")


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
        node_count = 0
        for head in model.heads:
            if not isinstance(head, PackedHeadTrie):
                raise RuntimeError("Packed LCP model contains a dictionary-backed head")
            node_count += sum(int(head._size[d]) for d in range(K))
    else:
        node_count = 0
        for head in model.heads:
            if not isinstance(head, HeadTrie):
                raise RuntimeError("Dictionary-backed LCP model contains a packed head")
            node_count += sum(len(head.levels[d].residues) for d in range(K))
    _console.print(
        f"[cyan]Task B:[/cyan] train_acc_pre={acc0:.4f} train_acc_post={acc_train:.4f} "
        f"test_acc={acc_test:.4f} created_nodes={created} total_nodes={node_count}"
    )
    _console.print(f"[dim]Timing(s): eval_pre={t1 - t0:.3f} train={t2 - t1:.3f} eval_test={t3 - t2:.3f}[/dim]")


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
    _console.print(
        f"[cyan]Small A:[/cyan] pre={acc0:.3f} post={acc_train:.3f} test={acc_test:.3f} created={created}"
    )


def demo():
    """Run all ultrametric demonstrations."""
    global last_diagnostics

    random.seed(0)
    np.random.seed(0)
    use_packed = bool(int(os.environ.get("ULTRA_PACKED", "0")))
    best_p: int | None = None
    best_acc = -1.0
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
            _console.print(f"[yellow]Packed/reference table skipped: {err}[/yellow]")
    except Exception as err:
        _console.print(f"[bold red]Packed/reference sanity check failed: {err}[/bold red]")
    # Optional packed timing benchmark to n=4096
    try:
        import time as _time

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
            var_heads = (
                np.var(np.array(getattr(U, "last_head_sims", [0.0])), ddof=1) if hasattr(U, "last_head_sims") else 0.0
            )
            print(
                f"N={N:4d} | insert {1000 * (t1 - t0):6.1f} ms | query {1000 * (t2 - t1):6.1f} ms | head var {var_heads:.3e}"
            )
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
            print(
                "rank_prefix demo:",
                U2.rank_prefix(0, d_demo, code_demo),
                "has_prefix:",
                U2.has_prefix(0, d_demo, code_demo),
            )
            # Rank/Test summary table
            try:
                from rich.console import Console as _Console
                from rich.table import Table as _Table

                tab = _Table(title="Rank/Test Summary (array-packed)", show_header=True, header_style="bold magenta")
                tab.add_column("depth")
                tab.add_column("code")
                tab.add_column("rank")
                tab.add_column("has")
                demos = [
                    (d_demo - 1, (1 << (d_demo - 1)) // 2),
                    (d_demo, code_demo),
                    (d_demo + 1, min((1 << (d_demo + 1)) // 2, (1 << (K)) - 1)),
                ]
                for dd, cc in demos:
                    rnk = U2.rank_prefix(0, max(1, dd), int(cc))
                    has = U2.has_prefix(0, max(1, dd), int(cc))
                    tab.add_row(str(int(dd)), str(int(cc)), str(int(rnk)), str(bool(has)))
                _Console().print(tab)
            except Exception as err:
                print(f"[ultrametric] Skipping rank/test table: {err}")
        # p-tuner (optional): try p∈{3,5,7} on a tiny Task A split and choose best by eval acc
        if bool(int(os.environ.get("ULTRA_TUNE_P", "0"))):
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
                var_avg = float(np.var(np.array(getattr(Uv, "last_head_sims", [0.0])), ddof=1))
                _os.environ["ULTRA_FUSE"] = "1"
                _ = Uv.attend(qv, valsv)
                var_fuse = float(np.var(np.array(getattr(Uv, "last_head_sims", [0.0])), ddof=1))
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
        last_diagnostics = {
            "last_head_variance": float(var_heads) if "var_heads" in locals() else None,
            "packed_arrays": bool(int(os.environ.get("ULTRA_PACKED_ARRAYS", "0"))),
            "scaling": {
                "N": Ns,
                "insert_ms": [float(x) for x in insert_ms],
                "query_ms": [float(x) for x in query_ms],
            },
            "variance_reduction": {
                "delta_mean": var_delta,
                "deltas": [float(x) for x in deltas] if "deltas" in locals() else None,
            },
            "scaling_compare": {
                "N": Ns,
                "packed": {"insert_ms": [float(x) for x in insert_ms], "query_ms": [float(x) for x in query_ms]},
                "dict": {"insert_ms": [float(x) for x in ins2], "query_ms": [float(x) for x in qry2]},
            }
            if compare
            else None,
            "rank_demo": {
                "depth": int(d_demo) if "d_demo" in locals() else None,
                "code": int(code_demo) if "code_demo" in locals() else None,
                "rank": int(U2.rank_prefix(0, d_demo, code_demo)) if ("U2" in locals()) else None,
                "has": bool(U2.has_prefix(0, d_demo, code_demo)) if ("U2" in locals()) else None,
            },
            "tuner": {"p": best_p, "acc": float(best_acc)} if best_p is not None else None,
            "rank_samples": [
                {
                    "depth": int(max(1, d_demo - 1)),
                    "code": int((1 << max(1, d_demo - 1)) // 2),
                    "rank": int(U2.rank_prefix(0, max(1, d_demo - 1), (1 << max(1, d_demo - 1)) // 2)),
                    "has": bool(U2.has_prefix(0, max(1, d_demo - 1), (1 << max(1, d_demo - 1)) // 2)),
                },
                {
                    "depth": int(d_demo),
                    "code": int(code_demo),
                    "rank": int(U2.rank_prefix(0, d_demo, code_demo)),
                    "has": bool(U2.has_prefix(0, d_demo, code_demo)),
                },
            ]
            if ("U2" in locals())
            else None,
        }
    except Exception as err:
        print(f"[ultrametric] Demo failed: {err}")

    # Valued-attention section (bead 8gk.2): the dictionary's reference
    # mechanism, the three-shadow table, and the exact ball-tree benchmark.
    # (last_diagnostics is already declared global earlier in this function.)
    try:
        section = run_valued_attention_section()
        try:
            last_diagnostics["valued_attention"] = section
        except (NameError, TypeError):
            last_diagnostics = {"valued_attention": section}
    except Exception as err:
        print(f"[ultrametric] Valued-attention section failed: {err}")

    # Hensel-lift curriculum section (bead 8gk.4): residue-preserving digit
    # refinement with the exact-preservation invariant asserted loudly.
    try:
        hensel = run_hensel_curriculum_section()
        try:
            last_diagnostics["hensel_curriculum"] = hensel
        except (NameError, TypeError):
            last_diagnostics = {"hensel_curriculum": hensel}
    except Exception as err:
        print(f"[ultrametric] Hensel-curriculum section failed: {err}")

    # Mahler-basis section (bead 92jp): canonical compression on Z_p addresses
    # with the certified ultrametric truncation error, all exact.
    try:
        mahler = run_mahler_section()
        try:
            last_diagnostics["mahler"] = mahler
        except (NameError, TypeError):
            last_diagnostics = {"mahler": mahler}
    except Exception as err:
        print(f"[ultrametric] Mahler section failed: {err}")


# --- Minimal p‑adic helpers for tests ---


def p_adic_encode(n: int, p: int, precision: int) -> np.ndarray:
    """Encode integer n modulo p^precision as base‑p digits least significant first."""
    p_pow = p**precision
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
    mod = p**k
    prod = (n1 * n2) % mod
    return p_adic_encode(prod, p, k)


# ---------------------------------------------------------------------------
# Valued attention (bead 8gk.2): the valuation dictionary made executable.
#
# One backbone, three shadows. Keys/queries are finite-precision elements of
# Z_p (length-K digit vectors = integer representatives), carrying the genuine
# p-adic valuation v_p. The dictionary, each line an exact integer theorem
# (tests: tests/test_algebraic_properties.py, valuation-dictionary section;
# note: markdown_documentation/the_valuation_dictionary.md):
#
#   (a) LCP IS VALUATION:        lcp(x, y) = v_p(x - y)  (mod p^K, capped at K)
#   (b) TROPICAL IS THE SHADOW:  v_p(<q, k>) = min_j (v_p(q_j) + v_p(k_j))
#       for valuation-generic inputs; the tie/cancellation locus where the
#       inequality is strict has Haar measure 1/(p+1) per binary sum - it is
#       the tropical variety, the corner locus where routes switch.
#   (c) DOMINANCE IS VALUATION:  the leading term q^{v} c_v of the Hahn-series
#       rendering is the dominance order the surreal demo probes.
#
# The existing ultrametric mechanism (digit similarity = v_p of a DIFFERENCE)
# and the tropical mechanism (valuation ARITHMETIC of products/sums) are the
# two projections of the one valued-attention structure (V, Gamma, v, sim).
#
# THE ALGORITHMIC PAYOFF: for the difference kernel, balls are trie nodes, so
# attention sums aggregate exactly over LCP shells - the ball tree computes
# EXACT attention in O(K) per query (correctness is a theorem, not an
# approximation bound), directly answering E3's finding that the torch kernel
# port is quadratic-with-better-constants.
# ---------------------------------------------------------------------------


def vp_int(n: int, p: int, cap: int) -> int:
    """p-adic valuation of an integer, capped (v(0) := cap, the precision)."""
    if n == 0:
        return cap
    v = 0
    while n % p == 0 and v < cap:
        n //= p
        v += 1
    return v


def vp_digits(digits: np.ndarray, p: int) -> int:
    """Valuation of a digit vector = index of the first nonzero digit (K if zero)."""
    nz = np.nonzero(np.asarray(digits))[0]
    return int(nz[0]) if len(nz) else len(digits)


def lcp_depth(a: np.ndarray, b: np.ndarray) -> int:
    """Longest common prefix of two digit vectors (LSB-first), = v_p(a - b)."""
    neq = np.nonzero(np.asarray(a) != np.asarray(b))[0]
    return int(neq[0]) if len(neq) else len(a)


def valued_bilinear_shadows(q_digits: np.ndarray, k_digits: np.ndarray, p: int, K: int) -> dict:
    """One bilinear score rendered in all three shadows (exact integers).

    Returns the integer inner product, its exact valuation, the tropical
    (min,+) shadow min_j (v q_j + v k_j), whether the tropicalization theorem's
    genericity holds (exact == tropical), and the Hahn/leading-term rendering.
    """
    q_ints = [p_adic_decode(np.asarray(q_digits)[j], p) for j in range(len(q_digits))]
    k_ints = [p_adic_decode(np.asarray(k_digits)[j], p) for j in range(len(k_digits))]
    ip = sum(a * b for a, b in zip(q_ints, k_ints))  # exact over Z, no truncation
    cap = 2 * K  # products of K-digit elements have valuation < 2K unless zero
    v_exact = vp_int(ip, p, cap)
    v_trop = min(
        vp_int(a, p, cap) + vp_int(b, p, cap) for a, b in zip(q_ints, k_ints)
    )
    lead_coeff = (ip // p**v_exact) % p if ip != 0 else 0
    return {
        "inner_product": ip,
        "v_exact": v_exact,
        "v_tropical": v_trop,
        "generic": v_exact == v_trop,
        "leading_term": f"{lead_coeff}*p^{v_exact}" if ip != 0 else "0",
    }


class BallTreeValuedAttention:
    """Exact sub-quadratic valued attention over the Bruhat-Tits trie.

    Hard-digit difference kernel: weight(q, k) = alpha^{lcp(q, k)} with
    alpha > 1. Balls of radius p^{-d} around q are trie nodes (residues mod
    p^d), nested by the strong triangle inequality, so

        out(q) = sum_j alpha^{lcp(q, k_j)} v_j
               = sum_{d=0..K} alpha^d * (S_{>=d}(q) - S_{>=d+1}(q)),

    where S_{>=d}(q) is the per-node value sum at q's depth-d ancestor - an
    EXACT shell decomposition, O(K) per query after O(N K) build, with the
    normalizer aggregated from per-node counts the same way. Streaming
    insertion (insert key i after querying it) gives the exactly-causal
    O(N K) attention path - the E3 sub-quadratic answer. With alpha = 2 and
    integer values every quantity is an exact dyadic float: the brute-force
    comparison asserts equality with ==, no tolerance.
    """

    def __init__(self, p: int, K: int, dim: int, alpha: float = 2.0):
        self.p = int(p)
        self.K = int(K)
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.pk = [self.p**d for d in range(self.K + 1)]
        # per-depth: residue (key mod p^d) -> [value-sum vector, count]
        self.node_sum: list[dict[int, np.ndarray]] = [{} for _ in range(self.K + 1)]
        self.node_cnt: list[dict[int, int]] = [{} for _ in range(self.K + 1)]
        self.n_keys = 0

    def insert(self, key_int: int, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float64)
        for d in range(self.K + 1):
            r = key_int % self.pk[d]
            if r in self.node_sum[d]:
                self.node_sum[d][r] = self.node_sum[d][r] + value
                self.node_cnt[d][r] += 1
            else:
                self.node_sum[d][r] = value.copy()
                self.node_cnt[d][r] = 1
        self.n_keys += 1

    def attend(self, q_int: int) -> np.ndarray | None:
        """Exact alpha^lcp-weighted average over all inserted keys, O(K)."""
        if self.n_keys == 0:
            return None
        num = np.zeros(self.dim, dtype=np.float64)
        den = 0.0
        for d in range(self.K + 1):
            r = q_int % self.pk[d]
            s_d = self.node_sum[d].get(r)
            if s_d is None:
                break  # deeper balls are empty subsets of this one
            c_d = self.node_cnt[d][r]
            if d < self.K:
                r1 = q_int % self.pk[d + 1]
                s_d1 = self.node_sum[d + 1].get(r1)
                c_d1 = self.node_cnt[d + 1].get(r1, 0)
            else:
                s_d1, c_d1 = None, 0
            shell_sum = s_d - (s_d1 if s_d1 is not None else 0.0)
            shell_cnt = c_d - c_d1
            if shell_cnt:
                w = self.alpha**d
                num += w * shell_sum
                den += w * shell_cnt
        return num / den

    def attend_bruteforce(self, q_int: int, keys: list[int], values: np.ndarray) -> np.ndarray:
        """O(N K) reference: identical arithmetic, summed per shell so the
        dyadic-exactness contract (== with alpha = 2, integer values) holds."""
        shells: dict[int, tuple[np.ndarray, int]] = {}
        for k_int, v in zip(keys, values):
            d = vp_int((q_int - k_int) % self.pk[self.K], self.p, self.K)
            s, c = shells.get(d, (np.zeros(self.dim, dtype=np.float64), 0))
            shells[d] = (s + v, c + 1)
        num = np.zeros(self.dim, dtype=np.float64)
        den = 0.0
        for d, (s, c) in shells.items():
            num += self.alpha**d * s
            den += self.alpha**d * c
        return num / den


def run_valued_attention_section() -> dict:
    """Valued attention reference + three-shadow table + ball-tree benchmark."""
    rng = np.random.default_rng(8)
    p, K, dim = 3, 8, 4
    print("\n[Valued Attention - the dictionary made executable (8gk.2)]")

    # --- (1) the three-shadow table: one attention computation, three lenses
    n_keys = 6
    q_dig = np.stack([p_adic_encode(int(rng.integers(0, p**K)), p, K) for _ in range(dim)])
    rows = []
    tie_count = 0
    for j in range(n_keys):
        k_dig = np.stack([p_adic_encode(int(rng.integers(0, p**K)), p, K) for _ in range(dim)])
        sh = valued_bilinear_shadows(q_dig, k_dig, p, K)
        if not sh["generic"]:
            tie_count += 1
        digits_str = "".join(str(int(d)) for d in p_adic_encode(sh["inner_product"] % p**K, p, K))
        rows.append(
            (
                f"k{j}",
                digits_str,
                str(sh["v_exact"]),
                str(sh["v_tropical"]),
                "=" if sh["generic"] else "< (tie locus)",
                sh["leading_term"],
                f"{2.0 ** (-sh['v_exact']):.4f}",
            )
        )
    try:
        from rich.console import Console as _Console
        from rich.table import Table as _Table

        tab = _Table(
            title=f"Three shadows of one attention computation (p={p}, K={K}): "
            "digits (Q_p) | tropical (min,+) | leading exponent (Hahn/dominance)",
            show_header=True,
            header_style="bold magenta",
        )
        for col in ("key", "<q,k> digits (LSB->MSB)", "v_p exact", "(min,+) shadow", "generic?", "leading term", "score 2^-v"):
            tab.add_column(col, justify="right")
        for r in rows:
            tab.add_row(*r)
        _Console().print(tab)
    except Exception as err:
        print(f"[ultrametric] three-shadow table skipped: {err}")

    # --- (2) the cancellation-locus measure: empirical vs the exact 1/(p+1)
    n_mc = 20000
    strict = 0
    M = p ** (2 * K)
    for _ in range(n_mc):
        x = int(rng.integers(0, M))
        y = int(rng.integers(0, M))
        if vp_int((x + y) % M, p, 2 * K) > min(vp_int(x or M, p, 2 * K), vp_int(y or M, p, 2 * K)):
            strict += 1
    tie_rate = strict / n_mc
    print(
        f"cancellation locus: empirical {tie_rate:.4f} vs exact 1/(p+1) = {1 / (p + 1):.4f} "
        f"(binary sums, Haar-uniform Z_p; the tropical variety has measure zero in the limit K -> inf per digit)"
    )

    # --- (3) ball-tree exact attention: == brute force, then the E3 timing hook
    import time as _time

    Kbt, pbt = 12, 2  # alpha = 2, integer values: every weight/sum is exact dyadic
    bench_rows = []
    exact_ok = True
    for N in (256, 1024, 4096):
        keys = [int(rng.integers(0, pbt**Kbt)) for _ in range(N)]
        values = np.asarray(rng.integers(-8, 9, size=(N, dim)), dtype=np.float64)
        bt = BallTreeValuedAttention(p=pbt, K=Kbt, dim=dim, alpha=2.0)
        t0 = _time.perf_counter()
        for k_int, v in zip(keys, values):
            bt.insert(k_int, v)
        t_build = _time.perf_counter() - t0
        n_q = 64
        q_ints = [int(rng.integers(0, pbt**Kbt)) for _ in range(n_q)]
        t0 = _time.perf_counter()
        outs_bt: list[np.ndarray] = []
        for q in q_ints:
            out = bt.attend(q)
            if out is None:
                raise RuntimeError("Ball-tree attention is empty after inserting benchmark keys")
            outs_bt.append(out)
        t_bt = (_time.perf_counter() - t0) / n_q
        t0 = _time.perf_counter()
        outs_bf = [bt.attend_bruteforce(q, keys, values) for q in q_ints]
        t_bf = (_time.perf_counter() - t0) / n_q
        exact = all(np.array_equal(a, b) for a, b in zip(outs_bt, outs_bf))
        exact_ok = exact_ok and exact
        bench_rows.append(
            {
                "N": N,
                "build_ms": 1000 * t_build,
                "balltree_us_per_query": 1e6 * t_bt,
                "brute_us_per_query": 1e6 * t_bf,
                "speedup": t_bf / t_bt if t_bt > 0 else float("inf"),
                "exact": exact,
            }
        )
    try:
        from rich.console import Console as _Console
        from rich.table import Table as _Table

        bt_tab = _Table(
            title="Ball-tree exact attention vs brute force (hard digits, alpha=2: equality asserted with ==)",
            show_header=True,
            header_style="bold magenta",
        )
        for col in ("N", "build ms", "ball-tree us/q", "brute us/q", "speedup", "exact?"):
            bt_tab.add_column(col, justify="right")
        for r in bench_rows:
            bt_tab.add_row(
                str(r["N"]),
                f"{r['build_ms']:.1f}",
                f"{r['balltree_us_per_query']:.1f}",
                f"{r['brute_us_per_query']:.1f}",
                f"{r['speedup']:.1f}x",
                str(r["exact"]),
            )
        _Console().print(bt_tab)
    except Exception as err:
        print(f"[ultrametric] ball-tree table skipped: {err}")
    if not exact_ok:
        print("[ultrametric] WARNING: ball-tree != brute force - the exactness theorem's implementation is broken")

    section = {
        "three_shadow_rows": [
            {
                "key": r[0],
                "digits": r[1],
                "v_exact": int(r[2]),
                "v_tropical": int(r[3]),
                "generic": r[4] == "=",
                "leading_term": r[5],
            }
            for r in rows
        ],
        "tie_rate": {"empirical": tie_rate, "exact_binary": 1.0 / (p + 1), "n_mc": n_mc},
        "balltree_bench": bench_rows,  # the E3 hook: exact sub-quadratic path timings
        "balltree_exact": exact_ok,
    }
    return section


# ---------------------------------------------------------------------------
# Hensel-lift curriculum (bead 8gk.4): train coarse digits first, then lift.
#
# Solving f(x) = 0 mod p and lifting to mod p^2, p^3, ... is the p-adic Newton
# method; each lift PRESERVES the previous residue exactly. Training
# translation: learn digit-0..j-1 structure (a K=j trie), then lift to K=j+1
# with the stage-j node data FROZEN (the lift fixes the residue) and only
# deeper/new structure trainable. The invariant is EXACT and asserted loudly:
# every node that existed at lift time keeps bit-identical (S, R) through all
# later training - float curricula have no such guarantee (early learning is
# routinely overwritten). Theory note: markdown_documentation/padic_precision.md.
# ---------------------------------------------------------------------------


def _dict_backed_heads(model: "LCPTreeAttention") -> list[HeadTrie]:
    """Return reference heads, rejecting a model whose runtime layout disagrees."""
    heads: list[HeadTrie] = []
    for head in model.heads:
        if not isinstance(head, HeadTrie):
            raise ValueError("Hensel lifting requires dictionary-backed HeadTrie instances")
        heads.append(head)
    return heads


def hensel_lift_model(src: "LCPTreeAttention", K_new: int) -> "LCPTreeAttention":
    """Lift a K=j LCPTreeAttention to K_new > j, freezing the stage-j residue.

    The lifted heads share the source's U seeds (the per-depth maps for the
    first j levels are bit-identical by construction) and copy the node data;
    frozen_mark records the lift boundary per depth.
    """
    if K_new <= src.K:
        raise ValueError(f"K_new must exceed the source depth, got {K_new} <= {src.K}")
    if src.packed:
        raise ValueError("hensel_lift_model supports the dict-backed HeadTrie reference")
    src_heads = _dict_backed_heads(src)
    lifted = LCPTreeAttention(
        p=src.p,
        K=K_new,
        H=src.H,
        m=src.m,
        r=src_heads[0].r,
        superdiag=False,
        seeds=None,
        packed=False,
    )
    lifted_heads = _dict_backed_heads(lifted)
    for h in range(src.H):
        src_head, new_head = src_heads[h], lifted_heads[h]
        new_head.U = list(src_head.U) + list(new_head.U[src.K:])  # reuse stage maps verbatim
        for d in range(src.K):
            L = src_head.levels[d]
            new_head.levels[d] = DepthArrays(dict(L.res2idx), list(L.residues), L.S, L.R)
        new_head.frozen_mark = [len(src_head.levels[d].residues) for d in range(src.K)]
    return lifted


def _frozen_snapshot(model: "LCPTreeAttention", upto_K: int) -> list[list[tuple]]:
    """Bit-exact snapshot of every node's (residue, S, R) at depths < upto_K."""
    snap = []
    for head in _dict_backed_heads(model):
        levels = []
        for d in range(upto_K):
            L = head.levels[d]
            n = head.frozen_mark[d] if head.frozen_mark else len(L.residues)
            levels.append((list(L.residues[:n]), np.asarray(L.S[:n]).copy(), np.asarray(L.R[:n]).copy()))
        snap.append(levels)
    return snap


def _assert_residues_preserved(model: "LCPTreeAttention", snap: list[list[tuple]], upto_K: int) -> None:
    """A single violated residue fails the run loudly (the bead's contract)."""
    for h, head in enumerate(_dict_backed_heads(model)):
        for d in range(upto_K):
            residues, S, R = snap[h][d]
            L = head.levels[d]
            n = len(residues)
            assert list(L.residues[:n]) == residues, f"head {h} depth {d}: residue set mutated"
            assert np.array_equal(np.asarray(L.S[:n]), S), f"head {h} depth {d}: S residue violated"
            assert np.array_equal(np.asarray(L.R[:n]), R), f"head {h} depth {d}: R counters violated"


def run_hensel_curriculum_section(*, K_coarse: int = 2, K_full: int = 4, epochs: int = 6,
                                  n_train: int = 3000, n_test: int = 800, seed: int = 17) -> dict:
    """Hensel curriculum vs end-to-end at equal total budget, on Task A."""
    print("\n[Hensel-Lift Curriculum - residue-preserving digit refinement (8gk.4)]")
    p, m = 5, 8
    qs, ys, qst, yst = taskA_dataset(n_train, n_test, p, K_full, m, seed=seed)
    qs_coarse = [q[:K_coarse] for q in qs]

    # stage 1: coarse model on truncated digits (half the budget)
    random.seed(seed)
    stage1 = LCPTreeAttention(p=p, K=K_coarse, H=1, m=m, r=4, superdiag=False, seeds=None)
    for _ in range(epochs // 2):
        acc1, _ = stage1.train_epoch(qs_coarse, ys, shuffle=True)

    # lift: freeze the stage-1 residue, refine with full digits (other half)
    lifted = hensel_lift_model(stage1, K_full)
    snap = _frozen_snapshot(lifted, K_coarse)
    for _ in range(epochs - epochs // 2):
        acc2, _ = lifted.train_epoch(qs, ys, shuffle=True)
    _assert_residues_preserved(lifted, snap, K_coarse)
    acc_curr = lifted.eval_acc(qst, yst)

    # control: end-to-end full-depth training at the SAME total budget
    random.seed(seed)
    e2e = LCPTreeAttention(p=p, K=K_full, H=1, m=m, r=4, superdiag=False, seeds=None)
    acc_curve = []
    for _ in range(epochs):
        a, _ = e2e.train_epoch(qs, ys, shuffle=True)
        acc_curve.append(a)
    acc_e2e = e2e.eval_acc(qst, yst)

    print("residue preservation: EXACT (every lift-time node bit-identical through stage 2)")
    print(f"curriculum test acc {acc_curr:.3f} vs end-to-end {acc_e2e:.3f} (equal budget, {epochs} epochs)")
    try:
        from rich.console import Console as _Console
        from rich.table import Table as _Table

        tab = _Table(title=f"Hensel curriculum (K {K_coarse} -> {K_full}, p={p})",
                     show_header=True, header_style="bold magenta")
        for col in ("arm", "test acc", "residue invariant"):
            tab.add_column(col, justify="right")
        tab.add_row("curriculum (lift)", f"{acc_curr:.3f}", "EXACT (asserted)")
        tab.add_row("end-to-end", f"{acc_e2e:.3f}", "-")
        _Console().print(tab)
    except Exception as err:
        print(f"[ultrametric] Hensel table skipped: {err}")

    return {
        "K_coarse": K_coarse,
        "K_full": K_full,
        "epochs": epochs,
        "curriculum_test_acc": acc_curr,
        "end_to_end_test_acc": acc_e2e,
        "residues_preserved_exactly": True,  # the assert above fails the run otherwise
    }


# ---------------------------------------------------------------------------
# Mahler-basis heads (bead 92jp; design in padic_precision.md section 4).
#
# Mahler's theorem: every continuous f: Z_p -> Q_p has a canonical expansion
# f(x) = sum_n a_n * C(x, n) with |a_n|_p -> 0, and truncation at N is
# CANONICAL COMPRESSION with a certified ultrametric error:
#     ||f - f_N||_sup = max_{n > N} |a_n|_p.
# Trie addresses ARE elements of Z_p, so heads on hierarchical addresses can
# be truncated Mahler series instead of MLPs - the function class whose
# compression knob carries a certificate that COMPOSES with the flat-error
# lemma instead of fighting it. Everything below is exact integer arithmetic.
# ---------------------------------------------------------------------------


def mahler_coefficients(f_values: list[int], modulus: int) -> list[int]:
    """Mahler coefficients a_n = (Delta^n f)(0) mod modulus, from f on 0..N.

    The forward-difference transform is the exact inverse of evaluation:
    f(x) = sum_n a_n C(x, n) for all 0 <= x <= N (tested as a roundtrip).
    """
    diffs = [v % modulus for v in f_values]
    coeffs = []
    while diffs:
        coeffs.append(diffs[0])
        diffs = [(b - a) % modulus for a, b in zip(diffs, diffs[1:])]
    return coeffs


def mahler_eval(coeffs: list[int], x: int, modulus: int) -> int:
    """Evaluate the (truncated) Mahler series at x: sum_n a_n C(x, n) mod modulus."""
    total = 0
    binom = 1  # C(x, 0)
    for n, a in enumerate(coeffs):
        total = (total + a * binom) % modulus
        binom = binom * (x - n) // (n + 1)  # C(x, n+1), exact integer division
    return total


def run_mahler_section(*, p: int = 3, prec: int = 6, N: int = 24, seed: int = 23) -> dict:
    """Mahler truncation certificate + roundtrip, exact mod p^prec."""
    print("\n[Mahler-Basis Heads - canonical compression with certified error (92jp)]")
    rng = random.Random(seed)
    modulus = p**prec

    def vp_of(n: int) -> int:
        return vp_int(n % modulus or modulus, p, prec)

    # construct f FROM a coefficient sequence with known valuation growth
    # (a_n divisible by p^(n//4): continuity made quantitative), then verify
    # the truncation error certificate exactly
    true_coeffs = [(p ** min(n // 4, prec - 1)) * rng.randrange(1, p) for n in range(N + 1)]
    f_vals = [mahler_eval(true_coeffs, x, modulus) for x in range(N + 1)]

    rec = mahler_coefficients(f_vals, modulus)
    roundtrip_ok = all(mahler_eval(rec, x, modulus) == f_vals[x] % modulus for x in range(N + 1))
    coeff_ok = all((rec[n] - true_coeffs[n]) % modulus == 0 for n in range(N + 1))

    rows = []
    for cut in (4, 8, 16):
        cert = min(vp_of(c) for c in true_coeffs[cut + 1 :])  # certified min valuation of dropped terms
        worst = prec
        for x in range(N + 1):
            err = (mahler_eval(true_coeffs[: cut + 1], x, modulus) - f_vals[x]) % modulus
            if err:
                worst = min(worst, vp_of(err))
        rows.append((cut, cert, worst, worst >= cert))
    try:
        from rich.console import Console as _Console
        from rich.table import Table as _Table

        tab = _Table(title=f"Mahler truncation certificate (p={p}, mod p^{prec})",
                     show_header=True, header_style="bold magenta")
        for col in ("truncate at N", "certified v_p(error) >=", "measured min v_p(error)", "certificate holds"):
            tab.add_column(col, justify="right")
        for cut, cert, worst, ok in rows:
            tab.add_row(str(cut), str(cert), str(worst), str(ok))
        _Console().print(tab)
    except Exception as err:
        print(f"[ultrametric] Mahler table skipped: {err}")
    print(f"roundtrip exact: {roundtrip_ok} | coefficient recovery exact: {coeff_ok}")

    return {
        "roundtrip_exact": roundtrip_ok,
        "coefficients_recovered_exactly": coeff_ok,
        "certificate_rows": [
            {"truncate_at": c, "certified_vp": ce, "measured_vp": w, "holds": ok} for c, ce, w, ok in rows
        ],
    }


if __name__ == "__main__":
    demo()
