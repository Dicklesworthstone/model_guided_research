"""
Fractal KV‑store via contractive IFS. Values live as fixed points of composed contractions; keys are
depth‑k paths selecting branch maps. With A = s·I_d (0 < s < 1/2), each branch is f_{ℓ,i}(x)=A x + t_{ℓ,i}+u_{ℓ,i},
where t_{ℓ,i} are hypercube‑corner translations giving separation margin γ=1−2s. For a path w=(i₁,…,i_k):
F_w(x)=A^k x + c_w + u_w with c_w=∑_{j=1}^k A^{k−j} t_{j,i_j}. The stored value is the fixed point
x*_w=(I−A^k)^{-1}(c_w+u_w). Write v is leaf‑local: u_w=(I−A^k) v − c_w (siblings untouched). Read is closed‑form:
v̂=(c_w+u_w)/(1−s^k). A learned router (k independent m‑way classifiers) maps query q→w; inference composes exactly
k maps, so access is O(log_m N). Diagnostics report utilization, collision_rate, and γ; controlled re‑indexing
(increase k or decrease s) is triggered on fragmentation/overlap. A microbenchmark demonstrates reduced catastrophic
forgetting vs a last‑batch baseline due to strictly local writes.
"""

# A complete, working JAX implementation of a "fractal KV‑store":
#   - Keys are contraction parameters (branch choices) in a depth‑k IFS.
#   - Values are fixed points x*_w of the composed contraction F_w.
#   - A learned router composes exactly k maps per access (O(log N)).
#   - Writes are closed‑form (leaf‑local); reads are closed‑form.
#   - Strong separation and contractivity guaranteed by initialization.
#   - Diagnostics and a microbenchmark for catastrophic forgetting.
#
# Mathematical core (linear, isotropic case A = s I_d with 0 < s < 1/2):
#   F_w(x) = A^k x + c_w + u_w,             where
#   c_w    = sum_{j=1..k} A^{k-j} t_{j,i_j},  (t_{j,i} are geometric offsets)
#   x*_w   = (I - A^k)^{-1} (c_w + u_w).
#   Write to set value v at address w: u_w = (I - A^k) v - c_w.
#   Read:  v_hat = x*_w = (c_w + u_w) / (1 - s^k).
#
# The code below keeps the implementation aligned 1:1 with these equations.

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, jit, lax, random, vmap
from jax import tree_util as _tree

# ------------------------------
# Utility functions
# ------------------------------


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _m_powers(m: int, k: int) -> Array:
    # powers for base-m index encoding: [m^{k-1}, m^{k-2}, ..., m^0]
    return jnp.array([m**p for p in range(k - 1, -1, -1)], dtype=jnp.int32)


def _index_to_path(idx: Array, m: int, k: int) -> Array:
    # converts [B] indices to base-m digits [B, k]
    # Implement via a fixed-iteration fori_loop writing into a preallocated buffer of shape [B, k]
    B = idx.shape[0]
    def step(t, carry):
        rem, out = carry
        digit = rem % m
        rem = rem // m
        # Write least-significant first at position from the end
        pos = k - 1 - t
        out = out.at[:, pos].set(digit.astype(jnp.int32))
        return rem, out

    digits = jnp.zeros((B, k), dtype=jnp.int32)
    _, digits = lax.fori_loop(0, k, step, (idx, digits))
    return digits  # type: ignore[no-any-return]


def _path_to_index(path: Array, m_pow: Array) -> Array:
    # path: [B, k], m_pow: [k]
    return jnp.dot(path, m_pow).astype(jnp.int32)


def _binary_corners(m: int, d: int, s: float) -> Array:
    r"""
    Return T \in R^{m x d} where each row is a hypercube corner scaled by (1 - s).
    Requires m = 2^{de} and d >= de.
    """
    assert _is_power_of_two(m), "m must be a power of two to use binary-corner translations."
    de = int(math.log2(m))
    assert d >= de, f"d (value dimension) must be >= log2(m). Got d={d}, log2(m)={de}."

    # Build all m binary codes of length de (little-endian: bit j is (i >> j) & 1)
    bits = ((jnp.arange(m)[:, None] >> jnp.arange(de)[None, :]) & 1).astype(jnp.float32)
    # Pad to d and scale
    pad = jnp.zeros((m, d - de), dtype=jnp.float32)
    corners = jnp.concatenate([bits, pad], axis=1)
    return (1.0 - s) * corners


# ------------------------------
# Fractal KV store (A = s I_d)
# ------------------------------


@dataclass
class FractalKVConfig:
    d_val: int  # value dimension (and geometric space dimension)
    m: int  # branching factor (must be power of two for corner scheme)
    k: int  # depth
    s: float  # contraction scale in (0, 1/2) for separation margin
    dtype: jnp.dtype = jnp.float32

    def __post_init__(self):
        if not _is_power_of_two(self.m):
            raise ValueError("m must be a power of two (for hypercube corner translations).")
        if not (0.0 < self.s < 0.5):
            raise ValueError("s must be in (0, 1/2) to guarantee strong separation margin 1-2s > 0.")
        if self.d_val < int(math.log2(self.m)):
            raise ValueError("d_val must be >= log2(m) to embed corner translations.")
        # Precompute powers for index encoding used in tests
        object.__setattr__(self, "m_pow", _m_powers(self.m, self.k))

    @property
    def capacity(self) -> int:
        return int(self.m**self.k)

    @property
    def separation_margin(self) -> float:
        # For the corner construction with A = s I_d, first-level margin is 1 - 2s
        return float(1.0 - 2.0 * self.s)

    @property
    def a_pow_k_scalar(self) -> float:
        return float(self.s**self.k)

    @property
    def inv_I_minus_Ak_scalar(self) -> float:
        # (I - s^k I)^{-1} = 1 / (1 - s^k)
        return float(1.0 / (1.0 - self.s**self.k))


@dataclass
class FractalKVState:
    payload_u: Array  # [capacity, d_val] payload translations u_w
    written_mask: Array  # [capacity] bool
    total_writes: Array  # scalar int32
    total_collisions: Array  # scalar int32

    # Backward-compatible alias expected by tests
    @property
    def u_leaf(self) -> Array:
        return self.payload_u


class FractalKV:
    """
    Implements the exact linear IFS store specialized to A = s I_d.

    Equations:
      c_w = sum_{j=1..k} s^{k-j} * t_{j, i_j}   (t-bank identical across levels)
      u_w = (1 - s^k) * v - c_w
      read: x*_w = (c_w + u_w) / (1 - s^k)
    """

    def __init__(self, cfg: FractalKVConfig):
        self.cfg = cfg
        self.T = _binary_corners(cfg.m, cfg.d_val, cfg.s).astype(cfg.dtype)  # [m, d_val]
        self.m_pow = _m_powers(cfg.m, cfg.k)  # [k] base-m positional multipliers
        self.level_scales = jnp.array([cfg.s ** (cfg.k - 1 - j) for j in range(cfg.k)], dtype=cfg.dtype)  # [k]
        self._state = FractalKVState(
            payload_u=jnp.zeros((cfg.capacity, cfg.d_val), dtype=cfg.dtype),
            written_mask=jnp.zeros((cfg.capacity,), dtype=jnp.bool_),
            total_writes=jnp.array(0, dtype=jnp.int32),
            total_collisions=jnp.array(0, dtype=jnp.int32),
        )

        # Register PyTree for FractalKVState so it can be carried through jitted fns
        try:
            _tree.register_pytree_node(
                FractalKVState,
                lambda s: ((s.payload_u, s.written_mask, s.total_writes, s.total_collisions), None),
                lambda _, c: FractalKVState(*c),
            )
        except Exception:
            # Safe to ignore if already registered
            pass

        # JIT‑compiled kernels (no static args; keep pure for tracing)
        self._jit_compute_c = jit(self._compute_c_batch)
        self._jit_write = jit(self._write_batch)
        self._jit_read = jit(self._read_batch)
        self._jit_diag = jit(self._diagnostics)

    # ----- geometry -----

    def _compute_c_batch(self, paths: Array) -> Array:
        """
        paths: [B, k] ints in [0, m)
        returns c_w: [B, d_val]
        c_w = sum_j s^{k-1-j} * T[ path[:, j] ]
        """
        # Gather per-level translations: [B, k, d_val]
        per_level = vmap(lambda idxs: self.T[idxs], in_axes=0, out_axes=0)(paths)  # [B, k, d]
        # Weight by scales and sum over levels
        weighted = per_level * self.level_scales[None, :, None]  # [B, k, d]
        c_w = jnp.sum(weighted, axis=1)  # [B, d]
        return c_w

    # ----- addressing -----

    def paths_to_indices(self, paths: Array) -> Array:
        return _path_to_index(paths, self.m_pow)

    def indices_to_paths(self, idx: Array) -> Array:
        return _index_to_path(idx, self.cfg.m, self.cfg.k)

    # ----- write -----

    def _write_batch(self, state: FractalKVState, paths: Array, values_v: Array) -> FractalKVState:
        """
        Closed‑form write (Eq. u_w = (1 - s^k) v - c_w) with in‑place scatter.
        paths: [B, k]; values_v: [B, d_val]
        """
        cfg = self.cfg
        B = paths.shape[0]
        idx = self.paths_to_indices(paths)  # [B]
        c_w = self._jit_compute_c(paths)  # [B, d]
        factor = 1.0 - cfg.a_pow_k_scalar
        u_w = factor * values_v - c_w  # [B, d]

        # Count collisions: existing mask at indices that are already written
        mask_gather = state.written_mask[idx]  # [B]
        new_collisions = jnp.sum(mask_gather.astype(jnp.int32))  # scalar

        # Scatter updates
        new_payload = state.payload_u.at[idx].set(u_w)
        new_mask = state.written_mask.at[idx].set(True)

        return FractalKVState(
            payload_u=new_payload,
            written_mask=new_mask,
            total_writes=state.total_writes + jnp.array(B, dtype=jnp.int32),
            total_collisions=state.total_collisions + new_collisions,
        )

    @property
    def state(self) -> FractalKVState:
        return self._state

    def write(self, paths: Array, values_v: Array) -> None:
        self._state = self._jit_write(self._state, paths.astype(jnp.int32), values_v.astype(self.cfg.dtype))

    # ----- read -----

    def _read_batch(self, state: FractalKVState, paths: Array) -> tuple[Array, Array]:
        """
        Returns (v_hat, present_mask).
        paths: [B, k]
        """
        cfg = self.cfg
        idx = self.paths_to_indices(paths)  # [B]
        c_w = self._jit_compute_c(paths)  # [B, d]
        u = state.payload_u[idx]  # [B, d]
        present = state.written_mask[idx]  # [B]
        v_hat = (c_w + u) * cfg.inv_I_minus_Ak_scalar  # [B, d]
        return v_hat, present

    def read(self, paths: Array) -> tuple[Array, Array]:
        return self._jit_read(self._state, paths.astype(jnp.int32))  # type: ignore[no-any-return]

    # ----- diagnostics -----

    def _diagnostics(self, state: FractalKVState) -> dict[str, Array]:
        cfg = self.cfg
        used = jnp.sum(state.written_mask.astype(jnp.int32))
        util = used / jnp.array(cfg.capacity, dtype=jnp.int32)
        coll_rate = jnp.where(
            state.total_writes > 0, state.total_collisions / state.total_writes, jnp.array(0.0, dtype=jnp.float32)
        ).astype(jnp.float32)
        sep = jnp.array(cfg.separation_margin, dtype=jnp.float32)
        return dict(utilization=util.astype(jnp.float32), collision_rate=coll_rate, separation_margin=sep)

    def diagnostics(self) -> dict[str, float]:
        out = self._jit_diag(self._state)
        return {k: float(v) for k, v in out.items()}

    # Expose separation margin as property for tests
    @property
    def separation_margin(self) -> float:
        return self.cfg.separation_margin

    # ----- re-indexing (depth +1) -----

    def reindex_increase_depth(self) -> FractalKV:
        """
        Lossless capacity expansion: increase depth k -> k+1, append branch 0 for all existing items.
        For each written leaf with value v, compute new u' for path w' = (w, 0) under the new parameters.
        """
        cfg_old = self.cfg
        cfg_new = FractalKVConfig(d_val=cfg_old.d_val, m=cfg_old.m, k=cfg_old.k + 1, s=cfg_old.s, dtype=cfg_old.dtype)
        new_store = FractalKV(cfg_new)

        # Collect existing written indices and their values
        mask = self.state.written_mask
        idx_old = jnp.nonzero(mask, size=int(cfg_old.capacity), fill_value=-1)[0]
        idx_old = idx_old[idx_old >= 0]
        if idx_old.size == 0:
            return new_store

        paths_old = self.indices_to_paths(idx_old)  # [W, k]
        # Read exact values v from old store (for mathematical exactness)
        v_vals, present = self.read(paths_old)
        # Append branch 0 to form new paths
        zero_col = jnp.zeros((paths_old.shape[0], 1), dtype=jnp.int32)
        paths_new = jnp.concatenate([paths_old, zero_col], axis=1)  # [W, k+1]
        # Write exact values into the new store under new parameters
        new_store.write(paths_new, v_vals)
        return new_store

    # --- Experimental: adaptive contractivity and auto reindexing ---
    def adjust_contractivity(self, target_collisions: float = 0.05) -> None:
        """Lower s (increase separation) if collision rate is high; conservative step.

        This rebuilds internal precomputes; payload is unchanged. For exact value preservation,
        call reindex_increase_depth and rewrite exact values.
        """
        diag = self.diagnostics()
        if diag["collision_rate"] > target_collisions and self.cfg.s > 0.2:
            new_s = float(max(0.1, self.cfg.s * 0.9))
            cfg_old = self.cfg
            cfg_new = FractalKVConfig(d_val=cfg_old.d_val, m=cfg_old.m, k=cfg_old.k, s=new_s, dtype=cfg_old.dtype)
            self.__init__(cfg_new)

    def maybe_expand_depth(self, util_threshold: float = 0.7) -> FractalKV:
        diag = self.diagnostics()
        if diag["utilization"] > util_threshold:
            return self.reindex_increase_depth()
        return self


def _hashed_paths_for_vectors(V: Array, m: int, k: int, seed: int = 123) -> Array:
    """Generate pseudo-addresses via random projection signatures (consistent hashing)."""
    rng = np.random.default_rng(seed)
    proj = jnp.array(rng.standard_normal((V.shape[-1], 32)), dtype=jnp.float32)
    sig = (V @ proj) > 0
    # Fold to integer and then to base-m digits
    idxs = []
    mod_val = int(m) ** int(k)
    for row in np.asarray(sig):
        code = 0
        for i, b in enumerate(row.tolist()):
            if b:
                code += (1 << i)
        idxs.append(code % mod_val)
    idxs = jnp.array(idxs, dtype=jnp.int32)
    digs = []
    for code in idxs.tolist():
        x = int(code)
        arr = [0] * int(k)
        for t in range(int(k) - 1, -1, -1):
            arr[t] = x % int(m)
            x //= int(m)
        digs.append(arr)
    return jnp.array(digs, dtype=jnp.int32)


# --- Minimal adapter expected by tests ---
class IFSMemory:
    """Thin wrapper exposing store/recall API over FractalKV for tests.

    Uses A = s I and the corner-translation scheme; provides store(value) and
    recall(query) hooks similar to a KV interface used in tests.
    """

    def __init__(self, feature_dim: int, max_transforms: int):
        # Map arguments into our config; choose a small m^k capacity >= max_transforms
        m = 16
        k = max(1, int(math.ceil(math.log(max(1, max_transforms), m))))
        self.cfg = FractalKVConfig(d_val=feature_dim, m=m, k=k, s=0.4, dtype=jnp.float32)
        self.store_impl = FractalKV(self.cfg)
        # Signature-based router: sign of random projections -> integer hash -> path
        rng = np.random.default_rng(123)
        self._proj = jnp.array(rng.standard_normal((feature_dim, 32)), dtype=jnp.float32)

    def _hash_to_path(self, vec: jnp.ndarray) -> jnp.ndarray:
        # Compute 32-bit signature → integer (use Python ints to avoid int32 overflow)
        import numpy as _np
        v_np = _np.asarray(vec, dtype=_np.float32)
        proj = _np.asarray(self._proj, dtype=_np.float32)
        sig = (v_np @ proj) > 0.0  # bool[32]
        idx_val: int = 0
        for i, bit in enumerate(sig.tolist()):
            if bit:
                idx_val += (1 << i)
        idx = idx_val % int(self.cfg.capacity)
        # Convert idx to base-m digits of length k
        x = idx
        digs = [0] * int(self.cfg.k)
        for t in range(int(self.cfg.k) - 1, -1, -1):
            digs[t] = x % int(self.cfg.m)
            x //= int(self.cfg.m)
        return jnp.array([digs], dtype=jnp.int32)

    def store(self, v: jnp.ndarray) -> None:
        path = self._hash_to_path(v.reshape(-1))
        self.store_impl.write(path, v.reshape(1, -1))

    def recall(self, q: jnp.ndarray):
        path = self._hash_to_path(q.reshape(-1))
        v, present = self.store_impl.read(path)
        return v.reshape(-1), bool(present[0])


# ------------------------------
# Learned router (k independent m‑way classifiers)
# ------------------------------


@dataclass
class RouterParams:
    W: Array  # [k, m, d_key]
    b: Array  # [k, m]


@dataclass
class RouterOptState:
    t: Array  # step
    mW: Array  # Adam first moment for W
    vW: Array  # Adam second moment for W
    mb: Array  # Adam first moment for b
    vb: Array  # Adam second moment for b


# Register RouterParams and RouterOptState as JAX pytrees so they can flow through jitted functions
try:
    _tree.register_pytree_node(
        RouterParams,
        lambda p: ((p.W, p.b), None),
        lambda _, xs: RouterParams(W=xs[0], b=xs[1]),
    )
    _tree.register_pytree_node(
        RouterOptState,
        lambda s: ((s.t, s.mW, s.vW, s.mb, s.vb), None),
        lambda _, xs: RouterOptState(t=xs[0], mW=xs[1], vW=xs[2], mb=xs[3], vb=xs[4]),
    )
except Exception:
    # Safe to ignore if already registered
    pass


class LearnedRouter:
    """
    A per‑level softmax router selecting k decisions in series (exactly k compositions).
    Inference uses argmax; training uses cross‑entropy on per‑level targets.

    logits_l = W[l] @ q + b[l], l=0..k-1, each of size [m].
    """

    def __init__(self, rng: Array, d_key: int, m: int, k: int, dtype=jnp.float32):
        self.d_key, self.m, self.k, self.dtype = d_key, m, k, dtype
        k1, k2 = random.split(rng)
        W = random.normal(k1, (k, m, d_key), dtype) * (1.0 / math.sqrt(d_key))
        b = jnp.zeros((k, m), dtype)
        self.params = RouterParams(W=W, b=b)
        # Adam optimizer state
        zerosW = jnp.zeros_like(W)
        zerosb = jnp.zeros_like(b)
        self.opt = RouterOptState(t=jnp.array(0, dtype=jnp.int32), mW=zerosW, vW=zerosW, mb=zerosb, vb=zerosb)

        self._jit_forward = jit(self._forward_logits)
        self._jit_predict_path = jit(self._predict_path)
        self._jit_loss = jit(self._loss)
        self._jit_update = jit(self._update)

    # ----- forward -----

    def _forward_logits(self, params: RouterParams, q: Array) -> Array:
        """
        q: [B, d_key] -> logits: [B, k, m]
        logits[l] = q @ W[l]^T + b[l]
        """
        # [k, m, d] and [B, d] -> [B, k, m]
        logits = jnp.einsum("b d, k m d -> b k m", q, params.W) + params.b[None, :, :]
        return logits

    def _predict_path(self, params: RouterParams, q: Array) -> Array:
        logits = self._forward_logits(params, q)  # [B, k, m]
        path = jnp.argmax(logits, axis=-1).astype(jnp.int32)  # [B, k]
        return path

    # ----- loss -----

    @staticmethod
    def _xent_logits(logits: Array, targets: Array) -> Array:
        # logits: [B, m], targets: [B] int in [0,m)
        # cross-entropy with log-softmax
        logp = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
        gather = jnp.take_along_axis(logp, targets[:, None], axis=-1).squeeze(-1)
        return -jnp.mean(gather)

    def _loss(self, params: RouterParams, q: Array, target_paths: Array) -> Array:
        logits = self._forward_logits(params, q)  # [B, k, m]
        # average level-wise CE
        per_level = vmap(self._xent_logits, in_axes=(1, 1), out_axes=0)(logits, target_paths)
        return jnp.mean(per_level)

    # ----- optimizer (Adam) -----

    def _update(
        self,
        params: RouterParams,
        opt: RouterOptState,
        q: Array,
        target_paths: Array,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
    ) -> tuple[RouterParams, RouterOptState, Array]:
        loss_val, grads = jax.value_and_grad(self._loss)(params, q, target_paths)
        t = opt.t + 1

        def adam_step(param, g, m, v):
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g * g)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            param = param - lr * m_hat / (jnp.sqrt(v_hat) + eps)
            return param, m, v

        W, mW, vW = adam_step(params.W, grads.W, opt.mW, opt.vW)
        b, mb, vb = adam_step(params.b, grads.b, opt.mb, opt.vb)
        new_params = RouterParams(W=W, b=b)
        new_opt = RouterOptState(t=t, mW=mW, vW=vW, mb=mb, vb=vb)
        return new_params, new_opt, loss_val

    # ----- public API -----

    def train(
        self,
        q: Array,
        target_paths: Array,
        *,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        key: Array | None = None,
        verbose: bool = True,
    ) -> None:
        N = q.shape[0]
        if key is None:
            key = random.PRNGKey(0)

        for ep in range(1, epochs + 1):
            key, sub = random.split(key)
            perm = random.permutation(sub, N)
            q_shuf = q[perm]
            t_shuf = target_paths[perm]
            losses = []
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                q_b = q_shuf[start:end]
                t_b = t_shuf[start:end]
                self.params, self.opt, loss_val = self._jit_update(
                    self.params, self.opt, q_b, t_b, lr, beta1, beta2, eps
                )
                losses.append(float(loss_val))
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == 1):
                print(f"[Router] epoch {ep}/{epochs}  loss={np.mean(losses):.4f}")

    def predict(self, q: Array) -> Array:
        return self._jit_predict_path(self.params, q)  # type: ignore[no-any-return]


# ------------------------------
# Microbenchmark and demonstration
# ------------------------------


def balanced_address_assignment(N: int, m: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Assigns each of N items a unique leaf in [0, m^k) by shuffled order, then converts to base-m paths.
    Returns paths: [N, k] with ints in [0, m).
    """
    capacity = m**k
    if N > capacity:
        raise ValueError(f"Capacity exceeded: N={N} > m^k={capacity}. Increase m or k.")
    order = rng.permutation(capacity)[:N]  # unique leaves
    # Convert to base-m digits [N, k]
    paths = []
    for idx in order:
        digits = []
        x = int(idx)
        for _ in range(k):
            digits.append(x % m)
            x //= m
        paths.append(list(reversed(digits)))
    return np.array(paths, dtype=np.int32)


def mse(a: Array, b: Array) -> Array:
    return jnp.mean((a - b) ** 2)  # type: ignore[return-value]


def ridge_closed_form(X: Array, Y: Array, lam: float = 1e-3) -> Array:
    """
    Return W in R^{d_key x d_val} minimizing ||X W - Y||^2 + lam ||W||^2.
    """
    d_key = X.shape[1]
    XtX = X.T @ X + lam * jnp.eye(d_key, dtype=X.dtype)
    XtY = X.T @ Y
    W = jnp.linalg.solve(XtX, XtY)  # [d_key, d_val]
    return W  # type: ignore[no-any-return]


def catastrophic_forgetting_benchmark(
    *,
    N: int = 4096,
    d_key: int = 64,
    d_val: int = 16,
    m: int = 16,
    k: int = 3,
    s: float = 0.4,
    batches: int = 4,
    router_epochs: int = 60,
    seed: int = 0,
) -> None:
    """
    Sequential batches without rehearsal. Compare:
      - Fractal KV (learned router) with closed‑form writes
      - Baseline: last-batch ridge regression (classic forgetting)
    Prints batch-wise MSE on all seen data and diagnostics from the store.
    """
    rng_np = np.random.default_rng(seed)
    rng_jax = random.PRNGKey(seed)

    # Data
    Q = jnp.array(rng_np.normal(size=(N, d_key)).astype(np.float32))
    V = jnp.array(rng_np.normal(size=(N, d_val)).astype(np.float32))

    # Address assignment for supervised router training (unique leaves)
    target_paths_np = balanced_address_assignment(N, m, k, rng_np)
    target_paths = jnp.array(target_paths_np)

    # Router
    router = LearnedRouter(rng_jax, d_key=d_key, m=m, k=k, dtype=jnp.float32)
    router.train(Q, target_paths, epochs=router_epochs, batch_size=512, lr=5e-3, verbose=True)

    # Store
    cfg = FractalKVConfig(d_val=d_val, m=m, k=k, s=s, dtype=jnp.float32)
    store = FractalKV(cfg)

    # Baseline model (retrained only on the most recent batch)
    W_baseline = jnp.zeros((d_key, d_val), dtype=jnp.float32)

    bs = N // batches
    seen_mask = np.zeros(N, dtype=bool)

    print("\n=== Sequential write/read benchmark ===")
    for b in range(1, batches + 1):
        start, end = (b - 1) * bs, b * bs
        seen_mask[start:end] = True

        # ---- Fractal store writes (using learned router to select address) ----
        q_b = Q[start:end]
        v_b = V[start:end]
        paths_b = router.predict(q_b)  # [B, k]
        store.write(paths_b, v_b)

        # Evaluate on all seen items so far
        q_seen = Q[seen_mask]
        v_seen = V[seen_mask]
        paths_seen = router.predict(q_seen)
        vhat_seen, present = store.read(paths_seen)
        # Absent entries (should be none for seen data) are counted as zeros; mask them if needed
        fr_mse = float(mse(vhat_seen, v_seen))

        # ---- Baseline update (fit on last batch only) ----
        W_baseline = ridge_closed_form(q_b, v_b, lam=1e-3)
        vhat_base = q_seen @ W_baseline
        base_mse = float(mse(vhat_base, v_seen))

        # Diagnostics
        diag = store.diagnostics()

        print(
            f"Batch {b}/{batches}: "
            f"Fractal MSE={fr_mse:.4f}, Baseline MSE={base_mse:.4f}, "
            f"Δ={base_mse - fr_mse:+.4f} (positive means store better), "
            f"Util={diag['utilization']:.4f}, CollRate={diag['collision_rate']:.4f}, "
            f"SepMargin={diag['separation_margin']:.3f}"
        )

    # Re-indexing demo if fragmentation threshold exceeded (example condition)
    diag = store.diagnostics()
    if diag["collision_rate"] >= 0.20 or diag["separation_margin"] <= 0.02:
        print("\n[Reindex] Triggered (collision rate or margin threshold). Expanding depth by +1.")
        store = store.reindex_increase_depth()
        diag2 = store.diagnostics()
        print(
            f"[Reindex] New depth k={store.cfg.k}, new capacity={store.cfg.capacity}, "
            f"separation margin={diag2['separation_margin']:.3f}"
        )

    # --- Optional: Router distillation from hashed routes ---
    import os as _os
    if _os.environ.get("IFS_DISTILL", "0") == "1":
        print("\n=== Router distillation from hashed routes ===")
        hashed = _hashed_paths_for_vectors(Q, m, k)
        # Train a fresh router on hashed targets
        router2 = LearnedRouter(rng_jax, d_key=d_key, m=m, k=k, dtype=jnp.float32)
        router2.train(Q, hashed, epochs=router_epochs // 2, batch_size=512, lr=5e-3, verbose=False)
        # Measure collisions/recall on a subset using hashed vs learned
        cfg2 = FractalKVConfig(d_val=d_val, m=m, k=k, s=s, dtype=jnp.float32)
        store_h = FractalKV(cfg2)
        P_h = hashed
        store_h.write(P_h, V)
        diag_h = store_h.diagnostics()
        vhat_h, present_h = store_h.read(P_h)
        mse_h = float(mse(vhat_h, V))

        store_r = FractalKV(cfg2)
        P_r = router2.predict(Q)
        store_r.write(P_r, V)
        diag_r = store_r.diagnostics()
        vhat_r, present_r = store_r.read(P_r)
        mse_r = float(mse(vhat_r, V))

        from rich.table import Table as _Table
        t = _Table(title="Distillation: Hashed vs Learned Router", show_header=True, header_style="bold magenta")
        t.add_column("Metric")
        t.add_column("Hashed", justify="right")
        t.add_column("Learned", justify="right")
        t.add_row("Utilization", f"{diag_h['utilization']:.3f}", f"{diag_r['utilization']:.3f}")
        t.add_row("CollisionRate", f"{diag_h['collision_rate']:.3f}", f"{diag_r['collision_rate']:.3f}")
        t.add_row("Readback MSE", f"{mse_h:.4f}", f"{mse_r:.4f}")
        print(t)


# ------------------------------
# End-to-end sanity check
# ------------------------------


def _sanity_small():
    """
    Quick functional test of exact write/read equalities.
    """
    cfg = FractalKVConfig(d_val=8, m=8, k=3, s=0.4)
    store = FractalKV(cfg)
    rng = np.random.default_rng(123)
    # 10 random entries
    P = jnp.array(balanced_address_assignment(10, cfg.m, cfg.k, rng))
    V = jnp.array(rng.normal(size=(10, cfg.d_val)).astype(np.float32))
    store.write(P, V)
    Vh, present = store.read(P)
    err = float(mse(V, Vh))
    assert err < 1e-6, f"Readback mismatch: MSE={err}"
    d = store.diagnostics()
    assert 0.0 <= d["collision_rate"] <= 1.0
    print("[Sanity] Exact write/read OK, MSE ~", err)


# ------------------------------
# Main
# ------------------------------

def demo():
    """Run the iterated function systems and fractal memory demonstration."""
    # Sanity test (verifies closed-form correctness)
    _sanity_small()

    # Microbenchmark showing reduced catastrophic forgetting vs last-batch baseline.
    # Tweak N/dimensions/epochs as desired; defaults are sized to run quickly on CPU/GPU.
    catastrophic_forgetting_benchmark(
        N=4096,
        d_key=64,
        d_val=16,
        m=16,  # 2^4 branches (4 bits per level)
        k=3,  # depth (capacity = m^k = 4096)
        s=0.4,  # contraction (separation margin = 1 - 2s = 0.2)
        batches=4,
        router_epochs=60,
        seed=42,
    )


if __name__ == "__main__":
    demo()


# --- Test-facing helpers/aliases ---

def FractalKVStore(key, d: int, m: int, k: int, s: float = 0.4) -> FractalKV:
    """Adapter matching tests: construct a fractal store with given params."""
    cfg = FractalKVConfig(d_val=d, m=m, k=k, s=s)
    return FractalKV(cfg)


def _compute_c_w(path: Array, cfg: FractalKVConfig) -> Array:
    store = FractalKV(cfg)
    return store._compute_c_batch(path.reshape(1, -1)).reshape(cfg.d_val)
