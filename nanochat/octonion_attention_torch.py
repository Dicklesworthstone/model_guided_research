"""
Octonion Attention Module (PyTorch)
Implements Octonion-based attention using the Cayley-Dickson construction over Quaternions.
Octonions are 8D hypercomplex numbers. Multiplication is non-associative.
"""

import os

import torch
import torch.nn.functional as F

from nanochat.model_utils import AttentionCore
from nanochat.quaternion_attention_torch import qconj, qmul

# Octonion Multiplication via Cayley-Dickson
# O1 = (a, b), O2 = (c, d) where a,b,c,d are Quaternions.
# O1 * O2 = (a*c - d_conj*b, d*a + b*c_conj)
# Note: Order matters! Octonions are non-associative.


def _cd_mul_tuples(x, y):
    """Exact Cayley-Dickson product on nested real tuples (import-time only,
    independent of the torch kernels below): (a, b) * (c, d)
    = (a*c - conj(d)*b, d*a + b*conj(c))."""

    def neg(t):
        if isinstance(t, float):
            return -t
        return tuple(neg(v) for v in t)

    def add(p, q):
        if isinstance(p, float):
            return p + q
        return tuple(add(u, v) for u, v in zip(p, q, strict=True))

    def conj(t):
        if isinstance(t, float):
            return t
        a, b = t
        return (conj(a), neg(b))

    if isinstance(x, float):
        return x * y
    a, b = x
    c, d = y
    return (
        add(_cd_mul_tuples(a, c), neg(_cd_mul_tuples(conj(d), b))),
        add(_cd_mul_tuples(d, a), _cd_mul_tuples(b, conj(c))),
    )


def _unit_blade_tuple(i: int, depth: int = 3):
    """Unit blade e_i as nested Cayley-Dickson tuples."""

    def zero(level: int):
        if level == 0:
            return 0.0
        return (zero(level - 1), zero(level - 1))

    def build(level: int, pos: int):
        if level == 0:
            return 1.0 if pos == 0 else 0.0
        half = 2 ** (level - 1)
        if pos < half:
            return (build(level - 1, pos), zero(level - 1))
        return (zero(level - 1), build(level - 1, pos - half))

    return build(depth, i)


def _flatten_tuple(x, out):
    if isinstance(x, tuple):
        for v in x:
            _flatten_tuple(v, out)
    else:
        out.append(float(x))
    return out


def _build_blade_sign_table() -> tuple[tuple[tuple[int, int], ...], ...]:
    """e_i * e_j -> (blade k, sign s) with e_i*e_j == s * e_k, derived once
    from the exact tuple recursion above (every entry is +-1/0 arithmetic)."""
    table = []
    for i in range(8):
        row = []
        for j in range(8):
            flat = _flatten_tuple(_cd_mul_tuples(_unit_blade_tuple(i), _unit_blade_tuple(j)), [])
            ks = [(idx, int(v)) for idx, v in enumerate(flat) if v != 0.0]
            if len(ks) != 1:
                raise AssertionError(f"blade product e{i}*e{j} is not a unit blade: {flat}")
            row.append(tuple(ks[0]))
        table.append(tuple(row))
    return tuple(table)


#: _BLADE_SIGN_TABLE[i][j] = (k, s) meaning e_i * e_j = s * e_k.
_BLADE_SIGN_TABLE = _build_blade_sign_table()


def omul(o1, o2):
    """
    Multiply octonion tensors o1 and o2.
    Shape: (..., 8)
    Splits into two quaternions (..., 4).
    """
    a, b = torch.split(o1, 4, dim=-1)
    c, d = torch.split(o2, 4, dim=-1)

    # a*c
    ac = qmul(a, c)
    # d_conj * b
    db = qmul(qconj(d), b)
    # d*a
    da = qmul(d, a)
    # b*c_conj
    bc = qmul(b, qconj(c))

    first = ac - db
    second = da + bc

    return torch.cat([first, second], dim=-1)


def _omul_blades(o1, o2):
    """Blade-table octonion product: mathematically IDENTICAL to ``omul``
    (same multiplication table, guarded by tests/test_algebraic_properties.py
    against an independent Cayley-Dickson oracle) but evaluated as 48 fused
    last-dim component products instead of chained quaternion blocks.

    Used by the tiled attention aggregate (bead 7b0.6): the chained form
    materializes ~a dozen (B, H, c, Tk, N, 8) intermediates per product and
    becomes memory-bandwidth-bound at tile heights that would otherwise be
    efficient, while this form holds roughly operands + 8 accumulators.
    """
    c1 = o1.unbind(-1)
    c2 = o2.unbind(-1)
    out: list[torch.Tensor | None] = [None] * 8
    for i in range(8):
        ai = c1[i]
        row = _BLADE_SIGN_TABLE[i]
        for j in range(8):
            k, s = row[j]
            if s == 0:
                continue
            term = ai * c2[j]
            acc = out[k]
            if acc is None:
                out[k] = term if s > 0 else -term
            elif s > 0:
                out[k] = acc + term
            else:
                out[k] = acc - term
    # _BLADE_SIGN_TABLE produces every output blade exactly once; the guard
    # converts that invariant into something the type checker can see.
    filled = [t for t in out if t is not None]
    assert len(filled) == 8, "corrupt blade table: some octonion component unproduced"
    return torch.stack(filled, dim=-1)


def oconj(o):
    """
    Conjugate of octonion o = (a, b) is (a_conj, -b).
    """
    a, b = torch.split(o, 4, dim=-1)
    return torch.cat([qconj(a), -b], dim=-1)


def onorm(o):
    return torch.norm(o, dim=-1, keepdim=True)


def onormalize(o):
    return F.normalize(o, p=2, dim=-1)


def _tile_budget_bytes(device: torch.device) -> int:
    """Peak-memory budget for ONE vectorized query tile (bead 7b0.6).

    The tiled aggregate materializes a (B, H, c, Tk, N, 8) working set per
    tile; this budget caps the tile height ``c`` via
    ``B*H*c*Tk*N*8*dtype_size <= budget``. Precedence:

    1. ``NANOCHAT_OCTONION_TILE_BUDGET_MB`` env var (positive float, MB).
    2. CUDA device: 1/16 of currently-free memory (floor 1 MiB).
    3. CPU: flat 8 MiB default - measured fastest on the bead's microbench
       shape (B=8, T=256, D=128): tiles stay cache-resident; larger budgets
       go bandwidth-bound on the fused product's working set.
    """
    env = os.environ.get("NANOCHAT_OCTONION_TILE_BUDGET_MB")
    if env is not None:
        mb = float(env)
        if mb <= 0:
            raise ValueError(f"NANOCHAT_OCTONION_TILE_BUDGET_MB must be > 0, got {env!r}")
        return int(mb * (1 << 20))
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        return max(int(free_bytes) // 16, 1 << 20)
    return 8 << 20


class OctonionCausalSelfAttention(AttentionCore):
    """Octonionic signal flow: standard scalar scores, non-associative mixing.

    The value update is Y_i = sum_j probs_ij * ((Q_i * conj(K_j)) * V_j) with
    the EXPLICIT parenthesization (rotor first, then value): octonions are
    non-associative, so unlike the quaternion rotor-gate the query CANNOT be
    factored out of the sum - the pairwise products are intrinsic to the
    mechanism and cost O(T^2 * D). Q/K are per-channel normalized so the
    octonion multiplications act as norm-preserving "rotors".
    """

    def __init__(self, config, layer_idx):
        # Standard linear projections; the output is interpreted as
        # head_dim/8 octonions per head.
        super().__init__(config, layer_idx)
        if self.n_embd % 8 != 0:
            raise ValueError("n_embd must be divisible by 8 for Octonion attention")
        if self.head_dim % 8 != 0:
            raise ValueError("head_dim must be divisible by 8 for Octonion attention")

    def score(self, q, k):
        return (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))

    def aggregate(self, weights, v, *, q, k, kv_cache, pos0):
        B = q.size(0)
        Tq = q.size(2)

        # Interpret as octonions: (..., D) -> (..., D/8, 8).
        q_o = onormalize(q.view(B, self.n_head, -1, self.head_dim // 8, 8))
        k_o = onormalize(k.view(B, self.n_head, -1, self.head_dim // 8, 8))
        v_o = v.view(B, self.n_head, -1, self.head_dim // 8, 8)

        # Chunked vectorization (bead model_guided_research-7b0.6): queries are
        # processed in tiles of height c chosen from _tile_budget_bytes so
        # B*H*c*Tk*N*8*dtype_size stays within the memory budget. The math is
        # UNCHANGED: each (query, key) rotor keeps the exact parenthesization
        # ((Q_i * conj(K_j)) * V_j); tiling only batches independent products
        # into one kernel launch per tile. Products use the fused blade-table
        # kernel (_omul_blades), which reproduces omul's multiplication table.
        k_conj = oconj(k_o)
        bytes_per_query_row = B * self.n_head * k_o.size(2) * k_o.size(3) * 8 * q_o.element_size()
        tile = max(1, min(Tq, _tile_budget_bytes(q_o.device) // max(bytes_per_query_row, 1)))
        y_tiles = []
        for t0 in range(0, Tq, tile):
            q_t = q_o[:, :, t0 : t0 + tile].unsqueeze(3)  # (B, H, c, 1, N, 8)
            r_t = _omul_blades(q_t, k_conj.unsqueeze(2))  # rotors Q*conj(K): (B, H, c, Tk, N, 8)
            term = _omul_blades(r_t, v_o.unsqueeze(2))  # (Q*conj(K))*V, same parenthesization
            p_t = weights[:, :, t0 : t0 + tile].unsqueeze(-1).unsqueeze(-1)  # (B, H, c, Tk, 1, 1)
            y_tiles.append((term * p_t).sum(dim=3))  # (B, H, c, N, 8)

        y_o = y_tiles[0] if len(y_tiles) == 1 else torch.cat(y_tiles, dim=2)
        return y_o.reshape(B, self.n_head, Tq, self.head_dim)
