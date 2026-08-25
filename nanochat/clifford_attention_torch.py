"""
Clifford Attention Module (PyTorch) - Cl(3,0)
=============================================

Production implementation of framework #12 (bead model_guided_research-mnn.3),
built on the A1 AttentionCore scaffold. The JAX demo
(clifford_algebra_and_geometric_attention.py) derives and validates the math;
this file realizes it against real training machinery.

Design (per the mnn.2 demo doc):
- head_dim is interpreted as chunks of 8 components = one Cl(3,0) multivector
  (1 scalar, e1..e3, e12/e13/e23, e123). Constraint validated at config time.
- Scores are the standard scaled dot product; the Clifford content lives in
  aggregation: y_i = sum_j p_ij * ((Q_i * reverse(K_j)) * V_j) with the EXACT
  parenthesization preserved (associative algebra, so no parenthesization
  hazards - but we still never reorder the product).
- Aggregation is chunked over queries (octonion_attention_torch.py precedent,
  bead 7b0.6): tiles of height c chosen so
  B*H*c*Tk*N*8*dtype_size <= budget (NANOCHAT_CLIFFORD_TILE_BUDGET_MB env
  override; CUDA free-memory fraction; flat CPU default).
- Diagnostic: ``last_rotor_norm`` records the mean coefficient norm of the
  rotor multivectors Q*reverse(K) computed during the most recent aggregate
  (D2 telemetry consumers read it post-forward).

The blade multiplication table is BUILT PROGRAMMATICALLY at import time -
identical derivation to the demo (sign-tracked canonical reordering plus
metric contraction), verified by tests against known identities.
"""

from __future__ import annotations

import os

import torch

from nanochat.model_utils import AttentionCore

# ---------------------------------------------------------------------------
# Programmatic Cl(3,0) structure tensor
# ---------------------------------------------------------------------------

_BLADES: tuple[tuple[int, ...], ...] = (
    (),
    (1,),
    (2,),
    (3,),
    (1, 2),
    (1, 3),
    (2, 3),
    (1, 2, 3),
)
_METRIC = {1: 1.0, 2: 1.0, 3: 1.0}
_BLADE_INDEX: dict[tuple[int, ...], int] = {blade: i for i, blade in enumerate(_BLADES)}

#: Grade of each canonical blade (for reversion signs).
_GRADES = (0, 1, 1, 1, 2, 2, 2, 3)


def _blade_product(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[tuple[int, ...], float]:
    seq: list[int] = list(a) + list(b)
    sign = 1.0
    moved = True
    while moved:
        moved = False
        for i in range(len(seq) - 1):
            if seq[i] > seq[i + 1]:
                seq[i], seq[i + 1] = seq[i + 1], seq[i]
                sign = -sign
                moved = True
    out: list[int] = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == seq[i + 1]:
            sign *= _METRIC[seq[i]]
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out), sign


def _build_structure_tensor() -> torch.Tensor:
    m = torch.zeros(8, 8, 8)
    for i, ba in enumerate(_BLADES):
        for j, bb in enumerate(_BLADES):
            blade, sign = _blade_product(ba, bb)
            m[i, j, _BLADE_INDEX[blade]] = sign
    return m


_STRUCTURE: torch.Tensor = _build_structure_tensor()
_REVERSION_SIGNS: torch.Tensor = torch.tensor([1.0 if (g * (g - 1) // 2) % 2 == 0 else -1.0 for g in _GRADES])


def cgp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Geometric product of multivectors (..., 8) via the derived table."""

    return torch.einsum("...i,...j,ijk->...k", a, b, _STRUCTURE.to(a.dtype))


def creverse(m: torch.Tensor) -> torch.Tensor:
    """Reversion: sign (-1)^{g(g-1)/2} per grade."""

    return m * _REVERSION_SIGNS.to(dtype=m.dtype, device=m.device)


# ---------------------------------------------------------------------------
# Memory-budgeted query tiling (7b0.6 precedent)
# ---------------------------------------------------------------------------


def _tile_budget_bytes(device: torch.device) -> int:
    """Peak-memory budget for ONE query tile (mirrors octonion 7b0.6):

    1. ``NANOCHAT_CLIFFORD_TILE_BUDGET_MB`` env var (positive float, MB).
    2. CUDA device: 1/16 of currently-free memory (floor 1 MiB).
    3. CPU: flat 32 MiB default - the gp einsum holds roughly operands plus
       one output working set; measured fastest around cache-resident tiles.
    """
    env = os.environ.get("NANOCHAT_CLIFFORD_TILE_BUDGET_MB")
    if env is not None:
        mb = float(env)
        if mb <= 0:
            raise ValueError(f"NANOCHAT_CLIFFORD_TILE_BUDGET_MB must be > 0, got {env!r}")
        return int(mb * (1 << 20))
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        return max(int(free_bytes) // 16, 1 << 20)
    return 32 << 20


# ---------------------------------------------------------------------------
# Mechanism
# ---------------------------------------------------------------------------


class CliffordCausalSelfAttention(AttentionCore):
    """Clifford signal flow: standard scalar scores, geometric-product mixing.

    The value update is Y_i = sum_j p_ij * ((Q_i * reverse(K_j)) * V_j).
    Q/K are per-channel normalized (house rotor-mechanism convention) so the
    products act as norm-controlled transformations.
    """

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        if self.n_embd % 8 != 0:
            raise ValueError("n_embd must be divisible by 8 for Clifford attention")
        if self.head_dim % 8 != 0:
            raise ValueError("head_dim must be divisible by 8 for Clifford attention")
        # Diagnostic for the D2 stream: mean coefficient norm of the rotors
        # Q * reverse(K) from the most recent aggregate.
        self.last_rotor_norm = float("nan")

    def score(self, q, k):
        return (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))

    def aggregate(self, weights, v, *, q, k, kv_cache, pos0):
        B, H, Tq = q.size(0), q.size(1), q.size(2)
        Tk = k.size(2)
        n = self.head_dim // 8

        # (B, H, T, D) -> (B, H, T, N, 8): one Cl(3,0) multivector per chunk.
        q_m = q.view(B, H, Tq, n, 8)
        k_m = k.view(B, H, Tk, n, 8)
        v_m = v.view(B, H, Tk, n, 8)

        k_rev = creverse(k_m)
        bytes_per_query_row = B * H * Tk * n * 8 * q_m.element_size()
        tile = max(1, min(Tq, _tile_budget_bytes(q_m.device) // max(bytes_per_query_row, 1)))

        rotor_norm_acc = 0.0
        y_tiles = []
        for t0 in range(0, Tq, tile):
            q_t = q_m[:, :, t0 : t0 + tile].unsqueeze(3)  # (B,H,c,1,N,8)
            r_t = cgp(q_t, k_rev.unsqueeze(2))  # rotors Q*reverse(K): (B,H,c,Tk,N,8)
            term = cgp(r_t, v_m.unsqueeze(2))  # ((Q*K~)*V)
            p_t = weights[:, :, t0 : t0 + tile].unsqueeze(-1).unsqueeze(-1)
            y_tiles.append((term * p_t).sum(dim=3))
            rotor_norm_acc += float(r_t.norm(dim=-1).mean().detach())

        self.last_rotor_norm = rotor_norm_acc / max(1, -(-Tq // tile))

        y_o = y_tiles[0] if len(y_tiles) == 1 else torch.cat(y_tiles, dim=2)

        return y_o.reshape(B, H, Tq, self.head_dim)
