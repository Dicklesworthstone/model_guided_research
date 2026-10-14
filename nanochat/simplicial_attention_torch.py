"""
Simplicial Attention Module (PyTorch)

Supports two tiers selectable via ``GPTConfig.simplicial_triplet``:
1. Proxy tier (``simplicial_triplet=False``, default for backward golden/ledger parity):
   2-hop pairwise random walk ``y = mix_1 * (A v) + mix_2 * (A (A v))`` on the 1-skeleton.
2. Faithful 2-simplex cochain tier (``simplicial_triplet=True``, bead jida.14):
   Over each query's causal window of size ``simplicial_window`` (default 8),
   constructs oriented causal 2-simplices ``sigma = (j_1, j_2, i)`` (``j_1 < j_2 <= i``)
   with antisymmetric coboundary wedge 2-cochain score:
       Omega^wedge_{i, j_1, j_2} = <q_i, k_{j_1} * U_wedge k_{j_2} - k_{j_2} * U_wedge k_{j_1}> / sqrt(d_h)
   and aggregates values via the oriented boundary operator ``partial_2``:
       y_2(i) = sum_{j_1 < j_2 <= i} alpha^{(2)}_{i, j_1, j_2} [ (v_{j_1} + v_{j_2}) / 2 + W_partial (v_{j_2} - v_{j_1}) ]
   which satisfies exact orientation antisymmetry (swapping j_1 <-> j_2 negates Omega^wedge),
   exact constant-vector mass conservation (sum_m gamma^+_m = 1 and sum_m gamma^-_m = 0),
   and exact reduction to standard causal SDPA when ``mix_1 = 1, mix_2 = 0``.
"""

from typing import Any

import torch
import torch.nn as nn

from nanochat.model_utils import AttentionCore


class SimplicialCausalSelfAttention(AttentionCore):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)

        # Mixing weights for 1-simplex (Edge) and 2-simplex (Triangle / 2-hop) paths
        self.mix_1 = nn.Parameter(torch.tensor(1.0))
        self.mix_2 = nn.Parameter(torch.tensor(0.5))

        self.simplicial_triplet = bool(getattr(config, "simplicial_triplet", False))
        self.simplicial_window = int(getattr(config, "simplicial_window", 8))
        self.simplicial_random_triangles = bool(getattr(config, "simplicial_random_triangles", False))
        self.last_simplicial_stats: dict[str, Any] | None = None

        if self.simplicial_triplet:
            self.wedge_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
            self.boundary_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        else:
            self.wedge_proj = None
            self.boundary_proj = None

    def _ensure_triplet_projections(self, device: torch.device, dtype: torch.dtype) -> tuple[nn.Linear, nn.Linear]:
        """Return (wedge_proj, boundary_proj), lazily constructing deterministic
        projections if ``simplicial_triplet`` was toggled on after ``__init__``."""
        if self.wedge_proj is None or self.boundary_proj is None:
            saved_rng = torch.random.get_rng_state()
            try:
                torch.manual_seed(1729 + int(self.layer_idx))
                self.wedge_proj = nn.Linear(self.head_dim, self.head_dim, bias=False).to(device=device, dtype=dtype)
                self.boundary_proj = nn.Linear(self.head_dim, self.head_dim, bias=False).to(device=device, dtype=dtype)
            finally:
                torch.random.set_rng_state(saved_rng)
        return self.wedge_proj, self.boundary_proj

    def compute_wedge_matrix(self, q: torch.Tensor, k_win: torch.Tensor) -> torch.Tensor:
        """Compute the antisymmetric 2-simplex coboundary wedge matrix ``M_i[a, b]``
        of shape ``(B, H, Tq, w, w)`` for query ``q`` ``(B, H, Tq, D)`` and windowed
        keys ``k_win`` ``(B, H, Tq, w, D)`` (or unwindowed ``(B, H, w, D)``):

            M_i[a, b] = <q_i, k_a * (U_wedge k_b) - k_b * (U_wedge k_a)> / sqrt(D)

        Satisfies exact skew-symmetry ``M_i[b, a] = -M_i[a, b]``.
        """
        wedge_proj, _ = self._ensure_triplet_projections(q.device, q.dtype)
        if k_win.dim() == 4:
            k_win = k_win.unsqueeze(2)
        # F_i[a] = q_i * k_a, G_i[b] = U_wedge k_b -> (B, H, Tq, w, D)
        f_mat = q.unsqueeze(-2) * k_win
        g_mat = wedge_proj(k_win)
        fg = torch.matmul(f_mat, g_mat.transpose(-2, -1))
        return (fg - fg.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))

    def _aggregate_triplet(
        self,
        weights: torch.Tensor,
        v: torch.Tensor,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        pos0: int | None,
    ) -> torch.Tensor:
        """Oriented 2-simplex cochain attention over causal triples ``(j_1 < j_2 <= i)``
        inside a local window of width ``w = min(Tk, max(2, simplicial_window))``."""
        B, H, Tq, Tk = weights.shape
        pos0_val = 0 if pos0 is None else int(pos0)
        w = min(Tk, max(2, int(self.simplicial_window)))
        if w < 2 or Tk < 2:
            self.last_simplicial_stats = {
                "triangle_entropy": 0.0,
                "hodge_2simplex_share": 0.0,
            }
            return weights @ v

        _, boundary_proj = self._ensure_triplet_projections(v.device, v.dtype)

        # Build causal window key positions for each query i in [0, Tq):
        # raw_pos[i, m] = (pos0_val + i) - (w - 1) + m, so raw_pos[i, -1] == pos0_val + i.
        offsets = torch.arange(-(w - 1), 1, device=q.device, dtype=torch.long)  # (w,)
        t_idx = torch.arange(pos0_val, pos0_val + Tq, device=q.device, dtype=torch.long).unsqueeze(1)  # (Tq, 1)
        raw_pos = t_idx + offsets.unsqueeze(0)  # (Tq, w)
        safe_pos = raw_pos.clamp(0, Tk - 1)  # (Tq, w)

        # Gather windowed keys, values, and 1-simplex attention weights
        k_win = k[:, :, safe_pos, :]  # (B, H, Tq, w, D)
        v_win = v[:, :, safe_pos, :]  # (B, H, Tq, w, D)
        bv_win = boundary_proj(v_win)  # (B, H, Tq, w, D)
        t_gather = torch.arange(Tq, device=q.device, dtype=torch.long).unsqueeze(1)
        w1_win = weights[:, :, t_gather, safe_pos]  # (B, H, Tq, w)

        # Antisymmetric wedge scores on all pairs (a, b) in the w x w window
        wedge_mat = self.compute_wedge_matrix(q, k_win)  # (B, H, Tq, w, w)
        log_w1 = torch.log(w1_win.clamp_min(1e-12))
        pair_scores = wedge_mat + log_w1.unsqueeze(-1) + log_w1.unsqueeze(-2)  # (B, H, Tq, w, w)

        # Extract upper-triangular oriented causal pairs 0 <= a < b < w
        tri_a, tri_b = torch.triu_indices(w, w, offset=1, device=q.device)
        num_tri = int(tri_a.numel())
        tri_scores = pair_scores[:, :, :, tri_a, tri_b]  # (B, H, Tq, num_tri)

        # A triangle (j_1, j_2, i) is valid iff both j_1 and j_2 are >= 0 and j_1 < j_2 <= t_i
        valid_tri = (raw_pos[:, tri_a] >= 0) & (raw_pos[:, tri_b] <= t_idx)  # (Tq, num_tri)
        has_any_tri = valid_tri.any(dim=-1)  # (Tq,)

        masked_scores = tri_scores.float().masked_fill(~valid_tri.view(1, 1, Tq, num_tri), float("-inf"))
        # Safe softmax: rows with no valid triangles (e.g. t_i == 0) get zeros
        safe_scores = torch.where(
            has_any_tri.view(1, 1, Tq, 1),
            masked_scores,
            torch.zeros_like(masked_scores),
        )
        alpha2 = torch.softmax(safe_scores, dim=-1)
        alpha2 = alpha2 * valid_tri.view(1, 1, Tq, num_tri).to(alpha2.dtype)

        if self.simplicial_random_triangles and num_tri > 1:
            # Placebo control (checklist item 2): cyclic derangement of valid triangles
            # preserves simplex mass and FLOPs while breaking edge-triangle incidence.
            perm = torch.roll(torch.arange(num_tri, device=q.device), shifts=1)
            alpha2_perm = alpha2[:, :, :, perm]
            alpha2_perm = alpha2_perm * valid_tri.view(1, 1, Tq, num_tri).to(alpha2.dtype)
            denom = alpha2_perm.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            alpha2 = torch.where(
                has_any_tri.view(1, 1, Tq, 1) & (denom > 1e-11),
                alpha2_perm / denom,
                alpha2,
            )

        # Assemble the w x w upper-triangular 2-simplex weight tensor A2[a, b]
        a2_mat = torch.zeros(B, H, Tq, w, w, device=q.device, dtype=v.dtype)
        a2_mat[:, :, :, tri_a, tri_b] = alpha2.to(v.dtype)

        # Unsigned midpoint vertex marginals gamma^+_m = 0.5 * (sum_{b>m} A2[m,b] + sum_{a<m} A2[a,m])
        # and oriented boundary weights gamma^-_m = (partial_2 A2)_m = sum_{a<m} A2[a,m] - sum_{b>m} A2[m,b]
        row_sum = a2_mat.sum(dim=-1)  # sum_b A2[m, b]
        col_sum = a2_mat.sum(dim=-2)  # sum_a A2[a, m]
        gamma_plus = 0.5 * (row_sum + col_sum)  # (B, H, Tq, w); sums to 1 when has_any_tri is True
        gamma_minus = col_sum - row_sum  # (B, H, Tq, w); sums to 0 identically!

        # For t_i == 0 (where has_any_tri is False), put unit mass on the query's own vertex (m = w - 1)
        # so constant value vectors pass through unchanged at every position.
        no_tri_mask = (~has_any_tri).view(1, 1, Tq, 1)
        fallback_plus = torch.zeros_like(gamma_plus)
        fallback_plus[:, :, :, -1] = 1.0
        gamma_plus = torch.where(no_tri_mask, fallback_plus, gamma_plus)

        y2 = (gamma_plus.unsqueeze(-1) * v_win).sum(dim=-2) + (gamma_minus.unsqueeze(-1) * bv_win).sum(dim=-2)

        with torch.no_grad():
            if bool(has_any_tri.any()) and num_tri > 1:
                p_valid = alpha2[:, :, has_any_tri, :].clamp_min(1e-12)
                ent = -(p_valid * torch.log(p_valid)).sum(dim=-1).mean() / float(torch.log(torch.tensor(float(num_tri))))
                tri_ent = float(ent.item())
            else:
                tri_ent = 0.0
            self.last_simplicial_stats = {
                "triangle_entropy": tri_ent,
                "boundary_zero_sum_err": float(gamma_minus.float().sum(dim=-1).abs().max().item()),
            }

        return y2

    def score(self, q, k):
        # Standard 1-Simplex Attention Weights
        return (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))

    def aggregate(self, weights, v, *, q, k, kv_cache, pos0):
        # 1-hop Aggregation (Edges / 1-Simplices)
        y1 = weights @ v

        if self.simplicial_triplet:
            y2 = self._aggregate_triplet(weights, v, q=q, k=k, pos0=pos0)
            out = self.mix_1 * y1 + self.mix_2 * y2
            with torch.no_grad():
                n1 = float((self.mix_1 * y1).float().norm().item())
                n2 = float((self.mix_2 * y2).float().norm().item())
                if self.last_simplicial_stats is not None:
                    self.last_simplicial_stats["hodge_2simplex_share"] = n2 / max(n1 + n2, 1e-12)
            return out

        # Proxy 2-hop Aggregation (Simplicial/Paths on 1-skeleton)
        #
        # Cache-backed 2-hop: y2 = A @ y1_all, where y1_all stores the 1-hop outputs
        # for every past token (and is updated for the current chunk starting at the
        # pre-insert cache position pos0).
        if kv_cache is None:
            y2 = weights @ y1  # A @ (A @ v)
        else:
            if pos0 is None:
                raise RuntimeError("Expected pos0 to be set when kv_cache is provided")
            kv_cache.ensure_simplicial_y1_cache(
                num_heads=self.n_head,
                head_dim=self.head_dim,
                dtype=y1.dtype,
                device=y1.device,
            )
            y1_all = kv_cache.insert_simplicial_y1(self.layer_idx, pos0, y1)  # (B, H, Tk, D)
            y2 = weights @ y1_all

        return self.mix_1 * y1 + self.mix_2 * y2

