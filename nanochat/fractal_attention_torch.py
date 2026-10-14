"""
Fractal Memory / Attention Module (PyTorch)
Implements attention over a "Fractal" memory structure (Iterated Function System / Valuation Tree).

Mathematical core (bead 8gk.5):
  - Address space {0..m-1}^L with per-branch contraction ratios r_i in (0, r_max)
    carries the weighted ultrametric d(a, b) = prod_{j <= LCP(a,b)} r_{a_j}.
  - By the Hutchinson-Moran theorem (thm-moran-capacity), the unique similarity
    dimension s = D_Moran solves sum_{i=0}^{m-1} r_i^s = 1.
  - Optimal radius allocation under branch probability mass p = (p_0, ..., p_{m-1}):
    r_i^* = p_i^{1/s}, equating geometric cylinder volume r_i^s to probability mass p_i.
  - When ``fractal_ifs=False`` (default), executes the router-simplex proxy for
    backward golden/ledger parity.
  - When ``fractal_ifs=True``, executes the weighted-ultrametric cylinder kernel
    and contractive Cayley-Hutchinson value maps w_i(v) = r_i R_i v + (1 - r_i) tau_i,
    recording ``fractal_moran_dimension``, ``fractal_max_contraction``, and
    ``fractal_radii_profile`` in the metrics stream.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.model_utils import AttentionCore


def compute_moran_dimension(radii: torch.Tensor, *, steps: int = 32) -> torch.Tensor:
    """Solve the exact Moran equation ``sum_{i=0}^{m-1} r_i^s = 1`` for ``s > 0``
    via monotone bisection in fp32.

    Requires ``0 < r_i < 1`` and ``m >= 2``. Since ``g(s) = sum_i r_i^s`` is strictly
    decreasing with ``g(0) = m > 1`` and ``lim_{s->inf} g(s) = 0``, bisection converges
    geometrically at rate ``2^{-steps}``.
    """
    r = radii.detach().float().clamp(min=1e-6, max=1.0 - 1e-6)
    m = int(r.numel())
    if m < 2:
        return torch.zeros((), dtype=torch.float32, device=radii.device)
    r_max = float(r.max().item())
    lo = torch.zeros((), dtype=torch.float32, device=r.device)
    hi_val = max(2.0, float(torch.log(torch.tensor(float(m))) / (-torch.log(torch.tensor(r_max))) + 1.0))
    hi = torch.tensor(hi_val, dtype=torch.float32, device=r.device)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        val = torch.pow(r, mid).sum()
        if float(val.item()) > 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def optimal_ifs_radii(
    branch_masses: torch.Tensor,
    target_moran_dim: float = 1.0,
    *,
    r_min: float = 1e-4,
    r_max: float = 0.999,
) -> torch.Tensor:
    """Closed-form optimal IFS contraction radii ``r_i^* = p_i^{1/s}`` from
    branch probability masses ``p_i`` and target Moran similarity dimension ``s > 0``
    (Proposition 2 of bead 8gk.5; consumed by 77l.4 geometry-informed init).

    When ``p_i > 0`` and ``sum_i p_i = 1``, ``r_i^* = p_i^{1/s}`` satisfies the
    Moran equation ``sum_i (r_i^*)^s = sum_i p_i = 1`` identically while allocating
    larger radii (finer resolution / less contraction) to higher-mass branches.
    """
    if target_moran_dim <= 0.0:
        raise ValueError(f"target_moran_dim must be > 0, got {target_moran_dim}")
    p = branch_masses.float().clamp_min(1e-12)
    p = p / p.sum()
    radii = torch.pow(p, 1.0 / float(target_moran_dim))
    return radii.clamp(min=r_min, max=r_max)


class FractalCausalSelfAttention(AttentionCore):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)

        # IFS Router: depth independent m-way classifiers mapping q, k -> branch distributions
        self.m = 4  # Branching factor
        self.depth = 4  # Depth of IFS valuation tree
        self.router = nn.Linear(self.head_dim, self.depth * self.m, bias=False)

        self.fractal_ifs = bool(getattr(config, "fractal_ifs", False))
        self.r_max = 0.85
        self.last_fractal_stats: dict[str, Any] | None = None

        if self.fractal_ifs:
            self.ifs_raw_radii = nn.Parameter(torch.zeros(self.m))
            self.ifs_skew_params = nn.Parameter(torch.zeros(self.m, self.head_dim, self.head_dim))
            self.ifs_translations = nn.Parameter(torch.zeros(self.m, self.head_dim))
            self.ifs_prefix_weight = nn.Parameter(torch.tensor(1.0))
            self.ifs_gate = nn.Parameter(torch.tensor(1.0))
        else:
            self.ifs_raw_radii = None
            self.ifs_skew_params = None
            self.ifs_translations = None
            self.ifs_prefix_weight = None
            self.ifs_gate = None

    def _ensure_ifs_parameters(self, device: torch.device, dtype: torch.dtype) -> None:
        """Lazily allocate deterministic IFS parameters if ``fractal_ifs`` was
        toggled on after ``__init__``."""
        if self.ifs_raw_radii is None:
            self.ifs_raw_radii = nn.Parameter(torch.zeros(self.m, device=device, dtype=torch.float32))
            self.ifs_skew_params = nn.Parameter(
                torch.zeros(self.m, self.head_dim, self.head_dim, device=device, dtype=torch.float32)
            )
            self.ifs_translations = nn.Parameter(
                torch.zeros(self.m, self.head_dim, device=device, dtype=dtype)
            )
            self.ifs_prefix_weight = nn.Parameter(torch.tensor(1.0, device=device, dtype=torch.float32))
            self.ifs_gate = nn.Parameter(torch.tensor(1.0, device=device, dtype=torch.float32))

    def branch_radii(self) -> torch.Tensor:
        """Return the per-branch contraction ratios ``r_i in (0, r_max)``."""
        if self.ifs_raw_radii is None:
            return torch.full((self.m,), 0.5 * self.r_max)
        return float(self.r_max) * torch.sigmoid(self.ifs_raw_radii.float())

    def branch_rotors(self) -> torch.Tensor:
        """Return the ``m`` exact Cayley orthogonal matrices ``R_i = (I - A_i)(I + A_i)^{-1}``
        in ``SO(head_dim)`` computed in fp32."""
        if self.ifs_skew_params is None:
            return torch.eye(self.head_dim).unsqueeze(0).expand(self.m, -1, -1)
        raw = self.ifs_skew_params.float()
        skew = 0.5 * (raw - raw.transpose(-2, -1))
        eye = torch.eye(self.head_dim, device=skew.device, dtype=torch.float32).unsqueeze(0)
        return torch.linalg.solve(eye + skew, eye - skew)

    def apply_branch_map(self, z: torch.Tensor, branch_idx: int) -> torch.Tensor:
        """Apply the single-branch Banach contraction ``w_i(z) = r_i R_i z + (1 - r_i) tau_i``."""
        self._ensure_ifs_parameters(z.device, z.dtype)
        radii = self.branch_radii().to(device=z.device, dtype=z.dtype)
        rotors = self.branch_rotors().to(device=z.device, dtype=z.dtype)
        assert self.ifs_translations is not None
        tau = self.ifs_translations[branch_idx].to(device=z.device, dtype=z.dtype)
        r_i = radii[branch_idx]
        r_mat = rotors[branch_idx]
        return r_i * torch.matmul(z, r_mat.transpose(-2, -1)) + (1.0 - r_i) * tau

    def score(self, q, k):
        # Fractal Addressing / Similarity
        B = q.size(0)
        q_route = self.router(q).view(B, self.n_head, -1, self.depth, self.m)
        k_route = self.router(k).view(B, self.n_head, -1, self.depth, self.m)

        # Softmax over m to get branch probabilities
        q_prob = F.softmax(q_route, dim=-1)
        k_prob = F.softmax(k_route, dim=-1)

        if self.fractal_ifs:
            self._ensure_ifs_parameters(q.device, q.dtype)
            assert self.ifs_prefix_weight is not None
            radii = self.branch_radii().to(device=q.device, dtype=q.dtype)
            mean_r = radii.mean()
            # Weighted ultrametric cylinder kernel: branches with smaller radius r_i
            # represent finer valuation cylinders and receive weight (mean_r / r_i)
            # modulated by ifs_prefix_weight (when ifs_prefix_weight == 0, weight == 1).
            pw = self.ifs_prefix_weight.to(device=q.device, dtype=q.dtype)
            branch_scale = (1.0 + pw * (mean_r / radii.clamp_min(1e-4) - 1.0)).clamp_min(0.1)
            sqrt_scale = torch.sqrt(branch_scale).view(1, 1, 1, 1, self.m)
            q_prob = q_prob * sqrt_scale
            k_prob = k_prob * sqrt_scale

        q_flat = q_prob.view(B, self.n_head, -1, self.depth * self.m)
        k_flat = k_prob.view(B, self.n_head, -1, self.depth * self.m)

        return (q_flat @ k_flat.transpose(-2, -1)) * (1.0 / (self.depth**0.5))

    def aggregate(self, weights, v, *, q, k, kv_cache, pos0):
        y0 = weights @ v
        if not self.fractal_ifs:
            return y0

        self._ensure_ifs_parameters(v.device, v.dtype)
        assert self.ifs_translations is not None and self.ifs_gate is not None
        B = q.size(0)
        q_route = self.router(q).view(B, self.n_head, -1, self.depth, self.m)
        q_prob = F.softmax(q_route, dim=-1)  # (B, H, Tq, depth, m)
        branch_weights = q_prob.mean(dim=-2)  # (B, H, Tq, m)

        radii = self.branch_radii().to(device=v.device, dtype=v.dtype)  # (m,)
        rotors = self.branch_rotors().to(device=v.device, dtype=v.dtype)  # (m, D, D)
        tau = self.ifs_translations.to(device=v.device, dtype=v.dtype)  # (m, D)

        # Evaluate all m contractive affine maps w_i(y0) = r_i (y0 R_i^T) + (1 - r_i) tau_i
        # y0: (B, H, Tq, D) -> rotated: (B, H, Tq, m, D)
        rotated = torch.einsum("bhtd,mfd->bhtmf", y0, rotors)
        r_view = radii.view(1, 1, 1, self.m, 1)
        tau_view = tau.view(1, 1, 1, self.m, self.head_dim)
        w_y0 = r_view * rotated + (1.0 - r_view) * tau_view  # (B, H, Tq, m, D)

        y_ifs = (branch_weights.unsqueeze(-1) * w_y0).sum(dim=-2)  # (B, H, Tq, D)
        gate = self.ifs_gate.to(device=v.device, dtype=v.dtype)
        out = (1.0 - gate) * y0 + gate * y_ifs

        with torch.no_grad():
            r_fp32 = self.branch_radii()
            moran_d = float(compute_moran_dimension(r_fp32).item())
            moran_res = float((torch.pow(r_fp32, moran_d).sum() - 1.0).abs().item())
            self.last_fractal_stats = {
                "moran_dimension": moran_d,
                "max_contraction": float(r_fp32.max().item()),
                "moran_residual": moran_res,
                "radii_profile": [float(x) for x in r_fp32.tolist()],
            }

        return out

