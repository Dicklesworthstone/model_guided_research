"""
Braid Attention Module (PyTorch)
Implements "Braid Attention": permutation + limited crossings with invariant-aware aggregation.

Program model (from JAX reference):
- Score each token (and an 'aggregator' token).
- Output a permutation pi.
- Execute 'braid words' w = sigma_1^k (aggregator swaps with neighbors k times).
- Crossing algebra: (x_a, y_a), (x_b, y_b) -> (x_a + y_b, y_a), (x_b + y_a, y_b).
- This restricted crossing accumulates payloads {y} onto the aggregator {x}.

PyTorch Implementation:
- "Prioritized Accumulation": We simulate the permutation and accumulation process
  using a soft scoring mechanism that respects the additive invariant.
- Instead of hard sorting and scanning (hard to autograd/GPU-optimize in PyTorch without custom kernels),
  we use a "Soft Braid" approximation:
  - Learn priority scores s_i.
  - Probability of crossing (swap) P(i crosses j) ~ Sigmoid(s_i - s_j).
  - Accumulate: x_i += sum_j P(j crosses i) * y_j.

Discrete mode (opt-in):
- Replace sigmoid weights with a hard threshold on the braid score matrix.
- Optionally record a discrete "braid schedule" (a per-head permutation + prefix length)
  for KV-cache decode (Tq==1) and verify local invariants.
- Optional crossing law: 'restricted' (fast, non-YBE) vs 'ybe' (swap-output, YBE-valid).

R-matrix mode (braid_crossing_law='rmatrix', bead u55.3):
- Crossings between positions i and j use the trigonometric six-vertex R-matrix of
  U_q(sl_2) in its one-particle braid form, with spectral parameter w = u_i - u_j:
      N(w) = [[c(w), b(w)], [b(w), c(w)]],
      b(w) = sinh(w)/sinh(w+eta),  c(w) = sinh(eta)/sinh(w+eta),
  per-head learned deformation q = e^eta (eta > 0) and per-position learned
  rapidities u_i, constrained monotone increasing so every causal argument
  w = u_i - u_j >= 0 stays in the stable region (pole only at w = -eta).
- N satisfies the spectral-parameter braid relation exactly (rapidities ride with
  strands), the inversion relation N(w) N(-w) = I, and N(0) = I (regularity).
- Attention for query i is the monodromy sweep of a fresh auxiliary strand of
  rapidity u_i through its prefix: A_ij = bg(w_ij) * prod_{j<j'<=i} cg(w_ij'),
  where (bg, cg) = (b, c)/(b + c) is the stochastic gauge of the same data
  (bg + cg = 1), giving the exact mass-partition charge
      Q1: sum_j A_ij + prod_{j<=i} cg(w_ij) = 1  (conserved, drift 0).
  The gauge is a per-crossing scalar; all YBE / transfer-matrix certificates use
  the a-normalized law (per-crossing stochastic normalization does NOT commute
  with the lifted braid relation - verified in tests).
- Conserved-charge tower: the inhomogeneous transfer matrices T(theta) built from
  the same (eta, rapidities) commute for all theta; their one-particle sector
  t(theta) (closed form in `one_particle_transfer`) supplies the charge matrices
  Q_k used by telemetry, certificates, and the charge-decoding probe.
"""

import math

import torch
import torch.nn as nn

from nanochat.model_utils import AttentionCore, causal_attn_mask


def _softplus_inv(y: float) -> float:
    # softplus(x) = y  =>  x = log(expm1(y))
    return math.log(math.expm1(y))


# Floors keeping the trigonometric weights inside the stable region:
# eta >= _RMATRIX_ETA_MIN > 0 (pole of N(w) sits at w = -eta) and strictly
# monotone rapidities (every causal argument w = u_i - u_j >= _RMATRIX_GAP_MIN > 0).
_RMATRIX_ETA_MIN = 1e-2
_RMATRIX_GAP_MIN = 1e-3


_LOG2 = math.log(2.0)


def _log_sinh_pos(x: torch.Tensor) -> torch.Tensor:
    """log sinh(x) for x > 0 without overflow: x + log(1 - e^{-2x}) - log 2."""
    return x + torch.log1p(-torch.exp(-2.0 * x)) - _LOG2


def rmatrix_bc(w: torch.Tensor, eta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A-normalized trigonometric six-vertex weights b(w), c(w) (broadcasting)."""
    den = torch.sinh(w + eta)
    return torch.sinh(w) / den, torch.sinh(eta) / den


def rmatrix_crossing(
    a: torch.Tensor, b_state: torch.Tensor, w: torch.Tensor, eta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Position-basis R-matrix crossing on a strand pair (swap encoded by the law).

    (s_i, s_j) <- (c*s_i + b*s_j, b*s_i + c*s_j) with w = r_i - r_j; the strands
    exchange positions, so the caller must also swap the rapidities riding on them.
    """
    bw, cw = rmatrix_bc(w, eta)
    return cw * a + bw * b_state, bw * a + cw * b_state


def one_particle_transfer(theta: float, u: torch.Tensor, eta: float) -> torch.Tensor:
    """One-particle sector of the inhomogeneous six-vertex transfer matrix t(theta).

    Closed form (fp64). With b_i = b(theta - u_i), c_i = c(theta - u_i):
      A-part (aux in |0>):  A_jj = b_j;            A_kj = c_j c_k prod_{j<l<k} b_l (k > j)
      D-part (aux in |1>):  D_jj = prod_{l!=j} b_l; D_kj = c_k c_j prod_{l<k} b_l prod_{l>j} b_l (k < j)
    t(theta) = A + D. The family {t(theta)} pairwise commutes for fixed (u, eta) -
    the conserved-charge tower of bead u55.3 - which tests verify both against this
    closed form and against the dense 2^T tensor construction on small chains.
    """
    u64 = u.detach().to(torch.float64)
    T = u64.numel()
    w = torch.as_tensor(theta, dtype=torch.float64) - u64
    if bool(torch.any(torch.abs(torch.sinh(w)) < 1e-12)) or bool(torch.any(torch.abs(torch.sinh(w + eta)) < 1e-12)):
        raise ValueError("one_particle_transfer: probe theta must avoid the rapidities and the pole at u_i - eta")
    bv, cv = rmatrix_bc(w, torch.as_tensor(eta, dtype=torch.float64))
    # prefix[k] = prod_{l<k} b_l, suffix[j] = prod_{l>j} b_l (exclusive products)
    prefix = torch.cat([torch.ones(1, dtype=torch.float64), torch.cumprod(bv, 0)[:-1]])
    rev = torch.flip(torch.cumprod(torch.flip(bv, [0]), 0), [0])
    suffix = torch.cat([rev[1:], torch.ones(1, dtype=torch.float64)])
    t = torch.diag(bv + prefix * suffix)  # A_jj = b_j ; D_jj = prod_{l!=j} b_l
    kk, jj = torch.meshgrid(torch.arange(T), torch.arange(T), indexing="ij")
    # A-part, k > j: c_j c_k prod_{j<l<k} b_l = c_j c_k prefix[k] / prefix[j+1]
    full_pref = torch.cat([prefix, torch.prod(bv).reshape(1)])  # prefix[T] = prod all
    upper = kk > jj
    ratio = prefix[kk.clamp(min=0)] / full_pref[(jj + 1).clamp(max=T)]
    t = t + torch.where(upper, cv[jj] * cv[kk] * ratio, torch.zeros((), dtype=torch.float64))
    # D-part, k < j: c_k c_j prod_{l<k} b_l prod_{l>j} b_l
    lower = kk < jj
    t = t + torch.where(lower, cv[kk] * cv[jj] * prefix[kk] * suffix[jj], torch.zeros((), dtype=torch.float64))
    return t


def _braid_triple_residual(states: torch.Tensor, raps: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Max |sigma1 sigma2 sigma1 - sigma2 sigma1 sigma2| on three strands.

    `states` is (..., 3, D), `raps` is (..., 3); rapidities ride with strands
    (they swap at every crossing, and the crossing argument is the rapidity
    difference of the two positions being crossed, read BEFORE the crossing).
    """

    def sigma(st: torch.Tensor, rp: torch.Tensor, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        w = (rp[..., i] - rp[..., i + 1]).unsqueeze(-1)
        a, b_state = st[..., i, :], st[..., i + 1, :]
        na, nb = rmatrix_crossing(a, b_state, w, eta.unsqueeze(-1))
        st = st.clone()
        st[..., i, :] = na
        st[..., i + 1, :] = nb
        rp = rp.clone()
        rp[..., [i, i + 1]] = rp[..., [i + 1, i]]
        return st, rp

    st1, rp1 = states, raps
    for i in (0, 1, 0):
        st1, rp1 = sigma(st1, rp1, i)
    st2, rp2 = states, raps
    for i in (1, 0, 1):
        st2, rp2 = sigma(st2, rp2, i)
    return torch.max(torch.abs(st1 - st2))


def rmatrix_braid_relation_residual(*, trials: int = 256, seed: int = 0) -> float:
    """Spectral-parameter braid relation residual for the trigonometric law (fp64)."""
    gen = torch.Generator().manual_seed(seed)
    eta = torch.rand((trials,), generator=gen, dtype=torch.float64) * 2.45 + 0.05
    # pairwise rapidity differences stay inside (-eta, eta): poles avoided
    raps = (torch.rand((trials, 3), generator=gen, dtype=torch.float64) - 0.5) * 0.8 * eta.unsqueeze(-1)
    states = torch.randn((trials, 3, 2), generator=gen, dtype=torch.float64)
    return float(_braid_triple_residual(states, raps, eta))


def rmatrix_inversion_residual(*, trials: int = 256, seed: int = 0) -> float:
    """Inversion relation N(w) N(-w) = I residual for the trigonometric law (fp64)."""
    gen = torch.Generator().manual_seed(seed)
    eta = torch.rand((trials, 1), generator=gen, dtype=torch.float64) * 2.45 + 0.05
    w = torch.rand((trials, 1), generator=gen, dtype=torch.float64) * 3.0
    a = torch.randn((trials, 2), generator=gen, dtype=torch.float64)
    b_state = torch.randn((trials, 2), generator=gen, dtype=torch.float64)
    fa, fb = rmatrix_crossing(a, b_state, w, eta)
    ra, rb = rmatrix_crossing(fa, fb, -w, eta)
    return float(torch.max(torch.abs(torch.stack([ra - a, rb - b_state]))))


class BraidCausalSelfAttention(AttentionCore):
    _ybe_checked: bool = False
    _rmatrix_checked: bool = False

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)

        self.braid_mode = str(getattr(config, "braid_mode", "soft")).strip().lower()
        self.braid_tau = float(getattr(config, "braid_tau", 0.0))
        self.braid_crossing_law = str(getattr(config, "braid_crossing_law", "restricted")).strip().lower()
        self.braid_record_schedule = bool(getattr(config, "braid_record_schedule", False))
        self.braid_verify = bool(getattr(config, "braid_verify", False))
        if self.braid_mode not in {"soft", "discrete"}:
            raise ValueError(f"braid_mode must be 'soft' or 'discrete', got {self.braid_mode!r}")
        if self.braid_crossing_law not in {"restricted", "ybe", "rmatrix"}:
            raise ValueError(
                f"braid_crossing_law must be 'restricted', 'ybe', or 'rmatrix', got {self.braid_crossing_law!r}"
            )
        self.last_braid_debug: dict[str, object] | None = None
        self.last_braid_charges: dict[str, object] | None = None

        if self.braid_crossing_law == "rmatrix":
            # Purely positional integrable mixing: no score network. Per-head
            # deformation eta = softplus(raw) + floor and per-position rapidity
            # increments (cumsum of softplus + floor => strictly monotone u_i).
            # Deterministic init (no RNG consumed; goldens of the default braid
            # path are unaffected because that path creates the same modules in
            # the same order as before).
            seq_len = int(getattr(config, "sequence_len", 0))
            if seq_len <= 0:
                raise ValueError("rmatrix crossing law requires config.sequence_len > 0 for the rapidity table")
            self.rmatrix_eta_raw = nn.Parameter(torch.full((self.n_head,), _softplus_inv(1.0)))
            # Rapidity table follows the rotary cache's 10x over-allocation so
            # the mechanism honors the same length-extrapolation contract
            # (held-out lengths 2x-8x train are the word-problem protocol).
            # Positions beyond the trained range keep the default increment:
            # uniform rapidity continuation, the integrable analogue of RoPE's
            # by-formula extrapolation.
            self.rmatrix_rho = nn.Parameter(torch.full((self.n_head, seq_len * 10), _softplus_inv(0.10)))
            # Spectral multi-view readout (bead 0i1v, the causal realization of
            # the TL/Markov-trace design in the theory note section 5): K extra
            # monodromy sweeps per query at probe rapidities ABOVE the query's
            # own (theta_k = u_i + delta_k, delta_k > 0, so every causal
            # argument stays in the stable region), gated into the output with
            # ZERO-INIT per-(head, probe) gates - at init the mechanism is
            # bitwise the base sweep; training opens the spectral views it
            # finds useful. 0 probes (the default) creates no parameters.
            self.rmatrix_n_probes = int(getattr(config, "braid_rmatrix_probes", 0))
            if self.rmatrix_n_probes < 0:
                raise ValueError(f"braid_rmatrix_probes must be >= 0, got {self.rmatrix_n_probes}")
            if self.rmatrix_n_probes > 0:
                self.rmatrix_probe_delta_raw = nn.Parameter(
                    torch.full((self.n_head, self.rmatrix_n_probes), _softplus_inv(0.5))
                )
                self.rmatrix_probe_gate = nn.Parameter(torch.zeros(self.n_head, self.rmatrix_n_probes))
            # The integrable law is purely positional: the scaffold-mandated
            # q/k projections are dead weights in this mode. Freeze them so
            # optimizers that demand a gradient for every parameter (Muon)
            # see only live parameters.
            self.c_q.weight.requires_grad_(False)
            self.c_k.weight.requires_grad_(False)
        else:
            # Braid Scoring Network
            # JAX: "sigmoid(w*[tag, value] + b)"
            # Here we learn a scalar score from the head dimension.
            self.braid_score = nn.Linear(self.head_dim, 1, bias=False)

        if self.braid_verify and self.braid_crossing_law == "ybe":
            self._verify_ybe_crossing_law_once()
        if self.braid_verify and self.braid_crossing_law == "rmatrix":
            self._verify_rmatrix_law_once()

    @staticmethod
    def _crossing_update_restricted(
        ax: torch.Tensor, ay: torch.Tensor, bx: torch.Tensor, by: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return ax + by, ay, bx + ay, by

    @staticmethod
    def _crossing_update_ybe(
        ax: torch.Tensor, ay: torch.Tensor, bx: torch.Tensor, by: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        # Swap-output version of the restricted crossing law.
        return bx + ay, by, ax + by, ay

    @classmethod
    def _verify_ybe_crossing_law_once(cls) -> None:
        if cls._ybe_checked:
            return
        cls._ybe_checked = True

        # Quick set-theoretic YBE (R3) sanity check on random tensors. A local
        # generator: reseeding the GLOBAL RNG here made every later parameter
        # draw depend on whether --braid-verify was set.
        gen = torch.Generator().manual_seed(0)
        n = 64
        d = 8
        ax, ay, bx, by, cx, cy = (torch.randn(n, d, generator=gen) for _ in range(6))

        def apply12(ax, ay, bx, by, cx, cy):
            nax, nay, nbx, nby = cls._crossing_update_ybe(ax, ay, bx, by)
            return nax, nay, nbx, nby, cx, cy

        def apply23(ax, ay, bx, by, cx, cy):
            nbx, nby, ncx, ncy = cls._crossing_update_ybe(bx, by, cx, cy)
            return ax, ay, nbx, nby, ncx, ncy

        lhs = apply12(*apply23(*apply12(ax, ay, bx, by, cx, cy)))
        rhs = apply23(*apply12(*apply23(ax, ay, bx, by, cx, cy)))
        err = torch.max(torch.abs(torch.stack(lhs, dim=-1) - torch.stack(rhs, dim=-1))).item()
        if err > 1e-5:
            raise RuntimeError(f"YBE crossing law check failed: max |lhs-rhs| = {err:.3e}")

    @classmethod
    def _verify_rmatrix_law_once(cls) -> None:
        if cls._rmatrix_checked:
            return
        cls._rmatrix_checked = True
        err = rmatrix_braid_relation_residual(trials=64, seed=0)
        if err > 1e-10:
            raise RuntimeError(f"R-matrix braid relation check failed: max residual = {err:.3e}")
        inv = rmatrix_inversion_residual(trials=64, seed=0)
        if inv > 1e-10:
            raise RuntimeError(f"R-matrix inversion relation check failed: max residual = {inv:.3e}")

    def score(self, q, k):
        # Braid Attention Logic
        # 1. Compute priority scores for Q (Aggregator) and K (Tokens).
        # In Braid model, aggregator is just a designated strand.
        # Here, every query i acts as a potential aggregator for its past.

        # Score(q): (B, H, Tq, 1)
        s_q = self.braid_score(q)
        # Score(k): (B, H, Tk, 1)
        s_k = self.braid_score(k)

        # Crossing Condition: Aggregator i interacts with Token j if i "sorts" past j?
        # Or if they are "compatible".
        # JAX: "Allowed set A = { j : p_j > tau }".
        # This implies interaction depends purely on the token j's score, relative to a threshold.
        # But in Attention, we need Q-dependence.
        # "Score = s_q + s_k" (additive interaction) gives the pairwise matrix.
        return s_q + s_k.transpose(-2, -1)  # (B, H, Tq, Tk)

    def rmatrix_eta(self) -> torch.Tensor:
        """Per-head deformation eta > 0 (fp32)."""
        return torch.nn.functional.softplus(self.rmatrix_eta_raw.float()) + _RMATRIX_ETA_MIN

    def rmatrix_rapidities(self) -> torch.Tensor:
        """Per-head strictly monotone rapidity profile u (n_head, sequence_len), fp32."""
        inc = torch.nn.functional.softplus(self.rmatrix_rho.float()) + _RMATRIX_GAP_MIN
        return torch.cumsum(inc, dim=-1)

    def _attend_rmatrix(self, q, k, v, *, kv_cache, pos0):
        # Monodromy sweep: a fresh auxiliary strand of rapidity u_i scatters
        # through the query's prefix; the stochastic gauge (bg + cg = 1) of the
        # trigonometric weights makes the kernel an exact mass partition:
        #   A_ij = bg(w_ij) * prod_{j<j'<=i} cg(w_ij'),
        #   sum_j A_ij + prod_{j<=i} cg(w_ij) = 1   (the Q1 charge).
        # The kernel is purely positional (B-independent): (H, Tq, Tk) only.
        Tq = q.size(2)
        Tk = k.size(2)
        u_all = self.rmatrix_rapidities()  # (H, S)
        if Tk > u_all.size(-1):
            raise ValueError(
                f"rmatrix rapidity table covers {u_all.size(-1)} positions but Tk={Tk}; increase config.sequence_len"
            )
        # The kernel is positional and tiny ((H, Tq, Tk), B-independent), so it
        # is evaluated in float64: the log-space gauge below accumulates a
        # Tk-term cumulative sum, and in fp32 that put the Q1 mass-partition
        # defect at ~1e-5 (the e2e charge gate); in fp64 it is ~1e-13. The
        # weights are cast to the value dtype only where they meet v.
        eta = self.rmatrix_eta().view(-1, 1, 1).double()  # (H, 1, 1)
        qpos = torch.arange(Tk - Tq, Tk, device=q.device)
        uq = u_all[:, qpos]  # (H, Tq)
        uk = u_all[:, :Tk]  # (H, Tk)
        w = (uq.unsqueeze(-1) - uk.unsqueeze(1)).double()  # (H, Tq, Tk); >= 0 iff j <= qpos_i
        valid = w >= 0

        log_sh_e = _log_sinh_pos(eta)  # eta >= _RMATRIX_ETA_MIN > 0

        def kernel_rows(w_arg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # (weights, leftover) of one monodromy sweep family; exact mass
            # partition per row: weights.sum(-1) + leftover = 1.
            #   bg = sinh(w) / (sinh(w) + sinh(eta)),  cg = sinh(eta) / (sinh(w) + sinh(eta))
            # sinh(w) overflows fp32 past w ~ 89 - a ~900-token causal span at
            # the init increment 0.1 - which turned every held-out-length
            # forward (the 2x-8x word-problem protocol) into NaN. The gauge is
            # therefore evaluated in log space. A query against its own
            # position has w == 0 exactly, where bg = 0 and cg = 1; those
            # entries get a finite placeholder argument so neither branch of
            # the where() can produce a non-finite gradient (0 * inf).
            strict = valid & (w_arg > 0)
            zero = torch.zeros((), device=w_arg.device, dtype=w_arg.dtype)
            w_safe = torch.where(strict, w_arg, torch.ones_like(w_arg))
            log_sh_w = _log_sinh_pos(w_safe)
            log_den = torch.logaddexp(log_sh_w, log_sh_e)
            bg = torch.where(strict, torch.exp(log_sh_w - log_den), zero)
            log_cg = torch.where(strict, log_sh_e - log_den, zero)
            cum = torch.cumsum(log_cg, dim=-1)  # (H, Tq, Tk)
            total = cum[..., -1:]
            return bg * torch.exp(total - cum), torch.exp(total).squeeze(-1)

        weights, leftover = kernel_rows(w)
        out = weights.unsqueeze(0).to(v.dtype) @ v  # (B, H, Tq, D)

        # Spectral multi-view readout (0i1v): extra monodromy sweeps at probe
        # rapidities theta_k = u_i + delta_k (delta_k > 0 keeps every causal
        # argument in the stable region), gated in with zero-init gates. Each
        # view satisfies its own mass partition; Q1 telemetry tracks the base
        # view (gated views are explicit, learned departures from it).
        if getattr(self, "rmatrix_n_probes", 0) > 0:
            delta = torch.nn.functional.softplus(self.rmatrix_probe_delta_raw.double()) + _RMATRIX_GAP_MIN
            for kp in range(self.rmatrix_n_probes):
                w_k = w + delta[:, kp].view(-1, 1, 1)
                weights_k, _ = kernel_rows(w_k)
                gate = self.rmatrix_probe_gate[:, kp].view(1, -1, 1, 1).to(v.dtype)
                out = out + gate * (weights_k.unsqueeze(0).to(v.dtype) @ v)

        with torch.no_grad():
            defect = torch.max(torch.abs(weights.sum(dim=-1) + leftover - 1.0))
            eta_flat = eta.detach().view(-1)
            self.last_braid_charges = {
                "crossing_law": "rmatrix",
                "q1_mass_defect": float(defect),
                "q2_braid_residual": self._live_braid_residual(),
                "eta": [float(x) for x in eta_flat],
                "rapidity_span": [float(x) for x in (uk[:, -1] - uk[:, 0]).detach()],
            }

        return out

    def _live_braid_residual(self) -> float:
        # Q2 telemetry: path-independence of transport, measured on the LIVE
        # learned parameters - compose the layer's actual crossing law on a
        # probe strand triple along the two braid-equivalent schedules.
        if self.braid_crossing_law == "rmatrix":
            # Telemetry math runs on CPU in fp64: the probe states come from a
            # CPU generator, so device-resident params must come down to meet
            # them (and the fp64 composition is host-side scalar work anyway).
            u_all = self.rmatrix_rapidities().detach().to("cpu", torch.float64)
            eta = self.rmatrix_eta().detach().to("cpu", torch.float64)
            H, S = u_all.shape
            idx = torch.tensor([0, S // 2, S - 1] if S >= 3 else [0] * 3)
            raps = u_all[:, idx]  # (H, 3)
            gen = torch.Generator().manual_seed(0)
            states = torch.randn((H, 3, 2), generator=gen, dtype=torch.float64)
            return float(_braid_triple_residual(states, raps, eta))
        # Heuristic laws: compose the constant 4-dim pair law on probe triples.
        gen = torch.Generator().manual_seed(0)
        parts = [torch.randn((8, 4), generator=gen) for _ in range(3)]
        law = self._crossing_update_ybe if self.braid_crossing_law == "ybe" else self._crossing_update_restricted

        def apply12(ax, ay, bx, by, cx, cy):
            nax, nay, nbx, nby = law(ax, ay, bx, by)
            return nax, nay, nbx, nby, cx, cy

        def apply23(ax, ay, bx, by, cx, cy):
            nbx, nby, ncx, ncy = law(bx, by, cx, cy)
            return ax, ay, nbx, nby, ncx, ncy

        six = (parts[0][:, 0], parts[0][:, 1], parts[1][:, 0], parts[1][:, 1], parts[2][:, 0], parts[2][:, 1])
        lhs = apply12(*apply23(*apply12(*six)))
        rhs = apply23(*apply12(*apply23(*six)))
        return float(torch.max(torch.abs(torch.stack(lhs, dim=-1) - torch.stack(rhs, dim=-1))))

    def attend(self, q, k, v, *, kv_cache, pos0):
        # Overrides the default pipeline: braid weights are sigmoid crossing
        # probabilities (soft) or a hard threshold (discrete) - additive
        # accumulation, NOT a softmax convex combination - and the discrete
        # decode path reads the masked score matrix to record its schedule.
        # The rmatrix law replaces score-gated accumulation entirely with the
        # integrable monodromy sweep (purely positional, exactly conserved Q1).
        if self.braid_crossing_law == "rmatrix":
            return self._attend_rmatrix(q, k, v, kv_cache=kv_cache, pos0=pos0)
        scores = self.score(q, k)

        # Masking (Causal)
        Tq = q.size(2)
        Tk = k.size(2)
        if kv_cache is None or Tq > 1:
            mask = causal_attn_mask(Tq, Tk, device=q.device)
            scores.masked_fill_(~mask, float("-inf"))

        # Interaction Strength
        # Soft mode: sigmoid(scores) in [0,1].
        # Discrete mode: (scores > tau) in {0,1}.
        # Since braid crossing is x += y, we don't normalize to sum=1.
        # We sum raw values weighted by crossing probability (or hard selection).

        # Accumulation: x += sum(p * y)
        # Note: Standard attention is convex combination (sum p = 1).
        # Braid is additive accumulation (sum p can be anything).
        # This can lead to explosion.
        # We add a scaling factor 1/sqrt(T) or similar, or rely on LayerNorm.
        # JAX code uses "MSE(pred_soft - gt)" where GT is sum.
        # So it expects to learn the scale.
        # We'll divide by sqrt(Tk) to keep variance stable at init.

        if self.braid_mode == "soft":
            attn_weights = torch.sigmoid(scores)
        else:
            # Hard gating by threshold: produces a discrete braid schedule (prefix after sort).
            attn_weights = (scores > self.braid_tau).to(dtype=v.dtype)

            if self.braid_record_schedule and Tq == 1:
                score_vec = scores.squeeze(2)  # (B, H, Tk)
                order = torch.argsort(score_vec, dim=-1, descending=True)  # (B, H, Tk)
                selected = score_vec > self.braid_tau  # (B, H, Tk)
                k = selected.sum(dim=-1)  # (B, H)
                self.last_braid_debug = {
                    "tau": float(self.braid_tau),
                    "crossing_law": str(self.braid_crossing_law),
                    "scores": score_vec.detach(),
                    "order": order.detach(),
                    "selected": selected.detach(),
                    "k": k.detach(),
                }

                if self.braid_verify:
                    ordered_scores = score_vec.gather(-1, order)
                    pos = torch.arange(Tk, device=score_vec.device)
                    prefix = pos.view(1, 1, Tk) < k.unsqueeze(-1)
                    # Sorted scores imply a prefix property for threshold selection.
                    ordered_selected = ordered_scores > self.braid_tau
                    if torch.any(ordered_selected != prefix):
                        raise RuntimeError(
                            "Discrete braid decode failed prefix verification (threshold/sort mismatch)."
                        )

                    # Invariant check: prefix-sum over permuted values equals mask-sum.
                    v_perm = v.gather(2, order.unsqueeze(-1).expand(-1, -1, -1, v.size(-1)))
                    sum_perm = (prefix.to(v.dtype).unsqueeze(-1) * v_perm).sum(dim=2)
                    sum_mask = (selected.to(v.dtype).unsqueeze(-1) * v).sum(dim=2)
                    max_err = torch.max(torch.abs(sum_perm - sum_mask)).item()
                    if max_err > 1e-4:
                        raise RuntimeError(
                            f"Discrete braid invariant check failed: max |perm-sum - mask-sum| = {max_err:.3e}"
                        )

        # Scale each row by the number of keys that query can SEE, not by the
        # total key count of the tensor. With 1/sqrt(Tk) the token at position
        # t was scaled by 1/sqrt(T_train) during training but by 1/sqrt(t+1)
        # during single-token decode (Tk == t+1), so generation ran a different
        # function than the one that was trained at every non-final position
        # (the KV-cache parity tests only compare the LAST position, where the
        # two agree). visible[i] = Tk - Tq + i + 1 covers training (Tq == Tk),
        # single-token decode (Tq == 1) and chunked prefill identically.
        visible = torch.arange(Tk - Tq + 1, Tk + 1, device=v.device, dtype=torch.float32)
        inv_scale = visible.rsqrt().to(v.dtype).view(1, 1, Tq, 1)

        with torch.no_grad():
            # Charge fingerprint of the heuristic modes (u55.3): additive
            # accumulation has no mass partition - transported mass is whatever
            # the gates sum to - so the Q1 defect is measurably nonzero, and Q2
            # records the law's path-dependence (restricted fails the braid
            # relation; the constant ybe law passes it but still fails Q1).
            row_mass = attn_weights.sum(dim=-1) * inv_scale.view(1, 1, Tq)
            self.last_braid_charges = {
                "crossing_law": str(self.braid_crossing_law),
                "q1_mass_defect": float(torch.max(torch.abs(row_mass - 1.0))),
                "q2_braid_residual": self._live_braid_residual(),
                "eta": None,
                "rapidity_span": None,
            }

        y = attn_weights @ v  # [B, H, Tq, D]
        return y * inv_scale
