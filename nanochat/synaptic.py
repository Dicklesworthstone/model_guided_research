# nanochat/synaptic.py
# Comprehensive synaptic modules for nanochat:
# - Presynaptic biophysics → attention logit augmentation
# - Postsynaptic dual-timescale linear with low-rank eligibility
# - Synaptic Self-Attention (RoPE, MQA-compatible)
# - Synaptic MLP
# - Synaptic MoE with router embeddings, contrastive updates & structural hooks
# - Structural plasticity utilities
#
# Design highlights (mapped from the JAX reference you provided):
#   • Synaptotagmin-1/7 mixed Ca2+ sensor, complexin clamp
#   • Munc13/18 priming, clathrin/dynamin endocytosis (delay queue)
#   • V-ATPase/VDAC energy coupling and per-edge cost model
#   • EMA normalization of quantal gain; optional stochastic release
#   • PSD-like low-rank eligibility U/V with CaMKII/PP1 gating (fast/slow)
#   • Septin-like distance barrier in attention logits
#   • Router embeddings + contrastive update; MoE top-k dispatch with fatigue
#
# This file is intentionally verbose and highly instrumented for clarity.

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, cast

from nanochat.model_utils import causal_attn_mask, repeat_kv_heads
from nanochat.torch_imports import F, Tensor, nn, torch

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def _rms(x: Tensor, eps=1e-6) -> Tensor:
    return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * x


def _tri(T: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tril(torch.ones(T, T, device=device, dtype=dtype)).view(1, 1, T, T)


def _softplus(x: Tensor, beta=1.0) -> Tensor:
    return (1.0 / beta) * F.softplus(beta * x)


def _cosine(u: Tensor, v: Tensor, eps=1e-8) -> Tensor:
    """Cosine similarity with safe normalization."""
    u = u / (u.norm(dim=-1, keepdim=True) + eps)
    v = v / (v.norm(dim=-1, keepdim=True) + eps)
    return (u * v).sum(dim=-1)

def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class SynapticConfig:
    enabled: bool = True
    # General
    rank_eligibility: int = 8
    attn_topk: int = 32
    stochastic_train_frac: float = 0.12

    # Presynaptic Biophysics
    tau_c: float = 0.85
    alpha_c: float = 0.55
    syt1_slope: float = 8.0
    syt7_slope: float = 3.0
    cpx_thresh: float = 0.55
    complexin_bias: float = 0.0
    doc2_gain: float = 0.08
    prime_rate: float = 0.075
    unprime_per_release: float = 0.05
    nsf_recover: float = 0.08
    rec_rate: float = 0.06
    endo_delay: int = 3
    amp_load: float = 0.02
    amp_leak: float = 0.006

    # Initial States
    init_rrp: float = 6.0
    init_reserve: float = 18.0
    init_snare: float = 0.7
    init_clamp: float = 0.6
    init_amp: float = 1.0
    init_energy: float = 0.85

    # Energy Dynamics
    energy_fill: float = 0.02
    energy_use: float = 0.02
    energy_max: float = 1.0

    # Attention
    lambda_loge: float = 1.0
    barrier_strength: float = 0.1
    epsilon: float = 1e-6
    use_flex_attention: bool = False

    # Postsynaptic Plasticity
    post_fast_decay: float = 0.95
    post_fast_lr: float = 1.5e-3
    post_slow_lr: float = 5e-4
    post_trace_decay: float = 0.96
    camkii_up: float = 0.05
    camkii_down: float = 0.02
    pp1_tau: float = 0.985
    camkii_thr: float = 1.0
    pp1_thr: float = 0.7
    bdnf_tau: float = 0.985
    bdnf_scale: float = 1.0

    # Structural Plasticity (MoE)
    structural_interval: int = 50000
    structural_tau_util: float = 0.2
    structural_age_bias: float = 1.0
    router_embed_dim: int = 24
    router_contrastive_lr: float = 1e-4
    router_contrastive_push: float = 0.1
    router_sim_threshold: float = 0.6

    # Genetics
    xi_dim: int = 4  # [alpha_fatigue, alpha_energy, camkii_gain, pp1_gain]

    # Native (Rust) Kernel Toggles
    native_presyn: bool = _env_bool("BIO_FUSED_PRESYN", default=False)
    native_metrics: bool = _env_bool("BIO_FUSED_METRICS", default=False)
    native_genetics: bool = _env_bool("BIO_FUSED_GENETICS", default=False)
    native_plasticity: bool = _env_bool("BIO_FUSED_PLASTICITY", default=False)


# -----------------------------------------------------------------------------
# Presynaptic biophysics
# -----------------------------------------------------------------------------


class SynapticPresyn(nn.Module):
    cfg: SynapticConfig
    """
    Vectorized presynaptic module with explicit Syt1/7 mix, complexin clamp,
    Munc13/18 priming, clathrin/dynamin endocytosis (queue), V-ATPase/VDAC
    coupling, EMA normalization, optional stochastic release on a fraction
    of edges, and a septin-like distance barrier for attention logits.
    """

    def __init__(self, d_head: int, cfg: SynapticConfig):
        super().__init__()
        object.__setattr__(self, "cfg", cfg)
        self.register_buffer("ema_e", torch.ones(1))

    def _mix_prob(self, c: Tensor, clamp: Tensor, sn: Tensor) -> Tensor:
        p1 = torch.sigmoid(self.cfg.syt1_slope * (c - 0.55))
        p7 = torch.sigmoid(self.cfg.syt7_slope * (c - 0.25))
        p = p1 * 0.8 + p7 * 0.2 + self.cfg.doc2_gain * torch.sigmoid(4 * (c - 0.12))
        cpx_gate = torch.sigmoid(8.0 * (c - self.cfg.cpx_thresh) - 2.0 * (clamp + self.cfg.complexin_bias))
        p = p * cpx_gate * sn
        return torch.clamp(p, 0, 0.999)

    def release(
        self,
        state: dict[str, Any],
        drive: Tensor,
        idx: Tensor,
        train: bool,
    ) -> Tensor:
        """
        Compute release and update state.
        drive: (B, H, T, K) - attention logits for top-k
        idx: (B, H, T, K) - indices of top-k keys
        """
        B, H, T, K = drive.shape
        cfg = self.cfg

        # Keep all presynaptic computations in the state's dtype so scatter_add_ destinations
        # and sources match even under autocast (e.g. LayerNorm → fp32, matmul → fp16).
        state_dtype = state["c"].dtype
        if drive.dtype != state_dtype:
            drive = drive.to(dtype=state_dtype)

        # Gather state for the selected keys
        # state tensors are (B, H, T_keys)
        # We gather along dim 2 (T_keys) using idx
        drive_isfinite = torch.isfinite(drive)
        drive_safe = torch.where(drive_isfinite, drive, torch.zeros_like(drive))
        T_key = state["c"].size(2)
        expand_shape = (B, H, T, T_key)
        c = state["c"].unsqueeze(2).expand(expand_shape).gather(3, idx)
        c = cfg.tau_c * c + cfg.alpha_c * F.softplus(drive_safe)

        sn = state["sn"].unsqueeze(2).expand(expand_shape).gather(3, idx)
        clamp = state["cl"].unsqueeze(2).expand(expand_shape).gather(3, idx)

        p = self._mix_prob(c, clamp, sn)
        rrp = state["rrp"].unsqueeze(2).expand(expand_shape).gather(3, idx)

        if train and cfg.stochastic_train_frac > 0:
            # Stochastic release on a fraction of edges
            mask = (torch.rand_like(p[..., 0]) < cfg.stochastic_train_frac).float().unsqueeze(-1)
            # Binomial sampling
            k_rel = torch.distributions.Binomial(
                total_count=torch.clamp(rrp, 0, 8).round(), probs=p
            ).sample()
            rel = mask * k_rel + (1 - mask) * (p * rrp)
        else:
            rel = p * rrp
        rel = torch.where(drive_isfinite, rel, torch.zeros_like(rel))

        amp = state["amp"].unsqueeze(2).expand(expand_shape).gather(3, idx)
        e = rel * amp

        # Scatter updates back to state
        # We need to sum updates for the same key if it appears multiple times (unlikely in top-k but possible across heads/batch if flattened, here we are per-head)
        # Actually, idx is (B, H, T_query, K). Keys are T_key.
        # We scatter_add_ into (B, H, T_key).

        # We should scatter into (B, H, T_key).
        # idx is (B, H, T, K). We flatten T and K?
        # scatter_add_ expects index to have same number of dimensions.
        # We want to scatter into dim 2.

        # Let's flatten B, H to batch dims, and T, K to source dims.
        # Target is (B, H, T_key).
        # We can use scatter_add on the last dim if we view it right.

        # Actually, the PDF code did:
        # add = torch.zeros(B,H,T,state['c'].size(2)...) -> This implies T_query=T, T_key=size(2).
        # But creating a (B,H,T,T) tensor is expensive if T is large.
        # However, the PDF code does:
        # add.scatter_add_(3, idx, rel)
        # This implies 'add' has 4 dims.
        # If T=2048, (B,H,2048,2048) is too big.
        # But wait, 'state' tensors are (B,H,T).
        # The updates should be accumulated over the query dimension (dim 2).
        # So we want to sum over T_query for each T_key.

        # Efficient scatter:
        # We want to add 'rel' (B,H,T,K) to 'state' (B,H,T_key) at indices 'idx' (B,H,T,K).
        # We can flatten T,K.

        flat_idx = idx.view(B, H, -1) # (B, H, T*K)
        flat_rel = rel.view(B, H, -1)
        flat_drive = drive_safe.view(B, H, -1)
        flat_valid = drive_isfinite.view(B, H, -1).to(flat_rel.dtype)
        flat_amp = amp.view(B, H, -1)

        # Accumulators
        add_vals = torch.zeros_like(state["c"]) # (B, H, T_key)
        drv_vals = torch.zeros_like(state["c"])
        snu_vals = torch.zeros_like(state["c"])
        rru_vals = torch.zeros_like(state["c"])
        ampu_vals = torch.zeros_like(state["c"])

        add_vals.scatter_add_(2, flat_idx, flat_rel)
        drv_vals.scatter_add_(2, flat_idx, flat_drive)
        snu_vals.scatter_add_(2, flat_idx, flat_valid) # Count of accesses
        rru_vals.scatter_add_(2, flat_idx, flat_rel)
        ampu_vals.scatter_add_(2, flat_idx, flat_amp)

        # Update dynamics
        c_up = cfg.tau_c * state["c"] + cfg.alpha_c * F.softplus(drv_vals)
        rrp_up = torch.clamp(state["rrp"] - add_vals, 0)

        # Endocytosis delay queue
        res_up = state["res"] + state["delay"][0]
        new_delay = state["delay"][1:] + [rru_vals * cfg.rec_rate]

        # Priming
        take = torch.minimum(res_up, torch.ones_like(res_up)) # Max 1 unit per step? Or just soft clamp?
        # PDF: take=torch.minimum(res_up, torch.ones_like(res_up))
        res_up = torch.clamp(res_up - cfg.prime_rate * take, 0)
        rrp_up = torch.clamp(rrp_up + cfg.prime_rate * take, 0, 30.0) # Cap RRP

        # SNARE / Clamp / AMPA / Energy
        sn_up = torch.clamp(state["sn"] * (1.0 - cfg.unprime_per_release * add_vals) + cfg.nsf_recover * (1.0 - state["sn"]), 0, 1)
        cl_up = torch.clamp(state["cl"] * 0.995 + 0.005, 0, 1)
        amp_up = torch.clamp(state["amp"] + cfg.amp_load * (1.2 - state["amp"]) - cfg.amp_leak * state["amp"], 0, 2)
        en_up = torch.clamp(state["en"] + cfg.energy_fill * (cfg.energy_max - state["en"]) - cfg.energy_use * add_vals, 0, cfg.energy_max)

        state.update({
            "c": c_up,
            "rrp": rrp_up,
            "res": res_up,
            "delay": new_delay,
            "sn": sn_up,
            "cl": cl_up,
            "amp": amp_up,
            "en": en_up
        })

        # EMA normalization
        ema = self.ema_e.detach().clone()
        s = e.detach().abs().mean().clamp_min(1e-3)
        self.ema_e.mul_(0.99).add_(0.01 * s)

        # IMPORTANT: Use the EMA value *from before* this update so outputs are causal within a chunk.
        return e / (ema + self.cfg.epsilon)


# -----------------------------------------------------------------------------
# Postsynaptic eligibility and linear
# -----------------------------------------------------------------------------


class PostsynapticHebb(nn.Module):
    cfg: SynapticConfig
    """Low-rank eligibility + CaMKII/PP1/BDNF gate controlling consolidation."""

    def __init__(self, d_k: int, d_v: int, cfg: SynapticConfig):
        super().__init__()
        object.__setattr__(self, "cfg", cfg)
        R = cfg.rank_eligibility
        self.fast = nn.Parameter(torch.zeros(d_v))
        self.slow = nn.Parameter(torch.zeros(d_v))
        self.U = nn.Parameter(torch.zeros(d_v, R))
        self.V = nn.Parameter(torch.zeros(R, d_v))

        self.register_buffer("camkii", torch.zeros(d_v))
        self.register_buffer("pp1", torch.ones(d_v) * 0.5)
        self.register_buffer("bdnf", torch.zeros(d_v))

        nn.init.normal_(self.U, std=0.02)
        nn.init.normal_(self.V, std=0.02)

    @torch.no_grad()
    def _delta_out_from_traces(self, traceU: Tensor, traceV: Tensor) -> Tensor:
        """
        Compute the per-output (diagonal) eligibility delta from low-rank traces.

        We interpret (traceU, traceV) as a factorization of an update matrix:
            dW ≈ traceU @ traceV
        where traceU has shape (d_k, R) and traceV has shape (R, d_v).

        Since PostsynapticHebb only stores per-output gains (fast/slow are (d_v,)),
        we reduce dW over the pre-synaptic dimension:
            delta[v] = mean_k dW[k, v]  ==  (mean_k traceU[k, :]) @ traceV[:, v].

        This is well-defined even when d_k == d_v; we intentionally do NOT use
        diag(dW), which would be an arbitrary input↔output index pairing.
        """
        if traceU.ndim != 2 or traceV.ndim != 2:
            raise ValueError(
                "PostsynapticHebb expects 2D traces: traceU (d_k, R), traceV (R, d_v). "
                f"Got traceU shape={tuple(traceU.shape)} (ndim={traceU.ndim}), "
                f"traceV shape={tuple(traceV.shape)} (ndim={traceV.ndim})."
            )
        if traceU.shape[1] != traceV.shape[0]:
            raise ValueError(
                "PostsynapticHebb trace rank mismatch: expected traceU.shape[1] == traceV.shape[0]. "
                f"Got traceU shape={tuple(traceU.shape)}, traceV shape={tuple(traceV.shape)}."
            )

        delta_out = traceU.mean(dim=0) @ traceV
        if delta_out.shape != self.fast.shape:
            raise ValueError(
                "PostsynapticHebb trace output mismatch: expected delta_out to match (d_v,). "
                f"Got delta_out shape={tuple(delta_out.shape)}, expected={tuple(self.fast.shape)}. "
                f"traceU shape={tuple(traceU.shape)}, traceV shape={tuple(traceV.shape)}."
            )
        return delta_out

    def forward(self, v: Tensor) -> Tensor:
        diag = 1.0 + self.fast + self.slow
        return v * diag + v @ (self.U @ self.V)

    @torch.no_grad()
    def update(self, y: Tensor, ca_proxy: Tensor):
        up = (ca_proxy > self.cfg.camkii_thr).float()
        down = (ca_proxy < self.cfg.pp1_thr).float()

        self.camkii.add_(self.cfg.camkii_up * up * (1 - self.camkii))
        self.camkii.clamp_(0, 1)

        self.pp1.mul_(self.cfg.pp1_tau).add_((1 - self.cfg.pp1_tau) * down)
        self.bdnf.mul_(self.cfg.bdnf_tau).add_((1 - self.cfg.bdnf_tau) * F.relu(self.camkii - 0.5))

    @torch.no_grad()
    def consolidate(self, traceU: Tensor, traceV: Tensor):
        delta = self._delta_out_from_traces(traceU, traceV)
        g = torch.sigmoid(self.camkii - 0.5) - 0.3
        self.slow.add_(self.cfg.post_slow_lr * (1.0 + self.cfg.bdnf_scale * self.bdnf) * delta * g)

    @torch.no_grad()
    def hebb_fast(self, traceU: Tensor, traceV: Tensor):
        delta = self._delta_out_from_traces(traceU, traceV)
        self.fast.mul_(self.cfg.post_fast_decay).add_(self.cfg.post_fast_lr * delta)


class SynapticLinear(nn.Module):
    cfg: SynapticConfig
    use_input_ln: bool
    bias: nn.Parameter | None
    input_ln: nn.LayerNorm | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cfg: SynapticConfig,
        bias: bool = True,
        use_input_ln: bool = False,
    ):
        super().__init__()
        object.__setattr__(self, "cfg", cfg)
        object.__setattr__(self, "use_input_ln", use_input_ln)

        # Standard weights
        self.w_slow = nn.Parameter(torch.empty(in_features, out_features))
        self.w_fast = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.trunc_normal_(self.w_slow, std=0.02)
        nn.init.trunc_normal_(self.w_fast, std=0.02)

        # Postsynaptic module (operates on output)
        self.post = PostsynapticHebb(in_features, out_features, cfg)

        # Eligibility buffers
        self.register_buffer("u_buf", torch.zeros(in_features, cfg.rank_eligibility))
        self.register_buffer("v_buf", torch.zeros(cfg.rank_eligibility, out_features))

        if use_input_ln:
            self.input_ln = nn.LayerNorm(in_features, eps=1e-5)
        else:
            object.__setattr__(self, "input_ln", None)

    def forward(
        self, x: Tensor, calcium: Tensor, energy: Tensor, update_mem: bool = True, genes: Tensor | None = None
    ):
        if self.input_ln is not None:
            x = self.input_ln(x)

        # Linear pass
        # We combine w_slow and w_fast.
        # Note: In the old code, w_fast was separate. In PDF, SynapticLinear has w (slow) and post has fast/slow diagonals.
        # We will blend them: Base linear uses w_slow + w_fast (matrix).
        # PostsynapticHebb applies diagonal modulation.

        W = self.w_slow + self.w_fast
        y = x @ W
        if self.bias is not None:
            y = y + self.bias

        # Postsynaptic modulation (diagonal fast/slow + low-rank)
        y = self.post(y)

        if update_mem:
            with torch.no_grad():
                # Update eligibility traces
                # u_buf: (in, R) <- x (B, in)
                # v_buf: (R, out) <- y (B, out)
                # We need to project x and y to rank R?
                # Or we accumulate outer products?
                # PDF: self.u_buf...add_(... einsum...)
                # Let's implement a simple Hebbian accumulation

                # Random projection for eligibility? Or learned?
                # The PDF PostsynapticHebb has U and V parameters.
                # We can use those to project.

                # Actually, let's just use the mean activity for now to keep it simple and fast
                u_mean = x.mean(0) # (in,)
                v_mean = y.mean(0) # (out,)

                # We need (in, R) and (R, out).
                # We can just broadcast or rotate.
                # Let's just update the buffers with a decay

                # Update logic from old code was:
                # U.mul_(rho).add_(eta * u.unsqueeze(-1))
                # V.mul_(rho).add_(eta * v.unsqueeze(0))
                # This creates rank-1 updates.

                # We will do similar here but on u_buf/v_buf
                self.u_buf.mul_(self.cfg.post_trace_decay).add_(0.05 * u_mean.unsqueeze(-1).expand(-1, self.cfg.rank_eligibility))
                self.v_buf.mul_(self.cfg.post_trace_decay).add_(0.05 * v_mean.unsqueeze(0).expand(self.cfg.rank_eligibility, -1))

                # Update Postsynaptic state
                # Proxy calcium from output activity
                y.norm(dim=-1).mean().clamp(0, 10.0) # Scalar proxy for the batch
                # We need a vector for per-neuron update?
                # y is (B, out). ca_proxy should be (out,).
                ca_vec = y.abs().mean(0).clamp(0, 10.0)

                self.post.update(y, ca_vec)
                self.post.hebb_fast(self.u_buf, self.v_buf)
                self.post.consolidate(self.u_buf, self.v_buf)

        return y


# -----------------------------------------------------------------------------
# Presyn state builder
# -----------------------------------------------------------------------------


def build_presyn_state(B: int, T: int, H: int, device, dtype, cfg: SynapticConfig):
    R = torch.ones(B, H, T, device=device, dtype=dtype) * cfg.init_rrp
    res = torch.ones_like(R) * cfg.init_reserve
    c = torch.zeros_like(R)
    sn = torch.ones_like(R) * cfg.init_snare
    cl = torch.ones_like(R) * cfg.init_clamp
    amp = torch.ones_like(R) * cfg.init_amp
    en = torch.ones_like(R) * cfg.init_energy
    delay = [torch.zeros_like(R) for _ in range(cfg.endo_delay)]

    # Map to old keys for compatibility if needed, or just use new keys
    return {
        "rrp": R, "res": res, "c": c, "sn": sn, "cl": cl, "amp": amp, "en": en, "delay": delay,
        # Aliases for old code compatibility (if any)
        "RRP": R, "RES": res, "C": c, "PR": sn, "CL": cl, "E": en, "BUF": torch.zeros_like(R)
    }


# -----------------------------------------------------------------------------
# FlexAttention (optional)
# -----------------------------------------------------------------------------

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except Exception:  # pragma: no cover - depends on torch build
    create_block_mask = None
    flex_attention = None
    _HAS_FLEX = False
else:
    _HAS_FLEX = True


class SynapticFlexAttention(nn.Module):
    cfg: SynapticConfig

    def __init__(self, cfg: SynapticConfig):
        super().__init__()
        object.__setattr__(self, "cfg", cfg)
        if not _HAS_FLEX:
            raise ImportError(
                "SynapticFlexAttention requires torch.nn.attention.flex_attention "
                "(FlexAttention, torch>=2.5)."
            )

    @torch.compiler.disable
    def _precompute_bio_factors(self, state: dict[str, Any]) -> tuple[Tensor, Tensor]:
        c = state["c"]  # (B, H, T)
        rrp = state["rrp"]
        cl = state["cl"]
        sn = state["sn"]
        amp = state["amp"]

        p1 = torch.sigmoid(self.cfg.syt1_slope * (c - 0.55))
        p7 = torch.sigmoid(self.cfg.syt7_slope * (c - 0.25))
        p = p1 * 0.8 + p7 * 0.2 + self.cfg.doc2_gain * torch.sigmoid(4 * (c - 0.12))
        cpx_gate = torch.sigmoid(8.0 * (c - self.cfg.cpx_thresh) - 2.0 * (cl + self.cfg.complexin_bias))
        mix_prob = torch.clamp(p * cpx_gate * sn, 0.0, 0.999)

        key_factor = mix_prob * rrp
        qamp = amp
        return key_factor, qamp

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        presyn_state: dict[str, Any],
        *,
        ema: Tensor,
        block_mask,
        enable_gqa: bool,
        prefix_len: int,
    ) -> Tensor:
        if not _HAS_FLEX or flex_attention is None:
            raise RuntimeError("FlexAttention requested but unavailable at runtime.")
        B, H, Tq, D = q.shape
        Tk = k.size(2)

        key_factor, qamp = self._precompute_bio_factors(presyn_state)
        # Materialize pointwise-computed tensors so torch.compile/Inductor can lower flex_attention.
        # Without this, Inductor may keep `key_factor` as a FlexibleLayout ComputedBuffer and fail.
        key_factor = key_factor.clone()
        qamp = qamp.clone()
        barrier_strength = self.cfg.barrier_strength
        epsilon = self.cfg.epsilon
        lambda_loge = self.cfg.lambda_loge
        # Make denom a scalar () tensor so score_mod returns a scalar per element (not shape (1,)).
        denom = (ema.to(dtype=key_factor.dtype, device=key_factor.device) + epsilon).reshape(())

        def score_mod(score, b, h, q_idx, kv_idx):
            # NOTE: `score` is already scaled by `scale` inside FlexAttention
            # (defaults to 1/sqrt(head_dim) when scale=None).
            scaled_score = score

            kf_val = key_factor[b, h, kv_idx]
            qa_val = qamp[b, h, kv_idx]

            release = kf_val * torch.sigmoid(scaled_score)
            e_norm = (release * qa_val) / denom
            bio_bias = lambda_loge * torch.log(e_norm + epsilon)

            abs_q = prefix_len + q_idx
            dist = torch.abs(abs_q - kv_idx) / float(max(1, Tk))
            barrier = barrier_strength * dist

            return scaled_score + bio_bias - barrier

        return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask, enable_gqa=enable_gqa)


# -----------------------------------------------------------------------------
# Attention and MLP
# -----------------------------------------------------------------------------


class SynapticCausalSelfAttention(nn.Module):
    cfg: SynapticConfig
    """
    Drop-in attention with synaptic augmentation. Uses standard Q,K,V projections,
    RoPE, multi-query key/value replication, and adds log(ε+q⋅n) to logits.
    """

    def __init__(
        self,
        n_embd,
        n_head,
        n_kv_head,
        rope_cos,
        rope_sin,
        cfg: SynapticConfig,
        layer_idx: int = 0,
        attn_drop=0.0,
        resid_drop=0.0,
    ):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.layer_idx = int(layer_idx)
        self.head_dim = n_embd // n_head
        object.__setattr__(self, "cfg", cfg)
        if cfg.use_flex_attention:
            if not _HAS_FLEX:
                raise ImportError(
                    "SynapticConfig.use_flex_attention=True but FlexAttention is unavailable "
                    "(requires torch>=2.5 and torch.nn.attention.flex_attention)."
                )
            self.flex = SynapticFlexAttention(cfg)
        else:
            self.flex = None
        self.q_proj = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_head * self.head_dim, n_embd, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.resid_drop = nn.Dropout(resid_drop)
        self.cos, self.sin = rope_cos, rope_sin
        self.pre = SynapticPresyn(self.head_dim, cfg)

    def _apply_rope(self, x: Tensor, T0: int):
        H = self.n_head if x.size(-1) == self.n_head * self.head_dim else self.n_kv_head
        D = self.head_dim
        x = x.view(x.size(0), x.size(1), H, D)
        # IMPORTANT: Ensure RoPE tables match the activation dtype.
        # Mixing bf16 RoPE buffers with fp16 activations will promote to fp32 and break
        # downstream scatter_add dtypes in the synaptic presyn path.
        cos = self.cos[:, T0 : T0 + x.size(1), : D // 2].to(device=x.device, dtype=x.dtype).unsqueeze(2)
        sin = self.sin[:, T0 : T0 + x.size(1), : D // 2].to(device=x.device, dtype=x.dtype).unsqueeze(2)
        x1, x2 = x.split(D // 2, dim=-1)
        xr = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return xr

    def _repeat_kv(self, x: Tensor):
        if self.n_head == self.n_kv_head:
            return x
        nrep = self.n_head // self.n_kv_head
        b, t, nh, d = x.shape
        return x.unsqueeze(2).expand(b, t, nh, nrep, d).reshape(b, t, self.n_head, d)

    def forward(self, x: Tensor, kv_cache=None, presyn_state=None, train_mode=True):
        B, Tq, C = x.shape
        H = self.n_head
        D = self.head_dim
        device = x.device
        dtype = x.dtype

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x).view(B, Tq, self.n_kv_head, D)

        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        q = self._apply_rope(q, T0)
        k = self._apply_rope(k, T0)
        q = _rms(q)
        k = _rms(k)

        # (B, T, H, D) -> (B, H, T, D)
        q = q.transpose(1, 2)  # (B, Hq, Tq, D)
        k = k.transpose(1, 2)  # (B, Hkv, Tq, D)
        v = v.transpose(1, 2)  # (B, Hkv, Tq, D)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)  # (B, Hkv, Tk, D)
        Tk = k.size(2)

        # Ensure presyn_state is sized to keys (Tk), not just the current query chunk (Tq).
        if presyn_state is None:
            presyn_state = build_presyn_state(B, Tk, H, device, dtype, self.cfg)
        elif presyn_state["c"].size(2) != Tk:
            old_Tk = presyn_state["c"].size(2)
            if old_Tk > Tk:
                raise ValueError(f"presyn_state has more keys than k/v: {old_Tk} > {Tk}")
            new_state = build_presyn_state(B, Tk, H, device, presyn_state["c"].dtype, self.cfg)
            for key in ("rrp", "res", "c", "sn", "cl", "amp", "en"):
                new_state[key][:, :, :old_Tk].copy_(presyn_state[key])
            for di in range(min(len(new_state["delay"]), len(presyn_state["delay"]))):
                new_state["delay"][di][:, :, :old_Tk].copy_(presyn_state["delay"][di])
            presyn_state = new_state

        # Expand KV heads for manual attention / state updates (GQA).
        k_full, v_full = repeat_kv_heads(k, v, n_head=H)

        # Standard attention logits
        dots = (q @ k_full.transpose(-1, -2)) / math.sqrt(D)  # (B, H, Tq, Tk)
        mask = causal_attn_mask(Tq, Tk, device=q.device).view(1, 1, Tq, Tk)
        dots = dots.masked_fill(~mask, float("-inf"))

        # Top-k selection for synaptic physics (efficiency)
        topk = min(self.cfg.attn_topk, Tk)
        # We select topk keys for each query
        # dots is (B, H, Tq, Tk)
        vals, idx = torch.topk(dots, topk, dim=-1)

        # Drive for presyn is the attention logits (pre-softmax)
        drive = vals

        # --- FlexAttention Path ---
        if self.flex is not None:
            if create_block_mask is None:
                raise RuntimeError("FlexAttention requested but create_block_mask is unavailable.")
            prefix_len = Tk - Tq

            def causal_mask(b, h, q_idx, kv_idx):
                return kv_idx <= (prefix_len + q_idx)

            block_mask = create_block_mask(causal_mask, B, H, Tq, Tk, device=device)
            enable_gqa = self.n_head != self.n_kv_head

            q_f = q
            k_f = k
            v_f = v
            if q_f.dtype != v_f.dtype:
                q_f = q_f.to(v_f.dtype)
            if k_f.dtype != v_f.dtype:
                k_f = k_f.to(v_f.dtype)

            y = self.flex(
                q_f,
                k_f,
                v_f,
                presyn_state,
                ema=self.pre.ema_e.detach().clone(),
                block_mask=block_mask,
                enable_gqa=enable_gqa,
                prefix_len=prefix_len,
            )

            # Update presynaptic state after producing outputs so outputs are causal within a chunk.
            _ = self.pre.release(presyn_state, drive, idx, train_mode)

            y = y.transpose(1, 2).contiguous().view(B, Tq, H * D)
            y = self.resid_drop(self.o_proj(y))
            return y, presyn_state

        # Manual attention path.
        #
        # Important: keep output computation causal within the current chunk by:
        #   1) computing the logit augmentation from the *current* state (no within-chunk mutation),
        #   2) updating presynaptic state only after producing outputs (mirrors FlexAttention path).
        c = presyn_state["c"]  # (B, H, Tk)
        rrp = presyn_state["rrp"]
        cl = presyn_state["cl"]
        sn = presyn_state["sn"]
        amp = presyn_state["amp"]

        p1 = torch.sigmoid(self.cfg.syt1_slope * (c - 0.55))
        p7 = torch.sigmoid(self.cfg.syt7_slope * (c - 0.25))
        p = p1 * 0.8 + p7 * 0.2 + self.cfg.doc2_gain * torch.sigmoid(4 * (c - 0.12))
        cpx_gate = torch.sigmoid(8.0 * (c - self.cfg.cpx_thresh) - 2.0 * (cl + self.cfg.complexin_bias))
        mix_prob = torch.clamp(p * cpx_gate * sn, 0.0, 0.999)

        key_factor = mix_prob * rrp  # (B, H, Tk)
        denom = self.pre.ema_e.detach().clone().to(dtype=key_factor.dtype, device=key_factor.device) + self.cfg.epsilon

        # Logit augmentation (dense): matches SynapticFlexAttention.score_mod
        e_norm = (key_factor[:, :, None, :] * torch.sigmoid(dots) * amp[:, :, None, :]) / denom
        aug = self.cfg.lambda_loge * torch.log(e_norm + self.cfg.epsilon)

        # Distance barrier
        prefix_len = Tk - Tq
        q_steps = torch.arange(Tq, device=device, dtype=torch.float32) + float(prefix_len)
        k_steps = torch.arange(Tk, device=device, dtype=torch.float32)
        dist = (q_steps.view(1, 1, Tq, 1) - k_steps.view(1, 1, 1, Tk)).abs() / float(max(1, Tk))
        logits = dots + aug - self.cfg.barrier_strength * dist.to(dots.dtype)

        P = F.softmax(logits, dim=-1)
        P = self.attn_drop(P)

        ctx = torch.matmul(P.to(v_full.dtype), v_full)
        y = ctx.transpose(1, 2).contiguous().view(B, Tq, H * D)
        y = self.resid_drop(self.o_proj(y))
        _ = self.pre.release(presyn_state, drive, idx, train_mode)
        return y, presyn_state


class SynapticMLP(nn.Module):
    cfg: SynapticConfig
    def __init__(self, n_embd: int, cfg: SynapticConfig, dropout: float = 0.0):
        super().__init__()
        object.__setattr__(self, "cfg", cfg)
        self.fc = SynapticLinear(n_embd, 4 * n_embd, cfg, bias=True, use_input_ln=True)
        self.proj = SynapticLinear(
            4 * n_embd, n_embd, cfg, bias=True, use_input_ln=False
        )
        self.drop = nn.Dropout(dropout)
        self.register_buffer("C0", torch.tensor(0.5))
        self.register_buffer("E0", torch.tensor(0.8))

    def forward(self, x: Tensor):
        B, T, C = x.shape
        c0 = cast(Tensor, self.C0)
        e0 = cast(Tensor, self.E0)
        c = c0.expand(B * T)
        e = e0.expand(B * T)
        h = self.fc(x.reshape(B * T, C), c, e)
        h = F.relu(h).square()
        h = self.drop(h.reshape(B, T, -1))
        y = self.proj(h.reshape(B * T, -1), c, e).reshape(B, T, C)
        return y


# -----------------------------------------------------------------------------
# Synaptic MoE (router embeddings, contrastive updates)
# -----------------------------------------------------------------------------


class SynapticExpert(nn.Module):
    def __init__(
        self, n_embd: int, hidden_mult: int, cfg: SynapticConfig, dropout: float = 0.0
    ):
        super().__init__()
        h = hidden_mult * n_embd
        self.fc1 = SynapticLinear(n_embd, h, cfg, bias=True, use_input_ln=False)
        self.fc2 = SynapticLinear(h, n_embd, cfg, bias=True, use_input_ln=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, energy_override: Tensor | None = None, genes: Tensor | None = None) -> Tensor:
        # x: (N, C)
        N = x.size(0)
        device = x.device

        if energy_override is not None:
            if energy_override.ndim == 0:
                e_tens = energy_override.expand(N)
            else:
                e_tens = energy_override.view(-1).expand(N)
        else:
            e_tens = torch.ones(N, device=device)

        c_tens = torch.ones(N, device=device)

        y = self.fc1(
            x,
            calcium=c_tens,
            energy=e_tens,
            genes=genes,
        )
        y = F.relu(y).square()
        y = self.drop(y)
        y = self.fc2(
            y,
            calcium=c_tens,
            energy=e_tens,
            genes=genes,
        )
        return y


class SynapticMoE(nn.Module):
    num_experts: int
    top_k: int
    cfg: SynapticConfig
    last_aux_loss: Tensor | None
    last_ctx: dict[str, Tensor]
    """Top-k sparse Synaptic MoE with router embeddings, expert fatigue/energy,
    contrastive router-embedding updates, and split/merge structural hooks."""

    def __init__(
        self,
        n_embd: int,
        num_experts: int,
        top_k: int,
        hidden_mult: int,
        cfg: SynapticConfig,
        dropout: float = 0.0,
    ):
        super().__init__()
        object.__setattr__(self, "num_experts", num_experts)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "cfg", cfg)
        self.router = nn.Linear(n_embd, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                SynapticExpert(n_embd, hidden_mult, cfg, dropout)
                for _ in range(num_experts)
            ]
        )
        # Projects token features into router embedding space for alignment bias
        self.router_probe = nn.Linear(n_embd, cfg.router_embed_dim, bias=False)
        self.register_buffer("fatigue", torch.zeros(num_experts))
        self.register_buffer("energy", torch.ones(num_experts))
        # Router embeddings (biological identity) with unit-norm constraint
        emb = torch.randn(num_experts, cfg.router_embed_dim)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        self.router_embeddings = nn.Parameter(
            emb, requires_grad=False
        )  # updated by EMA-style rule
        object.__setattr__(self, "last_aux_loss", None)
        object.__setattr__(self, "last_ctx", {})

        # Molecular Genetics: Xi (The Genome)
        self.Xi = nn.Parameter(torch.zeros(num_experts, cfg.xi_dim))
        nn.init.normal_(self.Xi, std=0.1)

    def _get_phenotype(self, xi: Tensor) -> Tensor:
        """Map Xi logits to biological range constants."""
        fatigue_rate = 0.01 * (torch.sigmoid(xi[..., 0]) * 2.0 + 0.1)
        energy_fill = 0.005 * (torch.sigmoid(xi[..., 1]) * 2.0 + 0.1)
        camkii_gain = F.softplus(xi[..., 2] + 1.0)
        pp1_gain = F.softplus(xi[..., 3] + 0.5)
        return torch.stack([fatigue_rate, energy_fill, camkii_gain, pp1_gain], dim=-1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        B, T, C = x.shape
        E = self.num_experts
        device = x.device
        fatigue_buf = cast(Tensor, self.fatigue)
        energy_buf = cast(Tensor, self.energy)

        pheno = self._get_phenotype(self.Xi) # (E, 4)
        alpha_fatigue = pheno[:, 0]
        alpha_energy = pheno[:, 1]

        logits = self.router(x)  # (B,T,E)

        # Router bias logic (same as before)
        tok_proxy = x.mean(dim=-1, keepdim=True)
        base_bias = 0.02 * tok_proxy.expand(-1, -1, E)
        router_gain = self.router_embeddings.norm(dim=-1).view(1, 1, -1)
        gain_bias = 0.02 * tok_proxy * router_gain
        probe_feat = self.router_probe(x)
        tok_unit = F.normalize(probe_feat, dim=-1)
        router_unit = F.normalize(self.router_embeddings, dim=-1)
        align_bias = 0.02 * torch.einsum("btd,ed->bte", tok_unit, router_unit)
        bias = base_bias + gain_bias + align_bias
        gene_bias = 0.05 * (alpha_energy - alpha_fatigue).view(1, 1, E)

        logits = (
            logits
            + gene_bias
            + bias
            + 0.1 * energy_buf.view(1, 1, E)
            - 0.1 * fatigue_buf.view(1, 1, E)
        )
        topk = min(self.top_k, E)
        g, idx = torch.topk(logits, topk, dim=-1)
        gates = F.softmax(g, dim=-1)

        out = torch.zeros_like(x)
        flat_out = out.view(-1, C)
        flat_x = x.view(-1, C)

        me = torch.zeros(E, device=device)
        pe = torch.zeros(E, device=device)

        for e in range(E):
            mask = idx == e
            sel = mask.any(dim=-1)
            if not sel.any():
                continue
            flat_idx = sel.view(-1).nonzero(as_tuple=False).squeeze(1)
            x_e = flat_x.index_select(0, flat_idx)

            gene_e = pheno[e]
            energy_e = energy_buf[e]

            y_e = self.experts[e](x_e, energy_override=energy_e, genes=gene_e)
            w = gates.masked_select(mask).unsqueeze(-1)
            flat_out.index_add_(0, flat_idx, w * y_e)

            me[e] = sel.sum()
            pe[e] = gates.masked_select(mask).sum()

        with torch.no_grad():
            util = me.clamp_min(1.0) / float(B * T)
            fatigue_buf.mul_(1.0 - alpha_fatigue).add_(alpha_fatigue * util)
            energy_buf.mul_(1.0 - alpha_energy).add_(alpha_energy * (1.0 - util))

            object.__setattr__(self, "last_ctx", {
                "x": x.detach(),
                "indices": idx.detach(),
                "gates": gates.detach()
            })

        me = me / float(B * T)
        pe = pe / float(B * T)
        aux_loss = E * torch.sum(pe * me)
        self.last_aux_loss = aux_loss

        # Contrastive router-embedding update
        with torch.no_grad():
            cooc = torch.zeros(E, E, device=device)
            for e in range(E):
                cooc[e, e] = pe[e]
            emb = self.router_embeddings
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
            sim = emb @ emb.T
            pull = cooc * (sim - 1.0)
            push = (1.0 - cooc) * (sim + 0.3) * self.cfg.router_contrastive_push
            grad = pull - push
            grad = grad - grad.mean()
            grad = grad.to(emb.dtype)
            delta = (grad @ emb) * self.cfg.router_contrastive_lr
            emb = emb - delta
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
            self.router_embeddings.copy_(emb)

        return out, aux_loss


# -----------------------------------------------------------------------------
# Structural plasticity utility
# -----------------------------------------------------------------------------


class StructuralPlasticity(nn.Module):
    cfg: SynapticConfig
    def __init__(self, cfg: SynapticConfig):
        super().__init__()
        object.__setattr__(self, "cfg", cfg)
        self.register_buffer("age", torch.zeros(1))
        self.register_buffer("util", torch.zeros(1))

    @torch.no_grad()
    def step(self, used: Tensor):
        age = cast(Tensor, self.age)
        age.add_(1.0)
        util = cast(Tensor, self.util)
        util.mul_(1.0 - self.cfg.structural_tau_util).add_(
            self.cfg.structural_tau_util * used.float()
        )

    @torch.no_grad()
    def decision(self):
        util = cast(Tensor, self.util)
        age = cast(Tensor, self.age)
        s = torch.sigmoid(
            10.0 * (util - 0.2)
            - self.cfg.structural_age_bias
            * (age / float(self.cfg.structural_interval))
        )
        return (torch.rand_like(s) > s).item()


def structural_plasticity_step(
    expert_states: list[nn.Module], cfg: SynapticConfig, global_step: int
):
    if cfg.structural_interval < 1 or global_step % cfg.structural_interval != 0:
        return
    for st in expert_states:
        st = cast(StructuralPlasticity, st)
        st.step(used=torch.tensor(1.0))
        if st.decision():
            for p in st.parameters():
                nn.init.trunc_normal_(p, std=0.02)
