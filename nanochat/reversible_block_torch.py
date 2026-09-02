"""
Reversible Block Module (PyTorch)

Two coupling modes on the half-split stream x = [x1, x2] (bead u55.5):

additive (the classic RevNet form — volume-preserving for ANY F, G):
    y1 = x1 + F(x2)          y2 = x2 + G(y1)
    x2 = y2 - G(y1)          x1 = y1 - F(x2)

symplectic (kick-kick / Stormer-Verlet, F and G are EXACT GRADIENTS of
scalar potentials — each kick is an exact symplectic shear, so the block
satisfies J^T Omega J = Omega identically; theory note
markdown_documentation/symplectic_transformer.md):
    y1 = x1 + grad(phi_F)(x2)        y2 = x2 - grad(phi_G)(y1)
    x2 = y2 + grad(phi_G)(y1)        x1 = y1 - grad(phi_F)(x2)

The MINUS on the second kick is load-bearing: it makes the block the
splitting integrator of the COERCIVE Hamiltonian H = phi_G(x1) + phi_F(x2),
whose shadow is conserved across tied depth and bounds activation norms.
With ++ signs the conserved quantity is the non-coercive DIFFERENCE
phi_F - phi_G and norms can blow up (validated: 6.5e4x at kick scale 0.2).

O(1)-memory training is ACTIVE on the additive path (bead a6k3): grad-enabled
forwards without a KV cache route through ReversibleFunction below, whose
backward RECOMPUTES the activations instead of storing them. Decode/generation
(kv_cache set) and no-grad eval stay eager. Symplectic stays eager too: its
kicks are double-backward (create_graph=True), which does not compose with the
detach-and-recompute trick. The kick inverse is the negative kick, exact to
machine epsilon.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F_torch


class _EnergyHead(nn.Module):
    """Scalar potential phi(x) = sum_t v^T tanh(W h_t) + (lambda/2) ||x||^2,
    h = inner(x) the wrapped attention/MLP block on the half-stream.

    Coercivity by construction (u55.5 polish round 3): the tanh head is
    BOUNDED (|phi_net| <= T ||v||_1), so the learnable confinement
    lambda = lambda_min + softplus(lambda_raw) >= lambda_min makes every
    level set of phi bounded, with the explicit activation bound
    ||x||^2 <= (2/lambda_min) (H + 2 T max ||v||_1) derived in the note.
    lambda_min = 0 is allowed as the registered falsification control."""

    def __init__(self, inner: nn.Module, dim: int, lambda_min: float, inner_takes_ctx: bool):
        super().__init__()
        self.inner = inner
        self.inner_takes_ctx = inner_takes_ctx
        self.W = nn.Linear(dim, dim, bias=False)
        self.v = nn.Parameter(torch.randn(dim) * 0.02)
        self.lambda_min = float(lambda_min)
        # softplus(0.0541) ~ 0.72 is too hot; pick raw so initial lambda ~ lambda_min + 0.05
        self.lambda_raw = nn.Parameter(torch.tensor(_softplus_inverse(0.05)))

    @property
    def confinement(self) -> torch.Tensor:
        return self.lambda_min + F_torch.softplus(self.lambda_raw)

    def forward(self, x, cos_sin=None, kv_cache=None) -> torch.Tensor:
        # x: (B, T, C_half) -> phi: (B,)  [token-coupled scalar per element]
        h = self.inner(x, cos_sin, kv_cache) if self.inner_takes_ctx else self.inner(x)
        e: torch.Tensor = torch.tanh(self.W(h)) @ self.v  # (B, T)
        phi: torch.Tensor = e.sum(dim=-1) + 0.5 * self.confinement * x.pow(2).sum(dim=(-2, -1))
        return phi


def _softplus_inverse(y: float) -> float:
    import math

    return math.log(math.expm1(y))


class SymplecticKick(nn.Module):
    """Exact-gradient kick: returns grad_x phi(x).

    Differentiability contract: under grad mode the kick output carries a
    create_graph=True second-order graph (the autograd-of-autograd training
    path — phi's parameters AND upstream activations both receive gradients
    through the kick). Under no_grad (eval/generation) the gradient is
    computed in a local enable_grad island and returned detached."""

    def __init__(self, energy: _EnergyHead):
        super().__init__()
        self.energy = energy
        self.last_phi: float | None = None  # telemetry, populated when recording

    def forward(self, x, cos_sin=None, kv_cache=None, record: bool = False) -> torch.Tensor:
        # The kick's training path differentiates THROUGH the energy gradient
        # (double backward); fused SDPA kernels have no second derivative, so
        # the energy evaluation pins the math backend (plain matmul+softmax,
        # double-differentiable everywhere).
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if torch.is_grad_enabled():
            xg = x if x.requires_grad else x.detach().requires_grad_(True)
            with sdpa_kernel(SDPBackend.MATH):
                phi = self.energy(xg, cos_sin, kv_cache)
                if record:
                    self.last_phi = float(phi.detach().mean())
                (g,) = torch.autograd.grad(phi.sum(), xg, create_graph=True)
            return g
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            xl = x.detach().requires_grad_(True)
            phi = self.energy(xl, cos_sin, kv_cache)
            if record:
                self.last_phi = float(phi.detach().mean())
            (g,) = torch.autograd.grad(phi.sum(), xl)
        return g.detach()


class ReversibleBlock(nn.Module):
    def __init__(self, config, layer_idx, f_block, g_block):
        super().__init__()
        self.layer_idx = layer_idx
        self.mode = str(getattr(config, "reversible_mode", "additive"))
        self.record_energy = bool(getattr(config, "reversible_record_energy", False))
        self.dim = config.n_embd
        if self.dim % 2 != 0:
            raise ValueError("n_embd must be even for Reversible Block")
        if self.mode == "symplectic":
            half = self.dim // 2
            lam_min = float(getattr(config, "reversible_lambda_min", 0.05))
            # attention rides inside phi_F (needs cos_sin/kv_cache), the MLP
            # inside phi_G; both reduced to scalars by the bounded energy head
            self.f_block = SymplecticKick(_EnergyHead(f_block, half, lam_min, inner_takes_ctx=True))
            self.g_block = SymplecticKick(_EnergyHead(g_block, half, lam_min, inner_takes_ctx=False))
        elif self.mode == "additive":
            self.f_block = f_block  # Attention-like
            self.g_block = g_block  # MLP-like
        else:
            raise ValueError(f"unknown reversible_mode {self.mode!r} (additive | symplectic)")
        # Per-call telemetry trace (train.py drains once per step). A TIED
        # block is the same object called n_layer times per forward, so the
        # trace carries one entry per LAYER in call order - the across-layer
        # shadow-energy band is the conservation observable (note section 2).
        self.energy_trace: list[dict[str, float]] = []

    def drain_energy_trace(self) -> list[dict[str, float]]:
        trace, self.energy_trace = self.energy_trace, []
        return trace

    def forward(self, x, cos_sin, kv_cache):
        # x: (B, T, C)
        x1, x2 = torch.chunk(x, 2, dim=-1)

        if self.mode == "symplectic":
            rec = self.record_energy
            f_out = self.f_block(x2, cos_sin, kv_cache, record=rec)
            y1 = x1 + f_out
            g_out = self.g_block(y1, record=rec)
            # MINUS: the corrected kick-kick sign — conserves the coercive
            # H = phi_G + phi_F instead of the unbounded difference (note 1.1)
            y2 = x2 - g_out
            if rec:
                with torch.no_grad():
                    phi_f = self.f_block.last_phi if self.f_block.last_phi is not None else float("nan")
                    phi_g = self.g_block.last_phi if self.g_block.last_phi is not None else float("nan")
                    self.energy_trace.append(
                        {
                            "phi_f": phi_f,
                            "phi_g": phi_g,
                            "shadow_energy": phi_f + phi_g,
                            "x1_norm": float(x1.detach().pow(2).sum(dim=(-2, -1)).mean()),
                            "x2_norm": float(x2.detach().pow(2).sum(dim=(-2, -1)).mean()),
                            "lambda_f": float(self.f_block.energy.confinement.detach()),
                            "lambda_g": float(self.g_block.energy.confinement.detach()),
                        }
                    )
            return torch.cat([y1, y2], dim=-1)

        # additive: y1 = x1 + F(x2); y2 = x2 + G(y1)
        if torch.is_grad_enabled() and kv_cache is None:
            # Training path (bead a6k3): the custom autograd Function below
            # stores only the block OUTPUT and recomputes F/G activations in
            # backward — this is what realizes the headline O(1)-in-depth
            # activation memory. Eager (graph-storing) coupling remains for
            # decode/generation (kv_cache set: re-running F inside backward
            # would append to the cache twice) and for no-grad eval.
            return ReversibleFunction.apply(x, cos_sin, kv_cache, self.f_block, self.g_block)
        f_out = self.f_block(x2, cos_sin, kv_cache)
        y1 = x1 + f_out
        g_out = self.g_block(y1)
        y2 = x2 + g_out
        return torch.cat([y1, y2], dim=-1)

    def inverse(self, y, cos_sin, kv_cache):
        y1, y2 = torch.chunk(y, 2, dim=-1)

        if self.mode == "symplectic":
            # kick inverse = negative kick, EXACT (machine eps; note 1.2)
            g_out = self.g_block(y1)
            x2 = y2 + g_out
            f_out = self.f_block(x2, cos_sin, kv_cache)
            x1 = y1 - f_out
            return torch.cat([x1, x2], dim=-1)

        # additive inverse: x2 = y2 - G(y1); x1 = y1 - F(x2)
        g_out = self.g_block(y1)
        x2 = y2 - g_out
        f_out = self.f_block(x2, cos_sin, kv_cache)
        x1 = y1 - f_out
        return torch.cat([x1, x2], dim=-1)


# Custom Autograd Function to enable memory saving
class ReversibleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos_sin, kv_cache, f_module, g_module):
        # We don't save x for backward!
        # We only save output y (or we can recompute x from y)
        # Saving y is standard.

        # Manual forward
        x1, x2 = torch.chunk(x, 2, dim=-1)

        # Disable grad tracking for forward pass to save graph memory?
        # But we need to track params.
        # The trick is: We detach inputs, run forward, and in backward we recompute inputs.

        with torch.no_grad():
            f_out = f_module(x2, cos_sin, kv_cache)
            y1 = x1 + f_out
            g_out = g_module(y1)
            y2 = x2 + g_out

        y = torch.cat([y1, y2], dim=-1)

        ctx.save_for_backward(y)  # Save output
        ctx.cos_sin = cos_sin
        ctx.kv_cache = kv_cache
        ctx.f_module = f_module
        ctx.g_module = g_module
        # Autograd runs a custom Function's backward with autocast DISABLED, so
        # under bf16 mixed precision the recompute below would re-run F/G in
        # fp32 while the forward ran them in bf16: x2 = y2 - G(y1) would then
        # be reconstructed from a different G than the one that produced y2,
        # and the gradients would belong to a different network than the
        # forward. Record the forward's autocast state and re-enter it in
        # backward (what torch.amp.custom_fwd/custom_bwd do for a fixed device).
        device_type = x.device.type
        ctx.autocast_state = (
            device_type,
            torch.is_autocast_enabled(device_type),
            torch.get_autocast_dtype(device_type),
        )

        return y

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        if len(grad_outputs) != 1:
            raise RuntimeError(f"ReversibleFunction expected one output gradient, got {len(grad_outputs)}")
        grad_y = grad_outputs[0]
        y = ctx.saved_tensors[0]
        cos_sin = ctx.cos_sin
        kv_cache = ctx.kv_cache
        f_module = ctx.f_module
        g_module = ctx.g_module

        y1, y2 = torch.chunk(y, 2, dim=-1)
        dy1, dy2 = torch.chunk(grad_y, 2, dim=-1)
        device_type, autocast_enabled, autocast_dtype = ctx.autocast_state

        # Every recompute below runs under the SAME autocast state as the
        # forward (see forward): the reconstruction x2 = y2 - G(y1) is only
        # exact against the G that actually produced y2.
        with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=autocast_enabled):
            # Reconstruct x2 (the only input the gradient computation below
            # needs). x1 = y1 - F(x2) is never used here, so its extra F forward
            # is skipped; requires_grad is set on detached clones, not these.
            with torch.no_grad():
                g_out = g_module(y1)
                x2 = y2 - g_out

            # RevNet backward: re-run G and F with grad on reconstructed inputs
            # so parameter gradients accumulate through autograd, then chain the
            # input gradients by hand along the coupling identities
            #   y2 = x2 + G(y1)   ->  dx2 += dy2,  dy1 += dG^T dy2
            #   y1 = x1 + F(x2)   ->  dx1  = dy1,  dx2 += dF^T dy1
            with torch.enable_grad():
                y1_detached = y1.detach().requires_grad_(True)
                g_out = g_module(y1_detached)
                g_out.backward(dy2)
                y1_grad = y1_detached.grad
                if y1_grad is None:
                    raise RuntimeError("Reversible G recomputation did not produce an input gradient")
                dy1_total = dy1 + y1_grad

                x2_detached = x2.detach().requires_grad_(True)
                f_out = f_module(x2_detached, cos_sin, kv_cache)
                f_out.backward(dy1_total)
                x2_grad = x2_detached.grad
                if x2_grad is None:
                    raise RuntimeError("Reversible F recomputation did not produce an input gradient")
                dx2_total = dy2 + x2_grad  # dy2 arrives through the identity path x2 -> y2
                dx1_total = dy1_total  # x1 -> y1 is the identity

        return torch.cat([dx1_total, dx2_total], dim=-1), None, None, None, None
