"""
HOSS (Hyperreal OU Shadow Step) optimizer for PyTorch.
Implements nonstandard analysis optimization with infinitesimal perturbations.

Core logic:
- Uses a shadow step `mean_update` based on curvature (Lanczos).
- Adds `noise` scaled by Lyapunov integral of the Hessian.
- Requires Hessian-Vector Products (HVP), so `closure` must be passed to `step`.
"""

import math

import torch
from torch.optim import Optimizer


def _symmetrize(M):
    return 0.5 * (M + M.T)


def _phi_delta_fraction(lam, delta):
    # (1 - exp(-delta * lam)) / lam
    # Handle small lambda for stability via Taylor or mask
    mask = torch.abs(lam) > 1e-12
    res = torch.zeros_like(lam)
    res[mask] = (1.0 - torch.exp(-delta * lam[mask])) / lam[mask]
    res[~mask] = delta
    return res


def _exp_delta_fraction(lam, delta):
    return torch.exp(-delta * lam)


def _lyapunov_integral_from_eigh(T, S, delta, min_curvature=None):
    # Computes int_0^delta exp(-sT) S exp(-sT) ds
    # via eigendecomposition of T.
    lam, V = torch.linalg.eigh(_symmetrize(T))
    if min_curvature is not None:
        lam = torch.clamp(lam, min=min_curvature)
    S_hat = V.T @ S @ V
    lam_i = lam[:, None]
    lam_j = lam[None, :]
    D = lam_i + lam_j

    mask = torch.abs(D) > 1e-12
    frac = torch.zeros_like(D)

    num = 1.0 - torch.exp(-delta * D[mask])
    frac[mask] = num / D[mask]
    frac[~mask] = delta

    C_hat = S_hat * frac
    C = V @ C_hat @ V.T
    return _symmetrize(C)


def _psd_matrix_sqrt(M):
    """Symmetric positive-semidefinite square root via eigendecomposition (zero Cholesky jitter)."""
    eigvals, eigvecs = torch.linalg.eigh(_symmetrize(M))
    sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=0.0))
    return (eigvecs * sqrt_eigvals) @ eigvecs.T


def lanczos_sym(hvp_fn, vec_g, r, device):
    # Symmetric Lanczos iteration to approximate Hessian T from HVP
    d = vec_g.shape[0]
    Q = torch.zeros((d, r), device=device, dtype=vec_g.dtype)
    alpha = torch.zeros((r,), device=device, dtype=vec_g.dtype)
    beta = torch.zeros((r,), device=device, dtype=vec_g.dtype)

    q_im1 = torch.zeros_like(vec_g)
    g_norm = torch.linalg.norm(vec_g)

    if g_norm > 1e-30:
        q_i = vec_g / g_norm
    else:
        q_i = torch.zeros_like(vec_g)

    beta_im1 = 0.0

    for i in range(r):
        v = hvp_fn(q_i)
        a_i = torch.dot(q_i, v)
        v = v - a_i * q_i - beta_im1 * q_im1
        b_i = torch.linalg.norm(v)

        if b_i > 1e-30:
            q_ip1 = v / b_i
        else:
            q_ip1 = torch.zeros_like(v)

        Q[:, i] = q_i
        alpha[i] = a_i
        beta[i] = b_i

        q_im1 = q_i
        q_i = q_ip1
        beta_im1 = b_i

    # Construct T (tridiagonal)
    T = torch.diag(alpha) + torch.diag(beta[:-1], 1) + torch.diag(beta[:-1], -1)

    return Q, T, g_norm


class HOSS(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        lanczos_rank=10,
        noise_scale=1.0,
        isotropic_noise_var=1e-4,
        min_curvature=1e-6,
        gradient_norm_clip=1.0,
        normalize_grad_rms=False,
    ):
        defaults = dict(
            lr=lr,
            lanczos_rank=lanczos_rank,
            noise_scale=noise_scale,
            isotropic_noise_var=isotropic_noise_var,
            min_curvature=min_curvature,
            gradient_norm_clip=gradient_norm_clip,
            normalize_grad_rms=normalize_grad_rms,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        closure: A closure that reevaluates the model and returns the loss.
                 REQUIRED for HOSS to compute Hessian-Vector Products (HVP).
        """
        loss = None
        if closure is not None:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
                loss = closure()

        if closure is None:
            raise RuntimeError("HOSS requires a closure that calculates loss to compute Hessian-Vector Products.")

        for group in self.param_groups:
            # Flatten params and grads for this group
            params_list = []
            grads_list = []
            for p in group["params"]:
                if p.grad is not None:
                    params_list.append(p)
                    grads_list.append(p.grad)

            if not params_list:
                continue

            # Flatten params and grads
            params_flat = torch.cat([p.view(-1) for p in params_list])
            grads_flat = torch.cat([g.view(-1) for g in grads_list])

            device = params_flat.device

            # Lanczos runs in float32 for stability; the parameter vector itself
            # is only needed for its device/layout (updates are cast back per
            # parameter below).
            grads_flat_f32 = grads_flat.float()

            # Gradient clipping
            if group["gradient_norm_clip"] is not None:
                g_norm = torch.linalg.norm(grads_flat_f32)
                if g_norm > group["gradient_norm_clip"]:
                    scale = group["gradient_norm_clip"] / g_norm
                    grads_flat_f32.mul_(scale)

            # Define HVP function
            # Uses autograd.grad on existing gradients (double backprop).
            # Parameters that enter linearly have g.requires_grad == False and
            # zero second derivative.
            def hvp_fn(v):
                active_grads = []
                active_params = []
                active_v = []
                active_indices = []
                offset = 0
                for idx, (p, g) in enumerate(zip(params_list, grads_list)):
                    numel = p.numel()
                    v_p = v[offset : offset + numel].view_as(p)
                    offset += numel
                    if g.requires_grad:
                        active_grads.append(g)
                        active_params.append(p)
                        active_v.append(v_p)
                        active_indices.append(idx)

                if not active_grads:
                    return torch.zeros_like(v).float()

                hvp_active = torch.autograd.grad(
                    outputs=active_grads,
                    inputs=active_params,
                    grad_outputs=active_v,
                    retain_graph=True,
                    allow_unused=True,
                )
                active_map = dict(zip(active_indices, hvp_active))

                hvp_flat = []
                for idx, p in enumerate(params_list):
                    g_hvp = active_map.get(idx)
                    if g_hvp is None:
                        hvp_flat.append(torch.zeros_like(p).view(-1))
                    else:
                        hvp_flat.append(g_hvp.reshape(-1))

                return torch.cat(hvp_flat).float()

            # Run Lanczos to get Krylov subspace Q and tridiagonal T
            lanczos_rank = int(group["lanczos_rank"])
            Q, T, g_norm = lanczos_sym(hvp_fn, grads_flat_f32, lanczos_rank, device)

            if group.get("normalize_grad_rms", False) and g_norm > 1e-30:
                rms_scale = math.sqrt(grads_flat_f32.numel()) / g_norm
                eff_grad = grads_flat_f32 * rms_scale
            else:
                eff_grad = grads_flat_f32

            # Eigen decomp of T (symmetrized and clamped to min_curvature)
            lam_T, V_T = torch.linalg.eigh(_symmetrize(T))
            min_curv = group["min_curvature"]
            if min_curv is not None:
                lam_T = torch.clamp(lam_T, min=min_curv)

            delta = group["lr"]

            # Mean update via phi function: -Q Phi_delta(T_clamped) Q^T g - delta (I - Q Q^T) g
            phi_delta_T = (V_T * _phi_delta_fraction(lam_T, delta)) @ V_T.T
            projected_grad = Q.T @ eff_grad
            ortho_grad = eff_grad - (Q @ projected_grad)
            mean_update_projected = -phi_delta_T @ projected_grad
            mean_update_flat = (Q @ mean_update_projected) - delta * ortho_grad

            # Noise injection (Lyapunov integral on the SAME clamped spectrum, zero Cholesky jitter)
            noise_scale = float(group["noise_scale"])
            noise_var = float(group["isotropic_noise_var"])
            if noise_scale != 0.0 and noise_var > 0.0:
                S_hat = noise_var * torch.eye(lanczos_rank, device=device, dtype=torch.float32)
                C_delta_T = _lyapunov_integral_from_eigh(T, S_hat, delta, min_curvature=min_curv)
                sqrt_C = _psd_matrix_sqrt(C_delta_T)
                z = torch.randn(lanczos_rank, device=device, dtype=torch.float32)
                noise_projected = sqrt_C @ z
                noise_flat = noise_scale * (Q @ noise_projected)
            else:
                noise_flat = torch.zeros_like(mean_update_flat)

            total_update = mean_update_flat + noise_flat

            # Apply update to parameters
            offset = 0
            for p in params_list:
                numel = p.numel()
                update_p = total_update[offset : offset + numel].view_as(p).to(p.dtype)
                p.data.add_(update_p)
                offset += numel

        return loss

