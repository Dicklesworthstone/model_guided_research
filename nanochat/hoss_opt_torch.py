"""
HOSS (Hyperreal OU Shadow Step) optimizer for PyTorch.
Implements nonstandard analysis optimization with infinitesimal perturbations.

Core logic:
- Uses a shadow step `mean_update` based on curvature (Lanczos).
- Adds `noise` scaled by Lyapunov integral of the Hessian.
- Requires Hessian-Vector Products (HVP), so `closure` must be passed to `step`.
"""

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

def _lyapunov_integral_from_eigh(T, S, delta):
    # Computes int_0^delta exp(-sT) S exp(-sT) ds
    # via eigendecomposition of T.
    lam, V = torch.linalg.eigh(_symmetrize(T))
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
    def __init__(self, params, lr=1e-3, lanczos_rank=10, noise_scale=1.0, isotropic_noise_var=1e-4, min_curvature=1e-6, gradient_norm_clip=1.0):
        defaults = dict(lr=lr, lanczos_rank=lanczos_rank, noise_scale=noise_scale,
                        isotropic_noise_var=isotropic_noise_var, min_curvature=min_curvature,
                        gradient_norm_clip=gradient_norm_clip)
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
            with torch.enable_grad():
                loss = closure()

        if closure is None:
            raise RuntimeError("HOSS requires a closure that calculates loss to compute Hessian-Vector Products.")

        for group in self.param_groups:
            # Flatten params and grads for this group
            params_list = []
            grads_list = []
            for p in group['params']:
                if p.grad is not None:
                    params_list.append(p)
                    grads_list.append(p.grad)

            if not params_list:
                continue

            if any(not g.requires_grad for g in grads_list):
                raise RuntimeError(
                    "HOSS requires gradients with create_graph=True. "
                    "Call loss.backward(create_graph=True) before HOSS.step()."
                )

            # Flatten params and grads
            params_flat = torch.cat([p.view(-1) for p in params_list])
            grads_flat = torch.cat([g.view(-1) for g in grads_list])

            device = params_flat.device

            # Cast to float32 for stability in Lanczos
            params_flat.float()
            grads_flat_f32 = grads_flat.float()

            # Gradient clipping
            if group['gradient_norm_clip'] is not None:
                g_norm = torch.linalg.norm(grads_flat_f32)
                if g_norm > group['gradient_norm_clip']:
                    scale = group['gradient_norm_clip'] / g_norm
                    grads_flat_f32.mul_(scale)

            # Define HVP function
            # Uses autograd.grad on existing gradients (double backprop)
            # Requires gradients to have been created with create_graph=True

            def hvp_fn(v):
                # v is vector of size params
                # Unflatten v to match params_list
                v_list = []
                offset = 0
                for p in params_list:
                    numel = p.numel()
                    v_list.append(v[offset:offset+numel].view_as(p))
                    offset += numel

                # Compute HVP: grad( dot(grads, v) )
                hvp_list = torch.autograd.grad(
                    outputs=grads_list,
                    inputs=params_list,
                    grad_outputs=v_list,
                    retain_graph=True,
                    allow_unused=True
                )

                # Flatten result
                hvp_flat = []
                for g, p in zip(hvp_list, params_list):
                    if g is None:
                        hvp_flat.append(torch.zeros_like(p).view(-1))
                    else:
                        hvp_flat.append(g.view(-1))

                return torch.cat(hvp_flat).float()

            # Run Lanczos to get Krylov subspace Q and tridiagonal T
            lanczos_rank = group['lanczos_rank']
            Q, T, g_norm = lanczos_sym(hvp_fn, grads_flat_f32, lanczos_rank, device)

            # Eigen decomp of T
            lam_T, V_T = torch.linalg.eigh(T)
            lam_T = torch.clamp(lam_T, min=group['min_curvature'])

            delta = group['lr']

            # Mean update via phi function
            phi_delta_T = (V_T * _phi_delta_fraction(lam_T, delta)) @ V_T.T

            # The mean update is -Phi(H)g.
            # Project g into Krylov basis: Q.T @ g = [norm, 0, ...]
            projected_grad = torch.zeros(lanczos_rank, device=device, dtype=torch.float32)
            projected_grad[0] = g_norm

            mean_update_projected = -phi_delta_T @ projected_grad
            mean_update_flat = Q @ mean_update_projected

            # Noise injection (Lyapunov integral)
            S_hat = group['isotropic_noise_var'] * torch.eye(lanczos_rank, device=device, dtype=torch.float32)
            C_delta_T = _lyapunov_integral_from_eigh(T, S_hat, delta)

            # Sample noise in Krylov space
            noise_projected = torch.distributions.MultivariateNormal(
                torch.zeros(lanczos_rank, device=device, dtype=torch.float32),
                covariance_matrix=C_delta_T + 1e-6 * torch.eye(lanczos_rank, device=device)
            ).sample()

            noise_flat = group['noise_scale'] * (Q @ noise_projected)

            total_update = mean_update_flat + noise_flat

            # Apply update to parameters
            offset = 0
            for p in params_list:
                numel = p.numel()
                update_p = total_update[offset:offset+numel].view_as(p).to(p.dtype)
                p.data.add_(update_p)
                offset += numel

        return loss
