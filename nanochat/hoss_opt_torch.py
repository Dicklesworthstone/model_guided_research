"""
HOSS (Hyperreal OU Shadow Step) optimizer for PyTorch.
"""

import math
import torch
from torch.optim import Optimizer

def _symmetrize(M):
    return 0.5 * (M + M.T)

def _phi_delta_fraction(lam, delta):
    # (1 - exp(-delta * lam)) / lam
    # Handle small lambda for stability
    mask = torch.abs(lam) > 1e-12
    res = torch.zeros_like(lam)
    res[mask] = (1.0 - torch.exp(-delta * lam[mask])) / lam[mask]
    res[~mask] = delta
    return res

def _exp_delta_fraction(lam, delta):
    return torch.exp(-delta * lam)

def _lyapunov_integral_from_eigh(T, S, delta):
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

    # Construct T
    # alpha on diag, beta[:-1] on off-diag
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
                 REQUIRED for HOSS to compute HVP.
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

            # Flatten
            # We need to handle flattening carefully to support HVP
            # But hvp_fn needs to call closure? 
            # No, closure returns loss. We need grad(loss, params).
            # We already have grads.
            # HVP: grad(grad(loss) @ v).
            
            # Concatenate all params into one vector
            params_flat = torch.cat([p.view(-1) for p in params_list])
            grads_flat = torch.cat([g.view(-1) for g in grads_list])
            
            device = params_flat.device
            dtype = params_flat.dtype
            
            # Cast to float32 for stability
            params_flat_f32 = params_flat.float()
            grads_flat_f32 = grads_flat.float()
            
            # Gradient clipping
            if group['gradient_norm_clip'] is not None:
                g_norm = torch.linalg.norm(grads_flat_f32)
                if g_norm > group['gradient_norm_clip']:
                    scale = group['gradient_norm_clip'] / g_norm
                    grads_flat_f32.mul_(scale)

            # Define HVP function
            # We need to re-compute gradients inside HVP?
            # HVP(v) = lim (grad(w + eps*v) - grad(w))/eps
            # Or using autograd.grad
            
            # The closure computes loss.
            # We need a function that takes params_flat and returns gradients_flat?
            # This is expensive if we assume closure re-runs forward pass.
            # Efficient HVP usually relies on `torch.autograd.grad(outputs=grads, inputs=params, grad_outputs=v)`.
            # But we need `grads` graph to be retained?
            # In standard training `loss.backward()` frees the graph.
            # So `p.grad` does NOT have a graph.
            
            # CRITICAL: To use HOSS, the training loop must call `loss.backward(create_graph=True)`.
            # I will update train.py to handle this for HOSS.
            
            # HVP function
            def hvp_fn(v):
                # v is vector of size params
                # We need to unflatten v to match params_list
                v_list = []
                offset = 0
                for p in params_list:
                    numel = p.numel()
                    v_list.append(v[offset:offset+numel].view_as(p))
                    offset += numel
                
                # Compute HVP
                # grads_list must have been created with create_graph=True
                # grad_outputs = v_list
                
                # We want d(grad_loss)/dw * v
                # = grad(dot(grad_loss, v), w)
                
                # First compute dot product
                # But we don't have grad_loss as a tensor connected to graph unless we kept it.
                # This requires `grads_list` to be valid tensors with grad_fn.
                
                # If `closure` is passed, we can re-compute gradients?
                # "Standard" Second Order optimizers (L-BFGS) in PyTorch re-evaluate closure.
                # But that's for line search.
                
                # If we require `create_graph=True` in backward, we can use the existing grads.
                # Let's assume `grads_list` has graph.
                
                # Check if grads have graph
                if not grads_list[0].requires_grad:
                     # Fallback: Use finite difference if no graph? 
                     # Or re-run backward?
                     # Re-running backward inside HVP is very expensive (R * Backward).
                     # Better to require create_graph=True once.
                     pass

                # d(loss)/dw
                # vector-Jacobian product of gradient?
                # torch.autograd.grad(outputs=grads_list, inputs=params_list, grad_outputs=v_list, retain_graph=True)
                
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

            # Run Lanczos
            lanczos_rank = group['lanczos_rank']
            Q, T, g_norm = lanczos_sym(hvp_fn, grads_flat_f32, lanczos_rank, device)
            
            # Eigen decomp
            lam_T, V_T = torch.linalg.eigh(T)
            lam_T = torch.clamp(lam_T, min=group['min_curvature'])
            
            delta = group['lr']
            
            # Mean update
            phi_delta_T = (V_T * _phi_delta_fraction(lam_T, delta)) @ V_T.T
            
            # The mean update is -Phi(H)g.
            # We project g: Q.T @ g
            # Note: Q[:, 0] is g/norm. So Q.T @ g = [norm, 0, ...]
            projected_grad = torch.zeros(lanczos_rank, device=device, dtype=torch.float32)
            projected_grad[0] = g_norm
            
            mean_update_projected = -phi_delta_T @ projected_grad
            mean_update_flat = Q @ mean_update_projected
            
            # Noise
            # Isotropic assumption
            S_hat = group['isotropic_noise_var'] * torch.eye(lanczos_rank, device=device, dtype=torch.float32)
            C_delta_T = _lyapunov_integral_from_eigh(T, S_hat, delta)
            
            # Sample noise
            noise_projected = torch.distributions.MultivariateNormal(
                torch.zeros(lanczos_rank, device=device, dtype=torch.float32), 
                covariance_matrix=C_delta_T + 1e-6 * torch.eye(lanczos_rank, device=device) # stability
            ).sample()
            
            noise_flat = group['noise_scale'] * (Q @ noise_projected)
            
            total_update = mean_update_flat + noise_flat
            
            # Apply update
            offset = 0
            for p in params_list:
                numel = p.numel()
                update_p = total_update[offset:offset+numel].view_as(p).to(p.dtype)
                p.data.add_(update_p)
                offset += numel

        return loss
