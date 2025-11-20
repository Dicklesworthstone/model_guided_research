"""
Reversible Block Module (PyTorch)
Implements a Reversible Block using Additive Coupling or Symplectic/Cayley transforms.
Allows O(1) memory training (in theory, via checkpointing/recomputation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nanochat.model_utils import norm

# Simple Reversible Block:
# x = [x1, x2]
# y1 = x1 + F(x2)
# y2 = x2 + G(y1)
# Inverse:
# x2 = y2 - G(y1)
# x1 = y1 - F(x2)

class ReversibleBlock(nn.Module):
    def __init__(self, config, layer_idx, f_block, g_block):
        super().__init__()
        self.layer_idx = layer_idx
        self.f_block = f_block # Attention-like
        self.g_block = g_block # MLP-like
        
        # For now, we assume f_block and g_block are instantiated modules
        # that take half the embedding dimension?
        # Or we act on the full dimension but use masks?
        # Standard RevNet splits channels in half.
        self.dim = config.n_embd
        if self.dim % 2 != 0:
            raise ValueError("n_embd must be even for Reversible Block")

    def forward(self, x, cos_sin, kv_cache):
        # x: (B, T, C)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        
        # Forward pass
        # Note: We need to pass cos_sin/kv_cache to F (Attention)
        # But G (MLP) doesn't need them.
        
        # F block (Attention side)
        # We need to pad x2 to full dim if F expects full dim?
        # Or F is designed for half dim.
        # Let's assume F and G are designed for C/2.
        
        # y1 = x1 + F(x2)
        f_out = self.f_block(x2, cos_sin, kv_cache)
        y1 = x1 + f_out
        
        # y2 = x2 + G(y1)
        g_out = self.g_block(y1)
        y2 = x2 + g_out
        
        return torch.cat([y1, y2], dim=-1)

    def inverse(self, y, cos_sin, kv_cache):
        y1, y2 = torch.chunk(y, 2, dim=-1)
        
        # Inverse pass
        # x2 = y2 - G(y1)
        g_out = self.g_block(y1)
        x2 = y2 - g_out
        
        # x1 = y1 - F(x2)
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
        
        ctx.save_for_backward(y) # Save output
        ctx.cos_sin = cos_sin
        ctx.kv_cache = kv_cache
        ctx.f_module = f_module
        ctx.g_module = g_module
        
        return y

    @staticmethod
    def backward(ctx, grad_y):
        y = ctx.saved_tensors[0]
        cos_sin = ctx.cos_sin
        kv_cache = ctx.kv_cache
        f_module = ctx.f_module
        g_module = ctx.g_module
        
        y1, y2 = torch.chunk(y, 2, dim=-1)
        dy1, dy2 = torch.chunk(grad_y, 2, dim=-1)
        
        # Reconstruct x
        with torch.no_grad():
            g_out = g_module(y1)
            x2 = y2 - g_out
            f_out = f_module(x2, cos_sin, kv_cache)
            x1 = y1 - f_out
        
        x1.requires_grad = True
        x2.requires_grad = True
        
        # Now recompute gradients
        # Backward G
        # y2 = x2 + G(y1)
        # dy2 flows to dx2 (identity) and dG(y1)
        # dG(y1) flows to params_G and dy1
        
        # Standard RevNet backward logic is complex to implement manually in PyTorch
        # without hooking into autograd for parameters.
        # We need to use `torch.autograd.backward` or run forward with grad enabled on reconstructed inputs.
        
        with torch.enable_grad():
            # Recompute G
            y1_detached = y1.detach()
            y1_detached.requires_grad = True
            g_out = g_module(y1_detached)
            
            g_out.backward(dy2, retain_graph=True)
            
            # Grads w.r.t params_G are accumulated.
            # Grads w.r.t y1 are in y1_detached.grad
            dy1_total = dy1 + y1_detached.grad
            
            # Recompute F
            x2_detached = x2.detach()
            x2_detached.requires_grad = True
            f_out = f_module(x2_detached, cos_sin, kv_cache)
            
            f_out.backward(dy1_total, retain_graph=True)
            
            # Grads w.r.t params_F accumulated.
            dx2_total = dy2 + x2_detached.grad # (dy2 comes from identity path x2->y2)
            dx1_total = dy1_total # x1 -> y1 is identity
            
        return torch.cat([dx1_total, dx2_total], dim=-1), None, None, None, None

