"""
Ordinal Scheduler (PyTorch)
Implements transfinite learning rate scheduling based on ordinal ranking.
Rank rho = omega^2 * A + omega * B + C
A: Restart budget
B: Anneal levels
C: Patience (steps)

Transitions:
- Step: C -> C-1 (if loss improves, C resets to P(B))
- Anneal (Limit of C): B -> B-1, lr -> lr * gamma, C -> P(B)
- Restart (Limit of B): A -> A-1, B -> B_init, lr -> lr_init, C -> P(B_init)
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler

class OrdinalLRScheduler:
    def __init__(self, optimizer, A_init=2, B_init=3, P_init=100, eta_init=1e-3, gamma=0.3, min_lr=1e-6):
        self.optimizer = optimizer
        self.A = A_init
        self.B_init = B_init
        self.B = B_init
        self.P_init = P_init
        self.C = P_init
        self.eta_init = eta_init
        self.gamma = gamma
        self.min_lr = min_lr
        
        self.best_loss = float('inf')
        self.ema_loss = None
        self.alpha = 0.1 # EMA smoothing factor
        
        # Set initial LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_init
            
    def step(self, loss):
        # Update EMA loss
        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = (1 - self.alpha) * self.ema_loss + self.alpha * loss
            
        # Check for improvement
        if self.ema_loss < self.best_loss:
            self.best_loss = self.ema_loss
            # Reset patience for current level B
            # Logic: If we improve, we stay in current B but refresh C?
            # Or does C always tick down?
            # JAX code: "If improved: keep (A,B,C). Else: C := max(C-1, 0)."
            # Wait, if improved, we DON'T decrement C.
            # This effectively resets patience if we keep improving.
            # But we don't reset C to max. We just don't decrement.
            # Actually, usually patience resets to max on improvement.
            # Let's follow JAX doc: "If improved: keep (A,B,C)". So C stays constant.
            pass
        else:
            # No improvement
            self.C -= 1
            
        # Check Limit Conditions
        if self.C <= 0:
            # Limit reached
            if self.B > 0:
                # Anneal
                self.B -= 1
                self.C = self.P_init # Reset patience
                # Decay LR
                new_lr = 0.0
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = max(self.min_lr, param_group['lr'] * self.gamma)
                    new_lr = param_group['lr']
                # Reset best loss to allow new exploration?
                # JAX: "reset best metric"
                self.best_loss = float('inf') 
                # print(f"Ordinal Anneal: B={self.B}, lr={new_lr:.2e}")
                
            elif self.A > 0:
                # Restart
                self.A -= 1
                self.B = self.B_init
                self.C = self.P_init
                # Reset LR to init
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.eta_init
                # Reset optimizer state?
                # "reset optimizer state"
                self.optimizer.state.clear()
                
                self.best_loss = float('inf')
                # print(f"Ordinal Restart: A={self.A}, lr={self.eta_init:.2e}")
            
            else:
                # Terminate? Or just stay at min LR?
                # Default: Just stay.
                pass

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
