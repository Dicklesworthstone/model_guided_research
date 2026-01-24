"""
Ordinal Scheduler (PyTorch)
Implements transfinite learning rate scheduling based on ordinal ranking.
Rank rho = omega^2 * A + omega * B + C
A: Restart budget (highest order)
B: Anneal levels / curriculum
C: Patience (steps)

Transitions (mirroring JAX ordinal logic):
- Step: Update EMA loss.
  - If improved: C is reset (or kept, per policy).
  - Else: C -> C-1.
- Limit (C=0):
  - Anneal (B>0): B->B-1, lr->lr*gamma, C->P(B).
  - Restart (B=0, A>0): A->A-1, B->B_init, lr->lr_init, C->P(B_init).
"""

import torch


class OrdinalLRScheduler:
    def __init__(self, optimizer, A_init=2, B_init=3, P_init=100, eta_init=1e-3, gamma=0.3, min_lr=1e-6):
        self.optimizer = optimizer
        if A_init < 0 or B_init < 0 or P_init < 1:
            raise ValueError("A_init and B_init must be >= 0 and P_init must be >= 1")
        if eta_init <= 0 or gamma <= 0 or min_lr <= 0:
            raise ValueError("eta_init, gamma, and min_lr must be positive")
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
        if torch.is_tensor(loss):
            loss = float(loss.detach().item())
        # Update EMA loss
        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = (1 - self.alpha) * self.ema_loss + self.alpha * loss

        # Check for improvement
        if self.ema_loss < self.best_loss:
            self.best_loss = self.ema_loss
            # JAX logic: "If improved: keep (A,B,C)".
            # This means we DON'T decrement C.
            # It effectively extends patience indefinitely as long as we improve.
            pass
        else:
            # No improvement
            self.C -= 1

        # Check Limit Conditions
        if self.C <= 0:
            # Limit reached
            if self.B > 0:
                # Anneal (omega-term drop)
                self.B -= 1
                self.C = self.P_init # Reset patience
                # Decay LR
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = max(self.min_lr, param_group['lr'] * self.gamma)
                    param_group['lr']
                # Reset best loss to allow new exploration (JAX: "reset best metric")
                self.best_loss = float('inf')

            elif self.A > 0:
                # Restart (omega^2-term drop)
                self.A -= 1
                self.B = self.B_init
                self.C = self.P_init
                # Reset LR to init
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.eta_init
                # Reset optimizer state
                self.optimizer.state.clear()

                self.best_loss = float('inf')

            else:
                # Terminate or plateau
                pass

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
