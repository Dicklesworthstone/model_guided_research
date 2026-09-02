"""
Ordinal Scheduler (PyTorch)
Implements transfinite learning rate scheduling based on ordinal ranking.
Rank rho = omega^2 * A + omega * B + C
A: Restart budget (highest order)
B: Anneal levels / curriculum
C: Patience (steps)

Transitions (mirroring JAX ordinal logic):
- Step: Update EMA loss.
  - If improved: C is kept (patience extends while the EMA keeps improving).
  - Else: C -> C-1.
- Limit (C=0):
  - Anneal (B>0): B->B-1, scale->scale*gamma, C->P.
  - Restart (B=0, A>0): A->A-1, B->B_init, scale->1, C->P, optimizer state cleared.

The schedule is a multiplicative SCALE applied to each param group's own
configured learning rate (``initial_lr`` when the optimizer factory recorded
one, else the group's lr at construction). Earlier versions overwrote every
group with a single ``eta_init``, which silently collapsed the embedding /
lm_head / matrix LR split that setup_optimizers builds (a 1/sqrt(d_model)
scaled 0.2 / 0.004 / 0.02 structure) into one flat value - so an
``--scheduler-type ordinal`` arm trained a different optimizer configuration
from the ``none`` arm before the schedule ever fired.
"""

import torch


class OrdinalLRScheduler:
    def __init__(self, optimizer, A_init=2, B_init=3, P_init=100, gamma=0.3, min_lr=1e-6):
        self.optimizer = optimizer
        if A_init < 0 or B_init < 0 or P_init < 1:
            raise ValueError("A_init and B_init must be >= 0 and P_init must be >= 1")
        if gamma <= 0 or min_lr <= 0:
            raise ValueError("gamma and min_lr must be positive")
        self.A = A_init
        self.B_init = B_init
        self.B = B_init
        self.P_init = P_init
        self.C = P_init
        self.gamma = gamma
        self.min_lr = min_lr

        self.best_loss = float("inf")
        self.ema_loss = None
        self.alpha = 0.1  # EMA smoothing factor

        # Per-group base LRs; the schedule only ever moves the shared scale.
        self.base_lrs = [float(group.get("initial_lr", group["lr"])) for group in self.optimizer.param_groups]
        self.scale = 1.0
        self._apply_lr()

    def _apply_lr(self) -> None:
        for group, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = max(self.min_lr, base * self.scale)

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
            # JAX logic: "If improved: keep (A,B,C)" - C is not decremented, so
            # patience extends indefinitely while the EMA keeps improving.
        else:
            # No improvement
            self.C -= 1

        # Check Limit Conditions
        if self.C <= 0:
            # Limit reached
            if self.B > 0:
                # Anneal (omega-term drop)
                self.B -= 1
                self.C = self.P_init  # Reset patience
                self.scale *= self.gamma
                self._apply_lr()
                # Reset best loss to allow new exploration (JAX: "reset best metric")
                self.best_loss = float("inf")

            elif self.A > 0:
                # Restart (omega^2-term drop)
                self.A -= 1
                self.B = self.B_init
                self.C = self.P_init
                self.scale = 1.0
                self._apply_lr()
                # Reset optimizer state
                self.optimizer.state.clear()

                self.best_loss = float("inf")

            else:
                # Terminate or plateau
                pass

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        """Mutable scheduler state for checkpoint/resume (bead rz8.1).

        Constructor hyperparameters (B_init/P_init/gamma/min_lr/alpha) and the
        per-group base LRs are included too: a resumed run must reproduce the
        limit transitions of the original run even if the resume command line
        drifts. Per-param-group LRs also live in the OPTIMIZER state_dict;
        load_state_dict re-derives them from base_lrs * scale so the two
        sources cannot disagree.
        """
        return {
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "B_init": self.B_init,
            "P_init": self.P_init,
            "gamma": self.gamma,
            "min_lr": self.min_lr,
            "best_loss": self.best_loss,
            "ema_loss": self.ema_loss,
            "alpha": self.alpha,
            "base_lrs": list(self.base_lrs),
            "scale": self.scale,
        }

    def load_state_dict(self, state: dict) -> None:
        keys = ("A", "B", "C", "B_init", "P_init", "gamma", "min_lr", "best_loss", "ema_loss", "alpha", "base_lrs", "scale")
        for key in keys:
            if key not in state:
                raise KeyError(f"OrdinalLRScheduler.load_state_dict missing key {key!r}")
        if len(state["base_lrs"]) != len(self.optimizer.param_groups):
            raise ValueError(
                f"OrdinalLRScheduler.load_state_dict: checkpoint has {len(state['base_lrs'])} param-group base LRs, "
                f"optimizer has {len(self.optimizer.param_groups)} groups"
            )
        for key in keys:
            setattr(self, key, state[key])
        self.base_lrs = [float(b) for b in state["base_lrs"]]
        self._apply_lr()
