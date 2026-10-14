"""
Ordinal Scheduler (PyTorch)
Implements transfinite learning rate scheduling based on ordinal ranking in omega^4:
Rank Phi(t) = omega^3 * A_t + omega^2 * B_t + omega^1 * (T_horizon - t_record(t)) + C_t
A: Restart budget (omega^3 level)
B: Anneal levels / curriculum (omega^2 level)
R = max(0, T_horizon - t_record): Remaining record-step horizon (omega^1 level)
C: Patience steps remaining (omega^0 level)

Transitions:
- Step t: Update EMA loss.
  - If improved (ema_loss < best_loss - min_delta): t_record <- t (strictly decreasing R = T_horizon - t_record
    at the omega^1 level), and reset C <- P_init (dominated by the omega^1 decrease, so Phi(t) < Phi(t-1)).
  - Else: C -> C - 1 (strictly decreasing at the omega^0 level).
- Limit (C == 0):
  - Anneal (B > 0): B -> B - 1 (omega^2 decrease), scale -> scale * gamma, C -> P_init.
  - Restart (B == 0, A > 0): A -> A - 1 (omega^3 decrease), B -> B_init, scale -> 1, C -> P_init,
    zero optimizer first-moment/momentum buffers while preserving AdamW second-moment preconditioning.

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
    def __init__(
        self,
        optimizer,
        A_init=2,
        B_init=3,
        P_init=100,
        gamma=0.3,
        min_lr=1e-6,
        max_steps=1_000_000,
        min_delta=0.0,
        reset_mode="momentum",
    ):
        self.optimizer = optimizer
        if A_init < 0 or B_init < 0 or P_init < 1:
            raise ValueError("A_init and B_init must be >= 0 and P_init must be >= 1")
        if gamma <= 0 or min_lr <= 0:
            raise ValueError("gamma and min_lr must be positive")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if min_delta < 0:
            raise ValueError("min_delta must be >= 0")
        if reset_mode not in ("momentum", "clear"):
            raise ValueError(f"reset_mode must be 'momentum' or 'clear', got {reset_mode!r}")
        self.A = A_init
        self.B_init = B_init
        self.B = B_init
        self.P_init = P_init
        self.C = P_init
        self.gamma = gamma
        self.min_lr = min_lr
        self.max_steps = int(max_steps)
        self.min_delta = float(min_delta)
        self.reset_mode = str(reset_mode)

        self.step_count = 0
        self.last_record_step = 0
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

    def _reset_optimizer_momentum(self) -> None:
        """Reset first-moment/momentum buffers on restart while preserving second-moment variance."""
        if self.reset_mode == "clear":
            self.optimizer.state.clear()
            return
        for param_state in self.optimizer.state.values():
            if not isinstance(param_state, dict):
                continue
            for key in ("exp_avg", "momentum_buffer"):
                buf = param_state.get(key)
                if torch.is_tensor(buf):
                    buf.zero_()

    def ordinal_rank_tuple(self) -> tuple[int, int, int, int]:
        """Cantor Normal Form coefficients (A, B, R, C) for rank omega^3*A + omega^2*B + omega*R + C."""
        rem_horizon = max(0, self.max_steps - self.last_record_step)
        return (int(self.A), int(self.B), int(rem_horizon), int(self.C))

    def step(self, loss):
        if torch.is_tensor(loss):
            loss = float(loss.detach().item())
        self.step_count += 1
        # Update EMA loss
        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = (1 - self.alpha) * self.ema_loss + self.alpha * loss

        # Check for improvement (new loss record)
        if self.ema_loss < self.best_loss - self.min_delta:
            self.best_loss = self.ema_loss
            self.last_record_step = self.step_count
            # Reset patience C to P_init: because last_record_step strictly increased,
            # R = max_steps - last_record_step strictly decreased at the omega^1 level,
            # dominating the omega^0 reset of C.
            self.C = self.P_init
        else:
            # No improvement
            self.C -= 1

        # Check Limit Conditions
        if self.C <= 0:
            # Limit reached
            if self.B > 0:
                # Anneal (omega^2-term drop)
                self.B -= 1
                self.C = self.P_init  # Reset patience
                self.scale *= self.gamma
                self._apply_lr()
                # Reset best loss to allow new exploration
                self.best_loss = float("inf")
                self.last_record_step = self.step_count

            elif self.A > 0:
                # Restart (omega^3-term drop)
                self.A -= 1
                self.B = self.B_init
                self.C = self.P_init
                self.scale = 1.0
                self._apply_lr()
                self._reset_optimizer_momentum()

                self.best_loss = float("inf")
                self.last_record_step = self.step_count

            else:
                # Terminal plateau
                self.C = 0

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
            "step_count": self.step_count,
            "last_record_step": self.last_record_step,
            "max_steps": self.max_steps,
            "min_delta": self.min_delta,
            "reset_mode": self.reset_mode,
        }

    def load_state_dict(self, state: dict) -> None:
        keys = (
            "A",
            "B",
            "C",
            "B_init",
            "P_init",
            "gamma",
            "min_lr",
            "best_loss",
            "ema_loss",
            "alpha",
            "base_lrs",
            "scale",
        )
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
        self.step_count = int(state.get("step_count", 0))
        self.last_record_step = int(state.get("last_record_step", 0))
        self.max_steps = int(state.get("max_steps", self.max_steps))
        self.min_delta = float(state.get("min_delta", self.min_delta))
        self.reset_mode = str(state.get("reset_mode", self.reset_mode))
        self.base_lrs = [float(b) for b in state["base_lrs"]]
        self._apply_lr()

