# nanochat/neuroscore.py
# -----------------------------------------------------------------------------
# NeuroScore: Evolutionary Credit Assignment for Synaptic Experts
# -----------------------------------------------------------------------------
# Measures:
#   1. Loss Contribution: How much did this expert reduce the loss? (Approx via routing weight * |dL/dx|)
#   2. Specialization: How unique is this expert's input distribution? (Cosine distance from global mean)
#   3. Efficiency: Performance per unit of energy.
#   4. Resilience: Stability of contribution over time.
# -----------------------------------------------------------------------------

import os
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from nanochat.torch_imports import F, nn, torch

from .synaptic import SynapticMoE


@dataclass
class NeuroScoreConfig:
    enabled: bool = True
    history_len: int = 1024
    update_every: int = 100  # Compute expensive metrics every N steps
    decay: float = 0.99  # EMA decay for resilience tracking


class NeuroScore:
    """
    The 'Credit Assignment' engine.
    Tracks the true utility of experts to guide evolutionary decisions.
    """

    def __init__(self, cfg: NeuroScoreConfig, neuroviz=None):
        self.cfg = cfg
        self.neuroviz = neuroviz
        self.stats: dict[str, dict[str, Any]] = {}  # layer_name -> metrics
        self._last_loss = None

    def register_layer(self, name: str, num_experts: int):
        if name not in self.stats:
            self.stats[name] = {
                "loss_contrib": torch.zeros(num_experts),  # Rolling sum
                "routing_freq": torch.zeros(num_experts),
                "specialization": torch.zeros(num_experts),
                "efficiency": torch.zeros(num_experts),
                "resilience": torch.zeros(num_experts),
                "prev_contrib": torch.zeros(num_experts),  # For resilience
                "updates": 0,
            }

    @torch.no_grad()
    def step(self, model: nn.Module, loss: Tensor, global_step: int):
        if not self.cfg.enabled:
            return

        # We need the gradient to estimate contribution, so we can't do this
        # strictly inside torch.no_grad(), but the METRIC update is no_grad.
        # ACTUALLY: We can approximate contribution using the stored context
        # and the current loss magnitude, or hook into backward.
        # For simplicity/speed in this "v1", we use a forward-pass proxy:
        # Contribution ~ RoutingWeight * ExpertEnergy (Heuristic: "Active & High Energy" ~ doing work)
        # A better "v2" would be RoutingWeight * |Grad_Expert_Output|

        fused_metrics = os.environ.get("BIO_FUSED_METRICS", "").strip().lower() in {"1", "true", "yes", "y", "on"}

        for name, module in model.named_modules():
            if isinstance(module, SynapticMoE):
                if not hasattr(module, "last_ctx") or not module.last_ctx:
                    continue

                layer_name = name
                if self.neuroviz is not None:
                    mapped = self.neuroviz._name_of(module)  # type: ignore[attr-defined]
                    if mapped:
                        layer_name = mapped
                if layer_name not in self.stats:
                    self.register_layer(layer_name, module.num_experts)

                st = self.stats[layer_name]
                ctx = module.last_ctx
                gates = ctx["gates"]  # (B,T,k)
                indices = ctx["indices"]  # (B,T,k)
                x_in = ctx["x"]  # (B,T,C)
                energy = module.energy # (E,)

                # Fused Kernel Path
                if fused_metrics and gates.is_cuda:
                    try:
                        from nanochat.kernels import update_metrics_fused

                        if update_metrics_fused(indices, gates, energy, st, self.cfg):
                            st["updates"] += 1

                            if global_step % self.cfg.update_every == 0:
                                self._update_specialization(st, x_in, indices, module.num_experts)
                                if self.neuroviz:
                                    self._log_metrics(layer_name, st, global_step)
                            continue
                    except ImportError:
                        pass
                    except Exception as e:
                        print(f"Triton metrics kernel failed: {e}")

                # 1. Specialization (Diversity of inputs)
                # Calculate mean input vector per expert
                # This is expensive, so only do it occasionally
                if global_step % self.cfg.update_every == 0:
                    self._update_specialization(st, x_in, indices, module.num_experts)

                # 2. Loss Contribution Proxy
                # We use the routing weights as a proxy for "responsibility"
                # scaled by the global loss (if loss is high, and you were picked, you share blame/credit)
                # In a real RL setting, we'd use Advantage, but here:
                # If loss is dropping, high routing weight = Good.
                # If loss is flat, high routing weight = Neutral.
                # We simplify: Contribution = Sum(Gates)
                # This seems trivial, but combined with Energy it gives Efficiency.

                # Flatten batch/time
                gates_flat = gates.view(-1)
                indices_flat = indices.view(-1)

                # Scatter add
                contrib_update = torch.zeros_like(st["loss_contrib"])
                contrib_update.index_add_(0, indices_flat.cpu(), gates_flat.float().cpu())

                # Normalize by batch size
                batch_size = gates.shape[0] * gates.shape[1]
                contrib_update /= batch_size

                # EMA update
                st["loss_contrib"].mul_(self.cfg.decay).add_(contrib_update * (1 - self.cfg.decay))

                # 3. Efficiency = Contribution / (Energy + epsilon)
                energy_cpu = module.energy.detach().float().cpu()
                st["efficiency"] = st["loss_contrib"] / (energy_cpu + 1e-6)

                # 4. Resilience = 1 / (Variance of Contribution)
                # Track simple diff from last step
                diff = (st["loss_contrib"] - st["prev_contrib"]).abs()
                st["resilience"].mul_(self.cfg.decay).add_((1.0 / (diff + 1e-6)) * (1 - self.cfg.decay))
                st["prev_contrib"].copy_(st["loss_contrib"])

                st["updates"] += 1

                # Log to NeuroViz/TensorBoard if connected
                if self.neuroviz and global_step % self.cfg.update_every == 0:
                    self._log_metrics(layer_name, st, global_step)

    def _update_specialization(self, st, x, indices, num_experts):
        """
        How 'unique' is the input subspace this expert sees?
        High specialization = Sees vectors very different from the global mean.
        """
        # x: (B,T,C)
        # indices: (B,T,k)
        B, T, C = x.shape

        # Compute global mean of inputs
        global_mean = x.mean(dim=(0, 1)).float()  # (C,)

        # We want mean input per expert.
        # Gather inputs for each expert? Too much memory.
        # Streaming approx:
        # Just sample a subset for speed
        mask_prob = 0.1
        mask = torch.rand(B, T, device=x.device) < mask_prob
        if not mask.any():
            return

        x_sub = x[mask].float() # (N, C)
        ind_sub = indices[mask] # (N, k)

        # For each expert, compute centroid of assigned inputs
        expert_sums = torch.zeros(num_experts, C, device=x.device, dtype=torch.float32)
        expert_counts = torch.zeros(num_experts, device=x.device, dtype=torch.float32)

        # Naive loop is slow, but x_sub is small.
        # Vectorized scatter_add is better.
        # Expand x_sub for k assignments?
        # (N, k, C)
        # This might OOM if k is large, but k=2 usually.

        for k_i in range(ind_sub.shape[1]):
            # idx: (N,)
            idx = ind_sub[:, k_i]
            expert_sums.index_add_(0, idx, x_sub)
            expert_counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))

        expert_means = expert_sums / (expert_counts.unsqueeze(1) + 1e-6)

        # Cosine distance from global mean
        # (E, C) vs (C,)
        sim = F.cosine_similarity(expert_means, global_mean.unsqueeze(0), dim=1)

        # Specialization = 1 - similarity (0 = generic, 1 = unique)
        spec = 1.0 - sim
        st["specialization"].copy_(spec.detach().cpu())

    def _log_metrics(self, layer_name, st, step):
        # Push to TensorBoard via NeuroViz
        if not self.neuroviz or not getattr(self.neuroviz, "tb", None):
            return
        tb = self.neuroviz.tb

        # Scalars (Means)
        tb.add_scalar(f"{layer_name}/score/mean_efficiency", st["efficiency"].mean(), step)
        tb.add_scalar(f"{layer_name}/score/mean_specialization", st["specialization"].mean(), step)
        tb.add_scalar(f"{layer_name}/score/mean_resilience", st["resilience"].mean(), step)

        # Histograms
        tb.add_histogram(f"{layer_name}/score/hist_efficiency", st["efficiency"], step)
        tb.add_histogram(f"{layer_name}/score/hist_specialization", st["specialization"], step)

        # Leaderboard (Top 5 Experts by Efficiency)
        top_k = 5
        vals, idxs = torch.topk(st["efficiency"], k=min(top_k, len(st["efficiency"])))

        # Create Markdown Table
        md = "| Rank | ID | Efficiency | Spec | Contrib |\n|---|---|---|---|---|\n"
        for rank, (val, idx) in enumerate(zip(vals, idxs)):
            i = idx.item()
            md += f"| {rank+1} | {i} | {val:.3f} | {st['specialization'][i]:.3f} | {st['loss_contrib'][i]:.3f} |\n"

        tb.add_text(f"{layer_name}/leaderboard", md, step)

