# nanochat/synaptic_splitmerge.py
# Split/Merge controller for Synaptic MoE layers
#
# This controller performs:
#   • MERGE: pick expert pairs with high cosine similarity in the router-embedding
#            space AND low health; merge loser into winner (weighted average),
#            then CLONE the winner (with small noise) back into loser slot.
#   • SPLIT: clone strong experts into the weakest slots (optional).
#   • Keeps expert count constant; updates router columns, embeddings, synaptic state,
#     zeroes optimizer moments for changed parameters, and can broadcast in DDP.
#
# Works with:
#   - SynapticMoE, SynapticExpert, SynapticLinear, PostsynapticHebb from nanochat/synaptic.py
#   - GPTSynaptic from nanochat/gpt_synaptic.py
#
# Usage:
#   ctrl = SplitMergeController(model, SplitMergeConfig(...))
#   ctrl.step(global_step, optimizer=opt)    # call periodically (e.g. every 50k steps)

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import torch.distributed as torch_dist

from nanochat.torch_imports import Tensor, nn, torch

dist = cast(Any, torch_dist)

from .synaptic import SynapticExpert, SynapticLinear, SynapticMoE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SplitMergeConfig:
    enabled: bool = True
    # MERGE criteria
    merge_cosine_threshold: float = 0.85  # router-embedding cosine sim threshold
    merge_health_max: float = 0.25  # both experts must be below this health to be merge-candidates
    merges_per_call: int = 1  # max merges per step
    # SPLIT criteria
    split_health_min: float = 0.80  # expert must be above this health to be split candidate
    splits_per_call: int = 1
    # Noise scales for cloned experts
    clone_noise_linear: float = 0.02  # noise scale for linear weights
    clone_noise_router: float = 0.01  # noise scale for router columns
    clone_noise_embed: float = 0.05  # noise scale (tangent) for router embedding
    # Scheduling
    min_step_interval: int = 10_000  # don't do anything more frequently than this
    warmup_steps: int = 20_000  # no changes until after warmup
    # DDP
    ddp_broadcast: bool = True  # broadcast parameters from rank 0 after changes
    # Expert weighting
    use_util_weighting: bool = True  # weight merge by winner/loser utilization (via fatigue proxy)
    # Logging
    verbose: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_rank0() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def _world_size() -> int:
    return 1 if (not dist.is_available()) or (not dist.is_initialized()) else dist.get_world_size()


@torch.no_grad()
def _cosine(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return a @ b.T


@torch.no_grad()
def _orthogonal_perturb_like(vec: Tensor, noise_scale: float) -> Tensor:
    """Return a unit-length vector: normalized(vec + noise in orthogonal subspace)."""
    vec.shape[-1]
    noise = torch.randn_like(vec)
    proj = (noise * vec).sum(dim=-1, keepdim=True) * vec
    tangent = noise - proj
    out = vec + noise_scale * tangent
    return out / (out.norm(dim=-1, keepdim=True) + 1e-8)


@torch.no_grad()
def _add_noise_(t: Tensor, scale: float):
    if scale <= 0:
        return
    t.add_(torch.randn_like(t) * scale)


@torch.no_grad()
def _zero_optim_moments_for(optimizer: torch.optim.Optimizer | None, params: Iterable[nn.Parameter]):
    if optimizer is None:
        return
    pset = set(params)
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p in pset:
                state = optimizer.state.get(p, None)
                if state:
                    for k in list(state.keys()):  # momentum, exp_avg, etc.
                        state[k].zero_() if torch.is_tensor(state[k]) else None


@torch.no_grad()
def _broadcast_module_params(module: nn.Module):
    if not dist.is_available() or not dist.is_initialized() or _world_size() == 1:
        return
    for t in module.state_dict().values():
        if torch.is_tensor(t):
            dist.broadcast(t, src=0)


# ---------------------------------------------------------------------------
# Parameter copy helpers (SynapticLinear & expert)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _copy_synaptic_linear_(dst: SynapticLinear, src: SynapticLinear):
    # weights & bias
    dst.w_slow.copy_(src.w_slow)
    dst.w_fast.copy_(src.w_fast)
    if (dst.bias is not None) and (src.bias is not None):
        cast(Tensor, dst.bias).copy_(cast(Tensor, src.bias))
    # postsyn state
    cast(Tensor, dst.post.U).copy_(cast(Tensor, src.post.U))
    cast(Tensor, dst.post.V).copy_(cast(Tensor, src.post.V))
    cast(Tensor, dst.post.H_fast).copy_(cast(Tensor, src.post.H_fast))
    cast(Tensor, dst.post.m_gate).copy_(cast(Tensor, src.post.m_gate))
    cast(Tensor, dst.post.camkii).copy_(cast(Tensor, src.post.camkii))
    cast(Tensor, dst.post.pp1).copy_(cast(Tensor, src.post.pp1))


@torch.no_grad()
def _merge_linear_into_(winner: SynapticLinear, loser: SynapticLinear, alpha: float, cfg: SplitMergeConfig):
    """winner = alpha * winner + (1-alpha) * loser; loser = winner + noise"""
    if cfg.enabled and winner.w_slow.is_cuda:  # Check if we can use fused kernel
        try:
            from nanochat.kernels import mix_and_shift_tensors

            # Weights
            mix_and_shift_tensors(winner.w_slow, loser.w_slow, alpha, cfg.clone_noise_linear)
            mix_and_shift_tensors(winner.w_fast, loser.w_fast, alpha, cfg.clone_noise_linear)
            if (winner.bias is not None) and (loser.bias is not None):
                mix_and_shift_tensors(
                    cast(Tensor, winner.bias), cast(Tensor, loser.bias), alpha, cfg.clone_noise_linear
                )

            # Postsynaptic state
            # For state, we might want less noise or different logic?
            # Original logic:
            # winner = alpha*winner + (1-alpha)*loser
            # loser = winner + noise (clone)
            # BUT original _clone_linear_from_ did:
            # dst.post.H_fast.zero_()
            # dst.post.U.mul_(0.5)
            # So simple mix_and_shift is NOT correct for state if we want to reset/dampen loser.

            # Let's stick to manual for state to preserve logic, or adapt kernel.
            # The kernel does: t2 = t1 + noise.
            # We want t2 = 0 or t2 = t1 * 0.5.

            # So we only use fused kernel for weights/biases which are the big tensors.
            # State tensors are small (rank R).

            # Manual state update (same as before)
            cast(Tensor, winner.post.U).mul_(alpha).add_((1.0 - alpha) * cast(Tensor, loser.post.U))
            cast(Tensor, winner.post.V).mul_(alpha).add_((1.0 - alpha) * cast(Tensor, loser.post.V))
            cast(Tensor, winner.post.H_fast).mul_(alpha).add_((1.0 - alpha) * cast(Tensor, loser.post.H_fast))
            cast(Tensor, winner.post.m_gate).mul_(0.9).add_(0.1 * cast(Tensor, loser.post.m_gate))
            cast(Tensor, winner.post.camkii).mul_(0.9).add_(0.1 * cast(Tensor, loser.post.camkii))
            cast(Tensor, winner.post.pp1).mul_(0.9).add_(0.1 * cast(Tensor, loser.post.pp1))

            # Clone state to loser (with reset logic)
            cast(Tensor, loser.post.U).copy_(cast(Tensor, winner.post.U)).mul_(0.5)
            cast(Tensor, loser.post.V).copy_(cast(Tensor, winner.post.V)).mul_(0.5)
            cast(Tensor, loser.post.H_fast).zero_()
            cast(Tensor, loser.post.m_gate).copy_(cast(Tensor, winner.post.m_gate))
            cast(Tensor, loser.post.camkii).copy_(cast(Tensor, winner.post.camkii))
            cast(Tensor, loser.post.pp1).copy_(cast(Tensor, winner.post.pp1))

            return
        except ImportError:
            pass

    # Fallback / CPU logic
    # winner = alpha * winner + (1-alpha) * loser
    winner.w_slow.mul_(alpha).add_((1.0 - alpha) * loser.w_slow)
    winner.w_fast.mul_(alpha).add_((1.0 - alpha) * loser.w_fast)
    if (winner.bias is not None) and (loser.bias is not None):
        cast(Tensor, winner.bias).mul_(alpha).add_((1.0 - alpha) * cast(Tensor, loser.bias))
    cast(Tensor, winner.post.U).mul_(alpha).add_((1.0 - alpha) * cast(Tensor, loser.post.U))
    cast(Tensor, winner.post.V).mul_(alpha).add_((1.0 - alpha) * cast(Tensor, loser.post.V))
    cast(Tensor, winner.post.H_fast).mul_(alpha).add_((1.0 - alpha) * cast(Tensor, loser.post.H_fast))
    # gate and enzymes: bias toward winner (more stable)
    cast(Tensor, winner.post.m_gate).mul_(0.9).add_(0.1 * cast(Tensor, loser.post.m_gate))
    cast(Tensor, winner.post.camkii).mul_(0.9).add_(0.1 * cast(Tensor, loser.post.camkii))
    cast(Tensor, winner.post.pp1).mul_(0.9).add_(0.1 * cast(Tensor, loser.post.pp1))

    # Clone back into loser (to keep count constant)
    _clone_linear_from_(loser, winner, cfg.clone_noise_linear)


@torch.no_grad()
def _clone_linear_from_(dst: SynapticLinear, src: SynapticLinear, noise_scale: float):
    _copy_synaptic_linear_(dst, src)
    _add_noise_(dst.w_slow, noise_scale)
    _add_noise_(dst.w_fast, noise_scale)
    if dst.bias is not None:
        _add_noise_(cast(Tensor, dst.bias), noise_scale)
    # reset fast Hebbian traces for cloned expert
    cast(Tensor, dst.post.H_fast).zero_()
    cast(Tensor, dst.post.U).mul_(0.5)
    cast(Tensor, dst.post.V).mul_(0.5)  # keep some eligibility but dampen


@torch.no_grad()
def _merge_expert_into_and_clone_(
    layer: SynapticMoE,
    winner_idx: int,
    loser_idx: int,
    alpha: float,
    cfg: SplitMergeConfig,
):
    """Merge loser into winner (weighted), then clone winner (+noise) into loser slot."""
    winner: SynapticExpert = layer.experts[winner_idx]
    loser: SynapticExpert = layer.experts[loser_idx]

    # 1) Merge parameters into winner AND clone to loser
    # We combined these steps in _merge_linear_into_ if fused
    _merge_linear_into_(winner.fc1, loser.fc1, alpha, cfg)
    _merge_linear_into_(winner.fc2, loser.fc2, alpha, cfg)

    # 3) Router columns: average into winner, clone into loser (with noise)
    W = layer.router.weight
    if W.is_cuda:
        try:
            from nanochat.kernels import mix_and_shift_rows

            # W is (E, n_embd). We want to mix row[winner] and row[loser].
            mix_and_shift_rows(W, winner_idx, loser_idx, alpha, cfg.clone_noise_router)
        except ImportError:
            # Fallback
            W_w = W[winner_idx]
            W_l = W[loser_idx]
            W_w.mul_(alpha).add_((1.0 - alpha) * W_l)
            W_l.copy_(W_w)
            _add_noise_(W_l, cfg.clone_noise_router)
    else:
        W_w = W[winner_idx]
        W_l = W[loser_idx]
        W_w.mul_(alpha).add_((1.0 - alpha) * W_l)
        W_l.copy_(W_w)
        _add_noise_(W_l, cfg.clone_noise_router)

    # 4) Router embeddings: keep winner embedding; clone loser as orthogonalized perturbed winner
    # This logic is specific (orthogonal perturb), so we keep it manual for now or write another kernel.
    # Since it's just 1 vector per expert, manual is fine.
    emb = layer.router_embeddings  # (E, D)
    e_w = emb[winner_idx : winner_idx + 1]  # (1,D)
    e_l = _orthogonal_perturb_like(e_w.clone(), cfg.clone_noise_embed)
    emb[loser_idx : loser_idx + 1].copy_(e_l)

    # 5) Reset stats
    fatigue = cast(Tensor, layer.fatigue)
    energy = cast(Tensor, layer.energy)
    fatigue[loser_idx] = 0.0
    energy[loser_idx] = 1.0


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class SplitMergeController:
    def __init__(self, model: nn.Module, cfg: SplitMergeConfig, logger: Any | None = None):
        self.model = model
        self.cfg = cfg
        self._last_step = -(10**12)  # ensure first call can run if warmup permits
        self._moe_layers: list[SynapticMoE] = self._find_moe_layers(model)
        self.logger = logger

    def _find_moe_layers(self, module: nn.Module) -> list[SynapticMoE]:
        moes: list[SynapticMoE] = []
        for m in module.modules():
            if isinstance(m, SynapticMoE):
                moes.append(m)
        return moes

    @torch.no_grad()
    def _health(self, layer: SynapticMoE) -> Tensor:
        # Higher is better: combine (1 - fatigue) with energy in [0,1]
        fat = cast(Tensor, layer.fatigue).clamp(0, 1)  # EMA utilization proxy
        eng = cast(Tensor, layer.energy).clamp(0, 1)
        health = (1.0 - fat) * (0.5 + 0.5 * eng)  # [0,1]
        return health

    @torch.no_grad()
    def _util_weight(self, layer: SynapticMoE, i: int, j: int) -> float:
        if not self.cfg.use_util_weighting:
            return 0.6  # mild bias toward first arg
        # invert fatigue → utilization proxy
        fat = cast(Tensor, layer.fatigue)
        u_i = (1.0 - fat[i]).clamp(0, 1)
        u_j = (1.0 - fat[j]).clamp(0, 1)
        s = (u_i + u_j).clamp_min(1e-6)
        return float((u_i / s).item())

    @torch.no_grad()
    def _pick_merge_pairs(self, layer: SynapticMoE) -> list[tuple[int, int]]:
        E = layer.num_experts
        emb = layer.router_embeddings  # (E, D)
        sim = _cosine(emb, emb)  # (E,E)
        health = self._health(layer)
        # candidate mask: high sim and both low health
        sim_mask = sim > self.cfg.merge_cosine_threshold
        low = health <= self.cfg.merge_health_max
        cand = sim_mask & low.unsqueeze(1) & low.unsqueeze(0)
        # remove diagonal
        idx = torch.arange(E, device=emb.device)
        cand[idx, idx] = False
        # score by similarity (higher first)
        scores = sim.masked_fill(~cand, -1.0)  # -1 for invalid
        pairs: list[tuple[int, int]] = []
        used = set()
        for _ in range(self.cfg.merges_per_call):
            # find max entry
            val, linear_idx = scores.view(-1).max(dim=0)
            if val <= 0:
                break
            i = (linear_idx // E).item()
            j = (linear_idx % E).item()
            if i in used or j in used:
                scores[i, :] = -1.0
                scores[:, i] = -1.0
                scores[j, :] = -1.0
                scores[:, j] = -1.0
                continue
            pairs.append((i, j))
            used.add(i)
            used.add(j)
            # invalidate rows/cols
            scores[i, :] = -1.0
            scores[:, i] = -1.0
            scores[j, :] = -1.0
            scores[:, j] = -1.0
        return pairs

    @torch.no_grad()
    def _pick_split_sources(self, layer: SynapticMoE) -> list[int]:
        health = self._health(layer)
        strong = (health >= self.cfg.split_health_min).nonzero(as_tuple=False).flatten().tolist()
        # take top k strongest
        strong_sorted = sorted(strong, key=lambda e: float(health[e].item()), reverse=True)
        return strong_sorted[: self.cfg.splits_per_call]

    @torch.no_grad()
    def _weakest_slots(self, layer: SynapticMoE, k: int) -> list[int]:
        health = self._health(layer)
        idx = torch.argsort(health)  # ascending
        return idx[:k].tolist()

    @torch.no_grad()
    def _split_into_slots(
        self,
        layer: SynapticMoE,
        sources: list[int],
        slots: list[int],
        optimizer: torch.optim.Optimizer | None,
        step: int,
    ):
        for src, dst in zip(sources, slots):
            if src == dst:
                continue
            # Clone src → dst, with noise & embedding tweak
            _clone_linear_from_(
                layer.experts[dst].fc1,
                layer.experts[src].fc1,
                self.cfg.clone_noise_linear,
            )
            _clone_linear_from_(
                layer.experts[dst].fc2,
                layer.experts[src].fc2,
                self.cfg.clone_noise_linear,
            )
            # router weight row (expert row)
            W = layer.router.weight
            W[dst].copy_(W[src])
            _add_noise_(W[dst], self.cfg.clone_noise_router)
            # embedding
            layer.router_embeddings[dst : dst + 1].copy_(
                _orthogonal_perturb_like(
                    layer.router_embeddings[src : src + 1].clone(),
                    self.cfg.clone_noise_embed,
                )
            )
            # reset stats
            fatigue = cast(Tensor, layer.fatigue)
            energy = cast(Tensor, layer.energy)
            fatigue[dst] = 0.0
            energy[dst] = 1.0
            # emit lineage event: split parent src -> child dst
            if self.logger is not None and hasattr(self.logger, "on_split"):
                try:
                    self.logger.on_split(layer, parent_idx=int(src), child_idx=int(dst), step=step)
                except Exception as _e:
                    if self.cfg.verbose:
                        print(f"[SplitMerge] logger.on_split failed: {_e}")
            # zero optimizer moments
            if optimizer is not None:
                changed = [
                    layer.experts[dst].fc1.w_slow,
                    layer.experts[dst].fc1.w_fast,
                    layer.experts[dst].fc2.w_slow,
                    layer.experts[dst].fc2.w_fast,
                    W,
                ]
                if layer.experts[dst].fc1.bias is not None:
                    changed.append(layer.experts[dst].fc1.bias)
                if layer.experts[dst].fc2.bias is not None:
                    changed.append(layer.experts[dst].fc2.bias)
                changed_params: list[nn.Parameter] = [param for param in changed if isinstance(param, nn.Parameter)]
                _zero_optim_moments_for(optimizer, changed_params)

    @torch.no_grad()
    def _do_merges(self, layer: SynapticMoE, optimizer: torch.optim.Optimizer | None, step: int):
        pairs = self._pick_merge_pairs(layer)
        if self.cfg.verbose and len(pairs) > 0:
            print(f"[SplitMerge] Merging pairs: {pairs}")
        for i, j in pairs:
            # winner = healthier of the two
            health = self._health(layer)
            if health[i] >= health[j]:
                winner, loser = i, j
            else:
                winner, loser = j, i
            alpha = self._util_weight(layer, winner, loser)
            _merge_expert_into_and_clone_(layer, winner, loser, alpha, self.cfg)
            # emit lineage event: merge parents (winner,loser) -> child lives at index loser (clone slot reused)
            if self.logger is not None and hasattr(self.logger, "on_merge"):
                try:
                    self.logger.on_merge(
                        layer,
                        parent_i=int(winner),
                        parent_j=int(loser),
                        child_idx=int(loser),
                        step=step,
                    )
                except Exception as _e:
                    if self.cfg.verbose:
                        print(f"[SplitMerge] logger.on_merge failed: {_e}")
            # zero optimizer moments for both experts + router rows
            if optimizer is not None:
                changed = [
                    layer.experts[winner].fc1.w_slow,
                    layer.experts[winner].fc1.w_fast,
                    layer.experts[winner].fc2.w_slow,
                    layer.experts[winner].fc2.w_fast,
                    layer.experts[loser].fc1.w_slow,
                    layer.experts[loser].fc1.w_fast,
                    layer.experts[loser].fc2.w_slow,
                    layer.experts[loser].fc2.w_fast,
                    layer.router.weight,
                ]
                if layer.experts[winner].fc1.bias is not None:
                    changed.append(layer.experts[winner].fc1.bias)
                if layer.experts[winner].fc2.bias is not None:
                    changed.append(layer.experts[winner].fc2.bias)
                if layer.experts[loser].fc1.bias is not None:
                    changed.append(layer.experts[loser].fc1.bias)
                if layer.experts[loser].fc2.bias is not None:
                    changed.append(layer.experts[loser].fc2.bias)
                changed_params = [param for param in changed if isinstance(param, nn.Parameter)]
                _zero_optim_moments_for(optimizer, changed_params)

    @torch.no_grad()
    def step(self, global_step: int, optimizer: torch.optim.Optimizer | None = None):
        if not self.cfg.enabled:
            return
        if global_step < self.cfg.warmup_steps:
            return
        if global_step - self._last_step < self.cfg.min_step_interval:
            return

        if not _is_rank0():
            # Non-zero ranks just wait for broadcast after rank 0 modifies params
            if self.cfg.ddp_broadcast:  # ensure we hit the barrier roughly in sync
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
            return

        if self.cfg.verbose:
            print(f"[SplitMerge] step @ {global_step}")

        # Perform operations layer-by-layer on rank 0
        for layer in self._moe_layers:
            # 1) merges
            self._do_merges(layer, optimizer, global_step)
            # 2) splits
            sources = self._pick_split_sources(layer)
            if len(sources) > 0 and self.cfg.splits_per_call > 0:
                slots = self._weakest_slots(layer, min(len(sources), self.cfg.splits_per_call))
                if self.cfg.verbose:
                    print(f"[SplitMerge] Splitting {list(zip(sources, slots))}")
                self._split_into_slots(layer, sources, slots, optimizer, global_step)

        # Broadcast updated params to all ranks (DDP)
        if self.cfg.ddp_broadcast and dist.is_available() and dist.is_initialized():
            for layer in self._moe_layers:
                _broadcast_module_params(layer)
            dist.barrier()

        self._last_step = global_step
