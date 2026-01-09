# nanochat/neuroviz.py
# ---------------------------------------------------------------------
# Visualization & logging for Synaptic-MoE:
#  - NeuroVizManager: orchestrates TensorBoard + static figures + lineage
#  - LineageBook: split/merge event ledger + timeline renders
#  - Expert plotting: UMAP/PCA map, radar of top experts, histograms
#
# Dependencies:
#   - required: numpy, matplotlib, tensorboard
#   - optional: umap-learn (preferred), scikit-learn (PCA fallback), plotly
#
# Usage:
#   viz = NeuroVizManager(log_dir="runs/brain1", image_every=10000, tb_every=1000)
#   viz.register_model(model)                 # once, after model creation
#   sm_ctrl = SplitMergeController(model, cfg, logger=viz)  # to receive events
#   ...
#   for step in ...:
#       ...
#       viz.step(model, step)                 # per-step logging
#   viz.close()
# ---------------------------------------------------------------------

import importlib
import importlib.util
import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, cast

import numpy as np
from nanochat.torch_imports import torch, nn, Tensor
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _maybe_import(module: str, attr: Optional[str] = None) -> Any:
    spec = importlib.util.find_spec(module)
    if spec is None:
        return None
    mod = importlib.import_module(module)
    return getattr(mod, attr) if attr else mod


UMAP = _maybe_import("umap", "UMAP")
_HAS_UMAP = UMAP is not None
PCA = _maybe_import("sklearn.decomposition", "PCA")
_HAS_SKLEARN = PCA is not None
go = _maybe_import("plotly.graph_objects")
_HAS_PLOTLY = go is not None

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None  # type: ignore[assignment]

# lazy import to avoid circulars
from .synaptic import SynapticMoE, SynapticExpert

if TYPE_CHECKING:
    from .neuroscore import NeuroScore, NeuroScoreConfig
else:
    try:
        from .neuroscore import NeuroScore, NeuroScoreConfig
    except Exception:
        NeuroScore = None  # type: ignore[assignment]
        NeuroScoreConfig = None  # type: ignore[assignment]

# --------------------------- utilities --------------------------------


def _to_np(x: torch.Tensor) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _cosine(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> float:
    u = u / (np.linalg.norm(u) + eps)
    v = v / (np.linalg.norm(v) + eps)
    return float(np.dot(u, v))


def _reduce_camkii(expert: SynapticExpert) -> float:
    # average camkii over fc1/fc2 posts
    vals = []
    for fc in (expert.fc1, expert.fc2):
        if hasattr(fc, "post") and hasattr(fc.post, "camkii"):
            t = cast(torch.Tensor, fc.post.camkii)
            vals.append(float(t.item()))
    return float(np.mean(vals)) if vals else 0.0


def _reduce_mgate(expert: SynapticExpert) -> float:
    vals = []
    for fc in (expert.fc1, expert.fc2):
        if hasattr(fc, "post") and hasattr(fc.post, "m_gate"):
            t = cast(torch.Tensor, fc.post.m_gate)
            vals.append(float(t.item()))
    return float(np.mean(vals)) if vals else 0.0


def _reduce_elig_norm(expert: SynapticExpert) -> float:
    vals = []
    for fc in (expert.fc1, expert.fc2):
        if hasattr(fc, "post"):
            u = cast(torch.Tensor, fc.post.U)
            v = cast(torch.Tensor, fc.post.V)
            h = cast(torch.Tensor, fc.post.H_fast)
            vals.append(
                float(
                    u.norm().item()
                    + v.norm().item()
                    + h.norm().item()
                )
            )
    return float(np.mean(vals)) if vals else 0.0


def _weight_energy_util(energy: np.ndarray, fatigue: np.ndarray) -> np.ndarray:
    energy = np.clip(energy, 0.0, 1.0)
    # fatigue is already ndarray here based on usage in _layer_metrics
    util = 1.0 - np.clip(fatigue, 0.0, 1.0)
    return energy * util


def _fit_2d(emb: np.ndarray) -> np.ndarray:
    if emb.shape[1] <= 2:
        return emb
    # Fallback for very small populations where UMAP/PCA might fail or be weird
    if emb.shape[0] < 4:
        # Just project to first 2 dims or random
        return emb[:, :2]
        
    if _HAS_UMAP:
        try:
            red = UMAP(
                n_neighbors=min(15, emb.shape[0] - 1),
                min_dist=0.3,
                metric="cosine",
                random_state=42,
            )
            return red.fit_transform(emb)
        except Exception as err:
            print(f"[neuroviz] UMAP failed, falling back to PCA/random: {err}")
            
    if _HAS_SKLEARN:
        return PCA(n_components=2).fit_transform(emb)
    # fallback: random projection
    W = np.random.normal(0, 1, (emb.shape[1], 2))
    Y = emb @ W
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    return Y


# --------------------------- lineage book ------------------------------


class LineageBook:
    """
    Keeps a per-layer log of events (split/merge). Provides:
      - timeline PNG render
      - optional interactive HTML (plotly)
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        _ensure_dir(save_dir)
        # layer_id -> list of events [(step, "merge", w,l,child), (step, "split", parent, child)]
        self.events: Dict[str, List[Tuple[int, str, List[int]]]] = {}

    def log_merge(
        self, layer_name: str, step: int, parent_i: int, parent_j: int, child_idx: int
    ):
        self.events.setdefault(layer_name, []).append(
            (step, "merge", [parent_i, parent_j, child_idx])
        )
        self._persist(layer_name)

    def log_split(self, layer_name: str, step: int, parent_idx: int, child_idx: int):
        self.events.setdefault(layer_name, []).append(
            (step, "split", [parent_idx, child_idx])
        )
        self._persist(layer_name)

    def _persist(self, layer_name: str):
        path = os.path.join(self.save_dir, f"{layer_name}_lineage.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.events.get(layer_name, []), f)

    def render_timeline_png(self, layer_name: str, step: int):
        evs = self.events.get(layer_name, [])
        if not evs:
            return
        fig, ax = plt.subplots(figsize=(12, 4))
        ys = {}
        y_next = 0
        for s, et, ids in evs:
            if et == "merge":
                i, j, c = ids
                for eid in (i, j, c):
                    if eid not in ys:
                        ys[eid] = y_next
                        y_next += 1
                ax.plot([s, s], [ys[i], ys[c]], color="purple", lw=2, alpha=0.7)
                ax.plot([s, s], [ys[j], ys[c]], color="purple", lw=2, alpha=0.7)
                ax.scatter([s], [ys[c]], color="purple", s=24, marker="o", zorder=5)
            elif et == "split":
                p, c = ids
                for eid in (p, c):
                    if eid not in ys:
                        ys[eid] = y_next
                        y_next += 1
                ax.plot([s, s], [ys[p], ys[c]], color="green", lw=2, alpha=0.7)
                ax.scatter([s], [ys[c]], color="green", s=24, marker="^", zorder=5)
        ax.set_title(f"Lineage — {layer_name} @ step {step:,}")
        ax.set_xlabel("step")
        ax.set_ylabel("expert ID (track)")
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        out = os.path.join(self.save_dir, f"{layer_name}_lineage_{step:09d}.png")
        plt.savefig(out, dpi=120)
        plt.close(fig)

    def render_interactive_html(self, layer_name: str, step: int):
        if not _HAS_PLOTLY:
            return
        evs = self.events.get(layer_name, [])
        if not evs:
            return
        # very lightweight: just scatter the events along time with text labels
        xs, ys, texts, colors = [], [], [], []
        ymap = {}
        ycur = 0
        for s, et, ids in evs:
            if et == "merge":
                i, j, c = ids
                for eid in (i, j, c):
                    if eid not in ymap:
                        ymap[eid] = ycur
                        ycur += 1
                xs += [s, s, s]
                ys += [ymap[i], ymap[j], ymap[c]]
                texts += [f"merge parent {i}", f"merge parent {j}", f"child {c}"]
                colors += ["purple", "purple", "purple"]
            elif et == "split":
                p, c = ids
                for eid in (p, c):
                    if eid not in ymap:
                        ymap[eid] = ycur
                        ycur += 1
                xs += [s, s]
                ys += [ymap[p], ymap[c]]
                texts += [f"split parent {p}", f"child {c}"]
                colors += ["green", "green"]
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    text=texts,
                    marker=dict(color=colors, size=10),
                )
            ]
        )
        fig.update_layout(
            title=f"Interactive Lineage — {layer_name} @ step {step:,}",
            xaxis_title="step",
            yaxis_title="expert track",
            template="plotly_dark",
            height=500,
        )
        out = os.path.join(self.save_dir, f"{layer_name}_lineage_{step:09d}.html")
        fig.write_html(out, include_plotlyjs="cdn")


# --------------------------- NeuroVizManager ---------------------------


@dataclass
class NeuroVizConfig:
    log_dir: str = "runs/neuroviz"
    tb_every: int = 1000
    image_every: int = 10000
    interactive_every: int = 25000
    top_n_radar: int = 6
    save_pngs: bool = True
    write_tensorboard: bool = True
    write_interactive_html: bool = True


class NeuroVizManager:
    """
    Central orchestrator for metrics, images, TensorBoard, and lineage logging.
    Acts as a logger for the SplitMergeController (via on_merge/on_split).
    """

    def __init__(self, cfg: NeuroVizConfig):
        self.cfg = cfg
        _ensure_dir(cfg.log_dir)
        if cfg.write_tensorboard:
            if SummaryWriter is None:
                raise ImportError(
                    "TensorBoard is not installed but `write_tensorboard=True`. "
                    "Install the optional dependencies (e.g. `uv sync --extra viz`) and retry."
                )
            self.tb = SummaryWriter(cfg.log_dir)
        else:
            self.tb = None
        self.layers: List[Tuple[str, SynapticMoE]] = []  # (name, module)
        self.lineage = LineageBook(os.path.join(cfg.log_dir, "lineage"))
        self._last_tb = -(10**12)
        self._last_img = -(10**12)
        self._last_html = -(10**12)
        
        # NeuroScore hook
        self.score = None
        if NeuroScore is not None:
            # Default score config
            sc_cfg = NeuroScoreConfig(enabled=True, update_every=100)
            self.score = NeuroScore(sc_cfg, neuroviz=self)

    # --------- registration & events (used by controller) ----------

    def register_model(self, model: nn.Module):
        """Record every SynapticMoE with a stable name (layer index)."""
        idx = 0
        for m in model.modules():
            if isinstance(m, SynapticMoE):
                name = f"moe_L{idx}"
                self.layers.append((name, m))
                if self.score is not None:
                    self.score.register_layer(name, m.num_experts)
                idx += 1

    def on_merge(
        self,
        moe: SynapticMoE,
        parent_i: int,
        parent_j: int,
        child_idx: int,
        step: Optional[int] = None,
    ):
        name = self._name_of(moe)
        if name:
            step = (
                step if step is not None else int(time.time())
            )  # fallback when caller didn't supply step
            self.lineage.log_merge(name, step, parent_i, parent_j, child_idx)

    def on_split(
        self,
        moe: SynapticMoE,
        parent_idx: int,
        child_idx: int,
        step: Optional[int] = None,
    ):
        name = self._name_of(moe)
        if name:
            step = step if step is not None else int(time.time())
            self.lineage.log_split(name, step, parent_idx, child_idx)

    def _name_of(self, moe: SynapticMoE) -> Optional[str]:
        for nm, m in self.layers:
            if m is moe:
                return nm
        return None

    def close(self):
        if self.tb is not None:
            self.tb.close()

    @torch.no_grad()
    def _layer_metrics(self, moe: SynapticMoE) -> Dict[str, np.ndarray]:
        emb = _to_np(cast(torch.Tensor, moe.router_embeddings))  # (E, D)
        fatigue = _to_np(cast(torch.Tensor, moe.fatigue))  # (E,)
        energy = _to_np(cast(torch.Tensor, moe.energy))  # (E,)
        util = 1.0 - np.clip(fatigue, 0.0, 1.0)  # util proxy
        health = _weight_energy_util(energy, fatigue)

        # expert-specific reductions
        mgate, camkii, elig = [], [], []
        for e in moe.experts:
            e = cast(SynapticExpert, e)
            mgate.append(_reduce_mgate(e))
            camkii.append(_reduce_camkii(e))
            elig.append(_reduce_elig_norm(e))

        mgate_arr = np.asarray(mgate, dtype=np.float32)
        camkii_arr = np.asarray(camkii, dtype=np.float32)
        elig_arr = np.asarray(elig, dtype=np.float32)

        # “quantal proxy”: weight norms of slow+fast in fc2 (downstream)
        qprox = []
        for e in moe.experts:
            e = cast(SynapticExpert, e)
            s = float(e.fc2.w_slow.norm().item() + e.fc2.w_fast.norm().item())
            qprox.append(s)
        qprox_arr = np.asarray(qprox, dtype=np.float32)

        return dict(
            embedding=emb,
            util=util,
            energy=energy,
            health=health,
            mgate=mgate_arr,
            camkii=camkii_arr,
            elig=elig_arr,
            qprox=qprox_arr,
        )

    # ------------------------- per-step logging ---------------------

    def step(self, model: nn.Module, step: int, loss: Optional[torch.Tensor] = None):
        # ensure we have layers registered
        if not self.layers:
            self.register_model(model)

        # NeuroScore update (if active)
        if self.score is not None and loss is not None:
            self.score.step(model, loss, step)

        # TensorBoard scalars/hists
        if self.tb is not None and step - self._last_tb >= self.cfg.tb_every:
            for name, moe in self.layers:
                self._log_tb_layer(name, moe, step)
            self._last_tb = step

        # Static images
        if self.cfg.save_pngs and step - self._last_img >= self.cfg.image_every:
            for name, moe in self.layers:
                self._write_images(name, moe, step)
                self.lineage.render_timeline_png(name, step)
            self._last_img = step

        # Optional interactive HTML
        if (
            self.cfg.write_interactive_html
            and _HAS_PLOTLY
            and step - self._last_html >= self.cfg.interactive_every
        ):
            for name, moe in self.layers:
                self.lineage.render_interactive_html(name, step)
            self._last_html = step

    # ------------------------- TensorBoard writers ------------------

    def _log_tb_layer(self, name: str, moe: SynapticMoE, step: int):
        m = self._layer_metrics(moe)
        # scalars
        if self.tb is None:
            return
        self.tb.add_scalar(f"{name}/population", moe.num_experts, step)
        self.tb.add_scalar(f"{name}/util_mean", float(np.mean(m["util"])), step)
        self.tb.add_scalar(f"{name}/energy_mean", float(np.mean(m["energy"])), step)
        self.tb.add_scalar(f"{name}/health_mean", float(np.mean(m["health"])), step)
        self.tb.add_scalar(f"{name}/mgate_mean", float(np.mean(m["mgate"])), step)
        self.tb.add_scalar(f"{name}/camkii_mean", float(np.mean(m["camkii"])), step)

        # histograms (downsample if huge)
        for key in ("util", "energy", "health", "mgate", "camkii", "elig", "qprox"):
            arr = m[key]
            self.tb.add_histogram(f"{name}/hist/{key}", arr, step)

        # embedding projector
        emb2d = _fit_2d(m["embedding"])
        meta = [
            f"id:{i} util:{m['util'][i]:.3f} E:{m['energy'][i]:.2f}"
            for i in range(moe.num_experts)
        ]
        # add_embedding expects N x D; for 2D, it still works; for larger, projector clusters in higher-D
        self.tb.add_embedding(
            torch.from_numpy(m["embedding"]),
            metadata=meta,
            tag=f"{name}/router_embedding",
            global_step=step,
        )

        # save a small 2D map as a figure
        fig = plt.figure(figsize=(5, 4))
        cs = m["health"] / (np.max(m["health"]) + 1e-6)
        plt.scatter(
            emb2d[:, 0],
            emb2d[:, 1],
            c=cs,
            s=30 + 200 * m["util"],
            cmap="viridis",
            edgecolors="k",
            linewidths=0.3,
        )
        plt.title(f"{name} — 2D map @ {step:,}")
        plt.axis("off")
        self.tb.add_figure(f"{name}/map2d", fig, step)
        plt.close(fig)

    # ------------------------- image writers ------------------------

    def _write_images(self, name: str, moe: SynapticMoE, step: int):
        outdir = os.path.join(self.cfg.log_dir, "images", name)
        _ensure_dir(outdir)
        m = self._layer_metrics(moe)
        # map
        emb2d = _fit_2d(m["embedding"])
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        size = 30 + 350 * m["util"] * m["health"]
        color = m["energy"] * m["camkii"] if np.max(m["camkii"]) > 0 else m["energy"]
        sc = ax.scatter(
            emb2d[:, 0],
            emb2d[:, 1],
            s=size,
            c=color,
            cmap="coolwarm",
            edgecolors="k",
            linewidths=0.3,
            alpha=0.9,
        )
        ax.set_title(f"{name} map — step {step:,} | E={moe.num_experts}")
        ax.axis("off")
        fig.colorbar(sc, ax=ax, shrink=0.8, label="energy×CaMKII")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{name}_map_{step:09d}.png"), dpi=140)
        plt.close(fig)

        # radar for top-N
        self._radar(name, m, step, outdir, self.cfg.top_n_radar)

        # histograms
        self._hists(name, m, step, outdir)
        
        # contribution plot (Bio vs Static)
        self._contribution_plot(name, moe, step, outdir)
        
        # educational plots (Presyn, Hebbian, Raster)
        self._plot_presynaptic_dynamics(name, moe, step, outdir)
        self._plot_hebbian_memory(name, moe, step, outdir)
        self._plot_expert_raster(name, moe, step, outdir)
        
        # genetics and metabolism
        self._plot_genetics(name, moe, step, outdir)
        self._plot_metabolism(name, moe, step, outdir)
        
        # router decision breakdown
        self._plot_router_decision(name, moe, step, outdir)

    def _plot_router_decision(self, name: str, moe: SynapticMoE, step: int, outdir: str):
        if not hasattr(moe, "last_ctx") or "decision_data" not in moe.last_ctx:
            return
            
        d = moe.last_ctx["decision_data"]
        # Convert to numpy
        data = {k: _to_np(v) for k, v in d.items()}
        
        self._save_json(data, os.path.join(outdir, f"{name}_decision_{step:09d}.json"))

    def _plot_genetics(self, name: str, moe: SynapticMoE, step: int, outdir: str):
        if not hasattr(moe, "Xi") or not hasattr(moe, "_get_phenotype"):
            return
            
        pheno = moe._get_phenotype(moe.Xi) # (E, 4)
        pheno_np = _to_np(pheno)
        
        # [0] fatigue rate, [1] energy refill, [2] camkii, [3] pp1
        data = {
            "fatigue_rate": pheno_np[:, 0],
            "energy_refill": pheno_np[:, 1],
            "camkii_gain": pheno_np[:, 2],
            "pp1_gain": pheno_np[:, 3],
            "utilization": _to_np(cast(Tensor, moe.fatigue)) # Use fatigue as proxy for util history
        }
        self._save_json(data, os.path.join(outdir, f"{name}_genetics_{step:09d}.json"))

    def _plot_metabolism(self, name: str, moe: SynapticMoE, step: int, outdir: str):
        energy = _to_np(cast(Tensor, moe.energy))
        fatigue = _to_np(cast(Tensor, moe.fatigue))
        
        # Sort by energy to show inequality
        sorted_idx = np.argsort(energy)
        
        data = {
            "energy": energy[sorted_idx],
            "fatigue": fatigue[sorted_idx],
            "ids": sorted_idx.tolist()
        }
        self._save_json(data, os.path.join(outdir, f"{name}_metabolism_{step:09d}.json"))

    def _save_json(self, data: Dict[str, Any], path: str):
        def default(obj):
            if isinstance(obj, (np.ndarray, np.generic)):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            return str(obj)
        with open(path, 'w') as f:
            json.dump(data, f, default=default)

    def _plot_presynaptic_dynamics(self, name: str, moe: SynapticMoE, step: int, outdir: str):
        # Simulate "Boredom": Attend to the same token repeatedly
        # We need a SynapticPresyn instance. We can't easily grab one from MoE (it's in Attention).
        # But we can instantiate a dummy one with the same config.
        from .synaptic import SynapticPresyn, build_presyn_state
        
        cfg = moe.cfg
        head_dim = 64 # Assumption, but doesn't matter for this simulation as we fake logits
        pre = SynapticPresyn(head_dim, cfg).to("cpu")
        
        T = 50
        # Scenario: Attend to token 0 for 20 steps, then token 1 for 20 steps
        logits = torch.zeros(1, 1, T, T)
        # Causal mask
        mask = torch.tril(torch.ones(T, T)).bool()
        
        # Set high logits for target
        # Steps 0-20: target 0
        logits[:, :, :25, 0] = 20.0
        # Steps 20-40: target 1
        logits[:, :, 25:, 1] = 20.0
        
        # Dummy q, k (needed for docking, but we can set them to be compatible)
        q = torch.randn(1, 1, T, head_dim)
        k = torch.randn(1, 1, T, head_dim)
        
        state = build_presyn_state(1, T, 1, "cpu", torch.float32, cfg)
        
        # Run forward
        # SynapticPresyn.forward returns (syn_logit, new_state)
        # But it updates state in-place or returns new tensors? It returns new tensors.
        # And it processes the whole sequence at once (parallel).
        
        with torch.no_grad():
            syn_logit, final_state = pre(q, k, logits, state, mask, train_mode=False)
            
        # We want to see the time-evolution of RRP, C, Release for the *target* tokens.
        # RRP is (B,H,T). It represents the pool available *at step t*.
        # Actually RRP is per-key (T_k). 
        # Wait, in SynapticPresyn:
        # RRP is (B,H,T) -> This is the RRP of the *key* token?
        # Yes: "RRP_refill = (rho_r * RRP + ...)"
        # And "used_rrp = release_frac.sum(dim=2)" -> sum over queries? No, dim=2 is T_q?
        # Let's check presyn code:
        # raw_release: (B,H,T_q, T_k)
        # used_rrp = release_frac.sum(dim=2) -> Sum over queries attending to this key.
        # So RRP[t] is the vesicle pool of token t (acting as a Key).
        
        # So we want to plot RRP of Token 0 and Token 1.
        rrp = final_state["RRP"][0, 0].numpy() # (T,)
        c_val = final_state["C"][0, 0].numpy()
        
        # Release probability? 
        # We can infer it from syn_logit or just re-calculate.
        # syn_logit is (B,H,T,T).
        # Let's look at syn_logit[:,:,t,0] (attention to token 0 at step t)
        syn_adjust = syn_logit[0, 0, :, 0].numpy() # (T,)
        
        # Save data for interactive dashboard
        data = {
            "rrp": rrp,
            "calcium": c_val,
            "logit_delta": syn_adjust,
            "steps": list(range(T))
        }
        self._save_json(data, os.path.join(outdir, f"{name}_presyn_{step:09d}.json"))
        
        fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        
        # Plot RRP of Token 0
        ax[0].plot(rrp, label="RRP (Vesicles)", color="green", lw=2)
        ax[0].set_title("Presynaptic State (Token 0)")
        ax[0].set_ylabel("Pool Size")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        
        # Plot Calcium of Token 0
        ax[1].plot(c_val, label="Calcium (Excitement)", color="orange", lw=2)
        ax[1].set_ylabel("Concentration")
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
        
        # Plot Synaptic Adjustment to Token 0
        ax[2].plot(syn_adjust, label="Logit Adjustment", color="red", lw=2)
        ax[2].set_ylabel("Logit Delta")
        ax[2].set_xlabel("Time Step")
        ax[2].legend()
        ax[2].grid(True, alpha=0.3)
        
        ax[2].text(5, -2, "Attending to Token 0...", fontsize=9, color="gray")
        ax[2].text(30, -2, "Switched to Token 1", fontsize=9, color="gray")
        
        fig.suptitle(f"The 'Boredom' Mechanism (Simulated) @ {step}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{name}_presyn_{step:09d}.png"), dpi=120)
        plt.close(fig)

    def _plot_hebbian_memory(self, name: str, moe: SynapticMoE, step: int, outdir: str):
        # Visualize H_fast of the most active expert
        # Find most active expert from last_ctx
        if not hasattr(moe, "last_ctx") or not moe.last_ctx:
            return
            
        # We can also just pick expert 0 for consistency
        e_idx = 0
        expert = cast(SynapticExpert, moe.experts[e_idx])
        
        # H_fast is (d_in, d_out). It might be large.
        # We'll take a slice.
        if not hasattr(expert.fc1.post, "H_fast"):
            return
            
        H = cast(torch.Tensor, expert.fc1.post.H_fast).detach().float().cpu().numpy()
        # Slice 50x50
        H_sub = H[:50, :50]
        
        # Save data
        self._save_json({"heatmap": H_sub}, os.path.join(outdir, f"{name}_hebbian_{step:09d}.json"))
        
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(H_sub, cmap="RdBu_r", vmin=-0.01, vmax=0.01)
        ax.set_title(f"Hebbian Trace (Expert {e_idx}) - Short Term Memory")
        ax.set_xlabel("Output Dim")
        ax.set_ylabel("Input Dim")
        fig.colorbar(im, ax=ax)
        
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{name}_hebbian_{step:09d}.png"), dpi=120)
        plt.close(fig)

    def _plot_expert_raster(self, name: str, moe: SynapticMoE, step: int, outdir: str):
        if not hasattr(moe, "last_ctx") or not moe.last_ctx:
            return
            
        # gates: (B, T, k)
        gates = _to_np(moe.last_ctx["gates"])
        indices = _to_np(moe.last_ctx["indices"])
        
        # Flatten B*T
        B, T, k = gates.shape
        flat_gates = gates.reshape(-1, k)
        flat_indices = indices.reshape(-1, k)
        
        # Take first 100 tokens
        L = min(100, B*T)
        
        # Create a matrix (Experts, Time)
        E = moe.num_experts
        raster = np.zeros((E, L))
        
        for t in range(L):
            for i in range(k):
                idx = int(flat_indices[t, i])
                val = flat_gates[t, i]
                if idx < E:
                    raster[idx, t] = val
        
        # Save data
        self._save_json({"raster": raster}, os.path.join(outdir, f"{name}_raster_{step:09d}.json"))
                    
        fig, ax = plt.subplots(figsize=(10, 6))
        # Use a dark background for "brain scan" look
        im = ax.imshow(raster, aspect="auto", cmap="magma", interpolation="nearest")
        ax.set_title(f"Expert Activation Raster (First {L} tokens)")
        ax.set_xlabel("Token Time")
        ax.set_ylabel("Expert ID")
        fig.colorbar(im, ax=ax, label="Gate Probability")
        
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{name}_raster_{step:09d}.png"), dpi=120)
        plt.close(fig)

    def _contribution_plot(self, name: str, moe: SynapticMoE, step: int, outdir: str):
        # Visualize the magnitude of fast weights vs slow weights
        # We'll sample a few experts
        experts = moe.experts[:min(5, len(moe.experts))]
        
        slow_norms = []
        fast_norms = []
        ids = []
        
        for i, e in enumerate(experts):
            e = cast(SynapticExpert, e)
            # L2 norm of weights
            s = e.fc1.w_slow.norm().item() + e.fc2.w_slow.norm().item()
            f = e.fc1.w_fast.norm().item() + e.fc2.w_fast.norm().item()
            # Add Hebbian trace norm if available
            if hasattr(e.fc1.post, "H_fast"):
                f += cast(torch.Tensor, e.fc1.post.H_fast).norm().item()
            if hasattr(e.fc2.post, "H_fast"):
                f += cast(torch.Tensor, e.fc2.post.H_fast).norm().item()
                
            slow_norms.append(s)
            fast_norms.append(f)
            ids.append(f"Exp {i}")
            
        # Save data
        self._save_json({
            "ids": ids,
            "slow_norms": slow_norms,
            "fast_norms": fast_norms
        }, os.path.join(outdir, f"{name}_contrib_{step:09d}.json"))
            
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(ids))
        width = 0.35
        
        ax.bar(x - width/2, slow_norms, width, label='Slow (Static)', color='#4472C4')
        ax.bar(x + width/2, fast_norms, width, label='Fast (Bio)', color='#ED7D31')
        
        ax.set_ylabel('Weight Norm (L2)')
        ax.set_title(f'{name} - Static vs Bio Weight Magnitude')
        ax.set_xticks(x)
        ax.set_xticklabels(ids)
        ax.legend()
        
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{name}_contrib_{step:09d}.png"), dpi=120)
        plt.close(fig)

    def _radar(
        self,
        name: str,
        m: Dict[str, np.ndarray],
        step: int,
        outdir: str,
        top_n: int = 6,
    ):
        N = min(top_n, len(m["util"]))
        order = np.argsort(-m["util"])[:N]
        labels = ["util", "energy", "camkii", "mgate", "elig", "qprox"]
        K = len(labels)
        th = np.linspace(0, 2 * np.pi, K, endpoint=False)
        
        # Compute global max per metric for fair comparison
        max_vals = np.array([
            np.max(m[key]) + 1e-6 for key in labels
        ], dtype=np.float32)
        
        fig = plt.figure(figsize=(7, 7))
        ax = plt.subplot(111, polar=True)
        for idx in order:
            vals = np.array(
                [
                    m["util"][idx],
                    m["energy"][idx],
                    m["camkii"][idx],
                    m["mgate"][idx],
                    m["elig"][idx],
                    m["qprox"][idx],
                ],
                dtype=np.float32,
            )
            # normalize by population max
            v = vals / max_vals
            ax.plot(np.r_[th, th[0]], np.r_[v, v[0]], lw=2, label=f"id {idx}")
            ax.fill(np.r_[th, th[0]], np.r_[v, v[0]], alpha=0.15)
        ax.set_xticks(th)
        ax.set_xticklabels(labels)
        ax.set_title(f"{name} — top-{N} profiles @ {step:,} (norm by pop max)")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{name}_radar_{step:09d}.png"), dpi=140)
        plt.close(fig)

    def _hists(self, name: str, m: Dict[str, np.ndarray], step: int, outdir: str):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        keys = ["util", "energy", "health", "camkii", "mgate", "qprox"]
        for ax, key in zip(axes.ravel(), keys):
            ax.hist(m[key], bins=20, color="#4472C4", alpha=0.85)
            ax.set_title(key)
            ax.grid(True, alpha=0.2)
        fig.suptitle(f"{name} distributions @ {step:,}")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(outdir, f"{name}_hists_{step:09d}.png"), dpi=140)
        plt.close(fig)
