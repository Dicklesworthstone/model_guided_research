"""
NeuroViz for JAX
Adapts the visualization logic from nanochat/neuroviz.py to work with JAX arrays.
"""

import os
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

# Optional imports
try:
    import umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

try:
    from sklearn.decomposition import PCA
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

def _to_np(x):
    if hasattr(x, "device_buffer"): # JAX array
        return np.array(x)
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _fit_2d(emb):
    emb = _to_np(emb)
    if emb.shape[1] <= 2:
        return emb
    if emb.shape[0] < 4:
        return emb[:, :2]
        
    if _HAS_UMAP:
        try:
            red = umap.UMAP(n_neighbors=min(15, emb.shape[0]-1), min_dist=0.3, metric="cosine", random_state=42)
            return red.fit_transform(emb)
        except:
            pass
            
    if _HAS_SKLEARN:
        return PCA(n_components=2).fit_transform(emb)
        
    # Fallback
    W = np.random.normal(0, 1, (emb.shape[1], 2))
    Y = emb @ W
    return Y

@dataclass
class NeuroVizConfig:
    log_dir: str = "runs/neuroviz_jax"
    image_every: int = 100
    save_pngs: bool = True

class NeuroVizManager:
    def __init__(self, cfg: NeuroVizConfig):
        self.cfg = cfg
        _ensure_dir(cfg.log_dir)
        self._last_img = -1e9
        
    def step(self, step: int, metrics: Dict[str, Any]):
        """
        metrics: Dictionary of layer_name -> {metric_name: value}
        """
        if self.cfg.save_pngs and step - self._last_img >= self.cfg.image_every:
            for layer_name, layer_metrics in metrics.items():
                self._write_images(layer_name, layer_metrics, step)
            self._last_img = step

    def _write_images(self, name: str, m: Dict[str, Any], step: int):
        outdir = os.path.join(self.cfg.log_dir, "images", name)
        _ensure_dir(outdir)
        
        # If we have embeddings, plot map
        if "embedding" in m:
            emb2d = _fit_2d(m["embedding"])
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # Color by some metric if available
            c = m.get("util", np.zeros(emb2d.shape[0]))
            s = 30 + 200 * c
            
            sc = ax.scatter(emb2d[:, 0], emb2d[:, 1], s=s, c=c, cmap="viridis", edgecolors="k", linewidths=0.3, alpha=0.9)
            ax.set_title(f"{name} map — step {step:,}")
            ax.axis("off")
            fig.colorbar(sc, ax=ax, label="utilization")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"{name}_map_{step:09d}.png"), dpi=140)
            plt.close(fig)
            
        # Histograms
        keys = [k for k in m.keys() if k != "embedding" and isinstance(m[k], (np.ndarray, jax.Array)) and m[k].ndim == 1]
        if keys:
            n_plots = len(keys)
            cols = 3
            rows = (n_plots + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
            axes = np.array(axes).reshape(-1)
            
            for i, key in enumerate(keys):
                ax = axes[i]
                val = _to_np(m[key])
                ax.hist(val, bins=20, color="#4472C4", alpha=0.85)
                ax.set_title(key)
                ax.grid(True, alpha=0.2)
                
            for i in range(n_plots, len(axes)):
                axes[i].axis("off")
                
            fig.suptitle(f"{name} distributions @ {step:,}")
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.savefig(os.path.join(outdir, f"{name}_hists_{step:09d}.png"), dpi=140)
            plt.close(fig)

        # Save raw data
        data_path = os.path.join(outdir, f"{name}_metrics_{step:09d}.json")
        serializable_m = {k: _to_np(v).tolist() if isinstance(v, (np.ndarray, jax.Array)) else v for k, v in m.items()}
        with open(data_path, "w") as f:
            json.dump(serializable_m, f)

