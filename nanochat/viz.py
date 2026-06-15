"""Model-state visualizations for nanochat (beads hi3, 7ow).

This module renders *insightful* snapshots of a transformer's internal state on
a small, seeded sample batch, so you can SEE what a mathematical attention
mechanism is doing rather than only reading scalar metrics. It deliberately
reuses the introspection buffers the attention mechanisms already populate
(``standard_record_attn_entropy`` -> ``attn_entropy_head_mean``,
``tropical_record_margins`` -> ``tropical_gamma_head_*``) plus a lightweight,
reversible runtime capture of the softmax attention matrices, so it adds **no**
new state to the hot path and never changes a forward result.

Two responsibilities, one shared pipeline (build/seed a model -> run a forward
on a sample batch -> harvest diagnostics -> render):

* ``state``   (hi3) -- >=3 model-state visualizations saved to ``artifacts/vis/``:
    per-head attention-entropy heatmap, per-head softmax attention maps, and
    the tropical route-margin heatmap (+ optional extras when present).
* ``entropy`` (7ow) -- per-head attention entropy AND cross-head route
    diversity for a baseline vs. a math-feature config, with tables + a grouped
    bar plot saved to ``artifacts/vis/entropy/``.

Standalone, reproducible, and import-light::

    python -m nanochat.viz state   --attention standard --out artifacts/vis/state_standard
    python -m nanochat.viz state   --attention tropical --out artifacts/vis/state_tropical
    python -m nanochat.viz entropy --baseline standard --feature tropical

Every render writes a ``summary.json`` manifest + an ``index.html`` and prints a
rich table; interpretation notes live in ``docs/visualizations.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Torch / matplotlib are imported lazily inside functions (mirrors cli.py) so
# that ``python -m nanochat.viz --help`` stays instant and import-cost is paid
# only when a visualization actually runs.

DEFAULT_VIS_ROOT = Path("artifacts/vis")

# A diagnostic-task-shaped default prompt: the trained corpus format, so a
# loaded checkpoint shows meaningful structure. Fresh models attend ~uniformly
# regardless, so the exact text barely matters for an untrained probe.
DEFAULT_TEXT = "TASK arith CMP 1.00e-02 2.00e+03 OUT 1 TASK dyck ( ( ) ( ) ) OUT 1"


def _console() -> Any:
    from rich.console import Console

    return Console()


def _safe_token_label(s: str) -> str:
    """A matplotlib/terminal-safe token label: newlines escaped, non-printable
    bytes (common in byte-level BPE, e.g. control chars) shown as a middle dot so
    they render without 'glyph missing from font' warnings or garbage."""
    out = "".join(ch if ch.isprintable() else "·" for ch in s.replace("\n", "\\n"))
    return out if out else "∅"


# --------------------------------------------------------------------------- #
# Model construction + sample batch                                           #
# --------------------------------------------------------------------------- #


# Per-mechanism head geometry that satisfies each block's validate_config (e.g.
# reversible needs n_kv_head <= n_head//2; quaternion/octonion need head_dim
# divisible by 4/8). These small, valid defaults keep the probe construction
# robust across mechanisms without the caller having to know each constraint.
_MECH_DEFAULTS: dict[str, dict[str, int]] = {
    "reversible": {"n_head": 4, "n_kv_head": 2, "n_embd": 64},
    "octonion": {"n_head": 4, "n_kv_head": 4, "n_embd": 128},
    "quaternion": {"n_head": 4, "n_kv_head": 4, "n_embd": 64},
}

# Record flags to flip on per mechanism so the introspection buffers populate.
_RECORD_FLAGS: dict[str, dict[str, bool]] = {
    "standard": {"standard_record_attn_entropy": True},
    "tropical": {"tropical_record_margins": True, "standard_record_attn_entropy": True},
    "reversible": {"reversible_record_energy": True},
}


def build_probe_model(
    attention_type: str,
    *,
    device: str = "cpu",
    seed: int = 0,
    n_layer: int = 4,
    n_head: int | None = None,
    n_kv_head: int | None = None,
    n_embd: int | None = None,
    sequence_len: int = 128,
    vocab_size: int = 256,
    extra_config: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Construct a small, seeded, eval-mode GPT for one mechanism.

    Returns ``(model, meta)`` where ``meta`` records the exact config used so
    the render is fully reproducible from the manifest.
    """
    import torch

    from nanochat.gpt import GPT, GPTConfig

    torch.manual_seed(int(seed))
    defaults = _MECH_DEFAULTS.get(attention_type, {})
    cfg_kwargs: dict[str, Any] = {
        "n_layer": int(n_layer),
        "n_head": int(n_head if n_head is not None else defaults.get("n_head", 4)),
        "n_kv_head": int(n_kv_head if n_kv_head is not None else defaults.get("n_kv_head", 4)),
        "n_embd": int(n_embd if n_embd is not None else defaults.get("n_embd", 64)),
        "sequence_len": int(sequence_len),
        "vocab_size": int(vocab_size),
        "attention_type": attention_type,
    }
    cfg_kwargs.update(_RECORD_FLAGS.get(attention_type, {}))
    if extra_config:
        cfg_kwargs.update(extra_config)
    config = GPTConfig(**cfg_kwargs)
    model = GPT(config).to(device)
    model.eval()
    meta = {"source": "fresh", "seed": int(seed), "config": dict(cfg_kwargs)}
    return model, meta


def load_probe_model(
    checkpoint: Path,
    *,
    step: int | None = None,
    device: str = "cpu",
    record_overrides: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a trained nanochat checkpoint for visualization.

    Mirrors the eval loader's deliberate-vocab-mismatch handling: nanochat
    configs use the padded default vocab (50304) with the 50257-token GPT-2
    tokenizer, so we never assert vocab equality. Record flags (which do not
    change parameter shapes) are merged in so the introspection buffers
    populate during the probe forward.
    """
    import torch

    from nanochat.checkpoint_manager import find_last_step, load_checkpoint
    from nanochat.gpt import GPT, GPTConfig

    resolved = step if step is not None else find_last_step(str(checkpoint))
    model_data, _optim, ckpt_meta = load_checkpoint(str(checkpoint), resolved, device)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    if ckpt_meta.get("model_type", "gpt") != "gpt":
        raise ValueError(f"viz supports model_type=gpt checkpoints, got {ckpt_meta.get('model_type')!r}")
    config_dict = dict(ckpt_meta["model_config"])
    if "semiring_beta_live" in ckpt_meta:
        config_dict["semiring_beta"] = ckpt_meta["semiring_beta_live"]
    attn = str(config_dict.get("attention_type", "standard"))
    config_dict.update(_RECORD_FLAGS.get(attn, {}))
    if record_overrides:
        config_dict.update(record_overrides)
    config = GPTConfig(**config_dict)
    model = GPT(config).to(device)
    dev = torch.device(device)
    if dev.type in {"cpu", "mps"}:
        model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}
    model.load_state_dict(model_data, strict=True)
    model.eval()
    meta = {"source": str(checkpoint), "step": int(resolved), "config": config_dict}
    return model, meta


def sample_batch(
    *,
    text: str | None,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
    device: str = "cpu",
) -> tuple[Any, list[str] | None]:
    """Build a seeded ``(B, T)`` long tensor of token ids.

    Prefers encoding ``text`` with the GPT-2 tokenizer (interpretable token
    labels for the attention-map axes); falls back to seeded random ids if the
    tokenizer is unavailable (offline boxes) or no text is given.
    """
    import torch

    gen = torch.Generator().manual_seed(int(seed))
    tok = None
    ids: list[int] | None = None
    if text:
        try:
            from nanochat.tokenizer import get_tokenizer

            tok = get_tokenizer()
            # Clamp ids into the probe vocab so a tiny fresh model stays valid.
            ids = [int(i) % int(vocab_size) for i in tok.encode(text)]
        except Exception:
            tok, ids = None, None
    if not ids:
        flat = torch.randint(0, int(vocab_size), (int(batch_size) * int(seq_len),), generator=gen)
        return flat.view(int(batch_size), int(seq_len)).to(device), None
    # Tile / truncate the encoded ids to exactly seq_len, THEN derive labels from
    # the final ids so token_labels lines up 1:1 with the sequence axis.
    ids = (ids * (seq_len // len(ids) + 1))[:seq_len] if len(ids) < seq_len else ids[:seq_len]
    token_labels = [_safe_token_label(tok.decode([i])) for i in ids] if tok is not None else None
    row = torch.tensor(ids, dtype=torch.long)
    idx = row.unsqueeze(0).repeat(int(batch_size), 1)
    return idx.to(device), token_labels


# --------------------------------------------------------------------------- #
# Diagnostics harvesting                                                       #
# --------------------------------------------------------------------------- #


def _iter_attention_modules(model: Any) -> Iterator[tuple[int, Any, str]]:
    """Yield ``(layer_idx, attn_module, kind)`` for each block's attention.

    ``kind`` is the per-layer attention type. Gauge/reversible special blocks
    are surfaced as their wrapped attention where one exists.
    """
    blocks = getattr(getattr(model, "transformer", None), "h", None)
    if blocks is None:
        return
    for i, block in enumerate(blocks):
        kind = str(getattr(block, "attention_type", "standard"))
        attn = getattr(block, "attn", None)
        if attn is not None:
            yield i, attn, kind
        else:
            # reversible/gauge own a special_block; expose it for energy/etc.
            special = getattr(block, "special_block", None)
            if special is not None:
                yield i, special, kind


@contextmanager
def _capture_attention_maps(model: Any, example_idx: int) -> Iterator[dict[int, Any]]:
    """Reversibly patch standard-attention ``attend`` to record the softmax map.

    The patch runs entirely under ``no_grad`` and delegates to the original
    method for the real output, so the forward result is bitwise unchanged. The
    recorded map for ``example_idx`` is a ``(H, Tq, Tk)`` cpu tensor per layer.
    Restored on exit even if the forward raises.
    """
    import torch

    from nanochat.gpt import CausalSelfAttention, causal_attn_mask

    captured: dict[int, Any] = {}
    patched: list[tuple[Any, Any]] = []  # (module, original_bound_attend)

    def make_patched(module: Any, layer_idx: int, original: Any):
        def attend(q, k, v, *, kv_cache, pos0):
            with torch.no_grad():
                tq, tk = q.size(2), k.size(2)
                enable_gqa = module.n_head != module.n_kv_head
                kk = k
                if enable_gqa:
                    rep = module.n_head // module.n_kv_head
                    kk = k.repeat_interleave(rep, dim=1)
                scale = 1.0 / math.sqrt(float(module.head_dim))
                scores = torch.matmul(q.detach().float(), kk.detach().float().transpose(-2, -1)).mul(scale)
                mask = causal_attn_mask(tq, tk, device=q.device)
                scores = scores.masked_fill(~mask, float("-inf"))
                p = torch.softmax(scores, dim=-1)
                if p.size(0) > example_idx:
                    captured[layer_idx] = p[example_idx].detach().cpu()
            return original(q, k, v, kv_cache=kv_cache, pos0=pos0)

        return attend

    for layer_idx, module, _kind in _iter_attention_modules(model):
        if isinstance(module, CausalSelfAttention):
            original = module.attend  # bound method
            module.attend = make_patched(module, layer_idx, original)  # type: ignore[method-assign]
            patched.append((module, original))
    try:
        yield captured
    finally:
        for module, original in patched:
            module.attend = original  # type: ignore[method-assign]


@dataclass
class StateDiagnostics:
    """Harvested internal state from one probe forward."""

    attention_type: str
    n_layer: int
    n_head: int
    # per-layer per-head attention entropy (standard buffer): [[h...], ...]
    entropy_layer_head: list[list[float]] = field(default_factory=list)
    # per-layer per-head tropical runner-up margins
    margin_layer_head: list[list[float]] = field(default_factory=list)
    margin_min_layer_head: list[list[float]] = field(default_factory=list)
    route_coverage: float | None = None
    # captured softmax maps for one example: layer_idx -> (H, Tq, Tk) tensor
    attn_maps: dict[int, Any] = field(default_factory=dict)
    token_labels: list[str] | None = None

    def has_entropy(self) -> bool:
        return any(any(math.isfinite(x) for x in row) for row in self.entropy_layer_head)

    def has_margins(self) -> bool:
        return any(any(math.isfinite(x) for x in row) for row in self.margin_layer_head)


def collect_state(model: Any, idx: Any, *, example_idx: int = 0, token_labels: list[str] | None = None) -> StateDiagnostics:
    """Run one forward and harvest entropy / margins / attention maps."""
    import torch

    config = model.config
    diag = StateDiagnostics(
        attention_type=str(getattr(config, "attention_type", "standard")),
        n_layer=int(getattr(config, "n_layer", 0)),
        n_head=int(getattr(config, "n_head", 0)),
        token_labels=token_labels,
    )
    with _capture_attention_maps(model, example_idx) as captured, torch.no_grad():
        _ = model(idx)
    diag.attn_maps = captured

    coverages: list[float] = []
    for _i, module, _kind in _iter_attention_modules(model):
        ent = getattr(module, "attn_entropy_head_mean", None)
        if torch.is_tensor(ent) and ent.ndim == 1 and torch.isfinite(ent).any():
            diag.entropy_layer_head.append([float(x) for x in ent.tolist()])
        gmean = getattr(module, "tropical_gamma_head_mean", None)
        gmin = getattr(module, "tropical_gamma_head_min", None)
        if torch.is_tensor(gmean) and gmean.ndim == 1:
            diag.margin_layer_head.append([float(x) for x in gmean.tolist()])
            if torch.is_tensor(gmin) and gmin.ndim == 1:
                diag.margin_min_layer_head.append([float(x) for x in gmin.tolist()])
        cov = getattr(module, "tropical_route_coverage", None)
        if torch.is_tensor(cov) and cov.numel() == 1 and math.isfinite(float(cov.item())):
            coverages.append(float(cov.item()))
    if coverages:
        diag.route_coverage = sum(coverages) / len(coverages)
    return diag


# --------------------------------------------------------------------------- #
# Cross-head route-diversity metric (7ow)                                      #
# --------------------------------------------------------------------------- #


def _js_divergence(p: Any, q: Any, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence (base-2, in [0, 1]) between two distributions.

    ``p`` and ``q`` are 1-D torch tensors (head attention maps flattened)."""
    p = p.double() / p.double().sum().clamp_min(eps)
    q = q.double() / q.double().sum().clamp_min(eps)
    m = 0.5 * (p + q)

    def _kl(a: Any, b: Any) -> Any:
        return (a * ((a + eps).log() - (b + eps).log())).sum()

    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    # JS is in [0, ln2] mathematically; clamp the normalized value to [0, 1] so
    # float rounding on near-identical distributions can't yield a tiny negative.
    return max(0.0, min(1.0, float(js / math.log(2.0))))


def head_route_diversity(diag: StateDiagnostics) -> float | None:
    """Mean pairwise Jensen-Shannon divergence between heads' attention maps.

    A high value means heads route the *same tokens* to *different places*
    (specialized heads); near-zero means heads are redundant. Computed per layer
    from the captured softmax maps (flattened over query x key) and averaged.
    Returns None when no maps were captured (non-standard mechanisms).
    """
    if not diag.attn_maps:
        return None
    per_layer: list[float] = []
    for _layer, maps in sorted(diag.attn_maps.items()):
        h = maps.size(0)
        if h < 2:
            continue
        flat = [maps[i].reshape(-1) for i in range(h)]
        pair_js: list[float] = []
        for i in range(h):
            for j in range(i + 1, h):
                pair_js.append(_js_divergence(flat[i], flat[j]))
        if pair_js:
            per_layer.append(sum(pair_js) / len(pair_js))
    return (sum(per_layer) / len(per_layer)) if per_layer else None


# --------------------------------------------------------------------------- #
# Rendering helpers                                                            #
# --------------------------------------------------------------------------- #


def _heatmap_png(
    matrix: list[list[float]],
    out: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str = "viridis",
    xticklabels: list[str] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    fig, ax = plt.subplots(figsize=(max(4.0, 0.5 * arr.shape[1] + 2), max(3.0, 0.45 * arr.shape[0] + 1.5)))
    im = ax.imshow(arr, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # annotate small grids
    if arr.shape[0] * arr.shape[1] <= 64 and xticklabels is None:
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                ax.text(c, r, f"{arr[r, c]:.2f}", ha="center", va="center", color="w", fontsize=6)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _rich_heatmap(matrix: list[list[float]], *, row_prefix: str = "L", col_prefix: str = "h") -> Any:
    """A compact terminal heatmap using background-shaded cells."""
    from rich.table import Table
    from rich.text import Text

    flat = [v for row in matrix for v in row if math.isfinite(v)]
    lo, hi = (min(flat), max(flat)) if flat else (0.0, 1.0)
    span = (hi - lo) or 1.0
    ramp = "·░▒▓█"  # lowest is a visible dot (not a blank) so the grid keeps its shape
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 0))
    table.add_column("", style="dim")
    ncols = max((len(r) for r in matrix), default=0)
    for c in range(ncols):
        table.add_column(f"{col_prefix}{c}", justify="center")
    for r, row in enumerate(matrix):
        cells: list[Any] = [f"{row_prefix}{r}"]
        for v in row:
            if not math.isfinite(v):
                cells.append(Text(" · ", style="dim"))
                continue
            frac = (v - lo) / span
            ch = ramp[min(len(ramp) - 1, int(frac * (len(ramp) - 1) + 0.5))]
            # 256-color cube ramp red(196)=(5,0,0) -> green(46)=(0,5,0), -30/step.
            hue = 196 - 30 * int(round(frac * 5))
            cells.append(Text(f" {ch} ", style=f"color({max(16, min(231, hue))})"))
        table.add_row(*cells)
    return table


def _write_index_html(out_dir: Path, title: str, images: list[Path], summary: dict[str, Any]) -> Path:
    """A tiny self-contained gallery so the PNGs are browsable."""
    rows = "\n".join(
        f'<figure><img src="{p.name}" style="max-width:760px;border:1px solid #ccc"/>'
        f"<figcaption>{p.name}</figcaption></figure>"
        for p in images
    )
    body = (
        f"<h1>{title}</h1>"
        f"<pre style='background:#f6f8fa;padding:12px;border-radius:6px'>"
        f"{json.dumps(summary, indent=2, sort_keys=True)}</pre>{rows}"
    )
    html = (
        "<!doctype html><meta charset='utf-8'>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;max-width:840px}"
        "figure{margin:18px 0}</style>" + body
    )
    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# state subcommand (hi3)                                                       #
# --------------------------------------------------------------------------- #


def render_state(diag: StateDiagnostics, out_dir: Path, *, console: Any = None) -> dict[str, Any]:
    """Render >=3 model-state visualizations + manifest. Returns the summary."""
    console = console or _console()
    out_dir.mkdir(parents=True, exist_ok=True)
    images: list[Path] = []
    visuals: list[str] = []

    from rich.panel import Panel

    console.rule(f"[bold cyan]model-state visualizations · {diag.attention_type}")

    # (1) Per-head attention-entropy heatmap (layers x heads).
    if diag.has_entropy():
        png = out_dir / "attention_entropy_heatmap.png"
        _heatmap_png(
            diag.entropy_layer_head, png,
            title=f"Per-head attention entropy ({diag.attention_type})",
            xlabel="head", ylabel="layer", cmap="magma",
        )
        images.append(png)
        visuals.append("attention_entropy_heatmap")
        console.print(Panel(_rich_heatmap(diag.entropy_layer_head), title="attention entropy (nats) · layer×head", border_style="magenta"))

    # (2) Per-head softmax attention maps for one example (grid for first layer).
    if diag.attn_maps:
        layer0 = sorted(diag.attn_maps)[0]
        png = out_dir / "attention_maps.png"
        _attention_maps_png(diag.attn_maps[layer0], png, layer=layer0, token_labels=diag.token_labels, attention_type=diag.attention_type)
        images.append(png)
        visuals.append("attention_maps")
        console.print(f"[green]✓[/green] attention maps for layer {layer0}: {diag.attn_maps[layer0].shape} -> {png.name}")

    # (3) Tropical route-margin heatmap (layers x heads).
    if diag.has_margins():
        png = out_dir / "tropical_route_margins.png"
        _heatmap_png(
            diag.margin_layer_head, png,
            title=f"Tropical runner-up margins ({diag.attention_type})",
            xlabel="head", ylabel="layer", cmap="viridis",
        )
        images.append(png)
        visuals.append("tropical_route_margins")
        console.print(Panel(_rich_heatmap(diag.margin_layer_head), title="tropical route margins · layer×head", border_style="green"))
        if diag.route_coverage is not None:
            console.print(f"[dim]route coverage (β-thresholded): {diag.route_coverage:.4f}[/dim]")

    summary = {
        "schema": "mgr.viz.state.v1",
        "attention_type": diag.attention_type,
        "n_layer": diag.n_layer,
        "n_head": diag.n_head,
        "visuals": visuals,
        "entropy_layer_head": diag.entropy_layer_head or None,
        "margin_layer_head": diag.margin_layer_head or None,
        "route_coverage": diag.route_coverage,
        "head_route_diversity_js": head_route_diversity(diag),
        "images": [p.name for p in images],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    idx_html = _write_index_html(out_dir, f"model-state · {diag.attention_type}", images, summary)
    console.print(f"\n[bold]wrote[/bold] {len(images)} visualization(s) + manifest -> [cyan]{out_dir}/[/cyan] (open {idx_html.name})")
    if len(visuals) < 3:
        console.print(
            f"[yellow]note:[/yellow] {len(visuals)} visual(s) for '{diag.attention_type}'. "
            "The 3-visual acceptance is met by running standard (entropy + maps) and tropical (margins) — "
            "see docs/visualizations.md."
        )
    return summary


def _attention_maps_png(maps: Any, out: Path, *, layer: int, token_labels: list[str] | None, attention_type: str) -> None:
    """Grid of per-head softmax attention matrices for one example."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = maps.size(0)
    cols = min(4, h)
    rows = (h + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), squeeze=False)
    for head in range(rows * cols):
        ax = axes[head // cols][head % cols]
        if head < h:
            mat = maps[head].numpy()
            im = ax.imshow(mat, aspect="auto", cmap="cividis", interpolation="nearest", vmin=0.0)
            ax.set_title(f"head {head}", fontsize=8)
            ax.set_xlabel("key pos", fontsize=7)
            ax.set_ylabel("query pos", fontsize=7)
            # Label the axes with actual tokens when the sequence is short enough
            # to stay legible (otherwise the positional axes are clearer).
            if token_labels is not None and len(token_labels) == mat.shape[0] <= 32:
                ax.set_xticks(range(len(token_labels)))
                ax.set_xticklabels(token_labels, rotation=90, fontsize=4)
                ax.set_yticks(range(len(token_labels)))
                ax.set_yticklabels(token_labels, fontsize=4)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")
    fig.suptitle(f"Softmax attention maps · {attention_type} · layer {layer}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def run_state(args: argparse.Namespace) -> int:
    console = _console()
    out_dir = Path(args.out) if args.out else (DEFAULT_VIS_ROOT / f"state_{args.attention}")
    if args.checkpoint:
        model, meta = load_probe_model(Path(args.checkpoint), step=args.step, device=args.device)
        attention_type = str(meta["config"].get("attention_type", "standard"))
        vocab = int(meta["config"].get("vocab_size", 50304))
        seq_len = min(int(args.seq_len), int(meta["config"].get("sequence_len", args.seq_len)))
    else:
        model, meta = build_probe_model(
            args.attention, device=args.device, seed=args.seed, n_layer=args.n_layer,
            n_head=args.n_head, n_kv_head=args.n_kv_head, n_embd=args.n_embd,
            sequence_len=args.seq_len, vocab_size=args.vocab_size,
        )
        attention_type = args.attention
        vocab = int(args.vocab_size)
        seq_len = int(args.seq_len)
    idx, labels = sample_batch(
        text=(None if args.random_input else args.text), batch_size=args.batch_size,
        seq_len=seq_len, vocab_size=vocab, seed=args.seed, device=args.device,
    )
    console.print(f"[dim]model={meta['source']} attention={attention_type} input={tuple(idx.shape)} seed={args.seed}[/dim]")
    diag = collect_state(model, idx, example_idx=0, token_labels=labels)
    diag.attention_type = attention_type
    summary = render_state(diag, out_dir, console=console)
    summary["meta"] = meta
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- #
# entropy subcommand (7ow)                                                     #
# --------------------------------------------------------------------------- #


def _per_head_entropy_summary(diag: StateDiagnostics) -> dict[str, Any]:
    import statistics

    head_means: list[float] = []
    if diag.entropy_layer_head:
        ncols = max(len(r) for r in diag.entropy_layer_head)
        for c in range(ncols):
            col = [r[c] for r in diag.entropy_layer_head if c < len(r) and math.isfinite(r[c])]
            head_means.append(sum(col) / len(col) if col else float("nan"))
    elif diag.margin_layer_head:
        # tropical fallback: per-head mean margin stands in for "peakedness".
        ncols = max(len(r) for r in diag.margin_layer_head)
        for c in range(ncols):
            col = [r[c] for r in diag.margin_layer_head if c < len(r) and math.isfinite(r[c])]
            head_means.append(sum(col) / len(col) if col else float("nan"))
    finite = [x for x in head_means if math.isfinite(x)]
    return {
        "per_head": head_means,
        "head_entropy_mean": (sum(finite) / len(finite)) if finite else None,
        "head_entropy_std": (statistics.pstdev(finite) if len(finite) > 1 else 0.0) if finite else None,
        "route_diversity_js": head_route_diversity(diag),
        "route_coverage": diag.route_coverage,
        "signal": "attention_entropy_nats" if diag.entropy_layer_head else ("tropical_margin" if diag.margin_layer_head else "none"),
    }


def render_entropy_diversity(
    configs: list[tuple[str, StateDiagnostics]], out_dir: Path, *, console: Any = None
) -> dict[str, Any]:
    """7ow: per-head entropy + cross-head route diversity for baseline vs feature."""
    console = console or _console()
    out_dir.mkdir(parents=True, exist_ok=True)
    from rich.table import Table

    rows: dict[str, dict[str, Any]] = {}
    for name, diag in configs:
        rows[name] = _per_head_entropy_summary(diag)

    table = Table(title="per-head entropy & route diversity", header_style="bold cyan")
    table.add_column("config")
    table.add_column("signal", style="dim")
    table.add_column("head entropy μ", justify="right")
    table.add_column("head entropy σ", justify="right")
    table.add_column("route diversity (JS)", justify="right")
    table.add_column("coverage", justify="right")
    for name, s in rows.items():
        def _f(v: Any, spec: str = ".4f") -> str:
            return format(v, spec) if isinstance(v, (int, float)) else "—"
        table.add_row(name, str(s["signal"]), _f(s["head_entropy_mean"]), _f(s["head_entropy_std"]),
                      _f(s["route_diversity_js"]), _f(s["route_coverage"]))
    console.print(table)

    # grouped bar plot of per-head signal for each config
    png = out_dir / "per_head_entropy_diversity.png"
    _grouped_bar_png(rows, png)

    summary = {
        "schema": "mgr.viz.entropy.v1",
        "configs": rows,
        "images": [png.name],
        "interpretation": (
            "head entropy μ = mean per-head attention entropy (nats; higher = flatter/less selective). "
            "head entropy σ = spread across heads (head specialization). "
            "route diversity (JS) = mean pairwise Jensen-Shannon divergence between heads' attention maps "
            "(0 = redundant heads, →1 = heads route identical tokens to different places). "
            "coverage = fraction of tropical routes above the β route-stability threshold."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_index_html(out_dir, "per-head entropy & route diversity", [png], summary)
    console.print(f"[bold]wrote[/bold] entropy/diversity report -> [cyan]{out_dir}/[/cyan]")
    return summary


def _grouped_bar_png(rows: dict[str, dict[str, Any]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(rows)
    max_heads = max((len(rows[n]["per_head"]) for n in names), default=0)
    x = np.arange(max_heads)
    width = 0.8 / max(1, len(names))
    fig, ax = plt.subplots(figsize=(max(5.0, 0.8 * max_heads + 2), 4.0))
    for i, n in enumerate(names):
        vals = rows[n]["per_head"] + [float("nan")] * (max_heads - len(rows[n]["per_head"]))
        ax.bar(x + i * width, vals, width, label=f"{n} ({rows[n]['signal']})")
    ax.set_xlabel("head index")
    ax.set_ylabel("per-head signal")
    ax.set_title("Per-head entropy / margin: baseline vs math feature")
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels([f"h{i}" for i in range(max_heads)])
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def run_entropy(args: argparse.Namespace) -> int:
    console = _console()
    out_dir = Path(args.out) if args.out else (DEFAULT_VIS_ROOT / "entropy" / f"{args.baseline}_vs_{args.feature}")
    configs: list[tuple[str, StateDiagnostics]] = []
    for name in (args.baseline, args.feature):
        model, meta = build_probe_model(
            name, device=args.device, seed=args.seed, n_layer=args.n_layer,
            n_head=args.n_head, n_kv_head=args.n_kv_head, n_embd=args.n_embd,
            sequence_len=args.seq_len, vocab_size=args.vocab_size,
        )
        vocab = int(meta["config"]["vocab_size"])
        idx, labels = sample_batch(
            text=(None if args.random_input else args.text), batch_size=args.batch_size,
            seq_len=args.seq_len, vocab_size=vocab, seed=args.seed, device=args.device,
        )
        diag = collect_state(model, idx, example_idx=0, token_labels=labels)
        diag.attention_type = name
        configs.append((name, diag))
    render_entropy_diversity(configs, out_dir, console=console)
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="cpu", help="cpu | cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=None)
    p.add_argument("--n-kv-head", type=int, default=None)
    p.add_argument("--n-embd", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--text", default=DEFAULT_TEXT, help="prompt text encoded for the sample batch")
    p.add_argument("--random-input", action="store_true", help="ignore --text; use seeded random token ids")
    p.add_argument("--out", default=None, help="output directory (default under artifacts/vis/)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m nanochat.viz", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("state", help="render >=3 model-state visualizations (hi3)")
    ps.add_argument("--attention", default="standard", help="attention type for a fresh probe model")
    ps.add_argument("--checkpoint", default=None, help="load a trained checkpoint dir instead of a fresh model")
    ps.add_argument("--step", type=int, default=None, help="checkpoint step (default latest)")
    _add_common(ps)
    ps.set_defaults(func=run_state)

    pe = sub.add_parser("entropy", help="per-head entropy & route diversity, baseline vs feature (7ow)")
    pe.add_argument("--baseline", default="standard", help="baseline attention type")
    pe.add_argument("--feature", default="tropical", help="math-feature attention type")
    _add_common(pe)
    pe.set_defaults(func=run_entropy)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
