"""
Minimal PyTorch training loop for Nanochat.

Intentionally lightweight: small model, short run, and a streaming parquet-backed dataloader.
"""

import argparse
import json
import math
import os
import random
import shlex
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
from rich import box
from rich.console import Console
from rich.table import Table
from torch.nn.parallel import DistributedDataParallel as DDP

from nanochat.checkpoint_manager import (
    checkpoint_file_paths,
    find_last_step,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
    verify_checkpoint_roundtrip,
)
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, print0
from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
from nanochat.dataset import ensure_min_parquet_files, list_parquet_files
from nanochat.gpt import (
    GPT,
    SUPPORTED_ATTENTION_TYPES,
    GPTConfig,
    attention_schedule_label,
    resolve_attention_schedule,
)
from nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from nanochat.loss_eval import evaluate_loss_and_bpb
from nanochat.ordinal_scheduler import OrdinalLRScheduler
from nanochat.profiling import ProfileConfig, render_profile_table, summarize_profile, torch_profiler
from nanochat.report import (
    MetricsStream,
    TrainingDashboard,
    build_provenance,
    get_git_info,
    get_gpu_info,
    get_system_info,
)
from nanochat.synaptic import SynapticConfig
from nanochat.tokenizer import (
    TASK_TOKENIZER_DEFAULT_VOCAB,
    TASK_TOKENIZER_DIRNAME,
    TASK_TOKENIZER_MIN_VOCAB,
    HuggingFaceTokenizer,
    get_tokenizer,
    padded_vocab_size,
    train_task_tokenizer,
)
from nanochat.tropical_attention_torch import set_semiring_beta

console = Console()

_SUPPORTED_MODEL_TYPES = ("gpt", "synaptic")
_SUPPORTED_OPTIMIZER_TYPES = ("adamw", "muon", "hoss")
_SUPPORTED_SCHEDULER_TYPES = ("none", "ordinal")
_SUPPORTED_TOKENIZERS = ("gpt2", "task")
_SUPPORTED_ATTENTION_TYPES = SUPPORTED_ATTENTION_TYPES

# FFN structure variants (bead 8gk.8): the semiring axis extended to the MLP.
_SUPPORTED_FFN_TYPES = ("standard", "tropical", "tropical-rational")


def _select_env_vars() -> dict[str, str]:
    prefixes = ("CUDA", "NCCL", "TORCH", "PYTORCH", "NANOCHAT")
    selected = {k: v for k, v in os.environ.items() if k.startswith(prefixes)}
    return dict(sorted(selected.items()))


def _write_artifacts(run_dir: Path, *, summary: dict[str, Any], report_md: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "run.md").write_text(report_md, encoding="utf-8")


def _attention_spec_mentions(spec: Any, mechanism: str) -> bool:
    if spec is None:
        return False
    if isinstance(spec, list | tuple):
        parts: list[str] = []
        for item in spec:
            parts.extend(str(item).split(","))
    else:
        parts = str(spec).split(",")
    return any(part.strip() == mechanism for part in parts)


def _parse_optional_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"auto", "none"}:
        return None
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value {value!r}; use true/false/auto.")


def _normalize_ca_rule(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "none", "off", "false", "0"}:
        return None
    if normalized in {"rule30", "30"}:
        return "rule30"
    if normalized in {"rule116", "116"}:
        return "rule116"
    raise ValueError(f"Invalid CA rule {value!r}; use none|rule30|rule116.")


def _summarize_nonfinite(t: torch.Tensor) -> dict[str, Any]:
    t = t.detach()
    if t.numel() == 0:
        return {"shape": tuple(t.shape), "dtype": str(t.dtype), "device": str(t.device), "numel": 0}
    is_finite = torch.isfinite(t)
    bad = ~is_finite
    bad_count = int(bad.sum().item())
    nan_count = int(torch.isnan(t).sum().item()) if (torch.is_floating_point(t) or torch.is_complex(t)) else 0
    inf_count = int(torch.isinf(t).sum().item()) if (torch.is_floating_point(t) or torch.is_complex(t)) else 0
    finite_vals = t[is_finite]
    finite_stats = finite_vals.abs() if torch.is_complex(finite_vals) else finite_vals
    if finite_stats.numel():
        finite_min = float(finite_stats.min().item())
        finite_max = float(finite_stats.max().item())
        finite_mean = float(finite_stats.float().mean().item())
    else:
        finite_min = float("nan")
        finite_max = float("nan")
        finite_mean = float("nan")
    return {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "numel": int(t.numel()),
        "nonfinite": bad_count,
        "nan": nan_count,
        "inf": inf_count,
        "finite_min": finite_min,
        "finite_max": finite_max,
        "finite_mean": finite_mean,
    }


def _load_synaptic_config(path: Path) -> SynapticConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"--synaptic-config must be a JSON object, got {type(raw).__name__}")

    allowed = {f.name for f in SynapticConfig.__dataclass_fields__.values()}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"--synaptic-config contains unknown SynapticConfig keys: {unknown}")

    cfg = SynapticConfig()
    for key, value in raw.items():
        setattr(cfg, key, value)
    return cfg


_SemiringSpec = tuple[str, float, float] | tuple[str, float, float, float]


def _parse_semiring_beta_spec(spec: str) -> _SemiringSpec:
    """--semiring-beta spec parser (beads 8gk.1 + 9jzb).

    'B'                      -> ('const', B, B)
    'linear:B0:B1'           -> open-loop linear anneal B0 -> B1
    'exp:B0:B1'              -> open-loop exponential anneal
    'coverage:B0:BMAX:FLOOR' -> CLOSED-LOOP certificate-driven control: raise
        beta only while measured route-stability coverage stays >= FLOOR,
        back off otherwise (requires --tropical-record-margins for the
        coverage telemetry). The schedule anneals as fast as the certificate
        allows instead of on a clock.
    'ordinal:B0:B1'          -> ORDINAL-ORCHESTRATED (T3.3 client): beta
        doubles toward B1 at each ordinal-scheduler transition (any (A, B)
        descent - an anneal or a restart, i.e. the scheduler's limit-ordinal
        events). Requires --scheduler-type ordinal.
    """
    s = str(spec).strip()
    if ":" in s:
        mode, _, rest = s.partition(":")
        mode = mode.strip().lower()
        parts = rest.split(":")
        if mode in ("linear", "exp", "ordinal"):
            if len(parts) != 2:
                raise ValueError(f"--semiring-beta {mode} schedule needs B0:B1, got {rest!r}")
            b0, b1 = float(parts[0]), float(parts[1])
            if not (b0 > 0 and b1 > 0):
                raise ValueError(f"--semiring-beta schedule endpoints must be > 0, got {b0}, {b1}")
            return mode, b0, b1
        if mode == "coverage":
            if len(parts) != 3:
                raise ValueError(f"--semiring-beta coverage schedule needs B0:BMAX:FLOOR, got {rest!r}")
            b0, bmax, floor = float(parts[0]), float(parts[1]), float(parts[2])
            if not (b0 > 0 and bmax >= b0):
                raise ValueError(f"--semiring-beta coverage needs 0 < B0 <= BMAX, got {b0}, {bmax}")
            if not (0.0 < floor < 1.0):
                raise ValueError(f"--semiring-beta coverage FLOOR must be in (0, 1), got {floor}")
            return "coverage", b0, bmax, floor
        raise ValueError(f"--semiring-beta schedule mode must be linear | exp | ordinal | coverage, got {mode!r}")
    b = float(s)
    if not (b > 0):
        raise ValueError(f"--semiring-beta must be > 0, got {b}")
    return "const", b, b


class _OrdinalBetaLadder:
    """Ordinal-orchestrated dequantization annealing (9jzb, T3.3 client):
    beta doubles toward BMAX at each ordinal-scheduler transition - any
    descent of the (A, B) counters, i.e. an anneal or a restart, the
    scheduler's limit-ordinal events. Between events beta holds: the
    transfinite clock, not the step clock, drives the homotopy.
    """

    RATIO = 2.0  # one beta doubling per ordinal event

    def __init__(self, b0: float, bmax: float):
        self.beta = float(b0)
        self.bmax = float(bmax)
        self._last_ab: tuple[int, int] | None = None

    def step(self, a: int, b: int) -> float:
        ab = (int(a), int(b))
        if self._last_ab is not None and ab != self._last_ab:
            self.beta = min(self.beta * self.RATIO, self.bmax)
        self._last_ab = ab
        return self.beta

    def state_dict(self) -> dict[str, Any]:
        """Round-trip payload (efzy): everything `step` reads or writes, in
        weights-only-safe types (lists, not tuples)."""
        return {"beta": self.beta, "last_ab": None if self._last_ab is None else list(self._last_ab)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.beta = float(state["beta"])
        last_ab = state.get("last_ab")
        self._last_ab = None if last_ab is None else (int(last_ab[0]), int(last_ab[1]))


class _CoverageBetaController:
    """Closed-loop dequantization annealing (9jzb): certificate-driven beta
    control. Per step, given the PREVIOUS forward's route-stability coverage:
    raise beta multiplicatively while coverage holds at or above the floor,
    back off when it dips, hold while there is no reading. INVARIANT (the
    preregistration's checkable-from-logs clause): beta is never RAISED on a
    step whose observed coverage is below the floor or missing.
    """

    UP = 1.02  # multiplicative climb per covered step (~35 steps per beta doubling)
    DOWN = 1.05  # divisor per uncovered step (back-off is faster than climb)

    def __init__(self, b0: float, bmax: float, floor: float):
        self.beta = float(b0)
        self.b0 = float(b0)
        self.bmax = float(bmax)
        self.floor = float(floor)

    def step(self, coverage: float | None) -> float:
        if coverage is not None and coverage >= self.floor:
            self.beta = min(self.beta * self.UP, self.bmax)
        elif coverage is not None:
            self.beta = max(self.beta / self.DOWN, self.b0)
        # coverage None -> hold (first step, or margins momentarily silent)
        return self.beta

    def state_dict(self) -> dict[str, Any]:
        """Round-trip payload (efzy): beta is the only mutable state. The
        checkpoint step's pending coverage reading is saved BESIDE this by the
        trainer (the live route-coverage buffers are non-persistent)."""
        return {"beta": self.beta}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.beta = float(state["beta"])


def _lr_warmdown_multiplier(step: int, max_steps: int, warmdown_ratio: float) -> float:
    """bio-parity WSD tail (bead kj8s): hold the LR flat, then decay linearly
    to 0 over the final ``warmdown_ratio`` fraction of the run (bio's
    get_lr_multiplier with warmup_ratio=0.0). Pure function of
    (step, max_steps): resumed runs land on the same trajectory without any
    extra checkpoint state."""
    if warmdown_ratio <= 0.0 or max_steps <= 1:
        return 1.0
    warmdown_start = (1.0 - warmdown_ratio) * max_steps
    if step < warmdown_start:
        return 1.0
    span = max(1, (max_steps - 1) - warmdown_start)
    frac = min((step - warmdown_start) / span, 1.0)
    return 1.0 - frac


def _semiring_beta_at(schedule: _SemiringSpec, step: int, max_steps: int) -> float:
    mode, b0, b1 = str(schedule[0]), float(schedule[1]), float(schedule[2])
    frac = min(max(step / max(max_steps - 1, 1), 0.0), 1.0)
    if mode == "linear":
        return b0 + (b1 - b0) * frac
    if mode == "exp":
        return float(b0 * (b1 / b0) ** frac)
    return b0  # "const"


def _collect_tropical_route_coverage(model: torch.nn.Module) -> float | None:
    """Mean certificate coverage across tropical attention layers (8gk.1):
    the fraction of routes whose tropical margin clears the route-stability
    threshold at the current beta. None when no layer has a finite reading."""
    vals: list[float] = []
    for module in model.modules():
        cov = getattr(module, "tropical_route_coverage", None)
        if torch.is_tensor(cov) and cov.numel() == 1:
            f = float(cov.item())
            if math.isfinite(f):
                vals.append(f)
    return (sum(vals) / len(vals)) if vals else None


def _collect_hyperbolic_stats(model: torch.nn.Module) -> dict[str, Any] | None:
    """Collect per-head curvature and radius from hyperbolic attention layers.

    Curvature is the mechanism-internal hierarchy readout: movement toward
    zero is the Euclidean null, while movement above the registered hierarchy
    threshold is evidence that a head is using negative curvature. Radius
    records whether the learned chart actually occupies hyperbolic depth.
    """
    from nanochat.hyperbolic_attention_torch import (
        EUCLIDEAN_CURVATURE_THRESHOLD,
        HIERARCHY_CURVATURE_THRESHOLD,
    )

    curvature_layers: list[torch.Tensor] = []
    radius_layers: list[torch.Tensor] = []
    for module in model.modules():
        curvature = getattr(module, "hyperbolic_curvature_head", None)
        radius = getattr(module, "hyperbolic_radius_head_mean", None)
        if not torch.is_tensor(curvature) or curvature.ndim != 1:
            continue
        curvature_f = curvature.detach().float().cpu()
        if not torch.isfinite(curvature_f).all():
            continue
        curvature_layers.append(curvature_f)
        if torch.is_tensor(radius) and radius.ndim == 1:
            radius_f = radius.detach().float().cpu()
            if torch.isfinite(radius_f).all():
                radius_layers.append(radius_f)
    if not curvature_layers:
        return None

    curvature_all = torch.cat(curvature_layers)
    stats: dict[str, Any] = {
        "curvature": {
            "layer_head": [[float(x) for x in layer.tolist()] for layer in curvature_layers],
            "mean": float(curvature_all.mean()),
            "min": float(curvature_all.min()),
            "max": float(curvature_all.max()),
        },
        "frac_heads_hier": float((curvature_all >= HIERARCHY_CURVATURE_THRESHOLD).float().mean()),
        "frac_heads_euclidean": float((curvature_all <= EUCLIDEAN_CURVATURE_THRESHOLD).float().mean()),
        "hierarchy_curvature_threshold": HIERARCHY_CURVATURE_THRESHOLD,
        "euclidean_curvature_threshold": EUCLIDEAN_CURVATURE_THRESHOLD,
    }
    if radius_layers:
        radius_all = torch.cat(radius_layers)
        stats["radius"] = {
            "layer_head": [[float(x) for x in layer.tolist()] for layer in radius_layers],
            "mean": float(radius_all.mean()),
            "min": float(radius_all.min()),
            "max": float(radius_all.max()),
        }
    return stats


def _collect_braid_charge_stats(model: torch.nn.Module) -> dict[str, Any] | None:
    """Collect the latest conserved-charge readings from braid attention layers
    (bead u55.3): Q1 = mass-partition defect (exactly 0 for the rmatrix law, the
    stochastic-gauge conservation theorem; measurably nonzero for the additive
    heuristic modes) and Q2 = path-independence (braid relation) residual of the
    layer's live crossing law. Per-head eta and rapidity spans ride along when
    the law is rmatrix."""
    q1: list[float] = []
    q2: list[float] = []
    etas: list[list[float]] = []
    spans: list[list[float]] = []
    law: str | None = None
    for module in model.modules():
        charges = getattr(module, "last_braid_charges", None)
        if not isinstance(charges, dict):
            continue
        d1 = charges.get("q1_mass_defect")
        d2 = charges.get("q2_braid_residual")
        if isinstance(d1, float) and math.isfinite(d1):
            q1.append(d1)
        if isinstance(d2, float) and math.isfinite(d2):
            q2.append(d2)
        if isinstance(charges.get("eta"), list):
            etas.append([float(x) for x in charges["eta"]])
        if isinstance(charges.get("rapidity_span"), list):
            spans.append([float(x) for x in charges["rapidity_span"]])
        law = str(charges.get("crossing_law")) if charges.get("crossing_law") is not None else law
    if not q1 and not q2:
        return None
    stats: dict[str, Any] = {
        "crossing_law": law,
        "q1_mass_defect_max": max(q1) if q1 else None,
        "q2_braid_residual_max": max(q2) if q2 else None,
    }
    if etas:
        stats["eta_per_layer"] = etas
    if spans:
        stats["rapidity_span_per_layer"] = spans
    return stats


def _collect_symplectic_energy_stats(model: torch.nn.Module) -> dict[str, Any] | None:
    """Drain the per-layer shadow-energy traces from reversible blocks (bead
    u55.5). A TIED block appears once in modules() but its trace carries one
    entry per layer call, so dedupe by identity and concatenate traces in
    call order. The across-layer band of the shadow energy within one step is
    the conservation observable (tied: the same symplectic map iterated)."""
    seen: set[int] = set()
    entries: list[dict[str, float]] = []
    for module in model.modules():
        drain = getattr(module, "drain_energy_trace", None)
        if drain is None or id(module) in seen:
            continue
        seen.add(id(module))
        entries.extend(drain())
    finite = [e for e in entries if math.isfinite(e.get("shadow_energy", float("nan")))]
    if not finite:
        return None
    energies = [e["shadow_energy"] for e in finite]
    x_norms = [e["x1_norm"] + e["x2_norm"] for e in finite]
    lams = [0.5 * (e["lambda_f"] + e["lambda_g"]) for e in finite]
    return {
        "shadow_energy_mean": sum(energies) / len(energies),
        "energy_band_layers": max(energies) - min(energies),
        "x_norm_mean": sum(x_norms) / len(x_norms),
        "lambda_mean": sum(lams) / len(lams),
        "layers_recorded": len(finite),
    }


def _collect_tropical_margin_stats(model: torch.nn.Module) -> dict[str, Any] | None:
    """Collect latest tropical margin stats from attention modules (if enabled)."""
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        return None
    blocks = getattr(model, "h", None)
    if blocks is None:
        try:
            blocks = transformer["h"]
        except Exception:
            return None

    layer_mins: list[float] = []
    head_means: list[torch.Tensor] = []
    head_mins: list[torch.Tensor] = []
    for block in blocks:
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        gamma_min = getattr(attn, "tropical_gamma_min", None)
        if not torch.is_tensor(gamma_min) or gamma_min.numel() != 1:
            continue
        gamma_min_val = float(gamma_min.detach().float().item())
        if math.isnan(gamma_min_val):
            continue
        layer_mins.append(gamma_min_val)

        gamma_head_mean = getattr(attn, "tropical_gamma_head_mean", None)
        if torch.is_tensor(gamma_head_mean) and gamma_head_mean.ndim == 1:
            head_means.append(gamma_head_mean.detach().float())
        gamma_head_min = getattr(attn, "tropical_gamma_head_min", None)
        if torch.is_tensor(gamma_head_min) and gamma_head_min.ndim == 1:
            head_mins.append(gamma_head_min.detach().float())

    if not layer_mins:
        return None

    stats: dict[str, Any] = {
        "layer_min": layer_mins,
        "gamma_min": min(layer_mins),
        "gamma_mean": sum(layer_mins) / len(layer_mins),
    }
    if head_means:
        mean = torch.stack(head_means, dim=0).mean(dim=0)
        stats["head_mean"] = [float(x) for x in mean.cpu().tolist()]
    if head_mins:
        amin = torch.stack(head_mins, dim=0).amin(dim=0)
        stats["head_min"] = [float(x) for x in amin.cpu().tolist()]
    return stats


def _transformer_blocks(model: torch.nn.Module) -> list[torch.nn.Module] | None:
    """The residual blocks of a GPT-style model (``model.transformer.h``), else None."""
    transformer = getattr(model, "transformer", None)
    blocks = getattr(transformer, "h", None) if transformer is not None else None
    return list(blocks) if blocks is not None else None


def _collect_block_grad_norms(model: torch.nn.Module) -> list[float] | None:
    """Per-block L2 gradient norm: the per-layer gradient telemetry the
    gradient-stability hypotheses wait on (hyp-gauge-gradient-stability,
    hyp-reversible-gradient-stability). Call after backward and before the next
    zero_grad. Tied blocks share parameters and therefore report equal norms."""
    blocks = _transformer_blocks(model)
    if blocks is None:
        return None
    out: list[float] = []
    for block in blocks:
        grads = [p.grad for p in block.parameters() if p.grad is not None]
        out.append(float(torch.nn.utils.get_total_norm(grads)) if grads else 0.0)
    return out


def _install_activation_rms_hooks(model: torch.nn.Module) -> dict[str, Any] | None:
    """Forward hooks that record every block's output RMS, the per-layer
    activation-magnitude telemetry hyp-octonion-norm-stability waits on.

    Readings are kept as 0-dim tensors and converted at log time, so the hooks
    add no per-forward device sync. A cursor assigns readings to block
    positions in forward order, which is what makes tied (shared) blocks
    report one value per position instead of overwriting each other. Not
    installed under torch.compile (module hooks and dynamo do not mix)."""
    blocks = _transformer_blocks(model)
    if not blocks:
        return None
    n_blocks = len(blocks)
    state: dict[str, Any] = {"rms": [None] * n_blocks, "cursor": 0, "handles": []}

    def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        if isinstance(tensor, torch.Tensor):
            state["rms"][state["cursor"]] = tensor.detach().float().pow(2).mean().sqrt()
        state["cursor"] = (state["cursor"] + 1) % n_blocks

    for block in {id(b): b for b in blocks}.values():  # one hook per distinct module
        state["handles"].append(block.register_forward_hook(hook))
    return state


def _summarize_depth_telemetry(
    grad_norms: list[float],
    block_grad_norms: list[list[float]],
    activation_rms: list[list[float]],
) -> dict[str, Any] | None:
    """Summary statistics over the logged steps for the depth-telemetry
    observables (metric paths under results.depth_telemetry). Ratios use the
    LAST logged reading; spike_ratio is max/median of the total gradient norm
    over the run (1.0 = perfectly flat)."""
    import statistics as st

    if not grad_norms:
        return None
    finite = [g for g in grad_norms if math.isfinite(g)]
    out: dict[str, Any] = {
        "grad_norm_max": max(finite) if finite else None,
        "grad_norm_median": st.median(finite) if finite else None,
        "grad_norm_spike_ratio": (max(finite) / st.median(finite)) if finite and st.median(finite) > 0 else None,
        "logged_steps": len(grad_norms),
    }

    def _by_block(series: list[list[float]], key: str) -> None:
        rows = [row for row in series if row and all(isinstance(x, int | float) and math.isfinite(x) for x in row)]
        if not rows:
            out[f"{key}_by_block_final"] = None
            out[f"{key}_by_block_mean"] = None
            out[f"{key}_depth_ratio"] = None
            return
        final = list(rows[-1])
        out[f"{key}_by_block_final"] = final
        out[f"{key}_by_block_mean"] = [st.fmean(col) for col in zip(*rows, strict=True)]
        out[f"{key}_depth_ratio"] = (final[-1] / final[0]) if len(final) > 1 and final[0] > 0 else None

    _by_block(block_grad_norms, "grad_norm")
    _by_block(activation_rms, "activation_rms")
    return out


def _collect_attn_entropy_stats(model: torch.nn.Module) -> dict[str, Any] | None:
    """Collect latest per-head attention entropy stats from standard attention modules (if enabled)."""
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        return None
    blocks = getattr(model, "h", None)
    if blocks is None:
        try:
            blocks = transformer["h"]
        except Exception:
            return None

    head_means: list[torch.Tensor] = []
    layer_head: list[list[float]] = []
    for block in blocks:
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        entropy = getattr(attn, "attn_entropy_head_mean", None)
        if not torch.is_tensor(entropy) or entropy.ndim != 1:
            continue
        entropy_f = entropy.detach().float().cpu()
        if entropy_f.numel() == 0:
            continue
        if not torch.isfinite(entropy_f).any():
            # Buffer exists for standard attention, but values stay NaN unless the feature is enabled.
            continue
        layer_head.append([float(x) for x in entropy_f.tolist()])
        head_means.append(entropy_f)

    if not head_means:
        return None

    mean = torch.stack(head_means, dim=0).mean(dim=0)
    return {
        "layer_head_mean": layer_head,
        "head_mean": [float(x) for x in mean.tolist()],
    }


def _extract_loss(output: object) -> torch.Tensor:
    if isinstance(output, tuple):
        if len(output) != 2:
            raise TypeError(f"Expected model output to be loss or (logits, loss), got tuple(len={len(output)})")
        loss = output[1]
        if loss is None:
            raise TypeError("Model returned (logits, None) during training; expected a loss tensor.")
        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"Expected loss to be torch.Tensor, got {type(loss).__name__}")
        return loss
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected model output to be torch.Tensor loss, got {type(output).__name__}")
    return output


def _validate_train_args(args, *, ddp_rank: int, device: torch.device) -> None:
    errors: list[str] = []
    warnings: list[str] = []

    model_type = str(getattr(args, "model_type", "gpt"))
    if model_type not in _SUPPORTED_MODEL_TYPES:
        errors.append(f"--model-type must be one of: {', '.join(_SUPPORTED_MODEL_TYPES)}")

    if int(getattr(args, "batch_size", 0)) < 1:
        errors.append("--batch-size must be >= 1")
    if int(getattr(args, "sequence_len", 0)) < 1:
        errors.append("--sequence-len must be >= 1")
    if int(getattr(args, "n_layer", 0)) < 1:
        errors.append("--n-layer must be >= 1")
    if int(getattr(args, "n_head", 0)) < 1:
        errors.append("--n-head must be >= 1")
    if int(getattr(args, "n_kv_head", 0)) < 1:
        errors.append("--n-kv-head must be >= 1")
    if int(getattr(args, "n_embd", 0)) < 1:
        errors.append("--n-embd must be >= 1")

    n_head = int(getattr(args, "n_head", 0))
    n_embd = int(getattr(args, "n_embd", 0))
    if n_head > 0 and n_embd > 0 and (n_embd % n_head != 0):
        errors.append(f"--n-embd must be divisible by --n-head (got n_embd={n_embd}, n_head={n_head})")

    n_kv_head = int(getattr(args, "n_kv_head", 0))
    if n_head > 0 and n_kv_head > 0 and not (n_kv_head <= n_head and n_head % n_kv_head == 0):
        errors.append(
            f"--n-kv-head must divide --n-head and be <= --n-head (got n_kv_head={n_kv_head}, n_head={n_head})"
        )

    optimizer_type = str(getattr(args, "optimizer_type", "adamw"))
    if optimizer_type not in _SUPPORTED_OPTIMIZER_TYPES:
        errors.append(f"--optimizer-type must be one of: {', '.join(_SUPPORTED_OPTIMIZER_TYPES)}")

    scheduler_type = str(getattr(args, "scheduler_type", "none"))
    if scheduler_type not in _SUPPORTED_SCHEDULER_TYPES:
        errors.append(f"--scheduler-type must be one of: {', '.join(_SUPPORTED_SCHEDULER_TYPES)}")
    lr_warmdown_ratio = float(getattr(args, "lr_warmdown_ratio", 0.0) or 0.0)
    if not (0.0 <= lr_warmdown_ratio < 1.0):
        errors.append("--lr-warmdown-ratio must be in [0.0, 1.0)")
    elif lr_warmdown_ratio > 0.0 and scheduler_type == "ordinal":
        errors.append(
            "--lr-warmdown-ratio owns the LR tail for flat-LR runs and conflicts with "
            "--scheduler-type ordinal (the ordinal scheduler drives the LR); use one or the other"
        )

    tokenizer_kind = str(getattr(args, "tokenizer", "gpt2"))
    if tokenizer_kind not in _SUPPORTED_TOKENIZERS:
        errors.append(f"--tokenizer must be one of: {', '.join(_SUPPORTED_TOKENIZERS)}")
    elif tokenizer_kind == "task":
        if getattr(args, "data_dir", None) is None:
            errors.append("--tokenizer task trains its vocabulary from --data-dir; pass a parquet corpus directory")
        if int(getattr(args, "vocab_size", GPTConfig().vocab_size)) != GPTConfig().vocab_size:
            errors.append("--vocab-size is derived from the task tokenizer; do not pass it with --tokenizer task")
        if int(getattr(args, "tokenizer_vocab_size", 0)) < TASK_TOKENIZER_MIN_VOCAB:
            errors.append(
                f"--tokenizer-vocab-size must be >= {TASK_TOKENIZER_MIN_VOCAB} "
                f"(256 byte symbols + special tokens), got {getattr(args, 'tokenizer_vocab_size', None)}"
            )

    attention_type = str(getattr(args, "attention_type", "standard"))
    if attention_type not in _SUPPORTED_ATTENTION_TYPES:
        errors.append(f"--attention-type must be one of: {', '.join(_SUPPORTED_ATTENTION_TYPES)}")
    attention_schedule_arg = getattr(args, "attention_schedule", None)
    if attention_schedule_arg is not None and str(attention_schedule_arg).strip():
        if model_type == "synaptic":
            pass
        else:
            schedule_n_kv_head = n_kv_head
            if n_head > 0 and _attention_spec_mentions(attention_schedule_arg, "reversible"):
                schedule_n_kv_head = n_head // 2
            try:
                resolve_attention_schedule(
                    GPTConfig(
                        n_layer=int(getattr(args, "n_layer", 0)),
                        n_head=n_head,
                        n_kv_head=schedule_n_kv_head,
                        n_embd=n_embd,
                        attention_type=str(attention_schedule_arg),
                    )
                )
            except (TypeError, ValueError) as exc:
                errors.append(f"--attention-schedule invalid: {exc}")

    ffn_type = str(getattr(args, "ffn_type", "standard"))
    if ffn_type not in _SUPPORTED_FFN_TYPES:
        errors.append(f"--ffn-type must be one of: {', '.join(_SUPPORTED_FFN_TYPES)}")
    ffn_beta = getattr(args, "ffn_beta", None)
    if ffn_beta is not None and not (float(ffn_beta) > 0):
        errors.append(f"--ffn-beta must be > 0 when set (got {ffn_beta}); omit it for the exact tropical endpoint.")
    if ffn_beta is not None and ffn_type == "standard":
        warnings.append("--ffn-beta has no effect with --ffn-type standard.")
    semiring_beta = getattr(args, "semiring_beta", None)
    if semiring_beta is not None:
        try:
            _parse_semiring_beta_spec(semiring_beta)
        except ValueError as exc:
            errors.append(str(exc))

    if bool(getattr(args, "use_flex_attention", False)) and not hasattr(torch.nn.attention, "flex_attention"):
        errors.append("--use-flex-attention requires torch>=2.5 (missing torch.nn.attention.flex_attention).")

    data_dir_arg = getattr(args, "data_dir", None)
    if data_dir_arg is not None and not Path(str(data_dir_arg)).is_dir():
        errors.append(f"--data-dir must be an existing directory, got {data_dir_arg!r}")

    # Checkpoint/resume flags (bead rz8.1).
    if int(getattr(args, "checkpoint_interval", 0)) < 0:
        errors.append("--checkpoint-interval must be >= 0 (0 disables checkpointing)")
    if int(getattr(args, "checkpoint_keep", 0)) < 0:
        errors.append("--checkpoint-keep must be >= 0 (0 keeps all checkpoints)")
    resume_from = getattr(args, "resume_from", None)
    if resume_from is not None:
        if str(resume_from) == "latest":
            if getattr(args, "checkpoint_dir", None) is None:
                errors.append("--resume-from latest requires --checkpoint-dir (the directory to scan)")
        elif not Path(str(resume_from)).is_dir():
            errors.append(f"--resume-from must be 'latest' or an existing checkpoint directory, got {resume_from!r}")
    if getattr(args, "resume_step", None) is not None and resume_from is None:
        warnings.append("--resume-step has no effect without --resume-from.")
    if model_type == "synaptic" and resume_from is not None:
        errors.append("--resume-from is currently only supported for --model-type gpt.")

    syn_cfg_path = getattr(args, "synaptic_config", None)
    if model_type == "synaptic" and syn_cfg_path is not None:
        path = Path(str(syn_cfg_path))
        if not path.is_file():
            errors.append(f"--synaptic-config path does not exist or is not a file: {path}")

    if model_type == "synaptic":
        if optimizer_type == "hoss":
            errors.append("--optimizer-type hoss is not supported for --model-type synaptic (no HVP closure).")
        if attention_type != "standard":
            warnings.append("--attention-type is ignored for --model-type synaptic.")
        if attention_schedule_arg is not None and str(attention_schedule_arg).strip():
            warnings.append("--attention-schedule is ignored for --model-type synaptic.")
        if bool(getattr(args, "use_flex_attention", False)):
            warnings.append("--use-flex-attention is ignored for --model-type synaptic.")
    else:
        if syn_cfg_path is not None:
            warnings.append("--synaptic-config is only used with --model-type synaptic; ignoring.")

    # Artifact path segments: --run-id is joined as ONE directory under
    # <artifacts-dir>/<kind>/<topic>/, and kind/topic may nest but must stay
    # inside the artifacts tree. Unchecked, '--run-id ../x' or an absolute
    # topic wrote summary.json/metrics.jsonl wherever the caller pointed.
    run_id_arg = getattr(args, "run_id", None)
    if run_id_arg is not None:
        rid = str(run_id_arg)
        if not rid.strip() or rid in {".", ".."} or "/" in rid or "\\" in rid or rid != rid.strip():
            errors.append(f"--run-id must be a single non-empty path segment, got {rid!r}")
    for flag in ("artifacts_kind", "artifacts_topic"):
        val = getattr(args, flag, None)
        if val is None:
            continue
        sval = str(val)
        segments = sval.replace("\\", "/").split("/")
        if not sval.strip() or sval.startswith("/") or any(seg in {"", ".", ".."} for seg in segments):
            errors.append(
                f"--{flag.replace('_', '-')} must be a relative path with no empty or '..' segments, got {sval!r}"
            )

    if errors:
        if ddp_rank == 0:
            console.print("[bold red]Invalid configuration[/bold red]")
            for msg in errors:
                console.print(f"[red]- {msg}[/red]")
            if warnings:
                console.print("[bold yellow]Additional notes[/bold yellow]")
                for msg in warnings:
                    console.print(f"[yellow]- {msg}[/yellow]")
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))

    if warnings and ddp_rank == 0:
        for msg in warnings:
            console.print(f"[yellow]warning[/yellow] {msg}")


def train(args) -> None:
    # Init distributed mode if necessary
    device_type = args.device
    if device_type == "auto":
        device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type=device_type, seed=args.seed)
    if ddp and device.type != "cuda":
        raise RuntimeError("DDP env detected, but distributed training is currently only supported on CUDA.")

    check_numerics = bool(getattr(args, "check_numerics", False))
    detect_anomaly = bool(getattr(args, "detect_anomaly", False))
    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True, check_nan=True)
        if ddp_rank == 0:
            console.print("[bold yellow]autograd anomaly detection enabled[/bold yellow] (very slow; debug only)")

    # Config
    model_type = str(getattr(args, "model_type", "gpt"))
    if model_type not in {"gpt", "synaptic"}:
        raise ValueError("--model-type must be one of: gpt, synaptic")

    compile_requested = bool(getattr(args, "compile", False))
    compile_backend = str(getattr(args, "compile_backend", "inductor"))
    compile_mode = getattr(args, "compile_mode", None)
    compile_fullgraph = bool(getattr(args, "compile_fullgraph", False))
    compile_dynamic = _parse_optional_bool(getattr(args, "compile_dynamic", "auto"))

    _validate_train_args(args, ddp_rank=ddp_rank, device=device)

    # ----- checkpoint/resume setup (bead rz8.1) -----
    checkpoint_interval = int(getattr(args, "checkpoint_interval", 0))
    checkpoint_keep = int(getattr(args, "checkpoint_keep", 0))
    checkpoint_verify = bool(getattr(args, "checkpoint_verify", False))
    resume_data_mode = str(getattr(args, "resume_data_mode", "exact"))

    resume_from = getattr(args, "resume_from", None)
    resume_meta: dict[str, Any] | None = None
    resume_train_state: dict[str, Any] | None = None
    resume_model_data: dict[str, Any] | None = None
    if resume_from is not None:
        resume_dir = str(getattr(args, "checkpoint_dir", None)) if str(resume_from) == "latest" else str(resume_from)
        resume_step_arg = getattr(args, "resume_step", None)
        resume_step = int(resume_step_arg) if resume_step_arg is not None else find_last_step(resume_dir)
        if ddp_rank == 0:
            console.print(f"[bold cyan]resume[/bold cyan] loading checkpoint step {resume_step} from {resume_dir}")
        t_load0 = time.perf_counter()
        resume_model_data, resume_train_state, resume_meta = load_checkpoint(
            resume_dir, resume_step, device, load_optimizer=True, rank=ddp_rank
        )
        if resume_train_state is None:
            raise RuntimeError(f"Checkpoint at {resume_dir} step {resume_step} has no rank-{ddp_rank} training state")
        # Defensive both-directions torch.compile key normalization: saves come
        # from the raw module (clean keys), but tolerate older artifacts.
        resume_model_data = {k.removeprefix("_orig_mod."): v for k, v in resume_model_data.items()}
        if ddp_rank == 0:
            load_bytes = sum(
                os.path.getsize(p)
                for p in checkpoint_file_paths(resume_dir, resume_step, rank=ddp_rank)
                if os.path.exists(p)
            )
            console.print(
                f"[bold cyan]resume[/bold cyan] read step={resume_step} "
                f"bytes={load_bytes:,} duration={time.perf_counter() - t_load0:.2f}s "
                f"components=model/optim/sched/dataloader/rng"
            )

    # Dequantization-annealing schedule (8gk.1): set while wiring the tropical
    # config below; consumed once per step in the training loop.
    semiring_schedule: _SemiringSpec | None = None
    semiring_controller: _CoverageBetaController | None = None
    semiring_ladder: _OrdinalBetaLadder | None = None
    # The checkpoint step's coverage reading, restored on resume (efzy): the
    # live route-coverage buffers are non-persistent (reload as nan), so the
    # resumed first step would otherwise see None (hold) where the
    # uninterrupted run consumes the checkpoint step's real reading.
    semiring_resume_coverage: float | None = None

    # Document order per epoch (bead r7qn): a small corpus replayed in file
    # order every epoch teaches the model which document follows which
    # instead of the task (the 1e12 copyops probe: train-stream loss 0.9 in
    # corpus order, 4.5 on the same documents shuffled). 'epoch' permutes each
    # row group's documents with a permutation seeded by (--seed, epoch,
    # position), so the stream stays a deterministic function of the seed.
    data_shuffle = str(getattr(args, "data_shuffle", "epoch"))
    data_shuffle_seed: int | None = int(args.seed) if data_shuffle == "epoch" else None

    # Task-scoped tokenizer (nanochat.tokenizer.train_task_tokenizer explains
    # why): trained from the corpus's train split BEFORE the config is built,
    # because the embedding table is sized from it. BPE training is
    # deterministic for a fixed corpus, so a resumed run rebuilds the identical
    # vocabulary and the resume config check still guards it.
    task_tokenizer: HuggingFaceTokenizer | None = None
    tokenizer_meta: dict[str, Any] = {"kind": "gpt2"}
    if str(getattr(args, "tokenizer", "gpt2")) == "task":
        corpus_files = list_parquet_files(str(args.data_dir))
        train_files = corpus_files[:-1] if len(corpus_files) > 1 else corpus_files  # last file is the val split
        task_tokenizer = train_task_tokenizer(train_files, int(args.tokenizer_vocab_size))
        tokenizer_meta = {
            "kind": "task",
            "dir": TASK_TOKENIZER_DIRNAME,
            "vocab_size": int(task_tokenizer.get_vocab_size()),
            "requested_vocab_size": int(args.tokenizer_vocab_size),
        }
        print0(
            f"[tokenizer] task-scoped BPE: {task_tokenizer.get_vocab_size()} tokens "
            f"(requested {args.tokenizer_vocab_size}) from {len(train_files)} train shard(s); "
            f"embedding table {padded_vocab_size(task_tokenizer.get_vocab_size())}"
        )

    attention_schedule: list[str] = []
    attention_label = model_type
    if model_type == "gpt":
        config = GPTConfig()
        config.vocab_size = (
            padded_vocab_size(task_tokenizer.get_vocab_size())
            if task_tokenizer is not None
            else int(getattr(args, "vocab_size", GPTConfig().vocab_size))
        )
        config.n_layer = args.n_layer
        config.n_head = args.n_head
        config.n_kv_head = args.n_kv_head
        config.parameterization = getattr(args, "parameterization", "current")
        config.activation_ckpt = getattr(args, "activation_checkpointing", "none")
        config.activation_ckpt_every_k = int(getattr(args, "activation_ckpt_every_k", 1))
        config.n_embd = args.n_embd
        config.sequence_len = args.sequence_len
        config.optimizer_type = args.optimizer_type
        attention_schedule_arg = getattr(args, "attention_schedule", None)
        config.attention_type = str(attention_schedule_arg).strip() if attention_schedule_arg else args.attention_type
        config.ffn_type = str(getattr(args, "ffn_type", "standard"))
        ffn_beta_arg = getattr(args, "ffn_beta", None)
        config.ffn_beta = float(ffn_beta_arg) if ffn_beta_arg is not None else None
        config.use_flex_attention = bool(getattr(args, "use_flex_attention", False))
        std_entropy = getattr(args, "standard_record_attn_entropy", None)
        if std_entropy is not None:
            config.standard_record_attn_entropy = bool(std_entropy)

        if _attention_spec_mentions(config.attention_type, "reversible"):
            if config.n_head % 2 != 0:
                raise ValueError("reversible attention requires n_head to be even")
            desired_n_kv_head = config.n_head // 2
            if config.n_kv_head != desired_n_kv_head and ddp_rank == 0:
                print0(
                    f"[reversible] Overriding n_kv_head from {config.n_kv_head} to {desired_n_kv_head} "
                    "to satisfy reversible KV-cache constraints."
                )
            config.n_kv_head = desired_n_kv_head

        attention_schedule = resolve_attention_schedule(config)
        attention_label = attention_schedule_label(config)
        has_standard = "standard" in attention_schedule
        has_tropical = "tropical" in attention_schedule
        has_ultrametric = "ultrametric" in attention_schedule
        has_reversible = "reversible" in attention_schedule
        has_braid = "braid" in attention_schedule

        if has_ultrametric:
            config.ultrametric_mode = str(getattr(args, "ultrametric_mode", config.ultrametric_mode))
            ultra_hard = getattr(args, "ultrametric_hard_digits", None)
            if ultra_hard is not None:
                config.ultrametric_hard_digits = bool(ultra_hard)
            ultra_k = getattr(args, "ultrametric_K", None)
            if ultra_k is not None:
                config.ultrametric_K = int(ultra_k)
        if has_standard:
            no_norms = getattr(args, "disable_block_norms", None)
            if no_norms is not None:
                config.disable_block_norms = bool(no_norms)
        if has_reversible:
            config.reversible_mode = str(getattr(args, "reversible_mode", config.reversible_mode))
            rev_tied = getattr(args, "reversible_tied", None)
            if rev_tied is not None:
                config.reversible_tied = bool(rev_tied)
            rev_lam = getattr(args, "reversible_lambda_min", None)
            if rev_lam is not None:
                config.reversible_lambda_min = float(rev_lam)
            rev_energy = getattr(args, "reversible_record_energy", None)
            if rev_energy is not None:
                config.reversible_record_energy = bool(rev_energy)
        if has_braid:
            config.braid_mode = str(getattr(args, "braid_mode", config.braid_mode))
            config.braid_tau = float(getattr(args, "braid_tau", config.braid_tau))
            config.braid_crossing_law = str(getattr(args, "braid_crossing_law", config.braid_crossing_law))
            braid_record = getattr(args, "braid_record_schedule", None)
            if braid_record is not None:
                config.braid_record_schedule = bool(braid_record)
            braid_verify = getattr(args, "braid_verify", None)
            if braid_verify is not None:
                config.braid_verify = bool(braid_verify)
            config.braid_rmatrix_probes = int(getattr(args, "braid_rmatrix_probes", config.braid_rmatrix_probes))
        if config.use_flex_attention and not has_standard:
            if ddp_rank == 0:
                print0("[flex] --use-flex-attention only applies to standard attention layers; disabling.")
            config.use_flex_attention = False
        if config.use_flex_attention and device.type != "cuda":
            if ddp_rank == 0:
                print0(f"[flex] FlexAttention requires CUDA; disabling (device={device.type}).")
            config.use_flex_attention = False
        if config.standard_record_attn_entropy and not has_standard:
            if ddp_rank == 0:
                print0("[entropy] --standard-record-attn-entropy only applies to standard attention layers; disabling.")
            config.standard_record_attn_entropy = False

        config.compile_backend = compile_backend
        config.compile_mode = compile_mode
        config.compile_fullgraph = compile_fullgraph
        config.compile_dynamic = compile_dynamic

        if config.use_flex_attention:
            compile_flex_flag = getattr(args, "compile_flex_attention", None)
            config.compile_flex_attention = (
                bool(compile_requested) if compile_flex_flag is None else bool(compile_flex_flag)
            )
        else:
            config.compile_flex_attention = False

        ca_rule = _normalize_ca_rule(getattr(args, "ca_init_rule", None))
        ca_alpha = float(getattr(args, "ca_init_alpha", 1.0))
        ca_seed = getattr(args, "ca_init_seed", None)
        if ca_seed is None:
            env_seed = os.environ.get("NANOCHAT_CA_INIT_SEED")
            ca_seed = int(env_seed) if env_seed is not None else int(args.seed)
        if ca_rule is not None:
            if not (0.0 <= ca_alpha <= 1.0):
                raise ValueError(f"--ca-init-alpha must be in [0, 1], got {ca_alpha}")
            if int(ca_seed) < 0:
                raise ValueError(f"--ca-init-seed must be non-negative, got {ca_seed}")
        config.ca_init_rule = ca_rule
        config.ca_init_alpha = ca_alpha
        config.ca_init_seed = int(ca_seed)

        # Tropical attention knobs (only relevant when --attention-type tropical).
        tropical_gauge_fix = getattr(args, "tropical_gauge_fix", None)
        tropical_score_center = getattr(args, "tropical_score_center", None)
        tropical_record_margins = getattr(args, "tropical_record_margins", None)
        tropical_log_margins = bool(getattr(args, "tropical_log_margins", False))
        semiring_beta_arg = getattr(args, "semiring_beta", None)
        if not has_tropical:
            if (
                tropical_gauge_fix is not None
                or tropical_score_center is not None
                or tropical_record_margins is not None
                or semiring_beta_arg is not None
            ) and ddp_rank == 0:
                print0("[tropical] Ignoring tropical flags because the attention schedule has no tropical layers.")
            if tropical_log_margins and ddp_rank == 0:
                print0(
                    "[tropical] Ignoring --tropical-log-margins because the attention schedule has no tropical layers."
                )
        else:
            if tropical_gauge_fix is not None:
                config.tropical_gauge_fix = bool(tropical_gauge_fix)
            if tropical_score_center is not None:
                config.tropical_score_center = bool(tropical_score_center)
            if tropical_record_margins is not None:
                config.tropical_record_margins = bool(tropical_record_margins)
            if tropical_log_margins:
                config.tropical_record_margins = True
            if semiring_beta_arg is not None:
                # Dequantization annealing (8gk.1/9jzb): constant beta lands
                # in the config; schedules start at b0 and update live modules
                # per step (see the semiring_schedule hook in the training loop).
                parsed = _parse_semiring_beta_spec(semiring_beta_arg)
                config.semiring_beta = float(parsed[1])
                if parsed[0] == "coverage" and len(parsed) == 4:
                    if not config.tropical_record_margins:
                        raise ValueError(
                            "--semiring-beta coverage:... is certificate-driven control: it needs the "
                            "coverage telemetry, so pass --tropical-record-margins as well."
                        )
                    semiring_controller = _CoverageBetaController(parsed[1], parsed[2], parsed[3])
                if parsed[0] == "ordinal":
                    if str(args.scheduler_type) != "ordinal":
                        raise ValueError(
                            "--semiring-beta ordinal:... is driven by the ordinal scheduler's "
                            "transitions, so pass --scheduler-type ordinal as well."
                        )
                    semiring_ladder = _OrdinalBetaLadder(float(parsed[1]), float(parsed[2]))
                if parsed[0] != "const":
                    semiring_schedule = parsed
    else:
        if args.optimizer_type == "hoss":
            raise ValueError("--optimizer-type hoss is not supported for --model-type synaptic (no HVP closure).")
        if getattr(args, "use_flex_attention", False) and ddp_rank == 0:
            print0("[flex] Ignoring --use-flex-attention for --model-type synaptic.")
        if _normalize_ca_rule(getattr(args, "ca_init_rule", None)) is not None and ddp_rank == 0:
            print0("[ca-init] Ignoring CA initializer flags for --model-type synaptic.")

        syn_cfg: SynapticConfig
        syn_cfg_path = getattr(args, "synaptic_config", None)
        if syn_cfg_path:
            syn_cfg = _load_synaptic_config(Path(syn_cfg_path))
        else:
            syn_cfg = SynapticConfig()

        config = GPTSynapticConfig()
        config.vocab_size = (
            padded_vocab_size(task_tokenizer.get_vocab_size())
            if task_tokenizer is not None
            else int(getattr(args, "vocab_size", GPTConfig().vocab_size))
        )
        config.n_layer = args.n_layer
        config.n_head = args.n_head
        config.n_kv_head = args.n_kv_head
        config.n_embd = args.n_embd
        config.sequence_len = args.sequence_len
        config.syn_cfg = syn_cfg

    # Dataset (optional auto-download). --data-dir points training at ANY
    # parquet corpus following the FineWeb convention (sorted; last file is
    # the val split) - e.g. an mgr gen-tasks output directory (bead kbj2).
    data_dir = getattr(args, "data_dir", None)
    data_dir = str(data_dir) if data_dir is not None else None
    required_parquet_files = max(2, int(args.min_parquet_files))
    if args.auto_download_data and data_dir is None:
        if ddp_rank == 0:
            info = ensure_min_parquet_files(min_count=required_parquet_files)
            present = len(info.get("paths", []))
            downloaded = info.get("downloaded", [])
            console.print(
                f"[bold cyan]dataset[/bold cyan] parquet_files={present} "
                f"downloaded={len(downloaded)} min_required={required_parquet_files}"
            )
            if downloaded:
                console.print(f"[dim]Downloaded shards:[/dim] {', '.join(downloaded)}")
        if ddp:
            dist.barrier()
    elif args.auto_download_data and data_dir is not None and ddp_rank == 0:
        console.print("[yellow]--auto-download-data is ignored with --data-dir (custom corpus).[/yellow]")

    parquet_files = list_parquet_files(data_dir)
    if len(parquet_files) < required_parquet_files:
        where = data_dir or "the nanochat cache directory"
        raise RuntimeError(
            f"No usable dataset found in {where} (need >={required_parquet_files} parquet shards; "
            "2 is the minimum: 1 train + 1 val). "
            "Either provide shards there or run with --auto-download-data (FineWeb cache only)."
        )

    # Model
    if model_type == "gpt":
        if not isinstance(config, GPTConfig):
            raise TypeError("GPT training requires GPTConfig")
        raw_model: GPT | GPTSynaptic = GPT(config)
    else:
        if not isinstance(config, GPTSynapticConfig):
            raise TypeError("synaptic training requires GPTSynapticConfig")
        raw_model = GPTSynaptic(config)
    raw_model.to(device)
    # Per-block activation RMS hooks (depth telemetry); skipped under
    # torch.compile, where module forward hooks and dynamo do not mix.
    activation_hooks = _install_activation_rms_hooks(raw_model) if not compile_requested else None
    raw_model.init_weights()
    if resume_meta is not None:
        # The resumed run must be the SAME model: config drift between the
        # checkpoint and the resume command line is an error, not a warning.
        saved_config = resume_meta.get("model_config") or {}
        current_config = asdict(config)
        diffs = {
            k: (saved_config.get(k), current_config.get(k))
            for k in sorted(set(saved_config) | set(current_config))
            if saved_config.get(k) != current_config.get(k)
        }
        if diffs:
            raise ValueError(
                "Resume config mismatch (checkpoint vs current args): "
                + ", ".join(f"{k}: {a!r} -> {b!r}" for k, (a, b) in diffs.items())
            )
        assert resume_model_data is not None
        raw_model.load_state_dict(resume_model_data, strict=True)
    config_artifact = asdict(config)
    if model_type == "gpt":
        config_artifact["resolved_attention_schedule"] = list(attention_schedule)
        config_artifact["attention_schedule_label"] = attention_label
    model: torch.nn.Module = raw_model
    compiled_model = False
    if compile_requested:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile requested but torch.compile is unavailable (requires torch>=2.0).")
        compile_kwargs: dict[str, Any] = {
            "backend": compile_backend,
            "mode": compile_mode,
            "fullgraph": compile_fullgraph,
            "dynamic": compile_dynamic,
        }
        if ddp_rank == 0:
            console.print(
                "[bold cyan]torch.compile[/bold cyan] enabled "
                f"(backend={compile_kwargs['backend']!r}, mode={compile_kwargs['mode']!r}, "
                f"fullgraph={compile_kwargs['fullgraph']}, dynamic={compile_kwargs['dynamic']})"
            )
        model = cast(torch.nn.Module, torch.compile(model, **compile_kwargs))
        compiled_model = True

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Optimizer
    unembedding_lr = (
        float(args.unembedding_lr) if getattr(args, "unembedding_lr", None) is not None else float(args.learning_rate)
    )
    embedding_lr = (
        float(args.embedding_lr) if getattr(args, "embedding_lr", None) is not None else float(args.learning_rate)
    )
    matrix_lr = float(args.matrix_lr) if getattr(args, "matrix_lr", None) is not None else float(args.learning_rate)
    weight_decay = float(getattr(args, "weight_decay", 0.0))
    grad_clip_norm = getattr(args, "grad_clip_norm", None)
    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)
        if grad_clip_norm <= 0:
            raise ValueError("--grad-clip-norm must be > 0 when set")

    optimizers = raw_model.setup_optimizers(
        unembedding_lr=unembedding_lr,
        embedding_lr=embedding_lr,
        matrix_lr=matrix_lr,
        weight_decay=weight_decay,
    )

    # Warmdown baselines captured BEFORE any resume-state overwrite: the
    # multiplier always applies to these, never to mid-decay restored LRs
    # (checkpointed LRs may already be inside a decayed tail).
    base_param_lrs: list[list[float]] = [[float(g["lr"]) for g in opt.param_groups] for opt in optimizers]

    # Scheduler
    schedulers: list[OrdinalLRScheduler] = []
    if args.scheduler_type == "ordinal":
        # We attach an ordinal scheduler to each optimizer
        # Note: Ordinal scheduler updates LR based on loss.
        for opt in optimizers:
            schedulers.append(OrdinalLRScheduler(opt))

    # Restore optimizer/scheduler state AFTER scheduler construction: the
    # OrdinalLRScheduler __init__ snapshots each param group's configured LR
    # as its base; optimizer.load_state_dict then restores the checkpointed
    # LRs, and scheduler.load_state_dict re-applies base * scale over them so
    # the two sources agree.
    if resume_train_state is not None:
        saved_opt_states = resume_train_state.get("optimizers") or []
        if len(saved_opt_states) != len(optimizers):
            raise RuntimeError(
                f"Resume optimizer count mismatch: checkpoint has {len(saved_opt_states)}, "
                f"current setup built {len(optimizers)} (optimizer_type changed?)"
            )
        for opt, opt_state in zip(optimizers, saved_opt_states):
            opt.load_state_dict(opt_state)
        saved_sched_states = resume_train_state.get("schedulers") or []
        if len(saved_sched_states) != len(schedulers):
            raise RuntimeError(
                f"Resume scheduler count mismatch: checkpoint has {len(saved_sched_states)}, "
                f"current setup built {len(schedulers)} (scheduler_type changed?)"
            )
        for sched, sched_state in zip(schedulers, saved_sched_states):
            sched.load_state_dict(sched_state)
        # Stateful dequantization-annealing controllers (efzy): like the
        # schedulers, their state must round-trip or the resumed beta
        # trajectory silently restarts at b0.
        if semiring_controller is not None:
            saved_ctl = resume_train_state.get("semiring_controller")
            if saved_ctl is None:
                raise RuntimeError(
                    "--semiring-beta coverage:... resume: the checkpoint has no semiring_controller "
                    "state (saved before efzy or with a different --semiring-beta mode); resuming "
                    "would restart the beta trajectory at b0"
                )
            semiring_controller.load_state_dict(saved_ctl)
            pending_cov = resume_train_state.get("semiring_pending_coverage")
            semiring_resume_coverage = None if pending_cov is None else float(pending_cov)
        if semiring_ladder is not None:
            saved_ladder = resume_train_state.get("semiring_ladder")
            if saved_ladder is None:
                raise RuntimeError(
                    "--semiring-beta ordinal:... resume: the checkpoint has no semiring_ladder "
                    "state (saved before efzy or with a different --semiring-beta mode); resuming "
                    "would restart the beta ladder at b0"
                )
            semiring_ladder.load_state_dict(saved_ladder)

    # Dataloader (with-state variant so checkpoints can capture the data position).
    batches_consumed = 0
    loader_resume_state: dict[str, int] | None = None
    if resume_meta is not None:
        batches_consumed = int(resume_meta.get("batches_consumed", 0))
        if resume_data_mode == "approximate":
            # The loader's native resume: skips to the next row group after the
            # recorded position. Cheap for huge runs; may skip a few documents
            # (never repeats), so trajectories are NOT bitwise-comparable.
            saved_loader = (resume_train_state or {}).get("dataloader") or {}
            loader_resume_state = {
                "pq_idx": int(saved_loader["pq_idx"]),
                "rg_idx": int(saved_loader["rg_idx"]),
                "epoch": int(saved_loader.get("epoch", 0)),
            }
    loader = tokenizing_distributed_data_loader_with_state(
        B=args.batch_size,
        T=config.sequence_len,
        split="train",
        device=device.type,
        resume_state_dict=loader_resume_state,
        data_dir=data_dir,
        prefetch_chunks=int(getattr(args, "dataloader_prefetch", 0)),
        tokenizer=task_tokenizer,
        shuffle_seed=data_shuffle_seed,
    )
    if resume_meta is not None and resume_data_mode == "exact" and batches_consumed > 0:
        # Exact resume: replay the deterministic stream from the beginning and
        # discard the batches the parent run already consumed. The token stream
        # (and the internal token_buffer remainder) is reconstructed exactly, so
        # the resumed trajectory can be bitwise-identical. Cost is one pass of
        # tokenization over the consumed prefix - linear in the parent run
        # length; use --resume-data-mode approximate for very long runs.
        t_ff0 = time.perf_counter()
        for _ in range(batches_consumed):
            next(loader)
        if ddp_rank == 0:
            console.print(
                f"[bold cyan]resume[/bold cyan] fast-forwarded {batches_consumed} batches "
                f"in {time.perf_counter() - t_ff0:.2f}s (exact data replay)"
            )

    def autocast_ctx():
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    # Byte length per token id for the validation tokenizer (set with the val
    # loader below); None for model types whose forward has no per-token loss.
    val_token_bytes: torch.Tensor | None = None

    @torch.no_grad()
    def evaluate_validation(val_loader_iter, num_batches: int) -> tuple[float, float | None]:
        """Validation cross-entropy per token and, when the model exposes a
        per-token loss, bits per byte (tokenizer-independent; see
        nanochat.loss_eval). Returns ``(val_ce, val_bpb_or_None)``."""
        model.eval()
        try:
            if val_token_bytes is not None:
                with autocast_ctx():
                    return evaluate_loss_and_bpb(model, val_loader_iter, num_batches, val_token_bytes)
            total_loss = 0.0
            count = 0
            for _ in range(num_batches):
                try:
                    val_inputs, val_targets = next(val_loader_iter)
                except StopIteration:
                    break
                with autocast_ctx():
                    output = model(val_inputs, val_targets)
                    loss = _extract_loss(output)
                if ddp:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / ddp_world_size
                total_loss += loss.item()
                count += 1
            return (total_loss / count if count > 0 else float("nan")), None
        finally:
            model.train()

    if ddp_rank == 0:
        console.print(
            f"[bold green]Starting training[/bold green] on [bold]{device}[/bold] (world_size={ddp_world_size})"
        )
        if model_type == "gpt":
            schedule_table = Table(title="Resolved attention schedule", box=box.ROUNDED)
            schedule_table.add_column("layer", justify="right", style="cyan")
            schedule_table.add_column("mechanism", style="bold")
            schedule_table.add_column("constraint check", style="green")
            for layer_idx, mechanism in enumerate(attention_schedule):
                schedule_table.add_row(str(layer_idx), mechanism, "ok")
            console.print(schedule_table)
        compile_flex_attention = bool(getattr(config, "compile_flex_attention", False))
        if compiled_model or compile_flex_attention:
            status_bits = [f"model={'enabled' if compiled_model else 'disabled'}"]
            if bool(getattr(config, "use_flex_attention", False)):
                status_bits.append(f"flex_attention={'enabled' if compile_flex_attention else 'disabled'}")
            console.print(
                f"[dim]compile[/dim] backend={compile_backend!r} mode={compile_mode!r} "
                f"fullgraph={compile_fullgraph} dynamic={compile_dynamic} " + " ".join(status_bits)
            )
        if check_numerics:
            console.print("[bold yellow]numerics checks enabled[/bold yellow] (NaN/Inf watchpoints)")

    # Training loop
    flops_per_token = int(raw_model.estimate_flops())
    tokens_per_step = int(args.batch_size) * int(config.sequence_len) * int(ddp_world_size)
    flops_per_step = flops_per_token * tokens_per_step

    if args.target_flops is not None:
        if args.target_flops <= 0:
            raise ValueError("--target-flops must be positive")
        max_steps = max(1, int(math.ceil(args.target_flops / flops_per_step)))
    else:
        max_steps = int(args.max_steps)
        if max_steps < 1:
            raise ValueError("--max-steps must be >= 1")

    start_step = 0
    parent_run_ids: list[str] = []
    if resume_meta is not None:
        # A resumed run honors the ORIGINAL budget: max_steps comes from the
        # checkpoint meta, so interrupt+resume consumes exactly the FLOPs the
        # first command line planned - regardless of the resume command line.
        saved_budget = resume_meta.get("budget") or {}
        saved_max_steps = int(saved_budget.get("max_steps", max_steps))
        if saved_max_steps != max_steps and ddp_rank == 0:
            console.print(
                f"[bold cyan]resume[/bold cyan] restoring original budget: max_steps {max_steps} -> "
                f"{saved_max_steps} (from checkpoint meta; the original --target-flops/--max-steps governs)"
            )
        max_steps = saved_max_steps
        start_step = int(resume_meta["step"]) + 1
        lineage = resume_meta.get("lineage") or {}
        parent_run_ids = list(lineage.get("parent_run_ids") or [])
        parent_id = lineage.get("run_id")
        if parent_id:
            parent_run_ids.append(str(parent_id))
        if start_step >= max_steps and ddp_rank == 0:
            console.print(
                f"[bold yellow]resume[/bold yellow] checkpoint step {start_step - 1} already reaches the "
                f"budget ({max_steps} steps); nothing to train."
            )

    if ddp_rank == 0:
        console.print(
            f"[bold cyan]budget[/bold cyan] steps={max_steps} start_step={start_step} warmup={args.warmup_steps} "
            f"tokens/step={tokens_per_step:,} flops/token={flops_per_token:,} "
            f"flops/step={flops_per_step:,}"
        )

    # Run identity is needed BEFORE the loop now: the default checkpoint dir
    # lives under the run's artifact directory.
    resolved_run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    artifacts_kind = str(getattr(args, "artifacts_kind", "baseline"))
    artifacts_topic = str(getattr(args, "artifacts_topic", "nanochat"))
    run_dir = Path(args.artifacts_dir) / artifacts_kind / artifacts_topic / resolved_run_id

    checkpoint_dir = str(getattr(args, "checkpoint_dir", None) or (run_dir / "checkpoints"))
    if task_tokenizer is not None and ddp_rank == 0:
        # The vocabulary travels with the weights: every checkpoint consumer
        # loads it through nanochat.tokenizer.checkpoint_tokenizer.
        task_tokenizer.save(str(Path(checkpoint_dir) / TASK_TOKENIZER_DIRNAME))
    checkpoint_saved_steps: list[int] = []

    # Per-step metrics stream (bead rz8.2): rank-0 only; the header (with the
    # tamper-evidence provenance block) is flushed immediately so even a
    # crashed run leaves an attributable artifact.
    provenance = build_provenance(config_artifact)
    metrics_stream: MetricsStream | None = None
    dashboard: TrainingDashboard | None = None
    if ddp_rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        # append on resume: truncating would erase the parent process's step
        # history (rz8.8 e2e finding); the splice is marked by a
        # "resume_header" record with this process's provenance
        metrics_stream = MetricsStream(run_dir / "metrics.jsonl", provenance=provenance, append=resume_meta is not None)
        if bool(getattr(args, "dashboard", False)):
            # Live rich dashboard (bead nyp): renders the SAME records the
            # metrics stream writes - zero overhead when the flag is off.
            dash_html = getattr(args, "dashboard_html", None)
            html_path = (
                Path(dash_html) if dash_html else Path(args.artifacts_dir) / "dashboard" / f"{resolved_run_id}.html"
            )
            dash_flags: dict[str, Any] = {
                "attention": attention_label,
                "ffn": getattr(config, "ffn_type", "standard"),
                "optimizer": str(args.optimizer_type),
                "scheduler": str(args.scheduler_type),
                "flex": bool(getattr(config, "use_flex_attention", False)),
                "device": device.type,
            }
            if len(set(attention_schedule)) > 1:
                dash_flags["schedule"] = list(attention_schedule)
            if "reversible" in attention_schedule:
                dash_flags["mode"] = getattr(config, "reversible_mode", "additive")
                dash_flags["tied"] = getattr(config, "reversible_tied", False)
            if "tropical" in attention_schedule and getattr(args, "semiring_beta", None) is not None:
                dash_flags["beta"] = str(args.semiring_beta)
            parameterization = getattr(config, "parameterization", "current")
            if parameterization != "current":
                dash_flags["param"] = str(parameterization)
            if isinstance(config, GPTConfig) and config.activation_ckpt != "none":
                dash_flags["ckpt"] = (
                    f"{config.activation_ckpt}@{config.activation_ckpt_every_k}"
                    if config.activation_ckpt == "every-k"
                    else str(config.activation_ckpt)
                )
            if "braid" in attention_schedule:
                dash_flags["law"] = getattr(config, "braid_crossing_law", "restricted")
            if all(name == "standard" for name in attention_schedule) and getattr(config, "disable_block_norms", False):
                dash_flags["no_norms"] = True
            dashboard = TrainingDashboard(
                run_id=resolved_run_id,
                mechanism=attention_label if model_type == "gpt" else model_type,
                max_steps=max_steps,
                flags=dash_flags,
                html_path=html_path,
            )
            dashboard.start()

    def _grad_norm() -> float:
        # Called in the log block AFTER opt.step() and BEFORE the next
        # iteration's zero_grad, so gradients are still populated.
        # clip_grad_norm_ with max_norm=inf is the canonical foreach-optimized
        # total-norm computation: no clipping happens, the return value is the
        # global L2 norm (a hand-rolled per-param pow/sum costs ~3x more).
        return float(torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=float("inf")))

    def save_training_checkpoint(step: int) -> None:
        """Capture FULL training state at a completed step (bead rz8.1)."""
        t_save0 = time.perf_counter()
        model_data = raw_model.state_dict()  # raw module: clean keys under torch.compile/DDP
        # RNG lanes serialized into weights-only-safe types (tensors/ints/lists)
        # so load_checkpoint can keep torch.load(weights_only=True).
        np_bit_gen, np_keys, np_pos, np_has_gauss, np_cached = np.random.get_state()
        py_version, py_internal, py_gauss = random.getstate()
        rng_state: dict[str, Any] = {
            "torch_cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "numpy": {
                "bit_generator": str(np_bit_gen),
                "keys": torch.from_numpy(np.asarray(np_keys, dtype=np.int64)),
                "pos": int(np_pos),
                "has_gauss": int(np_has_gauss),
                "cached_gaussian": float(np_cached),
            },
            # gauss slot is None or the cached Box-Muller float - normalize explicitly
            "python": [int(py_version), [int(v) for v in py_internal], None if py_gauss is None else float(py_gauss)],
        }
        train_state: dict[str, Any] = {
            "optimizers": [opt.state_dict() for opt in optimizers],
            "schedulers": [sched.state_dict() for sched in schedulers],
            "dataloader": {
                "batches_consumed": step + 1,
                "pq_idx": int(last_loader_state.get("pq_idx", 0)),
                "rg_idx": int(last_loader_state.get("rg_idx", 0)),
                "epoch": int(last_loader_state.get("epoch", 0)),
            },
            "rng": rng_state,
        }
        # Stateful dequantization-annealing controllers (efzy) ride beside the
        # schedulers. The coverage controller also saves the model's CURRENT
        # route-coverage reading: the live buffers are non-persistent, so this
        # is the value the next step's controller call must consume on resume.
        if semiring_controller is not None:
            train_state["semiring_controller"] = semiring_controller.state_dict()
            train_state["semiring_pending_coverage"] = _collect_tropical_route_coverage(raw_model)
        if semiring_ladder is not None:
            train_state["semiring_ladder"] = semiring_ladder.state_dict()
        meta_data: dict[str, Any] = {
            "step": step,
            "model_config": asdict(config),
            "model_type": model_type,
            "tokenizer": tokenizer_meta,
            "data_shuffle": data_shuffle,
            "optimizer_type": str(args.optimizer_type),
            "scheduler_type": str(args.scheduler_type),
            "seed": int(args.seed),
            "world_size": int(ddp_world_size),
            "batches_consumed": step + 1,
            # the annealed beta these weights were saved at (y2h9); loaders
            # must construct at this value, not model_config's schedule b0
            "semiring_beta_live": last_semiring_beta,
            "budget": {
                "max_steps": max_steps,
                "target_flops": args.target_flops,
                "flops_per_step_est": flops_per_step,
                "flops_consumed_est": (step + 1) * flops_per_step,
            },
            "lineage": {"run_id": resolved_run_id, "parent_run_ids": parent_run_ids},
        }
        save_checkpoint(checkpoint_dir, step, model_data, train_state, meta_data, rank=ddp_rank)
        checkpoint_saved_steps.append(step)
        if checkpoint_verify:
            mismatches = verify_checkpoint_roundtrip(
                checkpoint_dir, step, device, model_data=model_data, optimizer_data=train_state, rank=ddp_rank
            )
            if mismatches:
                raise RuntimeError(
                    f"--checkpoint-verify failed at step {step}: {len(mismatches)} tensor mismatches "
                    f"(first: {mismatches[:5]})"
                )
        pruned = prune_checkpoints(checkpoint_dir, checkpoint_saved_steps, keep=checkpoint_keep, rank=ddp_rank)
        if ddp_rank == 0:
            saved_bytes = sum(
                os.path.getsize(p)
                for p in checkpoint_file_paths(checkpoint_dir, step, rank=ddp_rank)
                if os.path.exists(p)
            )
            verify_note = " verify=ok" if checkpoint_verify else ""
            prune_note = f" pruned={pruned}" if pruned else ""
            console.print(
                f"[bold cyan]checkpoint[/bold cyan] step={step} dir={checkpoint_dir} "
                f"bytes={saved_bytes:,} duration={time.perf_counter() - t_save0:.2f}s "
                f"components=model/optim/sched/dataloader/rng{verify_note}{prune_note}"
            )

    # Restore RNG streams LAST, after every setup consumer of randomness
    # (model init, optimizer construction) has run - so the first resumed
    # step draws exactly what the uninterrupted run would have drawn.
    if resume_train_state is not None:
        saved_rng = resume_train_state.get("rng") or {}
        if "torch_cpu" in saved_rng:
            torch.set_rng_state(saved_rng["torch_cpu"].cpu().to(torch.uint8))
        if device.type == "cuda" and saved_rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in saved_rng["cuda"]])
        if saved_rng.get("numpy") is not None:
            np_state = saved_rng["numpy"]
            np.random.set_state(
                (
                    np_state["bit_generator"],
                    np_state["keys"].cpu().numpy().astype(np.uint32),
                    int(np_state["pos"]),
                    int(np_state["has_gauss"]),
                    float(np_state["cached_gaussian"]),
                )
            )
        if saved_rng.get("python") is not None:
            py_version, py_internal, py_gauss = saved_rng["python"]
            random.setstate((int(py_version), tuple(int(v) for v in py_internal), py_gauss))

    is_hoss = args.optimizer_type == "hoss"
    if is_hoss:
        # HOSS's HVP differentiates THROUGH the attention backward pass
        # (create_graph=True + autograd.grad of grads). The flash CPU SDPA
        # kernel ships no double-backward derivative, so HVP runs must record
        # their forward under the composed math kernel instead (rz8.3).
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def _hvp_safe_sdpa_ctx():
            return sdpa_kernel(SDPBackend.MATH)

    losses: list[float] = []
    # depth telemetry (logged steps only): total grad norm, per-block grad
    # norms, per-block activation RMS - summarized into results.depth_telemetry
    grad_norm_log: list[float] = []
    block_grad_norm_log: list[list[float]] = []
    activation_rms_log: list[list[float]] = []
    val_losses: list[tuple[int, float]] = []  # (step, val_loss) pairs
    val_bpbs: list[tuple[int, float]] = []  # (step, bits per byte) pairs; empty when the model has no per-token loss
    # Tie-locus trend trackers (bead y4r8): first/last certificate coverage
    # seen during training, promoted into the summary so the registered
    # hyp-tie-locus-density-decreases observable is adjudicable from the
    # train artifact alone (metrics.jsonl streams are not engine-readable).
    route_coverage_first: float | None = None
    route_coverage_last: float | None = None
    # Shadow-energy trend trackers (bead u55.5): first/final mean shadow
    # energy and final activation norm, promoted into the summary so the
    # symplectic no-norm program's diagnostics are engine-readable.
    symplectic_energy_first: float | None = None
    symplectic_energy_last: float | None = None
    symplectic_x_norm_last: float | None = None
    # Live semiring beta at the most recent step (bead y2h9): recorded into
    # every checkpoint meta (semiring_beta_live) and the summary
    # (results.semiring_beta_final) so post-hoc loaders reconstruct annealed
    # models at the value they actually ran at - the recorded model_config
    # keeps the schedule's b0, which is wrong for an annealed-to-32 endpoint.
    _config_beta0 = getattr(config, "semiring_beta", None)
    last_semiring_beta: float | None = float(_config_beta0) if _config_beta0 is not None else None
    step_times_s: list[float] = []
    last_log_step = -1

    # Validation setup
    val_interval = int(getattr(args, "val_interval", 0))
    val_batches = int(getattr(args, "val_batches", 10))
    val_loader = None
    if val_interval > 0:
        val_loader = tokenizing_distributed_data_loader(
            B=args.batch_size,
            T=config.sequence_len,
            split="val",
            device=device.type,
            data_dir=data_dir,
            tokenizer=task_tokenizer,
        )
        if model_type == "gpt":
            # bits per byte needs the byte length of every token id, padded to
            # the embedding table so any target id is a valid index.
            try:
                byte_lengths = (task_tokenizer if task_tokenizer is not None else get_tokenizer()).token_bytes()
            except Exception as exc:  # noqa: BLE001 - reported, and the run records val_bpb = null
                byte_lengths = None
                if ddp_rank == 0:
                    console.print(
                        f"[yellow]validation: tokenizer unavailable for bits-per-byte ({type(exc).__name__}: {exc}); "
                        "val_bpb will be null[/yellow]"
                    )
            if byte_lengths is not None:
                byte_lengths += [0] * max(0, int(config.vocab_size) - len(byte_lengths))
                val_token_bytes = torch.tensor(byte_lengths, dtype=torch.int64, device=device)
        if ddp_rank == 0:
            console.print(
                f"[bold cyan]validation[/bold cyan] interval={val_interval} batches={val_batches} "
                f"metrics=val_ce{' + val_bpb' if val_token_bytes is not None else ''}"
            )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    last_log_time = time.perf_counter()
    meas_start_time: float | None = None

    last_loader_state: dict[str, int] = {}
    last_completed_step = start_step - 1
    # pre-clip grad norm captured by the clipping call for reuse in metrics
    # (one-element list: assigned inside the loop, read in the log block)

    # Profiling window (b1l hooks wired via --profile, bead 6m35): profile
    # only the LAST --profile-steps steps so whole-run overhead stays zero
    # when the flag is off; the aggregate table + chrome trace land under
    # run_dir/profile/. Teardown is crash-safe via the finally below.
    prof_steps = max(1, int(getattr(args, "profile_steps", 5) or 5))
    prof_cfg = ProfileConfig(
        enabled=bool(getattr(args, "profile", False)),
        trace_dir=(run_dir / "profile") if getattr(args, "profile", False) else None,
    )
    prof_window_start = max(start_step, max_steps - prof_steps)
    prof_stack = None
    prof_handle = None
    last_preclip_grad_norm: list[float | None] = [None]
    try:
        for step, (inputs, targets, loader_state) in enumerate(loader, start=start_step):
            if step >= max_steps:
                break
            last_loader_state = loader_state

            if prof_cfg.enabled and prof_stack is None and step >= prof_window_start:
                from contextlib import ExitStack

                prof_stack = ExitStack()
                prof_handle = prof_stack.enter_context(torch_profiler(prof_cfg))
                if ddp_rank == 0:
                    console.print(
                        f"[bold cyan]profiling[/bold cyan] window: steps {step}..{max_steps - 1} -> {prof_cfg.trace_dir}"
                    )

            if step == args.warmup_steps:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                meas_start_time = time.perf_counter()

            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            current_semiring_beta: float | None
            if semiring_schedule is not None:
                # dequantization annealing (8gk.1/9jzb): one scalar with
                # algebraic meaning, updated on the live modules - never state
                if semiring_controller is not None:
                    # closed-loop: consume the PREVIOUS forward's certificate
                    # coverage; the controller never raises beta below the floor
                    if semiring_resume_coverage is not None:
                        # first step after resume (efzy): the checkpointed reading
                        coverage_reading: float | None = semiring_resume_coverage
                        semiring_resume_coverage = None
                    else:
                        coverage_reading = _collect_tropical_route_coverage(raw_model)
                    current_semiring_beta = semiring_controller.step(coverage_reading)
                elif semiring_ladder is not None and schedulers:
                    # ordinal-orchestrated: the transfinite clock drives beta
                    current_semiring_beta = semiring_ladder.step(schedulers[0].A, schedulers[0].B)
                else:
                    current_semiring_beta = _semiring_beta_at(semiring_schedule, step, max_steps)
                set_semiring_beta(raw_model, current_semiring_beta)
            else:
                config_beta = getattr(config, "semiring_beta", None)
                current_semiring_beta = float(config_beta) if config_beta is not None else None
            last_semiring_beta = current_semiring_beta

            if args.lr_warmdown_ratio > 0.0:
                # WSD tail (bead kj8s): multiply every param group's BASE lr by
                # the schedule factor. Pure function of (step, max_steps), so
                # resumed runs reproduce the trajectory with no extra state.
                mult = _lr_warmdown_multiplier(step, max_steps, args.lr_warmdown_ratio)
                for _opt, _bases in zip(optimizers, base_param_lrs):
                    for group, base in zip(_opt.param_groups, _bases):
                        group["lr"] = base * mult

            step_t0 = time.perf_counter()

            def closure(inputs=inputs, targets=targets):
                with autocast_ctx():
                    if is_hoss:
                        with _hvp_safe_sdpa_ctx():
                            loss = _extract_loss(model(inputs, targets))
                    else:
                        loss = _extract_loss(model(inputs, targets))
                if check_numerics and (not torch.isfinite(loss).all().item()):
                    if ddp_rank == 0:
                        console.print("[bold red]Non-finite loss detected[/bold red]")
                        console.print(_summarize_nonfinite(loss))
                    raise FloatingPointError("Non-finite loss detected (NaN/Inf).")
                loss.backward(create_graph=is_hoss)
                if grad_clip_norm is not None and not is_hoss:
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=grad_clip_norm)
                return loss

            if is_hoss:
                loss = optimizers[0].step(closure)
                if check_numerics:
                    bad_grads: list[tuple[str, dict[str, Any]]] = []
                    for n, p in raw_model.named_parameters():
                        if p.grad is None:
                            continue
                        if torch.isfinite(p.grad).all().item():
                            continue
                        bad_grads.append((n, _summarize_nonfinite(p.grad)))
                        if len(bad_grads) >= 12:
                            break
                    if bad_grads:
                        if ddp_rank == 0:
                            table = Table(title="Non-finite gradients detected", box=box.ROUNDED)
                            table.add_column("param", style="bold")
                            table.add_column("shape")
                            table.add_column("dtype")
                            table.add_column("device")
                            table.add_column("nonfinite", justify="right")
                            table.add_column("nan", justify="right")
                            table.add_column("inf", justify="right")
                            for name, stats in bad_grads:
                                table.add_row(
                                    name,
                                    str(stats.get("shape")),
                                    str(stats.get("dtype")),
                                    str(stats.get("device")),
                                    str(stats.get("nonfinite")),
                                    str(stats.get("nan")),
                                    str(stats.get("inf")),
                                )
                            console.print(table)
                        raise FloatingPointError("Non-finite gradients detected (NaN/Inf).")
            else:
                with autocast_ctx():
                    loss = _extract_loss(model(inputs, targets))
                if check_numerics and (not torch.isfinite(loss).all().item()):
                    if ddp_rank == 0:
                        console.print("[bold red]Non-finite loss detected[/bold red]")
                        console.print(_summarize_nonfinite(loss))
                    raise FloatingPointError("Non-finite loss detected (NaN/Inf).")
                loss.backward()
                if grad_clip_norm is not None:
                    # The clip call computes the pre-clip total norm anyway;
                    # reuse it in the metrics record (zero added cost).
                    last_preclip_grad_norm[0] = float(
                        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=grad_clip_norm)
                    )
                if check_numerics:
                    bad_grads: list[tuple[str, dict[str, Any]]] = []
                    for n, p in raw_model.named_parameters():
                        if p.grad is None:
                            continue
                        if torch.isfinite(p.grad).all().item():
                            continue
                        bad_grads.append((n, _summarize_nonfinite(p.grad)))
                        if len(bad_grads) >= 12:
                            break
                    if bad_grads:
                        if ddp_rank == 0:
                            table = Table(title="Non-finite gradients detected", box=box.ROUNDED)
                            table.add_column("param", style="bold")
                            table.add_column("shape")
                            table.add_column("dtype")
                            table.add_column("device")
                            table.add_column("nonfinite", justify="right")
                            table.add_column("nan", justify="right")
                            table.add_column("inf", justify="right")
                            for name, stats in bad_grads:
                                table.add_row(
                                    name,
                                    str(stats.get("shape")),
                                    str(stats.get("dtype")),
                                    str(stats.get("device")),
                                    str(stats.get("nonfinite")),
                                    str(stats.get("nan")),
                                    str(stats.get("inf")),
                                )
                            console.print(table)
                        raise FloatingPointError("Non-finite gradients detected (NaN/Inf).")
                for opt in optimizers:
                    opt.step()

            loss_for_log = loss.detach()
            if ddp:
                dist.all_reduce(loss_for_log, op=dist.ReduceOp.SUM)
                loss_for_log = loss_for_log / ddp_world_size

            loss_item = float(loss_for_log.item())
            losses.append(loss_item)
            last_completed_step = step

            # Ordinal scheduler transitions (loss-driven). Orphaned until rz8.1:
            # the schedulers were constructed but never stepped, so
            # --scheduler-type ordinal silently did nothing past the initial LR.
            for sched in schedulers:
                sched.step(loss_item)

            # Periodic full-state checkpoint (bead rz8.1).
            if checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
                save_training_checkpoint(step)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_t1 = time.perf_counter()
            step_times_s.append(step_t1 - step_t0)

            if ddp_rank == 0 and (step % args.log_interval == 0 or step == max_steps - 1):
                dt = step_t1 - last_log_time
                steps_since = step - last_log_step
                tokens_since = steps_since * tokens_per_step
                toks_s = tokens_since / dt if dt > 0 else float("nan")
                tflops = (flops_per_token * toks_s) / 1e12
                msg = (
                    f"[dim]step[/dim] {step:>6}  "
                    f"[dim]loss[/dim] {loss_item:>8.4f}  "
                    f"[dim]tok/s[/dim] {toks_s:>10.0f}  "
                    f"[dim]TFLOP/s(est)[/dim] {tflops:>7.2f}"
                )
                if (
                    bool(getattr(args, "tropical_log_margins", False))
                    and model_type == "gpt"
                    and "tropical" in attention_schedule
                ):
                    tropical = _collect_tropical_margin_stats(raw_model)
                    if tropical is not None:
                        gamma_min = float(tropical.get("gamma_min", float("nan")))
                        gamma_mean = float(tropical.get("gamma_mean", float("nan")))

                        def fmt(x: float) -> str:
                            if math.isnan(x):
                                return "nan"
                            if math.isinf(x):
                                return "inf" if x > 0 else "-inf"
                            return f"{x:.4g}"

                        msg += f"  [dim]γ_min[/dim] {fmt(gamma_min):>7}  [dim]γ_mean[/dim] {fmt(gamma_mean):>7}"
                        head_mean = tropical.get("head_mean")
                        if isinstance(head_mean, list) and head_mean:
                            head_snip = head_mean[: min(8, len(head_mean))]
                            msg += (
                                "  [dim]γ_head_mean[/dim] ["
                                + ", ".join(fmt(float(x)) for x in head_snip)
                                + (", …]" if len(head_mean) > len(head_snip) else "]")
                            )

                if model_type == "gpt" and "hyperbolic" in attention_schedule:
                    hyperbolic = _collect_hyperbolic_stats(raw_model)
                    if hyperbolic is not None:
                        curvature = hyperbolic["curvature"]
                        radius = hyperbolic.get("radius")
                        msg += f"  [dim]c_mean[/dim] {float(curvature['mean']):>7.4g}"
                        if isinstance(radius, dict):
                            msg += f"  [dim]r_mean[/dim] {float(radius['mean']):>7.4g}"

                console.print(msg)
                last_log_time = step_t1
                last_log_step = step

                if metrics_stream is not None:
                    record: dict[str, Any] = {
                        "type": "step",
                        "step": step,
                        "loss": loss_item,
                        "lr": float(optimizers[0].param_groups[0]["lr"]),
                        "lr_groups": [float(g["lr"]) for opt in optimizers for g in opt.param_groups],
                        "grad_norm": (
                            last_preclip_grad_norm[0] if last_preclip_grad_norm[0] is not None else _grad_norm()
                        ),
                        "tokens_per_s": float(toks_s),
                        "tflops": float(tflops),
                        "peak_mem_gb": (
                            float(torch.cuda.max_memory_allocated(device) / (1024**3))
                            if device.type == "cuda"
                            else None
                        ),
                        "elapsed_s": step_t1 - (meas_start_time or step_t1),
                    }
                    # depth telemetry: gradients are still populated here (see
                    # _grad_norm); activation readings come from this step's
                    # forward via the block hooks
                    record["grad_norm_by_block"] = _collect_block_grad_norms(raw_model)
                    record["activation_rms_by_block"] = (
                        [float(x) if x is not None else None for x in activation_hooks["rms"]]
                        if activation_hooks is not None
                        else None
                    )
                    if isinstance(record["grad_norm"], int | float):
                        grad_norm_log.append(float(record["grad_norm"]))
                    if record["grad_norm_by_block"] is not None:
                        block_grad_norm_log.append(list(record["grad_norm_by_block"]))
                    rms_readings = [x for x in (record["activation_rms_by_block"] or []) if isinstance(x, int | float)]
                    if record["activation_rms_by_block"] is not None and len(rms_readings) == len(
                        record["activation_rms_by_block"]
                    ):
                        activation_rms_log.append([float(x) for x in rms_readings])
                    if schedulers:
                        s0 = schedulers[0]
                        record["ordinal"] = {
                            "A": s0.A,
                            "B": s0.B,
                            "C": s0.C,
                            "best_loss": s0.best_loss,
                            "ema_loss": s0.ema_loss,
                        }
                    if (
                        model_type == "gpt"
                        and "tropical" in attention_schedule
                        and bool(getattr(config, "tropical_record_margins", False))
                    ):
                        trop_stats = _collect_tropical_margin_stats(raw_model)
                        if trop_stats is not None:
                            record["tropical_gamma_min"] = trop_stats.get("gamma_min")
                            record["tropical_gamma_mean"] = trop_stats.get("gamma_mean")
                            record["tropical_gamma_head_mean"] = trop_stats.get("head_mean")
                    if model_type == "gpt" and "tropical" in attention_schedule and current_semiring_beta is not None:
                        # D2 schema gains the annealing telemetry (8gk.1):
                        # the smoothing level and the certificate coverage
                        record["semiring_beta"] = float(current_semiring_beta)
                        coverage = _collect_tropical_route_coverage(raw_model)
                        if coverage is not None:
                            record["route_coverage"] = coverage
                            if route_coverage_first is None:
                                route_coverage_first = coverage
                            route_coverage_last = coverage
                    if model_type == "gpt" and "braid" in attention_schedule:
                        # D2 schema gains the conserved-charge telemetry (u55.3):
                        # Q1 mass-partition defect and Q2 braid-consistency residual
                        # separate integrable (rmatrix) from heuristic mixing live.
                        braid_stats = _collect_braid_charge_stats(raw_model)
                        if braid_stats is not None:
                            record["braid_crossing_law"] = braid_stats.get("crossing_law")
                            record["braid_q1_mass_defect_max"] = braid_stats.get("q1_mass_defect_max")
                            record["braid_q2_braid_residual_max"] = braid_stats.get("q2_braid_residual_max")
                            if "eta_per_layer" in braid_stats:
                                record["braid_eta_per_layer"] = braid_stats["eta_per_layer"]
                            if "rapidity_span_per_layer" in braid_stats:
                                record["braid_rapidity_span_per_layer"] = braid_stats["rapidity_span_per_layer"]
                    if model_type == "gpt" and "hyperbolic" in attention_schedule:
                        hyperbolic = _collect_hyperbolic_stats(raw_model)
                        if hyperbolic is not None:
                            record["hyperbolic_curvature"] = hyperbolic["curvature"]
                            record["hyperbolic_radius"] = hyperbolic.get("radius")
                            record["hyperbolic_frac_heads_hier"] = hyperbolic["frac_heads_hier"]
                            record["hyperbolic_frac_heads_euclidean"] = hyperbolic["frac_heads_euclidean"]
                    if (
                        model_type == "gpt"
                        and "reversible" in attention_schedule
                        and getattr(config, "reversible_record_energy", False)
                    ):
                        # D2 schema gains the shadow-energy telemetry (u55.5):
                        # the across-layer band of the conserved H-tilde plus
                        # activation norms and the live confinement lambda.
                        sym_stats = _collect_symplectic_energy_stats(raw_model)
                        if sym_stats is not None:
                            record["symplectic_shadow_energy_mean"] = sym_stats["shadow_energy_mean"]
                            record["symplectic_energy_band_layers"] = sym_stats["energy_band_layers"]
                            record["symplectic_x_norm_mean"] = sym_stats["x_norm_mean"]
                            record["symplectic_lambda_mean"] = sym_stats["lambda_mean"]
                            if symplectic_energy_first is None:
                                symplectic_energy_first = sym_stats["shadow_energy_mean"]
                            symplectic_energy_last = sym_stats["shadow_energy_mean"]
                            symplectic_x_norm_last = sym_stats["x_norm_mean"]
                    metrics_stream.write(record)
                    if dashboard is not None:
                        dashboard.observe(record)

            # Periodic validation evaluation
            if val_loader is not None and val_interval > 0 and (step + 1) % val_interval == 0:
                val_loss, val_bpb = evaluate_validation(val_loader, val_batches)
                val_losses.append((step, val_loss))
                if val_bpb is not None:
                    val_bpbs.append((step, val_bpb))
                if ddp_rank == 0:
                    console.print(
                        f"[bold magenta]val[/bold magenta] step={step}  "
                        f"[dim]val_ce[/dim] {val_loss:.4f}  "
                        + (f"[dim]val_bpb[/dim] {val_bpb:.4f}  " if val_bpb is not None else "")
                        + f"[dim]train_ce[/dim] {loss_item:.4f}"
                    )
                    if metrics_stream is not None:
                        metrics_stream.write(
                            {
                                "type": "val",
                                "step": step,
                                "val_loss": float(val_loss),
                                "val_bpb": (float(val_bpb) if val_bpb is not None else None),
                                "train_loss": loss_item,
                            }
                        )
                        metrics_stream.flush()  # val cadence is the bead's flush point
                    if dashboard is not None:
                        dashboard.observe(
                            {"type": "val", "step": step, "val_loss": float(val_loss), "train_loss": loss_item}
                        )
    finally:
        # Land the buffered metrics even on KeyboardInterrupt/crash, BEFORE
        # the process group teardown can get in the way.
        if metrics_stream is not None:
            metrics_stream.close()
        if dashboard is not None:
            # stops the live view and writes the HTML snapshot of the final
            # dashboard state (crash-safe: runs even on KeyboardInterrupt)
            dashboard.close()
        compute_cleanup()
        if prof_stack is not None:
            # Crash-safe teardown (same contract as the metrics/dashboard
            # cleanup above): stop the profiler, render the aggregate op
            # table, and export the chrome trace for the profiling window.
            prof_stack.close()
            if prof_handle is not None and ddp_rank == 0:
                prof_summary = summarize_profile(prof_handle)
                render_profile_table(prof_summary, title=f"Training profile (last {prof_steps} steps)", console=console)
                trace_dir = prof_cfg.trace_dir
                if trace_dir is None:
                    raise RuntimeError("enabled profiler is missing its trace directory")
                trace_path = trace_dir / f"{resolved_run_id}_steps.json"
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                prof_handle.export_chrome_trace(str(trace_path))
                console.print(f"[bold cyan]profile[/bold cyan] chrome trace -> {trace_path}")

    # Final checkpoint so downstream consumers (eval C2, teachers C6) always
    # have the end-of-run state, even when max_steps is not a multiple of the
    # interval. Skipped if the interval already saved this exact step, and
    # skipped when THIS run trained no steps (a resume of an already-complete
    # run has an empty last_loader_state - saving would record a bogus
    # data position; the parent's checkpoint already covers that step).
    if (
        checkpoint_interval > 0
        and last_completed_step >= start_step
        and last_completed_step not in checkpoint_saved_steps
    ):
        save_training_checkpoint(last_completed_step)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end_time = time.perf_counter()

    if meas_start_time is None:
        meas_start_time = end_time
    measured_steps = max(0, max_steps - int(args.warmup_steps))
    measured_tokens = measured_steps * tokens_per_step
    measured_time_s = max(1e-9, end_time - meas_start_time)
    tokens_per_second = (measured_tokens / measured_time_s) if measured_steps > 0 else 0.0
    est_tflops = (flops_per_token * tokens_per_second) / 1e12

    peak_mem_gb = None
    if device.type == "cuda":
        peak_mem_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3))

    if ddp_rank != 0:
        return

    git_info = get_git_info()
    gpu_info = get_gpu_info()
    sys_info = get_system_info()

    arg_str = shlex.join(sys.argv[1:])
    module_command = "python -m nanochat.train" + (f" {arg_str}" if arg_str else "")
    uv_command = "uv run " + module_command

    # resolved_run_id / run_dir were computed before the training loop (the
    # default checkpoint dir lives under the run dir).
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta: dict[str, Any] = {
        "run_id": resolved_run_id,
        "generated_at": generated_at,
        "kind": artifacts_kind,
        "topic": artifacts_topic,
        "git": git_info,
        "system": sys_info,
        "gpu": gpu_info,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "command": uv_command,
        "command_module": module_command,
        "command_argv": shlex.join(sys.argv),
        "argv": sys.argv,
        "env": _select_env_vars(),
        "ddp": {
            "enabled": bool(ddp),
            "world_size": ddp_world_size,
            "rank": ddp_rank,
            "local_rank": ddp_local_rank,
        },
        "device": str(device),
    }
    budget: dict[str, Any] = {
        "max_steps": max_steps,
        "warmup_steps": int(args.warmup_steps),
        "target_flops": args.target_flops,
        "flops_per_token_est": flops_per_token,
        "tokens_per_step_global": tokens_per_step,
        "flops_per_step_est": flops_per_step,
        "planned_total_flops_est": max_steps * flops_per_step,
    }
    checkpointing: dict[str, Any] = {
        "interval": checkpoint_interval,
        "dir": checkpoint_dir if checkpoint_interval > 0 else None,
        "keep": checkpoint_keep,
        "verify": checkpoint_verify,
        "saved_steps": checkpoint_saved_steps,
    }
    resume_record: dict[str, Any] | None = None
    if resume_meta is not None:
        resume_record = {
            "from": str(resume_from),
            "resume_step": start_step,
            "data_mode": resume_data_mode,
            "parent_run_ids": parent_run_ids,
        }
    # Compute final train/val CE statistics
    final_train_ce = losses[-1] if losses else float("nan")
    final_val_ce = val_losses[-1][1] if val_losses else None
    final_val_bpb = val_bpbs[-1][1] if val_bpbs else None

    results: dict[str, Any] = {
        "losses": losses,
        "start_step": start_step,
        "train_ce_final": final_train_ce,
        "val_losses": val_losses,
        "val_ce_final": final_val_ce,
        # bits per byte: comparable across tokenizers (val_ce is not); null
        # when validation is off or the model type has no per-token loss
        "val_bpbs": val_bpbs,
        "val_bpb_final": final_val_bpb,
        "step_times_s": step_times_s,
        "measured_steps": measured_steps,
        "measured_tokens": measured_tokens,
        "measured_time_s": measured_time_s,
        "tokens_per_second": tokens_per_second,
        "tflops_per_second_est": est_tflops,
        "peak_memory_allocated_gb": peak_mem_gb,
    }
    if model_type == "gpt" and "tropical" in attention_schedule:
        tropical = _collect_tropical_margin_stats(raw_model)
        if tropical is not None:
            results["tropical_margins"] = tropical
        # Tie-locus trend aggregates (bead y4r8): the last forward's coverage
        # buffer is the true final reading even when the metrics stream is off.
        final_cov = _collect_tropical_route_coverage(raw_model)
        if final_cov is not None:
            route_coverage_last = final_cov
            if route_coverage_first is None:
                route_coverage_first = final_cov
        if route_coverage_first is not None and route_coverage_last is not None:
            results["route_coverage_first"] = route_coverage_first
            results["route_coverage_final"] = route_coverage_last
            results["route_coverage_delta"] = route_coverage_last - route_coverage_first
    if model_type == "gpt" and "hyperbolic" in attention_schedule:
        hyperbolic = _collect_hyperbolic_stats(raw_model)
        if hyperbolic is not None:
            results["hyperbolic_curvature"] = hyperbolic["curvature"]
            results["hyperbolic_radius"] = hyperbolic.get("radius")
            results["hyperbolic_frac_heads_hier"] = hyperbolic["frac_heads_hier"]
            results["hyperbolic_frac_heads_euclidean"] = hyperbolic["frac_heads_euclidean"]
    # the beta the model ended training at (y2h9): null = exact tropical
    # endpoint or a non-semiring model; mirrors checkpoint semiring_beta_live
    results["semiring_beta_final"] = last_semiring_beta
    if symplectic_energy_first is not None:
        # shadow-energy trend (u55.5): movement BETWEEN steps is learning
        # (the potentials change); the conservation claim lives in the
        # per-step across-layer band recorded in the metrics stream
        results["symplectic_energy_first"] = symplectic_energy_first
        results["symplectic_energy_final"] = symplectic_energy_last
        results["symplectic_x_norm_final"] = symplectic_x_norm_last
    attn_entropy = _collect_attn_entropy_stats(raw_model)
    if attn_entropy is not None:
        results["attention_entropy"] = attn_entropy
    depth_telemetry = _summarize_depth_telemetry(grad_norm_log, block_grad_norm_log, activation_rms_log)
    if depth_telemetry is not None:
        results["depth_telemetry"] = depth_telemetry
    summary: dict[str, Any] = {
        "schema_version": "mgr.telemetry.v1",
        "meta": meta,
        "hparams": {
            "learning_rate": float(args.learning_rate),
            "unembedding_lr": float(unembedding_lr),
            "embedding_lr": float(embedding_lr),
            "matrix_lr": float(matrix_lr),
            "weight_decay": float(weight_decay),
            "grad_clip_norm": (float(grad_clip_norm) if grad_clip_norm is not None else None),
            "model_type": model_type,
            "tokenizer": tokenizer_meta,
            "data_shuffle": data_shuffle,
            "scheduler_type": str(args.scheduler_type),  # arm detection for the G2 verdict engine
            # arm detection for semiring_beta variants (rgyl): the RAW spec -
            # "linear:1:32" vs "32.0" vs null (exact tropical) - so annealed,
            # fixed-beta, and endpoint runs are distinguishable as evidence
            "semiring_beta_spec": (
                str(args.semiring_beta)
                if getattr(args, "semiring_beta", None) is not None and "tropical" in attention_schedule
                else None
            ),
            "synaptic_config": (asdict(config.syn_cfg) if isinstance(config, GPTSynapticConfig) else None),
            "val_interval": val_interval,
            "val_batches": val_batches if val_interval > 0 else None,
        },
        "compile": {
            "enabled": compiled_model,
            "backend": compile_backend,
            "mode": compile_mode,
            "fullgraph": compile_fullgraph,
            "dynamic": compile_dynamic,
            "compile_flex_attention": bool(getattr(config, "compile_flex_attention", False)),
        },
        "config": config_artifact,
        "dataset": {
            "data_dir": data_dir,  # null = the FineWeb cache
            "parquet_files_count": len(parquet_files),
            "parquet_files": parquet_files,
        },
        "budget": budget,
        "provenance": provenance,
        "checkpointing": checkpointing,
        "resume": resume_record,
        "results": results,
        "numerics": {
            "check_numerics": check_numerics,
            "detect_anomaly": detect_anomaly,
        },
    }

    report_table = Table(title="nanochat summary", box=box.ROUNDED)
    report_table.add_column("Field", style="cyan")
    report_table.add_column("Value", style="white")
    commit_label = git_info.get("commit_full") or git_info.get("commit") or "unknown"
    dirty = " (dirty)" if git_info.get("dirty") else ""
    report_table.add_row("Artifacts", f"{artifacts_kind}/{artifacts_topic}/{resolved_run_id}")
    report_table.add_row("Commit", f"{commit_label}{dirty}")
    report_table.add_row("Device", str(device))
    if model_type == "gpt":
        report_table.add_row("Attention schedule", attention_label)
    report_table.add_row("Steps", str(max_steps))
    report_table.add_row("Warmup Steps", str(args.warmup_steps))
    report_table.add_row("LR warmdown ratio", str(args.lr_warmdown_ratio))
    report_table.add_row("Parameterization", str(getattr(config, "parameterization", "current")))
    report_table.add_row("Activation ckpt", str(getattr(config, "activation_ckpt", "none")))
    report_table.add_row("check_numerics", str(check_numerics))
    report_table.add_row("detect_anomaly", str(detect_anomaly))
    report_table.add_row("torch.compile model", "enabled" if compiled_model else "disabled")
    if bool(getattr(config, "use_flex_attention", False)):
        report_table.add_row(
            "compile flex_attention",
            "enabled" if bool(getattr(config, "compile_flex_attention", False)) else "disabled",
        )
    ca_init_rule = getattr(config, "ca_init_rule", None)
    if ca_init_rule:
        report_table.add_row(
            "CA init",
            f"{ca_init_rule} (alpha={getattr(config, 'ca_init_alpha', None)}, seed={getattr(config, 'ca_init_seed', None)})",
        )
    if compiled_model or bool(getattr(config, "compile_flex_attention", False)):
        report_table.add_row("compile backend", repr(compile_backend))
        report_table.add_row("compile mode", repr(compile_mode))
        report_table.add_row("compile fullgraph", str(compile_fullgraph))
        report_table.add_row("compile dynamic", str(compile_dynamic))
    report_table.add_row("Tokens/s", f"{tokens_per_second:,.0f}")
    report_table.add_row("TFLOP/s (est)", f"{est_tflops:.2f}")
    if peak_mem_gb is not None:
        report_table.add_row("Peak Mem (GB)", f"{peak_mem_gb:.2f}")
    report_table.add_row("Final Train CE", f"{final_train_ce:.4f}")
    if val_interval > 0:
        report_table.add_row("Val Interval", str(val_interval))
        report_table.add_row("Val Batches", str(val_batches))
        if final_val_ce is not None:
            report_table.add_row("Final Val CE", f"{final_val_ce:.4f}")
        if final_val_bpb is not None:
            report_table.add_row("Final Val bits/byte", f"{final_val_bpb:.4f}")
    console.print(report_table)

    report_md = f"""# nanochat run (fixed FLOPs)

- Run ID: `{resolved_run_id}`
- Generated: {generated_at}
- Artifacts: `{artifacts_kind}/{artifacts_topic}/{resolved_run_id}`
- Commit: {commit_label}{dirty}
{"- Attention schedule: `" + attention_label + "`" if model_type == "gpt" else ""}

## Command

```bash
{uv_command}
```

## Budget

- steps: {max_steps}
- warmup_steps: {args.warmup_steps}
- tokens/step (global): {tokens_per_step:,}
- FLOPs/token (est): {flops_per_token:,}
- FLOPs/step (est): {flops_per_step:,}
- planned_total_FLOPs (est): {max_steps * flops_per_step:,}

## Compilation

- torch.compile: {compiled_model}
- compile_backend: {compile_backend!r}
- compile_mode: {compile_mode!r}
- compile_fullgraph: {compile_fullgraph}
- compile_dynamic: {compile_dynamic}
- compile_flex_attention: {bool(getattr(config, "compile_flex_attention", False))}

## Numerics (debug)

- check_numerics: {check_numerics}
- detect_anomaly: {detect_anomaly}

## Results (measured after warmup)

- measured_steps: {measured_steps}
- measured_tokens: {measured_tokens:,}
- measured_time_s: {measured_time_s:.3f}
- tokens/s: {tokens_per_second:,.0f}
- TFLOP/s (est): {est_tflops:.2f}
- peak_memory_allocated_gb: {peak_mem_gb if peak_mem_gb is not None else "n/a"}
- final_train_ce: {final_train_ce:.4f}
- val_interval: {val_interval}
- val_batches: {val_batches if val_interval > 0 else "n/a"}
- final_val_ce: {final_val_ce if final_val_ce is not None else "n/a"}
- final_val_bpb: {final_val_bpb if final_val_bpb is not None else "n/a"} (bits per byte; comparable across tokenizers)

See `summary.json` for full details.
"""

    _write_artifacts(run_dir, summary=summary, report_md=report_md)
    console.print(f"[bold green]Wrote artifacts[/bold green] → {run_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        type=str,
        default="gpt",
        choices=["gpt", "synaptic"],
        help="Model architecture: GPT (default) or GPTSynaptic.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument(
        "--unembedding-lr",
        type=float,
        default=None,
        help="Override lm_head LR (defaults to --learning-rate when unset).",
    )
    parser.add_argument(
        "--embedding-lr",
        type=float,
        default=None,
        help="Override token embedding LR (defaults to --learning-rate when unset).",
    )
    parser.add_argument(
        "--matrix-lr",
        type=float,
        default=None,
        help="Override matrix/Muon LR (defaults to --learning-rate when unset).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay (applies to embedding + lm_head groups).",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="Clip global grad norm to this value (disabled when unset).",
    )
    parser.add_argument("--optimizer-type", type=str, default="adamw", choices=_SUPPORTED_OPTIMIZER_TYPES)
    parser.add_argument("--attention-type", type=str, default="standard", choices=_SUPPORTED_ATTENTION_TYPES)
    parser.add_argument(
        "--attention-schedule",
        type=str,
        default=None,
        help=(
            "Comma-separated per-layer attention pattern. The pattern must be exactly n_layer entries "
            "or a shorter pattern whose length evenly divides n_layer, e.g. standard,tropical."
        ),
    )
    parser.add_argument(
        "--ffn-type",
        type=str,
        default="standard",
        choices=_SUPPORTED_FFN_TYPES,
        help="FFN structure: standard ReLU^2 MLP, pure max-plus (1-Lipschitz), or tropical-rational (8gk.8)",
    )
    parser.add_argument(
        "--ffn-beta",
        type=float,
        default=None,
        help="Maslov smoothing for tropical FFN modes: omit for the exact max endpoint, >0 for (+)_beta",
    )
    parser.add_argument(
        "--standard-record-attn-entropy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Standard attention only: record per-head attention entropy from the last forward pass "
            "(stored in summary.json; debug-only and can be expensive for large sequence lengths)."
        ),
    )
    parser.add_argument(
        "--braid-mode",
        type=str,
        default=os.environ.get("NANOCHAT_BRAID_MODE", "soft"),
        choices=["soft", "discrete"],
        help="Braid attention only: soft (sigmoid weights) vs discrete (hard threshold + optional schedule/verification).",
    )
    parser.add_argument(
        "--braid-tau",
        type=float,
        default=float(os.environ.get("NANOCHAT_BRAID_TAU", "0.0")),
        help="Braid attention only: threshold tau for discrete selection (applied to the braid score matrix).",
    )
    parser.add_argument(
        "--braid-crossing-law",
        type=str,
        default=os.environ.get("NANOCHAT_BRAID_CROSSING_LAW", "restricted"),
        choices=["restricted", "ybe", "rmatrix"],
        help=(
            "Braid attention only: restricted (fast, non-YBE) vs ybe (swap-output, YBE-valid) vs "
            "rmatrix (trigonometric U_q(sl2) R-matrix with learned per-head deformation eta and "
            "per-position rapidities; integrable mixing with conserved-charge telemetry, bead u55.3)."
        ),
    )
    parser.add_argument(
        "--braid-record-schedule",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Braid attention only: record per-head discrete schedule for KV-cache decode (Tq==1).",
    )
    parser.add_argument(
        "--braid-verify",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Braid attention only: verify discrete decode invariants; and check YBE once when braid-crossing-law=ybe.",
    )
    parser.add_argument(
        "--braid-rmatrix-probes",
        type=int,
        default=int(os.environ.get("NANOCHAT_BRAID_RMATRIX_PROBES", "0")),
        help=(
            "Braid rmatrix law only: number of spectral multi-view probe sweeps per head "
            "(0i1v; zero-init gates, 0 = off). The causal realization of the TL/Markov-trace readout."
        ),
    )
    parser.add_argument(
        "--ultrametric-mode",
        type=str,
        default=os.environ.get("NANOCHAT_ULTRAMETRIC_MODE", "kernel"),
        choices=["kernel", "trie", "balltree"],
        help=(
            "Ultrametric attention only: kernel (continuous, O(T^2 K)) vs trie (packed-prefix decode path) vs "
            "balltree (exact hard-digit ball-tree, O(K T log T), training/prefill/decode; bead 33dd)."
        ),
    )
    parser.add_argument(
        "--ultrametric-hard-digits",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ultrametric attention only: quantize digits before computing LCP weights (default: off).",
    )
    parser.add_argument(
        "--ultrametric-K",
        dest="ultrametric_K",
        type=int,
        default=None,
        help=(
            "Ultrametric attention only: number of p-adic digits K (default: GPTConfig.ultrametric_K = 8). "
            "Finer K extends the digit-truncation memory-fraction axis downward (k=1 reaches 1/K; bead kgj1)."
        ),
    )
    parser.add_argument(
        "--reversible-mode",
        type=str,
        default="additive",
        choices=["additive", "symplectic"],
        help=(
            "Reversible attention only: additive coupling (any F, G; volume-preserving) vs symplectic "
            "kick-kick coupling (exact-gradient kicks, corrected sign; conserves a coercive shadow "
            "Hamiltonian - bead u55.5, theory note symplectic_transformer.md)."
        ),
    )
    parser.add_argument(
        "--reversible-tied",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Reversible+symplectic only: share ONE block across all layers (the regime where shadow "
            "conservation is theory-backed; also a parameter-efficiency win). No kv-cache generation."
        ),
    )
    parser.add_argument(
        "--reversible-lambda-min",
        dest="reversible_lambda_min",
        type=float,
        default=None,
        help=(
            "Reversible+symplectic only: coercivity floor for the learnable confinement lambda "
            "(default GPTConfig 0.05; 0 = the registered unconfined falsification control)."
        ),
    )
    parser.add_argument(
        "--reversible-record-energy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reversible+symplectic only: stream shadow-energy/activation-norm telemetry into metrics.jsonl.",
    )
    parser.add_argument(
        "--disable-block-norms",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Standard attention only: strip the per-layer norms - the expected-failure falsification "
            "arm of the symplectic no-norm program (bead z4xx)."
        ),
    )
    parser.add_argument(
        "--dashboard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Live rich console dashboard (bead nyp): loss/val/grad-norm/throughput sparklines, feature "
            "flags, and an auto-adopted mathematical-invariant drift panel rendered from the metrics "
            "stream. HTML snapshot saved to artifacts/dashboard/<run_id>.html on close."
        ),
    )
    parser.add_argument(
        "--dashboard-html",
        type=str,
        default=None,
        help="Override the dashboard HTML export path (default: <artifacts-dir>/dashboard/<run_id>.html).",
    )
    parser.add_argument(
        "--tropical-gauge-fix",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Tropical attention only: enable per-vector gauge-fixing (subtract max so max=0).",
    )
    parser.add_argument(
        "--tropical-score-center",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Tropical attention only: center per-query scores by subtracting max over keys (pure gauge shift).",
    )
    parser.add_argument(
        "--tropical-record-margins",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Tropical attention only: compute runner-up margins (gamma) and store per-head stats for logging.",
    )
    parser.add_argument(
        "--tropical-log-margins",
        action="store_true",
        help="Tropical attention only: include gamma summary + per-head means in the training log (implies --tropical-record-margins).",
    )
    parser.add_argument(
        "--semiring-beta",
        type=str,
        default=None,
        help=(
            "Tropical attention only (8gk.1 dequantization annealing): Maslov smoothing level. "
            "A float fixes beta; 'linear:B0:B1' or 'exp:B0:B1' anneals beta from B0 to B1 over the run "
            "(beta -> infinity is the exact tropical endpoint; omit the flag for exact tropical). "
            "With --tropical-record-margins, logs semiring_beta + certificate route_coverage per step."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--ca-init-rule",
        type=str,
        default=os.environ.get("NANOCHAT_CA_INIT_RULE", "none"),
        help="Optional CA initializer for weights: none|rule30|rule116 (env: NANOCHAT_CA_INIT_RULE).",
    )
    parser.add_argument(
        "--ca-init-alpha",
        type=float,
        default=float(os.environ.get("NANOCHAT_CA_INIT_ALPHA", "1.0")),
        help="Mixing ratio alpha*CA + (1-alpha)*standard (env: NANOCHAT_CA_INIT_ALPHA).",
    )
    parser.add_argument(
        "--ca-init-seed",
        type=int,
        default=None,
        help="Seed for CA initializer; defaults to --seed when unset (env: NANOCHAT_CA_INIT_SEED).",
    )
    parser.add_argument("--n-layer", type=int, default=4, help="Number of transformer layers.")
    parser.add_argument("--n-head", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--n-kv-head", type=int, default=4, help="Number of key/value heads (GQA).")
    parser.add_argument("--n-embd", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--sequence-len", type=int, default=256, help="Sequence length.")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=GPTConfig().vocab_size,
        help="Vocabulary size (model embedding table size) for the GPT-2 tokenizer; derived automatically with --tokenizer task.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        choices=list(_SUPPORTED_TOKENIZERS),
        help="gpt2 = the shared 50,257-token tokenizer; task = train a byte-level BPE on --data-dir's train split "
        "and save it inside the checkpoint dir (at small width the GPT-2 vocabulary is ~97%% of every FLOP).",
    )
    parser.add_argument(
        "--data-shuffle",
        type=str,
        default="epoch",
        choices=["epoch", "none"],
        help="epoch = visit each parquet row group's documents in a per-epoch permutation seeded by --seed "
        "(a small corpus replayed in file order every epoch is memorized as a sequence); none = file order.",
    )
    parser.add_argument(
        "--tokenizer-vocab-size",
        type=int,
        default=TASK_TOKENIZER_DEFAULT_VOCAB,
        help=f"Target vocabulary for --tokenizer task (>= {TASK_TOKENIZER_MIN_VOCAB}); the embedding table is padded to a multiple of 64.",
    )
    parser.add_argument(
        "--synaptic-config",
        type=str,
        default=None,
        help="Path to JSON file with SynapticConfig overrides (only used for --model-type synaptic).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=20, help="Max training steps (ignored if --target-flops is set)."
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=0,
        help="Run validation every N steps (0 = disabled).",
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=10,
        help="Number of batches to evaluate during validation (default: 10).",
    )
    parser.add_argument(
        "--target-flops",
        type=float,
        default=None,
        help="Target total FLOPs budget (est). If set, compute steps from model.estimate_flops().",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=2, help="Warmup steps excluded from throughput measurement."
    )
    parser.add_argument("--log-interval", type=int, default=1, help="Log every N steps.")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Base directory for run artifacts (default: artifacts/).",
    )
    parser.add_argument(
        "--artifacts-kind",
        type=str,
        default="baseline",
        help="Artifacts category subdir under <artifacts-dir>/ (e.g., baseline, bench, perf).",
    )
    parser.add_argument(
        "--artifacts-topic",
        type=str,
        default="nanochat",
        help="Artifacts topic subdir under <artifacts-dir>/<artifacts-kind>/ (may include subdirs).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (directory name). Defaults to YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--auto-download-data",
        action="store_true",
        help="If dataset shards are missing, download a minimal set (>=2 parquet shards).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Train on a custom parquet corpus directory (FineWeb convention: sorted, last file is val) - "
            "e.g. an `mgr gen-tasks` output like artifacts/diagnostics/hier. Default: the nanochat cache."
        ),
    )
    parser.add_argument(
        "--min-parquet-files",
        type=int,
        default=2,
        help="Minimum number of parquet shards required (>=2 recommended: 1 train + 1 val).",
    )
    parser.add_argument(
        "--use-flex-attention",
        action="store_true",
        help="Use torch FlexAttention for standard attention (requires torch>=2.5).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for the model (optional; may improve throughput after warmup).",
    )
    parser.add_argument(
        "--compile-backend",
        type=str,
        default="inductor",
        help="torch.compile backend (e.g., inductor, aot_eager).",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default=None,
        help="torch.compile mode (e.g., default, reduce-overhead, max-autotune).",
    )
    parser.add_argument(
        "--compile-fullgraph",
        action="store_true",
        help="Pass fullgraph=True to torch.compile (stricter; may fail on graph breaks).",
    )
    parser.add_argument(
        "--compile-dynamic",
        type=str,
        default="auto",
        help="torch.compile dynamic setting: true|false|auto (auto maps to None).",
    )
    parser.add_argument(
        "--compile-flex-attention",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compile FlexAttention callable when --use-flex-attention; defaults to --compile when unset.",
    )
    parser.add_argument(
        "--check-numerics",
        action="store_true",
        help="Enable NaN/Inf watchpoints (loss + gradients); prints diagnostics and raises on failure.",
    )
    parser.add_argument(
        "--detect-anomaly",
        action="store_true",
        help="Enable torch.autograd anomaly detection (very slow; debug only).",
    )
    parser.add_argument("--scheduler-type", type=str, default="none", choices=_SUPPORTED_SCHEDULER_TYPES)
    parser.add_argument(
        "--lr-warmdown-ratio",
        type=float,
        default=0.0,
        help="Linearly decay the LR to 0 over the final FRACTION of total steps "
        "(WSD tail for non-ordinal runs, bio_inspired parity; default 0.0 = flat LR).",
    )
    parser.add_argument(
        "--parameterization",
        type=str,
        default="current",
        choices=["current", "nsa"],
        help="Width-scaling parameterization arm (lab.1/bp08): 'nsa' applies the "
        "derived per-stage corrections (tropical exact-E[max] constants); default 'current'.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        type=str,
        default="none",
        choices=["none", "full", "every-k"],
        help="Trade memory for compute: checkpoint every block (full) or every k-th block "
        "(every-k + --activation-ckpt-every-k) so backward recomputes instead of storing "
        "activations. Numerically identical trajectories; default none.",
    )
    parser.add_argument("--activation-ckpt-every-k", type=int, default=1, help="Block stride for every-k mode.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument(
        "--dataloader-prefetch",
        type=int,
        default=0,
        help="Tokenizer-chunk depth for the dataloader's background prefetch thread "
        "(bead atkp; hides encode refill stalls behind GPU steps). 0 = synchronous loading.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the last --profile-steps training steps (b1l torch.profiler hooks): aggregate op table + chrome trace under <run-dir>/profile/. Debug tool; zero overhead when off.",
    )
    parser.add_argument(
        "--profile-steps", type=int, default=5, help="How many of the FINAL steps to profile when --profile is on."
    )
    # Checkpoint/resume (bead rz8.1).
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Save a full training-state checkpoint every N steps (0 disables checkpointing).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory (default: <artifacts>/<kind>/<topic>/<run-id>/checkpoints).",
    )
    parser.add_argument(
        "--checkpoint-keep",
        type=int,
        default=0,
        help="Retain only the newest K checkpoints saved by THIS run (0 keeps all).",
    )
    parser.add_argument(
        "--checkpoint-verify",
        action="store_true",
        help="After each save, reload and compare every tensor bitwise (corruption tripwire).",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from a checkpoint directory, or 'latest' to scan --checkpoint-dir.",
    )
    parser.add_argument(
        "--resume-step",
        type=int,
        default=None,
        help="Checkpoint step to resume from (default: the last step found in the directory).",
    )
    parser.add_argument(
        "--resume-data-mode",
        type=str,
        default="exact",
        choices=["exact", "approximate"],
        help=(
            "exact: replay+discard the consumed data prefix (bitwise-resumable; cost linear in the prefix). "
            "approximate: the loader's native row-group skip (cheap; may skip a few documents)."
        ),
    )
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
