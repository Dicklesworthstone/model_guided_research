from __future__ import annotations

"""CA-init early-phase benchmark (bead model_guided_research-m32).

Compares cellular-automata weight initialization (``--ca-init-rule``) against
the standard init, and against an alpha-blended "mix", on small models at
matched FLOPs/seeds. For each (config, variant, seed) cell it captures:

* the early-phase **loss curve** (from the real training path -- we shell out
  to ``python -m nanochat.train`` exactly like ``scripts/cmaes_phase1.py`` so
  the numbers come from production training, not a re-implemented loop),
* per-step **grad-norm** statistics (from the run's ``metrics.jsonl``), and
* **init-time activation statistics** -- a cheap in-process forward pass that
  records the residual-stream RMS at each block. This is the most direct probe
  of init quality: a good init keeps activations well-scaled through depth; a
  bad one explodes or collapses them.

Two architectures are swept: one **attention-heavy** (more heads, longer
sequence) and one **MLP-heavy** (wider embedding, short sequence), per the m32
acceptance criteria. Results land under ``artifacts/ca_init/<run_id>/`` with a
machine-readable ``summary.json`` and a human-readable ``report.md``; every
exact training command is recorded for reproduction.

Example
-------
    uv run python scripts/ca_init_bench.py --run-id m32_cpu --device cpu \
        --max-steps 200 --seeds 0 1
"""

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()

SCHEMA_VERSION = "mgr.ca_init.bench.v1"
PENALTY_LOSS = float("inf")


@dataclass(frozen=True)
class ArchConfig:
    """A small model architecture under test."""

    name: str
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int
    sequence_len: int
    batch_size: int
    note: str

    def flags(self) -> list[str]:
        return [
            "--n-layer",
            str(self.n_layer),
            "--n-head",
            str(self.n_head),
            "--n-kv-head",
            str(self.n_kv_head),
            "--n-embd",
            str(self.n_embd),
            "--sequence-len",
            str(self.sequence_len),
            "--batch-size",
            str(self.batch_size),
        ]


@dataclass(frozen=True)
class InitVariant:
    """An initialization scheme: standard, pure-CA, or alpha-blended mix."""

    name: str
    ca_rule: str | None
    ca_alpha: float
    note: str

    def flags(self) -> list[str]:
        if self.ca_rule is None:
            return ["--ca-init-rule", "none"]
        return [
            "--ca-init-rule",
            str(self.ca_rule),
            "--ca-init-alpha",
            f"{self.ca_alpha:g}",
        ]


# Two regimes that stress different parameter blocks. Attention-heavy: 8 heads
# over a long 256-token window -> the O(T^2) attention and qkv projections
# carry most of the per-block work. MLP-heavy: a short 32-token window with
# more depth -> attention (~T^2) is negligible and the FFN's c_fc weights
# dominate the per-block work. Both use n_embd=128 and standard attention so
# the only difference is the attention-vs-FFN balance (and the short window
# keeps the MLP-heavy config cheap on CPU). CA init touches every
# nn.Linear/nn.Embedding regardless of mechanism, so standard is the cleanest
# probe of the initializer itself.
ARCH_CONFIGS: dict[str, ArchConfig] = {
    "attn_heavy": ArchConfig(
        name="attn_heavy",
        n_layer=4,
        n_head=8,
        n_kv_head=8,
        n_embd=128,
        sequence_len=256,
        batch_size=4,
        note="8 heads / 256-token window: attention + qkv projections dominate",
    ),
    "mlp_heavy": ArchConfig(
        name="mlp_heavy",
        n_layer=6,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
        sequence_len=32,
        batch_size=8,
        note="6 layers / 32-token window: FFN per-block work dominates, attention negligible",
    ),
}

INIT_VARIANTS: dict[str, InitVariant] = {
    "standard": InitVariant("standard", None, 1.0, "baseline normal_ init"),
    "ca_rule30": InitVariant("ca_rule30", "rule30", 1.0, "pure CA (rule30) init"),
    "ca_mix0.5": InitVariant("ca_mix0.5", "rule30", 0.5, "0.5*CA(rule30) + 0.5*standard"),
    "ca_mix0.25": InitVariant("ca_mix0.25", "rule30", 0.25, "0.25*CA(rule30) + 0.75*standard (bias-like)"),
}

# bench mode sweeps these three; ca_mix0.25 is reserved for the retention
# (bias-axis) experiment so adding it does not change the default bench sweep.
BENCH_VARIANTS = ["standard", "ca_rule30", "ca_mix0.5"]
RETENTION_VARIANTS = ["standard", "ca_rule30", "ca_mix0.5", "ca_mix0.25"]


@dataclass
class CellResult:
    config: str
    variant: str
    seed: int
    status: str
    command: str
    train_dir: str | None = None
    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    duration_s: float = 0.0
    activation: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def loss_first(self) -> float | None:
        return self.losses[0] if self.losses else None

    def loss_final(self) -> float | None:
        return self.losses[-1] if self.losses else None

    def loss_drop(self) -> float | None:
        if len(self.losses) >= 2:
            return self.losses[0] - self.losses[-1]
        return None

    def grad_stat(self, which: str) -> float | None:
        if not self.grad_norms:
            return None
        finite = [g for g in self.grad_norms if math.isfinite(g)]
        if not finite:
            return None
        if which == "mean":
            return sum(finite) / len(finite)
        if which == "max":
            return max(finite)
        if which == "final":
            return finite[-1]
        raise ValueError(which)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _probe_init_activations(
    arch: ArchConfig, variant: InitVariant, seed: int, *, device: str, vocab_size: int
) -> dict[str, Any]:
    """Build the model with the given init, run one forward, and record the
    residual-stream RMS at the output of every transformer block. Cheap
    (single forward, no training) and the most direct readout of init quality.
    """
    import torch

    from nanochat.gpt import GPT, GPTConfig

    cfg = GPTConfig()
    cfg.n_layer = arch.n_layer
    cfg.n_head = arch.n_head
    cfg.n_kv_head = arch.n_kv_head
    cfg.n_embd = arch.n_embd
    cfg.sequence_len = arch.sequence_len
    cfg.vocab_size = vocab_size
    cfg.attention_type = "standard"
    cfg.ca_init_rule = variant.ca_rule
    cfg.ca_init_alpha = variant.ca_alpha
    cfg.ca_init_seed = seed

    torch.manual_seed(seed)
    dev = torch.device(device if device != "auto" else "cpu")
    model = GPT(cfg).to(dev)
    model.init_weights()
    model.eval()

    block_rms: list[float] = []
    # Per-Linear output RMS, keyed by module name. The "input/expansion"
    # projections (c_q/c_k/c_v/c_fc) are NOT zeroed at init, so their output
    # scale directly reflects the initializer; c_proj/lm_head are deliberately
    # zeroed (the residual stream is constant at init by construction), so they
    # would mask the signal -- we report the input projections separately.
    proj_rms: dict[str, float] = {}
    handles = []

    def _rms(t: Any) -> float:
        return float(t.detach().float().pow(2).mean().sqrt().item())

    def _mk_block_hook() -> Any:
        def _hook(_module: Any, _inp: Any, out: Any) -> None:
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                block_rms.append(_rms(t))

        return _hook

    def _mk_linear_hook(qual: str) -> Any:
        def _hook(_module: Any, _inp: Any, out: Any) -> None:
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                proj_rms[qual] = _rms(t)

        return _hook

    for block in model._blocks():
        handles.append(block.register_forward_hook(_mk_block_hook()))
    for qual, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(_mk_linear_hook(qual)))

    # init-time weight scale, sampled over the matrix params (excludes the
    # deliberately-zeroed lm_head / c_proj weights, which would skew the std).
    wstats: list[float] = []
    for name, p in model.named_parameters():
        if p.dim() >= 2 and "c_proj" not in name and "lm_head" not in name:
            wstats.append(float(p.detach().float().std().item()))
    weight_std_mean = sum(wstats) / len(wstats) if wstats else None

    g = torch.Generator(device="cpu").manual_seed(seed + 7919)
    idx = torch.randint(0, vocab_size, (2, arch.sequence_len), generator=g).to(dev)
    with torch.no_grad():
        model(idx, targets=None)

    for h in handles:
        h.remove()

    finite_rms = [r for r in block_rms if math.isfinite(r)]
    # mean output RMS over the non-zeroed input/expansion projections
    input_proj = [
        v for k, v in proj_rms.items() if any(s in k for s in ("c_q", "c_k", "c_v", "c_fc")) and math.isfinite(v)
    ]
    return {
        "block_residual_rms": block_rms,
        "block_residual_rms_mean": (sum(finite_rms) / len(finite_rms)) if finite_rms else None,
        "block_residual_rms_max": (max(finite_rms) if finite_rms else None),
        "depth_rms_ratio": ((block_rms[-1] / block_rms[0]) if len(block_rms) >= 2 and block_rms[0] > 0 else None),
        "input_proj_rms_mean": (sum(input_proj) / len(input_proj)) if input_proj else None,
        "weight_std_mean": weight_std_mean,
        "all_finite": (all(math.isfinite(r) for r in block_rms) if block_rms else False)
        and all(math.isfinite(v) for v in proj_rms.values()),
    }


def _run_train_cell(
    *,
    arch: ArchConfig,
    variant: InitVariant,
    seed: int,
    run_dir: Path,
    artifacts_dir: Path,
    args: argparse.Namespace,
    checkpoint: bool = False,
) -> CellResult:
    topic = f"{run_dir.name}/runs/{arch.name}__{variant.name}"
    cell_run_id = f"seed{seed}"
    train_cmd = [
        sys.executable,
        "-m",
        "nanochat.train",
        "--device",
        str(args.device),
        "--seed",
        str(seed),
        "--attention-type",
        "standard",
        "--optimizer-type",
        str(args.optimizer_type),
        "--learning-rate",
        str(args.learning_rate),
        "--max-steps",
        str(args.max_steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--log-interval",
        "1",
        "--vocab-size",
        str(args.vocab_size),
        *arch.flags(),
        *variant.flags(),
        "--artifacts-dir",
        str(artifacts_dir),
        "--artifacts-kind",
        "ca_init",
        "--artifacts-topic",
        topic,
        "--run-id",
        cell_run_id,
    ]
    if checkpoint:
        # save the final model so retention can compare trained vs init weights
        train_cmd += ["--checkpoint-interval", str(args.max_steps), "--checkpoint-keep", "1"]
    if args.auto_download_data:
        train_cmd += ["--auto-download-data", "--min-parquet-files", str(args.min_parquet_files)]

    cmd_str = shlex.join(train_cmd)
    res = CellResult(config=arch.name, variant=variant.name, seed=seed, status="pending", command=cmd_str)

    t0 = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=float(args.timeout_s),
            check=False,
        )
        returncode = int(proc.returncode)
        stderr_tail = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        raw_stderr = exc.stderr or ""
        stderr_tail = raw_stderr.decode(errors="replace") if isinstance(raw_stderr, bytes) else raw_stderr
    res.duration_s = time.perf_counter() - t0

    train_dir = artifacts_dir / "ca_init" / topic / cell_run_id
    res.train_dir = str(train_dir.relative_to(artifacts_dir)) if train_dir.exists() else None
    summary_path = train_dir / "summary.json"
    metrics_path = train_dir / "metrics.jsonl"

    if timed_out:
        res.status = "timeout"
    elif returncode != 0 or not summary_path.exists():
        res.status = "error"
        if stderr_tail:
            res.notes.append("stderr_tail: " + stderr_tail.strip().splitlines()[-1][:200])
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        res.losses = [float(x) for x in summary.get("results", {}).get("losses", [])]
        if metrics_path.exists():
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("type") == "step" and rec.get("grad_norm") is not None:
                    res.grad_norms.append(float(rec["grad_norm"]))
        finite_loss = all(math.isfinite(x) for x in res.losses) if res.losses else False
        res.status = "ok" if (res.losses and finite_loss) else "error"
        if not finite_loss:
            res.notes.append("non-finite loss encountered")
    return res


def _build_summary(run_id: str, args: argparse.Namespace, cells: list[CellResult]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": shlex.join(["uv", "run", "python", "scripts/ca_init_bench.py"] + sys.argv[1:]),
        "settings": {
            "device": str(args.device),
            "max_steps": int(args.max_steps),
            "warmup_steps": int(args.warmup_steps),
            "learning_rate": float(args.learning_rate),
            "optimizer_type": str(args.optimizer_type),
            "vocab_size": int(args.vocab_size),
            "seeds": list(args.seeds),
            "configs": {k: vars(v) for k, v in ARCH_CONFIGS.items() if k in args.configs},
            "variants": {k: vars(v) for k, v in INIT_VARIANTS.items() if k in args.variants},
        },
        "cells": [
            {
                "config": c.config,
                "variant": c.variant,
                "seed": c.seed,
                "status": c.status,
                "duration_s": round(c.duration_s, 3),
                "loss_first": c.loss_first(),
                "loss_final": c.loss_final(),
                "loss_drop": c.loss_drop(),
                "grad_norm_mean": c.grad_stat("mean"),
                "grad_norm_max": c.grad_stat("max"),
                "grad_norm_final": c.grad_stat("final"),
                "activation": c.activation,
                "losses": c.losses,
                "train_dir": c.train_dir,
                "command": c.command,
                "notes": c.notes,
            }
            for c in cells
        ],
    }


def _aggregate(cells: list[CellResult]) -> dict[tuple[str, str], dict[str, float | None]]:
    """Mean over seeds of the key metrics per (config, variant)."""
    groups: dict[tuple[str, str], list[CellResult]] = {}
    for c in cells:
        groups.setdefault((c.config, c.variant), []).append(c)

    def _mean(vals: list[float | None]) -> float | None:
        ok = [v for v in vals if v is not None and math.isfinite(v)]
        return (sum(ok) / len(ok)) if ok else None

    out: dict[tuple[str, str], dict[str, float | None]] = {}
    for key, cs in groups.items():
        ok_cells = [c for c in cs if c.status == "ok"]
        # activation probe runs for every cell regardless of training outcome
        act_cells = [c for c in cs if c.activation and "error" not in c.activation]
        out[key] = {
            "n_ok": len(ok_cells),
            "n_total": len(cs),
            "loss_final": _mean([c.loss_final() for c in ok_cells]),
            "loss_drop": _mean([c.loss_drop() for c in ok_cells]),
            "grad_norm_mean": _mean([c.grad_stat("mean") for c in ok_cells]),
            "grad_norm_max": _mean([c.grad_stat("max") for c in ok_cells]),
            "act_input_proj_rms": _mean([c.activation.get("input_proj_rms_mean") for c in act_cells]),
            "act_rms_mean": _mean([c.activation.get("block_residual_rms_mean") for c in act_cells]),
            "act_depth_ratio": _mean([c.activation.get("depth_rms_ratio") for c in act_cells]),
            "weight_std_mean": _mean([c.activation.get("weight_std_mean") for c in act_cells]),
        }
    return out


def _render_report(run_id: str, args: argparse.Namespace, cells: list[CellResult]) -> str:
    agg = _aggregate(cells)
    lines: list[str] = []
    lines.append(f"# CA-init early-phase benchmark — `{run_id}`\n")
    lines.append(
        "Bead model_guided_research-m32. Compares CA weight init "
        "(`--ca-init-rule`) vs the standard init vs an alpha-blended mix on small "
        "attention-heavy and MLP-heavy models, at matched steps/seeds.\n"
    )
    lines.append(
        f"- Device: `{args.device}`  ·  steps: `{args.max_steps}`  ·  "
        f"warmup: `{args.warmup_steps}`  ·  lr: `{args.learning_rate}`  ·  "
        f"optimizer: `{args.optimizer_type}`  ·  seeds: `{list(args.seeds)}`\n"
    )

    lines.append("## Architectures\n")
    for name in args.configs:
        a = ARCH_CONFIGS[name]
        lines.append(f"- **{name}** — L{a.n_layer} H{a.n_head} d{a.n_embd} T{a.sequence_len} B{a.batch_size}: {a.note}")
    lines.append("")

    lines.append("## Results (mean over seeds)\n")
    lines.append(
        "Loss/grad columns aggregate trained cells; activation/weight columns come "
        "from the in-process init-time probe (runs for every cell).\n"
    )
    lines.append(
        "| config | variant | ok | final loss | loss drop | grad‖·‖ mean | grad‖·‖ max | in-proj act RMS | resid RMS | depth ratio | init w-std |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for cfg in args.configs:
        for var in args.variants:
            m = agg.get((cfg, var))
            if not m:
                continue

            def f(x: float | None, p: str = ".4g") -> str:
                return f"{x:{p}}" if isinstance(x, (int, float)) else "—"

            lines.append(
                f"| {cfg} | {var} | {m['n_ok']}/{m['n_total']} | {f(m['loss_final'])} | "
                f"{f(m['loss_drop'])} | {f(m['grad_norm_mean'])} | {f(m['grad_norm_max'])} | "
                f"{f(m['act_input_proj_rms'])} | {f(m['act_rms_mean'])} | "
                f"{f(m['act_depth_ratio'])} | {f(m['weight_std_mean'])} |"
            )
    lines.append("")

    # Interpretation: CA vs standard per config.
    lines.append("## Interpretation\n")
    for cfg in args.configs:
        base = agg.get((cfg, "standard"))
        if base is None:
            continue
        base_loss_final = base.get("loss_final")
        if base_loss_final is None:
            continue
        lines.append(f"### {cfg}\n")
        for var in args.variants:
            if var == "standard":
                continue
            m = agg.get((cfg, var))
            if m is None:
                continue
            loss_final = m.get("loss_final")
            if loss_final is None:
                continue
            dloss = loss_final - base_loss_final
            verdict = "worse" if dloss > 0.01 else ("better" if dloss < -0.01 else "~tied")
            lines.append(
                f"- **{var}** vs standard: final-loss Δ = `{dloss:+.4f}` ({verdict}); "
                f"act-RMS depth ratio `{m.get('act_depth_ratio')}` "
                f"(standard `{base.get('act_depth_ratio')}`)."
            )
        lines.append("")

    lines.append("## Reproduction\n")
    lines.append("Each cell's exact command is in `summary.json` (`cells[].command`). Top-level command:\n")
    lines.append("```\n" + shlex.join(["uv", "run", "python", "scripts/ca_init_bench.py"] + sys.argv[1:]) + "\n```\n")

    failures = [c for c in cells if c.status != "ok"]
    if failures:
        lines.append("## Failures\n")
        for c in failures:
            lines.append(
                f"- {c.config}/{c.variant}/seed{c.seed}: `{c.status}` {('— ' + '; '.join(c.notes)) if c.notes else ''}"
            )
        lines.append("")
    return "\n".join(lines)


def _print_table(cells: list[CellResult]) -> None:
    agg = _aggregate(cells)
    table = Table(title="CA-init early-phase benchmark (mean over seeds)", show_header=True, header_style="bold")
    for col in ("config", "variant", "ok", "final loss", "loss drop", "grad‖·‖ mean", "in-proj RMS", "init w-std"):
        table.add_column(col, justify="right" if col not in ("config", "variant") else "left")

    def f(x: float | None) -> str:
        return f"{x:.4g}" if isinstance(x, (int, float)) else "—"

    configs = sorted({c.config for c in cells})
    variants = ["standard", "ca_rule30", "ca_mix0.5"]
    for cfg in configs:
        for var in variants:
            m = agg.get((cfg, var))
            if not m:
                continue
            style = None
            base = agg.get((cfg, "standard"))
            loss_final = m.get("loss_final")
            base_loss_final = base.get("loss_final") if base else None
            if var != "standard" and loss_final is not None and base_loss_final is not None:
                style = (
                    "green"
                    if loss_final < base_loss_final - 0.01
                    else ("red" if loss_final > base_loss_final + 0.01 else "yellow")
                )
            table.add_row(
                cfg,
                var,
                f"{m['n_ok']}/{m['n_total']}",
                f(m["loss_final"]),
                f(m["loss_drop"]),
                f(m["grad_norm_mean"]),
                f(m["act_input_proj_rms"]),
                f(m["weight_std_mean"]),
                style=style,
            )
    console.print(table)


def _measure_retention(
    arch: ArchConfig, variant: InitVariant, seed: int, *, cell_train_dir: Path, device: str, vocab_size: int
) -> dict[str, Any]:
    """How much of the init's weight structure survives training: cosine
    similarity between the (deterministically reconstructed) init weights and
    the trained checkpoint, over the CA-initialized tensors (input projections
    c_q/c_k/c_v/c_fc + the wte embedding; c_proj/lm_head are zeroed at init so
    they carry no init structure). Returns mean cosine + mean relative L2 drift.
    """
    import torch
    import torch.nn.functional as F

    from nanochat.checkpoint_manager import find_last_step, load_checkpoint
    from nanochat.gpt import GPT, GPTConfig

    cfg = GPTConfig()
    cfg.n_layer = arch.n_layer
    cfg.n_head = arch.n_head
    cfg.n_kv_head = arch.n_kv_head
    cfg.n_embd = arch.n_embd
    cfg.sequence_len = arch.sequence_len
    cfg.vocab_size = vocab_size
    cfg.attention_type = "standard"
    cfg.ca_init_rule = variant.ca_rule
    cfg.ca_init_alpha = variant.ca_alpha
    cfg.ca_init_seed = seed
    torch.manual_seed(seed)
    model = GPT(cfg)
    model.init_weights()
    init_sd = {k: v.detach().float().clone() for k, v in model.state_dict().items()}

    ckpt_dir = cell_train_dir / "checkpoints"
    step = find_last_step(str(ckpt_dir))
    if step is None:
        return {"error": f"no checkpoint under {ckpt_dir}"}
    final_sd, _, _ = load_checkpoint(str(ckpt_dir), step, device=device)

    cos_by_tensor: dict[str, float] = {}
    rel_by_tensor: dict[str, float] = {}
    for k in init_sd:
        if not (any(s in k for s in ("c_q", "c_k", "c_v", "c_fc")) or k.endswith("wte.weight")):
            continue
        if k not in final_sd:
            continue
        a = init_sd[k].flatten()
        b = final_sd[k].float().flatten()
        if a.norm() == 0:
            continue
        cos_by_tensor[k] = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
        rel_by_tensor[k] = float((b - a).norm().item() / (a.norm().item() + 1e-9))

    cos_vals = list(cos_by_tensor.values())
    rel_vals = list(rel_by_tensor.values())
    return {
        "trained_step": int(step),
        "n_tensors": len(cos_vals),
        "cosine_mean": (sum(cos_vals) / len(cos_vals)) if cos_vals else None,
        "cosine_min": (min(cos_vals) if cos_vals else None),
        "rel_drift_mean": (sum(rel_vals) / len(rel_vals)) if rel_vals else None,
        "cosine_by_tensor": cos_by_tensor,
    }


def _run_retention(args: argparse.Namespace, run_dir: Path, artifacts_dir: Path) -> int:
    """Bead 827: does CA structure persist through training? Trains each
    alpha-bias variant (pure CA -> blended bias -> standard baseline) with a
    final checkpoint, then measures init->trained cosine retention.

    NOTE the freeze-channel arm of 827 would need a train.py --freeze-init-steps
    feature (no default-behavior change here); the implementable arm is the
    bias-axis (alpha) sweep, which directly answers the retention question.
    """
    run_id = run_dir.name
    console.rule(f"[bold magenta]CA-init retention[/bold magenta] · {run_id}")
    console.print(
        f"configs={args.configs}  variants={RETENTION_VARIANTS}  seeds={args.seeds}  "
        f"steps={args.max_steps}  device={args.device}"
    )

    rows: list[dict[str, Any]] = []
    total = len(args.configs) * len(RETENTION_VARIANTS) * len(args.seeds)
    with Progress(
        TextColumn("[bold magenta]retention[/bold magenta]"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("cells", total=total)
        for cfg_name in args.configs:
            arch = ARCH_CONFIGS[cfg_name]
            for var_name in RETENTION_VARIANTS:
                variant = INIT_VARIANTS[var_name]
                for seed in args.seeds:
                    cell = _run_train_cell(
                        arch=arch,
                        variant=variant,
                        seed=seed,
                        run_dir=run_dir,
                        artifacts_dir=artifacts_dir,
                        args=args,
                        checkpoint=True,
                    )
                    ret: dict[str, Any] = {"error": f"train status={cell.status}"}
                    if cell.status == "ok" and cell.train_dir is not None:
                        try:
                            ret = _measure_retention(
                                arch,
                                variant,
                                seed,
                                cell_train_dir=artifacts_dir / cell.train_dir,
                                device=args.device,
                                vocab_size=args.vocab_size,
                            )
                        except Exception as exc:  # noqa: BLE001
                            ret = {"error": f"{type(exc).__name__}: {exc}"}
                    rows.append(
                        {
                            "config": cfg_name,
                            "variant": var_name,
                            "seed": seed,
                            "alpha": variant.ca_alpha if variant.ca_rule else 0.0,
                            "status": cell.status,
                            "loss_final": cell.loss_final(),
                            "retention": ret,
                        }
                    )
                    prog.advance(task)

    # aggregate per (config, variant): mean cosine + rel drift + final loss
    def _mean(vals: list[float | None]) -> float | None:
        ok = [v for v in vals if v is not None and math.isfinite(v)]
        return (sum(ok) / len(ok)) if ok else None

    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for cfg_name in args.configs:
        for var_name in RETENTION_VARIANTS:
            grp = [r for r in rows if r["config"] == cfg_name and r["variant"] == var_name]
            agg[(cfg_name, var_name)] = {
                "alpha": INIT_VARIANTS[var_name].ca_alpha if INIT_VARIANTS[var_name].ca_rule else 0.0,
                "cosine_mean": _mean([r["retention"].get("cosine_mean") for r in grp]),
                "rel_drift_mean": _mean([r["retention"].get("rel_drift_mean") for r in grp]),
                "loss_final": _mean([r["loss_final"] for r in grp]),
                "n_ok": sum(1 for r in grp if r["status"] == "ok"),
                "n_total": len(grp),
            }

    _write_json(
        run_dir / "retention_summary.json",
        {
            "schema_version": "mgr.ca_init.retention.v1",
            "run_id": run_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": shlex.join(["uv", "run", "python", "scripts/ca_init_bench.py"] + sys.argv[1:]),
            "settings": {
                "max_steps": args.max_steps,
                "seeds": list(args.seeds),
                "configs": args.configs,
                "variants": RETENTION_VARIANTS,
                "learning_rate": args.learning_rate,
                "device": args.device,
            },
            "rows": rows,
        },
    )

    # report
    lines = [
        f"# CA-init structure retention — `{run_id}`\n",
        "Bead model_guided_research-827. After training, how much of the init's weight "
        "structure survives? Cosine similarity between the (deterministically reconstructed) "
        "init weights and the trained checkpoint, over the CA-initialized tensors "
        "(c_q/c_k/c_v/c_fc + wte). Higher cosine = more structure retained.\n",
        f"- Device `{args.device}` · steps `{args.max_steps}` · lr `{args.learning_rate}` · "
        f"seeds `{list(args.seeds)}`\n",
        "Bias axis: `ca_rule30` (alpha=1.0 pure CA) → `ca_mix0.5` → `ca_mix0.25` "
        "(more bias-like) → `standard` (random-init baseline).\n",
        "| config | variant | alpha | ok | retention cosine | rel L2 drift | final loss |",
        "|---|---|---|---|---|---|---|",
    ]

    def f(x: float | None) -> str:
        return f"{x:.4f}" if isinstance(x, (int, float)) else "—"

    for cfg_name in args.configs:
        for var_name in RETENTION_VARIANTS:
            m = agg[(cfg_name, var_name)]
            lines.append(
                f"| {cfg_name} | {var_name} | {m['alpha']:.2f} | {m['n_ok']}/{m['n_total']} | "
                f"{f(m['cosine_mean'])} | {f(m['rel_drift_mean'])} | {f(m['loss_final'])} |"
            )
    lines.append("")
    lines.append("## Reading it\n")
    lines.append(
        "- **Cosine → 1.0**: the trained weights still point the same way as init → CA "
        "structure is retained; SGD only nudged magnitudes."
    )
    lines.append(
        "- **Cosine ≪ 1.0**: training reoriented the weights → the init structure washed "
        "out (CA-init then acts as a fancy random seed, not a lasting prior)."
    )
    lines.append(
        "- Compare across the alpha axis: if lower-alpha (more bias-like) variants retain "
        "MORE structure at equal/again-better loss, the bias framing helps; if not, pure CA "
        "is as good as any blend.\n"
    )
    lines.append(
        "The freeze-channel arm (freeze CA channels for N warmup steps) is left as a "
        "follow-up: it needs a `train.py --freeze-init-steps` feature and is out of scope "
        "for a no-default-change experiment.\n"
    )
    lines.append(
        "## Reproduction\n```\n"
        + shlex.join(["uv", "run", "python", "scripts/ca_init_bench.py"] + sys.argv[1:])
        + "\n```\n"
    )
    _write_text(run_dir / "retention.md", "\n".join(lines))

    table = Table(title=f"CA-init structure retention — {run_id}", header_style="bold")
    for col in ("config", "variant", "alpha", "ok", "cosine", "rel drift", "final loss"):
        table.add_column(col, justify="left" if col in ("config", "variant") else "right")
    for cfg_name in args.configs:
        for var_name in RETENTION_VARIANTS:
            m = agg[(cfg_name, var_name)]
            table.add_row(
                cfg_name,
                var_name,
                f"{m['alpha']:.2f}",
                f"{m['n_ok']}/{m['n_total']}",
                f(m["cosine_mean"]),
                f(m["rel_drift_mean"]),
                f(m["loss_final"]),
            )
    console.print(table)
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    console.print(f"[bold green]done[/bold green] {n_ok}/{len(rows)} cells ok → {run_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="CA-init experiments: m32 early-phase bench + 827 retention.")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--optimizer-type", type=str, default="adamw")
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--mode",
        choices=["bench", "retention"],
        default="bench",
        help="bench: m32 early-phase loss/grad/activation sweep. "
        "retention: 827 structure-retention (init vs trained weight cosine).",
    )
    parser.add_argument("--configs", type=str, nargs="+", default=list(ARCH_CONFIGS), choices=list(ARCH_CONFIGS))
    parser.add_argument("--variants", type=str, nargs="+", default=BENCH_VARIANTS, choices=list(INIT_VARIANTS))
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--auto-download-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-parquet-files", type=int, default=2)
    parser.add_argument(
        "--skip-train", action="store_true", help="Only run the in-process activation probe (no training subprocess)."
    )
    args = parser.parse_args()

    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1")

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    artifacts_dir = Path(args.artifacts_dir)
    run_dir = artifacts_dir / "ca_init" / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run dir already exists and is non-empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "retention":
        return _run_retention(args, run_dir, artifacts_dir)

    console.rule(f"[bold cyan]CA-init benchmark[/bold cyan] · {run_id}")
    console.print(
        f"configs={args.configs}  variants={args.variants}  seeds={args.seeds}  "
        f"steps={args.max_steps}  device={args.device}"
    )

    cells: list[CellResult] = []
    total = len(args.configs) * len(args.variants) * len(args.seeds)
    with Progress(
        TextColumn("[bold cyan]ca-init[/bold cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("cells", total=total)
        for cfg_name in args.configs:
            arch = ARCH_CONFIGS[cfg_name]
            for var_name in args.variants:
                variant = INIT_VARIANTS[var_name]
                for seed in args.seeds:
                    # init-time activation probe (always; cheap)
                    try:
                        act = _probe_init_activations(
                            arch, variant, seed, device=args.device, vocab_size=args.vocab_size
                        )
                    except Exception as exc:  # noqa: BLE001 - record, never crash the sweep
                        act = {"error": f"{type(exc).__name__}: {exc}"}

                    if args.skip_train:
                        cell = CellResult(
                            config=cfg_name,
                            variant=var_name,
                            seed=seed,
                            status="probe_only",
                            command="(skip-train)",
                        )
                    else:
                        cell = _run_train_cell(
                            arch=arch,
                            variant=variant,
                            seed=seed,
                            run_dir=run_dir,
                            artifacts_dir=artifacts_dir,
                            args=args,
                        )
                    cell.activation = act
                    cells.append(cell)
                    prog.advance(task)

    summary = _build_summary(run_id, args, cells)
    _write_json(run_dir / "summary.json", summary)
    report = _render_report(run_id, args, cells)
    _write_text(run_dir / "report.md", report)

    _print_table(cells)
    n_ok = sum(1 for c in cells if c.status == "ok")
    console.print(f"[bold green]done[/bold green] {n_ok}/{len(cells)} cells ok → {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
