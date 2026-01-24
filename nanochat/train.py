"""
Minimal PyTorch training loop for Nanochat.

Intentionally lightweight: small model, short run, and a streaming parquet-backed dataloader.
"""

import argparse
import json
import math
import os
import shlex
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from rich import box
from rich.console import Console
from rich.table import Table
from torch.nn.parallel import DistributedDataParallel as DDP

from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, print0
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.dataset import ensure_min_parquet_files, list_parquet_files
from nanochat.gpt import GPT, GPTConfig
from nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from nanochat.ordinal_scheduler import OrdinalLRScheduler
from nanochat.report import get_git_info, get_gpu_info, get_system_info
from nanochat.synaptic import SynapticConfig

console = Console()

_SUPPORTED_MODEL_TYPES = ("gpt", "synaptic")
_SUPPORTED_OPTIMIZER_TYPES = ("adamw", "muon", "hoss")
_SUPPORTED_SCHEDULER_TYPES = ("none", "ordinal")
_SUPPORTED_ATTENTION_TYPES = (
    "standard",
    "tropical",
    "ultrametric",
    "simplicial",
    "quaternion",
    "braid",
    "fractal",
    "octonion",
    "surreal",
    "reversible",
    "gauge",
)


def _select_env_vars() -> dict[str, str]:
    prefixes = ("CUDA", "NCCL", "TORCH", "PYTORCH", "NANOCHAT")
    selected = {k: v for k, v in os.environ.items() if k.startswith(prefixes)}
    return dict(sorted(selected.items()))


def _write_artifacts(run_dir: Path, *, summary: dict[str, Any], report_md: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "run.md").write_text(report_md, encoding="utf-8")


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

    attention_type = str(getattr(args, "attention_type", "standard"))
    if attention_type not in _SUPPORTED_ATTENTION_TYPES:
        errors.append(f"--attention-type must be one of: {', '.join(_SUPPORTED_ATTENTION_TYPES)}")

    if bool(getattr(args, "use_flex_attention", False)) and not hasattr(torch.nn.attention, "flex_attention"):
        errors.append("--use-flex-attention requires torch>=2.5 (missing torch.nn.attention.flex_attention).")

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
        if bool(getattr(args, "use_flex_attention", False)):
            warnings.append("--use-flex-attention is ignored for --model-type synaptic.")
    else:
        if syn_cfg_path is not None:
            warnings.append("--synaptic-config is only used with --model-type synaptic; ignoring.")

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

    if model_type == "gpt":
        config = GPTConfig()
        config.n_layer = args.n_layer
        config.n_head = args.n_head
        config.n_kv_head = args.n_kv_head
        config.n_embd = args.n_embd
        config.sequence_len = args.sequence_len
        config.optimizer_type = args.optimizer_type
        config.attention_type = args.attention_type
        config.use_flex_attention = bool(getattr(args, "use_flex_attention", False))
        std_entropy = getattr(args, "standard_record_attn_entropy", None)
        if std_entropy is not None:
            config.standard_record_attn_entropy = bool(std_entropy)
        if config.attention_type == "ultrametric":
            config.ultrametric_mode = str(getattr(args, "ultrametric_mode", config.ultrametric_mode))
            ultra_hard = getattr(args, "ultrametric_hard_digits", None)
            if ultra_hard is not None:
                config.ultrametric_hard_digits = bool(ultra_hard)
        if config.attention_type == "braid":
            config.braid_mode = str(getattr(args, "braid_mode", config.braid_mode))
            config.braid_tau = float(getattr(args, "braid_tau", config.braid_tau))
            config.braid_crossing_law = str(getattr(args, "braid_crossing_law", config.braid_crossing_law))
            braid_record = getattr(args, "braid_record_schedule", None)
            if braid_record is not None:
                config.braid_record_schedule = bool(braid_record)
            braid_verify = getattr(args, "braid_verify", None)
            if braid_verify is not None:
                config.braid_verify = bool(braid_verify)
        if config.use_flex_attention and config.attention_type != "standard":
            if ddp_rank == 0:
                print0("[flex] --use-flex-attention only applies to --attention-type standard; disabling.")
            config.use_flex_attention = False
        if config.use_flex_attention and device.type != "cuda":
            if ddp_rank == 0:
                print0(f"[flex] FlexAttention requires CUDA; disabling (device={device.type}).")
            config.use_flex_attention = False
        if config.standard_record_attn_entropy and config.attention_type != "standard":
            if ddp_rank == 0:
                print0("[entropy] --standard-record-attn-entropy only applies to --attention-type standard; disabling.")
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

        if config.attention_type == "reversible":
            if config.n_head % 2 != 0:
                raise ValueError("reversible attention requires n_head to be even")
            desired_n_kv_head = config.n_head // 2
            if config.n_kv_head != desired_n_kv_head and ddp_rank == 0:
                print0(
                    f"[reversible] Overriding n_kv_head from {config.n_kv_head} to {desired_n_kv_head} "
                    "to satisfy reversible KV-cache constraints."
                )
            config.n_kv_head = desired_n_kv_head

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
        if config.attention_type != "tropical":
            if (
                tropical_gauge_fix is not None
                or tropical_score_center is not None
                or tropical_record_margins is not None
            ) and ddp_rank == 0:
                print0("[tropical] Ignoring tropical flags because --attention-type is not tropical.")
            if tropical_log_margins and ddp_rank == 0:
                print0("[tropical] Ignoring --tropical-log-margins because --attention-type is not tropical.")
        else:
            if tropical_gauge_fix is not None:
                config.tropical_gauge_fix = bool(tropical_gauge_fix)
            if tropical_score_center is not None:
                config.tropical_score_center = bool(tropical_score_center)
            if tropical_record_margins is not None:
                config.tropical_record_margins = bool(tropical_record_margins)
            if tropical_log_margins:
                config.tropical_record_margins = True
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
        config.vocab_size = int(getattr(args, "vocab_size", GPTConfig().vocab_size))
        config.n_layer = args.n_layer
        config.n_head = args.n_head
        config.n_kv_head = args.n_kv_head
        config.n_embd = args.n_embd
        config.sequence_len = args.sequence_len
        config.syn_cfg = syn_cfg

    # Dataset (optional auto-download)
    required_parquet_files = max(2, int(args.min_parquet_files))
    if args.auto_download_data:
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

    parquet_files = list_parquet_files()
    if len(parquet_files) < required_parquet_files:
        raise RuntimeError(
            f"No usable dataset found (need >={required_parquet_files} parquet shards; 2 is the minimum: 1 train + 1 val). "
            "Either download shards into the nanochat cache directory or run with --auto-download-data."
        )

    # Model
    raw_model = GPT(config) if model_type == "gpt" else GPTSynaptic(config)
    raw_model.to(device)
    raw_model.init_weights()
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
        model = torch.compile(model, **compile_kwargs)
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

    # Scheduler
    schedulers: list[OrdinalLRScheduler] = []
    if args.scheduler_type == "ordinal":
        # We attach an ordinal scheduler to each optimizer
        # Note: Ordinal scheduler updates LR based on loss.
        for opt in optimizers:
            schedulers.append(OrdinalLRScheduler(opt, eta_init=args.learning_rate))

    # Dataloader
    loader = tokenizing_distributed_data_loader(
        B=args.batch_size,
        T=config.sequence_len,
        split="train",
        device=device.type,
    )

    def autocast_ctx():
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @torch.no_grad()
    def evaluate_validation(val_loader_iter, num_batches: int) -> float:
        """Evaluate cross-entropy loss on validation data."""
        model.eval()
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
        model.train()
        return total_loss / count if count > 0 else float("nan")

    if ddp_rank == 0:
        console.print(
            f"[bold green]Starting training[/bold green] on [bold]{device}[/bold] (world_size={ddp_world_size})"
        )
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

    if ddp_rank == 0:
        console.print(
            f"[bold cyan]budget[/bold cyan] steps={max_steps} warmup={args.warmup_steps} "
            f"tokens/step={tokens_per_step:,} flops/token={flops_per_token:,} "
            f"flops/step={flops_per_step:,}"
        )

    is_hoss = args.optimizer_type == "hoss"

    losses: list[float] = []
    val_losses: list[tuple[int, float]] = []  # (step, val_loss) pairs
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
        )
        if ddp_rank == 0:
            console.print(f"[bold cyan]validation[/bold cyan] interval={val_interval} batches={val_batches}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    last_log_time = time.perf_counter()
    meas_start_time: float | None = None

    try:
        for step, (inputs, targets) in enumerate(loader):
            if step >= max_steps:
                break

            if step == args.warmup_steps:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                meas_start_time = time.perf_counter()

            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            step_t0 = time.perf_counter()

            def closure(inputs=inputs, targets=targets):
                with autocast_ctx():
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
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=grad_clip_norm)
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
                    and getattr(config, "attention_type", None) == "tropical"
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

                console.print(msg)
                last_log_time = step_t1
                last_log_step = step

            # Periodic validation evaluation
            if val_loader is not None and val_interval > 0 and (step + 1) % val_interval == 0:
                val_loss = evaluate_validation(val_loader, val_batches)
                val_losses.append((step, val_loss))
                if ddp_rank == 0:
                    console.print(
                        f"[bold magenta]val[/bold magenta] step={step}  "
                        f"[dim]val_ce[/dim] {val_loss:.4f}  "
                        f"[dim]train_ce[/dim] {loss_item:.4f}"
                    )
    finally:
        compute_cleanup()

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

    resolved_run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    artifacts_kind = str(getattr(args, "artifacts_kind", "baseline"))
    artifacts_topic = str(getattr(args, "artifacts_topic", "nanochat"))
    run_dir = Path(args.artifacts_dir) / artifacts_kind / artifacts_topic / resolved_run_id

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
    # Compute final train/val CE statistics
    final_train_ce = losses[-1] if losses else float("nan")
    final_val_ce = val_losses[-1][1] if val_losses else None

    results: dict[str, Any] = {
        "losses": losses,
        "train_ce_final": final_train_ce,
        "val_losses": val_losses,
        "val_ce_final": final_val_ce,
        "step_times_s": step_times_s,
        "measured_steps": measured_steps,
        "measured_tokens": measured_tokens,
        "measured_time_s": measured_time_s,
        "tokens_per_second": tokens_per_second,
        "tflops_per_second_est": est_tflops,
        "peak_memory_allocated_gb": peak_mem_gb,
    }
    if model_type == "gpt" and getattr(config, "attention_type", None) == "tropical":
        tropical = _collect_tropical_margin_stats(raw_model)
        if tropical is not None:
            results["tropical_margins"] = tropical
    attn_entropy = _collect_attn_entropy_stats(raw_model)
    if attn_entropy is not None:
        results["attention_entropy"] = attn_entropy
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
            "synaptic_config": (asdict(config.syn_cfg) if model_type == "synaptic" else None),
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
        "config": asdict(config),
        "dataset": {
            "parquet_files_count": len(parquet_files),
            "parquet_files": parquet_files,
        },
        "budget": budget,
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
    report_table.add_row("Steps", str(max_steps))
    report_table.add_row("Warmup Steps", str(args.warmup_steps))
    report_table.add_row("check_numerics", str(check_numerics))
    report_table.add_row("detect_anomaly", str(detect_anomaly))
    report_table.add_row("torch.compile model", "enabled" if compiled_model else "disabled")
    if bool(getattr(config, "use_flex_attention", False)):
        report_table.add_row(
            "compile flex_attention",
            "enabled" if bool(getattr(config, "compile_flex_attention", False)) else "disabled",
        )
    if getattr(config, "ca_init_rule", None):
        report_table.add_row(
            "CA init",
            f"{config.ca_init_rule} (alpha={getattr(config, 'ca_init_alpha', None)}, seed={getattr(config, 'ca_init_seed', None)})",
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
    console.print(report_table)

    report_md = f"""# nanochat run (fixed FLOPs)

- Run ID: `{resolved_run_id}`
- Generated: {generated_at}
- Artifacts: `{artifacts_kind}/{artifacts_topic}/{resolved_run_id}`
- Commit: {commit_label}{dirty}

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

See `summary.json` for full details.
"""

    _write_artifacts(run_dir, summary=summary, report_md=report_md)
    console.print(f"[bold green]Wrote artifacts[/bold green] → {run_dir}")


if __name__ == "__main__":
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
        choices=["restricted", "ybe"],
        help="Braid attention only: restricted (fast, non-YBE) vs ybe (swap-output, YBE-valid; used for verification).",
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
        "--ultrametric-mode",
        type=str,
        default=os.environ.get("NANOCHAT_ULTRAMETRIC_MODE", "kernel"),
        choices=["kernel", "trie"],
        help="Ultrametric attention only: kernel (continuous) vs trie (packed-prefix lookup) decode path.",
    )
    parser.add_argument(
        "--ultrametric-hard-digits",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ultrametric attention only: quantize digits before computing LCP weights (default: off).",
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
        help="Vocabulary size (model embedding table size).",
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
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    args = parser.parse_args()

    train(args)
