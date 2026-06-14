"""Profiling instrumentation for nanochat (bead b1l).

Optional, default-off hooks for measuring where time and memory go in the
attention / optimizer hot paths:

* ``nvtx_range(name)`` -- an NVTX marker for nsys / Nsight Systems timelines.
  A true no-op without CUDA, so it is free to sprinkle through the trainer.
* ``torch_profiler(config)`` -- a context manager yielding a configured
  ``torch.profiler.profile`` (CPU + CUDA activities, shapes, memory), or
  ``None`` when disabled — so a disabled profiler costs nothing.
* ``summarize_profile`` / ``render_profile_table`` -- turn a finished profile
  into a device-agnostic kernel/memory breakdown (a rich table + JSON).

These are the reusable hooks the trainer wraps around its hot paths; this module
is also runnable standalone as a microbench that captures a sample trace and the
kernel/memory breakdown for a mechanism (and FlexAttention on/off for standard)::

    python -m nanochat.profiling bench --attention standard --out artifacts/profiles/std
    python -m nanochat.profiling bench --attention tropical --steps 8
    python -m nanochat.profiling bench --attention standard --compare-flex   # flex on vs off

Each run writes ``summary.json`` + a Chrome trace (``trace.json``, load in
chrome://tracing or Perfetto) and prints a rich table. See ``docs/profiling.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_PROFILE_ROOT = Path("artifacts/profiles")


def _console() -> Any:
    from rich.console import Console

    return Console()


# --------------------------------------------------------------------------- #
# Hooks (the reusable instrumentation the trainer wraps hot paths with)        #
# --------------------------------------------------------------------------- #


@contextmanager
def nvtx_range(name: str):
    """Mark an NVTX range for nsys/Nsight timelines; a no-op without CUDA.

    The CUDA check is cheap and evaluated once on entry, so leaving these around
    attention/optimizer calls is free on CPU and when profiling is off.
    """
    import torch

    active = torch.cuda.is_available()
    if active:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if active:
            torch.cuda.nvtx.range_pop()


@dataclass
class ProfileConfig:
    """Toggle + knobs for a profiling session (default disabled = no overhead)."""

    enabled: bool = False
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    row_limit: int = 15
    trace_dir: Path | None = None
    extra_activities: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> ProfileConfig:
        """Build from NANOCHAT_PROFILE* env vars (parity with the CLI flags).

        NANOCHAT_PROFILE=1 enables; NANOCHAT_PROFILE_DIR sets the trace dir;
        NANOCHAT_PROFILE_ROWS / _MEMORY / _STACK tune the rest.
        """
        import os

        e = env if env is not None else os.environ
        enabled = str(e.get("NANOCHAT_PROFILE", "")).lower() in {"1", "true", "yes", "on"}
        trace_dir = e.get("NANOCHAT_PROFILE_DIR")
        return cls(
            enabled=enabled,
            profile_memory=str(e.get("NANOCHAT_PROFILE_MEMORY", "1")).lower() not in {"0", "false", "no", "off"},
            with_stack=str(e.get("NANOCHAT_PROFILE_STACK", "")).lower() in {"1", "true", "yes", "on"},
            row_limit=int(e.get("NANOCHAT_PROFILE_ROWS", "15")),
            trace_dir=Path(trace_dir) if trace_dir else None,
        )


@contextmanager
def torch_profiler(config: ProfileConfig):
    """Yield a configured ``torch.profiler.profile`` or ``None`` when disabled.

    When disabled the body runs with zero added work (no profiler constructed),
    which is the negligible-overhead-when-off requirement.
    """
    if not config.enabled:
        yield None
        return
    import torch
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(
        activities=activities,
        record_shapes=config.record_shapes,
        profile_memory=config.profile_memory,
        with_stack=config.with_stack,
    ) as prof:
        yield prof


# --------------------------------------------------------------------------- #
# Summarization                                                                #
# --------------------------------------------------------------------------- #


def _device_time_us(event: Any) -> float:
    # torch >= 2.x renamed self_cuda_time_total -> self_device_time_total.
    val = getattr(event, "self_device_time_total", None)
    if val is None:
        val = getattr(event, "self_cuda_time_total", 0)
    return float(val or 0.0)


def _device_mem_bytes(event: Any) -> int:
    val = getattr(event, "device_memory_usage", None)
    if val is None:
        val = getattr(event, "cuda_memory_usage", 0)
    return int(val or 0)


def summarize_profile(prof: Any, *, row_limit: int = 15) -> dict[str, Any]:
    """Top ops by self-time + a memory/time roll-up; device-agnostic fields."""
    import torch

    cuda = torch.cuda.is_available()
    ka = list(prof.key_averages())
    sort_key = _device_time_us if cuda else (lambda e: float(e.self_cpu_time_total))
    top = sorted(ka, key=sort_key, reverse=True)[: int(row_limit)]
    ops: list[dict[str, Any]] = []
    for e in top:
        ops.append({
            "op": str(e.key),
            "count": int(e.count),
            "self_cpu_us": float(e.self_cpu_time_total),
            "cpu_total_us": float(e.cpu_time_total),
            "self_device_us": _device_time_us(e),
            "cpu_mem_bytes": int(getattr(e, "cpu_memory_usage", 0) or 0),
            "device_mem_bytes": _device_mem_bytes(e),
        })
    return {
        "device": "cuda" if cuda else "cpu",
        "totals": {
            "self_cpu_us": float(sum(float(e.self_cpu_time_total) for e in ka)),
            "self_device_us": float(sum(_device_time_us(e) for e in ka)),
            "n_ops": len(ka),
        },
        "ops": ops,
    }


def _fmt_us(us: float) -> str:
    if us >= 1e6:
        return f"{us / 1e6:.2f} s"
    if us >= 1e3:
        return f"{us / 1e3:.2f} ms"
    return f"{us:.1f} µs"


def _fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(x) < 1024.0:
            return f"{x:.1f} {unit}"
        x /= 1024.0
    return f"{x:.1f} TB"


def render_profile_table(summary: dict[str, Any], *, title: str, console: Any = None) -> None:
    from rich.table import Table

    console = console or _console()
    dev = summary["device"]
    time_col = "self CUDA" if dev == "cuda" else "self CPU"
    mem_col = "CUDA mem" if dev == "cuda" else "CPU mem"
    table = Table(title=f"{title}  ·  device={dev}", header_style="bold cyan")
    table.add_column("op", overflow="fold")
    table.add_column("#", justify="right", style="dim")
    table.add_column(time_col, justify="right")
    table.add_column("CPU total", justify="right", style="dim")
    table.add_column(mem_col, justify="right")
    for op in summary["ops"]:
        dev_t = op["self_device_us"] if dev == "cuda" else op["self_cpu_us"]
        mem = op["device_mem_bytes"] if dev == "cuda" else op["cpu_mem_bytes"]
        table.add_row(op["op"], str(op["count"]), _fmt_us(dev_t), _fmt_us(op["cpu_total_us"]), _fmt_bytes(mem))
    console.print(table)
    tot = summary["totals"]
    key_t = tot["self_device_us"] if dev == "cuda" else tot["self_cpu_us"]
    console.print(f"[dim]total self {('CUDA' if dev == 'cuda' else 'CPU')} time across {tot['n_ops']} ops: {_fmt_us(key_t)}[/dim]")


# --------------------------------------------------------------------------- #
# Standalone microbench                                                        #
# --------------------------------------------------------------------------- #


def profile_model(
    attention_type: str,
    *,
    device: str = "cpu",
    steps: int = 5,
    warmup: int = 2,
    backward: bool = True,
    use_flex_attention: bool = False,
    seed: int = 0,
    n_layer: int = 4,
    n_head: int | None = None,
    n_kv_head: int | None = None,
    n_embd: int | None = None,
    seq_len: int = 128,
    batch_size: int = 8,
    vocab_size: int = 256,
    trace_dir: Path | None = None,
    row_limit: int = 15,
) -> dict[str, Any]:
    """Build a probe model and profile ``steps`` forward(+backward) iterations.

    Returns a summary dict (device, top ops, totals, peak memory, meta) and, if
    ``trace_dir`` is set, exports a Chrome trace there.
    """
    import torch

    from nanochat.viz import build_probe_model, sample_batch

    extra: dict[str, Any] = {}
    if use_flex_attention:
        extra["use_flex_attention"] = True
    model, meta = build_probe_model(
        attention_type, device=device, seed=seed, n_layer=n_layer, n_head=n_head,
        n_kv_head=n_kv_head, n_embd=n_embd, sequence_len=seq_len, vocab_size=vocab_size,
        extra_config=extra,
    )
    if backward:
        model.train()
    idx, _labels = sample_batch(
        text=None, batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size, seed=seed, device=device,
    )
    targets = idx.clone()

    def _step() -> None:
        if backward:
            model.zero_grad(set_to_none=True)
            loss = model(idx, targets=targets)
            loss.backward()
        else:
            with torch.no_grad():
                _ = model(idx)

    for _ in range(max(0, int(warmup))):  # warm caches / lazy init out of the trace
        _step()
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)

    cfg = ProfileConfig(enabled=True, trace_dir=trace_dir, row_limit=row_limit)
    with torch_profiler(cfg) as prof, nvtx_range(f"{attention_type}-microbench"):
        for i in range(max(1, int(steps))):
            with nvtx_range(f"step{i}"):
                _step()
    if dev.type == "cuda":
        torch.cuda.synchronize()

    summary = summarize_profile(prof, row_limit=row_limit)
    peak_mem = None
    if dev.type == "cuda":
        peak_mem = int(torch.cuda.max_memory_allocated(dev))
    trace_path = None
    if trace_dir is not None:
        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "trace.json"
        prof.export_chrome_trace(str(trace_path))
    summary["meta"] = {
        "attention_type": attention_type,
        "use_flex_attention": bool(use_flex_attention),
        "backward": bool(backward),
        "steps": int(steps),
        "warmup": int(warmup),
        "config": meta["config"],
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "peak_device_mem_bytes": peak_mem,
        "trace": (str(trace_path) if trace_path else None),
    }
    return summary


def run_bench(args: argparse.Namespace) -> int:
    console = _console()
    out_dir = Path(args.out) if args.out else (DEFAULT_PROFILE_ROOT / f"{args.attention}")
    out_dir.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, bool]] = [(args.attention, False)]
    if args.compare_flex:
        if args.attention != "standard":
            console.print("[yellow]--compare-flex only applies to --attention standard; ignoring.[/yellow]")
        else:
            variants.append((args.attention, True))

    results: list[dict[str, Any]] = []
    for attn, flex in variants:
        label = f"{attn}{' +flex' if flex else ''}"
        sub_dir = out_dir / ("flex_on" if flex else "flex_off") if args.compare_flex else out_dir
        try:
            summary = profile_model(
                attn, device=args.device, steps=args.steps, warmup=args.warmup,
                backward=(not args.forward_only), use_flex_attention=flex, seed=args.seed,
                n_layer=args.n_layer, n_head=args.n_head, n_kv_head=args.n_kv_head,
                n_embd=args.n_embd, seq_len=args.seq_len, batch_size=args.batch_size,
                vocab_size=args.vocab_size, trace_dir=sub_dir, row_limit=args.row_limit,
            )
        except ImportError as exc:  # FlexAttention unavailable: skip cleanly
            console.print(f"[yellow]skipped {label}: {exc}[/yellow]")
            continue
        console.rule(f"[bold cyan]profile · {label}")
        render_profile_table(summary, title=f"top ops · {label}", console=console)
        if summary["meta"]["peak_device_mem_bytes"] is not None:
            console.print(f"[dim]peak device memory: {_fmt_bytes(summary['meta']['peak_device_mem_bytes'])}[/dim]")
        (sub_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        console.print(f"[dim]trace: {summary['meta']['trace']}[/dim]")
        results.append(summary)

    if len(results) == 2:  # flex on/off comparison roll-up
        _render_flex_compare(results, console)
    console.print(f"\n[bold]wrote[/bold] {len(results)} profile(s) -> [cyan]{out_dir}/[/cyan]")
    return 0


def _render_flex_compare(results: list[dict[str, Any]], console: Any) -> None:
    from rich.table import Table

    off, on = results[0], results[1]
    dev = off["device"]
    key = "self_device_us" if dev == "cuda" else "self_cpu_us"
    t_off, t_on = off["totals"][key], on["totals"][key]
    table = Table(title="FlexAttention on vs off", header_style="bold magenta")
    table.add_column("variant")
    table.add_column(f"self {dev.upper()} time", justify="right")
    table.add_column("speedup", justify="right")
    table.add_row("flex off", _fmt_us(t_off), "1.00×")
    table.add_row("flex on", _fmt_us(t_on), f"{(t_off / t_on):.2f}×" if t_on else "—")
    console.print(table)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m nanochat.profiling", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("bench", help="profile a mechanism's forward(+backward) hot path (b1l)")
    pb.add_argument("--attention", default="standard")
    pb.add_argument("--compare-flex", action="store_true", help="standard only: run FlexAttention on vs off")
    pb.add_argument("--forward-only", action="store_true", help="profile forward only (default: fwd+bwd)")
    pb.add_argument("--device", default="cpu")
    pb.add_argument("--steps", type=int, default=5)
    pb.add_argument("--warmup", type=int, default=2)
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("--n-layer", type=int, default=4)
    pb.add_argument("--n-head", type=int, default=None)
    pb.add_argument("--n-kv-head", type=int, default=None)
    pb.add_argument("--n-embd", type=int, default=None)
    pb.add_argument("--seq-len", type=int, default=128)
    pb.add_argument("--batch-size", type=int, default=8)
    pb.add_argument("--vocab-size", type=int, default=256)
    pb.add_argument("--row-limit", type=int, default=15)
    pb.add_argument("--out", default=None, help="output dir (default artifacts/profiles/<attention>)")
    pb.set_defaults(func=run_bench)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
