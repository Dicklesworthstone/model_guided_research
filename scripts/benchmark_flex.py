from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table

from nanochat.gpt import GPT, GPTConfig
from nanochat.report import get_git_info, get_gpu_info, get_system_info

console = Console()


def _flex_available() -> bool:
    try:
        from torch.nn.attention.flex_attention import flex_attention  # noqa: F401
    except Exception:
        return False
    return True


def _autocast_ctx(device: torch.device, dtype: torch.dtype):
    if dtype == torch.float32:
        return nullcontext()
    if device.type == "cpu" and dtype is torch.float16:
        raise ValueError("float16 autocast is not supported on CPU")
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def _ms_per_iter_cuda(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _ms_per_iter_cpu(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return float((t1 - t0) * 1000.0 / iters)


def _build_config(args: argparse.Namespace, *, use_flex_attention: bool) -> GPTConfig:
    return GPTConfig(
        sequence_len=args.sequence_len,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        attention_type="standard",
        use_flex_attention=use_flex_attention,
    )


def _default_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _write_artifacts(run_dir: Path, *, summary: dict[str, Any], report_md: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "run.md").write_text(report_md, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark FlexAttention vs SDPA (standard GPT).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-len", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-kv-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=512)

    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use torch.compile (recommended for FlexAttention fused kernels).",
    )
    parser.add_argument(
        "--write-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write summary.json + run.md under artifacts/perf/flex_attention/<run_id>/",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Base directory for run artifacts.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (directory name). Defaults to YYYYMMDD_HHMMSS.",
    )
    args = parser.parse_args()

    if not _flex_available():
        console.print("[yellow]FlexAttention is unavailable in this torch build; skipping.[/yellow]")
        return 0

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]CUDA requested but not available; skipping.[/yellow]")
        return 0

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    if device.type == "cpu" and dtype is torch.float16:
        console.print("[yellow]CPU does not support float16 autocast; use float32/bfloat16.[/yellow]")
        return 2

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    cfg_sdpa = _build_config(args, use_flex_attention=False)
    cfg_flex = _build_config(args, use_flex_attention=True)

    model_sdpa = GPT(cfg_sdpa).to(device).train(False)
    model_flex = GPT(cfg_flex).to(device).train(False)
    model_flex.load_state_dict(model_sdpa.state_dict(), strict=True)

    if args.compile:
        model_sdpa = torch.compile(model_sdpa)
        model_flex = torch.compile(model_flex)

    x = torch.randint(0, cfg_sdpa.vocab_size, (args.batch_size, args.sequence_len), device=device, dtype=torch.long)

    def _bench(model: GPT) -> tuple[float, float, float]:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        def run_once():
            with torch.inference_mode(), _autocast_ctx(device, dtype):
                _ = model(x)

        if device.type == "cuda":
            ms = _ms_per_iter_cuda(run_once, warmup=args.warmup, iters=args.iters)
            peak = float(torch.cuda.max_memory_allocated() / (1024**2))
        else:
            ms = _ms_per_iter_cpu(run_once, warmup=args.warmup, iters=args.iters)
            peak = 0.0
        toks_per_s = (args.batch_size * args.sequence_len) / (ms / 1000.0)
        return float(ms), float(toks_per_s), peak

    ms_sdpa, tps_sdpa, peak_sdpa = _bench(model_sdpa)
    ms_flex, tps_flex, peak_flex = _bench(model_flex)

    table = Table(title="FlexAttention benchmark (standard GPT forward)")
    table.add_column("mode", style="bold")
    table.add_column("ms/iter", justify="right")
    table.add_column("tokens/s", justify="right")
    if device.type == "cuda":
        table.add_column("peak MB", justify="right")

    if device.type == "cuda":
        table.add_row("sdpa", f"{ms_sdpa:.2f}", f"{tps_sdpa:,.0f}", f"{peak_sdpa:.1f}")
        table.add_row("flex", f"{ms_flex:.2f}", f"{tps_flex:,.0f}", f"{peak_flex:.1f}")
    else:
        table.add_row("sdpa", f"{ms_sdpa:.2f}", f"{tps_sdpa:,.0f}")
        table.add_row("flex", f"{ms_flex:.2f}", f"{tps_flex:,.0f}")

    speedup = (ms_sdpa / ms_flex) if ms_flex > 0 else float("nan")
    console.print(table)
    console.print(f"Speedup (sdpa/flex): {speedup:.3f}×")

    if args.write_artifacts:
        run_id = args.run_id or _default_run_id()
        run_dir = Path(args.artifacts_dir) / "perf" / "flex_attention" / run_id

        summary: dict[str, Any] = {
            "git": get_git_info(),
            "system": get_system_info(),
            "gpu": get_gpu_info(),
            "python": {"executable": sys.executable, "version": sys.version},
            "command": f"uv run python {shlex.join(sys.argv)}",
            "argv": sys.argv,
            "args": vars(args),
            "dtype": args.dtype,
            "device": str(device),
            "config": {"sdpa": asdict(cfg_sdpa), "flex": asdict(cfg_flex)},
            "results": {
                "ms_per_iter": {"sdpa": ms_sdpa, "flex": ms_flex},
                "tokens_per_s": {"sdpa": tps_sdpa, "flex": tps_flex},
                "peak_mem_mb": {"sdpa": peak_sdpa, "flex": peak_flex},
                "speedup_sdpa_over_flex": speedup,
            },
        }

        report_md = f"""# FlexAttention perf benchmark (SDPA vs Flex)

- Run ID: `{run_id}`
- Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}

## Command

```bash
uv run python {shlex.join(sys.argv)}
```

## Config

- device: `{device}`
- dtype: `{args.dtype}`
- batch_size: {args.batch_size}
- sequence_len: {args.sequence_len}
- n_layer/n_head/n_kv_head/n_embd: {args.n_layer}/{args.n_head}/{args.n_kv_head}/{args.n_embd}
- torch.compile: {bool(args.compile)}

## Results

- ms/iter: sdpa={ms_sdpa:.2f}  flex={ms_flex:.2f}
- tokens/s: sdpa={tps_sdpa:,.0f}  flex={tps_flex:,.0f}
- speedup (sdpa/flex): {speedup:.3f}×
"""
        if device.type == "cuda":
            report_md += f"- peak MB: sdpa={peak_sdpa:.1f}  flex={peak_flex:.1f}\n"

        report_md += "\nSee `summary.json` for full details.\n"

        _write_artifacts(run_dir, summary=summary, report_md=report_md)
        console.print(f"[dim]Wrote artifacts → {run_dir}[/dim]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
