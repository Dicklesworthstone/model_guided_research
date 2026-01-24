from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table

from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig
from nanochat.report import get_git_info, get_gpu_info, get_system_info

console = Console()


@dataclass(frozen=True)
class CheckResult:
    name: str
    max_abs: float
    mean_abs: float
    max_rel: float
    passed: bool
    note: str = ""


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


def _tensor_diffs(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    diff = (a - b).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    denom = b.abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    return max_abs, mean_abs, max_rel


def _check_close(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    note: str = "",
) -> CheckResult:
    a32 = a.float()
    b32 = b.float()
    max_abs, mean_abs, max_rel = _tensor_diffs(a32, b32)
    try:
        torch.testing.assert_close(a32, b32, rtol=rtol, atol=atol)
    except AssertionError:
        return CheckResult(name=name, max_abs=max_abs, mean_abs=mean_abs, max_rel=max_rel, passed=False, note=note)
    return CheckResult(name=name, max_abs=max_abs, mean_abs=mean_abs, max_rel=max_rel, passed=True, note=note)


def _build_config(args: argparse.Namespace) -> GPTConfig:
    return GPTConfig(
        sequence_len=args.sequence_len,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        attention_type="standard",
        use_flex_attention=bool(args.use_flex_attention),
        compile_flex_attention=bool(getattr(args, "compile", False)),
    )


def _default_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _write_artifacts(run_dir: Path, *, summary: dict[str, Any], report_md: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "run.md").write_text(report_md, encoding="utf-8")


def _make_model(cfg: GPTConfig, device: torch.device) -> GPT:
    model = GPT(cfg).to(device).train(False)
    return model


def _kv_cache_for(cfg: GPTConfig, *, batch_size: int, total_len: int) -> KVCache:
    return KVCache(
        batch_size=batch_size,
        num_heads=cfg.n_kv_head,
        seq_len=total_len,
        head_dim=cfg.n_embd // cfg.n_head,
        num_layers=cfg.n_layer,
    )


def _run_chunk_decode(model: GPT, cfg: GPTConfig, *, prefix: torch.Tensor, chunk: torch.Tensor) -> torch.Tensor:
    kv_cache = _kv_cache_for(cfg, batch_size=prefix.size(0), total_len=prefix.size(1) + chunk.size(1))
    _ = model(prefix, kv_cache=kv_cache)
    logits = model(chunk, kv_cache=kv_cache)
    return logits


def _run_checks(
    cfg_ref: GPTConfig,
    cfg_flex: GPTConfig,
    *,
    label: str,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    batch_size: int,
    sequence_len: int,
    kv_prefix_len: int,
    kv_chunk_len: int,
    rtol: float,
    atol: float,
    compile: bool,
) -> list[CheckResult]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model_ref = _make_model(cfg_ref, device)
    model_flex = _make_model(cfg_flex, device)
    model_flex.load_state_dict(model_ref.state_dict(), strict=True)

    if compile:
        model_ref = torch.compile(model_ref)
        model_flex = torch.compile(model_flex)

    results: list[CheckResult] = []

    ids = torch.randint(0, cfg_ref.vocab_size, (batch_size, sequence_len), device=device, dtype=torch.long)
    with torch.inference_mode(), _autocast_ctx(device, dtype):
        out_ref = model_ref(ids)
        out_flex = model_flex(ids)
    results.append(_check_close(f"{label}/full_forward", out_ref, out_flex, rtol=rtol, atol=atol))

    # KV-cache: decode last token
    ids2 = torch.randint(0, cfg_ref.vocab_size, (batch_size, sequence_len), device=device, dtype=torch.long)
    kv_ref = _kv_cache_for(cfg_ref, batch_size=batch_size, total_len=sequence_len)
    kv_flex = _kv_cache_for(cfg_flex, batch_size=batch_size, total_len=sequence_len)
    with torch.inference_mode(), _autocast_ctx(device, dtype):
        _ = model_ref(ids2[:, :-1], kv_cache=kv_ref)
        _ = model_flex(ids2[:, :-1], kv_cache=kv_flex)
        last_ref = model_ref(ids2[:, -1:], kv_cache=kv_ref)[:, -1, :]
        last_flex = model_flex(ids2[:, -1:], kv_cache=kv_flex)[:, -1, :]
    results.append(_check_close(f"{label}/kv_decode_last", last_ref, last_flex, rtol=rtol, atol=atol))

    # KV-cache: chunk decode
    if kv_prefix_len + kv_chunk_len > cfg_ref.sequence_len:
        results.append(
            CheckResult(
                name=f"{label}/kv_chunk_decode",
                max_abs=0.0,
                mean_abs=0.0,
                max_rel=0.0,
                passed=True,
                note="skipped: prefix+chunk > sequence_len",
            )
        )
    else:
        prefix = torch.randint(0, cfg_ref.vocab_size, (batch_size, kv_prefix_len), device=device, dtype=torch.long)
        chunk = torch.randint(0, cfg_ref.vocab_size, (batch_size, kv_chunk_len), device=device, dtype=torch.long)
        with torch.inference_mode(), _autocast_ctx(device, dtype):
            out_ref = _run_chunk_decode(model_ref, cfg_ref, prefix=prefix, chunk=chunk)
            out_flex = _run_chunk_decode(model_flex, cfg_flex, prefix=prefix, chunk=chunk)
        results.append(_check_close(f"{label}/kv_chunk_decode", out_ref, out_flex, rtol=rtol, atol=atol))

    return results


def _run_checks_synaptic(
    cfg_ref,
    cfg_flex,
    *,
    label: str,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    batch_size: int,
    sequence_len: int,
    kv_prefix_len: int,
    kv_chunk_len: int,
    rtol: float,
    atol: float,
    compile: bool,
) -> list[CheckResult]:
    from nanochat.gpt_synaptic import GPTSynaptic

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model_ref = GPTSynaptic(cfg_ref).to(device).train(False)
    model_flex = GPTSynaptic(cfg_flex).to(device).train(False)
    model_flex.load_state_dict(model_ref.state_dict(), strict=True)

    if compile:
        # NOTE: For synaptic KV-cache paths, `torch.compile(dynamic=None)` can trigger Inductor/SymInt
        # lowering failures (e.g., AttributeError: 'int' object has no attribute 'is_Add' from torch.sym_max).
        # `dynamic=False` avoids this by specializing shapes for each seen sequence length.
        model_ref = torch.compile(model_ref, dynamic=False)
        model_flex = torch.compile(model_flex, dynamic=False)

    results: list[CheckResult] = []

    ids = torch.randint(0, cfg_ref.vocab_size, (batch_size, sequence_len), device=device, dtype=torch.long)
    with torch.inference_mode(), _autocast_ctx(device, dtype):
        out_ref, _ = model_ref(ids, train_mode=False)
        out_flex, _ = model_flex(ids, train_mode=False)
    results.append(_check_close(f"{label}/full_forward", out_ref, out_flex, rtol=rtol, atol=atol))

    # KV-cache: decode last token
    ids2 = torch.randint(0, cfg_ref.vocab_size, (batch_size, sequence_len), device=device, dtype=torch.long)
    kv_ref = KVCache(
        batch_size=batch_size,
        num_heads=cfg_ref.n_kv_head,
        seq_len=sequence_len,
        head_dim=cfg_ref.n_embd // cfg_ref.n_head,
        num_layers=cfg_ref.n_layer,
    )
    kv_flex = KVCache(
        batch_size=batch_size,
        num_heads=cfg_flex.n_kv_head,
        seq_len=sequence_len,
        head_dim=cfg_flex.n_embd // cfg_flex.n_head,
        num_layers=cfg_flex.n_layer,
    )
    with torch.inference_mode(), _autocast_ctx(device, dtype):
        _ = model_ref(ids2[:, :-1], kv_cache=kv_ref, train_mode=False)
        _ = model_flex(ids2[:, :-1], kv_cache=kv_flex, train_mode=False)
        last_ref, _ = model_ref(ids2[:, -1:], kv_cache=kv_ref, train_mode=False)
        last_flex, _ = model_flex(ids2[:, -1:], kv_cache=kv_flex, train_mode=False)
        last_ref = last_ref[:, -1, :]
        last_flex = last_flex[:, -1, :]
    results.append(_check_close(f"{label}/kv_decode_last", last_ref, last_flex, rtol=rtol, atol=atol))

    # KV-cache: chunk decode
    if kv_prefix_len + kv_chunk_len > cfg_ref.sequence_len:
        results.append(
            CheckResult(
                name=f"{label}/kv_chunk_decode",
                max_abs=0.0,
                mean_abs=0.0,
                max_rel=0.0,
                passed=True,
                note="skipped: prefix+chunk > sequence_len",
            )
        )
        return results

    prefix = torch.randint(0, cfg_ref.vocab_size, (batch_size, kv_prefix_len), device=device, dtype=torch.long)
    chunk = torch.randint(0, cfg_ref.vocab_size, (batch_size, kv_chunk_len), device=device, dtype=torch.long)
    kv_ref = KVCache(
        batch_size=batch_size,
        num_heads=cfg_ref.n_kv_head,
        seq_len=kv_prefix_len + kv_chunk_len,
        head_dim=cfg_ref.n_embd // cfg_ref.n_head,
        num_layers=cfg_ref.n_layer,
    )
    kv_flex = KVCache(
        batch_size=batch_size,
        num_heads=cfg_flex.n_kv_head,
        seq_len=kv_prefix_len + kv_chunk_len,
        head_dim=cfg_flex.n_embd // cfg_flex.n_head,
        num_layers=cfg_flex.n_layer,
    )
    with torch.inference_mode(), _autocast_ctx(device, dtype):
        _ = model_ref(prefix, kv_cache=kv_ref, train_mode=False)
        _ = model_flex(prefix, kv_cache=kv_flex, train_mode=False)
        out_ref, _ = model_ref(chunk, kv_cache=kv_ref, train_mode=False)
        out_flex, _ = model_flex(chunk, kv_cache=kv_flex, train_mode=False)
    results.append(_check_close(f"{label}/kv_chunk_decode", out_ref, out_flex, rtol=rtol, atol=atol))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify FlexAttention correctness vs SDPA (standard GPT).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model",
        choices=["standard", "synaptic"],
        default="standard",
        help="Which model to compare (standard GPT flex vs SDPA, or synaptic flex path vs non-flex).",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run a small masking/caching suite across MQA/GQA/MHA head configs.",
    )

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-len", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-kv-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=128)

    parser.add_argument("--kv-prefix-len", type=int, default=13)
    parser.add_argument("--kv-chunk-len", type=int, default=7)

    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Relative tolerance for correctness checks (default depends on dtype).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Absolute tolerance for correctness checks (default depends on dtype).",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--write-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write summary.json + run.md under artifacts/certs/flex_attention/<run_id>/",
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

    rtol = float(args.rtol) if args.rtol is not None else 1e-3
    atol = float(args.atol) if args.atol is not None else (1e-2 if dtype is torch.float32 else 2e-2)
    args.rtol = rtol
    args.atol = atol

    results: list[CheckResult] = []

    if args.model == "standard":
        if args.suite:
            suite: list[tuple[str, int, int]] = [("mqa", args.n_head, 1)]
            if args.n_head % 2 == 0:
                suite.append(("gqa", args.n_head, args.n_head // 2))
            suite.append(("mha", args.n_head, args.n_head))

            for label, n_head, n_kv_head in suite:
                cfg_ref = _build_config(
                    argparse.Namespace(
                        **{
                            **vars(args),
                            "n_head": n_head,
                            "n_kv_head": n_kv_head,
                            "use_flex_attention": False,
                        }
                    )
                )
                cfg_flex = _build_config(
                    argparse.Namespace(
                        **{
                            **vars(args),
                            "n_head": n_head,
                            "n_kv_head": n_kv_head,
                            "use_flex_attention": True,
                        }
                    )
                )
                results.extend(
                    _run_checks(
                        cfg_ref,
                        cfg_flex,
                        label=label,
                        device=device,
                        dtype=dtype,
                        seed=args.seed,
                        batch_size=args.batch_size,
                        sequence_len=args.sequence_len,
                        kv_prefix_len=args.kv_prefix_len,
                        kv_chunk_len=args.kv_chunk_len,
                        rtol=rtol,
                        atol=atol,
                        compile=args.compile,
                    )
                )
        else:
            cfg_ref = _build_config(argparse.Namespace(**{**vars(args), "use_flex_attention": False}))
            cfg_flex = _build_config(argparse.Namespace(**{**vars(args), "use_flex_attention": True}))
            results.extend(
                _run_checks(
                    cfg_ref,
                    cfg_flex,
                    label="single",
                    device=device,
                    dtype=dtype,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    sequence_len=args.sequence_len,
                    kv_prefix_len=args.kv_prefix_len,
                    kv_chunk_len=args.kv_chunk_len,
                    rtol=rtol,
                    atol=atol,
                    compile=args.compile,
                )
            )
    else:
        from nanochat.gpt_synaptic import GPTSynapticConfig
        from nanochat.synaptic import SynapticConfig

        def build_syn_cfg(*, use_flex_attention: bool) -> SynapticConfig:
            return SynapticConfig(use_flex_attention=use_flex_attention, stochastic_train_frac=0.0)

        def build_cfg(*, use_flex_attention: bool) -> GPTSynapticConfig:
            return GPTSynapticConfig(
                sequence_len=args.sequence_len,
                vocab_size=args.vocab_size,
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_kv_head=args.n_kv_head,
                n_embd=args.n_embd,
                syn_cfg=build_syn_cfg(use_flex_attention=use_flex_attention),
                dropout=0.0,
            )

        cfg_ref = build_cfg(use_flex_attention=False)
        cfg_flex = build_cfg(use_flex_attention=True)
        results.extend(
            _run_checks_synaptic(
                cfg_ref,
                cfg_flex,
                label="synaptic",
                device=device,
                dtype=dtype,
                seed=args.seed,
                batch_size=args.batch_size,
                sequence_len=args.sequence_len,
                kv_prefix_len=args.kv_prefix_len,
                kv_chunk_len=args.kv_chunk_len,
                rtol=rtol,
                atol=atol,
                compile=args.compile,
            )
        )

    table_title = (
        "FlexAttention correctness (SDPA vs Flex)" if args.model == "standard" else "Synaptic FlexAttention correctness"
    )
    table = Table(title=table_title)
    table.add_column("case", style="bold")
    table.add_column("max_abs", justify="right")
    table.add_column("mean_abs", justify="right")
    table.add_column("max_rel", justify="right")
    table.add_column("pass", justify="center")
    table.add_column("note")

    ok = True
    for r in results:
        ok = ok and r.passed
        table.add_row(
            r.name,
            f"{r.max_abs:.3e}",
            f"{r.mean_abs:.3e}",
            f"{r.max_rel:.3e}",
            "[green]✓[/green]" if r.passed else "[red]✗[/red]",
            r.note,
        )

    console.print(table)
    if ok:
        console.print("[green]All checks passed.[/green]")
        exit_code = 0
    else:
        console.print("[red]One or more checks failed.[/red]")
        exit_code = 1

    if args.write_artifacts:
        run_id = args.run_id or _default_run_id()
        topic = "flex_attention" if args.model == "standard" else "synaptic_flex_attention"
        run_dir = Path(args.artifacts_dir) / "certs" / topic / run_id

        if args.model == "standard":
            cfg_ref = _build_config(argparse.Namespace(**{**vars(args), "use_flex_attention": False}))
            cfg_flex = _build_config(argparse.Namespace(**{**vars(args), "use_flex_attention": True}))
            cfg_payload = {"ref": asdict(cfg_ref), "flex": asdict(cfg_flex)}
        else:
            from nanochat.gpt_synaptic import GPTSynapticConfig
            from nanochat.synaptic import SynapticConfig

            cfg_ref = GPTSynapticConfig(
                sequence_len=args.sequence_len,
                vocab_size=args.vocab_size,
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_kv_head=args.n_kv_head,
                n_embd=args.n_embd,
                syn_cfg=SynapticConfig(use_flex_attention=False, stochastic_train_frac=0.0),
                dropout=0.0,
            )
            cfg_flex = GPTSynapticConfig(
                sequence_len=args.sequence_len,
                vocab_size=args.vocab_size,
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_kv_head=args.n_kv_head,
                n_embd=args.n_embd,
                syn_cfg=SynapticConfig(use_flex_attention=True, stochastic_train_frac=0.0),
                dropout=0.0,
            )
            cfg_payload = {"ref": asdict(cfg_ref), "flex": asdict(cfg_flex)}

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
            "config": cfg_payload,
            "results": {
                "ok": ok,
                "checks": [asdict(r) for r in results],
            },
        }

        report_title = "FlexAttention correctness (SDPA vs Flex)" if args.model == "standard" else "Synaptic FlexAttention correctness (non-flex vs flex)"
        report_md = f"""# {report_title}

- Run ID: `{run_id}`
- Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}
- Overall: {'PASS' if ok else 'FAIL'}

## Command

```bash
uv run python {shlex.join(sys.argv)}
```

## Notes

- This script compares SDPA (reference) vs FlexAttention for standard GPT across:
  - full forward
  - KV-cache last-token decode
  - optional KV chunk decode

See `summary.json` for full details.
"""

        _write_artifacts(run_dir, summary=summary, report_md=report_md)
        console.print(f"[dim]Wrote artifacts → {run_dir}[/dim]")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
