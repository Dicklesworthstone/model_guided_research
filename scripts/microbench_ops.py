"""Per-op microbenchmark for nanochat attention and FFN ops (bead 2cg).

Isolates a SINGLE attention op and a SINGLE FFN op (one transformer block's
sub-modules, not the whole model / training loop) and times forward and
forward+backward across mechanisms, shapes, and dtypes. This is the foundation
the Triton hotspot work (bead c6h) plugs into: it is where a fused max-plus
kernel's before/after numbers live (artifacts/microbench/).

Why per-op (not per-model): docs/flops_validation.md (bead bks) showed tropical
max-plus is 3-6x less hardware-efficient than matmul at the *model* level. This
harness localizes that to the attention op vs the FFN op so a kernel target is
unambiguous.

Usage:
    uv run python scripts/microbench_ops.py                 # default sweep, CPU
    uv run python scripts/microbench_ops.py --device cuda --dtype bf16
    uv run python scripts/microbench_ops.py --out artifacts/microbench/ops.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from nanochat.gpt import GPT, GPTConfig
from nanochat.model_utils import norm
from nanochat.octonion_attention_torch import OctonionCausalSelfAttention, oconj, omul, onormalize

console = Console()
MAIN = Path(__file__).resolve().parents[1]

# (label, attention_type, ffn_type) — mechanisms with a normal .attn slot so we
# can time the attention op in isolation. standard = the matmul baseline,
# tropical = the measured bandwidth-bound max-plus target.
MECHS = [
    ("standard", "standard", "standard"),
    ("tropical", "tropical", "tropical"),
]


def _build_block(attention_type: str, ffn_type: str, n_embd: int, n_head: int, n_kv_head: int, seq: int):
    cfg = GPTConfig(
        sequence_len=seq,
        vocab_size=50304,
        n_layer=1,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
        attention_type=attention_type,
        ffn_type=ffn_type,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        model = GPT(cfg)
    block = model.transformer.h[0]
    cos_sin = (model.cos[:, :seq], model.sin[:, :seq])
    return block, cos_sin


def _time(fn, *, warmup: int, iters: int, device: str) -> float:
    sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] * 1000.0  # median ms


def bench_op(
    kind: str, label: str, attn_t: str, ffn_t: str, *, B, T, n_embd, n_head, n_kv_head, dtype, device, warmup, iters
):
    torch.manual_seed(0)
    block, cos_sin = _build_block(attn_t, ffn_t, n_embd, n_head, n_kv_head, T)
    block = block.to(device=device, dtype=dtype)
    cos_sin = tuple(c.to(device=device) for c in cos_sin)  # rotary stays fp32-ish; module handles cast
    x = torch.randn(B, T, n_embd, device=device, dtype=dtype, requires_grad=True)
    op = (lambda: block.attn(norm(x), cos_sin, None)) if kind == "attn" else (lambda: block.mlp(norm(x)))

    def fwd():
        with torch.no_grad():
            op()

    def fwdbwd():
        if x.grad is not None:
            x.grad = None
        block.zero_grad(set_to_none=True)
        op().pow(2).mean().backward()

    ms_fwd = _time(fwd, warmup=warmup, iters=iters, device=device)
    ms_fb = _time(fwdbwd, warmup=warmup, iters=iters, device=device)
    head_dim = n_embd // n_head
    # analytical peak intermediate for the attention score path: standard keeps a
    # (B,H,T,T) score matrix; tropical max-plus materializes (B,H,T,T,head_dim)
    # before the reduce (tropical_attention_torch.py:21) - the bandwidth cost.
    score_elems = B * n_head * T * T
    interm = score_elems * (head_dim if (kind == "attn" and attn_t == "tropical") else 1)
    return {
        "op": kind,
        "mech": label,
        "attn_type": attn_t,
        "ffn_type": ffn_t,
        "B": B,
        "T": T,
        "n_embd": n_embd,
        "n_head": n_head,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "ms_fwd": round(ms_fwd, 4),
        "ms_fwd_bwd": round(ms_fb, 4),
        "peak_intermediate_elems": interm,
    }


# ---------------------------------------------------------------------------
# Octonion aggregate: per-query loop (pre-7b0.6 reference) vs chunked tiles
# ---------------------------------------------------------------------------


def _octonion_ref_aggregate(self, weights, v, *, q, k, kv_cache, pos0):
    """Verbatim pre-7b0.6 OctonionCausalSelfAttention.aggregate loop."""
    B = q.size(0)
    Tq = q.size(2)
    q_o = onormalize(q.view(B, self.n_head, -1, self.head_dim // 8, 8))
    k_o = onormalize(k.view(B, self.n_head, -1, self.head_dim // 8, 8))
    v_o = v.view(B, self.n_head, -1, self.head_dim // 8, 8)
    k_conj = oconj(k_o)
    y_list = []
    for i in range(Tq):
        q_i = q_o[:, :, i : i + 1]
        r_i = omul(q_i, k_conj)
        term = omul(r_i, v_o)
        p_i = weights[:, :, i].unsqueeze(-1).unsqueeze(-1)
        y_list.append((term * p_i).sum(dim=2).unsqueeze(2))
    return torch.cat(y_list, dim=2).view(B, self.n_head, Tq, self.head_dim)


def _git_info() -> dict:
    def _run(*cmd: str) -> str:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=MAIN, timeout=10).stdout.strip()

    return {
        "commit": _run("git", "rev-parse", "--short", "HEAD"),
        "dirty": bool(_run("git", "status", "--porcelain")),
    }


def bench_octonion_tile(
    *, B: int, T: int, n_embd: int, n_head: int, n_kv_head: int, dtype, device: str, warmup: int, iters: int
) -> dict:
    """Time the octonion attention op with the tiled vectorized aggregate
    (production) against the pre-7b0.6 per-query Python loop (reference,
    bound onto the module instance), plus a Tq==1 decode-shape aggregate.
    Records the fp32 parity gap between the two paths as evidence."""
    torch.manual_seed(0)
    block, cos_sin = _build_block("octonion", "standard", n_embd, n_head, n_kv_head, T)
    block = block.to(device=device, dtype=dtype)
    attn: OctonionCausalSelfAttention = block.attn
    cos_sin = tuple(c.to(device=device) for c in cos_sin)
    x = torch.randn(B, T, n_embd, device=device, dtype=dtype, requires_grad=True)
    head_dim = attn.head_dim

    def run_once(use_reference: bool):
        if use_reference:
            attn.__dict__["aggregate"] = _octonion_ref_aggregate.__get__(attn)
        else:
            attn.__dict__.pop("aggregate", None)
        try:
            return attn(norm(x), cos_sin, None)
        finally:
            attn.__dict__.pop("aggregate", None)

    with torch.no_grad():
        out_ref = run_once(True)
        out_tiled = run_once(False)
    parity_max_abs = float((out_ref - out_tiled).abs().max())

    def timed(use_reference: bool, backward: bool) -> float:
        def fn():
            if backward:
                if x.grad is not None:
                    x.grad = None
                block.zero_grad(set_to_none=True)
                run_once(use_reference).pow(2).mean().backward()
            else:
                with torch.no_grad():
                    run_once(use_reference)

        return _time(fn, warmup=warmup, iters=iters, device=device)

    # Decode shape: ONE query attending the full T-key prefix - aggregate only,
    # no scaffold, matching generation-step geometry (Tq==1 must not regress).
    q1 = torch.randn(B, n_head, 1, head_dim, device=device, dtype=dtype)
    k_full = torch.randn(B, n_head, T, head_dim, device=device, dtype=dtype)
    v_full = torch.randn(B, n_head, T, head_dim, device=device, dtype=dtype)
    w_decode = torch.softmax(attn.score(q1, k_full), dim=-1)

    def decode_aggregate(use_reference: bool):
        agg = _octonion_ref_aggregate.__get__(attn) if use_reference else attn.aggregate
        with torch.no_grad():
            agg(w_decode, v_full, q=q1, k=k_full, kv_cache=None, pos0=None)

    loop_fwd = timed(True, False)
    tiled_fwd = timed(False, False)
    loop_fb = timed(True, True)
    tiled_fb = timed(False, True)
    decode_loop = _time(lambda: decode_aggregate(True), warmup=warmup, iters=iters, device=device)
    decode_tiled = _time(lambda: decode_aggregate(False), warmup=warmup, iters=iters, device=device)
    row = {
        "B": B,
        "T": T,
        "n_embd": n_embd,
        "n_head": n_head,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "parity_max_abs_fp32": parity_max_abs if dtype == torch.float32 else None,
        "loop_ms_fwd": round(loop_fwd, 4),
        "tiled_ms_fwd": round(tiled_fwd, 4),
        "speedup_fwd": round(loop_fwd / max(tiled_fwd, 1e-9), 2),
        "loop_ms_fwd_bwd": round(loop_fb, 4),
        "tiled_ms_fwd_bwd": round(tiled_fb, 4),
        "speedup_fwd_bwd": round(loop_fb / max(tiled_fb, 1e-9), 2),
        "decode_loop_ms": round(decode_loop, 4),
        "decode_tiled_ms": round(decode_tiled, 4),
        "decode_speedup": round(decode_loop / max(decode_tiled, 1e-9), 2),
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--out", default=str(MAIN / "artifacts" / "microbench" / "ops_cpu.json"))
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-kv-head", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seqs", default="256,512", help="comma-separated T values to sweep")
    ap.add_argument(
        "--octonion-tile",
        action="store_true",
        help="bench the octonion tiled aggregate (bead 7b0.6) vs the per-query loop reference",
    )
    ap.add_argument(
        "--torch-threads",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="torch CPU thread cap for the --octonion-tile arm (per-op thread-sync storms on a "
             "shared box otherwise swamp both arms; ratios stay honest either way)",
    )
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    seqs = [int(s) for s in args.seqs.split(",") if s.strip()]
    if args.device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]--device cuda requested but CUDA unavailable; falling back to cpu[/]")
        args.device = "cpu"
    if args.octonion_tile:
        saved_threads = torch.get_num_threads()
        torch.set_num_threads(max(1, args.torch_threads))
        try:
            _octonion_tile_main(args, dtype, seqs)
        finally:
            torch.set_num_threads(saved_threads)
        return 0

    rows = []
    for T in seqs:
        for kind in ("attn", "ffn"):
            for label, attn_t, ffn_t in MECHS:
                rows.append(
                    bench_op(
                        kind,
                        label,
                        attn_t,
                        ffn_t,
                        B=args.batch_size,
                        T=T,
                        n_embd=args.n_embd,
                        n_head=args.n_head,
                        n_kv_head=args.n_kv_head,
                        dtype=dtype,
                        device=args.device,
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                )

    table = Table(
        title=f"per-op microbench — {args.device}/{args.dtype} (B={args.batch_size}, D={args.n_embd}, H={args.n_head})",
        border_style="cyan",
    )
    for col in ("op", "mech", "T", "ms_fwd", "ms_fwd+bwd", "peak_interm_elems", "vs_standard"):
        table.add_column(col, justify="right" if col not in ("op", "mech") else "left")
    # relative slowdown vs the standard arm of the same (op,T)
    base = {(r["op"], r["T"]): r["ms_fwd_bwd"] for r in rows if r["mech"] == "standard"}
    for r in rows:
        b = base.get((r["op"], r["T"]))
        rel = f"{r['ms_fwd_bwd'] / b:.2f}x" if b else "-"
        style = "yellow" if (b and r["ms_fwd_bwd"] / b > 1.5) else ""
        table.add_row(
            r["op"],
            r["mech"],
            str(r["T"]),
            f"{r['ms_fwd']:.2f}",
            f"{r['ms_fwd_bwd']:.2f}",
            f"{r['peak_intermediate_elems']:,}",
            f"[{style}]{rel}[/{style}]" if style else rel,
        )
    console.print(table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "device": args.device,
            "dtype": args.dtype,
            "B": args.batch_size,
            "n_embd": args.n_embd,
            "n_head": args.n_head,
            "n_kv_head": args.n_kv_head,
            "seqs": seqs,
            "warmup": args.warmup,
            "iters": args.iters,
        },
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2))
    console.print(f"[green]wrote[/] {out}")
    return 0


def _octonion_tile_main(args, dtype, seqs) -> None:
    rows = [
        bench_octonion_tile(
            B=args.batch_size,
            T=T,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_kv_head=args.n_kv_head,
            dtype=dtype,
            device=args.device,
            warmup=args.warmup,
            iters=args.iters,
        )
        for T in seqs
    ]
    table = Table(
        title=f"octonion aggregate: tiled vs per-query loop — {args.device}/{args.dtype} "
        f"(B={args.batch_size}, D={args.n_embd}, H={args.n_head}, threads={args.torch_threads})",
        border_style="cyan",
    )
    for col in ("T", "loop fwd", "tiled fwd", "fwd x", "loop f+b", "tiled f+b", "f+b x", "decode x", "parity"):
        table.add_column(col, justify="right")
    for r in rows:
        table.add_row(
            str(r["T"]),
            f"{r['loop_ms_fwd']:.1f}",
            f"{r['tiled_ms_fwd']:.1f}",
            f"{r['speedup_fwd']:.1f}x",
            f"{r['loop_ms_fwd_bwd']:.1f}",
            f"{r['tiled_ms_fwd_bwd']:.1f}",
            f"{r['speedup_fwd_bwd']:.1f}x",
            f"{r['decode_speedup']:.2f}x",
            f"{r['parity_max_abs_fp32']:.2e}" if r["parity_max_abs_fp32"] is not None else "-",
        )
    console.print(table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bead": "model_guided_research-7b0.6",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        "git": _git_info(),
        "hardware": {
            "host": platform.node(),
            "platform": sys.platform,
            "cpu_count_logical": os.cpu_count(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
        "config": {
            "device": args.device,
            "dtype": args.dtype,
            "B": args.batch_size,
            "n_embd": args.n_embd,
            "n_head": args.n_head,
            "n_kv_head": args.n_kv_head,
            "seqs": seqs,
            "warmup": args.warmup,
            "iters": args.iters,
            "torch_threads": args.torch_threads,
        },
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2))
    console.print(f"[green]wrote[/] {out}")


if __name__ == "__main__":
    raise SystemExit(main())
