"""Frozen-worktree campaign launcher (bead nm9j).

Every preregistered campaign trains from a DETACHED WORKTREE pinned at a
clean SHA, never from the shared mutable checkout: on 2026-06-11 two live
campaigns lost runs because a concurrent agent's uncommitted (and entirely
legitimate) edits flipped `git_dirty` mid-flight, and the ci-v2+ verdict
engine rightly refuses tainted evidence. The frozen worktree makes campaign
provenance immune to everything except the campaign's own launch state.

Hard-won rules encoded here:
- the worktree must be CLEAN at launch (verified) and at least as NEW as any
  checkpoint it will load (config fields added later break older loaders);
- every run gets --checkpoint-interval > 0 (a run without a final checkpoint
  cannot be evaluated - cost one probe cycle to learn);
- the FIRST run's provenance header is verified (sha + dirty=False) before
  the rest of the matrix is trusted to the same fate;
- artifacts/data paths are absolute into the MAIN repo so evidence pools in
  one place regardless of which worktree trained it.

Usage:
    uv run python scripts/run_campaign.py \
        --combo dyck:braid --combo dyck:standard --seeds 0,1,2 \
        --target-flops 1e14 --topic e2 [--sha <clean-sha>] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

MAIN = Path(__file__).resolve().parents[1]
console = Console()


def build_plan(combos: list[str], seeds: list[int]) -> list[tuple[str, str, int]]:
    plan = []
    for combo in combos:
        task, _, mech = combo.partition(":")
        if not task or not mech:
            raise SystemExit(f"--combo must be task:mechanism, got {combo!r}")
        for seed in seeds:
            plan.append((task, mech, seed))
    return plan


def freeze_worktree(sha: str, path: Path) -> str:
    resolved = subprocess.run(["git", "rev-parse", sha], cwd=MAIN, capture_output=True, text=True)
    if resolved.returncode != 0:
        raise SystemExit(f"cannot resolve --sha {sha!r}: {resolved.stderr.strip()}")
    full = resolved.stdout.strip()
    if path.exists():
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True).stdout.strip()
        if head != full:
            raise SystemExit(f"worktree {path} pinned at {head[:9]}, wanted {full[:9]}: pass a fresh --worktree path")
    else:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(path), full], cwd=MAIN, check=True, capture_output=True
        )
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=path, capture_output=True, text=True).stdout
    if dirty.strip():
        raise SystemExit(f"frozen worktree {path} is DIRTY - campaigns demand clean provenance:\n{dirty}")
    return full


def run_one(wt: Path, args: argparse.Namespace, task: str, mech: str, seed: int) -> tuple[str, int]:
    run_id = f"{task}-{mech}-s{seed}"
    log = Path(args.log_dir) / f"{run_id}.log"
    cmd = (
        [
            str(MAIN / ".venv" / "bin" / "python"),
            "-m",
            "nanochat.train",
            "--device",
            args.device,
            "--attention-type",
            mech,
            "--data-dir",
            str(MAIN / args.data_root / task),
            "--target-flops",
            str(args.target_flops),
            "--n-layer",
            str(args.n_layer),
            "--n-head",
            str(args.n_head),
            "--n-kv-head",
            str(args.n_kv_head),
            "--n-embd",
            str(args.n_embd),
            "--sequence-len",
            str(args.sequence_len),
            "--batch-size",
            str(args.batch_size),
            "--checkpoint-interval",
            str(args.checkpoint_interval),
            "--seed",
            str(seed),
            "--artifacts-dir",
            str(MAIN / "artifacts"),
            "--artifacts-kind",
            "campaigns",
            "--artifacts-topic",
            args.topic,
            "--run-id",
            run_id,
        ]
        + (
            # val cadence (z4xx fresh-eyes finding): train.py defaults
            # val_interval to 0, so campaign artifacts recorded
            # results.val_ce_final = null - any registration on that metric
            # (hyp-symplectic-nonorm-*, hyp-ordinal-*) was unfulfillable by
            # campaign evidence. Opt-in to preserve old campaign behavior.
            ["--val-interval", str(args.val_interval), "--val-batches", str(args.val_batches)]
            if args.val_interval > 0
            else []
        )
        + (shlex.split(args.extra_args) if args.extra_args else [])
    )
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(args.threads)
    with log.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=wt, env=env)
    return run_id, proc.returncode


def verify_provenance(topic: str, run_id: str, want_sha: str) -> bool:
    metrics = MAIN / "artifacts" / "campaigns" / topic / run_id / "metrics.jsonl"
    if not metrics.exists():
        return False
    header = json.loads(metrics.open().readline())
    return header.get("git_sha") == want_sha and header.get("git_dirty") is False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--combo", action="append", required=True, help="task:mechanism (repeatable)")
    parser.add_argument("--seeds", default="0,1,2", help="comma-separated training seeds")
    parser.add_argument("--target-flops", default="1e14")
    parser.add_argument("--topic", required=True, help="artifacts/campaigns/<topic>/ destination")
    parser.add_argument("--sha", default="HEAD", help="clean commit to freeze the campaign at")
    parser.add_argument("--worktree", type=Path, default=None, help="worktree path (default /data/tmp/wt-<topic>)")
    parser.add_argument("--data-root", default="artifacts/diagnostics_e1")
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-kv-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--sequence-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=100_000)  # >0: final checkpoint always
    parser.add_argument(
        "--val-interval",
        type=int,
        default=0,
        help="validation cadence in steps (0 = off, the historical campaign default; "
        "REQUIRED >0 for any registration on train:results.val_ce_final)",
    )
    parser.add_argument("--val-batches", type=int, default=16, help="batches per validation evaluation")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=6, help="OMP threads per trainer")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--extra-args", default="", help="extra nanochat.train flags, shell-quoted")
    parser.add_argument("--log-dir", default="/tmp/campaign_logs")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    plan = build_plan(args.combo, seeds)
    table = Table(title=f"campaign plan — topic={args.topic} ({len(plan)} runs)", border_style="cyan")
    for col in ("task", "mechanism", "seed"):
        table.add_column(col)
    for task, mech, seed in plan:
        table.add_row(task, mech, str(seed))
    console.print(table)
    if args.dry_run:
        return 0

    wt = args.worktree or Path(f"/data/tmp/wt-{args.topic}")
    sha = freeze_worktree(args.sha, wt)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    console.print(
        Panel(
            f"frozen worktree [bold]{wt}[/bold] @ {sha[:9]} (clean) · {args.workers} workers × {args.threads} threads",
            border_style="green",
        )
    )

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, wt, args, t, m, s): (t, m, s) for t, m, s in plan}
        first_verified = False
        for fut in list(futures):
            run_id, code = fut.result()
            ok = code == 0
            if ok and not first_verified:
                if not verify_provenance(args.topic, run_id, sha):
                    console.print(
                        f"[bold red]PROVENANCE MISMATCH on {run_id} - aborting trust in this campaign[/bold red]"
                    )
                    failures.append(f"{run_id} (provenance)")
                first_verified = True
            console.print(f"[{'green' if ok else 'red'}]{run_id} exit={code}[/{'green' if ok else 'red'}]")
            if not ok:
                failures.append(run_id)
    if failures:
        console.print(Panel("\n".join(failures), title="[red]failures[/red]", border_style="red"))
        return 1
    console.print(
        Panel(
            f"[bold green]{len(plan)} runs complete, provenance pinned at {sha[:9]}[/bold green]", border_style="green"
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
