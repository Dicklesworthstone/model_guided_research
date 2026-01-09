from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from cmaes import CMA
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from nanochat.synaptic import SynapticConfig

console = Console()

PENALTY_SCORE = 1e9


Kind = Literal["linear", "log10"]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: Kind
    low: float
    high: float

    def clip_x(self, x: float) -> float:
        return float(min(self.high, max(self.low, x)))

    def decode(self, x: float) -> float:
        x = self.clip_x(float(x))
        if self.kind == "linear":
            return float(x)
        if self.kind == "log10":
            return float(10.0 ** x)
        raise ValueError(f"Unknown kind: {self.kind}")

    def encode(self, value: float) -> float:
        v = float(value)
        if self.kind == "linear":
            return self.clip_x(v)
        if self.kind == "log10":
            if v <= 0:
                raise ValueError(f"Cannot encode non-positive value for log10 param {self.name}: {v}")
            return self.clip_x(math.log10(v))
        raise ValueError(f"Unknown kind: {self.kind}")


PARAM_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec("tau_c", "linear", 0.70, 0.99),
    ParamSpec("alpha_c", "linear", 0.10, 1.00),
    ParamSpec("init_rrp", "linear", 1.0, 18.0),
    ParamSpec("prime_rate", "linear", 0.01, 0.20),
    ParamSpec("rec_rate", "linear", 0.01, 0.20),
    ParamSpec("lambda_loge", "linear", 0.0, 4.0),
    ParamSpec("barrier_strength", "linear", 0.0, 0.50),
    ParamSpec("stochastic_train_frac", "linear", 0.0, 0.40),
    ParamSpec("post_fast_lr", "log10", -4.5, -2.0),
    ParamSpec("post_slow_lr", "log10", -5.5, -3.0),
)


def _default_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _bounds() -> np.ndarray:
    return np.array([(p.low, p.high) for p in PARAM_SPECS], dtype=np.float64)


def _decode_x(x: np.ndarray) -> dict[str, float]:
    if x.shape != (len(PARAM_SPECS),):
        raise ValueError(f"Expected x shape {(len(PARAM_SPECS),)}, got {x.shape}")
    decoded: dict[str, float] = {}
    for i, p in enumerate(PARAM_SPECS):
        decoded[p.name] = p.decode(float(x[i]))
    return decoded


def _encode_syn_defaults() -> np.ndarray:
    base = SynapticConfig()
    xs: list[float] = []
    for p in PARAM_SPECS:
        xs.append(p.encode(float(getattr(base, p.name))))
    return np.array(xs, dtype=np.float64)


def _mean_tail(values: list[float], *, tail: int) -> float:
    if not values:
        return float("inf")
    tail = max(1, int(tail))
    window = values[-tail:]
    return float(sum(window) / len(window))


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _train_eval(
    *,
    artifacts_dir: Path,
    search_run_id: str,
    gen: int,
    cand: int,
    eval_seed: int,
    decoded: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidate_dir = artifacts_dir / "cmaes" / "phase1" / search_run_id / "eval" / f"gen_{gen:04d}" / f"cand_{cand:04d}"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    syn_cfg_path = candidate_dir / "synaptic_config.json"
    _write_json(syn_cfg_path, decoded)

    eval_id = f"seed_{eval_seed}"
    train_cmd = [
        sys.executable,
        "-m",
        "nanochat.train",
        "--model-type",
        "synaptic",
        "--synaptic-config",
        str(syn_cfg_path),
        "--device",
        str(args.device),
        "--seed",
        str(eval_seed),
        "--batch-size",
        str(args.batch_size),
        "--sequence-len",
        str(args.sequence_len),
        "--vocab-size",
        str(args.vocab_size),
        "--n-layer",
        str(args.n_layer),
        "--n-head",
        str(args.n_head),
        "--n-kv-head",
        str(args.n_kv_head),
        "--n-embd",
        str(args.n_embd),
        "--learning-rate",
        str(args.learning_rate),
        "--target-flops",
        str(args.target_flops),
        "--warmup-steps",
        str(args.warmup_steps),
        "--log-interval",
        str(args.log_interval),
        "--artifacts-dir",
        str(artifacts_dir),
        "--artifacts-kind",
        "cmaes",
        "--artifacts-topic",
        f"phase1/{search_run_id}/eval/gen_{gen:04d}/cand_{cand:04d}",
        "--run-id",
        eval_id,
    ]
    if args.auto_download_data:
        train_cmd.append("--auto-download-data")
        train_cmd.extend(["--min-parquet-files", str(args.min_parquet_files)])

    cmd_str = shlex.join(train_cmd)
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
        proc_stdout = proc.stdout
        proc_stderr = proc.stderr
        returncode = int(proc.returncode)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        proc_stdout = exc.stdout or ""
        proc_stderr = exc.stderr or ""
        returncode = 124
    t1 = time.perf_counter()

    train_dir = candidate_dir / eval_id
    stdout_path = candidate_dir / f"{eval_id}.stdout.txt"
    stderr_path = candidate_dir / f"{eval_id}.stderr.txt"
    _write_text(stdout_path, proc_stdout)
    _write_text(stderr_path, proc_stderr)

    summary_path = train_dir / "summary.json"
    status = "timeout" if timed_out else ("ok" if returncode == 0 and summary_path.exists() else "error")
    score = float(PENALTY_SCORE)
    losses: list[float] = []
    if status == "ok":
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        results = summary.get("results", {})
        losses = list(results.get("losses", []))
        score = _mean_tail([float(x) for x in losses], tail=int(args.score_tail))
        if not math.isfinite(score):
            status = "error"
            score = float(PENALTY_SCORE)

    return {
        "status": status,
        "score": float(score),
        "duration_s": float(t1 - t0),
        "command": cmd_str,
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout_path": str(stdout_path.relative_to(artifacts_dir)),
        "stderr_path": str(stderr_path.relative_to(artifacts_dir)),
        "train_summary_path": str(summary_path.relative_to(artifacts_dir)) if summary_path.exists() else None,
        "loss_tail_mean": float(score) if math.isfinite(score) else None,
        "losses": losses[-min(len(losses), int(args.score_tail)) :],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CMA-ES Phase 1 pilot for nanochat synaptic knobs.")
    parser.add_argument("--run-id", type=str, default=None, help="Run identifier (directory name).")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Base directory for artifacts.")

    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.30)

    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=123)

    parser.add_argument("--target-flops", type=float, default=1e10)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--score-tail", type=int, default=3, help="Mean of last N losses used as score.")
    parser.add_argument("--timeout-s", type=float, default=600.0)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-len", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-kv-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=6e-4)

    parser.add_argument(
        "--auto-download-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download minimal dataset shards if missing (recommended for reproducibility).",
    )
    parser.add_argument("--min-parquet-files", type=int, default=2)

    args = parser.parse_args()

    if args.generations < 1:
        raise ValueError("--generations must be >= 1")
    if args.population_size < 4:
        raise ValueError("--population-size must be >= 4 (CMA-ES requires mu>=2; smaller values break adaptation)")
    if args.sigma <= 0:
        raise ValueError("--sigma must be > 0")
    if args.target_flops <= 0:
        raise ValueError("--target-flops must be > 0")

    run_id = args.run_id or _default_run_id()
    artifacts_dir = Path(args.artifacts_dir)
    run_dir = artifacts_dir / "cmaes" / "phase1" / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run dir already exists and is non-empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    bounds = _bounds()
    mean0 = _encode_syn_defaults()

    opt = CMA(
        mean=mean0,
        sigma=float(args.sigma),
        bounds=bounds,
        seed=int(args.search_seed),
        population_size=int(args.population_size),
    )

    run_spec: dict[str, Any] = {
        "schema_version": "mgr.cmaes.phase1.v1",
        "run_id": run_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": shlex.join(["uv", "run", "python", "scripts/cmaes_phase1.py"] + sys.argv[1:]),
        "cmaes": {
            "library": "cmaes",
            "population_size": int(args.population_size),
            "sigma": float(args.sigma),
            "search_seed": int(args.search_seed),
        },
        "objective": {
            "model_type": "synaptic",
            "device": str(args.device),
            "target_flops": float(args.target_flops),
            "eval_seed": int(args.eval_seed),
            "score_tail": int(args.score_tail),
            "train_args": {
                "batch_size": int(args.batch_size),
                "sequence_len": int(args.sequence_len),
                "vocab_size": int(args.vocab_size),
                "n_layer": int(args.n_layer),
                "n_head": int(args.n_head),
                "n_kv_head": int(args.n_kv_head),
                "n_embd": int(args.n_embd),
                "learning_rate": float(args.learning_rate),
                "warmup_steps": int(args.warmup_steps),
                "log_interval": int(args.log_interval),
            },
        },
        "param_space": {
            "dim": len(PARAM_SPECS),
            "specs": [asdict(p) for p in PARAM_SPECS],
        },
    }
    _write_json(run_dir / "run.json", run_spec)

    progress_path = run_dir / "progress.csv"
    best_path = run_dir / "best.json"

    with progress_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gen",
                "cand",
                "eval_seed",
                "status",
                "score",
                "duration_s",
                "train_summary_path",
            ],
        )
        writer.writeheader()

        best: dict[str, Any] | None = None

        total_evals = int(args.generations) * int(args.population_size)
        with Progress(
            TextColumn("[bold cyan]cmaes[/bold cyan]"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("eval", total=total_evals)

            for gen in range(int(args.generations)):
                solutions: list[tuple[np.ndarray, float]] = []

                for cand in range(int(args.population_size)):
                    x = opt.ask()
                    decoded = _decode_x(x)
                    eval_seed = int(args.eval_seed)
                    result = _train_eval(
                        artifacts_dir=artifacts_dir,
                        search_run_id=run_id,
                        gen=gen,
                        cand=cand,
                        eval_seed=eval_seed,
                        decoded=decoded,
                        args=args,
                    )

                    score = float(result["score"])
                    solutions.append((x, score))

                    row = {
                        "gen": gen,
                        "cand": cand,
                        "eval_seed": eval_seed,
                        "status": result["status"],
                        "score": score,
                        "duration_s": float(result["duration_s"]),
                        "train_summary_path": result["train_summary_path"],
                    }
                    writer.writerow(row)
                    f.flush()

                    if best is None or score < float(best["score"]):
                        best = {
                            "score": score,
                            "gen": gen,
                            "cand": cand,
                            "x": [float(v) for v in x.tolist()],
                            "decoded": decoded,
                            "train_summary_path": result["train_summary_path"],
                        }
                        _write_json(best_path, best)

                    prog.advance(task)

                opt.tell(solutions)

    if best_path.exists():
        best_obj = json.loads(best_path.read_text(encoding="utf-8"))
        table = Table(title="CMA-ES Phase 1 best", show_header=True, header_style="bold")
        table.add_column("param")
        table.add_column("value", justify="right")
        for p in PARAM_SPECS:
            table.add_row(p.name, f"{float(best_obj['decoded'][p.name]):.6g}")
        console.print(table)
        console.print(f"[bold green]best score[/bold green] = {float(best_obj['score']):.6f}")

        summary_md = f"""# CMA-ES Phase 1 (pilot)

- Run ID: `{run_id}`
- Best score: `{float(best_obj['score']):.6f}`
- Best candidate: gen `{best_obj['gen']}`, cand `{best_obj['cand']}`
- Best train summary: `{best_obj.get('train_summary_path')}`

## Go / No-Go (Phase 2)

Go if the run completes reliably and best score improves meaningfully over baseline within the tiny budget.
No-Go if failures dominate or scores are flat/noisy.
"""
        _write_text(run_dir / "summary.md", summary_md)

    console.print(f"[bold green]done[/bold green] → {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
