from __future__ import annotations

"""CMA-ES Phase 1 search for nanochat synaptic knobs, with production-grade
robustness for longer / preemptible runs.

Beyond the original pilot (bead model_guided_research-68v) this harness adds:

* **Multi-seed evaluation** (bead q8f): each candidate is scored over a set of
  ``--eval-seeds``; the per-seed scores are aggregated (mean / worst /
  mean+std) so the search is not fooled by single-seed luck.
* **Budget / auto-stop guard** (bead 2mj): ``--max-evals``,
  ``--max-wall-seconds``, ``--patience`` (generations without improvement) and
  ``--max-crash-rate`` stop the search gracefully with a recorded reason
  instead of running a fixed generation count blind.
* **Resume / preemption robustness** (bead a3u): the CMA optimizer state and a
  search ledger are checkpointed (atomically) after every generation; ``--resume``
  reloads them and continues, honoring the original budget.
* **Dataset snapshot / split pinning** (bead wiz): the parquet corpus is
  fingerprinted (path/size/mtime) into ``run.json``; on resume a drift in the
  fingerprint is surfaced as a loud warning so candidates are never silently
  compared across different data.

The objective shells out to ``python -m nanochat.train`` (the real training
path) at a fixed FLOPs budget and scores the mean of the last N losses.

Example
-------
    uv run python scripts/cmaes_phase1.py --run-id phase1 --device cpu \
        --generations 6 --population-size 6 --eval-seeds 0 1 \
        --max-wall-seconds 7200 --patience 3
    # ... preempted ...
    uv run python scripts/cmaes_phase1.py --run-id phase1 --resume
"""

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
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

from nanochat.dataset import list_parquet_files
from nanochat.synaptic import SynapticConfig

console = Console()

PENALTY_SCORE = 1e9
STATE_SCHEMA = "mgr.cmaes.phase1.state.v1"


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
            return float(10.0**x)
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


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write-then-rename so a kill mid-write never corrupts the checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    _atomic_write_bytes(path, (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8"))


# --------------------------------------------------------------------------- #
# Dataset snapshot / split pinning (bead wiz)
# --------------------------------------------------------------------------- #
def _dataset_fingerprint(data_dir: str | None) -> dict[str, Any]:
    """Fingerprint the parquet corpus training will actually read.

    Mirrors the dataloader's resolution (``list_parquet_files``; data_dir=None
    -> the FineWeb cache, sorted, last file is the val split). We hash file
    metadata (name, size, mtime_ns) rather than 100s of MB of content -- enough
    to detect a corpus that changed between generations without the I/O cost.
    """
    try:
        paths = list_parquet_files(data_dir)
    except Exception as exc:  # noqa: BLE001 - never block the search on this
        return {"resolved": False, "error": f"{type(exc).__name__}: {exc}", "data_dir": data_dir}
    files: list[dict[str, Any]] = []
    hasher = hashlib.sha256()
    for p in paths:
        pp = Path(p)
        try:
            st = pp.stat()
            entry = {"name": pp.name, "size_bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
        except OSError as exc:
            entry = {"name": pp.name, "error": str(exc)}
        files.append(entry)
        hasher.update(json.dumps(entry, sort_keys=True).encode("utf-8"))
    return {
        "resolved": True,
        "data_dir": data_dir,
        "n_files": len(files),
        "train_files": max(0, len(files) - 1),
        "val_files": 1 if files else 0,
        "files": files,
        "digest": hasher.hexdigest(),
        "method": "metadata-sha256(name,size,mtime_ns)",
    }


@dataclass
class CellEval:
    """Result of one (candidate, seed) training subprocess."""

    seed: int
    status: str
    score: float
    duration_s: float
    command: str
    returncode: int
    train_summary_path: str | None
    losses_tail: list[float]


def _train_eval(
    *,
    artifacts_dir: Path,
    search_run_id: str,
    gen: int,
    cand: int,
    eval_seed: int,
    decoded: dict[str, float],
    args: argparse.Namespace,
) -> CellEval:
    candidate_dir = (
        artifacts_dir / "cmaes" / "phase1" / search_run_id / "eval" / f"gen_{gen:04d}" / f"cand_{cand:04d}"
    )
    candidate_dir.mkdir(parents=True, exist_ok=True)

    syn_cfg_path = candidate_dir / "synaptic_config.json"
    _write_json(syn_cfg_path, decoded)

    eval_id = f"seed_{eval_seed}"
    train_cmd = [
        sys.executable, "-m", "nanochat.train",
        "--model-type", "synaptic",
        "--synaptic-config", str(syn_cfg_path),
        "--device", str(args.device),
        "--seed", str(eval_seed),
        "--batch-size", str(args.batch_size),
        "--sequence-len", str(args.sequence_len),
        "--vocab-size", str(args.vocab_size),
        "--n-layer", str(args.n_layer),
        "--n-head", str(args.n_head),
        "--n-kv-head", str(args.n_kv_head),
        "--n-embd", str(args.n_embd),
        "--learning-rate", str(args.learning_rate),
        "--target-flops", str(args.target_flops),
        "--warmup-steps", str(args.warmup_steps),
        "--log-interval", str(args.log_interval),
        "--artifacts-dir", str(artifacts_dir),
        "--artifacts-kind", "cmaes",
        "--artifacts-topic", f"phase1/{search_run_id}/eval/gen_{gen:04d}/cand_{cand:04d}",
        "--run-id", eval_id,
    ]
    if args.data_dir is not None:
        train_cmd += ["--data-dir", str(args.data_dir)]
    if args.auto_download_data:
        train_cmd += ["--auto-download-data", "--min-parquet-files", str(args.min_parquet_files)]

    cmd_str = shlex.join(train_cmd)
    t0 = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            train_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=float(args.timeout_s), check=False,
        )
        proc_stdout, proc_stderr = proc.stdout, proc.stderr
        returncode = int(proc.returncode)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        proc_stdout = exc.stdout or ""
        proc_stderr = exc.stderr or ""
        returncode = 124
    duration_s = time.perf_counter() - t0

    train_dir = candidate_dir / eval_id
    _write_text(candidate_dir / f"{eval_id}.stdout.txt", proc_stdout)
    _write_text(candidate_dir / f"{eval_id}.stderr.txt", proc_stderr)

    summary_path = train_dir / "summary.json"
    status = "timeout" if timed_out else ("ok" if returncode == 0 and summary_path.exists() else "error")
    score = float(PENALTY_SCORE)
    losses: list[float] = []
    if status == "ok":
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        losses = [float(x) for x in summary.get("results", {}).get("losses", [])]
        score = _mean_tail(losses, tail=int(args.score_tail))
        if not math.isfinite(score):
            status = "error"
            score = float(PENALTY_SCORE)

    return CellEval(
        seed=eval_seed,
        status=status,
        score=float(score),
        duration_s=float(duration_s),
        command=cmd_str,
        returncode=returncode,
        train_summary_path=str(summary_path.relative_to(artifacts_dir)) if summary_path.exists() else None,
        losses_tail=losses[-min(len(losses), int(args.score_tail)):],
    )


def _aggregate_seed_scores(evals: list[CellEval], *, how: str, lam: float) -> float:
    """Combine per-seed scores into the value CMA-ES minimizes.

    Failed seeds keep their PENALTY score so a config that crashes on some
    seeds is correctly penalized regardless of aggregation mode.
    """
    scores = [e.score for e in evals]
    if not scores:
        return float(PENALTY_SCORE)
    if how == "worst":
        return float(max(scores))
    mean = float(sum(scores) / len(scores))
    if how == "mean":
        return mean
    if how == "mean_std":
        if len(scores) < 2:
            return mean
        std = float(np.std(scores, ddof=1))
        return mean + lam * std
    raise ValueError(f"Unknown seed aggregation: {how}")


@dataclass
class SearchState:
    """Everything needed to resume the search after preemption (bead a3u)."""

    generation: int  # next generation index to run
    eval_count: int  # cumulative per-seed training evals executed
    crash_count: int  # cumulative non-ok evals
    wall_accum_s: float  # wall time accumulated across prior segments
    no_improve_streak: int
    best: dict[str, Any] | None

    def to_json(self) -> dict[str, Any]:
        return {"schema_version": STATE_SCHEMA, **asdict(self)}


def _save_checkpoint(state_dir: Path, opt: CMA, state: SearchState) -> None:
    _atomic_write_bytes(state_dir / "cma_state.pkl", pickle.dumps(opt))
    _atomic_write_json(state_dir / "search_state.json", state.to_json())


def _load_checkpoint(state_dir: Path) -> tuple[CMA, SearchState]:
    opt = pickle.loads((state_dir / "cma_state.pkl").read_bytes())
    raw = json.loads((state_dir / "search_state.json").read_text(encoding="utf-8"))
    state = SearchState(
        generation=int(raw["generation"]),
        eval_count=int(raw["eval_count"]),
        crash_count=int(raw["crash_count"]),
        wall_accum_s=float(raw["wall_accum_s"]),
        no_improve_streak=int(raw["no_improve_streak"]),
        best=raw.get("best"),
    )
    return opt, state


def _check_budget(state: SearchState, args: argparse.Namespace, wall_now_s: float) -> str | None:
    """Return a stop reason if any budget guard trips, else None."""
    if args.max_evals is not None and state.eval_count >= int(args.max_evals):
        return f"max_evals reached ({state.eval_count} >= {args.max_evals})"
    total_wall = state.wall_accum_s + wall_now_s
    if args.max_wall_seconds is not None and total_wall >= float(args.max_wall_seconds):
        return f"max_wall_seconds reached ({total_wall:.0f}s >= {args.max_wall_seconds:.0f}s)"
    if args.patience is not None and state.no_improve_streak >= int(args.patience):
        return f"patience exhausted ({state.no_improve_streak} >= {args.patience} gens w/o improvement)"
    if args.max_crash_rate is not None and state.eval_count > 0:
        rate = state.crash_count / state.eval_count
        if state.eval_count >= int(args.population_size) and rate > float(args.max_crash_rate):
            return f"crash_rate too high ({rate:.0%} > {float(args.max_crash_rate):.0%})"
    return None


def _restore_args_from_run_json(args: argparse.Namespace, prev: dict[str, Any], argv: list[str]) -> None:
    """Resume must reuse the ORIGINAL search-defining args (the CMA object has
    population_size etc. baked in; argv defaults would corrupt ``tell``). Those
    are always restored from run.json. Budget guards are restored too, but an
    explicitly-passed flag on resume wins so the budget can be *extended*.
    """
    obj = prev.get("objective", {})
    cma = prev.get("cmaes", {})
    budget = prev.get("budget", {})
    ta = obj.get("train_args", {})

    # always-restore (search identity): changing these mid-run is incoherent
    args.population_size = int(cma.get("population_size", args.population_size))
    args.sigma = float(cma.get("sigma", args.sigma))
    args.search_seed = int(cma.get("search_seed", args.search_seed))
    args.target_flops = float(obj.get("target_flops", args.target_flops))
    args.eval_seeds = list(obj.get("eval_seeds", args.eval_seeds))
    args.seed_agg = str(obj.get("seed_agg", args.seed_agg))
    args.seed_agg_lambda = float(obj.get("seed_agg_lambda", args.seed_agg_lambda))
    args.score_tail = int(obj.get("score_tail", args.score_tail))
    args.device = str(obj.get("device", args.device))
    ds_dir = (prev.get("dataset") or {}).get("data_dir")
    if "--data-dir" not in " ".join(argv):
        args.data_dir = ds_dir
    for k in ("batch_size", "sequence_len", "vocab_size", "n_layer", "n_head",
              "n_kv_head", "n_embd", "learning_rate", "warmup_steps", "log_interval"):
        if k in ta:
            setattr(args, k, ta[k])

    # budget guards: restore unless explicitly overridden on the resume command
    def _passed(flag: str) -> bool:
        return any(a == flag or a.startswith(flag + "=") for a in argv)

    if not _passed("--generations"):
        args.generations = int(budget.get("generations", args.generations))
    if not _passed("--max-evals"):
        args.max_evals = budget.get("max_evals", args.max_evals)
    if not _passed("--max-wall-seconds"):
        args.max_wall_seconds = budget.get("max_wall_seconds", args.max_wall_seconds)
    if not _passed("--patience"):
        args.patience = budget.get("patience", args.patience)
    if not _passed("--max-crash-rate"):
        args.max_crash_rate = budget.get("max_crash_rate", args.max_crash_rate)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CMA-ES Phase 1 search for nanochat synaptic knobs.")
    parser.add_argument("--run-id", type=str, default=None, help="Run identifier (directory name).")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Base directory for artifacts.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted run from its saved CMA + ledger checkpoint.")

    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--generations", type=int, default=2, help="Maximum number of generations (a budget guard may stop earlier).")
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.30)

    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[123],
                        help="Per-candidate training seed(s); >1 enables robust multi-seed averaging (bead q8f).")
    parser.add_argument("--seed-agg", choices=["mean", "worst", "mean_std"], default="mean",
                        help="How to combine per-seed scores into the CMA-ES objective.")
    parser.add_argument("--seed-agg-lambda", type=float, default=1.0,
                        help="Std penalty weight when --seed-agg=mean_std.")

    # Budget / auto-stop guard (bead 2mj)
    parser.add_argument("--max-evals", type=int, default=None, help="Stop after this many per-seed training evals.")
    parser.add_argument("--max-wall-seconds", type=float, default=None, help="Stop once cumulative wall time exceeds this.")
    parser.add_argument("--patience", type=int, default=None, help="Stop after N generations without best-score improvement.")
    parser.add_argument("--min-improve", type=float, default=1e-4, help="Improvement smaller than this does not reset patience.")
    parser.add_argument("--max-crash-rate", type=float, default=None, help="Stop if the fraction of failed evals exceeds this (0..1).")

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

    parser.add_argument("--data-dir", type=str, default=None,
                        help="Pin training to a specific parquet corpus (default: the FineWeb cache).")
    parser.add_argument(
        "--auto-download-data", action=argparse.BooleanOptionalAction, default=True,
        help="Download minimal dataset shards if missing (recommended for reproducibility).",
    )
    parser.add_argument("--min-parquet-files", type=int, default=2)
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if args.generations < 1:
        raise ValueError("--generations must be >= 1")
    if args.population_size < 4:
        raise ValueError("--population-size must be >= 4 (CMA-ES requires mu>=2; smaller values break adaptation)")
    if args.sigma <= 0:
        raise ValueError("--sigma must be > 0")
    if args.target_flops <= 0:
        raise ValueError("--target-flops must be > 0")
    if not args.eval_seeds:
        raise ValueError("--eval-seeds must list at least one seed")

    run_id = args.run_id or _default_run_id()
    artifacts_dir = Path(args.artifacts_dir)
    run_dir = artifacts_dir / "cmaes" / "phase1" / run_id
    state_dir = run_dir / "state"
    progress_path = run_dir / "progress.csv"
    best_path = run_dir / "best.json"

    bounds = _bounds()
    fingerprint = _dataset_fingerprint(args.data_dir)

    # ---- fresh vs resume ---------------------------------------------------
    progress_fields = [
        "gen", "cand", "eval_seeds", "n_ok", "n_fail", "agg_score",
        "per_seed_scores", "duration_s", "best_summary_path",
    ]
    if args.resume:
        if not (state_dir / "cma_state.pkl").exists():
            raise FileNotFoundError(f"--resume given but no checkpoint at {state_dir}")
        opt, state = _load_checkpoint(state_dir)
        prev = json.loads((run_dir / "run.json").read_text(encoding="utf-8")) if (run_dir / "run.json").exists() else {}
        _restore_args_from_run_json(args, prev, sys.argv)
        # recompute the fingerprint now that data_dir is restored, so the drift
        # check compares the same corpus the original run pinned.
        fingerprint = _dataset_fingerprint(args.data_dir)
        prev_fp = (prev.get("dataset") or {}).get("digest")
        if prev_fp and fingerprint.get("digest") and prev_fp != fingerprint["digest"]:
            console.print("[bold red]⚠ DATASET DRIFT[/bold red]: parquet fingerprint changed since the "
                          f"original run ({prev_fp[:12]} → {fingerprint['digest'][:12]}). "
                          "Resumed candidates are NOT comparable to earlier ones.")
        console.print(f"[bold cyan]resume[/bold cyan] gen={state.generation} eval_count={state.eval_count} "
                      f"best={(state.best or {}).get('score')}")
        csv_handle = progress_path.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_handle, fieldnames=progress_fields)
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Run dir already exists and is non-empty: {run_dir} (use --resume to continue)")
        run_dir.mkdir(parents=True, exist_ok=True)
        opt = CMA(
            mean=_encode_syn_defaults(), sigma=float(args.sigma), bounds=bounds,
            seed=int(args.search_seed), population_size=int(args.population_size),
        )
        state = SearchState(generation=0, eval_count=0, crash_count=0,
                            wall_accum_s=0.0, no_improve_streak=0, best=None)
        _write_json(run_dir / "run.json", {
            "schema_version": "mgr.cmaes.phase1.v1",
            "run_id": run_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": shlex.join(["uv", "run", "python", "scripts/cmaes_phase1.py"] + sys.argv[1:]),
            "cmaes": {
                "library": "cmaes", "population_size": int(args.population_size),
                "sigma": float(args.sigma), "search_seed": int(args.search_seed),
            },
            "objective": {
                "model_type": "synaptic", "device": str(args.device),
                "target_flops": float(args.target_flops),
                "eval_seeds": list(args.eval_seeds), "seed_agg": str(args.seed_agg),
                "seed_agg_lambda": float(args.seed_agg_lambda), "score_tail": int(args.score_tail),
                "train_args": {
                    "batch_size": int(args.batch_size), "sequence_len": int(args.sequence_len),
                    "vocab_size": int(args.vocab_size), "n_layer": int(args.n_layer),
                    "n_head": int(args.n_head), "n_kv_head": int(args.n_kv_head),
                    "n_embd": int(args.n_embd), "learning_rate": float(args.learning_rate),
                    "warmup_steps": int(args.warmup_steps), "log_interval": int(args.log_interval),
                },
            },
            "budget": {
                "generations": int(args.generations), "max_evals": args.max_evals,
                "max_wall_seconds": args.max_wall_seconds, "patience": args.patience,
                "max_crash_rate": args.max_crash_rate,
            },
            "dataset": fingerprint,
            "param_space": {"dim": len(PARAM_SPECS), "specs": [asdict(p) for p in PARAM_SPECS]},
        })
        csv_handle = progress_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_handle, fieldnames=progress_fields)
        writer.writeheader()
        csv_handle.flush()

    # ---- search loop -------------------------------------------------------
    segment_start = time.perf_counter()
    stop_reason: str | None = None
    n_seeds = len(args.eval_seeds)

    console.rule(f"[bold cyan]CMA-ES phase1[/bold cyan] · {run_id}")
    remaining_gens = max(0, int(args.generations) - state.generation)
    with Progress(
        TextColumn("[bold cyan]cmaes[/bold cyan]"), BarColumn(), MofNCompleteColumn(),
        TaskProgressColumn(), TimeElapsedColumn(), console=console,
    ) as prog:
        task = prog.add_task("eval", total=remaining_gens * int(args.population_size))

        while state.generation < int(args.generations):
            wall_now = time.perf_counter() - segment_start
            stop_reason = _check_budget(state, args, wall_now)
            if stop_reason is not None:
                console.print(f"[bold yellow]stop[/bold yellow]: {stop_reason}")
                break

            gen = state.generation
            best_before = float(state.best["score"]) if state.best else float("inf")
            solutions: list[tuple[np.ndarray, float]] = []

            for cand in range(int(args.population_size)):
                x = opt.ask()
                decoded = _decode_x(x)

                seed_evals: list[CellEval] = []
                for eval_seed in args.eval_seeds:
                    ev = _train_eval(
                        artifacts_dir=artifacts_dir, search_run_id=run_id, gen=gen, cand=cand,
                        eval_seed=int(eval_seed), decoded=decoded, args=args,
                    )
                    seed_evals.append(ev)
                    state.eval_count += 1
                    if ev.status != "ok":
                        state.crash_count += 1

                agg_score = _aggregate_seed_scores(
                    seed_evals, how=args.seed_agg, lam=float(args.seed_agg_lambda)
                )
                solutions.append((x, agg_score))

                n_ok = sum(1 for e in seed_evals if e.status == "ok")
                cand_summary = next((e.train_summary_path for e in seed_evals if e.train_summary_path), None)
                writer.writerow({
                    "gen": gen, "cand": cand,
                    "eval_seeds": ";".join(str(s) for s in args.eval_seeds),
                    "n_ok": n_ok, "n_fail": n_seeds - n_ok,
                    "agg_score": agg_score,
                    "per_seed_scores": ";".join(f"{e.score:.6g}" for e in seed_evals),
                    "duration_s": sum(e.duration_s for e in seed_evals),
                    "best_summary_path": cand_summary,
                })
                csv_handle.flush()

                if state.best is None or agg_score < float(state.best["score"]):
                    state.best = {
                        "score": agg_score, "gen": gen, "cand": cand,
                        "x": [float(v) for v in x.tolist()], "decoded": decoded,
                        "eval_seeds": list(args.eval_seeds), "seed_agg": str(args.seed_agg),
                        "per_seed_scores": [e.score for e in seed_evals],
                        "train_summary_path": cand_summary,
                    }
                    _write_json(best_path, state.best)

                prog.advance(task)

            opt.tell(solutions)

            # Patience (resume-safe): compare this generation's best to the
            # all-time best as it stood BEFORE the generation. A meaningful
            # improvement resets the streak; otherwise it grows. Only persisted
            # fields are used, so resume reconstructs the streak exactly.
            gen_best = min((s for _, s in solutions), default=float("inf"))
            if gen_best < best_before - float(args.min_improve):
                state.no_improve_streak = 0
            else:
                state.no_improve_streak += 1
            state.generation = gen + 1

            # Checkpoint with cumulative wall time folded in so resume continues
            # the budget rather than restarting it.
            seg_elapsed = time.perf_counter() - segment_start
            chk = SearchState(
                generation=state.generation, eval_count=state.eval_count,
                crash_count=state.crash_count, wall_accum_s=state.wall_accum_s + seg_elapsed,
                no_improve_streak=state.no_improve_streak, best=state.best,
            )
            _save_checkpoint(state_dir, opt, chk)

    # Persist final accumulated wall time.
    state.wall_accum_s += time.perf_counter() - segment_start
    _save_checkpoint(state_dir, opt, state)

    if stop_reason is None and state.generation >= int(args.generations):
        stop_reason = f"completed all {args.generations} generations"

    csv_handle.close()
    _finalize(run_dir, run_id, args, state, stop_reason)
    console.print(f"[bold green]done[/bold green] ({stop_reason}) → {run_dir}")
    return 0


def _finalize(run_dir: Path, run_id: str, args: argparse.Namespace, state: SearchState, stop_reason: str | None) -> None:
    if not state.best:
        _write_text(run_dir / "summary.md", f"# CMA-ES Phase 1 ({run_id})\n\nNo successful candidates. Stop: {stop_reason}\n")
        return

    table = Table(title="CMA-ES Phase 1 best", show_header=True, header_style="bold")
    table.add_column("param")
    table.add_column("value", justify="right")
    for p in PARAM_SPECS:
        table.add_row(p.name, f"{float(state.best['decoded'][p.name]):.6g}")
    console.print(table)
    console.print(f"[bold green]best score[/bold green] = {float(state.best['score']):.6f}")

    crash_rate = (state.crash_count / state.eval_count) if state.eval_count else 0.0
    params_md = "\n".join(f"- `{p.name}` = `{float(state.best['decoded'][p.name]):.6g}`" for p in PARAM_SPECS)
    summary_md = f"""# CMA-ES Phase 1 — `{run_id}`

- Best score: `{float(state.best["score"]):.6f}` (gen `{state.best["gen"]}`, cand `{state.best["cand"]}`)
- Eval seeds: `{state.best.get("eval_seeds")}` · aggregation: `{state.best.get("seed_agg")}`
- Per-seed scores at best: `{state.best.get("per_seed_scores")}`
- Best train summary: `{state.best.get("train_summary_path")}`

## Budget usage

- Generations run: `{state.generation}` / `{args.generations}`
- Per-seed evals: `{state.eval_count}`  (crashes: `{state.crash_count}`, rate `{crash_rate:.0%}`)
- Wall time: `{state.wall_accum_s:.0f}s`
- No-improvement streak at stop: `{state.no_improve_streak}`
- **Stop reason**: {stop_reason}

## Best parameters

{params_md}

## Go / No-Go (Phase 2)

Go if the run completed reliably and best score improves meaningfully over baseline
within budget. No-Go if failures dominate or scores are flat/noisy.
"""
    _write_text(run_dir / "summary.md", summary_md)


if __name__ == "__main__":
    raise SystemExit(main())
