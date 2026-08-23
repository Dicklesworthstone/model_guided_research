#!/usr/bin/env python3
"""
Model Guided Research CLI - Run experimental mathematical models for ML research
"""

import importlib
import json
import math
import os
import platform
import shlex
import statistics
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

app = typer.Typer(
    name="model-guided-research",
    help="Run experimental mathematical models for machine learning research",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()

_ALLOWED_CMDS = {"git"}


def _run_command(cmd: str) -> str | None:
    try:
        parts = shlex.split(cmd)
        if not parts or parts[0] not in _ALLOWED_CMDS:
            return None
        result = subprocess.run(parts, shell=False, capture_output=True, text=True, timeout=5)  # nosec B603
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except Exception:
        return None


def _get_git_info() -> dict[str, Any]:
    commit = _run_command("git rev-parse --short HEAD") or "unknown"
    commit_full = _run_command("git rev-parse HEAD") or "unknown"
    branch = _run_command("git rev-parse --abbrev-ref HEAD") or "unknown"
    status = _run_command("git status --porcelain")
    dirty = bool(status) if status is not None else False
    return {"commit": commit, "commit_full": commit_full, "branch": branch, "dirty": dirty}


def _default_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _write_artifacts(run_dir: Path, *, summary: dict[str, Any], report_md: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "run.md").write_text(report_md, encoding="utf-8")


def _sparkline(values: list[float], *, width: int = 20) -> str:
    bars = "▁▂▃▄▅▆▇█"
    if not values or width <= 0:
        return ""
    if len(values) > width:
        # Take the tail: most useful for training loss curves.
        values = values[-width:]
    lo = min(values)
    hi = max(values)
    if not math.isfinite(lo) or not math.isfinite(hi):
        return ""
    if hi - lo < 1e-12:
        return bars[0] * len(values)
    idxs = [int((v - lo) / (hi - lo) * (len(bars) - 1)) for v in values]
    return "".join(bars[i] for i in idxs)


_T_CRIT_975: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _summary_stats(values: list[float]) -> dict[str, Any]:
    vals = [
        float(v) for v in values if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(float(v))
    ]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": None}
    mean = float(statistics.fmean(vals))
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci95": None}
    var = float(sum((x - mean) ** 2 for x in vals) / (n - 1))
    std = math.sqrt(var)
    t = _T_CRIT_975.get(n - 1, 1.96)
    ci = float(t * std / math.sqrt(n))
    return {"n": n, "mean": mean, "std": std, "ci95": ci}


def _aggregate_per_head(samples: list[tuple[int, list[float]]]) -> dict[str, Any] | None:
    valid = [
        (seed, vec)
        for seed, vec in samples
        if isinstance(vec, list) and vec and all(isinstance(x, int | float) and not isinstance(x, bool) for x in vec)
    ]
    if not valid:
        return None
    n_head = min(len(vec) for _, vec in valid)
    means: list[float] = []
    stds: list[float] = []
    ci95s: list[float | None] = []
    ns: list[int] = []
    for i in range(n_head):
        vals = [float(vec[i]) for _, vec in valid if i < len(vec) and math.isfinite(float(vec[i]))]
        stats = _summary_stats(vals)
        ns.append(int(stats["n"]))
        means.append(float(stats["mean"]) if stats["mean"] is not None else float("nan"))
        stds.append(float(stats["std"]) if stats["std"] is not None else float("nan"))
        ci95s.append(float(stats["ci95"]) if stats["ci95"] is not None else None)
    return {
        "n_head": n_head,
        "seeds": [int(seed) for seed, _ in valid],
        "samples": {str(seed): vec[:n_head] for seed, vec in valid},
        "n": ns,
        "mean": means,
        "std": stds,
        "ci95": ci95s,
    }


def _resolve_summary_path(path: Path, *, artifacts_dir: Path) -> Path:
    candidates: list[Path] = [path]
    if not path.exists():
        candidates.append(artifacts_dir / path)
    for cand in candidates:
        if cand.is_dir():
            summary_path = cand / "summary.json"
            if summary_path.is_file():
                return summary_path
        if cand.is_file():
            return cand
    raise typer.BadParameter(f"Could not find summary.json for {path} (tried: {', '.join(str(c) for c in candidates)})")


def _get_nested(obj: dict[str, Any], keys: tuple[str, ...]) -> Any:
    cur: Any = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _as_float(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    return None


def _extract_metric(summary: dict[str, Any], *, metric: str, variant: str | None) -> float | None:
    """Extract a canonical metric from heterogeneous summary.json shapes."""
    # Suite summaries: choose by attention_type.
    if isinstance(summary.get("runs"), list) and variant:
        for run in summary["runs"]:
            if isinstance(run, dict) and run.get("attention_type") == variant:
                if metric == "final_loss":
                    return _as_float(run.get("final_loss")) or _as_float(run.get("score"))
                if metric == "tokens_per_second":
                    return _as_float(run.get("tokens_per_second")) or _as_float(run.get("tokens_per_s"))
                if metric == "tflops_per_second_est":
                    return _as_float(run.get("tflops_per_second_est"))
                if metric == "peak_memory_allocated_gb":
                    return _as_float(run.get("peak_memory_allocated_gb"))

    results = summary.get("results")
    if not isinstance(results, dict):
        results = {}

    if metric == "final_loss":
        losses = results.get("losses")
        if isinstance(losses, list) and losses:
            vals = [_as_float(v) for v in losses]
            vals2 = [v for v in vals if v is not None]
            return vals2[-1] if vals2 else None
        for k in ("final_loss", "loss", "score"):
            v = _as_float(results.get(k))
            if v is not None:
                return v
        return _as_float(summary.get("final_loss")) or _as_float(summary.get("score"))

    if metric == "tokens_per_second":
        v = results.get("tokens_per_second")
        if isinstance(v, dict):
            return _as_float(v.get(variant)) if variant else None
        v2 = _as_float(v)
        if v2 is not None:
            return v2
        v = results.get("tokens_per_s")
        if isinstance(v, dict):
            return _as_float(v.get(variant)) if variant else None
        return _as_float(v) or _as_float(summary.get("tokens_per_second")) or _as_float(summary.get("tokens_per_s"))

    if metric == "tflops_per_second_est":
        v = results.get("tflops_per_second_est")
        if isinstance(v, dict):
            return _as_float(v.get(variant)) if variant else None
        return _as_float(v) or _as_float(summary.get("tflops_per_second_est"))

    if metric == "peak_memory_allocated_gb":
        v = results.get("peak_memory_allocated_gb")
        if isinstance(v, dict):
            return _as_float(v.get(variant)) if variant else None
        v2 = _as_float(v)
        if v2 is not None:
            return v2
        peak_mb = results.get("peak_mem_mb")
        if isinstance(peak_mb, dict):
            mb = _as_float(peak_mb.get(variant)) if variant else None
            return (mb / 1024.0) if mb is not None else None
        mb = _as_float(peak_mb)
        return (mb / 1024.0) if mb is not None else None

    raise ValueError(f"Unknown metric {metric!r}")


def _extract_loss_series(summary: dict[str, Any]) -> list[float]:
    results = summary.get("results")
    if not isinstance(results, dict):
        return []
    losses = results.get("losses")
    if not isinstance(losses, list):
        return []
    out: list[float] = []
    for v in losses:
        fv = _as_float(v)
        if fv is not None:
            out.append(fv)
    return out


def _certify_comparisons(
    baseline_obj: dict[str, Any],
    candidate_obj: dict[str, Any],
    *,
    degrade_factor: float,
) -> list[dict[str, Any]]:
    """Build regression rows from two `mgr certify` summaries (bead 5ki.3).

    A certificate check regresses when it flips green -> red (pass ->
    fail/error) or when a still-passing measured invariant degrades by more
    than `degrade_factor` relative to the baseline snapshot (e.g. reversible
    round-trip error growing 10x while still under tolerance). Direction
    follows the check's comparator: 'le' tolerances degrade upward, 'ge'
    floors degrade downward. Tightening-only policy: loosening a tolerance in
    cli.py requires an explicit justification note in the bead/commit.
    """
    base_checks: dict[tuple[str, str], dict[str, Any]] = {
        (str(c.get("mechanism")), str(c.get("check"))): c
        for c in baseline_obj.get("checks", [])
        if isinstance(c, dict)
    }
    cand_checks: dict[tuple[str, str], dict[str, Any]] = {
        (str(c.get("mechanism")), str(c.get("check"))): c
        for c in candidate_obj.get("checks", [])
        if isinstance(c, dict)
    }
    keys = sorted(set(base_checks) | set(cand_checks))

    comparisons: list[dict[str, Any]] = []
    for mechanism, check_name in keys:
        key = f"cert:{mechanism}.{check_name}"
        b = base_checks.get((mechanism, check_name))
        c = cand_checks.get((mechanism, check_name))
        if b is None or c is None:
            comparisons.append(
                {
                    "metric": key,
                    "label": key,
                    "direction": "lower",
                    "baseline": _as_float(b.get("measured")) if b else None,
                    "candidate": _as_float(c.get("measured")) if c else None,
                    "delta": None,
                    "delta_rel": None,
                    "status": "missing",
                }
            )
            continue

        comparator = str(b.get("comparator") or c.get("comparator") or "le")
        direction = "lower" if comparator == "le" else "higher"
        b_val = _as_float(b.get("measured"))
        c_val = _as_float(c.get("measured"))
        b_pass = b.get("status") == "pass"
        c_pass = c.get("status") == "pass"
        delta = float(c_val - b_val) if (b_val is not None and c_val is not None) else None
        delta_rel = float(delta / b_val) if (delta is not None and b_val) else None
        status = "ok"

        if b_pass and not c_pass:
            status = "regression"
        elif b_pass and c_pass and b_val is not None and c_val is not None and b_val > 0:
            worsened = c_val > b_val if comparator == "le" else c_val < b_val
            ratio = (c_val / b_val) if comparator == "le" else (b_val / c_val if c_val else None)
            if worsened and ratio is not None and ratio > degrade_factor:
                status = "regression"

        comparisons.append(
            {
                "metric": f"cert:{mechanism}.{check_name}",
                "label": key,
                "direction": direction,
                "baseline": b_val,
                "candidate": c_val,
                "delta": delta,
                "delta_rel": delta_rel,
                "status": status,
            }
        )
    return comparisons


def _summarize_provenance(summary: dict[str, Any]) -> dict[str, Any]:
    meta = summary.get("meta") if isinstance(summary.get("meta"), dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    git = summary.get("git") if isinstance(summary.get("git"), dict) else meta.get("git", {})
    if not isinstance(git, dict):
        git = {}
    cfg = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    if not isinstance(cfg, dict):
        cfg = {}
    return {
        "run_id": summary.get("run_id") or meta.get("run_id"),
        "kind": meta.get("kind") or summary.get("kind"),
        "device": meta.get("device") or summary.get("device"),
        "commit": git.get("commit") or git.get("commit_full"),
        "dirty": git.get("dirty"),
        "attention_type": cfg.get("attention_type"),
        "use_flex_attention": cfg.get("use_flex_attention"),
        "compile": _get_nested(summary, ("compile", "enabled")) or cfg.get("compile"),
        "command": meta.get("command") or summary.get("command"),
    }


# Map of available demos
DEMOS = {
    "ifs-fractal": {
        "module": "iterated_function_systems_and_fractal_memory",
        "description": "Iterated Function Systems and Fractal Memory structures",
        "func": "demo",
    },
    "clifford": {
        "module": "clifford_algebra_and_geometric_attention",
        "description": "Clifford algebra (Cl(3,0)) geometric products, rotors, and equivariance",
        "func": "demo",
    },
    "knot-braid": {
        "module": "knot_theoretic_programs_and_braid_based_attention",
        "description": "Knot-theoretic programs and braid-based attention mechanisms",
        "func": "demo",
    },
    "matrix-gauge": {
        "module": "matrix_exponential_gauge_learning",
        "description": "Matrix exponential gauge learning with Lie groups",
        "func": "demo",
    },
    "nonstandard": {
        "module": "nonstandard_analysis_and_hyperreal_training",
        "description": "Nonstandard analysis and hyperreal training methods",
        "func": "demo",
    },
    "octonion": {
        "module": "octonionic_quaternionic_signal_flow",
        "description": "Octonionic and quaternionic signal flow processing",
        "func": "demo",
    },
    "ordinal": {
        "module": "ordinal_schedules_and_well_founded_optimization",
        "description": "Ordinal schedules and well-founded optimization",
        "func": "demo",
    },
    "reversible": {
        "module": "reversible_computation_and_measure_preserving_learning",
        "description": "Reversible computation and measure-preserving learning",
        "func": "demo",
    },
    "simplicial": {
        "module": "simplicial_complexes_and_higher_order_attention",
        "description": "Simplicial complexes and higher-order attention",
        "func": "demo",
    },
    "surreal": {
        "module": "surreal_numbers_transseries_and_scaling",
        "description": "Surreal numbers, transseries and scaling methods",
        "func": "demo",
    },
    "tropical": {
        "module": "tropical_geometry_and_idempotent_algebra",
        "description": "Tropical geometry and idempotent algebra",
        "func": "demo",
    },
    "ultrametric": {
        "module": "ultrametric_worlds_and_p_adic_computation",
        "description": "Ultrametric worlds and p-adic computation",
        "func": "demo",
    },
}


@app.command("list")
def list_demos():
    """List all available demos with descriptions"""
    table = Table(
        title="[bold cyan]Available Model Demos[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Demo Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Module", style="dim white")

    for name, info in DEMOS.items():
        table.add_row(name, info["description"], info["module"])

    console.print(table)
    console.print("\n[dim]Run a demo with:[/dim] [bold green]mgr run <demo-name>[/bold green]")
    console.print("[dim]Get info about a demo:[/dim] [bold green]mgr info <demo-name>[/bold green]")


@app.command()
def run(
    demo_name: Annotated[
        str,
        typer.Argument(
            help="Name of the demo to run",
            autocompletion=lambda: DEMOS.keys(),  # type: ignore[call-arg]
        ),
    ],
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to JSON config file (see config.example.json)")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show verbose output")] = False,
    verbose_level: Annotated[
        int | None,
        typer.Option("--verbose-level", min=0, max=3, help="Verbosity level: 0=silent, 1=normal, 2=detailed, 3=debug"),
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", "-s", help="Random seed for reproducibility")] = None,
    max_iterations: Annotated[
        int | None,
        typer.Option(
            "--max-iterations", min=1, help="Override ProjectConfig.max_iterations (for demos that respect it)"
        ),
    ] = None,
    no_rich: Annotated[bool, typer.Option("--no-rich", help="Disable rich formatting for plain text output")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Enable debug mode with numerical checking")] = False,
    ultra_packed: Annotated[
        bool,
        typer.Option(
            "--ultra-packed",
            help="Use packed bit-trie implementation in ultrametric demo (and set ULTRA_PACKED for tests)",
        ),
    ] = False,
    tropical_cert: Annotated[
        bool, typer.Option("--tropical-cert", help="Compute a tropical attention robustness margin certificate")
    ] = False,
    simplicial_hodge: Annotated[
        bool, typer.Option("--simplicial-hodge", help="Demonstrate Hodge-based readout coefficients on a tiny graph")
    ] = False,
    simplicial_signed: Annotated[
        bool, typer.Option("--simplicial-signed", help="Demonstrate signed (orientation-aware) diffusion vs unsigned")
    ] = False,
    rev_cayley: Annotated[
        bool, typer.Option("--rev-cayley", help="Demonstrate Cayley orthogonal property check (skew → orthogonal)")
    ] = False,
    rev_cayley_o1: Annotated[
        bool,
        typer.Option(
            "--rev-cayley-o1/--no-rev-cayley-o1", help="Use O(1)-memory custom gradient for Cayley step (default on)"
        ),
    ] = True,
    rev_cayley_iters: Annotated[
        int,
        typer.Option("--rev-cayley-iters", help="Cayley fixed-point iterations (trade compute for accuracy)", min=1),
    ] = 1,
    rev_symplectic: Annotated[
        bool, typer.Option("--rev-symplectic", help="Demonstrate symplectic Cayley property check (S^T J S ≈ J)")
    ] = False,
    rev_inv_iters: Annotated[
        int, typer.Option("--rev-inv-iters", help="Inverse fixed-point iteration count for Cayley inverse", min=1)
    ] = 1,
    rev_pareto: Annotated[
        bool, typer.Option("--rev-pareto", help="Run a small Cayley-iterations Pareto sweep (time vs memory)")
    ] = False,
    rev_symp_hybrid: Annotated[
        bool, typer.Option("--rev-symplectic-hybrid", help="Enable a symplectic leapfrog step inside coupling (hybrid)")
    ] = False,
    rev_givens: Annotated[
        bool, typer.Option("--rev-givens", help="Use strict Givens mixing (exact inverse; det=1)")
    ] = False,
    rev_generating: Annotated[
        bool, typer.Option("--rev-generating", help="Enable generating-function symplectic step (exact inverse)")
    ] = False,
    rev_gen_vjp: Annotated[
        bool, typer.Option("--rev-gen-vjp", help="Use custom VJP for generating step (O(1) grads; ignores ∂/∂(a,b,c))")
    ] = False,
    gauge_structured: Annotated[
        bool, typer.Option("--gauge-structured", help="Enable structured SO/SPD/Sp channel blocks in matrix-gauge demo")
    ] = False,
    gauge_bch_compact: Annotated[
        bool, typer.Option("--gauge-bch-compact", help="Print only compact BCH summary table (skip heatmap)")
    ] = False,
    gauge_alt_struct: Annotated[
        bool,
        typer.Option("--gauge-alt-struct", help="Alternate structured/unstructured on odd blocks in matrix-gauge demo"),
    ] = False,
    export_json: Annotated[
        Path | None, typer.Option("--export-json", help="Write a JSON artifact with any computed certificates/readouts")
    ] = None,
    artifacts_dir: Annotated[
        Path | None,
        typer.Option(
            "--artifacts-dir",
            help="Write run artifacts under this directory using the standard layout (see artifacts/README.md).",
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id", help="Run identifier (directory name) when writing artifacts. Defaults to YYYYMMDD_HHMMSS."
        ),
    ] = None,
):
    """Run a specific demo by name"""

    # Configure settings
    from config import ProjectConfig, set_config
    from utils import seed_everything

    # Load config from file if provided
    if config_file and config_file.exists():
        config = ProjectConfig.from_file(config_file)
        if verbose:
            console.print(f"[dim]Loaded config from {config_file}[/dim]")
    else:
        config = ProjectConfig()

    # Override with command-line arguments
    if verbose:
        config.verbose = True
    if verbose_level is not None:
        config.verbose_level = verbose_level
    if seed is not None:
        config.random_seed = seed
    if max_iterations is not None:
        config.max_iterations = max_iterations
    if no_rich:
        config.use_rich_output = False
    if debug:
        config.debug_mode = True
        config.check_numerics = True
        config.jax_debug_nans = True
        config.jax_debug_infs = True

    # Set the global config
    set_config(config)
    seed_everything(config.random_seed)

    # Optional environment knobs for tests/internals
    if ultra_packed:
        import os as _os

        _os.environ["ULTRA_PACKED"] = "1"
    if gauge_structured:
        import os as _os

        _os.environ["GAUGE_STRUCTURED"] = "1"
    if gauge_bch_compact:
        import os as _os

        _os.environ["GAUGE_BCH_COMPACT"] = "1"
    if gauge_alt_struct:
        import os as _os

        _os.environ["GAUGE_ALT_STRUCT"] = "1"
    if rev_givens:
        import os as _os

        _os.environ["REV_GIVENS"] = "1"
    if rev_generating:
        import os as _os

        _os.environ["REV_GENERATING"] = "1"
    if rev_gen_vjp:
        import os as _os

        _os.environ["REV_GEN_VJP"] = "1"

    if demo_name not in DEMOS:
        console.print(f"[bold red]Error:[/bold red] Demo '{demo_name}' not found")
        console.print("\nAvailable demos:")
        for name in DEMOS:
            console.print(f"  • {name}")
        raise typer.Exit(1)

    demo_info = DEMOS[demo_name]

    # Display what we're running
    panel = Panel(
        f"[bold cyan]{demo_info['description']}[/bold cyan]\n[dim]Module: {demo_info['module']}.py[/dim]",
        title=f"Running Demo: {demo_name}",
        box=box.ROUNDED,
    )
    console.print(panel)
    console.print()

    try:
        artifacts: dict = {"demo": demo_name, "certificates": {}}
        # Import the module dynamically
        if verbose:
            console.print(f"[dim]Importing module: {demo_info['module']}[/dim]")

        module = importlib.import_module(demo_info["module"])

        # Get the demo function
        func_name = demo_info["func"]
        if hasattr(module, func_name):
            demo_func = getattr(module, func_name)

            if verbose:
                console.print(f"[dim]Running function: {func_name}()[/dim]\n")

            # Pre-demo feature showcases
            if demo_name == "tropical" and tropical_cert:
                import numpy as _np

                from tropical_geometry_and_idempotent_algebra import TropicalAttention

                Q_np = _np.random.randn(32, 16)
                K_np = _np.random.randn(32, 16)
                V_np = _np.random.randn(32, 16)
                attn = TropicalAttention(16)
                _ = attn(Q_np, K_np, V_np)
                table = Table(title="Tropical Robustness Certificate", show_header=True, header_style="bold magenta")
                table.add_column("Min (best−second) margin", justify="center")
                margin = float(getattr(attn, "last_min_margin", 0.0))
                table.add_row(f"{margin:.4f}")
                console.print(table)
                artifacts["certificates"]["tropical_min_margin"] = margin
                # Toggle ASCII summary of K if matrix-gauge demo is run too
                # (No-op here; matrix-gauge prints uniformization K when demo runs.)

            if demo_name == "simplicial" and simplicial_hodge:
                import numpy as _np

                from simplicial_complexes_and_higher_order_attention import hodge_readout

                n = 8
                A = _np.zeros((n, n))
                for _ in range(12):
                    i, j = _np.random.randint(0, n, 2)
                    if i != j:
                        A[i, j] = A[j, i] = 1
                flow = _np.random.randn(n)
                coeff = hodge_readout(flow, A, k_small=3)
                t = Table(title="Hodge Readout Coefficients (k=3)", show_header=True, header_style="bold magenta")
                t.add_column("Mode", justify="center")
                t.add_column("Coeff", justify="right")
                for i, c in enumerate(coeff):
                    t.add_row(str(i), f"{float(c):.4f}")
                console.print(t)
                artifacts["certificates"]["simplicial_hodge_coeffs"] = [float(c) for c in coeff]

            if (demo_name == "reversible") and (
                rev_cayley or rev_symplectic or rev_pareto or rev_symp_hybrid or (rev_inv_iters != 1)
            ):
                import numpy as _np

                from matrix_exponential_gauge_learning import cayley_orthogonal_from_skew, symplectic_cayley

                # Cayley orthogonal check
                if rev_cayley:
                    try:
                        from reversible_computation_and_measure_preserving_learning import (
                            set_reversible_cayley,
                            set_reversible_cayley_iters,
                            set_reversible_cayley_o1,
                        )

                        set_reversible_cayley(True)
                        set_reversible_cayley_o1(bool(rev_cayley_o1))
                        set_reversible_cayley_iters(int(rev_cayley_iters))
                        import os as _os

                        _os.environ["REV_LAYER_CERT"] = "1"
                    except Exception:
                        pass
                    M = _np.random.randn(16, 16)
                    A = 0.1 * (M - M.T)  # skew
                    import jax.numpy as _jnp

                    Q = cayley_orthogonal_from_skew(_jnp.array(A))
                    eye_q = _jnp.eye(Q.shape[-1])
                    err = float(_jnp.linalg.norm(Q.T @ Q - eye_q))
                    table = Table(title="Cayley Orthogonality Check", show_header=True, header_style="bold magenta")
                    table.add_column("||Q^T Q − I||_F", justify="right")
                    table.add_row(f"{err:.2e}")
                    console.print(table)
                    artifacts["certificates"]["reversible_cayley_orth_err"] = err
                # Symplectic check
                if rev_symplectic:
                    n = 8
                    H = _np.random.randn(2 * n, 2 * n)
                    H = 0.1 * (H + H.T)
                    import jax.numpy as _jnp

                    S = symplectic_cayley(_jnp.array(H))
                    Z = _jnp.zeros((n, n))
                    eye_n = _jnp.eye(n)
                    J = _jnp.block([[Z, eye_n], [-eye_n, Z]])
                    err = float(_jnp.linalg.norm(S.T @ J @ S - J))
                    t2 = Table(title="Symplectic Cayley Check", show_header=True, header_style="bold magenta")
                    t2.add_column("||S^T J S − J||_F", justify="right")
                    t2.add_row(f"{err:.2e}")
                    console.print(t2)
                    artifacts["certificates"]["reversible_symplectic_err"] = err
                if rev_pareto:
                    import os as _os

                    _os.environ["REV_PARETO"] = "1"
                if rev_inv_iters and rev_inv_iters != 1:
                    try:
                        import os as _os

                        _os.environ["REV_INV_ITERS"] = str(int(rev_inv_iters))
                    except Exception:
                        pass
                if rev_symp_hybrid:
                    try:
                        from reversible_computation_and_measure_preserving_learning import set_reversible_symplectic

                        set_reversible_symplectic(True)
                    except Exception:
                        pass

            # Run the demo
            with console.status("[bold green]Running demo...[/bold green]"):
                demo_func()

            # Collect module-level diagnostics if present
            try:
                diag = getattr(module, "last_diagnostics", None)
                if diag is not None:
                    artifacts.setdefault("diagnostics", {})[demo_name] = diag
            except Exception:
                pass

        # Write artifacts if requested (legacy path)
        if export_json is not None:
            try:
                export_json.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            with export_json.open("w", encoding="utf-8") as f:
                json.dump(artifacts, f, indent=2)
            if verbose:
                console.print(f"[dim]Wrote JSON artifact to {export_json}[/dim]")

        # Write artifacts using the unified artifacts layout
        if artifacts_dir is not None:
            resolved_run_id = run_id or _default_run_id()
            run_dir = artifacts_dir / "certs" / "demos" / demo_name / resolved_run_id

            meta = {
                "demo": demo_name,
                "run_id": resolved_run_id,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "git": _get_git_info(),
                "python": {
                    "executable": sys.executable,
                    "version": platform.python_version(),
                },
                "argv": sys.argv,
                "seed": int(config.random_seed),
                "config_file": str(config_file) if config_file is not None else None,
                "flags": {
                    "max_iterations": int(config.max_iterations),
                    "ultra_packed": bool(ultra_packed),
                    "tropical_cert": bool(tropical_cert),
                    "simplicial_hodge": bool(simplicial_hodge),
                    "rev_givens": bool(rev_givens),
                    "rev_generating": bool(rev_generating),
                    "rev_gen_vjp": bool(rev_gen_vjp),
                    "rev_pareto": bool(rev_pareto),
                    "rev_inv_iters": rev_inv_iters,
                    "rev_symp_hybrid": bool(rev_symp_hybrid),
                    "gauge_structured": bool(gauge_structured),
                    "gauge_bch_compact": bool(gauge_bch_compact),
                    "gauge_alt_struct": bool(gauge_alt_struct),
                },
            }

            summary = {"meta": meta, "artifacts": artifacts}
            report_md = f"""# Demo certificate run: `{demo_name}`

- Run ID: `{resolved_run_id}`
- Generated: {meta["generated_at"]}
- Commit: {meta["git"]["commit_full"]}{" (dirty)" if meta["git"]["dirty"] else " (clean)"}

## Command

```bash
{shlex.join(sys.argv)}
```

## Certificates

This run writes `summary.json` (machine-readable) and `run.md` (human-readable) under:

`{run_dir}`
"""
            _write_artifacts(run_dir, summary=summary, report_md=report_md)
            console.print(f"[dim]Wrote artifacts → {run_dir}[/dim]")

    except ImportError as e:
        console.print(f"[bold red]Import Error:[/bold red] {e}")
        console.print("\n[dim]Make sure all dependencies are installed:[/dim]")
        console.print("[bold]uv sync --extra dev[/bold]")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
        raise typer.Exit(0) from None
    except Exception as e:
        console.print(f"[bold red]Error running demo:[/bold red] {e}")
        if verbose:
            import traceback

            console.print("[dim]Traceback:[/dim]")
            traceback.print_exc()
        raise typer.Exit(1) from e


@app.command()
def info(
    demo_name: str = typer.Argument(
        ...,
        help="Name of the demo to get info about",
        autocompletion=lambda: DEMOS.keys(),  # type: ignore[call-arg]
    ),
):
    """Show detailed information about a specific demo"""

    if demo_name not in DEMOS:
        console.print(f"[bold red]Error:[/bold red] Demo '{demo_name}' not found")
        console.print("\nAvailable demos:")
        for name in DEMOS:
            console.print(f"  • {name}")
        raise typer.Exit(1)

    demo_info = DEMOS[demo_name]
    module_file = Path(f"{demo_info['module']}.py")

    # Display demo information
    panel = Panel(
        f"[bold cyan]{demo_info['description']}[/bold cyan]\n\n"
        f"[bold]Module:[/bold] {demo_info['module']}.py\n"
        f"[bold]Function:[/bold] {demo_info['func']}()\n"
        f"[bold]File exists:[/bold] {'✓' if module_file.exists() else '✗'}",
        title=f"Demo: {demo_name}",
        box=box.ROUNDED,
    )
    console.print(panel)

    # Try to extract and display the module docstring
    if module_file.exists():
        try:
            with open(module_file, encoding="utf-8") as f:
                lines = f.readlines()

            # Find module docstring
            in_docstring = False
            docstring_lines = []
            for _i, line in enumerate(lines[:50]):  # Check first 50 lines
                if '"""' in line:
                    if not in_docstring:
                        in_docstring = True
                        # Check if it's a one-liner
                        if line.count('"""') == 2:
                            docstring_lines.append(line.strip().replace('"""', ""))
                            break
                    else:
                        in_docstring = False
                        break
                elif in_docstring:
                    docstring_lines.append(line.rstrip())

            if docstring_lines:
                console.print("\n[bold]Module Documentation:[/bold]")
                console.print(
                    Panel(
                        "\n".join(docstring_lines),
                        box=box.ROUNDED,
                        padding=(1, 2),
                    )
                )

        except Exception as e:
            if str(e):  # Only show error if it has a message
                console.print(f"[dim]Could not read module documentation: {e}[/dim]")

    console.print(f"\n[dim]Run this demo with:[/dim] [bold green]mgr run {demo_name}[/bold green]")


@app.command()
def config(
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output path for config file")] = None,
    show: Annotated[bool, typer.Option("--show", help="Show current configuration")] = False,
):
    """Generate example config file or show current configuration"""

    import json

    from config import get_config

    if show:
        # Show current configuration
        current = get_config()
        console.print("[bold cyan]Current Configuration:[/bold cyan]\n")

        config_dict = {}
        for field in current.__dataclass_fields__:
            value = getattr(current, field)
            if isinstance(value, Path):
                value = str(value)
            config_dict[field] = value

        console.print(json.dumps(config_dict, indent=2))
        return

    # Generate example config
    output_path = output or Path("config.json")

    if output_path.exists():
        if not typer.confirm(f"File {output_path} exists. Overwrite?"):
            raise typer.Exit(0)

    example_config = {
        "use_gpu": False,
        "jax_precision": "float32",
        "random_seed": 42,
        "jax_debug_nans": False,
        "jax_debug_infs": False,
        "jax_disable_jit": False,
        "verbose": True,
        "verbose_level": 1,
        "save_outputs": False,
        "output_dir": "outputs",
        "save_checkpoints": False,
        "checkpoint_dir": "checkpoints",
        "log_metrics": True,
        "log_interval": 100,
        "use_rich_output": True,
        "show_progress_bars": True,
        "debug_mode": False,
        "check_numerics": False,
        "profile_performance": False,
        "max_iterations": 1000,
        "convergence_threshold": 1e-6,
        "early_stopping_patience": 10,
        "default_learning_rate": 0.001,
        "default_batch_size": 32,
        "gradient_clip_norm": None,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(example_config, f, indent=2)

    console.print(f"[green]✓ Example config written to {output_path}[/green]")
    console.print(f"\n[dim]Use it with:[/dim] [bold]mgr run <demo> --config {output_path}[/bold]")


@app.command()
def run_all(
    delay: Annotated[int, typer.Option("--delay", "-d", help="Delay in seconds between demos")] = 2,
    seed: Annotated[int | None, typer.Option("--seed", "-s", help="Random seed for reproducibility")] = None,
    skip_errors: Annotated[
        bool, typer.Option("--skip-errors/--stop-on-error", help="Continue running demos even if one fails")
    ] = True,
):
    """Run all available demos in sequence"""

    from config import ProjectConfig, set_config
    from utils import seed_everything

    config = ProjectConfig()
    if seed is not None:
        config.random_seed = seed
    set_config(config)
    seed_everything(config.random_seed)

    console.print("[bold cyan]Running all demos...[/bold cyan]\n")

    success_count = 0
    error_count = 0

    for i, (name, info) in enumerate(DEMOS.items(), 1):
        console.rule(f"[bold]Demo {i}/{len(DEMOS)}: {name}[/bold]")

        try:
            # Import and run the demo
            module = importlib.import_module(info["module"])
            func_name = info["func"]

            if hasattr(module, func_name):
                console.print(f"[cyan]{info['description']}[/cyan]\n")

                demo_func = getattr(module, func_name)
                demo_func()

                success_count += 1
                console.print(f"\n[green]✓ Demo '{name}' completed successfully[/green]")
            else:
                raise AttributeError(f"Function '{func_name}' not found")

        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped by user[/yellow]")
            break
        except Exception as e:
            error_count += 1
            console.print(f"\n[red]✗ Demo '{name}' failed: {e}[/red]")

            if not skip_errors:
                console.print("[red]Stopping due to error (use --skip-errors to continue)[/red]")
                break

        # Add delay between demos (except after the last one)
        if i < len(DEMOS) and delay > 0:
            import time

            console.print(f"\n[dim]Waiting {delay} seconds before next demo...[/dim]")
            time.sleep(delay)

    # Summary
    console.rule("[bold]Summary[/bold]")
    console.print(f"[green]Successful:[/green] {success_count}")
    console.print(f"[red]Failed:[/red] {error_count}")
    console.print(f"[dim]Total:[/dim] {len(DEMOS)}")


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-V", help="Show version information"),
):
    """
    Model Guided Research CLI - Run experimental mathematical models for ML research

    This CLI provides easy access to various experimental mathematical models
    and algorithms for machine learning research, including fractal memories,
    knot-theoretic attention, gauge learning, and more.
    """
    if version:
        console.print("[bold]Model Guided Research[/bold] v0.1.0")
        raise typer.Exit()


@app.command("eval")
def evaluate(
    ultra_packed: Annotated[
        bool,
        typer.Option(
            "--ultra-packed", help="Use packed bit-trie implementation for ultrametric tests (sets ULTRA_PACKED=1)"
        ),
    ] = False,
    seed: Annotated[int | None, typer.Option("--seed", "-s", help="Random seed for reproducibility")] = None,
    export_json: Annotated[
        Path | None, typer.Option("--export-json", help="Write a combined JSON artifact of the practical utility suite")
    ] = None,
    artifacts_dir: Annotated[
        Path | None,
        typer.Option(
            "--artifacts-dir",
            help="Write run artifacts under this directory using the standard layout (see artifacts/README.md).",
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id", help="Run identifier (directory name) when writing artifacts. Defaults to YYYYMMDD_HHMMSS."
        ),
    ] = None,
    print_ultra_table: Annotated[
        bool, typer.Option("--print-ultra-table", help="Print ultrametric exponent table")
    ] = False,
    print_trop_table: Annotated[
        bool, typer.Option("--print-trop-table", help="Print tropical Lipschitz table")
    ] = False,
):
    """Run the practical utility test suite and optionally export a JSON artifact."""
    from config import ProjectConfig, set_config
    from utils import seed_everything

    config = ProjectConfig()
    if seed is not None:
        config.random_seed = seed
    set_config(config)
    seed_everything(config.random_seed)

    import os as _os

    if ultra_packed:
        _os.environ["ULTRA_PACKED"] = "1"
    if print_ultra_table:
        _os.environ["PRINT_ULTRA_TABLE"] = "1"
    if print_trop_table:
        _os.environ["PRINT_TROP_TABLE"] = "1"

    from tests.test_practical_utility import run_all_utility_tests

    results = run_all_utility_tests()

    if export_json is not None:
        payload = []
        for r in results:
            payload.append(
                {
                    "approach": r.approach_name,
                    "claim": r.claim,
                    "baseline": float(r.baseline_metric),
                    "proposed": float(r.proposed_metric),
                    "improvement": float(r.improvement_ratio),
                    "is_better": bool(r.is_better),
                    "verdict": r.verdict,
                    "details": r.details,
                }
            )
        try:
            export_json.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        with export_json.open("w", encoding="utf-8") as f:
            json.dump({"results": payload}, f, indent=2)
        console.print(f"[dim]Wrote suite JSON to {export_json}[/dim]")

    if artifacts_dir is not None:
        resolved_run_id = run_id or _default_run_id()
        run_dir = artifacts_dir / "bench" / "practical_utility" / resolved_run_id

        summary = {
            "meta": {
                "suite": "practical_utility",
                "run_id": resolved_run_id,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "git": _get_git_info(),
                "python": {
                    "executable": sys.executable,
                    "version": platform.python_version(),
                },
                "argv": sys.argv,
                "seed": int(config.random_seed),
                "flags": {
                    "ultra_packed": bool(ultra_packed),
                    "print_ultra_table": bool(print_ultra_table),
                    "print_trop_table": bool(print_trop_table),
                },
            },
            "results": payload,
        }

        report_md = f"""# Practical Utility Suite

- Run ID: `{resolved_run_id}`
- Generated: {summary["meta"]["generated_at"]}
- Commit: {summary["meta"]["git"]["commit_full"]}{" (dirty)" if summary["meta"]["git"]["dirty"] else " (clean)"}

## Command

```bash
{shlex.join(sys.argv)}
```

See `summary.json` for full details.
"""
        _write_artifacts(run_dir, summary=summary, report_md=report_md)
        console.print(f"[dim]Wrote artifacts → {run_dir}[/dim]")


def _bench_welch_delta(cand: list[float], base: list[float]) -> dict[str, Any] | None:
    """Two-sample Welch t-test of cand vs base (unequal variances) — the
    standard A/B significance test for the feature-ablation suite (bead z7r).

    Returns the mean difference cand-base, its 95% CI (Welch t interval with
    Satterthwaite df), the t statistic, the two-sided p-value, and df. None
    if either arm has < 2 observations (variance undefined at n=1).

    This is a DESCRIPTIVE benchmark comparison, deliberately separate from the
    preregistered mgr adjudicate engine (which tests a registered threshold
    with floor/power gates): here H0 is simply 'equal means', the textbook
    feature-vs-vanilla question. Zero-variance arms are handled explicitly
    rather than trusting scipy's nan."""
    import statistics as st

    if len(cand) < 2 or len(base) < 2:
        return None
    mc, mb = st.mean(cand), st.mean(base)
    vc, vb = st.variance(cand), st.variance(base)
    nc, nb = len(cand), len(base)
    delta = mc - mb
    se = (vc / nc + vb / nb) ** 0.5
    common = {"n_cand": nc, "n_base": nb, "mean_cand": mc, "mean_base": mb, "delta": delta}
    if se <= 0.0:
        # both arms have zero spread: a point comparison, no t distribution
        if delta == 0.0:
            return {**common, "ci95": [0.0, 0.0], "t_stat": 0.0, "p_value": 1.0, "df": float(nc + nb - 2)}
        return {
            **common,
            "ci95": [delta, delta],
            "t_stat": float("inf") if delta > 0 else float("-inf"),
            "p_value": 0.0,
            "df": float(nc + nb - 2),
        }
    from scipy import stats as sps

    df = (vc / nc + vb / nb) ** 2 / ((vc / nc) ** 2 / (nc - 1) + (vb / nb) ** 2 / (nb - 1))
    crit = float(sps.t.ppf(0.975, df))
    res = sps.ttest_ind(cand, base, equal_var=False)
    return {
        **common,
        "ci95": [delta - crit * se, delta + crit * se],
        "t_stat": float(res.statistic),
        "p_value": float(res.pvalue),
        "df": float(df),
    }


@app.command("bench-fixed-flops")
def bench_fixed_flops(
    attention_types: Annotated[
        list[str],
        typer.Option(
            "--attention-type",
            "-a",
            help="Nanochat attention type or comma-separated attention schedule to benchmark (repeatable).",
        ),
    ] = ("standard", "tropical", "ultrametric", "simplicial", "reversible", "gauge"),
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Device for nanochat training runs (passed through to nanochat.train).",
        ),
    ] = "cpu",
    target_flops: Annotated[
        float,
        typer.Option(
            "--target-flops",
            help="Target total FLOPs budget (est) per run.",
            min=1e6,
        ),
    ] = 2e9,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Training seed (single-seed mode; ignored when --seeds is given).",
        ),
    ] = 0,
    seeds: Annotated[
        str,
        typer.Option(
            "--seeds",
            help=(
                "Comma-separated seeds for multi-seed A/B with mean±std, 95% CI, and a Welch "
                "t-test delta vs the baseline arm (bead z7r). Empty = single-seed via --seed."
            ),
        ),
    ] = "",
    val_interval: Annotated[
        int,
        typer.Option(
            "--val-interval",
            help=(
                "Validation cadence in steps. Default 0 = OFF (preserves the throughput-benchmark "
                "behavior); pass >0 for the A/B quality use so runs record results.val_ce_final and "
                "the suite scores on val CE instead of the train-loss tail (bead z7r)."
            ),
            min=0,
        ),
    ] = 0,
    val_batches: Annotated[
        int,
        typer.Option(
            "--val-batches",
            help="Batches per validation evaluation.",
            min=1,
        ),
    ] = 10,
    score_tail: Annotated[
        int,
        typer.Option(
            "--score-tail",
            help="Score is mean of last N losses from nanochat summary.json.",
            min=1,
        ),
    ] = 3,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Batch size for nanochat training.",
            min=1,
        ),
    ] = 8,
    sequence_len: Annotated[
        int,
        typer.Option(
            "--sequence-len",
            help="Sequence length for nanochat training.",
            min=8,
        ),
    ] = 256,
    n_layer: Annotated[
        int,
        typer.Option(
            "--n-layer",
            help="Number of transformer layers.",
            min=1,
        ),
    ] = 4,
    n_head: Annotated[
        int,
        typer.Option(
            "--n-head",
            help="Number of attention heads.",
            min=1,
        ),
    ] = 4,
    n_kv_head: Annotated[
        int,
        typer.Option(
            "--n-kv-head",
            help="Number of KV heads (GQA).",
            min=1,
        ),
    ] = 4,
    n_embd: Annotated[
        int,
        typer.Option(
            "--n-embd",
            help="Embedding dimension.",
            min=16,
        ),
    ] = 128,
    optimizer_type: Annotated[
        str,
        typer.Option(
            "--optimizer-type",
            help="nanochat optimizer type (passed through).",
        ),
    ] = "adamw",
    learning_rate: Annotated[
        float,
        typer.Option(
            "--learning-rate",
            help="Base learning rate for nanochat.train.",
            min=1e-8,
        ),
    ] = 6e-4,
    warmup_steps: Annotated[
        int,
        typer.Option(
            "--warmup-steps",
            help="Warmup steps excluded from throughput measurement.",
            min=0,
        ),
    ] = 0,
    log_interval: Annotated[
        int,
        typer.Option(
            "--log-interval",
            help="Train logging interval (steps).",
            min=1,
        ),
    ] = 1,
    check_numerics: Annotated[
        bool,
        typer.Option(
            "--check-numerics",
            help="Enable NaN/Inf watchpoints inside nanochat.train.",
        ),
    ] = False,
    compile: Annotated[
        bool,
        typer.Option(
            "--compile/--no-compile",
            help="Enable torch.compile in nanochat.train (optional).",
        ),
    ] = False,
    auto_download_data: Annotated[
        bool,
        typer.Option(
            "--auto-download-data/--no-auto-download-data",
            help="Auto-download minimal dataset shards if missing.",
        ),
    ] = True,
    min_parquet_files: Annotated[
        int,
        typer.Option(
            "--min-parquet-files",
            help="Minimum number of parquet shards required (>=2 recommended).",
            min=2,
        ),
    ] = 2,
    include_demo_certs: Annotated[
        bool,
        typer.Option(
            "--include-demo-certs/--no-include-demo-certs",
            help="Also run a few demo certificate runs (math-specific diagnostics) and link them in the suite report.",
        ),
    ] = False,
    artifacts_dir: Annotated[
        Path,
        typer.Option(
            "--artifacts-dir",
            help="Base directory for artifacts (default: artifacts/).",
        ),
    ] = Path("artifacts"),
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help="Suite run identifier (directory name). Defaults to YYYYMMDD_HHMMSS.",
        ),
    ] = None,
    timeout_s: Annotated[
        float,
        typer.Option(
            "--timeout-s",
            help="Per-run timeout (seconds) for subprocess invocations.",
            min=1.0,
        ),
    ] = 1800.0,
):
    attention_types = [str(attn).strip() for attn in attention_types if str(attn).strip()]
    """Benchmark nanochat attention variants under a fixed FLOPs budget.

    Per-run nanochat artifacts:
    - `artifacts/bench/fixed_flops/nanochat/<suite_run_id>/<attention_type>/seed_<seed>/`

    Suite aggregation:
    - `artifacts/bench/fixed_flops/nanochat/<suite_run_id>/summary.json`
    - `artifacts/bench/fixed_flops/nanochat/<suite_run_id>/run.md`
    """
    if not attention_types:
        raise typer.BadParameter("--attention-type must be provided at least once")
    seed_list = [int(s) for s in seeds.split(",") if s.strip()] or [int(seed)]

    suite_run_id = run_id or _default_run_id()
    suite_dir = artifacts_dir / "bench" / "fixed_flops" / "nanochat" / suite_run_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = suite_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    bench_meta = {
        "suite": "bench_fixed_flops",
        "run_id": suite_run_id,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git": _get_git_info(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "argv": sys.argv,
        "device": device,
        "target_flops": float(target_flops),
        "seed": int(seed),
        "seeds": seed_list,
        "val_interval": int(val_interval),
        "val_batches": int(val_batches),
        "score_tail": int(score_tail),
        "train_config": {
            "batch_size": int(batch_size),
            "sequence_len": int(sequence_len),
            "n_layer": int(n_layer),
            "n_head": int(n_head),
            "n_kv_head": int(n_kv_head),
            "n_embd": int(n_embd),
            "optimizer_type": str(optimizer_type),
            "learning_rate": float(learning_rate),
            "warmup_steps": int(warmup_steps),
            "log_interval": int(log_interval),
            "check_numerics": bool(check_numerics),
            "compile": bool(compile),
        },
        "attention_types": list(attention_types),
    }

    def _run_train(attn: str, run_seed: int) -> dict[str, Any]:
        run_topic = f"fixed_flops/nanochat/{suite_run_id}/{attn}"
        run_id_local = f"seed_{run_seed}"
        attention_flag = "--attention-schedule" if "," in attn else "--attention-type"
        train_cmd = [
            sys.executable,
            "-m",
            "nanochat.train",
            "--device",
            device,
            "--seed",
            str(run_seed),
            "--batch-size",
            str(batch_size),
            "--sequence-len",
            str(sequence_len),
            "--n-layer",
            str(n_layer),
            "--n-head",
            str(n_head),
            "--n-kv-head",
            str(n_kv_head),
            "--n-embd",
            str(n_embd),
            "--learning-rate",
            str(learning_rate),
            "--optimizer-type",
            str(optimizer_type),
            attention_flag,
            attn,
            "--target-flops",
            str(float(target_flops)),
            "--warmup-steps",
            str(int(warmup_steps)),
            "--log-interval",
            str(int(log_interval)),
            "--artifacts-dir",
            str(artifacts_dir),
            "--artifacts-kind",
            "bench",
            "--artifacts-topic",
            run_topic,
            "--run-id",
            run_id_local,
        ]
        if val_interval > 0:
            # val CE is the A/B score metric (bead z7r); train defaults val off
            train_cmd += ["--val-interval", str(int(val_interval)), "--val-batches", str(int(val_batches))]
        if compile:
            train_cmd.append("--compile")
        if check_numerics:
            train_cmd.append("--check-numerics")
        if auto_download_data:
            train_cmd.extend(["--auto-download-data", "--min-parquet-files", str(min_parquet_files)])

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(  # nosec B603
                train_cmd,
                capture_output=True,
                text=True,
                timeout=float(timeout_s),
                check=False,
            )
            stdout = proc.stdout
            stderr = proc.stderr
            returncode = int(proc.returncode)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            returncode = 124
        t1 = time.perf_counter()

        stdout_path = logs_dir / f"nanochat_{attn}_seed{run_seed}.stdout.txt"
        stderr_path = logs_dir / f"nanochat_{attn}_seed{run_seed}.stderr.txt"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

        summary_path = artifacts_dir / "bench" / run_topic / run_id_local / "summary.json"
        status = "ok" if returncode == 0 and summary_path.exists() else ("timeout" if returncode == 124 else "error")

        losses: list[float] = []
        final_loss: float | None = None
        score: float | None = None
        ppl: float | None = None
        val_ce_final: float | None = None
        tokens_s: float | None = None
        tflops_s: float | None = None
        peak_mem_gb: float | None = None
        nan_inf_count: int | None = None
        if status == "ok":
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            res = payload.get("results", {})
            losses = [float(x) for x in res.get("losses", [])]
            if losses:
                final_loss = float(losses[-1])
                tail = losses[-min(len(losses), int(score_tail)) :]
                score = float(sum(tail) / len(tail))
                ppl = float(math.exp(score))
                # instability flag: NaN/Inf in the recorded train-loss stream
                nan_inf_count = sum(1 for x in losses if not math.isfinite(x))
            if isinstance(res.get("val_ce_final"), int | float) and math.isfinite(float(res["val_ce_final"])):
                val_ce_final = float(res["val_ce_final"])
            if isinstance(res.get("tokens_per_second"), int | float):
                tokens_s = float(res["tokens_per_second"])
            if isinstance(res.get("tflops_per_second_est"), int | float):
                tflops_s = float(res["tflops_per_second_est"])
            if isinstance(res.get("peak_memory_allocated_gb"), int | float):
                peak_mem_gb = float(res["peak_memory_allocated_gb"])

        return {
            "attention_type": attn,
            "seed": int(run_seed),
            "status": status,
            "returncode": int(returncode),
            "duration_s": float(t1 - t0),
            "command": shlex.join(train_cmd),
            "stdout_path": str(stdout_path.relative_to(artifacts_dir)),
            "stderr_path": str(stderr_path.relative_to(artifacts_dir)),
            "summary_path": str(summary_path.relative_to(artifacts_dir)) if summary_path.exists() else None,
            "score": score,
            "final_loss": final_loss,
            "perplexity_est": ppl,
            "val_ce_final": val_ce_final,
            "tokens_per_second": tokens_s,
            "tflops_per_second_est": tflops_s,
            "peak_memory_allocated_gb": peak_mem_gb,
            "nan_inf_count": nan_inf_count,
        }

    results: list[dict[str, Any]] = []
    total = len(attention_types) * len(seed_list)
    with Progress(
        TextColumn("[bold cyan]fixed-FLOPs bench[/bold cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("runs", total=total)
        for attn in attention_types:
            for run_seed in seed_list:
                console.print(
                    Panel(f"[bold]nanochat[/bold] variant={attn!r} seed={run_seed}", box=box.ROUNDED)
                )
                results.append(_run_train(attn, run_seed))
                prog.advance(task)

    demo_certs: list[dict[str, Any]] = []
    if include_demo_certs:
        console.print(Panel("[bold]Demo certificates[/bold] (math-specific diagnostics)", box=box.ROUNDED))
        cert_runs: list[tuple[str, list[str]]] = [
            ("tropical", ["--tropical-cert"]),
            ("reversible", ["--rev-cayley", "--rev-symplectic"]),
            ("matrix-gauge", ["--gauge-structured"]),
            ("simplicial", ["--simplicial-hodge"]),
            ("ultrametric", ["--ultra-packed"]),
        ]
        for demo_name, extra_flags in cert_runs:
            cert_run_id = f"{suite_run_id}_{demo_name}"
            cmd = [
                sys.executable,
                "-m",
                "cli",
                "run",
                demo_name,
                "--max-iterations",
                "50",
                "--seed",
                str(seed),
                "--artifacts-dir",
                str(artifacts_dir),
                "--run-id",
                cert_run_id,
                *extra_flags,
            ]
            try:
                proc = subprocess.run(  # nosec B603
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=float(timeout_s),
                    check=False,
                )
                stdout = proc.stdout
                stderr = proc.stderr
                returncode = int(proc.returncode)
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                returncode = 124

            (logs_dir / f"demo_{demo_name}.stdout.txt").write_text(stdout, encoding="utf-8")
            (logs_dir / f"demo_{demo_name}.stderr.txt").write_text(stderr, encoding="utf-8")

            cert_dir = artifacts_dir / "certs" / "demos" / demo_name / cert_run_id
            cert_summary = cert_dir / "summary.json"
            demo_certs.append(
                {
                    "demo": demo_name,
                    "status": "ok"
                    if returncode == 0 and cert_summary.exists()
                    else ("timeout" if returncode == 124 else "error"),
                    "returncode": int(returncode),
                    "command": shlex.join(cmd),
                    "summary_path": str(cert_summary.relative_to(artifacts_dir)) if cert_summary.exists() else None,
                }
            )

    baseline_attn = "standard" if "standard" in attention_types else attention_types[0]

    # ---- per-attention aggregation across seeds (bead z7r) ----
    import statistics as _st

    def _finite(attn: str, field: str) -> list[float]:
        return [
            float(r[field])
            for r in results
            if r["attention_type"] == attn
            and r.get("status") == "ok"
            and isinstance(r.get(field), (int, float))
            and math.isfinite(float(r[field]))
        ]

    # the A/B score metric: val CE when every arm recorded it, else train-tail
    arms_ok = [a for a in attention_types if _finite(a, "score")]
    metric_key = (
        "val_ce_final" if arms_ok and all(_finite(a, "val_ce_final") for a in arms_ok) else "score"
    )
    metric_label = "val CE" if metric_key == "val_ce_final" else "train-loss tail"

    def _mean_std(vals: list[float]) -> tuple[float | None, float | None]:
        if not vals:
            return None, None
        return _st.mean(vals), (_st.stdev(vals) if len(vals) > 1 else 0.0)

    aggregates: dict[str, dict[str, Any]] = {}
    for attn in attention_types:
        metric_vals = _finite(attn, metric_key)
        m_mean, m_std = _mean_std(metric_vals)
        tok_mean, _ = _mean_std(_finite(attn, "tokens_per_second"))
        mem_mean, _ = _mean_std(_finite(attn, "peak_memory_allocated_gb"))
        attn_runs = [r for r in results if r["attention_type"] == attn]
        nan_total = sum(int(r["nan_inf_count"]) for r in attn_runs if isinstance(r.get("nan_inf_count"), int))
        aggregates[attn] = {
            "n_ok": len(metric_vals),
            "n_total": len(attn_runs),
            "metric_key": metric_key,
            "metric_mean": m_mean,
            "metric_std": m_std,
            "metric_values": metric_vals,
            "tokens_per_second_mean": tok_mean,
            "peak_memory_allocated_gb_mean": mem_mean,
            "nan_inf_total": nan_total,
            "any_failed": any(r.get("status") != "ok" for r in attn_runs),
        }

    base_vals = aggregates.get(baseline_attn, {}).get("metric_values", [])
    comparisons: dict[str, dict[str, Any]] = {}
    for attn in attention_types:
        if attn == baseline_attn:
            continue
        comparisons[attn] = _bench_welch_delta(aggregates[attn]["metric_values"], base_vals) or {
            "delta": None,
            "reason": "insufficient_seeds_for_welch (need >=2 per arm)",
        }

    table = Table(
        title=f"Fixed-FLOPs A/B vs {baseline_attn!r} ({metric_label}, n={len(seed_list)} seed(s))", box=box.ROUNDED
    )
    table.add_column("attention_type", style="bold")
    table.add_column(f"{metric_label} (mean±std)", justify="right")
    table.add_column("Δ vs base", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("p (Welch)", justify="right")
    table.add_column("tokens/s", justify="right")
    table.add_column("mem GB", justify="right")
    table.add_column("NaN", justify="right")

    for attn in attention_types:
        agg = aggregates[attn]
        cmp = comparisons.get(attn)
        mm, ms = agg["metric_mean"], agg["metric_std"]
        if attn == baseline_attn:
            delta_s = ci_s = p_s = "[dim]—[/dim]"
        elif cmp and cmp.get("delta") is not None:
            delta_s = f"{cmp['delta']:+.4f}"
            ci_s = f"[{cmp['ci95'][0]:+.3f},{cmp['ci95'][1]:+.3f}]"
            p = cmp["p_value"]
            star = " *" if p < 0.05 else ""
            p_s = f"{p:.3f}{star}"
        else:
            delta_s = ci_s = p_s = "n/a"
        table.add_row(
            str(attn),
            f"{mm:.4f}±{ms:.4f}" if isinstance(mm, float) else "n/a",
            delta_s,
            ci_s,
            p_s,
            f"{agg['tokens_per_second_mean']:,.0f}" if isinstance(agg["tokens_per_second_mean"], float) else "n/a",
            f"{agg['peak_memory_allocated_gb_mean']:.2f}"
            if isinstance(agg["peak_memory_allocated_gb_mean"], float)
            else "n/a",
            f"[red]{agg['nan_inf_total']}[/red]" if agg["nan_inf_total"] else "0",
        )

    console.print(table)
    console.print(
        f"[dim]score metric: {metric_label} (lower is better) · * = Welch p<0.05 vs baseline · "
        "descriptive A/B, distinct from mgr adjudicate's preregistered engine[/dim]"
    )

    summary = {
        "schema_version": "mgr.bench.fixed_flops.v2",
        "meta": bench_meta,
        "baseline_attention_type": baseline_attn,
        "score_metric": metric_key,
        "runs": results,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "demo_certs": demo_certs,
    }

    # per-feature CSV (bead z7r): one row per non-baseline arm's delta vs base
    csv_lines = ["attention_type,n_ok,metric,mean,std,delta_vs_base,ci95_lo,ci95_hi,welch_p,nan_inf,tokens_s_mean"]
    for attn in attention_types:
        agg = aggregates[attn]
        cmp = comparisons.get(attn) if attn != baseline_attn else None
        d = cmp.get("delta") if cmp else None
        ci = cmp.get("ci95") if cmp and cmp.get("delta") is not None else None
        p = cmp.get("p_value") if cmp and cmp.get("delta") is not None else None
        csv_lines.append(
            ",".join(
                str(x)
                for x in [
                    attn,
                    agg["n_ok"],
                    metric_key,
                    f"{agg['metric_mean']:.6f}" if isinstance(agg["metric_mean"], float) else "",
                    f"{agg['metric_std']:.6f}" if isinstance(agg["metric_std"], float) else "",
                    f"{d:.6f}" if isinstance(d, float) else "",
                    f"{ci[0]:.6f}" if ci else "",
                    f"{ci[1]:.6f}" if ci else "",
                    f"{p:.6f}" if isinstance(p, float) else "",
                    agg["nan_inf_total"],
                    f"{agg['tokens_per_second_mean']:.3f}"
                    if isinstance(agg["tokens_per_second_mean"], float)
                    else "",
                ]
            )
        )
    (suite_dir / "feature_ablate.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    # rank arms by aggregate metric mean (lower CE is better)
    ranked = sorted(
        (a for a in attention_types if isinstance(aggregates[a]["metric_mean"], float)),
        key=lambda a: aggregates[a]["metric_mean"],
    )
    best = {"attention_type": ranked[0], "metric_mean": aggregates[ranked[0]]["metric_mean"]} if ranked else None

    def _md_row(values: list[str]) -> str:
        return "| " + " | ".join(values) + " |"

    md_lines: list[str] = []
    md_lines.append("# Fixed-FLOPs nanochat benchmark")
    md_lines.append("")
    md_lines.append(f"- Run ID: `{suite_run_id}`")
    md_lines.append(f"- Baseline: `{baseline_attn}`")
    md_lines.append(f"- Device: `{device}`")
    md_lines.append(f"- Target FLOPs/run (est): `{float(target_flops):.3e}`")
    md_lines.append(f"- Seeds: `{seed_list}`")
    md_lines.append(f"- Score metric: `{metric_label}` (lower is better)")
    md_lines.append("")
    md_lines.append("## A/B vs baseline (mean ± std over seeds, Welch t-test)")
    md_lines.append("")
    md_lines.append(
        _md_row([f"attention_type ({metric_label})", "n_ok", "mean±std", "Δ vs base", "95% CI", "Welch p",
                 "tokens/s", "mem GB", "NaN/Inf"])
    )
    md_lines.append(_md_row(["---"] * 9))

    for attn in attention_types:
        agg = aggregates[attn]
        cmp = comparisons.get(attn) if attn != baseline_attn else None
        mm, ms = agg["metric_mean"], agg["metric_std"]
        if attn == baseline_attn:
            delta_s = ci_s = p_s = "—"
        elif cmp and cmp.get("delta") is not None:
            delta_s = f"{cmp['delta']:+.4f}"
            ci_s = f"[{cmp['ci95'][0]:+.3f}, {cmp['ci95'][1]:+.3f}]"
            p_s = f"{cmp['p_value']:.3f}" + (" \\*" if cmp["p_value"] < 0.05 else "")
        else:
            delta_s = ci_s = p_s = "n/a"
        md_lines.append(
            _md_row(
                [
                    str(attn),
                    str(agg["n_ok"]),
                    f"{mm:.4f}±{ms:.4f}" if isinstance(mm, float) else "n/a",
                    delta_s,
                    ci_s,
                    p_s,
                    f"{agg['tokens_per_second_mean']:,.0f}"
                    if isinstance(agg["tokens_per_second_mean"], float)
                    else "n/a",
                    f"{agg['peak_memory_allocated_gb_mean']:.2f}"
                    if isinstance(agg["peak_memory_allocated_gb_mean"], float)
                    else "n/a",
                    str(agg["nan_inf_total"]),
                ]
            )
        )

    md_lines.append("")
    md_lines.append("`\\*` = Welch two-sample p < 0.05 vs baseline. This is a descriptive A/B benchmark, "
                    "deliberately distinct from the preregistered `mgr adjudicate` engine (which tests a "
                    "registered threshold with floor/power gates). Per-feature deltas in `feature_ablate.csv`.")
    md_lines.append("")
    md_lines.append("## Conclusions")
    md_lines.append("")
    if best is None:
        md_lines.append("- No successful runs; see `logs/` for stdout/stderr.")
    else:
        md_lines.append(
            f"- Best (lowest {metric_label}): `{best['attention_type']}` mean=`{best['metric_mean']:.6f}`"
        )
        sig_better, sig_worse = [], []
        for attn in attention_types:
            cmp = comparisons.get(attn)
            if attn == baseline_attn or not cmp or cmp.get("delta") is None:
                continue
            if cmp["p_value"] < 0.05:
                (sig_better if cmp["delta"] < 0 else sig_worse).append((attn, cmp["delta"], cmp["p_value"]))
        if sig_better:
            md_lines.append("- Significantly BETTER than baseline (p<0.05): "
                            + ", ".join(f"`{a}` (Δ{d:+.4f}, p={p:.3f})" for a, d, p in sig_better))
        if sig_worse:
            md_lines.append("- Significantly WORSE than baseline (p<0.05): "
                            + ", ".join(f"`{a}` (Δ{d:+.4f}, p={p:.3f})" for a, d, p in sig_worse))
        if not sig_better and not sig_worse:
            md_lines.append("- No arm differs significantly from baseline at p<0.05 "
                            f"(n={len(seed_list)} seed(s); widen seeds to gain power).")
        unstable = [a for a in attention_types if aggregates[a]["nan_inf_total"] > 0]
        if unstable:
            md_lines.append("- Instability (NaN/Inf in train stream): " + ", ".join(f"`{a}`" for a in unstable))
        failed = sorted({r["attention_type"] for r in results if r.get("status") != "ok"})
        if failed:
            md_lines.append("- Run failures: " + ", ".join(f"`{a}`" for a in failed))

    if demo_certs:
        md_lines.append("")
        md_lines.append("## Demo Certificates (Diagnostics)")
        md_lines.append("")
        for d in demo_certs:
            md_lines.append(f"- `{d['demo']}` status={d['status']} summary={d.get('summary_path')}")

    md_lines.append("")
    md_lines.append("## Command")
    md_lines.append("")
    md_lines.append("```bash")
    md_lines.append(shlex.join(sys.argv))
    md_lines.append("```")
    report_md = "\n".join(md_lines) + "\n"

    _write_artifacts(suite_dir, summary=summary, report_md=report_md)
    console.print(f"[dim]Wrote suite artifacts → {suite_dir}[/dim]")


# -----------------------------------------------------------------------------
# Scaling sweep harness (bead w94.1): resumable model-size ladders per mechanism
#
# WHY: scaling-law comparisons need many training runs across a parameter
# ladder per attention mechanism, and individual rungs can take hours. The
# sweep is therefore resumable at BOTH levels:
#   - mid-sweep: manifest.json tracks per-(rung, seed) status; re-invoking
#     with the same --run-id skips completed work;
#   - mid-rung: when checkpoints were enabled for a run, an interrupted seed
#     continues via nanochat's D1 resume (`--resume-from latest`), which also
#     restores the ORIGINAL step budget from the checkpoint meta.
#
# LADDER RATIONALE (the bead demands documented rationale):
#   - head_dim is FIXED at 32 on every rung: wide enough that the narrowest
#     useful rung (n_embd=64) is still multi-head, divisible by 8 so the
#     quaternion (%4) and octonion (%8) mechanisms are feasible everywhere,
#     and every ladder n_head (= n_embd // head_dim) is even, satisfying the
#     reversible mechanism's even-n_head constraint on every rung (reversible
#     additionally halves the attention heads, so it trains with n_kv_head =
#     n_head // 2 — see _scaling_kv_heads; every other mechanism uses GQA-off
#     n_kv_head = n_head).
#   - aspect ratio n_embd / n_layer = 32 is CONSTANT across rungs: depth
#     grows in lockstep with width (the standard deep-narrow ladder shape).
#   - vocab stays at the GPTConfig default (50304) because the FineWeb
#     tokenizer's id range requires it. The untied wte/lm_head pair therefore
#     dominates small rungs; param counts below are exact TOTALS (including
#     embeddings), instantiated and measured per rung, never estimated.
#
# TOKEN BUDGETS: research ladders spend tokens ~= token_multiplier x params
# (default 20x, the Chinchilla heuristic), expressed to nanochat.train as a
# per-rung derived --target-flops. The smoke ladder pins explicit tiny step
# counts instead: it verifies PLUMBING end-to-end on CPU in minutes, not
# science.

SCALING_HEAD_DIM = 32


def _scaling_rung(name: str, n_layer: int, n_embd: int, *, max_steps: int | None = None) -> dict[str, Any]:
    n_head = n_embd // SCALING_HEAD_DIM
    if n_embd % SCALING_HEAD_DIM != 0 or n_head < 1:
        raise ValueError(f"ladder rung {name!r}: n_embd={n_embd} must be a positive multiple of {SCALING_HEAD_DIM}")
    if n_head % 2 != 0:
        raise ValueError(f"ladder rung {name!r}: n_head={n_head} must stay even (reversible constraint)")
    rung: dict[str, Any] = {"name": name, "n_layer": int(n_layer), "n_embd": int(n_embd)}
    if max_steps is not None:
        rung["max_steps"] = int(max_steps)
    return rung


SCALING_LADDERS: dict[str, list[dict[str, Any]]] = {
    # Smoke: same shapes as the two smallest research rungs, pinned tiny step
    # counts -> CPU plumbing verification in minutes (CI-lite friendly).
    "smoke": [
        _scaling_rung("smoke_6M", 2, 64, max_steps=25),
        _scaling_rung("smoke_14M", 4, 128, max_steps=40),
    ],
    # Small: single-GPU / CPU-patient ladder spanning ~7x in parameters.
    "small": [
        _scaling_rung("6M", 2, 64),
        _scaling_rung("14M", 4, 128),
        _scaling_rung("32M", 8, 256),
    ],
    # Full: adds the 60M and 102M rungs (~16x span) for scaling fits.
    "full": [
        _scaling_rung("6M", 2, 64),
        _scaling_rung("14M", 4, 128),
        _scaling_rung("32M", 8, 256),
        _scaling_rung("60M", 12, 384),
        _scaling_rung("102M", 16, 512),
    ],
}

# Per-mechanism honesty notes surfaced in the feasibility table and recorded
# in the manifest. Soft gates only: nothing here blocks a rung.
_SCALING_MECHANISM_NOTES: dict[str, str] = {
    "octonion": (
        "per-query Python loop makes larger rungs wall-clock-infeasible until "
        "vectorization lands (bead 7b0.6); soft-gated, runs are NOT blocked"
    ),
    "gauge": "cached eval unsupported until A5; TRAINING runs are unaffected",
}


def _scaling_kv_heads(mechanism: str, n_head: int) -> int:
    """GQA width for a rung. Every mechanism runs full attention (n_kv_head =
    n_head) EXCEPT reversible, whose coupling halves the head count and so
    requires n_kv_head to divide (and not exceed) n_head // 2 — the natural
    choice n_head // 2 satisfies both (gpt.py Block validation)."""
    return n_head // 2 if mechanism == "reversible" else n_head


def _scaling_feasibility_rows(
    mechanism: str,
    ladder_name: str,
    *,
    batch_size: int,
    sequence_len: int,
    token_multiplier: float,
) -> list[dict[str, Any]]:
    """Instantiate each rung once on CPU to measure EXACT totals and validity."""
    from nanochat.gpt import GPT, GPTConfig

    rows: list[dict[str, Any]] = []
    for idx, rung in enumerate(SCALING_LADDERS[ladder_name]):
        n_head = rung["n_embd"] // SCALING_HEAD_DIM
        n_kv_head = _scaling_kv_heads(mechanism, n_head)
        row: dict[str, Any] = {
            "index": idx,
            **rung,
            "n_head": n_head,
            "n_kv_head": n_kv_head,
            "feasible": True,
            "reason": None,
            "notes": _SCALING_MECHANISM_NOTES.get(mechanism),
            "param_count": None,
            "flops_per_token_est": None,
            "token_budget": None,
            "planned_max_steps": None,
            "target_flops_est": None,
        }
        try:
            model = GPT(
                GPTConfig(
                    sequence_len=int(sequence_len),
                    n_layer=int(rung["n_layer"]),
                    n_head=n_head,
                    n_kv_head=n_kv_head,
                    n_embd=int(rung["n_embd"]),
                    attention_type=mechanism,
                )
            )
            row["param_count"] = int(sum(p.numel() for p in model.parameters()))
            row["flops_per_token_est"] = int(model.estimate_flops())
            del model
        except ValueError as exc:
            row["feasible"] = False
            row["reason"] = str(exc)
            rows.append(row)
            continue
        if rung.get("max_steps") is not None:
            row["planned_max_steps"] = int(rung["max_steps"])
        else:
            tokens = int(round(float(token_multiplier) * row["param_count"]))
            row["token_budget"] = tokens
            row["planned_max_steps"] = max(1, math.ceil(tokens / (int(batch_size) * int(sequence_len))))
            row["target_flops_est"] = int(row["flops_per_token_est"]) * tokens
        rows.append(row)
    return rows


def _scaling_train_command(
    row: dict[str, Any],
    *,
    mechanism: str,
    suite_run_id: str,
    seed: int,
    device: str,
    batch_size: int,
    sequence_len: int,
    learning_rate: float,
    optimizer_type: str,
    warmup_steps: int,
    val_interval: int,
    val_batches: int,
    artifacts_dir: Path,
    checkpoint_interval: int,
    checkpoint_keep: int,
    data_dir: Path | None,
    auto_download_data: bool,
    min_parquet_files: int,
    continue_from_checkpoint: bool,
) -> list[str]:
    """Build one nanochat.train invocation; layout mirrors bench-fixed-flops."""
    topic = f"{mechanism}/{suite_run_id}/rung_{row['index']}"
    seed_run_id = f"seed_{seed}"
    seed_dir = artifacts_dir / "scaling" / topic / seed_run_id
    cmd = [
        sys.executable,
        "-m",
        "nanochat.train",
        "--device",
        device,
        "--seed",
        str(int(seed)),
        "--batch-size",
        str(int(batch_size)),
        "--sequence-len",
        str(int(sequence_len)),
        "--n-layer",
        str(int(row["n_layer"])),
        "--n-head",
        str(int(row["n_head"])),
        "--n-kv-head",
        str(int(row["n_kv_head"])),
        "--n-embd",
        str(int(row["n_embd"])),
        "--learning-rate",
        str(float(learning_rate)),
        "--optimizer-type",
        str(optimizer_type),
        "--warmup-steps",
        str(int(warmup_steps)),
        "--attention-type",
        mechanism,
    ]
    if row.get("target_flops_est") is not None:
        cmd += ["--target-flops", str(int(row["target_flops_est"]))]
    elif row.get("planned_max_steps") is not None:
        cmd += ["--max-steps", str(int(row["planned_max_steps"]))]
    if int(val_interval) > 0:
        cmd += ["--val-interval", str(int(val_interval)), "--val-batches", str(int(val_batches))]
    cmd += [
        "--artifacts-dir",
        str(artifacts_dir),
        "--artifacts-kind",
        "scaling",
        "--artifacts-topic",
        topic,
        "--run-id",
        seed_run_id,
    ]
    if int(checkpoint_interval) > 0:
        cmd += ["--checkpoint-interval", str(int(checkpoint_interval)), "--checkpoint-keep", str(int(checkpoint_keep))]
    if continue_from_checkpoint:
        # D1 within-run resume: 'latest' scans --checkpoint-dir; train restores
        # the original step budget from the checkpoint meta on its own.
        cmd += ["--resume-from", "latest", "--checkpoint-dir", str(seed_dir / "checkpoints")]
    if data_dir is not None:
        cmd += ["--data-dir", str(data_dir)]
    if auto_download_data:
        cmd += ["--auto-download-data", "--min-parquet-files", str(int(min_parquet_files))]
    return cmd


def _scaling_launch_train(cmd: list[str], *, timeout_s: float) -> tuple[int, str, str]:
    """Run one training subprocess; returns (returncode, stdout, stderr).

    Module-level so tests can monkeypatch the launcher without spawning
    real training (mirrors how test_bench exercises aggregation only).
    """
    try:
        proc = subprocess.run(  # nosec B603
            cmd,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            check=False,
        )
        return int(proc.returncode), proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return 124, stdout, stderr


def _scaling_seed_dir(artifacts_dir: Path, mechanism: str, suite_run_id: str, row: dict[str, Any], seed: int) -> Path:
    return artifacts_dir / "scaling" / mechanism / suite_run_id / f"rung_{row['index']}" / f"seed_{seed}"


def _scaling_write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _scaling_manifest_rung(row: dict[str, Any], seeds: list[int], prior_runs: dict[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
    """Fresh manifest entry for one feasibility row, adopting prior per-seed
    run records (status/metrics) when a run (index, seed) was seen before."""
    runs = []
    for seed in seeds:
        prior = prior_runs.get((int(row["index"]), int(seed)))
        if prior is not None:
            runs.append({**prior})
        else:
            runs.append({"seed": int(seed), "status": "pending", "summary_path": None, "wall_seconds": None})
    return {
        "index": row["index"],
        "name": row["name"],
        "n_layer": row["n_layer"],
        "n_embd": row["n_embd"],
        "n_head": row["n_head"],
        "n_kv_head": row["n_kv_head"],
        "max_steps": row.get("max_steps"),
        "feasible": bool(row["feasible"]),
        "infeasible_reason": row["reason"],
        "notes": row["notes"],
        "param_count": row["param_count"],
        "flops_per_token_est": row["flops_per_token_est"],
        "token_budget": row["token_budget"],
        "planned_max_steps": row["planned_max_steps"],
        "target_flops_est": row["target_flops_est"],
        "runs": runs,
        # Rung status is always DERIVED from its seed runs; the execution loop
        # refreshes it after each pass so resumed sweeps recompute honestly.
        "status": _scaling_rung_status(runs, feasible=bool(row["feasible"])),
    }


def _scaling_rung_status(runs: list[dict[str, Any]], *, feasible: bool) -> str:
    if not feasible:
        return "infeasible"
    statuses = [str(run["status"]) for run in runs]
    if statuses and all(s == "done" for s in statuses):
        return "done"
    if any(s == "failed" for s in statuses):
        return "failed"
    return "pending"


def _scaling_new_manifest(
    *,
    mechanism: str,
    ladder_name: str,
    rows: list[dict[str, Any]],
    seeds: list[int],
    sweep_config: dict[str, Any],
    dataset_fp: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite": "scaling_sweep",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mechanism": mechanism,
        "ladder": ladder_name,
        "git": _get_git_info(),
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "argv": sys.argv,
        "sweep_config": sweep_config,
        "dataset_fingerprint": dataset_fp,
        "rungs": [_scaling_manifest_rung(row, seeds, {}) for row in rows],
    }


_SCALING_RESUME_CONTRACT_KEYS = (
    "mechanism",
    "ladder",
)


def _scaling_load_manifest_for_resume(
    existing: dict[str, Any],
    *,
    mechanism: str,
    ladder_name: str,
    sweep_config: dict[str, Any],
    dataset_fp: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Reuse per-(rung, seed) statuses when the sweep definition matches.

    Returns (manifest, warnings). Raises typer.Exit on an incompatible reuse:
    silently mixing sweeps would corrupt the scaling comparison.
    """
    problems: list[str] = []
    wanted = {"mechanism": mechanism, "ladder": ladder_name}
    for key in _SCALING_RESUME_CONTRACT_KEYS:
        if existing.get(key) != wanted[key]:
            problems.append(f"{key}: manifest={existing.get(key)!r} request={wanted[key]!r}")
    old_cfg = existing.get("sweep_config") or {}
    for key in ("batch_size", "sequence_len", "token_multiplier", "optimizer_type", "learning_rate"):
        if old_cfg.get(key) != sweep_config.get(key):
            problems.append(f"sweep_config.{key}: manifest={old_cfg.get(key)!r} request={sweep_config.get(key)!r}")
    if problems:
        details = "\n".join(f"  - {p}" for p in problems)
        console.print(f"[bold red]manifest mismatch under --run-id[/bold red]\n{details}")
        console.print("[bold red]Refusing to mix sweeps; choose a new --run-id or delete the stale manifest.[/bold red]")
        raise typer.Exit(code=2)
    warnings: list[str] = []
    old_fp = existing.get("dataset_fingerprint") or {}
    if old_fp.get("resolved") and dataset_fp.get("resolved") and old_fp.get("digest") != dataset_fp.get("digest"):
        warnings.append(
            f"dataset changed since this sweep started (digest {str(old_fp.get('digest'))[:12]}… -> "
            f"{str(dataset_fp.get('digest'))[:12]}…); completed rungs trained on different data"
        )
    existing["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    existing["sweep_config"] = sweep_config
    existing["dataset_fingerprint"] = dataset_fp
    return existing, warnings


def _scaling_report_md(manifest: dict[str, Any]) -> str:
    """Human-readable sweep report; written incrementally after every rung."""
    m = manifest
    lines: list[str] = [
        "# Scaling sweep report",
        "",
        f"- Suite: `{m['suite']}` (schema v{m['schema_version']})",
        f"- Mechanism: `{m['mechanism']}` | ladder: `{m['ladder']}`",
        f"- Updated: {m['updated_at']}",
        f"- Sweep config: `{json.dumps(m['sweep_config'], sort_keys=True)}`",
        "",
        "## Ladder",
        "",
        "| rung | n_layer | n_embd | n_head | params | flops/token | tokens | steps | target FLOPs | feasible |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in m["rungs"]:
        fmt = lambda v: ("—" if v is None else f"{v:,}")  # noqa: E731
        lines.append(
            f"| {r['name']} | {r['n_layer']} | {r['n_embd']} | {r['n_head']} | {fmt(r['param_count'])} | "
            f"{fmt(r['flops_per_token_est'])} | {fmt(r['token_budget'])} | {fmt(r['planned_max_steps'])} | "
            f"{fmt(r['target_flops_est'])} | {r['status'] if not r['feasible'] else 'yes'} |"
        )
    lines += ["", "## Runs", "", "| rung | seed | status | final loss | val CE | tok/s | wall s | summary |", "|---|---|---|---|---|---|---|---|"]
    any_notes = False
    for r in m["rungs"]:
        if r.get("notes"):
            any_notes = True
        for run in r["runs"]:
            metrics = run.get("metrics") or {}
            loss = metrics.get("final_loss")
            val_ce = metrics.get("val_ce_final")
            tps = metrics.get("tokens_per_second")
            fmt2 = lambda v: ("—" if v is None else f"{v:.4g}")  # noqa: E731
            summary = run.get("summary_path") or "—"
            lines.append(
                f"| {r['name']} | {run['seed']} | {run['status']} | {fmt2(loss)} | {fmt2(val_ce)} | "
                f"{fmt2(tps)} | {fmt2(run.get('wall_seconds'))} | `{summary}` |"
            )
        if r["status"] == "failed":
            lines.append("")
            lines.append(f"Rung `{r['name']}` FAILED: inspect stderr logs under its `seed_*/logs/` directory.")
    if any_notes:
        lines += ["", "## Mechanism notes", ""]
        for r in m["rungs"]:
            if r.get("notes"):
                lines.append(f"- `{m['mechanism']}`: {r['notes']}")
    lines += ["", "## Command", "", "```bash", shlex.join(sys.argv), "```", ""]
    return "\n".join(lines)


def _scaling_extract_metrics(summary_path: Path | None) -> dict[str, float | None]:
    """Pull headline scalars out of a train summary.json (missing -> Nones)."""
    empty: dict[str, float | None] = {"final_loss": None, "val_ce_final": None, "tokens_per_second": None}
    if summary_path is None or not summary_path.exists():
        return empty
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    results = payload.get("results", {}) if isinstance(payload, dict) else {}
    losses = [float(x) for x in results.get("losses", []) if isinstance(x, int | float)]
    metrics: dict[str, float | None] = {"final_loss": losses[-1] if losses else None}
    val = results.get("val_ce_final")
    metrics["val_ce_final"] = float(val) if isinstance(val, int | float) else None
    tps = results.get("tokens_per_second")
    metrics["tokens_per_second"] = float(tps) if isinstance(tps, int | float) else None
    return metrics


@app.command("scaling-sweep")
def scaling_sweep(
    mechanism: Annotated[
        str,
        typer.Option("--mechanism", "-m", help="Attention mechanism to scale (nanochat attention type)."),
    ],
    ladder: Annotated[
        str,
        typer.Option("--ladder", help="Preset rung ladder: smoke (CPU plumbing), small, or full."),
    ] = "small",
    device: Annotated[
        str,
        typer.Option("--device", help="Device passed through to nanochat.train."),
    ] = "cpu",
    seeds: Annotated[
        int,
        typer.Option("--seeds", help="Number of seeds per rung (seeds are 0..N-1).", min=1),
    ] = 1,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Batch size for every rung.", min=1),
    ] = 8,
    sequence_len: Annotated[
        int,
        typer.Option("--sequence-len", help="Sequence length for every rung.", min=8),
    ] = 256,
    token_multiplier: Annotated[
        float,
        typer.Option(
            "--token-multiplier",
            help=(
                "Research ladders: tokens ~= multiplier x params (Chinchilla-style), "
                "expressed as a derived per-rung --target-flops. Ignored by smoke rungs "
                "(they pin explicit max-steps)."
            ),
            min=0.000001,
        ),
    ] = 20.0,
    learning_rate: Annotated[
        float,
        typer.Option("--learning-rate", help="Base learning rate.", min=1e-8),
    ] = 6e-4,
    optimizer_type: Annotated[
        str,
        typer.Option("--optimizer-type", help="nanochat optimizer type (passed through)."),
    ] = "adamw",
    warmup_steps: Annotated[
        int,
        typer.Option("--warmup-steps", help="Warmup steps.", min=0),
    ] = 0,
    val_interval: Annotated[
        int,
        typer.Option(
            "--val-interval",
            help="Validation cadence in steps; 0 disables (pass >0 so runs record results.val_ce_final).",
            min=0,
        ),
    ] = 0,
    val_batches: Annotated[
        int,
        typer.Option("--val-batches", help="Batches per validation evaluation.", min=1),
    ] = 10,
    checkpoint_interval: Annotated[
        int,
        typer.Option(
            "--checkpoint-interval",
            help=(
                "Steps between D1 checkpoints (enables MID-RUNG resume on retry). "
                "0 disables checkpointing."
            ),
            min=0,
        ),
    ] = 500,
    checkpoint_keep: Annotated[
        int,
        typer.Option("--checkpoint-keep", help="Retain only the newest K checkpoints per run (0 keeps all).", min=0),
    ] = 1,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help=("Parquet corpus dir (sorted files, LAST file = val split). Default: the FineWeb cache. "
                  "Point at an `mgr gen-tasks` output for fast CPU smoke runs."),
        ),
    ] = None,
    auto_download_data: Annotated[
        bool,
        typer.Option("--auto-download-data/--no-auto-download-data", help="Auto-download minimal dataset shards if missing."),
    ] = True,
    min_parquet_files: Annotated[
        int,
        typer.Option("--min-parquet-files", help="Minimum number of parquet shards required (>=2 recommended).", min=2),
    ] = 2,
    artifacts_dir: Annotated[
        Path,
        typer.Option("--artifacts-dir", help="Base artifacts directory."),
    ] = Path("artifacts"),
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Suite run identifier (directory name). Defaults to YYYYMMDD_HHMMSS. Reuse to RESUME."),
    ] = None,
    timeout_s: Annotated[
        float,
        typer.Option("--timeout-s", help="Per-training-run timeout in seconds.", min=1.0),
    ] = 3600.0,
    resume_sweep: Annotated[
        bool,
        typer.Option(
            "--resume-sweep/--fresh-sweep",
            help="With an existing manifest under --run-id: reuse per-(rung, seed) statuses (skip done work).",
        ),
    ] = True,
    resume_train: Annotated[
        bool,
        typer.Option(
            "--resume-train/--no-resume-train",
            help="Continue interrupted seeds from their newest D1 checkpoint when one exists.",
        ),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the feasibility table and planned commands; launch NOTHING, write NOTHING."),
    ] = False,
):
    """Train one attention mechanism across a resumable model-size ladder.

    Artifacts land under artifacts/scaling/<mechanism>/<run_id>/ :
      manifest.json          sweep-level state machine (resume contract)
      report.md              human-readable ladder + run tables
      rung_<i>/seed_<k>/     nanochat train dir (summary.json, metrics.jsonl, logs/)
    """
    from nanochat.dataset import dataset_fingerprint
    from nanochat.gpt import SUPPORTED_ATTENTION_TYPES

    if mechanism not in SUPPORTED_ATTENTION_TYPES:
        console.print(
            f"[bold red]unknown mechanism {mechanism!r}; expected one of: {', '.join(SUPPORTED_ATTENTION_TYPES)}[/bold red]"
        )
        raise typer.Exit(code=2)
    if ladder not in SCALING_LADDERS:
        console.print(f"[bold red]unknown ladder {ladder!r}; expected one of: {', '.join(SCALING_LADDERS)}[/bold red]")
        raise typer.Exit(code=2)

    seed_list = list(range(int(seeds)))
    rows = _scaling_feasibility_rows(
        mechanism,
        ladder,
        batch_size=batch_size,
        sequence_len=sequence_len,
        token_multiplier=token_multiplier,
    )

    # ---- feasibility table (always printed; the upfront validation gate) ----
    table = Table(title=f"scaling ladder [{ladder}] × {mechanism}", box=box.SIMPLE_HEAVY)
    for col in ("rung", "shape", "params", "flops/token", "tokens", "steps", "target FLOPs", "feasible"):
        table.add_column(col)
    for r in rows:
        shape = f"L{r['n_layer']} E{r['n_embd']} H{r['n_head']}"
        fmt = lambda v: ("—" if v is None else f"{v:,}")  # noqa: E731
        feas = "[green]yes[/green]" if r["feasible"] else f"[red]{r['reason']}[/red]"
        note = f"\n[dim]{r['notes']}[/dim]" if r["notes"] and r["feasible"] else ""
        table.add_row(
            r["name"],
            shape,
            fmt(r["param_count"]),
            fmt(r["flops_per_token_est"]),
            fmt(r["token_budget"]),
            fmt(r["planned_max_steps"]),
            fmt(r["target_flops_est"]),
            feas + note,
        )
    console.print(table)

    suite_run_id = run_id or _default_run_id()
    suite_dir = artifacts_dir / "scaling" / mechanism / suite_run_id

    def _cmd_preview(row: dict[str, Any], seed: int) -> str:
        return shlex.join(
            _scaling_train_command(
                row,
                mechanism=mechanism,
                suite_run_id=suite_run_id,
                seed=seed,
                device=device,
                batch_size=batch_size,
                sequence_len=sequence_len,
                learning_rate=learning_rate,
                optimizer_type=optimizer_type,
                warmup_steps=warmup_steps,
                val_interval=val_interval,
                val_batches=val_batches,
                artifacts_dir=artifacts_dir,
                checkpoint_interval=checkpoint_interval,
                checkpoint_keep=checkpoint_keep,
                data_dir=data_dir,
                auto_download_data=auto_download_data,
                min_parquet_files=min_parquet_files,
                continue_from_checkpoint=False,
            )
        )

    if dry_run:
        console.print("[bold yellow]dry run:[/bold yellow] suite dir would be " + str(suite_dir))
        first = next((r for r in rows if r["feasible"]), None)
        if first is not None:
            console.print(f"[dim]first command would be:[/dim]\n{_cmd_preview(first, seed_list[0])}")
        return

    # ---- manifest: fresh or resumed ----------------------------------------
    dataset_fp = dataset_fingerprint(str(data_dir) if data_dir is not None else None)
    sweep_config: dict[str, Any] = {
        "device": device,
        "batch_size": int(batch_size),
        "sequence_len": int(sequence_len),
        "token_multiplier": float(token_multiplier),
        "learning_rate": float(learning_rate),
        "optimizer_type": str(optimizer_type),
        "warmup_steps": int(warmup_steps),
        "val_interval": int(val_interval),
        "checkpoint_interval": int(checkpoint_interval),
        "checkpoint_keep": int(checkpoint_keep),
    }
    suite_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = suite_dir / "manifest.json"
    resume_warnings: list[str] = []
    if manifest_path.exists() and resume_sweep:
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            console.print(
                f"[bold red]manifest at {manifest_path} is corrupt ({exc}); "
                "start a new --run-id or delete the file.[/bold red]"
            )
            raise typer.Exit(code=2) from exc
        manifest, resume_warnings = _scaling_load_manifest_for_resume(
            existing,
            mechanism=mechanism,
            ladder_name=ladder,
            sweep_config=sweep_config,
            dataset_fp=dataset_fp,
        )
        # Structural drift (ladder redefined in-code) invalidates old indexes.
        if [r["name"] for r in manifest["rungs"]] != [r["name"] for r in rows]:
            console.print("[bold red]ladder definition changed since the manifest was written; start a new --run-id.[/bold red]")
            raise typer.Exit(code=2)
        prior_runs = {(int(r["index"]), int(run["seed"])): run for r in manifest["rungs"] for run in r["runs"]}
        manifest["rungs"] = [_scaling_manifest_rung(row, seed_list, prior_runs) for row in rows]
    else:
        if manifest_path.exists():
            console.print("[yellow]existing manifest found but --fresh-sweep given; ignoring stored statuses.[/yellow]")
        manifest = _scaling_new_manifest(
            mechanism=mechanism,
            ladder_name=ladder,
            rows=rows,
            seeds=seed_list,
            sweep_config=sweep_config,
            dataset_fp=dataset_fp,
        )
    for warning in resume_warnings:
        console.print(f"[bold yellow]warning:[/bold yellow] {warning}")

    def _save() -> None:
        manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _scaling_write_json_atomic(manifest_path, manifest)

    _save()
    (suite_dir / "report.md").write_text(_scaling_report_md(manifest), encoding="utf-8")
    logs_root = suite_dir / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    # ---- execute pending work ----------------------------------------------
    any_failed = False
    for rung in manifest["rungs"]:
        if rung["status"] == "infeasible":
            continue
        for run in rung["runs"]:
            if run["status"] == "done":
                continue
            seed = int(run["seed"])
            seed_dir = _scaling_seed_dir(artifacts_dir, mechanism, suite_run_id, rung, seed)
            ckpt_dir = seed_dir / "checkpoints"
            continue_from = (
                resume_train
                and int(checkpoint_interval) > 0
                and run["status"] in {"failed", "running"}
                and ckpt_dir.is_dir()
                and any(ckpt_dir.glob("*.pt"))
            )
            row = next(r for r in rows if r["index"] == rung["index"])
            cmd = _scaling_train_command(
                row,
                mechanism=mechanism,
                suite_run_id=suite_run_id,
                seed=seed,
                device=device,
                batch_size=batch_size,
                sequence_len=sequence_len,
                learning_rate=learning_rate,
                optimizer_type=optimizer_type,
                warmup_steps=warmup_steps,
                val_interval=val_interval,
                val_batches=val_batches,
                artifacts_dir=artifacts_dir,
                checkpoint_interval=checkpoint_interval,
                checkpoint_keep=checkpoint_keep,
                data_dir=data_dir,
                auto_download_data=auto_download_data,
                min_parquet_files=min_parquet_files,
                continue_from_checkpoint=continue_from,
            )
            label = f"{rung['name']}/seed_{seed}"
            suffix = " [cyan](D1 resume)[/cyan]" if continue_from else ""
            console.print(f"[bold cyan]▶ training {label}[/bold cyan]{suffix}")
            run["status"] = "running"
            _save()

            t0 = time.perf_counter()
            returncode, stdout, stderr = _scaling_launch_train(cmd, timeout_s=timeout_s)
            wall = time.perf_counter() - t0

            (logs_root / f"{rung['name']}_seed{seed}.stdout.txt").write_text(stdout, encoding="utf-8")
            (logs_root / f"{rung['name']}_seed{seed}.stderr.txt").write_text(stderr, encoding="utf-8")
            summary_path = seed_dir / "summary.json"
            ok = returncode == 0 and summary_path.exists()
            run["status"] = "done" if ok else "failed"
            run["wall_seconds"] = round(wall, 2)
            run["summary_path"] = str(summary_path) if ok else None
            run["metrics"] = _scaling_extract_metrics(summary_path if ok else None)
            run["returncode"] = returncode
            if not ok:
                any_failed = True
                tail = "\n".join(stderr.splitlines()[-5:])
                console.print(f"[bold red]✗ {label} failed (rc={returncode})[/bold red]\n[dim]{tail}[/dim]")
            else:
                metrics = run.get("metrics") or {}
                console.print(
                    f"[green]✓ {label} done[/green] "
                    f"loss={metrics.get('final_loss')} val_ce={metrics.get('val_ce_final')} "
                    f"({wall:.1f}s)"
                )
            _save()
            (suite_dir / "report.md").write_text(_scaling_report_md(manifest), encoding="utf-8")
        rung["status"] = _scaling_rung_status(rung["runs"], feasible=bool(rung["feasible"]))
        _save()

    # ---- wrap up ------------------------------------------------------------
    (suite_dir / "report.md").write_text(_scaling_report_md(manifest), encoding="utf-8")
    statuses = [rung["status"] for rung in manifest["rungs"]]
    summary_panel = Panel(
        "\n".join(
            [
                f"mechanism: [bold]{mechanism}[/bold]  ladder: [bold]{ladder}[/bold]  run_id: [bold]{suite_run_id}[/bold]",
                f"rungs: {statuses.count('done')}/{len(statuses)} done, {statuses.count('failed')} failed, "
                f"{statuses.count('infeasible')} infeasible",
                f"artifacts: {suite_dir}",
            ]
        ),
        title="scaling sweep complete",
        box=box.ROUNDED,
    )
    console.print(summary_panel)
    no_feasible_work = not any(r["feasible"] for r in rows)
    if any_failed or (no_feasible_work and statuses.count("done") == 0):
        raise typer.Exit(code=1)


# -----------------------------------------------------------------------------
# Scaling-law fits & comparative report (bead w94.2): consumes w94.1 sweep
# manifests, retrains NOTHING.
#
# HONESTY CONTRACT (hard-coded into every report):
#   - fits describe ONLY the measured compute range; no extrapolation claims;
#   - bootstrap CIs are shown with explicit wide-CI cautions; resisting
#     over-claim is a deliverable, so when no pairwise exponent difference is
#     significant the headline says exactly that;
#   - thin ladders (<3 rungs) get the plain power law only - a 2-point
#     saturating fit is underdetermined theater and is refused.


def _scaling_fmt(x: Any) -> str:
    """Fixed-precision float rendering; keeps regenerated reports byte-stable."""
    if x is None:
        return "—"
    return f"{float(x):.6g}"


def _scaling_report_tail_loss(values: list[float], fraction: float) -> float | None:
    """Mean of the last `fraction` of a loss series (noise reduction)."""
    if not values:
        return None
    n_tail = max(1, math.ceil(len(values) * float(fraction)))
    tail = values[-n_tail:]
    return sum(tail) / len(tail)


def _scaling_report_read_losses(seed_dir: Path, fraction: float) -> tuple[float | None, str]:
    """Tail loss for one seed run: D2 metrics.jsonl stream first (rz8.2),
    train summary.json losses as fallback."""
    metrics_path = seed_dir / "metrics.jsonl"
    if metrics_path.exists():
        losses: list[float] = []
        try:
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and rec.get("type") == "step" and isinstance(rec.get("loss"), int | float):
                    losses.append(float(rec["loss"]))
        except OSError:
            losses = []
        tail = _scaling_report_tail_loss(losses, fraction)
        if tail is not None:
            return tail, "metrics"
    summary_path = seed_dir / "summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            losses = [float(x) for x in payload.get("results", {}).get("losses", [])]
        except (OSError, json.JSONDecodeError, AttributeError, TypeError):
            return None, "missing"
        tail = _scaling_report_tail_loss(losses, fraction)
        if tail is not None:
            return tail, "summary"
    return None, "missing"


def _scaling_report_rung_compute(rung: dict[str, Any], manifest: dict[str, Any]) -> float | None:
    """Total training compute C (FLOPs) for one rung, from manifest fields alone."""
    if rung.get("target_flops_est") is not None:
        return float(rung["target_flops_est"])
    fpt = rung.get("flops_per_token_est")
    steps = rung.get("planned_max_steps")
    cfg = manifest.get("sweep_config") or {}
    batch = cfg.get("batch_size")
    seq = cfg.get("sequence_len")
    if fpt is None or steps is None or not batch or not seq:
        return None
    return float(fpt) * int(batch) * int(seq) * int(steps)


def _scaling_report_load_series(run_path: Path, tail_fraction: float) -> dict[str, Any] | None:
    """Load one suite directory (or manifest path) into a fit-ready series."""
    manifest_path = run_path / "manifest.json" if run_path.is_dir() else run_path
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    points: list[dict[str, Any]] = []
    for rung in manifest.get("rungs", []):
        if rung.get("status") != "done" or not rung.get("feasible", False):
            continue
        compute_c = _scaling_report_rung_compute(rung, manifest)
        if compute_c is None:
            continue
        tails: list[float] = []
        source = "metrics"
        for run in rung.get("runs", []):
            if run.get("status") != "done":
                continue
            summary_path = run.get("summary_path")
            if summary_path:
                seed_dir = Path(summary_path).parent
            else:
                seed_dir = (
                    manifest_path.parent / f"rung_{rung['index']}" / f"seed_{run.get('seed', 0)}"
                )
            tail, src = _scaling_report_read_losses(seed_dir, tail_fraction)
            if tail is not None:
                tails.append(tail)
                source = src
        if tails:
            points.append(
                {
                    "C": compute_c,
                    "loss": sum(tails) / len(tails),
                    "n_seeds": len(tails),
                    "rung": rung["name"],
                    "source": source,
                }
            )
    if not points:
        return None
    return {
        "mechanism": manifest.get("mechanism"),
        "ladder": manifest.get("ladder"),
        "manifest_path": str(manifest_path),
        "points": sorted(points, key=lambda p: p["C"]),
    }


def _scaling_report_fit_saturating(cs: Any, ls: Any) -> dict[str, float] | None:
    """L(C) = a*C^-b + c via robust (soft_l1) least squares on the original scale.

    Raw FLOPs spans many decades, so the solve runs on x = C/max(C) (well
    conditioned) with the amplitude converted back afterwards; two starts
    (plain-power-law anchored + span heuristic) guard the soft_l1 basin.
    """
    import numpy as np
    from scipy.optimize import least_squares

    cs_arr = np.asarray(cs, dtype=float)
    ls_arr = np.asarray(ls, dtype=float)
    c_ref = float(np.max(cs_arr))
    x = cs_arr / c_ref
    span = float(np.max(ls_arr) - np.min(ls_arr))
    floor0 = max(float(np.min(ls_arr)) * 0.5, 1e-9)

    def model(p: Any) -> Any:
        return p[0] * x ** (-p[1]) + p[2]

    starts: list[list[float]] = []
    plain = _scaling_report_fit_plain(cs_arr, ls_arr)
    if plain is not None:
        b0 = min(max(float(plain["b"]), 0.0), 2.0 - 1e-9)
        starts.append([max(float(plain["k"]) * c_ref ** (-float(plain["b"])), 1e-9), b0, floor0])
    starts.append([max(span, 1e-9), 0.5, floor0])

    best: dict[str, Any] | None = None
    for p0 in starts:
        try:
            res = least_squares(
                lambda p: model(p) - ls_arr,
                x0=np.array(p0),
                bounds=([1e-12, 0.0, 0.0], [np.inf, 2.0 - 1e-9, np.inf]),
                loss="soft_l1",
                f_scale=max(span, 1e-9) * 0.1,
            )
        except Exception:  # noqa: BLE001 - non-convergence becomes "no fit"
            continue
        if not res.success:
            continue
        if best is None or float(res.cost) < best["cost"]:
            best = {"cost": float(res.cost), "x": res.x}
    if best is None:
        return None
    a_scaled, b_fit, c_fit = (float(v) for v in best["x"])
    pred = a_scaled * x ** (-b_fit) + c_fit
    ss_res = float(np.sum((ls_arr - pred) ** 2))
    ss_tot = float(np.sum((ls_arr - np.mean(ls_arr)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"a": a_scaled * c_ref**b_fit, "b": b_fit, "c": c_fit, "r2": r2}


def _scaling_report_fit_plain(cs: Any, ls: Any) -> dict[str, float] | None:
    """Plain power law L = k*C^-b via log-log linear regression."""
    import numpy as np

    slope, intercept = np.polyfit(np.log10(cs), np.log10(ls), 1)
    log_pred = slope * np.log10(cs) + intercept
    ss_res = float(np.sum((np.log10(ls) - log_pred) ** 2))
    ss_tot = float(np.sum((np.log10(ls) - np.mean(np.log10(ls))) ** 2))
    r2_log = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"k": float(10.0**intercept), "b": float(-slope), "r2_log": r2_log}


def _scaling_report_bootstrap(
    cs: Any,
    ls: Any,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any] | None:
    """Deterministic bootstrap over rungs; returns exponent b CI + samples."""
    import numpy as np

    rng = np.random.default_rng(int(seed))
    n = len(cs)
    b_samples: list[float] = []
    a_samples: list[float] = []
    c_samples: list[float] = []
    for _ in range(int(draws)):
        idx = rng.integers(0, n, size=n)
        if len(set(idx.tolist())) < 3:
            continue  # saturating fit needs >=3 distinct rungs
        fit = _scaling_report_fit_saturating(cs[idx], ls[idx])
        if fit is not None and math.isfinite(fit["b"]):
            b_samples.append(fit["b"])
            a_samples.append(fit["a"])
            c_samples.append(fit["c"])
    if len(b_samples) < max(50, draws // 20):
        return None
    arr = np.array(b_samples)
    return {
        "b_ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "a_mean": float(np.mean(a_samples)),
        "c_mean": float(np.mean(c_samples)),
        "n_draws_ok": len(b_samples),
        "_samples": arr,
    }


def _scaling_report_pairwise(samples_by_mech: dict[str, Any], confidence: float) -> list[dict[str, Any]]:
    """Bootstrap-overlap test for exponent differences between mechanisms."""
    import numpy as np

    mechs = sorted(samples_by_mech)
    lo_q = 100.0 * (1.0 - confidence) / 2.0
    hi_q = 100.0 - lo_q
    rows: list[dict[str, Any]] = []
    for i, m1 in enumerate(mechs):
        for m2 in mechs[i + 1 :]:
            s1, s2 = samples_by_mech[m1], samples_by_mech[m2]
            n = min(len(s1), len(s2))
            diffs = s1[:n] - s2[:n]
            lo, hi = (float(x) for x in np.percentile(diffs, [lo_q, hi_q]))
            rows.append(
                {
                    "pair": f"{m1} vs {m2}",
                    "delta_b": float(np.mean(s1) - np.mean(s2)),
                    "ci": [lo, hi],
                    "significant": not (lo <= 0.0 <= hi),
                }
            )
    return rows


@app.command("scaling-report")
def scaling_report(
    runs: Annotated[
        list[Path],
        typer.Option(
            "--runs",
            help=(
                "Suite directory (containing manifest.json) or direct manifest path "
                "from mgr scaling-sweep; repeatable - one series per mechanism."
            ),
        ),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for the report, fits JSON, and plot."),
    ] = Path("artifacts/scaling/report"),
    tail_fraction: Annotated[
        float,
        typer.Option("--tail-fraction", help="Loss = mean of the last N-fraction of each run's loss stream.", min=0.01, max=1.0),
    ] = 0.1,
    bootstrap: Annotated[
        int,
        typer.Option("--bootstrap", help="Bootstrap resamples for exponent CIs.", min=100),
    ] = 2000,
    bootstrap_seed: Annotated[
        int,
        typer.Option("--bootstrap-seed", help="Deterministic bootstrap RNG seed."),
    ] = 1729,
    confidence: Annotated[
        float,
        typer.Option("--confidence", help="CI / test confidence level.", min=0.5, max=0.999),
    ] = 0.95,
    no_plot: Annotated[
        bool,
        typer.Option("--no-plot", help="Skip matplotlib rendering (report + JSON only)."),
    ] = False,
):
    """Fit scaling laws from EXISTING sweep artifacts; never trains.

    Fits L(C) = a*C^-b + c per mechanism (plain power-law alongside),
    bootstraps exponent CIs, tests pairwise exponent differences, renders a
    log-log overlay + per-mechanism panels, and emits a G2-compatible
    mgr.scaling.v1 JSON block. Regenerating from fixed artifacts is
    byte-stable (deterministic bootstrap seed, fixed formatting).
    """
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    series: list[dict[str, Any]] = []
    for rp in runs:
        loaded = _scaling_report_load_series(rp, tail_fraction)
        if loaded is None:
            console.print(f"[yellow]skipping {rp}: no readable manifest with completed rungs[/yellow]")
            continue
        series.append(loaded)
    if not series:
        console.print("[bold red]no usable series found; pass mgr scaling-sweep artifact directories.[/bold red]")
        raise typer.Exit(code=2)

    # ---- fits ----------------------------------------------------------------
    mech_fits: dict[str, dict[str, Any]] = {}
    samples_by_mech: dict[str, Any] = {}
    for s in sorted(series, key=lambda x: str(x["mechanism"])):
        mech = str(s["mechanism"])
        cs = np.array([p["C"] for p in s["points"]], dtype=float)
        ls = np.array([p["loss"] for p in s["points"]], dtype=float)
        entry: dict[str, Any] = {
            "ladder": s["ladder"],
            "manifest_path": s["manifest_path"],
            "points": [
                {
                    "rung": p["rung"],
                    "C_flops": p["C"],
                    "tail_loss": p["loss"],
                    "n_seeds": p["n_seeds"],
                    "loss_source": p["source"],
                }
                for p in s["points"]
            ],
        }
        sat = _scaling_report_fit_saturating(cs, ls) if len(cs) >= 3 else None
        plain = _scaling_report_fit_plain(cs, ls) if len(cs) >= 2 else None
        if sat is not None:
            entry["saturating"] = sat
        if plain is not None:
            entry["plain"] = plain
        if sat is not None and len(cs) >= 3:
            boot = _scaling_report_bootstrap(cs, ls, draws=bootstrap, seed=bootstrap_seed)
            if boot is not None:
                samples_by_mech[mech] = boot.pop("_samples")
                entry["bootstrap"] = boot
                width = boot["b_ci95"][1] - boot["b_ci95"][0]
                entry["b_ci_wide"] = bool(width > 0.5 * abs(sat["b"])) if sat["b"] > 0 else True
        mech_fits[mech] = entry

    pairwise = _scaling_report_pairwise(samples_by_mech, confidence) if len(samples_by_mech) >= 2 else []
    significant = [row for row in pairwise if row["significant"]]

    # ---- machine-readable core (byte-stable: sorted keys, fixed floats) ------
    def _clean(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: _clean(v) for k, v in node.items() if k != "b_ci_wide"}
        if isinstance(node, float):
            return float(f"{node:.10g}")
        if isinstance(node, list):
            return [_clean(v) for v in node]
        return node

    g2_json = {
        "schema": "mgr.scaling.v1",
        "generated_from": sorted(str(p) for p in runs),
        "config": {
            "tail_fraction": tail_fraction,
            "bootstrap_draws": bootstrap,
            "bootstrap_seed": bootstrap_seed,
            "confidence": confidence,
        },
        "fits": {
            mech: {
                "exponent_b": entry.get("saturating", {}).get("b"),
                "exponent_b_ci95": entry.get("bootstrap", {}).get("b_ci95"),
                "amplitude_a": entry.get("saturating", {}).get("a"),
                "floor_c": entry.get("saturating", {}).get("c"),
                "r2_original_scale": entry.get("saturating", {}).get("r2"),
                "plain_power_law_b": entry.get("plain", {}).get("b"),
                "n_rungs": len(entry["points"]),
                "n_seeds_total": sum(p["n_seeds"] for p in entry["points"]),
            }
            for mech, entry in mech_fits.items()
        },
        "pairwise_exponent_tests": [
            {"pair": row["pair"], "delta_b": _clean(row["delta_b"]), "ci95": _clean(row["ci"]), "significant": row["significant"]}
            for row in pairwise
        ],
    }
    fits_json_text = json.dumps(_clean(g2_json), indent=2, sort_keys=True) + "\n"
    (out / "fits.json").write_text(fits_json_text, encoding="utf-8")

    # ---- plot -----------------------------------------------------------------
    plot_name = "scaling_overlay.png"
    if not no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 9})
        fig, axes = plt.subplots(1, max(2, len(mech_fits)), figsize=(4.2 * max(2, len(mech_fits)), 3.6), squeeze=False)
        ax_all = axes[0][0]
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for pos, (mech, entry) in enumerate(sorted(mech_fits.items())):
            ax = axes[0][pos] if pos < axes.shape[1] else ax_all
            pts = entry["points"]
            xs = [p["C_flops"] for p in pts]
            ys = [p["tail_loss"] for p in pts]
            color = colors[pos % len(colors)]
            ax.scatter(xs, ys, color=color, label=f"{mech} data", zorder=3)
            sat = entry.get("saturating")
            boot = entry.get("bootstrap")
            if sat is not None:
                a_pt, c_pt = sat["a"], sat["c"]
                grid = np.geomspace(min(xs), max(xs), 120)
                base = lambda b: a_pt * grid ** (-b) + c_pt  # noqa: E731
                ax.plot(grid, base(sat["b"]), color=color, label=f"{mech} fit")
                if boot is not None:
                    ax.fill_between(
                        grid,
                        base(boot["b_ci95"][0]),
                        base(boot["b_ci95"][1]),
                        color=color,
                        alpha=0.15,
                        label="b CI band",
                    )
                if ax is not ax_all or len(mech_fits) == 1:
                    ax.set_title(mech)
                    ax.set_xlabel("compute C (FLOPs)")
                    ax.set_ylabel("tail loss")
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.legend(fontsize=6)
        ax_all.set_title(f"scaling overlay ({len(mech_fits)} mechanisms)")
        ax_all.set_xscale("log")
        ax_all.set_yscale("log")
        fig.tight_layout()
        fig.savefig(out / plot_name)
        plt.close(fig)

    # ---- markdown report -------------------------------------------------------
    lines: list[str] = [
        "# Scaling-law comparative report",
        "",
        f"- Inputs: `{sorted(str(p) for p in runs)}`",
        f"- Config: tail_fraction={tail_fraction}, bootstrap={bootstrap}, seed={bootstrap_seed}, confidence={confidence}",
        "- Loss definition: mean of the last tail_fraction of each run's loss stream "
        "(D2 metrics.jsonl preferred, summary.json fallback); multi-seed rungs averaged.",
        "",
        "> **HONESTY CAVEATS (fixed):** these fits describe ONLY the measured compute range — ",
        "> extrapolation beyond it is unsupported by construction. Thin ladders carry wide CIs; ",
        "> treat wide-CI exponent comparisons as undecided, not as evidence of equality.",
        "",
    ]
    if significant:
        headline = "; ".join(f"{r['pair']} (Δb={_scaling_fmt(r['delta_b'])})" for r in significant)
        lines += [f"**Headline:** significant exponent differences at this scale: {headline}.", ""]
    elif pairwise:
        lines += [
            "**Headline:** NO pairwise exponent differences are significant at this compute scale — "
            "treat all exponent rankings among these mechanisms as ties pending larger ladders.",
            "",
        ]
    else:
        lines += [
            "**Headline:** fewer than two mechanisms produced saturating fits with valid bootstraps — "
            "no cross-mechanism comparison is possible yet.",
            "",
        ]
    for mech, entry in sorted(mech_fits.items()):
        lines += [f"## {mech}", "", "| rung | C (FLOPs) | tail loss | seeds | loss source |", "|---|---|---|---|---|"]
        for p in entry["points"]:
            lines.append(
                f"| {p['rung']} | {_scaling_fmt(p['C_flops'])} | {_scaling_fmt(p['tail_loss'])} | {p['n_seeds']} | {p['loss_source']} |"
            )
        lines.append("")
        sat = entry.get("saturating")
        plain = entry.get("plain")
        boot = entry.get("bootstrap")
        if sat is None:
            lines.append(
                f"- Saturating fit REFUSED ({len(entry['points'])} rungs < 3): a 2-parameter-plus-floor fit on two points is underdetermined."
            )
        else:
            ci_txt = (
                f"b ∈ [{_scaling_fmt(boot['b_ci95'][0])}, {_scaling_fmt(boot['b_ci95'][1])}]"
                if boot
                else "bootstrap failed to converge sufficiently"
            )
            wide = ", **WIDE CI — comparison undecided**" if entry.get("b_ci_wide") else ""
            lines.append(
                f"- Saturating fit: a={_scaling_fmt(sat['a'])}, b={_scaling_fmt(sat['b'])}, c={_scaling_fmt(sat['c'])}, "
                f"R²(original scale)={_scaling_fmt(sat['r2'])}; {ci_txt}{wide}"
            )
        if plain is not None:
            lines.append(
                f"- Plain power law: k={_scaling_fmt(plain['k'])}, b={_scaling_fmt(plain['b'])}, R²(log space)={_scaling_fmt(plain['r2_log'])}"
            )
        lines.append("")
    if pairwise:
        lines += ["## Pairwise exponent tests", "", "| pair | Δb | CI | significant |", "|---|---|---|---|"]
        for r in pairwise:
            lines.append(
                f"| {r['pair']} | {_scaling_fmt(r['delta_b'])} | [{_scaling_fmt(r['ci'][0])}, {_scaling_fmt(r['ci'][1])}] | "
                f"{'yes' if r['significant'] else 'no'} |"
            )
        lines.append("")
    lines += [
        "## G2-compatible result block (`mgr.scaling.v1`)",
        "",
        "```json",
        fits_json_text.rstrip("\n"),
        "```",
        "",
        "Registry note: no hypothesis prediction currently targets `mgr.scaling.v1`; when one is "
        "registered (G1) `mgr adjudicate` can consume this block verbatim.",
        "",
    ]
    (out / "scaling_report.md").write_text("\n".join(lines), encoding="utf-8")
    console.print(
        Panel(
            "\n".join(
                [
                    f"mechanisms fitted: [bold]{len(mech_fits)}[/bold]; pairwise significant: [bold]{len(significant)}[/bold]",
                    f"artifacts: {out}/{{scaling_report.md, fits.json{'' if no_plot else ', ' + plot_name}}}",
                ]
            ),
            title="scaling report complete",
            box=box.ROUNDED,
        )
    )


@app.command("per-head-metrics")
def per_head_metrics(
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Device for nanochat training runs (passed through to nanochat.train).",
        ),
    ] = "cpu",
    seeds: Annotated[
        list[int],
        typer.Option(
            "--seed",
            "-s",
            help="Training seeds (repeatable).",
        ),
    ] = (0, 1, 2),
    target_flops: Annotated[
        float,
        typer.Option(
            "--target-flops",
            help="Target total FLOPs budget (est) per run.",
            min=1e6,
        ),
    ] = 2e8,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Batch size for nanochat training.",
            min=1,
        ),
    ] = 8,
    sequence_len: Annotated[
        int,
        typer.Option(
            "--sequence-len",
            help="Sequence length for nanochat training.",
            min=8,
        ),
    ] = 256,
    n_layer: Annotated[
        int,
        typer.Option(
            "--n-layer",
            help="Number of transformer layers.",
            min=1,
        ),
    ] = 4,
    n_head: Annotated[
        int,
        typer.Option(
            "--n-head",
            help="Number of attention heads.",
            min=1,
        ),
    ] = 4,
    n_kv_head: Annotated[
        int,
        typer.Option(
            "--n-kv-head",
            help="Number of KV heads (GQA).",
            min=1,
        ),
    ] = 4,
    n_embd: Annotated[
        int,
        typer.Option(
            "--n-embd",
            help="Embedding dimension.",
            min=16,
        ),
    ] = 128,
    optimizer_type: Annotated[
        str,
        typer.Option(
            "--optimizer-type",
            help="nanochat optimizer type (passed through).",
        ),
    ] = "adamw",
    learning_rate: Annotated[
        float,
        typer.Option(
            "--learning-rate",
            help="Base learning rate for nanochat.train.",
            min=1e-8,
        ),
    ] = 6e-4,
    warmup_steps: Annotated[
        int,
        typer.Option(
            "--warmup-steps",
            help="Warmup steps excluded from throughput measurement.",
            min=0,
        ),
    ] = 0,
    log_interval: Annotated[
        int,
        typer.Option(
            "--log-interval",
            help="Train logging interval (steps).",
            min=1,
        ),
    ] = 1,
    auto_download_data: Annotated[
        bool,
        typer.Option(
            "--auto-download-data/--no-auto-download-data",
            help="Auto-download minimal dataset shards if missing.",
        ),
    ] = True,
    min_parquet_files: Annotated[
        int,
        typer.Option(
            "--min-parquet-files",
            help="Minimum number of parquet shards required (>=2 recommended).",
            min=2,
        ),
    ] = 2,
    artifacts_dir: Annotated[
        Path,
        typer.Option(
            "--artifacts-dir",
            help="Base directory for artifacts (default: artifacts/).",
        ),
    ] = Path("artifacts"),
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help="Suite run identifier (directory name). Defaults to YYYYMMDD_HHMMSS.",
        ),
    ] = None,
    timeout_s: Annotated[
        float,
        typer.Option(
            "--timeout-s",
            help="Per-run timeout (seconds) for subprocess invocations.",
            min=1.0,
        ),
    ] = 1800.0,
):
    """Compute per-head stability/error bars across multiple seeds for a few small nanochat configs.

    Note: the FlexAttention variant requires CUDA and is skipped otherwise.
    """
    seeds = list(seeds)
    if not seeds:
        raise typer.BadParameter("--seed must be provided at least once")

    suite_run_id = run_id or _default_run_id()
    suite_dir = artifacts_dir / "bench" / "feature_ablate" / "per_head_metrics" / suite_run_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = suite_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "suite": "per_head_metrics",
        "run_id": suite_run_id,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git": _get_git_info(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "argv": sys.argv,
        "device": device,
        "seeds": [int(s) for s in seeds],
        "train_config": {
            "batch_size": int(batch_size),
            "sequence_len": int(sequence_len),
            "n_layer": int(n_layer),
            "n_head": int(n_head),
            "n_kv_head": int(n_kv_head),
            "n_embd": int(n_embd),
            "optimizer_type": str(optimizer_type),
            "learning_rate": float(learning_rate),
            "warmup_steps": int(warmup_steps),
            "log_interval": int(log_interval),
            "target_flops": float(target_flops),
        },
    }

    variants: list[dict[str, Any]] = [
        {
            "key": "standard",
            "label": "standard (SDPA)",
            "attention_type": "standard",
            "use_flex_attention": False,
            "extra_flags": ["--standard-record-attn-entropy"],
        },
        {
            "key": "standard_flex",
            "label": "standard (FlexAttention)",
            "attention_type": "standard",
            "use_flex_attention": True,
            "extra_flags": ["--standard-record-attn-entropy"],
        },
        {
            "key": "tropical",
            "label": "tropical (margins)",
            "attention_type": "tropical",
            "use_flex_attention": False,
            "extra_flags": ["--tropical-record-margins"],
        },
    ]

    def _run_train(variant: dict[str, Any], *, seed: int) -> dict[str, Any]:
        run_topic = f"feature_ablate/per_head_metrics/{suite_run_id}/{variant['key']}"
        run_id_local = f"seed_{seed}"
        train_cmd = [
            sys.executable,
            "-m",
            "nanochat.train",
            "--device",
            device,
            "--seed",
            str(seed),
            "--batch-size",
            str(batch_size),
            "--sequence-len",
            str(sequence_len),
            "--n-layer",
            str(n_layer),
            "--n-head",
            str(n_head),
            "--n-kv-head",
            str(n_kv_head),
            "--n-embd",
            str(n_embd),
            "--learning-rate",
            str(learning_rate),
            "--optimizer-type",
            str(optimizer_type),
            "--attention-type",
            str(variant["attention_type"]),
            "--target-flops",
            str(float(target_flops)),
            "--warmup-steps",
            str(int(warmup_steps)),
            "--log-interval",
            str(int(log_interval)),
            "--artifacts-dir",
            str(artifacts_dir),
            "--artifacts-kind",
            "bench",
            "--artifacts-topic",
            run_topic,
            "--run-id",
            run_id_local,
            *list(variant.get("extra_flags", [])),
        ]
        if bool(variant.get("use_flex_attention", False)):
            train_cmd.append("--use-flex-attention")
        if auto_download_data:
            train_cmd.extend(["--auto-download-data", "--min-parquet-files", str(min_parquet_files)])

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(  # nosec B603
                train_cmd,
                capture_output=True,
                text=True,
                timeout=float(timeout_s),
                check=False,
            )
            stdout = proc.stdout
            stderr = proc.stderr
            returncode = int(proc.returncode)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            returncode = 124
        t1 = time.perf_counter()

        stdout_path = logs_dir / f"nanochat_{variant['key']}_seed_{seed}.stdout.txt"
        stderr_path = logs_dir / f"nanochat_{variant['key']}_seed_{seed}.stderr.txt"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

        summary_path = artifacts_dir / "bench" / run_topic / run_id_local / "summary.json"
        status = "ok" if returncode == 0 and summary_path.exists() else ("timeout" if returncode == 124 else "error")

        final_loss: float | None = None
        tokens_s: float | None = None
        tflops_s: float | None = None
        peak_mem_gb: float | None = None
        use_flex_actual: bool | None = None
        entropy_head_mean: list[float] | None = None
        tropical_gamma_head_mean: list[float] | None = None
        if status == "ok":
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            res = payload.get("results", {}) if isinstance(payload.get("results"), dict) else {}
            cfg = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
            use_flex_actual = bool(cfg.get("use_flex_attention", False))

            losses = res.get("losses")
            if isinstance(losses, list) and losses:
                try:
                    final_loss = float(losses[-1])
                except Exception:
                    final_loss = None
            if isinstance(res.get("tokens_per_second"), int | float):
                tokens_s = float(res["tokens_per_second"])
            if isinstance(res.get("tflops_per_second_est"), int | float):
                tflops_s = float(res["tflops_per_second_est"])
            if isinstance(res.get("peak_memory_allocated_gb"), int | float):
                peak_mem_gb = float(res["peak_memory_allocated_gb"])

            entropy = res.get("attention_entropy")
            if isinstance(entropy, dict) and isinstance(entropy.get("head_mean"), list):
                try:
                    entropy_head_mean = [float(x) for x in entropy["head_mean"]]
                except Exception:
                    entropy_head_mean = None
            tropical = res.get("tropical_margins")
            if isinstance(tropical, dict) and isinstance(tropical.get("head_mean"), list):
                try:
                    tropical_gamma_head_mean = [float(x) for x in tropical["head_mean"]]
                except Exception:
                    tropical_gamma_head_mean = None

            expect_flex = bool(variant.get("use_flex_attention", False))
            if expect_flex != bool(use_flex_actual):
                status = "mismatch"

        return {
            "variant": str(variant["key"]),
            "seed": int(seed),
            "status": status,
            "returncode": int(returncode),
            "duration_s": float(t1 - t0),
            "command": shlex.join(train_cmd),
            "stdout_path": str(stdout_path.relative_to(artifacts_dir)),
            "stderr_path": str(stderr_path.relative_to(artifacts_dir)),
            "summary_path": str(summary_path.relative_to(artifacts_dir)) if summary_path.exists() else None,
            "final_loss": final_loss,
            "tokens_per_second": tokens_s,
            "tflops_per_second_est": tflops_s,
            "peak_memory_allocated_gb": peak_mem_gb,
            "use_flex_attention": use_flex_actual,
            "attention_entropy_head_mean": entropy_head_mean,
            "tropical_gamma_head_mean": tropical_gamma_head_mean,
        }

    results: list[dict[str, Any]] = []
    cuda_available = False
    if device == "auto":
        try:
            import torch
        except Exception:
            cuda_available = False
        else:
            cuda_available = bool(torch.cuda.is_available())
    flex_capable_device = device == "cuda" or (device == "auto" and cuda_available)
    runnable_variants = [v for v in variants if not bool(v.get("use_flex_attention", False)) or flex_capable_device]
    skipped_variants = [v for v in variants if v not in runnable_variants]
    for v in skipped_variants:
        for seed in seeds:
            results.append(
                {
                    "variant": str(v["key"]),
                    "seed": int(seed),
                    "status": "skipped",
                    "returncode": None,
                    "duration_s": 0.0,
                    "command": None,
                    "stdout_path": None,
                    "stderr_path": None,
                    "summary_path": None,
                    "final_loss": None,
                    "tokens_per_second": None,
                    "tflops_per_second_est": None,
                    "peak_memory_allocated_gb": None,
                    "use_flex_attention": None,
                    "attention_entropy_head_mean": None,
                    "tropical_gamma_head_mean": None,
                    "note": f"variant requires CUDA (device={device!r})",
                }
            )

    total = len(runnable_variants) * len(seeds)
    with Progress(
        TextColumn("[bold cyan]per-head metrics[/bold cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("runs", total=total)
        for variant in runnable_variants:
            console.print(Panel(f"[bold]nanochat[/bold] variant={variant['key']!r}", box=box.ROUNDED))
            for seed in seeds:
                results.append(_run_train(variant, seed=int(seed)))
                prog.advance(task)

    variants_out: list[dict[str, Any]] = []
    for variant in variants:
        key = str(variant["key"])
        runs = [r for r in results if r.get("variant") == key]
        loss_vals = [float(r["final_loss"]) for r in runs if isinstance(r.get("final_loss"), int | float)]
        loss_stats = _summary_stats(loss_vals)

        entropy_agg = _aggregate_per_head(
            [
                (int(r["seed"]), r["attention_entropy_head_mean"])
                for r in runs
                if isinstance(r.get("seed"), int) and isinstance(r.get("attention_entropy_head_mean"), list)
            ]
        )
        gamma_agg = _aggregate_per_head(
            [
                (int(r["seed"]), r["tropical_gamma_head_mean"])
                for r in runs
                if isinstance(r.get("seed"), int) and isinstance(r.get("tropical_gamma_head_mean"), list)
            ]
        )
        metrics: dict[str, Any] = {}
        if entropy_agg is not None:
            metrics["attention_entropy"] = entropy_agg
        if gamma_agg is not None:
            metrics["tropical_margin_gamma"] = gamma_agg

        variants_out.append(
            {
                "variant": key,
                "label": str(variant.get("label", key)),
                "attention_type": str(variant.get("attention_type")),
                "expected_use_flex_attention": bool(variant.get("use_flex_attention", False)),
                "runs": runs,
                "final_loss": loss_stats,
                "metrics": metrics,
            }
        )

    def _md_row(values: list[str]) -> str:
        return "| " + " | ".join(values) + " |"

    def _md_per_head_table(metric: dict[str, Any], *, value_name: str) -> list[str]:
        n_head = int(metric.get("n_head", 0))
        means = metric.get("mean", [])
        stds = metric.get("std", [])
        ci95s = metric.get("ci95", [])
        lines: list[str] = []
        lines.append(_md_row(["head", value_name, "std", "ci95"]))
        lines.append(_md_row(["---"] * 4))
        for i in range(n_head):
            mean = means[i] if i < len(means) else float("nan")
            std = stds[i] if i < len(stds) else float("nan")
            ci = ci95s[i] if i < len(ci95s) else None
            lines.append(
                _md_row(
                    [
                        str(i),
                        f"{float(mean):.6f}" if isinstance(mean, int | float) and math.isfinite(float(mean)) else "n/a",
                        f"{float(std):.6f}" if isinstance(std, int | float) and math.isfinite(float(std)) else "n/a",
                        f"{float(ci):.6f}" if isinstance(ci, int | float) and math.isfinite(float(ci)) else "n/a",
                    ]
                )
            )
        return lines

    md_lines: list[str] = []
    md_lines.append("# Per-head metrics suite")
    md_lines.append("")
    md_lines.append(f"- Run ID: `{suite_run_id}`")
    md_lines.append(f"- Device: `{device}`")
    md_lines.append(f"- Seeds: `{', '.join(str(s) for s in seeds)}`")
    md_lines.append(f"- Target FLOPs/run (est): `{float(target_flops):.3e}`")
    md_lines.append("")
    md_lines.append("## Variants")
    md_lines.append("")

    for v in variants_out:
        md_lines.append(f"### {v['variant']}")
        md_lines.append("")
        md_lines.append(f"- label: `{v['label']}`")
        md_lines.append(f"- attention_type: `{v['attention_type']}`")
        md_lines.append(f"- expected_use_flex_attention: `{v['expected_use_flex_attention']}`")
        md_lines.append(
            f"- final_loss: mean=`{v['final_loss']['mean']}` std=`{v['final_loss']['std']}` ci95=`{v['final_loss']['ci95']}` n=`{v['final_loss']['n']}`"
        )
        md_lines.append("")
        metrics = v.get("metrics", {})
        if isinstance(metrics, dict) and metrics.get("attention_entropy") is not None:
            md_lines.append("#### attention_entropy (per head)")
            md_lines.append("")
            md_lines.extend(_md_per_head_table(metrics["attention_entropy"], value_name="entropy_mean"))
            md_lines.append("")
        if isinstance(metrics, dict) and metrics.get("tropical_margin_gamma") is not None:
            md_lines.append("#### tropical margin gamma (per head)")
            md_lines.append("")
            md_lines.extend(_md_per_head_table(metrics["tropical_margin_gamma"], value_name="gamma_mean"))
            md_lines.append("")

        md_lines.append("#### Runs")
        md_lines.append("")
        md_lines.append(_md_row(["seed", "status", "final_loss", "summary"]))
        md_lines.append(_md_row(["---"] * 4))
        for r in v.get("runs", []):
            md_lines.append(
                _md_row(
                    [
                        str(r.get("seed")),
                        str(r.get("status")),
                        f"{float(r['final_loss']):.6f}" if isinstance(r.get("final_loss"), int | float) else "n/a",
                        str(r.get("summary_path") or "n/a"),
                    ]
                )
            )
        md_lines.append("")

    md_lines.append("## Command")
    md_lines.append("")
    md_lines.append("```bash")
    md_lines.append(shlex.join(sys.argv))
    md_lines.append("```")
    report_md = "\n".join(md_lines) + "\n"

    summary = {
        "schema_version": "mgr.bench.per_head_metrics.v1",
        "meta": meta,
        "variants": variants_out,
    }

    for v in variants_out:
        metrics = v.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        panel_title = f"[bold]{v['variant']}[/bold] ({v['label']})"
        console.print(Panel(panel_title, box=box.ROUNDED))

        if metrics.get("attention_entropy") is not None:
            ent = metrics["attention_entropy"]
            table = Table(title="attention entropy per head (mean ± ci95)", box=box.ROUNDED)
            table.add_column("head", justify="right", style="cyan")
            table.add_column("mean", justify="right")
            table.add_column("std", justify="right")
            table.add_column("ci95", justify="right")
            for i in range(int(ent.get("n_head", 0))):
                mean = ent["mean"][i]
                std = ent["std"][i]
                ci = ent["ci95"][i]
                table.add_row(
                    str(i),
                    f"{float(mean):.6f}" if math.isfinite(float(mean)) else "n/a",
                    f"{float(std):.6f}" if math.isfinite(float(std)) else "n/a",
                    f"{float(ci):.6f}" if isinstance(ci, int | float) and math.isfinite(float(ci)) else "n/a",
                )
            console.print(table)

        if metrics.get("tropical_margin_gamma") is not None:
            gam = metrics["tropical_margin_gamma"]
            table = Table(title="tropical gamma per head (mean ± ci95)", box=box.ROUNDED)
            table.add_column("head", justify="right", style="cyan")
            table.add_column("mean", justify="right")
            table.add_column("std", justify="right")
            table.add_column("ci95", justify="right")
            for i in range(int(gam.get("n_head", 0))):
                mean = gam["mean"][i]
                std = gam["std"][i]
                ci = gam["ci95"][i]
                table.add_row(
                    str(i),
                    f"{float(mean):.6f}" if math.isfinite(float(mean)) else "n/a",
                    f"{float(std):.6f}" if math.isfinite(float(std)) else "n/a",
                    f"{float(ci):.6f}" if isinstance(ci, int | float) and math.isfinite(float(ci)) else "n/a",
                )
            console.print(table)

    _write_artifacts(suite_dir, summary=summary, report_md=report_md)
    console.print(f"[dim]Wrote suite artifacts → {suite_dir}[/dim]")


@app.command()
def regressions(
    baseline: Annotated[
        Path,
        typer.Option(
            "--baseline",
            "-b",
            help="Baseline run directory or summary.json (relative paths also searched under --artifacts-dir).",
        ),
    ],
    candidate: Annotated[
        Path,
        typer.Option(
            "--candidate",
            "-c",
            help="Candidate run directory or summary.json (relative paths also searched under --artifacts-dir).",
        ),
    ],
    baseline_variant: Annotated[
        str | None,
        typer.Option(
            "--baseline-variant",
            help="Optional sub-run selector inside baseline (e.g. attention_type in suite summaries; or 'sdpa'/'flex' in flex perf summaries).",
        ),
    ] = None,
    candidate_variant: Annotated[
        str | None,
        typer.Option(
            "--candidate-variant",
            help="Optional sub-run selector inside candidate (e.g. attention_type in suite summaries; or 'sdpa'/'flex' in flex perf summaries).",
        ),
    ] = None,
    loss_abs: Annotated[
        float,
        typer.Option(
            "--loss-abs",
            help="Absolute loss regression threshold (candidate > baseline + loss_abs).",
            min=0.0,
        ),
    ] = 0.01,
    loss_rel: Annotated[
        float,
        typer.Option(
            "--loss-rel",
            help="Relative loss regression threshold (candidate > baseline*(1+loss_rel)).",
            min=0.0,
        ),
    ] = 0.01,
    throughput_rel: Annotated[
        float,
        typer.Option(
            "--throughput-rel",
            help="Relative throughput regression threshold (candidate < baseline*(1-throughput_rel)).",
            min=0.0,
            max=1.0,
        ),
    ] = 0.05,
    tflops_rel: Annotated[
        float,
        typer.Option(
            "--tflops-rel",
            help="Relative TFLOP/s regression threshold (candidate < baseline*(1-tflops_rel)).",
            min=0.0,
            max=1.0,
        ),
    ] = 0.05,
    memory_rel: Annotated[
        float,
        typer.Option(
            "--memory-rel",
            help="Relative memory regression threshold (candidate > baseline*(1+memory_rel)).",
            min=0.0,
            max=10.0,
        ),
    ] = 0.05,
    artifacts_dir: Annotated[
        Path,
        typer.Option(
            "--artifacts-dir",
            help="Artifacts base directory (default: artifacts/).",
        ),
    ] = Path("artifacts"),
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help="Regression report run id (directory name). Defaults to YYYYMMDD_HHMMSS.",
        ),
    ] = None,
    write_artifacts: Annotated[
        bool,
        typer.Option(
            "--write-artifacts/--no-write-artifacts",
            help="Write summary.json + run.md under artifacts/regressions/<run_id>/.",
        ),
    ] = True,
    html: Annotated[
        bool,
        typer.Option(
            "--html/--no-html",
            help="Also write a minimal HTML report alongside run.md (when --write-artifacts is on).",
        ),
    ] = True,
    fail_on_regression: Annotated[
        bool,
        typer.Option(
            "--fail-on-regression/--no-fail-on-regression",
            help="Exit with code 1 if any metric is flagged as a regression (useful for guardrails/CI).",
        ),
    ] = False,
    fail_on_missing: Annotated[
        bool,
        typer.Option(
            "--fail-on-missing/--no-fail-on-missing",
            help="Also treat missing metrics as failures when --fail-on-regression is enabled.",
        ),
    ] = False,
    cert_degrade_factor: Annotated[
        float,
        typer.Option(
            "--cert-degrade-factor",
            help=(
                "Certificate mode: flag a still-passing invariant as a regression when its "
                "measured value degrades by more than this factor vs the baseline snapshot "
                "(e.g. reversible round-trip error growing 10x while still under tolerance)."
            ),
            min=1.0,
        ),
    ] = 10.0,
):
    """Compare two artifact snapshots and highlight regressions/improvements.

    Supports common `summary.json` shapes used in this repo:
    - nanochat.train run summaries (results.losses, tokens_per_second, tflops_per_second_est, peak_memory_allocated_gb)
    - bench suite summaries (top-level runs[]; select a sub-run via --*-variant=attention_type)
    - flex perf summaries (results.tokens_per_s / results.peak_mem_mb; select via --*-variant=sdpa|flex)
    - `mgr certify` certificate summaries (kind == "certify"): a green->red
      check flip is a regression, and a still-passing measured invariant that
      degraded by more than --cert-degrade-factor vs baseline is flagged.
      Tolerances follow a tightening-only policy: loosening one requires an
      explicit justification note in the bead/commit.
    """
    baseline_path = _resolve_summary_path(baseline, artifacts_dir=artifacts_dir)
    candidate_path = _resolve_summary_path(candidate, artifacts_dir=artifacts_dir)

    base_obj = json.loads(baseline_path.read_text(encoding="utf-8"))
    cand_obj = json.loads(candidate_path.read_text(encoding="utf-8"))
    if not isinstance(base_obj, dict) or not isinstance(cand_obj, dict):
        raise typer.BadParameter("Expected both summaries to be JSON objects (dicts).")

    base_meta = _summarize_provenance(base_obj)
    cand_meta = _summarize_provenance(cand_obj)
    if (base_obj.get("kind") == "certify") != (cand_obj.get("kind") == "certify"):
        raise typer.BadParameter(
            "Snapshot kind mismatch: comparing a 'certify' certificate against a "
            "non-certificate snapshot is not meaningful."
        )

    if base_obj.get("kind") == "certify":
        comparisons: list[dict[str, Any]] = _certify_comparisons(
            base_obj, cand_obj, degrade_factor=cert_degrade_factor
        )
    else:
        metrics = [
            ("final_loss", "Final loss", "lower"),
            ("tokens_per_second", "Tokens/s", "higher"),
            ("tflops_per_second_est", "TFLOP/s (est)", "higher"),
            ("peak_memory_allocated_gb", "Peak mem (GB)", "lower"),
        ]

        comparisons = []
        for key, label, direction in metrics:
            b = _extract_metric(base_obj, metric=key, variant=baseline_variant)
            c = _extract_metric(cand_obj, metric=key, variant=candidate_variant)
            if b is None or c is None:
                comparisons.append(
                    {
                        "metric": key,
                        "label": label,
                        "direction": direction,
                        "baseline": b,
                        "candidate": c,
                        "delta": None,
                        "delta_rel": None,
                        "status": "missing",
                    }
                )
                continue

            delta = float(c - b)
            delta_rel = float(delta / b) if b != 0 else None

            status = "ok"
            if direction == "lower":
                if c > b + loss_abs or (delta_rel is not None and delta_rel > loss_rel):
                    status = "regression"
            else:
                thr = tflops_rel if key.startswith("tflops") else throughput_rel
                if c < b * (1.0 - thr):
                    status = "regression"
            if direction == "lower" and key.startswith("peak_memory"):
                if c > b * (1.0 + memory_rel):
                    status = "regression"

            comparisons.append(
                {
                    "metric": key,
                    "label": label,
                    "direction": direction,
                    "baseline": b,
                    "candidate": c,
                    "delta": delta,
                    "delta_rel": delta_rel,
                    "status": status,
                }
            )


    # Rich output
    header = Table.grid(padding=(0, 2))
    header.add_column(justify="left")
    header.add_column(justify="left")
    header.add_row("[bold]Baseline[/bold]", str(baseline_path))
    header.add_row("[bold]Candidate[/bold]", str(candidate_path))
    if baseline_variant or candidate_variant:
        header.add_row("[bold]Variants[/bold]", f"baseline={baseline_variant!r} candidate={candidate_variant!r}")
    console.print(Panel(header, title="Regression Diff", border_style="cyan"))

    meta_t = Table(title="Run provenance", box=box.SIMPLE_HEAVY)
    meta_t.add_column("field")
    meta_t.add_column("baseline", overflow="fold")
    meta_t.add_column("candidate", overflow="fold")
    for k in ("run_id", "device", "commit", "dirty", "attention_type", "use_flex_attention", "compile"):
        meta_t.add_row(str(k), str(base_meta.get(k)), str(cand_meta.get(k)))
    console.print(meta_t)

    loss_spark_base = _sparkline(_extract_loss_series(base_obj), width=24)
    loss_spark_cand = _sparkline(_extract_loss_series(cand_obj), width=24)

    t = Table(title="Metrics", box=box.SIMPLE_HEAVY)
    t.add_column("metric")
    t.add_column("baseline", justify="right")
    t.add_column("candidate", justify="right")
    t.add_column("Δ", justify="right")
    t.add_column("Δ%", justify="right")
    t.add_column("status", justify="right")
    for row in comparisons:
        b = row["baseline"]
        c = row["candidate"]
        d = row["delta"]
        dr = row["delta_rel"]
        status = row["status"]
        style = {"regression": "bold red", "ok": "green", "missing": "dim"}.get(status, "")
        t.add_row(
            row["label"],
            "-" if b is None else f"{b:.6g}",
            "-" if c is None else f"{c:.6g}",
            "-" if d is None else f"{d:+.6g}",
            "-" if dr is None else f"{(100.0 * dr):+.2f}%",
            f"[{style}]{status}[/{style}]" if style else status,
        )
    console.print(t)
    if loss_spark_base or loss_spark_cand:
        spark_t = Table(title="Loss sparklines (tail)", box=box.SIMPLE_HEAVY)
        spark_t.add_column("baseline")
        spark_t.add_column("candidate")
        spark_t.add_row(loss_spark_base or "-", loss_spark_cand or "-")
        console.print(spark_t)

    regressions_found = [row for row in comparisons if row.get("status") == "regression"]
    missing_found = [row for row in comparisons if row.get("status") == "missing"]

    thresholds = {
        "loss_abs": float(loss_abs),
        "loss_rel": float(loss_rel),
        "throughput_rel": float(throughput_rel),
        "tflops_rel": float(tflops_rel),
        "memory_rel": float(memory_rel),
        "cert_degrade_factor": float(cert_degrade_factor),
    }

    report_run_id = run_id or _default_run_id()
    report_dir = artifacts_dir / "regressions" / report_run_id

    md_lines: list[str] = []
    md_lines.append("# Regression Report")
    md_lines.append("")
    md_lines.append(f"- generated_at: `{time.strftime('%Y-%m-%d %H:%M:%S %Z')}`")
    md_lines.append(f"- baseline: `{baseline_path}`")
    md_lines.append(f"- candidate: `{candidate_path}`")
    if baseline_variant or candidate_variant:
        md_lines.append(f"- variants: baseline=`{baseline_variant}` candidate=`{candidate_variant}`")
    md_lines.append("")
    md_lines.append("## Thresholds")
    md_lines.append("")
    md_lines.append("```json")
    md_lines.append(json.dumps(thresholds, indent=2, sort_keys=True))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## Metrics")
    md_lines.append("")
    md_lines.append("| metric | baseline | candidate | delta | delta% | status |")
    md_lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for row in comparisons:
        b = row["baseline"]
        c = row["candidate"]
        d = row["delta"]
        dr = row["delta_rel"]
        md_lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    "-" if b is None else f"{b:.6g}",
                    "-" if c is None else f"{c:.6g}",
                    "-" if d is None else f"{d:+.6g}",
                    "-" if dr is None else f"{(100.0 * dr):+.2f}%",
                    row["status"],
                ]
            )
            + " |"
        )
    if loss_spark_base or loss_spark_cand:
        md_lines.append("")
        md_lines.append("## Loss sparklines (tail)")
        md_lines.append("")
        md_lines.append(f"- baseline: `{loss_spark_base}`")
        md_lines.append(f"- candidate: `{loss_spark_cand}`")

    md_lines.append("")
    md_lines.append("## Command")
    md_lines.append("")
    md_lines.append("```bash")
    md_lines.append(shlex.join(sys.argv))
    md_lines.append("```")
    report_md = "\n".join(md_lines) + "\n"

    summary = {
        "meta": {
            "run_id": report_run_id,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "git": _get_git_info(),
            "argv": sys.argv,
            "baseline_path": str(baseline_path),
            "candidate_path": str(candidate_path),
            "baseline_variant": baseline_variant,
            "candidate_variant": candidate_variant,
        },
        "thresholds": thresholds,
        "baseline": base_meta,
        "candidate": cand_meta,
        "comparisons": comparisons,
    }

    if write_artifacts:
        _write_artifacts(report_dir, summary=summary, report_md=report_md)
        if html:
            html_table = [
                "<table><thead><tr><th>metric</th><th>baseline</th><th>candidate</th><th>delta</th><th>delta%</th><th>status</th></tr></thead><tbody>"
            ]
            for row in comparisons:
                b = row["baseline"]
                c = row["candidate"]
                d = row["delta"]
                dr = row["delta_rel"]
                status = row["status"]
                cls = "regression" if status == "regression" else ("ok" if status == "ok" else "missing")
                html_table.append(
                    "<tr class='"
                    + cls
                    + "'>"
                    + "".join(
                        f"<td>{cell}</td>"
                        for cell in [
                            row["label"],
                            "-" if b is None else f"{b:.6g}",
                            "-" if c is None else f"{c:.6g}",
                            "-" if d is None else f"{d:+.6g}",
                            "-" if dr is None else f"{(100.0 * dr):+.2f}%",
                            status,
                        ]
                    )
                    + "</tr>"
                )
            html_table.append("</tbody></table>")
            html_doc = "\n".join(
                [
                    "<!doctype html>",
                    "<meta charset='utf-8'/>",
                    "<title>Regression Report</title>",
                    "<style>",
                    "body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:24px;}",
                    "table{border-collapse:collapse;width:100%;}",
                    "th,td{border:1px solid #ddd;padding:8px;}",
                    "th{background:#f6f6f6;text-align:left;}",
                    "tr.ok td{background:#e9f7ef;}",
                    "tr.regression td{background:#fdecea;}",
                    "tr.missing td{color:#666;}",
                    "</style>",
                    "<h1>Regression Report</h1>",
                    f"<p><b>Baseline</b>: <code>{baseline_path}</code></p>",
                    f"<p><b>Candidate</b>: <code>{candidate_path}</code></p>",
                    "\n".join(html_table),
                ]
            )
            (report_dir / "report.html").write_text(html_doc + "\n", encoding="utf-8")
        console.print(f"[bold green]Wrote regression report[/bold green] → {report_dir}")

    if fail_on_regression:
        failing_rows = list(regressions_found)
        if fail_on_missing:
            failing_rows.extend(missing_found)

        if failing_rows:
            failing_metrics = ", ".join(row.get("metric", "?") for row in failing_rows)
            console.print(
                Panel(
                    f"[bold red]Regressions detected[/bold red]: {failing_metrics}",
                    title="Guardrail",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# mgr certify — per-mechanism mathematical invariant checks (br: model_guided_research-5ki.1)
#
# Verifies each nanochat attention mechanism's DEFINING mathematical invariant
# directly against the production torch modules, with dtype-aware tolerances.
# The shared causality certificate (no gradient flow from the future) runs for
# every mechanism. Exit code is nonzero if any check fails.
# ---------------------------------------------------------------------------

_CERTIFY_MECHANISMS: list[str] = [
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
]


def _certify_tiny_config(mechanism: str):
    """Build a tiny GPTConfig for `mechanism` satisfying its structural constraints."""
    from nanochat.gpt import GPTConfig

    kwargs: dict[str, Any] = dict(
        sequence_len=64,
        vocab_size=128,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=64,  # head_dim = 32: divisible by 4 (quaternion) and 8 (octonion), even (RoPE/gauge)
        attention_type=mechanism,
    )
    if mechanism == "reversible":
        # Sub-attention gets n_head//2 = 1 query head; n_kv_head must divide it.
        kwargs["n_kv_head"] = 1
    if mechanism == "tropical":
        kwargs["tropical_record_margins"] = True
    return GPTConfig(**kwargs)


def _certify_cos_sin(T: int, head_dim: int, device: Any, dtype: Any) -> tuple[Any, Any]:
    """Rotary cos/sin matching GPT._precompute_rotary_embeddings, in the requested dtype."""
    import torch

    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000 ** (channel_range / head_dim))
    t = torch.arange(T, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos().to(dtype), freqs.sin().to(dtype)
    return cos[None, :, None, :], sin[None, :, None, :]


def _run_certify_checks(
    mechanisms: list[str],
    *,
    device_str: str = "cpu",
    dtype_str: str = "fp32",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Run the invariant checks for the requested mechanisms and return one record per
    check: {mechanism, check, family, status, measured, tolerance, comparator,
    duration_ms, detail}. status is "pass" | "fail" | "error".

    Pass criterion: measured <= tolerance (comparator "le", default) or
    measured >= tolerance (comparator "ge", used for separation witnesses that
    must be bounded AWAY from zero, e.g. octonion non-associativity).
    """
    import copy

    import torch

    from nanochat.gpt import Block
    from nanochat.model_utils import apply_rotary_emb, causal_attn_mask
    from nanochat.model_utils import norm as rmsnorm

    device = torch.device(device_str)
    if dtype_str not in {"fp32", "bf16"}:
        raise ValueError(f"dtype must be 'fp32' or 'bf16', got {dtype_str!r}")
    dtype = torch.float32 if dtype_str == "fp32" else torch.bfloat16

    def fp_tol(base: float) -> float:
        # bf16 has ~8 bits of mantissa; loosen float-comparison tolerances accordingly.
        return base if dtype_str == "fp32" else max(base * 512.0, 5e-2)

    checks: list[dict[str, Any]] = []

    def add_check(
        mechanism: str,
        name: str,
        family: str,
        fn: Any,
        *,
        tolerance: float,
        comparator: str = "le",
        detail: str = "",
    ) -> None:
        t0 = time.perf_counter()
        try:
            measured = float(fn())
            if comparator == "le":
                status = "pass" if measured <= tolerance else "fail"
            else:
                status = "pass" if measured >= tolerance else "fail"
            error = None
        except Exception as exc:  # noqa: BLE001 - report, don't crash the suite
            measured = float("nan")
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        duration_ms = (time.perf_counter() - t0) * 1000.0
        checks.append(
            {
                "mechanism": mechanism,
                "check": name,
                "family": family,
                "status": status,
                "measured": measured,
                "tolerance": tolerance,
                "comparator": comparator,
                "duration_ms": round(duration_ms, 3),
                "detail": detail if error is None else f"{detail} [{error}]".strip(),
            }
        )

    def make_block(mechanism: str) -> Any:
        torch.manual_seed(seed)
        cfg = _certify_tiny_config(mechanism)
        block = Block(cfg, 0).to(device=device, dtype=dtype)
        block.eval()
        return block, cfg

    # ----- shared causality certificate (the single most important check) -----
    def causality_measure(mechanism: str) -> float:
        block, cfg = make_block(mechanism)
        T = 8
        head_dim = cfg.n_embd // cfg.n_head
        cos_sin = _certify_cos_sin(T, head_dim, device, dtype)
        x = torch.randn(1, T, cfg.n_embd, device=device, dtype=dtype, requires_grad=True)
        y = block(x, cos_sin, None)
        worst = 0.0
        past_signal = 0.0
        for t in (0, T // 2, T - 2):
            (g,) = torch.autograd.grad(y[0, t].sum(), x, retain_graph=True, allow_unused=True)
            if g is None:
                continue  # no dependence on x at all: past_signal stays 0 -> vacuity guard trips below
            if t + 1 < T:
                worst = max(worst, float(g[0, t + 1 :].abs().max()))
            past_signal = max(past_signal, float(g[0, : t + 1].abs().max()))
        if past_signal == 0.0:
            # Vacuity guard: an output that depends on NO input would trivially satisfy
            # the future-gradient condition. That is a broken mechanism, not a causal one.
            return float("inf")
        return worst

    for mech in mechanisms:
        add_check(
            mech,
            "causality_no_future_grad",
            "causality",
            lambda m=mech: causality_measure(m),
            tolerance=1e-12,
            detail=(
                "max |d y[t] / d x[t+k]| over t in {0, T/2, T-2}, k >= 1 (must be exactly 0); "
                "inf = vacuity guard tripped (output has no input dependence at all)"
            ),
        )

    # ----- standard: scaffolding invariants -----
    if "standard" in mechanisms:

        def rope_norm_measure() -> float:
            torch.manual_seed(seed)
            T, H, D = 8, 2, 32
            cos, sin = _certify_cos_sin(T, D, device, torch.float32)
            x = torch.randn(1, T, H, D, device=device)
            y = apply_rotary_emb(x, cos, sin)
            d = D // 2
            pn_in = torch.sqrt(x[..., :d] ** 2 + x[..., d:] ** 2)
            pn_out = torch.sqrt(y[..., :d] ** 2 + y[..., d:] ** 2)
            return float((pn_in - pn_out).abs().max())

        def mask_structure_measure() -> float:
            mismatches = 0
            m = causal_attn_mask(8, 8, device=device)
            mismatches += int((m != torch.tril(torch.ones(8, 8, dtype=torch.bool, device=device))).sum())
            m = causal_attn_mask(1, 9, device=device)
            mismatches += int((~m).sum())
            m = causal_attn_mask(4, 12, device=device)
            expect = torch.zeros(4, 12, dtype=torch.bool, device=device)
            expect[:, :8] = True
            expect[:, 8:] = torch.tril(torch.ones(4, 4, dtype=torch.bool, device=device))
            mismatches += int((m != expect).sum())
            return float(mismatches)

        def rmsnorm_measure() -> float:
            torch.manual_seed(seed)
            x = torch.randn(4, 8, 64, device=device)
            y = rmsnorm(x)
            rms = torch.sqrt((y**2).mean(dim=-1))
            return float((rms - 1.0).abs().max())

        def softmax_rows_measure() -> float:
            torch.manual_seed(seed)
            scores = torch.randn(1, 2, 8, 8, device=device)
            scores = scores.masked_fill(~causal_attn_mask(8, 8, device=device), float("-inf"))
            p = torch.softmax(scores, dim=-1)
            return float((p.sum(dim=-1) - 1.0).abs().max())

        add_check("standard", "rope_pairwise_norm_preservation", "classical", rope_norm_measure, tolerance=1e-5)
        add_check(
            "standard",
            "causal_mask_structure",
            "classical",
            mask_structure_measure,
            tolerance=0.0,
            detail="mismatch count vs spec across train/decode/chunk regimes",
        )
        add_check("standard", "rmsnorm_unit_rms", "classical", rmsnorm_measure, tolerance=1e-3)
        add_check("standard", "softmax_row_stochastic", "classical", softmax_rows_measure, tolerance=1e-6)

    # ----- tropical: 1-Lipschitz, gauge invariance, margin consistency -----
    if "tropical" in mechanisms:
        from nanochat.tropical_attention_torch import tropical_max_plus_attention

        def _trop_inputs():
            torch.manual_seed(seed)
            q = torch.randn(1, 2, 8, 32, device=device)
            k = torch.randn(1, 2, 8, 32, device=device)
            v = torch.randn(1, 2, 8, 32, device=device)
            return q, k, v

        def trop_lipschitz_measure(which: str) -> float:
            q, k, v = _trop_inputs()
            eps = 0.123
            y0, _ = tropical_max_plus_attention(q, k, v, gauge_fix=False, score_center=False, return_margins=False)
            delta = torch.empty_like(q if which == "q" else v).uniform_(-eps, eps)
            if which == "q":
                y1, _ = tropical_max_plus_attention(
                    q + delta, k, v, gauge_fix=False, score_center=False, return_margins=False
                )
            else:
                y1, _ = tropical_max_plus_attention(
                    q, k, v + delta, gauge_fix=False, score_center=False, return_margins=False
                )
            return float((y1 - y0).abs().max() / eps)

        def trop_center_invariance_measure() -> float:
            q, k, v = _trop_inputs()
            y0, _ = tropical_max_plus_attention(q, k, v, gauge_fix=False, score_center=False, return_margins=False)
            y1, _ = tropical_max_plus_attention(q, k, v, gauge_fix=False, score_center=True, return_margins=False)
            diff = y1 - y0  # must be constant across the feature dim (a pure per-query gauge shift)
            spread = diff.amax(dim=-1) - diff.amin(dim=-1)
            return float(spread.abs().max())

        def trop_margin_measure() -> float:
            q, k, v = _trop_inputs()
            _, gamma = tropical_max_plus_attention(q, k, v, gauge_fix=False, score_center=False, return_margins=True)
            if gamma is None:
                raise RuntimeError("tropical_max_plus_attention returned no margins despite return_margins=True")
            scores = torch.max(q.unsqueeze(3) + k.unsqueeze(2), dim=-1).values
            scores = scores.masked_fill(~causal_attn_mask(8, 8, device=device), float("-inf"))
            logits = scores.unsqueeze(-1) + v.unsqueeze(2)
            srt, _ = torch.sort(logits, dim=3, descending=True)
            gamma_bf = (srt[..., 0, :] - srt[..., 1, :]).amin(dim=-1)
            both_finite = torch.isfinite(gamma) & torch.isfinite(gamma_bf)
            inf_mismatch = float((torch.isfinite(gamma) != torch.isfinite(gamma_bf)).sum())
            max_diff = float((gamma - gamma_bf).abs().masked_fill(~both_finite, 0.0).max())
            return max_diff + inf_mismatch

        add_check(
            "tropical",
            "lipschitz_1_sup_norm_q",
            "classical",
            lambda: trop_lipschitz_measure("q"),
            tolerance=1.0 + 1e-6,
            detail="sup-norm output change / eps under q-perturbation",
        )
        add_check(
            "tropical",
            "lipschitz_1_sup_norm_v",
            "classical",
            lambda: trop_lipschitz_measure("v"),
            tolerance=1.0 + 1e-6,
            detail="sup-norm output change / eps under v-perturbation",
        )
        add_check(
            "tropical",
            "score_center_pure_gauge_shift",
            "classical",
            trop_center_invariance_measure,
            tolerance=1e-5,
            detail="centering must shift outputs by a per-query constant only",
        )
        add_check(
            "tropical",
            "margin_matches_bruteforce",
            "classical",
            trop_margin_measure,
            tolerance=1e-6,
            detail="recorded gamma vs sort-based runner-up gap (+ count of inf mismatches)",
        )

        def trop_endpoint_measure() -> float:
            # Reduction to the known special case (bead uvjq): finite-beta
            # (Maslov) tropical attention converges to the EXACT (beta=None)
            # tropical endpoint, with |y_beta - y_inf|_inf bounded by the
            # LSE-max sandwich (log D + log m)/beta (thm-lse-max-sandwich). The
            # measure is the gap as a fraction of the bound (<= 1). fp64: the
            # inequality is exact in real arithmetic.
            torch.manual_seed(seed)
            qd = torch.randn(1, 2, 8, 16, dtype=torch.float64, device=device)
            kd = torch.randn(1, 2, 8, 16, dtype=torch.float64, device=device)
            vd = torch.randn(1, 2, 8, 16, dtype=torch.float64, device=device)
            y_inf, _ = tropical_max_plus_attention(
                qd, kd, vd, gauge_fix=False, score_center=True, return_margins=False, beta=None
            )
            beta = 32.0
            y_b, _ = tropical_max_plus_attention(
                qd, kd, vd, gauge_fix=False, score_center=True, return_margins=False, beta=beta
            )
            bound = (math.log(qd.size(-1)) + math.log(kd.size(2))) / beta
            return float((y_b - y_inf).abs().max() / bound)

        add_check(
            "tropical",
            "maslov_endpoint_within_sandwich",
            "reduction",
            trop_endpoint_measure,
            tolerance=1.0 + 1e-6,
            detail="finite-beta tropical attention -> exact tropical endpoint, gap within the LSE-max sandwich",
        )

        # --- tropical FFN (bead 8gk.8): the certified chain's MLP piece ---
        def _ffn_tiny(ffn_type: str, dtype_=None):
            from nanochat.gpt import GPTConfig as _Cfg
            from nanochat.tropical_attention_torch import TropicalMLP as _TMLP

            torch.manual_seed(seed)
            cfg_f = _Cfg(sequence_len=16, vocab_size=64, n_layer=1, n_head=2, n_kv_head=2, n_embd=16, ffn_type=ffn_type)
            mlp = _TMLP(cfg_f)
            return mlp.double() if dtype_ == torch.float64 else mlp

        def ffn_lipschitz_measure() -> float:
            # fp64: the inequality is EXACT in real arithmetic; fp32 rounding
            # of (x + d) alone can push the ratio a few ulps above 1
            mlp = _ffn_tiny("tropical", dtype_=torch.float64)
            gen = torch.Generator().manual_seed(seed)
            worst = 0.0
            with torch.no_grad():
                for scale in (1e-2, 1.0, 10.0):
                    x = torch.randn(8, 16, generator=gen, dtype=torch.float64)
                    d = torch.randn(8, 16, generator=gen, dtype=torch.float64) * scale
                    num = (mlp(x + d) - mlp(x)).abs().amax(dim=-1)
                    den = d.abs().amax(dim=-1).clamp_min(1e-15)
                    worst = max(worst, float((num / den).max()))
            return worst

        def ffn_collapse_measure() -> float:
            mlp = _ffn_tiny("tropical", dtype_=torch.float64)
            from nanochat.tropical_attention_torch import tropical_maxplus_layer as _layer

            with torch.no_grad():
                m, b2 = mlp.collapsed_weight()
                gen = torch.Generator().manual_seed(seed + 1)
                x = torch.randn(32, 16, generator=gen, dtype=torch.float64)
                return float((mlp(x) - _layer(x, m, b2)).abs().max())

        add_check(
            "tropical",
            "ffn_lipschitz_1_sup_norm",
            "classical",
            ffn_lipschitz_measure,
            tolerance=1.0 + 1e-9,
            detail="pure max-plus FFN, fp64: sup-norm output change / input change (thm-maxplus-ffn-lipschitz)",
        )
        add_check(
            "tropical",
            "ffn_collapse_single_layer",
            "classical",
            ffn_collapse_measure,
            tolerance=1e-9,
            detail="two-stage pure stack equals its collapsed tropical-affine map, fp64 (thm-maxplus-ffn-collapse)",
        )

    # ----- quaternion: algebra laws + rotor norm preservation (fp64) -----
    if "quaternion" in mechanisms:
        from nanochat.quaternion_attention_torch import qconj, qmul, qnormalize

        def _quats() -> tuple[Any, Any, Any]:
            torch.manual_seed(seed)
            a = torch.randn(512, 4, dtype=torch.float64, device=device)
            b = torch.randn(512, 4, dtype=torch.float64, device=device)
            c = torch.randn(512, 4, dtype=torch.float64, device=device)
            return a, b, c

        def q_assoc_measure() -> float:
            a, b, c = _quats()
            return float((qmul(qmul(a, b), c) - qmul(a, qmul(b, c))).abs().max())

        def q_norm_mult_measure() -> float:
            a, b, _ = _quats()
            lhs = qmul(a, b).norm(dim=-1)
            return float((lhs - a.norm(dim=-1) * b.norm(dim=-1)).abs().max())

        def q_conj_antihom_measure() -> float:
            a, b, _ = _quats()
            return float((qconj(qmul(a, b)) - qmul(qconj(b), qconj(a))).abs().max())

        def q_rotor_norm_measure() -> float:
            a, b, _ = _quats()
            rotor = qnormalize(a)
            return float((qmul(rotor, b).norm(dim=-1) - b.norm(dim=-1)).abs().max())

        add_check("quaternion", "qmul_associativity", "classical", q_assoc_measure, tolerance=1e-10)
        add_check("quaternion", "qmul_norm_multiplicative", "classical", q_norm_mult_measure, tolerance=1e-10)
        add_check("quaternion", "qconj_antihomomorphism", "classical", q_conj_antihom_measure, tolerance=1e-10)
        add_check("quaternion", "rotor_norm_preservation", "classical", q_rotor_norm_measure, tolerance=1e-10)

    # ----- octonion: division-algebra laws + non-associativity witness (fp64) -----
    if "octonion" in mechanisms:
        from nanochat.octonion_attention_torch import oconj, omul

        def _octs() -> tuple[Any, Any, Any]:
            torch.manual_seed(seed)
            a = torch.randn(512, 8, dtype=torch.float64, device=device)
            b = torch.randn(512, 8, dtype=torch.float64, device=device)
            c = torch.randn(512, 8, dtype=torch.float64, device=device)
            return a, b, c

        def o_norm_mult_measure() -> float:
            a, b, _ = _octs()
            return float((omul(a, b).norm(dim=-1) - a.norm(dim=-1) * b.norm(dim=-1)).abs().max())

        def o_alternativity_measure() -> float:
            a, b, _ = _octs()
            left = (omul(a, omul(a, b)) - omul(omul(a, a), b)).abs().max()
            right = (omul(omul(b, a), a) - omul(b, omul(a, a))).abs().max()
            return float(torch.maximum(left, right))

        def o_nonassoc_witness_measure() -> float:
            a, b, c = _octs()
            return float((omul(omul(a, b), c) - omul(a, omul(b, c))).abs().max())

        def o_conj_scalar_measure() -> float:
            a, _, _ = _octs()
            prod = omul(a, oconj(a))
            scalar_err = (prod[..., 0] - a.norm(dim=-1) ** 2).abs().max()
            imag_err = prod[..., 1:].abs().max()
            return float(torch.maximum(scalar_err, imag_err))

        def o_reduces_to_quaternion_measure() -> float:
            # Reduction-to-a-known-mechanism anchor (bead uvjq / the
            # new-mechanism gate): octonion (q, 0) elements form a quaternion
            # subalgebra under Cayley-Dickson, with omul((a,0),(c,0)) ==
            # (qmul(a,c), 0). So octonion attention restricted to this
            # subalgebra IS quaternion attention - the exact special case that
            # makes the mechanism's numbers independently checkable.
            from nanochat.quaternion_attention_torch import qmul

            a, c, _ = _octs()
            aq, cq = a[..., :4], c[..., :4]
            za = torch.cat([aq, torch.zeros_like(aq)], dim=-1)
            zc = torch.cat([cq, torch.zeros_like(cq)], dim=-1)
            prod = omul(za, zc)
            sub_err = (prod[..., :4] - qmul(aq, cq)).abs().max()  # matches qmul on the subalgebra
            leak_err = prod[..., 4:].abs().max()  # stays in the subalgebra (no leakage into the (0,b) part)
            return float(torch.maximum(sub_err, leak_err))

        add_check("octonion", "omul_norm_multiplicative", "classical", o_norm_mult_measure, tolerance=1e-9)
        add_check("octonion", "omul_alternativity", "classical", o_alternativity_measure, tolerance=1e-9)
        add_check(
            "octonion",
            "omul_nonassociativity_witness",
            "classical",
            o_nonassoc_witness_measure,
            tolerance=1e-2,
            comparator="ge",
            detail="associator must be bounded AWAY from zero (guards against an associative shortcut)",
        )
        add_check("octonion", "o_times_conj_is_norm_squared", "classical", o_conj_scalar_measure, tolerance=1e-9)
        add_check(
            "octonion",
            "reduces_to_quaternion_on_subalgebra",
            "reduction",
            o_reduces_to_quaternion_measure,
            tolerance=1e-9,
            detail="omul on the (q,0) Cayley-Dickson subalgebra == qmul (exact reduction to the quaternion mechanism)",
        )

    # ----- reversible: round-trip + custom-autograd gradient parity -----
    if "reversible" in mechanisms:

        def rev_roundtrip_measure() -> float:
            block, cfg = make_block("reversible")
            sb = block.special_block
            T = 8
            cos_sin = _certify_cos_sin(T, 32, device, dtype)
            x = torch.randn(1, T, cfg.n_embd, device=device, dtype=dtype)
            with torch.no_grad():
                y = sb(x, cos_sin, None)
                xr = sb.inverse(y, cos_sin, None)
            return float((x - xr).abs().max())

        def rev_grad_parity_measure() -> float:
            from nanochat.reversible_block_torch import ReversibleFunction

            block, cfg = make_block("reversible")
            sb = block.special_block.float()
            sb_ref = copy.deepcopy(sb)
            T = 8
            cos_sin = _certify_cos_sin(T, 32, device, torch.float32)
            torch.manual_seed(seed + 1)
            x = torch.randn(1, T, cfg.n_embd, device=device)
            w = torch.randn_like(x)

            # TRUE naive reference: run the coupling math directly instead of
            # sb(...), which since a6k3 routes through ReversibleFunction and
            # therefore records no graph for the coupling parameters.
            def naive_forward(mod, x_in):
                x1, x2 = torch.chunk(x_in, 2, dim=-1)
                y1 = x1 + mod.f_block(x2, cos_sin, None)
                y2 = x2 + mod.g_block(y1)
                return torch.cat([y1, y2], dim=-1)

            xa = x.clone().requires_grad_(True)
            loss_a = (naive_forward(sb_ref, xa) * w).sum()
            grads_a = torch.autograd.grad(loss_a, [xa, *sb_ref.parameters()])

            xb = x.clone().requires_grad_(True)
            yb = ReversibleFunction.apply(xb, cos_sin, None, sb.f_block, sb.g_block)
            (yb * w).sum().backward()
            grads_b = [xb.grad, *(p.grad for p in sb.parameters())]

            worst = 0.0
            for ga, gb in zip(grads_a, grads_b, strict=True):
                if gb is None:
                    return float("inf")
                denom = float(ga.abs().max()) + 1e-8
                worst = max(worst, float((ga - gb).abs().max()) / denom)
            return worst

        add_check(
            "reversible",
            "forward_inverse_roundtrip",
            "classical",
            rev_roundtrip_measure,
            tolerance=fp_tol(1e-5),
            detail="max |x - inverse(forward(x))|",
        )
        add_check(
            "reversible",
            "custom_autograd_grad_parity",
            "classical",
            rev_grad_parity_measure,
            tolerance=1e-4,
            detail="max relative grad diff: ReversibleFunction vs naive autograd",
        )

        def _symplectic_block() -> tuple[Any, Any]:
            from nanochat.gpt import GPTConfig

            torch.manual_seed(seed)
            base = _certify_tiny_config("reversible")
            cfg = GPTConfig(**{**vars(base), "reversible_mode": "symplectic"})
            block = Block(cfg, 0).to(device=device, dtype=torch.float64)
            block.eval()
            return block.special_block, cfg

        def rev_symplectic_jacobian_measure() -> float:
            # u55.5 (bead 7vwo): the SHIPPED kick-kick block is an exact
            # symplectic map - J^T Omega J = Omega on the (x1, x2) phase
            # space, fp64, through the real sub-attention energy head.
            sb, cfg = _symplectic_block()
            T = 3
            half = cfg.n_embd // 2
            cos_sin = _certify_cos_sin(T, 32, device, torch.float64)
            dim = T * half

            def chunked(z: torch.Tensor) -> torch.Tensor:
                x1 = z[:dim].reshape(1, T, half)
                x2 = z[dim:].reshape(1, T, half)
                y = sb(torch.cat([x1, x2], dim=-1), cos_sin, None)
                y1, y2 = torch.chunk(y, 2, dim=-1)
                return torch.cat([y1.reshape(-1), y2.reshape(-1)])

            z0 = torch.randn(2 * dim, device=device, dtype=torch.float64)
            jac = torch.autograd.functional.jacobian(chunked, z0)
            omega = torch.zeros(2 * dim, 2 * dim, device=device, dtype=torch.float64)
            omega[:dim, dim:] = torch.eye(dim, device=device, dtype=torch.float64)
            omega[dim:, :dim] = -torch.eye(dim, device=device, dtype=torch.float64)
            return float((jac.T @ omega @ jac - omega).abs().max())

        def rev_energy_drift_separation_measure() -> float:
            # The load-bearing SIGN: the corrected kick-kick (y2 = x2 - grad)
            # holds the coercive H = phi_F + phi_G in a band across 64 tied
            # layers; the ++ form conserves only the non-coercive difference.
            # Measure = band(corrected) / band(++) - far below 1 when the
            # shipped sign is right, ~1 (or worse) if it ever regresses.
            sb, cfg = _symplectic_block()
            f_kick, g_kick = sb.f_block, sb.g_block
            T = 3
            half = cfg.n_embd // 2
            cos_sin = _certify_cos_sin(T, 32, device, torch.float64)

            def band(sign: float) -> float:
                torch.manual_seed(seed + 7)
                x1 = 0.5 * torch.randn(1, T, half, device=device, dtype=torch.float64)
                x2 = 0.5 * torch.randn(1, T, half, device=device, dtype=torch.float64)
                energies = []
                for _ in range(64):
                    y1 = x1 + f_kick(x2, cos_sin, None)
                    y2 = x2 + sign * g_kick(y1)
                    x1, x2 = y1.detach(), y2.detach()
                    with torch.no_grad():
                        energies.append(
                            float(f_kick.energy(x2, cos_sin, None).mean() + g_kick.energy(x1).mean())
                        )
                e = torch.tensor(energies)
                return float(e.max() - e.min())

            return band(-1.0) / max(band(+1.0), 1e-12)

        add_check(
            "reversible",
            "symplectic_jacobian",
            "classical",
            rev_symplectic_jacobian_measure,
            tolerance=fp_tol(1e-9),
            detail="max |J^T Omega J - Omega| of the shipped kick-kick block (fp64)",
        )
        add_check(
            "reversible",
            "energy_drift_separation",
            "classical",
            rev_energy_drift_separation_measure,
            tolerance=0.2,
            detail="energy band ratio over 64 tied layers: corrected sign / ++ sign",
        )

    # ----- gauge: rotation orthogonality, inverse consistency, additivity -----
    if "gauge" in mechanisms:

        def _gauge_setup() -> tuple[Any, Any, Any]:
            block, cfg = make_block("gauge")
            gb = block.special_block.float()
            torch.manual_seed(seed + 2)
            x = torch.randn(1, 8, cfg.n_embd, device=device)
            th = torch.randn(1, 8, cfg.n_embd // 2, device=device)
            return gb, x, th

        def gauge_roundtrip_measure() -> float:
            gb, x, th = _gauge_setup()
            xr = gb._apply_rotations(gb._apply_rotations(x, th), th, inverse=True)
            return float((x - xr).abs().max())

        def gauge_norm_measure() -> float:
            gb, x, th = _gauge_setup()
            y = gb._apply_rotations(x, th)
            pn_in = torch.sqrt(x[..., 0::2] ** 2 + x[..., 1::2] ** 2)
            pn_out = torch.sqrt(y[..., 0::2] ** 2 + y[..., 1::2] ** 2)
            return float((pn_in - pn_out).abs().max())

        def gauge_additivity_measure() -> float:
            gb, x, th = _gauge_setup()
            torch.manual_seed(seed + 12)
            th2 = torch.randn_like(th)
            y_seq = gb._apply_rotations(gb._apply_rotations(x, th), th2)
            y_sum = gb._apply_rotations(x, th + th2)
            return float((y_seq - y_sum).abs().max())

        add_check(
            "gauge",
            "rotation_inverse_roundtrip",
            "classical",
            gauge_roundtrip_measure,
            tolerance=1e-5,
            detail="R(-theta) R(theta) = I (orthogonality of the Givens transport)",
        )
        add_check("gauge", "rotation_pairwise_norm_preservation", "classical", gauge_norm_measure, tolerance=1e-5)
        add_check(
            "gauge",
            "rotation_additivity_cumsum_law",
            "classical",
            gauge_additivity_measure,
            tolerance=1e-5,
            detail="R(t2) R(t1) = R(t1+t2): the property that justifies cumsum-as-transport",
        )

        def gauge_kv_decode_measure() -> float:
            # Cached-mode gauge invariance (bead 7b0.5): the fp32 cumulative-angle
            # lane must rebuild the SAME global frame token-by-token that the
            # full forward builds in one cumsum, so cached keys (stored already
            # in the global frame) stay valid for every later query.
            from nanochat.engine import KVCache
            from nanochat.gpt import GPT

            torch.manual_seed(seed)
            cfg = _certify_tiny_config("gauge")
            # No dtype cast: GPT params default to fp32 and the rotary buffers
            # must STAY bf16 (GPT.forward asserts it). The fp32-vs-bf16 sweep
            # of the other certify checks does not apply to a whole-model run.
            model = GPT(cfg).to(device=device).eval()
            ids = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
            with torch.inference_mode():
                full = model(ids).float()
                kv = KVCache(
                    batch_size=1,
                    num_heads=cfg.n_kv_head,
                    seq_len=ids.size(1),
                    head_dim=cfg.n_embd // cfg.n_head,
                    num_layers=cfg.n_layer,
                )
                steps = [model(ids[:, t : t + 1], kv_cache=kv)[:, -1, :].float() for t in range(ids.size(1))]
            decoded = torch.stack(steps, dim=1)
            return float((decoded - full).abs().max())

        add_check(
            "gauge",
            "kv_decode_matches_full_forward",
            "classical",
            gauge_kv_decode_measure,
            tolerance=1e-4,
            detail="token-by-token decode through the fp32 angle lane vs one full forward (frame invariance)",
        )

    # ----- ultrametric: strong triangle inequality on hard digits (exact) -----
    if "ultrametric" in mechanisms:

        def ultra_triangle_measure() -> float:
            block, _cfg = make_block("ultrametric")
            attn = block.attn.float()
            torch.manual_seed(seed + 3)
            feats = torch.randn(1, 2, 12, 32, device=device)
            digits = attn._digits_hard_int(attn.to_digits_q(feats))  # (1, 2, 12, K) integer digits
            d = digits[0, 0]  # (12, K)
            n = d.size(0)
            matches = (d.unsqueeze(0) == d.unsqueeze(1)).to(torch.int64)  # (n, n, K)
            lcp = matches.cumprod(dim=-1).sum(dim=-1)  # (n, n) exact LCP depths
            worst = 0
            for i in range(n):
                for j in range(n):
                    for kk in range(n):
                        lower = min(int(lcp[i, j]), int(lcp[j, kk]))
                        worst = max(worst, lower - int(lcp[i, kk]))
            return float(worst)

        add_check(
            "ultrametric",
            "strong_triangle_inequality_lcp",
            "classical",
            ultra_triangle_measure,
            tolerance=0.0,
            detail="lcp(x,z) >= min(lcp(x,y), lcp(y,z)) over all triples (exact, integer)",
        )

        def ultra_kernel_trie_equiv_measure() -> float:
            # Reduction to the exact implementation (bead uvjq): the packed
            # prefix-trie decode (33dd) reproduces hard-digit alpha^LCP kernel
            # attention EXACTLY in O(K T log T). Build a tiny 2-layer GPT,
            # prefill + one decode step in both modes, compare last-token
            # logits. Trie state is CPU-side, so this check pins CPU regardless
            # of --device (the equivalence is device-independent anyway).
            from nanochat.engine import KVCache
            from nanochat.gpt import GPT as _GPT
            from nanochat.gpt import GPTConfig as _GPTConfig

            torch.manual_seed(seed)
            ids = torch.randint(0, 128, (1, 12), dtype=torch.long)

            def _run(mode: str) -> Any:
                cfg = _GPTConfig(
                    sequence_len=32, vocab_size=128, n_layer=2, n_head=4, n_kv_head=2, n_embd=64,
                    attention_type="ultrametric", ultrametric_mode=mode,
                    ultrametric_hard_digits=True, ultrametric_lcp_beta=128.0,
                )
                model = _GPT(cfg).train(False)
                kv = KVCache(
                    batch_size=1, num_heads=cfg.n_kv_head, seq_len=ids.size(1),
                    head_dim=cfg.n_embd // cfg.n_head, num_layers=cfg.n_layer,
                )
                with torch.inference_mode():
                    _ = model(ids[:, :-1], kv_cache=kv)
                    return model(ids[:, -1:], kv_cache=kv)[:, -1, :].float()

            torch.manual_seed(seed)
            out_kernel = _run("kernel")
            torch.manual_seed(seed)  # identical init so only the attention MODE differs
            out_trie = _run("trie")
            return float((out_trie - out_kernel).abs().max())

        add_check(
            "ultrametric",
            "trie_decode_matches_hard_kernel",
            "reduction",
            ultra_kernel_trie_equiv_measure,
            tolerance=1e-3,
            detail="packed-trie decode reproduces hard-digit alpha^LCP kernel attention exactly",
        )

    # ----- braid: YBE law + restricted-law separation + payload invariance -----
    if "braid" in mechanisms:
        from nanochat.braid_attention_torch import BraidCausalSelfAttention as _Braid

        def _braid_strands() -> list[Any]:
            torch.manual_seed(seed + 4)
            return [torch.randn(64, 8, dtype=torch.float64, device=device) for _ in range(6)]

        def _ybe_residual(update: Any) -> float:
            ax, ay, bx, by, cx, cy = _braid_strands()

            def apply12(ax, ay, bx, by, cx, cy):
                nax, nay, nbx, nby = update(ax, ay, bx, by)
                return nax, nay, nbx, nby, cx, cy

            def apply23(ax, ay, bx, by, cx, cy):
                nbx, nby, ncx, ncy = update(bx, by, cx, cy)
                return ax, ay, nbx, nby, ncx, ncy

            lhs = apply12(*apply23(*apply12(ax, ay, bx, by, cx, cy)))
            rhs = apply23(*apply12(*apply23(ax, ay, bx, by, cx, cy)))
            return float(torch.max(torch.abs(torch.stack(lhs, dim=-1) - torch.stack(rhs, dim=-1))))

        def braid_payload_measure() -> float:
            ax, ay, bx, by, _, _ = _braid_strands()
            worst = 0.0
            for update in (_Braid._crossing_update_restricted, _Braid._crossing_update_ybe):
                _, nay, _, nby = update(ax, ay, bx, by)
                inp = torch.sort(torch.stack([ay, by], dim=0), dim=0).values
                out = torch.sort(torch.stack([nay, nby], dim=0), dim=0).values
                worst = max(worst, float((inp - out).abs().max()))
            return worst

        add_check(
            "braid",
            "ybe_law_holds",
            "classical",
            lambda: _ybe_residual(_Braid._crossing_update_ybe),
            tolerance=1e-10,
            detail="R3 residual for the swap-output crossing law",
        )
        add_check(
            "braid",
            "restricted_law_violates_ybe",
            "classical",
            lambda: _ybe_residual(_Braid._crossing_update_restricted),
            tolerance=1e-3,
            comparator="ge",
            detail="separation witness: the heuristic law must NOT satisfy YBE (proves the test has teeth)",
        )
        add_check(
            "braid",
            "payload_multiset_invariance",
            "classical",
            braid_payload_measure,
            tolerance=0.0,
            detail="crossings must preserve the payload multiset {y} exactly",
        )

        # Integrable (rmatrix) law certificates - bead u55.3. fp64 throughout:
        # these are theorems, not training artifacts, so no dtype relaxation.
        from nanochat.braid_attention_torch import (
            one_particle_transfer as _braid_t1p,
        )
        from nanochat.braid_attention_torch import (
            rmatrix_braid_relation_residual as _rmx_braid,
        )
        from nanochat.braid_attention_torch import (
            rmatrix_inversion_residual as _rmx_inv,
        )

        def _rmx_transfer_comm(perturb: float) -> float:
            # Closed-form one-particle transfer matrices of the inhomogeneous
            # six-vertex chain; commutator must vanish for the exact law and
            # detectably break under an epsilon-perturbation of one Boltzmann
            # weight (the teeth witness: commutativity is NOT generic).
            torch.manual_seed(seed + 6)
            T_, eta_ = 12, 0.9
            u = torch.cumsum(torch.rand(T_, dtype=torch.float64) * 0.25 + 0.05, dim=0)
            t1 = _braid_t1p(float(u[-1]) + 1.1, u, eta_)
            t2 = _braid_t1p(float(u[-1]) + 2.3, u, eta_)
            if perturb != 0.0:
                t1 = t1.clone()
                t1[0, 1] = t1[0, 1] * (1.0 + perturb) + perturb
            comm = t1 @ t2 - t2 @ t1
            return float(comm.norm() / (t1.norm() * t2.norm()))

        add_check(
            "braid",
            "rmatrix_braid_relation_holds",
            "classical",
            lambda: _rmx_braid(trials=256, seed=seed),
            tolerance=1e-10,
            detail="spectral-parameter braid relation for the trigonometric six-vertex law (rapidities ride)",
        )
        add_check(
            "braid",
            "rmatrix_inversion_relation_holds",
            "classical",
            lambda: _rmx_inv(trials=256, seed=seed),
            tolerance=1e-10,
            detail="inversion relation N(w) N(-w) = I (protected memory: the mixing is exactly invertible)",
        )
        add_check(
            "braid",
            "rmatrix_transfer_matrices_commute",
            "classical",
            lambda: _rmx_transfer_comm(0.0),
            tolerance=1e-10,
            detail="[T(u), T(v)] = 0 for the inhomogeneous transfer family (one-particle closed form, fp64)",
        )
        add_check(
            "braid",
            "rmatrix_perturbed_transfer_separates",
            "classical",
            lambda: _rmx_transfer_comm(1e-1),
            tolerance=1e-6,
            comparator="ge",
            detail="separation witness: perturbing one Boltzmann weight must break commutativity (teeth)",
        )

        def _rmx_mass_partition() -> float:
            # Q1 conservation on a live forward through the actual module.
            from nanochat.gpt import GPT, GPTConfig

            torch.manual_seed(seed + 7)
            cfg = GPTConfig(
                n_layer=1,
                n_head=2,
                n_kv_head=2,
                n_embd=32,
                sequence_len=24,
                vocab_size=64,
                attention_type="braid",
                braid_crossing_law="rmatrix",
            )
            model = GPT(cfg)
            model.eval()
            with torch.no_grad():
                model(torch.randint(0, 64, (1, 24)))
            worst = 0.0
            for module in model.modules():
                charges = getattr(module, "last_braid_charges", None)
                if isinstance(charges, dict) and isinstance(charges.get("q1_mass_defect"), float):
                    worst = max(worst, charges["q1_mass_defect"])
            return worst

        add_check(
            "braid",
            "rmatrix_mass_partition_charge_conserved",
            "classical",
            _rmx_mass_partition,
            tolerance=1e-5,
            detail="Q1: sum_j A_ij + leftover_i = 1 exactly (stochastic gauge; live module forward)",
        )

        def _heuristic_mass_partition() -> float:
            # The heuristic (soft/restricted) modes are additive accumulations
            # with no mass partition: their Q1 defect must be measurably
            # NONZERO - the other half of the charge fingerprint, in the same
            # artifact the conservation claim is adjudicated from (xas7).
            from nanochat.gpt import GPT, GPTConfig

            torch.manual_seed(seed + 8)
            cfg = GPTConfig(
                n_layer=1,
                n_head=2,
                n_kv_head=2,
                n_embd=32,
                sequence_len=24,
                vocab_size=64,
                attention_type="braid",
                braid_crossing_law="restricted",
            )
            model = GPT(cfg)
            model.eval()
            with torch.no_grad():
                model(torch.randint(0, 64, (1, 24)))
            worst = 0.0
            for module in model.modules():
                charges = getattr(module, "last_braid_charges", None)
                if isinstance(charges, dict) and isinstance(charges.get("q1_mass_defect"), float):
                    worst = max(worst, charges["q1_mass_defect"])
            return worst

        add_check(
            "braid",
            "heuristic_mass_partition_violated",
            "classical",
            _heuristic_mass_partition,
            tolerance=1e-3,
            comparator="ge",
            detail="separation witness: additive heuristic accumulation has NO mass partition (Q1 defect >> 0)",
        )

    # ----- simplicial: mass conservation through 1-hop and 2-hop paths -----
    if "simplicial" in mechanisms:

        def simplicial_mass_measure() -> float:
            torch.manual_seed(seed + 5)
            scores = torch.randn(1, 2, 8, 8, device=device)
            scores = scores.masked_fill(~causal_attn_mask(8, 8, device=device), float("-inf"))
            att = torch.softmax(scores, dim=-1)
            c = 0.731
            v = torch.full((1, 2, 8, 32), c, device=device)
            y1 = att @ v
            y2 = att @ y1
            return float(torch.maximum((y1 - c).abs().max(), (y2 - c).abs().max()))

        add_check(
            "simplicial",
            "mass_conservation_two_hop",
            "classical",
            simplicial_mass_measure,
            tolerance=1e-5,
            detail="row-stochastic 1-hop and 2-hop aggregation must preserve constants",
        )

    # ----- fractal: router branch distributions are simplex-valued -----
    if "fractal" in mechanisms:

        def fractal_router_measure() -> float:
            import torch.nn.functional as t_func

            block, _cfg = make_block("fractal")
            attn = block.attn.float()
            torch.manual_seed(seed + 6)
            feats = torch.randn(1, 2, 8, 32, device=device)
            route = attn.router(feats).view(1, 2, 8, attn.depth, attn.m)
            probs = t_func.softmax(route, dim=-1)
            sums_err = (probs.sum(dim=-1) - 1.0).abs().max()
            neg_err = (-probs).clamp_min(0.0).max()
            return float(torch.maximum(sums_err, neg_err))

        add_check(
            "fractal",
            "router_branch_simplex",
            "classical",
            fractal_router_measure,
            tolerance=1e-6,
            detail="per-depth branch distributions must lie on the m-simplex",
        )

    # ----- surreal: scale/direction decomposition laws -----
    if "surreal" in mechanisms:
        from nanochat.surreal_torch import SurrealLayer

        def _surreal_layer() -> Any:
            torch.manual_seed(seed + 7)
            return SurrealLayer(64, 64, bias=False).to(device=device).float()

        def surreal_norm_measure() -> float:
            import torch.nn.functional as t_func

            layer = _surreal_layer()
            with torch.no_grad():
                layer.weight_s.uniform_(-1.0, 1.0)
                w = torch.exp(layer.weight_s) * t_func.normalize(layer.weight_v, dim=1)
                return float((w.norm(dim=1, keepdim=True) - torch.exp(layer.weight_s)).abs().max())

        def surreal_linearity_measure() -> float:
            layer = _surreal_layer()
            torch.manual_seed(seed + 8)
            x = torch.randn(4, 64, device=device)
            c = 3.7
            with torch.no_grad():
                return float((layer(c * x) - c * layer(x)).abs().max())

        def surreal_s_shift_measure() -> float:
            layer = _surreal_layer()
            torch.manual_seed(seed + 9)
            x = torch.randn(4, 64, device=device)
            delta = 0.9
            with torch.no_grad():
                y0 = layer(x)
                layer.weight_s.add_(delta)
                y1 = layer(x)
                return float((y1 - math.exp(delta) * y0).abs().max())

        add_check(
            "surreal",
            "row_norm_equals_exp_scale",
            "classical",
            surreal_norm_measure,
            tolerance=1e-5,
            detail="||w_row|| = exp(s_row): the scale/direction decomposition",
        )
        add_check("surreal", "layer_linearity", "classical", surreal_linearity_measure, tolerance=1e-4)
        add_check(
            "surreal",
            "scale_shift_equivariance",
            "classical",
            surreal_s_shift_measure,
            tolerance=1e-4,
            detail="s -> s + d must scale outputs by exp(d) exactly",
        )

    return checks


@app.command()
def certify(
    mechanism: Annotated[
        list[str] | None,
        typer.Option("--mechanism", "-m", help="Mechanism(s) to certify (default: all)"),
    ] = None,
    device: Annotated[str, typer.Option(help="Device: cpu | cuda")] = "cpu",
    dtype: Annotated[str, typer.Option(help="Dtype for module-level checks: fp32 | bf16")] = "fp32",
    seed: Annotated[int, typer.Option(help="Seed for all randomized checks")] = 42,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show per-check detail column")] = False,
    artifacts_dir: Annotated[Path | None, typer.Option(help="Write summary.json/run.md under this dir")] = None,
    run_id: Annotated[str | None, typer.Option(help="Run id (default: timestamp)")] = None,
):
    """
    Certify the mathematical invariants of the nanochat attention mechanisms.

    Runs the shared causality certificate for every mechanism plus per-mechanism
    invariant checks (Lipschitz bounds, algebra laws, round-trips, conservation).
    Exits nonzero if any check fails.
    """
    mechanisms = list(mechanism) if mechanism else list(_CERTIFY_MECHANISMS)
    unknown = [m for m in mechanisms if m not in _CERTIFY_MECHANISMS]
    if unknown:
        console.print(f"[bold red]Unknown mechanism(s):[/bold red] {unknown}. Valid: {_CERTIFY_MECHANISMS}")
        raise typer.Exit(code=2)

    git_info = _get_git_info()
    rid = run_id or _default_run_id()
    console.print(
        Panel(
            f"mechanisms: [cyan]{', '.join(mechanisms)}[/cyan]\n"
            f"device: [cyan]{device}[/cyan] · dtype: [cyan]{dtype}[/cyan] · seed: [cyan]{seed}[/cyan]\n"
            f"git: [cyan]{git_info['commit']}[/cyan]{' (dirty)' if git_info['dirty'] else ''}",
            title="[bold]mgr certify — mathematical invariant checks[/bold]",
            border_style="blue",
        )
    )

    t_start = time.perf_counter()
    checks = _run_certify_checks(mechanisms, device_str=device, dtype_str=dtype, seed=seed)
    total_s = time.perf_counter() - t_start

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Mechanism", style="cyan")
    table.add_column("Check")
    table.add_column("Family", style="dim")
    table.add_column("Status")
    table.add_column("Measured", justify="right")
    table.add_column("Tolerance", justify="right")
    table.add_column("ms", justify="right", style="dim")
    if verbose:
        table.add_column("Detail", style="dim", max_width=60)
    n_pass = n_fail = n_error = 0
    for c in checks:
        if c["status"] == "pass":
            n_pass += 1
            status = "[green]PASS[/green]"
        elif c["status"] == "fail":
            n_fail += 1
            status = "[bold red]FAIL[/bold red]"
        else:
            n_error += 1
            status = "[bold red]ERROR[/bold red]"
        cmp_sym = "<=" if c["comparator"] == "le" else ">="
        row = [
            c["mechanism"],
            c["check"],
            c["family"],
            status,
            f"{c['measured']:.3e}",
            f"{cmp_sym} {c['tolerance']:.3e}",
            f"{c['duration_ms']:.1f}",
        ]
        if verbose:
            row.append(c["detail"])
        table.add_row(*row)
    console.print(table)

    color = "green" if (n_fail + n_error) == 0 else "red"
    console.print(
        Panel(
            f"[bold {color}]{n_pass} passed · {n_fail} failed · {n_error} errored[/bold {color}]"
            f" · {len(checks)} checks in {total_s:.1f}s",
            border_style=color,
        )
    )
    for c in checks:
        if c["status"] != "pass":
            cmp_sym = "<=" if c["comparator"] == "le" else ">="
            console.print(
                f"[red]✗ {c['mechanism']}.{c['check']}[/red] measured={c['measured']:.3e} "
                f"({cmp_sym} {c['tolerance']:.3e} required) — "
                f"repro: [bold]mgr certify -m {c['mechanism']} --seed {seed} --device {device} --dtype {dtype}[/bold]"
                + (f"\n  [dim]{c['detail']}[/dim]" if c["detail"] else "")
            )

    if artifacts_dir is not None:
        run_dir = artifacts_dir / "certs" / "nanochat" / rid
        summary = {
            "schema_version": 1,
            "kind": "certify",
            "run_id": rid,
            "seed": seed,
            "device": device,
            "dtype": dtype,
            "git": git_info,
            "mechanisms": mechanisms,
            "checks": checks,
            "counts": {"pass": n_pass, "fail": n_fail, "error": n_error},
            "duration_s": round(total_s, 3),
        }
        lines = [
            "# mgr certify",
            "",
            f"- run_id: `{rid}` · seed: {seed} · device: {device} · dtype: {dtype}",
            f"- git: `{git_info['commit']}`{' (dirty)' if git_info['dirty'] else ''}",
            f"- result: **{n_pass} passed / {n_fail} failed / {n_error} errored** in {total_s:.1f}s",
            "",
            "| Mechanism | Check | Family | Status | Measured | Tolerance | ms |",
            "|---|---|---|---|---:|---:|---:|",
        ]
        for c in checks:
            cmp_sym = "<=" if c["comparator"] == "le" else ">="
            lines.append(
                f"| {c['mechanism']} | {c['check']} | {c['family']} | {c['status']} "
                f"| {c['measured']:.3e} | {cmp_sym} {c['tolerance']:.3e} | {c['duration_ms']:.1f} |"
            )
        _write_artifacts(run_dir, summary=summary, report_md="\n".join(lines) + "\n")
        console.print(f"[bold green]Wrote certificate artifacts[/bold green] → {run_dir}")

    if n_fail + n_error > 0:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# mgr fuzz — numerical robustness sweep (br: model_guided_research-5ki.4)
#
# Stresses every mechanism across dtypes, input scales, sequence-length
# boundaries, and adversarial input patterns, recording NaN/Inf in outputs AND
# gradients. Complements `mgr certify` (invariants on benign inputs): fuzz asks
# what happens on the inputs nobody writes unit tests for.
# ---------------------------------------------------------------------------

_FUZZ_PATTERNS: list[str] = ["randn", "all_equal", "alternating", "zeros"]


def _fuzz_make_input(pattern: str, *, B: int, T: int, C: int, scale: float, device: Any, dtype: Any) -> Any:
    import torch

    if pattern == "randn":
        x = torch.randn(B, T, C, device=device, dtype=torch.float32) * scale
    elif pattern == "all_equal":
        # Every token identical and every channel equal: maximal ties in every argmax/softmax.
        x = torch.full((B, T, C), 1.0, device=device, dtype=torch.float32) * scale
    elif pattern == "alternating":
        # Extreme +/- cancellation pattern along both time and channels.
        t_sign = (-1.0) ** torch.arange(T, device=device, dtype=torch.float32)
        c_sign = (-1.0) ** torch.arange(C, device=device, dtype=torch.float32)
        x = (t_sign.view(1, T, 1) * c_sign.view(1, 1, C)).expand(B, T, C).clone() * scale
    elif pattern == "zeros":
        # Zero-norm everything: rmsnorm of zeros, zero-norm quaternions/octonions, etc.
        x = torch.zeros(B, T, C, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown fuzz pattern: {pattern!r}")
    return x.to(dtype=dtype)


def _run_fuzz_cells(
    mechanisms: list[str],
    *,
    device_str: str = "cpu",
    dtypes: list[str] | None = None,
    scales: list[float] | None = None,
    lengths: list[int] | None = None,
    patterns: list[str] | None = None,
    seed: int = 42,
    warn_out_abs: float = 1e4,
    warn_grad_abs: float = 1e6,
    progress: Any = None,
) -> list[dict[str, Any]]:
    """
    Run the robustness sweep and return one record per cell:
    {mechanism, dtype, scale, T, pattern, status, out_nan_inf, grad_nan_inf,
     max_abs_out, max_abs_grad, duration_ms, recipe}.

    status: "fail" if any NaN/Inf appears in the output or input-gradient;
    "warn" if magnitudes exceed the warn thresholds; "error" on exception;
    else "pass". The `recipe` string is a complete reproduction command.

    Note: the `zeros` pattern is scale-invariant, so it runs once per
    (mechanism, dtype, T) regardless of the scales list.
    """
    import torch

    from nanochat.gpt import Block

    device = torch.device(device_str)
    dtypes = dtypes if dtypes is not None else ["fp32", "bf16"]
    scales = scales if scales is not None else [1e-3, 1.0, 1e3]
    # Boundary lengths around the tiny-config block size (1, 2, just-below, at).
    lengths = lengths if lengths is not None else [1, 2, 63, 64]
    patterns = patterns if patterns is not None else list(_FUZZ_PATTERNS)

    records: list[dict[str, Any]] = []
    for mech in mechanisms:
        for dtype_str in dtypes:
            dtype = torch.float32 if dtype_str == "fp32" else torch.bfloat16
            for T in lengths:
                for pattern in patterns:
                    cell_scales = [1.0] if pattern == "zeros" else scales
                    for scale in cell_scales:
                        t0 = time.perf_counter()
                        recipe = (
                            f"mechanism={mech} dtype={dtype_str} scale={scale:g} T={T} pattern={pattern} seed={seed}"
                        )
                        rec: dict[str, Any] = {
                            "mechanism": mech,
                            "dtype": dtype_str,
                            "scale": scale,
                            "T": T,
                            "pattern": pattern,
                            "recipe": recipe,
                        }
                        try:
                            torch.manual_seed(seed)
                            cfg = _certify_tiny_config(mech)
                            block = Block(cfg, 0).to(device=device, dtype=dtype)
                            block.eval()
                            head_dim = cfg.n_embd // cfg.n_head
                            cos_sin = _certify_cos_sin(T, head_dim, device, dtype)
                            x = _fuzz_make_input(
                                pattern, B=2, T=T, C=cfg.n_embd, scale=scale, device=device, dtype=dtype
                            )
                            x.requires_grad_(True)
                            y = block(x, cos_sin, None)
                            (g,) = torch.autograd.grad(y.float().sum(), x, allow_unused=True)
                            y_f = y.detach().float()
                            out_bad = int((~torch.isfinite(y_f)).sum())
                            if g is None:
                                grad_bad = 0
                                max_grad = 0.0
                            else:
                                g_f = g.detach().float()
                                grad_bad = int((~torch.isfinite(g_f)).sum())
                                max_grad = float(g_f.abs().max()) if g_f.numel() else 0.0
                            max_out = float(y_f.abs().max()) if y_f.numel() else 0.0
                            rec.update(
                                out_nan_inf=out_bad,
                                grad_nan_inf=grad_bad,
                                max_abs_out=max_out,
                                max_abs_grad=max_grad,
                            )
                            if out_bad or grad_bad:
                                rec["status"] = "fail"
                            elif max_out > warn_out_abs or max_grad > warn_grad_abs:
                                rec["status"] = "warn"
                            else:
                                rec["status"] = "pass"
                        except Exception as exc:  # noqa: BLE001 - a crash is a finding, not a harness bug
                            rec.update(
                                status="error",
                                out_nan_inf=-1,
                                grad_nan_inf=-1,
                                max_abs_out=float("nan"),
                                max_abs_grad=float("nan"),
                                detail=f"{type(exc).__name__}: {exc}",
                            )
                        rec["duration_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
                        records.append(rec)
                        if progress is not None:
                            progress()
    return records


@app.command()
def fuzz(
    mechanism: Annotated[
        list[str] | None,
        typer.Option("--mechanism", "-m", help="Mechanism(s) to fuzz (default: all)"),
    ] = None,
    device: Annotated[str, typer.Option(help="Device: cpu | cuda")] = "cpu",
    seed: Annotated[int, typer.Option(help="Seed for randomized patterns")] = 42,
    artifacts_dir: Annotated[Path | None, typer.Option(help="Write summary.json/run.md under this dir")] = None,
    run_id: Annotated[str | None, typer.Option(help="Run id (default: timestamp)")] = None,
    fail_on_warn: Annotated[bool, typer.Option(help="Treat warns as failures for the exit code")] = False,
):
    """
    Numerical robustness fuzz: dtype x scale x boundary-length x adversarial-pattern
    sweep over the attention mechanisms, recording NaN/Inf in outputs and gradients.

    Every non-pass cell prints its full reproduction recipe. Exit code 1 if any
    cell fails (or warns, with --fail-on-warn).
    """
    mechanisms = list(mechanism) if mechanism else list(_CERTIFY_MECHANISMS)
    unknown = [m for m in mechanisms if m not in _CERTIFY_MECHANISMS]
    if unknown:
        console.print(f"[bold red]Unknown mechanism(s):[/bold red] {unknown}. Valid: {_CERTIFY_MECHANISMS}")
        raise typer.Exit(code=2)

    git_info = _get_git_info()
    rid = run_id or _default_run_id()
    n_cells_per_mech = 2 * len([1, 2, 63, 64]) * (3 * 3 + 1)  # dtypes * lengths * (3 patterns x 3 scales + zeros)
    total_cells = n_cells_per_mech * len(mechanisms)
    console.print(
        Panel(
            f"mechanisms: [cyan]{', '.join(mechanisms)}[/cyan]\n"
            f"grid: dtype {{fp32, bf16}} x scale {{1e-3, 1, 1e3}} x T {{1, 2, 63, 64}} x "
            f"pattern {{{', '.join(_FUZZ_PATTERNS)}}} = [cyan]{total_cells}[/cyan] cells\n"
            f"device: [cyan]{device}[/cyan] · seed: [cyan]{seed}[/cyan] · git: [cyan]{git_info['commit']}[/cyan]",
            title="[bold]mgr fuzz — numerical robustness sweep[/bold]",
            border_style="blue",
        )
    )

    t_start = time.perf_counter()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("fuzzing", total=total_cells)
        records = _run_fuzz_cells(
            mechanisms,
            device_str=device,
            seed=seed,
            progress=lambda: prog.advance(task),
        )
    total_s = time.perf_counter() - t_start

    by_mech: dict[str, dict[str, int]] = {}
    for r in records:
        agg = by_mech.setdefault(r["mechanism"], {"pass": 0, "warn": 0, "fail": 0, "error": 0, "cells": 0})
        agg[r["status"]] += 1
        agg["cells"] += 1

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Mechanism", style="cyan")
    table.add_column("Cells", justify="right")
    table.add_column("Pass", justify="right", style="green")
    table.add_column("Warn", justify="right", style="yellow")
    table.add_column("Fail", justify="right", style="red")
    table.add_column("Error", justify="right", style="red")
    table.add_column("Worst cell", style="dim", max_width=60)
    for mech, agg in by_mech.items():
        worst = ""
        bad = [r for r in records if r["mechanism"] == mech and r["status"] in ("fail", "error")]
        if not bad:
            bad = [r for r in records if r["mechanism"] == mech and r["status"] == "warn"]
        if bad:
            worst = bad[0]["recipe"]
        table.add_row(
            mech,
            str(agg["cells"]),
            str(agg["pass"]),
            str(agg["warn"]),
            str(agg["fail"]),
            str(agg["error"]),
            worst,
        )
    console.print(table)

    n_pass = sum(1 for r in records if r["status"] == "pass")
    n_warn = sum(1 for r in records if r["status"] == "warn")
    n_fail = sum(1 for r in records if r["status"] == "fail")
    n_error = sum(1 for r in records if r["status"] == "error")
    color = "green" if (n_fail + n_error) == 0 else "red"
    console.print(
        Panel(
            f"[bold {color}]{n_pass} pass · {n_warn} warn · {n_fail} fail · {n_error} error[/bold {color}]"
            f" · {len(records)} cells in {total_s:.1f}s",
            border_style=color,
        )
    )
    for r in records:
        if r["status"] in ("fail", "error"):
            console.print(
                f"[red]✗[/red] {r['recipe']} — out_nan_inf={r['out_nan_inf']} grad_nan_inf={r['grad_nan_inf']}"
                + (f" [{r.get('detail', '')}]" if r.get("detail") else "")
            )
        elif r["status"] == "warn":
            console.print(
                f"[yellow]△[/yellow] {r['recipe']} — max|out|={r['max_abs_out']:.3e} max|grad|={r['max_abs_grad']:.3e}"
            )

    if artifacts_dir is not None:
        run_dir = artifacts_dir / "certs" / "fuzz" / rid
        summary = {
            "schema_version": 1,
            "kind": "fuzz",
            "run_id": rid,
            "seed": seed,
            "device": device,
            "git": git_info,
            "mechanisms": mechanisms,
            "counts": {"pass": n_pass, "warn": n_warn, "fail": n_fail, "error": n_error},
            "cells": records,
            "duration_s": round(total_s, 3),
        }
        lines = [
            "# mgr fuzz",
            "",
            f"- run_id: `{rid}` · seed: {seed} · device: {device} · git: `{git_info['commit']}`",
            f"- result: **{n_pass} pass / {n_warn} warn / {n_fail} fail / {n_error} error** "
            f"({len(records)} cells, {total_s:.1f}s)",
            "",
            "| Mechanism | Cells | Pass | Warn | Fail | Error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for mech, agg in by_mech.items():
            lines.append(
                f"| {mech} | {agg['cells']} | {agg['pass']} | {agg['warn']} | {agg['fail']} | {agg['error']} |"
            )
        non_pass = [r for r in records if r["status"] != "pass"]
        if non_pass:
            lines += ["", "## Non-pass cells", ""]
            for r in non_pass:
                lines.append(
                    f"- `{r['status']}` {r['recipe']} (out_nan_inf={r['out_nan_inf']}, "
                    f"grad_nan_inf={r['grad_nan_inf']}, max|out|={r['max_abs_out']:.3e})"
                )
        _write_artifacts(run_dir, summary=summary, report_md="\n".join(lines) + "\n")
        console.print(f"[bold green]Wrote fuzz artifacts[/bold green] → {run_dir}")

    if n_fail + n_error > 0 or (fail_on_warn and n_warn > 0):
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# mgr doctor — environment diagnostics (br: model_guided_research-rz8.7)
#
# Preflight for humans and agents: checks the four-way version matrix
# (Python/torch/JAX/CUDA), data presence, tokenizer, Triton, disk space, and
# runs a tiny end-to-end forward pass. Every failing row carries an actionable
# fix-it hint. Exit codes: 0 = all ok, 1 = warnings, 2 = failures.
# ---------------------------------------------------------------------------


def _run_doctor_checks() -> list[dict[str, Any]]:
    """Return one record per check: {name, status: ok|warn|fail, detail, hint}."""
    import shutil
    import sys as _sys

    rows: list[dict[str, Any]] = []

    def add(name: str, status: str, detail: str, hint: str = "") -> None:
        rows.append({"name": name, "status": status, "detail": detail, "hint": hint})

    # Python + tooling
    py = _sys.version_info
    if (py.major, py.minor) >= (3, 13):
        add("python", "ok", f"{py.major}.{py.minor}.{py.micro}")
    else:
        add("python", "fail", f"{py.major}.{py.minor}.{py.micro}", "This project targets Python 3.13+ (see AGENTS.md)")
    add(
        "uv",
        "ok" if shutil.which("uv") else "warn",
        shutil.which("uv") or "not on PATH",
        "" if shutil.which("uv") else "Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh",
    )
    in_venv = _sys.prefix != _sys.base_prefix
    add(
        "venv",
        "ok" if in_venv else "warn",
        _sys.prefix if in_venv else "no virtualenv active",
        "" if in_venv else "Run: uv venv && source .venv/bin/activate && uv sync --extra dev",
    )

    # torch
    try:
        import torch

        detail = f"torch {torch.__version__}"
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            bf16 = torch.cuda.is_bf16_supported()
            detail += f" · CUDA {torch.version.cuda} · {props.name} ({props.total_memory // (1 << 20)} MiB)"
            detail += f" · bf16={'yes' if bf16 else 'no'}"
            add("torch", "ok", detail)
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            add("torch", "ok", detail + " · MPS available (no CUDA)")
        else:
            add("torch", "warn", detail + " · CPU only", "GPU optional; nanochat training is much faster on CUDA")
    except Exception as exc:  # noqa: BLE001
        add("torch", "fail", f"{type(exc).__name__}: {exc}", "uv sync --extra dev")

    # JAX
    try:
        import jax

        from utils import get_device_info

        info = get_device_info()
        add("jax", "ok", f"jax {jax.__version__} · backend={info['default_backend']} · devices={info['devices']}")
    except Exception as exc:  # noqa: BLE001
        add("jax", "fail", f"{type(exc).__name__}: {exc}", "uv sync --extra dev (JAX powers the mgr demos)")

    # Triton (A7 kernel prerequisite; optional)
    try:
        import triton  # noqa: F401

        add("triton", "ok", f"triton {getattr(triton, '__version__', '?')}")
    except Exception:  # noqa: BLE001
        add("triton", "warn", "not importable", "Optional: needed only for custom GPU kernels (tropical Triton path)")

    # Training data (parquet shards)
    try:
        from nanochat.dataset import DATA_DIR, list_parquet_files

        files = list_parquet_files()
        if len(files) >= 2:
            add("training data", "ok", f"{len(files)} parquet shard(s) in {DATA_DIR}")
        elif len(files) == 1:
            add(
                "training data",
                "warn",
                f"only 1 parquet shard in {DATA_DIR} (train/val split needs >= 2)",
                "python -m nanochat.train --auto-download-data, or stage shards into the data dir",
            )
        else:
            add(
                "training data",
                "warn",
                f"no parquet shards in {DATA_DIR}",
                "python -m nanochat.train --auto-download-data downloads FineWeb-Edu shards",
            )
    except Exception as exc:  # noqa: BLE001
        add("training data", "warn", f"{type(exc).__name__}: {exc}", "check NANOCHAT_BASE_DIR")

    # Tokenizer
    try:
        from nanochat.tokenizer import get_tokenizer

        tok = get_tokenizer()
        n_vocab = tok.get_vocab_size() if hasattr(tok, "get_vocab_size") else "?"
        add("tokenizer", "ok", f"{type(tok).__name__} (vocab={n_vocab})")
    except Exception as exc:  # noqa: BLE001
        add(
            "tokenizer",
            "warn",
            f"{type(exc).__name__}: {exc}",
            "tokenizer assets load on first training run; check network access if this persists",
        )

    # Disk space at the nanochat cache dir
    try:
        from nanochat.common import get_base_dir

        base = get_base_dir()
        usage = shutil.disk_usage(base)
        free_gb = usage.free / (1 << 30)
        status = "ok" if free_gb >= 5.0 else "warn"
        add(
            "disk space",
            status,
            f"{free_gb:.1f} GiB free at {base}",
            "" if status == "ok" else "training data + checkpoints need headroom; free space or move NANOCHAT_BASE_DIR",
        )
    except Exception as exc:  # noqa: BLE001
        add("disk space", "warn", f"{type(exc).__name__}: {exc}", "")

    # End-to-end smoke: tiny GPT forward on the best available device
    try:
        import torch

        from nanochat.gpt import GPT, GPTConfig

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = GPTConfig(sequence_len=32, vocab_size=128, n_layer=1, n_head=2, n_kv_head=2, n_embd=64)
        model = GPT(cfg).to(device)
        # GPT.forward requires bf16 rotary buffers (asserted in forward); init_weights would
        # recompute them, but for a smoke test the constructor buffers are already bf16.
        idx = torch.randint(0, 128, (1, 16), device=device)
        with torch.no_grad():
            logits = model(idx)
        if logits.shape == (1, 16, 128) and bool(torch.isfinite(logits).all()):
            add("model smoke", "ok", f"tiny GPT forward on {device}: logits {tuple(logits.shape)}, finite")
        else:
            add("model smoke", "fail", f"unexpected logits: shape={tuple(logits.shape)}", "investigate nanochat.gpt")
    except Exception as exc:  # noqa: BLE001
        add("model smoke", "fail", f"{type(exc).__name__}: {exc}", "uv sync --extra dev; then rerun mgr doctor")

    return rows


@app.command()
def doctor(
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON only")] = False,
):
    """
    Environment diagnostics: versions, devices, data, tokenizer, disk, and a tiny
    end-to-end forward pass — with an actionable fix-it hint on every failing row.

    Exit codes: 0 = all ok, 1 = warnings present, 2 = failures present.
    """
    rows = _run_doctor_checks()
    n_warn = sum(1 for r in rows if r["status"] == "warn")
    n_fail = sum(1 for r in rows if r["status"] == "fail")

    if json_output:
        payload = {
            "schema_version": 1,
            "kind": "doctor",
            "checks": rows,
            "counts": {"ok": len(rows) - n_warn - n_fail, "warn": n_warn, "fail": n_fail},
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        table = Table(box=box.SIMPLE_HEAVY, title="mgr doctor")
        table.add_column("Check", style="cyan")
        table.add_column("Status")
        table.add_column("Detail", max_width=70)
        table.add_column("Fix hint", style="dim", max_width=50)
        style = {"ok": "[green]OK[/green]", "warn": "[yellow]WARN[/yellow]", "fail": "[bold red]FAIL[/bold red]"}
        for r in rows:
            table.add_row(r["name"], style[r["status"]], r["detail"], r["hint"])
        console.print(table)
        color = "green" if n_fail == 0 and n_warn == 0 else ("yellow" if n_fail == 0 else "red")
        console.print(
            Panel(
                f"[bold {color}]{len(rows) - n_warn - n_fail} ok · {n_warn} warn · {n_fail} fail[/bold {color}]",
                border_style=color,
            )
        )

    if n_fail > 0:
        raise typer.Exit(code=2)
    if n_warn > 0:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# mgr profile-data — model-free DATA-GEOMETRY PROFILER (bead 77l.1, EPIC GEO)
#
# Measures the geometric properties the theory epics condition on - delta-
# hyperbolicity, ultrametricity, dynamic range, order sensitivity, hierarchy
# depth - on a corpus BEFORE any training. Estimator math + planted-geometry
# calibration generators live in geometry_profile.py; this command is the
# corpus-facing surface. Values are representation-relative: the registered
# claims (77l.2) are about ORDERINGS across corpora, never absolute numbers.
# ---------------------------------------------------------------------------


@app.command("gen-tasks")
def gen_tasks(
    task: Annotated[str, typer.Option(help="Task name or 'all' (synthetic battery; realhier is --include-real)")] = "all",
    out: Annotated[Path, typer.Option(help="Output root; datasets land in <out>/<task>/")] = Path(
        "artifacts/diagnostics"
    ),
    size: Annotated[int, typer.Option(help="Documents per task (split ~80/10/10 train/val/heldout-test)")] = 2000,
    seed: Annotated[int, typer.Option(help="Generator seed (same seed -> byte-identical parquet)")] = 42,
    dial: Annotated[
        list[str] | None, typer.Option(help="Difficulty dial override name=value (repeatable)")
    ] = None,
    include_real: Annotated[
        bool, typer.Option("--include-real", help="Include the realhier task in --task all")
    ] = False,
    list_tasks: Annotated[bool, typer.Option("--list", help="List tasks, mechanisms, hypotheses, dials")] = False,
) -> None:
    """Generate the theory-aligned diagnostic task battery (bead vdc.1)."""
    from nanochat.diagnostics_data import DEFAULT_TASKS, TASKS, generate_task

    if list_tasks:
        table = Table(title="Diagnostic task battery (vdc.1)", box=box.SIMPLE_HEAVY)
        table.add_column("task", style="bold")
        table.add_column("targets")
        table.add_column("dials")
        table.add_column("hypothesis")
        for name, spec in TASKS.items():
            dials = ", ".join(f"{d.name}={d.default} [{d.lo},{d.hi}]" for d in spec.dials) or "-"
            table.add_row(name, ", ".join(spec.target_mechanisms) or "(control)", dials, spec.hypothesis)
        console.print(table)
        return

    dial_overrides: dict[str, float] = {}
    for pair in dial or []:
        key, sep, value = pair.partition("=")
        if not sep:
            console.print(f"[bold red]--dial expects name=value, got {pair!r}[/bold red]")
            raise typer.Exit(code=2)
        try:
            dial_overrides[key.strip()] = float(value)
        except ValueError:
            console.print(f"[bold red]--dial value must be numeric, got {pair!r}[/bold red]")
            raise typer.Exit(code=2) from None

    if task == "all":
        names = list(DEFAULT_TASKS) + (["realhier"] if include_real else [])
        if dial_overrides:
            console.print("[bold red]--dial applies to a single --task, not --task all.[/bold red]")
            raise typer.Exit(code=2)
    elif task in TASKS:
        names = [task]
    else:
        console.print(f"[bold red]Unknown task {task!r}; available: {', '.join(sorted(TASKS))} or 'all'[/bold red]")
        raise typer.Exit(code=2)

    summary = Table(title=f"gen-tasks → {out} (seed={seed}, size={size})", box=box.SIMPLE_HEAVY)
    summary.add_column("task", style="bold")
    summary.add_column("train", justify="right")
    summary.add_column("val", justify="right")
    summary.add_column("test", justify="right")
    summary.add_column("doc words min/med/max", justify="right")
    summary.add_column("dials")
    summary.add_column("sha256 (train)", style="dim")

    t0 = time.perf_counter()
    for name in names:
        with console.status(f"[bold cyan]generating {name}…[/bold cyan]"):
            manifest = generate_task(
                name, out_dir=out, size=size, seed=seed, dial_overrides=dial_overrides or None
            )
        import pyarrow.parquet as _pq

        train_path = out / name / "train_000.parquet"
        lengths = sorted(len(t.split()) for t in _pq.read_table(train_path).column("text").to_pylist())
        med = lengths[len(lengths) // 2]
        sizes = manifest["split_sizes"]
        dials_txt = ", ".join(f"{k}={v:g}" for k, v in manifest["dials"].items()) or "-"
        summary.add_row(
            name,
            str(sizes["train"]),
            str(sizes["val"]),
            str(sizes["test"]),
            f"{lengths[0]}/{med}/{lengths[-1]}",
            dials_txt,
            manifest["sha256"]["train_000.parquet"][:12],
        )
    console.print(summary)
    console.print(f"[bold green]Generated {len(names)} task(s)[/bold green] in {time.perf_counter() - t0:.1f}s")


def _load_eval_checkpoint(
    checkpoint_dir: Path, step: int | None, device: Any, *, model_overrides: dict[str, Any] | None = None
) -> tuple[Any, dict[str, Any], int]:
    """Load a nanochat GPT checkpoint for evaluation.

    Deliberately NOT checkpoint_manager.build_model: that helper asserts the
    tokenizer vocab matches the model config, but nanochat training configs
    use the padded default (50304) with the 50257-token GPT-2 tokenizer -
    a deliberate mismatch (every tokenizer id is still a valid input).

    model_overrides (bead 8gk.4): eval-time config knobs (e.g.
    ultrametric_digits_k for valuation-truncation sweeps) merged into the
    checkpoint's recorded config BEFORE construction; the caller must record
    the merged config in the artifact so override arms are variant-selectable.
    The override may not change any parameter-shaping field (the state dict
    must still load strictly).

    Annealed checkpoints (bead y2h9): when the meta records semiring_beta_live
    (the schedule's live value at save time), construction uses it in place of
    the recorded config's b0 and the returned meta's model_config reflects it,
    so eval artifacts carry the beta the model actually ran at. An explicit
    model_overrides semiring_beta still wins (applied after the fold).
    """
    import torch

    from nanochat.checkpoint_manager import find_last_step, load_checkpoint
    from nanochat.gpt import GPT, GPTConfig

    resolved_step = step if step is not None else find_last_step(str(checkpoint_dir))
    model_data, _optim, meta = load_checkpoint(str(checkpoint_dir), resolved_step, device)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    if meta.get("model_type", "gpt") != "gpt":
        raise ValueError(f"eval-tasks supports model_type=gpt checkpoints, got {meta.get('model_type')!r}")
    config_dict = dict(meta["model_config"])
    if "semiring_beta_live" in meta:
        config_dict["semiring_beta"] = meta["semiring_beta_live"]
        meta = {**meta, "model_config": config_dict}
    if model_overrides:
        # validate against the CURRENT GPTConfig schema, not the checkpoint's
        # recorded dict: eval-time knobs (ultrametric_digits_k) legitimately
        # postdate old checkpoints, whose configs simply lack the field
        import dataclasses

        known_fields = {f.name for f in dataclasses.fields(GPTConfig)}
        unknown = sorted(set(model_overrides) - known_fields)
        if unknown:
            raise ValueError(f"--model-override names unknown GPTConfig field(s): {unknown}")
        config_dict.update(model_overrides)
        meta = {**meta, "model_config": config_dict}
    config = GPTConfig(**config_dict)
    model = GPT(config).to(device)
    if device.type in {"cpu", "mps"}:
        model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}
    model.load_state_dict(model_data, strict=True)
    quant_bits = config_dict.get("eval_weight_quant_bits")
    if quant_bits is not None:
        _fake_quantize_weights(model, int(quant_bits))
    model.eval()
    return model, meta, resolved_step


def _fake_quantize_weights(model: Any, bits: int) -> None:
    """Symmetric uniform per-tensor fake-quantization of Linear/Embedding
    weights (bead tcuy): w -> round(w/delta) * delta with delta =
    max|w| / (2^(bits-1) - 1). The archimedean baseline arm for the
    digit-truncation precision curves (hyp-padic-truncation-graceful):
    per-layer rounding errors here COMPOUND through depth, which is exactly
    what the flat-error lemma forbids on the p-adic-native path."""
    import torch

    if not (2 <= bits <= 16):
        raise ValueError(f"eval_weight_quant_bits must be in [2, 16], got {bits}")
    qmax = float(2 ** (bits - 1) - 1)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                w = module.weight
                scale = w.abs().max()
                if float(scale) == 0.0:
                    continue
                delta = scale / qmax
                w.copy_(torch.clamp(torch.round(w / delta), -qmax, qmax) * delta)


def _eval_tasks_provenance(
    ckpt_meta: dict[str, Any],
    *,
    seeds: list[int],
    examples: int,
    decode_modes: list[str],
    dials: dict[str, float] | None = None,
) -> dict[str, Any]:
    from nanochat.report import build_provenance

    eval_recipe: dict[str, Any] = {"seeds": seeds, "examples": examples, "decode_modes": decode_modes}
    if dials:
        # dial overrides change the evidence-producing recipe (doc difficulty),
        # so they are part of the tamper-evident hash, not an invisible knob
        eval_recipe["dials"] = dials
    return build_provenance(
        {
            "model_config": ckpt_meta.get("model_config"),
            "eval": eval_recipe,
        }
    )


def _read_train_provenance(checkpoint_dir: Path) -> dict[str, Any] | None:
    """The TRAINING run's provenance block, from the run summary train.py
    writes beside the checkpoints directory (dz9i: training taint must
    propagate through eval artifacts). None when absent or unreadable."""
    summary_path = Path(checkpoint_dir).resolve().parent / "summary.json"
    if not summary_path.exists():
        return None
    try:
        prov = json.loads(summary_path.read_text(encoding="utf-8")).get("provenance")
    except (OSError, json.JSONDecodeError):
        return None
    return prov if isinstance(prov, dict) else None


_EVAL_TASKS_SCHEMA_VERSION = "mgr.evaltasks.v3"
# SCHEMA CONTRACT (consumed by C4 scorecards and the G2 verdict engine):
# {"schema_version", "kind": "eval-tasks",
#  "meta": {run_id, generated_at, checkpoint{dir, step, attention_type, n_params,
#           budget, lineage}, device, seeds, examples_per_seed, decode_modes, git,
#           receipts},
#  "train_provenance": {<the TRAINING run's provenance block>} | null,
#  "tasks": {<name>: {
#     "difficulty_axis": str|null, "in_range_max": float|null,
#     "exact_match": {<mode>: {"in_range"|"held_out":
#         {"mean": float, "ci95": [lo, hi], "per_seed": [float, ...]}}} | null,
#     "answer_prior": {"in_range"|"held_out":
#         {"mean": float, "per_seed": [float|null, ...], "majority_answer": str}} | null,
#     "perplexity": {"in_range": float, "held_out": float},
#     "curve": {"buckets": [...], "accuracy": [...], "counts": [...]} | null,
#     "length_slope": {"held_out"|"all":
#         {"slope": float, "ci95": [lo, hi], "intercept": float, "n_docs": int,
#          "basis": str} | null,
#       "by_category"?: {<cat>: <same shape> | null}} | null,
#     "skipped_too_long": int, "n_examples_per_seed": int}}}
# Changing this layout requires a schema_version bump and a migration note.
# v1 -> v2 (2026-06-10, beads 27q3/i4eq): adds (a) per-task answer_prior - the
# exact-match score of the best CONSTANT-answer policy on the very docs this
# run scored, i.e. the floor a model that learned only the answer format +
# majority token would hit; the ci-v2 verdict engine prefers this recorded
# floor over registered fallbacks - and (b) per-example receipts in
# generations.jsonl (meta.receipts), so behavioral claims about checkpoints
# (e.g. "every arm answers 'lt' on every prompt") are committed evidence, not
# session lore. v1 artifacts remain readable by the verdict engine.
# v2 -> v3 (2026-06-11, bead dz9i): adds train_provenance - the TRAINING
# run's provenance block, read from the run summary beside the checkpoint.
# The hqwi audit found that a clean-time eval of a TAINTED-TRAINING
# checkpoint looked clean to the verdict engine (eval artifacts only carried
# eval-time provenance); the engine now refuses an evaltasks artifact when
# EITHER provenance is tainted. null = no train summary found (legacy /
# synthetic checkpoints): engine falls back to eval-time provenance only.
# 2026-06-11 (bead u55.3, additive within v2 - no consumer reads break): adds
# per-task length_slope, the doc-level OLS slope (with CI95) of greedy
# exact-match against the difficulty axis, fit separately on held-out docs and
# on all docs. This is the length-generalization observable the preregistered
# word-problem hypotheses (hyp-rmatrix-*) adjudicate against; null when the
# task has no difficulty axis or too few scored docs.
# 2026-06-11 (fresh-eyes review, additive within v2): meta.checkpoint gains
# model_config - the checkpoint's full recorded GPTConfig - so variant
# selectors (rgyl) can distinguish within-mechanism arms (braid_crossing_law,
# ultrametric_mode, ...) on evaltasks evidence. Artifacts predating the field
# simply cannot match a variant-selected arm (the selector requires the
# recorded knob), which fails loudly as no_candidate_artifacts rather than
# silently pooling arms.


def _eval_split_examples(
    spec: Any, *, n: int, seed: int, dials: dict[str, float] | None = None
) -> tuple[list[str], list[str], float | None]:
    """(in-range docs, held-out docs, in_range_max difficulty)."""
    splits = spec.generate(max(10 * n, 30), seed, spec.resolve_dials(dials))
    in_range = splits["train"][:n]
    held_out = splits["test"][:n]
    in_range_max = None
    if spec.difficulty is not None:
        in_range_max = max(spec.difficulty(d) for d in splits["train"])
    return in_range, held_out, in_range_max


def _eval_doc_perplexity(model: Any, tok: Any, doc: str, device: Any) -> float:
    import torch

    ids = tok.encode(doc)[: model.config.sequence_len + 1]
    if len(ids) < 2:
        return float("nan")
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    with torch.inference_mode():
        loss = model(x, targets=y)
    return float(torch.exp(loss).item())


def _eval_score_doc(
    model: Any,
    tok: Any,
    spec: Any,
    doc: str,
    *,
    device: Any,
    temperature: float,
    seed: int,
) -> tuple[bool | None, str, str] | None:
    """(correct, expected, got) for answer-bearing docs; None when the prompt
    does not fit the rotary cache (caller counts these as skipped)."""
    sp = spec.split_prompt(doc)
    if sp is None:
        return (None, "", "")
    prompt, expected = sp
    prompt_ids = tok.encode(prompt)
    expected_words = expected.split()
    # answer + a small margin; canonicalize via whitespace split so tokenizer
    # quirks (leading spaces, merged pieces) cannot fail a correct answer
    max_new = len(tok.encode(" " + expected)) + 2
    if len(prompt_ids) + max_new >= model.rotary_seq_len:
        return None
    # Stop at the document separator: training corpora pack docs with BOS/EOT
    # between them, so a model that has learned the format emits the answer
    # IMMEDIATELY followed by <|endoftext|> - and decode() glues that marker
    # onto the answer with no whitespace, which would fail a correct answer
    # under whitespace canonicalization (found by the first real campaign, kbj2).
    stop_id = None
    get_bos = getattr(tok, "get_bos_token_id", None)
    if callable(get_bos):
        stop_id = get_bos()
    pieces: list[int] = []
    for piece in model.generate(prompt_ids, max_tokens=max_new, temperature=temperature, seed=seed):
        if stop_id is not None and piece == stop_id:
            break
        pieces.append(piece)
    got = tok.decode(pieces)
    got_words = got.split()
    correct = got_words[: len(expected_words)] == expected_words
    return (correct, expected, got)


@app.command("eval-tasks")
def eval_tasks(
    checkpoint: Annotated[Path, typer.Option(help="Checkpoint directory (rz8.1 layout)")],
    step: Annotated[int | None, typer.Option(help="Checkpoint step (default: latest in the directory)")] = None,
    task: Annotated[list[str] | None, typer.Option("--task", "-t", help="Task name (repeatable)")] = None,
    all_tasks: Annotated[bool, typer.Option("--all-tasks", help="Evaluate the full synthetic battery")] = False,
    device_str: Annotated[str, typer.Option("--device", help="cpu | cuda")] = "cpu",
    seeds: Annotated[str, typer.Option(help="Comma-separated eval seeds (multi-seed mean +/- CI)")] = "0",
    examples: Annotated[int, typer.Option(help="Examples per split per seed")] = 24,
    sampled: Annotated[bool, typer.Option("--sampled", help="Also score temperature-1 sampled decoding")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", help="Log per-example failures (capped per task)")] = False,
    dial: Annotated[
        list[str] | None,
        typer.Option(help="Difficulty dial override name=value (repeatable; recorded in provenance)"),
    ] = None,
    model_override: Annotated[
        list[str] | None,
        typer.Option(
            help=(
                "Eval-time model-config override key=value (repeatable; JSON-parsed scalars; e.g. "
                "ultrametric_digits_k=4 for valuation-truncation sweeps, bead 8gk.4). Recorded in "
                "meta.checkpoint.model_config, so override arms are variant-selectable evidence."
            )
        ),
    ] = None,
    artifacts_dir: Annotated[Path, typer.Option(help="Artifacts root")] = Path("artifacts"),
    run_id: Annotated[str | None, typer.Option(help="Run identifier (default: timestamp)")] = None,
) -> None:
    """Evaluate a trained checkpoint on the diagnostic task battery (bead vdc.2).

    Headline output: extrapolation curves (accuracy vs difficulty, in-range vs
    held-out marked) per task, exact-match via the vdc.1 brute-force format,
    per-task perplexity as the secondary metric. Writes summary.json (schema
    mgr.evaltasks.v3 - a versioned contract), per-example receipts
    (generations.jsonl), run.md, and curve PNGs.
    """
    import statistics as stats_mod

    import torch

    from nanochat.diagnostics_data import DEFAULT_TASKS, TASKS
    from nanochat.tokenizer import get_tokenizer

    device = torch.device(device_str)
    if task and all_tasks:
        console.print("[bold red]Provide --task or --all-tasks, not both.[/bold red]")
        raise typer.Exit(code=2)
    if all_tasks:
        names = list(DEFAULT_TASKS)  # realhier stays opt-in via explicit --task realhier
    elif task:
        unknown = [t for t in task if t not in TASKS]
        if unknown:
            console.print(f"[bold red]Unknown task(s) {unknown}; available: {', '.join(sorted(TASKS))}[/bold red]")
            raise typer.Exit(code=2)
        names = list(task)
    else:
        console.print("[bold red]Provide --task NAME (repeatable) or --all-tasks.[/bold red]")
        raise typer.Exit(code=2)

    seed_list = [int(s) for s in seeds.split(",") if s.strip() != ""]
    if not seed_list:
        console.print("[bold red]--seeds must contain at least one integer.[/bold red]")
        raise typer.Exit(code=2)

    dial_overrides: dict[str, float] = {}
    for pair in dial or []:
        key, sep, value = pair.partition("=")
        if not sep:
            console.print(f"[bold red]--dial expects name=value, got {pair!r}[/bold red]")
            raise typer.Exit(code=2)
        try:
            dial_overrides[key.strip()] = float(value)
        except ValueError:
            console.print(f"[bold red]--dial value must be numeric, got {pair!r}[/bold red]")
            raise typer.Exit(code=2) from None
    if dial_overrides:
        known = {d.name for n in names for d in TASKS[n].dials}
        unknown_dials = sorted(set(dial_overrides) - known)
        if unknown_dials:
            console.print(f"[bold red]--dial names {unknown_dials} match no dial of the selected tasks.[/bold red]")
            raise typer.Exit(code=2)

    model_overrides: dict[str, Any] = {}
    for pair in model_override or []:
        key, sep, value = pair.partition("=")
        if not sep:
            console.print(f"[bold red]--model-override expects key=value, got {pair!r}[/bold red]")
            raise typer.Exit(code=2)
        try:
            model_overrides[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            model_overrides[key.strip()] = value  # bare strings pass through

    model, ckpt_meta, resolved_step = _load_eval_checkpoint(
        checkpoint, step, device, model_overrides=model_overrides or None
    )
    tok = get_tokenizer()
    n_params = sum(p.numel() for p in model.parameters())

    # Run header: every eval artifact is self-identifying (D1 meta lineage).
    header = Table(title="eval-tasks — checkpoint header", box=box.SIMPLE_HEAVY)
    header.add_column("field", style="bold")
    header.add_column("value")
    header.add_row("checkpoint", f"{checkpoint} @ step {resolved_step}")
    header.add_row("attention_type", str(ckpt_meta.get("model_config", {}).get("attention_type")))
    header.add_row("n_params", f"{n_params:,}")
    header.add_row("training budget", json.dumps(ckpt_meta.get("budget", {})))
    header.add_row("lineage", json.dumps(ckpt_meta.get("lineage", {})))
    header.add_row("seeds", repr(seed_list))
    console.print(header)

    decode_modes = [("greedy", 0.0)] + ([("sampled", 1.0)] if sampled else [])
    results: dict[str, Any] = {}
    receipt_rows: list[dict[str, Any]] = []  # per-example evidence -> generations.jsonl
    failure_log_cap = 10

    for name in names:
        spec = TASKS[name]
        # per_seed lists stay ALIGNED with seed_list: a seed whose docs were
        # all skipped (too long for the rotary cache) records None, never a
        # silent gap - downstream consumers can pair per_seed[i] with seeds[i].
        per_seed_em: dict[str, dict[str, list[float | None]]] = {
            mode: {"in_range": [], "held_out": []} for mode, _ in decode_modes
        }
        # expected answers per region/seed (greedy pass; docs are identical
        # across modes) -> the best-constant-policy floor for this exact sample
        prior_counts: dict[str, list[dict[str, int]]] = {"in_range": [], "held_out": []}
        ppl_in: list[float] = []
        ppl_out: list[float] = []
        curve_points: list[tuple[float, bool, str, str | None]] = []
        skipped = 0
        failures_logged = 0
        in_range_max: float | None = None

        spec_dials = {k: v for k, v in dial_overrides.items() if k in {d.name for d in spec.dials}}
        with console.status(f"[bold cyan]evaluating {name}…[/bold cyan]"):
            for eval_seed in seed_list:
                in_docs, out_docs, seed_in_range_max = _eval_split_examples(
                    spec, n=examples, seed=eval_seed, dials=spec_dials or None
                )
                # the train-regime boundary is the max over EVERY seed's train split
                if seed_in_range_max is not None:
                    in_range_max = seed_in_range_max if in_range_max is None else max(in_range_max, seed_in_range_max)
                ppl_in.append(
                    stats_mod.mean(_eval_doc_perplexity(model, tok, d, device) for d in in_docs)
                )
                ppl_out.append(
                    stats_mod.mean(_eval_doc_perplexity(model, tok, d, device) for d in out_docs)
                )
                if spec.answer_marker is None:
                    continue
                for mode, temp in decode_modes:
                    for region, docs in (("in_range", in_docs), ("held_out", out_docs)):
                        correct_n = 0
                        scored_n = 0
                        seed_expected: dict[str, int] = {}
                        for doc_idx, doc in enumerate(docs):
                            scored = _eval_score_doc(
                                model, tok, spec, doc, device=device, temperature=temp, seed=eval_seed
                            )
                            if scored is None:
                                # each doc is scored once per mode; count its
                                # skip once (greedy always runs)
                                if mode == "greedy":
                                    skipped += 1
                                continue
                            correct, expected, got = scored
                            if correct is None:
                                continue
                            scored_n += 1
                            correct_n += int(correct)
                            receipt_rows.append(
                                {
                                    "task": name,
                                    "mode": mode,
                                    "region": region,
                                    "eval_seed": eval_seed,
                                    "doc_index": doc_idx,
                                    "expected": expected,
                                    "got": got,
                                    "correct": correct,
                                }
                            )
                            if mode == "greedy":
                                seed_expected[expected] = seed_expected.get(expected, 0) + 1
                                diff = spec.difficulty(doc) if spec.difficulty else None
                                if diff is not None:
                                    cat = spec.category(doc) if spec.category else None
                                    curve_points.append((diff, correct, region, cat))
                            if verbose and not correct and failures_logged < failure_log_cap:
                                failures_logged += 1
                                console.print(
                                    f"[yellow]{name} fail[/yellow] [{mode}/{region}] "
                                    f"expected={expected!r} got={got!r}"
                                )
                        per_seed_em[mode][region].append(correct_n / scored_n if scored_n else None)
                        if mode == "greedy":
                            prior_counts[region].append(seed_expected)

        def agg(values: list[float | None]) -> dict[str, Any] | None:
            valid = [v for v in values if v is not None]
            if not valid:
                return None
            mean = stats_mod.mean(valid)
            if len(valid) > 1:
                half = 1.96 * stats_mod.stdev(valid) / (len(valid) ** 0.5)
            else:
                half = 0.0
            return {"mean": mean, "ci95": [mean - half, mean + half], "per_seed": values}

        exact_match: dict[str, Any] | None = None
        if spec.answer_marker is not None:
            exact_match = {
                mode: {region: agg(vals) for region, vals in regions.items()}
                for mode, regions in per_seed_em.items()
            }

        def prior_agg(counters: list[dict[str, int]]) -> dict[str, Any] | None:
            # best-constant-answer score on this exact sample, per seed: any
            # model that learned only the answer format + one fixed token sits
            # AT or BELOW this line (the ci-v2 floor gate's preferred floor)
            per: list[float | None] = []
            for c in counters:
                total = sum(c.values())
                per.append(max(c.values()) / total if total else None)
            valid = [v for v in per if v is not None]
            if not valid:
                return None
            pooled: dict[str, int] = {}
            for c in counters:
                for ans, n in c.items():
                    pooled[ans] = pooled.get(ans, 0) + n
            majority = max(pooled, key=lambda k: pooled[k])
            return {"mean": stats_mod.mean(valid), "per_seed": per, "majority_answer": majority}

        answer_prior: dict[str, Any] | None = None
        if spec.answer_marker is not None:
            answer_prior = {region: prior_agg(counters) for region, counters in prior_counts.items()}

        curve: dict[str, Any] | None = None
        if curve_points:
            buckets = sorted({d for d, _, _, _ in curve_points})
            accuracy = []
            counts = []
            for b in buckets:
                hits = [c for d, c, _, _ in curve_points if d == b]
                accuracy.append(sum(hits) / len(hits))
                counts.append(len(hits))
            curve = {"buckets": buckets, "accuracy": accuracy, "counts": counts}

        def slope_fit(points: list[tuple[float, bool]]) -> dict[str, Any] | None:
            # Doc-level OLS of correctness on the difficulty axis: the
            # length-generalization slope (bead u55.3 - the preregistered
            # word-problem predictions compare these slopes across mechanisms).
            # CI95 from the slope's standard error; degenerate spreads -> None.
            if len(points) < 3:
                return None
            xs = [float(d) for d, _ in points]
            ys = [float(c) for _, c in points]
            n = len(xs)
            x_mean = sum(xs) / n
            y_mean = sum(ys) / n
            sxx = sum((x - x_mean) ** 2 for x in xs)
            if sxx <= 0:
                return None
            slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sxx
            intercept = y_mean - slope * x_mean
            if n > 2:
                rss = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
                se = (rss / (n - 2) / sxx) ** 0.5
            else:
                se = 0.0
            half = 1.96 * se
            return {
                "slope": slope,
                "ci95": [slope - half, slope + half],
                "intercept": intercept,
                "n_docs": n,
                "basis": "doc-level OLS of greedy exact-match on the difficulty axis",
            }

        length_slope: dict[str, Any] | None = None
        if curve_points:
            length_slope = {
                "held_out": slope_fit([(d, c) for d, c, r, _ in curve_points if r == "held_out"]),
                "all": slope_fit([(d, c) for d, c, _, _ in curve_points]),
            }
            categories = sorted({cat for _, _, _, cat in curve_points if cat is not None})
            if categories:
                # per-category held-out slopes: the mechanism-specificity
                # breakdown (e.g. the group task's S5/A5 vs Z60/S3 controls)
                length_slope["by_category"] = {
                    cat: slope_fit([(d, c) for d, c, r, cc in curve_points if r == "held_out" and cc == cat])
                    for cat in categories
                }

        results[name] = {
            "difficulty_axis": spec.difficulty_axis,
            "in_range_max": in_range_max,
            "exact_match": exact_match,
            "answer_prior": answer_prior,
            "perplexity": {"in_range": stats_mod.mean(ppl_in), "held_out": stats_mod.mean(ppl_out)},
            "curve": curve,
            "length_slope": length_slope,
            "skipped_too_long": skipped,
            "n_examples_per_seed": examples,
        }

    resolved_run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = artifacts_dir / "evals" / "tasks" / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Per-example receipts: every scored (prompt, expected, got, correct) row,
    # so claims about model behavior are auditable from the artifact alone.
    with (run_dir / "generations.jsonl").open("w", encoding="utf-8") as fh:
        for row in receipt_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    # Extrapolation curves (matplotlib, headless)
    curve_files: dict[str, str] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for name, rec in results.items():
            if not rec["curve"]:
                continue
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(rec["curve"]["buckets"], rec["curve"]["accuracy"], marker="o")
            if rec["in_range_max"] is not None:
                ax.axvline(rec["in_range_max"], linestyle="--", color="red", label="train-regime max")
                ax.legend(fontsize=7)
            ax.set_xlabel(rec["difficulty_axis"] or "difficulty")
            ax.set_ylabel("exact-match accuracy (greedy)")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{name}: extrapolation curve")
            fig.tight_layout()
            out_png = run_dir / f"curve_{name}.png"
            fig.savefig(out_png, dpi=120)
            plt.close(fig)
            curve_files[name] = out_png.name
    except Exception as exc:  # noqa: BLE001 - curves are best-effort; the JSON is the contract
        console.print(f"[yellow]curve rendering skipped: {exc}[/yellow]")

    summary: dict[str, Any] = {
        "schema_version": _EVAL_TASKS_SCHEMA_VERSION,
        "kind": "eval-tasks",
        "meta": {
            "run_id": resolved_run_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checkpoint": {
                "dir": str(checkpoint),
                "step": resolved_step,
                "attention_type": ckpt_meta.get("model_config", {}).get("attention_type"),
                "n_params": n_params,
                "budget": ckpt_meta.get("budget"),
                "lineage": ckpt_meta.get("lineage"),
                # the full recorded model config (additive in v2): variant
                # selectors (rgyl) resolve config knobs - braid_crossing_law,
                # ultrametric_mode, ... - against evaltasks evidence; without
                # this, any within-mechanism arm distinction is invisible to
                # the verdict engine and its hypotheses block at adjudication
                "model_config": ckpt_meta.get("model_config"),
            },
            "device": str(device),
            "seeds": seed_list,
            "examples_per_seed": examples,
            "decode_modes": [m for m, _ in decode_modes],
            "dials": dial_overrides or None,
            "git": _get_git_info(),
            "receipts": "generations.jsonl",
        },
        # Tamper-evidence (rz8.2 contract): the G2 verdict engine refuses
        # tainted artifacts. config_hash here covers the EVAL parameters plus
        # the checkpoint's model config (the full evidence-producing recipe).
        "provenance": _eval_tasks_provenance(
            ckpt_meta,
            seeds=seed_list,
            examples=examples,
            decode_modes=[m for m, _ in decode_modes],
            dials=dial_overrides or None,
        ),
        # dz9i: a checkpoint trained on a dirty tree must taint its evals even
        # when the EVAL itself runs on a clean tree - carry the TRAINING run's
        # provenance (the summary train.py writes beside the checkpoints dir);
        # null = no train summary found (legacy / synthetic checkpoints).
        "train_provenance": _read_train_provenance(checkpoint),
        "tasks": results,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    table = Table(title=f"eval-tasks — {checkpoint} @ {resolved_step}", box=box.SIMPLE_HEAVY)
    table.add_column("task", style="bold")
    table.add_column("EM in-range", justify="right")
    table.add_column("EM held-out", justify="right")
    table.add_column("ppl in/held", justify="right")
    table.add_column("slope held-out [CI95]", justify="right")
    table.add_column("axis")
    md_rows = []
    for name, rec in results.items():
        em = rec["exact_match"]
        em_in = em_out = "-"
        if em and em.get("greedy"):
            g_in, g_out = em["greedy"].get("in_range"), em["greedy"].get("held_out")
            em_in = f"{g_in['mean']:.3f}" if g_in else "-"
            em_out = f"{g_out['mean']:.3f}" if g_out else "-"
        ppl = f"{rec['perplexity']['in_range']:.1f}/{rec['perplexity']['held_out']:.1f}"
        slope_cell = "-"
        ls = rec.get("length_slope") or {}
        held = ls.get("held_out") if isinstance(ls, dict) else None
        if held:
            slope_cell = f"{held['slope']:+.4f} [{held['ci95'][0]:+.4f},{held['ci95'][1]:+.4f}]"
        table.add_row(name, em_in, em_out, ppl, slope_cell, rec["difficulty_axis"] or "(lm-only)")
        curve_cell = f"[curve]({curve_files[name]})" if name in curve_files else "-"
        md_rows.append(f"| {name} | {em_in} | {em_out} | {ppl} | {slope_cell} | {curve_cell} |")
    console.print(table)

    report_md = (
        f"# eval-tasks — {resolved_run_id}\n\n"
        f"- checkpoint: `{checkpoint}` @ step {resolved_step}\n"
        f"- attention_type: {summary['meta']['checkpoint']['attention_type']}\n"
        f"- n_params: {n_params:,}\n"
        f"- seeds: {seed_list} · examples/seed: {examples} · decode: {[m for m, _ in decode_modes]}\n\n"
        "| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |\n|---|---|---|---|---|---|\n"
        + "\n".join(md_rows)
        + "\n\nSee `summary.json` (schema mgr.evaltasks.v3) for the full contract output and "
        "`generations.jsonl` for per-example receipts.\n"
    )
    (run_dir / "run.md").write_text(report_md)
    console.print(f"[bold green]Wrote eval artifacts[/bold green] → {run_dir}")


@app.command("probe-charges")
def probe_charges(
    checkpoint: Annotated[Path, typer.Option(help="Checkpoint directory (rz8.1 layout); must be braid/rmatrix")],
    step: Annotated[int | None, typer.Option(help="Checkpoint step (default: latest)")] = None,
    task: Annotated[str, typer.Option(help="Diagnostic task with a category extractor (default: group)")] = "group",
    examples: Annotated[int, typer.Option(help="Documents to probe (split 80/20 train/test per category)")] = 240,
    seed: Annotated[int, typer.Option(help="Generator + probe-fit seed")] = 0,
    device_str: Annotated[str, typer.Option("--device", help="cpu | cuda")] = "cpu",
    dial: Annotated[
        list[str] | None, typer.Option(help="Difficulty dial override name=value (repeatable)")
    ] = None,
    artifacts_dir: Annotated[Path, typer.Option(help="Artifacts root")] = Path("artifacts"),
    run_id: Annotated[str | None, typer.Option(help="Run identifier (default: timestamp)")] = None,
) -> None:
    """Charge-decoding probe (bead u55.3, referee round 3 mandatory deliverable).

    Conservation alone proves nothing - conservation of a USELESS quantity is
    free. This probe tests the second half of the protected-memory story: are
    the conserved charges DECODABLE to the task's ground-truth group state?
    For each document, the per-head charge observables q_hk = <v, t_h(theta_k) v>
    (t = one-particle transfer matrices built from the FINAL braid layer's
    learned eta/rapidities; v = that layer's value sequences) feed a linear
    probe and a small-MLP probe predicting the composed group element.
    Preregistered expectation (hyp-rmatrix-charge-decodability): decodable for
    the abelian control (Z60), at or near chance for the non-solvable groups
    whose product is order-sensitive - if charges decode nothing anywhere, the
    honest conclusion is 'integrable structure conserves the wrong quantities
    for this task', reported as such.
    """
    import torch

    from nanochat.braid_attention_torch import BraidCausalSelfAttention, one_particle_transfer
    from nanochat.diagnostics_data import TASKS
    from nanochat.tokenizer import get_tokenizer

    device = torch.device(device_str)
    if task not in TASKS:
        console.print(f"[bold red]Unknown task {task!r}; available: {', '.join(sorted(TASKS))}[/bold red]")
        raise typer.Exit(code=2)
    spec = TASKS[task]
    if spec.category is None or spec.answer_marker is None:
        console.print(f"[bold red]Task {task!r} has no category extractor / answer marker.[/bold red]")
        raise typer.Exit(code=2)

    dial_overrides: dict[str, float] = {}
    for pair in dial or []:
        key, sep, value = pair.partition("=")
        if not sep:
            console.print(f"[bold red]--dial expects name=value, got {pair!r}[/bold red]")
            raise typer.Exit(code=2)
        dial_overrides[key.strip()] = float(value)

    model, ckpt_meta, resolved_step = _load_eval_checkpoint(checkpoint, step, device)
    model_cfg = ckpt_meta.get("model_config", {})
    if model_cfg.get("attention_type") != "braid" or model_cfg.get("braid_crossing_law") != "rmatrix":
        console.print(
            "[bold red]probe-charges requires a braid/rmatrix checkpoint "
            f"(got attention_type={model_cfg.get('attention_type')!r}, "
            f"braid_crossing_law={model_cfg.get('braid_crossing_law')!r}).[/bold red]"
        )
        raise typer.Exit(code=2)
    tok = get_tokenizer()

    braid_layers = [m for m in model.modules() if isinstance(m, BraidCausalSelfAttention)]
    final_attn = braid_layers[-1]
    captured: dict[str, torch.Tensor] = {}

    def _capture(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        captured["x"] = args[0].detach()

    hook = final_attn.register_forward_pre_hook(_capture, with_kwargs=True)

    eta = final_attn.rmatrix_eta().to(torch.float64)  # (H,)
    u_all = final_attn.rmatrix_rapidities().to(torch.float64)  # (H, S)
    n_head = int(model.config.n_head)
    head_dim = int(model.config.n_embd) // n_head
    # two probe points beyond the rapidity range generate the charge tower
    theta_offsets = (0.7, 1.6)
    transfer_cache: dict[tuple[int, int, int], torch.Tensor] = {}

    def charge_features(v_heads: torch.Tensor) -> list[float]:
        # v_heads: (T, H, D) fp32 -> n_head * len(theta_offsets) features
        T = v_heads.size(0)
        feats: list[float] = []
        v64 = v_heads.to(torch.float64)
        for h in range(n_head):
            vh = v64[:, h, :]  # (T, D)
            denom = float((vh * vh).sum()) or 1.0
            for k, off in enumerate(theta_offsets):
                key = (h, T, k)
                if key not in transfer_cache:
                    u_h = u_all[h, :T]
                    transfer_cache[key] = one_particle_transfer(float(u_h[-1]) + off, u_h, float(eta[h]))
                t_mat = transfer_cache[key]
                feats.append(float((vh * (t_mat @ vh)).sum()) / denom)
        return feats

    splits = spec.generate(max(examples, 30), seed, spec.resolve_dials(dial_overrides or None))
    docs = (splits["train"] + splits["test"])[:examples]
    by_cat: dict[str, list[tuple[list[float], str]]] = {}
    skipped = 0
    with console.status("[bold cyan]extracting charge features…[/bold cyan]"):
        for doc in docs:
            parts = spec.split_prompt(doc)
            cat = spec.category(doc)
            if parts is None or cat is None:
                continue
            prompt, expected = parts
            ids = tok.encode(prompt)
            if len(ids) > model.config.sequence_len:
                skipped += 1
                continue
            with torch.inference_mode():
                model(torch.tensor([ids], dtype=torch.long, device=device))
                x = captured["x"]  # (1, T, C)
                # c_v projects to n_kv_head value heads (GQA); repeat them to
                # the n_head query heads exactly as the attention layer does,
                # so charge features pair each query head's (eta, rapidities)
                # with the value sequence it actually transports
                n_kv = int(model.config.n_kv_head)
                v_kv = final_attn.c_v(x).view(x.size(1), n_kv, head_dim)
                v = v_kv.repeat_interleave(n_head // n_kv, dim=1)
                feats = charge_features(v)
            by_cat.setdefault(cat, []).append((feats, expected))
    hook.remove()

    def fit_probe(rows: list[tuple[list[float], str]], hidden: int | None) -> tuple[float, float, int, float]:
        # (train_acc, test_acc, n_classes, majority_floor); deterministic fp32
        # full-batch fit. majority_floor = the best-constant-class accuracy on
        # the test split (the house answer-prior convention): at small samples
        # the theoretical 1/|G| floor wildly understates chance, since a random
        # probe already scores ~1/n_observed_classes.
        gen = torch.Generator().manual_seed(seed)
        labels = sorted({y for _, y in rows})
        idx = {y: i for i, y in enumerate(labels)}
        X = torch.tensor([f for f, _ in rows], dtype=torch.float32)
        y = torch.tensor([idx[lab] for _, lab in rows], dtype=torch.long)
        perm = torch.randperm(len(rows), generator=gen)
        n_test = max(1, len(rows) // 5)
        te, tr = perm[:n_test], perm[n_test:]
        # standardize with TRAIN statistics only (no test-set leakage)
        mu, sd = X[tr].mean(0), X[tr].std(0)
        X = (X - mu) / (sd + 1e-8)
        majority_floor = float(torch.bincount(y[te]).max()) / float(len(te))
        if hidden:
            net: torch.nn.Module = torch.nn.Sequential(
                torch.nn.Linear(X.size(1), hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, len(labels))
            )
        else:
            net = torch.nn.Linear(X.size(1), len(labels))
        torch.manual_seed(seed)
        for m in net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        for _ in range(300):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(X[tr]), y[tr])
            loss.backward()
            opt.step()
        with torch.no_grad():
            tr_acc = float((net(X[tr]).argmax(-1) == y[tr]).float().mean())
            te_acc = float((net(X[te]).argmax(-1) == y[te]).float().mean())
        return tr_acc, te_acc, len(labels), majority_floor

    group_sizes = {"s5": 120, "a5": 60, "z60": 60, "s3": 6}
    results: dict[str, Any] = {}
    table = Table(title=f"charge-decoding probe — {checkpoint} @ {resolved_step}", box=box.SIMPLE_HEAVY)
    for col in ("category", "n_docs", "classes", "floor", "linear test acc", "mlp test acc", "verdict"):
        table.add_column(col, justify="right" if col != "category" else "left")
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        if len(rows) < 10:
            results[cat] = {"n_docs": len(rows), "skipped": "fewer than 10 docs"}
            table.add_row(cat, str(len(rows)), "-", "-", "-", "-", "[yellow]skipped[/yellow]")
            continue
        lin_tr, lin_te, n_cls, floor_lin = fit_probe(rows, hidden=None)
        mlp_tr, mlp_te, _, _ = fit_probe(rows, hidden=64)
        chance = 1.0 / group_sizes.get(cat, n_cls)
        # the operative floor: best-constant accuracy on the test split (never
        # below the theoretical 1/|G|) - a random probe already scores
        # ~1/n_observed at small samples, so 1/|G| alone wildly inflates
        # "over-chance" for the large groups
        floor_used = max(chance, floor_lin)
        decodable = max(lin_te, mlp_te) > 2.0 * floor_used
        results[cat] = {
            "n_docs": len(rows),
            "n_classes_observed": n_cls,
            "chance": chance,
            "majority_floor": floor_lin,
            "floor_used": floor_used,
            "linear": {"train_acc": lin_tr, "test_acc": lin_te},
            "mlp": {"train_acc": mlp_tr, "test_acc": mlp_te},
            "decodable_at_2x_chance": decodable,
        }
        verdict = "[green]decodable[/green]" if decodable else "[red]~chance[/red]"
        table.add_row(cat, str(len(rows)), str(n_cls), f"{floor_used:.3f}", f"{lin_te:.3f}", f"{mlp_te:.3f}", verdict)
    console.print(table)
    if skipped:
        console.print(f"[yellow]{skipped} docs skipped (prompt longer than the rotary cache)[/yellow]")

    # Dissociation observable (the preregistered hyp-rmatrix-charge-decodability
    # contract): abelian over-chance ratio vs the BEST non-solvable over-chance
    # ratio (clamped below at 1.0 so below-chance probe noise cannot inflate
    # the dissociation). decodable-everywhere and decodable-nowhere both push
    # the ratio toward 1 - only the predicted dissociation pattern clears 2.0.
    def _over_chance(cat: str) -> float | None:
        rec = results.get(cat)
        if not isinstance(rec, dict) or "floor_used" not in rec:
            return None
        best = max(rec["linear"]["test_acc"], rec["mlp"]["test_acc"])
        return best / rec["floor_used"] if rec["floor_used"] > 0 else None

    dissociation: dict[str, Any] | None = None
    abelian = _over_chance("z60")
    nonsolvable_vals = [v for v in (_over_chance("s5"), _over_chance("a5")) if v is not None]
    if abelian is not None and nonsolvable_vals:
        nonsolvable = max(nonsolvable_vals)
        dissociation = {
            "abelian_over_chance": abelian,
            "nonsolvable_over_chance": nonsolvable,
            "ratio": abelian / max(nonsolvable, 1.0),
            "basis": (
                "max(linear, mlp) test_acc / floor_used, where floor_used = "
                "max(1/|G|, best-constant accuracy on the probe's test split); "
                "ratio denominator clamped at 1.0"
            ),
        }

    from nanochat.report import build_provenance

    resolved_run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = artifacts_dir / "probes" / "charges" / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "mgr.chargeprobe.v1",
        "kind": "probe-charges",
        "meta": {
            "run_id": resolved_run_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # evaltasks-style checkpoint block: the verdict engine's arm
            # matching, variant selectors, and budget cohorts read these
            "checkpoint": {
                "dir": str(checkpoint),
                "step": resolved_step,
                "attention_type": model_cfg.get("attention_type"),
                "braid_crossing_law": model_cfg.get("braid_crossing_law"),
                "n_params": sum(p.numel() for p in model.parameters()),
                "budget": ckpt_meta.get("budget"),
                "lineage": ckpt_meta.get("lineage"),
            },
            "task": task,
            "examples": examples,
            "seed": seed,
            "dials": dial_overrides or None,
            "theta_offsets": list(theta_offsets),
            "feature_basis": "per-head <v, t(theta_k) v>/<v, v> at the final braid layer",
            "git": _get_git_info(),
        },
        "provenance": build_provenance(
            {
                "model_config": ckpt_meta.get("model_config"),
                "probe": {"task": task, "examples": examples, "seed": seed, "dials": dial_overrides or None},
            }
        ),
        "categories": results,
        "dissociation": dissociation,
        "skipped_too_long": skipped,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    md = [
        f"# probe-charges — {resolved_run_id}",
        "",
        f"- checkpoint: `{checkpoint}` @ step {resolved_step}",
        f"- task: {task} · examples: {examples} · seed: {seed}",
        "- features: per-head charge observables `<v, t(theta_k) v>` of the final braid layer",
        "",
        "| category | n | classes | floor | linear test | mlp test | decodable@2x |",
        "|---|---|---|---|---|---|---|",
    ]
    for cat, rec in sorted(results.items()):
        if "skipped" in rec:
            md.append(f"| {cat} | {rec['n_docs']} | - | - | - | - | skipped |")
        else:
            md.append(
                f"| {cat} | {rec['n_docs']} | {rec['n_classes_observed']} | {rec['floor_used']:.3f} "
                f"| {rec['linear']['test_acc']:.3f} | {rec['mlp']['test_acc']:.3f} "
                f"| {rec['decodable_at_2x_chance']} |"
            )
    md.append("")
    md.append(
        "Both halves of the protected-memory story are required: conservation "
        "(certify: braid.rmatrix_mass_partition_charge_conserved) AND decodability (this probe). "
        "If conservation holds but decodability fails, the conclusion is that the integrable "
        "structure conserves the wrong quantities for this task - a result, not a bug."
    )
    (run_dir / "run.md").write_text("\n".join(md) + "\n")
    console.print(f"[bold green]Wrote charge-probe artifacts[/bold green] → {run_dir}")


# ---------------------------------------------------------------------------
# Shared curve math for the precision-axis bench commands (precision-curve,
# depth-curve). Quality = ppl_full/ppl_point per sweep point, curves anchored
# at (1.0, 1.0), trapezoid AUC over an explicit window normalized by span.
# ---------------------------------------------------------------------------


def _curve_eval_summary(artifacts_dir: Path, rid: str) -> dict[str, Any]:
    path = artifacts_dir / "evals" / "tasks" / rid / "summary.json"
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


def _curve_task_ppl(summary: dict[str, Any], task: str) -> float:
    return float(summary["tasks"][task]["perplexity"]["in_range"])


def _curve_model_config(summary: dict[str, Any], rid: str) -> dict[str, Any]:
    mc = summary["meta"]["checkpoint"].get("model_config")
    if not isinstance(mc, dict):
        raise ValueError(f"{rid}: artifact does not record meta.checkpoint.model_config")
    return mc


def _curve_knob_points(
    artifacts_dir: Path,
    runs: list[str],
    task: str,
    knob: str,
    denom_from: Any,
) -> list[tuple[float, float]]:
    pts = []
    for rid in runs:
        d = _curve_eval_summary(artifacts_dir, rid)
        mc = _curve_model_config(d, rid)
        val = mc.get(knob)
        if val is None:
            raise ValueError(f"{rid}: artifact does not record {knob} (run with --model-override)")
        denom = denom_from(mc)
        pts.append((float(val) / denom, _curve_task_ppl(d, task)))
    return pts


def _curve_quality(pts: list[tuple[float, float]], anchor_ppl: float) -> list[tuple[float, float]]:
    return sorted([(f, anchor_ppl / p if p > 0 else 0.0) for f, p in pts] + [(1.0, 1.0)])


def _curve_interp(curve: list[tuple[float, float]], f: float) -> float:
    for (f0, q0), (f1, q1) in zip(curve, curve[1:]):
        if f0 <= f <= f1:
            return q0 if f1 == f0 else q0 + (q1 - q0) * (f - f0) / (f1 - f0)
    return curve[0][1] if f < curve[0][0] else curve[-1][1]


def _curve_auc(curve: list[tuple[float, float]], window_lo: float, window_hi: float = 1.0) -> float:
    clipped = (
        [(window_lo, _curve_interp(curve, window_lo))]
        + [(f, q) for f, q in curve if window_lo < f < window_hi]
        + [(window_hi, _curve_interp(curve, window_hi))]
    )
    area = 0.0
    for (f0, q0), (f1, q1) in zip(clipped, clipped[1:]):
        area += (f1 - f0) * (q0 + q1) / 2.0
    span = clipped[-1][0] - clipped[0][0]
    return area / span if span > 0 else 0.0


@app.command("precision-curve")
def precision_curve(
    digit_run: Annotated[list[str], typer.Option(help="eval-tasks run-id of one digit-truncation sweep point (repeatable)")],
    float_run: Annotated[list[str], typer.Option(help="eval-tasks run-id of one float-quantization sweep point (repeatable)")],
    digit_full: Annotated[str, typer.Option(help="run-id of the digit arm's FULL-precision eval (the anchor)")],
    float_full: Annotated[str, typer.Option(help="run-id of the float arm's FULL-precision eval (the anchor)")],
    task: Annotated[str, typer.Option(help="Task whose in-range perplexity is the quality metric")] = "hier",
    seed: Annotated[int, typer.Option(help="Checkpoint-seed pairing this artifact represents (one artifact per seed)")] = 0,
    window_hi: Annotated[
        float,
        typer.Option(
            help=(
                "Upper edge of the AUC window (default 1.0 = full axis). A registered sub-1.0 edge isolates the "
                "deep-compression region where graceful-vs-cliff actually discriminates (bead kgj1: the tcuy "
                "refutation showed full-axis AUC dilutes a narrow cliff with the wide near-lossless region)."
            )
        ),
    ] = 1.0,
    artifacts_dir: Annotated[Path, typer.Option(help="Artifacts root")] = Path("artifacts"),
    run_id: Annotated[str | None, typer.Option(help="Run identifier (default: timestamp)")] = None,
) -> None:
    """Precision curves: digit truncation vs float quantization (bead tcuy).

    Reads per-point eval artifacts (the knob - ultrametric_digits_k or
    eval_weight_quant_bits - is taken from each artifact's recorded
    meta.checkpoint.model_config), normalizes both arms onto a shared
    memory-fraction axis (digits: k/K of the digit path; float: bits/32 of
    the weights), computes quality = ppl_full / ppl_point per point, and the
    trapezoid AUC per arm anchored at (1.0, 1.0). The headline scalar is
    results.auc_ratio = AUC_digit / AUC_float - the preregistered
    graceful-vs-cliff observable of hyp-padic-truncation-graceful (and, over
    the registered deep window via --window-hi, of its K=16 successor). One
    artifact per checkpoint-seed pairing: the verdict engine counts each as
    one observation (bench schema, budget-exempt).
    """
    from nanochat.report import build_provenance

    digit_anchor = _curve_task_ppl(_curve_eval_summary(artifacts_dir, digit_full), task)
    float_anchor = _curve_task_ppl(_curve_eval_summary(artifacts_dir, float_full), task)
    digit_pts = _curve_knob_points(
        artifacts_dir, digit_run, task, "ultrametric_digits_k", lambda mc: float(mc.get("ultrametric_K", 8))
    )
    float_pts = _curve_knob_points(artifacts_dir, float_run, task, "eval_weight_quant_bits", lambda mc: 32.0)

    curve_digit_pts = _curve_quality(digit_pts, digit_anchor)
    curve_float_pts = _curve_quality(float_pts, float_anchor)
    # "At matched memory" (the registered phrase): both AUCs integrate over
    # the COMMON fraction window [max(arm minima), window_hi]. Normalizing
    # each arm over its OWN span would penalize whichever arm was swept
    # deeper into compression - a bias in the registered direction, caught
    # in review before any evidence was produced.
    window_lo = max(curve_digit_pts[0][0], curve_float_pts[0][0])
    if not window_lo < window_hi <= 1.0:
        raise ValueError(f"window must satisfy lo < hi <= 1.0, got [{window_lo}, {window_hi}]")

    auc_digit = _curve_auc(curve_digit_pts, window_lo, window_hi)
    auc_float = _curve_auc(curve_float_pts, window_lo, window_hi)
    curve_digit = [{"fraction": f, "quality": q} for f, q in curve_digit_pts]
    curve_float = [{"fraction": f, "quality": q} for f, q in curve_float_pts]
    ratio = auc_digit / auc_float if auc_float > 0 else float("inf")

    table = Table(title=f"precision curves — {task} (seed pairing {seed})", box=box.SIMPLE_HEAVY)
    for col in ("arm", "points", "AUC (quality x memory-fraction)"):
        table.add_column(col, justify="right")
    table.add_row("digit truncation", str(len(digit_pts)), f"{auc_digit:.4f}")
    table.add_row("float quantization", str(len(float_pts)), f"{auc_float:.4f}")
    table.add_row("ratio (digit/float)", "-", f"{ratio:.3f}")
    console.print(table)

    resolved_run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = artifacts_dir / "bench" / "precision_curves" / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "mgr.bench.precision_curves.v1",
        "kind": "precision-curve",
        "mechanism": "ultrametric",
        "meta": {
            "run_id": resolved_run_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "task": task,
            "seed": seed,
            "device": "cpu",
            "digit_runs": list(digit_run),
            "float_runs": list(float_run),
            "digit_full": digit_full,
            "float_full": float_full,
            "normalization": (
                "digits: k/K of the digit path; float: bits/32 of weights; quality = ppl_full/ppl_point; "
                f"both AUCs integrate over the COMMON fraction window [max(arm minima), {window_hi}] "
                "(matched memory), anchored at (1,1), normalized by the window span"
            ),
            "git": _get_git_info(),
        },
        "provenance": build_provenance(
            {
                "precision_curve": {
                    "task": task,
                    "seed": seed,
                    "digit_runs": list(digit_run),
                    "float_runs": list(float_run),
                    "window_hi": window_hi,
                }
            }
        ),
        "results": {
            "auc_digit": auc_digit,
            "auc_float": auc_float,
            # The headline ratio's KEY encodes the window protocol: bench arm
            # matching has no variant selectors (engine matches on mechanism
            # only), so full-axis and deep-window artifacts must expose
            # DISJOINT observables or re-adjudication would pool them across
            # hypotheses - the campaign-killer caught at kgj1 registration.
            ("auc_ratio" if window_hi >= 1.0 else "auc_ratio_deepwindow"): ratio,
            "common_window_lo": window_lo,
            "common_window_hi": window_hi,
            "curve_digit": curve_digit,
            "curve_float": curve_float,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    md = [
        f"# precision-curve — {resolved_run_id}",
        "",
        f"- task {task} · seed pairing {seed}",
        f"- AUC digit {auc_digit:.4f} vs float {auc_float:.4f} -> ratio {ratio:.3f}",
        "",
        "| arm | fraction | quality |",
        "|---|---|---|",
    ]
    for name, curve in (("digit", curve_digit), ("float", curve_float)):
        for pt in curve:
            md.append(f"| {name} | {pt['fraction']:.3f} | {pt['quality']:.4f} |")
    (run_dir / "run.md").write_text("\n".join(md) + "\n")
    console.print(f"[bold green]Wrote precision-curve artifacts[/bold green] → {run_dir}")


@app.command("depth-curve")
def depth_curve(
    shallow_run: Annotated[
        list[str], typer.Option(help="digit-truncation eval run-id at the SHALLOW depth (repeatable)")
    ],
    deep_run: Annotated[list[str], typer.Option(help="digit-truncation eval run-id at the DEEP depth (repeatable)")],
    shallow_full: Annotated[str, typer.Option(help="run-id of the shallow depth's FULL-precision eval (anchor)")],
    deep_full: Annotated[str, typer.Option(help="run-id of the deep depth's FULL-precision eval (anchor)")],
    float_shallow_run: Annotated[
        list[str] | None, typer.Option(help="OPTIONAL float-quant eval run-id at the shallow depth (diagnostic)")
    ] = None,
    float_deep_run: Annotated[
        list[str] | None, typer.Option(help="OPTIONAL float-quant eval run-id at the deep depth (diagnostic)")
    ] = None,
    float_shallow_full: Annotated[
        str | None, typer.Option(help="run-id of the shallow float arm's FULL-precision eval (anchor)")
    ] = None,
    float_deep_full: Annotated[
        str | None, typer.Option(help="run-id of the deep float arm's FULL-precision eval (anchor)")
    ] = None,
    task: Annotated[str, typer.Option(help="Task whose in-range perplexity is the quality metric")] = "hier",
    seed: Annotated[int, typer.Option(help="Checkpoint-seed pairing this artifact represents")] = 0,
    artifacts_dir: Annotated[Path, typer.Option(help="Artifacts root")] = Path("artifacts"),
    run_id: Annotated[str | None, typer.Option(help="Run identifier (default: timestamp)")] = None,
) -> None:
    """Depth contrast of digit-truncation curves (bead 9qeq).

    The flat-error lemma predicts the digit-truncation quality curve is
    INDEPENDENT of model depth on the hard-digit path (per-layer truncation
    errors live below digit k and cannot accumulate through depth). The
    headline scalar is results.digit_depth_auc_absdev =
    |AUC_digit(deep)/AUC_digit(shallow) - 1| - the preregistered observable
    of hyp-padic-truncation-depth-independent (comparator <=). AUCs are
    within-arm (each depth normalized to its own full-precision anchor) over
    the common fraction window across the two depths. The float-quantization
    depth contrast, when float runs are supplied, is recorded as
    results.float_depth_auc_absdev - a DIAGNOSTIC for the compounding
    contrast clause, deliberately not the adjudicated scalar. One artifact
    per checkpoint-seed pairing (bench schema, budget-exempt).
    """
    from nanochat.report import build_provenance

    def _digit_group(runs: list[str], full: str, label: str) -> tuple[list[tuple[float, float]], float, int]:
        """(quality curve, anchor ppl, n_layer) for one depth's digit arm, with hygiene checks."""
        n_layers = set()
        for rid in [*runs, full]:
            mc = _curve_model_config(_curve_eval_summary(artifacts_dir, rid), rid)
            if mc.get("attention_type") != "ultrametric" or not mc.get("ultrametric_hard_digits"):
                raise ValueError(f"{rid}: digit arm requires a hard-digit ultrametric checkpoint ({label})")
            if not isinstance(mc.get("n_layer"), int):
                raise ValueError(f"{rid}: artifact does not record n_layer")
            n_layers.add(int(mc["n_layer"]))
        if len(n_layers) != 1:
            raise ValueError(f"{label} digit arm mixes depths {sorted(n_layers)} - one depth per group")
        anchor = _curve_task_ppl(_curve_eval_summary(artifacts_dir, full), task)
        pts = _curve_knob_points(
            artifacts_dir, runs, task, "ultrametric_digits_k", lambda mc: float(mc.get("ultrametric_K", 8))
        )
        return _curve_quality(pts, anchor), anchor, n_layers.pop()

    def _float_group(runs: list[str], full: str, label: str) -> tuple[list[tuple[float, float]], int]:
        n_layers = set()
        for rid in [*runs, full]:
            mc = _curve_model_config(_curve_eval_summary(artifacts_dir, rid), rid)
            if mc.get("attention_type") != "standard":
                raise ValueError(f"{rid}: float arm requires a standard checkpoint ({label})")
            if not isinstance(mc.get("n_layer"), int):
                raise ValueError(f"{rid}: artifact does not record n_layer")
            n_layers.add(int(mc["n_layer"]))
        if len(n_layers) != 1:
            raise ValueError(f"{label} float arm mixes depths {sorted(n_layers)} - one depth per group")
        anchor = _curve_task_ppl(_curve_eval_summary(artifacts_dir, full), task)
        pts = _curve_knob_points(artifacts_dir, runs, task, "eval_weight_quant_bits", lambda mc: 32.0)
        return _curve_quality(pts, anchor), n_layers.pop()

    curve_shallow, _, n_shallow = _digit_group(shallow_run, shallow_full, "shallow")
    curve_deep, _, n_deep = _digit_group(deep_run, deep_full, "deep")
    if not n_shallow < n_deep:
        raise ValueError(f"shallow n_layer {n_shallow} must be strictly less than deep n_layer {n_deep}")
    digit_window_lo = max(curve_shallow[0][0], curve_deep[0][0])
    auc_digit_shallow = _curve_auc(curve_shallow, digit_window_lo)
    auc_digit_deep = _curve_auc(curve_deep, digit_window_lo)
    digit_absdev = abs(auc_digit_deep / auc_digit_shallow - 1.0) if auc_digit_shallow > 0 else float("inf")

    float_results: dict[str, Any] = {
        "auc_float_shallow": None,
        "auc_float_deep": None,
        "float_depth_auc_absdev": None,
        "float_window_lo": None,
    }
    have_float = bool(float_shallow_run) and bool(float_deep_run)
    if have_float:
        if float_shallow_full is None or float_deep_full is None:
            raise ValueError("float diagnostic requires both --float-shallow-full and --float-deep-full anchors")
        fcurve_shallow, fn_shallow = _float_group(float_shallow_run or [], float_shallow_full, "shallow")
        fcurve_deep, fn_deep = _float_group(float_deep_run or [], float_deep_full, "deep")
        if (fn_shallow, fn_deep) != (n_shallow, n_deep):
            raise ValueError(
                f"float arm depths ({fn_shallow}, {fn_deep}) do not match digit arm depths ({n_shallow}, {n_deep})"
            )
        float_window_lo = max(fcurve_shallow[0][0], fcurve_deep[0][0])
        auc_float_shallow = _curve_auc(fcurve_shallow, float_window_lo)
        auc_float_deep = _curve_auc(fcurve_deep, float_window_lo)
        float_results = {
            "auc_float_shallow": auc_float_shallow,
            "auc_float_deep": auc_float_deep,
            "float_depth_auc_absdev": (
                abs(auc_float_deep / auc_float_shallow - 1.0) if auc_float_shallow > 0 else float("inf")
            ),
            "float_window_lo": float_window_lo,
        }

    table = Table(title=f"depth curves — {task} (seed pairing {seed}, L{n_shallow} vs L{n_deep})", box=box.SIMPLE_HEAVY)
    for col in ("arm", "AUC shallow", "AUC deep", "|deep/shallow - 1|"):
        table.add_column(col, justify="right")
    table.add_row("digit truncation", f"{auc_digit_shallow:.4f}", f"{auc_digit_deep:.4f}", f"{digit_absdev:.4f}")
    if have_float:
        table.add_row(
            "float quantization (diagnostic)",
            f"{float_results['auc_float_shallow']:.4f}",
            f"{float_results['auc_float_deep']:.4f}",
            f"{float_results['float_depth_auc_absdev']:.4f}",
        )
    console.print(table)

    resolved_run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = artifacts_dir / "bench" / "depth_curves" / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "mgr.bench.depth_curves.v1",
        "kind": "depth-curve",
        "mechanism": "ultrametric",
        "meta": {
            "run_id": resolved_run_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "task": task,
            "seed": seed,
            "device": "cpu",
            "shallow_runs": list(shallow_run),
            "deep_runs": list(deep_run),
            "shallow_full": shallow_full,
            "deep_full": deep_full,
            "float_shallow_runs": list(float_shallow_run or []),
            "float_deep_runs": list(float_deep_run or []),
            "float_shallow_full": float_shallow_full,
            "float_deep_full": float_deep_full,
            "normalization": (
                "within-arm: each depth's quality = own ppl_full/ppl_point over digit fractions k/K, "
                "anchored at (1,1); AUCs integrate over the common fraction window across the two depths, "
                "normalized by the window span; headline |AUC_deep/AUC_shallow - 1| on the digit arm only"
            ),
            "git": _get_git_info(),
        },
        "provenance": build_provenance(
            {
                "depth_curve": {
                    "task": task,
                    "seed": seed,
                    "shallow_runs": list(shallow_run),
                    "deep_runs": list(deep_run),
                    "n_layer_shallow": n_shallow,
                    "n_layer_deep": n_deep,
                }
            }
        ),
        "results": {
            "n_layer_shallow": n_shallow,
            "n_layer_deep": n_deep,
            "auc_digit_shallow": auc_digit_shallow,
            "auc_digit_deep": auc_digit_deep,
            "digit_depth_auc_absdev": digit_absdev,
            "digit_window_lo": digit_window_lo,
            "curve_digit_shallow": [{"fraction": f, "quality": q} for f, q in curve_shallow],
            "curve_digit_deep": [{"fraction": f, "quality": q} for f, q in curve_deep],
            **float_results,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    md = [
        f"# depth-curve — {resolved_run_id}",
        "",
        f"- task {task} · seed pairing {seed} · L{n_shallow} vs L{n_deep}",
        f"- digit AUC {auc_digit_shallow:.4f} (L{n_shallow}) vs {auc_digit_deep:.4f} (L{n_deep}) "
        f"-> |deep/shallow - 1| = {digit_absdev:.4f}",
    ]
    if have_float:
        md.append(
            f"- float diagnostic AUC {float_results['auc_float_shallow']:.4f} vs "
            f"{float_results['auc_float_deep']:.4f} -> absdev {float_results['float_depth_auc_absdev']:.4f}"
        )
    md += ["", "| arm | depth | fraction | quality |", "|---|---|---|---|"]
    for name, nl, curve in (("digit", n_shallow, curve_shallow), ("digit", n_deep, curve_deep)):
        for f, q in curve:
            md.append(f"| {name} | L{nl} | {f:.3f} | {q:.4f} |")
    (run_dir / "run.md").write_text("\n".join(md) + "\n")
    console.print(f"[bold green]Wrote depth-curve artifacts[/bold green] → {run_dir}")


@app.command("bench-ultrametric")
def bench_ultrametric(
    context_lengths: Annotated[str, typer.Option(help="Comma-separated T values")] = "1024,4096",
    repeats: Annotated[int, typer.Option(help="Best-of-N timing per point")] = 3,
    seed: Annotated[int, typer.Option(help="Init/data seed")] = 0,
    artifacts_dir: Annotated[Path, typer.Option(help="Artifacts root")] = Path("artifacts"),
    run_id: Annotated[str | None, typer.Option(help="Run identifier (default: timestamp)")] = None,
) -> None:
    """Wall-clock benchmark of the ultrametric attention paths (bead 33dd).

    kernel (O(T^2 K)) vs balltree (O(K T log T)) full prefill, and kernel vs
    trie per-token decode - the versioned artifact
    (mgr.bench.ultrametric_paths.v1) that hyp-balltree-valued-attention-speedup
    and hyp-ultrametric-trie-decode-speedup adjudicate against. The two paths
    compute the same function (thm-balltree-exact-attention), so wall-clock is
    the entire claim; the recorded load average documents box contention.
    """
    from nanochat.report import build_provenance
    from nanochat.ultrametric_attention_torch import bench_ultrametric_paths

    lengths = tuple(int(t.strip()) for t in context_lengths.split(",") if t.strip())
    if not lengths:
        console.print("[bold red]--context-lengths must name at least one T.[/bold red]")
        raise typer.Exit(code=2)
    load_before = os.getloadavg()
    with console.status("[bold cyan]benchmarking ultrametric paths…[/bold cyan]"):
        results = bench_ultrametric_paths(context_lengths=lengths, repeats=repeats, seed=seed)

    table = Table(title="ultrametric paths — wall clock (CPU)", box=box.SIMPLE_HEAVY)
    for col in ("T", "kernel ms", "balltree ms", "fwd speedup", "trie-decode speedup"):
        table.add_column(col, justify="right")
    decode_by_t = {int(r["Tk"]): r for r in results["decode"]}
    for row in results["forward"]:
        t = int(row["T"])
        dec = decode_by_t.get(t)
        table.add_row(
            str(t),
            f"{1000 * row['kernel_s']:.1f}",
            f"{1000 * row['balltree_s']:.1f}",
            f"{row['speedup']:.1f}x",
            f"{dec['speedup']:.1f}x" if dec else "-",
        )
    console.print(table)
    console.print(f"[dim]loadavg at start: {load_before} (timings on a busy box understate speedups)[/dim]")

    resolved_run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = artifacts_dir / "bench" / "ultrametric_paths" / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "mgr.bench.ultrametric_paths.v1",
        "kind": "bench-ultrametric",
        "mechanism": "ultrametric",
        "meta": {
            "run_id": resolved_run_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed": seed,
            "repeats": repeats,
            "context_lengths": list(lengths),
            "device": "cpu",
            "loadavg": list(load_before),
            "git": _get_git_info(),
        },
        "provenance": build_provenance(
            {"bench": {"kind": "ultrametric_paths", "seed": seed, "repeats": repeats, "lengths": list(lengths)}}
        ),
        "results": results,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    md_lines = [
        f"# bench-ultrametric — {resolved_run_id}",
        "",
        f"- seed {seed} · repeats {repeats} · loadavg {load_before}",
        "",
        "| T | kernel ms | balltree ms | fwd speedup | trie-decode speedup |",
        "|---|---|---|---|---|",
    ]
    for row in results["forward"]:
        t = int(row["T"])
        dec = decode_by_t.get(t)
        md_lines.append(
            f"| {t} | {1000 * row['kernel_s']:.1f} | {1000 * row['balltree_s']:.1f} "
            f"| {row['speedup']:.1f}x | {dec['speedup']:.1f}x |" if dec else
            f"| {t} | {1000 * row['kernel_s']:.1f} | {1000 * row['balltree_s']:.1f} | {row['speedup']:.1f}x | - |"
        )
    (run_dir / "run.md").write_text("\n".join(md_lines) + "\n")
    console.print(f"[bold green]Wrote bench artifacts[/bold green] → {run_dir}")


def _sample_stream(
    model: Any,
    tok: Any,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int,
    stop_at_separator: bool,
    on_token: Any = None,
) -> dict[str, Any]:
    """The testable core of mgr sample: stream one generation, returning
    {tokens, text, elapsed_s, tokens_per_s}. Stops at the document separator
    (the trained format's answer terminator) unless told otherwise - the same
    contract the eval scorer uses (kbj2 finding)."""
    stop_id = None
    if stop_at_separator:
        get_bos = getattr(tok, "get_bos_token_id", None)
        if callable(get_bos):
            stop_id = get_bos()
    pieces: list[int] = []
    t0 = time.perf_counter()
    for piece in model.generate(prompt_ids, max_tokens=max_tokens, temperature=temperature, top_k=top_k, seed=seed):
        if stop_id is not None and piece == stop_id:
            break
        pieces.append(piece)
        if on_token is not None:
            on_token(tok.decode(pieces))
    elapsed = time.perf_counter() - t0
    return {
        "tokens": pieces,
        "text": tok.decode(pieces),
        "elapsed_s": elapsed,
        "tokens_per_s": (len(pieces) / elapsed) if elapsed > 0 and pieces else 0.0,
    }


def _sample_peak_memory(device: Any) -> str:
    import torch

    if device.type == "cuda":
        return f"{torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB (CUDA peak)"
    import resource

    # ru_maxrss is KiB on Linux; process-lifetime peak (coarse but honest on CPU)
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB (process peak RSS)"


@app.command("sample")
def sample(
    checkpoint: Annotated[Path | None, typer.Option(help="Checkpoint directory (rz8.1 layout)")] = None,
    compare: Annotated[
        list[Path] | None,
        typer.Option("--compare", help="Additional checkpoint dir(s) for side-by-side comparison (repeatable)"),
    ] = None,
    prompt: Annotated[
        str,
        typer.Option(help="Prompt text. Diagnostic-task formats work best, e.g. 'TASK arith CMP 1.00e-02 2.00e+03 OUT'"),
    ] = "",
    max_tokens: Annotated[int, typer.Option(help="Max new tokens")] = 64,
    temperature: Annotated[float, typer.Option(help="0 = greedy (deterministic); >0 samples")] = 0.0,
    top_k: Annotated[int | None, typer.Option(help="Top-k filtering (sampled mode)")] = None,
    seed: Annotated[int, typer.Option(help="Sampling seed (greedy mode ignores it)")] = 0,
    step: Annotated[int | None, typer.Option(help="Checkpoint step (default: latest in the directory)")] = None,
    device_str: Annotated[str, typer.Option("--device", help="cpu | cuda")] = "cpu",
    stop_at_separator: Annotated[
        bool,
        typer.Option(
            "--stop-at-separator/--no-stop-at-separator",
            help="Stop at the <|endoftext|> document separator (the trained format's answer terminator)",
        ),
    ] = True,
    json_out: Annotated[bool, typer.Option("--json", help="Machine-readable output (no live display)")] = False,
) -> None:
    """Generate text from trained checkpoints (bead rz8.5): streaming display,
    tokens/s + peak memory, and side-by-side mechanism comparison via --compare
    (same prompt, same seed, same sampling params).

    EXPECTATION SETTING: research-scale checkpoints (2-15M params) produce
    barely-coherent text. The point is (a) end-to-end decode-path validation
    in anger, (b) RELATIVE comparisons between mechanisms at matched budgets -
    never absolute prose quality.
    """
    import torch

    from nanochat.tokenizer import get_tokenizer

    ckpts: list[Path] = ([checkpoint] if checkpoint else []) + list(compare or [])
    if not ckpts:
        console.print("[bold red]Provide --checkpoint DIR (and optionally --compare DIR, repeatable).[/bold red]")
        raise typer.Exit(code=2)
    if not prompt:
        console.print("[bold red]--prompt is required.[/bold red]")
        raise typer.Exit(code=2)

    device = torch.device(device_str)
    tok = get_tokenizer()
    prompt_ids = tok.encode(prompt)
    results: list[dict[str, Any]] = []

    for ckpt_dir in ckpts:
        model, ckpt_meta, resolved_step = _load_eval_checkpoint(ckpt_dir, step, device)
        attn = str((ckpt_meta.get("model_config") or {}).get("attention_type", "?"))
        n_params = sum(p.numel() for p in model.parameters())
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        title = f"{attn} · {ckpt_dir} @ step {resolved_step}"
        if json_out:
            rec = _sample_stream(
                model, tok, prompt_ids, max_tokens=max_tokens, temperature=temperature,
                top_k=top_k, seed=seed, stop_at_separator=stop_at_separator,
            )
        else:
            from rich.live import Live

            shown = {"text": ""}

            def render() -> Panel:
                return Panel(shown["text"] or "[dim]…[/dim]", title=title, border_style="cyan")

            with Live(render(), console=console, refresh_per_second=12) as live:

                def on_token(text_so_far: str) -> None:
                    shown["text"] = text_so_far
                    live.update(render())

                rec = _sample_stream(
                    model, tok, prompt_ids, max_tokens=max_tokens, temperature=temperature,
                    top_k=top_k, seed=seed, stop_at_separator=stop_at_separator, on_token=on_token,
                )
        rec.update(
            {
                "checkpoint": str(ckpt_dir),
                "step": resolved_step,
                "attention_type": attn,
                "n_params": n_params,
                "peak_memory": _sample_peak_memory(device),
            }
        )
        results.append(rec)

    if json_out:
        payload = {
            "prompt": prompt,
            "seed": seed,
            "temperature": temperature,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "device": str(device),
            "stop_at_separator": stop_at_separator,
            "results": [
                {k: v for k, v in r.items() if k != "tokens"} | {"n_tokens": len(r["tokens"])} for r in results
            ],
        }
        console.print_json(json.dumps(payload))
        return

    table = Table(title="mgr sample — generation stats", box=box.SIMPLE_HEAVY)
    table.add_column("mechanism", style="bold")
    table.add_column("checkpoint")
    table.add_column("params", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("tokens/s", justify="right")
    table.add_column("peak memory", justify="right")
    for r in results:
        table.add_row(
            r["attention_type"], f"{r['checkpoint']} @ {r['step']}", f"{r['n_params']:,}",
            str(len(r["tokens"])), f"{r['tokens_per_s']:.1f}", r["peak_memory"],
        )
    console.print(table)
    if len(results) > 1:
        from rich.columns import Columns

        panels = [
            Panel(
                r["text"] or "[dim](no tokens before the separator)[/dim]",
                title=f"{r['attention_type']} @ step {r['step']}",
                border_style="cyan",
                width=46,
            )
            for r in results
        ]
        console.print(Columns(panels, equal=True))


@app.command("profile-data")
def profile_data(
    data: Annotated[Path | None, typer.Option(help="Directory (or file) of .txt/.md/.parquet documents")] = None,
    task: Annotated[str | None, typer.Option(help="Profile a generated diagnostic task (requires vdc.1 gen-tasks)")] = None,
    mode: Annotated[str, typer.Option(help="Representation mode: tokens | activations")] = "tokens",
    sample: Annotated[int, typer.Option(help="Max documents to sample")] = 256,
    points: Annotated[int, typer.Option(help="Points in the distance matrix (cost is O(points^2))")] = 96,
    doc_tokens: Annotated[int, typer.Option(help="Tokens kept per document")] = 256,
    seed: Annotated[int, typer.Option(help="Seed for sampling/bootstrap (determinism)")] = 42,
    out: Annotated[Path | None, typer.Option(help="Write profile.json to this path")] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Emit the profile JSON to stdout")] = False,
) -> None:
    """Profile a corpus's data geometry (mgr profile-data --data DIR | --task NAME[:dial=v,...])."""
    import geometry_profile as gp

    if task is not None and data is not None:
        console.print("[bold red]Provide --data or --task, not both.[/bold red]")
        raise typer.Exit(code=2)
    if task is None and data is None:
        console.print(
            "[bold red]Provide --data DIR (documents) or --task NAME[:dial=v,...] "
            "(generated diagnostic task; see mgr gen-tasks --list).[/bold red]"
        )
        raise typer.Exit(code=2)

    if task is not None:
        from nanochat.diagnostics_data import TASKS, generate_texts

        task_name, _, dial_str = task.partition(":")
        if task_name not in TASKS:
            console.print(f"[bold red]Unknown task {task_name!r}; available: {', '.join(sorted(TASKS))}[/bold red]")
            raise typer.Exit(code=2)
        dial_overrides: dict[str, float] = {}
        if dial_str:
            for pair in dial_str.split(","):
                key, _, value = pair.partition("=")
                try:
                    dial_overrides[key.strip()] = float(value)
                except ValueError:
                    console.print(f"[bold red]--task dial must be name=number, got {pair!r}[/bold red]")
                    raise typer.Exit(code=2) from None
        try:
            texts = generate_texts(task_name, size=sample, seed=seed, dial_overrides=dial_overrides)
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            raise typer.Exit(code=2) from exc
        corpus_label = f"task:{task}"
    else:
        import numpy as _np

        assert data is not None  # guarded above: one of --data/--task is required
        rng = _np.random.default_rng(seed)
        try:
            texts = gp.load_corpus_texts(data, max_docs=sample, rng=rng)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            raise typer.Exit(code=2) from exc
        corpus_label = str(data)

    cfg = gp.ProfileConfig(
        mode=mode, sample_docs=sample, n_points=points, doc_tokens=doc_tokens, seed=seed, corpus_label=corpus_label
    )
    t0 = time.perf_counter()
    with console.status(f"[bold cyan]profiling {len(texts)} docs ({mode} mode, {points} points)…[/bold cyan]"):
        profile = gp.profile_from_texts(texts, cfg)
    elapsed = time.perf_counter() - t0

    schema_errors = gp.validate_profile_schema(profile)
    if schema_errors:  # defensive: should be impossible for our own output
        for e in schema_errors:
            console.print(f"[bold red]profile schema error: {e}[/bold red]")
        raise typer.Exit(code=1)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(gp.profile_to_json(profile), encoding="utf-8")
    if json_out:
        print(gp.profile_to_json(profile))
    else:
        est = profile["estimators"]
        table = Table(title=f"Data-geometry profile — {corpus_label} ({mode} mode)", box=box.SIMPLE_HEAVY)
        table.add_column("estimator", style="bold")
        table.add_column("value", justify="right")
        table.add_column("detail")
        dh = est["delta_hyperbolicity"]
        table.add_row(
            "delta-hyperbolicity",
            f"{dh['mean']:.4f}",
            f"CI95 [{dh['ci95'][0]:.4f}, {dh['ci95'][1]:.4f}] · 0=tree-like, higher=flat",
        )
        um = est["ultrametricity"]
        table.add_row(
            "ultrametricity violation",
            f"{um['violation_mean']:.4f}",
            f"violating frac: {um['violation_fraction']:.2f} · cophenetic r={um['cophenetic_correlation']:.3f}",
        )
        dr = est["dynamic_range"]
        table.add_row(
            "dynamic range",
            f"{dr['mean_decades']:.2f} dec",
            f"{dr['numbers_found']} numerals · Hill tail={dr['hill_tail_exponent']:.2f}",
        )
        osens = est["order_sensitivity"]
        table.add_row(
            "order sensitivity",
            f"{osens['relative_delta']:.4f}",
            f"bigram NLL {osens['nll_original']:.3f} -> {osens['nll_transposed']:.3f} under transposition",
        )
        hd = est["hierarchy_depth"]
        table.add_row(
            "hierarchy depth",
            f"{hd['normalized_mean_depth']:.3f}",
            f"mean {hd['mean_depth']:.1f} / max {hd['max_depth']:.0f} merges · ~1=balanced hierarchy, >>1=flat/chained",
        )
        console.print(table)
        for w in profile["warnings"]:
            console.print(Panel(f"[yellow]{w}[/yellow]", border_style="yellow"))
        console.print(
            Panel(
                f"[dim]{profile['interpretation_note']}[/dim]\n"
                f"docs={profile['sample']['docs']} points={profile['sample']['points']} "
                f"seed={seed} elapsed={elapsed:.1f}s"
                + (f" · written to {out}" if out else ""),
                border_style="blue",
            )
        )


# ---------------------------------------------------------------------------
# mgr theorems — the THEOREM REGISTRY (bead model_guided_research-vnl.1)
#
# The mathematical twin of the (future) hypothesis registry (hij.1): every
# load-bearing mathematical claim in the project lives in
# hypotheses/theorems.yaml with a stable id, precise statement, proof status
# (conjecture | proved-on-paper | lean-checked), source-note anchor, numerical
# check pointers (certify check names / pytest node ids / demo checks), and
# its consumers. `validate` enforces the schema and pointer integrity; the
# fast tier runs always, `--deep` additionally resolves pytest refs against a
# live `pytest --collect-only`.
#
# SERIALIZATION DECISION (binding for hij.1, which lands second and inherits):
# YAML via PyYAML safe_load — registry entries carry multi-sentence prose
# statements, and YAML block scalars keep those hand-editable in review.
# ---------------------------------------------------------------------------

_THEOREM_STATUSES: tuple[str, ...] = ("conjecture", "proved-on-paper", "lean-checked")
_THEOREM_CHECK_KINDS: tuple[str, ...] = ("certify", "pytest", "demo")
_THEOREM_ID_RE = r"^(thm|conj)-[a-z0-9][a-z0-9-]*$"

# Canonical literal check names from _run_certify_checks (mechanism.check).
# Kept in sync with the certify implementation by a source-scan test in
# tests/test_theorem_registry.py — if you add an add_check call, add it here.
_CERTIFY_NAMED_CHECKS: frozenset[str] = frozenset(
    {
        "braid.heuristic_mass_partition_violated",
        "braid.payload_multiset_invariance",
        "braid.restricted_law_violates_ybe",
        "braid.rmatrix_braid_relation_holds",
        "braid.rmatrix_inversion_relation_holds",
        "braid.rmatrix_mass_partition_charge_conserved",
        "braid.rmatrix_perturbed_transfer_separates",
        "braid.rmatrix_transfer_matrices_commute",
        "braid.ybe_law_holds",
        "fractal.router_branch_simplex",
        "gauge.kv_decode_matches_full_forward",
        "gauge.rotation_additivity_cumsum_law",
        "gauge.rotation_inverse_roundtrip",
        "gauge.rotation_pairwise_norm_preservation",
        "octonion.o_times_conj_is_norm_squared",
        "octonion.omul_alternativity",
        "octonion.omul_nonassociativity_witness",
        "octonion.omul_norm_multiplicative",
        "octonion.reduces_to_quaternion_on_subalgebra",
        "quaternion.qconj_antihomomorphism",
        "quaternion.qmul_associativity",
        "quaternion.qmul_norm_multiplicative",
        "quaternion.rotor_norm_preservation",
        "reversible.custom_autograd_grad_parity",
        "reversible.energy_drift_separation",
        "reversible.forward_inverse_roundtrip",
        "reversible.symplectic_jacobian",
        "simplicial.mass_conservation_two_hop",
        "standard.causal_mask_structure",
        "standard.rmsnorm_unit_rms",
        "standard.rope_pairwise_norm_preservation",
        "standard.softmax_row_stochastic",
        "surreal.layer_linearity",
        "surreal.row_norm_equals_exp_scale",
        "surreal.scale_shift_equivariance",
        "tropical.ffn_collapse_single_layer",
        "tropical.ffn_lipschitz_1_sup_norm",
        "tropical.lipschitz_1_sup_norm_q",
        "tropical.lipschitz_1_sup_norm_v",
        "tropical.margin_matches_bruteforce",
        "tropical.maslov_endpoint_within_sandwich",
        "tropical.score_center_pure_gauge_shift",
        "ultrametric.strong_triangle_inequality_lcp",
        "ultrametric.trie_decode_matches_hard_kernel",
    }
)


def _certify_known_check_names() -> frozenset[str]:
    """All resolvable certify check names: literals + per-mechanism causality."""
    causality = {f"{m}.causality_no_future_grad" for m in _CERTIFY_MECHANISMS}
    return _CERTIFY_NAMED_CHECKS | causality


def _theorems_registry_path() -> Path:
    return Path(__file__).resolve().parent / "hypotheses" / "theorems.yaml"


def _load_theorem_registry(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Load the registry YAML. Returns (data, load_errors)."""
    if not path.exists():
        return None, [f"registry file not found: {path}"]
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        return None, [f"YAML parse error: {exc}"]
    if not isinstance(data, dict):
        return None, ["registry root must be a mapping with schema_version + theorems"]
    return data, []


def _anchor_in_note(note_path: Path, anchor: str) -> bool:
    """anchor is a case-insensitive substring of some markdown heading line."""
    try:
        text = note_path.read_text(encoding="utf-8")
    except OSError:
        return False
    needle = anchor.lower()
    return any(line.lstrip().startswith("#") and needle in line.lower() for line in text.splitlines())


def _pytest_ref_resolves(ref: str, collected: list[str]) -> bool:
    """A ref resolves if it names a collected node id or a prefix of one
    (class refs and parametrized tests resolve via their children)."""
    return any(cid == ref or cid.startswith(ref + "::") or cid.startswith(ref + "[") for cid in collected)


def _collect_pytest_node_ids(repo_root: Path, scope_files: list[str]) -> tuple[list[str], str | None]:
    """Run pytest --collect-only -q over the referenced files; return node ids."""
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *scope_files]
    try:
        result = subprocess.run(  # nosec B603 - fixed argv, no shell
            cmd, cwd=repo_root, capture_output=True, text=True, timeout=180
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"pytest collection failed to run: {exc}"
    ids = [line.strip() for line in result.stdout.splitlines() if "::" in line]
    if not ids:
        tail = (result.stdout + result.stderr).strip().splitlines()[-3:]
        return [], f"pytest collected no node ids (exit {result.returncode}): {' | '.join(tail)}"
    return ids, None


def _validate_theorem_registry(
    data: dict[str, Any] | None,
    load_errors: list[str],
    repo_root: Path,
    *,
    collected_pytest_ids: list[str] | None = None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Validate the registry. Returns (errors, warnings, summary).

    Static tier (always): schema shape, id uniqueness/format, status enum,
    source-note path+anchor existence (path may be the literal "pending"),
    certify pointer resolution against the canonical check-name list,
    pytest pointer FORMAT, depends_on referential integrity, lean-checked
    proof-file existence. Deep tier (collected_pytest_ids provided): pytest
    pointer resolution against the live collection.
    """
    import re as _re

    errors: list[str] = list(load_errors)
    warnings: list[str] = []
    summary: dict[str, Any] = {"entries": 0, "by_status": {}, "pending_notes": 0, "checks": {"certify": 0, "pytest": 0, "demo": 0}}
    if data is None:
        return errors, warnings, summary

    if not isinstance(data.get("schema_version"), int):
        errors.append("schema_version must be an integer")
    theorems = data.get("theorems")
    if not isinstance(theorems, list) or not theorems:
        errors.append("theorems must be a non-empty list")
        return errors, warnings, summary

    known_checks = _certify_known_check_names()
    id_re = _re.compile(_THEOREM_ID_RE)
    seen_ids: set[str] = set()
    all_ids = {t.get("id") for t in theorems if isinstance(t, dict)}
    summary["entries"] = len(theorems)

    for idx, t in enumerate(theorems):
        where = f"theorems[{idx}]"
        if not isinstance(t, dict):
            errors.append(f"{where}: entry must be a mapping")
            continue
        tid = t.get("id")
        where = f"{tid or where}"
        if not isinstance(tid, str) or not id_re.match(tid):
            errors.append(f"{where}: id missing or not matching {_THEOREM_ID_RE}")
        elif tid in seen_ids:
            errors.append(f"{where}: duplicate id")
        else:
            seen_ids.add(tid)

        status = t.get("status")
        if status not in _THEOREM_STATUSES:
            errors.append(f"{where}: status {status!r} not in {_THEOREM_STATUSES}")
        else:
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
            if isinstance(tid, str):
                if tid.startswith("conj-") and status != "conjecture":
                    warnings.append(f"{where}: id has conj- prefix but status is {status} (rename on promotion)")
                if tid.startswith("thm-") and status == "conjecture":
                    warnings.append(f"{where}: id has thm- prefix but status is conjecture")

        if not isinstance(t.get("statement"), str) or not t["statement"].strip():
            errors.append(f"{where}: statement must be a non-empty string")
        mechanisms = t.get("mechanisms")
        if not isinstance(mechanisms, list) or not all(isinstance(m, str) for m in mechanisms):
            errors.append(f"{where}: mechanisms must be a list of strings")
        if not isinstance(t.get("proof_location"), str) or not t["proof_location"].strip():
            errors.append(f"{where}: proof_location must be a non-empty string")

        note = t.get("source_note")
        if not isinstance(note, dict) or "path" not in note or "anchor" not in note:
            errors.append(f"{where}: source_note must be a mapping with path and anchor keys")
        else:
            npath, anchor = note["path"], note["anchor"]
            if npath == "pending":
                summary["pending_notes"] += 1
            elif not isinstance(npath, str):
                errors.append(f"{where}: source_note.path must be a string or 'pending'")
            else:
                note_file = repo_root / npath
                if not note_file.exists():
                    errors.append(f"{where}: source_note.path does not exist: {npath}")
                elif anchor is not None:
                    if not isinstance(anchor, str) or not anchor.strip():
                        errors.append(f"{where}: source_note.anchor must be null or a non-empty string")
                    elif not _anchor_in_note(note_file, anchor):
                        errors.append(f"{where}: anchor {anchor!r} not found in any heading of {npath}")

        if status == "lean-checked":
            ploc = t.get("proof_location", "")
            lean_tokens = [tok for tok in str(ploc).replace(",", " ").split() if "proofs/" in tok]
            if not lean_tokens:
                errors.append(f"{where}: lean-checked requires proof_location naming a proofs/ file")
            else:
                lean_file = lean_tokens[0].split("::")[0].rstrip(";:")
                if not (repo_root / lean_file).exists():
                    errors.append(f"{where}: lean-checked proof file does not exist: {lean_file}")

        checks = t.get("numerical_checks")
        if not isinstance(checks, list):
            errors.append(f"{where}: numerical_checks must be a list (may be empty)")
            checks = []
        for c in checks:
            if not isinstance(c, dict) or "kind" not in c or "ref" not in c:
                errors.append(f"{where}: each numerical_check needs kind and ref")
                continue
            kind, ref = c["kind"], c["ref"]
            if kind not in _THEOREM_CHECK_KINDS:
                errors.append(f"{where}: check kind {kind!r} not in {_THEOREM_CHECK_KINDS}")
                continue
            summary["checks"][kind] = summary["checks"].get(kind, 0) + 1
            if kind == "certify":
                if ref not in known_checks:
                    errors.append(f"{where}: certify check {ref!r} is not a known check name")
            elif kind == "pytest":
                if not (isinstance(ref, str) and ref.startswith("tests/")):
                    errors.append(f"{where}: pytest ref must start with tests/: {ref!r}")
                elif collected_pytest_ids is not None and not _pytest_ref_resolves(ref, collected_pytest_ids):
                    errors.append(f"{where}: pytest ref does not resolve against live collection: {ref}")

        depends_on = t.get("depends_on")
        if not isinstance(depends_on, list):
            # without this check a malformed string value would iterate per
            # character and emit one confusing error per letter
            errors.append(f"{where}: depends_on must be a list (may be empty)")
        else:
            for dep in depends_on:
                if dep not in all_ids:
                    errors.append(f"{where}: depends_on references unknown theorem id {dep!r}")
        used_by = t.get("used_by")
        if not isinstance(used_by, list):
            errors.append(f"{where}: used_by must be a list (may be empty)")

    return errors, warnings, summary


_STATUS_STYLE = {"conjecture": "yellow", "proved-on-paper": "cyan", "lean-checked": "green"}

theorems_app = typer.Typer(
    help="Theorem registry — the project's mathematical claims with proof status and check pointers.",
    rich_markup_mode="rich",
)
app.add_typer(theorems_app, name="theorems")


def _theorems_load_or_exit() -> dict[str, Any]:
    data, load_errors = _load_theorem_registry(_theorems_registry_path())
    if data is None:
        for e in load_errors:
            console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)
    return data


@theorems_app.command("list")
def theorems_list(
    status: Annotated[str | None, typer.Option(help="Filter by status: conjecture | proved-on-paper | lean-checked")] = None,
    mechanism: Annotated[str | None, typer.Option(help="Filter by mechanism (e.g. tropical)")] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON instead of a table")] = False,
) -> None:
    """List registered theorems with status, mechanisms, and check counts."""
    data = _theorems_load_or_exit()
    rows = data.get("theorems", [])
    if status:
        rows = [t for t in rows if t.get("status") == status]
    if mechanism:
        rows = [t for t in rows if mechanism in (t.get("mechanisms") or [])]
    if json_out:
        console.print_json(json.dumps({"schema_version": data.get("schema_version"), "theorems": rows}))
        return
    table = Table(title=f"Theorem registry — {len(rows)} entr{'y' if len(rows) == 1 else 'ies'}", box=box.SIMPLE_HEAVY)
    table.add_column("id", style="bold")
    table.add_column("status")
    table.add_column("mechanisms")
    table.add_column("checks", justify="right")
    table.add_column("source note")
    for t in rows:
        st = t.get("status", "?")
        note = t.get("source_note") or {}
        npath = note.get("path", "?")
        src = "[dim]pending[/dim]" if npath == "pending" else Path(str(npath)).name
        table.add_row(
            str(t.get("id")),
            f"[{_STATUS_STYLE.get(st, 'white')}]{st}[/{_STATUS_STYLE.get(st, 'white')}]",
            ", ".join(t.get("mechanisms") or []),
            str(len(t.get("numerical_checks") or [])),
            src,
        )
    console.print(table)
    by_status: dict[str, int] = {}
    for t in rows:
        by_status[t.get("status", "?")] = by_status.get(t.get("status", "?"), 0) + 1
    console.print(
        Panel(
            " · ".join(f"[{_STATUS_STYLE.get(k, 'white')}]{k}: {v}[/{_STATUS_STYLE.get(k, 'white')}]" for k, v in sorted(by_status.items()))
            or "no entries",
            border_style="blue",
        )
    )


@theorems_app.command("show")
def theorems_show(
    theorem_id: Annotated[str, typer.Argument(help="Theorem id, e.g. thm-route-stability")],
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON")] = False,
) -> None:
    """Show one theorem entry in full."""
    data = _theorems_load_or_exit()
    match = next((t for t in data.get("theorems", []) if t.get("id") == theorem_id), None)
    if match is None:
        console.print(f"[bold red]No theorem with id {theorem_id!r}.[/bold red] Try: mgr theorems list")
        raise typer.Exit(code=1)
    if json_out:
        console.print_json(json.dumps(match))
        return
    st = match.get("status", "?")
    note = match.get("source_note") or {}
    checks = match.get("numerical_checks") or []
    lines = [
        f"[bold]{match.get('statement', '').strip()}[/bold]",
        "",
        f"status: [{_STATUS_STYLE.get(st, 'white')}]{st}[/{_STATUS_STYLE.get(st, 'white')}]",
        f"mechanisms: {', '.join(match.get('mechanisms') or []) or '—'}",
        f"source note: {note.get('path')}" + (f"  (anchor: {note.get('anchor')})" if note.get("anchor") else ""),
        f"proof: {match.get('proof_location')}",
        "checks: " + (", ".join(f"{c.get('kind')}:{c.get('ref')}" for c in checks) or "—"),
        f"used by: {', '.join(match.get('used_by') or []) or '—'}",
        f"depends on: {', '.join(match.get('depends_on') or []) or '—'}",
    ]
    console.print(Panel("\n".join(lines), title=f"[bold]{theorem_id}[/bold]", border_style=_STATUS_STYLE.get(st, "white")))


@theorems_app.command("validate")
def theorems_validate(
    deep: Annotated[bool, typer.Option("--deep", help="Also resolve pytest refs against a live pytest --collect-only")] = False,
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON report")] = False,
) -> None:
    """Validate the registry: schema, anchors, pointer integrity. Exit 1 on errors."""
    repo_root = Path(__file__).resolve().parent
    data, load_errors = _load_theorem_registry(_theorems_registry_path())
    collected: list[str] | None = None
    collect_problem: str | None = None
    if deep and data is not None:
        ref_files = sorted(
            {
                str(c["ref"]).split("::")[0]
                for t in data.get("theorems", [])
                if isinstance(t, dict)
                for c in (t.get("numerical_checks") or [])
                if isinstance(c, dict) and c.get("kind") == "pytest" and isinstance(c.get("ref"), str)
            }
        )
        if ref_files:
            collected, collect_problem = _collect_pytest_node_ids(repo_root, ref_files)
            if collect_problem:
                collected = None
    errors, warnings, summary = _validate_theorem_registry(data, load_errors, repo_root, collected_pytest_ids=collected)
    if collect_problem:
        errors.append(f"--deep: {collect_problem}")
    if json_out:
        console.print_json(json.dumps({"errors": errors, "warnings": warnings, "summary": summary, "deep": deep}))
    else:
        if errors:
            etable = Table(title=f"[red]{len(errors)} error(s)[/red]", box=box.SIMPLE)
            etable.add_column("error", style="red")
            for e in errors:
                etable.add_row(e)
            console.print(etable)
        if warnings:
            wtable = Table(title=f"[yellow]{len(warnings)} warning(s)[/yellow]", box=box.SIMPLE)
            wtable.add_column("warning", style="yellow")
            for w in warnings:
                wtable.add_row(w)
            console.print(wtable)
        color = "red" if errors else ("yellow" if warnings else "green")
        if not deep:
            tier = "static"
        elif collected is not None:
            tier = "deep (pytest collection resolved)"
        elif collect_problem:
            tier = "deep (pytest collection FAILED)"
        else:
            tier = "deep (no pytest refs to resolve)"
        console.print(
            Panel(
                f"[bold {color}]{summary['entries']} entries · "
                f"{summary['by_status']} · {summary['pending_notes']} pending notes · "
                f"checks {summary['checks']} · tier: {tier} · "
                f"{len(errors)} errors / {len(warnings)} warnings[/bold {color}]",
                title="[bold]mgr theorems validate[/bold]",
                border_style=color,
            )
        )
    if errors:
        raise typer.Exit(code=1)


# =============================================================================
# Hypothesis registry (bead hij.1) — empirical twin of the theorem registry.
# Same serialization (YAML/PyYAML, decided by vnl.1), same loader shape.
# =============================================================================

_HYP_ID_RE = r"^hyp-[a-z0-9][a-z0-9-]*$"
_HYP_STATUSES = ("open", "supported", "refuted", "inconclusive", "blocked")
_HYP_VERDICTS = ("supported", "refuted", "inconclusive")
_HYP_SOURCE_KINDS = ("human", "model")
_HYP_MECHANISMS = frozenset(
    {
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
        "hoss",
        "ordinal",
    }
)
_HYP_METRIC_SCHEMAS = ("evaltasks", "train", "certify", "bench", "chargeprobe")
_HYP_COMPARATORS = (">=", "<=")
_HYP_THRESHOLD_KINDS = ("absolute_delta", "ratio")


def _hypotheses_registry_path() -> Path:
    return Path(__file__).resolve().parent / "hypotheses" / "registry.yaml"


def _load_hypothesis_registry(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Same loader contract as _load_theorem_registry (shared format decision)."""
    if not path.exists():
        return None, [f"registry file not found: {path}"]
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        return None, [f"YAML parse error: {exc}"]
    if not isinstance(data, dict):
        return None, ["registry root must be a mapping with schema_version + hypotheses"]
    return data, []


def _load_parent_hypothesis_registry(repo_root: Path) -> dict[str, Any] | None:
    """The committed (HEAD) version of the registry, for append-only checks.
    Returns None when unavailable (file new in this commit, or not a git repo)."""
    try:
        result = subprocess.run(  # nosec B603 - fixed argv, no shell
            ["git", "show", "HEAD:hypotheses/registry.yaml"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        parent = yaml.safe_load(result.stdout)
    except yaml.YAMLError:
        return None
    return parent if isinstance(parent, dict) else None


def _validate_hypothesis_prediction(pred: Any, where: str, errors: list[str], warnings: list[str]) -> None:
    if not isinstance(pred, dict):
        errors.append(f"{where}: prediction must be a mapping or null")
        return
    metric_path = pred.get("metric_path")
    if not isinstance(metric_path, str) or ":" not in metric_path:
        errors.append(f"{where}: prediction.metric_path must be '<schema>:<dotted.path>'")
    else:
        schema, _, dotted = metric_path.partition(":")
        if schema not in _HYP_METRIC_SCHEMAS:
            errors.append(f"{where}: metric_path schema {schema!r} not in {sorted(_HYP_METRIC_SCHEMAS)}")
        if not dotted or not all(seg.isidentifier() or seg.isdigit() for seg in dotted.split(".")):
            errors.append(f"{where}: metric_path dotted path {dotted!r} is malformed")
        elif schema == "evaltasks":
            segs = dotted.split(".")
            if len(segs) < 2 or segs[0] != "tasks":
                errors.append(f"{where}: evaltasks metric_path must start with 'tasks.<task>.'")
            else:
                from nanochat.diagnostics_data import TASKS as _diag_tasks

                if segs[1] not in _diag_tasks:
                    errors.append(
                        f"{where}: evaltasks metric_path names unknown task {segs[1]!r} "
                        f"(known: {', '.join(sorted(_diag_tasks))})"
                    )
        elif schema == "certify":
            segs = dotted.split(".")
            if len(segs) != 3 or segs[2] != "measured":
                errors.append(f"{where}: certify metric_path must be '<mechanism>.<check>.measured'")
            elif f"{segs[0]}.{segs[1]}" not in _certify_known_check_names():
                errors.append(f"{where}: certify metric_path names unknown check {segs[0]}.{segs[1]!r}")
        elif schema == "chargeprobe":
            if not (dotted.startswith("categories.") or dotted.startswith("dissociation.")):
                errors.append(
                    f"{where}: chargeprobe metric_path must start with 'categories.' or 'dissociation.'"
                )
    if pred.get("comparator") not in _HYP_COMPARATORS:
        errors.append(f"{where}: prediction.comparator must be one of {_HYP_COMPARATORS}")
    if pred.get("threshold_kind") not in _HYP_THRESHOLD_KINDS:
        errors.append(f"{where}: prediction.threshold_kind must be one of {_HYP_THRESHOLD_KINDS}")
    threshold = pred.get("threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        errors.append(f"{where}: prediction.threshold must be a number")
    elif pred.get("threshold_kind") == "ratio" and threshold <= 0:
        errors.append(f"{where}: ratio threshold must be > 0, got {threshold}")
    def _check_variant(variant: Any, label: str) -> None:
        # rgyl: a variant selector restricts an arm WITHIN a mechanism by
        # recorded hparams/config values; scalars only (matched by equality)
        if variant is None:
            return
        if not isinstance(variant, dict) or not variant:
            errors.append(f"{where}: {label} must be a non-empty mapping of recorded knob -> value")
            return
        for k, v in variant.items():
            if not isinstance(k, str) or not k.isidentifier():
                errors.append(f"{where}: {label} key {k!r} must be an identifier-like string")
            if isinstance(v, (dict, list)):
                errors.append(f"{where}: {label}[{k!r}] must be a scalar (matched by equality), got {type(v).__name__}")

    baseline = pred.get("baseline")
    if baseline is None and "baseline" in pred:
        # ci-v3 single-arm prediction: an explicit `baseline: null` makes the
        # claim a threshold on the candidate arm alone - absolute thresholds
        # only (a ratio without a denominator arm is meaningless).
        if pred.get("threshold_kind") == "ratio":
            errors.append(f"{where}: single-arm predictions (baseline: null) require threshold_kind absolute_delta")
    elif not isinstance(baseline, dict):
        errors.append(f"{where}: prediction.baseline must be a mapping, or explicitly null for single-arm claims")
    else:
        if baseline.get("mechanism") not in _HYP_MECHANISMS:
            errors.append(f"{where}: baseline.mechanism {baseline.get('mechanism')!r} unknown")
        if baseline.get("equal_flops") is not True:
            errors.append(
                f"{where}: baseline.equal_flops must be true - equal-budget comparison is "
                "the registry's fairness invariant"
            )
        _check_variant(baseline.get("variant"), "baseline.variant")
    _check_variant(pred.get("candidate_variant"), "prediction.candidate_variant")
    min_seeds = pred.get("min_seeds")
    if not isinstance(min_seeds, int) or isinstance(min_seeds, bool) or min_seeds < 1:
        errors.append(f"{where}: prediction.min_seeds must be an integer >= 1")
    elif min_seeds < 3:
        warnings.append(f"{where}: min_seeds={min_seeds} is below the house convention of 3")
    validity = pred.get("validity")
    if validity is not None:
        if not isinstance(validity, dict):
            errors.append(f"{where}: prediction.validity must be a mapping or omitted")
        else:
            unknown_keys = set(validity) - {"baseline_floor", "floor_margin", "floor_source"}
            if unknown_keys:
                errors.append(f"{where}: prediction.validity has unknown key(s) {sorted(unknown_keys)}")
            floor = validity.get("baseline_floor")
            if not isinstance(floor, (int, float)) or isinstance(floor, bool):
                errors.append(f"{where}: validity.baseline_floor must be a number")
            margin = validity.get("floor_margin", 0.0)
            if not isinstance(margin, (int, float)) or isinstance(margin, bool) or margin < 0:
                errors.append(f"{where}: validity.floor_margin must be a number >= 0")
            floor_source = validity.get("floor_source")
            if not isinstance(floor_source, str) or not floor_source.strip():
                errors.append(f"{where}: validity.floor_source must document how the floor was computed")
    elif isinstance(metric_path, str) and ".exact_match." in metric_path:
        warnings.append(
            f"{where}: exact-match prediction without validity.baseline_floor - the ci-v2 "
            "floor gate will rely on artifact-recorded answer priors only"
        )


def _validate_hypothesis_registry(
    data: dict[str, Any] | None,
    load_errors: list[str],
    repo_root: Path,
    *,
    parent: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Validate the registry. Returns (errors, warnings, summary).

    Checks: schema shape, id format/uniqueness, status/source enums, date
    format, mechanism vocabulary, prediction contract (metric_path schema +
    evaltasks task names, comparator/threshold/baseline/min_seeds),
    prediction-null entries carry an operationalization_note, theorem_refs
    resolve against hypotheses/theorems.yaml, verdict_history entry shape,
    and - when the committed parent version is available - APPEND-ONLY
    history (no entry deletions, no verdict_history rewrites, no
    date_registered edits).
    """
    import re as _re

    errors: list[str] = list(load_errors)
    warnings: list[str] = []
    summary: dict[str, Any] = {"entries": 0, "by_status": {}, "operationalized": 0, "needs_operationalization": 0}
    if data is None:
        return errors, warnings, summary

    if not isinstance(data.get("schema_version"), int):
        errors.append("schema_version must be an integer")
    entries = data.get("hypotheses")
    if not isinstance(entries, list) or not entries:
        errors.append("hypotheses must be a non-empty list")
        return errors, warnings, summary
    summary["entries"] = len(entries)

    theorem_ids: set[str] = set()
    th_data, _th_errors = _load_theorem_registry(_theorems_registry_path())
    if th_data is not None:
        theorem_ids = {
            str(t["id"]) for t in th_data.get("theorems", []) if isinstance(t, dict) and isinstance(t.get("id"), str)
        }

    id_re = _re.compile(_HYP_ID_RE)
    date_re = _re.compile(r"^\d{4}-\d{2}-\d{2}$")
    seen_ids: set[str] = set()

    for idx, h in enumerate(entries):
        where = f"hypotheses[{idx}]"
        if not isinstance(h, dict):
            errors.append(f"{where}: entry must be a mapping")
            continue
        hid = h.get("id")
        if not isinstance(hid, str) or not id_re.match(hid):
            errors.append(f"{where}: id {hid!r} must match {_HYP_ID_RE}")
        elif hid in seen_ids:
            errors.append(f"{where}: duplicate id {hid!r}")
        else:
            seen_ids.add(hid)
            where = hid
        if not isinstance(h.get("statement"), str) or not h["statement"].strip():
            errors.append(f"{where}: statement must be non-empty prose")
        mechanisms = h.get("mechanisms")
        if not isinstance(mechanisms, list) or not mechanisms:
            errors.append(f"{where}: mechanisms must be a non-empty list")
        else:
            unknown = [m for m in mechanisms if m not in _HYP_MECHANISMS]
            if unknown:
                errors.append(f"{where}: unknown mechanism(s) {unknown} (vocab: {sorted(_HYP_MECHANISMS)})")
        source = h.get("source")
        if not isinstance(source, dict) or source.get("kind") not in _HYP_SOURCE_KINDS:
            errors.append(f"{where}: source.kind must be one of {_HYP_SOURCE_KINDS}")
        elif not isinstance(source.get("provenance"), str) or not source["provenance"].strip():
            errors.append(f"{where}: source.provenance must trace the claim to its prose origin")
        if not isinstance(h.get("date_registered"), str) or not date_re.match(h["date_registered"]):
            errors.append(f"{where}: date_registered must be YYYY-MM-DD")
        status = h.get("status")
        if status not in _HYP_STATUSES:
            errors.append(f"{where}: status {status!r} must be one of {_HYP_STATUSES}")
        for ref in h.get("theorem_refs") or []:
            if theorem_ids and ref not in theorem_ids:
                errors.append(f"{where}: theorem_refs entry {ref!r} not found in hypotheses/theorems.yaml")

        pred = h.get("prediction", "MISSING")
        if pred == "MISSING":
            errors.append(f"{where}: prediction key is required (may be null with an operationalization_note)")
        elif pred is None:
            summary["needs_operationalization"] += 1
            note = h.get("operationalization_note")
            if not isinstance(note, str) or not note.strip():
                errors.append(
                    f"{where}: prediction is null - an operationalization_note naming what is missing is required"
                )
            if status not in ("open", "blocked"):
                errors.append(f"{where}: status {status!r} requires an operationalized prediction")
        else:
            summary["operationalized"] += 1
            _validate_hypothesis_prediction(pred, where, errors, warnings)

        if not isinstance(h.get("evidence"), list):
            errors.append(f"{where}: evidence must be a list (may be empty)")
        history = h.get("verdict_history")
        if not isinstance(history, list):
            errors.append(f"{where}: verdict_history must be a list (may be empty)")
        else:
            for j, v in enumerate(history):
                vw = f"{where}.verdict_history[{j}]"
                if not isinstance(v, dict):
                    errors.append(f"{vw}: must be a mapping")
                    continue
                if not isinstance(v.get("date"), str) or not date_re.match(v["date"]):
                    errors.append(f"{vw}: date must be YYYY-MM-DD")
                if v.get("verdict") not in _HYP_VERDICTS:
                    errors.append(f"{vw}: verdict must be one of {_HYP_VERDICTS}")
                if not isinstance(v.get("artifacts"), list) or not v["artifacts"]:
                    errors.append(f"{vw}: artifacts must be a non-empty list of artifact paths/run ids")
                if not isinstance(v.get("adjudicator"), str) or not v["adjudicator"].strip():
                    errors.append(f"{vw}: adjudicator must name a human or engine version")
            if history and status in ("open", "blocked"):
                warnings.append(f"{where}: has verdicts but status is still {status!r}")
            if (
                history
                and status in _HYP_VERDICTS
                and isinstance(history[-1], dict)
                and history[-1].get("verdict") != status
            ):
                warnings.append(
                    f"{where}: status {status!r} disagrees with the latest verdict {history[-1].get('verdict')!r}"
                )

        # manual_hold (srtf): optional human-control block. Toggling it is NOT
        # an append-only violation (the governance check below guards only
        # verdict_history/date_registered/statement) - it is the sanctioned way
        # a human freezes a hand-checked verdict against mechanical re-stamping.
        hold = h.get("manual_hold")
        if hold is not None:
            if not isinstance(hold, dict):
                errors.append(f"{where}: manual_hold must be a mapping with held/reason")
            else:
                if not isinstance(hold.get("held"), bool):
                    errors.append(f"{where}: manual_hold.held must be a boolean")
                if hold.get("held") and (not isinstance(hold.get("reason"), str) or not hold["reason"].strip()):
                    errors.append(f"{where}: manual_hold.held=true requires a non-empty reason (why the verdict is frozen)")
                hd = hold.get("date")
                if hd is not None and (not isinstance(hd, str) or not date_re.match(hd)):
                    errors.append(f"{where}: manual_hold.date must be YYYY-MM-DD")

        if status == "blocked" and pred is not None and pred != "MISSING":
            warnings.append(f"{where}: blocked entries normally carry prediction: null until unblocked")

    # ---- append-only governance against the committed parent ----
    if parent is not None and isinstance(parent.get("hypotheses"), list):
        new_by_id = {h.get("id"): h for h in entries if isinstance(h, dict)}
        for ph in parent["hypotheses"]:
            if not isinstance(ph, dict) or not isinstance(ph.get("id"), str):
                continue
            pid = ph["id"]
            nh = new_by_id.get(pid)
            if nh is None:
                errors.append(f"{pid}: entry deleted - registry entries are retired by status, never removed")
                continue
            old_hist = ph.get("verdict_history") or []
            new_hist = nh.get("verdict_history") or []
            if new_hist[: len(old_hist)] != old_hist:
                errors.append(
                    f"{pid}: verdict_history rewritten - history is APPEND-ONLY "
                    f"(parent has {len(old_hist)} entr{'y' if len(old_hist) == 1 else 'ies'}; "
                    "they must be an unchanged prefix)"
                )
            if ph.get("date_registered") != nh.get("date_registered"):
                errors.append(f"{pid}: date_registered changed - registration dates are immutable")
            if ph.get("statement") != nh.get("statement"):
                warnings.append(
                    f"{pid}: statement text changed - claims should be retired and re-registered, not morphed"
                )

    for h in entries:
        if isinstance(h, dict):
            st = h.get("status", "?")
            summary["by_status"][st] = summary["by_status"].get(st, 0) + 1
    return errors, warnings, summary


_HYP_STATUS_STYLE = {
    "open": "yellow",
    "supported": "green",
    "refuted": "red",
    "inconclusive": "magenta",
    "blocked": "dim",
}

hypotheses_app = typer.Typer(
    help="Hypothesis registry — the project's empirical claims with preregistered predictions.",
    rich_markup_mode="rich",
)
app.add_typer(hypotheses_app, name="hypotheses")


def _hypotheses_load_or_exit() -> dict[str, Any]:
    data, load_errors = _load_hypothesis_registry(_hypotheses_registry_path())
    if data is None:
        for e in load_errors:
            console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)
    return data


@hypotheses_app.command("list")
def hypotheses_list(
    status: Annotated[str | None, typer.Option(help=f"Filter by status: {' | '.join(_HYP_STATUSES)}")] = None,
    mechanism: Annotated[str | None, typer.Option(help="Filter by mechanism (e.g. tropical)")] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON instead of a table")] = False,
) -> None:
    """List registered hypotheses with status, mechanisms, and prediction state."""
    data = _hypotheses_load_or_exit()
    rows = data.get("hypotheses", [])
    if status:
        rows = [h for h in rows if h.get("status") == status]
    if mechanism:
        rows = [h for h in rows if mechanism in (h.get("mechanisms") or [])]
    if json_out:
        console.print_json(json.dumps({"schema_version": data.get("schema_version"), "hypotheses": rows}))
        return
    table = Table(
        title=f"Hypothesis registry — {len(rows)} entr{'y' if len(rows) == 1 else 'ies'}", box=box.SIMPLE_HEAVY
    )
    table.add_column("id", style="bold")
    table.add_column("status")
    table.add_column("mechanisms")
    table.add_column("prediction")
    table.add_column("source")
    for h in rows:
        st = h.get("status", "?")
        pred = h.get("prediction")
        pred_cell = pred.get("metric_path", "?") if isinstance(pred, dict) else "[dim]needs operationalization[/dim]"
        table.add_row(
            str(h.get("id")),
            f"[{_HYP_STATUS_STYLE.get(st, 'white')}]{st}[/{_HYP_STATUS_STYLE.get(st, 'white')}]",
            ", ".join(h.get("mechanisms") or []),
            pred_cell,
            (h.get("source") or {}).get("kind", "?"),
        )
    console.print(table)
    by_status: dict[str, int] = {}
    for h in rows:
        by_status[h.get("status", "?")] = by_status.get(h.get("status", "?"), 0) + 1
    console.print(
        Panel(
            " · ".join(
                f"[{_HYP_STATUS_STYLE.get(k, 'white')}]{k}: {v}[/{_HYP_STATUS_STYLE.get(k, 'white')}]"
                for k, v in sorted(by_status.items())
            )
            or "no entries",
            border_style="blue",
        )
    )


@hypotheses_app.command("show")
def hypotheses_show(
    hypothesis_id: Annotated[str, typer.Argument(help="Hypothesis id, e.g. hyp-ultrametric-hier-heldout-depth")],
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON")] = False,
) -> None:
    """Show one hypothesis entry in full."""
    data = _hypotheses_load_or_exit()
    match = next((h for h in data.get("hypotheses", []) if h.get("id") == hypothesis_id), None)
    if match is None:
        console.print(f"[bold red]No hypothesis with id {hypothesis_id!r}.[/bold red] Try: mgr hypotheses list")
        raise typer.Exit(code=1)
    if json_out:
        console.print_json(json.dumps(match))
        return
    st = match.get("status", "?")
    pred = match.get("prediction")
    lines = [
        f"[bold]{str(match.get('statement', '')).strip()}[/bold]",
        "",
        f"status: [{_HYP_STATUS_STYLE.get(st, 'white')}]{st}[/{_HYP_STATUS_STYLE.get(st, 'white')}]",
        f"mechanisms: {', '.join(match.get('mechanisms') or [])}",
        f"source: {(match.get('source') or {}).get('kind', '?')} — {(match.get('source') or {}).get('provenance', '?')}",
        f"registered: {match.get('date_registered', '?')}",
    ]
    hold = match.get("manual_hold")
    if isinstance(hold, dict) and hold.get("held"):
        lines.append(
            f"[bold red]🔒 manual hold[/bold red]: {str(hold.get('reason', '')).strip()} "
            f"(since {hold.get('date', '?')}, by {hold.get('by', '?')})"
        )
    if match.get("theorem_refs"):
        lines.append(f"theorem refs: {', '.join(match['theorem_refs'])}")
    if isinstance(pred, dict):
        lines.append("")
        lines.append("[bold]prediction[/bold]")
        lines.append(f"  metric: {pred.get('metric_path')}")
        baseline = pred.get("baseline") or {}
        lines.append(
            f"  claim: metric {pred.get('comparator')} baseline[{baseline.get('mechanism')}] "
            f"{'x' if pred.get('threshold_kind') == 'ratio' else '+'} {pred.get('threshold')} "
            f"({pred.get('threshold_kind')}, equal FLOPs, >= {pred.get('min_seeds')} seeds)"
        )
        validity = pred.get("validity")
        if isinstance(validity, dict):
            lines.append(
                f"  validity: baseline must clear floor {validity.get('baseline_floor')} "
                f"(+{validity.get('floor_margin', 0.0)}) for a refutation to stand "
                f"— {validity.get('floor_source', '?')}"
            )
        if pred.get("scale_caveats"):
            lines.append(f"  caveats: {pred['scale_caveats']}")
    else:
        lines.append("")
        lines.append(f"[dim]needs operationalization: {match.get('operationalization_note', '(no note)')}[/dim]")
    if match.get("verdict_history"):
        lines.append("")
        lines.append("[bold]verdict history[/bold]")
        for v in match["verdict_history"]:
            lines.append(f"  {v.get('date')}: {v.get('verdict')} (adjudicator: {v.get('adjudicator')})")
    console.print(Panel("\n".join(lines), title=f"[bold]{hypothesis_id}[/bold]", border_style="blue"))


@hypotheses_app.command("validate")
def hypotheses_validate(
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON report")] = False,
) -> None:
    """Validate the registry: schema, vocab, prediction contract, append-only history. Exit 1 on errors."""
    repo_root = Path(__file__).resolve().parent
    data, load_errors = _load_hypothesis_registry(_hypotheses_registry_path())
    parent = _load_parent_hypothesis_registry(repo_root)
    errors, warnings, summary = _validate_hypothesis_registry(data, load_errors, repo_root, parent=parent)
    if json_out:
        console.print_json(
            json.dumps(
                {"errors": errors, "warnings": warnings, "summary": summary, "parent_checked": parent is not None}
            )
        )
    else:
        if errors:
            etable = Table(title=f"[red]{len(errors)} error(s)[/red]", box=box.SIMPLE)
            etable.add_column("error", style="red")
            for e in errors:
                etable.add_row(e)
            console.print(etable)
        if warnings:
            wtable = Table(title=f"[yellow]{len(warnings)} warning(s)[/yellow]", box=box.SIMPLE)
            wtable.add_column("warning", style="yellow")
            for w in warnings:
                wtable.add_row(w)
            console.print(wtable)
        color = "red" if errors else ("yellow" if warnings else "green")
        governance = "append-only vs HEAD" if parent is not None else "no committed parent (first commit)"
        console.print(
            Panel(
                f"[bold {color}]{summary['entries']} entries · {summary['by_status']} · "
                f"{summary['operationalized']} operationalized / {summary['needs_operationalization']} need "
                f"operationalization · governance: {governance} · "
                f"{len(errors)} errors / {len(warnings)} warnings[/bold {color}]",
                title="[bold]mgr hypotheses validate[/bold]",
                border_style=color,
            )
        )
    if errors:
        raise typer.Exit(code=1)


@hypotheses_app.command("add")
def hypotheses_add(
    hypothesis_id: Annotated[str, typer.Option("--id", help="Stable slug, e.g. hyp-tropical-foo")],
    statement: Annotated[str, typer.Option(help="One-sentence prose claim")],
    mechanism: Annotated[list[str], typer.Option("--mechanism", "-m", help="Mechanism (repeatable)")],
    source_kind: Annotated[str, typer.Option(help="human | model")],
    provenance: Annotated[str, typer.Option(help="Where the claim comes from (session, doc, table row)")],
    metric_path: Annotated[
        str | None, typer.Option(help="'<schema>:<dotted.path>' (omit for a not-yet-operationalized claim)")
    ] = None,
    comparator: Annotated[str, typer.Option(help=">= | <=")] = ">=",
    threshold_kind: Annotated[str, typer.Option(help="absolute_delta | ratio")] = "absolute_delta",
    threshold: Annotated[float, typer.Option(help="Effect size (delta or ratio)")] = 0.05,
    baseline_mechanism: Annotated[
        str, typer.Option(help="Baseline arm (equal FLOPs is implied and required)")
    ] = "standard",
    min_seeds: Annotated[int, typer.Option(help="Minimum seeds per arm")] = 3,
    scale_caveats: Annotated[str | None, typer.Option(help="Scale caveats for the prediction")] = None,
    baseline_floor: Annotated[
        float | None,
        typer.Option(help="validity.baseline_floor - answer-prior fallback floor for the ci-v2 gate"),
    ] = None,
    floor_margin: Annotated[
        float | None, typer.Option(help="validity.floor_margin (default 0.0 when floor registered)")
    ] = None,
    floor_source: Annotated[
        str | None, typer.Option(help="validity.floor_source - how the floor was computed (required with --baseline-floor)")
    ] = None,
    note: Annotated[
        str | None, typer.Option(help="operationalization_note (required when --metric-path is omitted)")
    ] = None,
    theorem_ref: Annotated[list[str] | None, typer.Option(help="theorems.yaml id (repeatable)")] = None,
) -> None:
    """Append a new hypothesis (text-append: hand-written comments are preserved), then validate."""
    if metric_path is None and not (note and note.strip()):
        console.print(
            "[bold red]--note (operationalization_note) is required when --metric-path is omitted.[/bold red]"
        )
        raise typer.Exit(code=2)
    if baseline_floor is not None and not (floor_source and floor_source.strip()):
        console.print("[bold red]--floor-source is required when --baseline-floor is given.[/bold red]")
        raise typer.Exit(code=2)
    path = _hypotheses_registry_path()
    original = path.read_text(encoding="utf-8")

    def yq(s: str) -> str:
        return json.dumps(str(s))  # JSON string quoting is valid YAML

    lines = [
        "",
        f"  - id: {hypothesis_id}",
        "    statement: >-",
        *[f"      {chunk}" for chunk in statement.strip().splitlines()],
        f"    mechanisms: [{', '.join(mechanism)}]",
        "    source:",
        f"      kind: {source_kind}",
        f"      provenance: {yq(provenance)}",
        f'    date_registered: "{time.strftime("%Y-%m-%d")}"',
    ]
    if theorem_ref:
        lines.append(f"    theorem_refs: [{', '.join(theorem_ref)}]")
    if metric_path is not None:
        lines.extend(
            [
                "    prediction:",
                f"      metric_path: {yq(metric_path)}",
                f'      comparator: "{comparator}"',
                f"      threshold_kind: {threshold_kind}",
                f"      threshold: {threshold}",
                f"      baseline: {{mechanism: {baseline_mechanism}, equal_flops: true}}",
                f"      min_seeds: {min_seeds}",
            ]
        )
        if baseline_floor is not None:
            lines.append("      validity:")
            lines.append(f"        baseline_floor: {baseline_floor}")
            if floor_margin is not None:
                lines.append(f"        floor_margin: {floor_margin}")
            lines.append(f"        floor_source: {yq(floor_source or '')}")
        if scale_caveats:
            lines.append(f"      scale_caveats: {yq(scale_caveats)}")
        lines.append("    status: open")
    else:
        lines.extend(
            [
                "    prediction: null",
                f"    operationalization_note: {yq(note or '')}",
                "    status: blocked",
            ]
        )
    lines.extend(["    evidence: []", "    verdict_history: []", ""])

    path.write_text(original.rstrip("\n") + "\n" + "\n".join(lines), encoding="utf-8")
    repo_root = Path(__file__).resolve().parent
    data, load_errors = _load_hypothesis_registry(path)
    parent = _load_parent_hypothesis_registry(repo_root)
    errors, _warnings, _summary = _validate_hypothesis_registry(data, load_errors, repo_root, parent=parent)
    if errors:
        path.write_text(original, encoding="utf-8")  # roll back: never leave the registry invalid
        for e in errors:
            console.print(f"[red]{e}[/red]")
        console.print("[bold red]add rolled back: the new entry failed validation.[/bold red]")
        raise typer.Exit(code=1)
    console.print(f"[bold green]Registered {hypothesis_id}[/bold green] → {path}")


def _registry_set_manual_hold(path: Path, hyp_id: str, *, held: bool, reason: str, by: str) -> None:
    """Textual surgery: set (held=True) or clear (held=False -> block removed) a
    hypothesis's manual_hold block, preserving every hand-written comment. The
    caller validates the result and rolls back on any error (mirrors `add` and
    `_registry_append_verdict`)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == f"- id: {hyp_id}")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("- id: ")), len(lines))
    # drop any existing manual_hold block (its 6-space-indented children)
    mh = next((i for i in range(start, end) if lines[i].startswith("    manual_hold:")), None)
    if mh is not None:
        j = mh + 1
        while j < end and lines[j].startswith("      "):
            j += 1
        del lines[mh:j]
        end -= j - mh
    if held:
        status_idx = next(i for i in range(start, end) if lines[i].startswith("    status:"))
        block = [
            "    manual_hold:",
            "      held: true",
            f"      reason: {json.dumps(reason)}",
            f'      date: "{time.strftime("%Y-%m-%d")}"',
            f"      by: {json.dumps(by)}",
        ]
        lines[status_idx + 1 : status_idx + 1] = block
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _hypotheses_revalidate_or_rollback(path: Path, original: str, action: str) -> None:
    """Re-run full registry validation after a textual edit; restore the saved
    text and exit 1 if anything regressed."""
    repo_root = Path(__file__).resolve().parent
    new_data, load_errors = _load_hypothesis_registry(path)
    parent = _load_parent_hypothesis_registry(repo_root)
    errors, _w, _s = _validate_hypothesis_registry(new_data, load_errors, repo_root, parent=parent)
    if errors:
        path.write_text(original, encoding="utf-8")
        for e in errors:
            console.print(f"[red]{e}[/red]")
        console.print(f"[bold red]{action} rolled back: the edit failed validation.[/bold red]")
        raise typer.Exit(code=1)


@hypotheses_app.command("hold")
def hypotheses_hold(
    hypothesis_id: Annotated[str, typer.Argument(help="Hypothesis id to freeze")],
    reason: Annotated[str, typer.Option("--reason", help="Why the verdict is frozen (required)")],
    by: Annotated[str, typer.Option("--by", help="Who placed the hold")] = "human",
) -> None:
    """Freeze a hypothesis's verdict: the engine refuses mechanical adjudication
    (BLOCKED manual_hold) so a later `adjudicate -H`/`--all` cannot re-stamp a
    hand-checked verdict by recency (srtf). Clear with `mgr hypotheses release`."""
    if not reason.strip():
        console.print("[bold red]--reason is required (document why the verdict is frozen).[/bold red]")
        raise typer.Exit(code=2)
    path = _hypotheses_registry_path()
    data = _hypotheses_load_or_exit()
    if not any(h.get("id") == hypothesis_id for h in data.get("hypotheses", [])):
        console.print(f"[bold red]No hypothesis with id {hypothesis_id!r}.[/bold red] Try: mgr hypotheses list")
        raise typer.Exit(code=1)
    original = path.read_text(encoding="utf-8")
    _registry_set_manual_hold(path, hypothesis_id, held=True, reason=reason, by=by)
    _hypotheses_revalidate_or_rollback(path, original, "hold")
    console.print(
        f"[bold yellow]🔒 {hypothesis_id} held[/bold yellow] — the engine will refuse mechanical "
        f"adjudication until `mgr hypotheses release {hypothesis_id}`."
    )


@hypotheses_app.command("release")
def hypotheses_release(
    hypothesis_id: Annotated[str, typer.Argument(help="Hypothesis id to release")],
) -> None:
    """Clear a manual hold (a deliberate human action) so the engine can
    adjudicate the hypothesis again."""
    path = _hypotheses_registry_path()
    data = _hypotheses_load_or_exit()
    match = next((h for h in data.get("hypotheses", []) if h.get("id") == hypothesis_id), None)
    if match is None:
        console.print(f"[bold red]No hypothesis with id {hypothesis_id!r}.[/bold red]")
        raise typer.Exit(code=1)
    hold = match.get("manual_hold")
    if not (isinstance(hold, dict) and hold.get("held")):
        console.print(f"[yellow]{hypothesis_id} is not on hold; nothing to release.[/yellow]")
        return
    original = path.read_text(encoding="utf-8")
    _registry_set_manual_hold(path, hypothesis_id, held=False, reason="", by="")
    _hypotheses_revalidate_or_rollback(path, original, "release")
    console.print(f"[bold green]🔓 {hypothesis_id} released[/bold green] — mechanical adjudication re-enabled.")


@hypotheses_app.command("power")
def hypotheses_power(
    hypothesis: Annotated[
        list[str] | None, typer.Option("--hypothesis", "-H", help="Hypothesis id (repeatable; default: all)")
    ] = None,
    artifacts: Annotated[
        list[Path] | None, typer.Option(help="Artifact root(s) to scan (repeatable)")
    ] = None,
) -> None:
    """Power analysis per hypothesis (bead hij.4): achieved power to detect
    the REGISTERED effect size at the current evidence's variance and seed
    count, and the per-arm seed count needed for 80%. Read-only: runs the
    verdict engine's evidence pipeline without touching the registry.

    "3 seeds, 34% power - needs 8" is the actionable output for a SUPPORTED
    verdict: a SUPPORTED arm below the 50% power-to-confirm floor carries the
    UNDERPOWERED qualifier in adjudication (ci-v6), and this command tells you
    how many seeds fix it. A REFUTED arm is adequate by its CI excluding the
    threshold (refutation_margin), so low power-to-confirm is not a weakness
    there.
    """
    data = _hypotheses_load_or_exit()
    entries = [h for h in data.get("hypotheses", []) if isinstance(h, dict)]
    if hypothesis:
        wanted = set(hypothesis)
        unknown = wanted - {h.get("id") for h in entries}
        if unknown:
            console.print(f"[bold red]Unknown hypothesis id(s): {sorted(unknown)}[/bold red]")
            raise typer.Exit(code=2)
        entries = [h for h in entries if h.get("id") in wanted]
    roots = [Path(p) for p in (artifacts or [Path("artifacts")])]
    index = _adj_collect_artifacts(roots)
    console.print(
        f"[bold cyan]power[/bold cyan] {len(entries)} hypothesis(es) · "
        f"{len(index)} artifact(s) · floor {_ADJ_POWER_FLOOR:.0%} · target {_ADJ_POWER_TARGET:.0%}"
    )
    table = Table(title=f"power analysis (policy {_ADJ_POLICY_VERSION})", box=box.SIMPLE_HEAVY)
    table.add_column("hypothesis", style="bold")
    table.add_column("arm")
    table.add_column("n (c/b)", justify="right")
    table.add_column("power", justify="right")
    table.add_column(f"n for {_ADJ_POWER_TARGET:.0%}", justify="right")
    table.add_column("note")
    for h in entries:
        v = _adjudicate_hypothesis(h, index)
        hid = str(v["id"])
        if v["verdict"] == "blocked":
            table.add_row(hid, "", "", "", "", f"[dim]blocked: {v['reason_code']}[/dim]")
            continue
        for mech, a in v["arms"].items():
            power = a.get("power")
            if power is None:
                note = "power undefined: threshold is the no-effect point"
                table.add_row(hid, mech, f"{a['n_candidate']}/{a['n_baseline']}", "-", "-", note)
                continue
            # ci-v6: UNDERPOWERED is a SUPPORTED-side concern, so mirror exactly
            # what adjudication stamped (arm["underpowered"]) rather than
            # re-deriving power<floor - a REFUTED arm with low power-to-confirm
            # is decisive (refutation_margin), not weak.
            under = bool(a.get("underpowered"))
            margin = a.get("refutation_margin")
            if under:
                note = "UNDERPOWERED: low power to confirm the registered effect"
                style_p = "red"
            elif margin is not None:
                note = f"refuted decisively (margin {margin:.1f}×)"
                style_p = "green" if power >= _ADJ_POWER_TARGET else "yellow"
            else:
                note = ""
                style_p = "green" if power >= _ADJ_POWER_TARGET else "yellow"
            table.add_row(
                hid,
                mech,
                f"{a['n_candidate']}/{a['n_baseline']}",
                f"[{style_p}]{power:.0%}[/{style_p}]",
                str(a.get("n_for_80pct", "-")),
                note,
            )
    console.print(table)


# =============================================================================
# Verdict engine (bead hij.2) — mgr adjudicate. Artifacts in, verdicts out,
# deterministically. The integrity core is REFUSAL: missing/weak/tainted
# evidence yields BLOCKED with a machine-readable reason, never a soft verdict.
# =============================================================================

_ADJ_POLICY_VERSION = "ci-v6"
# ci-v6 (bead 4b82) makes the UNDERPOWERED qualifier SIGN-AWARE. The ci-v4
# power floor asks "could this design have CONFIRMED the registered effect?" -
# the right adequacy lens for a SUPPORTED verdict, the WRONG one for a REFUTED
# verdict. A REFUTED arm already carries a CI that EXCLUDES the threshold (that
# IS the refutation's significance) and the floor gate already downgrades
# powerless refutations to inconclusive; stamping it UNDERPOWERED because the
# test had low power to confirm the OPPOSITE-sign +threshold effect reads as
# "weak refutation" when it is decisive (8h0e: effect -0.245, CI excludes +0.05
# by >3x, was stamped UNDERPOWERED at 29% power-to-confirm). Under ci-v6 the
# UNDERPOWERED qualifier fires on SUPPORTED arms only (byte-identical to ci-v5);
# a REFUTED arm instead records refutation_margin = |effect - threshold| /
# CI-half-width (how many half-widths the interval sits past the threshold on
# the failing side; > 1 by construction, larger = more decisive). `power` (the
# power to confirm the registered effect) is still recorded on every arm as an
# informational quantity. SUPPORTED/INCONCLUSIVE verdicts, p-values, FDR, floor
# gates, and the registry status vocabulary are unchanged; the version bumps
# because a ci-v6 entry's qualifier semantics differ from a ci-v5 entry's.
# ci-v5 (bead qtdq) EXTENDS ci-v4 with the slope-path floor gate: a
# length_slope verdict computed while BOTH arms' held-out exact-match means
# sit at/below the recorded answer prior is downgraded to floor_effect
# inconclusive in BOTH directions (supported and refuted are equally vacuous
# when slopes are fit over floor noise - the o85g audit). All other verdicts
# are computed exactly as under ci-v4. The gate is conservative: unrecorded
# EM/prior leaves the mechanical verdict untouched. This is a
# VERDICT-CHANGING policy step (unlike v3->v4's instrumentation-only step),
# which is precisely why the version bumps: a ledger entry stamped ci-v5
# is distinguishable from one an earlier engine would have stamped.
# ci-v4 (bead hij.4) EXTENDS ci-v3 with statistical-integrity instrumentation;
# every verdict's CI machinery, observation units, floor gate, and budget
# cohorts are computed exactly as under ci-v3. Prior verdicts keep their
# stamps (append-only); re-adjudication under ci-v4 supersedes by appending.
#   - power (per arm): achieved power to detect the REGISTERED effect size -
#     the distance from the no-effect point (delta 0 / ratio 1) to the
#     registered threshold - at the observed variance and seed count, normal
#     approximation, two-sided alpha 0.05. Mildly optimistic at n=3; the
#     power floor is a coarse honesty gate, not an inference. Power is
#     undefined (null) when the threshold IS the no-effect point. Zero
#     observed spread = power 1 by convention (deterministic observable).
#     n_for_80pct: the per-arm seed count solving for 80% power under
#     balanced-arm growth (ratio arms: 1/sqrt(n) scaling of the bootstrap
#     SE by the smaller arm).
#   - UNDERPOWERED qualifier: a SUPPORTED or REFUTED arm (ci-v6 narrows this
#     to SUPPORTED only - see the ci-v6 block above) whose achieved power is
#     below 0.5 carries underpowered: true - a visible asterisk,
#     never a clean verdict (a test that could not have detected the
#     registered effect is noise laundered as evidence). The verdict STATUS
#     is unchanged (the registry vocabulary stays supported/refuted/
#     inconclusive); the qualifier rides in the arms record and rendering.
#   - p_value (per arm): one-sided p for the registered claim against its
#     own threshold (H0 at the threshold, H1 the registered direction) -
#     the test the CI rule already implies (CI95 bound clears threshold
#     <=> one-sided p <= 0.025). Welch/Student arms: parametric t. Ratio
#     arms: add-one-smoothed proportion of the SAME deterministic bootstrap
#     resamples on the wrong side of the threshold (minimum resolvable p =
#     1/(N+1) ~ 1e-4 at N=10_000 - far below the q=0.10 decision band).
#     Record-level p_value = max over arms (a for-all claim is as strong as
#     its weakest arm).
#   - ledger-level FDR: each adjudication run computes Benjamini-Hochberg
#     q-values across ITS family (every non-blocked verdict with a p-value
#     in the run); the headline always reports both "N supported" and "M
#     survive FDR at q=0.10" - the gap IS the honesty. q-values are family-
#     relative and live in run reports (verdicts.json), never the ledger.
#     Directional preregistration already bars forking-paths inflation;
#     FDR is the backstop for family SIZE, not a substitute for it. The
#     report flags families whose ledger mixes policy versions.
# ci-v3 (bead xas7) EXTENDS ci-v2 by appending capabilities; every two-arm
# verdict is computed exactly as under ci-v2 (same observation units, same
# Welch t / bootstrap CIs, same floor gate, same budget cohorts):
#   - certify artifacts (kind: "certify", integer schema_version 1) are
#     readable evidence: metric paths "certify:<mechanism>.<check>.measured"
#     resolve against the checks list; taint derives from the recorded git
#     dirty flag (certify predates the provenance block); budget cohorts do
#     not apply (mathematical invariants are budget-free by construction).
#   - chargeprobe artifacts (mgr.chargeprobe.v1) are readable evidence with
#     evaltasks-style arm matching via meta.checkpoint.
#   - single-arm predictions (baseline: null): the claim is a THRESHOLD on
#     the candidate arm alone (absolute_delta only) - one-sample t CI95
#     (Student t, df = n-1; zero spread -> point CI); SUPPORTED iff the CI
#     lies entirely on the predicted side, REFUTED iff entirely on the
#     failing side. No floor gate (there is no baseline to power-check);
#     within-run trend claims (e.g. train:results.route_coverage_delta > 0)
#     are this shape.
# ci-v2 statistical policy (recorded in every verdict so a future policy
# change cannot silently reinterpret history). ci-v1 verdicts remain in the
# ledger stamped ci-v1; ci-v2 supersedes by APPENDING, never rewriting.
#
#   - observation unit (the ci-v1 -> ci-v2 fix #1): for evaltasks artifacts,
#     ONE observation per TRAINED MODEL - the artifact's `.mean`, which
#     already averages its eval seeds. Eval seeds are repeated measurements
#     of the same checkpoint, not independent replications; ci-v1 pooled
#     them as i.i.d. (3 train seeds x 3 eval seeds = "n=9"), an
#     anti-conservative pseudo-replication. Re-evals of the same checkpoint
#     (lineage run_id + step) are deduped to the newest/richest artifact.
#     train artifacts: per_seed values when present (those ARE training
#     seeds), else the single value. CIs need >= max(2, min_seeds) obs/arm.
#   - absolute_delta: effect = mean(C) - mean(B); Welch t CI95 with
#     Satterthwaite df (fix #2: at the n=3 clustering produces, the normal
#     1.96 understates the interval; t(df~4) ~ 2.78). Zero spread in both
#     arms -> a degenerate point CI at the effect.
#   - ratio: effect = mean(C) / mean(B), compared against the threshold IN
#     RATIO SPACE (for negative-valued metrics like length_slope, dividing by
#     B flips inequalities - the registry header documents the convention);
#     percentile bootstrap CI95
#     (10_000 resamples, numpy default_rng(1234) - deterministic).
#   - SUPPORTED: the CI95 lies entirely on the predicted side of the
#     threshold. REFUTED: entirely on the failing side (the registered
#     effect size is part of the claim). INCONCLUSIVE: the CI straddles.
#   - floor validity gate (fix #3, the pilot1/kbj2 lesson): a REFUTED arm
#     additionally requires the BASELINE to have demonstrably cleared the
#     answer-prior floor - mean(B) > floor + floor_margin. Below it, every
#     arm sits at the best-constant-answer score, the experiment has no
#     power to see a mechanism advantage, and a null effect is evidence of
#     NO POWER, not evidence of absence: the arm downgrades to INCONCLUSIVE
#     with floor_effect: true. The floor is the artifact-recorded
#     answer_prior (mgr.evaltasks.v2: the best constant-answer score on the
#     exact docs scored) when every baseline artifact records it, else the
#     registered prediction.validity.baseline_floor. SUPPORTED needs no
#     gate: clearing baseline + threshold proves the candidate learned.
#   - budget cohorts (fix #4): qualifying artifacts are grouped into
#     planned-FLOPs cohorts (within 5% of a cohort anchor); the verdict uses
#     the LARGEST-budget cohort where both arms reach min_seeds. Bigger-
#     budget evidence supersedes smaller automatically; cross-budget pools
#     never mix. No cohort containing both arms -> BLOCKED budget_mismatch;
#     both arms present but too few runs -> BLOCKED insufficient_seeds.
_ADJ_BUDGET_RTOL = 0.05
_ADJ_BOOTSTRAP_SEED = 1234
_ADJ_BOOTSTRAP_N = 10_000
_ADJ_POWER_FLOOR = 0.5  # below this, SUPPORTED arms carry the UNDERPOWERED qualifier (ci-v6; supported/refuted in ci-v4)
_ADJ_POWER_TARGET = 0.8  # the n_for_80pct column solves for this
_ADJ_FDR_Q = 0.10  # Benjamini-Hochberg level for the ledger-family headline (ci-v4)
_ADJ_ATTENTION_MECHS = frozenset(
    {"standard", "tropical", "ultrametric", "simplicial", "quaternion", "braid", "fractal", "octonion", "surreal", "reversible", "gauge"}
)

_ADJ_BLOCK_REASONS = {
    "prediction_not_operationalized": "prediction is null; operationalization_note names the blocker",
    "no_candidate_artifacts": "no qualifying artifact for the candidate arm",
    "no_baseline_artifacts": "no qualifying artifact for the baseline arm",
    "tainted_evidence": "only tainted artifacts available (dirty tree / missing provenance) - evidence not attributable to a committed code state",
    "insufficient_seeds": "no equal-FLOPs cohort gives both arms at least min_seeds (>= 2) training runs - mgr hypotheses power computes the seed count the registered effect actually needs",
    "budget_mismatch": "no equal-FLOPs cohort (5% tolerance) contains both arms, or an artifact cannot prove its planned budget",
    "metric_missing": "qualifying artifact lacks the metric at the registered path",
    "manual_hold": "hypothesis is on manual hold - a human froze the verdict; the engine refuses to re-stamp it mechanically. Release with `mgr hypotheses release <id>` before adjudicating.",
}


def _adj_walk_path(data: Any, dotted: str) -> Any:
    cur = data
    for seg in dotted.split("."):
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        elif isinstance(cur, list) and seg.isdigit() and int(seg) < len(cur):
            cur = cur[int(seg)]
        else:
            return None
    return cur


def _adj_collect_artifacts(roots: list[Path]) -> list[dict[str, Any]]:
    """Index every summary.json under the roots by schema kind."""
    index: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("summary.json")):
            parts = path.parts
            if ("probes", "sizing") in zip(parts, parts[1:]):
                # probes/sizing/ is the evidence-pool quarantine (bead dzor):
                # sizing/calibration runs SELECT rungs and budgets, so
                # admitting them as evidence is selection bias by
                # construction. probes/charges stays evidence - chargeprobe
                # instruments measure, they do not select.
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            sv = data.get("schema_version")
            if sv in ("mgr.evaltasks.v1", "mgr.evaltasks.v2", "mgr.evaltasks.v3"):
                schema = "evaltasks"
            elif sv == "mgr.telemetry.v1":
                schema = "train"
            elif sv == "mgr.chargeprobe.v1":
                schema = "chargeprobe"
            elif sv in (
                "mgr.bench.ultrametric_paths.v1",
                "mgr.bench.precision_curves.v1",
                "mgr.bench.depth_curves.v1",
                "mgr.bench.coord_curves.v1",
            ):
                # coord_curves (bp08): coordinate-check curves from
                # nanochat.parameterization.write_coord_check_artifact -
                # mechanism/meta/results shape, results.parameterization is
                # the rgyl arm selector; makes hyp-coordcheck-clt-flat and
                # hyp-tropical-evt-miscoupling engine-adjudicable.
                schema = "bench"
            elif sv == 1 and data.get("kind") == "certify":
                schema = "certify"
            else:
                continue
            if schema == "certify":
                # certify predates the provenance block; the evidence-producing
                # recipe is the committed code + seed, so taint = dirty tree
                git = data.get("git")
                tainted = (not isinstance(git, dict)) or bool(git.get("dirty", True))
            else:
                prov = data.get("provenance")
                tainted = (not isinstance(prov, dict)) or bool(prov.get("tainted", True))
                if schema == "evaltasks":
                    # dz9i: training taint propagates - a clean-time eval of a
                    # dirty-trained checkpoint is still tainted evidence. Only
                    # enforceable when the artifact carries train_provenance
                    # (v3); legacy artifacts keep eval-time-only semantics.
                    train_prov = data.get("train_provenance")
                    if isinstance(train_prov, dict):
                        tainted = tainted or bool(train_prov.get("tainted", True))
            index.append({"path": str(path), "schema": schema, "data": data, "tainted": tainted})
    return index


def _adj_variant_matches(art: dict[str, Any], variant: dict[str, Any] | None) -> bool:
    """Variant selector (rgyl): every key in `variant` must equal the
    artifact's recorded value - looked up in hparams first, then config for
    train artifacts; in meta.checkpoint for evaltasks. A null value selects
    artifacts where the knob is recorded null OR absent (the all-defaults
    arm), so 'exact tropical' (semiring_beta_spec: null) matches both new
    artifacts that record the field and pre-rgyl artifacts that predate it.
    This is what lets annealed / fixed-beta / endpoint runs of the SAME
    mechanism be distinguished as evidence (hyp-maslov-anneal-loss-retention).
    For bench artifacts (bhjf) knobs are looked up in results then meta, so
    two bench-backed hypotheses sharing a metric path stay disjoint via
    selectors rather than per-protocol observable keys."""
    if not variant:
        return True
    data = art["data"]
    if art["schema"] in ("evaltasks", "chargeprobe"):
        ckpt = (data.get("meta") or {}).get("checkpoint") or {}
        # knobs live either on the checkpoint block itself (step, attention
        # type, probe extras) or in its recorded model_config (every GPTConfig
        # knob - braid_crossing_law, ultrametric_mode, ...)
        model_cfg = ckpt.get("model_config")
        sources: list[dict[str, Any]] = [ckpt, model_cfg if isinstance(model_cfg, dict) else {}]
    elif art["schema"] == "bench":
        # bench protocols (bhjf): window bounds and derived knobs land in
        # results; task/seed/run parameters land in meta
        res = data.get("results")
        bench_meta = data.get("meta")
        sources = [res if isinstance(res, dict) else {}, bench_meta if isinstance(bench_meta, dict) else {}]
    else:
        sources = [data.get("hparams") or {}, data.get("config") or {}]
    for key, wanted in variant.items():
        found = None
        for src in sources:
            if key in src:
                found = src[key]
                break
        if found != wanted:
            return False
    return True


def _adj_artifact_matches_arm(art: dict[str, Any], mechanism: str, variant: dict[str, Any] | None = None) -> bool:
    """Arm membership: an arm changes exactly ITS axis from the all-defaults
    baseline (attention standard, scheduler none, optimizer non-hoss); an
    optional variant selector further restricts within the mechanism."""
    data = art["data"]
    if art["schema"] == "certify":
        # certify runs are per-mechanism invariant suites, not trained models:
        # the artifact witnesses a mechanism iff it carries that mechanism's
        # check records (variant selectors do not apply - the law is encoded
        # in the check NAME, e.g. braid.rmatrix_*).
        checks = data.get("checks")
        return isinstance(checks, list) and any(
            isinstance(c, dict) and c.get("mechanism") == mechanism for c in checks
        )
    if art["schema"] == "bench":
        # path benchmarks record the mechanism they exercise top-level; the
        # variant selector separates protocol variants sharing a metric path
        # (bhjf - previously selectors were silently ignored here)
        return bool(data.get("mechanism") == mechanism) and _adj_variant_matches(art, variant)
    if art["schema"] in ("evaltasks", "chargeprobe"):
        attn = ((data.get("meta") or {}).get("checkpoint") or {}).get("attention_type")
        return attn == (mechanism if mechanism in _ADJ_ATTENTION_MECHS else "standard") and _adj_variant_matches(
            art, variant
        )
    config = data.get("config") or {}
    hparams = data.get("hparams") or {}
    attn = config.get("attention_type")
    sched = hparams.get("scheduler_type", "none")
    opt = config.get("optimizer_type", "adamw")
    if mechanism == "ordinal":
        return attn == "standard" and sched == "ordinal" and _adj_variant_matches(art, variant)
    if mechanism == "hoss":
        return attn == "standard" and opt == "hoss" and _adj_variant_matches(art, variant)
    return attn == mechanism and sched != "ordinal" and opt != "hoss" and _adj_variant_matches(art, variant)


def _adj_planned_flops(art: dict[str, Any]) -> float | None:
    data = art["data"]
    if art["schema"] in ("evaltasks", "chargeprobe"):
        budget = ((data.get("meta") or {}).get("checkpoint") or {}).get("budget") or {}
    else:
        budget = data.get("budget") or {}
    target = budget.get("target_flops")
    if isinstance(target, (int, float)) and target:
        return float(target)
    planned = budget.get("planned_total_flops_est")
    if isinstance(planned, (int, float)) and planned:
        return float(planned)
    max_steps = budget.get("max_steps")
    per_step = budget.get("flops_per_step_est")
    if isinstance(max_steps, int) and isinstance(per_step, (int, float)):
        return float(max_steps * per_step)
    return None


def _adj_dedupe_evaltasks(arts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One eval artifact per trained checkpoint (ci-v2): re-evals of the same
    training run (lineage run_id + step) are repeated measurements of the same
    model, never extra evidence. Keep the richest/newest artifact (schema v2
    over v1, then generated_at, then path - all deterministic)."""
    best: dict[Any, tuple[tuple[str, str, str], dict[str, Any]]] = {}
    out: list[dict[str, Any]] = []
    for a in arts:
        if a["schema"] not in ("evaltasks", "chargeprobe", "certify", "bench"):
            out.append(a)
            continue
        meta = a["data"].get("meta") or {}
        if a["schema"] == "certify":
            # re-runs of the same certify seed are byte-replays, never extra
            # evidence; distinct seeds ARE replications and all survive
            key: Any = ("certify", a["data"].get("seed"), a["data"].get("device"), a["data"].get("dtype"))
            rank = ("", str(a["data"].get("run_id", "")), str(a["path"]))
        elif a["schema"] == "bench":
            # one bench observation per (seed, device); timings are best-of-N
            key = ("bench", a["data"].get("schema_version"), meta.get("seed"), meta.get("device"))
            rank = ("", str(meta.get("generated_at", "")), str(a["path"]))
        else:
            ckpt = meta.get("checkpoint") or {}
            lineage = ckpt.get("lineage") or {}
            run_id = lineage.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                if a["schema"] == "chargeprobe":
                    # probe artifacts key on (checkpoint dir, step, task, seed)
                    key = ("chargeprobe", ckpt.get("dir"), ckpt.get("step"), meta.get("task"), meta.get("seed"))
                else:
                    out.append(a)  # no lineage -> cannot dedupe; keep as-is
                    continue
            else:
                key = (a["schema"], run_id, ckpt.get("step"), meta.get("task"), meta.get("seed")) if a[
                    "schema"
                ] == "chargeprobe" else (run_id, ckpt.get("step"))
            rank = (
                str(a["data"].get("schema_version", "")),  # "...v2" > "...v1"
                str(meta.get("generated_at", "")),
                str(a["path"]),
            )
        prev = best.get(key)
        if prev is None or rank > prev[0]:
            best[key] = (rank, a)
    out.extend(a for _, a in best.values())
    return sorted(out, key=lambda a: str(a["path"]))


def _adj_observations(art: dict[str, Any], dotted: str) -> list[float] | None:
    """ci-v2 observation units. evaltasks: ONE observation per artifact - the
    `.mean` value, which already averages that checkpoint's eval seeds (eval
    seeds are repeated measurements, not replications). train: per_seed values
    when present (those ARE training seeds), else the single value.
    certify (ci-v3): "<mechanism>.<check>.<field>" resolves against the checks
    list - one observation per certify run (runs at distinct seeds replicate).
    chargeprobe (ci-v3): plain dotted walk into the summary."""
    if art["schema"] == "certify":
        segs = dotted.split(".")
        if len(segs) != 3:
            return None
        mech, check, field = segs
        for c in art["data"].get("checks") or []:
            if isinstance(c, dict) and c.get("mechanism") == mech and c.get("check") == check:
                value = c.get(field)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return [float(value)]
                return None
        return None
    value = _adj_walk_path(art["data"], dotted)
    if art["schema"] != "evaltasks" and dotted.endswith(".mean"):
        parent = _adj_walk_path(art["data"], dotted.rsplit(".", 1)[0])
        if isinstance(parent, dict) and isinstance(parent.get("per_seed"), list):
            obs = [float(v) for v in parent["per_seed"] if isinstance(v, (int, float))]
            return obs or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    return None


def _adj_slope_floor_paths(dotted: str) -> tuple[str, str] | None:
    """For length_slope metric paths, derive the held-out EM and answer-prior
    paths that decide whether the slopes were fit over floor noise (qtdq):
    tasks.<t>.length_slope.<...> ->
      (tasks.<t>.exact_match.greedy.held_out.mean,
       tasks.<t>.answer_prior.held_out.mean).
    None for non-slope metrics - they have their own gate (_adj_prior_path)."""
    segs = dotted.split(".")
    if "length_slope" not in segs:
        return None
    base = segs[: segs.index("length_slope")]
    return (
        ".".join(base + ["exact_match", "greedy", "held_out", "mean"]),
        ".".join(base + ["answer_prior", "held_out", "mean"]),
    )


def _adj_prior_path(dotted: str) -> str | None:
    """Derive the recorded-floor path from an exact-match metric path:
    tasks.<t>.exact_match.<mode>.<region>.mean -> tasks.<t>.answer_prior.<region>.mean
    (None for non-exact-match metrics - the recorded floor only exists there)."""
    segs = dotted.split(".")
    if segs[-1] != "mean" or "exact_match" not in segs:
        return None
    i = segs.index("exact_match")
    if len(segs) != i + 4:  # exact_match.<mode>.<region>.mean
        return None
    return ".".join(segs[:i] + ["answer_prior", segs[i + 2], "mean"])


def _adj_ci(cand: list[float], base: list[float], threshold_kind: str) -> tuple[float, float, float]:
    """(effect, ci_low, ci_high) under the ci-v2 policy."""
    import statistics as stats_mod

    import numpy as _np

    if threshold_kind == "absolute_delta":
        effect = stats_mod.mean(cand) - stats_mod.mean(base)
        vc, vb = stats_mod.variance(cand), stats_mod.variance(base)
        nc, nb = len(cand), len(base)
        se = (vc / nc + vb / nb) ** 0.5
        if se == 0.0:
            return effect, effect, effect  # zero spread in both arms: a point CI
        # Welch t with Satterthwaite df: honest widths at the small n that
        # per-training-run clustering produces (n=3 -> crit ~2.78, not 1.96)
        df = (vc / nc + vb / nb) ** 2 / ((vc / nc) ** 2 / (nc - 1) + (vb / nb) ** 2 / (nb - 1))
        from scipy import stats as _scipy_stats

        crit = float(_scipy_stats.t.ppf(0.975, df))
        return effect, effect - crit * se, effect + crit * se
    c = _np.asarray(cand, dtype=float)
    b = _np.asarray(base, dtype=float)
    effect = float(c.mean() / b.mean())
    ratios = _adj_bootstrap_ratios(cand, base)
    lo, hi = _np.percentile(ratios, [2.5, 97.5])
    return effect, float(lo), float(hi)


def _adj_bootstrap_ratios(cand: list[float], base: list[float]) -> list[float]:
    """The engine's single deterministic bootstrap stream for ratio arms
    (fixed seed, fixed resample count): the CI percentiles, the ci-v4
    p-value, and the ci-v4 power estimate all read the SAME resamples, so
    they can never disagree about the sampling distribution."""
    import numpy as _np

    rng = _np.random.default_rng(_ADJ_BOOTSTRAP_SEED)
    c = _np.asarray(cand, dtype=float)
    b = _np.asarray(base, dtype=float)
    ratios: list[float] = []
    for _ in range(_ADJ_BOOTSTRAP_N):
        cs = c[rng.integers(0, len(c), len(c))]
        bs = b[rng.integers(0, len(b), len(b))]
        if bs.mean() != 0:
            ratios.append(float(cs.mean() / bs.mean()))
    return ratios


def _adj_ci_single(obs: list[float]) -> tuple[float, float, float]:
    """(mean, ci_low, ci_high) for a single-arm prediction (ci-v3): one-sample
    Student t CI95 (df = n-1); zero spread -> a degenerate point CI."""
    import statistics as stats_mod

    effect = stats_mod.mean(obs)
    n = len(obs)
    var = stats_mod.variance(obs) if n > 1 else 0.0
    se = (var / n) ** 0.5
    if se == 0.0:
        return effect, effect, effect
    from scipy import stats as _scipy_stats

    crit = float(_scipy_stats.t.ppf(0.975, n - 1))
    return effect, effect - crit * se, effect + crit * se


def _adj_null_reference(threshold_kind: str) -> float:
    """The no-effect point the registered effect size is measured FROM:
    a delta of 0 (absolute_delta) or a ratio of 1 (ratio)."""
    return 1.0 if threshold_kind == "ratio" else 0.0


def _adj_p_value(
    cand: list[float],
    base: list[float],
    threshold: float,
    threshold_kind: str,
    comparator: str,
    single_arm: bool,
) -> float | None:
    """One-sided p-value for the registered claim against its own threshold
    (ci-v4): H0 places the true effect AT the threshold, H1 is the registered
    direction - the test the CI verdict rule already implies (CI95 bound
    clears threshold <=> one-sided p <= 0.025), so BH over these p-values
    measures exactly which verdicts survive the family. Welch/Student arms:
    parametric t. Ratio arms: add-one-smoothed proportion of the engine's
    deterministic bootstrap resamples on the wrong side of the threshold
    (minimum resolvable p = 1/(N+1)). Zero spread degenerates to 0 or 1."""
    import statistics as stats_mod

    if threshold_kind == "absolute_delta":
        if single_arm:
            n = len(cand)
            effect = stats_mod.mean(cand)
            var = stats_mod.variance(cand) if n > 1 else 0.0
            se = (var / n) ** 0.5
            df = float(n - 1)
        else:
            effect = stats_mod.mean(cand) - stats_mod.mean(base)
            vc, vb = stats_mod.variance(cand), stats_mod.variance(base)
            nc, nb = len(cand), len(base)
            se = (vc / nc + vb / nb) ** 0.5
            df = (
                (vc / nc + vb / nb) ** 2 / ((vc / nc) ** 2 / (nc - 1) + (vb / nb) ** 2 / (nb - 1))
                if se > 0.0
                else 1.0
            )
        if se == 0.0:
            ok = effect >= threshold if comparator == ">=" else effect <= threshold
            return 0.0 if ok else 1.0
        from scipy import stats as _scipy_stats

        sf = float(_scipy_stats.t.sf((effect - threshold) / se, df))
        return sf if comparator == ">=" else 1.0 - sf
    ratios = _adj_bootstrap_ratios(cand, base)
    if not ratios:
        return None
    if comparator == ">=":
        wrong = sum(1 for r in ratios if r < threshold)
    else:
        wrong = sum(1 for r in ratios if r > threshold)
    return (wrong + 1) / (len(ratios) + 1)


def _adj_power(
    cand: list[float],
    base: list[float],
    threshold: float,
    threshold_kind: str,
    single_arm: bool,
) -> dict[str, Any] | None:
    """Achieved power (ci-v4, normal approximation, two-sided alpha 0.05) to
    detect the REGISTERED effect size - the distance from the no-effect point
    to the registered threshold - at the observed variance and seed count,
    plus the per-arm seed count needed for the 80% target under balanced-arm
    growth. None when the threshold IS the no-effect point (effect size zero:
    power undefined). Zero observed spread = power 1 by convention (a
    deterministic observable). Mildly optimistic at n=3 (normal, not t);
    the power floor is a coarse honesty gate, not an inference."""
    import statistics as stats_mod

    effect_size = abs(threshold - _adj_null_reference(threshold_kind))
    if effect_size == 0.0:
        return None
    from scipy import stats as _scipy_stats

    # z for the CI's two-sided alpha and z for the power TARGET - derived
    # from the policy constant so the n_for_80pct column can never disagree
    # with _ADJ_POWER_TARGET (norm.ppf(0.8) = 0.8416...)
    z975 = 1.959963984540054
    z80 = float(_scipy_stats.norm.ppf(_ADJ_POWER_TARGET))
    if threshold_kind == "absolute_delta":
        if single_arm:
            n_now = len(cand)
            basis = stats_mod.variance(cand) if n_now > 1 else 0.0  # SE(n)^2 = basis/n
        else:
            nc, nb = len(cand), len(base)
            n_now = min(nc, nb)
            basis = (stats_mod.variance(cand) / nc + stats_mod.variance(base) / nb) * n_now
    else:
        ratios = _adj_bootstrap_ratios(cand, base)
        if len(ratios) < 2:
            return None
        n_now = min(len(cand), len(base))
        basis = stats_mod.variance(ratios) * n_now  # 1/sqrt(n) scaling by the smaller arm
    se_now = (basis / n_now) ** 0.5
    if se_now == 0.0:
        return {"power": 1.0, "n_for_80pct": n_now}
    power = float(_scipy_stats.norm.cdf(effect_size / se_now - z975))
    n80 = math.ceil(basis * (z975 + z80) ** 2 / (effect_size * effect_size))
    return {"power": power, "n_for_80pct": max(n80, 2)}


def _adj_bh_qvalues(pvals: dict[str, float]) -> dict[str, float]:
    """Benjamini-Hochberg adjusted q-values (step-up, monotone-enforced):
    q_(i) = min over j >= i of p_(j) * m / j on the ascending-sorted p's.
    Deterministic tie-break on the key keeps reports stable."""
    items = sorted(pvals.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    q = [0.0] * m
    running = 1.0
    for i in range(m - 1, -1, -1):
        running = min(running, items[i][1] * m / (i + 1))
        q[i] = running
    return {items[i][0]: q[i] for i in range(m)}


def _adjudicate_hypothesis(hyp: dict[str, Any], artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """One hypothesis -> a verdict record (verdict in supported/refuted/
    inconclusive) or a refusal (verdict 'blocked' + reason_code)."""
    hid = hyp.get("id")
    # manual_hold (srtf): a human-frozen verdict. The mechanical engine reads
    # the same evidence every run, so without this guard `adjudicate --all`
    # silently RE-STAMPS a hand-checked verdict by recency (e.g. the z4xx
    # symplectic val_ce ratio ~0.004 reads SUPPORTED, clobbering the
    # INCONCLUSIVE metric-commensurability hand-check). A blocked verdict is
    # report-only and never appended, so the status line and verdict_history
    # are left untouched - for both `-H` and `--all` - until a human releases.
    hold = hyp.get("manual_hold")
    if isinstance(hold, dict) and bool(hold.get("held")):
        return {"id": hid, "verdict": "blocked", "reason_code": "manual_hold",
                "reason": _ADJ_BLOCK_REASONS["manual_hold"], "hold_reason": str(hold.get("reason", ""))}
    pred = hyp.get("prediction")
    if not isinstance(pred, dict):
        return {"id": hid, "verdict": "blocked", "reason_code": "prediction_not_operationalized",
                "reason": _ADJ_BLOCK_REASONS["prediction_not_operationalized"]}
    schema, _, dotted = str(pred.get("metric_path", "")).partition(":")
    # ci-v3 single-arm predictions: an explicit `baseline: null` makes the
    # claim a threshold on the candidate arm alone (no contrast, no floor gate)
    single_arm = "baseline" in pred and pred.get("baseline") is None
    baseline_mech = (pred.get("baseline") or {}).get("mechanism", "standard")
    min_seeds = int(pred.get("min_seeds", 3))
    comparator = pred.get("comparator")
    threshold = float(pred.get("threshold", 0.0))
    threshold_kind = str(pred.get("threshold_kind", "absolute_delta"))
    validity_raw = pred.get("validity")
    validity: dict[str, Any] = validity_raw if isinstance(validity_raw, dict) else {}
    # Variant selectors (rgyl): restrict arms WITHIN a mechanism by recorded
    # hparams/config values (e.g. semiring_beta_spec distinguishes annealed
    # vs fixed-beta vs exact-tropical evidence). Absent selectors = the
    # pre-rgyl behavior, identically.
    base_variant_raw = (pred.get("baseline") or {}).get("variant")
    base_variant: dict[str, Any] | None = base_variant_raw if isinstance(base_variant_raw, dict) else None
    cand_variant_raw = pred.get("candidate_variant")
    cand_variant: dict[str, Any] | None = cand_variant_raw if isinstance(cand_variant_raw, dict) else None
    need = max(2, min_seeds)

    pool = [a for a in artifacts if a["schema"] == schema]
    arms: dict[str, dict[str, Any]] = {}
    used_paths: list[str] = []
    saw_tainted = False
    floor_gated = False
    # A multi-mechanism entry is a FOR-ALL claim: every listed mechanism must
    # adjudicate; the overall verdict is the worst case across arms.
    for mech in hyp.get("mechanisms") or []:
        cand_all = [a for a in pool if _adj_artifact_matches_arm(a, mech, cand_variant)]
        base_all = (
            [] if single_arm else [a for a in pool if _adj_artifact_matches_arm(a, baseline_mech, base_variant)]
        )
        saw_tainted = saw_tainted or any(a["tainted"] for a in cand_all + base_all)
        cand = _adj_dedupe_evaltasks([a for a in cand_all if not a["tainted"]])
        base = _adj_dedupe_evaltasks([a for a in base_all if not a["tainted"]])
        if not cand:
            code = "tainted_evidence" if cand_all else "no_candidate_artifacts"
            return {"id": hid, "verdict": "blocked", "reason_code": code, "mechanism": mech,
                    "reason": _ADJ_BLOCK_REASONS[code]}
        if not base and not single_arm:
            code = "tainted_evidence" if base_all else "no_baseline_artifacts"
            return {"id": hid, "verdict": "blocked", "reason_code": code, "mechanism": mech,
                    "reason": _ADJ_BLOCK_REASONS[code]}
        # qualifying = carries the registered metric (keeps evidence lists
        # honest: an arith-trained model's eval cannot witness a hier claim)
        cand_m = [a for a in cand if _adj_observations(a, dotted)]
        base_m = [a for a in base if _adj_observations(a, dotted)]
        if not cand_m or (not base_m and not single_arm):
            return {"id": hid, "verdict": "blocked", "reason_code": "metric_missing", "mechanism": mech,
                    "reason": _ADJ_BLOCK_REASONS["metric_missing"]}
        # budget cohorts: every artifact must prove its planned FLOPs; the
        # verdict uses the LARGEST cohort where both arms reach min_seeds.
        # certify artifacts are budget-exempt (ci-v3): mathematical invariant
        # suites have no training budget; all qualifying runs form one cohort.
        anchor: float | None
        cc: list[dict[str, Any]]
        bb: list[dict[str, Any]]
        if schema in ("certify", "bench"):
            # budget-exempt schemas (ci-v3): invariant suites and path
            # microbenchmarks have no training budget by construction; both
            # arms (the baseline arm only for two-arm predictions) must still
            # reach min_seeds.
            n_c = sum(len(_adj_observations(a, dotted) or []) for a in cand_m)
            n_b = need if single_arm else sum(len(_adj_observations(a, dotted) or []) for a in base_m)
            if n_c < need or n_b < need:
                return {"id": hid, "verdict": "blocked", "reason_code": "insufficient_seeds", "mechanism": mech,
                        "reason": _ADJ_BLOCK_REASONS["insufficient_seeds"]}
            anchor, cc, bb = None, cand_m, ([] if single_arm else base_m)
        else:
            cand_f = [(a, f) for a in cand_m if (f := _adj_planned_flops(a)) is not None]
            base_f = [(a, f) for a in base_m if (f := _adj_planned_flops(a)) is not None]
            if not cand_f or (not base_f and not single_arm):
                return {"id": hid, "verdict": "blocked", "reason_code": "budget_mismatch", "mechanism": mech,
                        "reason": _ADJ_BLOCK_REASONS["budget_mismatch"]}
            anchors = sorted({f for _, f in cand_f + base_f}, reverse=True)
            chosen: tuple[float, list[dict[str, Any]], list[dict[str, Any]]] | None = None
            both_arms_seen = False
            for cohort_anchor in anchors:
                lo_bound = cohort_anchor * (1.0 - _ADJ_BUDGET_RTOL)
                cc_try = [a for a, f in cand_f if lo_bound <= f <= cohort_anchor]
                bb_try = [a for a, f in base_f if lo_bound <= f <= cohort_anchor]
                if not cc_try or (not bb_try and not single_arm):
                    continue
                both_arms_seen = True
                n_c = sum(len(_adj_observations(a, dotted) or []) for a in cc_try)
                n_b = sum(len(_adj_observations(a, dotted) or []) for a in bb_try)
                if n_c >= need and (single_arm or n_b >= need):
                    chosen = (cohort_anchor, cc_try, bb_try)
                    break
            if chosen is None:
                code = "insufficient_seeds" if both_arms_seen else "budget_mismatch"
                return {"id": hid, "verdict": "blocked", "reason_code": code, "mechanism": mech,
                        "reason": _ADJ_BLOCK_REASONS[code]}
            anchor, cc, bb = chosen
        cand_obs = [v for a in cc for v in (_adj_observations(a, dotted) or [])]
        base_obs = [v for a in bb for v in (_adj_observations(a, dotted) or [])]
        if single_arm:
            effect, lo, hi = _adj_ci_single(cand_obs)
        else:
            effect, lo, hi = _adj_ci(cand_obs, base_obs, threshold_kind)
        if comparator == ">=":
            arm_verdict = "supported" if lo >= threshold else ("refuted" if hi < threshold else "inconclusive")
        else:  # "<="
            arm_verdict = "supported" if hi <= threshold else ("refuted" if lo > threshold else "inconclusive")
        arm: dict[str, Any] = {
            "effect": effect,
            "ci95": [lo, hi],
            "n_candidate": len(cand_obs),
            "n_baseline": len(base_obs),
            "budget_flops": anchor,
        }
        if single_arm:
            arm["single_arm"] = True
        # ---- floor validity gate (ci-v2; no baseline -> no gate) ----
        floor: float | None = None
        floor_src: str | None = None
        prior_dotted = _adj_prior_path(dotted) if not single_arm else None
        if prior_dotted is not None:
            priors = [_adj_walk_path(a["data"], prior_dotted) for a in bb]
            vals = [float(p) for p in priors if isinstance(p, (int, float)) and not isinstance(p, bool)]
            if vals and len(vals) == len(bb):
                floor = sum(vals) / len(vals)
                floor_src = "recorded_answer_prior"
        if floor is None and not single_arm:
            reg = validity.get("baseline_floor")
            if isinstance(reg, (int, float)) and not isinstance(reg, bool):
                floor = float(reg)
                floor_src = "registered_baseline_floor"
        if floor is not None:
            margin = validity.get("floor_margin", 0.0)
            margin = float(margin) if isinstance(margin, (int, float)) and not isinstance(margin, bool) else 0.0
            baseline_mean = sum(base_obs) / len(base_obs)
            arm["baseline_mean"] = baseline_mean
            arm["baseline_floor"] = floor
            arm["floor_source"] = floor_src
            if arm_verdict == "refuted" and baseline_mean <= floor + margin:
                # the baseline never demonstrably learned the task: a null
                # effect here is evidence of NO POWER, not evidence of absence
                arm_verdict = "inconclusive"
                arm["floor_effect"] = True
                floor_gated = True
        # ---- slope-path floor gate (qtdq) ----
        # A length-generalization slope fit while BOTH arms sit at/below the
        # recorded answer-prior floor is fit over floor noise: a zero slope is
        # flat-because-dead, a nonzero one is sign noise. Unlike the EM gate
        # above this fires on supported AND refuted - both directions are
        # equally vacuous - closing the hole where mechanical verdicts
        # overrode the registered hand-checks by recency (o85g audit).
        slope_paths = _adj_slope_floor_paths(dotted) if not single_arm else None
        if slope_paths is not None and arm_verdict in ("supported", "refuted"):
            em_path, em_prior_path = slope_paths

            def _arm_mean(arts: list[dict[str, Any]], path: str) -> float | None:
                vals = [_adj_walk_path(a["data"], path) for a in arts]
                nums = [float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
                return (sum(nums) / len(nums)) if nums and len(nums) == len(arts) else None

            cand_em = _arm_mean(cc, em_path)
            base_em = _arm_mean(bb, em_path)
            em_floor = _arm_mean(bb, em_prior_path)
            if (
                em_floor is not None
                and cand_em is not None
                and base_em is not None
                and cand_em <= em_floor
                and base_em <= em_floor
            ):
                arm_verdict = "inconclusive"
                arm["floor_effect"] = True
                arm["floor_source"] = "slope_em_floor"
                arm["candidate_em_mean"] = cand_em
                arm["baseline_em_mean"] = base_em
                arm["em_floor"] = em_floor
                floor_gated = True
        # ---- ci-v4 statistical integrity: p-value + power per arm ----
        arm["p_value"] = _adj_p_value(
            cand_obs, base_obs, threshold, threshold_kind, str(comparator), single_arm
        )
        power_info = _adj_power(cand_obs, base_obs, threshold, threshold_kind, single_arm)
        if power_info is not None:
            arm["power"] = power_info["power"]
            arm["n_for_80pct"] = power_info["n_for_80pct"]
            # ci-v6: the registered-effect power floor is the SUPPORTED-side
            # adequacy lens ("could this design have CONFIRMED the +threshold
            # effect?") - noise laundered as evidence when low (hij.4). It is
            # the WRONG lens for a refutation, so the UNDERPOWERED qualifier
            # fires on SUPPORTED arms only (byte-identical to ci-v4/v5 there).
            if arm_verdict == "supported" and power_info["power"] < _ADJ_POWER_FLOOR:
                arm["underpowered"] = True
        # ci-v6: a refutation's adequacy is its PRECISION - how decisively the
        # CI excludes the threshold - not the power to confirm the opposite-sign
        # effect. refutation_margin counts CI-half-widths from the effect to the
        # threshold (> 1 by construction for a REFUTED arm; the floor gate has
        # already removed powerless refutations), so a decisive refutation
        # (large margin) is never misread as weak (bead 4b82).
        if arm_verdict == "refuted":
            half = (hi - lo) / 2.0
            if half > 0.0:
                arm["refutation_margin"] = abs(effect - threshold) / half
        arm["verdict"] = arm_verdict
        arms[mech] = arm
        used_paths.extend(sorted({a["path"] for a in cc + bb}))

    order = {"refuted": 0, "inconclusive": 1, "supported": 2}
    verdict = min((a["verdict"] for a in arms.values()), key=lambda v: order[v])
    record: dict[str, Any] = {
        "id": hid,
        "verdict": verdict,
        "arms": arms,
        "artifacts": sorted(set(used_paths)),
        "policy_version": _ADJ_POLICY_VERSION,
        "tainted_artifacts_seen": saw_tainted,
    }
    # ci-v4: a for-all claim is as strong as its weakest arm
    arm_ps = [a.get("p_value") for a in arms.values()]
    record["p_value"] = (
        None if any(p is None for p in arm_ps) else max(p for p in arm_ps if p is not None)
    )
    if any(a.get("underpowered") for a in arms.values()):
        record["underpowered"] = True
    if floor_gated:
        record["floor_effect"] = True
    return record


def _registry_append_verdict(path: Path, hyp_id: str, entry_line: str, new_status: str) -> None:
    """Textual ledger surgery: append a single-line flow-mapping history item
    and update the status line, preserving every hand-written comment. The
    caller validates the result and rolls back on any error."""
    lines = path.read_text(encoding="utf-8").splitlines()
    # locate the entry block
    start = next(i for i, ln in enumerate(lines) if ln.strip() == f"- id: {hyp_id}")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("- id: ")), len(lines))
    block = list(range(start, end))
    status_idx = next(i for i in block if lines[i].startswith("    status:"))
    lines[status_idx] = f"    status: {new_status}"
    vh_idx = next(i for i in block if lines[i].startswith("    verdict_history:"))
    if lines[vh_idx].strip() == "verdict_history: []":
        lines[vh_idx] = "    verdict_history:"
        lines.insert(vh_idx + 1, entry_line)
    else:
        # append AFTER the last existing item (history is chronological)
        insert_at = vh_idx + 1
        while insert_at < end and (
            lines[insert_at].startswith("      - ") or lines[insert_at].startswith("        ")
        ):
            insert_at += 1
        lines.insert(insert_at, entry_line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report_train_row(art: dict[str, Any]) -> dict[str, Any]:
    """One roll-up row per train summary (mgr.telemetry.v1)."""
    data = art["data"]
    cfg = data.get("config") or {}
    ds = (data.get("dataset") or {}).get("data_dir")
    res = data.get("results") or {}
    meta = data.get("meta") or {}
    argv = [str(a) for a in (meta.get("argv") or [])]
    seed = "?"
    if "--seed" in argv and argv.index("--seed") + 1 < len(argv):
        seed = argv[argv.index("--seed") + 1]
    return {
        "run_id": str(meta.get("run_id") or Path(art["path"]).parent.name),
        "task": Path(ds).name if ds else "fineweb",
        "mechanism": str(cfg.get("attention_type") or "?"),
        "seed": seed,
        "target_flops": (data.get("budget") or {}).get("target_flops"),
        "train_ce_final": res.get("train_ce_final"),
        "val_ce_final": res.get("val_ce_final"),
        "minutes": float(res.get("measured_time_s") or 0.0) / 60.0,
        "tainted": bool(art["tainted"]),
    }


def _report_eval_rows(art: dict[str, Any]) -> list[dict[str, Any]]:
    """One roll-up row per (eval artifact, task): the artifact's held-out EM
    mean is the ci-v2 observation unit (one per trained checkpoint)."""
    data = art["data"]
    ckpt = (data.get("meta") or {}).get("checkpoint") or {}
    lineage = ckpt.get("lineage") or {}
    rows: list[dict[str, Any]] = []
    for task, rec in (data.get("tasks") or {}).items():
        if not isinstance(rec, dict):
            continue
        em = ((rec.get("exact_match") or {}).get("greedy") or {}).get("held_out") or {}
        prior = (rec.get("answer_prior") or {}).get("held_out") or {}
        rows.append({
            "run_id": str(lineage.get("run_id") or "?"),
            "step": ckpt.get("step"),
            "task": str(task),
            "mechanism": str(ckpt.get("attention_type") or "?"),
            "target_flops": (ckpt.get("budget") or {}).get("target_flops"),
            "em_held_out": em.get("mean"),
            "answer_prior": prior.get("mean"),
            "tainted": bool(art["tainted"]),
        })
    return rows


def _report_fmt(v: Any, spec: str = ".4g") -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return format(v, spec)
    return str(v)


def _report_mean_sd(vals: list[float]) -> str:
    if not vals:
        return "-"
    m = statistics.mean(vals)
    if len(vals) == 1:
        return f"{m:.4g}"
    return f"{m:.4g} ± {statistics.stdev(vals):.2g}"


@app.command("report")
def report(
    artifacts: Annotated[
        list[Path] | None, typer.Option(help="Artifact root(s) to scan (repeatable)")
    ] = None,
    out: Annotated[Path, typer.Option(help="Output root for the roll-up")] = Path("artifacts/reports"),
    run_id: Annotated[str | None, typer.Option(help="Report id (default: UTC timestamp)")] = None,
) -> None:
    """Roll-up report across an artifacts tree (bead yhy): the cross-run
    aggregation that per-run run.md files cannot give. Train runs and eval
    evidence are grouped into campaign-style arm tables - mean ± sd over
    CLEAN, lineage-deduped observations (the ci-v2 unit: one per trained
    checkpoint) - with tainted evidence counted but excluded from the stats.
    Replaces the hand-written campaign.md arm tables; report.md +
    report.json land under artifacts/reports/<id>/ with provenance."""
    roots = [Path(p) for p in (artifacts or [Path("artifacts")])]
    index = _adj_collect_artifacts(roots)
    counts: dict[str, dict[str, int]] = {}
    for a in index:
        c = counts.setdefault(a["schema"], {"total": 0, "tainted": 0})
        c["total"] += 1
        c["tainted"] += int(a["tainted"])
    console.print(
        f"[bold cyan]report[/bold cyan] {len(index)} artifact(s) indexed from {[str(r) for r in roots]}"
    )

    # ---- train runs ----
    train_rows = sorted(
        (_report_train_row(a) for a in index if a["schema"] == "train"),
        key=lambda r: (r["task"], r["mechanism"], r["target_flops"] or 0.0, r["run_id"]),
    )
    train_groups: list[dict[str, Any]] = []
    seen_tg: dict[tuple[str, str, Any], dict[str, Any]] = {}
    for r in train_rows:
        g = seen_tg.setdefault(
            (r["task"], r["mechanism"], r["target_flops"]),
            {"task": r["task"], "mechanism": r["mechanism"], "target_flops": r["target_flops"],
             "n": 0, "n_tainted": 0, "ce": [], "minutes": 0.0},
        )
        g["n"] += 1
        g["minutes"] += r["minutes"]
        if r["tainted"]:
            g["n_tainted"] += 1
        elif isinstance(r["train_ce_final"], (int, float)):
            g["ce"].append(float(r["train_ce_final"]))
    # None-safe ordering: legacy artifacts may lack budget.target_flops
    def _group_order(k: tuple[str, str, Any]) -> tuple[str, str, float]:
        return (k[0], k[1], k[2] if k[2] is not None else -1.0)

    for key in sorted(seen_tg, key=_group_order):
        g = seen_tg[key]
        g["train_ce_final"] = _report_mean_sd(g.pop("ce"))
        train_groups.append(g)

    # ---- eval evidence (clean, lineage-deduped: the ci-v2 observation set) ----
    evals_all = [a for a in index if a["schema"] == "evaltasks"]
    evals_clean = _adj_dedupe_evaltasks([a for a in evals_all if not a["tainted"]])
    eval_rows = sorted(
        (row for a in evals_clean for row in _report_eval_rows(a)),
        key=lambda r: (r["task"], r["mechanism"], r["target_flops"] or 0.0, r["run_id"]),
    )
    n_eval_tainted = sum(1 for a in evals_all if a["tainted"])
    eval_groups: list[dict[str, Any]] = []
    seen_eg: dict[tuple[str, str, Any], dict[str, Any]] = {}
    for r in eval_rows:
        g = seen_eg.setdefault(
            (r["task"], r["mechanism"], r["target_flops"]),
            {"task": r["task"], "mechanism": r["mechanism"], "target_flops": r["target_flops"],
             "n_models": 0, "em": [], "prior": []},
        )
        g["n_models"] += 1
        if isinstance(r["em_held_out"], (int, float)):
            g["em"].append(float(r["em_held_out"]))
        if isinstance(r["answer_prior"], (int, float)):
            g["prior"].append(float(r["answer_prior"]))
    for key in sorted(seen_eg, key=_group_order):
        g = seen_eg[key]
        g["em_held_out"] = _report_mean_sd(g.pop("em"))
        priors = g.pop("prior")
        g["answer_prior"] = f"{statistics.mean(priors):.4g}" if priors else "-"
        eval_groups.append(g)

    # ---- render ----
    overview = Table(title="artifact overview", box=box.SIMPLE_HEAVY)
    for col in ("schema", "total", "tainted"):
        overview.add_column(col)
    for schema in sorted(counts):
        c = counts[schema]
        overview.add_row(schema, str(c["total"]), str(c["tainted"]))
    console.print(overview)

    tg_table = Table(title="train runs by arm (CE over clean runs)", box=box.SIMPLE_HEAVY)
    for col in ("task", "mechanism", "FLOPs", "n", "tainted", "train CE (mean ± sd)", "total min"):
        tg_table.add_column(col)
    for g in train_groups:
        tg_table.add_row(
            g["task"], g["mechanism"], _report_fmt(g["target_flops"], ".3g"), str(g["n"]),
            str(g["n_tainted"]), g["train_ce_final"], f"{g['minutes']:.0f}",
        )
    console.print(tg_table)

    eg_table = Table(
        title="eval evidence by arm (held-out EM over clean deduped checkpoints)", box=box.SIMPLE_HEAVY
    )
    for col in ("task", "mechanism", "FLOPs", "n models", "EM held-out (mean ± sd)", "answer prior"):
        eg_table.add_column(col)
    for g in eval_groups:
        eg_table.add_row(
            g["task"], g["mechanism"], _report_fmt(g["target_flops"], ".3g"),
            str(g["n_models"]), g["em_held_out"], g["answer_prior"],
        )
    console.print(eg_table)
    if n_eval_tainted:
        console.print(f"[yellow]{n_eval_tainted} tainted eval artifact(s) excluded from arm stats[/yellow]")

    # ---- persist ----
    resolved = run_id or time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_dir = out / resolved
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "report",
        "schema_version": "mgr.report.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": _get_git_info(),
        "roots": [str(r) for r in roots],
        "counts": counts,
        "train_runs": train_rows,
        "train_groups": train_groups,
        "eval_rows": eval_rows,
        "eval_groups": eval_groups,
        "eval_tainted_excluded": n_eval_tainted,
    }
    (run_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md = [
        f"# Artifact roll-up — {payload['generated_at']}",
        "",
        f"- roots: {payload['roots']}",
        "- counts: " + ", ".join(f"{k}: {v['total']} ({v['tainted']} tainted)" for k, v in sorted(counts.items())),
        "",
        "## Train runs by arm",
        "",
        "| task | mechanism | FLOPs | n | tainted | train CE (mean ± sd) | total min |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in train_groups:
        md.append(
            f"| {g['task']} | {g['mechanism']} | {_report_fmt(g['target_flops'], '.3g')} | {g['n']} "
            f"| {g['n_tainted']} | {g['train_ce_final']} | {g['minutes']:.0f} |"
        )
    md += [
        "",
        "## Eval evidence by arm (held-out EM; clean, lineage-deduped — ci-v2 observation units)",
        "",
        "| task | mechanism | FLOPs | n models | EM held-out (mean ± sd) | answer prior |",
        "|---|---|---|---|---|---|",
    ]
    for g in eval_groups:
        md.append(
            f"| {g['task']} | {g['mechanism']} | {_report_fmt(g['target_flops'], '.3g')} | {g['n_models']} "
            f"| {g['em_held_out']} | {g['answer_prior']} |"
        )
    if n_eval_tainted:
        md.append(f"\n{n_eval_tainted} tainted eval artifact(s) excluded from arm stats.")
    md.append("")
    (run_dir / "report.md").write_text("\n".join(md))
    console.print(Panel(f"report → [bold]{run_dir}[/bold]", border_style="blue"))


def _status_age(ts: float, now: float) -> str:
    mins = max(0.0, (now - ts) / 60.0)
    if mins < 90:
        return f"{mins:.0f}m"
    if mins < 48 * 60:
        return f"{mins / 60:.1f}h"
    return f"{mins / 1440:.1f}d"


@app.command("status")
def status(
    artifacts_dir: Annotated[Path, typer.Option(help="Artifacts root")] = Path("artifacts"),
    json_out: Annotated[bool, typer.Option("--json", help="Machine-readable payload only")] = False,
    write_index: Annotated[
        bool, typer.Option("--write-index", help="Persist the payload as artifacts/index.json (+ index.md)")
    ] = False,
) -> None:
    """Project state in one call (bead 2vs): what exists, how old it is, and
    what is stale - so sessions stop spelunking artifacts/. Latest artifact
    per kind with ages; per-mechanism certificate freshness (a cert older
    than its mechanism's source file is STALE); which registered hypotheses
    the engine would rule on TODAY vs refuse (and why). --json is the agent
    schema; --write-index persists the same payload as the durable manifest."""
    now = time.time()
    repo_root = Path(__file__).resolve().parent

    # ---- latest artifact per top-level kind ----
    kinds: dict[str, dict[str, Any]] = {}
    if artifacts_dir.exists():
        for kind_dir in sorted(p for p in artifacts_dir.iterdir() if p.is_dir()):
            newest: tuple[float, Path] | None = None
            count = 0
            for s in kind_dir.rglob("summary.json"):
                count += 1
                mt = s.stat().st_mtime
                if newest is None or mt > newest[0]:
                    newest = (mt, s)
            if newest is None:
                continue
            kinds[kind_dir.name] = {
                "count": count,
                "newest_path": str(newest[1].parent),
                "newest_age": _status_age(newest[0], now),
                "newest_mtime": newest[0],
            }

    # ---- certificate freshness per mechanism ----
    index = _adj_collect_artifacts([artifacts_dir])
    cert_state: dict[str, dict[str, Any]] = {}
    for art in index:
        if art["schema"] != "certify":
            continue
        mt = Path(art["path"]).stat().st_mtime
        for c in art["data"].get("checks") or []:
            mech = c.get("mechanism") if isinstance(c, dict) else None
            if not isinstance(mech, str):
                continue
            cur = cert_state.get(mech)
            if cur is None or mt > cur["cert_mtime"]:
                cert_state[mech] = {"cert_mtime": mt, "cert_path": art["path"]}
    certs: list[dict[str, Any]] = []
    for mech in sorted(cert_state):
        st = cert_state[mech]
        mech_file = repo_root / "nanochat" / f"{mech}_attention_torch.py"
        stale: bool | None = None
        if mech_file.exists():
            stale = mech_file.stat().st_mtime > st["cert_mtime"]
        certs.append({
            "mechanism": mech,
            "cert_age": _status_age(st["cert_mtime"], now),
            "stale": stale,  # null = no per-mechanism source file to compare
            "cert_path": st["cert_path"],
        })

    # ---- what would the engine rule on today? ----
    reg_data, _load_errors = _load_hypothesis_registry(_hypotheses_registry_path())
    entries = [h for h in (reg_data or {}).get("hypotheses", []) if isinstance(h, dict)]
    adjudicable: list[dict[str, Any]] = []
    blocked_reasons: dict[str, int] = {}
    for h in entries:
        v = _adjudicate_hypothesis(h, index)
        if v["verdict"] == "blocked":
            blocked_reasons[v["reason_code"]] = blocked_reasons.get(v["reason_code"], 0) + 1
            adjudicable.append({"id": str(v["id"]), "state": "blocked", "reason": v["reason_code"]})
        else:
            adjudicable.append({"id": str(v["id"]), "state": "would_rule", "reason": v["verdict"]})
    ledger_counts: dict[str, int] = {}
    for h in entries:
        st_h = str(h.get("status", "?"))
        ledger_counts[st_h] = ledger_counts.get(st_h, 0) + 1

    payload: dict[str, Any] = {
        "kind": "status",
        "schema_version": "mgr.status.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "git": _get_git_info(),
        "artifact_kinds": {k: {kk: vv for kk, vv in v.items() if kk != "newest_mtime"} for k, v in kinds.items()},
        "evidence_pool": {
            "indexed": len(index),
            "tainted": sum(1 for a in index if a["tainted"]),
            "by_schema": {
                s: sum(1 for a in index if a["schema"] == s)
                for s in sorted({a["schema"] for a in index})
            },
        },
        "certificates": certs,
        "ledger": ledger_counts,
        "engine_today": {
            "would_rule": sum(1 for a in adjudicable if a["state"] == "would_rule"),
            "blocked": sum(1 for a in adjudicable if a["state"] == "blocked"),
            "blocked_reasons": blocked_reasons,
            "hypotheses": adjudicable,
        },
    }

    if write_index:
        (artifacts_dir / "index.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        md_lines = [
            f"# Artifacts index — {payload['generated_at']}",
            "",
            "| kind | artifacts | newest | age |",
            "|---|---|---|---|",
        ]
        for k in sorted(kinds):
            v = kinds[k]
            md_lines.append(f"| {k} | {v['count']} | {v['newest_path']} | {v['newest_age']} |")
        md_lines += ["", f"Evidence pool: {payload['evidence_pool']['indexed']} indexed, "
                         f"{payload['evidence_pool']['tainted']} tainted.", ""]
        (artifacts_dir / "index.md").write_text("\n".join(md_lines))

    if json_out:
        console.print_json(json.dumps(payload))
        return

    kt = Table(title="artifacts", box=box.SIMPLE_HEAVY)
    for col in ("kind", "n", "newest", "age"):
        kt.add_column(col)
    for k in sorted(kinds):
        v = kinds[k]
        kt.add_row(k, str(v["count"]), v["newest_path"], v["newest_age"])
    console.print(kt)

    if certs:
        ct = Table(title="certificates (stale = cert predates mechanism source)", box=box.SIMPLE_HEAVY)
        for col in ("mechanism", "age", "fresh?"):
            ct.add_column(col)
        for c in certs:
            mark = "?" if c["stale"] is None else ("[red]STALE[/red]" if c["stale"] else "[green]fresh[/green]")
            ct.add_row(c["mechanism"], c["cert_age"], mark)
        console.print(ct)

    eng = payload["engine_today"]
    reasons = ", ".join(f"{k}: {n}" for k, n in sorted(blocked_reasons.items())) or "-"
    console.print(
        Panel(
            f"ledger: {' · '.join(f'{k}: {n}' for k, n in sorted(ledger_counts.items()))}\n"
            f"evidence pool: {payload['evidence_pool']['indexed']} artifacts "
            f"({payload['evidence_pool']['tainted']} tainted)\n"
            f"engine today: would rule on [bold]{eng['would_rule']}[/bold], "
            f"refuses [bold]{eng['blocked']}[/bold] ({reasons})"
            + ("\nindex written → artifacts/index.json" if write_index else ""),
            border_style="blue",
        )
    )


@app.command("adjudicate")
def adjudicate(
    hypothesis: Annotated[list[str] | None, typer.Option("--hypothesis", "-H", help="Hypothesis id (repeatable)")] = None,
    adjudicate_all: Annotated[bool, typer.Option("--all", help="Adjudicate every operationalized hypothesis")] = False,
    artifacts: Annotated[list[Path] | None, typer.Option(help="Artifact root(s) to scan (repeatable)")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Report verdicts without touching the registry")] = False,
    artifacts_dir: Annotated[Path, typer.Option(help="Output root for the adjudication report")] = Path("artifacts"),
    run_id: Annotated[str | None, typer.Option(help="Report run id (default: date)")] = None,
) -> None:
    """Adjudicate registry hypotheses against artifact evidence (bead hij.2).

    Verdicts are deterministic (fixed bootstrap seed) and stamped with the
    statistical policy version. The engine REFUSES weak evidence: missing
    artifacts, low seed counts, mismatched budgets, or tainted provenance
    yield BLOCKED with a machine-readable reason - never a soft verdict.
    Refusals are report-only; only supported/refuted/inconclusive verdicts
    append to the registry ledger (append-only, validated, rolled back on
    any validation failure).
    """
    repo_root = Path(__file__).resolve().parent
    data = _hypotheses_load_or_exit()
    entries = [h for h in data.get("hypotheses", []) if isinstance(h, dict)]
    if hypothesis:
        wanted = set(hypothesis)
        unknown = wanted - {h.get("id") for h in entries}
        if unknown:
            console.print(f"[bold red]Unknown hypothesis id(s): {sorted(unknown)}[/bold red]")
            raise typer.Exit(code=2)
        entries = [h for h in entries if h.get("id") in wanted]
    elif not adjudicate_all:
        console.print("[bold red]Provide --hypothesis ID (repeatable) or --all.[/bold red]")
        raise typer.Exit(code=2)

    roots = [Path(p) for p in (artifacts or [Path("artifacts")])]
    index = _adj_collect_artifacts(roots)
    console.print(
        f"[bold cyan]adjudicate[/bold cyan] {len(entries)} hypothesis(es) · "
        f"{len(index)} artifact(s) indexed from {[str(r) for r in roots]} · policy {_ADJ_POLICY_VERSION}"
    )

    today = time.strftime("%Y-%m-%d")
    verdicts = [_adjudicate_hypothesis(h, index) for h in entries]

    # ---- ledger-level FDR (ci-v4): BH q-values across this run's family ----
    # q-values are family-relative: they live in run reports, never the ledger.
    family = [v for v in verdicts if v["verdict"] != "blocked" and v.get("p_value") is not None]
    if family:
        qmap = _adj_bh_qvalues({str(v["id"]): float(v["p_value"]) for v in family})
        for v in family:
            v["q_value"] = qmap[str(v["id"])]
    n_supported = sum(1 for v in verdicts if v["verdict"] == "supported")
    fdr_survivors = sorted(
        str(v["id"]) for v in family if v["verdict"] == "supported" and v["q_value"] <= _ADJ_FDR_Q
    )
    # flag families aggregating heterogeneous statistical machinery: the
    # ledger's latest recorded verdicts may predate this policy version
    ledger_policies = sorted(
        {
            str((h.get("verdict_history") or [{}])[-1].get("policy_version") or "unstamped")
            for h in data.get("hypotheses", [])
            if isinstance(h, dict) and h.get("verdict_history")
        }
    )

    # ---- ledger append (real verdicts only; refusals are report-only) ----
    registry_path = _hypotheses_registry_path()
    applied = 0
    if not dry_run:
        original = registry_path.read_text(encoding="utf-8")
        try:
            for v in verdicts:
                if v["verdict"] == "blocked":
                    continue
                arms_txt = json.dumps(v["arms"], sort_keys=True)
                entry_line = (
                    "      - {date: "
                    + json.dumps(today)
                    + ", verdict: "
                    + v["verdict"]
                    + ", artifacts: "
                    + json.dumps(v["artifacts"])
                    + ", adjudicator: "
                    + json.dumps(f"engine:{_ADJ_POLICY_VERSION}")
                    + ", policy_version: "
                    + json.dumps(_ADJ_POLICY_VERSION)
                    + ", arms: "
                    + arms_txt
                    + "}"
                )
                _registry_append_verdict(registry_path, str(v["id"]), entry_line, v["verdict"])
                applied += 1
            if applied:
                new_data, load_errors = _load_hypothesis_registry(registry_path)
                parent = _load_parent_hypothesis_registry(repo_root)
                errors, _w, _s = _validate_hypothesis_registry(new_data, load_errors, repo_root, parent=parent)
                if errors:
                    raise RuntimeError("post-adjudication registry validation failed: " + "; ".join(errors[:5]))
        except Exception as exc:  # roll back: never leave the ledger invalid
            registry_path.write_text(original, encoding="utf-8")
            console.print(f"[bold red]ledger update rolled back: {exc}[/bold red]")
            raise typer.Exit(code=1) from exc

    # ---- report ----
    resolved_run_id = run_id or today
    run_dir = artifacts_dir / "adjudications" / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_version": _ADJ_POLICY_VERSION,
        "date": today,
        "verdicts": verdicts,
        "fdr": {
            "q_level": _ADJ_FDR_Q,
            "family_size": len(family),
            "family_is_full_ledger": bool(adjudicate_all),
            "supported": n_supported,
            "supported_fdr_survivors": fdr_survivors,
            "ledger_policy_versions": ledger_policies,
        },
    }
    (run_dir / "verdicts.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    style = {"supported": "green", "refuted": "red", "inconclusive": "magenta", "blocked": "dim"}
    order = {"refuted": 0, "supported": 1, "inconclusive": 2, "blocked": 3}
    by_id = {h.get("id"): h for h in entries}
    rows = sorted(
        verdicts, key=lambda v: (order[v["verdict"]], ",".join(by_id[v["id"]].get("mechanisms") or []), str(v["id"]))
    )
    table = Table(title=f"adjudication — {today} (policy {_ADJ_POLICY_VERSION})", box=box.SIMPLE_HEAVY)
    table.add_column("hypothesis", style="bold")
    table.add_column("verdict")
    table.add_column("q", justify="right")
    table.add_column("detail")
    md_rows = []
    for v in rows:
        st = v["verdict"]
        st_label = st + ("-underpowered" if v.get("underpowered") else "")
        q_txt = f"{v['q_value']:.3g}" if v.get("q_value") is not None else ""
        if st == "blocked":
            detail = f"{v['reason_code']}" + (f" [{v.get('mechanism')}]" if v.get("mechanism") else "")
        else:
            detail = "; ".join(
                f"{m}: effect={a['effect']:.4g} ci95=[{a['ci95'][0]:.4g},{a['ci95'][1]:.4g}] "
                f"(n={a['n_candidate']}/{a['n_baseline']})"
                + (f" power={a['power']:.0%}" if a.get("power") is not None else "")
                + (
                    f" UNDERPOWERED(need n≈{a['n_for_80pct']})"
                    if a.get("underpowered")
                    else ""
                )
                + (
                    # ci-v6: a refutation's decisiveness is its CI-half-widths
                    # past the threshold, not the (irrelevant) confirm-power
                    f" refutation_margin={a['refutation_margin']:.1f}×"
                    if a.get("refutation_margin") is not None
                    else ""
                )
                + (
                    # the EM gate records baseline_mean/floor in metric units;
                    # the slope gate (qtdq) records em-vs-prior instead
                    (
                        f" FLOOR(em {a['baseline_em_mean']:.4g} <= prior {a['em_floor']:.4g}, slopes vacuous)"
                        if a.get("floor_source") == "slope_em_floor"
                        else f" FLOOR(base {a['baseline_mean']:.4g} <= {a['baseline_floor']:.4g}, no power)"
                    )
                    if a.get("floor_effect")
                    else ""
                )
                for m, a in v["arms"].items()
            )
        table.add_row(str(v["id"]), f"[{style[st]}]{st_label}[/{style[st]}]", q_txt, detail)
        md_rows.append(f"| {v['id']} | {st_label} | {q_txt} | {detail} |")
    console.print(table)
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    summary_line = " · ".join(f"{k}: {n}" for k, n in sorted(counts.items()))
    # the mandatory two-number headline (ci-v4): the gap IS the honesty
    fdr_line = (
        f"{n_supported} supported, of which {len(fdr_survivors)} survive FDR at q={_ADJ_FDR_Q:g} "
        f"(family: {len(family)} adjudicated"
        + ("" if adjudicate_all else " - PARTIAL run, not the whole ledger")
        + ")"
    )
    policy_note = (
        f"ledger note: latest recorded verdicts span policies {ledger_policies}; q-values here "
        f"are computed fresh under {_ADJ_POLICY_VERSION} for this run's family only."
        if len(ledger_policies) > 1
        else ""
    )
    report_md = (
        f"# Adjudication — {today}\n\n"
        f"- policy: `{_ADJ_POLICY_VERSION}`\n"
        f"- artifacts indexed: {len(index)} from {[str(r) for r in roots]}\n"
        f"- ledger entries appended: {applied}{' (dry run)' if dry_run else ''}\n"
        f"- verdicts: {summary_line}\n"
        f"- **{fdr_line}**\n"
        + (f"- {policy_note}\n" if policy_note else "")
        + "\n| hypothesis | verdict | q | detail |\n|---|---|---|---|\n" + "\n".join(md_rows) + "\n\n"
        "BLOCKED rows are refusals, not adjudications: the engine declines to rule on weak, "
        "mismatched, or tainted evidence. UNDERPOWERED verdicts cleared their threshold at a test "
        "with under 50% power to detect the registered effect - an asterisk, not a clean verdict. "
        "See `verdicts.json` for machine-readable reasons, p-values, and q-values.\n"
    )
    (run_dir / "report.md").write_text(report_md)
    if policy_note:
        console.print(f"[yellow]{policy_note}[/yellow]")
    console.print(
        Panel(
            f"[bold]{summary_line}[/bold]\n[bold cyan]{fdr_line}[/bold cyan]\n"
            f"ledger appends: {applied}{' (dry run)' if dry_run else ''} · → {run_dir}",
            border_style="blue",
        )
    )


@app.command("visualize")
def visualize(
    mode: Annotated[
        str,
        typer.Argument(help="'state' (hi3: model-state visuals) or 'entropy' (7ow: per-head entropy/diversity)"),
    ] = "state",
    attention: Annotated[str, typer.Option("--attention", help="Attention type for a fresh probe (state mode)")] = "standard",
    baseline: Annotated[str, typer.Option("--baseline", help="Baseline mechanism (entropy mode)")] = "standard",
    feature: Annotated[str, typer.Option("--feature", help="Math-feature mechanism (entropy mode)")] = "tropical",
    checkpoint: Annotated[
        Path | None, typer.Option("--checkpoint", help="Load a trained checkpoint dir instead of a fresh probe (state mode)")
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Output dir (default under artifacts/vis/)")] = None,
    device: Annotated[str, typer.Option("--device", help="cpu | cuda")] = "cpu",
    seed: Annotated[int, typer.Option("--seed", "-s", help="Seeds model init + sample batch")] = 0,
    seq_len: Annotated[int, typer.Option("--seq-len", help="Sample-batch sequence length")] = 64,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Sample-batch size")] = 4,
) -> None:
    """Model-state visualizations (beads hi3, 7ow).

    'state' renders >=3 visuals (per-head attention-entropy heatmap, softmax
    attention maps, tropical route margins) for a fresh probe or a checkpoint;
    'entropy' tabulates per-head entropy + cross-head route diversity for a
    baseline vs a math-feature mechanism. Thin wrapper over `python -m
    nanochat.viz`; outputs under artifacts/vis/. See docs/visualizations.md.
    """
    from nanochat import viz

    if mode == "state":
        argv = ["state", "--attention", attention, "--device", device, "--seed", str(seed),
                "--seq-len", str(seq_len), "--batch-size", str(batch_size)]
        if checkpoint is not None:
            argv += ["--checkpoint", str(checkpoint)]
    elif mode == "entropy":
        argv = ["entropy", "--baseline", baseline, "--feature", feature, "--device", device,
                "--seed", str(seed), "--seq-len", str(seq_len), "--batch-size", str(batch_size)]
    else:
        console.print(f"[bold red]Unknown mode {mode!r}. Use 'state' or 'entropy'.[/bold red]")
        raise typer.Exit(code=2)
    if out is not None:
        argv += ["--out", str(out)]
    raise typer.Exit(code=viz.main(argv))


@app.command("profile")
def profile(
    attention: Annotated[str, typer.Option("--attention", help="Attention type to profile")] = "standard",
    compare_flex: Annotated[bool, typer.Option("--compare-flex", help="standard only: FlexAttention on vs off")] = False,
    forward_only: Annotated[bool, typer.Option("--forward-only", help="Profile forward only (default: fwd+bwd)")] = False,
    device: Annotated[str, typer.Option("--device", help="cpu | cuda")] = "cpu",
    steps: Annotated[int, typer.Option("--steps", help="Profiled steps (excludes warmup)")] = 5,
    out: Annotated[Path | None, typer.Option("--out", help="Output dir (default artifacts/profiles/<attention>)")] = None,
) -> None:
    """Profile a mechanism's attention/optimizer hot path (bead b1l).

    torch.profiler kernel/memory breakdown + a Chrome trace, with optional
    FlexAttention on-vs-off comparison (standard). Thin wrapper over `python -m
    nanochat.profiling bench`; outputs under artifacts/profiles/. See
    docs/profiling.md.
    """
    from nanochat import profiling

    argv = ["bench", "--attention", attention, "--device", device, "--steps", str(steps)]
    if compare_flex:
        argv.append("--compare-flex")
    if forward_only:
        argv.append("--forward-only")
    if out is not None:
        argv += ["--out", str(out)]
    raise typer.Exit(code=profiling.main(argv))


if __name__ == "__main__":
    app()
