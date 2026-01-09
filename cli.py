#!/usr/bin/env python3
"""
Model Guided Research CLI - Run experimental mathematical models for ML research
"""

import importlib
import json
import math
import platform
import statistics
import shlex
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import typer
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
        float(v)
        for v in values
        if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(float(v))
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
        if isinstance(vec, list)
        and vec
        and all(isinstance(x, int | float) and not isinstance(x, bool) for x in vec)
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
        table.add_row(
            name,
            info["description"],
            info["module"]
        )

    console.print(table)
    console.print("\n[dim]Run a demo with:[/dim] [bold green]mgr run <demo-name>[/bold green]")
    console.print("[dim]Get info about a demo:[/dim] [bold green]mgr info <demo-name>[/bold green]")


@app.command()
def run(
    demo_name: Annotated[str, typer.Argument(
        help="Name of the demo to run",
        autocompletion=lambda: DEMOS.keys()  # type: ignore[call-arg]
    )],
    config_file: Annotated[Path | None, typer.Option(
        "--config", "-c",
        help="Path to JSON config file (see config.example.json)"
    )] = None,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Show verbose output"
    )] = False,
    verbose_level: Annotated[int | None, typer.Option(
        "--verbose-level",
        min=0, max=3,
        help="Verbosity level: 0=silent, 1=normal, 2=detailed, 3=debug"
    )] = None,
    seed: Annotated[int | None, typer.Option(
        "--seed", "-s",
        help="Random seed for reproducibility"
    )] = None,
    max_iterations: Annotated[int | None, typer.Option(
        "--max-iterations",
        min=1,
        help="Override ProjectConfig.max_iterations (for demos that respect it)"
    )] = None,
    no_rich: Annotated[bool, typer.Option(
        "--no-rich",
        help="Disable rich formatting for plain text output"
    )] = False,
    debug: Annotated[bool, typer.Option(
        "--debug",
        help="Enable debug mode with numerical checking"
    )] = False,
    ultra_packed: Annotated[bool, typer.Option(
        "--ultra-packed",
        help="Use packed bit-trie implementation in ultrametric demo (and set ULTRA_PACKED for tests)"
    )] = False,
    tropical_cert: Annotated[bool, typer.Option(
        "--tropical-cert",
        help="Compute a tropical attention robustness margin certificate"
    )] = False,
    simplicial_hodge: Annotated[bool, typer.Option(
        "--simplicial-hodge",
        help="Demonstrate Hodge-based readout coefficients on a tiny graph"
    )] = False,
    simplicial_signed: Annotated[bool, typer.Option(
        "--simplicial-signed",
        help="Demonstrate signed (orientation-aware) diffusion vs unsigned"
    )] = False,
    rev_cayley: Annotated[bool, typer.Option(
        "--rev-cayley",
        help="Demonstrate Cayley orthogonal property check (skew → orthogonal)"
    )] = False,
    rev_cayley_o1: Annotated[bool, typer.Option(
        "--rev-cayley-o1/--no-rev-cayley-o1",
        help="Use O(1)-memory custom gradient for Cayley step (default on)"
    )] = True,
    rev_cayley_iters: Annotated[int, typer.Option(
        "--rev-cayley-iters",
        help="Cayley fixed-point iterations (trade compute for accuracy)",
        min=1
    )] = 1,
    rev_symplectic: Annotated[bool, typer.Option(
        "--rev-symplectic",
        help="Demonstrate symplectic Cayley property check (S^T J S ≈ J)"
    )] = False,
    rev_inv_iters: Annotated[int, typer.Option(
        "--rev-inv-iters",
        help="Inverse fixed-point iteration count for Cayley inverse",
        min=1
    )] = 1,
    rev_pareto: Annotated[bool, typer.Option(
        "--rev-pareto",
        help="Run a small Cayley-iterations Pareto sweep (time vs memory)"
    )] = False,
    rev_symp_hybrid: Annotated[bool, typer.Option(
        "--rev-symplectic-hybrid",
        help="Enable a symplectic leapfrog step inside coupling (hybrid)"
    )] = False,
    rev_givens: Annotated[bool, typer.Option(
        "--rev-givens",
        help="Use strict Givens mixing (exact inverse; det=1)"
    )] = False,
    rev_generating: Annotated[bool, typer.Option(
        "--rev-generating",
        help="Enable generating-function symplectic step (exact inverse)"
    )] = False,
    rev_gen_vjp: Annotated[bool, typer.Option(
        "--rev-gen-vjp",
        help="Use custom VJP for generating step (O(1) grads; ignores ∂/∂(a,b,c))"
    )] = False,
    gauge_structured: Annotated[bool, typer.Option(
        "--gauge-structured",
        help="Enable structured SO/SPD/Sp channel blocks in matrix-gauge demo"
    )] = False,
    gauge_bch_compact: Annotated[bool, typer.Option(
        "--gauge-bch-compact",
        help="Print only compact BCH summary table (skip heatmap)"
    )] = False,
    gauge_alt_struct: Annotated[bool, typer.Option(
        "--gauge-alt-struct",
        help="Alternate structured/unstructured on odd blocks in matrix-gauge demo"
    )] = False,
    export_json: Annotated[Path | None, typer.Option(
        "--export-json",
        help="Write a JSON artifact with any computed certificates/readouts"
    )] = None,
    artifacts_dir: Annotated[Path | None, typer.Option(
        "--artifacts-dir",
        help="Write run artifacts under this directory using the standard layout (see artifacts/README.md)."
    )] = None,
    run_id: Annotated[str | None, typer.Option(
        "--run-id",
        help="Run identifier (directory name) when writing artifacts. Defaults to YYYYMMDD_HHMMSS."
    )] = None,
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
        f"[bold cyan]{demo_info['description']}[/bold cyan]\n"
        f"[dim]Module: {demo_info['module']}.py[/dim]",
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

        module = importlib.import_module(demo_info['module'])

        # Get the demo function
        func_name = demo_info['func']
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
                margin = float(getattr(attn, 'last_min_margin', 0.0))
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

            if (demo_name == "reversible") and (rev_cayley or rev_symplectic or rev_pareto or rev_symp_hybrid or (rev_inv_iters != 1)):
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
- Generated: {meta['generated_at']}
- Commit: {meta['git']['commit_full']}{' (dirty)' if meta['git']['dirty'] else ' (clean)'}

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
        autocompletion=lambda: DEMOS.keys()  # type: ignore[call-arg]
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
                            docstring_lines.append(line.strip().replace('"""', ''))
                            break
                    else:
                        in_docstring = False
                        break
                elif in_docstring:
                    docstring_lines.append(line.rstrip())

            if docstring_lines:
                console.print("\n[bold]Module Documentation:[/bold]")
                console.print(Panel(
                    '\n'.join(docstring_lines),
                    box=box.ROUNDED,
                    padding=(1, 2),
                ))

        except Exception as e:
            if str(e):  # Only show error if it has a message
                console.print(f"[dim]Could not read module documentation: {e}[/dim]")

    console.print(f"\n[dim]Run this demo with:[/dim] [bold green]mgr run {demo_name}[/bold green]")


@app.command()
def config(
    output: Annotated[Path | None, typer.Option(
        "--output", "-o",
        help="Output path for config file"
    )] = None,
    show: Annotated[bool, typer.Option(
        "--show",
        help="Show current configuration"
    )] = False,
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
        "gradient_clip_norm": None
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(example_config, f, indent=2)

    console.print(f"[green]✓ Example config written to {output_path}[/green]")
    console.print(f"\n[dim]Use it with:[/dim] [bold]mgr run <demo> --config {output_path}[/bold]")


@app.command()
def run_all(
    delay: Annotated[int, typer.Option(
        "--delay", "-d",
        help="Delay in seconds between demos"
    )] = 2,
    seed: Annotated[int | None, typer.Option(
        "--seed", "-s",
        help="Random seed for reproducibility"
    )] = None,
    skip_errors: Annotated[bool, typer.Option(
        "--skip-errors/--stop-on-error",
        help="Continue running demos even if one fails"
    )] = True,
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
            module = importlib.import_module(info['module'])
            func_name = info['func']

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
    version: bool = typer.Option(
        False, "--version", "-V",
        help="Show version information"
    ),
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
    ultra_packed: Annotated[bool, typer.Option(
        "--ultra-packed",
        help="Use packed bit-trie implementation for ultrametric tests (sets ULTRA_PACKED=1)"
    )] = False,
    seed: Annotated[int | None, typer.Option(
        "--seed", "-s",
        help="Random seed for reproducibility"
    )] = None,
    export_json: Annotated[Path | None, typer.Option(
        "--export-json",
        help="Write a combined JSON artifact of the practical utility suite"
    )] = None,
    artifacts_dir: Annotated[Path | None, typer.Option(
        "--artifacts-dir",
        help="Write run artifacts under this directory using the standard layout (see artifacts/README.md)."
    )] = None,
    run_id: Annotated[str | None, typer.Option(
        "--run-id",
        help="Run identifier (directory name) when writing artifacts. Defaults to YYYYMMDD_HHMMSS."
    )] = None,
    print_ultra_table: Annotated[bool, typer.Option(
        "--print-ultra-table",
        help="Print ultrametric exponent table"
    )] = False,
    print_trop_table: Annotated[bool, typer.Option(
        "--print-trop-table",
        help="Print tropical Lipschitz table"
    )] = False,
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
            payload.append({
                "approach": r.approach_name,
                "claim": r.claim,
                "baseline": float(r.baseline_metric),
                "proposed": float(r.proposed_metric),
                "improvement": float(r.improvement_ratio),
                "is_better": bool(r.is_better),
                "verdict": r.verdict,
                "details": r.details,
            })
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
- Generated: {summary['meta']['generated_at']}
- Commit: {summary['meta']['git']['commit_full']}{' (dirty)' if summary['meta']['git']['dirty'] else ' (clean)'}

## Command

```bash
{shlex.join(sys.argv)}
```

See `summary.json` for full details.
"""
        _write_artifacts(run_dir, summary=summary, report_md=report_md)
        console.print(f"[dim]Wrote artifacts → {run_dir}[/dim]")

@app.command("bench-fixed-flops")
def bench_fixed_flops(
    attention_types: Annotated[list[str], typer.Option(
        "--attention-type", "-a",
        help="Nanochat attention types to benchmark (repeatable).",
    )] = ("standard", "tropical", "ultrametric", "simplicial", "reversible", "gauge"),
    device: Annotated[str, typer.Option(
        "--device",
        help="Device for nanochat training runs (passed through to nanochat.train).",
    )] = "cpu",
    target_flops: Annotated[float, typer.Option(
        "--target-flops",
        help="Target total FLOPs budget (est) per run.",
        min=1e6,
    )] = 2e9,
    seed: Annotated[int, typer.Option(
        "--seed",
        help="Training seed (same seed used for each attention type).",
    )] = 0,
    score_tail: Annotated[int, typer.Option(
        "--score-tail",
        help="Score is mean of last N losses from nanochat summary.json.",
        min=1,
    )] = 3,
    batch_size: Annotated[int, typer.Option(
        "--batch-size",
        help="Batch size for nanochat training.",
        min=1,
    )] = 8,
    sequence_len: Annotated[int, typer.Option(
        "--sequence-len",
        help="Sequence length for nanochat training.",
        min=8,
    )] = 256,
    n_layer: Annotated[int, typer.Option(
        "--n-layer",
        help="Number of transformer layers.",
        min=1,
    )] = 4,
    n_head: Annotated[int, typer.Option(
        "--n-head",
        help="Number of attention heads.",
        min=1,
    )] = 4,
    n_kv_head: Annotated[int, typer.Option(
        "--n-kv-head",
        help="Number of KV heads (GQA).",
        min=1,
    )] = 4,
    n_embd: Annotated[int, typer.Option(
        "--n-embd",
        help="Embedding dimension.",
        min=16,
    )] = 128,
    optimizer_type: Annotated[str, typer.Option(
        "--optimizer-type",
        help="nanochat optimizer type (passed through).",
    )] = "adamw",
    learning_rate: Annotated[float, typer.Option(
        "--learning-rate",
        help="Base learning rate for nanochat.train.",
        min=1e-8,
    )] = 6e-4,
    warmup_steps: Annotated[int, typer.Option(
        "--warmup-steps",
        help="Warmup steps excluded from throughput measurement.",
        min=0,
    )] = 0,
    log_interval: Annotated[int, typer.Option(
        "--log-interval",
        help="Train logging interval (steps).",
        min=1,
    )] = 1,
    check_numerics: Annotated[bool, typer.Option(
        "--check-numerics",
        help="Enable NaN/Inf watchpoints inside nanochat.train.",
    )] = False,
    compile: Annotated[bool, typer.Option(
        "--compile/--no-compile",
        help="Enable torch.compile in nanochat.train (optional).",
    )] = False,
    auto_download_data: Annotated[bool, typer.Option(
        "--auto-download-data/--no-auto-download-data",
        help="Auto-download minimal dataset shards if missing.",
    )] = True,
    min_parquet_files: Annotated[int, typer.Option(
        "--min-parquet-files",
        help="Minimum number of parquet shards required (>=2 recommended).",
        min=2,
    )] = 2,
    include_demo_certs: Annotated[bool, typer.Option(
        "--include-demo-certs/--no-include-demo-certs",
        help="Also run a few demo certificate runs (math-specific diagnostics) and link them in the suite report.",
    )] = False,
    artifacts_dir: Annotated[Path, typer.Option(
        "--artifacts-dir",
        help="Base directory for artifacts (default: artifacts/).",
    )] = Path("artifacts"),
    run_id: Annotated[str | None, typer.Option(
        "--run-id",
        help="Suite run identifier (directory name). Defaults to YYYYMMDD_HHMMSS.",
    )] = None,
    timeout_s: Annotated[float, typer.Option(
        "--timeout-s",
        help="Per-run timeout (seconds) for subprocess invocations.",
        min=1.0,
    )] = 1800.0,
):
    attention_types = list(attention_types)
    """Benchmark nanochat attention variants under a fixed FLOPs budget.

    Per-run nanochat artifacts:
    - `artifacts/bench/fixed_flops/nanochat/<suite_run_id>/<attention_type>/seed_<seed>/`

    Suite aggregation:
    - `artifacts/bench/fixed_flops/nanochat/<suite_run_id>/summary.json`
    - `artifacts/bench/fixed_flops/nanochat/<suite_run_id>/run.md`
    """
    if not attention_types:
        raise typer.BadParameter("--attention-type must be provided at least once")

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

    def _run_train(attn: str) -> dict[str, Any]:
        run_topic = f"fixed_flops/nanochat/{suite_run_id}/{attn}"
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

        stdout_path = logs_dir / f"nanochat_{attn}.stdout.txt"
        stderr_path = logs_dir / f"nanochat_{attn}.stderr.txt"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

        summary_path = artifacts_dir / "bench" / run_topic / run_id_local / "summary.json"
        status = "ok" if returncode == 0 and summary_path.exists() else ("timeout" if returncode == 124 else "error")

        losses: list[float] = []
        final_loss: float | None = None
        score: float | None = None
        ppl: float | None = None
        tokens_s: float | None = None
        tflops_s: float | None = None
        peak_mem_gb: float | None = None
        if status == "ok":
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            res = payload.get("results", {})
            losses = [float(x) for x in res.get("losses", [])]
            if losses:
                final_loss = float(losses[-1])
                tail = losses[-min(len(losses), int(score_tail)) :]
                score = float(sum(tail) / len(tail))
                ppl = float(math.exp(score))
            if isinstance(res.get("tokens_per_second"), int | float):
                tokens_s = float(res["tokens_per_second"])
            if isinstance(res.get("tflops_per_second_est"), int | float):
                tflops_s = float(res["tflops_per_second_est"])
            if isinstance(res.get("peak_memory_allocated_gb"), int | float):
                peak_mem_gb = float(res["peak_memory_allocated_gb"])

        return {
            "attention_type": attn,
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
            "tokens_per_second": tokens_s,
            "tflops_per_second_est": tflops_s,
            "peak_memory_allocated_gb": peak_mem_gb,
        }

    results: list[dict[str, Any]] = []
    total = len(attention_types)
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
            console.print(Panel(f"[bold]nanochat[/bold] attention_type={attn!r}", box=box.ROUNDED))
            results.append(_run_train(attn))
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
                    "status": "ok" if returncode == 0 and cert_summary.exists() else ("timeout" if returncode == 124 else "error"),
                    "returncode": int(returncode),
                    "command": shlex.join(cmd),
                    "summary_path": str(cert_summary.relative_to(artifacts_dir)) if cert_summary.exists() else None,
                }
            )

    baseline_attn = "standard" if "standard" in attention_types else attention_types[0]
    baseline = next((r for r in results if r["attention_type"] == baseline_attn), None)

    table = Table(title="Fixed-FLOPs nanochat benchmark (train loss)", box=box.ROUNDED)
    table.add_column("attention_type", style="bold")
    table.add_column("status")
    table.add_column("score", justify="right")
    table.add_column("Δ vs baseline", justify="right")
    table.add_column("tokens/s", justify="right")
    table.add_column("TFLOP/s(est)", justify="right")
    table.add_column("peak_mem_gb", justify="right")

    for r in results:
        score = r.get("score")
        delta = None
        if baseline and baseline.get("score") is not None and score is not None:
            base = float(baseline["score"])
            delta = (float(score) - base) / base if base != 0 else None

        table.add_row(
            str(r["attention_type"]),
            str(r["status"]),
            f"{float(score):.6f}" if isinstance(score, int | float) else "n/a",
            f"{float(delta):+.2%}" if isinstance(delta, int | float) else "n/a",
            f"{float(r['tokens_per_second']):,.0f}" if isinstance(r.get("tokens_per_second"), int | float) else "n/a",
            f"{float(r['tflops_per_second_est']):.2f}" if isinstance(r.get("tflops_per_second_est"), int | float) else "n/a",
            f"{float(r['peak_memory_allocated_gb']):.2f}" if isinstance(r.get("peak_memory_allocated_gb"), int | float) else "n/a",
        )

    console.print(table)

    summary = {
        "schema_version": "mgr.bench.fixed_flops.v1",
        "meta": bench_meta,
        "baseline_attention_type": baseline_attn,
        "runs": results,
        "demo_certs": demo_certs,
    }

    ok_runs = [r for r in results if r.get("status") == "ok" and isinstance(r.get("score"), (int, float))]
    ok_runs_sorted = sorted(ok_runs, key=lambda r: float(r["score"]))
    best = ok_runs_sorted[0] if ok_runs_sorted else None

    def _md_row(values: list[str]) -> str:
        return "| " + " | ".join(values) + " |"

    md_lines: list[str] = []
    md_lines.append("# Fixed-FLOPs nanochat benchmark")
    md_lines.append("")
    md_lines.append(f"- Run ID: `{suite_run_id}`")
    md_lines.append(f"- Baseline: `{baseline_attn}`")
    md_lines.append(f"- Device: `{device}`")
    md_lines.append(f"- Target FLOPs/run (est): `{float(target_flops):.3e}`")
    md_lines.append(f"- Seed: `{seed}`")
    md_lines.append("")
    md_lines.append("## Results")
    md_lines.append("")
    md_lines.append(
        _md_row(["attention_type", "status", "score", "Δ vs baseline", "tokens/s", "TFLOP/s(est)", "peak_mem_gb"])
    )
    md_lines.append(_md_row(["---"] * 7))

    for r in results:
        score = r.get("score")
        delta = None
        if baseline and baseline.get("score") is not None and score is not None:
            base = float(baseline["score"])
            delta = (float(score) - base) / base if base != 0 else None
        md_lines.append(
            _md_row(
                [
                    str(r["attention_type"]),
                    str(r["status"]),
                    f"{float(score):.6f}" if isinstance(score, (int, float)) else "n/a",
                    f"{float(delta):+.2%}" if isinstance(delta, (int, float)) else "n/a",
                    f"{float(r['tokens_per_second']):,.0f}" if isinstance(r.get("tokens_per_second"), (int, float)) else "n/a",
                    f"{float(r['tflops_per_second_est']):.2f}" if isinstance(r.get("tflops_per_second_est"), (int, float)) else "n/a",
                    f"{float(r['peak_memory_allocated_gb']):.2f}" if isinstance(r.get("peak_memory_allocated_gb"), (int, float)) else "n/a",
                ]
            )
        )

    md_lines.append("")
    md_lines.append("## Conclusions")
    md_lines.append("")
    if best is None:
        md_lines.append("- No successful runs; see `logs/` for stdout/stderr.")
    else:
        md_lines.append(f"- Best (lowest score): `{best['attention_type']}` score=`{float(best['score']):.6f}`")
        if baseline and baseline.get("score") is not None:
            base = float(baseline["score"])
            md_lines.append(f"- Baseline `{baseline_attn}` score=`{base:.6f}`; best Δ=`{(float(best['score']) - base) / base:+.2%}`")
        better = []
        worse = []
        if baseline and baseline.get("score") is not None:
            base = float(baseline["score"])
            for r in ok_runs_sorted:
                if r["attention_type"] == baseline_attn:
                    continue
                d = (float(r["score"]) - base) / base if base != 0 else 0.0
                (better if d < 0 else worse).append((r["attention_type"], d))
        if better:
            md_lines.append("- Better than baseline: " + ", ".join(f"`{a}` ({d:+.2%})" for a, d in better))
        if worse:
            md_lines.append("- Worse than baseline: " + ", ".join(f"`{a}` ({d:+.2%})" for a, d in worse))
        failed = [r for r in results if r.get("status") != "ok"]
        if failed:
            md_lines.append("- Failures: " + ", ".join(f"`{f['attention_type']}`" for f in failed))

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


@app.command("per-head-metrics")
def per_head_metrics(
    device: Annotated[str, typer.Option(
        "--device",
        help="Device for nanochat training runs (passed through to nanochat.train).",
    )] = "cpu",
    seeds: Annotated[list[int], typer.Option(
        "--seed", "-s",
        help="Training seeds (repeatable).",
    )] = (0, 1, 2),
    target_flops: Annotated[float, typer.Option(
        "--target-flops",
        help="Target total FLOPs budget (est) per run.",
        min=1e6,
    )] = 2e8,
    batch_size: Annotated[int, typer.Option(
        "--batch-size",
        help="Batch size for nanochat training.",
        min=1,
    )] = 8,
    sequence_len: Annotated[int, typer.Option(
        "--sequence-len",
        help="Sequence length for nanochat training.",
        min=8,
    )] = 256,
    n_layer: Annotated[int, typer.Option(
        "--n-layer",
        help="Number of transformer layers.",
        min=1,
    )] = 4,
    n_head: Annotated[int, typer.Option(
        "--n-head",
        help="Number of attention heads.",
        min=1,
    )] = 4,
    n_kv_head: Annotated[int, typer.Option(
        "--n-kv-head",
        help="Number of KV heads (GQA).",
        min=1,
    )] = 4,
    n_embd: Annotated[int, typer.Option(
        "--n-embd",
        help="Embedding dimension.",
        min=16,
    )] = 128,
    optimizer_type: Annotated[str, typer.Option(
        "--optimizer-type",
        help="nanochat optimizer type (passed through).",
    )] = "adamw",
    learning_rate: Annotated[float, typer.Option(
        "--learning-rate",
        help="Base learning rate for nanochat.train.",
        min=1e-8,
    )] = 6e-4,
    warmup_steps: Annotated[int, typer.Option(
        "--warmup-steps",
        help="Warmup steps excluded from throughput measurement.",
        min=0,
    )] = 0,
    log_interval: Annotated[int, typer.Option(
        "--log-interval",
        help="Train logging interval (steps).",
        min=1,
    )] = 1,
    auto_download_data: Annotated[bool, typer.Option(
        "--auto-download-data/--no-auto-download-data",
        help="Auto-download minimal dataset shards if missing.",
    )] = True,
    min_parquet_files: Annotated[int, typer.Option(
        "--min-parquet-files",
        help="Minimum number of parquet shards required (>=2 recommended).",
        min=2,
    )] = 2,
    artifacts_dir: Annotated[Path, typer.Option(
        "--artifacts-dir",
        help="Base directory for artifacts (default: artifacts/).",
    )] = Path("artifacts"),
    run_id: Annotated[str | None, typer.Option(
        "--run-id",
        help="Suite run identifier (directory name). Defaults to YYYYMMDD_HHMMSS.",
    )] = None,
    timeout_s: Annotated[float, typer.Option(
        "--timeout-s",
        help="Per-run timeout (seconds) for subprocess invocations.",
        min=1.0,
    )] = 1800.0,
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
            [(int(r["seed"]), r["attention_entropy_head_mean"]) for r in runs if isinstance(r.get("seed"), int) and isinstance(r.get("attention_entropy_head_mean"), list)]
        )
        gamma_agg = _aggregate_per_head(
            [(int(r["seed"]), r["tropical_gamma_head_mean"]) for r in runs if isinstance(r.get("seed"), int) and isinstance(r.get("tropical_gamma_head_mean"), list)]
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
        md_lines.append(f"- final_loss: mean=`{v['final_loss']['mean']}` std=`{v['final_loss']['std']}` ci95=`{v['final_loss']['ci95']}` n=`{v['final_loss']['n']}`")
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
    baseline: Annotated[Path, typer.Option(
        "--baseline",
        "-b",
        help="Baseline run directory or summary.json (relative paths also searched under --artifacts-dir).",
    )],
    candidate: Annotated[Path, typer.Option(
        "--candidate",
        "-c",
        help="Candidate run directory or summary.json (relative paths also searched under --artifacts-dir).",
    )],
    baseline_variant: Annotated[str | None, typer.Option(
        "--baseline-variant",
        help="Optional sub-run selector inside baseline (e.g. attention_type in suite summaries; or 'sdpa'/'flex' in flex perf summaries).",
    )] = None,
    candidate_variant: Annotated[str | None, typer.Option(
        "--candidate-variant",
        help="Optional sub-run selector inside candidate (e.g. attention_type in suite summaries; or 'sdpa'/'flex' in flex perf summaries).",
    )] = None,
    loss_abs: Annotated[float, typer.Option(
        "--loss-abs",
        help="Absolute loss regression threshold (candidate > baseline + loss_abs).",
        min=0.0,
    )] = 0.01,
    loss_rel: Annotated[float, typer.Option(
        "--loss-rel",
        help="Relative loss regression threshold (candidate > baseline*(1+loss_rel)).",
        min=0.0,
    )] = 0.01,
    throughput_rel: Annotated[float, typer.Option(
        "--throughput-rel",
        help="Relative throughput regression threshold (candidate < baseline*(1-throughput_rel)).",
        min=0.0,
        max=1.0,
    )] = 0.05,
    tflops_rel: Annotated[float, typer.Option(
        "--tflops-rel",
        help="Relative TFLOP/s regression threshold (candidate < baseline*(1-tflops_rel)).",
        min=0.0,
        max=1.0,
    )] = 0.05,
    memory_rel: Annotated[float, typer.Option(
        "--memory-rel",
        help="Relative memory regression threshold (candidate > baseline*(1+memory_rel)).",
        min=0.0,
        max=10.0,
    )] = 0.05,
    artifacts_dir: Annotated[Path, typer.Option(
        "--artifacts-dir",
        help="Artifacts base directory (default: artifacts/).",
    )] = Path("artifacts"),
    run_id: Annotated[str | None, typer.Option(
        "--run-id",
        help="Regression report run id (directory name). Defaults to YYYYMMDD_HHMMSS.",
    )] = None,
    write_artifacts: Annotated[bool, typer.Option(
        "--write-artifacts/--no-write-artifacts",
        help="Write summary.json + run.md under artifacts/regressions/<run_id>/.",
    )] = True,
    html: Annotated[bool, typer.Option(
        "--html/--no-html",
        help="Also write a minimal HTML report alongside run.md (when --write-artifacts is on).",
    )] = True,
    fail_on_regression: Annotated[bool, typer.Option(
        "--fail-on-regression/--no-fail-on-regression",
        help="Exit with code 1 if any metric is flagged as a regression (useful for guardrails/CI).",
    )] = False,
    fail_on_missing: Annotated[bool, typer.Option(
        "--fail-on-missing/--no-fail-on-missing",
        help="Also treat missing metrics as failures when --fail-on-regression is enabled.",
    )] = False,
):
    """Compare two artifact snapshots and highlight regressions/improvements.

    Supports common `summary.json` shapes used in this repo:
    - nanochat.train run summaries (results.losses, tokens_per_second, tflops_per_second_est, peak_memory_allocated_gb)
    - bench suite summaries (top-level runs[]; select a sub-run via --*-variant=attention_type)
    - flex perf summaries (results.tokens_per_s / results.peak_mem_mb; select via --*-variant=sdpa|flex)
    """
    baseline_path = _resolve_summary_path(baseline, artifacts_dir=artifacts_dir)
    candidate_path = _resolve_summary_path(candidate, artifacts_dir=artifacts_dir)

    base_obj = json.loads(baseline_path.read_text(encoding="utf-8"))
    cand_obj = json.loads(candidate_path.read_text(encoding="utf-8"))
    if not isinstance(base_obj, dict) or not isinstance(cand_obj, dict):
        raise typer.BadParameter("Expected both summaries to be JSON objects (dicts).")

    base_meta = _summarize_provenance(base_obj)
    cand_meta = _summarize_provenance(cand_obj)

    metrics = [
        ("final_loss", "Final loss", "lower"),
        ("tokens_per_second", "Tokens/s", "higher"),
        ("tflops_per_second_est", "TFLOP/s (est)", "higher"),
        ("peak_memory_allocated_gb", "Peak mem (GB)", "lower"),
    ]

    comparisons: list[dict[str, Any]] = []
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
            html_table = ["<table><thead><tr><th>metric</th><th>baseline</th><th>candidate</th><th>delta</th><th>delta%</th><th>status</th></tr></thead><tbody>"]
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


if __name__ == "__main__":
    app()
