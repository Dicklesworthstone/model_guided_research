#!/usr/bin/env python3
"""
Model Guided Research CLI - Run experimental mathematical models for ML research
"""

import importlib
import json
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="model-guided-research",
    help="Run experimental mathematical models for machine learning research",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()

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


@app.command()
def list():
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
):
    """Run a specific demo by name"""

    # Configure settings
    from config import ProjectConfig, set_config

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
    if no_rich:
        config.use_rich_output = False
    if debug:
        config.debug_mode = True
        config.check_numerics = True
        config.jax_debug_nans = True

    # Set the global config
    set_config(config)

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



        # Write artifacts if requested
        if export_json is not None:
            try:
                export_json.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            with export_json.open("w") as f:
                json.dump(artifacts, f, indent=2)
            if verbose:
                console.print(f"[dim]Wrote JSON artifact to {export_json}[/dim]")


    except ImportError as e:
        console.print(f"[bold red]Import Error:[/bold red] {e}")
        console.print("\n[dim]Make sure all dependencies are installed:[/dim]")
        console.print("[bold]uv pip install -e .[/bold]")
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
            with open(module_file) as f:
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

    with open(output_path, 'w') as f:
        json.dump(example_config, f, indent=2)

    console.print(f"[green]✓ Example config written to {output_path}[/green]")
    console.print(f"\n[dim]Use it with:[/dim] [bold]mgr run <demo> --config {output_path}[/bold]")


@app.command()
def run_all(
    delay: Annotated[int, typer.Option(
        "--delay", "-d",
        help="Delay in seconds between demos"
    )] = 2,
    skip_errors: Annotated[bool, typer.Option(
        "--skip-errors/--stop-on-error",
        help="Continue running demos even if one fails"
    )] = True,
):
    """Run all available demos in sequence"""

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


if __name__ == "__main__":
    app()
@app.command("eval")
def evaluate(
    ultra_packed: Annotated[bool, typer.Option(
        "--ultra-packed",
        help="Use packed bit-trie implementation for ultrametric tests (sets ULTRA_PACKED=1)"
    )] = False,
    export_json: Annotated[Path | None, typer.Option(
        "--export-json",
        help="Write a combined JSON artifact of the practical utility suite"
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
        with export_json.open("w") as f:
            json.dump({"results": payload}, f, indent=2)
        console.print(f"[dim]Wrote suite JSON to {export_json}[/dim]")
