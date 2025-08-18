#!/usr/bin/env python3
"""
Model Guided Research CLI - Run experimental mathematical models for ML research
"""

import importlib
from pathlib import Path

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
    demo_name: str = typer.Argument(
        ...,
        help="Name of the demo to run",
        autocompletion=lambda: DEMOS.keys()  # type: ignore[call-arg]
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show verbose output"
    ),
):
    """Run a specific demo by name"""

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

            # Run the demo
            with console.status("[bold green]Running demo...[/bold green]"):
                demo_func()
        else:
            console.print(f"[bold red]Error:[/bold red] Function '{func_name}' not found in module")
            raise typer.Exit(1)

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
def run_all(
    delay: int = typer.Option(
        2,
        "--delay", "-d",
        help="Delay in seconds between demos"
    ),
    skip_errors: bool = typer.Option(
        True,
        "--skip-errors/--stop-on-error",
        help="Continue running demos even if one fails"
    ),
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
