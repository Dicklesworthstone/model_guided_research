#!/usr/bin/env python3
"""
Beautiful test runner with rich output for model-guided research.
Updated to harmonize with latest code changes and linting fixes.
"""

import importlib
import sys
import time
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

console = Console()

# Module configuration updated to reflect code structure
DEMO_MODULES = [
    "iterated_function_systems_and_fractal_memory",
    "knot_theoretic_programs_and_braid_based_attention",
    "matrix_exponential_gauge_learning",
    "nonstandard_analysis_and_hyperreal_training",
    "octonionic_quaternionic_signal_flow",
    "ordinal_schedules_and_well_founded_optimization",
    "reversible_computation_and_measure_preserving_learning",
    "simplicial_complexes_and_higher_order_attention",
    "surreal_numbers_transseries_and_scaling",
    "tropical_geometry_and_idempotent_algebra",
    "ultrametric_worlds_and_p_adic_computation",
]

# Code quality checks based on our linting fixes
CODE_QUALITY_CHECKS = {
    "ambiguous_names": ["def l(", "= l ", "= l,", " l,", " l =", "O = "],
    "old_formatting": ["% ("],
    "exception_chaining": ["raise typer.Exit(1)", "raise typer.Exit(0)"],
}


class TestRunner:
    """Rich test runner for mathematical demos."""

    def __init__(self, check_code_quality: bool = True):
        self.results: dict[str, dict[str, Any]] = {}
        self.start_time: float | None = None
        self.current_test: str | None = None
        self.check_code_quality = check_code_quality
        self.code_issues: dict[str, list[str]] = {}

    def run_all_tests(self) -> bool:
        """Run all tests with beautiful progress display."""
        self.start_time = time.time()

        console.print(
            Panel.fit(
                "[bold cyan]Model-Guided Research Test Suite[/bold cyan]\n"
                "[dim]Testing mathematical implementations from AI-generated research directions[/dim]",
                box=box.DOUBLE,
            )
        )

        # Test imports
        console.print("\n[bold]Phase 1: Testing Module Imports[/bold]")
        console.print("[dim]Verifying all modules can be imported successfully...[/dim]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            import_task = progress.add_task("Testing imports...", total=len(DEMO_MODULES))

            for module_name in DEMO_MODULES:
                self.current_test = module_name
                result = self._test_import(module_name)
                self.results[f"import_{module_name}"] = result

                if result["success"]:
                    progress.console.print(f"  ✅ {module_name}")
                else:
                    progress.console.print(f"  ❌ {module_name}: {result['error']}")

                progress.update(import_task, advance=1)

        # Code quality checks (new)
        if self.check_code_quality:
            console.print("\n[bold]Phase 2: Code Quality Checks[/bold]")
            console.print("[dim]Verifying linting fixes have been applied...[/dim]\n")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                quality_task = progress.add_task("Checking code quality...", total=len(DEMO_MODULES))

                for module_name in DEMO_MODULES:
                    self.current_test = module_name
                    result = self._test_code_quality(module_name)
                    self.results[f"quality_{module_name}"] = result

                    if result["success"]:
                        progress.console.print(f"  ✅ {module_name}: Clean")
                    else:
                        issue_count = len(result.get("issues", []))
                        progress.console.print(f"  ⚠️  {module_name}: {issue_count} potential issues")

                    progress.update(quality_task, advance=1)

        # Test demo functions
        phase_num = 3 if self.check_code_quality else 2
        console.print(f"\n[bold]Phase {phase_num}: Testing Demo Functions[/bold]")
        console.print("[dim]Checking for required demo() functions...[/dim]\n")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            demo_task = progress.add_task("Testing demo functions...", total=len(DEMO_MODULES))

            for module_name in DEMO_MODULES:
                self.current_test = module_name
                result = self._test_demo_exists(module_name)
                self.results[f"demo_{module_name}"] = result

                if result["success"]:
                    progress.console.print(f"  ✅ {module_name}: demo() found")
                else:
                    progress.console.print(f"  ❌ {module_name}: {result['error']}")

                progress.update(demo_task, advance=1)

        # Test documentation
        phase_num = 4 if self.check_code_quality else 3
        console.print(f"\n[bold]Phase {phase_num}: Testing Documentation[/bold]")
        console.print("[dim]Verifying markdown documentation exists...[/dim]\n")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            doc_task = progress.add_task("Checking documentation...", total=len(DEMO_MODULES))

            for module_name in DEMO_MODULES:
                self.current_test = module_name
                result = self._test_documentation(module_name)
                self.results[f"doc_{module_name}"] = result

                if result["success"]:
                    progress.console.print(f"  ✅ {module_name}.md: {result['size']} bytes")
                else:
                    progress.console.print(f"  ❌ {module_name}.md: missing")

                progress.update(doc_task, advance=1)

        # Display summary
        self._display_summary()

        # Return True if all tests passed
        return all(r["success"] for r in self.results.values())

    def _test_code_quality(self, module_name: str) -> dict[str, Any]:
        """Test if code quality fixes have been applied."""
        try:
            file_path = Path(f"{module_name}.py")
            if not file_path.exists():
                return {"success": False, "error": "Source file not found"}

            content = file_path.read_text()
            issues = []

            # Check for issues we fixed
            for category, patterns in CODE_QUALITY_CHECKS.items():
                for pattern in patterns:
                    if pattern in content:
                        # Special handling for some patterns
                        if pattern == "O = " and "num_offsets" in content:
                            continue  # This was fixed
                        if pattern in ["raise typer.Exit(1)", "raise typer.Exit(0)"]:
                            # Check if it has proper exception chaining
                            lines = content.split("\n")
                            for i, line in enumerate(lines):
                                if pattern in line and "from" not in line:
                                    issues.append(f"{category}: Line {i + 1}")
                        else:
                            issues.append(f"{category}: Found '{pattern}'")

            if issues:
                self.code_issues[module_name] = issues
                return {"success": False, "issues": issues}

            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _test_import(self, module_name: str) -> dict[str, Any]:
        """Test if a module can be imported."""
        try:
            start = time.perf_counter()
            module = importlib.import_module(module_name)
            elapsed = time.perf_counter() - start
            return {"success": True, "module": module, "time": elapsed}
        except Exception as e:
            return {"success": False, "error": str(e), "time": 0}

    def _test_demo_exists(self, module_name: str) -> dict[str, Any]:
        """Test if a module has a demo function."""
        try:
            module = importlib.import_module(module_name)
            has_demo = hasattr(module, "demo")
            is_callable = callable(getattr(module, "demo", None))

            if has_demo and is_callable:
                return {"success": True}
            elif has_demo:
                return {"success": False, "error": "demo exists but not callable"}
            else:
                return {"success": False, "error": "no demo function"}
        except Exception as e:
            return {"success": False, "error": f"import failed: {e}"}

    def _test_documentation(self, module_name: str) -> dict[str, Any]:
        """Test if documentation exists for a module."""
        doc_path = Path("markdown_documentation") / f"{module_name}.md"
        if doc_path.exists():
            size = doc_path.stat().st_size
            return {"success": True, "size": size}
        return {"success": False}

    def _display_summary(self):
        """Display a beautiful summary table."""
        elapsed = time.time() - self.start_time

        # Count successes
        import_success = sum(1 for k, v in self.results.items() if k.startswith("import_") and v["success"])
        quality_success = sum(1 for k, v in self.results.items() if k.startswith("quality_") and v["success"])
        demo_success = sum(1 for k, v in self.results.items() if k.startswith("demo_") and v["success"])
        doc_success = sum(1 for k, v in self.results.items() if k.startswith("doc_") and v["success"])

        # Create summary table
        table = Table(title="Test Summary", box=box.ROUNDED, show_header=True)
        table.add_column("Category", style="cyan")
        table.add_column("Passed", style="green")
        table.add_column("Failed", style="red")
        table.add_column("Total", style="white")

        table.add_row(
            "Module Imports", str(import_success), str(len(DEMO_MODULES) - import_success), str(len(DEMO_MODULES))
        )

        if self.check_code_quality:
            table.add_row(
                "Code Quality", str(quality_success), str(len(DEMO_MODULES) - quality_success), str(len(DEMO_MODULES))
            )

        table.add_row(
            "Demo Functions", str(demo_success), str(len(DEMO_MODULES) - demo_success), str(len(DEMO_MODULES))
        )
        table.add_row("Documentation", str(doc_success), str(len(DEMO_MODULES) - doc_success), str(len(DEMO_MODULES)))

        console.print("\n")
        console.print(table)

        # Overall result
        total_tests = len(self.results)
        total_passed = sum(1 for v in self.results.values() if v["success"])

        if total_passed == total_tests:
            console.print(
                Panel(
                    f"[bold green]✅ ALL TESTS PASSED![/bold green]\n"
                    f"[dim]{total_passed}/{total_tests} tests successful in {elapsed:.2f}s[/dim]",
                    box=box.ROUNDED,
                    style="green",
                )
            )
        else:
            console.print(
                Panel(
                    f"[bold red]❌ SOME TESTS FAILED[/bold red]\n"
                    f"[dim]{total_passed}/{total_tests} tests passed in {elapsed:.2f}s[/dim]",
                    box=box.ROUNDED,
                    style="red",
                )
            )

        # Display failed tests
        failures = [(k, v) for k, v in self.results.items() if not v["success"]]
        if failures:
            console.print("\n[bold red]Failed Tests:[/bold red]")
            for name, result in failures:
                if "issues" in result:
                    console.print(f"  • {name}: {len(result['issues'])} code quality issues")
                else:
                    error = result.get("error", "unknown error")
                    console.print(f"  • {name}: {error}")

        # Display code quality issues if any
        if self.code_issues:
            console.print("\n[bold yellow]Code Quality Issues Found:[/bold yellow]")
            console.print("[dim]These may indicate incomplete linting fixes:[/dim]\n")
            for module, issues in self.code_issues.items():
                console.print(f"  [cyan]{module}:[/cyan]")
                for issue in issues[:3]:  # Show first 3 issues
                    console.print(f"    • {issue}")
                if len(issues) > 3:
                    console.print(f"    • ... and {len(issues) - 3} more")


def run_jax_diagnostics():
    """Run JAX diagnostics with rich output."""
    console.print("\n")
    console.rule("[bold]JAX Environment Diagnostics[/bold]", style="cyan")
    console.print()

    table = Table(box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    import jax

    # Get device info
    devices = jax.devices()
    table.add_row("Available Devices", str(len(devices)))
    table.add_row("Default Backend", jax.default_backend())

    for i, device in enumerate(devices):
        table.add_row(f"Device {i}", str(device))

    # JAX version
    try:
        table.add_row("JAX Version", jax.__version__)
    except AttributeError:
        table.add_row("JAX Version", "Unknown")

    # Test basic operations
    try:
        x = jax.numpy.array([1.0, 2.0, 3.0])
        _ = jax.numpy.sum(x)
        table.add_row("Basic Operations", "✅ Working")
    except Exception as e:
        table.add_row("Basic Operations", f"❌ Failed: {e}")

    # Test JIT compilation
    try:

        @jax.jit
        def test_func(x):
            return x * 2

        test_func(jax.numpy.ones(3))
        table.add_row("JIT Compilation", "✅ Working")
    except Exception as e:
        table.add_row("JIT Compilation", f"❌ Failed: {e}")

    console.print(table)


def main():
    """Main test runner entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Model-Guided Research Test Suite")
    parser.add_argument("--no-quality", action="store_true", help="Skip code quality checks")
    parser.add_argument("--no-jax", action="store_true", help="Skip JAX diagnostics")

    args = parser.parse_args()

    # Display header with version info
    header_text = Text.assemble(
        ("🧮 Model-Guided Research Test Suite 🧮\n", "bold magenta"),
        ("Version 1.1 - ", "dim"),
        ("Harmonized with latest code changes", "dim italic"),
    )

    console.print(Panel(header_text, box=box.DOUBLE_EDGE, style="magenta"))

    # Run JAX diagnostics
    if not args.no_jax:
        run_jax_diagnostics()

    # Run all tests
    runner = TestRunner(check_code_quality=not args.no_quality)
    success = runner.run_all_tests()

    # Show final message
    console.print("\n")
    if success:
        console.print("[bold green]✨ Test suite completed successfully![/bold green]")
    else:
        console.print("[bold yellow]⚠️  Test suite completed with some issues.[/bold yellow]")
        console.print("[dim]Review the failures above for details.[/dim]")

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
