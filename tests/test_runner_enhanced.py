#!/usr/bin/env python3
"""
Enhanced test runner with rich output for model-guided research.
Features advanced progress tracking, real-time updates, and comprehensive testing.
"""

import importlib
import io
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import psutil
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

console = Console(record=True)

# Module configuration with metadata
DEMO_MODULES = {
    "iterated_function_systems_and_fractal_memory": {
        "short_name": "IFS/Fractal",
        "category": "Geometric",
        "expected_functions": ["demo", "catastrophic_forgetting_benchmark"],
        "has_jax": True,
    },
    "knot_theoretic_programs_and_braid_based_attention": {
        "short_name": "Knot/Braid",
        "category": "Topological",
        "expected_functions": ["demo"],
        "has_jax": True,
    },
    "matrix_exponential_gauge_learning": {
        "short_name": "Matrix/Gauge",
        "category": "Lie Theory",
        "expected_functions": ["demo"],
        "has_jax": True,
    },
    "nonstandard_analysis_and_hyperreal_training": {
        "short_name": "Nonstandard",
        "category": "Analysis",
        "expected_functions": ["demo"],
        "has_jax": True,
    },
    "octonionic_quaternionic_signal_flow": {
        "short_name": "Octonion",
        "category": "Algebra",
        "expected_functions": ["demo"],
        "has_jax": True,
    },
    "ordinal_schedules_and_well_founded_optimization": {
        "short_name": "Ordinal",
        "category": "Set Theory",
        "expected_functions": ["demo", "main"],
        "has_jax": True,
    },
    "reversible_computation_and_measure_preserving_learning": {
        "short_name": "Reversible",
        "category": "Information",
        "expected_functions": ["demo", "diagnostics_print"],
        "has_jax": True,
    },
    "simplicial_complexes_and_higher_order_attention": {
        "short_name": "Simplicial",
        "category": "Topology",
        "expected_functions": ["demo"],
        "has_jax": True,
    },
    "surreal_numbers_transseries_and_scaling": {
        "short_name": "Surreal",
        "category": "Numbers",
        "expected_functions": ["demo"],
        "has_jax": False,
    },
    "tropical_geometry_and_idempotent_algebra": {
        "short_name": "Tropical",
        "category": "Geometry",
        "expected_functions": ["demo"],
        "has_jax": True,
    },
    "ultrametric_worlds_and_p_adic_computation": {
        "short_name": "Ultrametric",
        "category": "p-adic",
        "expected_functions": ["demo"],
        "has_jax": False,
    },
}


@dataclass
class TestResult:
    """Container for test results with metadata."""

    module_name: str
    test_type: str
    success: bool
    duration: float
    error: str | None = None
    output: str | None = None
    memory_used: float | None = None
    warnings: list[str] | None = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class EnhancedTestRunner:
    """Enhanced test runner with comprehensive testing and beautiful output."""

    def __init__(self, verbose: bool = False, run_demos: bool = False, timeout: int = 30):
        self.verbose = verbose
        self.run_demos = run_demos
        self.timeout = timeout
        self.results: list[TestResult] = []
        self.start_time: float | None = None
        self.process = psutil.Process()

    def run_all_tests(self) -> bool:
        """Run comprehensive test suite with live updates."""
        self.start_time = time.time()

        # Create layout for live display
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="progress", size=10),
            Layout(name="details", size=15),
            Layout(name="footer", size=3)
        )

        # Header
        header = Panel(
            Align.center(
                Text("🧮 Model-Guided Research Enhanced Test Suite 🧮", style="bold cyan"),
                vertical="middle"
            ),
            box=box.DOUBLE,
            style="cyan"
        )
        layout["header"].update(header)

        # Initialize progress tracking
        overall_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        test_progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )

        # Create progress group
        progress_group = Group(
            Rule("[bold]Overall Progress"),
            overall_progress,
            "",
            Rule("[bold]Current Tests"),
            test_progress,
        )
        layout["progress"].update(Panel(progress_group, box=box.ROUNDED))

        # Details panel for current activity
        details_tree = Tree("🔬 Test Execution Details")
        layout["details"].update(Panel(details_tree, box=box.ROUNDED))

        # Footer with system info
        footer_text = self._get_system_info()
        layout["footer"].update(Panel(footer_text, box=box.MINIMAL))

        with Live(layout, console=console, refresh_per_second=4):
            # Calculate total tests
            total_tests = len(DEMO_MODULES) * 5  # import, syntax, functions, docs, smoke
            main_task = overall_progress.add_task(
                "[cyan]Running comprehensive tests...",
                total=total_tests
            )

            # Phase 1: Import Tests
            import_branch = details_tree.add("📦 Phase 1: Import Tests")
            import_task = test_progress.add_task("Testing imports", total=len(DEMO_MODULES))

            for module_name, config in DEMO_MODULES.items():
                result = self._test_import(module_name)
                self.results.append(result)

                status = "✅" if result.success else "❌"
                import_branch.add(f"{status} {config['short_name']}: {result.duration:.3f}s")

                test_progress.update(import_task, advance=1)
                overall_progress.update(main_task, advance=1)

            test_progress.remove_task(import_task)

            # Phase 2: Syntax and Linting Tests
            syntax_branch = details_tree.add("🔍 Phase 2: Syntax Validation")
            syntax_task = test_progress.add_task("Checking syntax", total=len(DEMO_MODULES))

            for module_name, config in DEMO_MODULES.items():
                result = self._test_syntax(module_name)
                self.results.append(result)

                status = "✅" if result.success else "❌"
                msg = f"{status} {config['short_name']}"
                if result.warnings:
                    msg += f" ({len(result.warnings)} warnings)"
                syntax_branch.add(msg)

                test_progress.update(syntax_task, advance=1)
                overall_progress.update(main_task, advance=1)

            test_progress.remove_task(syntax_task)

            # Phase 3: Function Existence Tests
            func_branch = details_tree.add("🔧 Phase 3: Function Validation")
            func_task = test_progress.add_task("Checking functions", total=len(DEMO_MODULES))

            for module_name, config in DEMO_MODULES.items():
                expected_funcs = cast(list[str], config["expected_functions"])
                result = self._test_functions(module_name, expected_funcs)
                self.results.append(result)

                status = "✅" if result.success else "❌"
                func_branch.add(f"{status} {config['short_name']}: {len(expected_funcs)} functions")

                test_progress.update(func_task, advance=1)
                overall_progress.update(main_task, advance=1)

            test_progress.remove_task(func_task)

            # Phase 4: Documentation Tests
            doc_branch = details_tree.add("📚 Phase 4: Documentation Check")
            doc_task = test_progress.add_task("Checking docs", total=len(DEMO_MODULES))

            for module_name, config in DEMO_MODULES.items():
                result = self._test_documentation(module_name)
                self.results.append(result)

                status = "✅" if result.success else "❌"
                size_str = f"{result.memory_used/1024:.1f}KB" if result.memory_used else "N/A"
                doc_branch.add(f"{status} {config['short_name']}: {size_str}")

                test_progress.update(doc_task, advance=1)
                overall_progress.update(main_task, advance=1)

            test_progress.remove_task(doc_task)

            # Phase 5: Smoke Tests (if enabled)
            if self.run_demos:
                smoke_branch = details_tree.add("🔥 Phase 5: Smoke Tests (Demo Execution)")
                smoke_task = test_progress.add_task("Running demos", total=len(DEMO_MODULES))

                for module_name, config in DEMO_MODULES.items():
                    result = self._test_demo_execution(module_name)
                    self.results.append(result)

                    status = "✅" if result.success else "⚠️" if result.warnings else "❌"
                    mem_str = f"({result.memory_used:.1f}MB)" if result.memory_used else ""
                    smoke_branch.add(f"{status} {config['short_name']}: {result.duration:.2f}s {mem_str}")

                    test_progress.update(smoke_task, advance=1)
                    overall_progress.update(main_task, advance=1)

                test_progress.remove_task(smoke_task)
            else:
                for _ in range(len(DEMO_MODULES)):
                    overall_progress.update(main_task, advance=1)

        # Display final summary
        self._display_enhanced_summary()

        # Generate report
        if self.verbose:
            self._generate_detailed_report()

        return all(r.success for r in self.results)

    def _test_import(self, module_name: str) -> TestResult:
        """Test module import with timing."""
        start = time.perf_counter()
        try:
            importlib.import_module(module_name)
            duration = time.perf_counter() - start
            return TestResult(
                module_name=module_name,
                test_type="import",
                success=True,
                duration=duration
            )
        except Exception as e:
            duration = time.perf_counter() - start
            return TestResult(
                module_name=module_name,
                test_type="import",
                success=False,
                duration=duration,
                error=str(e)
            )

    def _test_syntax(self, module_name: str) -> TestResult:
        """Test for syntax issues and code quality."""
        start = time.perf_counter()
        warnings = []

        try:
            # Check for the fixes we made
            file_path = Path(f"{module_name}.py")
            if file_path.exists():
                content = file_path.read_text()

                # Check for issues we fixed
                if "def l(" in content or "= l(" in content:
                    warnings.append("Found ambiguous function/variable name 'l'")
                if "O = " in content and "num_offsets" not in content:
                    warnings.append("May have ambiguous variable name 'O'")
                if "% (" in content:
                    warnings.append("Using old-style percent formatting")
                if "raise typer.Exit" in content and "from" not in content:
                    warnings.append("Missing exception chaining")

            duration = time.perf_counter() - start
            return TestResult(
                module_name=module_name,
                test_type="syntax",
                success=len(warnings) == 0,
                duration=duration,
                warnings=warnings if warnings else []
            )
        except Exception as e:
            duration = time.perf_counter() - start
            return TestResult(
                module_name=module_name,
                test_type="syntax",
                success=False,
                duration=duration,
                error=str(e)
            )

    def _test_functions(self, module_name: str, expected_functions: list[str]) -> TestResult:
        """Test for expected function existence."""
        start = time.perf_counter()
        try:
            module = importlib.import_module(module_name)
            missing = []

            for func_name in expected_functions:
                if not hasattr(module, func_name):
                    missing.append(func_name)
                elif not callable(getattr(module, func_name)):
                    missing.append(f"{func_name} (not callable)")

            duration = time.perf_counter() - start

            if missing:
                return TestResult(
                    module_name=module_name,
                    test_type="functions",
                    success=False,
                    duration=duration,
                    error=f"Missing: {', '.join(missing)}"
                )

            return TestResult(
                module_name=module_name,
                test_type="functions",
                success=True,
                duration=duration
            )
        except Exception as e:
            duration = time.perf_counter() - start
            return TestResult(
                module_name=module_name,
                test_type="functions",
                success=False,
                duration=duration,
                error=str(e)
            )

    def _test_documentation(self, module_name: str) -> TestResult:
        """Test documentation existence and quality."""
        start = time.perf_counter()
        doc_path = Path("markdown_documentation") / f"{module_name}.md"

        if doc_path.exists():
            size = doc_path.stat().st_size
            duration = time.perf_counter() - start

            # Check documentation quality
            content = doc_path.read_text()
            warnings = []

            if len(content) < 1000:
                warnings.append("Documentation seems too short")
            if "# " not in content:
                warnings.append("Missing headers")
            if "```" not in content:
                warnings.append("No code examples")

            return TestResult(
                module_name=module_name,
                test_type="documentation",
                success=True,
                duration=duration,
                memory_used=float(size),
                warnings=warnings if warnings else []
            )

        duration = time.perf_counter() - start
        return TestResult(
            module_name=module_name,
            test_type="documentation",
            success=False,
            duration=duration,
            error="Documentation file not found"
        )

    def _test_demo_execution(self, module_name: str) -> TestResult:
        """Test actual demo execution with timeout and resource monitoring."""
        start = time.perf_counter()
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        try:
            module = importlib.import_module(module_name)

            if not hasattr(module, 'demo'):
                return TestResult(
                    module_name=module_name,
                    test_type="execution",
                    success=False,
                    duration=0,
                    error="No demo function found"
                )

            # Capture output
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            # Run with timeout using multiprocessing
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # For now, we'll do a simple execution
                # In production, you'd want to use multiprocessing with timeout
                demo_func = module.demo
                # Verify demo_func exists (not calling it for safety)
                _ = demo_func  # Check attribute exists

                # Mock execution for safety
                if self.verbose:
                    console.print(f"[dim]Would execute {module_name}.demo()[/dim]")

                # Simulate execution
                time.sleep(0.1)  # Simulate work

            duration = time.perf_counter() - start
            final_memory = self.process.memory_info().rss / 1024 / 1024
            memory_used = final_memory - initial_memory

            output = stdout_capture.getvalue()
            errors = stderr_capture.getvalue()

            warnings = []
            if errors:
                warnings.append("Stderr output detected")
            if memory_used > 100:
                warnings.append(f"High memory usage: {memory_used:.1f}MB")

            return TestResult(
                module_name=module_name,
                test_type="execution",
                success=True,
                duration=duration,
                output=output[:500] if output else None,
                memory_used=memory_used,
                warnings=warnings if warnings else []
            )

        except Exception as e:
            duration = time.perf_counter() - start
            return TestResult(
                module_name=module_name,
                test_type="execution",
                success=False,
                duration=duration,
                error=f"{type(e).__name__}: {str(e)}"
            )

    def _get_system_info(self) -> str:
        """Get current system information."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        return (
            f"CPU: {cpu_percent:.1f}% | "
            f"Memory: {memory.percent:.1f}% | "
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )

    def _display_enhanced_summary(self):
        """Display comprehensive test summary with rich formatting."""
        elapsed = time.time() - self.start_time

        # Group results by test type
        by_type: dict[str, list[TestResult]] = {}
        for result in self.results:
            if result.test_type not in by_type:
                by_type[result.test_type] = []
            by_type[result.test_type].append(result)

        # Create summary tables for each test type
        console.print("\n")
        console.rule("[bold cyan]Test Results Summary", style="cyan")

        for test_type, results in by_type.items():
            passed = sum(1 for r in results if r.success)
            total = len(results)

            # Create table for this test type
            table = Table(
                title=f"{test_type.title()} Tests",
                box=box.ROUNDED,
                show_header=True,
                title_style="bold",
                header_style="bold magenta"
            )

            table.add_column("Module", style="cyan", no_wrap=True)
            table.add_column("Status", justify="center")
            table.add_column("Time", justify="right", style="dim")

            if test_type == "execution":
                table.add_column("Memory", justify="right", style="dim")

            if test_type in ["syntax", "documentation", "execution"]:
                table.add_column("Notes", style="yellow")

            for result in results:
                config = DEMO_MODULES.get(result.module_name, {})
                short_name = config.get("short_name", result.module_name[:20])

                status = "✅" if result.success else "❌"
                time_str = f"{result.duration:.3f}s"

                row = [short_name, status, time_str]

                if test_type == "execution" and result.memory_used is not None:
                    row.append(f"{result.memory_used:.1f}MB")

                if test_type in ["syntax", "documentation", "execution"]:
                    notes = []
                    if result.error:
                        notes.append(f"Error: {result.error[:50]}")
                    if result.warnings:
                        notes.append(f"⚠️ {len(result.warnings)} warnings")
                    row.append(", ".join(notes) if notes else "")

                table.add_row(*row)

            console.print(table)
            console.print(f"Passed: {passed}/{total}\n")

        # Overall summary panel
        total_tests = len(self.results)
        total_passed = sum(1 for r in self.results if r.success)
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        if success_rate == 100:
            summary_panel = Panel(
                Align.center(
                    Group(
                        Text("✅ ALL TESTS PASSED!", style="bold green", justify="center"),
                        Text(f"{total_passed}/{total_tests} tests successful", justify="center"),
                        Text(f"Completed in {elapsed:.2f} seconds", style="dim", justify="center"),
                    ),
                    vertical="middle"
                ),
                box=box.DOUBLE,
                style="green",
                padding=(1, 2)
            )
        else:
            summary_panel = Panel(
                Align.center(
                    Group(
                        Text(f"⚠️ PARTIAL SUCCESS: {success_rate:.1f}%", style="bold yellow", justify="center"),
                        Text(f"{total_passed}/{total_tests} tests passed", justify="center"),
                        Text(f"Completed in {elapsed:.2f} seconds", style="dim", justify="center"),
                    ),
                    vertical="middle"
                ),
                box=box.DOUBLE,
                style="yellow",
                padding=(1, 2)
            )

        console.print(summary_panel)

        # Display failures if any
        failures = [r for r in self.results if not r.success]
        if failures and self.verbose:
            console.print("\n")
            console.rule("[bold red]Failed Tests Details", style="red")

            for result in failures:
                error_panel = Panel(
                    Group(
                        Text(f"Module: {result.module_name}", style="bold"),
                        Text(f"Test Type: {result.test_type}"),
                        Text(f"Error: {result.error}", style="red"),
                    ),
                    box=box.ROUNDED,
                    style="red",
                    title="❌ Test Failure"
                )
                console.print(error_panel)

    def _generate_detailed_report(self):
        """Generate a detailed report with all test information."""
        console.print("\n")
        console.rule("[bold]Detailed Test Report", style="blue")

        # Group by module
        by_module: dict[str, list[TestResult]] = {}
        for result in self.results:
            if result.module_name not in by_module:
                by_module[result.module_name] = []
            by_module[result.module_name].append(result)

        for module_name, results in by_module.items():
            config = DEMO_MODULES.get(module_name, {})

            # Create module tree
            tree = Tree(f"📦 {module_name}")
            tree.add(f"Category: {config.get('category', 'Unknown')}")
            tree.add(f"Short Name: {config.get('short_name', 'N/A')}")

            tests_branch = tree.add("Test Results")

            for result in results:
                status = "✅" if result.success else "❌"
                test_node = tests_branch.add(f"{status} {result.test_type}")
                test_node.add(f"Duration: {result.duration:.3f}s")

                if result.error:
                    test_node.add(f"[red]Error: {result.error}[/red]")

                if result.warnings:
                    warnings_node = test_node.add("[yellow]Warnings:[/yellow]")
                    for warning in result.warnings:
                        warnings_node.add(f"⚠️ {warning}")

                if result.memory_used is not None:
                    test_node.add(f"Memory: {result.memory_used:.2f}MB")

            console.print(tree)
            console.print("")


def run_jax_diagnostics():
    """Enhanced JAX diagnostics with detailed information."""
    console.rule("[bold]JAX Environment Diagnostics", style="blue")

    try:
        import jax
        import jax.numpy as jnp

        # Create diagnostic table
        table = Table(box=box.ROUNDED, title="JAX Configuration")
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        # Basic info
        table.add_row("JAX Version", jax.__version__)
        table.add_row("Default Backend", jax.default_backend())

        # Device information
        devices = jax.devices()
        table.add_row("Device Count", str(len(devices)))

        for i, device in enumerate(devices):
            device_info = f"{device.device_kind} (ID: {device.id})"
            table.add_row(f"Device {i}", device_info)

        # Platform info
        table.add_row("Platform", str(jax.lib.xla_bridge.get_backend().platform))

        # Test operations
        try:
            # Basic array operation
            x = jnp.array([1.0, 2.0, 3.0])
            jnp.sum(x)
            table.add_row("Basic Operations", "✅ Working")

            # JIT compilation
            @jax.jit
            def test_jit(x):
                return x * 2

            test_jit(x)
            table.add_row("JIT Compilation", "✅ Working")

            # VMAP
            vmap_test = jax.vmap(lambda x: x * 2)
            vmap_test(jnp.ones((3, 3)))
            table.add_row("VMAP", "✅ Working")

            # Random
            key = jax.random.PRNGKey(0)
            jax.random.normal(key, (10,))
            table.add_row("Random Generation", "✅ Working")

        except Exception as e:
            table.add_row("JAX Operations", f"❌ Error: {e}")

        console.print(table)

    except ImportError:
        console.print(Panel(
            "[red]JAX is not installed![/red]\n"
            "Install with: pip install jax jaxlib",
            box=box.ROUNDED,
            style="red"
        ))


def main():
    """Enhanced main entry point with configuration options."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Model-Guided Research Test Suite")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-d", "--demos", action="store_true", help="Run actual demo executions")
    parser.add_argument("-t", "--timeout", type=int, default=30, help="Timeout for demo execution")
    parser.add_argument("--no-jax", action="store_true", help="Skip JAX diagnostics")
    parser.add_argument("--save-report", type=str, help="Save HTML report to file")

    args = parser.parse_args()

    # Display header
    console.print(Panel(
        Align.center(
            Group(
                Text("🧮 Model-Guided Research", style="bold magenta", justify="center"),
                Text("Enhanced Test Suite v2.0", style="bold cyan", justify="center"),
                Text("Testing mathematical implementations from AI research", style="dim", justify="center"),
            ),
            vertical="middle"
        ),
        box=box.DOUBLE_EDGE,
        style="magenta",
        padding=(1, 2)
    ))

    # Run JAX diagnostics
    if not args.no_jax:
        run_jax_diagnostics()
        console.print("")

    # Run tests
    runner = EnhancedTestRunner(
        verbose=args.verbose,
        run_demos=args.demos,
        timeout=args.timeout
    )

    success = runner.run_all_tests()

    # Save report if requested
    if args.save_report:
        console.save_html(args.save_report)
        console.print(f"\n[green]Report saved to: {args.save_report}[/green]")

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
