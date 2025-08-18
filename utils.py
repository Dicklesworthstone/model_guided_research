"""
Shared utilities for model-guided research implementations.
"""

import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import jax
import jax.numpy as jnp
from rich import box
from rich.console import Console
from rich.table import Table

console = Console()


def timer(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        console.print(f"[dim]Execution time for {func.__name__}: {end - start:.3f}s[/dim]")
        return result
    return wrapper


def print_metrics(metrics: dict[str, Any], title: str = "Metrics") -> None:
    """Print metrics in a nice table format."""
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    for key, value in metrics.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    console.print(table)


def check_nan_inf(x: jnp.ndarray, name: str = "tensor") -> None:
    """Check for NaN or Inf values in tensor."""
    has_nan = jnp.any(jnp.isnan(x))
    has_inf = jnp.any(jnp.isinf(x))

    if has_nan or has_inf:
        console.print(f"[bold red]Warning:[/bold red] {name} contains {'NaN' if has_nan else ''}{' and ' if has_nan and has_inf else ''}{'Inf' if has_inf else ''} values")


def get_device_info() -> dict[str, Any]:
    """Get information about available compute devices."""
    devices = jax.devices()
    return {
        "device_count": len(devices),
        "devices": [str(d) for d in devices],
        "default_backend": jax.default_backend(),
    }


def print_model_summary(params: Any, name: str = "Model") -> None:
    """Print a summary of model parameters."""
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    param_bytes = sum(x.nbytes for x in jax.tree_util.tree_leaves(params))

    table = Table(title=f"{name} Summary", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Total Parameters", f"{param_count:,}")
    table.add_row("Memory (MB)", f"{param_bytes / 1024 / 1024:.2f}")
    table.add_row("Parameter Groups", str(len(jax.tree_util.tree_leaves(params))))

    console.print(table)
