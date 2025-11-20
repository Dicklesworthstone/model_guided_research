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
from nanochat.torch_imports import torch

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
        issues = []
        if has_nan:
            issues.append('NaN')
        if has_inf:
            issues.append('Inf')
        console.print(f"[bold red]Warning:[/bold red] {name} contains {' and '.join(issues)} values")


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


def conditional_print(message: str, level: int = 1) -> None:
    """Print message only if verbose level is high enough."""
    from config import get_config
    config = get_config()
    if config.verbose and config.verbose_level >= level:
        if config.use_rich_output:
            console.print(message)
        else:
            print(message)


def log_metrics_conditionally(step: int, metrics: dict[str, Any]) -> None:
    """Log metrics based on config settings."""
    from config import get_config
    config = get_config()

    if not config.log_metrics:
        return

    if step % config.log_interval != 0 and step != 0:
        return

    if config.use_rich_output:
        # Format metrics nicely
        metric_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                 for k, v in metrics.items()])
        console.print(f"[cyan]Step {step}[/cyan] | {metric_str}")
    else:
        print(f"Step {step} | " + " | ".join([f"{k}: {v}" for k, v in metrics.items()]))


def save_checkpoint(params: Any, step: int, metrics: dict[str, Any] | None = None) -> None:
    """Save model checkpoint if configured."""
    from config import get_config

    config = get_config()
    if not config.save_checkpoints:
        return

    checkpoint_path = config.checkpoint_dir / f"checkpoint_step_{step}.pkl"
    checkpoint_data = {
        "params": params,
        "step": step,
        "metrics": metrics or {}
    }

    torch.save(checkpoint_data, checkpoint_path)

    conditional_print(f"[dim]Checkpoint saved to {checkpoint_path}[/dim]", level=2)


def create_progress_bar(total: int, description: str = "Processing") -> Any:
    """Create a progress bar if configured."""
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

    from config import get_config

    config = get_config()
    if not config.show_progress_bars or not config.use_rich_output:
        return None

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    )
    progress.start()
    progress.add_task(description, total=total)
    return progress


def safe_divide(numerator: jnp.ndarray, denominator: jnp.ndarray,
                 epsilon: float = 1e-8) -> jnp.ndarray:
    """Safe division avoiding NaN/Inf."""
    return numerator / (denominator + epsilon)


def gradient_norm(grads: Any) -> float:
    """Calculate L2 norm of gradients."""
    leaves = jax.tree_util.tree_leaves(grads)
    return float(jnp.sqrt(sum(jnp.sum(g**2) for g in leaves)))


def clip_gradients(grads: Any, max_norm: float | None = None) -> Any:
    """Clip gradients by global norm."""
    from config import get_config

    config = get_config()
    max_norm = max_norm or config.gradient_clip_norm

    if max_norm is None:
        return grads

    g_norm = gradient_norm(grads)
    scale = jnp.minimum(1.0, max_norm / (g_norm + 1e-8))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)
