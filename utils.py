"""
Shared utilities for model-guided research implementations.
"""

import random
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

from nanochat.torch_imports import torch

console = Console()


def seed_everything(seed: int):
    """Seed Python, NumPy, and PyTorch RNGs; return a JAX PRNGKey for convenience."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return jax.random.PRNGKey(seed)


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


def check_nan_inf(
    x: Any,
    name: str = "tensor",
    *,
    enabled: bool | None = None,
    raise_on_error: bool = False,
) -> bool:
    """Optional NaN/Inf watchpoint with rich diagnostics.

    By default this is controlled by `ProjectConfig.check_numerics` (via `config.get_config()`).
    Returns True when finite, False when NaN/Inf is detected.
    """
    if enabled is None:
        from config import get_config

        enabled = bool(get_config().check_numerics)
    if not enabled:
        return True

    backend = "unknown"
    shape = "?"
    dtype = "?"
    nan_count = 0
    inf_count = 0

    if isinstance(x, torch.Tensor):
        backend = "torch"
        shape = tuple(x.shape)
        dtype = str(x.dtype)
        if torch.is_floating_point(x) or torch.is_complex(x):
            nan_count = int(torch.isnan(x).sum().item())
            inf_count = int(torch.isinf(x).sum().item())
    elif isinstance(x, np.ndarray):
        backend = "numpy"
        shape = x.shape
        dtype = str(x.dtype)
        if np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.complexfloating):
            nan_count = int(np.isnan(x).sum())
            inf_count = int(np.isinf(x).sum())
    else:
        try:
            arr = jnp.asarray(x)
        except Exception:
            arr = None
        if arr is not None:
            backend = "jax"
            shape = tuple(arr.shape)
            dtype = str(arr.dtype)
            if jnp.issubdtype(arr.dtype, jnp.floating) or jnp.issubdtype(arr.dtype, jnp.complexfloating):
                nan_count = int(jnp.isnan(arr).sum())
                inf_count = int(jnp.isinf(arr).sum())

    if nan_count == 0 and inf_count == 0:
        return True

    issues: list[str] = []
    if nan_count:
        issues.append("NaN")
    if inf_count:
        issues.append("Inf")
    issues_str = " and ".join(issues)
    console.print(f"[bold red]Numerics:[/bold red] {name} contains {issues_str}")

    table = Table(title=f"Numerics diagnostics: {name}", box=box.ROUNDED)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("backend", backend)
    table.add_row("shape", str(shape))
    table.add_row("dtype", dtype)
    table.add_row("nan_count", str(nan_count))
    table.add_row("inf_count", str(inf_count))
    console.print(table)

    if raise_on_error:
        raise ValueError(f"{name} contains {issues_str} (nan_count={nan_count}, inf_count={inf_count})")
    return False


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
        metric_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
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
    checkpoint_data = {"params": params, "step": step, "metrics": metrics or {}}

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
        console=console,
    )
    progress.start()
    progress.add_task(description, total=total)
    return progress


def safe_divide(numerator: jnp.ndarray, denominator: jnp.ndarray, epsilon: float = 1e-8) -> jnp.ndarray:
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
