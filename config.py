"""
Configuration management for model-guided research.
"""

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import get_args, get_origin


@dataclass
class ProjectConfig:
    """Global configuration for the project."""

    # Compute settings
    use_gpu: bool = False
    jax_precision: str = "float32"  # or "float64"
    random_seed: int = 42
    jax_debug_nans: bool = False  # Enable NaN checking in JAX
    jax_disable_jit: bool = False  # Disable JIT for debugging

    # Output settings
    verbose: bool = True
    verbose_level: int = 1  # 0=silent, 1=normal, 2=detailed, 3=debug
    save_outputs: bool = False
    output_dir: Path = Path("outputs")
    save_checkpoints: bool = False
    checkpoint_dir: Path = Path("checkpoints")

    # Logging settings
    log_metrics: bool = True
    log_interval: int = 100  # Log every N steps
    use_rich_output: bool = True  # Use rich formatting for console output
    show_progress_bars: bool = True

    # Debug settings
    debug_mode: bool = False
    check_numerics: bool = False  # Check for NaN/Inf in computations
    profile_performance: bool = False  # Enable performance profiling

    # Demo settings
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    early_stopping_patience: int = 10

    # Training settings
    default_learning_rate: float = 1e-3
    default_batch_size: int = 32
    gradient_clip_norm: float | None = None

    @classmethod
    def from_file(cls, path: Path) -> "ProjectConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        # Convert any string paths back to Path objects based on field annotations
        for field in fields(cls):
            # Check if the field type involves Path
            field_type = field.type
            origin = get_origin(field_type)
            args = get_args(field_type)

            # Handle direct Path, Optional[Path], Union[Path, ...], etc.
            is_path_type = (
                field_type == Path or
                (origin is not None and Path in args) or
                (isinstance(field_type, type) and issubclass(field_type, Path))
            )

            if is_path_type and field.name in data and isinstance(data[field.name], str):
                data[field.name] = Path(data[field.name])
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                data[key] = str(value)
            else:
                data[key] = value
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def setup_jax(self) -> None:
        """Configure JAX based on settings."""
        import jax

        if self.jax_precision == "float64":
            jax.config.update("jax_enable_x64", True)

        if not self.use_gpu:
            jax.config.update("jax_platform_name", "cpu")

        if self.jax_debug_nans:
            jax.config.update("jax_debug_nans", True)

        if self.jax_disable_jit:
            jax.config.update("jax_disable_jit", True)

        # Create output directories if needed
        if self.save_outputs:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.save_checkpoints:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config: ProjectConfig | None = None


def get_config() -> ProjectConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = ProjectConfig()
    return _config


def set_config(config: ProjectConfig) -> None:
    """Set global configuration instance."""
    global _config
    _config = config
    _config.setup_jax()
