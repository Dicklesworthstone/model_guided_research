"""
Configuration management for model-guided research.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectConfig:
    """Global configuration for the project."""

    # Compute settings
    use_gpu: bool = False
    jax_precision: str = "float32"  # or "float64"
    random_seed: int = 42

    # Output settings
    verbose: bool = True
    save_outputs: bool = False
    output_dir: Path = Path("outputs")

    # Demo settings
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6

    @classmethod
    def from_file(cls, path: Path) -> "ProjectConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        # Convert any string paths back to Path objects based on field annotations
        from dataclasses import fields
        for field in fields(cls):
            if field.type == Path or (hasattr(field.type, '__origin__') and field.type.__origin__ is Path):
                if field.name in data and isinstance(data[field.name], str):
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
