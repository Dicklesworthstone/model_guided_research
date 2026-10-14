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
    jax_debug_infs: bool = False  # Enable Inf checking in JAX
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
                field_type == Path
                or (origin is not None and Path in args)
                or (isinstance(field_type, type) and issubclass(field_type, Path))
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
        with open(path, "w") as f:
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

        if self.jax_debug_infs:
            jax.config.update("jax_debug_infs", True)

        if self.jax_disable_jit:
            jax.config.update("jax_disable_jit", True)

        if self.check_numerics:
            # `check_numerics` is a higher-level "turn on the alarms" switch.
            # Enable both NaN and Inf checks in JAX regardless of the lower-level flags.
            jax.config.update("jax_debug_nans", True)
            jax.config.update("jax_debug_infs", True)

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


# ---------------------------------------------------------------------------
# Declarative Demo Feature-Flag Registry (br: model_guided_research-rz8.6)
# ---------------------------------------------------------------------------

import difflib
import os
import warnings
from typing import Any


@dataclass(frozen=True)
class DemoFlagSpec:
    """Metadata and validation contract for a JAX demo environment flag."""

    name: str
    demo: str
    type: str  # "bool" | "int" | "float" | "enum" | "str"
    default: Any
    description: str
    choices: tuple[str, ...] | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


DEMO_FLAG_PREFIXES: tuple[str, ...] = ("GAUGE_", "TROP_", "REV_", "ULTRA_", "BRAID_")

# Internal scratch/test variables that are not user-facing demo flags.
_INTERNAL_DEMO_ENV_VARS: frozenset[str] = frozenset({"GAUGE_UNIF_REP_SEED", "ULTRA_HEADS"})

_DEMO_NAME_ALIASES: dict[str, str] = {
    "gauge": "matrix-gauge",
    "matrix-gauge": "matrix-gauge",
    "matrix_exponential_gauge_learning": "matrix-gauge",
    "tropical": "tropical",
    "tropical_geometry_and_idempotent_algebra": "tropical",
    "reversible": "reversible",
    "reversible_computation_and_measure_preserving_learning": "reversible",
    "ultrametric": "ultrametric",
    "ultrametric_worlds_and_p_adic_computation": "ultrametric",
    "braid": "knot-braid",
    "knot-braid": "knot-braid",
    "knot_theoretic_programs_and_braid_based_attention": "knot-braid",
}

DEMO_FLAG_REGISTRY: dict[str, DemoFlagSpec] = {
    # --- matrix-gauge (9 flags) ---
    "GAUGE_ALT_STRUCT": DemoFlagSpec(
        name="GAUGE_ALT_STRUCT",
        demo="matrix-gauge",
        type="bool",
        default=False,
        description="Alternate structured and unstructured Lie generators across odd/even blocks",
    ),
    "GAUGE_BCH_FUSION": DemoFlagSpec(
        name="GAUGE_BCH_FUSION",
        demo="matrix-gauge",
        type="bool",
        default=True,
        description="Run Baker-Campbell-Hausdorff generator fusion mini-experiment",
    ),
    "GAUGE_SAMPLE_SCHEDULE": DemoFlagSpec(
        name="GAUGE_SAMPLE_SCHEDULE",
        demo="matrix-gauge",
        type="bool",
        default=False,
        description="Compare deterministic vs multi-repetition stochastic uniformization schedule",
    ),
    "GAUGE_SAMPLE_TRAIN": DemoFlagSpec(
        name="GAUGE_SAMPLE_TRAIN",
        demo="matrix-gauge",
        type="bool",
        default=False,
        description="Run deterministic vs stochastic uniformization training comparison",
    ),
    "GAUGE_STRUCTURED": DemoFlagSpec(
        name="GAUGE_STRUCTURED",
        demo="matrix-gauge",
        type="bool",
        default=False,
        description="Enable structured SO/SPD/Sp Lie algebra channel blocks",
    ),
    "GAUGE_UNIF_CAP_K": DemoFlagSpec(
        name="GAUGE_UNIF_CAP_K",
        demo="matrix-gauge",
        type="int",
        default=None,
        description="Optional cap on Poisson truncation depth K in uniformization (unset/empty = uncapped)",
    ),
    "GAUGE_UNIF_OFF": DemoFlagSpec(
        name="GAUGE_UNIF_OFF",
        demo="matrix-gauge",
        type="bool",
        default=False,
        description="Bypass uniformization mixing in gauge block forward pass (ablation)",
    ),
    "GAUGE_UNIF_REPS": DemoFlagSpec(
        name="GAUGE_UNIF_REPS",
        demo="matrix-gauge",
        type="int",
        default=1,
        description="Number of Monte Carlo repetitions when stochastic uniformization sampling is enabled",
    ),
    "GAUGE_UNIF_SAMPLE": DemoFlagSpec(
        name="GAUGE_UNIF_SAMPLE",
        demo="matrix-gauge",
        type="bool",
        default=False,
        description="Enable stochastic single-chain Poisson uniformization sampling",
    ),
    # --- tropical (8 flags) ---
    "TROP_SPARSE_KS": DemoFlagSpec(
        name="TROP_SPARSE_KS",
        demo="tropical",
        type="str",
        default="2,4,8",
        description="Comma-separated support sizes k for sparse tropical mixture sweep",
    ),
    "TROP_SPARSE_LAMBDAS": DemoFlagSpec(
        name="TROP_SPARSE_LAMBDAS",
        demo="tropical",
        type="str",
        default="0.25,0.5,1.0",
        description="Comma-separated mixture weights lambda for sparse tropical mixture sweep",
    ),
    "TROP_SPARSE_LR": DemoFlagSpec(
        name="TROP_SPARSE_LR",
        demo="tropical",
        type="float",
        default=0.05,
        description="Step size for sparse-support tropical classifier projection training",
    ),
    "TROP_SPARSE_MIX": DemoFlagSpec(
        name="TROP_SPARSE_MIX",
        demo="tropical",
        type="bool",
        default=False,
        description="Enable sparse mixtures of tropical polynomials sweep",
    ),
    "TROP_SPARSE_STEPS": DemoFlagSpec(
        name="TROP_SPARSE_STEPS",
        demo="tropical",
        type="int",
        default=5,
        description="Number of projected top-k update steps in sparse tropical training",
    ),
    "TROP_SPARSE_TRAIN": DemoFlagSpec(
        name="TROP_SPARSE_TRAIN",
        demo="tropical",
        type="bool",
        default=False,
        description="Enable sparse-support top-k classifier training experiment",
    ),
    "TROP_SPARSE_TRAIN_K": DemoFlagSpec(
        name="TROP_SPARSE_TRAIN_K",
        demo="tropical",
        type="int",
        default=8,
        description="Default top-k support size for sparse tropical training when TROP_SPARSE_TRAIN_KS is unset",
    ),
    "TROP_SPARSE_TRAIN_KS": DemoFlagSpec(
        name="TROP_SPARSE_TRAIN_KS",
        demo="tropical",
        type="str",
        default="",
        description="Optional comma-separated top-k support grid overriding TROP_SPARSE_TRAIN_K",
    ),
    # --- reversible (6 flags) ---
    "REV_GENERATING": DemoFlagSpec(
        name="REV_GENERATING",
        demo="reversible",
        type="bool",
        default=False,
        description="Enable generating-function symplectic coupling step",
    ),
    "REV_GEN_PARAMS": DemoFlagSpec(
        name="REV_GEN_PARAMS",
        demo="reversible",
        type="bool",
        default=False,
        description="Derive generating-function coefficients (a, b, c) from learned coupling mix parameters",
    ),
    "REV_GEN_VJP": DemoFlagSpec(
        name="REV_GEN_VJP",
        demo="reversible",
        type="bool",
        default=False,
        description="Use custom VJP rule for the generating-function symplectic step",
    ),
    "REV_GIVENS": DemoFlagSpec(
        name="REV_GIVENS",
        demo="reversible",
        type="bool",
        default=False,
        description="Use strict Givens-rotation orthogonal mixing (exact inverse, det=1)",
    ),
    "REV_LAYER_CERT": DemoFlagSpec(
        name="REV_LAYER_CERT",
        demo="reversible",
        type="bool",
        default=False,
        description="Run per-layer Cayley orthogonality and invertibility certificate checks",
    ),
    "REV_PARETO": DemoFlagSpec(
        name="REV_PARETO",
        demo="reversible",
        type="bool",
        default=False,
        description="Run Cayley composition steps vs depth compute/memory Pareto sweep",
    ),
    # --- ultrametric (5 flags) ---
    "ULTRA_FUSE": DemoFlagSpec(
        name="ULTRA_FUSE",
        demo="ultrametric",
        type="bool",
        default=False,
        description="Fuse multi-head ultrametric candidates by maximizing summed head similarity",
    ),
    "ULTRA_PACKED": DemoFlagSpec(
        name="ULTRA_PACKED",
        demo="ultrametric",
        type="bool",
        default=False,
        description="Use packed bit-trie implementation for ultrametric LCP attention tasks",
    ),
    "ULTRA_PACKED_ARRAYS": DemoFlagSpec(
        name="ULTRA_PACKED_ARRAYS",
        demo="ultrametric",
        type="bool",
        default=False,
        description="Use flat array-backed buckets with O(1) prefix occupancy rank/test",
    ),
    "ULTRA_SCALE_COMPARE": DemoFlagSpec(
        name="ULTRA_SCALE_COMPARE",
        demo="ultrametric",
        type="bool",
        default=False,
        description="Compare packed bit-trie vs dictionary-backed scaling across sequence lengths",
    ),
    "ULTRA_TUNE_P": DemoFlagSpec(
        name="ULTRA_TUNE_P",
        demo="ultrametric",
        type="bool",
        default=False,
        description="Run prime-base p in {3, 5, 7} tuner on a Task A validation split",
    ),
    # --- knot-braid (1 flag) ---
    "BRAID_CROSSING_LAW": DemoFlagSpec(
        name="BRAID_CROSSING_LAW",
        demo="knot-braid",
        type="enum",
        default="restricted",
        choices=("restricted", "ybe", "heuristic"),
        description="Braid crossing update law ('restricted', 'ybe', or 'heuristic')",
    ),
}


def normalize_demo_name(demo: str | None) -> str | None:
    """Normalize a demo CLI alias or module name to its canonical registry demo key."""
    if demo is None:
        return None
    key = demo.strip().lower()
    return _DEMO_NAME_ALIASES.get(key, key)


def get_flag(name: str, default: Any = None) -> Any:
    """Read and validate a registered demo flag from ``os.environ``.

    Returns the parsed value (matching ``spec.type``) or ``spec.default`` when
    unset. Raises ``KeyError`` if ``name`` is not in ``DEMO_FLAG_REGISTRY`` and
    ``ValueError`` with a descriptive message if the environment variable has an
    invalid format for its declared type.
    """
    spec = DEMO_FLAG_REGISTRY.get(name)
    if spec is None:
        raise KeyError(f"Unknown demo flag {name!r}; registered flags: {sorted(DEMO_FLAG_REGISTRY)}")

    fallback = spec.default if default is None else default
    raw = os.environ.get(name)
    if raw is None:
        return fallback

    s = raw.strip()
    if spec.type == "bool":
        if s == "":
            return bool(fallback)
        low = s.lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ValueError(
            f"Invalid value for boolean flag {name}={raw!r}: expected '0' or '1' (or true/false)"
        )

    if spec.type == "int":
        if s == "":
            return fallback
        try:
            return int(s)
        except ValueError:
            raise ValueError(
                f"Invalid value for integer flag {name}={raw!r}: expected an integer (e.g. {name}=1)"
            ) from None

    if spec.type == "float":
        if s == "":
            return fallback
        try:
            return float(s)
        except ValueError:
            raise ValueError(
                f"Invalid value for float flag {name}={raw!r}: expected a float (e.g. {name}=0.05)"
            ) from None

    if spec.type == "enum":
        if s == "":
            return fallback
        if spec.choices is not None and s not in spec.choices:
            choices_str = ", ".join(repr(c) for c in spec.choices)
            raise ValueError(
                f"Invalid value for enum flag {name}={raw!r}: expected one of ({choices_str})"
            )
        return s

    if spec.type == "str":
        if name in ("TROP_SPARSE_KS", "TROP_SPARSE_TRAIN_KS") and s:
            try:
                for part in s.split(","):
                    if part.strip():
                        int(part.strip())
            except ValueError:
                raise ValueError(
                    f"Invalid value for {name}={raw!r}: expected comma-separated integers (e.g. '2,4,8')"
                ) from None
        elif name == "TROP_SPARSE_LAMBDAS" and s:
            try:
                for part in s.split(","):
                    if part.strip():
                        float(part.strip())
            except ValueError:
                raise ValueError(
                    f"Invalid value for {name}={raw!r}: expected comma-separated floats (e.g. '0.25,0.5,1.0')"
                ) from None
        return raw

    return raw


def validate_demo_flags(demo: str | None = None) -> list[str]:
    """Validate demo environment variables at demo startup.

    1. Warns loudly (Rich yellow panel + ``UserWarning``) on any set environment
       variable that prefix-matches a known demo family (``GAUGE_``, ``TROP_``,
       ``REV_``, ``ULTRA_``, ``BRAID_``) but is not in ``DEMO_FLAG_REGISTRY``.
    2. Validates the value types of all set registered flags (scoped to ``demo``
       when provided, or all registered flags when ``demo is None``), raising
       ``ValueError`` on malformed values.

    Returns the list of warning messages emitted for unknown/typo flags.
    """
    canon_demo = normalize_demo_name(demo)
    warning_messages: list[str] = []

    for env_key, env_val in sorted(os.environ.items()):
        if not env_key.startswith(DEMO_FLAG_PREFIXES):
            continue
        if env_key in DEMO_FLAG_REGISTRY or env_key in _INTERNAL_DEMO_ENV_VARS:
            continue
        close = difflib.get_close_matches(env_key, list(DEMO_FLAG_REGISTRY.keys()), n=1, cutoff=0.6)
        hint = f" Did you mean '{close[0]}'?" if close else " Run 'mgr flags' to list valid flags."
        msg = f"Unknown demo flag {env_key}={env_val!r} is set and will be ignored.{hint}"
        warning_messages.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
        try:
            from rich.console import Console
            from rich.panel import Panel

            Console().print(
                Panel(
                    f"[bold yellow]{msg}[/bold yellow]",
                    title="Unrecognized Demo Flag",
                    border_style="yellow",
                )
            )
        except Exception:
            print(f"WARNING: {msg}")

    for flag_name, spec in DEMO_FLAG_REGISTRY.items():
        if canon_demo is not None and spec.demo != canon_demo:
            continue
        if flag_name in os.environ:
            get_flag(flag_name)

    return warning_messages

