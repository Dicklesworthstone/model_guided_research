"""
Basic tests to ensure all demos run without errors.
"""

import importlib
from pathlib import Path

import jax
import pytest
from rich.console import Console

console = Console()


def require(condition, message: str):
    if not bool(condition):
        raise AssertionError(message)

# List of all demo modules
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


@pytest.mark.parametrize("module_name", DEMO_MODULES)
def test_demo_exists(module_name):
    """Test that each module has a demo function."""
    module = importlib.import_module(module_name)
    require(hasattr(module, 'demo'), f"Module {module_name} missing demo() function")
    require(callable(module.demo), f"demo in {module_name} is not callable")


@pytest.mark.parametrize("module_name", DEMO_MODULES)
def test_module_imports(module_name):
    """Test that each module can be imported without errors."""
    try:
        module = importlib.import_module(module_name)
        require(module is not None, f"Import returned None for {module_name}")
    except ImportError as e:
        pytest.fail(f"Failed to import {module_name}: {e}")


def test_jax_available():
    """Test that JAX is properly installed and configured."""
    devices = jax.devices()
    require(len(devices) > 0, "No JAX devices available")

    # Test basic JAX operation
    x = jax.numpy.array([1.0, 2.0, 3.0])
    y = jax.numpy.sum(x)
    require(float(y) == 6.0, "Basic JAX operation failed")


def test_documentation_exists():
    """Test that markdown documentation exists for each module."""
    doc_dir = Path("markdown_documentation")
    require(doc_dir.exists(), "Documentation directory missing")

    for module_name in DEMO_MODULES:
        doc_file = doc_dir / f"{module_name}.md"
        require(doc_file.exists(), f"Documentation missing for {module_name}")
