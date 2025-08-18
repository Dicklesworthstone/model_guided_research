#!/usr/bin/env python3
"""Quick test to verify all demo modules can be imported."""

import importlib
import os

# Set JAX to CPU mode
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Suppress JAX warnings
os.environ['JAX_LOG_LEVEL'] = 'ERROR'

demos = [
    "matrix_exponential_gauge_learning",
    "ultrametric_worlds_and_p_adic_computation",
    "simplicial_complexes_and_higher_order_attention",
    "nonstandard_analysis_and_hyperreal_training",
    "octonionic_quaternionic_signal_flow",
    "ordinal_schedules_and_well_founded_optimization",
    "reversible_computation_and_measure_preserving_learning",
    "iterated_function_systems_and_fractal_memory",
    "knot_theoretic_programs_and_braid_based_attention",
    "surreal_numbers_transseries_and_scaling",
    "tropical_geometry_and_idempotent_algebra"
]

print("Testing module imports...")
print("=" * 50)

for demo_name in demos:
    try:
        module = importlib.import_module(demo_name)
        # Check if demo function exists
        if hasattr(module, 'demo'):
            print(f"✓ {demo_name}: imported successfully, demo() exists")
        else:
            print(f"⚠ {demo_name}: imported but no demo() function")
    except Exception as e:
        print(f"✗ {demo_name}: {str(e)[:100]}")

print("=" * 50)
print("Import test complete!")
