import os
import time

import jax
import jax.numpy as jnp

from matrix_exponential_gauge_learning import GaugeAttentionBlock, GaugeTransformerConfig

# Force CPU for debugging
os.environ["JAX_PLATFORM_NAME"] = "cpu"


def debug_run():
    print("--- Starting Debug Run ---")

    # 1. Setup Config (Minimal)
    D = 64
    H = 4
    dh = 16
    N = 32

    cfg = GaugeTransformerConfig(
        d_model=D,
        n_heads=H,
        d_head=dh,
        mlp_hidden=4 * D,
        offsets=[-1, 1],  # Minimal offsets
        use_structured_blocks=True,
    )

    print(f"Config: N={N}, D={D}, H={H}")

    # 2. Initialize Block
    block = GaugeAttentionBlock(cfg, name="debug_block")
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1, N, D))

    print("Initializing block...")
    variables = block.init(key, x, train=False, return_debug=True)
    print("Block initialized.")

    # 3. Inspect Parameters to guess K
    # The time parameter determines loop length
    t_param = variables["params"]["time_scale"]
    t_val = float(jax.nn.softplus(t_param)[0])
    print(f"Time scale t ≈ {t_val:.4f}")

    # 4. Run Forward (JIT disabled to see progress)
    print("Running forward pass (JIT DISABLED)...")
    with jax.disable_jit():
        start = time.time()
        y, dbg = block.apply(variables, x, train=False, return_debug=True)
        end = time.time()

    print(f"Forward pass finished in {end - start:.4f}s")

    # 5. Check Diagnostics
    k_max = int(jnp.max(dbg["uniformization_K"]))
    print(f"Uniformization K_max used: {k_max}")
    print("Output shape:", y.shape)


if __name__ == "__main__":
    debug_run()
