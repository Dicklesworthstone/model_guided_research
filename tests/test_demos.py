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


@pytest.mark.parametrize(
    "attention_type",
    [
        "standard",
        "tropical",
        "ultrametric",
        "simplicial",
        "braid",
        "fractal",
        "quaternion",
        "octonion",
        "surreal",
        "reversible",
    ],
)
def test_nanochat_kv_cache_last_token_matches_full_forward(attention_type: str):
    """KV-cache decode (Tq==1) should match a full forward pass for the last token."""
    import torch

    from nanochat.engine import KVCache
    from nanochat.gpt import GPT, GPTConfig

    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=2,  # exercise GQA paths
        n_embd=64,
        attention_type=attention_type,
    )
    model = GPT(config).train(False)

    ids = torch.randint(0, config.vocab_size, (1, 8), dtype=torch.long)
    with torch.inference_mode():
        full_last = model(ids)[:, -1, :].float()

        kv_cache = KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,
            seq_len=ids.size(1),
            head_dim=config.n_embd // config.n_head,
            num_layers=config.n_layer,
        )
        _ = model(ids[:, :-1], kv_cache=kv_cache)
        cached_last = model(ids[:, -1:], kv_cache=kv_cache)[:, -1, :].float()

    # Shape-specific kernels (e.g. (Tq,Tk)=(1,8) vs (8,8)) can yield tiny numeric drift.
    torch.testing.assert_close(cached_last, full_last, rtol=1e-3, atol=1e-2)


def test_nanochat_braid_discrete_mode_records_schedule_and_matches_kv_cache():
    """Discrete braid mode should be KV-cache consistent and record a verifiable schedule."""
    import torch

    from nanochat.engine import KVCache
    from nanochat.gpt import GPT, GPTConfig

    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=2,  # exercise GQA paths
        n_embd=64,
        attention_type="braid",
        braid_mode="discrete",
        braid_tau=0.0,
        braid_crossing_law="ybe",
        braid_record_schedule=True,
        braid_verify=True,
    )
    model = GPT(config).train(False)

    ids = torch.randint(0, config.vocab_size, (1, 8), dtype=torch.long)
    with torch.inference_mode():
        full_last = model(ids)[:, -1, :].float()

        kv_cache = KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,
            seq_len=ids.size(1),
            head_dim=config.n_embd // config.n_head,
            num_layers=config.n_layer,
        )
        _ = model(ids[:, :-1], kv_cache=kv_cache)
        cached_last = model(ids[:, -1:], kv_cache=kv_cache)[:, -1, :].float()

    torch.testing.assert_close(cached_last, full_last, rtol=1e-3, atol=1e-2)

    attn0 = model.transformer["h"][0].attn
    debug = getattr(attn0, "last_braid_debug", None)
    require(isinstance(debug, dict), "Expected braid attention to record last_braid_debug in discrete mode")
    for key in ("order", "selected", "k", "scores", "tau", "crossing_law"):
        require(key in debug, f"Missing debug key: {key}")
    order = debug["order"]
    selected = debug["selected"]
    k = debug["k"]
    require(isinstance(order, torch.Tensor) and isinstance(selected, torch.Tensor) and isinstance(k, torch.Tensor), "Bad debug tensor types")
    require(order.shape[-1] == ids.size(1), "Expected schedule to cover all causal keys for the last token")


def test_nanochat_standard_attention_entropy_records_per_head_stats():
    """Standard attention should optionally record a finite per-head entropy summary."""
    import torch

    from nanochat.gpt import GPT, GPTConfig

    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=2,  # exercise GQA paths
        n_embd=64,
        attention_type="standard",
        standard_record_attn_entropy=True,
    )
    model = GPT(config).train(False)

    ids = torch.randint(0, config.vocab_size, (1, 8), dtype=torch.long)
    with torch.inference_mode():
        _ = model(ids)

    attn0 = model.transformer["h"][0].attn
    entropy = getattr(attn0, "attn_entropy_head_mean", None)
    require(torch.is_tensor(entropy), "Expected attn_entropy_head_mean to be a tensor")
    require(tuple(entropy.shape) == (config.n_head,), f"Expected entropy shape ({config.n_head},), got {tuple(entropy.shape)}")
    require(bool(torch.isfinite(entropy).all().item()), "Expected per-head entropy values to be finite")


@pytest.mark.parametrize(
    "attention_type",
    [
        "standard",
        "tropical",
        "ultrametric",
        "simplicial",
        "braid",
        "fractal",
        "quaternion",
        "octonion",
        "surreal",
        "reversible",
    ],
)
def test_nanochat_kv_cache_chunk_is_causal(attention_type: str):
    """Changing a future token in a chunk must not affect earlier chunk outputs."""
    import torch

    from nanochat.engine import KVCache
    from nanochat.gpt import GPT, GPTConfig

    torch.manual_seed(1)
    config = GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=2,  # exercise GQA paths
        n_embd=64,
        attention_type=attention_type,
    )
    model = GPT(config).train(False)

    prefix = torch.randint(0, config.vocab_size, (1, 5), dtype=torch.long)
    chunk = torch.randint(0, config.vocab_size, (1, 3), dtype=torch.long)
    chunk_alt = chunk.clone()
    chunk_alt[0, -1] = (chunk_alt[0, -1] + 1) % config.vocab_size

    def run(prefix_ids: torch.Tensor, chunk_ids: torch.Tensor) -> torch.Tensor:
        kv_cache = KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,
            seq_len=prefix_ids.size(1) + chunk_ids.size(1),
            head_dim=config.n_embd // config.n_head,
            num_layers=config.n_layer,
        )
        _ = model(prefix_ids, kv_cache=kv_cache)
        return model(chunk_ids, kv_cache=kv_cache)[:, :-1, :].float()

    with torch.inference_mode():
        out_a = run(prefix, chunk)
        out_b = run(prefix, chunk_alt)

    torch.testing.assert_close(out_a, out_b, rtol=1e-4, atol=1e-4)


def test_nanochat_ultrametric_trie_decode_matches_hard_kernel():
    """Trie-mode ultrametric decode should match the (hard-digit) kernel baseline."""
    import torch

    from nanochat.engine import KVCache
    from nanochat.gpt import GPT, GPTConfig

    torch.manual_seed(0)
    ids = torch.randint(0, 128, (1, 12), dtype=torch.long)

    def run(mode: str) -> torch.Tensor:
        cfg = GPTConfig(
            sequence_len=32,
            vocab_size=128,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=64,
            attention_type="ultrametric",
            ultrametric_mode=mode,
            ultrametric_hard_digits=True,
            ultrametric_lcp_beta=128.0,  # make mismatches effectively 0-prob so kernel == exact LCP
        )
        model = GPT(cfg).train(False)
        kv_cache = KVCache(
            batch_size=1,
            num_heads=cfg.n_kv_head,
            seq_len=ids.size(1),
            head_dim=cfg.n_embd // cfg.n_head,
            num_layers=cfg.n_layer,
        )
        _ = model(ids[:, :-1], kv_cache=kv_cache)
        return model(ids[:, -1:], kv_cache=kv_cache)[:, -1, :].float()

    with torch.inference_mode():
        torch.manual_seed(0)
        out_kernel = run("kernel")
        torch.manual_seed(0)
        out_trie = run("trie")

    torch.testing.assert_close(out_trie, out_kernel, rtol=1e-3, atol=1e-2)


def test_kv_cache_prefill_expands_batch_dimension():
    """KVCache.prefill should broadcast a batch-1 prefix to a larger batch."""
    import torch

    from nanochat.engine import KVCache

    torch.manual_seed(0)
    other = KVCache(batch_size=1, num_heads=2, seq_len=8, head_dim=4, num_layers=2)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    _ = other.insert_kv(0, k, v)
    _ = other.insert_kv(1, k, v)
    require(other.get_pos() == 3, "Expected other KVCache pos to advance after last layer insert")

    expanded = KVCache(batch_size=2, num_heads=2, seq_len=8, head_dim=4, num_layers=2)
    expanded.prefill(other)
    require(expanded.get_pos() == other.get_pos(), "Prefilled KVCache pos mismatch")
    require(expanded.kv_cache is not None and other.kv_cache is not None, "KV caches must be initialized after prefill")

    # Both batch rows should match the single source prefix exactly.
    torch.testing.assert_close(expanded.kv_cache[:, :, 0, :, : other.pos, :], other.kv_cache[:, :, 0, :, : other.pos, :])
    torch.testing.assert_close(expanded.kv_cache[:, :, 1, :, : other.pos, :], other.kv_cache[:, :, 0, :, : other.pos, :])


def test_cli_regressions_smoke(tmp_path: Path):
    """Regression dashboard CLI should run on minimal synthetic summaries."""
    import json

    from typer.testing import CliRunner

    import cli as mgr_cli

    base_dir = tmp_path / "baseline"
    cand_dir = tmp_path / "candidate"
    base_dir.mkdir(parents=True, exist_ok=True)
    cand_dir.mkdir(parents=True, exist_ok=True)

    base_summary = {
        "git": {"commit": "abc123", "dirty": False},
        "config": {"attention_type": "standard", "use_flex_attention": False},
        "results": {"losses": [3.0, 2.0], "tokens_per_second": 1000.0, "tflops_per_second_est": 1.0, "peak_memory_allocated_gb": 2.0},
    }
    cand_summary = {
        "git": {"commit": "def456", "dirty": True},
        "config": {"attention_type": "standard", "use_flex_attention": True},
        "results": {"losses": [3.0, 2.1], "tokens_per_second": 950.0, "tflops_per_second_est": 0.95, "peak_memory_allocated_gb": 2.2},
    }
    (base_dir / "summary.json").write_text(json.dumps(base_summary), encoding="utf-8")
    (cand_dir / "summary.json").write_text(json.dumps(cand_summary), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        mgr_cli.app,
        [
            "regressions",
            "-b",
            str(base_dir),
            "-c",
            str(cand_dir),
            "--no-write-artifacts",
            "--no-html",
        ],
    )
    require(result.exit_code == 0, f"regressions CLI failed: {result.stdout}\n{result.exception}")

    result_fail = runner.invoke(
        mgr_cli.app,
        [
            "regressions",
            "-b",
            str(base_dir),
            "-c",
            str(cand_dir),
            "--no-write-artifacts",
            "--no-html",
            "--fail-on-regression",
        ],
    )
    require(
        result_fail.exit_code == 1,
        f"regressions --fail-on-regression should exit 1: {result_fail.stdout}\n{result_fail.exception}",
    )


def test_nanochat_gpt_synaptic_kv_cache_chunk_is_causal():
    """Changing a future token in a chunk must not affect earlier chunk outputs (synaptic)."""
    import torch

    from nanochat.engine import KVCache
    from nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
    from nanochat.synaptic import SynapticConfig

    torch.manual_seed(2)
    syn_cfg = SynapticConfig(use_flex_attention=False, stochastic_train_frac=0.0)
    config = GPTSynapticConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=2,  # exercise GQA paths
        n_embd=64,
        syn_cfg=syn_cfg,
        dropout=0.0,
    )
    model = GPTSynaptic(config).train(False)

    prefix = torch.randint(0, config.vocab_size, (1, 5), dtype=torch.long)
    chunk = torch.randint(0, config.vocab_size, (1, 3), dtype=torch.long)
    chunk_alt = chunk.clone()
    chunk_alt[0, -1] = (chunk_alt[0, -1] + 1) % config.vocab_size

    def run(prefix_ids: torch.Tensor, chunk_ids: torch.Tensor) -> torch.Tensor:
        kv_cache = KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,
            seq_len=prefix_ids.size(1) + chunk_ids.size(1),
            head_dim=config.n_embd // config.n_head,
            num_layers=config.n_layer,
        )
        _ = model(prefix_ids, kv_cache=kv_cache)
        logits, _ = model(chunk_ids, kv_cache=kv_cache)
        return logits[:, :-1, :].float()

    with torch.inference_mode():
        out_a = run(prefix, chunk)
        out_b = run(prefix, chunk_alt)

    torch.testing.assert_close(out_a, out_b, rtol=1e-4, atol=1e-4)


def test_nanochat_gpt_synaptic_flexattention_smoke():
    """FlexAttention smoke test: forward/backward should run without NaNs on CUDA (skips on non-GPU CI)."""
    import torch

    try:
        from torch.nn.attention.flex_attention import flex_attention  # noqa: F401
    except Exception:
        pytest.skip("FlexAttention is unavailable in this torch build.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FlexAttention smoke test.")

    from nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
    from nanochat.synaptic import SynapticConfig

    torch.manual_seed(0)
    syn_cfg = SynapticConfig(use_flex_attention=True, stochastic_train_frac=0.0)
    config = GPTSynapticConfig(
        sequence_len=128,
        vocab_size=1024,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        syn_cfg=syn_cfg,
        dropout=0.0,
    )
    device = torch.device("cuda:0")
    dtype = torch.float16

    model = GPTSynaptic(config).to(device).to(dtype).train(True)
    model = torch.compile(model)

    x = torch.randint(0, config.vocab_size, (2, 128), device=device, dtype=torch.long)
    y = torch.randint(0, config.vocab_size, (2, 128), device=device, dtype=torch.long)

    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        logits, loss = model(x, y)
    require(torch.isfinite(loss).item(), "FlexAttention loss must be finite")
    loss.backward()

    for p in model.parameters():
        if p.grad is None:
            continue
        require(torch.isfinite(p.grad).all().item(), "FlexAttention gradients must be finite")


def test_nanochat_gauge_attention_rejects_kv_cache():
    import torch

    from nanochat.engine import KVCache
    from nanochat.gpt import GPT, GPTConfig

    config = GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=1,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        attention_type="gauge",
    )
    model = GPT(config).train(False)
    ids = torch.randint(0, config.vocab_size, (1, 4), dtype=torch.long)
    kv_cache = KVCache(
        batch_size=1,
        num_heads=config.n_kv_head,
        seq_len=ids.size(1),
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer,
    )
    with pytest.raises(NotImplementedError):
        _ = model(ids, kv_cache=kv_cache)


def test_execution_execute_code_allows_disabling_memory_limit():
    from nanochat.execution import execute_code

    result = execute_code("print('hello')", timeout=2.0, maximum_memory_bytes=None)
    require(result.success, f"execute_code failed unexpectedly: {result}")
    require(result.stdout.strip() == "hello", f"unexpected stdout: {result.stdout!r}")


def test_nanochat_synaptic_modules_import():
    # These modules are optional/experimental, but should import cleanly.
    import nanochat.gpt_synaptic  # noqa: F401
    import nanochat.synaptic  # noqa: F401


def test_postsynaptic_hebb_fast_uses_mean_over_presyn_dim():
    import torch

    from nanochat.synaptic import PostsynapticHebb, SynapticConfig

    # Use d_k == d_v to catch the previous (incorrect) diag(dW) update rule.
    cfg = SynapticConfig(rank_eligibility=2)
    post = PostsynapticHebb(d_k=4, d_v=4, cfg=cfg).train(False)

    traceU = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=torch.float32,
    )
    traceV = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.5, -0.5, 1.5, -1.5],
        ],
        dtype=torch.float32,
    )

    expected_delta = traceU.mean(dim=0) @ traceV
    post.hebb_fast(traceU, traceV)
    torch.testing.assert_close(post.fast, cfg.post_fast_lr * expected_delta)


def test_postsynaptic_hebb_raises_on_trace_shape_mismatch():
    import torch

    from nanochat.synaptic import PostsynapticHebb, SynapticConfig

    cfg = SynapticConfig(rank_eligibility=2)
    post = PostsynapticHebb(d_k=4, d_v=5, cfg=cfg).train(False)

    # Rank mismatch (R differs): should raise rather than silently no-op.
    traceU = torch.zeros(4, 2, dtype=torch.float32)
    traceV = torch.zeros(3, 5, dtype=torch.float32)
    with pytest.raises(ValueError):
        post.hebb_fast(traceU, traceV)


def test_synaptic_linear_updates_postsynaptic_state():
    import torch

    from nanochat.synaptic import SynapticConfig, SynapticLinear

    torch.manual_seed(0)
    cfg = SynapticConfig(rank_eligibility=3, camkii_thr=-1.0, stochastic_train_frac=0.0)
    layer = SynapticLinear(in_features=6, out_features=4, cfg=cfg, bias=False).train(False)

    with torch.no_grad():
        layer.w_slow.fill_(0.1)
        layer.w_fast.zero_()
        layer.post.U.zero_()
        layer.post.V.zero_()
        layer.post.fast.zero_()
        layer.post.slow.zero_()
        layer.u_buf.zero_()
        layer.v_buf.zero_()

    x = torch.ones(8, 6, dtype=torch.float32)
    c = torch.ones(8, dtype=torch.float32)
    e = torch.ones(8, dtype=torch.float32)
    y = layer(x, calcium=c, energy=e, update_mem=True)

    require(torch.isfinite(y).all().item(), "SynapticLinear output must be finite")
    require(torch.isfinite(layer.post.fast).all().item(), "Postsynaptic fast must be finite")
    require(torch.isfinite(layer.post.slow).all().item(), "Postsynaptic slow must be finite")
    require(layer.post.fast.abs().sum().item() > 0.0, "Postsynaptic fast update should not be a no-op")
    require(layer.post.slow.abs().sum().item() > 0.0, "Postsynaptic slow consolidation should not be a no-op")


def test_tropical_attention_gauge_fix_is_invariant_to_per_vector_shifts():
    import torch

    from nanochat.tropical_attention_torch import tropical_max_plus_attention

    torch.manual_seed(0)
    B, H, T, D = 2, 3, 5, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    y0, gamma0 = tropical_max_plus_attention(
        q,
        k,
        v,
        gauge_fix=True,
        score_center=True,
        return_margins=True,
    )

    # Apply independent (per-vector) gauge shifts; gauge-fixing should remove them.
    q_shift = torch.randn(B, H, T, 1)
    k_shift = torch.randn(B, H, T, 1)
    v_shift = torch.randn(B, H, T, 1)
    y1, gamma1 = tropical_max_plus_attention(
        q + q_shift,
        k + k_shift,
        v + v_shift,
        gauge_fix=True,
        score_center=True,
        return_margins=True,
    )

    torch.testing.assert_close(y0, y1)
    torch.testing.assert_close(gamma0, gamma1)


def test_execution_sandbox_smoke():
    from nanochat.execution import execute_code

    result = execute_code("print('hello')", timeout=2.0, maximum_memory_bytes=128 * 1024 * 1024)
    require(result.success is True, f"Expected execute_code to succeed, got: {result!r}")
    require("hello" in result.stdout, f"Expected stdout to contain 'hello', got: {result.stdout!r}")


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main(sys.argv[1:] or [__file__]))
