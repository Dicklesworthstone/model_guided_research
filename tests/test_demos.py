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
    "clifford_algebra_and_geometric_attention",
    "hyperbolic_geometry_and_negative_curvature_attention",
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
    require(hasattr(module, "demo"), f"Module {module_name} missing demo() function")
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


def test_attention_schedule_instantiates_per_layer_dispatch():
    import torch

    from nanochat.gpt import GPT, GPTConfig, resolve_attention_schedule

    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=4,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
        attention_type="standard,tropical",
    )
    model = GPT(config).train(False)
    assert resolve_attention_schedule(config) == ["standard", "tropical", "standard", "tropical"]
    assert [block.attention_type for block in model.transformer.h] == [
        "standard",
        "tropical",
        "standard",
        "tropical",
    ]


def test_attention_schedule_validation_errors_name_bad_layer():
    from nanochat.gpt import GPT, GPTConfig

    with pytest.raises(ValueError, match="length 2 does not evenly divide n_layer=3"):
        GPT(GPTConfig(n_layer=3, n_head=4, n_kv_head=2, n_embd=64, attention_type="standard,tropical"))

    with pytest.raises(ValueError, match="entry 1 has unknown mechanism 'bogus'"):
        GPT(GPTConfig(n_layer=2, n_head=4, n_kv_head=2, n_embd=64, attention_type="standard,bogus"))

    with pytest.raises(ValueError, match="layer 1 uses octonion.*head_dim % 8 == 0"):
        GPT(GPTConfig(n_layer=2, n_head=4, n_kv_head=2, n_embd=24, attention_type="standard,octonion"))


@pytest.mark.parametrize(
    ("attention_type", "entry_index"),
    [
        ("standard,", 1),
        (",standard", 0),
        ("standard,,tropical", 1),
        (["standard", ""], 1),
    ],
)
def test_attention_schedule_validation_rejects_empty_entries(attention_type, entry_index):
    from nanochat.gpt import GPT, GPTConfig

    with pytest.raises(ValueError, match=f"attention schedule entry {entry_index} is empty"):
        GPT(GPTConfig(n_layer=2, n_head=4, n_kv_head=2, n_embd=64, attention_type=attention_type))


def test_attention_schedule_kv_cache_single_and_chunk_match_full_forward():
    import torch

    from nanochat.engine import KVCache
    from nanochat.gpt import GPT, GPTConfig

    torch.manual_seed(2)
    config = GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
        attention_type=["standard", "tropical"],
    )
    model = GPT(config).train(False)
    ids = torch.randint(0, config.vocab_size, (1, 8), dtype=torch.long)

    with torch.inference_mode():
        full = model(ids).float()

        kv_cache = KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,
            seq_len=ids.size(1),
            head_dim=config.n_embd // config.n_head,
            num_layers=config.n_layer,
        )
        _ = model(ids[:, :-1], kv_cache=kv_cache)
        cached_last = model(ids[:, -1:], kv_cache=kv_cache)[:, -1, :].float()
        torch.testing.assert_close(cached_last, full[:, -1, :], rtol=1e-3, atol=1e-2)

        prefix = ids[:, :5]
        chunk = ids[:, 5:]
        kv_cache = KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,
            seq_len=ids.size(1),
            head_dim=config.n_embd // config.n_head,
            num_layers=config.n_layer,
        )
        _ = model(prefix, kv_cache=kv_cache)
        cached_chunk = model(chunk, kv_cache=kv_cache).float()
        torch.testing.assert_close(cached_chunk, full[:, 5:, :], rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize(
    "attention_type",
    [
        "standard",
        "tropical",
        "ultrametric",
        "simplicial",
        "braid",
        "clifford",
        "hyperbolic",
        "fractal",
        "quaternion",
        "octonion",
        "surreal",
        "reversible",
        "gauge",
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


def test_nanochat_hyperbolic_energy_reduction_and_telemetry():
    """mnn.6: the production scorer has the standard-attention anchor and
    exposes finite per-head geometry observables after a real forward."""
    import math

    import torch

    from nanochat.gpt import GPT, GPTConfig
    from nanochat.hyperbolic_attention_torch import energy_gromov_scores
    from nanochat.model_utils import causal_attn_mask, norm

    torch.manual_seed(13)
    q = norm(torch.randn(2, 4, 7, 16))
    k = norm(torch.randn(2, 4, 7, 16))
    c = torch.full((1, 4, 1, 1), 1e-7)
    scale = 1.0 / math.sqrt(q.size(-1))
    mask = causal_attn_mask(7, 7, device=q.device)
    hyp = torch.softmax((energy_gromov_scores(q, k, c) * scale).masked_fill(~mask, float("-inf")), dim=-1)
    standard = torch.softmax(((q @ k.transpose(-1, -2)) * scale).masked_fill(~mask, float("-inf")), dim=-1)
    torch.testing.assert_close(hyp, standard, rtol=2e-5, atol=2e-6)

    config = GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=1,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
        attention_type="hyperbolic",
    )
    model = GPT(config).train(False)
    ids = torch.randint(0, config.vocab_size, (2, 8), dtype=torch.long)
    with torch.inference_mode():
        logits = model(ids)
    attn = model.transformer.h[0].attn
    assert torch.isfinite(logits).all()
    assert torch.isfinite(attn.hyperbolic_curvature_head).all()
    assert torch.isfinite(attn.hyperbolic_radius_head_mean).all()
    assert attn.hyperbolic_curvature_head.shape == (config.n_head,)
    assert attn.hyperbolic_radius_head_mean.shape == (config.n_head,)

    v = torch.randn_like(q)
    with torch.no_grad():
        zero_radius = attn.attend(q, k, v, kv_cache=None, pos0=None)
        attn.radial_q.normal_(mean=0.0, std=0.5)
        attn.radial_k.normal_(mean=0.0, std=0.5)
        learned_radius = attn.attend(q, k, v, kv_cache=None, pos0=None)
    assert float((learned_radius - zero_radius).abs().max()) > 1e-5, (
        "learned radial projections must make finite curvature affect routing"
    )


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
    require(
        isinstance(order, torch.Tensor) and isinstance(selected, torch.Tensor) and isinstance(k, torch.Tensor),
        "Bad debug tensor types",
    )
    require(order.shape[-1] == ids.size(1), "Expected schedule to cover all causal keys for the last token")


def test_nanochat_braid_rmatrix_kv_cache_parity_and_charges():
    """R-matrix braid law (u55.3): KV-cache decode parity + conserved-charge telemetry.

    The integrable law must satisfy the house decode contract AND report a
    mass-partition defect (Q1) at fp32 noise level and a braid-consistency
    residual (Q2) at fp64 noise level - the two-charge fingerprint that
    separates integrable from heuristic mixing.
    """
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
        braid_crossing_law="rmatrix",
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
    charges = getattr(attn0, "last_braid_charges", None)
    require(isinstance(charges, dict), "Expected rmatrix attention to record last_braid_charges")
    require(charges["crossing_law"] == "rmatrix", "Expected the rmatrix law fingerprint")
    require(float(charges["q1_mass_defect"]) < 1e-5, f"Q1 mass defect too large: {charges['q1_mass_defect']}")
    require(float(charges["q2_braid_residual"]) < 1e-10, f"Q2 braid residual too large: {charges['q2_braid_residual']}")
    require(isinstance(charges["eta"], list) and len(charges["eta"]) == config.n_head, "Expected per-head eta")


def test_nanochat_braid_rmatrix_spectral_probes_zero_init_and_kv_parity():
    """Spectral multi-view probes (0i1v): zero-init gates make the probed
    mechanism bitwise the base sweep at init; gates receive gradients; the
    house KV-decode contract holds with probes enabled."""
    import torch

    from nanochat.engine import KVCache
    from nanochat.gpt import GPT, GPTConfig

    def build(n_probes: int):
        torch.manual_seed(0)
        cfg = GPTConfig(
            sequence_len=16,
            vocab_size=128,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=64,
            attention_type="braid",
            braid_crossing_law="rmatrix",
            braid_rmatrix_probes=n_probes,
        )
        return GPT(cfg), cfg

    ids = torch.randint(0, 128, (1, 8), dtype=torch.long, generator=torch.Generator().manual_seed(1))
    base, _ = build(0)
    probed, cfg = build(2)
    base.eval()
    probed.eval()
    # probe params are deterministic-init and appended AFTER the shared ones,
    # so the shared parameters are identical across the two builds
    with torch.inference_mode():
        out_base = base(ids)
        out_probed = probed(ids)
    torch.testing.assert_close(out_probed, out_base, rtol=0.0, atol=0.0)

    # gates receive gradients once training touches them
    probed.train()
    loss = probed(ids, ids)
    loss.backward()
    grads = [p.grad for n, p in probed.named_parameters() if "rmatrix_probe_gate" in n]
    require(all(g is not None for g in grads), "probe gates must receive gradients")

    probed.eval()
    with torch.inference_mode():
        full_last = probed(ids)[:, -1, :].float()
        kv_cache = KVCache(
            batch_size=1,
            num_heads=cfg.n_kv_head,
            seq_len=ids.size(1),
            head_dim=cfg.n_embd // cfg.n_head,
            num_layers=cfg.n_layer,
        )
        _ = probed(ids[:, :-1], kv_cache=kv_cache)
        cached_last = probed(ids[:, -1:], kv_cache=kv_cache)[:, -1, :].float()
    torch.testing.assert_close(cached_last, full_last, rtol=1e-3, atol=1e-2)


def test_nanochat_ultrametric_balltree_kv_parity_and_kernel_agreement():
    """Ball-tree mode (33dd): KV-cache decode parity (the house contract) and
    agreement with the hard-digit kernel - the two paths compute the same
    alpha^lcp attention; the kernel's soft-lcp approximation (exp(-beta d^2)
    instead of exact 0/1 digit matches) bounds the gap."""
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
        attention_type="ultrametric",
        ultrametric_mode="balltree",
        ultrametric_hard_digits=True,
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

    # same weights, kernel mode with hard digits: outputs must closely agree
    torch.manual_seed(0)
    config_k = GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
        attention_type="ultrametric",
        ultrametric_mode="kernel",
        ultrametric_hard_digits=True,
    )
    model_k = GPT(config_k).train(False)
    model_k.load_state_dict(model.state_dict())
    with torch.inference_mode():
        out_k = model_k(ids)[:, -1, :].float()
    torch.testing.assert_close(out_k, full_last, rtol=1e-3, atol=1e-2)


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
    require(
        tuple(entropy.shape) == (config.n_head,),
        f"Expected entropy shape ({config.n_head},), got {tuple(entropy.shape)}",
    )
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
        "gauge",
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
    torch.testing.assert_close(
        expanded.kv_cache[:, :, 0, :, : other.pos, :], other.kv_cache[:, :, 0, :, : other.pos, :]
    )
    torch.testing.assert_close(
        expanded.kv_cache[:, :, 1, :, : other.pos, :], other.kv_cache[:, :, 0, :, : other.pos, :]
    )


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
        "results": {
            "losses": [3.0, 2.0],
            "tokens_per_second": 1000.0,
            "tflops_per_second_est": 1.0,
            "peak_memory_allocated_gb": 2.0,
        },
    }
    cand_summary = {
        "git": {"commit": "def456", "dirty": True},
        "config": {"attention_type": "standard", "use_flex_attention": True},
        "results": {
            "losses": [3.0, 2.1],
            "tokens_per_second": 950.0,
            "tflops_per_second_est": 0.95,
            "peak_memory_allocated_gb": 2.2,
        },
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


def _tiny_gauge_model_and_cache(seq_len: int = 16, n_layer: int = 2):
    import torch

    from nanochat.engine import KVCache
    from nanochat.gpt import GPT, GPTConfig

    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=seq_len,
        vocab_size=128,
        n_layer=n_layer,
        n_head=4,
        n_kv_head=2,  # exercise the GQA path 7b0.5 unlocked
        n_embd=64,
        attention_type="gauge",
    )
    model = GPT(config).train(False)

    def make_cache(total_len: int) -> KVCache:
        return KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,
            seq_len=total_len,
            head_dim=config.n_embd // config.n_head,
            num_layers=config.n_layer,
        )

    return model, config, make_cache


def test_nanochat_gauge_token_by_token_decode_matches_full_forward():
    """Gauge KV-cache decode (7b0.5): the fp32 cumulative-angle lane must reproduce
    the full-forward gauge field exactly enough for token-by-token parity."""
    import torch

    model, config, make_cache = _tiny_gauge_model_and_cache()
    ids = torch.randint(0, config.vocab_size, (1, 8), dtype=torch.long)
    with torch.inference_mode():
        full = model(ids).float()
        kv_cache = make_cache(ids.size(1))
        steps = [model(ids[:, t : t + 1], kv_cache=kv_cache)[:, -1, :].float() for t in range(ids.size(1))]
    decoded = torch.stack(steps, dim=1)
    torch.testing.assert_close(decoded, full, rtol=1e-3, atol=1e-2)


def test_nanochat_gauge_long_decode_drift_is_bounded():
    """2k-step decode: fp32 angle accumulation must not drift away from the
    full-forward gauge field (bead 7b0.5 acceptance criterion)."""
    import torch

    model, config, make_cache = _tiny_gauge_model_and_cache(seq_len=256, n_layer=1)
    total = 2048  # within the 10x rotary overcompute for sequence_len=256
    ids = torch.randint(0, config.vocab_size, (1, total), dtype=torch.long)
    with torch.inference_mode():
        full_last = model(ids)[:, -1, :].float()
        kv_cache = make_cache(total)
        # prefill all but the final token in chunks, then decode the last one
        for start in range(0, total - 1, 256):
            _ = model(ids[:, start : min(start + 256, total - 1)], kv_cache=kv_cache)
        cached_last = model(ids[:, -1:], kv_cache=kv_cache)[:, -1, :].float()
    drift = (cached_last - full_last).abs().max().item()
    assert drift < 1e-2, f"gauge long-decode drift {drift:.3e} exceeds bound 1e-2 after {total} positions"


def test_nanochat_gauge_cache_desync_fails_loudly():
    """The gauge angle lane ACCUMULATES, so re-running a forward against the
    same cache (or rewinding pos without reset()) would silently double-count
    angles. The lane tracks its token count and must raise instead."""
    import torch

    model, config, make_cache = _tiny_gauge_model_and_cache()
    ids = torch.randint(0, config.vocab_size, (1, 4), dtype=torch.long)
    with torch.inference_mode():
        kv_cache = make_cache(8)
        _ = model(ids, kv_cache=kv_cache)
        kv_cache.pos = 0  # rewind WITHOUT reset(): the lane still holds 4 tokens
        with pytest.raises(RuntimeError, match="gauge angle lane desync"):
            _ = model(ids, kv_cache=kv_cache)


def test_nanochat_gauge_cache_reset_clears_angle_lane():
    """KVCache.reset() must zero the gauge cumsum lane: it ACCUMULATES (unlike
    the positionally-overwritten kv/simplicial lanes), so a reused cache would
    otherwise start from a stale gauge field."""
    import torch

    model, config, make_cache = _tiny_gauge_model_and_cache()
    ids = torch.randint(0, config.vocab_size, (1, 6), dtype=torch.long)
    with torch.inference_mode():
        kv_cache = make_cache(ids.size(1))
        first = model(ids, kv_cache=kv_cache)[:, -1, :].float()
        assert kv_cache.gauge_cum_angles is not None
        assert kv_cache.gauge_cum_angles.abs().sum() > 0
        kv_cache.reset()
        assert kv_cache.gauge_cum_angles.abs().sum() == 0
        second = model(ids, kv_cache=kv_cache)[:, -1, :].float()
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_nanochat_gauge_training_escapes_zero_init():
    """Regression for model_guided_research-1fr6: without the residual skeleton,
    zero-init c_proj/lm_head mutually annihilated ALL gradients (total |grad|
    was exactly 0.0) and gauge training was frozen at ln(vocab) forever."""
    import torch

    from nanochat.gpt import GPT, GPTConfig

    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=32,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        attention_type="gauge",
    )
    model = GPT(config)
    model.init_weights()
    ids = torch.randint(0, config.vocab_size, (2, 32), dtype=torch.long)
    targets = torch.randint(0, config.vocab_size, (2, 32), dtype=torch.long)
    loss = model(ids, targets=targets)
    loss.backward()
    total_grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    require(total_grad > 0.0, "gauge zero-init gradient deadlock is back: total |grad| == 0.0")


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


def test_certify_full_suite_passes():
    """The complete certificate suite must be green on CPU fp32 (br-5ki.1 acceptance)."""
    from cli import _CERTIFY_MECHANISMS, _run_certify_checks

    checks = _run_certify_checks(list(_CERTIFY_MECHANISMS), device_str="cpu", dtype_str="fp32", seed=42)
    require(len(checks) >= 30, f"Expected a comprehensive check suite, got only {len(checks)} checks")
    mechanisms_covered = {c["mechanism"] for c in checks}
    require(
        mechanisms_covered == set(_CERTIFY_MECHANISMS),
        f"Causality coverage gap: {set(_CERTIFY_MECHANISMS) - mechanisms_covered}",
    )
    failing = [c for c in checks if c["status"] != "pass"]
    require(not failing, f"Certificate checks failed: {[(c['mechanism'], c['check'], c['detail']) for c in failing]}")
    # Every check must carry measured value, tolerance, and timing (schema contract for B3/G2 consumers).
    for c in checks:
        require(isinstance(c["measured"], float), f"measured missing/wrong type in {c['check']}")
        require(isinstance(c["duration_ms"], float), f"duration_ms missing in {c['check']}")
        require(c["comparator"] in {"le", "ge"}, f"bad comparator in {c['check']}")


def test_certify_detects_violation(monkeypatch):
    """A deliberately broken invariant must produce a FAIL status (br-5ki.1 acceptance: failures fail)."""
    import nanochat.quaternion_attention_torch as quat_mod
    from cli import _run_certify_checks

    real_qmul = quat_mod.qmul

    def broken_qmul(a, b):
        out = real_qmul(a, b)
        return out * 1.01  # break norm multiplicativity (and associativity scaling)

    monkeypatch.setattr(quat_mod, "qmul", broken_qmul)
    checks = _run_certify_checks(["quaternion"], device_str="cpu", dtype_str="fp32", seed=42)
    norm_check = next(c for c in checks if c["check"] == "qmul_norm_multiplicative")
    require(
        norm_check["status"] == "fail",
        f"Broken qmul must fail the norm-multiplicativity certificate, got {norm_check['status']}",
    )


def test_clifford_property_checks_green():
    """mnn.2: all in-demo Clifford property checks must pass."""
    import clifford_algebra_and_geometric_attention as clifford

    checks = clifford.run_property_checks(seed=7)
    require(len(checks) == 6, f"expected 6 property checks, got {len(checks)}")
    failing = [(name, detail) for name, ok, detail in checks if not ok]
    require(not failing, f"Clifford property checks failed: {failing}")


def test_clifford_rotor_extrapolation_smoke():
    """mnn.2 payoff: rotor model trained near identity extrapolates to far
    rotations at near-machine precision, below the MLP baseline."""
    import clifford_algebra_and_geometric_attention as clifford

    res = clifford.run_generalization_experiment(seed=7, steps=150)
    require(res["rotor_test_far_mse"] < 1e-6, f"rotor far-rotation MSE too high: {res}")
    require(
        res["rotor_test_far_mse"] < res["mlp_test_far_mse"],
        f"rotor model should beat MLP on far rotations: {res}",
    )


def test_certify_cli_exit_codes(tmp_path):
    """CLI: exit 0 on green subset, nonzero on unknown mechanism; artifacts written when requested."""
    import json as json_mod

    from typer.testing import CliRunner

    import cli as mgr_cli

    runner = CliRunner()
    result = runner.invoke(
        mgr_cli.app,
        ["certify", "-m", "braid", "-m", "surreal", "--artifacts-dir", str(tmp_path), "--run-id", "testrun"],
    )
    require(result.exit_code == 0, f"certify on green mechanisms must exit 0, got {result.exit_code}: {result.output}")
    summary_path = tmp_path / "certs" / "nanochat" / "testrun" / "summary.json"
    require(summary_path.exists(), f"Expected certificate summary at {summary_path}")
    summary = json_mod.loads(summary_path.read_text())
    require(summary["kind"] == "certify", "summary.json must identify itself as a certify artifact")
    require(summary["counts"]["fail"] == 0 and summary["counts"]["error"] == 0, "green run must record zero failures")
    require((tmp_path / "certs" / "nanochat" / "testrun" / "run.md").exists(), "run.md report must be written")

    result_bad = runner.invoke(mgr_cli.app, ["certify", "-m", "nonexistent-mechanism"])
    require(result_bad.exit_code != 0, "certify must exit nonzero for an unknown mechanism")


def _cert_summary(tmp_path, name, checks):
    import json as json_mod

    path = tmp_path / f"{name}.json"
    passed = sum(1 for c in checks if c["status"] == "pass")
    payload = {
        "schema_version": 1,
        "kind": "certify",
        "run_id": name,
        "seed": 42,
        "device": "cpu",
        "dtype": "fp32",
        "mechanisms": sorted({c["mechanism"] for c in checks}),
        "checks": checks,
        "counts": {
            "pass": passed,
            "fail": sum(1 for c in checks if c["status"] == "fail"),
            "error": sum(1 for c in checks if c["status"] == "error"),
        },
        "duration_s": 0.1,
    }
    path.write_text(json_mod.dumps(payload))
    return path


def _check(mechanism, check, *, status, measured, tolerance=1e-4, comparator="le"):
    return {
        "mechanism": mechanism,
        "check": check,
        "family": "classical",
        "status": status,
        "measured": measured,
        "tolerance": tolerance,
        "comparator": comparator,
        "duration_ms": 1.0,
        "detail": "",
    }


def test_regressions_flags_certificate_green_to_red_flip(tmp_path):
    """5ki.3: a certificate check flipping pass -> fail between snapshots is
    a regression; --fail-on-regression must turn it into exit 1."""
    from typer.testing import CliRunner

    import cli as mgr_cli

    base = _cert_summary(
        tmp_path, "flip-base", [_check("reversible", "round_trip_error", status="pass", measured=1e-7)]
    )
    cand_bad = _cert_summary(
        tmp_path, "flip-cand-bad", [_check("reversible", "round_trip_error", status="fail", measured=1e-2)]
    )
    cand_good = _cert_summary(
        tmp_path, "flip-cand-good", [_check("reversible", "round_trip_error", status="pass", measured=2e-7)]
    )

    runner = CliRunner()
    bad = runner.invoke(
        mgr_cli.app,
        ["regressions", "--baseline", str(base), "--candidate", str(cand_bad), "--fail-on-regression"],
    )
    require(bad.exit_code == 1, f"green->red flip must exit 1, got {bad.exit_code}")
    require("round_trip_error" in bad.output, "failing metric name should appear in the guardrail output")

    good = runner.invoke(
        mgr_cli.app,
        ["regressions", "--baseline", str(base), "--candidate", str(cand_good)],
    )
    require(good.exit_code == 0, f"still-green candidate must exit 0, got {good.exit_code}: {good.output}")


def test_regressions_flags_still_passing_degradation_with_factor(tmp_path):
    """5ki.3: an invariant that stays under tolerance but degrades by more
    than --cert-degrade-factor is flagged; raising the factor clears it."""
    from typer.testing import CliRunner

    import cli as mgr_cli

    base = _cert_summary(
        tmp_path, "deg-base", [_check("reversible", "round_trip_error", status="pass", measured=1e-7)]
    )
    cand = _cert_summary(
        tmp_path, "deg-cand", [_check("reversible", "round_trip_error", status="pass", measured=8e-5)]
    )

    runner = CliRunner()
    flagged = runner.invoke(
        mgr_cli.app,
        [
            "regressions", "--baseline", str(base), "--candidate", str(cand),
            "--cert-degrade-factor", "10", "--fail-on-regression",
        ],
    )
    require(flagged.exit_code == 1, f"800x degradation must be flagged, got exit {flagged.exit_code}")

    cleared = runner.invoke(
        mgr_cli.app,
        [
            "regressions", "--baseline", str(base), "--candidate", str(cand),
            "--cert-degrade-factor", "10000", "--fail-on-regression",
        ],
    )
    require(cleared.exit_code == 0, f"degradation under the factor must pass, got {cleared.exit_code}")


def test_regressions_rejects_certificate_kind_mismatch(tmp_path):
    """5ki.3: comparing a certificate against a non-certificate snapshot is a
    hard parameter error, never silent nonsense."""
    import json as json_mod

    from typer.testing import CliRunner

    import cli as mgr_cli

    cert = _cert_summary(
        tmp_path, "mix-cert", [_check("standard", "causality", status="pass", measured=0.0)]
    )
    other = tmp_path / "plain.json"
    other.write_text(json_mod.dumps({"results": {"losses": [1.0, 2.0]}}))

    runner = CliRunner()
    result = runner.invoke(mgr_cli.app, ["regressions", "--baseline", str(cert), "--candidate", str(other)])
    require(result.exit_code != 0, "kind mismatch must exit nonzero")


def test_certify_gauge_forward_runs():
    """Regression for br-hn5: gauge block forward must run (RoPE layout bug found by certify)."""
    import torch

    from cli import _certify_cos_sin, _certify_tiny_config
    from nanochat.gpt import Block

    torch.manual_seed(0)
    cfg = _certify_tiny_config("gauge")
    block = Block(cfg, 0)
    T = 8  # deliberately != n_head: the broken layout only crashed when n_head != T
    require(cfg.n_head != T, "test must use n_head != T to exercise the regression")
    x = torch.randn(1, T, cfg.n_embd)
    cos_sin = _certify_cos_sin(T, cfg.n_embd // cfg.n_head, torch.device("cpu"), torch.float32)
    y = block(x, cos_sin, None)
    require(y.shape == x.shape, f"gauge forward output shape mismatch: {y.shape} vs {x.shape}")
    require(bool(torch.isfinite(y).all()), "gauge forward produced non-finite values")


def test_fuzz_runner_well_formed_and_green_on_benign_grid():
    """Fuzz harness contract: well-formed records; benign reduced grid is all-pass (br-5ki.4)."""
    from cli import _run_fuzz_cells

    records = _run_fuzz_cells(
        ["standard", "tropical"],
        device_str="cpu",
        dtypes=["fp32"],
        scales=[1.0],
        lengths=[1, 2],
        patterns=["randn", "all_equal", "zeros"],
        seed=42,
    )
    require(len(records) == 2 * 1 * 2 * 3, f"unexpected cell count: {len(records)}")
    for r in records:
        for key in ("mechanism", "dtype", "scale", "T", "pattern", "status", "recipe", "duration_ms"):
            require(key in r, f"fuzz record missing key {key}: {r}")
        require("seed=42" in r["recipe"], "recipe must carry the seed for reproduction")
        require(r["status"] == "pass", f"benign grid must pass, got {r['status']} for {r['recipe']}")


@pytest.mark.parametrize(
    "pattern,length",
    [
        ("all_equal", 4),  # maximal score ties (argmax/softmax tie-handling)
        ("zeros", 4),  # zero-norm inputs (rmsnorm/quaternion/octonion normalization paths)
        ("randn", 1),  # T=1 boundary (single-token causal path)
    ],
)
def test_fuzz_adversarial_regressions(pattern, length):
    """Permanent regressions for the adversarial cases the bead names: ties, zero-norm, T=1 (br-5ki.4)."""
    from cli import _CERTIFY_MECHANISMS, _run_fuzz_cells

    records = _run_fuzz_cells(
        list(_CERTIFY_MECHANISMS),
        device_str="cpu",
        dtypes=["fp32"],
        scales=[1.0],
        lengths=[length],
        patterns=[pattern],
        seed=42,
    )
    bad = [r for r in records if r["status"] in ("fail", "error")]
    require(not bad, f"NaN/Inf or crash on {pattern}/T={length}: {[r['recipe'] for r in bad]}")


def test_fuzz_detects_nan_injection(monkeypatch):
    """A mechanism emitting NaN must be recorded as FAIL with a complete repro recipe (br-5ki.4)."""
    import torch

    import nanochat.gpt as gpt_mod
    from cli import _run_fuzz_cells

    class NaNBlock(torch.nn.Module):
        def __init__(self, cfg, layer_idx):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x, cos_sin, kv_cache):
            y = x + self.p
            y = y.clone()
            y[..., 0] = float("nan")
            return y

    monkeypatch.setattr(gpt_mod, "Block", NaNBlock)
    records = _run_fuzz_cells(
        ["standard"], device_str="cpu", dtypes=["fp32"], scales=[1.0], lengths=[2], patterns=["randn"], seed=42
    )
    require(len(records) == 1, f"expected one cell, got {len(records)}")
    require(records[0]["status"] == "fail", f"NaN output must be a FAIL, got {records[0]['status']}")
    require(records[0]["out_nan_inf"] > 0, "out_nan_inf must count the injected NaNs")


def test_doctor_runs_and_reports(monkeypatch):
    """mgr doctor: runs on CPU-only env, JSON schema stable, missing-data hint present (br-rz8.7)."""
    import json as json_mod

    from typer.testing import CliRunner

    import cli as mgr_cli
    import nanochat.dataset as ds_mod

    monkeypatch.setattr(ds_mod, "list_parquet_files", lambda data_dir=None: [])
    runner = CliRunner()
    result = runner.invoke(mgr_cli.app, ["doctor", "--json"])
    # CPU-only CI: warns are expected (exit 1); failures are not (exit 2).
    require(result.exit_code in (0, 1), f"doctor must not FAIL on a healthy CPU env: {result.output}")
    payload = json_mod.loads(result.output)
    require(payload["kind"] == "doctor", "JSON payload must identify itself")
    names = {r["name"] for r in payload["checks"]}
    for expected in ("python", "torch", "jax", "training data", "tokenizer", "disk space", "model smoke"):
        require(expected in names, f"doctor missing check: {expected}")
    data_row = next(r for r in payload["checks"] if r["name"] == "training data")
    require(data_row["status"] == "warn", "empty data dir must produce a WARN")
    require("auto-download-data" in data_row["hint"], "missing-data hint must name the fix flag")
    smoke = next(r for r in payload["checks"] if r["name"] == "model smoke")
    require(smoke["status"] == "ok", f"model smoke must pass: {smoke}")


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main(sys.argv[1:] or [__file__]))
