"""Tests for nanochat.viz (beads hi3 + 7ow): model-state visualizations.

Covers the harvesting pipeline (entropy / tropical margins / softmax-map
capture), the cross-head route-diversity metric, determinism, the
no-side-effects guarantee of the reversible attend-capture, and that the
renderers actually write the expected artifacts.
"""

from __future__ import annotations

import json
import math

import torch

from nanochat import viz


def _standard_diag(seed: int = 0, seq_len: int = 16):
    model, _meta = viz.build_probe_model(
        "standard", seed=seed, n_layer=2, n_head=4, n_kv_head=4, n_embd=64,
        sequence_len=seq_len, vocab_size=128,
    )
    idx, labels = viz.sample_batch(text=None, batch_size=2, seq_len=seq_len, vocab_size=128, seed=seed)
    return viz.collect_state(model, idx, token_labels=labels)


def test_standard_collects_entropy_and_maps():
    diag = _standard_diag()
    assert diag.attention_type == "standard"
    assert diag.has_entropy()
    # one entropy row per layer, n_head columns, all finite and non-negative
    assert len(diag.entropy_layer_head) == 2
    for row in diag.entropy_layer_head:
        assert len(row) == 4
        assert all(math.isfinite(x) and x >= 0.0 for x in row)
    # softmax maps captured for both layers, shape (H, T, T), rows sum ~ 1
    assert set(diag.attn_maps) == {0, 1}
    maps = diag.attn_maps[0]
    assert tuple(maps.shape) == (4, 16, 16)
    row_sums = maps.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_tropical_collects_margins():
    model, _meta = viz.build_probe_model(
        "tropical", seed=0, n_layer=2, n_head=4, n_kv_head=4, n_embd=64,
        sequence_len=16, vocab_size=128,
    )
    idx, _ = viz.sample_batch(text=None, batch_size=2, seq_len=16, vocab_size=128, seed=0)
    diag = viz.collect_state(model, idx)
    assert diag.has_margins()
    assert len(diag.margin_layer_head) == 2
    for row in diag.margin_layer_head:
        assert len(row) == 4
        assert all(math.isfinite(x) for x in row)


def test_mixed_schedule_collects_per_mechanism_diagnostics():
    model, meta = viz.build_probe_model(
        "standard,tropical", seed=0, n_layer=2, n_head=4, n_kv_head=4, n_embd=64,
        sequence_len=16, vocab_size=128,
    )
    assert meta["config"]["standard_record_attn_entropy"] is True
    assert meta["config"]["tropical_record_margins"] is True
    assert [block.attention_type for block in model.transformer.h] == ["standard", "tropical"]

    idx, _ = viz.sample_batch(text=None, batch_size=2, seq_len=16, vocab_size=128, seed=0)
    diag = viz.collect_state(model, idx)
    assert diag.has_entropy()
    assert diag.has_margins()


def test_mixed_schedule_probe_defaults_cover_reversible_geometry():
    model, meta = viz.build_probe_model("standard,reversible", seed=0, n_layer=2, sequence_len=16, vocab_size=128)
    assert meta["config"]["n_kv_head"] == 2
    assert meta["config"]["reversible_record_energy"] is True
    assert [block.attention_type for block in model.transformer.h] == ["standard", "reversible"]


def test_route_diversity_in_unit_interval():
    diag = _standard_diag()
    div = viz.head_route_diversity(diag)
    assert div is not None
    assert 0.0 <= div <= 1.0


def test_determinism_same_seed():
    a = _standard_diag(seed=7)
    b = _standard_diag(seed=7)
    assert a.entropy_layer_head == b.entropy_layer_head
    assert torch.equal(a.attn_maps[0], b.attn_maps[0])


def test_capture_patch_is_reversible_and_output_unchanged():
    """The attend-capture must restore the original method and not perturb the
    forward result (it runs under no_grad and only delegates)."""
    from nanochat.gpt import CausalSelfAttention

    model, _ = viz.build_probe_model(
        "standard", seed=1, n_layer=2, n_head=4, n_kv_head=4, n_embd=64,
        sequence_len=16, vocab_size=128,
    )
    idx, _ = viz.sample_batch(text=None, batch_size=2, seq_len=16, vocab_size=128, seed=1)
    with torch.no_grad():
        baseline = model(idx).clone()
    # attend is the class method before any patch
    attn0 = model.transformer.h[0].attn
    assert attn0.attend.__func__ is CausalSelfAttention.attend
    diag = viz.collect_state(model, idx)
    # restored to the class method afterward
    assert attn0.attend.__func__ is CausalSelfAttention.attend
    assert diag.attn_maps  # capture happened
    with torch.no_grad():
        after = model(idx)
    assert torch.equal(baseline, after)


def test_sample_batch_random_is_seeded_and_shaped():
    a, la = viz.sample_batch(text=None, batch_size=3, seq_len=10, vocab_size=64, seed=5)
    b, lb = viz.sample_batch(text=None, batch_size=3, seq_len=10, vocab_size=64, seed=5)
    assert tuple(a.shape) == (3, 10)
    assert torch.equal(a, b)
    assert la is None and lb is None
    assert int(a.max()) < 64 and int(a.min()) >= 0


def test_render_state_writes_artifacts(tmp_path):
    diag = _standard_diag()
    out = tmp_path / "state"
    from rich.console import Console

    summary = viz.render_state(diag, out, console=Console(file=open(tmp_path / "_log.txt", "w")))
    assert summary["schema"] == "mgr.viz.state.v1"
    assert (out / "summary.json").is_file()
    assert (out / "attention_entropy_heatmap.png").is_file()
    assert (out / "attention_maps.png").is_file()
    assert (out / "index.html").is_file()
    loaded = json.loads((out / "summary.json").read_text())
    assert loaded["schema"] == "mgr.viz.state.v1"
    assert "attention_entropy_heatmap" in loaded["visuals"]
    assert "attention_maps" in loaded["visuals"]


def test_render_entropy_diversity_writes_artifacts(tmp_path):
    configs = []
    for name in ("standard", "tropical"):
        model, meta = viz.build_probe_model(
            name, seed=0, n_layer=2, n_head=4, n_kv_head=4, n_embd=64,
            sequence_len=16, vocab_size=128,
        )
        idx, labels = viz.sample_batch(text=None, batch_size=2, seq_len=16, vocab_size=128, seed=0)
        diag = viz.collect_state(model, idx, token_labels=labels)
        diag.attention_type = name
        configs.append((name, diag))
    out = tmp_path / "entropy"
    from rich.console import Console

    summary = viz.render_entropy_diversity(configs, out, console=Console(file=open(tmp_path / "_log.txt", "w")))
    assert (out / "summary.json").is_file()
    assert (out / "per_head_entropy_diversity.png").is_file()
    assert summary["schema"] == "mgr.viz.entropy.v1"
    assert set(summary["configs"]) == {"standard", "tropical"}
    # standard reports JS route diversity; tropical reports a margin signal
    assert summary["configs"]["standard"]["route_diversity_js"] is not None
    assert summary["configs"]["tropical"]["signal"] == "tropical_margin"


def test_mgr_visualize_state_command(tmp_path):
    """The `mgr visualize state` wrapper delegates to viz and writes artifacts."""
    from typer.testing import CliRunner

    import cli

    out = tmp_path / "state"
    result = CliRunner().invoke(
        cli.app,
        ["visualize", "state", "--attention", "standard", "--seq-len", "16",
         "--batch-size", "2", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert (out / "summary.json").is_file()
    assert (out / "attention_entropy_heatmap.png").is_file()


def test_mgr_visualize_entropy_command(tmp_path):
    from typer.testing import CliRunner

    import cli

    out = tmp_path / "entropy"
    result = CliRunner().invoke(
        cli.app,
        ["visualize", "entropy", "--baseline", "standard", "--feature", "tropical",
         "--seq-len", "16", "--batch-size", "2", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert (out / "summary.json").is_file()


def test_mgr_visualize_unknown_mode_errors():
    from typer.testing import CliRunner

    import cli

    result = CliRunner().invoke(cli.app, ["visualize", "bogus"])
    assert result.exit_code == 2
