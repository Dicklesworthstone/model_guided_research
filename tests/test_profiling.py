"""Tests for nanochat.profiling (bead b1l): torch.profiler / NVTX hooks.

Covers the toggle semantics (disabled = no profiler, negligible overhead), env
configuration, the NVTX no-op on CPU, and the standalone microbench producing a
device-agnostic kernel/memory breakdown + a Chrome trace.
"""

from __future__ import annotations

import json

from nanochat import profiling


def test_profile_config_from_env():
    assert profiling.ProfileConfig.from_env({}).enabled is False
    c = profiling.ProfileConfig.from_env({"NANOCHAT_PROFILE": "1", "NANOCHAT_PROFILE_ROWS": "7"})
    assert c.enabled is True
    assert c.row_limit == 7


def test_disabled_profiler_is_noop():
    cfg = profiling.ProfileConfig(enabled=False)
    with profiling.torch_profiler(cfg) as prof:
        with profiling.nvtx_range("x"):
            pass
    assert prof is None


def test_enabled_profiler_yields_profile():
    import torch

    cfg = profiling.ProfileConfig(enabled=True)
    with profiling.torch_profiler(cfg) as prof:
        x = torch.randn(8, 8)
        _ = (x @ x.t()).sum()
    assert prof is not None
    summary = profiling.summarize_profile(prof, row_limit=5)
    assert summary["device"] in {"cpu", "cuda"}
    assert summary["totals"]["n_ops"] >= 1
    assert len(summary["ops"]) <= 5


def test_nvtx_range_is_safe_on_cpu():
    # must not raise on a CPU-only box
    with profiling.nvtx_range("a"):
        with profiling.nvtx_range("b"):
            pass


def test_profile_model_standard_writes_trace(tmp_path):
    out = tmp_path / "prof"
    summary = profiling.profile_model(
        "standard", device="cpu", steps=1, warmup=0, backward=True,
        n_layer=1, n_head=4, n_kv_head=4, n_embd=64, seq_len=16, batch_size=2,
        vocab_size=128, trace_dir=out, row_limit=8,
    )
    assert summary["device"] == "cpu"
    assert summary["ops"], "expected at least one profiled op"
    # the dominant transformer op is a matmul
    assert any("mm" in op["op"] for op in summary["ops"])
    assert summary["totals"]["self_cpu_us"] > 0.0
    assert summary["meta"]["attention_type"] == "standard"
    assert summary["meta"]["n_params"] > 0
    # Chrome trace exported and loadable as JSON
    trace = out / "trace.json"
    assert trace.is_file()
    data = json.loads(trace.read_text())
    assert "traceEvents" in data or isinstance(data, list)


def test_profile_model_forward_only_runs():
    summary = profiling.profile_model(
        "tropical", device="cpu", steps=1, warmup=0, backward=False,
        n_layer=1, n_head=4, n_kv_head=4, n_embd=64, seq_len=16, batch_size=2,
        vocab_size=128, trace_dir=None, row_limit=5,
    )
    assert summary["meta"]["backward"] is False
    assert summary["meta"]["trace"] is None
    assert summary["ops"]
