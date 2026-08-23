"""Tests for the NSA width-scaling parameterization (bead lab.1): the
extreme-value primitives, the per-mechanism scaling table, and the
coordinate-check harness. The EVT claims are validated independently of any
network (the Gumbel ingredient); the coordinate check is validated on the
CLT control (standard must be flat in width)."""

import math

import pytest

from nanochat import parameterization as P


def test_exact_expected_max_matches_monte_carlo():
    """E[max of n N(0,1)] via quadrature matches a Monte-Carlo estimate within
    sampling noise across the width ladder -- the finite-n location the table
    carries."""
    import torch

    torch.manual_seed(0)
    for n in (16, 64, 256, 1024):
        mc = float(torch.randn(60000, n).max(dim=1).values.mean())
        exact = P.exact_expected_max(n)
        assert abs(exact - mc) / mc < 0.015, f"n={n}: exact {exact:.4f} vs MC {mc:.4f}"


def test_asymptotic_location_is_off_at_small_n():
    """The 2nd-order asymptote captures the LAW but not the small-n CONSTANT:
    it must be visibly off at n=16 (the finite-n lesson) and converge toward
    the exact value as n grows."""
    err16 = abs(P.gumbel_asymptotic_location(16) - P.exact_expected_max(16)) / P.exact_expected_max(16)
    err4096 = abs(P.gumbel_asymptotic_location(4096) - P.exact_expected_max(4096)) / P.exact_expected_max(4096)
    assert err16 > 0.05, f"asymptote should be >5% off at n=16, got {err16:.3f}"
    assert err4096 < err16, "asymptote must converge to exact as n grows"


def test_gumbel_scaling_law_slope():
    """E[max] grows ~sqrt(2 ln N): on a log scale vs ln N the exact location
    tracks sqrt(2 ln N) -- the SCALING LAW that makes max-plus an EVT class."""
    import numpy as np

    ns = [16, 64, 256, 1024, 4096]
    exact = np.array([P.exact_expected_max(n) for n in ns])
    asymp = np.array([math.sqrt(2 * math.log(n)) for n in ns])
    # ratio exact/asymptote approaches 1 monotonically from below
    ratios = exact / asymp
    assert all(0.7 < r < 1.0 for r in ratios)
    assert ratios[-1] > ratios[0], "exact/asymptote ratio must rise toward 1"


def test_scaling_table_covers_core_mechanisms():
    for mech in ("standard", "tropical", "ultrametric", "quaternion", "octonion", "reversible"):
        rule = P.scaling_rule(mech)
        assert rule.mechanism == mech
        assert rule.concentration_class
        assert rule.notes
    # tropical is the EVT showcase; quaternion/octonion split the LR exponent
    assert "EVT" in P.scaling_rule("tropical").concentration_class
    assert P.scaling_rule("quaternion").lr_exponent == -1.0
    # unclassified falls back to CLT conservatively, never KeyErrors
    assert "CLT" in P.scaling_rule("does-not-exist").concentration_class


def test_coordinate_check_standard_is_flat():
    """The CLT control: standard attention's activation RMS is flat in width
    (|log-log slope| small). If this drifts, the harness is broken, not the
    theory -- so it gates everything else."""
    import warnings

    warnings.filterwarnings("ignore")
    res = P.coordinate_check("standard", [64, 128, 256, 512], seed=0)
    assert abs(res["loglog_slope"]) < 0.05, f"standard not flat: slope {res['loglog_slope']:.4f}"
    assert all(v > 0 for v in res["activation_rms"].values())


@pytest.mark.parametrize("mech", ["tropical", "quaternion", "reversible"])
def test_coordinate_check_runs_for_mechanisms(mech):
    """The harness is generic over mechanisms: each produces finite per-width
    scales and a finite slope (the per-mechanism interpretation lives in the
    theory note; here we assert the apparatus runs and is well-formed)."""
    import warnings

    warnings.filterwarnings("ignore")
    res = P.coordinate_check(mech, [64, 128, 256], seed=0)
    assert res["concentration_class"]
    assert len(res["activation_rms"]) == 3
    assert all(math.isfinite(v) and v > 0 for v in res["activation_rms"].values())
    assert math.isfinite(res["loglog_slope"])


def test_nsa_mlp_bias_uses_exact_emax_constants():
    """bp08 part 1: under parameterization='nsa' the tropical MLP stage-1 bias
    carries the EXACT finite-N E[max](fan_in) constant; 'current' keeps the
    asymptote. The exact constant is SMALLER (the asymptote overshoots at
    small fan-in), so the arms must differ by the table's prediction."""
    import torch

    from nanochat.gpt import GPT, GPTConfig

    def stage1_bias(mode):
        torch.manual_seed(0)
        cfg = GPTConfig(
            n_layer=1, n_head=2, n_kv_head=2, n_embd=16, sequence_len=8,
            vocab_size=32, attention_type="tropical", ffn_type="tropical",
            parameterization=mode,
        )
        model = GPT(cfg)
        mlps = [m for m in model.modules() if type(m).__name__ == "TropicalMLP"]
        assert mlps, "no TropicalMLP in a tropical-FFN model"
        return float(mlps[0].b1.detach()[0])

    cur = stage1_bias("current")
    nsa = stage1_bias("nsa")
    fan_in = 16  # stage-1 max runs over the residual width, not d_ff
    assert abs(cur + math.sqrt(2 * math.log(fan_in))) < 1e-5, f"current arm left the asymptote: {cur}"
    assert abs(nsa + P.exact_expected_max(fan_in)) < 1e-5, f"nsa arm left E[max]: {nsa}"
    assert nsa > cur, "exact E[max] < asymptote at small n: nsa bias must sit higher"


def test_nsa_attention_shift_is_exact_and_centering_composes():
    """bp08 part 1: with score centering OFF the nsa arm shifts attention
    outputs by exactly E[max](head_dim); with centering ON the two arms agree
    (a global shift cancels in the per-query centering - the note's claim that
    empirical and theoretical location control commute)."""
    import torch

    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tropical_attention_torch import TropicalCausalSelfAttention

    def build(**kw):
        base = dict(n_layer=1, n_head=2, n_kv_head=2, n_embd=16, sequence_len=8, vocab_size=32, attention_type="tropical")
        return GPT(GPTConfig(**base, **kw))

    torch.manual_seed(3)
    q, k, v = (torch.randn(1, 2, 8, 8) for _ in range(3))

    uncentered = build(tropical_gauge_fix=False, tropical_score_center=False, parameterization="nsa")
    block = [m for m in uncentered.modules() if isinstance(m, TropicalCausalSelfAttention)][0]
    y_nsa = block.attend(q, k, v, kv_cache=None, pos0=0)
    block.parameterization = "current"
    y_cur = block.attend(q, k, v, kv_cache=None, pos0=0)
    delta = float((y_nsa - y_cur).abs().max())
    assert abs(delta - P.exact_expected_max(8)) < 1e-5, f"shift {delta} != E[max(8)]"

    centered = build(tropical_gauge_fix=False, parameterization="nsa")
    block2 = [m for m in centered.modules() if isinstance(m, TropicalCausalSelfAttention)][0]
    y2_nsa = block2.attend(q, k, v, kv_cache=None, pos0=0)
    block2.parameterization = "current"
    y2_cur = block2.attend(q, k, v, kv_cache=None, pos0=0)
    assert float((y2_nsa - y2_cur).abs().max()) < 1e-5, "centered arms must agree exactly-ish"


def test_parameterization_validation():
    """Unknown arms are rejected; nsa on mechanisms it does not touch is
    refused as a silent no-op rather than polluting run metadata."""
    from nanochat.gpt import GPT, GPTConfig

    with pytest.raises(ValueError, match="parameterization"):
        GPT(GPTConfig(n_layer=1, n_head=2, n_kv_head=2, n_embd=16, sequence_len=8, vocab_size=32, parameterization="bogus"))
    with pytest.raises(ValueError, match="silent no-op|parameterization"):
        GPT(GPTConfig(n_layer=1, n_head=2, n_kv_head=2, n_embd=16, sequence_len=8, vocab_size=32, parameterization="nsa"))


def test_coord_check_artifact_schema(tmp_path):
    """bp08 part 2: coordinate-check results wrap into the versioned
    mgr.bench.coord_curves.v1 payload in the house bench shape (mechanism /
    meta / results) so the verdict engine's arm matching, observation dedupe,
    and `bench:results.*` metric paths work unchanged."""
    import json

    res = {"attention_type": "standard", "widths": [16, 32], "activation_rms": {"16": 1.0, "32": 1.01}, "loglog_slope": 0.006, "r_squared": 0.9}
    path = tmp_path / "coord" / "standard_current"
    out = P.write_coord_check_artifact(res, path, parameterization="current", seed=7)
    assert out == path / "summary.json" and out.exists()
    payload = json.loads(out.read_text())
    assert payload["schema_version"] == P.COORD_CURVES_SCHEMA == "mgr.bench.coord_curves.v1"
    assert payload["bead"] == "model_guided_research-bp08"
    assert payload["kind"] == "coord-curve"
    assert payload["mechanism"] == "standard"
    assert payload["meta"]["seed"] == 7 and payload["meta"]["device"] == "cpu"
    assert payload["results"]["parameterization"] == "current"
    assert payload["results"]["loglog_slope"] == res["loglog_slope"]
    # writer creates parents and round-trips the harness's own result shape
    live = P.coordinate_check("standard", [64, 128], seed=0)
    out2 = P.write_coord_check_artifact(live, tmp_path / "nested" / "dir" / "std.json")
    live_payload = json.loads(out2.read_text())
    # JSON object keys stringify: int widths come back as "64"/"128"
    assert set(live_payload["results"]["activation_rms"]) == {"64", "128"}
    assert isinstance(live_payload["meta"]["generated_at"], str)


def test_measure_activation_scale_passes_parameterization_through():
    """The harness must expose both arms to the coordinate check (the note's
    separation requirement): the kwarg reaches GPTConfig without error."""
    import warnings

    warnings.filterwarnings("ignore")
    rms = P.measure_activation_scale("tropical", 64, seq_len=8, batch_size=2, parameterization="nsa")
    assert math.isfinite(rms) and rms > 0


def test_coordinate_check_exposes_both_evt_arms():
    """bp08: the harness must run the current-vs-nsa arms through the
    TROPICAL FFN (where the E[max] bias lives); the standard-FFN default
    keeps historical controls unchanged. Thresholds/flattening claims belong
    to the preregistered campaign (sizing-probe first) - here we pin only
    that both arms run, are finite, and the CLT control stays flat."""
    import warnings

    warnings.filterwarnings("ignore")
    arms = {
        mode: P.coordinate_check("tropical", [64, 128], seed=0, parameterization=mode, ffn_type="tropical")
        for mode in ("current", "nsa")
    }
    for mode, res in arms.items():
        assert math.isfinite(res["loglog_slope"]), f"{mode}: non-finite slope"
        assert all(v > 0 for v in res["activation_rms"].values())
    ctl = P.coordinate_check("standard", [64, 128], seed=0)
    assert abs(ctl["loglog_slope"]) < 0.05
