"""Tests for the fixed-FLOPs feature-ablation A/B harness (bead z7r):
the Welch two-sample delta helper and the multi-seed aggregation contract.

The harness trains via subprocess (covered by a tiny smoke run in the e2e
regression-gate scenario); here we test the pure statistical layer and the
aggregation/CSV emission against synthetic per-run summaries written to a
temp artifacts tree, so the suite is fast and deterministic.
"""

import json
import math
from pathlib import Path

from typer.testing import CliRunner

import cli

runner = CliRunner()


def test_welch_delta_identical_samples():
    d = cli._bench_welch_delta([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert d["delta"] == 0.0 and d["p_value"] == 1.0 and d["ci95"] == [0.0, 0.0]


def test_welch_delta_separated_significant():
    d = cli._bench_welch_delta([2.0, 2.1, 1.9, 2.05], [1.0, 1.1, 0.9, 0.95])
    assert d["delta"] > 0 and d["p_value"] < 0.05
    assert d["ci95"][0] > 0  # CI excludes zero on the positive side


def test_welch_delta_matches_scipy():
    from scipy import stats as sps

    a, b = [2.0, 2.1, 1.9, 2.05], [1.0, 1.1, 0.9, 0.95]
    d = cli._bench_welch_delta(a, b)
    ref = sps.ttest_ind(a, b, equal_var=False)
    assert abs(d["p_value"] - float(ref.pvalue)) < 1e-12
    assert abs(d["t_stat"] - float(ref.statistic)) < 1e-12


def test_welch_delta_insufficient_seeds():
    assert cli._bench_welch_delta([1.0], [1.0, 2.0]) is None
    assert cli._bench_welch_delta([1.0, 2.0], [1.0]) is None


def test_welch_delta_zero_variance_different_means():
    d = cli._bench_welch_delta([2.0, 2.0], [1.0, 1.0])
    assert d["delta"] == 1.0 and d["p_value"] == 0.0 and math.isinf(d["t_stat"]) and d["t_stat"] > 0
    d2 = cli._bench_welch_delta([1.0, 1.0], [2.0, 2.0])
    assert d2["t_stat"] == float("-inf") and d2["p_value"] == 0.0


def _write_bench_run(root: Path, suite: str, attn: str, seed: int, *, val_ce: float | None,
                     losses: list[float], tokens_s: float = 1000.0, mem: float | None = None) -> None:
    """Emit a nanochat train summary where bench-fixed-flops expects it."""
    run_dir = root / "bench" / "fixed_flops" / "nanochat" / suite / attn / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"losses": losses, "tokens_per_second": tokens_s}
    if val_ce is not None:
        results["val_ce_final"] = val_ce
    if mem is not None:
        results["peak_memory_allocated_gb"] = mem
    (run_dir / "summary.json").write_text(json.dumps({"schema_version": "mgr.telemetry.v1", "results": results}))


def test_aggregation_via_synthetic_summaries(tmp_path, monkeypatch):
    """Drive the real aggregation/CSV path by stubbing the per-run trainer to
    read pre-written synthetic summaries: standard is the baseline, tropical
    is clearly better on val CE, ultrametric is noisier/worse. Verifies the v2
    summary aggregates, the Welch comparisons, and the CSV."""
    suite = "synthetic-ab"
    arts = tmp_path / "artifacts"
    # standard ~ 3.0, tropical ~ 2.5 (better), ultrametric ~ 3.2 (worse)
    plan = {
        "standard": [3.00, 3.02, 2.98],
        "tropical": [2.50, 2.52, 2.48],
        "ultrametric": [3.20, 3.18, 3.25],
    }
    for attn, vals in plan.items():
        for seed, v in zip((0, 1, 2), vals):
            _write_bench_run(arts, suite, attn, seed, val_ce=v, losses=[v + 0.1, v], mem=0.5)

    # stub _run_train so the command does NOT spawn nanochat: return the dict
    # the real one would build from the synthetic summary we just wrote.
    real_summary_read = {}
    for attn, vals in plan.items():
        for seed, v in zip((0, 1, 2), vals):
            real_summary_read[(attn, seed)] = v

    import subprocess as _sub

    class _FakeProc:
        def __init__(self):
            self.stdout = ""
            self.stderr = ""
            self.returncode = 0

    def _fake_run(cmd, **kw):
        # find attn + seed from the argv and ensure the summary path exists
        # (already written above); the real code reads it back.
        return _FakeProc()

    monkeypatch.setattr(_sub, "run", _fake_run)

    result = runner.invoke(cli.app, [
        "bench-fixed-flops", "-a", "standard", "-a", "tropical", "-a", "ultrametric",
        "--seeds", "0,1,2", "--device", "cpu", "--target-flops", "1e6",
        "--no-auto-download-data", "--artifacts-dir", str(arts), "--run-id", suite,
    ])
    assert result.exit_code == 0, result.output

    summary = json.loads(
        (arts / "bench" / "fixed_flops" / "nanochat" / suite / "summary.json").read_text()
    )
    assert summary["schema_version"] == "mgr.bench.fixed_flops.v2"
    assert summary["score_metric"] == "val_ce_final"
    agg = summary["aggregates"]
    assert agg["standard"]["n_ok"] == 3
    assert abs(agg["tropical"]["metric_mean"] - 2.50) < 1e-9
    # tropical significantly better (negative delta, p < 0.05); ultrametric worse
    cmp_trop = summary["comparisons"]["tropical"]
    cmp_ultra = summary["comparisons"]["ultrametric"]
    assert cmp_trop["delta"] < 0 and cmp_trop["p_value"] < 0.05 and cmp_trop["ci95"][1] < 0
    assert cmp_ultra["delta"] > 0 and cmp_ultra["p_value"] < 0.05

    csv = (arts / "bench" / "fixed_flops" / "nanochat" / suite / "feature_ablate.csv").read_text()
    assert csv.splitlines()[0].startswith("attention_type,n_ok,metric")
    assert "tropical" in csv and "ultrametric" in csv


def test_score_metric_falls_back_to_train_tail_without_val(tmp_path, monkeypatch):
    """When val CE is absent (val-interval off), the suite scores on the
    train-loss tail rather than emitting a null-metric comparison."""
    suite = "synthetic-noval"
    arts = tmp_path / "artifacts"
    for attn, base in (("standard", 3.0), ("tropical", 2.5)):
        for seed in (0, 1):
            _write_bench_run(arts, suite, attn, seed, val_ce=None,
                             losses=[base + 0.2, base + 0.1, base])

    import subprocess as _sub

    class _FakeProc:
        stdout = ""
        stderr = ""
        returncode = 0

    monkeypatch.setattr(_sub, "run", lambda cmd, **kw: _FakeProc())

    result = runner.invoke(cli.app, [
        "bench-fixed-flops", "-a", "standard", "-a", "tropical",
        "--seeds", "0,1", "--device", "cpu", "--target-flops", "1e6", "--val-interval", "0",
        "--no-auto-download-data", "--artifacts-dir", str(arts), "--run-id", suite,
    ])
    assert result.exit_code == 0, result.output
    summary = json.loads(
        (arts / "bench" / "fixed_flops" / "nanochat" / suite / "summary.json").read_text()
    )
    assert summary["score_metric"] == "score"  # train-tail fallback
    assert summary["aggregates"]["tropical"]["metric_mean"] is not None


def test_schedule_arm_uses_attention_schedule_flag(tmp_path, monkeypatch):
    suite = "synthetic-schedule-arm"
    arts = tmp_path / "artifacts"
    for attn, val_ce in (("standard", 3.0), ("standard,tropical", 2.8)):
        _write_bench_run(arts, suite, attn, 0, val_ce=val_ce, losses=[val_ce + 0.1, val_ce])

    import subprocess as _sub

    commands: list[list[str]] = []

    class _FakeProc:
        stdout = ""
        stderr = ""
        returncode = 0

    def _fake_run(cmd, **kw):
        commands.append(list(cmd))
        return _FakeProc()

    monkeypatch.setattr(_sub, "run", _fake_run)

    result = runner.invoke(
        cli.app,
        [
            "bench-fixed-flops",
            "-a",
            "standard",
            "-a",
            "standard,tropical",
            "--seeds",
            "0",
            "--device",
            "cpu",
            "--target-flops",
            "1e6",
            "--no-auto-download-data",
            "--artifacts-dir",
            str(arts),
            "--run-id",
            suite,
        ],
    )
    assert result.exit_code == 0, result.output

    standard_cmd = next(cmd for cmd in commands if "--attention-type" in cmd)
    schedule_cmd = next(cmd for cmd in commands if "--attention-schedule" in cmd)
    assert standard_cmd[standard_cmd.index("--attention-type") + 1] == "standard"
    assert schedule_cmd[schedule_cmd.index("--attention-schedule") + 1] == "standard,tropical"
    assert "--attention-type" not in schedule_cmd

    summary = json.loads(
        (arts / "bench" / "fixed_flops" / "nanochat" / suite / "summary.json").read_text()
    )
    assert "standard,tropical" in summary["aggregates"]
