"""Tests for the scaling-law report harness (bead w94.2).

The report must be re-derivable from artifacts ALONE, so these tests feed it
synthetic sweep suites whose ground truth is KNOWN BY CONSTRUCTION: a fake
rung ladder generated from L(C) = a*C^-b + c with seeded noise must yield a
fitted exponent that recovers the truth inside its own bootstrap CI. We also
pin byte-stable regeneration, the thin-ladder refusal, the no-significance
headline, and the G2 JSON block contract.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import cli

runner = CliRunner()


# ---------------------------------------------------------------------------
# synthetic artifact factory


def _write_seed_run(
    suite_dir: Path,
    rung_index: int,
    seed: int,
    *,
    target_loss: float,
    n_steps: int = 120,
    rng: np.random.Generator | None = None,
) -> None:
    """One fake train run: metrics.jsonl decaying into target_loss + summary."""
    rng = rng or np.random.default_rng(seed)
    seed_dir = suite_dir / f"rung_{rung_index}" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    start = target_loss * 3.0
    steps = np.arange(1, n_steps + 1, dtype=float)
    curve = target_loss + (start - target_loss) * np.exp(-steps / (n_steps / 6.0))
    curve += rng.normal(0.0, target_loss * 0.002, size=n_steps)
    lines = [
        json.dumps({"type": "header", "schema_version": 1}),
        *(
            json.dumps({"type": "step", "step": i + 1, "loss": float(curve[i])})
            for i in range(n_steps)
        ),
    ]
    (seed_dir / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (seed_dir / "summary.json").write_text(
        json.dumps({"results": {"losses": [float(v) for v in curve[-5:]]}}),
        encoding="utf-8",
    )


def _make_suite(
    root: Path,
    mechanism: str,
    *,
    cs_flops: list[float],
    true_losses: list[float],
    seeds: int = 1,
    rng_seed: int = 7,
) -> Path:
    """A w94.1-shaped suite whose rung losses follow a known law."""
    suite = root / "artifacts" / "scaling" / mechanism / f"sweep_{mechanism}"
    rng = np.random.default_rng(rng_seed)
    rungs = []
    for idx, (c, loss) in enumerate(zip(cs_flops, true_losses)):
        name = f"rung{idx}"
        for seed in range(seeds):
            _write_seed_run(suite, idx, seed, target_loss=loss, rng=rng)
        rungs.append(
            {
                "index": idx,
                "name": name,
                "feasible": True,
                "status": "done",
                "target_flops_est": c,
                "flops_per_token_est": None,
                "planned_max_steps": None,
                "runs": [
                    {"seed": s, "status": "done", "summary_path": str(suite / f"rung_{idx}" / f"seed_{s}" / "summary.json")}
                    for s in range(seeds)
                ],
            }
        )
    (suite / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite": "scaling_sweep",
                "mechanism": mechanism,
                "ladder": "synthetic",
                "sweep_config": {"batch_size": 8, "sequence_len": 256},
                "rungs": rungs,
            }
        ),
        encoding="utf-8",
    )
    return suite


def _invoke(out: Path, *suite_dirs: Path, extra: tuple[str, ...] = ()):
    argv = ["scaling-report"]
    for s in suite_dirs:
        argv += ["--runs", str(s)]
    argv += ["--out", str(out), "--bootstrap", "400", "--no-plot", *extra]
    return runner.invoke(cli.app, argv)


# ---------------------------------------------------------------------------
# pure helpers


def test_tail_loss_is_mean_of_last_fraction():
    vals = [10.0] * 90 + [1.0] * 10
    assert cli._scaling_report_tail_loss(vals, 0.1) == pytest.approx(1.0)
    assert cli._scaling_report_tail_loss(vals, 1.0) == pytest.approx(9.1)
    assert cli._scaling_report_tail_loss([], 0.1) is None


def _known_law_points(a=1.0e4, b_true=0.35, c=0.7):
    cs = [1e12, 3e12, 1e13, 3e13, 1e14]
    # NOTE: comprehension uses x so the floor c is not shadowed by the loop var
    return cs, [a * x ** (-b_true) + c for x in cs]



def test_read_losses_prefers_metrics_falls_back_to_summary(tmp_path):
    rng = np.random.default_rng(3)
    _write_seed_run(tmp_path / "run_a", 0, 0, target_loss=2.0, rng=rng)
    val, src = cli._scaling_report_read_losses(tmp_path / "run_a" / "rung_0" / "seed_0", 0.1)
    assert src == "metrics" and val == pytest.approx(2.0, rel=0.01)
    d = tmp_path / "run_b" / "rung_0" / "seed_0"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"results": {"losses": [9.0, 8.0, 7.0]}}))
    val, src = cli._scaling_report_read_losses(d, 0.34)
    assert src == "summary" and val == pytest.approx((8.0 + 7.0) / 2)


def test_rung_compute_uses_target_or_derives_from_budget():
    assert cli._scaling_report_rung_compute({"target_flops_est": 123}, {}) == 123.0
    derived = cli._scaling_report_rung_compute(
        {"flops_per_token_est": 10, "planned_max_steps": 5},
        {"sweep_config": {"batch_size": 4, "sequence_len": 16}},
    )
    assert derived == 10 * 4 * 16 * 5


# ---------------------------------------------------------------------------
# end-to-end over synthetic artifacts


def test_synthetic_recovery_within_bootstrap_ci(tmp_path):
    a, b_true, c = 1.0e4, 0.35, 0.7
    cs, ls = _known_law_points(a, b_true, c)
    suite = _make_suite(tmp_path, "synthA", cs_flops=cs, true_losses=ls, rng_seed=11)
    out = tmp_path / "report"
    result = _invoke(out, suite)
    assert result.exit_code == 0
    fits = json.loads((out / "fits.json").read_text())["fits"]["synthA"]
    lo, hi = fits["exponent_b_ci95"]
    assert lo <= b_true <= hi, f"truth {b_true} outside recovered CI [{lo}, {hi}]"
    assert abs(fits["exponent_b"] - b_true) < 0.05
    assert abs(fits["floor_c"] - c) < 0.05
    assert fits["r2_original_scale"] > 0.99


def test_report_regenerates_byte_stably(tmp_path):
    cs, ls = _known_law_points()
    suite = _make_suite(tmp_path, "stableM", cs_flops=cs, true_losses=ls, rng_seed=5)
    out1, out2 = tmp_path / "r1", tmp_path / "r2"
    assert _invoke(out1, suite).exit_code == 0
    assert _invoke(out2, suite).exit_code == 0
    for name in ("fits.json", "scaling_report.md"):
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes(), name


def test_thin_ladder_refuses_saturating_fit_but_keeps_plain(tmp_path):
    cs = [1e12, 1e13]
    suite = _make_suite(tmp_path, "thin", cs_flops=cs, true_losses=[5.0, 3.0])
    out = tmp_path / "report"
    result = _invoke(out, suite)
    assert result.exit_code == 0
    md = (out / "scaling_report.md").read_text()
    assert "Saturating fit REFUSED" in md and "underdetermined" in md
    assert "Plain power law:" in md
    fits = json.loads((out / "fits.json").read_text())["fits"]["thin"]
    assert fits["exponent_b"] is None and fits["plain_power_law_b"] is not None


def test_no_significance_headline_when_cis_overlap(tmp_path):
    # two mechanisms drawn from the SAME law: any honest test must find no separation
    cs, ls = _known_law_points()
    s1 = _make_suite(tmp_path, "twinX", cs_flops=cs, true_losses=ls, rng_seed=21)
    s2 = _make_suite(tmp_path, "twinY", cs_flops=cs, true_losses=ls, rng_seed=22)
    out = tmp_path / "report"
    result = _invoke(out, s1, s2)
    assert result.exit_code == 0
    md = (out / "scaling_report.md").read_text()
    assert "NO pairwise exponent differences are significant" in md
    block = json.loads(_g2_block(md))
    assert len(block["pairwise_exponent_tests"]) == 1
    assert block["pairwise_exponent_tests"][0]["significant"] is False


def _g2_block(md_text: str) -> str:
    marker = "```json\n"
    start = md_text.index(marker) + len(marker)
    end = md_text.index("```", start)
    return md_text[start:end]


def test_g2_schema_contract_present(tmp_path):
    cs, ls = _known_law_points()
    suite = _make_suite(tmp_path, "g2m", cs_flops=cs, true_losses=ls)
    out = tmp_path / "report"
    assert _invoke(out, suite).exit_code == 0
    md = (out / "scaling_report.md").read_text()
    block = json.loads(_g2_block(md))
    assert block["schema"] == "mgr.scaling.v1"
    fit = block["fits"]["g2m"]
    for key in ("exponent_b", "exponent_b_ci95", "amplitude_a", "floor_c", "r2_original_scale", "n_rungs"):
        assert key in fit
    assert math.isfinite(fit["exponent_b"])
