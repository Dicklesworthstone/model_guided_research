"""Tests for mgr report (bead yhy, roll-up half): cross-run aggregation over
an artifacts tree. The arm tables must use the engine's evidence semantics -
clean artifacts only, lineage-deduped (one observation per trained
checkpoint) - so a hand-written campaign.md and the roll-up can never
disagree about what the evidence says.
"""

import json
from pathlib import Path

from typer.testing import CliRunner

import cli

runner = CliRunner()

CLEAN_PROV = {"schema_version": "mgr.metrics.v1", "git_sha": "deadbeef", "git_dirty": False,
              "config_hash": "abc", "data_snapshot_hash": None, "tainted": False}
TAINTED_PROV = {**CLEAN_PROV, "git_dirty": True, "tainted": True}


def _train_summary(root: Path, name: str, *, mechanism: str, task: str, seed: int,
                   ce: float, flops: float = 3e14, tainted: bool = False) -> None:
    run_dir = root / "campaigns" / "t" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps({
        "schema_version": "mgr.telemetry.v1",
        "meta": {"run_id": name, "argv": ["train.py", "--seed", str(seed)]},
        "config": {"attention_type": mechanism},
        "dataset": {"data_dir": f"artifacts/diagnostics_e1/{task}"},
        "budget": {"target_flops": flops},
        "provenance": TAINTED_PROV if tainted else CLEAN_PROV,
        "results": {"train_ce_final": ce, "val_ce_final": None, "measured_time_s": 600.0},
    }))


def _eval_summary(root: Path, name: str, *, mechanism: str, task: str, em: float,
                  run_id: str | None = None, step: int = 100, prior: float = 0.625,
                  flops: float = 3e14, tainted: bool = False,
                  generated_at: str = "2026-06-12T00:00:00Z") -> None:
    run_dir = root / "evals" / "tasks" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps({
        "schema_version": "mgr.evaltasks.v2",
        "kind": "eval-tasks",
        "meta": {
            "generated_at": generated_at,
            "checkpoint": {"attention_type": mechanism, "step": step,
                           "budget": {"target_flops": flops},
                           "lineage": {"run_id": run_id or name, "parent_run_ids": []}},
            "seeds": [0, 1, 2],
        },
        "provenance": TAINTED_PROV if tainted else CLEAN_PROV,
        "tasks": {task: {
            "exact_match": {"greedy": {"held_out": {"mean": em, "per_seed": [em] * 3}}},
            "answer_prior": {"held_out": {"mean": prior, "per_seed": [prior] * 3,
                                          "majority_answer": "x"}},
        }},
    }))


def test_report_rolls_up_arms_with_engine_semantics(tmp_path):
    """The arm tables aggregate over CLEAN, lineage-deduped evidence: a
    tainted eval is counted-but-excluded and a same-(run_id, step) re-eval
    never double-counts."""
    arts = tmp_path / "artifacts"
    _train_summary(arts, "dyck-braid-s30", mechanism="braid", task="dyck", seed=30, ce=0.80)
    _train_summary(arts, "dyck-braid-s31", mechanism="braid", task="dyck", seed=31, ce=0.90)
    _train_summary(arts, "dyck-standard-s30", mechanism="standard", task="dyck", seed=30,
                   ce=1.20, tainted=True)
    _eval_summary(arts, "e2-dyck-braid-s30", mechanism="braid", task="dyck", em=0.95,
                  run_id="dyck-braid-s30")
    _eval_summary(arts, "e2-dyck-braid-s31", mechanism="braid", task="dyck", em=0.97,
                  run_id="dyck-braid-s31")
    # re-eval of the SAME checkpoint: deduped, never a second observation
    _eval_summary(arts, "e2-dyck-braid-s30-r2", mechanism="braid", task="dyck", em=0.10,
                  run_id="dyck-braid-s30", generated_at="2026-06-11T00:00:00Z")
    # tainted eval: counted, excluded from stats
    _eval_summary(arts, "e2-dyck-standard-s30", mechanism="standard", task="dyck", em=0.91,
                  run_id="dyck-standard-s30", tainted=True)

    result = runner.invoke(cli.app, [
        "report", "--artifacts", str(arts), "--out", str(tmp_path / "reports"), "--run-id", "r1",
    ])
    assert result.exit_code == 0, result.output

    payload = json.loads((tmp_path / "reports" / "r1" / "report.json").read_text())
    assert payload["schema_version"] == "mgr.report.v1"
    assert payload["counts"]["train"] == {"total": 3, "tainted": 1}
    assert payload["counts"]["evaltasks"] == {"total": 4, "tainted": 1}
    assert payload["eval_tainted_excluded"] == 1

    # train arm grouping: braid clean pair averaged; tainted standard CE excluded
    tg = {(g["task"], g["mechanism"]): g for g in payload["train_groups"]}
    braid_tg = tg[("dyck", "braid")]
    assert braid_tg["n"] == 2 and braid_tg["n_tainted"] == 0
    assert braid_tg["train_ce_final"].startswith("0.85")
    std_tg = tg[("dyck", "standard")]
    assert std_tg["n"] == 1 and std_tg["n_tainted"] == 1
    assert std_tg["train_ce_final"] == "-"

    # eval arm grouping: dedupe keeps the NEWEST braid-s30 eval (em=0.95, not
    # the older 0.10 re-eval); tainted standard arm contributes no group
    eg = {(g["task"], g["mechanism"]): g for g in payload["eval_groups"]}
    braid_eg = eg[("dyck", "braid")]
    assert braid_eg["n_models"] == 2
    assert braid_eg["em_held_out"].startswith("0.96")  # mean(0.95, 0.97)
    assert braid_eg["answer_prior"] == "0.625"
    assert ("dyck", "standard") not in eg

    md = (tmp_path / "reports" / "r1" / "report.md").read_text()
    assert "| dyck | braid | 3e+14 | 2 | 0.96" in md
    assert "1 tainted eval artifact(s) excluded" in md
    assert payload["git"] is not None and payload["roots"] == [str(arts)]


def test_report_handles_empty_tree(tmp_path):
    """An empty root produces an empty (but valid) report, not a crash."""
    result = runner.invoke(cli.app, [
        "report", "--artifacts", str(tmp_path / "nothing"),
        "--out", str(tmp_path / "reports"), "--run-id", "empty",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "reports" / "empty" / "report.json").read_text())
    assert payload["counts"] == {} and payload["train_groups"] == [] and payload["eval_groups"] == []


def _cert_summary(root: Path, name: str, *, mechanism: str) -> Path:
    run_dir = root / "certs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / "summary.json"
    p.write_text(json.dumps({
        "schema_version": 1, "kind": "certify",
        "git": {"dirty": False, "sha": "deadbeef"},
        "checks": [{"mechanism": mechanism, "name": f"{mechanism}.smoke", "measured": 0.0}],
    }))
    return p


def test_status_reports_staleness_and_engine_view(tmp_path, monkeypatch):
    """2vs: mgr status answers existence (kinds + ages), cert freshness
    (mtime vs mechanism source), and what the engine would rule on today
    (planted: one adjudicable hypothesis, one blocked for missing
    artifacts); --json is schema-versioned; --write-index persists it."""
    import os
    import time as time_mod

    arts = tmp_path / "artifacts"
    _train_summary(arts, "dyck-braid-s0", mechanism="braid", task="dyck", seed=0, ce=0.9)
    _eval_summary(arts, "e-ultra-0", mechanism="ultrametric", task="hier", em=0.8, flops=1e14)
    _eval_summary(arts, "e-ultra-1", mechanism="ultrametric", task="hier", em=0.82, flops=1e14)
    _eval_summary(arts, "e-std-0", mechanism="standard", task="hier", em=0.50, flops=1e14)
    _eval_summary(arts, "e-std-1", mechanism="standard", task="hier", em=0.51, flops=1e14)
    stale_cert = _cert_summary(arts, "trop-old", mechanism="tropical")
    os.utime(stale_cert, (time_mod.time() - 10 * 86400,) * 2)  # cert predates source
    fresh_cert = _cert_summary(arts, "ultra-new", mechanism="ultrametric")
    os.utime(fresh_cert, (time_mod.time() + 3600,) * 2)  # newer than any source file

    registry = tmp_path / "registry.yaml"
    blocked = _hyp_status()
    blocked["id"] = "hyp-status-blocked"
    blocked["mechanisms"] = ["braid"]  # no braid eval artifacts -> blocked
    registry.write_text(
        "schema_version: 1\nhypotheses:\n" + _yaml_entry_status(_hyp_status())
        + _yaml_entry_status(blocked)
    )
    monkeypatch.setattr(cli, "_hypotheses_registry_path", lambda: registry)

    result = runner.invoke(cli.app, [
        "status", "--artifacts-dir", str(arts), "--json", "--write-index",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads((arts / "index.json").read_text())
    assert payload["schema_version"] == "mgr.status.v1"
    assert payload["artifact_kinds"]["campaigns"]["count"] == 1
    assert payload["artifact_kinds"]["evals"]["count"] == 4
    assert payload["evidence_pool"]["indexed"] == 7
    certs = {c["mechanism"]: c for c in payload["certificates"]}
    assert certs["tropical"]["stale"] is True
    assert certs["ultrametric"]["stale"] is False
    eng = payload["engine_today"]
    assert eng["would_rule"] == 1 and eng["blocked"] == 1
    states = {h["id"]: h for h in eng["hypotheses"]}
    assert states["hyp-status-ok"]["state"] == "would_rule"
    assert states["hyp-status-blocked"]["state"] == "blocked"
    assert (arts / "index.md").read_text().startswith("# Artifacts index")


def _hyp_status():
    return {
        "id": "hyp-status-ok",
        "statement": "status test claim",
        "mechanisms": ["ultrametric"],
        "prediction": {
            "metric_path": "evaltasks:tasks.hier.exact_match.greedy.held_out.mean",
            "comparator": ">=", "threshold_kind": "absolute_delta", "threshold": 0.05,
            "baseline": {"mechanism": "standard", "equal_flops": True}, "min_seeds": 1,
        },
        "status": "open",
    }


def _yaml_entry_status(h) -> str:
    pred = h["prediction"]
    return "\n".join([
        f"  - id: {h['id']}",
        f"    statement: {json.dumps(h['statement'])}",
        f"    mechanisms: [{', '.join(h['mechanisms'])}]",
        "    source: {kind: human, provenance: test}",
        '    date_registered: "2026-06-12"',
        "    prediction:",
        f"      metric_path: {json.dumps(pred['metric_path'])}",
        f"      comparator: \"{pred['comparator']}\"",
        f"      threshold_kind: {pred['threshold_kind']}",
        f"      threshold: {pred['threshold']}",
        f"      baseline: {{mechanism: {pred['baseline']['mechanism']}, equal_flops: true}}",
        f"      min_seeds: {pred['min_seeds']}",
        "    status: open",
        "    evidence: []",
        "    verdict_history: []",
        "",
    ])


def test_report_survives_mixed_none_flops(tmp_path):
    """Fresh-eyes regression: legacy artifacts may lack budget.target_flops;
    a None mixed with floats in the group-sort key must not TypeError."""
    arts = tmp_path / "artifacts"
    _train_summary(arts, "a", mechanism="braid", task="dyck", seed=0, ce=0.9, flops=3e14)
    # same (task, mechanism) with a missing budget -> None group key member
    run_dir = arts / "campaigns" / "t" / "b"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({
        "schema_version": "mgr.telemetry.v1",
        "meta": {"run_id": "b", "argv": []},
        "config": {"attention_type": "braid"},
        "dataset": {"data_dir": "artifacts/diagnostics_e1/dyck"},
        "budget": {},
        "provenance": CLEAN_PROV,
        "results": {"train_ce_final": 1.0, "measured_time_s": 60.0},
    }))
    result = runner.invoke(cli.app, [
        "report", "--artifacts", str(arts), "--out", str(tmp_path / "r"), "--run-id", "mix",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "r" / "mix" / "report.json").read_text())
    flops_seen = {g["target_flops"] for g in payload["train_groups"]}
    assert flops_seen == {3e14, None}


# ---------------------------------------------------------------------------
# Live training dashboard (bead nyp): the renderer half of nanochat.report.
# No Live loop in tests - render through a recording console.


def test_sparkline_shapes_and_nan_marker():
    from nanochat.report import sparkline

    assert sparkline([]) == ""
    assert set(sparkline([1.0, 1.0, 1.0])) == {"▄"}  # constant series, mid block
    s = sparkline([0.0, float("nan"), 1.0])
    assert "✕" in s and len(s) == 3
    assert sparkline(list(range(100)), width=16).startswith("▁") and len(sparkline(list(range(100)), width=16)) == 16


def test_drift_flag_trailing_band():
    from nanochat.report import _drift_flag

    stable = [1.0, 1.001, 0.999, 1.0, 1.0005, 0.9995, 1.0, 1.0002, 1.0001]
    assert not _drift_flag(stable)
    excursion = stable[:-1] + [1.5]
    assert _drift_flag(excursion)
    assert not _drift_flag([1.0, 2.0])  # too little history


def test_training_dashboard_renders_and_exports_html(tmp_path):
    """End-to-end without Live: synthetic records (core keys + a symplectic
    diagnostic + a val record) render all panels; close() writes the HTML
    snapshot with the run header and the auto-adopted invariant."""
    from rich.console import Console

    from nanochat.report import TrainingDashboard

    html = tmp_path / "dash" / "run-x.html"
    dash = TrainingDashboard(
        run_id="run-x", mechanism="reversible", max_steps=10,
        flags={"attention": "reversible", "mode": "symplectic", "tied": True},
        html_path=html,
    )
    for step in range(10):
        dash.observe({
            "type": "step", "step": step, "loss": 4.0 - 0.1 * step, "lr": 1e-3,
            "grad_norm": 0.5, "tokens_per_s": 1000.0, "tflops": 0.04, "peak_mem_gb": None,
            "elapsed_s": float(step), "symplectic_shadow_energy_mean": 0.01 * step,
        })
    dash.observe({"type": "val", "step": 9, "val_loss": 3.2, "train_loss": 3.1})

    console = Console(record=True, width=120, force_terminal=True)
    console.print(dash._render())
    text = console.export_text()
    for needle in ("run-x", "loss", "val_ce", "grad_norm", "feature flags",
                   "symplectic_shadow_energy_mean", "invariants"):
        assert needle in text, f"missing {needle!r} in rendered dashboard"
    assert "NaN" not in text  # finite run shows the green check, not the NaN flag

    dash.observe({"type": "step", "step": 10, "loss": float("nan"), "lr": 1e-3,
                  "grad_norm": 0.5, "tokens_per_s": 1000.0, "tflops": 0.04,
                  "peak_mem_gb": None, "elapsed_s": 10.0})
    console2 = Console(record=True, width=120, force_terminal=True)
    console2.print(dash._render())
    assert "NaN" in console2.export_text()  # the NaN flag latches visibly

    dash.close({"train_ce_final": 3.0, "val_ce_final": 3.2})
    content = html.read_text()
    assert "run-x" in content and "symplectic_shadow_energy_mean" in content


def test_dashboard_per_head_heatmap(tmp_path):
    """bead 92m: per-head metric vectors (route margins / entropy) carried in a
    step record render as a heatmap panel inline AND in the HTML export, with a
    cross-head diversity (σ) summary. Scalars stay sparklines, not heat rows."""
    from rich.console import Console

    from nanochat.report import TrainingDashboard

    html = tmp_path / "dash" / "trop.html"
    dash = TrainingDashboard(run_id="trop", mechanism="tropical", max_steps=4, html_path=html)
    for step in range(4):
        dash.observe({
            "type": "step", "step": step, "loss": 4.0 - step, "lr": 1e-3,
            "grad_norm": 0.5, "tokens_per_s": 1000.0, "tflops": 0.04, "peak_mem_gb": None,
            "elapsed_s": float(step),
            # per-head vector (4 heads) — the field train.py already emits
            "tropical_gamma_head_mean": [0.10 + 0.01 * step, 0.05, 0.20, 0.02],
            "attn_entropy_head_mean": [2.1, 2.0, 1.5, 2.2],
            "lr_groups": [1e-3, 1e-3],  # a core list field: must NOT become a heat row
        })

    assert "tropical_gamma_head_mean" in dash._head_latest
    assert "attn_entropy_head_mean" in dash._head_latest
    assert "lr_groups" not in dash._head_latest  # excluded (core key)
    assert dash._head_latest["tropical_gamma_head_mean"][0] == 0.13  # latest step wins

    console = Console(record=True, width=120, force_terminal=True)
    console.print(dash._render())
    text = console.export_text()
    assert "per-head route diversity" in text
    assert "tropical_gamma" in text and "attn_entropy" in text
    assert "div σ" in text

    dash.close()
    assert "per-head route diversity" in html.read_text()


def test_dashboard_head_heatmap_toggle_off():
    """show_head_heatmaps=False suppresses both capture and panel."""
    from rich.console import Console

    from nanochat.report import TrainingDashboard

    dash = TrainingDashboard(run_id="x", mechanism="tropical", max_steps=2, show_head_heatmaps=False)
    dash.observe({
        "type": "step", "step": 0, "loss": 3.0, "lr": 1e-3, "grad_norm": 0.5,
        "tokens_per_s": 1.0, "tflops": 0.0, "peak_mem_gb": None, "elapsed_s": 0.0,
        "tropical_gamma_head_mean": [0.1, 0.2, 0.3, 0.4],
    })
    assert dash._head_latest == {}
    console = Console(record=True, width=120, force_terminal=True)
    console.print(dash._render())
    assert "per-head route diversity" not in console.export_text()
