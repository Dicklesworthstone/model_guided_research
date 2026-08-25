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
    assert d is not None
    assert d["delta"] == 0.0 and d["p_value"] == 1.0 and d["ci95"] == [0.0, 0.0]


def test_welch_delta_separated_significant():
    d = cli._bench_welch_delta([2.0, 2.1, 1.9, 2.05], [1.0, 1.1, 0.9, 0.95])
    assert d is not None
    assert d["delta"] > 0 and d["p_value"] < 0.05
    assert d["ci95"][0] > 0  # CI excludes zero on the positive side


def test_welch_delta_matches_scipy():
    from scipy import stats as sps

    a, b = [2.0, 2.1, 1.9, 2.05], [1.0, 1.1, 0.9, 0.95]
    d = cli._bench_welch_delta(a, b)
    assert d is not None
    ref = sps.ttest_ind(a, b, equal_var=False)
    assert abs(d["p_value"] - float(ref.pvalue)) < 1e-12
    assert abs(d["t_stat"] - float(ref.statistic)) < 1e-12


def test_welch_delta_insufficient_seeds():
    assert cli._bench_welch_delta([1.0], [1.0, 2.0]) is None
    assert cli._bench_welch_delta([1.0, 2.0], [1.0]) is None


def test_welch_delta_zero_variance_different_means():
    d = cli._bench_welch_delta([2.0, 2.0], [1.0, 1.0])
    assert d is not None
    assert d["delta"] == 1.0 and d["p_value"] == 0.0 and math.isinf(d["t_stat"]) and d["t_stat"] > 0
    d2 = cli._bench_welch_delta([1.0, 1.0], [2.0, 2.0])
    assert d2 is not None
    assert d2["t_stat"] == float("-inf") and d2["p_value"] == 0.0


def _write_bench_run(
    root: Path,
    suite: str,
    attn: str,
    seed: int,
    *,
    val_ce: float | None,
    losses: list[float],
    tokens_s: float = 1000.0,
    mem: float | None = None,
) -> None:
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

    result = runner.invoke(
        cli.app,
        [
            "bench-fixed-flops",
            "-a",
            "standard",
            "-a",
            "tropical",
            "-a",
            "ultrametric",
            "--seeds",
            "0,1,2",
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

    summary = json.loads((arts / "bench" / "fixed_flops" / "nanochat" / suite / "summary.json").read_text())
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
            _write_bench_run(arts, suite, attn, seed, val_ce=None, losses=[base + 0.2, base + 0.1, base])

    import subprocess as _sub

    class _FakeProc:
        stdout = ""
        stderr = ""
        returncode = 0

    monkeypatch.setattr(_sub, "run", lambda cmd, **kw: _FakeProc())

    result = runner.invoke(
        cli.app,
        [
            "bench-fixed-flops",
            "-a",
            "standard",
            "-a",
            "tropical",
            "--seeds",
            "0,1",
            "--device",
            "cpu",
            "--target-flops",
            "1e6",
            "--val-interval",
            "0",
            "--no-auto-download-data",
            "--artifacts-dir",
            str(arts),
            "--run-id",
            suite,
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads((arts / "bench" / "fixed_flops" / "nanochat" / suite / "summary.json").read_text())
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

    summary = json.loads((arts / "bench" / "fixed_flops" / "nanochat" / suite / "summary.json").read_text())
    assert "standard,tropical" in summary["aggregates"]


def _fake_scorecard_generator(task: str, *, out_dir: Path, size: int, seed: int) -> dict:
    task_dir = out_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"task": task, "size": size, "seed": seed}
    (task_dir / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def _fake_scorecard_flops(mechanisms: list[str], **_kwargs) -> dict[str, int]:
    return {mechanism: 100_000 for mechanism in mechanisms}


def _write_fake_scorecard_train(cmd: list[str]) -> None:
    artifacts = Path(cmd[cmd.index("--artifacts-dir") + 1])
    topic = cmd[cmd.index("--artifacts-topic") + 1]
    run_id = cmd[cmd.index("--run-id") + 1]
    mechanism = cmd[cmd.index("--attention-type") + 1]
    budget = float(cmd[cmd.index("--target-flops") + 1])
    run_dir = artifacts / "runs" / topic / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    summary = {
        "schema_version": "mgr.telemetry.v1",
        "kind": "runs",
        "config": {"attention_type": mechanism},
        "budget": {"target_flops": budget, "planned_total_flops_est": budget},
        "provenance": {"tainted": False},
        "results": {"losses": [3.0, 2.5], "tokens_per_second": 100.0},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))


def _write_fake_scorecard_eval(cmd: list[str]) -> None:
    artifacts = Path(cmd[cmd.index("--artifacts-dir") + 1])
    run_id = cmd[cmd.index("--run-id") + 1]
    checkpoint = Path(cmd[cmd.index("--checkpoint") + 1])
    task = cmd[cmd.index("--task") + 1]
    mechanism = checkpoint.parent.parent.name
    train_summary = json.loads((checkpoint.parent / "summary.json").read_text())
    budget = train_summary["budget"]["target_flops"]
    metric = 0.9 if mechanism == "tropical" else 1.0
    run_dir = artifacts / "evals" / "tasks" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "mgr.evaltasks.v3",
        "kind": "eval-tasks",
        "meta": {
            "run_id": run_id,
            "checkpoint": {
                "dir": str(checkpoint),
                "step": 0,
                "attention_type": mechanism,
                "budget": {"target_flops": budget},
                "lineage": {"run_id": str(checkpoint.parent)},
                "model_config": {"attention_type": mechanism},
            },
        },
        "provenance": {"tainted": False},
        "train_provenance": {"tainted": False},
        "tasks": {
            task: {
                "exact_match": {
                    "greedy": {
                        "in_range": {"mean": metric, "per_seed": [metric]},
                        "held_out": {"mean": metric, "per_seed": [metric]},
                    }
                },
                "answer_prior": {
                    "in_range": {"mean": 0.5, "per_seed": [0.5]},
                    "held_out": {"mean": 0.5, "per_seed": [0.5]},
                },
                "perplexity": {"in_range": metric, "held_out": metric},
            }
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))


def _fake_scorecard_success(cmd: list[str], *, timeout_s: float) -> tuple[int, str, str]:
    assert timeout_s > 0
    if "nanochat.train" in cmd:
        _write_fake_scorecard_train(cmd)
    else:
        assert "eval-tasks" in cmd
        _write_fake_scorecard_eval(cmd)
    return 0, "ok", ""


def test_scorecard_full_sparse_matrix_is_exactly_29_claim_covered_cells():
    from nanochat.diagnostics_data import DEFAULT_TASKS, TASKS
    from nanochat.gpt import SUPPORTED_ATTENTION_TYPES

    requested = [
        "ultrametric:hier",
        "fractal:hier",
        "hyperbolic:hier",
        "simplicial:rel",
        "quaternion:rot",
        "octonion:rot",
        "braid:copyops",
        "reversible:copyops",
        "surreal:arith",
        "gauge:bag",
        "reversible:bag",
        "tropical:placebo",
    ]
    mechanisms, tasks, matrix = cli._scorecard_matrix(
        mechanism_options=None,
        task_options=None,
        cell_options=requested,
        supported_mechanisms=list(SUPPORTED_ATTENTION_TYPES),
        supported_tasks=set(TASKS),
        default_tasks=list(DEFAULT_TASKS),
    )
    assert len(matrix) == 29
    assert len(set(matrix)) == 29
    assert mechanisms == [
        "standard",
        "ultrametric",
        "fractal",
        "hyperbolic",
        "simplicial",
        "quaternion",
        "octonion",
        "braid",
        "reversible",
        "surreal",
        "gauge",
        "tropical",
    ]
    assert tasks == ["hier", "rel", "rot", "copyops", "arith", "bag", "placebo"]
    counts = {mechanism: sum(pair[0] == mechanism for pair in matrix) for mechanism in mechanisms}
    assert counts == {
        "standard": 7,
        "ultrametric": 2,
        "fractal": 2,
        "hyperbolic": 2,
        "simplicial": 2,
        "quaternion": 2,
        "octonion": 2,
        "braid": 2,
        "reversible": 3,
        "surreal": 2,
        "gauge": 2,
        "tropical": 1,
    }
    hypotheses = cli._scorecard_hypothesis_snapshot(
        [
            "hyp-ultrametric-hier-heldout-depth",
            "hyp-fractal-hier-heldout-depth",
            "hyp-hyperbolic-hierarchy-retrieval",
            "hyp-simplicial-two-hop-composition",
            "hyp-quaternion-rotation-composition",
            "hyp-octonion-rotation-composition",
            "hyp-braid-length-generalization",
            "hyp-reversible-copyops-inversion",
            "hyp-surreal-wide-dynamic-range",
            "hyp-gauge-bag-conservation",
            "hyp-placebo-no-winner",
            "hyp-hyperbolic-placebo-specificity",
        ]
    )
    cli._scorecard_validate_hypothesis_coverage(hypotheses, matrix)


def test_scorecard_sparse_cells_snapshot_claims_and_resolve_reversible_kv(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_scorecard_generate_task", _fake_scorecard_generator)
    monkeypatch.setattr(cli, "_scorecard_flops_per_step", _fake_scorecard_flops)
    monkeypatch.setattr(
        cli,
        "_get_git_info",
        lambda: {"commit": "abc1234", "commit_full": "a" * 40, "branch": "main", "dirty": False},
    )
    launched_kv_heads: list[tuple[str, int]] = []

    def launch(cmd: list[str], *, timeout_s: float) -> tuple[int, str, str]:
        if "nanochat.train" in cmd:
            mechanism = cmd[cmd.index("--attention-type") + 1]
            kv_heads = int(cmd[cmd.index("--n-kv-head") + 1])
            launched_kv_heads.append((mechanism, kv_heads))
        return _fake_scorecard_success(cmd, timeout_s=timeout_s)

    monkeypatch.setattr(cli, "_scorecard_launch", launch)
    artifacts = tmp_path / "artifacts"
    result = runner.invoke(
        cli.app,
        [
            "scorecard",
            "--cell",
            "braid:copyops",
            "--cell",
            "reversible:copyops",
            "-H",
            "hyp-braid-length-generalization",
            "--seeds",
            "1",
            "--n-kv-head",
            "2",
            "--budget",
            "1e6",
            "--dataset-size",
            "6",
            "--examples",
            "1",
            "--artifacts-dir",
            str(artifacts),
            "--run-id",
            "sparse-contract",
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads((artifacts / "scorecards" / "sparse-contract" / "manifest.json").read_text())
    assert manifest["config"]["matrix"] == [
        {"mechanism": "standard", "task": "copyops"},
        {"mechanism": "braid", "task": "copyops"},
        {"mechanism": "reversible", "task": "copyops"},
        {"mechanism": "standard", "task": "placebo"},
        {"mechanism": "braid", "task": "placebo"},
        {"mechanism": "reversible", "task": "placebo"},
    ]
    assert manifest["config"]["hypothesis_ids"] == ["hyp-braid-length-generalization"]
    assert [item["id"] for item in manifest["config"]["hypothesis_snapshot"]] == ["hyp-braid-length-generalization"]
    assert not ({"status", "evidence", "verdict_history", "manual_hold"} & manifest["config"]["hypothesis_snapshot"][0].keys())
    assert [(cell["mechanism"], cell["resolved_n_kv_head"]) for cell in manifest["cells"]] == [
        ("standard", 2),
        ("braid", 2),
        ("reversible", 2),
        ("standard", 2),
        ("braid", 2),
        ("reversible", 2),
    ]
    assert launched_kv_heads == [
        ("standard", 2),
        ("braid", 2),
        ("reversible", 2),
        ("standard", 2),
        ("braid", 2),
        ("reversible", 2),
    ]


def test_scorecard_sparse_cells_reject_cartesian_options_before_writing(tmp_path):
    artifacts = tmp_path / "artifacts"
    result = runner.invoke(
        cli.app,
        [
            "scorecard",
            "--cell",
            "braid:copyops",
            "--mechanism",
            "braid",
            "--artifacts-dir",
            str(artifacts),
            "--dry-run",
        ],
    )
    assert result.exit_code == 2
    assert "--cell cannot be combined" in result.output
    assert not artifacts.exists()


def test_scorecard_rejects_reversible_campaign_with_mismatched_kv_geometry(tmp_path):
    artifacts = tmp_path / "artifacts"
    result = runner.invoke(
        cli.app,
        [
            "scorecard",
            "--cell",
            "reversible:copyops",
            "--n-head",
            "4",
            "--n-kv-head",
            "4",
            "--artifacts-dir",
            str(artifacts),
            "--dry-run",
        ],
    )
    assert result.exit_code == 2
    assert "campaign-global --n-kv-head" in result.output
    assert not artifacts.exists()


def test_scorecard_rejects_unknown_hypothesis_before_writing(tmp_path):
    artifacts = tmp_path / "artifacts"
    result = runner.invoke(
        cli.app,
        [
            "scorecard",
            "--hypothesis",
            "hyp-does-not-exist",
            "--artifacts-dir",
            str(artifacts),
            "--dry-run",
        ],
    )
    assert result.exit_code == 2
    assert "unknown hypotheses" in result.output
    assert not artifacts.exists()


def test_scorecard_rejects_hypothesis_matrix_gap_before_writing(tmp_path):
    artifacts = tmp_path / "artifacts"
    result = runner.invoke(
        cli.app,
        [
            "scorecard",
            "--cell",
            "hyperbolic:placebo",
            "-H",
            "hyp-braid-length-generalization",
            "--artifacts-dir",
            str(artifacts),
            "--dry-run",
        ],
    )
    assert result.exit_code == 2
    assert "hypothesis/matrix coverage failed" in result.output
    assert "braid:copyops" in result.output
    assert not artifacts.exists()


def test_scorecard_rejects_dirty_claim_bearing_run_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli,
        "_get_git_info",
        lambda: {"commit": "abc1234", "commit_full": "a" * 40, "branch": "main", "dirty": True},
    )
    artifacts = tmp_path / "artifacts"
    result = runner.invoke(
        cli.app,
        [
            "scorecard",
            "--cell",
            "hyperbolic:placebo",
            "-H",
            "hyp-hyperbolic-placebo-specificity",
            "--artifacts-dir",
            str(artifacts),
        ],
    )
    assert result.exit_code == 2
    assert "clean, known Git commit" in result.output
    assert not artifacts.exists()


def test_scorecard_off_floor_gate_is_training_seed_strict_and_budget_local(tmp_path):
    cells: list[dict] = []
    for budget_index, budget_flops, score in ((0, 1e6, 0.5), (1, 2e6, 0.8)):
        for seed in range(3):
            cell = {
                "id": f"b{budget_index}-copyops-standard-s{seed}",
                "budget_index": budget_index,
                "budget_flops": budget_flops,
                "mechanism": "standard",
                "task": "copyops",
                "seed": seed,
                "step_qualified": True,
                "status": "done",
            }
            cells.append(cell)
            eval_dir = cli._scorecard_eval_dir(tmp_path, cell)
            eval_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "tasks": {
                    "copyops": {
                        "exact_match": {"greedy": {"held_out": {"mean": score}}},
                        "answer_prior": {"held_out": {"mean": 0.5}},
                    }
                }
            }
            (eval_dir / "summary.json").write_text(json.dumps(payload))

    floors = cli._scorecard_off_floor_tasks(cells, tmp_path)
    assert floors["1000000.0"]["copyops"]["qualified"] is False
    assert "did not clear" in floors["1000000.0"]["copyops"]["reason"]
    assert floors["2000000.0"]["copyops"]["qualified"] is True
    assert abs(floors["2000000.0"]["copyops"]["lower_ci95"] - 0.8) < 1e-12


def test_scorecard_resume_executes_exactly_the_unfinished_cells(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_scorecard_generate_task", _fake_scorecard_generator)
    monkeypatch.setattr(cli, "_scorecard_flops_per_step", _fake_scorecard_flops)
    state = {"train_calls": 0, "interrupt_once": True}

    def launch(cmd: list[str], *, timeout_s: float) -> tuple[int, str, str]:
        if "nanochat.train" in cmd:
            if state["train_calls"] == 2 and state["interrupt_once"]:
                state["interrupt_once"] = False
                raise KeyboardInterrupt
            state["train_calls"] += 1
        return _fake_scorecard_success(cmd, timeout_s=timeout_s)

    monkeypatch.setattr(cli, "_scorecard_launch", launch)
    artifacts = tmp_path / "artifacts"
    args = [
        "scorecard",
        "-m",
        "tropical",
        "-t",
        "placebo",
        "--seeds",
        "2",
        "--budget",
        "1e6",
        "--dataset-size",
        "6",
        "--examples",
        "1",
        "--artifacts-dir",
        str(artifacts),
        "--run-id",
        "resume-contract",
    ]
    first = runner.invoke(cli.app, args)
    assert first.exit_code == 130, first.output
    manifest_path = artifacts / "scorecards" / "resume-contract" / "manifest.json"
    first_manifest = json.loads(manifest_path.read_text())
    assert sum(cell["status"] == "done" for cell in first_manifest["cells"]) == 2
    assert sum(cell["status"] == "interrupted" for cell in first_manifest["cells"]) == 1
    created_command = first_manifest["created_command"]

    before_resume = state["train_calls"]
    second = runner.invoke(cli.app, args)
    assert second.exit_code == 0, second.output
    assert state["train_calls"] - before_resume == 2
    final_manifest = json.loads(manifest_path.read_text())
    assert all(cell["status"] == "done" for cell in final_manifest["cells"])
    assert [cell["attempts"] for cell in final_manifest["cells"]] == [1, 1, 2, 1]
    assert final_manifest["created_command"] == created_command
    assert len(final_manifest["resume_history"]) == 1
    suite = manifest_path.parent
    for required in ("summary.json", "report.md", "report.html"):
        assert (suite / required).exists()
    summary = json.loads((suite / "summary.json").read_text())
    assert summary["adjudications"]["placebo"]["publication_blocked"] is True
    assert "universal placebo guard" in " ".join(summary["adjudications"]["placebo"]["blockers"])

    before_fresh = state["train_calls"]
    fresh = runner.invoke(cli.app, [*args, "--fresh"])
    assert fresh.exit_code == 2
    assert "choose a new --run-id" in fresh.output
    assert state["train_calls"] == before_fresh


def test_scorecard_claim_resume_uses_frozen_snapshot_live_hold_and_original_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_scorecard_generate_task", _fake_scorecard_generator)
    monkeypatch.setattr(cli, "_scorecard_flops_per_step", _fake_scorecard_flops)
    clean_a = {"commit": "aaaaaaa", "commit_full": "a" * 40, "branch": "main", "dirty": False}
    monkeypatch.setattr(cli, "_get_git_info", lambda: clean_a)
    state = {"train_calls": 0, "interrupt_once": True}

    def launch(cmd: list[str], *, timeout_s: float) -> tuple[int, str, str]:
        if "nanochat.train" in cmd:
            if state["train_calls"] == 1 and state["interrupt_once"]:
                state["interrupt_once"] = False
                raise KeyboardInterrupt
            state["train_calls"] += 1
        return _fake_scorecard_success(cmd, timeout_s=timeout_s)

    monkeypatch.setattr(cli, "_scorecard_launch", launch)
    artifacts = tmp_path / "artifacts"
    args = [
        "scorecard",
        "--cell",
        "hyperbolic:placebo",
        "-H",
        "hyp-hyperbolic-placebo-specificity",
        "--seeds",
        "2",
        "--budget",
        "1e6",
        "--dataset-size",
        "6",
        "--examples",
        "1",
        "--artifacts-dir",
        str(artifacts),
        "--run-id",
        "claim-resume-contract",
    ]
    monkeypatch.setattr(cli.sys, "argv", ["mgr", *args])
    first = runner.invoke(cli.app, args)
    assert first.exit_code == 130, first.output
    manifest_path = artifacts / "scorecards" / "claim-resume-contract" / "manifest.json"
    first_manifest = json.loads(manifest_path.read_text())
    created_command = first_manifest["created_command"]

    def reject_live_snapshot(_hypothesis_ids):
        raise AssertionError("resume must use the manifest snapshot")

    monkeypatch.setattr(cli, "_scorecard_hypothesis_snapshot", reject_live_snapshot)
    monkeypatch.setattr(
        cli,
        "_scorecard_live_manual_holds",
        lambda _ids: (
            {
                "hyp-hyperbolic-placebo-specificity": {
                    "held": True,
                    "reason": "operator review",
                    "date": "2026-08-25",
                }
            },
            None,
        ),
    )
    resume_args = [*args, "--timeout-s", "7200"]
    monkeypatch.setattr(cli.sys, "argv", ["mgr", *resume_args])
    second = runner.invoke(cli.app, resume_args)
    assert second.exit_code == 0, second.output
    final_manifest = json.loads(manifest_path.read_text())
    assert final_manifest["created_command"] == created_command
    assert "7200" not in final_manifest["created_command"]
    assert "7200" in final_manifest["resume_history"][-1]["command"]
    summary = json.loads((manifest_path.parent / "summary.json").read_text())
    assert summary["adjudications"]["verdicts"][0]["reason_code"] == "manual_hold"

    clean_b = {"commit": "bbbbbbb", "commit_full": "b" * 40, "branch": "main", "dirty": False}
    monkeypatch.setattr(cli, "_get_git_info", lambda: clean_b)
    monkeypatch.setattr(cli.sys, "argv", ["mgr", *args])
    mixed_sha = runner.invoke(cli.app, args)
    assert mixed_sha.exit_code == 2
    assert "original clean producer commit" in mixed_sha.output


def test_scorecard_rejects_run_id_path_traversal(tmp_path):
    artifacts = tmp_path / "artifacts"
    result = runner.invoke(
        cli.app,
        [
            "scorecard",
            "--run-id",
            "../escape",
            "--artifacts-dir",
            str(artifacts),
            "--dry-run",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value: --run-id" in result.output
    assert not artifacts.exists()
    assert not (tmp_path / "escape").exists()


def test_scorecard_failure_is_recorded_without_stopping_other_cells(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_scorecard_generate_task", _fake_scorecard_generator)
    monkeypatch.setattr(cli, "_scorecard_flops_per_step", _fake_scorecard_flops)
    launched: list[str] = []

    def launch(cmd: list[str], *, timeout_s: float) -> tuple[int, str, str]:
        if "nanochat.train" in cmd:
            mechanism = cmd[cmd.index("--attention-type") + 1]
            launched.append(mechanism)
            if mechanism == "tropical":
                return 7, "", "deliberate failure"
        return _fake_scorecard_success(cmd, timeout_s=timeout_s)

    monkeypatch.setattr(cli, "_scorecard_launch", launch)
    artifacts = tmp_path / "artifacts"
    result = runner.invoke(
        cli.app,
        [
            "scorecard",
            "-m",
            "tropical",
            "-t",
            "placebo",
            "--seeds",
            "1",
            "--budget",
            "1e6",
            "--dataset-size",
            "6",
            "--examples",
            "1",
            "--artifacts-dir",
            str(artifacts),
            "--run-id",
            "failure-contract",
        ],
    )
    assert result.exit_code == 1, result.output
    assert launched == ["standard", "tropical"]
    suite = artifacts / "scorecards" / "failure-contract"
    manifest = json.loads((suite / "manifest.json").read_text())
    status = {cell["mechanism"]: cell["status"] for cell in manifest["cells"]}
    assert status == {"standard": "done", "tropical": "failed"}
    tropical = next(cell for cell in manifest["cells"] if cell["mechanism"] == "tropical")
    assert tropical["returncode"] == 7 and tropical["stage"] == "train"
    assert "deliberate failure" in (suite / tropical["stderr_path"]).read_text()
    assert (suite / "summary.json").exists() and (suite / "report.html").exists()


def _write_scorecard_placebo_evidence(root: Path, *, budget: float, mechanism: str, seed: int, value: float) -> None:
    run_id = f"b{int(budget)}-{mechanism}-s{seed}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "mgr.evaltasks.v3",
        "kind": "eval-tasks",
        "meta": {
            "run_id": run_id,
            "checkpoint": {
                "dir": f"checkpoints/{run_id}",
                "step": 0,
                "attention_type": mechanism,
                "budget": {"target_flops": budget},
                "lineage": {"run_id": run_id},
                "model_config": {"attention_type": mechanism},
            },
        },
        "provenance": {"tainted": False},
        "train_provenance": {"tainted": False},
        "tasks": {"placebo": {"perplexity": {"in_range": value, "held_out": value}}},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))


def test_scorecard_reports_scale_flips_and_two_sided_placebo_gate(tmp_path, monkeypatch):
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        """schema_version: 1
hypotheses:
  - id: hyp-placebo-no-winner
    statement: no mechanism wins on placebo
    mechanisms: [tropical]
    source: {kind: human, provenance: test}
    date_registered: '2026-08-24'
    prediction:
      metric_path: evaltasks:tasks.placebo.perplexity.in_range
      comparator: <=
      threshold_kind: ratio
      threshold: 1.02
      baseline: {mechanism: standard, equal_flops: true}
      min_seeds: 3
    status: open
    evidence: []
    verdict_history: []
"""
    )
    monkeypatch.setattr(cli, "_hypotheses_registry_path", lambda: registry)
    evidence = tmp_path / "scorecard"
    for seed in range(3):
        _write_scorecard_placebo_evidence(evidence, budget=1e6, mechanism="standard", seed=seed, value=1.0)
        _write_scorecard_placebo_evidence(evidence, budget=1e6, mechanism="tropical", seed=seed, value=1.0)
        _write_scorecard_placebo_evidence(evidence, budget=2e6, mechanism="standard", seed=seed, value=1.0)
        _write_scorecard_placebo_evidence(evidence, budget=2e6, mechanism="tropical", seed=seed, value=1.1)

    hypotheses = cli._scorecard_hypothesis_snapshot(["hyp-placebo-no-winner"])
    degraded = cli._scorecard_adjudications(
        evidence,
        [1e6, 2e6],
        hypotheses=hypotheses,
        mechanisms=["standard", "tropical"],
    )
    assert degraded["by_budget"]["1000000.0"][0]["verdict"] == "supported"
    assert degraded["by_budget"]["2000000.0"][0]["verdict"] == "refuted"
    assert degraded["verdict_flips"] == [
        {
            "id": "hyp-placebo-no-winner",
            "by_budget": {"1000000.0": "supported", "2000000.0": "refuted"},
        }
    ]
    assert degraded["placebo"]["publication_blocked"] is True

    for seed in range(3):
        _write_scorecard_placebo_evidence(evidence, budget=2e6, mechanism="tropical", seed=seed, value=0.9)
    improved = cli._scorecard_adjudications(
        evidence,
        [1e6, 2e6],
        hypotheses=hypotheses,
        mechanisms=["standard", "tropical"],
    )
    assert improved["verdicts"][0]["verdict"] == "supported"
    assert improved["placebo"]["publication_blocked"] is True
    assert "significant improvement below 0.98x" in " ".join(improved["placebo"]["blockers"])


def test_scorecard_fdr_reports_only_bh_surviving_supported_rows():
    verdicts = [
        {"id": "hyp-a", "verdict": "supported", "p_value": 0.01},
        {"id": "hyp-b", "verdict": "supported", "p_value": 0.08},
        {"id": "hyp-c", "verdict": "refuted", "p_value": 0.5},
        {"id": "hyp-d", "verdict": "blocked"},
    ]
    fdr = cli._scorecard_fdr(verdicts)
    assert fdr == {
        "q_level": 0.1,
        "family_size": 3,
        "supported": 2,
        "supported_fdr_survivors": ["hyp-a"],
    }
    assert verdicts[0]["q_value"] == 0.03
    assert verdicts[1]["q_value"] == 0.12
    assert verdicts[2]["q_value"] == 0.5
    assert "q_value" not in verdicts[3]


def test_scorecard_placebo_gate_blocks_missing_blocked_and_inconclusive_specific_controls(tmp_path):
    universal = {
        "id": "hyp-placebo-no-winner",
        "mechanisms": ["tropical"],
        "prediction": {
            "metric_path": "evaltasks:tasks.placebo.perplexity.in_range",
            "comparator": "<=",
            "threshold_kind": "ratio",
            "threshold": 1.02,
            "baseline": {"mechanism": "standard", "equal_flops": True},
            "min_seeds": 3,
        },
    }
    hyperbolic = {
        "id": "hyp-hyperbolic-placebo-specificity",
        "mechanisms": ["hyperbolic"],
        "prediction": {
            "metric_path": "evaltasks:tasks.placebo.perplexity.in_range",
            "comparator": "<=",
            "threshold_kind": "ratio",
            "threshold": 1.02,
            "baseline": {"mechanism": "standard", "equal_flops": True},
            "min_seeds": 3,
        },
    }
    for seed in range(3):
        _write_scorecard_placebo_evidence(tmp_path, budget=1e6, mechanism="standard", seed=seed, value=1.0)
        _write_scorecard_placebo_evidence(tmp_path, budget=1e6, mechanism="tropical", seed=seed, value=1.0)

    blocked = cli._scorecard_adjudications(
        tmp_path,
        [1e6],
        hypotheses=[universal, hyperbolic],
        mechanisms=["standard", "tropical", "hyperbolic"],
    )
    assert next(row for row in blocked["placebo"]["rows"] if row["id"] == hyperbolic["id"])["verdict"] == "blocked"
    assert "registered placebo guard is BLOCKED" in " ".join(blocked["placebo"]["blockers"])

    uncovered = cli._scorecard_adjudications(
        tmp_path,
        [1e6],
        hypotheses=[universal],
        mechanisms=["standard", "tropical", "hyperbolic"],
    )
    assert "hyperbolic: no selected placebo hypothesis" in " ".join(uncovered["placebo"]["blockers"])

    for seed, value in enumerate((0.95, 1.02, 1.09)):
        _write_scorecard_placebo_evidence(tmp_path, budget=1e6, mechanism="hyperbolic", seed=seed, value=value)
    inconclusive = cli._scorecard_adjudications(
        tmp_path,
        [1e6],
        hypotheses=[universal, hyperbolic],
        mechanisms=["standard", "tropical", "hyperbolic"],
    )
    hyperbolic_row = next(
        row for row in inconclusive["placebo"]["rows"] if row["id"] == hyperbolic["id"]
    )
    assert hyperbolic_row["verdict"] == "inconclusive"
    assert "registered placebo guard is INCONCLUSIVE" in " ".join(inconclusive["placebo"]["blockers"])
