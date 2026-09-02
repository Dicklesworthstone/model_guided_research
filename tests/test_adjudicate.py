"""Tests for the verdict engine, mgr adjudicate (bead hij.2).

The acceptance matrix: fixture artifacts drive one hypothesis to EACH verdict
state (supported / refuted / inconclusive / blocked-for-every-reason);
verdicts are deterministic (fixed bootstrap seed -> byte-identical
verdicts.json); the ledger append survives validation, preserves hand-written
registry comments, never rewrites prior history, and rolls back atomically
when validation fails. The integrity core - REFUSING weak or tainted
evidence - is tested explicitly per the bead.
"""

import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

import cli

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parent.parent

CLEAN_PROV = {
    "schema_version": "mgr.metrics.v1",
    "git_sha": "deadbeef",
    "git_dirty": False,
    "config_hash": "abc",
    "data_snapshot_hash": None,
    "tainted": False,
}
TAINTED_PROV = {**CLEAN_PROV, "git_dirty": True, "tainted": True}


def _evaltasks_artifact(
    root: Path,
    name: str,
    *,
    mechanism: str,
    per_seed: list[float],
    task: str = "hier",
    target_flops: float = 2e9,
    tainted: bool = False,
    run_id: str | None = None,
    step: int = 100,
    generated_at: str = "2026-06-10T20:00:00Z",
    schema_version: str = "mgr.evaltasks.v2",
    answer_prior: float | None = None,
) -> Path:
    """One eval artifact == one trained checkpoint (ci-v2 observation unit).
    per_seed are EVAL seeds inside the artifact; the engine consumes only the
    mean. run_id defaults to the artifact name (distinct training runs)."""
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    mean = sum(per_seed) / len(per_seed)
    task_rec: dict = {
        "exact_match": {
            "greedy": {
                "held_out": {"mean": mean, "per_seed": per_seed},
                "in_range": {"mean": mean, "per_seed": per_seed},
            }
        },
        "perplexity": {"in_range": mean, "held_out": mean},
    }
    if answer_prior is not None:
        task_rec["answer_prior"] = {
            "held_out": {"mean": answer_prior, "per_seed": [answer_prior] * len(per_seed), "majority_answer": "x"},
            "in_range": {"mean": answer_prior, "per_seed": [answer_prior] * len(per_seed), "majority_answer": "x"},
        }
    summary = {
        "schema_version": schema_version,
        "kind": "eval-tasks",
        "meta": {
            "generated_at": generated_at,
            "checkpoint": {
                "attention_type": mechanism,
                "step": step,
                "budget": {"max_steps": 100, "target_flops": target_flops, "flops_per_step_est": 1e7},
                "lineage": {"run_id": run_id or name, "parent_run_ids": []},
            },
            "seeds": list(range(len(per_seed))),
        },
        "provenance": TAINTED_PROV if tainted else CLEAN_PROV,
        "tasks": {task: task_rec},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    return run_dir / "summary.json"


def _arm_artifacts(root: Path, prefix: str, mechanism: str, means: list[float], **kw) -> None:
    """One artifact per training-run mean (the ci-v2 observation unit)."""
    for i, m in enumerate(means):
        _evaltasks_artifact(root, f"{prefix}{i}", mechanism=mechanism, per_seed=[m, m, m], **kw)


def _hyp(per_pred_overrides=None, **overrides):
    pred = {
        "metric_path": "evaltasks:tasks.hier.exact_match.greedy.held_out.mean",
        "comparator": ">=",
        "threshold_kind": "absolute_delta",
        "threshold": 0.05,
        "baseline": {"mechanism": "standard", "equal_flops": True},
        "min_seeds": 3,
    }
    pred.update(per_pred_overrides or {})
    base = {
        "id": "hyp-engine-test",
        "statement": "engine test claim",
        "mechanisms": ["ultrametric"],
        "source": {"kind": "human", "provenance": "test"},
        "date_registered": "2026-06-10",
        "prediction": pred,
        "status": "open",
        "evidence": [],
        "verdict_history": [],
    }
    base.update(overrides)
    return base


def _index(root: Path):
    return cli._adj_collect_artifacts([root])


# ---------------------------------------------------------------------------
# verdict states


def test_supported_when_ci_clears_threshold(tmp_path):
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.80, 0.82, 0.81])
    _arm_artifacts(tmp_path, "base", "standard", [0.50, 0.51, 0.52])
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    assert v["verdict"] == "supported", v
    arm = v["arms"]["ultrametric"]
    assert arm["ci95"][0] >= 0.05 and abs(arm["effect"] - 0.30) < 0.02
    assert arm["n_candidate"] == 3 and arm["n_baseline"] == 3  # one obs per trained model
    assert v["policy_version"] == "ci-v6"


# ---------------------------------------------------------------------------
# manual_hold: a human-frozen verdict the engine must refuse to re-stamp (srtf)


def test_manual_hold_blocks_a_would_be_supported_hypothesis(tmp_path):
    # Identical evidence to the supported case above: without the hold the
    # engine rules SUPPORTED; WITH it, the engine refuses (blocked: manual_hold)
    # so a mechanical -H/--all cannot clobber a hand-checked verdict by recency
    # (the z4xx symplectic val-CE incommensurability case the bead motivates).
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.80, 0.82, 0.81])
    _arm_artifacts(tmp_path, "base", "standard", [0.50, 0.51, 0.52])
    pool = _index(tmp_path)
    assert cli._adjudicate_hypothesis(_hyp(), pool)["verdict"] == "supported"
    held = cli._adjudicate_hypothesis(
        _hyp(manual_hold={"held": True, "reason": "val-CE incommensurable; hand-checked INCONCLUSIVE"}), pool
    )
    assert held["verdict"] == "blocked"
    assert held["reason_code"] == "manual_hold"
    assert "incommensurable" in held["hold_reason"]


def test_manual_hold_false_is_a_noop(tmp_path):
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.80, 0.82, 0.81])
    _arm_artifacts(tmp_path, "base", "standard", [0.50, 0.51, 0.52])
    # held=false is a released/cleared hold: the engine adjudicates normally.
    v = cli._adjudicate_hypothesis(_hyp(manual_hold={"held": False, "reason": "released"}), _index(tmp_path))
    assert v["verdict"] == "supported"


def test_refuted_when_ci_clears_opposite_side(tmp_path):
    # no floor info in fixture or prediction -> the gate stays out of the way
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.50, 0.51, 0.50])
    _arm_artifacts(tmp_path, "base", "standard", [0.50, 0.50, 0.51])
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    assert v["verdict"] == "refuted", v  # effect ~0, CI well below the +0.05 claim


def test_decisive_refutation_is_not_underpowered_civ6(tmp_path):
    """ci-v6 (bead 4b82), 8h0e in miniature: a large OPPOSITE-sign effect whose
    CI excludes the +0.05 threshold decisively, but with enough spread that the
    power to DETECT the registered +0.05 is < 50%. Under ci-v5 this was stamped
    UNDERPOWERED (the wrong lens - "could you have CONFIRMED +0.05?"); under
    ci-v6 a refutation's adequacy is its precision, recorded as
    refutation_margin, and the confirm-power qualifier no longer fires."""
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.30, 0.36, 0.42])
    _arm_artifacts(tmp_path, "base", "standard", [0.55, 0.60, 0.65])
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    assert v["verdict"] == "refuted", v
    arm = v["arms"]["ultrametric"]
    assert arm["power"] < 0.5  # low power to CONFIRM the registered +0.05...
    assert "underpowered" not in arm  # ...but a decisive refutation is NOT weak (ci-v6)
    assert "underpowered" not in v
    assert arm["refutation_margin"] > 1.5  # the CI sits well past the threshold
    assert v["policy_version"] == "ci-v6"


def test_inconclusive_when_ci_straddles(tmp_path):
    # high variance across training runs: CI spans the 0.05 threshold
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.40, 0.75, 0.55])
    _arm_artifacts(tmp_path, "base", "standard", [0.45, 0.55, 0.50])
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    assert v["verdict"] == "inconclusive", v


def test_welch_t_not_normal_critical_value(tmp_path):
    """ci-v2 fix #2: at n=3 the CI must use the t critical value, not 1.96 -
    the interval is strictly wider than the Welch-normal one."""
    import statistics as stats_mod

    cand_means, base_means = [0.50, 0.55, 0.60], [0.50, 0.51, 0.52]
    _arm_artifacts(tmp_path, "cand", "ultrametric", cand_means)
    _arm_artifacts(tmp_path, "base", "standard", base_means)
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    arm = v["arms"]["ultrametric"]
    se = (stats_mod.variance(cand_means) / 3 + stats_mod.variance(base_means) / 3) ** 0.5
    half = (arm["ci95"][1] - arm["ci95"][0]) / 2
    assert half > 1.96 * se * 1.05, f"t interval must be wider than normal: half={half}, z-half={1.96 * se}"


def test_ratio_threshold_kind_supported(tmp_path):
    hyp = _hyp(
        {
            "comparator": "<=",
            "threshold_kind": "ratio",
            "threshold": 0.7,
            "metric_path": "evaltasks:tasks.hier.perplexity.held_out",
        }
    )
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.30, 0.31, 0.29])
    _arm_artifacts(tmp_path, "base", "standard", [0.60, 0.61, 0.59])
    v = cli._adjudicate_hypothesis(hyp, _index(tmp_path))
    assert v["verdict"] == "supported", v
    assert abs(v["arms"]["ultrametric"]["effect"] - 0.5) < 0.05


# ---------------------------------------------------------------------------
# blocked reasons (the integrity core)


def test_blocked_reasons(tmp_path):
    # no artifacts at all
    v = cli._adjudicate_hypothesis(_hyp(), [])
    assert v["verdict"] == "blocked" and v["reason_code"] == "no_candidate_artifacts"

    # candidate present, baseline missing
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.8, 0.8, 0.8])
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    assert v["verdict"] == "blocked" and v["reason_code"] == "no_baseline_artifacts"

    # insufficient training runs (2 < min_seeds=3; eval seeds inside an
    # artifact cannot substitute - that is the ci-v1 pseudo-replication)
    root2 = tmp_path / "few"
    _arm_artifacts(root2, "cand", "ultrametric", [0.8, 0.8])
    _arm_artifacts(root2, "base", "standard", [0.5, 0.5])
    v = cli._adjudicate_hypothesis(_hyp(), _index(root2))
    assert v["verdict"] == "blocked" and v["reason_code"] == "insufficient_seeds"

    # budget mismatch beyond 5%: no cohort contains both arms
    root3 = tmp_path / "budget"
    _arm_artifacts(root3, "cand", "ultrametric", [0.8, 0.8, 0.8], target_flops=2e9)
    _arm_artifacts(root3, "base", "standard", [0.5, 0.5, 0.5], target_flops=4e9)
    v = cli._adjudicate_hypothesis(_hyp(), _index(root3))
    assert v["verdict"] == "blocked" and v["reason_code"] == "budget_mismatch"

    # metric missing at the registered path
    root4 = tmp_path / "metric"
    _arm_artifacts(root4, "cand", "ultrametric", [0.8, 0.8, 0.8], task="dyck")
    _arm_artifacts(root4, "base", "standard", [0.5, 0.5, 0.5], task="dyck")
    v = cli._adjudicate_hypothesis(_hyp(), _index(root4))  # hyp points at tasks.hier
    assert v["verdict"] == "blocked" and v["reason_code"] == "metric_missing"

    # not operationalized
    v = cli._adjudicate_hypothesis(_hyp() | {"prediction": None}, [])
    assert v["verdict"] == "blocked" and v["reason_code"] == "prediction_not_operationalized"


def test_tainted_evidence_refused_and_clean_twin_adjudicates(tmp_path):
    """The bead's explicit pair: tainted fixture -> BLOCKED with the correct
    reason; a clean twin of the same numbers adjudicates normally."""
    dirty = tmp_path / "dirty"
    _arm_artifacts(dirty, "cand", "ultrametric", [0.8, 0.82, 0.81], tainted=True)
    _arm_artifacts(dirty, "base", "standard", [0.5, 0.51, 0.52])
    v = cli._adjudicate_hypothesis(_hyp(), _index(dirty))
    assert v["verdict"] == "blocked" and v["reason_code"] == "tainted_evidence", v

    clean = tmp_path / "clean"
    _arm_artifacts(clean, "cand", "ultrametric", [0.8, 0.82, 0.81])
    _arm_artifacts(clean, "base", "standard", [0.5, 0.51, 0.52])
    v = cli._adjudicate_hypothesis(_hyp(), _index(clean))
    assert v["verdict"] == "supported"

    # missing provenance entirely == tainted (pre-rz8.2 artifacts)
    noprov = tmp_path / "noprov"
    _arm_artifacts(noprov, "cand", "ultrametric", [0.8, 0.82, 0.81])
    for sub in noprov.iterdir():
        p = sub / "summary.json"
        data = json.loads(p.read_text())
        del data["provenance"]
        p.write_text(json.dumps(data))
    _arm_artifacts(noprov, "base", "standard", [0.5, 0.51, 0.52])
    v = cli._adjudicate_hypothesis(_hyp(), _index(noprov))
    assert v["verdict"] == "blocked" and v["reason_code"] == "tainted_evidence"


def test_forall_multi_mechanism_worst_case(tmp_path):
    """A multi-mechanism entry is a FOR-ALL claim: worst arm decides."""
    _arm_artifacts(tmp_path, "c1-", "ultrametric", [0.80, 0.81, 0.82])
    _arm_artifacts(tmp_path, "c2-", "fractal", [0.50, 0.51, 0.50])
    _arm_artifacts(tmp_path, "base", "standard", [0.50, 0.51, 0.52])
    hyp = _hyp(mechanisms=["ultrametric", "fractal"])
    v = cli._adjudicate_hypothesis(hyp, _index(tmp_path))
    assert v["verdict"] == "refuted"  # fractal arm fails the +0.05 claim decisively
    assert v["arms"]["ultrametric"]["verdict"] == "supported"
    assert v["arms"]["fractal"]["verdict"] == "refuted"

    # any arm without evidence blocks the whole FOR-ALL claim
    hyp = _hyp(mechanisms=["ultrametric", "braid"])
    v = cli._adjudicate_hypothesis(hyp, _index(tmp_path))
    assert v["verdict"] == "blocked" and v["mechanism"] == "braid"


def test_train_schema_arm_detection(tmp_path):
    """ordinal/hoss arms resolve via hparams/config, not attention_type."""

    def train_artifact(name, *, scheduler="none", optimizer="adamw", val_ce=2.0):
        run = tmp_path / name
        run.mkdir(parents=True)
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "mgr.telemetry.v1",
                    "config": {"attention_type": "standard", "optimizer_type": optimizer},
                    "hparams": {"scheduler_type": scheduler},
                    "budget": {"max_steps": 100, "target_flops": 1e9, "flops_per_step_est": 1e7},
                    "provenance": CLEAN_PROV,
                    "results": {"val_ce_final": val_ce},
                }
            )
        )

    for i, ce in enumerate([1.90, 1.92, 1.91]):
        train_artifact(f"ord{i}", scheduler="ordinal", val_ce=ce)
    for i, ce in enumerate([2.00, 2.02, 2.01]):
        train_artifact(f"std{i}", val_ce=ce)
    hyp = _hyp(
        {"metric_path": "train:results.val_ce_final", "comparator": "<=", "threshold_kind": "ratio", "threshold": 0.98},
        mechanisms=["ordinal"],
    )
    v = cli._adjudicate_hypothesis(hyp, _index(tmp_path))
    assert v["verdict"] == "supported", v
    assert v["arms"]["ordinal"]["n_candidate"] == 3


# ---------------------------------------------------------------------------
# ci-v2: floor validity gate, budget cohorts, lineage dedupe


def test_floor_gate_recorded_prior_downgrades_refuted(tmp_path):
    """The pilot1/kbj2 regime: every arm at the best-constant-answer score.
    A null effect there is no-power, not evidence of absence."""
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.0208, 0.0208, 0.0208], answer_prior=0.0208)
    _arm_artifacts(tmp_path, "base", "standard", [0.0208, 0.0208, 0.0208], answer_prior=0.0208)
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    assert v["verdict"] == "inconclusive", v
    arm = v["arms"]["ultrametric"]
    assert arm["floor_effect"] is True and arm["floor_source"] == "recorded_answer_prior"
    assert arm["baseline_mean"] == pytest.approx(0.0208)
    assert v.get("floor_effect") is True


def test_floor_gate_registered_fallback_for_v1_artifacts(tmp_path):
    """v1 artifacts record no answer_prior; the registered validity floor
    must gate instead."""
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.5, 0.5, 0.5], schema_version="mgr.evaltasks.v1")
    _arm_artifacts(tmp_path, "base", "standard", [0.5, 0.5, 0.5], schema_version="mgr.evaltasks.v1")
    hyp = _hyp({"validity": {"baseline_floor": 0.521, "floor_source": "population prior (test)"}})
    v = cli._adjudicate_hypothesis(hyp, _index(tmp_path))
    assert v["verdict"] == "inconclusive", v
    assert v["arms"]["ultrametric"]["floor_source"] == "registered_baseline_floor"


def test_floor_gate_leaves_real_refutations_alone(tmp_path):
    """Baseline far above its floor: a tight null effect is a REAL refutation."""
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.40, 0.41, 0.40], answer_prior=0.05)
    _arm_artifacts(tmp_path, "base", "standard", [0.40, 0.40, 0.41], answer_prior=0.05)
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    assert v["verdict"] == "refuted", v
    arm = v["arms"]["ultrametric"]
    assert "floor_effect" not in arm and arm["baseline_mean"] > arm["baseline_floor"]


def test_budget_cohorts_largest_qualifying_wins(tmp_path):
    """Pilot evidence at 1e13 (at floor) + E1 evidence at 1e14 (off floor):
    the verdict must come from the E1 cohort, never a cross-budget mix."""
    _arm_artifacts(tmp_path, "pc", "ultrametric", [0.02, 0.02, 0.02], target_flops=1e13, answer_prior=0.02)
    _arm_artifacts(tmp_path, "pb", "standard", [0.02, 0.02, 0.02], target_flops=1e13, answer_prior=0.02)
    _arm_artifacts(tmp_path, "ec", "ultrametric", [0.80, 0.81, 0.82], target_flops=1e14, answer_prior=0.02)
    _arm_artifacts(tmp_path, "eb", "standard", [0.50, 0.51, 0.52], target_flops=1e14, answer_prior=0.02)
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    assert v["verdict"] == "supported", v
    arm = v["arms"]["ultrametric"]
    assert arm["budget_flops"] == pytest.approx(1e14)
    assert arm["n_candidate"] == 3 and arm["n_baseline"] == 3
    assert all("/ec" in p or "/eb" in p for p in v["artifacts"]), v["artifacts"]

    # larger cohort lacking min_seeds falls back to the next one down
    root2 = tmp_path / "partial"
    _arm_artifacts(root2, "pc", "ultrametric", [0.50, 0.51, 0.50], target_flops=1e13)
    _arm_artifacts(root2, "pb", "standard", [0.50, 0.50, 0.51], target_flops=1e13)
    _arm_artifacts(root2, "ec", "ultrametric", [0.9], target_flops=1e14)  # 1 run only
    _arm_artifacts(root2, "eb", "standard", [0.5], target_flops=1e14)
    v = cli._adjudicate_hypothesis(_hyp(), _index(root2))
    assert v["verdict"] == "refuted", v
    assert v["arms"]["ultrametric"]["budget_flops"] == pytest.approx(1e13)


def test_lineage_dedupe_re_evals_never_double_count(tmp_path):
    """Re-evaluating the same checkpoints (e.g. after an eval upgrade) must
    not inflate n; the newest/richest artifact per training run wins."""
    for s in range(3):
        _evaltasks_artifact(
            tmp_path,
            f"old-c{s}",
            mechanism="ultrametric",
            per_seed=[0.0208] * 3,
            run_id=f"u{s}",
            schema_version="mgr.evaltasks.v1",
            generated_at="2026-06-10T18:00:00Z",
        )
        _evaltasks_artifact(
            tmp_path,
            f"new-c{s}",
            mechanism="ultrametric",
            per_seed=[0.0208] * 3,
            run_id=f"u{s}",
            answer_prior=0.0208,
            generated_at="2026-06-10T23:00:00Z",
        )
        _evaltasks_artifact(
            tmp_path,
            f"old-b{s}",
            mechanism="standard",
            per_seed=[0.0208] * 3,
            run_id=f"s{s}",
            schema_version="mgr.evaltasks.v1",
            generated_at="2026-06-10T18:00:00Z",
        )
        _evaltasks_artifact(
            tmp_path,
            f"new-b{s}",
            mechanism="standard",
            per_seed=[0.0208] * 3,
            run_id=f"s{s}",
            answer_prior=0.0208,
            generated_at="2026-06-10T23:00:00Z",
        )
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    arm = v["arms"]["ultrametric"]
    assert arm["n_candidate"] == 3 and arm["n_baseline"] == 3, arm
    assert all("/new-" in p for p in v["artifacts"]), v["artifacts"]
    # and the v2 twins' recorded priors drive the gate
    assert v["verdict"] == "inconclusive" and arm["floor_effect"] is True


# ---------------------------------------------------------------------------
# determinism + CLI + ledger


def test_determinism_byte_identical_verdicts(tmp_path, monkeypatch):
    _arm_artifacts(tmp_path / "a", "cand", "ultrametric", [0.6, 0.7, 0.65])
    _arm_artifacts(tmp_path / "a", "base", "standard", [0.5, 0.55, 0.52])
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "schema_version: 1\nhypotheses:\n" + _yaml_entry(_hyp({"threshold_kind": "ratio", "threshold": 1.05}))
    )
    monkeypatch.setattr(cli, "_hypotheses_registry_path", lambda: registry)
    monkeypatch.setattr(cli, "_load_parent_hypothesis_registry", lambda repo_root: None)

    outs = []
    for run in ("r1", "r2"):
        result = runner.invoke(
            cli.app,
            [
                "adjudicate",
                "--all",
                "--dry-run",
                "--artifacts",
                str(tmp_path / "a"),
                "--artifacts-dir",
                str(tmp_path / run),
                "--run-id",
                "x",
            ],
        )
        assert result.exit_code == 0, result.output
        outs.append((tmp_path / run / "adjudications" / "x" / "verdicts.json").read_bytes())
    assert outs[0] == outs[1], "same artifacts must produce byte-identical verdicts (fixed bootstrap seed)"


def _yaml_entry(h):
    pred = h["prediction"]
    lines = [
        f"  - id: {h['id']}",
        f"    statement: {json.dumps(h['statement'])}",
        f"    mechanisms: [{', '.join(h['mechanisms'])}]",
        "    source: {kind: human, provenance: test}",
        f'    date_registered: "{h["date_registered"]}"',
    ]
    if pred is None:
        lines += ["    prediction: null", "    operationalization_note: test", "    status: blocked"]
    else:
        lines += [
            "    prediction:",
            f"      metric_path: {json.dumps(pred['metric_path'])}",
            f'      comparator: "{pred["comparator"]}"',
            f"      threshold_kind: {pred['threshold_kind']}",
            f"      threshold: {pred['threshold']}",
            f"      baseline: {{mechanism: {pred['baseline']['mechanism']}, equal_flops: true}}",
            f"      min_seeds: {pred['min_seeds']}",
            "    status: open",
        ]
    lines += ["    evidence: []", "    verdict_history: []", ""]
    return "\n".join(lines)


def test_ledger_append_preserves_comments_and_is_append_only(tmp_path, monkeypatch):
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "# HAND-WRITTEN HEADER COMMENT - must survive ledger surgery\n"
        "schema_version: 1\nhypotheses:\n" + _yaml_entry(_hyp())
    )
    monkeypatch.setattr(cli, "_hypotheses_registry_path", lambda: registry)
    monkeypatch.setattr(cli, "_load_parent_hypothesis_registry", lambda repo_root: None)
    _arm_artifacts(tmp_path / "a", "cand", "ultrametric", [0.8, 0.82, 0.81])
    _arm_artifacts(tmp_path / "a", "base", "standard", [0.5, 0.51, 0.52])

    result = runner.invoke(
        cli.app,
        [
            "adjudicate",
            "--all",
            "--artifacts",
            str(tmp_path / "a"),
            "--artifacts-dir",
            str(tmp_path / "out"),
            "--run-id",
            "x",
        ],
    )
    assert result.exit_code == 0, result.output
    text = registry.read_text()
    assert "HAND-WRITTEN HEADER COMMENT" in text, "ledger surgery must preserve comments"
    data, load_errors = cli._load_hypothesis_registry(registry)
    assert data is not None, load_errors
    entry = data["hypotheses"][0]
    assert entry["status"] == "supported"
    assert len(entry["verdict_history"]) == 1
    first = entry["verdict_history"][0]
    assert first["verdict"] == "supported" and first["adjudicator"] == "engine:ci-v6"
    assert first["policy_version"] == "ci-v6" and first["artifacts"]

    # second adjudication APPENDS; the first entry is untouched
    result = runner.invoke(
        cli.app,
        [
            "adjudicate",
            "--all",
            "--artifacts",
            str(tmp_path / "a"),
            "--artifacts-dir",
            str(tmp_path / "out2"),
            "--run-id",
            "x",
        ],
    )
    assert result.exit_code == 0, result.output
    data, load_errors = cli._load_hypothesis_registry(registry)
    assert data is not None, load_errors
    history = data["hypotheses"][0]["verdict_history"]
    assert len(history) == 2 and history[0] == first, "prior entries must never mutate"

    errors, _, _ = cli._validate_hypothesis_registry(data, [], REPO_ROOT, parent=None)
    assert errors == [], f"ledgered registry must stay valid: {errors}"


def test_dry_run_leaves_registry_untouched(tmp_path, monkeypatch):
    registry = tmp_path / "registry.yaml"
    registry.write_text("schema_version: 1\nhypotheses:\n" + _yaml_entry(_hyp()))
    before = registry.read_text()
    monkeypatch.setattr(cli, "_hypotheses_registry_path", lambda: registry)
    monkeypatch.setattr(cli, "_load_parent_hypothesis_registry", lambda repo_root: None)
    _arm_artifacts(tmp_path / "a", "cand", "ultrametric", [0.8, 0.82, 0.81])
    _arm_artifacts(tmp_path / "a", "base", "standard", [0.5, 0.51, 0.52])
    result = runner.invoke(
        cli.app,
        [
            "adjudicate",
            "--all",
            "--dry-run",
            "--artifacts",
            str(tmp_path / "a"),
            "--artifacts-dir",
            str(tmp_path / "out"),
            "--run-id",
            "x",
        ],
    )
    assert result.exit_code == 0, result.output
    assert registry.read_text() == before


def test_blocked_refusals_do_not_touch_the_ledger(tmp_path, monkeypatch):
    registry = tmp_path / "registry.yaml"
    registry.write_text("schema_version: 1\nhypotheses:\n" + _yaml_entry(_hyp()))
    before = registry.read_text()
    monkeypatch.setattr(cli, "_hypotheses_registry_path", lambda: registry)
    monkeypatch.setattr(cli, "_load_parent_hypothesis_registry", lambda repo_root: None)
    result = runner.invoke(
        cli.app,
        [
            "adjudicate",
            "--all",
            "--artifacts",
            str(tmp_path / "empty"),
            "--artifacts-dir",
            str(tmp_path / "out"),
            "--run-id",
            "x",
        ],
    )
    assert result.exit_code == 0, result.output
    assert registry.read_text() == before, "refusals are report-only"
    verdicts = json.loads((tmp_path / "out" / "adjudications" / "x" / "verdicts.json").read_text())
    assert verdicts["verdicts"][0]["verdict"] == "blocked"


def test_cli_argument_errors(tmp_path):
    result = runner.invoke(cli.app, ["adjudicate"])
    assert result.exit_code == 2
    result = runner.invoke(cli.app, ["adjudicate", "--hypothesis", "hyp-does-not-exist", "--dry-run"])
    assert result.exit_code == 2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))


def test_variant_selector_distinguishes_semiring_beta_arms(tmp_path):
    """rgyl: annealed vs fixed-beta vs exact-tropical runs share a mechanism;
    variant selectors split them into arms. A null variant value matches both
    recorded-null and knob-absent (pre-rgyl) artifacts."""

    def train_artifact(name, *, spec, val_ce, record_key=True):
        run = tmp_path / name
        run.mkdir(parents=True)
        hparams = {"scheduler_type": "none"}
        if record_key:
            hparams["semiring_beta_spec"] = spec
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "mgr.telemetry.v1",
                    "config": {"attention_type": "tropical", "optimizer_type": "adamw"},
                    "hparams": hparams,
                    "budget": {"max_steps": 100, "target_flops": 1e9, "flops_per_step_est": 1e7},
                    "provenance": CLEAN_PROV,
                    "results": {"val_ce_final": val_ce},
                }
            )
        )

    for i, ce in enumerate([1.90, 1.92, 1.91]):
        train_artifact(f"anneal{i}", spec="linear:1:32", val_ce=ce)
    for i, ce in enumerate([2.00, 2.02, 2.01]):
        train_artifact(f"fixed{i}", spec="1.0", val_ce=ce)
    # pre-rgyl artifact: the knob is ABSENT entirely (legacy summary)
    for i, ce in enumerate([2.50, 2.52, 2.51]):
        train_artifact(f"legacy{i}", spec=None, val_ce=ce, record_key=(i == 0))

    hyp = _hyp(
        {
            "metric_path": "train:results.val_ce_final",
            "comparator": "<=",
            "threshold_kind": "ratio",
            "threshold": 0.98,
            "baseline": {"mechanism": "tropical", "equal_flops": True, "variant": {"semiring_beta_spec": "1.0"}},
            "candidate_variant": {"semiring_beta_spec": "linear:1:32"},
        },
        mechanisms=["tropical"],
    )
    v = cli._adjudicate_hypothesis(hyp, _index(tmp_path))
    assert v["verdict"] == "supported", v
    arm = v["arms"]["tropical"]
    assert arm["n_candidate"] == 3 and arm["n_baseline"] == 3
    assert abs(arm["effect"] - 1.91 / 2.01) < 0.01
    assert all("anneal" in p or "fixed" in p for p in v["artifacts"]), v["artifacts"]

    # null variant = the exact-tropical arm: catches recorded-null AND absent
    hyp_null = _hyp(
        {
            "metric_path": "train:results.val_ce_final",
            "comparator": "<=",
            "threshold_kind": "ratio",
            "threshold": 0.98,
            "baseline": {"mechanism": "tropical", "equal_flops": True, "variant": {"semiring_beta_spec": None}},
            "candidate_variant": {"semiring_beta_spec": "linear:1:32"},
        },
        mechanisms=["tropical"],
    )
    v = cli._adjudicate_hypothesis(hyp_null, _index(tmp_path))
    assert v["verdict"] == "supported", v
    assert v["arms"]["tropical"]["n_baseline"] == 3  # all three legacy/null artifacts
    assert all("anneal" in p or "legacy" in p for p in v["artifacts"]), v["artifacts"]

    # no selector = pre-rgyl behavior: every tropical run pools into one arm
    hyp_plain = _hyp(
        {"metric_path": "train:results.val_ce_final", "comparator": "<=", "threshold_kind": "ratio", "threshold": 0.98},
        mechanisms=["tropical"],
    )
    hyp_plain["prediction"]["baseline"] = {"mechanism": "tropical", "equal_flops": True}
    v = cli._adjudicate_hypothesis(hyp_plain, _index(tmp_path))
    assert v["arms"]["tropical"]["n_candidate"] == 9


# ---------------------------------------------------------------------------
# ci-v3 (bead xas7): certify + chargeprobe schemas, single-arm predictions


def _certify_artifact(
    root: Path,
    name: str,
    *,
    seed: int = 0,
    measured: float = 1e-6,
    heuristic_measured: float = 1.2,
    dirty: bool = False,
) -> None:
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "certify",
                "run_id": name,
                "seed": seed,
                "device": "cpu",
                "dtype": "fp32",
                "git": {"commit": "deadbeef", "dirty": dirty},
                "mechanisms": ["braid"],
                "checks": [
                    {
                        "mechanism": "braid",
                        "check": "rmatrix_mass_partition_charge_conserved",
                        "family": "classical",
                        "status": "pass",
                        "measured": measured,
                        "tolerance": 1e-5,
                        "comparator": "le",
                        "duration_ms": 1.0,
                        "detail": "",
                    },
                    {
                        "mechanism": "braid",
                        "check": "heuristic_mass_partition_violated",
                        "family": "classical",
                        "status": "pass",
                        "measured": heuristic_measured,
                        "tolerance": 1e-3,
                        "comparator": "ge",
                        "duration_ms": 1.0,
                        "detail": "",
                    },
                ],
                "counts": {"pass": 2, "fail": 0, "error": 0},
            }
        )
    )


def _chargeprobe_artifact(
    root: Path,
    name: str,
    *,
    ratio: float,
    seed: int = 0,
    run_id: str | None = None,
    target_flops: float = 1e14,
    crossing_law: str = "rmatrix",
) -> None:
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "mgr.chargeprobe.v1",
                "kind": "probe-charges",
                "meta": {
                    "run_id": name,
                    "generated_at": "2026-06-11T20:00:00Z",
                    "checkpoint": {
                        "dir": f"ck/{name}",
                        "step": 100,
                        "attention_type": "braid",
                        "braid_crossing_law": crossing_law,
                        "n_params": 1000,
                        "budget": {"max_steps": 100, "target_flops": target_flops, "flops_per_step_est": 1e12},
                        "lineage": {"run_id": run_id or name, "parent_run_ids": []},
                    },
                    "task": "group",
                    "seed": seed,
                },
                "provenance": CLEAN_PROV,
                "categories": {
                    "z60": {"chance": 0.017, "floor_used": 0.1, "linear": {"test_acc": 0.2}, "mlp": {"test_acc": 0.25}}
                },
                "dissociation": {"abelian_over_chance": ratio, "nonsolvable_over_chance": 1.0, "ratio": ratio},
            }
        )
    )


def test_single_arm_certify_supported_budget_exempt(tmp_path):
    """3 distinct-seed certify runs adjudicate a single-arm threshold claim;
    certify artifacts carry no training budget (cohort-exempt by design)."""
    for s in range(3):
        _certify_artifact(tmp_path, f"cert{s}", seed=s, measured=1e-6 + s * 1e-8)
    hyp = _hyp(
        {
            "metric_path": "certify:braid.rmatrix_mass_partition_charge_conserved.measured",
            "comparator": "<=",
            "threshold_kind": "absolute_delta",
            "threshold": 1e-5,
            "baseline": None,
        },
        mechanisms=["braid"],
    )
    v = cli._adjudicate_hypothesis(hyp, _index(tmp_path))
    assert v["verdict"] == "supported", v
    arm = v["arms"]["braid"]
    assert arm["single_arm"] is True
    assert arm["budget_flops"] is None
    assert arm["n_candidate"] == 3 and arm["n_baseline"] == 0
    assert v["policy_version"] == "ci-v6"


def test_single_arm_certify_refuted_and_seed_dedupe(tmp_path):
    """Same-seed certify re-runs are byte-replays (deduped, never extra
    evidence); a measured value above threshold refutes."""
    for s in range(3):
        _certify_artifact(tmp_path, f"bad{s}", seed=s, measured=2e-3)
    hyp = _hyp(
        {
            "metric_path": "certify:braid.rmatrix_mass_partition_charge_conserved.measured",
            "comparator": "<=",
            "threshold_kind": "absolute_delta",
            "threshold": 1e-5,
            "baseline": None,
        },
        mechanisms=["braid"],
    )
    v = cli._adjudicate_hypothesis(hyp, _index(tmp_path))
    assert v["verdict"] == "refuted", v

    # dedupe: three artifacts sharing one seed collapse to a single observation
    dup_root = tmp_path / "dup"
    for i in range(3):
        _certify_artifact(dup_root, f"dup{i}", seed=7, measured=1e-6)
    v = cli._adjudicate_hypothesis(hyp | {"id": "hyp-dup"}, _index(dup_root))
    assert v["verdict"] == "blocked" and v["reason_code"] == "insufficient_seeds", v


def test_certify_taint_derives_from_git_dirty(tmp_path):
    """certify predates the provenance block: a dirty tree taints the run."""
    for s in range(3):
        _certify_artifact(tmp_path, f"dirty{s}", seed=s, measured=1e-6, dirty=True)
    hyp = _hyp(
        {
            "metric_path": "certify:braid.rmatrix_mass_partition_charge_conserved.measured",
            "comparator": "<=",
            "threshold_kind": "absolute_delta",
            "threshold": 1e-5,
            "baseline": None,
        },
        mechanisms=["braid"],
    )
    v = cli._adjudicate_hypothesis(hyp, _index(tmp_path))
    assert v["verdict"] == "blocked" and v["reason_code"] == "tainted_evidence", v


def test_single_arm_chargeprobe_dissociation_with_variant_and_cohorts(tmp_path):
    """chargeprobe artifacts adjudicate via meta.checkpoint (arm matching,
    variant selector, budget cohorts) like evaltasks; the dissociation ratio
    is the registered observable."""
    for s in range(3):
        _chargeprobe_artifact(tmp_path, f"probe{s}", ratio=4.0 + 0.1 * s, seed=s)
    # a non-rmatrix probe and a small-budget probe must both be excluded
    _chargeprobe_artifact(tmp_path, "probe-soft", ratio=50.0, seed=9, crossing_law="restricted")
    _chargeprobe_artifact(tmp_path, "probe-small", ratio=50.0, seed=10, target_flops=1e9)
    hyp = _hyp(
        {
            "metric_path": "chargeprobe:dissociation.ratio",
            "comparator": ">=",
            "threshold_kind": "absolute_delta",
            "threshold": 2.0,
            "baseline": None,
            "candidate_variant": {"braid_crossing_law": "rmatrix"},
        },
        mechanisms=["braid"],
    )
    v = cli._adjudicate_hypothesis(hyp, _index(tmp_path))
    assert v["verdict"] == "supported", v
    arm = v["arms"]["braid"]
    assert arm["n_candidate"] == 3  # soft-law and small-budget probes excluded
    assert arm["budget_flops"] == 1e14
    assert abs(arm["effect"] - 4.1) < 1e-9


def test_single_arm_train_trend_route_coverage_delta(tmp_path):
    """The y4r8 within-run trend: route_coverage_delta > 0 adjudicates as a
    single-arm train-schema prediction."""
    for i, delta in enumerate([0.21, 0.18, 0.25]):
        run = tmp_path / f"trend{i}"
        run.mkdir(parents=True)
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "mgr.telemetry.v1",
                    "config": {"attention_type": "tropical", "optimizer_type": "adamw"},
                    "hparams": {"scheduler_type": "none"},
                    "budget": {"max_steps": 100, "target_flops": 1e9, "flops_per_step_est": 1e7},
                    "provenance": CLEAN_PROV,
                    "results": {
                        "route_coverage_first": 0.0,
                        "route_coverage_final": delta,
                        "route_coverage_delta": delta,
                    },
                }
            )
        )
    hyp = _hyp(
        {
            "metric_path": "train:results.route_coverage_delta",
            "comparator": ">=",
            "threshold_kind": "absolute_delta",
            "threshold": 0.05,
            "baseline": None,
        },
        mechanisms=["tropical"],
    )
    v = cli._adjudicate_hypothesis(hyp, _index(tmp_path))
    assert v["verdict"] == "supported", v
    assert v["arms"]["tropical"]["single_arm"] is True


def test_two_arm_verdicts_unchanged_under_ci_v3(tmp_path):
    """ci-v3/ci-v4 are append-only policy: a classic two-arm adjudication
    produces the same verdict and CI as under ci-v2 (the stamp changes, and
    ci-v4 ADDS power/p_value instrumentation without touching the decision)."""
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.50, 0.52, 0.54], answer_prior=0.01)
    _arm_artifacts(tmp_path, "base", "standard", [0.30, 0.31, 0.32], answer_prior=0.01)
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    assert v["verdict"] == "supported"
    arm = v["arms"]["ultrametric"]
    assert "single_arm" not in arm
    assert arm["n_candidate"] == 3 and arm["n_baseline"] == 3
    assert v["policy_version"] == "ci-v6"


def test_evaltasks_variant_selector_resolves_via_model_config(tmp_path):
    """Variant selectors on evaltasks evidence read the checkpoint's recorded
    model_config (fresh-eyes fix): a braid_crossing_law selector must pick the
    rmatrix arm out of mixed braid artifacts - and must NOT match artifacts
    that predate the model_config field (loud no_candidate_artifacts, never
    silent arm pooling)."""

    def braid_eval(name, *, law, slope, with_model_config=True):
        path = _evaltasks_artifact(tmp_path, name, mechanism="braid", per_seed=[0.5, 0.5, 0.5])
        data = json.loads(path.read_text())
        data["tasks"]["group"] = {
            "length_slope": {
                "held_out": {
                    "slope": slope,
                    "ci95": [slope - 0.001, slope + 0.001],
                    "intercept": 1.0,
                    "n_docs": 50,
                    "basis": "test",
                },
                "by_category": {
                    "s5": {
                        "slope": slope,
                        "ci95": [slope - 0.001, slope + 0.001],
                        "intercept": 1.0,
                        "n_docs": 20,
                        "basis": "test",
                    }
                },
            },
        }
        if with_model_config:
            data["meta"]["checkpoint"]["model_config"] = {"attention_type": "braid", "braid_crossing_law": law}
        path.write_text(json.dumps(data))

    for i in range(3):
        braid_eval(f"rmx{i}", law="rmatrix", slope=-0.002)
        braid_eval(f"soft{i}", law="restricted", slope=-0.02)
    _arm_artifacts(tmp_path, "std", "standard", [0.5, 0.5, 0.5])
    for p in sorted(tmp_path.glob("std*/summary.json")):
        data = json.loads(p.read_text())
        data["tasks"]["group"] = {
            "length_slope": {
                "held_out": {"slope": -0.01, "ci95": [-0.011, -0.009], "intercept": 1.0, "n_docs": 50, "basis": "test"},
                "by_category": {
                    "s5": {"slope": -0.01, "ci95": [-0.011, -0.009], "intercept": 1.0, "n_docs": 20, "basis": "test"}
                },
            },
        }
        p.write_text(json.dumps(data))

    # ratio-space semantics: both slopes negative, "2x flatter" is M/B <= 0.5
    # (the registered convention; ">=" would invert under the negative baseline)
    hyp = _hyp(
        {
            "metric_path": "evaltasks:tasks.group.length_slope.by_category.s5.slope",
            "comparator": "<=",
            "threshold_kind": "ratio",
            "threshold": 0.5,
            "candidate_variant": {"braid_crossing_law": "rmatrix"},
        },
        mechanisms=["braid"],
    )
    v = cli._adjudicate_hypothesis(hyp, _index(tmp_path))
    assert v["verdict"] == "supported", v
    arm = v["arms"]["braid"]
    assert arm["n_candidate"] == 3  # restricted-law artifacts excluded
    assert abs(arm["effect"] - 0.2) < 0.01  # -0.002 / -0.01: 5x flatter
    assert all("rmx" in p or "std" in p for p in v["artifacts"]), v["artifacts"]

    # the inverted form must NOT support a 5x-flatter candidate (the bug the
    # fresh-eyes review caught in the originally drafted registrations)
    hyp_bad = _hyp(
        {
            "metric_path": "evaltasks:tasks.group.length_slope.by_category.s5.slope",
            "comparator": ">=",
            "threshold_kind": "ratio",
            "threshold": 0.5,
            "candidate_variant": {"braid_crossing_law": "rmatrix"},
        },
        mechanisms=["braid"],
    )
    hyp_bad["id"] = "hyp-inverted"
    v_bad = cli._adjudicate_hypothesis(hyp_bad, _index(tmp_path))
    assert v_bad["verdict"] == "refuted", v_bad

    # artifacts WITHOUT model_config cannot satisfy the selector
    legacy_root = tmp_path / "legacy"
    for i in range(3):
        _evaltasks_artifact(legacy_root, f"old{i}", mechanism="braid", per_seed=[0.5])
    v2 = cli._adjudicate_hypothesis(hyp, _index(legacy_root))
    assert v2["verdict"] == "blocked" and v2["reason_code"] == "no_candidate_artifacts", v2


def test_slope_floor_gate_vacuous_in_both_directions(tmp_path):
    """qtdq (o85g audit): a length_slope verdict computed while BOTH arms sit
    at/below the recorded answer prior is fit over floor noise - the gate
    downgrades supported AND refuted to floor_effect inconclusive (the EM
    gate only protects refutations). Off-floor arms pass through untouched,
    and artifacts with no recorded EM/prior leave the gate conservative."""

    def slope_eval(root, name, *, mechanism, slope, em, prior):
        path = _evaltasks_artifact(root, name, mechanism=mechanism, per_seed=[em] * 3)
        data = json.loads(path.read_text())
        data["tasks"]["group"] = {
            "exact_match": {
                "greedy": {
                    "held_out": {"mean": em, "per_seed": [em] * 3},
                    "in_range": {"mean": em, "per_seed": [em] * 3},
                }
            },
            "answer_prior": {"held_out": {"mean": prior}, "in_range": {"mean": prior}},
            "length_slope": {
                "held_out": {
                    "slope": slope,
                    "ci95": [slope - 0.001, slope + 0.001],
                    "intercept": 1.0,
                    "n_docs": 50,
                    "basis": "test",
                },
                "by_category": {
                    "s5": {
                        "slope": slope,
                        "ci95": [slope - 0.001, slope + 0.001],
                        "intercept": 1.0,
                        "n_docs": 20,
                        "basis": "test",
                    }
                },
            },
        }
        path.write_text(json.dumps(data))

    # floored regime: the o85g shape - candidate slopes exactly 0, baseline
    # slopes tiny-negative, every EM far below the 0.097 prior
    for i in range(3):
        slope_eval(tmp_path, f"cand{i}", mechanism="braid", slope=0.0, em=0.01, prior=0.097)
        slope_eval(tmp_path, f"base{i}", mechanism="standard", slope=-0.002, em=0.008, prior=0.097)

    spec = {
        "metric_path": "evaltasks:tasks.group.length_slope.by_category.s5.slope",
        "comparator": "<=",
        "threshold_kind": "ratio",
        "threshold": 0.5,
    }
    v = cli._adjudicate_hypothesis(_hyp(spec, mechanisms=["braid"]), _index(tmp_path))
    arm = v["arms"]["braid"]
    assert v["verdict"] == "inconclusive", v
    assert arm["floor_effect"] is True and arm["floor_source"] == "slope_em_floor"
    assert abs(arm["em_floor"] - 0.097) < 1e-12 and arm["baseline_em_mean"] < 0.097

    # the refuted direction is equally vacuous and equally gated
    spec_rev = {**spec, "comparator": ">="}
    hyp_rev = _hyp(spec_rev, mechanisms=["braid"])
    hyp_rev["id"] = "hyp-rev"
    v_rev = cli._adjudicate_hypothesis(hyp_rev, _index(tmp_path))
    assert v_rev["verdict"] == "inconclusive", v_rev
    assert v_rev["arms"]["braid"]["floor_source"] == "slope_em_floor"

    # off-floor control: same slopes with both arms ABOVE the prior - the
    # mechanical verdict stands (zero candidate slope vs negative baseline
    # gives ratio -0 <= 0.5 -> supported, per the registered sign caveat)
    off = tmp_path / "off"
    for i in range(3):
        slope_eval(off, f"cand{i}", mechanism="braid", slope=0.0, em=0.5, prior=0.097)
        slope_eval(off, f"base{i}", mechanism="standard", slope=-0.002, em=0.4, prior=0.097)
    v_off = cli._adjudicate_hypothesis(_hyp(spec, mechanisms=["braid"]), _index(off))
    assert v_off["verdict"] == "supported", v_off
    assert not v_off["arms"]["braid"].get("floor_effect")


def test_training_taint_propagates_through_eval_artifacts(tmp_path):
    """dz9i (hqwi audit finding): a clean-time eval of a TAINTED-TRAINING
    checkpoint must be refused - the engine taints an evaltasks artifact when
    EITHER provenance block is tainted. Legacy artifacts (no train_provenance)
    keep eval-time-only semantics."""
    _arm_artifacts(tmp_path, "base", "standard", [0.5, 0.5, 0.5])
    _arm_artifacts(tmp_path, "cand", "ultrametric", [0.8, 0.8, 0.8])
    # poison the candidate artifacts: clean eval provenance, tainted training
    for sub in tmp_path.iterdir():
        if not sub.name.startswith("cand"):
            continue
        p = sub / "summary.json"
        data = json.loads(p.read_text())
        data["train_provenance"] = {**CLEAN_PROV, "git_dirty": True, "tainted": True}
        p.write_text(json.dumps(data))
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    assert v["verdict"] == "blocked" and v["reason_code"] == "tainted_evidence", v

    # clean train_provenance adjudicates normally
    for sub in tmp_path.iterdir():
        if not sub.name.startswith("cand"):
            continue
        p = sub / "summary.json"
        data = json.loads(p.read_text())
        data["train_provenance"] = dict(CLEAN_PROV)
        p.write_text(json.dumps(data))
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path))
    assert v["verdict"] == "supported", v


# ---------------------------------------------------------------------------
# tcuy: precision-curve artifacts (the graceful-vs-cliff evidence producer)


def _precision_eval_artifact(
    root: Path,
    rid: str,
    *,
    ppl: float,
    knob: str | None = None,
    value: float | None = None,
    K: int = 8,
    attention: str = "ultrametric",
    hard_digits: bool | None = None,
    n_layer: int | None = None,
) -> None:
    run = root / "evals" / "tasks" / rid
    run.mkdir(parents=True, exist_ok=True)
    mc: dict = {"attention_type": attention, "ultrametric_K": K}
    if hard_digits is not None:
        mc["ultrametric_hard_digits"] = hard_digits
    if n_layer is not None:
        mc["n_layer"] = n_layer
    if knob:
        mc[knob] = value
    (run / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "mgr.evaltasks.v3",
                "meta": {"checkpoint": {"model_config": mc}},
                "tasks": {"hier": {"perplexity": {"in_range": ppl, "held_out": ppl}}},
            }
        )
    )


def test_precision_curve_auc_math_and_engine_ingestion(tmp_path, monkeypatch):
    """Hand-computable AUC case: digit arm flat (quality 1.0 everywhere) ->
    AUC 1.0; float arm cliff (quality 0 at every truncated point) -> AUC of
    the anchored trapezoid; the produced artifact is engine-readable bench
    evidence with the auc_ratio observable."""
    _precision_eval_artifact(tmp_path, "dfull", ppl=2.0)
    _precision_eval_artifact(tmp_path, "ffull", ppl=2.0)
    # digit points: NO degradation (ppl stays 2.0) at k = 4 and 2 of K = 8
    _precision_eval_artifact(tmp_path, "d-k4", ppl=2.0, knob="ultrametric_digits_k", value=4)
    _precision_eval_artifact(tmp_path, "d-k2", ppl=2.0, knob="ultrametric_digits_k", value=2)
    # float points: total collapse (ppl -> huge) at 8 and 16 of 32 bits
    _precision_eval_artifact(tmp_path, "f-b8", ppl=2.0e9, knob="eval_weight_quant_bits", value=8)
    _precision_eval_artifact(tmp_path, "f-b16", ppl=2.0e9, knob="eval_weight_quant_bits", value=16)

    result = runner.invoke(
        cli.app,
        [
            "precision-curve",
            "--digit-run",
            "d-k4",
            "--digit-run",
            "d-k2",
            "--float-run",
            "f-b8",
            "--float-run",
            "f-b16",
            "--digit-full",
            "dfull",
            "--float-full",
            "ffull",
            "--task",
            "hier",
            "--seed",
            "0",
            "--artifacts-dir",
            str(tmp_path),
            "--run-id",
            "pc-test",
        ],
    )
    assert result.exit_code == 0, result.output
    art = json.loads((tmp_path / "bench" / "precision_curves" / "pc-test" / "summary.json").read_text())
    r = art["results"]
    assert abs(r["auc_digit"] - 1.0) < 1e-9  # flat at quality 1.0 from 0.25 to 1.0
    # float curve: ~0 at 0.25 and 0.5, then trapezoid up to (1.0, 1.0):
    # spans 0.25->0.5 (~0) and 0.5->1.0 (avg ~0.5) over total span 0.75
    assert abs(r["auc_float"] - (0.5 * 0.5) / 0.75) < 1e-6
    assert r["auc_ratio"] > 2.9

    # the artifact is engine-readable bench evidence for the ultrametric arm
    arts = cli._adj_collect_artifacts([tmp_path / "bench"])
    assert len(arts) == 1 and arts[0]["schema"] == "bench"
    assert cli._adj_artifact_matches_arm(arts[0], "ultrametric", None)
    assert cli._adj_observations(arts[0], "results.auc_ratio")


def test_fake_quantize_weights_monotone_and_bounds():
    """More bits -> strictly smaller reconstruction error; range validated."""
    import torch

    torch.manual_seed(0)
    lin = torch.nn.Linear(32, 32, bias=False)
    ref = lin.weight.detach().clone()
    errs = []
    for bits in (2, 4, 8, 12):
        lin.weight.data.copy_(ref)
        cli._fake_quantize_weights(lin, bits)
        errs.append(float((lin.weight - ref).abs().max()))
    assert errs == sorted(errs, reverse=True), errs  # error shrinks with bits
    assert errs[-1] < 1e-3  # 12-bit is near-identity at this scale
    try:
        cli._fake_quantize_weights(lin, 1)
        raise AssertionError("bits=1 must be rejected")
    except ValueError:
        pass


def test_precision_curve_common_window_kills_span_bias(tmp_path):
    """Arms swept to DIFFERENT depths must be compared on the common window
    (matched memory): without it, the deeper-swept arm's AUC is penalized by
    construction - the bias the fresh-eyes review caught pre-evidence."""
    _precision_eval_artifact(tmp_path, "dfull2", ppl=2.0)
    _precision_eval_artifact(tmp_path, "ffull2", ppl=2.0)
    # digit arm swept SHALLOW (only k=4 of 8 -> fraction 0.5), no degradation
    _precision_eval_artifact(tmp_path, "d2-k4", ppl=2.0, knob="ultrametric_digits_k", value=4)
    # float arm swept DEEP (2 bits -> fraction 0.0625) where it collapses,
    # but IDENTICAL quality (1.0) inside the common window [0.5, 1.0]
    _precision_eval_artifact(tmp_path, "f2-b16", ppl=2.0, knob="eval_weight_quant_bits", value=16)
    _precision_eval_artifact(tmp_path, "f2-b2", ppl=2.0e9, knob="eval_weight_quant_bits", value=2)

    result = runner.invoke(
        cli.app,
        [
            "precision-curve",
            "--digit-run",
            "d2-k4",
            "--float-run",
            "f2-b16",
            "--float-run",
            "f2-b2",
            "--digit-full",
            "dfull2",
            "--float-full",
            "ffull2",
            "--task",
            "hier",
            "--seed",
            "1",
            "--artifacts-dir",
            str(tmp_path),
            "--run-id",
            "pc-window",
        ],
    )
    assert result.exit_code == 0, result.output
    r = json.loads((tmp_path / "bench" / "precision_curves" / "pc-window" / "summary.json").read_text())["results"]
    assert abs(r["common_window_lo"] - 0.5) < 1e-9
    # inside [0.5, 1.0] both arms are flat at quality 1.0: the ratio must be
    # ~1 (no advantage), NOT inflated by the float arm's deep-sweep collapse
    assert abs(r["auc_digit"] - 1.0) < 1e-9
    assert abs(r["auc_float"] - 1.0) < 1e-6, r["auc_float"]
    assert abs(r["auc_ratio"] - 1.0) < 1e-6


def test_precision_curve_window_hi_isolates_deep_compression(tmp_path):
    """kgj1: a registered sub-1.0 window edge keeps the wide near-lossless
    region from diluting the cliff. Hand-computable: digit flat at 1.0 over
    [0.25, 0.5] -> AUC 1; float collapses at 0.0625 and recovers by 0.5, so
    over the window its interpolated AUC is 5/7 -> ratio exactly 7/5."""
    _precision_eval_artifact(tmp_path, "dfull3", ppl=2.0)
    _precision_eval_artifact(tmp_path, "ffull3", ppl=2.0)
    _precision_eval_artifact(tmp_path, "d3-k4", ppl=2.0, knob="ultrametric_digits_k", value=4)
    _precision_eval_artifact(tmp_path, "d3-k2", ppl=2.0, knob="ultrametric_digits_k", value=2)
    _precision_eval_artifact(tmp_path, "f3-b16", ppl=2.0, knob="eval_weight_quant_bits", value=16)
    _precision_eval_artifact(tmp_path, "f3-b2", ppl=2.0e9, knob="eval_weight_quant_bits", value=2)

    result = runner.invoke(
        cli.app,
        [
            "precision-curve",
            "--digit-run",
            "d3-k4",
            "--digit-run",
            "d3-k2",
            "--float-run",
            "f3-b16",
            "--float-run",
            "f3-b2",
            "--digit-full",
            "dfull3",
            "--float-full",
            "ffull3",
            "--task",
            "hier",
            "--seed",
            "2",
            "--window-hi",
            "0.5",
            "--artifacts-dir",
            str(tmp_path),
            "--run-id",
            "pc-deepwin",
        ],
    )
    assert result.exit_code == 0, result.output
    r = json.loads((tmp_path / "bench" / "precision_curves" / "pc-deepwin" / "summary.json").read_text())["results"]
    assert (r["common_window_lo"], r["common_window_hi"]) == (0.25, 0.5)
    assert abs(r["auc_digit"] - 1.0) < 1e-9
    assert abs(r["auc_float"] - 5.0 / 7.0) < 1e-6, r["auc_float"]
    # deep-window artifacts expose a DISJOINT observable key as
    # defense-in-depth (bench variant selectors landed later, bhjf): the
    # full-axis hypothesis and the deep-window one must not be able to
    # ingest each other's artifacts even without selectors
    assert "auc_ratio" not in r
    assert abs(r["auc_ratio_deepwindow"] - 7.0 / 5.0) < 1e-6
    arts = cli._adj_collect_artifacts([tmp_path / "bench"])
    assert cli._adj_observations(arts[0], "results.auc_ratio") is None
    assert cli._adj_observations(arts[0], "results.auc_ratio_deepwindow")

    # a window edge at or below the data-dependent lo refuses loudly
    result = runner.invoke(
        cli.app,
        [
            "precision-curve",
            "--digit-run",
            "d3-k4",
            "--digit-run",
            "d3-k2",
            "--float-run",
            "f3-b16",
            "--float-run",
            "f3-b2",
            "--digit-full",
            "dfull3",
            "--float-full",
            "ffull3",
            "--task",
            "hier",
            "--seed",
            "2",
            "--window-hi",
            "0.25",
            "--artifacts-dir",
            str(tmp_path),
            "--run-id",
            "pc-badwin",
        ],
    )
    assert result.exit_code != 0


def test_bench_arm_matching_honors_variant_selectors():
    """bhjf: bench-backed hypotheses sharing a metric path must be separable
    by variant selectors - knobs are looked up in results then meta. Before
    this, the bench branch ignored selectors entirely and disjointness relied
    on per-protocol observable keys (kgj1's auc_ratio_deepwindow workaround)."""

    def bench_art(task: str, window_hi: float | None = None) -> dict:
        results: dict = {"auc_ratio": 1.5}
        if window_hi is not None:
            results["common_window_hi"] = window_hi
        return {
            "path": f"mem://{task}-{window_hi}",
            "schema": "bench",
            "tainted": False,
            "data": {
                "schema_version": "mgr.bench.precision_curves.v1",
                "kind": "precision_curve",
                "mechanism": "ultrametric",
                "meta": {"task": task, "seed": 0},
                "results": results,
            },
        }

    coarse = bench_art("hier")
    deep = bench_art("hier", window_hi=0.25)
    other_task = bench_art("arith", window_hi=0.25)

    # no selector: all same-mechanism bench artifacts pool (the pre-bhjf hazard)
    assert cli._adj_artifact_matches_arm(coarse, "ultrametric", None)
    assert cli._adj_artifact_matches_arm(deep, "ultrametric", None)

    # a results-level knob separates protocol variants sharing the metric path
    assert cli._adj_artifact_matches_arm(deep, "ultrametric", {"common_window_hi": 0.25})
    assert not cli._adj_artifact_matches_arm(coarse, "ultrametric", {"common_window_hi": 0.25})
    # null selects the knob-absent (default-protocol) artifacts only
    assert cli._adj_artifact_matches_arm(coarse, "ultrametric", {"common_window_hi": None})
    assert not cli._adj_artifact_matches_arm(deep, "ultrametric", {"common_window_hi": None})
    # meta-level knobs (task, seed) select the same way
    assert cli._adj_artifact_matches_arm(other_task, "ultrametric", {"task": "arith"})
    assert not cli._adj_artifact_matches_arm(deep, "ultrametric", {"task": "arith"})
    # a selector never rescues a mechanism mismatch
    assert not cli._adj_artifact_matches_arm(deep, "tropical", {"common_window_hi": 0.25})


# ---------------------------------------------------------------------------
# 9qeq: depth-curve artifacts (hyp-padic-truncation-depth-independent)


def _depth_eval_set(root: Path, prefix: str, *, n_layer: int, ppl_full: float, ppl_k: float) -> None:
    """One depth's digit-arm eval set: full anchor + k = 4, 2 of K = 8."""
    _precision_eval_artifact(root, f"{prefix}-full", ppl=ppl_full, hard_digits=True, n_layer=n_layer)
    for k in (4, 2):
        _precision_eval_artifact(
            root, f"{prefix}-k{k}", ppl=ppl_k, knob="ultrametric_digits_k", value=k, hard_digits=True, n_layer=n_layer
        )


def test_depth_curve_flat_across_depths_and_engine_ingestion(tmp_path):
    """Depth-independent digit arm (flat quality 1.0 at both depths) ->
    absdev 0; the artifact is engine-readable bench evidence exposing the
    digit_depth_auc_absdev observable and NOT the precision-curve one."""
    _depth_eval_set(tmp_path, "l2", n_layer=2, ppl_full=2.0, ppl_k=2.0)
    _depth_eval_set(tmp_path, "l8", n_layer=8, ppl_full=2.0, ppl_k=2.0)

    result = runner.invoke(
        cli.app,
        [
            "depth-curve",
            "--shallow-run",
            "l2-k4",
            "--shallow-run",
            "l2-k2",
            "--shallow-full",
            "l2-full",
            "--deep-run",
            "l8-k4",
            "--deep-run",
            "l8-k2",
            "--deep-full",
            "l8-full",
            "--task",
            "hier",
            "--seed",
            "0",
            "--artifacts-dir",
            str(tmp_path),
            "--run-id",
            "dc-flat",
        ],
    )
    assert result.exit_code == 0, result.output
    r = json.loads((tmp_path / "bench" / "depth_curves" / "dc-flat" / "summary.json").read_text())["results"]
    assert (r["n_layer_shallow"], r["n_layer_deep"]) == (2, 8)
    assert abs(r["auc_digit_shallow"] - 1.0) < 1e-9
    assert abs(r["auc_digit_deep"] - 1.0) < 1e-9
    assert r["digit_depth_auc_absdev"] < 1e-9
    assert r["float_depth_auc_absdev"] is None  # diagnostic absent unless float runs supplied

    arts = cli._adj_collect_artifacts([tmp_path / "bench"])
    assert len(arts) == 1 and arts[0]["schema"] == "bench"
    assert cli._adj_artifact_matches_arm(arts[0], "ultrametric", None)
    assert cli._adj_observations(arts[0], "results.digit_depth_auc_absdev") == [0.0]
    # cross-schema hygiene: the graceful hypothesis's observable must NOT
    # resolve against depth-curve artifacts (and the engine skips them)
    assert cli._adj_observations(arts[0], "results.auc_ratio") is None


def test_depth_curve_detects_depth_dependence_with_float_diagnostic(tmp_path):
    """Hand-computable contrast: shallow flat (AUC 1.0); deep degrades to
    quality 0.5 at k = 4, 2 -> AUC 0.5/0.75 = 2/3 -> absdev 1/3. Float
    diagnostic: shallow flat, deep collapses at 8 bits -> absdev 1/6."""
    _depth_eval_set(tmp_path, "s", n_layer=2, ppl_full=2.0, ppl_k=2.0)
    _depth_eval_set(tmp_path, "d", n_layer=8, ppl_full=2.0, ppl_k=4.0)
    for prefix, nl, b8_ppl in (("fs", 2, 2.0), ("fd", 8, 2.0e9)):
        _precision_eval_artifact(tmp_path, f"{prefix}-full", ppl=2.0, attention="standard", n_layer=nl)
        _precision_eval_artifact(
            tmp_path,
            f"{prefix}-b16",
            ppl=2.0,
            knob="eval_weight_quant_bits",
            value=16,
            attention="standard",
            n_layer=nl,
        )
        _precision_eval_artifact(
            tmp_path,
            f"{prefix}-b8",
            ppl=b8_ppl,
            knob="eval_weight_quant_bits",
            value=8,
            attention="standard",
            n_layer=nl,
        )

    result = runner.invoke(
        cli.app,
        [
            "depth-curve",
            "--shallow-run",
            "s-k4",
            "--shallow-run",
            "s-k2",
            "--shallow-full",
            "s-full",
            "--deep-run",
            "d-k4",
            "--deep-run",
            "d-k2",
            "--deep-full",
            "d-full",
            "--float-shallow-run",
            "fs-b16",
            "--float-shallow-run",
            "fs-b8",
            "--float-shallow-full",
            "fs-full",
            "--float-deep-run",
            "fd-b16",
            "--float-deep-run",
            "fd-b8",
            "--float-deep-full",
            "fd-full",
            "--task",
            "hier",
            "--seed",
            "1",
            "--artifacts-dir",
            str(tmp_path),
            "--run-id",
            "dc-contrast",
        ],
    )
    assert result.exit_code == 0, result.output
    r = json.loads((tmp_path / "bench" / "depth_curves" / "dc-contrast" / "summary.json").read_text())["results"]
    # deep curve (0.25, 0.5), (0.5, 0.5), (1.0, 1.0) over window [0.25, 1.0]:
    # 0.25*0.5 + 0.5*0.75 = 0.5 of area over span 0.75 -> 2/3
    assert abs(r["auc_digit_deep"] - 2.0 / 3.0) < 1e-9
    assert abs(r["digit_depth_auc_absdev"] - 1.0 / 3.0) < 1e-9
    # float deep curve (0.25, ~0), (0.5, 1.0), (1.0, 1.0): area 0.125 + 0.5
    # over span 0.75 -> 5/6 -> absdev 1/6
    assert abs(r["float_depth_auc_absdev"] - 1.0 / 6.0) < 1e-6


def test_depth_curve_hygiene_rejects_mislabeled_groups(tmp_path):
    """Group hygiene: mixed depths in one group, soft-digit checkpoints in
    the digit arm, and shallow >= deep all refuse loudly - a mislabeled
    run-id must not silently corrupt the registered observable."""
    _depth_eval_set(tmp_path, "g2", n_layer=2, ppl_full=2.0, ppl_k=2.0)
    _depth_eval_set(tmp_path, "g8", n_layer=8, ppl_full=2.0, ppl_k=2.0)

    # mixed depths inside the shallow group
    result = runner.invoke(
        cli.app,
        [
            "depth-curve",
            "--shallow-run",
            "g2-k4",
            "--shallow-run",
            "g8-k2",
            "--shallow-full",
            "g2-full",
            "--deep-run",
            "g8-k4",
            "--deep-full",
            "g8-full",
            "--artifacts-dir",
            str(tmp_path),
            "--run-id",
            "dc-mixed",
        ],
    )
    assert result.exit_code != 0

    # soft-digit checkpoint in the digit arm (the wave-1 trap: a soft-digit
    # eval is a channel drop, not a valuation truncation)
    _precision_eval_artifact(
        tmp_path, "soft-k4", ppl=2.0, knob="ultrametric_digits_k", value=4, hard_digits=False, n_layer=2
    )
    result = runner.invoke(
        cli.app,
        [
            "depth-curve",
            "--shallow-run",
            "soft-k4",
            "--shallow-full",
            "g2-full",
            "--deep-run",
            "g8-k4",
            "--deep-full",
            "g8-full",
            "--artifacts-dir",
            str(tmp_path),
            "--run-id",
            "dc-soft",
        ],
    )
    assert result.exit_code != 0

    # shallow depth must be strictly less than deep
    result = runner.invoke(
        cli.app,
        [
            "depth-curve",
            "--shallow-run",
            "g8-k4",
            "--shallow-full",
            "g8-full",
            "--deep-run",
            "g2-k4",
            "--deep-full",
            "g2-full",
            "--artifacts-dir",
            str(tmp_path),
            "--run-id",
            "dc-inverted",
        ],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# ci-v4 (bead hij.4): power-gated verdicts + ledger-level FDR


def test_bh_qvalues_textbook():
    """BH step-up against the hand-computed canonical case: sorted p
    [0.005, 0.01, 0.03, 0.04], m=4 -> raw p*m/i [0.02, 0.02, 0.04, 0.04]
    (monotone enforcement leaves them unchanged here)."""
    q = cli._adj_bh_qvalues({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    assert abs(q["d"] - 0.02) < 1e-12
    assert abs(q["a"] - 0.02) < 1e-12
    assert abs(q["c"] - 0.04) < 1e-12
    assert abs(q["b"] - 0.04) < 1e-12
    # monotone enforcement: a later small raw q pulls earlier ones down
    q2 = cli._adj_bh_qvalues({"x": 0.02, "y": 0.021})
    # raw: x -> 0.04, y -> 0.021; min-from-top: x -> min(0.04, 0.021) = 0.021
    assert abs(q2["x"] - 0.021) < 1e-12 and abs(q2["y"] - 0.021) < 1e-12
    # singleton family: q == p
    assert cli._adj_bh_qvalues({"only": 0.03})["only"] == 0.03


def test_power_closed_form_two_sample():
    """Power validated against the closed-form normal expression written out
    independently here: power = Phi(D/SE - z975) with SE^2 = vc/nc + vb/nb."""
    from scipy import stats as scipy_stats

    out = cli._adj_power([1.0, 2.0, 3.0], [0.0, 1.0, 2.0], 2.0, "absolute_delta", False)
    assert out is not None
    se = math.sqrt(1.0 / 3 + 1.0 / 3)  # both sample variances are exactly 1
    expected = float(scipy_stats.norm.cdf(2.0 / se - 1.959963984540054))
    assert abs(out["power"] - expected) < 1e-12
    # n for 80%: ceil((vc+vb) * (z975+z80)^2 / D^2) = ceil(2 * 7.8489 / 4) = 4
    assert out["n_for_80pct"] == 4

    # threshold AT the no-effect point: effect size zero, power undefined
    assert cli._adj_power([1.0, 2.0], [1.0, 2.0], 0.0, "absolute_delta", False) is None
    # zero spread in both arms: a deterministic observable has full power
    out0 = cli._adj_power([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 0.5, "absolute_delta", False)
    assert out0 is not None
    assert out0["power"] == 1.0 and out0["n_for_80pct"] == 3

    # single-arm Student case
    out1 = cli._adj_power([1.0, 2.0, 3.0], [], 2.0, "absolute_delta", True)
    assert out1 is not None
    se1 = math.sqrt(1.0 / 3)
    expected1 = float(scipy_stats.norm.cdf(2.0 / se1 - 1.959963984540054))
    assert abs(out1["power"] - expected1) < 1e-12
    assert out1["n_for_80pct"] == 2


def test_underpowered_qualifier_on_supported_verdict(tmp_path):
    """A SUPPORTED verdict whose test had under 50% power to detect the
    registered effect carries the UNDERPOWERED qualifier; a tight-variance
    SUPPORTED verdict stays clean. Hand-built case: cand [0.65, 0.70, 0.75]
    vs base [0.50, 0.50, 0.50], threshold 0.05 -> SE = sqrt(0.0025/3) ~ 0.0289,
    achieved power ~ 41%, but the CI (df=2, crit 4.303) still clears."""
    _arm_artifacts(tmp_path / "weak", "cand", "ultrametric", [0.65, 0.70, 0.75])
    _arm_artifacts(tmp_path / "weak", "base", "standard", [0.50, 0.50, 0.50])
    v = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path / "weak"))
    assert v["verdict"] == "supported", v
    arm = v["arms"]["ultrametric"]
    assert arm["underpowered"] is True and v["underpowered"] is True
    assert 0.30 < arm["power"] < 0.50
    assert arm["n_for_80pct"] == 8  # ceil(0.0025 * 7.8489 / 0.0025)
    assert arm["p_value"] <= 0.025  # supported <=> one-sided p clears 2.5%

    _arm_artifacts(tmp_path / "tight", "cand", "ultrametric", [0.80, 0.82, 0.81])
    _arm_artifacts(tmp_path / "tight", "base", "standard", [0.50, 0.51, 0.52])
    v2 = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path / "tight"))
    assert v2["verdict"] == "supported"
    arm2 = v2["arms"]["ultrametric"]
    assert arm2["power"] > 0.99 and "underpowered" not in arm2
    assert "underpowered" not in v2


def test_p_value_matches_ci_decision_welch(tmp_path):
    """The ci-v4 p-value tests H0 at the registered threshold, so it must
    agree with the existing CI verdict rule: supported <=> p <= 0.025 and an
    inconclusive straddle <=> p > 0.025 (for >= claims)."""
    _arm_artifacts(tmp_path / "s", "cand", "ultrametric", [0.80, 0.82, 0.81])
    _arm_artifacts(tmp_path / "s", "base", "standard", [0.50, 0.51, 0.52])
    vs = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path / "s"))
    assert vs["verdict"] == "supported" and vs["arms"]["ultrametric"]["p_value"] <= 0.025
    assert vs["p_value"] == vs["arms"]["ultrametric"]["p_value"]

    _arm_artifacts(tmp_path / "i", "cand", "ultrametric", [0.50, 0.60, 0.55])
    _arm_artifacts(tmp_path / "i", "base", "standard", [0.48, 0.52, 0.50])
    vi = cli._adjudicate_hypothesis(_hyp(), _index(tmp_path / "i"))
    assert vi["verdict"] == "inconclusive" and vi["arms"]["ultrametric"]["p_value"] > 0.025


def test_ratio_bootstrap_p_value_deterministic_and_bounded():
    """Ratio-arm p-values reuse the engine's fixed-seed bootstrap stream:
    deterministic across calls, add-one smoothed (never exactly 0), and
    directionally sane."""
    cand, base = [2.0, 2.1, 1.9], [1.0, 1.05, 0.95]
    p1 = cli._adj_p_value(cand, base, 1.5, "ratio", ">=", False)
    p2 = cli._adj_p_value(cand, base, 1.5, "ratio", ">=", False)
    assert p1 is not None and p2 is not None
    assert p1 == p2, "fixed seed must make the bootstrap p deterministic"
    assert p1 >= 1.0 / (cli._ADJ_BOOTSTRAP_N + 1)  # minimum resolvable p
    assert p1 < 0.025  # ratio ~2 vs threshold 1.5: clearly supported
    p_flip = cli._adj_p_value(cand, base, 1.5, "ratio", "<=", False)
    assert p_flip is not None
    assert p_flip > 0.5  # the same evidence is terrible for a <= 1.5 claim


def test_fdr_headline_and_qvalues_in_run_report(tmp_path, monkeypatch):
    """Command-level ci-v4: verdicts.json carries the fdr block + per-verdict
    q_value, and the report headline shows BOTH numbers (N supported / M
    survive). Two clearly-supported hypotheses at tiny p -> both survive."""
    registry = tmp_path / "registry.yaml"
    h1 = _hyp()
    h2 = _hyp()
    h2["id"] = "hyp-engine-test-2"
    h2["prediction"]["metric_path"] = "evaltasks:tasks.hier.exact_match.greedy.in_range.mean"
    registry.write_text("schema_version: 1\nhypotheses:\n" + _yaml_entry(h1) + _yaml_entry(h2))
    monkeypatch.setattr(cli, "_hypotheses_registry_path", lambda: registry)
    monkeypatch.setattr(cli, "_load_parent_hypothesis_registry", lambda repo_root: None)
    _arm_artifacts(tmp_path / "a", "cand", "ultrametric", [0.80, 0.82, 0.81])
    _arm_artifacts(tmp_path / "a", "base", "standard", [0.50, 0.51, 0.52])

    result = runner.invoke(
        cli.app,
        [
            "adjudicate",
            "--all",
            "--dry-run",
            "--artifacts",
            str(tmp_path / "a"),
            "--artifacts-dir",
            str(tmp_path / "out"),
            "--run-id",
            "fdr",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "out" / "adjudications" / "fdr" / "verdicts.json").read_text())
    fdr = payload["fdr"]
    assert fdr["q_level"] == 0.10 and fdr["family_size"] == 2 and fdr["supported"] == 2
    assert sorted(fdr["supported_fdr_survivors"]) == ["hyp-engine-test", "hyp-engine-test-2"]
    for v in payload["verdicts"]:
        assert v["q_value"] <= 0.10 and v["p_value"] is not None
    report = (tmp_path / "out" / "adjudications" / "fdr" / "report.md").read_text()
    assert "2 supported, of which 2 survive FDR at q=0.1" in report


def test_sizing_probe_quarantine_excluded_from_pools(tmp_path):
    """dzor: artifacts under probes/sizing/ never enter evidence pools - the
    runs that SELECT a rung must not adjudicate it. probes/charges (chargeprobe
    instruments) remains readable evidence."""
    _evaltasks_artifact(tmp_path / "evals" / "tasks", "real", mechanism="ultrametric", per_seed=[0.8, 0.8, 0.8])
    _evaltasks_artifact(
        tmp_path / "probes" / "sizing" / "e2-probe", "probe-eval", mechanism="standard", per_seed=[0.9, 0.9, 0.9]
    )
    arts = cli._adj_collect_artifacts([tmp_path])
    paths = [a["path"] for a in arts]
    assert len(arts) == 1 and paths[0].endswith("real/summary.json"), paths


def _coord_artifact(root: Path, arm: str, seed: int, slope: float, provenance: dict | None = None):
    """A real mgr.bench.coord_curves.v1 artifact via the bp08 writer."""
    from nanochat.parameterization import write_coord_check_artifact

    res = {
        "attention_type": "tropical",
        "widths": [64, 128, 256],
        "activation_rms": {"64": 1.0, "128": 1.02, "256": 1.03},
        "loglog_slope": slope,
        "r_squared": 0.9,
        "concentration_class": "EVT (Gumbel max)",
    }
    return write_coord_check_artifact(
        res, root / f"coord_{arm}_s{seed}", parameterization=arm, seed=seed, provenance=provenance
    )


def test_coord_curves_ingest_as_bench_with_arm_selectors(tmp_path):
    """bp08 part 2: coord-curve artifacts collect as bench evidence; the rgyl
    selector on results.parameterization separates arms sharing the metric
    path, and bench:results.loglog_slope resolves through the generic walk."""
    for arm, slope in (("current", 0.12), ("nsa", 0.014)):
        _coord_artifact(tmp_path, arm, seed=0, slope=slope)
    idx = cli._adj_collect_artifacts([tmp_path])
    coord = [
        a for a in idx if a["schema"] == "bench" and a["data"].get("schema_version") == "mgr.bench.coord_curves.v1"
    ]
    assert len(coord) == 2, "coord-curves artifacts must ingest with schema=bench"

    cur = [a for a in coord if cli._adj_variant_matches(a, {"parameterization": "current"})]
    nsa = [a for a in coord if cli._adj_variant_matches(a, {"parameterization": "nsa"})]
    assert len(cur) == 1 and len(nsa) == 1, "arm selectors must partition the artifacts"
    assert cli._adj_observations(cur[0], "results.loglog_slope") == [0.12]
    assert cli._adj_observations(nsa[0], "results.loglog_slope") == [0.014]
    assert all(cli._adj_artifact_matches_arm(a, "tropical") for a in coord)
    assert not any(cli._adj_artifact_matches_arm(a, "standard") for a in coord)


def test_coord_curve_single_arm_hypothesis_adjudicates(tmp_path):
    """End-to-end engine path: a single-arm flatness claim (baseline: null) on
    clean-provenance nsa-arm coord curves reaches SUPPORTED through ci-v6."""
    for seed, slope in ((0, 0.010), (1, 0.020), (2, 0.015)):
        # explicit CLEAN provenance: production writers record real (dirty =>
        # tainted) provenance; this fixture exercises the verdict path
        _coord_artifact(tmp_path, "nsa", seed=seed, slope=slope, provenance=CLEAN_PROV)
    hyp = _hyp(
        {
            "metric_path": "bench:results.loglog_slope",
            "comparator": "<=",
            "threshold_kind": "absolute_delta",
            "threshold": 0.05,
            "baseline": None,
            "candidate_variant": {"parameterization": "nsa"},
            "min_seeds": 3,
        },
        mechanisms=["tropical"],
    )
    v = cli._adjudicate_hypothesis(hyp, cli._adj_collect_artifacts([tmp_path]))
    assert v["verdict"] == "supported", v


def test_status_flags_verdicts_whose_cited_artifacts_are_absent(tmp_path):
    """A ledger verdict must be re-derivable from the checkout. artifacts/ is
    gitignored and force-added piecemeal, so a SUPPORTED entry can cite paths
    nobody committed (six entries did on 2026-09-01); mgr status now lists
    them instead of letting the status line stand unsupported."""
    (tmp_path / "artifacts" / "bench" / "kept").mkdir(parents=True)
    (tmp_path / "artifacts" / "bench" / "kept" / "summary.json").write_text("{}")
    entries = [
        {
            "id": "hyp-kept",
            "status": "supported",
            "verdict_history": [{"verdict": "supported", "artifacts": ["artifacts/bench/kept/summary.json"]}],
        },
        {
            "id": "hyp-lost",
            "status": "supported",
            "verdict_history": [
                {"verdict": "inconclusive", "artifacts": ["artifacts/bench/kept/summary.json"]},
                {
                    "verdict": "supported",
                    "artifacts": ["artifacts/bench/kept/summary.json", "artifacts/bench/gone/summary.json"],
                },
            ],
        },
        {"id": "hyp-open", "status": "open"},
        {"id": "hyp-malformed", "status": "blocked", "verdict_history": ["not a dict"]},
    ]
    flagged = cli._status_evidence_missing(entries, tmp_path)
    assert [row["id"] for row in flagged] == ["hyp-lost"]
    assert flagged[0]["missing"] == ["artifacts/bench/gone/summary.json"]
    assert flagged[0]["cited"] == 2 and flagged[0]["verdict"] == "supported"
