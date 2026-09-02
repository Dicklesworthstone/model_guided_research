"""Tests for the hypothesis registry + mgr hypotheses CLI (bead hij.1).

Mirrors tests/test_theorem_registry.py: the seeded registry must validate
green; malformed fixtures must be rejected with actionable errors; the
append-only governance must reject history rewrites/truncations against a
parent version; the add command must text-append (preserving hand-written
comments) and roll back on validation failure.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import cli

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parent.parent


def _validate(data, parent=None):
    return cli._validate_hypothesis_registry(data, [], REPO_ROOT, parent=parent)


def _entry(**overrides):
    base = {
        "id": "hyp-test-entry",
        "statement": "test claim",
        "mechanisms": ["tropical"],
        "source": {"kind": "human", "provenance": "unit test"},
        "date_registered": "2026-06-10",
        "prediction": {
            "metric_path": "evaltasks:tasks.dyck.exact_match.greedy.held_out.mean",
            "comparator": ">=",
            "threshold_kind": "absolute_delta",
            "threshold": 0.05,
            "baseline": {"mechanism": "standard", "equal_flops": True},
            "min_seeds": 3,
        },
        "status": "open",
        "evidence": [],
        "verdict_history": [],
    }
    base.update(overrides)
    return base


def _registry(*entries):
    return {"schema_version": 1, "hypotheses": list(entries)}


# ---------------------------------------------------------------------------
# the committed registry


def test_seeded_registry_validates_green():
    data, load_errors = cli._load_hypothesis_registry(cli._hypotheses_registry_path())
    assert data is not None, load_errors
    errors, warnings, summary = _validate(data)
    assert load_errors == [] and errors == [], f"seeded registry must be green: {errors}"
    assert summary["entries"] >= 20, "the seeding (README + docs claims) is the bulk of the bead"
    assert summary["operationalized"] >= 12
    # visible debt, never silent omission: blocked entries carry notes
    for h in data["hypotheses"]:
        if h["prediction"] is None:
            assert h.get("operationalization_note"), f"{h['id']}: null prediction without a note"


def test_seeded_registry_theorem_refs_resolve():
    data, load_errors = cli._load_hypothesis_registry(cli._hypotheses_registry_path())
    th_data, theorem_load_errors = cli._load_theorem_registry(cli._theorems_registry_path())
    assert data is not None, load_errors
    assert th_data is not None, theorem_load_errors
    theorem_ids = {t["id"] for t in th_data["theorems"]}
    refs = [r for h in data["hypotheses"] for r in (h.get("theorem_refs") or [])]
    assert refs, "seeded registry should cross-link at least some theorems"
    assert all(r in theorem_ids for r in refs)


def test_readme_table_replaced_by_registry_pointer():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "| Tropical | Robustness |" not in readme, "the speculative table must not return to the README"
    assert "hypotheses/registry.yaml" in readme
    assert "mgr hypotheses list" in readme


# ---------------------------------------------------------------------------
# schema validation


def test_malformed_entries_rejected():
    cases = [
        (_entry(id="BadId"), "must match"),
        (_entry(status="confirmed"), "status"),
        (_entry(source={"kind": "oracle", "provenance": "x"}), "source.kind"),
        (_entry(date_registered="June 10"), "YYYY-MM-DD"),
        (_entry(mechanisms=["warp-drive"]), "unknown mechanism"),
        (_entry(statement="  "), "statement"),
        (_entry(evidence="not-a-list"), "evidence"),
    ]
    for entry, fragment in cases:
        errors, _, _ = _validate(_registry(entry))
        assert any(fragment in e for e in errors), f"expected {fragment!r} error, got {errors}"


def test_duplicate_ids_rejected():
    errors, _, _ = _validate(_registry(_entry(), _entry()))
    assert any("duplicate id" in e for e in errors)


def test_manual_hold_schema():
    # a well-formed hold validates clean
    errors, _, _ = _validate(_registry(_entry(manual_hold={"held": True, "reason": "frozen by hand-check"})))
    assert errors == [], errors
    # held=false (released) needs no reason
    errors, _, _ = _validate(_registry(_entry(manual_hold={"held": False})))
    assert errors == [], errors
    # malformed holds are rejected with actionable errors
    cases = [
        (_entry(manual_hold="frozen"), "must be a mapping"),
        (_entry(manual_hold={"held": "yes", "reason": "x"}), "held must be a boolean"),
        (_entry(manual_hold={"held": True}), "non-empty reason"),
        (_entry(manual_hold={"held": True, "reason": "x", "date": "June 13"}), "manual_hold.date"),
    ]
    for entry, fragment in cases:
        errors, _, _ = _validate(_registry(entry))
        assert any(fragment in e for e in errors), f"expected {fragment!r}, got {errors}"


def test_manual_hold_toggle_is_not_append_only_violation():
    # A hold is a human control field, not a claim: adding it (and later
    # clearing it) must pass append-only governance against the committed parent
    # (srtf). Only verdict_history/date_registered/statement are guarded.
    parent = _registry(_entry())
    held = _registry(_entry(manual_hold={"held": True, "reason": "frozen pending re-test"}))
    errors, _, _ = _validate(held, parent=parent)
    assert errors == [], f"placing a hold must not trip append-only governance: {errors}"
    # and releasing it (parent held -> child cleared) is likewise clean
    errors, _, _ = _validate(_registry(_entry()), parent=held)
    assert errors == [], f"releasing a hold must not trip governance: {errors}"


def test_prediction_contract_enforced():
    bad_pred = dict(_entry()["prediction"])
    cases = [
        ({**bad_pred, "metric_path": "nope"}, "metric_path"),
        ({**bad_pred, "metric_path": "magic:tasks.dyck.x"}, "schema 'magic'"),
        ({**bad_pred, "metric_path": "evaltasks:tasks.notatask.x"}, "unknown task"),
        # an answer metric on an LM-only task is null on every eval (the regime
        # hypotheses were registered that way for 9 days, bead w76r)
        ({**bad_pred, "metric_path": "evaltasks:tasks.placebo.exact_match.greedy.held_out.mean"}, "no answer marker"),
        ({**bad_pred, "metric_path": "evaltasks:foo.dyck.x"}, "must start with 'tasks."),
        ({**bad_pred, "comparator": "=="}, "comparator"),
        ({**bad_pred, "threshold_kind": "percent"}, "threshold_kind"),
        ({**bad_pred, "threshold": "big"}, "threshold must be a number"),
        ({**bad_pred, "threshold_kind": "ratio", "threshold": -1}, "ratio threshold"),
        ({**bad_pred, "baseline": {"mechanism": "standard", "equal_flops": False}}, "equal_flops"),
        ({**bad_pred, "baseline": {"mechanism": "alien", "equal_flops": True}}, "baseline.mechanism"),
        ({**bad_pred, "min_seeds": 0}, "min_seeds"),
        ({**bad_pred, "validity": "not-a-mapping"}, "validity must be a mapping"),
        ({**bad_pred, "validity": {"baseline_floor": "tall", "floor_source": "x"}}, "baseline_floor must be a number"),
        ({**bad_pred, "validity": {"baseline_floor": 0.5}}, "floor_source"),
        ({**bad_pred, "validity": {"baseline_floor": 0.5, "floor_margin": -0.1, "floor_source": "x"}}, "floor_margin"),
        ({**bad_pred, "validity": {"baseline_floor": 0.5, "floor_ceiling": 1.0, "floor_source": "x"}}, "unknown key"),
    ]
    for pred, fragment in cases:
        errors, _, _ = _validate(_registry(_entry(prediction=pred)))
        assert any(fragment in e for e in errors), f"expected {fragment!r} error, got {errors}"


def test_exact_match_prediction_without_floor_warns():
    """ci-v2: EM predictions should register an answer-prior fallback floor;
    without one the gate has only artifact-recorded priors to lean on."""
    errors, warnings, _ = _validate(_registry(_entry()))  # dyck EM metric, no validity
    assert errors == []
    assert any("floor gate" in w for w in warnings), warnings

    with_floor = _entry()
    with_floor["prediction"] = {
        **with_floor["prediction"],
        "validity": {"baseline_floor": 0.512, "floor_source": "population prior (test)"},
    }
    errors, warnings, _ = _validate(_registry(with_floor))
    assert errors == [] and not any("floor gate" in w for w in warnings)


def test_null_prediction_requires_note_and_open_or_blocked_status():
    entry = _entry(prediction=None)
    errors, _, _ = _validate(_registry(entry))
    assert any("operationalization_note" in e for e in errors)

    entry = _entry(prediction=None, operationalization_note="needs D2", status="blocked")
    errors, _, _ = _validate(_registry(entry))
    assert errors == []

    entry = _entry(prediction=None, operationalization_note="needs D2", status="supported")
    errors, _, _ = _validate(_registry(entry))
    assert any("requires an operationalized prediction" in e for e in errors)


def test_unknown_theorem_ref_rejected():
    errors, _, _ = _validate(_registry(_entry(theorem_refs=["thm-does-not-exist"])))
    assert any("thm-does-not-exist" in e for e in errors)


def test_verdict_history_entry_shape_enforced():
    bad = _entry(
        verdict_history=[{"date": "soon", "verdict": "maybe", "artifacts": [], "adjudicator": ""}],
        status="supported",
    )
    errors, _, _ = _validate(_registry(bad))
    joined = "\n".join(errors)
    for fragment in (
        "date must be YYYY-MM-DD",
        "verdict must be one of",
        "artifacts must be a non-empty",
        "adjudicator",
    ):
        assert fragment in joined, f"missing {fragment!r} in: {errors}"


# ---------------------------------------------------------------------------
# append-only governance


def _verdict(date="2026-06-11", verdict="supported"):
    return {
        "date": date,
        "verdict": verdict,
        "artifacts": ["artifacts/evals/tasks/x/summary.json"],
        "adjudicator": "human",
    }


def test_append_only_allows_appends():
    parent = _registry(_entry(verdict_history=[_verdict()], status="supported"))
    child = _registry(
        _entry(verdict_history=[_verdict(), _verdict(date="2026-07-01", verdict="refuted")], status="refuted")
    )
    errors, _, _ = _validate(child, parent=parent)
    assert errors == [], f"append must be allowed: {errors}"


def test_append_only_rejects_rewrite_truncation_and_deletion():
    parent = _registry(_entry(verdict_history=[_verdict()], status="supported"))

    truncated = _registry(_entry(verdict_history=[], status="open"))
    errors, _, _ = _validate(truncated, parent=parent)
    assert any("APPEND-ONLY" in e for e in errors)

    rewritten = _registry(_entry(verdict_history=[_verdict(verdict="refuted")], status="refuted"))
    errors, _, _ = _validate(rewritten, parent=parent)
    assert any("APPEND-ONLY" in e for e in errors)

    deleted = _registry(_entry(id="hyp-different-entry"))
    errors, _, _ = _validate(deleted, parent=parent)
    assert any("entry deleted" in e for e in errors)


def test_append_only_rejects_registration_date_edits():
    parent = _registry(_entry())
    child = _registry(_entry(date_registered="2020-01-01"))
    errors, _, _ = _validate(child, parent=parent)
    assert any("date_registered changed" in e for e in errors)


def test_statement_morphing_warns():
    parent = _registry(_entry())
    child = _registry(_entry(statement="a conveniently weaker claim"))
    errors, warnings, _ = _validate(child, parent=parent)
    assert errors == []
    assert any("statement text changed" in w for w in warnings)


# ---------------------------------------------------------------------------
# CLI surface


def test_cli_list_show_validate():
    result = runner.invoke(cli.app, ["hypotheses", "list", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    ids = [h["id"] for h in payload["hypotheses"]]
    assert "hyp-placebo-no-winner" in ids

    result = runner.invoke(cli.app, ["hypotheses", "list", "--status", "blocked", "--json"])
    assert result.exit_code == 0
    blocked = json.loads(result.output)["hypotheses"]
    assert blocked and all(h["status"] == "blocked" and h["prediction"] is None for h in blocked)

    result = runner.invoke(cli.app, ["hypotheses", "show", "hyp-ultrametric-hier-heldout-depth"])
    assert result.exit_code == 0 and "evaltasks:tasks.hier" in result.output

    result = runner.invoke(cli.app, ["hypotheses", "show", "hyp-nope"])
    assert result.exit_code == 1

    result = runner.invoke(cli.app, ["hypotheses", "validate"])
    assert result.exit_code == 0, result.output


def test_cli_add_appends_preserving_comments_and_rolls_back_on_error(tmp_path, monkeypatch):
    registry_copy = tmp_path / "registry.yaml"
    registry_copy.write_text(cli._hypotheses_registry_path().read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(cli, "_hypotheses_registry_path", lambda: registry_copy)
    # isolate from git HEAD: the tmp copy has no parent version
    monkeypatch.setattr(cli, "_load_parent_hypothesis_registry", lambda repo_root: None)

    sentinel = "# HYPOTHESIS REGISTRY"  # a hand-written header comment that must survive
    assert sentinel in registry_copy.read_text(encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "hypotheses",
            "add",
            "--id",
            "hyp-test-added",
            "--statement",
            "added by the CLI test",
            "--mechanism",
            "tropical",
            "--source-kind",
            "human",
            "--provenance",
            "tests/test_hypotheses_registry.py",
            "--metric-path",
            "evaltasks:tasks.dyck.exact_match.greedy.held_out.mean",
        ],
    )
    assert result.exit_code == 0, result.output
    text = registry_copy.read_text(encoding="utf-8")
    assert sentinel in text, "text-append must preserve hand-written comments"
    assert "hyp-test-added" in text
    data, _ = cli._load_hypothesis_registry(registry_copy)
    errors, _, _ = _validate(data)
    assert errors == []

    before = registry_copy.read_text(encoding="utf-8")
    result = runner.invoke(
        cli.app,
        [
            "hypotheses",
            "add",
            "--id",
            "hyp-test-added",  # duplicate -> must roll back
            "--statement",
            "dup",
            "--mechanism",
            "tropical",
            "--source-kind",
            "human",
            "--provenance",
            "x",
            "--metric-path",
            "evaltasks:tasks.dyck.exact_match.greedy.held_out.mean",
        ],
    )
    assert result.exit_code == 1
    assert registry_copy.read_text(encoding="utf-8") == before, "failed add must roll the file back"

    result = runner.invoke(
        cli.app,
        [
            "hypotheses",
            "add",
            "--id",
            "hyp-no-pred",
            "--statement",
            "x",
            "--mechanism",
            "tropical",
            "--source-kind",
            "human",
            "--provenance",
            "y",
        ],
    )
    assert result.exit_code == 2, "omitting both --metric-path and --note must be rejected"


def test_cli_validate_catches_malformed_file(tmp_path, monkeypatch):
    bad = tmp_path / "registry.yaml"
    bad.write_text("schema_version: 1\nhypotheses:\n  - id: BadId\n    statement: x\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_hypotheses_registry_path", lambda: bad)
    monkeypatch.setattr(cli, "_load_parent_hypothesis_registry", lambda repo_root: None)
    result = runner.invoke(cli.app, ["hypotheses", "validate"])
    assert result.exit_code == 1
    assert "must match" in result.output


def test_parent_loader_reads_git_head():
    parent = cli._load_parent_hypothesis_registry(REPO_ROOT)
    # Before this bead's commit there is no parent; after it there is.
    # Either way the loader must not raise and must return None or a mapping.
    assert parent is None or isinstance(parent, dict)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))


def test_variant_selector_validation():
    """rgyl: variant selectors must be non-empty scalar-valued mappings."""
    good = _entry()
    good["prediction"] = {
        **good["prediction"],
        "baseline": {"mechanism": "tropical", "equal_flops": True, "variant": {"semiring_beta_spec": "1.0"}},
        "candidate_variant": {"semiring_beta_spec": "linear:1:32"},
    }
    errors, _, _ = _validate(_registry(good))
    assert errors == [], errors

    cases = [
        ({"variant": "not-a-mapping"}, "baseline.variant must be a non-empty mapping"),
        ({"variant": {}}, "baseline.variant must be a non-empty mapping"),
        ({"variant": {"semiring_beta_spec": {"nested": 1}}}, "must be a scalar"),
        ({"variant": {"not an identifier": 1}}, "identifier-like"),
    ]
    for variant_patch, fragment in cases:
        bad = _entry()
        bad["prediction"] = {
            **bad["prediction"],
            "baseline": {"mechanism": "standard", "equal_flops": True, **variant_patch},
        }
        errors, _, _ = _validate(_registry(bad))
        assert any(fragment in e for e in errors), f"expected {fragment!r}, got {errors}"

    bad = _entry()
    bad["prediction"] = {**bad["prediction"], "candidate_variant": []}
    errors, _, _ = _validate(_registry(bad))
    assert any("candidate_variant must be a non-empty mapping" in e for e in errors)
