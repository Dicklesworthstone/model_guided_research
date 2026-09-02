"""Tests for the theorem registry + mgr theorems CLI (bead model_guided_research-vnl.1).

Covers, with informative failure messages throughout:
  - the REAL registry: loads, validates statically with zero errors, has >= 25
    entries, spans all three theory pillars, and its pytest refs resolve
    against a live (scoped) pytest collection;
  - schema validation against malformed fixtures: duplicate ids, bad status,
    dangling depends_on, unknown certify refs, bogus/missing anchors,
    lean-checked without a proof file, conj-/thm- prefix-status mismatches;
  - the pure pytest-ref resolution semantics (exact / class prefix /
    parametrized);
  - the certify known-check-name constant staying in sync with the certify
    implementation in cli.py (source-scan, read-only);
  - the CLI surface via typer's CliRunner: list/show/validate, --json modes,
    and exit codes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

import cli

REPO_ROOT = Path(cli.__file__).resolve().parent
runner = CliRunner()


# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------


def entry(**overrides: Any) -> dict[str, Any]:
    """A minimal valid registry entry; override fields per test."""
    base: dict[str, Any] = {
        "id": "thm-test-entry",
        "statement": "A test statement.",
        "status": "proved-on-paper",
        "mechanisms": ["tropical"],
        "source_note": {"path": "pending", "anchor": None},
        "proof_location": "test fixture",
        "numerical_checks": [],
        "used_by": [],
        "depends_on": [],
    }
    base.update(overrides)
    return base


def registry(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "theorems": list(entries)}


def validate(data: dict[str, Any], repo_root: Path = REPO_ROOT, **kw: Any):
    return cli._validate_theorem_registry(data, [], repo_root, **kw)


# ---------------------------------------------------------------------------
# the real registry
# ---------------------------------------------------------------------------


def test_real_registry_loads_and_is_statically_valid() -> None:
    data, load_errors = cli._load_theorem_registry(cli._theorems_registry_path())
    assert not load_errors, f"registry failed to load: {load_errors}"
    assert data is not None
    errors, warnings, summary = validate(data)
    assert not errors, "real registry has validation errors:\n" + "\n".join(errors)
    assert not warnings, "real registry has warnings:\n" + "\n".join(warnings)
    assert summary["entries"] >= 25, f"bead vnl.1 requires >= 25 seeded entries, found {summary['entries']}"
    for status in summary["by_status"]:
        assert status in cli._THEOREM_STATUSES


def test_real_registry_spans_all_three_theory_pillars() -> None:
    data, _ = cli._load_theorem_registry(cli._theorems_registry_path())
    assert data is not None
    used_by = " ".join(" ".join(t.get("used_by") or []) for t in data["theorems"])
    for epic in ("model_guided_research-8gk", "model_guided_research-u55", "model_guided_research-lab"):
        assert epic in used_by, f"registry must span pillar epic {epic}; none of its beads appear in used_by"


def test_real_registry_pytest_refs_resolve_against_live_collection() -> None:
    data, _ = cli._load_theorem_registry(cli._theorems_registry_path())
    assert data is not None
    refs = [c["ref"] for t in data["theorems"] for c in (t.get("numerical_checks") or []) if c.get("kind") == "pytest"]
    assert refs, "real registry should carry at least one pytest pointer"
    scope_files = sorted({r.split("::")[0] for r in refs})
    collected, problem = cli._collect_pytest_node_ids(REPO_ROOT, scope_files)
    assert problem is None, f"pytest collection failed: {problem}"
    unresolved = [r for r in refs if not cli._pytest_ref_resolves(r, collected)]
    assert not unresolved, f"pytest refs that do not resolve against live collection over {scope_files}: {unresolved}"


# ---------------------------------------------------------------------------
# schema validation against malformed fixtures
# ---------------------------------------------------------------------------


def test_duplicate_id_rejected() -> None:
    errors, _, _ = validate(registry(entry(), entry()))
    assert any("duplicate id" in e for e in errors), f"expected duplicate-id error, got: {errors}"


def test_bad_status_rejected() -> None:
    errors, _, _ = validate(registry(entry(status="proven")))
    assert any("status" in e and "proven" in e for e in errors), f"expected status error, got: {errors}"


def test_bad_id_format_rejected() -> None:
    errors, _, _ = validate(registry(entry(id="Theorem_One")))
    assert any("id missing or not matching" in e for e in errors), f"expected id-format error, got: {errors}"


def test_dangling_depends_on_rejected() -> None:
    errors, _, _ = validate(registry(entry(depends_on=["thm-does-not-exist"])))
    assert any("unknown theorem id 'thm-does-not-exist'" in e for e in errors), f"got: {errors}"


def test_string_depends_on_rejected_with_single_error() -> None:
    """A malformed string depends_on must produce ONE clear type error, not
    one confusing unknown-id error per character."""
    errors, _, _ = validate(registry(entry(depends_on="thm-other")))
    assert any("depends_on must be a list" in e for e in errors), f"got: {errors}"
    assert not any("unknown theorem id" in e for e in errors), f"char-iteration leak: {errors}"


def test_nonstring_mechanism_rejected() -> None:
    errors, _, _ = validate(registry(entry(mechanisms=["tropical", None])))
    assert any("mechanisms must be a list of strings" in e for e in errors), f"got: {errors}"


def test_unknown_certify_ref_rejected() -> None:
    errors, _, _ = validate(registry(entry(numerical_checks=[{"kind": "certify", "ref": "tropical.not_a_real_check"}])))
    assert any("not a known check name" in e for e in errors), f"got: {errors}"


def test_known_certify_ref_accepted() -> None:
    errors, _, _ = validate(
        registry(entry(numerical_checks=[{"kind": "certify", "ref": "tropical.lipschitz_1_sup_norm_q"}]))
    )
    assert not errors, f"known certify ref should validate, got: {errors}"


def test_bad_check_kind_rejected() -> None:
    errors, _, _ = validate(registry(entry(numerical_checks=[{"kind": "doctest", "ref": "x"}])))
    assert any("check kind" in e for e in errors), f"got: {errors}"


def test_missing_note_path_rejected(tmp_path: Path) -> None:
    errors, _, _ = validate(
        registry(entry(source_note={"path": "markdown_documentation/nope.md", "anchor": None})),
        repo_root=tmp_path,
    )
    assert any("does not exist" in e for e in errors), f"got: {errors}"


def test_bogus_anchor_rejected(tmp_path: Path) -> None:
    note = tmp_path / "note.md"
    note.write_text("# Title\n\n## Real Section\n\nbody\n", encoding="utf-8")
    bad = entry(source_note={"path": "note.md", "anchor": "Imaginary Section"})
    errors, _, _ = validate(registry(bad), repo_root=tmp_path)
    assert any("anchor" in e and "Imaginary Section" in e for e in errors), f"got: {errors}"


def test_real_anchor_accepted_case_insensitive(tmp_path: Path) -> None:
    note = tmp_path / "note.md"
    note.write_text("# Title\n\n## Exact Uncertainty Bounds\n", encoding="utf-8")
    good = entry(source_note={"path": "note.md", "anchor": "exact uncertainty"})
    errors, _, _ = validate(registry(good), repo_root=tmp_path)
    assert not errors, f"substring/case-insensitive anchor should validate, got: {errors}"


def test_pending_note_is_counted_not_errored() -> None:
    errors, _, summary = validate(registry(entry()))
    assert not errors
    assert summary["pending_notes"] == 1


def test_lean_checked_requires_existing_proof_file(tmp_path: Path) -> None:
    lc = entry(status="lean-checked", proof_location="proofs/Tropical.lean RouteStability.route_stable")
    errors, _, _ = validate(registry(lc), repo_root=tmp_path)
    assert any("proof file does not exist" in e for e in errors), f"got: {errors}"
    (tmp_path / "proofs").mkdir()
    (tmp_path / "proofs" / "Tropical.lean").write_text("-- lemma\n", encoding="utf-8")
    # an existing file alone is not enough any more: the label must be bound
    # to a lemma the axiom audit prints (formalization block, scope full)
    errors_unbound, _, _ = validate(registry(lc), repo_root=tmp_path)
    assert any("requires a formalization block" in e for e in errors_unbound), f"got: {errors_unbound}"
    (tmp_path / "proofs" / "AxiomCheck.lean").write_text(
        "#print axioms RouteStability.route_stable\n", encoding="utf-8"
    )
    bound = entry(
        status="lean-checked",
        proof_location="proofs/Tropical.lean RouteStability.route_stable",
        formalization={"file": "proofs/Tropical.lean", "theorems": ["RouteStability.route_stable"], "scope": "full"},
    )
    errors2, _, _ = validate(registry(bound), repo_root=tmp_path)
    assert not errors2, f"lean-checked bound to an audited lemma should validate, got: {errors2}"


def test_lean_checked_without_proofs_token_rejected() -> None:
    lc = entry(status="lean-checked", proof_location="formalized somewhere")
    errors, _, _ = validate(registry(lc))
    assert any("requires proof_location naming a proofs/ file" in e for e in errors), f"got: {errors}"


def test_conj_prefix_with_proved_status_warns_not_errors() -> None:
    errors, warnings, _ = validate(registry(entry(id="conj-now-proved", status="proved-on-paper")))
    assert not errors, f"prefix mismatch must be a warning, got errors: {errors}"
    assert any("conj- prefix" in w for w in warnings), f"expected prefix warning, got: {warnings}"


def test_deep_tier_flags_unresolved_pytest_ref() -> None:
    bad = entry(numerical_checks=[{"kind": "pytest", "ref": "tests/test_theorem_registry.py::test_no_such"}])
    errors, _, _ = validate(registry(bad), collected_pytest_ids=["tests/test_theorem_registry.py::test_real"])
    assert any("does not resolve against live collection" in e for e in errors), f"got: {errors}"


# ---------------------------------------------------------------------------
# pytest-ref resolution semantics (pure function)
# ---------------------------------------------------------------------------


def test_pytest_ref_resolution_semantics() -> None:
    collected = [
        "tests/test_x.py::test_exact",
        "tests/test_x.py::TestClass::test_method",
        "tests/test_x.py::test_param[case-1]",
    ]
    assert cli._pytest_ref_resolves("tests/test_x.py::test_exact", collected), "exact id must resolve"
    assert cli._pytest_ref_resolves("tests/test_x.py::TestClass", collected), "class prefix must resolve"
    assert cli._pytest_ref_resolves("tests/test_x.py::test_param", collected), "parametrized base must resolve"
    assert not cli._pytest_ref_resolves("tests/test_x.py::test_missing", collected), "unknown must not resolve"
    assert not cli._pytest_ref_resolves("tests/test_x.py::test_exa", collected), "partial name must not resolve"


# ---------------------------------------------------------------------------
# certify known-name constant stays in sync with the certify implementation
# ---------------------------------------------------------------------------


def test_certify_known_names_in_sync_with_cli_source() -> None:
    src = (REPO_ROOT / "cli.py").read_text(encoding="utf-8")
    literal = {f"{m}.{n}" for m, n in re.findall(r'add_check\(\s*"([a-z_]+)"\s*,\s*"([a-z0-9_]+)"', src, re.S)}
    assert literal == set(cli._CERTIFY_NAMED_CHECKS), (
        "literal add_check names in cli.py and _CERTIFY_NAMED_CHECKS have drifted apart.\n"
        f"  in source but not constant: {sorted(literal - set(cli._CERTIFY_NAMED_CHECKS))}\n"
        f"  in constant but not source: {sorted(set(cli._CERTIFY_NAMED_CHECKS) - literal)}"
    )
    known = cli._certify_known_check_names()
    for mech in cli._CERTIFY_MECHANISMS:
        assert f"{mech}.causality_no_future_grad" in known, f"missing causality check name for {mech}"


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_theorems_list_runs_green() -> None:
    result = runner.invoke(cli.app, ["theorems", "list"])
    assert result.exit_code == 0, f"list failed: {result.output}"
    # rich may truncate long ids in narrow test terminals - assert on the
    # table title and a short id that always fits
    assert "Theorem registry" in result.output
    assert "thm-flat-error" in result.output


def test_cli_theorems_list_json_and_filters() -> None:
    result = runner.invoke(cli.app, ["theorems", "list", "--json", "--status", "conjecture"])
    assert result.exit_code == 0, f"list --json failed: {result.output}"
    payload = json.loads(result.output)
    assert payload["theorems"], "expected at least one conjecture"
    assert all(t["status"] == "conjecture" for t in payload["theorems"])
    result2 = runner.invoke(cli.app, ["theorems", "list", "--json", "--mechanism", "ordinal"])
    payload2 = json.loads(result2.output)
    assert all("ordinal" in t["mechanisms"] for t in payload2["theorems"])
    assert payload2["theorems"], "expected ordinal-mechanism entries"


def test_cli_theorems_show_known_and_unknown() -> None:
    ok = runner.invoke(cli.app, ["theorems", "show", "thm-flat-error"])
    assert ok.exit_code == 0, f"show failed: {ok.output}"
    assert "strong triangle" in ok.output or "k >= 0" in ok.output
    missing = runner.invoke(cli.app, ["theorems", "show", "thm-not-real"])
    assert missing.exit_code == 1, "unknown id must exit 1"
    assert "No theorem" in missing.output


def test_cli_theorems_validate_green_and_json() -> None:
    result = runner.invoke(cli.app, ["theorems", "validate"])
    assert result.exit_code == 0, f"validate failed:\n{result.output}"
    result_json = runner.invoke(cli.app, ["theorems", "validate", "--json"])
    assert result_json.exit_code == 0
    payload = json.loads(result_json.output)
    assert payload["errors"] == []
    assert payload["summary"]["entries"] >= 25


# ---------------------------------------------------------------------------
# formalization block: a label is bound to the Lean artifact the CI audits
# ---------------------------------------------------------------------------


def _formalization(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "file": "proofs/MGRProofs/RouteStability.lean",
        "theorems": ["MGRProofs.lse_max_sandwich"],
        "scope": "full",
    }
    base.update(overrides)
    return base


def test_lean_checked_requires_formalization_block() -> None:
    errors, _, _ = validate(
        registry(entry(status="lean-checked", proof_location="Lean: proofs/MGRProofs/RouteStability.lean (x)"))
    )
    assert any("requires a formalization block" in e for e in errors), f"got: {errors}"


def test_lean_checked_with_full_formalization_validates() -> None:
    errors, _, _ = validate(
        registry(
            entry(
                status="lean-checked",
                proof_location="Lean: proofs/MGRProofs/RouteStability.lean (lse_max_sandwich)",
                formalization=_formalization(),
            )
        )
    )
    assert not errors, f"a fully formalized, audited lemma must validate, got: {errors}"


def test_formalization_theorem_must_be_audited_by_axiom_check() -> None:
    """Kill witness: naming a Lean theorem the axiom gate never prints leaves
    the label unbound to any CI-checked artifact."""
    errors, _, _ = validate(
        registry(
            entry(
                status="lean-checked",
                proof_location="Lean: proofs/MGRProofs/RouteStability.lean (x)",
                formalization=_formalization(theorems=["MGRProofs.not_a_real_lemma"]),
            )
        )
    )
    assert any("not audited by proofs/AxiomCheck.lean" in e for e in errors), f"got: {errors}"


def test_partial_formalization_cannot_carry_lean_checked_status() -> None:
    errors, _, _ = validate(
        registry(
            entry(
                status="lean-checked",
                proof_location="Lean: proofs/MGRProofs/RouteStability.lean (x)",
                formalization=_formalization(scope="partial", unformalized="the k-ary case"),
            )
        )
    )
    assert any("scope must be full" in e for e in errors), f"got: {errors}"


def test_partial_formalization_requires_unformalized_note() -> None:
    errors, _, _ = validate(registry(entry(formalization=_formalization(scope="partial"))))
    assert any("requires 'unformalized'" in e for e in errors), f"got: {errors}"


def test_partial_formalization_on_paper_theorem_validates() -> None:
    errors, _, _ = validate(
        registry(entry(formalization=_formalization(scope="partial", unformalized="the boundary clause")))
    )
    assert not errors, f"got: {errors}"


def test_conjecture_cannot_carry_formalization() -> None:
    errors, _, _ = validate(registry(entry(id="conj-test-entry", status="conjecture", formalization=_formalization())))
    assert any("conjecture cannot carry a formalization" in e for e in errors), f"got: {errors}"


def test_real_registry_formalized_theorems_are_all_audited() -> None:
    """Every Lean lemma the registry points at is printed by AxiomCheck.lean,
    so the CI sorryAx grep actually covers every lean-checked label."""
    data, load_errors = cli._load_theorem_registry(REPO_ROOT / "hypotheses" / "theorems.yaml")
    assert data is not None and not load_errors
    audited = {
        line.strip()[len("#print axioms ") :]
        for line in (REPO_ROOT / "proofs" / "AxiomCheck.lean").read_text().splitlines()
        if line.strip().startswith("#print axioms ")
    }
    formalized = [t for t in data["theorems"] if isinstance(t.get("formalization"), dict)]
    assert formalized, "the real registry should carry formalization blocks"
    for t in formalized:
        for name in t["formalization"]["theorems"]:
            assert name in audited, f"{t['id']}: {name} is not audited by AxiomCheck.lean"
    lean_checked = [t for t in data["theorems"] if t.get("status") == "lean-checked"]
    assert all(t["formalization"]["scope"] == "full" for t in lean_checked)
