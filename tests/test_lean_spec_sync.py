"""Spec-sync gate for the ordinal orchestrator (bead vnl.3, relation lab.3).

The Lean file `proofs/MGRProofs/OrdinalTermination.lean` documents the
REFERENCE event alphabet for lab.3's Python orchestrator (lines of the form
`- evt: NAME` in the Event-Alphabet docstring). This test reads BOTH sides
(read-only; no code-modifying scripts per AGENTS.md) and fails if the
Python event enum drifts from the Lean spec.

While lab.3's orchestrator module does not exist yet, the sync test SKIPs
with an explicit reason instead of failing - the Lean side remains the
contract lab.3 must implement.
"""

import re
from pathlib import Path

import pytest

LEAN_FILE = Path(__file__).resolve().parent.parent / "proofs" / "MGRProofs" / "OrdinalTermination.lean"


def _lean_event_alphabet() -> list[str]:
    """Parse `- evt: NAME` entries from the Lean spec docstring."""
    text = LEAN_FILE.read_text(encoding="utf-8")
    events = re.findall(r"^\s*-\s*evt:\s*(\S+)\s*$", text, flags=re.M)
    assert events, "Lean spec lost its event alphabet - fix the docstring"
    return events


def _lab3_module_exists() -> bool:
    import importlib.util

    return importlib.util.find_spec("nanochat.ordinal_orchestrator") is not None


def _python_event_alphabet() -> set[str]:
    """Import lab.3's orchestrator event enum (must exist when called)."""
    import importlib

    mod = importlib.import_module("nanochat.ordinal_orchestrator")
    return {member.name for member in mod.OrchestratorEvent}  # type: ignore[attr-defined]


def test_lean_spec_declares_expected_alphabet():
    """The Lean reference alphabet is present and non-trivial."""
    events = _lean_event_alphabet()
    for expected in ("PHASE_ADVANCE", "WITHIN_PHASE_RETRY", "PHASE_ESCALATE"):
        assert expected in events


def test_python_events_stay_in_sync_with_lean_spec():
    """The Python event enum mirrors the Lean spec exactly.

    Skips (with reason) until lab.3 lands its orchestrator module; once it
    exists this becomes a hard gate against drift.
    """
    lean = set(_lean_event_alphabet())
    if not _lab3_module_exists():
        pytest.skip("lab.3 orchestrator module not implemented yet")
    py = _python_event_alphabet()
    missing_in_python = sorted(lean - py)
    extra_in_python = sorted(py - lean)
    assert not missing_in_python and not extra_in_python, (
        "orchestrator event enum drifted from the Lean spec: "
        f"python-only={extra_in_python} lean-only={missing_in_python}"
    )
