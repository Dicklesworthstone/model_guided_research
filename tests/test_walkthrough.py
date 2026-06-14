"""Tests for nanochat.walkthrough (bead 9f1): pedagogical walkthrough mode.

Covers the env gate, both narration modes (live nanochat mini-run + conceptual
framework demo), the bracket-safe vector formatter, and that every mechanism's
teaching note points at a doc that actually exists.
"""

from __future__ import annotations

import math
from pathlib import Path

from rich.console import Console

from nanochat import walkthrough

REPO_ROOT = Path(__file__).resolve().parent.parent


def _silent() -> Console:
    import io

    return Console(file=io.StringIO(), width=100)


def test_walkthrough_enabled_env():
    assert walkthrough.walkthrough_enabled({}) is False
    assert walkthrough.walkthrough_enabled({"MGR_WALKTHROUGH": "1"}) is True
    assert walkthrough.walkthrough_enabled({"MGR_WALKTHROUGH": "true"}) is True
    assert walkthrough.walkthrough_enabled({"MGR_WALKTHROUGH": "0"}) is False


def test_vec_formatter_has_no_square_brackets():
    s = walkthrough._vec([1.234, -5.0, 0.0], prec=2)
    assert s == "(1.23, -5.00, 0.00)"
    assert "[" not in s and "]" not in s


def test_every_note_doc_exists():
    for key, note in walkthrough.MECHANISM_NOTES.items():
        target = (REPO_ROOT / "markdown_documentation" / note.doc).resolve()
        assert target.is_file(), f"{key}: missing doc {note.doc_path()} -> {target}"


def test_narrate_demo_all_topics_run():
    for topic in walkthrough.MECHANISM_NOTES:
        res = walkthrough.narrate_demo(topic, console=_silent())
        assert res["ok"] is True, topic


def test_narrate_demo_unknown_topic():
    res = walkthrough.narrate_demo("does-not-exist", console=_silent())
    assert res["ok"] is False


def test_narrate_run_standard_learns():
    res = walkthrough.narrate_run("standard", seed=0, steps=3, console=_silent())
    assert res["n_params"] > 0
    losses = res["losses"]
    assert len(losses) == 3
    assert all(math.isfinite(x) for x in losses)
    # a few AdamW steps at lr=1e-2 reduce the loss on this tiny model
    assert losses[-1] < losses[0]


def test_narrate_run_reversible_geometry():
    # reversible needs n_kv_head <= n_head//2; narrate_run handles the geometry
    res = walkthrough.narrate_run("reversible", seed=0, steps=2, console=_silent())
    assert res["attention_type"] == "reversible"
    assert len(res["losses"]) == 2


def test_demo_hook_symbols_importable():
    # the env-gated hook in the reversible JAX demo imports these
    from nanochat.walkthrough import narrate_demo, walkthrough_enabled

    assert callable(narrate_demo) and callable(walkthrough_enabled)
