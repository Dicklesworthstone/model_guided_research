"""Tests for mgr sample (bead rz8.5).

- Determinism: greedy + fixed seed must produce byte-identical text across
  invocations (the bead's regression-testable acceptance).
- Comparison mode: --compare renders both mechanisms with per-checkpoint
  tokens/s in the JSON contract.
- Separator stop: the same document-separator contract the eval scorer uses
  (kbj2 finding), tested through stubs - no network, no GPT-2.
- CLI argument errors exit 2.

Reuses the tiny-checkpoint helpers from tests/test_eval_tasks.py.
"""

import json

import pytest
from typer.testing import CliRunner

import cli
from tests.test_eval_tasks import (
    _build_golden_checkpoint,
    _StubModel,
    _StubTok,
    _tokenizer_or_skip,
    _train_tiny_checkpoint,
)

runner = CliRunner()

PROMPT = "TASK arith CMP 1.00e-02 2.00e+03 OUT"


def test_sample_greedy_same_seed_byte_identical(tmp_path):
    """Greedy + fixed seed -> identical output across runs. --no-stop-at-separator
    pins the token count so the test cannot pass vacuously on an empty string."""
    _tokenizer_or_skip()
    ckpt = _build_golden_checkpoint(tmp_path)
    payloads = []
    for _ in range(2):
        result = runner.invoke(
            cli.app,
            [
                "sample",
                "--checkpoint",
                str(ckpt),
                "--prompt",
                PROMPT,
                "--max-tokens",
                "16",
                "--no-stop-at-separator",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payloads.append(json.loads(result.output))
    a, b = (p["results"][0] for p in payloads)
    assert a["n_tokens"] == 16 and b["n_tokens"] == 16
    assert a["text"] == b["text"], "greedy same-seed generations must be byte-identical"
    assert a["attention_type"] == "standard" and a["tokens_per_s"] > 0


def test_sample_compare_mechanisms(monkeypatch, tmp_path):
    _tokenizer_or_skip()
    a = _train_tiny_checkpoint(tmp_path, "standard", monkeypatch)
    b = _train_tiny_checkpoint(tmp_path, "ultrametric", monkeypatch)
    result = runner.invoke(
        cli.app,
        [
            "sample",
            "--checkpoint",
            str(a),
            "--compare",
            str(b),
            "--prompt",
            "hello",
            "--max-tokens",
            "8",
            "--no-stop-at-separator",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {r["attention_type"] for r in payload["results"]} == {"standard", "ultrametric"}
    assert all(r["n_tokens"] == 8 and r["tokens_per_s"] > 0 for r in payload["results"])
    assert all(r["peak_memory"] for r in payload["results"])


def test_sample_stream_stops_at_separator():
    class _StopTok(_StubTok):
        BOS = 0xE000

        def get_bos_token_id(self):
            return self.BOS

    class _StopModel(_StubModel):
        def generate(self, tokens, max_tokens, temperature=0.0, seed=0, *, top_k=None):
            yield ord("h")
            yield ord("i")
            yield _StopTok.BOS
            yield ord("X")  # next-doc babble that must never be shown

    rec = cli._sample_stream(
        _StopModel(""),
        _StopTok(),
        [1],
        max_tokens=8,
        temperature=0.0,
        top_k=None,
        seed=0,
        stop_at_separator=True,
    )
    assert rec["text"] == "hi", rec
    assert len(rec["tokens"]) == 2

    rec = cli._sample_stream(
        _StopModel(""),
        _StopTok(),
        [1],
        max_tokens=8,
        temperature=0.0,
        top_k=None,
        seed=0,
        stop_at_separator=False,
    )
    assert chr(_StopTok.BOS) in rec["text"], "separator must pass through when stopping is disabled"


def test_sample_cli_argument_errors(tmp_path):
    result = runner.invoke(cli.app, ["sample", "--prompt", "x"])
    assert result.exit_code == 2  # no checkpoint
    result = runner.invoke(cli.app, ["sample", "--checkpoint", str(tmp_path)])
    assert result.exit_code == 2  # no prompt


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
