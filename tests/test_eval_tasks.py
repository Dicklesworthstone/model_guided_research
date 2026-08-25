"""Tests for mgr eval-tasks (bead vdc.2).

- Metric unit tests against hand-computed values, including the answer-parser
  canonicalization edge cases (leading whitespace, trailing tokens, tokenizer
  merge quirks) driven through stub models/tokenizers - no network, no GPT-2.
- Golden-fixture drift tripwire: a deterministic seeded-RANDOM checkpoint
  (committed as a small score fixture, not megabytes of weights) must
  reproduce its pinned scores exactly; host/torch-pinned with capture-mode
  recapture, mirroring tests/test_attention_core_goldens.py. Any metric
  change must update the fixture deliberately (set MGR_CAPTURE_EVAL_GOLDEN=1)
  with a justification in the commit message.
- End-to-end: train a tiny CPU checkpoint (synthetic loader), eval it through
  the CLI, validate the mgr.evaltasks.v3 schema + artifacts (answer_prior
  floors + generations.jsonl receipts included).
"""

import json
import os
import platform
import socket
from pathlib import Path

import pytest
import torch

from nanochat.diagnostics_data import TASKS

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "eval_tasks_golden.json"
CAPTURE_ENV = "MGR_CAPTURE_EVAL_GOLDEN"
GOLDEN_SEED = 20260610
GOLDEN_TASKS = ("arith", "copyops", "bag")


def _tokenizer_or_skip():
    try:
        from nanochat.tokenizer import get_tokenizer

        return get_tokenizer()
    except Exception as exc:
        pytest.skip(f"tokenizer unavailable: {exc}")


def _host_fingerprint() -> dict[str, str]:
    return {"hostname": socket.gethostname(), "machine": platform.machine(), "torch_version": torch.__version__}


# ---------------------------------------------------------------------------
# answer parsing / canonicalization (stub model + tokenizer; pure unit tests)


class _StubTok:
    """Whitespace 'tokenizer': one token per character group, ids are ords."""

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _StubModel:
    def __init__(self, reply: str, rotary_seq_len: int = 4096):
        self.reply = reply
        self.rotary_seq_len = rotary_seq_len

    def generate(self, tokens, max_tokens, temperature=0.0, seed=0):
        for c in self.reply[:max_tokens]:
            yield ord(c)


def _score(reply: str, doc: str, task: str = "arith"):
    from cli import _eval_score_doc

    spec = TASKS[task]
    return _eval_score_doc(
        _StubModel(reply), _StubTok(), spec, doc, device=torch.device("cpu"), temperature=0.0, seed=0
    )


def _difficulty(task: str, doc: str) -> float:
    difficulty = TASKS[task].difficulty
    assert difficulty is not None, f"{task} task must define a difficulty axis"
    return difficulty(doc)


def test_parser_accepts_exact_answer():
    correct, expected, got = _score(" lt", "TASK arith CMP 1.00e-02 2.00e+03 OUT lt")
    assert correct is True and expected == "lt"


def test_parser_canonicalizes_leading_whitespace_and_trailing_tokens():
    # leading spaces and trailing junk after the answer must not fail a correct answer
    correct, _, _ = _score("   lt TASK garbage", "TASK arith CMP 1.00e-02 2.00e+03 OUT lt")
    assert correct is True


def test_parser_rejects_wrong_answer():
    correct, _, got = _score(" gt", "TASK arith CMP 1.00e-02 2.00e+03 OUT lt")
    assert correct is False and got.split()[0] == "gt"


def test_parser_multiword_answer_requires_full_match():
    doc = "TASK copyops OP REV SEQ a b c OUT c b a"
    assert _score(" c b a extra", doc, task="copyops")[0] is True
    assert _score(" c b", doc, task="copyops")[0] is False  # truncated answer is wrong
    assert _score(" c a b", doc, task="copyops")[0] is False


def test_parser_stops_at_document_separator():
    """Regression for the kbj2 campaign finding: models trained on BOS-packed
    docs emit the answer immediately followed by <|endoftext|>, and decode()
    glues that marker onto the answer with no whitespace - 'lt<|endoftext|>TASK'
    must not fail a correct 'lt'."""

    class _StopTok(_StubTok):
        BOS = 0xE000  # sentinel id the stub model can emit

        def get_bos_token_id(self):
            return self.BOS

    class _StopModel(_StubModel):
        def generate(self, tokens, max_tokens, temperature=0.0, seed=0):
            yielded = 0
            for c in self.reply:
                if yielded >= max_tokens:
                    return
                yield ord(c)
                yielded += 1
            while yielded < max_tokens:  # answer, then separator, then next-doc text
                yield _StopTok.BOS
                yielded += 1

    from cli import _eval_score_doc

    spec = TASKS["arith"]
    out = _eval_score_doc(
        _StopModel(" lt"),
        _StopTok(),
        spec,
        "TASK arith CMP 1.00e-02 2.00e+03 OUT lt",
        device=torch.device("cpu"),
        temperature=0.0,
        seed=0,
    )
    assert out is not None
    correct, expected, got = out
    assert correct is True, f"answer followed by the separator must score correct: got={got!r}"
    assert chr(_StopTok.BOS) not in got, "the separator must be truncated from the decoded answer"


def test_parser_skips_prompts_exceeding_rotary_cache():
    from cli import _eval_score_doc

    spec = TASKS["arith"]
    tiny = _StubModel(" lt", rotary_seq_len=8)
    out = _eval_score_doc(
        tiny,
        _StubTok(),
        spec,
        "TASK arith CMP 1.00e-02 2.00e+03 OUT lt",
        device=torch.device("cpu"),
        temperature=0.0,
        seed=0,
    )
    assert out is None, "over-long prompts must be reported as skipped, not crash"


def test_split_prompt_uses_last_marker_occurrence():
    spec = TASKS["bag"]
    doc = "TASK bag INS a ; QUERY a OUT c1"
    split = spec.split_prompt(doc)
    assert split is not None
    prompt, expected = split
    assert prompt.endswith(" OUT") and expected == "c1"
    assert spec.split_prompt("TASK bag no marker here") is None
    assert TASKS["regime"].split_prompt("TASK regime SEG m2 a1 n1 n3") is None  # LM-only


def test_difficulty_axes_hand_computed():
    assert _difficulty("dyck", "TASK dyck SEQ ( ( ) ) LABEL valid") == 2.0
    assert _difficulty("copyops", "TASK copyops OP REV SEQ a b c OUT c b a") == 3.0
    assert _difficulty("hier", "TASK hier TREE ( a v1 ) PATH a OUT v1") == 1.0
    assert _difficulty("rel", "TASK rel FACT a r b ; FACT b r c QUERY a r r OUT c") == 2.0
    assert _difficulty("arith", "TASK arith CMP 1.00e-05 2.00e+03 OUT lt") == 5.0
    assert _difficulty("bag", "TASK bag INS a ; INS b ; QUERY a OUT c1") == 2.0


# ---------------------------------------------------------------------------
# golden-fixture drift tripwire


def _build_golden_checkpoint(tmp_path: Path) -> Path:
    """Deterministic seeded-RANDOM model (NO init_weights: zero-init lm_head
    would make every logit identical and the golden degenerate). Saved through
    the real checkpoint layout so the loader path is exercised."""
    from nanochat.checkpoint_manager import save_checkpoint
    from nanochat.gpt import GPT, GPTConfig

    torch.manual_seed(GOLDEN_SEED)
    config = GPTConfig(
        sequence_len=64, vocab_size=50304, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, attention_type="standard"
    )
    model = GPT(config)
    ckpt_dir = tmp_path / "golden_ckpt"
    meta = {
        "step": 0,
        "model_config": dict(vars(config)),
        "model_type": "gpt",
        "budget": {"max_steps": 0},
        "lineage": {"run_id": "golden", "parent_run_ids": []},
    }
    save_checkpoint(str(ckpt_dir), 0, model.state_dict(), None, meta, rank=0)
    return ckpt_dir


def _run_golden_eval(tmp_path: Path) -> dict:
    from typer.testing import CliRunner

    import cli as mgr_cli

    ckpt = _build_golden_checkpoint(tmp_path)
    runner = CliRunner()
    args = [
        "eval-tasks",
        "--checkpoint",
        str(ckpt),
        "--examples",
        "8",
        "--seeds",
        "0",
        "--artifacts-dir",
        str(tmp_path / "artifacts"),
        "--run-id",
        "golden",
    ]
    for t in GOLDEN_TASKS:
        args.extend(["--task", t])
    result = runner.invoke(mgr_cli.app, args)
    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "artifacts" / "evals" / "tasks" / "golden" / "summary.json").read_text())
    scores = {}
    for t in GOLDEN_TASKS:
        rec = summary["tasks"][t]
        scores[t] = {
            "em_in": rec["exact_match"]["greedy"]["in_range"]["mean"],
            "em_held": rec["exact_match"]["greedy"]["held_out"]["mean"],
            "ppl_in": rec["perplexity"]["in_range"],
            "ppl_held": rec["perplexity"]["held_out"],
        }
    return scores


def test_golden_fixture_scores_reproduce(tmp_path):
    tok = _tokenizer_or_skip()
    del tok
    capture = os.environ.get(CAPTURE_ENV, "").strip() not in {"", "0"}
    scores = _run_golden_eval(tmp_path)

    if capture:
        GOLDEN_PATH.write_text(
            json.dumps({"host": _host_fingerprint(), "seed": GOLDEN_SEED, "scores": scores}, indent=2, sort_keys=True)
            + "\n"
        )
        return
    if not GOLDEN_PATH.exists():
        pytest.fail(f"Missing golden fixture {GOLDEN_PATH}; capture with {CAPTURE_ENV}=1")
    golden = json.loads(GOLDEN_PATH.read_text())
    if golden["host"] != _host_fingerprint():
        pytest.skip(f"eval golden pinned to {golden['host']}; recapture locally with {CAPTURE_ENV}=1")
    assert scores == golden["scores"], (
        "eval-tasks metrics drifted from the pinned golden. If the change is an intentional metric "
        f"fix, recapture deliberately with {CAPTURE_ENV}=1 and justify in the commit message.\n"
        f"pinned: {golden['scores']}\n     got: {scores}"
    )


# ---------------------------------------------------------------------------
# end-to-end on a trained tiny checkpoint + mechanism coverage


def _train_tiny_checkpoint(tmp_path: Path, attention_type: str, monkeypatch) -> Path:
    import nanochat.train as train_mod

    def fake_loader(B, T, split, device="cpu", resume_state_dict=None, **kwargs):
        gen = torch.Generator().manual_seed(1234)
        idx = 0
        while True:
            t = torch.randint(0, 1000, (B, T + 1), generator=gen, dtype=torch.long)
            yield t[:, :-1].contiguous(), t[:, 1:].contiguous(), {"pq_idx": 0, "rg_idx": idx}
            idx += 1

    monkeypatch.setattr(train_mod, "tokenizing_distributed_data_loader_with_state", fake_loader)
    monkeypatch.setattr(train_mod, "list_parquet_files", lambda data_dir=None: ["a.parquet", "b.parquet"])
    run_id = f"e2e-{attention_type}"
    args = train_mod.build_parser().parse_args(
        [
            "--device",
            "cpu",
            "--max-steps",
            "4",
            "--batch-size",
            "2",
            "--sequence-len",
            "32",
            "--n-layer",
            "1",
            "--n-head",
            "2",
            "--n-kv-head",
            "2",
            "--n-embd",
            "32",
            "--seed",
            "7",
            "--warmup-steps",
            "0",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--run-id",
            run_id,
            "--checkpoint-interval",
            "4",
            "--attention-type",
            attention_type,
        ]
    )
    train_mod.train(args)
    return tmp_path / "artifacts" / "baseline" / "nanochat" / run_id / "checkpoints"


@pytest.mark.parametrize("attention_type", ["standard", "ultrametric", "simplicial", "gauge"])
def test_e2e_trained_checkpoint_evaluates(attention_type, monkeypatch, tmp_path):
    """Mechanism coverage incl. the special-cache mechanisms (ultrametric,
    simplicial) and gauge - includable since 7b0.5 landed KV-cache decode
    (the bead's 'exclude gauge until A5' predates that landing)."""
    tok = _tokenizer_or_skip()
    del tok
    from typer.testing import CliRunner

    import cli as mgr_cli

    ckpt = _train_tiny_checkpoint(tmp_path, attention_type, monkeypatch)
    runner = CliRunner()
    result = runner.invoke(
        mgr_cli.app,
        [
            "eval-tasks",
            "--checkpoint",
            str(ckpt),
            "--task",
            "arith",
            "--examples",
            "4",
            "--seeds",
            "0",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--run-id",
            f"eval-{attention_type}",
        ],
    )
    assert result.exit_code == 0, result.output
    run_dir = tmp_path / "artifacts" / "evals" / "tasks" / f"eval-{attention_type}"
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["schema_version"] == "mgr.evaltasks.v3"
    # dz9i: a real trained checkpoint has a run summary beside it -> the
    # TRAINING provenance must be carried (taintedness mirrors whatever
    # tree state the tiny train ran under - assert shape, not state)
    assert isinstance(summary["train_provenance"], dict)
    assert isinstance(summary["train_provenance"]["tainted"], bool)
    assert summary["meta"]["checkpoint"]["attention_type"] == attention_type
    assert summary["meta"]["receipts"] == "generations.jsonl"
    rec = summary["tasks"]["arith"]
    assert 0.0 <= rec["exact_match"]["greedy"]["in_range"]["mean"] <= 1.0
    assert rec["perplexity"]["in_range"] > 0
    # v2: the recorded floor - what the best constant-answer policy scores on
    # the exact docs scored (the ci-v2 gate's preferred floor)
    prior = rec["answer_prior"]["in_range"]
    assert 0.0 < prior["mean"] <= 1.0
    assert len(prior["per_seed"]) == 1 and isinstance(prior["majority_answer"], str)
    # per-example receipts: every scored example, machine-readable
    receipts = [json.loads(line) for line in (run_dir / "generations.jsonl").read_text().splitlines()]
    assert receipts, "receipts must not be empty"
    assert {"task", "mode", "region", "eval_seed", "doc_index", "expected", "got", "correct"} <= set(receipts[0])
    assert all(r["task"] == "arith" and isinstance(r["correct"], bool) for r in receipts)
    assert (run_dir / "run.md").exists()


def test_e2e_lm_only_task_and_sampled_mode(monkeypatch, tmp_path):
    tok = _tokenizer_or_skip()
    del tok
    from typer.testing import CliRunner

    import cli as mgr_cli

    ckpt = _train_tiny_checkpoint(tmp_path, "standard", monkeypatch)
    runner = CliRunner()
    result = runner.invoke(
        mgr_cli.app,
        [
            "eval-tasks",
            "--checkpoint",
            str(ckpt),
            "--task",
            "regime",
            "--task",
            "dyck",
            "--sampled",
            "--examples",
            "4",
            "--seeds",
            "0,1",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--run-id",
            "eval-mixed",
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "artifacts" / "evals" / "tasks" / "eval-mixed" / "summary.json").read_text())
    regime = summary["tasks"]["regime"]
    assert regime["exact_match"] is None and regime["curve"] is None  # LM-only: perplexity is the metric
    assert regime["answer_prior"] is None  # no answers -> no constant-answer floor
    assert regime["perplexity"]["in_range"] > 0
    dyck = summary["tasks"]["dyck"]
    assert set(dyck["exact_match"]) == {"greedy", "sampled"}
    assert summary["meta"]["decode_modes"] == ["greedy", "sampled"]
    assert len(dyck["exact_match"]["greedy"]["in_range"]["per_seed"]) == 2  # multi-seed aggregation
    assert len(dyck["answer_prior"]["in_range"]["per_seed"]) == 2  # priors aligned with eval seeds


def test_eval_cli_argument_errors(tmp_path):
    from typer.testing import CliRunner

    import cli as mgr_cli

    runner = CliRunner()
    result = runner.invoke(mgr_cli.app, ["eval-tasks", "--checkpoint", str(tmp_path)])
    assert result.exit_code == 2  # neither --task nor --all-tasks
    result = runner.invoke(mgr_cli.app, ["eval-tasks", "--checkpoint", str(tmp_path), "--task", "not-a-task"])
    assert result.exit_code == 2
