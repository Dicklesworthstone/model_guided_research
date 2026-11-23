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
  the CLI, validate the mgr.evaltasks.v5 schema + artifacts (answer_prior
  floors + generations.jsonl receipts included).
"""

import hashlib
import hmac
import json
import os
import platform
import shutil
import socket
import subprocess
from dataclasses import asdict
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
    """Character 'tokenizer': one token per character, ids are ords; carries the
    project tokenizer contract the evaluator relies on (``prepend`` and a BOS id)."""

    BOS = 0xE000  # sentinel id: prepended to prompts, emitted by stub models as the separator

    def encode(self, text, prepend=None):
        ids = [ord(c) for c in text]
        return ([prepend] + ids) if prepend is not None else ids

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def get_bos_token_id(self):
        return self.BOS


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
    correct, expected, got, gamma_min = _score(" lt", "TASK arith CMP 1.00e-02 2.00e+03 OUT lt")
    assert correct is True and expected == "lt"
    assert gamma_min is None  # the stub model records no tropical margins


def test_parser_canonicalizes_leading_whitespace_and_trailing_tokens():
    # leading spaces and trailing junk after the answer must not fail a correct answer
    correct, _, _, _ = _score("   lt TASK garbage", "TASK arith CMP 1.00e-02 2.00e+03 OUT lt")
    assert correct is True


def test_parser_rejects_wrong_answer():
    correct, _, got, _ = _score(" gt", "TASK arith CMP 1.00e-02 2.00e+03 OUT lt")
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
    correct, expected, got, _ = out
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
    assert TASKS["placebo"].split_prompt("TASK placebo x y z") is None  # LM-only
    regime = TASKS["regime"].split_prompt("TASK regime SEG m2 a1 n1 n3 OUT n5 n7")
    assert regime == ("TASK regime SEG m2 a1 n1 n3 OUT", "n5 n7")  # v2: the answer continues the final regime


def test_difficulty_axes_hand_computed():
    assert _difficulty("dyck", "TASK dyck SEQ ( ( ) ) LABEL valid") == 2.0
    assert _difficulty("copyops", "TASK copyops OP REV SEQ a b c OUT c b a") == 3.0
    assert _difficulty("hier", "TASK hier TREE ( a v1 ) PATH a OUT v1") == 1.0
    assert _difficulty("rel", "TASK rel FACT a r b ; FACT b r c QUERY a r r OUT c") == 2.0
    assert _difficulty("arith", "TASK arith CMP 1.00e-05 2.00e+03 OUT lt") == 5.0
    assert _difficulty("bag", "TASK bag INS a ; INS b ; QUERY a OUT c1") == 2.0
    assert _difficulty("regime", "TASK regime SEG m2 a1 n1 n3 SEG m3 a0 n0 n3 OUT n6 n9") == 2.0


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


@pytest.mark.slow
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
@pytest.mark.slow
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
    assert summary["schema_version"] == "mgr.evaltasks.v5"
    assert summary["meta"]["perturbation"] == {
        "perturb_eps": [],
        "perturb_seed": 0,
        "spec": "nanochat.diagnostics_data.apply_embedding_perturbation",
    }
    # dz9i: a real trained checkpoint has a run summary beside it -> the
    # TRAINING provenance must be carried (taintedness mirrors whatever
    # tree state the tiny train ran under - assert shape, not state)
    assert isinstance(summary["train_provenance"], dict)
    assert isinstance(summary["train_provenance"]["tainted"], bool)
    checkpoint = summary["meta"]["checkpoint"]
    assert checkpoint["attention_type"] == attention_type
    assert checkpoint["training_seed"] == 7 and checkpoint["model_overrides"] == {}
    weights = ckpt / f"model_{checkpoint['step']:06d}.pt"
    with weights.open("rb") as handle:
        assert hmac.compare_digest(checkpoint["weights_sha256"], hashlib.file_digest(handle, "sha256").hexdigest())
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

    if attention_type == "standard":
        rounded = runner.invoke(
            mgr_cli.app,
            [
                "eval-tasks",
                "--checkpoint",
                str(ckpt),
                "--task",
                "arith",
                "--examples",
                "1",
                "--seeds",
                "0",
                "--model-override",
                "eval_weight_quant_bits=4",
                "--artifacts-dir",
                str(tmp_path / "artifacts"),
                "--run-id",
                "eval-rounded",
            ],
        )
        assert rounded.exit_code == 0, rounded.output
        rounded_path = run_dir.parent / "eval-rounded" / "summary.json"
        rounded_checkpoint = json.loads(rounded_path.read_text())["meta"]["checkpoint"]
        assert hmac.compare_digest(rounded_checkpoint["weights_sha256"], checkpoint["weights_sha256"])
        assert rounded_checkpoint["training_seed"] == checkpoint["training_seed"]
        assert rounded_checkpoint["model_overrides"] == {"eval_weight_quant_bits": 4}


@pytest.mark.slow
def test_checkpoint_sweeps_adjudicate_from_portable_bound_sources(tmp_path, monkeypatch):
    """Exercise real weights and all three public command stages in clean Git history."""
    from typer.testing import CliRunner

    import cli as mgr_cli
    from nanochat.checkpoint_manager import save_checkpoint
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.report import build_provenance
    from nanochat.tokenizer import HuggingFaceTokenizer

    runner = CliRunner()
    workspace = tmp_path / "repo"
    workspace.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(workspace), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Curve control")
    git("config", "user.email", "test@example.invalid")
    git("config", "commit.gpgsign", "false")
    git("commit", "-qm", "Before curve registration", "--allow-empty")
    before = git("rev-parse", "HEAD")
    tokenizer = HuggingFaceTokenizer.train_from_iterator(
        ["2 + 3 = 5", "4 + 6 = 10", "1 + 1 = 2", "11 + 9 = 20"] * 16, 300
    )
    configs = {
        role: GPTConfig(
            sequence_len=64,
            vocab_size=320,
            n_layer=depth,
            n_head=2,
            n_kv_head=2,
            n_embd=32,
            attention_type=kind,
            ultrametric_hard_digits=kind == "ultrametric",
        )
        for role, kind, depth in (("std", "standard", 1), ("small", "ultrametric", 1), ("deep", "ultrametric", 2))
    }
    knobs = {
        "std": ("eval_weight_quant_bits", 4),
        "small": ("ultrametric_digits_k", 2),
        "deep": ("ultrametric_digits_k", 2),
    }
    planned_flops = 100_000_000.0

    def cohort(mapping):
        roles = {}
        for label, config_name in mapping.items():
            model = asdict(configs[config_name])
            knob, value = knobs[config_name]
            del model[knob], model["ca_init_seed"]
            roles[label] = dict(
                model=model, sweep=[value], planned_flops=planned_flops, optimizer="adamw", scheduler="none"
            )
        design = dict(
            task="arith",
            window_hi=1.0,
            evaluation=dict(seeds=[0], examples_per_seed=2, dials={}, device="cpu"),
            roles=roles,
        )
        return hashlib.sha256(json.dumps(design, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    claims = [
        ("precision", "auc_ratio", cohort({"digit": "small", "float": "std"})),
        ("depth", "digit_depth_auc_absdev", cohort({"shallow": "small", "deep": "deep"})),
    ]
    hypotheses = [
        dict(
            id=f"hyp-curve-{kind}-control",
            statement=f"Verify {kind} reader on tiny checkpoint controls",
            mechanisms=["ultrametric"],
            source=dict(kind="human", provenance="test fixture"),
            date_registered="2026-01-01",
            status="open",
            evidence=[],
            verdict_history=[],
            prediction=dict(
                metric_path=f"bench:results.{metric}",
                comparator=">=",
                threshold=-100.0,
                threshold_kind="absolute_delta",
                baseline=None,
                min_seeds=2,
                candidate_variant={"curve_cohort": coordinate},
            ),
        )
        for kind, metric, coordinate in claims
    ]
    hypotheses.append(
        dict(
            hypotheses[0],
            id="hyp-curve-direct-control",
            statement="Only ordinary evaluations may establish the direct metric",
            mechanisms=["standard"],
            prediction=dict(
                metric_path="evaltasks:tasks.arith.perplexity.in_range",
                comparator=">=",
                threshold=-100.0,
                threshold_kind="absolute_delta",
                baseline=None,
                min_seeds=2,
            ),
        )
    )
    registry = workspace / "hypotheses" / "registry.yaml"
    registry.parent.mkdir()
    registry.write_text(json.dumps(dict(schema_version=1, hypotheses=hypotheses)))
    git("add", "hypotheses/registry.yaml")
    git("commit", "-qm", "Register both curve controls")
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(mgr_cli, "_research_workspace", lambda: workspace)
    artifacts = tmp_path / "original"

    def cli(*args, **options):
        cmd = list(args)
        for name, value in options.items():
            cmd.extend([f"--{name.replace('_', '-')}", str(value)])
        result = runner.invoke(mgr_cli.app, cmd)
        assert result.exit_code == 0, (result.output, result.exception)

    def read_json(path):
        return json.loads(path.read_bytes())

    def save_json(path, value):
        path.write_text(json.dumps(value) + "\n")

    for seed in (10, 11):
        for role, config in configs.items():
            torch.manual_seed(seed)
            model = GPT(config)
            model.init_weights()
            tokens = tokenizer.encode("2 + 3 = 5", prepend=tokenizer.get_bos_token_id())
            sequence = torch.tensor([tokens])
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            model(sequence[:, :-1], targets=sequence[:, 1:]).backward()
            optimizer.step()
            checkpoint = tmp_path / "weights" / f"{role}-{seed}" / "checkpoints"
            checkpoint.mkdir(parents=True)
            tokenizer.save(str(checkpoint / "tokenizer"))
            save_checkpoint(
                str(checkpoint),
                1,
                model.state_dict(),
                None,
                dict(
                    model_config=asdict(config),
                    optimizer_type="adamw",
                    scheduler_type="none",
                    seed=seed,
                    tokenizer={"kind": "task", "dir": "tokenizer"},
                    budget={"target_flops": planned_flops},
                    lineage={"run_id": f"{role}-{seed}", "parent_run_ids": []},
                ),
                rank=0,
            )
            save_json(
                checkpoint.parent / "summary.json", {"provenance": build_provenance({"seed": seed, "role": role})}
            )
            for label, flags in (("full", []), ("point", ["--model-override", f"{knobs[role][0]}={knobs[role][1]}"])):
                cli(
                    "eval-tasks",
                    *flags,
                    checkpoint=checkpoint,
                    task="arith",
                    examples=2,
                    seeds=0,
                    artifacts_dir=artifacts,
                    run_id=f"{role}-{seed}-{label}",
                )
            if role == "std":
                cli(
                    "eval-tasks",
                    checkpoint=checkpoint,
                    task="arith",
                    examples=2,
                    seeds=0,
                    artifacts_dir=tmp_path / "probes" / "sizing",
                    run_id=f"sizing-{seed}",
                )

    def produce(root, kind, seed, run_id, qualified=True):
        roles = (
            (("digit", "small"), ("float", "std")) if kind == "precision" else (("shallow", "small"), ("deep", "deep"))
        )
        inputs = {
            f"{label}_{option}": f"{name}-{seed}-{suffix}"
            for label, name in roles
            for option, suffix in (("full", "full"), ("run", "point"))
        }
        cli(f"{kind}-curve", task="arith", seed=seed, artifacts_dir=root, run_id=run_id, **inputs)
        result = root / "bench" / f"{kind}_curves" / run_id / "summary.json"
        expected = next(hash_value for name, _, hash_value in claims if name == kind)
        actual = read_json(result)["meta"]["curve_cohort"]
        if qualified:
            assert isinstance(actual, str) and hmac.compare_digest(actual, expected)
        else:
            assert actual is None
        return result

    for kind, _, _ in claims:
        for seed in (10, 11):
            result = produce(artifacts, kind, seed, f"{kind}-{seed}")
            if kind == "precision":
                assert "normalized precision" in read_json(result)["meta"]["normalization"]
                assert "normalized precision" in (result.parent / "run.md").read_text()
    portable = tmp_path / "relocated"
    shutil.copytree(artifacts, portable)

    def adjudicate(kind, root, name):
        cli(
            "adjudicate",
            "--dry-run",
            hypothesis=f"hyp-curve-{kind}-control",
            artifacts=root,
            artifacts_dir=tmp_path / "reports",
            run_id=name,
        )
        report = read_json(tmp_path / "reports" / "adjudications" / name / "verdicts.json")
        return report["verdicts"][0]

    for kind, _, _ in claims:
        verdict = adjudicate(kind, portable, kind)
        assert verdict["verdict"] == "supported" and verdict["arms"]["ultrametric"]["n_candidate"] == 2
        assert all(str(portable) in path for path in verdict["artifacts"])

    ordinary = adjudicate("direct", portable, "ordinary-evaluations")
    assert ordinary["verdict"] == "supported" and ordinary["arms"]["standard"]["n_candidate"] == 2
    relocated_sizing = tmp_path / "relocated-sizing"
    shutil.copytree(tmp_path / "probes" / "sizing", relocated_sizing)
    moved = adjudicate("direct", relocated_sizing, "relocated-sizing")
    assert moved["verdict"] == "blocked"
    assert [row["reason_code"] for row in moved["excluded_evidence"]] == ["quarantined_input"] * 2

    # Copies and relabeling may change the output location, but cannot add training units.
    repeats = tmp_path / "repeats"
    shutil.copytree(portable / "evals", repeats / "evals")
    for label in ("one", "copy", "relabeled"):
        produce(repeats, "precision", 10, label)
    assert adjudicate("precision", repeats, "copies")["reason_code"] == "insufficient_seeds"
    # A relabeled seed and run still share the exact loaded weight digests.
    for role in ("small", "std"):
        for suffix in ("full", "point"):
            prefix = repeats / "evals" / "tasks"
            data = read_json(prefix / f"{role}-10-{suffix}" / "summary.json")
            new_id = f"{role}-12-{suffix}"
            data["meta"]["run_id"] = new_id
            data["meta"]["checkpoint"]["training_seed"] = 12
            data["meta"]["checkpoint"]["lineage"]["run_id"] = f"{role}-12"
            dest = prefix / new_id / "summary.json"
            dest.parent.mkdir()
            save_json(dest, data)
    produce(repeats, "precision", 12, "same-weights-new-seed")
    assert adjudicate("precision", repeats, "weight-alias")["reason_code"] == "insufficient_seeds"

    # The producer accepts a new digest for an older but otherwise identical source.
    # Only the adjudicator can check it against the registration in Git.
    for field in ("train_provenance", "provenance"):
        older = tmp_path / field
        shutil.copytree(portable / "evals", older / "evals")
        old_point = older / "evals" / "tasks" / "small-10-point" / "summary.json"
        data = read_json(old_point)
        data[field]["git_sha"] = before
        save_json(old_point, data)
        produce(older, "precision", 10, "pre-registration")
        assert adjudicate("precision", older, field)["excluded_evidence"][0]["reason_code"] == "pre_registration"

    sizing = tmp_path / "sizing"
    shutil.copytree(portable / "evals", sizing / "evals")
    sizing_point = sizing / "evals" / "tasks" / "small-10-point" / "summary.json"
    data = read_json(sizing_point)
    data["meta"]["sizing_input"] = True
    save_json(sizing_point, data)
    produce(sizing, "precision", 10, "quarantined")
    assert (
        adjudicate("precision", sizing, "copied-sizing")["excluded_evidence"][0]["reason_code"] == "quarantined_input"
    )

    missing = tmp_path / "without-evaluations"
    shutil.copytree(portable / "bench", missing / "bench")
    errors = adjudicate("precision", missing, "missing-companions")["excluded_evidence"]
    assert len(errors) == 2 and {row["reason_code"] for row in errors} == {"missing_input_lineage"}

    curve = portable / "bench" / "precision_curves" / "precision-10" / "summary.json"
    legacy = tmp_path / "legacy" / "bench" / "precision_curves" / "old" / "summary.json"
    legacy.parent.mkdir(parents=True)
    data = read_json(curve)
    data["schema_version"] = "mgr.bench.precision_curves.v1"
    save_json(legacy, data)
    assert adjudicate("precision", tmp_path / "legacy", "older-schema")["excluded_evidence"][0]["reason_code"] == (
        "missing_input_lineage"
    )
    data["schema_version"] = "mgr.bench.coord_curves.v1"
    save_json(legacy, data)
    assert adjudicate("precision", tmp_path / "legacy", "other-schema")["excluded_evidence"][0]["reason_code"] == (
        "curve_schema_required"
    )

    invalid = tmp_path / "invalid"
    shutil.copytree(portable / "evals", invalid / "evals")
    invalid_point = invalid / "evals" / "tasks" / "small-10-point" / "summary.json"
    baseline_point = read_json(invalid_point)
    for label, path, value in (
        ("weights", ("meta", "checkpoint", "weights_sha256"), "f" * 64),
        ("seed", ("meta", "checkpoint", "training_seed"), 99),
        ("arm", ("meta", "checkpoint", "attention_type"), "standard"),
        ("depth", ("meta", "checkpoint", "model_config", "n_layer"), 7),
        ("extra-override", ("meta", "checkpoint", "model_overrides"), {"ultrametric_digits_k": 2, "n_layer": 1}),
        ("split", ("tasks", "arith", "perplexity_input_sha256"), "b" * 64),
        ("eval-seed", ("meta", "seeds"), [3]),
    ):
        data = json.loads(json.dumps(baseline_point))
        node = data
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        save_json(invalid_point, data)
        produce(invalid, "precision", 10, label, qualified=False)
    forged = invalid / "bench" / "precision_curves" / "eval-seed" / "summary.json"
    data = read_json(forged)
    data["meta"]["curve_cohort"] = claims[0][2]
    save_json(forged, data)
    assert adjudicate("precision", invalid, "forged-cohort")["excluded_evidence"][0]["reason_code"] == (
        "invalid_curve_protocol"
    )

    data = read_json(curve)
    data["results"]["auc_ratio"] += 0.25
    save_json(curve, data)
    assert adjudicate("precision", portable, "forged")["excluded_evidence"][0]["reason_code"] == "curve_result_mismatch"

    control_point = portable / "evals" / "tasks" / "small-11-point" / "summary.json"
    data = read_json(control_point)
    data["tasks"]["arith"]["perplexity"]["in_range"] += 0.25
    save_json(control_point, data)
    reasons = {row["reason_code"] for row in adjudicate("precision", portable, "changed-input")["excluded_evidence"]}
    assert reasons == {"curve_result_mismatch", "curve_input_digest_mismatch"}


@pytest.mark.slow
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
    # regime v2 scores the OUT continuation: the registered metric for the
    # ordinal/hoss regime hypotheses (exact_match.greedy.held_out.mean) exists
    assert set(regime["exact_match"]) == {"greedy", "sampled"}
    assert regime["exact_match"]["greedy"]["held_out"]["mean"] is not None
    assert regime["answer_prior"]["held_out"]["mean"] is not None  # constant-answer floor recorded
    assert regime["difficulty_axis"] == "shift_count" and regime["curve"] is not None
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


def test_perturbation_mode_is_identity_at_zero_and_degrades_at_large_eps(monkeypatch, tmp_path):
    """v5 (bead jida.21): --perturb-eps 0 is byte-identical to the default
    run; a large eps lowers (or at worst ties) exact match and is recorded in
    meta.perturbation, so a robustness ladder's runs are distinct arms for the
    engine's selectors. A standard checkpoint records no certified fraction."""
    _tokenizer_or_skip()
    from typer.testing import CliRunner

    import cli as mgr_cli

    ckpt = _train_tiny_checkpoint(tmp_path, "standard", monkeypatch)
    runner = CliRunner()

    def run(run_id: str, *extra: str) -> dict:
        result = runner.invoke(
            mgr_cli.app,
            [
                "eval-tasks",
                "--checkpoint",
                str(ckpt),
                "--task",
                "arith",
                "--examples",
                "6",
                "--seeds",
                "0",
                "--artifacts-dir",
                str(tmp_path / "artifacts"),
                "--run-id",
                run_id,
                *extra,
            ],
        )
        assert result.exit_code == 0, result.output
        return json.loads((tmp_path / "artifacts" / "evals" / "tasks" / run_id / "summary.json").read_text())

    plain = run("plain")
    ladder = run("ladder", "--perturb-eps", "0.001,25.0", "--perturb-seed", "3")
    # the unperturbed score is the same field in both runs, ladder or not
    assert plain["tasks"]["arith"]["exact_match"] == ladder["tasks"]["arith"]["exact_match"]
    assert plain["meta"]["perturbation"]["perturb_eps"] == []
    assert plain["tasks"]["arith"]["robustness"] == {"perturb_seed": 0, "ladder": {}}
    assert ladder["meta"]["perturbation"] == {
        "perturb_eps": [0.001, 25.0],
        "perturb_seed": 3,
        "spec": "nanochat.diagnostics_data.apply_embedding_perturbation",
    }
    rungs = ladder["tasks"]["arith"]["robustness"]["ladder"]
    assert set(rungs) == {"e0p001", "e25"}
    em_plain = plain["tasks"]["arith"]["exact_match"]["greedy"]["in_range"]["mean"]
    small = rungs["e0p001"]
    big = rungs["e25"]
    assert small["eps"] == 0.001 and big["eps"] == 25.0
    # a 1e-3 sup-norm nudge of unit-RMS embeddings leaves greedy answers alone;
    # a sup-norm 25 perturbation cannot help
    assert small["exact_match"]["in_range"]["mean"] == pytest.approx(em_plain)
    assert small["degradation"]["in_range"]["mean"] == pytest.approx(0.0)
    assert big["exact_match"]["in_range"]["mean"] <= em_plain
    assert big["degradation"]["in_range"]["mean"] == pytest.approx(em_plain - big["exact_match"]["in_range"]["mean"])
    # standard attention records no margins: no certificate, no violation rate
    assert big["certified_fraction"] == {"in_range": None, "held_out": None}
    assert big["certificate_violation_rate"] == {"in_range": None, "held_out": None}
    bad = runner.invoke(
        mgr_cli.app,
        [
            "eval-tasks",
            "--checkpoint",
            str(ckpt),
            "--task",
            "arith",
            "--perturb-eps",
            "0,0.1",
            "--artifacts-dir",
            str(tmp_path / "x"),
        ],
    )
    assert bad.exit_code == 2 and "must be > 0" in bad.output


def test_tropical_margins_vs_length_recorded_for_tropical_checkpoints_only(monkeypatch, tmp_path):
    """v5 (bead jida.24): with margins on (an eval-time model override or a
    checkpoint trained with them) a tropical checkpoint's summary carries the
    mean routing margin per prompt length and the slope of margin vs ln T over
    documents; standard attention records nothing. The group task has several
    prompt lengths, which is what makes the fit defined."""
    _tokenizer_or_skip()
    from typer.testing import CliRunner

    import cli as mgr_cli

    runner = CliRunner()

    def run(attention_type: str, run_id: str, *extra: str) -> dict:
        ckpt = _train_tiny_checkpoint(tmp_path / attention_type, attention_type, monkeypatch)
        result = runner.invoke(
            mgr_cli.app,
            [
                "eval-tasks",
                "--checkpoint",
                str(ckpt),
                "--task",
                "group",
                "--examples",
                "8",
                "--seeds",
                "0",
                "--artifacts-dir",
                str(tmp_path / "artifacts"),
                "--run-id",
                run_id,
                *extra,
            ],
        )
        assert result.exit_code == 0, result.output
        return json.loads((tmp_path / "artifacts" / "evals" / "tasks" / run_id / "summary.json").read_text())

    trop = run("tropical", "trop-margins", "--model-override", "tropical_record_margins=true")
    block = trop["tasks"]["group"]["tropical_margins"]
    assert block is not None, "a tropical checkpoint with margins on must record the length-axis block"
    assert len(block["gamma_by_length"]) >= 2
    fit = block["gamma_vs_lnT"]
    assert fit["n_docs"] >= 2 and fit["ci95"] is not None and len(fit["ci95"]) == 2
    assert all(isinstance(v, float) for v in (fit["slope"], fit["intercept"]))
    assert trop["meta"]["checkpoint"]["model_config"]["tropical_record_margins"] is True  # the override is recorded
    std = run("standard", "std-margins")
    assert std["tasks"]["group"]["tropical_margins"] is None


def test_adj_collect_artifacts_indexes_current_eval_tasks_schema(tmp_path):
    """Guard against drift between _EVAL_TASKS_SCHEMA_VERSION and _adj_collect_artifacts."""
    import cli as mgr_cli

    run_dir = tmp_path / "evals" / "tasks" / "eval-v5"
    run_dir.mkdir(parents=True)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": mgr_cli._EVAL_TASKS_SCHEMA_VERSION,
                "kind": "eval-tasks",
                "provenance": {"tainted": False},
                "train_provenance": {"tainted": False},
                "meta": {"checkpoint": {"attention_type": "standard", "budget": {"target_flops": 1e6}}},
                "tasks": {},
            }
        )
    )
    indexed = mgr_cli._adj_collect_artifacts([tmp_path])
    assert len(indexed) == 1
    assert indexed[0]["schema"] == "evaltasks"
    assert indexed[0]["tainted"] is False


def test_adjusted_rand_index_hand_computed_fixtures() -> None:
    """Verify compute_adjusted_rand_index on hand-computed fixtures."""
    import cli as mgr_cli

    # Perfect match -> 1.0
    assert abs(mgr_cli.compute_adjusted_rand_index([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]) - 1.0) < 1e-9
    # Label permutation invariance -> 1.0
    assert abs(mgr_cli.compute_adjusted_rand_index([0, 0, 1, 1, 2, 2], [9, 9, 3, 3, 7, 7]) - 1.0) < 1e-9
    # Orthogonal balanced 2x2 split on 4 items:
    # A = [0, 0, 1, 1], B = [0, 1, 0, 1] -> n_ij all 1 -> sum C(n_ij,2)=0, a_i=2, b_j=2 -> ARI = -0.5
    ari_orth = mgr_cli.compute_adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1])
    assert abs(ari_orth - (-0.5)) < 1e-6
    # Single cluster vs single cluster -> 1.0
    assert mgr_cli.compute_adjusted_rand_index([5, 5, 5], [1, 1, 1]) == 1.0


def test_extract_route_trace_determinism_and_backend_conformance() -> None:
    """Verify extract_route_trace determinism, brute-force cross-check, and per-backend conformance."""
    import cli as mgr_cli
    from nanochat.diagnostics_data import (
        ROBUSTNESS_PROBE_DEFAULT_EPS,
        ROBUSTNESS_PROBE_EPS_LADDER,
        ROBUSTNESS_PROBE_SPEC_VERSION,
    )
    from nanochat.gpt import GPT, GPTConfig

    assert ROBUSTNESS_PROBE_SPEC_VERSION == "vdc.1-linf-embedding-v1"
    assert ROBUSTNESS_PROBE_DEFAULT_EPS in ROBUSTNESS_PROBE_EPS_LADDER

    torch.manual_seed(123)
    tokens = torch.randint(0, 16, (1, 12), dtype=torch.long)

    # 1. Traceable mechanisms: tropical, ultrametric, braid, fractal
    for mech in ("tropical", "ultrametric", "braid", "fractal"):
        cfg = GPTConfig(
            sequence_len=16,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_kv_head=2,
            n_embd=16,
            attention_type=mech,
        )
        model = GPT(cfg).eval()
        gen_a = torch.Generator(device="cpu").manual_seed(42)
        trace_a = mgr_cli.extract_route_trace(
            model,
            tokens,
            mechanism=mech,
            perturb=(ROBUSTNESS_PROBE_DEFAULT_EPS, gen_a),
        )
        gen_b = torch.Generator(device="cpu").manual_seed(42)
        trace_b = mgr_cli.extract_route_trace(
            model,
            tokens,
            mechanism=mech,
            perturb=(ROBUSTNESS_PROBE_DEFAULT_EPS, gen_b),
        )
        assert trace_a["traceable_layers"] == 1
        assert trace_a["layers"][0]["status"] == "ok", f"Expected status='ok' for {mech}, got {trace_a}"
        # Bitwise-identical determinism check
        assert json.dumps(trace_a, sort_keys=True) == json.dumps(trace_b, sort_keys=True)
        assert len(trace_a["decisions"]) == 2 * 12
        if mech == "tropical":
            assert trace_a["bruteforce_verified"] is True

    # 2. Non-traceable mechanisms emit status="skipped" with explicit reason (never silently pass)
    std_cfg = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        attention_type="standard",
    )
    std_model = GPT(std_cfg).eval()
    std_trace = mgr_cli.extract_route_trace(std_model, tokens, mechanism="standard")
    assert std_trace["skipped_layers"] == 1
    assert std_trace["layers"][0]["status"] == "skipped"
    assert "discrete argmax route" in std_trace["layers"][0]["reason"].lower()


def test_route_controls_do_not_recover_labels_or_construct_a_margin_correlation():
    import random

    import cli

    spec = TASKS["hier"]
    docs = spec.generate(96, 42, spec.resolve_dials(None))["train"][:32]
    traces = []
    for i, doc in enumerate(docs):
        label, features = cli._extract_doc_latent_label_and_trace_features(spec, doc, {"decisions": [0]})
        traces.append(
            {
                "decisions": [0],
                "latent_label": label,
                "trace_cluster_features": features,
                "prompt_control_features": cli._route_prompt_control(spec, doc),
                "attention_entropy_vector": [0.0],
                "min_margin": i,
                "scored_draws": len(docs),
                "surviving_draws": i,
            }
        )
    store = {"schema_version": cli.ROUTE_STORE_SCHEMA_VERSION, "traces": traces, "meta": {"seed": 42}}
    results = cli.analyze_route_trace_store(store)["results"]
    cl = results["clustering"]
    assert cl["ari_trace"] == cl["ari_entropy_baseline"] == cl["ari_gain"] == 0
    assert cl["ari_prompt_only_control"] == 1
    assert results["stability"]["spearman_margin_vs_stability"] == pytest.approx(1)
    labels = [tr["latent_label"] for tr in traces]
    random.Random(42).shuffle(labels)
    for i, tr in enumerate(traces):
        tr["latent_label"] = labels[i]
        tr["surviving_draws"] = len(docs) - i
    reversed_results = cli.analyze_route_trace_store(store)["results"]
    assert reversed_results["stability"]["spearman_margin_vs_stability"] == pytest.approx(-1)
    assert reversed_results["clustering"]["ari_trace"] == 0
    assert reversed_results["clustering"]["ari_prompt_only_control"] < 0.15
    for tr in traces:
        tr["min_margin"] = 1
        tr["latent_label"] = 0
    degenerate = cli.analyze_route_trace_store(store)["results"]
    assert degenerate["stability"]["spearman_margin_vs_stability"] is None
    assert degenerate["clustering"]["ari_gain"] is None
    assert degenerate["stability"]["mean_paraphrase_stability"] is None
    empty = cli.analyze_route_trace_store({**store, "traces": []})["results"]
    assert empty["clustering"]["ari_trace"] is None and not empty["bruteforce_verified"]
    assert cli._route_agreement([1, 2], [1, 2, 3]) == (2, 3)
    assert cli._route_agreement([1, 2, 3], [1]) == (1, 3)
    assert cli._route_agreement([], []) == (0, 0)
    with pytest.raises(ValueError, match="v2"):
        cli.analyze_route_trace_store({**store, "schema_version": "mgr.routes.store.v1"})


def test_routes_cli_and_from_store_regeneration(monkeypatch, tmp_path: Path) -> None:
    """Verify mgr routes writes traces.parquet/json + summary.json + report.md and regenerates via --from-store."""
    _tokenizer_or_skip()
    from typer.testing import CliRunner

    import cli as mgr_cli

    ckpt_dir = _train_tiny_checkpoint(tmp_path, "tropical", monkeypatch)

    out_dir = tmp_path / "routes_out"
    runner = CliRunner()
    res = runner.invoke(
        mgr_cli.app,
        [
            "routes",
            "--checkpoint",
            str(ckpt_dir),
            "--task",
            "dyck",
            "--sample",
            "8",
            "--seed",
            "42",
            "--out",
            str(out_dir),
            "--no-region-count",
        ],
    )
    assert res.exit_code == 0, res.output
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"
    store_json_path = out_dir / "traces.json"
    assert summary_path.exists()
    assert report_path.exists()
    assert store_json_path.exists()

    summary_orig = json.loads(summary_path.read_text())
    assert summary_orig["schema_version"] == "mgr.bench.routes.v2"
    assert summary_orig["results"]["bruteforce_verified"] is True
    assert "ari_trace" in summary_orig["results"]["clustering"]
    assert "spearman_margin_vs_stability" in summary_orig["results"]["stability"]

    # Regenerate summary.json and report.md from traces.json alone in a fresh directory
    regen_dir = tmp_path / "routes_regen"
    res_regen = runner.invoke(
        mgr_cli.app,
        [
            "routes",
            "--from-store",
            str(store_json_path),
            "--out",
            str(regen_dir),
        ],
    )
    assert res_regen.exit_code == 0, res_regen.output
    summary_regen = json.loads((regen_dir / "summary.json").read_text())
    assert summary_regen["results"] == summary_orig["results"]
    assert (regen_dir / "report.md").exists()


def test_tropical_region_count_propositions_and_structured_support() -> None:
    """Verify Propositions 0, 1, 2, ReLU flat baseline, and tropical_structured_support parameter parity."""
    import cli as mgr_cli
    from nanochat.gpt import GPTConfig
    from nanochat.tropical_attention_torch import (
        TropicalCausalSelfAttention,
        count_tropical_realized_regions,
        cyclic_polytope_support_lift,
    )

    # 1. Verify Propositions 0, 1, 2 and positive T-slope via run_tropical_region_count_protocol
    proto = mgr_cli.run_tropical_region_count_protocol(
        checkpoint=None,
        seq_lengths=(4, 8, 16, 32),
        n_samples=64,
        seed=42,
        device_str="cpu",
    )
    assert proto["schema_version"] == "mgr.bench.tropical_regions.v2"
    assert proto["meta"]["curve_weight_source"] == "generated"
    assert proto["results"]["support_arm_bound_violations"] == {"dense": 0, "structured": 0}
    assert proto["results"]["total_bound_violations"] == 0
    assert proto["results"]["tropical_regions_per_param_slope"] > 0.0
    assert proto["results"]["relu_mlp_regions_per_param_slope"] == 0.0
    assert proto["results"]["efficiency_slope_gap"] > 0.0
    assert proto["results"]["structured_vs_dense_multi_head_gain"] > 0

    # 2. Verify cyclic_polytope_support_lift & tropical_structured_support parameter parity and exact bounds
    lift, scale = cyclic_polytope_support_lift(2, 8, device=torch.device("cpu"), dtype=torch.float32)
    assert lift.shape == (1, 2, 1, 8)
    assert scale.shape == (1, 2, 1, 8)

    cfg_dense = GPTConfig(
        sequence_len=32,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        attention_type="tropical",
        tropical_structured_support=False,
    )
    cfg_struct = GPTConfig(
        sequence_len=32,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        attention_type="tropical",
        tropical_structured_support=True,
    )
    torch.manual_seed(99)
    attn_dense = TropicalCausalSelfAttention(cfg_dense, layer_idx=0).eval()
    attn_struct = TropicalCausalSelfAttention(cfg_struct, layer_idx=0).eval()
    attn_struct.load_state_dict(attn_dense.state_dict())

    # Parameter counts must be identical
    dense_params = sum(p.numel() for p in attn_dense.parameters())
    struct_params = sum(p.numel() for p in attn_struct.parameters())
    assert dense_params == struct_params

    # Both must satisfy Propositions 0, 1, and 2 with zero bound violations
    x_samples = torch.randn(48, 12, 16)
    res_dense = count_tropical_realized_regions(attn_dense, x_samples)
    res_struct = count_tropical_realized_regions(attn_struct, x_samples)
    assert res_dense["bound_violations"] == 0
    assert res_struct["bound_violations"] == 0
    assert res_struct["realized_q_space_fixed_kv"] <= res_struct["prop0_bound"]
    assert res_struct["realized_single_head"] <= res_struct["prop1_bound"]
    assert res_struct["realized_multi_head"] <= res_struct["prop2_bound"]


@pytest.mark.slow
def test_routes_early_checkpoint_uses_saved_training_init_and_replays_parquet(tmp_path: Path) -> None:
    import hashlib
    import hmac

    from typer.testing import CliRunner

    import cli
    from nanochat.diagnostics_data import generate_task
    from nanochat.train import build_parser, train

    dataset = tmp_path / "data"
    generate_task("hier", out_dir=dataset, size=96, seed=99)
    train(
        build_parser().parse_args(
            [
                "--device",
                "cpu",
                "--tokenizer",
                "task",
                "--data-dir",
                str(dataset / "hier"),
                "--attention-type",
                "tropical",
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
                "16",
                "--max-steps",
                "6",
                "--warmup-steps",
                "0",
                "--seed",
                "11",
                "--run-id",
                "native-routes",
                "--checkpoint-interval",
                "2",
                "--save-initial-weights",
                "--artifacts-dir",
                str(tmp_path),
            ]
        )
    )
    checkpoints = tmp_path / "baseline" / "nanochat" / "native-routes" / "checkpoints"
    assert sorted(p.name for p in checkpoints.glob("meta_*.json")) == [
        "meta_000001.json",
        "meta_000003.json",
        "meta_000005.json",
    ]

    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    output = tmp_path / "early"
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "routes",
            "--checkpoint",
            str(checkpoints),
            "--step",
            "1",
            "--task",
            "hier",
            "--sample",
            "4",
            "--seed",
            "42",
            "--no-region-count",
            "--out",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads((output / "summary.json").read_text())
    assert summary["meta"]["training_seed"] == 11
    assert summary["meta"]["lineage"]["run_id"] == "native-routes"
    assert hmac.compare_digest(summary["meta"]["weights_sha256"], digest(checkpoints / "model_000001.pt"))
    assert hmac.compare_digest(
        summary["meta"]["initial_weights_sha256"], digest(checkpoints / "initial" / "model_000000.pt")
    )
    assert [row["step"] for row in summary["results"]["reuse"]["checkpoint_curve"]] == [-1, 1]
    store = json.loads((output / "traces.json").read_text())
    stability = summary["results"]["stability"]
    assert stability["perturbation_draws_scored"] == sum(t["scored_draws"] for t in store["traces"]) > 0
    assert stability["perturbation_draws_survived"] == sum(t["surviving_draws"] for t in store["traces"])
    replay = tmp_path / "replayed"
    result = runner.invoke(cli.app, ["routes", "--from-store", str(output / "traces.parquet"), "--out", str(replay)])
    assert result.exit_code == 0, result.output
    assert (replay / "summary.json").read_bytes() == (output / "summary.json").read_bytes()
    assert (replay / "report.md").read_bytes() == (output / "report.md").read_bytes()
    # A store replay must enforce the same selected endpoint as extraction.
    store["checkpoint_series"].append(
        {
            **store["checkpoint_series"][-1],
            "step": 3,
            "weights_sha256": digest(checkpoints / "model_000003.pt"),
        }
    )
    future = tmp_path / "future.json"
    future.write_text(json.dumps(store))
    result = runner.invoke(cli.app, ["routes", "--from-store", str(future), "--out", str(tmp_path / "invalid")])
    assert result.exit_code != 0
