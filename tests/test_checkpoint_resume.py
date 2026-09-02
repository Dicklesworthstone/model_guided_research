"""Checkpoint/resume acceptance suite for nanochat.train (bead rz8.1).

The centerpiece is the interruption test: a run checkpointed mid-flight and
resumed must reproduce the uninterrupted run's loss trajectory BITWISE on CPU
fp32. This is achievable because every state component round-trips exactly:
model weights, optimizer momenta (AdamW + Muon), ordinal-scheduler counters,
the data position (exact replay fast-forward), and all RNG streams.

The parquet dataloader is replaced by a deterministic synthetic token stream
(seeded independently of consumption count), so the suite runs in CI without
the fineweb shards while still exercising the trainer's full state machinery
including the with-state loader contract and the fast-forward path.
"""

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest
import torch

import nanochat.train as train_mod

VOCAB_FAKE = 1000  # < GPTConfig default vocab_size, so targets are always valid


def _fake_loader_factory(record: list[str] | None = None):
    """A deterministic stand-in for tokenizing_distributed_data_loader_with_state.

    Yields the SAME batch sequence on every instantiation (generator seeded by
    a constant, independent of global RNG), mirroring the real loader's
    determinism. Honors the resume_state_dict contract approximately the way
    the real loader does: resumes from the NEXT index after the recorded one.
    """

    def fake_loader(B, T, split, device="cpu", resume_state_dict=None, **kwargs):
        gen = torch.Generator().manual_seed(1234)
        idx = 0
        skip_to = None
        if resume_state_dict is not None:
            skip_to = int(resume_state_dict["rg_idx"]) + 1
        while True:
            tokens = torch.randint(0, VOCAB_FAKE, (B, T + 1), generator=gen, dtype=torch.long)
            state = {"pq_idx": 0, "rg_idx": idx}
            if skip_to is None or idx >= skip_to:
                if record is not None:
                    record.append(hashlib.sha256(tokens.numpy().tobytes()).hexdigest())
                yield tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous(), state
            idx += 1

    return fake_loader


def _train_args(tmp_path: Path, run_id: str, **overrides) -> Namespace:
    argv = [
        "--device",
        "cpu",
        "--max-steps",
        "12",
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
    ]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if value is None:
            argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return train_mod.build_parser().parse_args(argv)


def _run_train(
    monkeypatch,
    tmp_path: Path,
    run_id: str,
    record: list[str] | None = None,
    **overrides,
) -> dict[str, Any]:
    monkeypatch.setattr(train_mod, "tokenizing_distributed_data_loader_with_state", _fake_loader_factory(record))
    monkeypatch.setattr(
        train_mod, "list_parquet_files", lambda data_dir=None: ["fake_train.parquet", "fake_val.parquet"]
    )
    train_mod.train(_train_args(tmp_path, run_id, **overrides))
    summary_path = tmp_path / "artifacts" / "baseline" / "nanochat" / run_id / "summary.json"
    assert summary_path.exists(), f"missing summary at {summary_path}"
    summary: dict = json.loads(summary_path.read_text())
    return summary


def _ckpt_dir(tmp_path: Path, run_id: str) -> Path:
    return tmp_path / "artifacts" / "baseline" / "nanochat" / run_id / "checkpoints"


def _assert_bitwise_trajectory(resumed: list[float], reference: list[float], *, offset: int) -> None:
    """Bead requirement: on mismatch, pinpoint the FIRST divergent step with both values."""
    assert len(resumed) == len(reference), f"trajectory length {len(resumed)} != reference {len(reference)}"
    for i, (a, b) in enumerate(zip(resumed, reference)):
        if a != b:
            pytest.fail(
                f"resume diverged at step {offset + i}: resumed loss {a!r} != uninterrupted loss {b!r} "
                f"(first divergence; {sum(1 for x, y in zip(resumed, reference) if x != y)} steps differ in total)"
            )


def test_mixed_attention_schedule_train_artifacts(monkeypatch, tmp_path):
    """7b0.2: tiny CPU training run records the resolved heterogeneous stack."""
    summary = _run_train(
        monkeypatch,
        tmp_path,
        "schedule-smoke",
        max_steps=2,
        n_layer=2,
        attention_schedule="standard,tropical",
    )
    run_dir = tmp_path / "artifacts" / "baseline" / "nanochat" / "schedule-smoke"
    assert summary["config"]["attention_type"] == "standard,tropical"
    assert summary["config"]["resolved_attention_schedule"] == ["standard", "tropical"]
    assert summary["config"]["attention_schedule_label"] == "standard,tropical"
    assert len(summary["results"]["losses"]) == 2
    assert "Attention schedule: `standard,tropical`" in (run_dir / "run.md").read_text()

    metrics_lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert metrics_lines, "metrics.jsonl must be present and parseable"
    assert all(json.loads(line) for line in metrics_lines)


@pytest.mark.slow
def test_interrupted_run_resumes_bitwise(monkeypatch, tmp_path):
    """THE acceptance test: checkpoint at step 5, resume, steps 6-11 bitwise-match
    the uninterrupted run on CPU fp32 (AdamW + Muon both round-trip)."""
    hashes_a: list[str] = []
    summary_a = _run_train(monkeypatch, tmp_path, "uninterrupted", record=hashes_a)
    losses_a = summary_a["results"]["losses"]
    assert len(losses_a) == 12

    hashes_b1: list[str] = []
    summary_b1 = _run_train(monkeypatch, tmp_path, "parent", record=hashes_b1, checkpoint_interval=6)
    assert summary_b1["checkpointing"]["saved_steps"] == [5, 11]
    # Checkpointing must not perturb the trajectory itself.
    _assert_bitwise_trajectory(summary_b1["results"]["losses"], losses_a, offset=0)

    hashes_b2: list[str] = []
    summary_b2 = _run_train(
        monkeypatch,
        tmp_path,
        "resumed",
        record=hashes_b2,
        resume_from=str(_ckpt_dir(tmp_path, "parent")),
        resume_step=5,
    )
    losses_b2 = summary_b2["results"]["losses"]
    assert summary_b2["results"]["start_step"] == 6
    assert len(losses_b2) == 6
    _assert_bitwise_trajectory(losses_b2, losses_a[6:], offset=6)

    # Batch-hash assertion (no repeated/skipped batches): the resumed run's
    # full yielded stream (fast-forward replays 0..5, then trains 6..11)
    # must equal the uninterrupted run's stream batch-for-batch.
    assert hashes_b2 == hashes_a, "resumed run consumed a different batch sequence than the uninterrupted run"

    # Resume lineage recorded.
    assert summary_b2["resume"]["parent_run_ids"] == ["parent"]
    assert summary_b2["resume"]["resume_step"] == 6


@pytest.mark.slow
def test_resume_restores_original_budget(monkeypatch, tmp_path):
    """A resumed run honors the ORIGINAL budget even if the resume command line disagrees."""
    _run_train(monkeypatch, tmp_path, "budget-parent", checkpoint_interval=4, max_steps=8)
    summary = _run_train(
        monkeypatch,
        tmp_path,
        "budget-resumed",
        resume_from=str(_ckpt_dir(tmp_path, "budget-parent")),
        resume_step=3,
        max_steps=999,  # must be overridden by the checkpoint's recorded budget
    )
    assert summary["budget"]["max_steps"] == 8
    assert summary["results"]["start_step"] == 4
    assert len(summary["results"]["losses"]) == 4


@pytest.mark.slow
def test_resume_with_ordinal_scheduler_round_trips_counters(monkeypatch, tmp_path):
    """Scheduler counters survive the round trip and the trajectory stays bitwise.

    Also covers the orphan fix: the ordinal scheduler is now actually stepped
    (P_init patience consumes on non-improving steps), so its state is
    no longer frozen at construction values.
    """
    summary_a = _run_train(monkeypatch, tmp_path, "sched-uninterrupted", scheduler_type="ordinal")
    summary_b1 = _run_train(monkeypatch, tmp_path, "sched-parent", scheduler_type="ordinal", checkpoint_interval=6)
    state = torch.load(_ckpt_dir(tmp_path, "sched-parent") / "optim_000005_rank0.pt", weights_only=False)
    assert len(state["schedulers"]) == len(state["optimizers"]) > 0
    sched_state = state["schedulers"][0]
    assert sched_state["ema_loss"] is not None, "scheduler was never stepped - the orphan bug is back"
    summary_b2 = _run_train(
        monkeypatch,
        tmp_path,
        "sched-resumed",
        scheduler_type="ordinal",
        resume_from=str(_ckpt_dir(tmp_path, "sched-parent")),
        resume_step=5,
    )
    _assert_bitwise_trajectory(summary_b2["results"]["losses"], summary_a["results"]["losses"][6:], offset=6)
    assert summary_b1["checkpointing"]["saved_steps"] == [5, 11]


def test_checkpoint_retention_prunes_oldest(monkeypatch, tmp_path):
    summary = _run_train(monkeypatch, tmp_path, "retention", checkpoint_interval=2, checkpoint_keep=2, max_steps=8)
    ckpt = _ckpt_dir(tmp_path, "retention")
    # Saves at steps 1,3,5,7; keep=2 retains only the newest two.
    assert summary["checkpointing"]["saved_steps"] == [5, 7]
    present = sorted(p.name for p in ckpt.glob("model_*.pt"))
    assert present == ["model_000005.pt", "model_000007.pt"]
    for step in (5, 7):
        assert (ckpt / f"optim_{step:06d}_rank0.pt").exists()
        assert (ckpt / f"meta_{step:06d}.json").exists()
    for step in (1, 3):
        assert not (ckpt / f"meta_{step:06d}.json").exists()


def test_checkpoint_verify_happy_path(monkeypatch, tmp_path):
    summary = _run_train(monkeypatch, tmp_path, "verified", checkpoint_interval=6, checkpoint_verify=None)
    assert summary["checkpointing"]["verify"] is True
    assert summary["checkpointing"]["saved_steps"] == [5, 11]


@pytest.mark.slow
def test_resume_config_mismatch_is_actionable(monkeypatch, tmp_path):
    _run_train(monkeypatch, tmp_path, "mismatch-parent", checkpoint_interval=6)
    with pytest.raises(ValueError, match="n_embd"):
        _run_train(
            monkeypatch,
            tmp_path,
            "mismatch-resumed",
            resume_from=str(_ckpt_dir(tmp_path, "mismatch-parent")),
            n_embd=64,  # checkpoint was trained with 32
        )


@pytest.mark.slow
def test_resume_approximate_mode_passes_loader_state(monkeypatch, tmp_path):
    """Approximate mode hands the recorded (pq_idx, rg_idx) to the loader's
    native resume instead of fast-forwarding; trajectories need not be bitwise."""
    _run_train(monkeypatch, tmp_path, "approx-parent", checkpoint_interval=6)
    summary = _run_train(
        monkeypatch,
        tmp_path,
        "approx-resumed",
        resume_from=str(_ckpt_dir(tmp_path, "approx-parent")),
        resume_step=5,
        resume_data_mode="approximate",
    )
    assert summary["resume"]["data_mode"] == "approximate"
    assert summary["results"]["start_step"] == 6
    assert len(summary["results"]["losses"]) == 6


@pytest.mark.slow
def test_resume_under_torch_compile_round_trips(monkeypatch, tmp_path):
    """torch.compile both directions: the checkpoint is saved from the raw module
    (clean keys) and resume loads before the compile wrap (aot_eager keeps eager
    numerics, so compiled-parent -> compiled-resume stays bitwise)."""
    summary_a = _run_train(
        monkeypatch, tmp_path, "compile-uninterrupted", compile=None, compile_backend="aot_eager", max_steps=6
    )
    _run_train(
        monkeypatch,
        tmp_path,
        "compile-parent",
        compile=None,
        compile_backend="aot_eager",
        checkpoint_interval=3,
        max_steps=6,
    )
    model_sd = torch.load(_ckpt_dir(tmp_path, "compile-parent") / "model_000002.pt", weights_only=True)
    assert not any(k.startswith("_orig_mod.") for k in model_sd), "checkpoint keys must be clean under compile"
    summary_b = _run_train(
        monkeypatch,
        tmp_path,
        "compile-resumed",
        compile=None,
        compile_backend="aot_eager",
        resume_from=str(_ckpt_dir(tmp_path, "compile-parent")),
        resume_step=2,
        max_steps=6,
    )
    _assert_bitwise_trajectory(summary_b["results"]["losses"], summary_a["results"]["losses"][3:], offset=3)


def test_validation_loss_evaluated_and_recorded(monkeypatch, tmp_path):
    """Verification for bead 5fp (validation loss): --val-interval/--val-batches
    drive periodic val CE evaluation and both train/val CE land in summary.json.
    (The support landed untested in c6951ae; this pins it.)"""

    def fake_plain_loader(B, T, split, device="cpu", **kwargs):
        gen = torch.Generator().manual_seed(99 if split == "val" else 1234)
        while True:
            tokens = torch.randint(0, VOCAB_FAKE, (B, T + 1), generator=gen, dtype=torch.long)
            yield tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

    monkeypatch.setattr(train_mod, "tokenizing_distributed_data_loader", fake_plain_loader)
    summary = _run_train(monkeypatch, tmp_path, "with-val", val_interval=3, val_batches=2)

    val_losses = summary["results"]["val_losses"]
    assert [step for step, _ in val_losses] == [2, 5, 8, 11], f"unexpected val steps: {val_losses}"
    assert all(v == v and v > 0 for _, v in val_losses), f"non-finite val CE: {val_losses}"
    assert summary["results"]["val_ce_final"] == val_losses[-1][1]
    assert summary["hparams"]["val_interval"] == 3
    assert summary["hparams"]["val_batches"] == 2


def test_metrics_stream_written_and_readable(monkeypatch, tmp_path):
    """rz8.2: metrics.jsonl exists with a provenance header, one step record
    per log interval, val records at the val cadence, and the reader
    round-trips it with schema validation."""
    import nanochat.train as train_mod
    from nanochat.report import METRICS_SCHEMA_VERSION, read_metrics_jsonl

    def fake_plain_loader(B, T, split, device="cpu", **kwargs):
        gen = torch.Generator().manual_seed(99)
        while True:
            tokens = torch.randint(0, VOCAB_FAKE, (B, T + 1), generator=gen, dtype=torch.long)
            yield tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

    monkeypatch.setattr(train_mod, "tokenizing_distributed_data_loader", fake_plain_loader)
    _run_train(monkeypatch, tmp_path, "metrics-run", val_interval=4, val_batches=2, scheduler_type="ordinal")

    path = tmp_path / "artifacts" / "baseline" / "nanochat" / "metrics-run" / "metrics.jsonl"
    assert path.exists(), "metrics.jsonl must be written next to summary.json"
    header, records, problems = read_metrics_jsonl(path)
    assert problems == [], f"clean run must produce a clean stream: {problems}"
    assert header is not None and header["schema_version"] == METRICS_SCHEMA_VERSION
    for key in ("git_sha", "git_dirty", "config_hash", "tainted"):
        assert key in header, f"provenance header missing {key}"

    steps = [r for r in records if r["type"] == "step"]
    vals = [r for r in records if r["type"] == "val"]
    assert [r["step"] for r in steps] == list(range(12)), "default log-interval 1 -> one record per step"
    assert [r["step"] for r in vals] == [3, 7, 11], "val records at the val cadence"
    for r in steps:
        for key in ("loss", "lr", "lr_groups", "grad_norm", "tokens_per_s", "tflops", "elapsed_s"):
            assert key in r, f"step record missing {key}"
        assert r["grad_norm"] > 0, "gradients exist at logging time (post-step, pre-zero)"
        assert "ordinal" in r and set(r["ordinal"]) == {"A", "B", "C", "best_loss", "ema_loss"}

    # summary.json carries the same provenance block
    summary = json.loads((path.parent / "summary.json").read_text())
    assert summary["provenance"]["config_hash"] == header["config_hash"]


def test_metrics_stream_tropical_gamma_passthrough(monkeypatch, tmp_path):
    from nanochat.report import read_metrics_jsonl

    _run_train(
        monkeypatch,
        tmp_path,
        "metrics-tropical",
        max_steps=4,
        attention_type="tropical",
        tropical_record_margins=None,  # boolean flag
    )
    path = tmp_path / "artifacts" / "baseline" / "nanochat" / "metrics-tropical" / "metrics.jsonl"
    _header, records, problems = read_metrics_jsonl(path)
    assert problems == []
    steps = [r for r in records if r["type"] == "step"]
    assert steps and all("tropical_gamma_min" in r and "tropical_gamma_head_mean" in r for r in steps)


def test_metrics_reader_tolerates_malformed_lines(tmp_path):
    from nanochat.report import METRICS_SCHEMA_VERSION, read_metrics_jsonl

    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "header",
                "schema_version": METRICS_SCHEMA_VERSION,
                "git_sha": "x",
                "git_dirty": False,
                "config_hash": "y",
                "tainted": False,
            }
        )
        + "\n"
        + '{"type": "step", "step": 0, "loss": 1.0}\n'
        + "{not json at all\n"
        + '{"no_type_field": true}\n'
        + '{"type": "step", "step": 1, "loss": 0.9}\n'
    )
    header, records, problems = read_metrics_jsonl(path)
    assert header is not None
    assert [r["step"] for r in records] == [0, 1], "good lines survive bad neighbors"
    assert len(problems) == 2, f"each bad line reported once: {problems}"


def test_provenance_taint_semantics(monkeypatch):
    from nanochat import report as report_mod

    def fake_git(info):
        return lambda: dict(info)

    monkeypatch.setattr(report_mod, "get_git_info", fake_git({"commit_full": "abc123", "dirty": False}))
    clean = report_mod.build_provenance({"a": 1})
    assert clean["tainted"] is False and clean["git_sha"] == "abc123"

    monkeypatch.setattr(report_mod, "get_git_info", fake_git({"commit_full": "abc123", "dirty": True}))
    dirty = report_mod.build_provenance({"a": 1})
    assert dirty["tainted"] is True and dirty["git_dirty"] is True

    monkeypatch.setattr(report_mod, "get_git_info", fake_git({"commit_full": "unknown", "dirty": False}))
    nogit = report_mod.build_provenance({"a": 1})
    assert nogit["tainted"] is True and nogit["git_sha"] is None

    # config_hash is canonical: key order must not matter
    assert (
        report_mod.build_provenance({"a": 1, "b": 2})["config_hash"]
        == report_mod.build_provenance({"b": 2, "a": 1})["config_hash"]
    )


def test_data_dir_trains_on_generated_task_corpus(tmp_path):
    """kbj2: --data-dir points training at an mgr gen-tasks corpus through the
    REAL dataloader + tokenizer (no monkeypatching) - the campaign path."""
    try:
        from nanochat.tokenizer import get_tokenizer

        get_tokenizer()
    except Exception as exc:
        pytest.skip(f"tokenizer unavailable: {exc}")

    import nanochat.train as train_mod
    from nanochat.diagnostics_data import generate_task

    generate_task("arith", out_dir=tmp_path / "corpus", size=200, seed=7)
    args = train_mod.build_parser().parse_args(
        [
            "--device",
            "cpu",
            "--max-steps",
            "3",
            "--batch-size",
            "2",
            "--sequence-len",
            "64",
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
            "datadir",
            "--data-dir",
            str(tmp_path / "corpus" / "arith"),
            "--val-interval",
            "2",
            "--val-batches",
            "1",
        ]
    )
    train_mod.train(args)
    summary = json.loads((tmp_path / "artifacts" / "baseline" / "nanochat" / "datadir" / "summary.json").read_text())
    ds = summary["dataset"]
    assert ds["data_dir"] == str(tmp_path / "corpus" / "arith")
    assert [Path(p).name for p in ds["parquet_files"]] == ["train_000.parquet", "val_000.parquet"]
    assert len(summary["results"]["losses"]) == 3
    assert summary["results"]["val_losses"], "val split (last parquet) must feed validation"

    # nonexistent dir -> actionable validation error
    bad = train_mod.build_parser().parse_args(
        ["--device", "cpu", "--max-steps", "1", "--data-dir", str(tmp_path / "nope")]
    )
    with pytest.raises(ValueError, match="data-dir"):
        train_mod.train(bad)


@pytest.mark.parametrize(
    "flag, value, match",
    [
        ("--run-id", "../escape", "run-id"),
        ("--run-id", "nested/run", "run-id"),
        ("--run-id", "   ", "run-id"),
        ("--artifacts-topic", "../../etc", "artifacts-topic"),
        ("--artifacts-topic", "/abs/path", "artifacts-topic"),
        ("--artifacts-kind", "bench/../x", "artifacts-kind"),
    ],
)
def test_artifact_path_segments_are_validated(tmp_path, flag, value, match):
    """--run-id is one directory under <artifacts-dir>/<kind>/<topic>/ and
    kind/topic may nest but never escape: previously '--run-id ../x' wrote
    summary.json and metrics.jsonl outside the artifacts tree."""
    from nanochat import train as train_mod

    args = train_mod.build_parser().parse_args(
        ["--device", "cpu", "--max-steps", "1", "--artifacts-dir", str(tmp_path / "artifacts"), flag, value]
    )
    with pytest.raises(ValueError, match=match):
        train_mod.train(args)
    assert not (tmp_path / "escape").exists() and not (tmp_path / "etc").exists()


def test_optimizer_type_selects_distinct_optimizers():
    """`--optimizer-type adamw | muon | hoss` must build three DIFFERENT
    optimizer sets. Before 2026-09 setup_optimizers branched only on hoss, so
    'adamw' and 'muon' both produced the AdamW+Muon split and every artifact
    labelled adamw had trained its matrices with Muon."""
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.hoss_opt_torch import HOSS
    from nanochat.muon import Muon

    def build(optimizer_type):
        cfg = GPTConfig(
            n_layer=1, n_head=2, n_kv_head=2, n_embd=16, sequence_len=8, vocab_size=64, optimizer_type=optimizer_type
        )
        return GPT(cfg).setup_optimizers()

    adamw = build("adamw")
    muon = build("muon")
    hoss = build("hoss")
    assert [type(o) for o in adamw] == [torch.optim.AdamW]
    assert [type(o) for o in muon] == [torch.optim.AdamW, Muon]
    assert [type(o) for o in hoss] == [HOSS]
    # Under pure AdamW every trainable parameter is owned by that single
    # optimizer; under the split the block matrices belong to Muon only.
    n_params = lambda opts: sum(p.numel() for o in opts for g in o.param_groups for p in g["params"])  # noqa: E731
    assert n_params(adamw) == n_params(muon) == n_params(hoss)
    muon_matrix_params = sum(p.numel() for g in muon[1].param_groups for p in g["params"])
    assert muon_matrix_params > 0
    with pytest.raises(ValueError, match="optimizer_type must be"):
        build("sgd")


def test_ordinal_scheduler_preserves_param_group_lr_ratios():
    """The schedule is a multiplicative scale on each group's OWN base LR.
    The old implementation overwrote every group with one eta_init, which
    silently flattened the embedding / lm_head / matrix LR split."""
    from nanochat.ordinal_scheduler import OrdinalLRScheduler

    p1, p2 = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.SGD([{"params": [p1], "lr": 0.2}, {"params": [p2], "lr": 0.02}])
    sched = OrdinalLRScheduler(opt, A_init=1, B_init=1, P_init=2, gamma=0.5)
    assert sched.get_last_lr() == [0.2, 0.02]
    for x in [1.0, 1.1, 1.2]:  # non-improving: patience 2 expires -> anneal
        sched.step(x)
    assert sched.B == 0
    assert sched.get_last_lr() == pytest.approx([0.1, 0.01])
    for x in [1.3, 1.4, 1.5]:  # expires again -> restart back to the base LRs
        sched.step(x)
    assert (sched.A, sched.B) == (0, 1)
    assert sched.get_last_lr() == pytest.approx([0.2, 0.02])


def test_ordinal_scheduler_state_dict_round_trip():
    """Unit: a restored scheduler continues the exact LR trajectory of the original."""
    from nanochat.ordinal_scheduler import OrdinalLRScheduler

    def make():
        params = [torch.nn.Parameter(torch.zeros(2))]
        opt = torch.optim.SGD(params, lr=0.1)
        return OrdinalLRScheduler(opt, A_init=1, B_init=2, P_init=3, gamma=0.5)

    # Non-improving losses force C to count down and trigger anneals/restarts.
    losses = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    ref = make()
    for x in losses:
        ref.step(x)
    ref_lrs = ref.get_last_lr()

    half = make()
    for x in losses[:5]:
        half.step(x)
    snapshot = half.state_dict()

    restored = make()
    restored.load_state_dict(snapshot)
    # LR is optimizer state; carry it over the way train.py does (optimizer
    # state_dict restore happens alongside the scheduler restore).
    for group, lr in zip(restored.optimizer.param_groups, half.get_last_lr()):
        group["lr"] = lr
    for x in losses[5:]:
        restored.step(x)
    assert restored.get_last_lr() == ref_lrs
    assert (restored.A, restored.B, restored.C) == (ref.A, ref.B, ref.C)
    assert restored.best_loss == ref.best_loss and restored.ema_loss == ref.ema_loss


def test_metrics_stream_append_preserves_history_across_resume(tmp_path):
    """A resumed run must APPEND to metrics.jsonl, never truncate it: a resume
    previously erased the parent process's entire step history (found by the
    rz8.8 e2e resume scenario). The splice is marked by a resume_header record
    carrying the resuming process's provenance."""
    from nanochat.report import MetricsStream, read_metrics_jsonl

    path = tmp_path / "metrics.jsonl"
    prov = {
        "schema_version": "mgr.metrics.v1",
        "git_sha": "aaa",
        "git_dirty": False,
        "config_hash": "h1",
        "data_snapshot_hash": None,
        "tainted": False,
    }
    first = MetricsStream(path, provenance=prov, flush_every=1)
    for step in range(3):
        first.write({"type": "step", "step": step, "loss": 1.0})
    first.close()

    resumed = MetricsStream(path, provenance={**prov, "config_hash": "h2"}, flush_every=1, append=True)
    for step in range(3, 6):
        resumed.write({"type": "step", "step": step, "loss": 0.9})
    resumed.close()

    header, records, problems = read_metrics_jsonl(path)
    assert header is not None and not problems, problems
    steps = [r["step"] for r in records if r.get("type") == "step"]
    assert steps == [0, 1, 2, 3, 4, 5], "pre-kill history must survive the resume"
    splices = [r for r in records if r.get("type") == "resume_header"]
    assert len(splices) == 1 and splices[0]["config_hash"] == "h2", "splice must carry the resumer's provenance"


def test_dequantization_annealing_schedule_telemetry(monkeypatch, tmp_path):
    """8gk.1 e2e smoke: a tropical run with --semiring-beta linear:1:32 logs a
    monotone beta ladder and certificate route_coverage per step, and the
    schedule actually reaches its endpoint."""
    import json as json_mod

    monkeypatch.setattr(train_mod, "tokenizing_distributed_data_loader_with_state", _fake_loader_factory())
    monkeypatch.setattr(train_mod, "list_parquet_files", lambda data_dir=None: ["a.parquet", "b.parquet"])
    args = _train_args(
        tmp_path,
        "anneal-smoke",
        attention_type="tropical",
        semiring_beta="linear:1:32",
        tropical_record_margins=None,  # store_true flag
        log_interval="1",
    )
    train_mod.train(args)
    metrics = tmp_path / "artifacts" / "baseline" / "nanochat" / "anneal-smoke" / "metrics.jsonl"
    betas, coverages = [], []
    for line in metrics.read_text().splitlines():
        rec = json_mod.loads(line)
        if "semiring_beta" in rec:
            betas.append(rec["semiring_beta"])
        if "route_coverage" in rec:
            coverages.append(rec["route_coverage"])
    assert len(betas) >= 10, f"expected per-step beta telemetry, got {len(betas)} records"
    assert all(b2 >= b1 for b1, b2 in zip(betas, betas[1:], strict=False)), "beta must anneal monotonically"
    assert abs(betas[0] - 1.0) < 1e-6 and abs(betas[-1] - 32.0) < 1e-6, (betas[0], betas[-1])
    assert coverages and all(0.0 <= c <= 1.0 for c in coverages), "coverage must be a fraction when margins are on"
    # rgyl: the RAW schedule spec must be durably recorded for arm detection
    summary = json_mod.loads(
        (tmp_path / "artifacts" / "baseline" / "nanochat" / "anneal-smoke" / "summary.json").read_text()
    )
    assert summary["hparams"]["semiring_beta_spec"] == "linear:1:32"


def test_annealed_checkpoint_reloads_at_live_beta(monkeypatch, tmp_path):
    """y2h9: an annealed-to-32 checkpoint must reload at beta=32, not the
    config's b0=1. The checkpoint meta records semiring_beta_live, the summary
    records results.semiring_beta_final, and the eval loader constructs at the
    live value (with the artifact-recorded model_config reflecting it); an
    explicit model_overrides semiring_beta still wins."""
    summary = _run_train(
        monkeypatch,
        tmp_path,
        "anneal-reload",
        attention_type="tropical",
        semiring_beta="linear:1:32",
        checkpoint_interval="6",
    )
    assert abs(summary["results"]["semiring_beta_final"] - 32.0) < 1e-6

    from nanochat.checkpoint_manager import find_last_step
    from nanochat.tropical_attention_torch import TropicalCausalSelfAttention

    ckpt_dir = _ckpt_dir(tmp_path, "anneal-reload")
    last = find_last_step(str(ckpt_dir))
    meta = json.loads((ckpt_dir / f"meta_{last:06d}.json").read_text())
    assert abs(meta["semiring_beta_live"] - 32.0) < 1e-6
    # the recorded config keeps the schedule's b0 - the live value lives beside it
    assert meta["model_config"]["semiring_beta"] == 1.0

    from cli import _load_eval_checkpoint

    model, loaded_meta, _step = _load_eval_checkpoint(ckpt_dir, None, torch.device("cpu"))
    betas = [m.semiring_beta for m in model.modules() if isinstance(m, TropicalCausalSelfAttention)]
    assert betas and all(abs(b - 32.0) < 1e-6 for b in betas), betas
    # eval artifacts record the beta the model actually runs at (variant-selectable)
    assert abs(loaded_meta["model_config"]["semiring_beta"] - 32.0) < 1e-6

    model2, _meta2, _ = _load_eval_checkpoint(
        ckpt_dir, None, torch.device("cpu"), model_overrides={"semiring_beta": 4.0}
    )
    betas2 = [m.semiring_beta for m in model2.modules() if isinstance(m, TropicalCausalSelfAttention)]
    assert betas2 and all(abs(b - 4.0) < 1e-6 for b in betas2), betas2


def test_coverage_beta_controller_transitions_exact():
    """9jzb closed-loop controller: raise/hold/back-off transitions are exact,
    bounds are respected, and beta is NEVER raised on a below-floor or missing
    reading (the preregistration's checkable-from-logs invariant)."""
    ctl = train_mod._CoverageBetaController(1.0, 64.0, 0.7)
    up, down = ctl.UP, ctl.DOWN

    assert ctl.step(None) == 1.0  # no reading -> hold
    assert ctl.step(0.9) == 1.0 * up  # covered -> raise
    assert ctl.step(0.7) == 1.0 * up * up  # floor is inclusive
    b = ctl.beta
    assert ctl.step(0.69) == max(b / down, ctl.b0)  # dip -> back off, clamped at B0
    held = ctl.beta
    assert ctl.step(None) == held  # missing -> hold

    # never-raise-below-floor invariant over an adversarial random stream
    import random as _random

    rng = _random.Random(9362)
    ctl = train_mod._CoverageBetaController(1.0, 64.0, 0.7)
    prev = ctl.beta
    for _ in range(500):
        cov = rng.choice([None, rng.random()])
        beta = ctl.step(cov)
        assert 1.0 <= beta <= 64.0
        if beta > prev:
            assert cov is not None and cov >= 0.7, "raised beta without floor-clearing coverage"
        prev = beta

    # saturation at BMAX under sustained coverage
    ctl = train_mod._CoverageBetaController(1.0, 4.0, 0.5)
    for _ in range(200):
        ctl.step(1.0)
    assert ctl.beta == 4.0


def test_semiring_beta_spec_parser_coverage_mode():
    assert train_mod._parse_semiring_beta_spec("coverage:1:64:0.7") == ("coverage", 1.0, 64.0, 0.7)
    for bad in ("coverage:1:64", "coverage:0:64:0.7", "coverage:8:4:0.7", "coverage:1:64:1.5", "sigmoid:1:2"):
        with pytest.raises(ValueError):
            train_mod._parse_semiring_beta_spec(bad)


def test_closed_loop_schedule_e2e_invariant(monkeypatch, tmp_path):
    """Closed-loop e2e smoke: metrics stream shows beta never raised on a
    step whose PREVIOUS coverage reading was below the floor."""
    import json as json_mod

    monkeypatch.setattr(train_mod, "tokenizing_distributed_data_loader_with_state", _fake_loader_factory())
    monkeypatch.setattr(train_mod, "list_parquet_files", lambda data_dir=None: ["a.parquet", "b.parquet"])
    args = _train_args(
        tmp_path,
        "coverage-smoke",
        attention_type="tropical",
        semiring_beta="coverage:1:16:0.5",
        tropical_record_margins=None,
        log_interval="1",
    )
    train_mod.train(args)
    metrics = tmp_path / "artifacts" / "baseline" / "nanochat" / "coverage-smoke" / "metrics.jsonl"
    rows = []
    for line in metrics.read_text().splitlines():
        rec = json_mod.loads(line)
        if "semiring_beta" in rec:
            rows.append((rec["semiring_beta"], rec.get("route_coverage")))
    assert len(rows) >= 10, f"expected per-step telemetry, got {len(rows)}"
    assert all(1.0 <= b <= 16.0 for b, _ in rows)
    for (b_prev, cov_prev), (b_next, _) in zip(rows, rows[1:], strict=False):
        if b_next > b_prev:
            assert cov_prev is not None and cov_prev >= 0.5, (
                f"beta raised {b_prev} -> {b_next} after coverage {cov_prev}"
            )


def test_coverage_spec_requires_margins(monkeypatch, tmp_path):
    monkeypatch.setattr(train_mod, "tokenizing_distributed_data_loader_with_state", _fake_loader_factory())
    monkeypatch.setattr(train_mod, "list_parquet_files", lambda data_dir=None: ["a.parquet", "b.parquet"])
    args = _train_args(
        tmp_path,
        "coverage-no-margins",
        attention_type="tropical",
        semiring_beta="coverage:1:16:0.5",
    )
    with pytest.raises(ValueError, match="tropical-record-margins"):
        train_mod.train(args)


def test_ordinal_beta_ladder_transitions_exact():
    """9jzb ordinal mode: beta doubles (capped) exactly on (A, B) descents -
    the ordinal scheduler's anneal/restart events - and holds otherwise."""
    ladder = train_mod._OrdinalBetaLadder(1.0, 8.0)
    assert ladder.step(2, 3) == 1.0  # first observation initializes, no event
    assert ladder.step(2, 3) == 1.0  # no descent -> hold
    assert ladder.step(2, 2) == 2.0  # anneal (B descent) -> double
    assert ladder.step(2, 2) == 2.0  # hold
    assert ladder.step(1, 3) == 4.0  # restart (A descent, B reset) -> double
    assert ladder.step(1, 2) == 8.0  # anneal -> double, hits cap
    assert ladder.step(1, 1) == 8.0  # capped at BMAX
    assert train_mod._parse_semiring_beta_spec("ordinal:1:8") == ("ordinal", 1.0, 8.0)


def test_ordinal_beta_mode_requires_ordinal_scheduler(monkeypatch, tmp_path):
    monkeypatch.setattr(train_mod, "tokenizing_distributed_data_loader_with_state", _fake_loader_factory())
    monkeypatch.setattr(train_mod, "list_parquet_files", lambda data_dir=None: ["a.parquet", "b.parquet"])
    args = _train_args(tmp_path, "ord-beta-no-sched", attention_type="tropical", semiring_beta="ordinal:1:8")
    with pytest.raises(ValueError, match="scheduler-type ordinal"):
        train_mod.train(args)


def test_ordinal_beta_mode_e2e_holds_absent_transitions(monkeypatch, tmp_path):
    """With default patience no ordinal event fires in 12 steps: beta must
    hold at B0 the whole run (the transfinite clock, not the step clock)."""
    import json as json_mod

    monkeypatch.setattr(train_mod, "tokenizing_distributed_data_loader_with_state", _fake_loader_factory())
    monkeypatch.setattr(train_mod, "list_parquet_files", lambda data_dir=None: ["a.parquet", "b.parquet"])
    args = _train_args(
        tmp_path,
        "ord-beta-smoke",
        attention_type="tropical",
        semiring_beta="ordinal:2:16",
        scheduler_type="ordinal",
        log_interval="1",
    )
    train_mod.train(args)
    metrics = tmp_path / "artifacts" / "baseline" / "nanochat" / "ord-beta-smoke" / "metrics.jsonl"
    betas, saw_ordinal = [], False
    for line in metrics.read_text().splitlines():
        rec = json_mod.loads(line)
        if "semiring_beta" in rec:
            betas.append(rec["semiring_beta"])
            saw_ordinal = saw_ordinal or "ordinal" in rec
    assert len(betas) >= 10 and all(b == 2.0 for b in betas), betas[:5]
    assert saw_ordinal, "ordinal scheduler telemetry must accompany the beta stream"


def test_semiring_controller_and_ladder_state_round_trip():
    """efzy: state_dict/load_state_dict reproduce the exact decision state of
    both stateful beta controllers."""
    ctl = train_mod._CoverageBetaController(1.0, 64.0, 0.7)
    ctl.step(0.9)
    ctl.step(0.9)
    ctl.step(0.1)
    clone = train_mod._CoverageBetaController(1.0, 64.0, 0.7)
    clone.load_state_dict(ctl.state_dict())
    assert clone.beta == ctl.beta
    assert clone.step(0.9) == ctl.step(0.9)

    lad = train_mod._OrdinalBetaLadder(1.0, 32.0)
    lad.step(3, 5)
    lad.step(3, 4)  # descent -> double
    clone2 = train_mod._OrdinalBetaLadder(1.0, 32.0)
    clone2.load_state_dict(lad.state_dict())
    assert clone2.beta == lad.beta and clone2._last_ab == lad._last_ab
    assert clone2.step(3, 3) == lad.step(3, 3)  # descent seen identically
    assert clone2.step(3, 3) == lad.step(3, 3)  # hold seen identically


@pytest.mark.slow
def test_coverage_beta_mode_resumes_bitwise(monkeypatch, tmp_path):
    """efzy: the coverage controller's beta and the checkpoint step's pending
    coverage reading round-trip through optim_*.pt, so a resumed closed-loop
    run reproduces the uninterrupted run's beta trajectory bitwise.

    The fake coverage reader is a PURE FUNCTION of the model weights: identical
    model state yields identical readings (regardless of how often it is
    called), so any beta divergence is controller state, not reader noise."""
    import json as json_mod

    def fake_coverage(model):
        p = next(model.parameters()).detach().flatten()
        return float((p[0].abs() * 997.0) % 1.0)

    monkeypatch.setattr(train_mod, "_collect_tropical_route_coverage", fake_coverage)

    def beta_series(run_id: str) -> dict[int, float]:
        metrics = tmp_path / "artifacts" / "baseline" / "nanochat" / run_id / "metrics.jsonl"
        out: dict[int, float] = {}
        for line in metrics.read_text().splitlines():
            rec = json_mod.loads(line)
            if rec.get("type") == "step" and "semiring_beta" in rec:
                out[rec["step"]] = rec["semiring_beta"]
        return out

    summary_a = _run_train(
        monkeypatch,
        tmp_path,
        "cov-uninterrupted",
        attention_type="tropical",
        semiring_beta="coverage:1:16:0.5",
        tropical_record_margins=None,
        log_interval="1",
    )
    _run_train(
        monkeypatch,
        tmp_path,
        "cov-parent",
        checkpoint_interval=6,
        attention_type="tropical",
        semiring_beta="coverage:1:16:0.5",
        tropical_record_margins=None,
        log_interval="1",
    )

    betas_a = beta_series("cov-uninterrupted")
    assert len(set(betas_a.values())) > 1, "fake coverage must move beta or this test is vacuous"

    state = torch.load(_ckpt_dir(tmp_path, "cov-parent") / "optim_000005_rank0.pt", weights_only=False)
    assert "beta" in state["semiring_controller"]
    assert "semiring_pending_coverage" in state

    summary_b = _run_train(
        monkeypatch,
        tmp_path,
        "cov-resumed",
        resume_from=str(_ckpt_dir(tmp_path, "cov-parent")),
        resume_step=5,
        attention_type="tropical",
        semiring_beta="coverage:1:16:0.5",
        tropical_record_margins=None,
        log_interval="1",
    )
    assert summary_b["results"]["start_step"] == 6
    betas_b = beta_series("cov-resumed")
    assert betas_b, "resumed run logged no beta telemetry"
    for step in sorted(betas_b):
        assert betas_b[step] == betas_a[step], (
            f"beta diverged at step {step}: resumed {betas_b[step]} != uninterrupted {betas_a[step]}"
        )
    _assert_bitwise_trajectory(summary_b["results"]["losses"], summary_a["results"]["losses"][6:], offset=6)


@pytest.mark.slow
def test_stateful_beta_resume_refuses_missing_controller_state(monkeypatch, tmp_path):
    """efzy: resuming a coverage/ordinal beta run from a checkpoint saved under
    a DIFFERENT --semiring-beta mode (no controller state in the payload) must
    fail loudly, not silently restart the beta trajectory at b0."""
    _run_train(
        monkeypatch,
        tmp_path,
        "stateless-parent",
        checkpoint_interval=6,
        attention_type="tropical",
        semiring_beta="linear:1:8",
        tropical_record_margins=None,
    )
    _run_train(
        monkeypatch,
        tmp_path,
        "stateless-parent-ord",
        checkpoint_interval=6,
        attention_type="tropical",
        semiring_beta="linear:1:8",
        scheduler_type="ordinal",
    )
    cases = (
        ("coverage:1:8:0.5", "stateless-parent", {"tropical_record_margins": None}),
        ("ordinal:1:8", "stateless-parent-ord", {"scheduler_type": "ordinal"}),
    )
    for spec, parent, extra in cases:
        args = _train_args(
            tmp_path,
            f"stateful-resume-{spec.split(':')[0]}",
            attention_type="tropical",
            semiring_beta=spec,
            resume_from=str(_ckpt_dir(tmp_path, parent)),
            resume_step=5,
            **extra,
        )
        with pytest.raises(RuntimeError, match="no semiring_"):
            train_mod.train(args)


def test_lr_warmdown_multiplier_shape():
    """kj8s: flat until the tail starts, linear to exactly 0 at the final step,
    degenerate cases stay at 1.0 (pure function — no hidden state)."""
    f = train_mod._lr_warmdown_multiplier
    assert all(f(s, 12, 0.5) == 1.0 for s in range(6)), "must hold flat before the tail"
    assert f(6, 12, 0.5) == 1.0, "first tail step is still the full LR"
    shape = [f(s, 12, 0.5) for s in range(6, 12)]
    assert shape == pytest.approx([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]), f"linear decay broken: {shape}"
    assert f(11, 12, 0.5) == 0.0, "final step must reach 0"
    assert f(0, 10, 0.0) == 1.0, "ratio 0 must be a no-op"
    assert f(0, 1, 0.9) == 1.0, "single-step run has no tail"


@pytest.mark.slow
def test_lr_warmdown_tail_lands_in_metrics(monkeypatch, tmp_path):
    """kj8s end-to-end: ratio 0.5 over 12 steps — per-step recorded LRs stay
    flat through the stable phase then decay linearly to ~0; the default run
    stays perfectly flat (log_interval=1 so every step records an lr)."""
    summary = _run_train(monkeypatch, tmp_path, "warmdown-run", lr_warmdown_ratio=0.5, log_interval=1)
    assert len(summary["results"]["losses"]) == 12
    metrics = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "baseline" / "nanochat" / "warmdown-run" / "metrics.jsonl")
        .read_text()
        .splitlines()
    ]
    lrs = [rec["lr"] for rec in metrics if rec.get("type") == "step"]
    base = lrs[0]
    assert base > 0
    assert lrs[:7] == [base] * 7, f"stable phase must hold the base LR: {lrs[:7]}"
    tail = lrs[7:]
    assert tail[-1] < 1e-3 * base, f"tail must decay to ~0: {tail}"
    assert all(a >= b for a, b in zip(tail, tail[1:])), f"tail must be non-increasing: {tail}"

    _run_train(monkeypatch, tmp_path, "flat-run")
    flat = [
        json.loads(line)["lr"]
        for line in (tmp_path / "artifacts" / "baseline" / "nanochat" / "flat-run" / "metrics.jsonl")
        .read_text()
        .splitlines()
        if json.loads(line).get("type") == "step"
    ]
    assert len(set(flat)) == 1, f"default must remain flat LR: {set(flat)}"


def test_lr_warmdown_conflicts_with_ordinal_scheduler(tmp_path):
    """kj8s: the warmdown tail owns the LR only on flat-LR runs; combining it
    with the ordinal scheduler is a configuration error, not silent double-drive."""
    argv_extra = ["--lr-warmdown-ratio", "0.2", "--scheduler-type", "ordinal"]
    args = train_mod.build_parser().parse_args(
        [
            "--device",
            "cpu",
            "--max-steps",
            "2",
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
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--run-id",
            "conflict",
            *argv_extra,
        ]
    )
    with pytest.raises(ValueError, match="lr-warmdown-ratio"):
        train_mod.train(args)


@pytest.mark.slow
def test_hoss_end_to_end_training_smoke(monkeypatch, tmp_path):
    """rz8.3: nanochat trains end-to-end with --optimizer-type hoss on the
    CPU smoke config - the closure path (loss.backward(create_graph=True)
    feeding HOSS's HVP) runs inside the REAL training loop, and the recorded
    losses stay finite throughout."""
    summary = _run_train(
        monkeypatch,
        tmp_path,
        "hoss-smoke",
        max_steps=6,
        optimizer_type="hoss",
        check_numerics=None,
    )
    losses = summary["results"]["losses"]
    assert len(losses) == 6, f"expected 6 HOSS steps, got {len(losses)}"
    assert all(loss == loss and abs(loss) != float("inf") for loss in losses), (
        f"non-finite HOSS training losses: {losses}"
    )


def test_parameterization_recorded_in_run_artifacts(monkeypatch, tmp_path):
    """bp08: E1/E2 runs must record WHICH parameterization arm they used —
    the summary config carries it end-to-end from the CLI flag."""
    summary = _run_train(
        monkeypatch,
        tmp_path,
        "nsa-flag",
        max_steps=2,
        attention_type="tropical",
        ffn_type="tropical",
        parameterization="nsa",
    )
    assert summary["config"]["parameterization"] == "nsa"
    assert len(summary["results"]["losses"]) == 2
    assert all(v == v for v in summary["results"]["losses"]), "NaN loss under nsa arm"

    flat = _run_train(monkeypatch, tmp_path, "current-default", max_steps=2)
    assert flat["config"]["parameterization"] == "current"


def test_profile_flag_writes_summary_and_trace(monkeypatch, tmp_path):
    """6m35: --profile profiles the FINAL --profile-steps steps and lands a
    chrome trace under <run-dir>/profile/; the run itself completes normally."""
    summary = _run_train(
        monkeypatch,
        tmp_path,
        "profiled",
        max_steps=6,
        profile=None,  # store_true -> bare flag
        profile_steps=2,
    )
    assert len(summary["results"]["losses"]) == 6
    trace_dir = tmp_path / "artifacts" / "baseline" / "nanochat" / "profiled" / "profile"
    traces = sorted(trace_dir.glob("*.json"))
    assert traces, f"no chrome trace written under {trace_dir}"
    assert all(t.stat().st_size > 0 for t in traces)


def test_clifford_attention_training_smoke(monkeypatch, tmp_path):
    """mnn.3: --attention-type clifford trains end-to-end through the real
    loop (chunked geometric-product aggregate + KV scaffold) with finite
    losses throughout."""
    summary = _run_train(
        monkeypatch,
        tmp_path,
        "clifford-smoke",
        max_steps=6,
        attention_type="clifford",
    )
    losses = summary["results"]["losses"]
    assert len(losses) == 6
    assert all(loss == loss and abs(loss) != float("inf") for loss in losses), (
        f"non-finite clifford training losses: {losses}"
    )


def test_hyperbolic_attention_training_smoke(monkeypatch, tmp_path):
    """mnn.6: hyperbolic attention trains through the real CPU loop and
    publishes curvature/radius observables into the summary results."""
    summary = _run_train(
        monkeypatch,
        tmp_path,
        "hyperbolic-smoke",
        max_steps=6,
        attention_type="hyperbolic",
    )
    losses = summary["results"]["losses"]
    assert len(losses) == 6
    assert all(loss == loss and abs(loss) != float("inf") for loss in losses), (
        f"non-finite hyperbolic training losses: {losses}"
    )
    assert summary["results"]["hyperbolic_curvature"]["mean"] > 0.0
    assert summary["results"]["hyperbolic_radius"]["mean"] >= 0.0
    assert 0.0 <= summary["results"]["hyperbolic_frac_heads_hier"] <= 1.0
    assert 0.0 <= summary["results"]["hyperbolic_frac_heads_euclidean"] <= 1.0


def test_find_last_step_discovers_through_meta_commit_point(tmp_path):
    """Commit-point discovery (save_checkpoint contract): an orphaned
    model_*.pt whose meta never landed (SIGKILL mid-save) must NOT be picked
    as the resume step — discovery goes through meta_*.json, and load
    falls back to the last committed step instead of dying on missing meta."""
    from nanochat.checkpoint_manager import find_last_step

    d = tmp_path / "ckpts"
    d.mkdir()
    # committed step 5: full triple
    (d / "model_000005.pt").write_bytes(b"x")
    (d / "optim_000005_rank0.pt").write_bytes(b"x")
    (d / "meta_000005.json").write_text("{}")
    # orphaned step 7: model landed, SIGKILL before the meta commit point
    (d / "model_000007.pt").write_bytes(b"x")

    assert find_last_step(str(d)) == 5

    # empty dir -> loud error naming the commit-point requirement
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        find_last_step(str(empty))


@pytest.mark.slow
def test_activation_checkpointing_preserves_trajectories_bitwise(monkeypatch, tmp_path):
    """saew: full / every-k checkpointing recomputes activations instead of
    storing them; with no dropout the recompute is numerically identical, so
    loss trajectories must match the default none-mode run BITWISE on CPU."""
    runs = {}
    summary = None
    for mode in ("none", "full", "every-k"):
        if mode == "every-k":
            summary = _run_train(
                monkeypatch,
                tmp_path,
                f"ckpt-{mode}",
                max_steps=6,
                activation_checkpointing=mode,
                activation_ckpt_every_k=2,
            )
        else:
            summary = _run_train(
                monkeypatch,
                tmp_path,
                f"ckpt-{mode}",
                max_steps=6,
                activation_checkpointing=mode,
            )
        runs[mode] = summary["results"]["losses"]
        assert len(runs[mode]) == 6
        assert all(v == v for v in runs[mode]), f"{mode}: NaN loss"
    assert runs["full"] == runs["none"], "full ckpt changed the trajectory"
    assert runs["every-k"] == runs["none"], "every-k ckpt changed the trajectory"
    assert summary is not None and summary["config"]["activation_ckpt"] == "every-k"


def test_task_tokenizer_travels_with_the_checkpoint(tmp_path):
    """`--tokenizer task` trains a byte-level BPE on the corpus, sizes the
    embedding table from it, saves it INSIDE the checkpoint dir, and every
    checkpoint consumer loads it back through checkpoint_tokenizer. With the
    GPT-2 tokenizer this tiny model would spend ~97% of its FLOPs on the
    50k-row vocabulary matrices."""
    from typer.testing import CliRunner

    import cli
    from nanochat.diagnostics_data import generate_task
    from nanochat.tokenizer import TASK_TOKENIZER_DIRNAME, checkpoint_tokenizer, padded_vocab_size

    generate_task("arith", out_dir=tmp_path / "corpus", size=60, seed=3)
    data_dir = tmp_path / "corpus" / "arith"
    args = train_mod.build_parser().parse_args(
        [
            "--device", "cpu", "--max-steps", "2", "--n-layer", "1", "--n-head", "2", "--n-kv-head", "2",
            "--n-embd", "32", "--sequence-len", "32", "--batch-size", "2", "--warmup-steps", "0",
            "--checkpoint-interval", "2", "--data-dir", str(data_dir), "--tokenizer", "task",
            "--tokenizer-vocab-size", "320", "--artifacts-dir", str(tmp_path / "artifacts"), "--run-id", "tok",
        ]
    )  # fmt: skip
    train_mod.train(args)
    run_dir = tmp_path / "artifacts" / "baseline" / "nanochat" / "tok"
    ckpt_dir = run_dir / "checkpoints"
    assert (ckpt_dir / TASK_TOKENIZER_DIRNAME / "tokenizer.json").exists()
    meta = json.loads(sorted(ckpt_dir.glob("meta_*.json"))[-1].read_text())
    tok_meta = meta["tokenizer"]
    assert tok_meta["kind"] == "task" and tok_meta["dir"] == TASK_TOKENIZER_DIRNAME
    assert 256 < tok_meta["vocab_size"] <= 320
    assert meta["model_config"]["vocab_size"] == padded_vocab_size(tok_meta["vocab_size"]) < 1000
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["hparams"]["tokenizer"]["kind"] == "task"

    tok = checkpoint_tokenizer(ckpt_dir, meta)
    doc = "TASK arith CMP 1.00e-02 2.00e+03 OUT"
    ids = tok.encode(doc)
    assert max(ids) < meta["model_config"]["vocab_size"]
    assert tok.decode(ids) == doc
    # Resume retrains the tokenizer from the same corpus instead of reloading
    # it, so BPE training must be deterministic for a fixed corpus.
    from nanochat.dataset import list_parquet_files
    from nanochat.tokenizer import train_task_tokenizer

    retrained = train_task_tokenizer(list_parquet_files(str(data_dir))[:-1], 320)
    assert retrained.get_vocab_size() == tok_meta["vocab_size"]
    assert retrained.encode(doc) == ids

    # the sampler must decode with the checkpoint's tokenizer, not GPT-2 ids
    # (which would index past this 320-row embedding table)
    result = CliRunner().invoke(
        cli.app,
        ["sample", "--checkpoint", str(ckpt_dir), "--prompt", doc, "--max-tokens", "3", "--device", "cpu", "--json"],
    )
    assert result.exit_code == 0, result.output[-2000:]
    assert json.loads(result.output)["results"][0]["n_tokens"] <= 3


def test_task_tokenizer_flag_validation(tmp_path):
    for extra, match in (
        (["--tokenizer", "task"], "data-dir"),
        (["--tokenizer", "task", "--data-dir", str(tmp_path), "--vocab-size", "1024"], "vocab-size"),
        (["--tokenizer", "task", "--data-dir", str(tmp_path), "--tokenizer-vocab-size", "100"], "tokenizer-vocab-size"),
    ):
        args = train_mod.build_parser().parse_args(["--device", "cpu", "--max-steps", "1", *extra])
        with pytest.raises(ValueError, match=match):
            train_mod.train(args)


def test_validation_reports_bits_per_byte(tmp_path):
    """Validation records bits per byte next to cross-entropy. bpb divides the
    summed loss by the BYTES the target tokens spell, so it stays comparable
    when the vocabulary changes (val_ce does not: a coarser tokenizer packs
    more bytes per token and reports a higher per-token loss for the same
    model). Special tokens count for neither."""
    import math

    from nanochat.diagnostics_data import generate_task
    from nanochat.tokenizer import checkpoint_tokenizer

    generate_task("arith", out_dir=tmp_path / "corpus", size=60, seed=4)
    data_dir = tmp_path / "corpus" / "arith"
    args = train_mod.build_parser().parse_args(
        [
            "--device", "cpu", "--max-steps", "2", "--n-layer", "1", "--n-head", "2", "--n-kv-head", "2",
            "--n-embd", "32", "--sequence-len", "32", "--batch-size", "2", "--warmup-steps", "0",
            "--checkpoint-interval", "2", "--val-interval", "1", "--val-batches", "2",
            "--data-dir", str(data_dir), "--tokenizer", "task", "--tokenizer-vocab-size", "320",
            "--artifacts-dir", str(tmp_path / "artifacts"), "--run-id", "bpb",
        ]
    )  # fmt: skip
    train_mod.train(args)
    run_dir = tmp_path / "artifacts" / "baseline" / "nanochat" / "bpb"
    results = json.loads((run_dir / "summary.json").read_text())["results"]
    ce, bpb = results["val_ce_final"], results["val_bpb_final"]
    assert ce is not None and bpb is not None and math.isfinite(bpb) and bpb > 0
    # every counted token is at least one byte, so bpb <= ce / ln 2
    assert bpb <= ce / math.log(2) + 1e-6
    assert len(results["val_bpbs"]) == len(results["val_losses"]) == 2
    val_records = [
        json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if '"type": "val"' in line
    ]
    assert val_records and all(math.isfinite(r["val_bpb"]) for r in val_records)

    # the byte table itself: special tokens weigh 0, and the byte-level BPE's
    # token lengths add up to the document's UTF-8 length exactly
    meta = json.loads(sorted((run_dir / "checkpoints").glob("meta_*.json"))[-1].read_text())
    tok = checkpoint_tokenizer(run_dir / "checkpoints", meta)
    table = tok.token_bytes()
    assert len(table) == tok.get_vocab_size()
    assert table[tok.get_bos_token_id()] == 0
    doc = "TASK arith CMP 3.10e+01 9.00e-03 OUT"
    assert sum(table[i] for i in tok.encode(doc)) == len(doc.encode("utf-8"))
    assert min(table[i] for i in tok.encode(doc)) >= 1


def test_depth_telemetry_lands_in_metrics_and_summary(monkeypatch, tmp_path):
    """Per-block gradient norms and activation RMS are recorded on every logged
    step and summarized under results.depth_telemetry: the per-layer telemetry
    the gradient-stability and norm-stability hypotheses register against."""
    import math

    summary = _run_train(monkeypatch, tmp_path, "depth-telemetry", max_steps=3, n_layer=2, log_interval=1)
    run_dir = tmp_path / "artifacts" / "baseline" / "nanochat" / "depth-telemetry"
    steps = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    steps = [rec for rec in steps if rec.get("type") == "step"]
    assert len(steps) == 3
    for rec in steps:
        # step 0 legitimately reports zero block gradients: lm_head is
        # zero-initialized, so nothing flows back into the blocks until the
        # first optimizer step has moved it
        assert len(rec["grad_norm_by_block"]) == 2 and all(
            math.isfinite(x) and x >= 0 for x in rec["grad_norm_by_block"]
        )
        assert len(rec["activation_rms_by_block"]) == 2 and all(
            math.isfinite(x) and x > 0 for x in rec["activation_rms_by_block"]
        )
    assert all(x > 0 for x in steps[-1]["grad_norm_by_block"]), "block gradients must be live after the first step"
    telemetry = summary["results"]["depth_telemetry"]
    assert telemetry["logged_steps"] == 3
    assert telemetry["grad_norm_spike_ratio"] >= 1.0
    assert len(telemetry["grad_norm_by_block_final"]) == 2 and len(telemetry["activation_rms_by_block_mean"]) == 2
    assert telemetry["grad_norm_depth_ratio"] > 0 and telemetry["activation_rms_depth_ratio"] > 0
