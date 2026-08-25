"""Tests for the scaling sweep harness (bead w94.1).

The sweep trains via subprocess (a real smoke run belongs to the CI-lite
composition, not the unit suite); here we test the pure decision layer the
same way tests/test_bench.py does: ladder invariants, per-mechanism GQA
rules, budget derivation math, command construction, manifest persistence,
resume semantics (skip done work, D1-continue interrupted work), mismatch
refusal, and report rendering — against a monkeypatched launcher and
synthetic train summaries in a temp artifacts tree.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

import cli

runner = CliRunner()


# ---------------------------------------------------------------------------
# helpers


def _flag(cmd: list[str], name: str) -> str | None:
    """Value of a flag in a train argv, or None."""
    if name not in cmd:
        return None
    return cmd[cmd.index(name) + 1]


def _suite_dir(artifacts: Path, mechanism: str, run_id: str) -> Path:
    return artifacts / "scaling" / mechanism / run_id


def _seed_dir(artifacts: Path, mechanism: str, run_id: str, rung_index: int, seed: int) -> Path:
    return _suite_dir(artifacts, mechanism, run_id) / f"rung_{rung_index}" / f"seed_{seed}"


def _write_summary(run_dir: Path, *, losses=(1.5, 1.2, 1.1), val_ce=None, tokens_per_second=1234.0) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"results": {"losses": list(losses), "tokens_per_second": tokens_per_second}}
    if val_ce is not None:
        payload["results"]["val_ce_final"] = val_ce
    (run_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def _ok_launcher(tmp_artifacts: Path):
    """Launcher stub that 'trains' successfully by writing a summary."""

    def launch(cmd: list[str], *, timeout_s: float) -> tuple[int, str, str]:
        topic = _flag(cmd, "--artifacts-topic")
        seed_run_id = _flag(cmd, "--run-id")
        assert topic is not None and seed_run_id is not None
        run_dir = tmp_artifacts / "scaling" / topic / seed_run_id
        _write_summary(run_dir)
        (run_dir / "metrics.jsonl").write_text('{"type": "step", "step": 0}\n', encoding="utf-8")
        return 0, "out", ""

    return launch


def _invoke(argv: list[str], monkeypatch: pytest.MonkeyPatch, launcher) -> Result:
    monkeypatch.setattr(cli, "_scaling_launch_train", launcher)
    return runner.invoke(cli.app, ["scaling-sweep", *argv])


# ---------------------------------------------------------------------------
# ladder + feasibility pure logic


def test_ladder_tables_are_well_formed():
    for ladder_name, rungs in cli.SCALING_LADDERS.items():
        assert len(rungs) >= 2, ladder_name
        widths = [rung["n_embd"] for rung in rungs]
        assert widths == sorted(widths)
        for rung in rungs:
            n_head = rung["n_embd"] // cli.SCALING_HEAD_DIM
            assert rung["n_embd"] % cli.SCALING_HEAD_DIM == 0
            # even heads everywhere: reversible's hard constraint
            assert n_head % 2 == 0
            # constant aspect ratio n_embd / n_layer == head_dim (lockstep)
            assert rung["n_layer"] * cli.SCALING_HEAD_DIM == rung["n_embd"]
            # smoke pins steps; research ladders derive budgets from tokens
            if ladder_name == "smoke":
                assert rung.get("max_steps") is not None
            else:
                assert rung.get("max_steps") is None


def test_kv_heads_reversible_halves_others_do_not():
    assert cli._scaling_kv_heads("reversible", 8) == 4
    assert cli._scaling_kv_heads("standard", 8) == 8
    assert cli._scaling_kv_heads("tropical", 2) == 2


def test_feasibility_rows_measure_exact_params_and_budgets():
    rows = cli._scaling_feasibility_rows("standard", "smoke", batch_size=8, sequence_len=64, token_multiplier=20.0)
    assert [r["name"] for r in rows] == ["smoke_6M", "smoke_14M"]
    for r in rows:
        assert r["feasible"] and r["param_count"] > 0 and r["flops_per_token_est"] > 0
        assert r["target_flops_est"] is None  # smoke pins steps instead
        assert r["planned_max_steps"] == r["max_steps"]
    # research ladders derive Chinchilla-style budgets; check the arithmetic
    rows = cli._scaling_feasibility_rows("standard", "small", batch_size=8, sequence_len=256, token_multiplier=20.0)
    r0 = rows[0]
    tokens = int(round(20.0 * r0["param_count"]))
    assert r0["token_budget"] == tokens
    assert r0["planned_max_steps"] == -(-tokens // (8 * 256))  # ceil division
    assert r0["target_flops_est"] == r0["flops_per_token_est"] * tokens


def test_infeasible_rung_reports_reason_not_crash(monkeypatch):
    # The feasibility gate must convert a construction-time ValueError into an
    # explicit infeasible row (reason recorded), never crash the sweep table.
    import nanochat.gpt as gpt_mod

    def explode(_cfg):
        raise ValueError("synthetic constraint violation")

    monkeypatch.setattr(gpt_mod, "GPT", explode)
    rows = cli._scaling_feasibility_rows("standard", "smoke", batch_size=8, sequence_len=64, token_multiplier=20.0)
    assert len(rows) == 2
    assert all(not r["feasible"] and "synthetic constraint violation" in str(r["reason"]) for r in rows)


# ---------------------------------------------------------------------------
# command construction


def _build_cmd(
    row: dict,
    artifacts_dir: Path,
    *,
    val_interval: int = 0,
    val_batches: int = 10,
    checkpoint_interval: int = 500,
    continue_from_checkpoint: bool = False,
) -> list[str]:
    return cli._scaling_train_command(
        row,
        mechanism="tropical",
        suite_run_id="R",
        seed=0,
        device="cpu",
        batch_size=8,
        sequence_len=256,
        learning_rate=6e-4,
        optimizer_type="adamw",
        warmup_steps=0,
        val_interval=val_interval,
        val_batches=val_batches,
        artifacts_dir=artifacts_dir,
        checkpoint_interval=checkpoint_interval,
        checkpoint_keep=1,
        data_dir=None,
        auto_download_data=True,
        min_parquet_files=2,
        continue_from_checkpoint=continue_from_checkpoint,
    )


def test_command_targets_flops_for_research_ladder(tmp_path):
    row = {
        "index": 1,
        "name": "14M",
        "n_layer": 4,
        "n_embd": 128,
        "n_head": 4,
        "n_kv_head": 4,
        "planned_max_steps": 1000,
        "target_flops_est": 123456,
    }
    cmd = _build_cmd(row, tmp_path)
    assert cmd[:3] == [cmd[0], "-m", "nanochat.train"]
    assert _flag(cmd, "--target-flops") == "123456"
    assert "--max-steps" not in cmd
    assert _flag(cmd, "--artifacts-topic") == "tropical/R/rung_1"
    assert _flag(cmd, "--run-id") == "seed_0"
    assert "--val-interval" not in cmd
    assert _flag(cmd, "--checkpoint-interval") == "500"
    assert "--resume-from" not in cmd


def test_command_pins_steps_for_smoke_and_gates_val_and_resume(tmp_path):
    row = {
        "index": 0,
        "name": "s",
        "n_layer": 2,
        "n_embd": 64,
        "n_head": 2,
        "n_kv_head": 1,
        "planned_max_steps": 25,
        "target_flops_est": None,
    }
    cmd = _build_cmd(row, tmp_path)
    assert _flag(cmd, "--max-steps") == "25"
    assert "--target-flops" not in cmd
    cmd_val = _build_cmd(row, tmp_path, val_interval=50, val_batches=7)
    assert _flag(cmd_val, "--val-interval") == "50" and _flag(cmd_val, "--val-batches") == "7"
    cmd_resume = _build_cmd(row, tmp_path, continue_from_checkpoint=True)
    ckpt_flag = _flag(cmd_resume, "--checkpoint-dir")
    assert ckpt_flag is not None and ckpt_flag.endswith("rung_0/seed_0/checkpoints")
    cmd_nockpt = _build_cmd(row, tmp_path, checkpoint_interval=0)
    assert "--checkpoint-interval" not in cmd_nockpt


# ---------------------------------------------------------------------------
# end-to-end sweep behavior over the monkeypatched launcher


def _sweep_argv(run_id: str, *, seeds=1, extra=()):
    return [
        "--mechanism",
        "tropical",
        "--ladder",
        "smoke",
        "--device",
        "cpu",
        "--seeds",
        str(seeds),
        "--artifacts-dir",
        "artifacts",
        "--run-id",
        run_id,
        "--no-auto-download-data",
        *extra,
    ]


def test_dry_run_launches_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def boom(cmd, *, timeout_s):
        raise AssertionError("dry run must not launch training")

    result = _invoke(_sweep_argv("DRY", extra=("--dry-run",)), monkeypatch, boom)
    assert result.exit_code == 0
    assert not (tmp_path / "artifacts").exists()


def test_full_sweep_marks_all_done_and_writes_manifest_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    launch = _ok_launcher(tmp_path / "artifacts")
    result = _invoke(_sweep_argv("OK1"), monkeypatch, launch)
    assert result.exit_code == 0
    suite = _suite_dir(tmp_path / "artifacts", "tropical", "OK1")
    manifest = json.loads((suite / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert all(r["status"] == "done" for r in manifest["rungs"])
    run0 = manifest["rungs"][0]["runs"][0]
    assert run0["status"] == "done" and run0["metrics"]["final_loss"] == pytest.approx(1.1)
    report = (suite / "report.md").read_text()
    assert "| rung |" in report and "smoke_6M" in report
    logs = list((suite / "logs").glob("*.stdout.txt"))
    assert len(logs) == len(manifest["rungs"])  # one seed per rung


def test_failed_training_marks_failed_exits_nonzero_keeps_going(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []

    def launch(cmd: list[str], *, timeout_s: float) -> tuple[int, str, str]:
        calls.append(cmd)
        return 1, "", "boom: simulated training failure"

    result = _invoke(_sweep_argv("FAIL1"), monkeypatch, launch)
    assert result.exit_code == 1
    assert len(calls) == 2  # continued past the first failure
    suite = _suite_dir(tmp_path / "artifacts", "tropical", "FAIL1")
    manifest = json.loads((suite / "manifest.json").read_text())
    assert all(r["status"] == "failed" for r in manifest["rungs"])
    assert any("FAILED" in line for line in (suite / "report.md").read_text().splitlines())


def test_resume_skips_done_and_continues_interrupted_with_d1(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifacts = tmp_path / "artifacts"

    # generation 1: rung 0 succeeds, rung 1 fails mid-training WITH checkpoints
    real_seed_dir = _seed_dir(artifacts, "tropical", "RES1", 1, 0)
    (real_seed_dir / "checkpoints").mkdir(parents=True)
    (real_seed_dir / "checkpoints" / "step_000010.pt").touch()

    def mixed_launcher(cmd: list[str], *, timeout_s: float) -> tuple[int, str, str]:
        topic = _flag(cmd, "--artifacts-topic")
        assert topic is not None
        idx = int(topic.split("rung_")[1])
        if idx == 0:
            run_id_flag = _flag(cmd, "--run-id")
            assert run_id_flag is not None
            run_dir = artifacts / "scaling" / topic / run_id_flag
            _write_summary(run_dir)
            return 0, "", ""
        return 1, "", "interrupted"

    result = _invoke(_sweep_argv("RES1"), monkeypatch, mixed_launcher)
    assert result.exit_code == 1

    # generation 2: same --run-id, everything succeeds now
    seen_cmds: list[list[str]] = []
    launch2 = _ok_launcher(artifacts)

    def recording_launcher(cmd: list[str], *, timeout_s: float):
        seen_cmds.append(cmd)
        return launch2(cmd, timeout_s=timeout_s)

    result = _invoke(_sweep_argv("RES1"), monkeypatch, recording_launcher)
    assert result.exit_code == 0
    topics_launched = [_flag(c, "--artifacts-topic") for c in seen_cmds]
    assert topics_launched == ["tropical/RES1/rung_1"], "done rung must be skipped, failed rung retried"
    resumed_cmd = seen_cmds[0]
    assert _flag(resumed_cmd, "--resume-from") == "latest", "checkpoint present => D1 continuation"
    manifest = json.loads((_suite_dir(artifacts, "tropical", "RES1") / "manifest.json").read_text())
    assert all(r["status"] == "done" for r in manifest["rungs"])


def test_resume_refuses_mismatched_sweep_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    launch = _ok_launcher(tmp_path / "artifacts")
    result = _invoke(
        [
            "--mechanism",
            "braid",
            "--ladder",
            "smoke",
            "--device",
            "cpu",
            "--seeds",
            "1",
            "--artifacts-dir",
            "artifacts",
            "--run-id",
            "MIX1",
            "--no-auto-download-data",
        ],
        monkeypatch,
        launch,
    )
    assert result.exit_code == 0
    result = _invoke(
        [
            "--mechanism",
            "braid",
            "--ladder",
            "smoke",
            "--device",
            "cpu",
            "--batch-size",
            "16",
            "--artifacts-dir",
            "artifacts",
            "--run-id",
            "MIX1",
            "--no-auto-download-data",
        ],
        monkeypatch,
        launch,
    )
    assert result.exit_code == 2  # refuses to mix sweeps under one run-id


def test_fresh_flag_ignores_stored_statuses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    launch = _ok_launcher(tmp_path / "artifacts")
    assert _invoke(_sweep_argv("FRSH"), monkeypatch, launch).exit_code == 0
    calls: list[list[str]] = []

    def counting(cmd, *, timeout_s):
        calls.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(cli, "_scaling_launch_train", counting)
    result = runner.invoke(cli.app, ["scaling-sweep", *_sweep_argv("FRSH", extra=("--fresh-sweep",))])
    assert result.exit_code == 0
    assert len(calls) == 2, "--fresh-sweep must retrain every rung"


# ---------------------------------------------------------------------------
# metric extraction + status derivation


def test_extract_metrics_reads_synthetic_summary(tmp_path):
    run_dir = tmp_path / "run"
    _write_summary(run_dir, losses=(2.0, 1.5), val_ce=1.234, tokens_per_second=999.0)
    m = cli._scaling_extract_metrics(run_dir / "summary.json")
    assert m["final_loss"] == pytest.approx(1.5)
    assert m["val_ce_final"] == pytest.approx(1.234)
    assert m["tokens_per_second"] == pytest.approx(999.0)
    missing = cli._scaling_extract_metrics(tmp_path / "nope.json")
    assert missing == {"final_loss": None, "val_ce_final": None, "tokens_per_second": None}


def test_rung_status_derivation():
    done = [{"status": "done"}, {"status": "done"}]
    assert cli._scaling_rung_status(done, feasible=True) == "done"
    assert cli._scaling_rung_status([{"status": "failed"}, {"status": "pending"}], feasible=True) == "failed"
    assert cli._scaling_rung_status([{"status": "pending"}], feasible=True) == "pending"
    assert cli._scaling_rung_status([], feasible=False) == "infeasible"
