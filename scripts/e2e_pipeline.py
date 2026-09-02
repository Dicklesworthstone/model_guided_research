"""End-to-end research-loop pipeline scenarios (bead rz8.8).

Every other test in this repo exercises one command; integration failures
live where commands COMPOSE (artifact schema drift, checkpoint format
mismatches, CLI flag renames, separator semantics). These scenarios run the
full loop the way a user would, on tiny CPU budgets, with logging rich enough
that any failure is diagnosable WITHOUT rerunning.

Scenarios (run one with --scenario, or all):
  full-loop       gen-tasks -> train standard+tropical (SIGKILL one mid-run,
                  resume it) -> certify -> eval-tasks -> sample -> regressions
                  -> adjudicate --dry-run (G2 maintenance-contract stage: the
                  engine must REFUSE this under-seeded evidence with
                  machine-readable reasons, never rule on it)
  resume          randomized-kill resume robustness: SIGKILL at a random step
                  (logged seed), resume, assert step coverage 0..N-1 with no
                  holes and the planned final step reached
  determinism     greedy same-seed sampling across the mechanism zoo is
                  byte-identical across two invocations; tokens/s recorded
                  (octonion SKIPPED until 7b0.6 vectorizes its loop)
  regression-gate bench-fixed-flops two variants -> mgr regressions passes on
                  self-vs-self and TRIPS (--fail-on-regression, exit 1) on a
                  deliberately degraded fixture copy
  scorecard       one standard x placebo x seed fixed-FLOPs cell -> atomic
                  manifest, eval evidence, ci-v6 report JSON/Markdown/HTML
  word-problem    (bead u55.3) group word-problem mini-eval: gen-tasks group ->
                  train tiny braid/rmatrix + tiny standard -> eval-tasks both
                  -> assert length_slope (held-out OLS + per-group breakdown)
                  and conserved-charge telemetry land in the artifacts. The
                  assertion is that the harness runs and produces slope
                  tables, NOT that the tiny models separate.

Reports: <workdir>/reports/<scenario>/e2e_report.{md,json} - stage matrix
(pass/fail/skipped), durations, artifact inventory, repro commands. Exit is
nonzero if any required stage fails. The workdir lives OUTSIDE the repo so
runs never dirty the tree (provenance hygiene).

MAINTENANCE CONTRACT (binding, from the bead): G2/C4/E1-class beads extend
these scenarios with their stage as part of their OWN acceptance criteria.
Append-only infrastructure - never allowed to go stale.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, str(REPO / "cli.py")]
TRAIN = [sys.executable, "-m", "nanochat.train"]
PROMPT = "TASK arith CMP 1.00e-02 2.00e+03 OUT"
TINY_MODEL = [
    "--n-layer",
    "1",
    "--n-head",
    "2",
    "--n-kv-head",
    "2",
    "--n-embd",
    "32",
    "--sequence-len",
    "64",
    "--batch-size",
    "4",
    "--device",
    "cpu",
    "--warmup-steps",
    "0",
]

console = Console()


class StageFailure(Exception):
    pass


@dataclass
class Stage:
    name: str
    cmd: str
    status: str  # pass | fail | skipped
    elapsed_s: float
    note: str = ""
    log: str = ""


@dataclass
class Runner:
    workdir: Path
    scenario: str
    stages: list[Stage] = field(default_factory=list)

    @property
    def logdir(self) -> Path:
        d = self.workdir / "logs" / self.scenario
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "4")
        return env

    def skip(self, name: str, reason: str) -> None:
        console.print(Panel(f"[yellow]SKIPPED[/yellow]: {reason}", title=f"stage: {name}", border_style="yellow"))
        self.stages.append(Stage(name, "", "skipped", 0.0, note=reason))

    def run(
        self,
        name: str,
        argv: list[str],
        *,
        required: bool = True,
        expect_exit: int = 0,
        timeout: int = 1200,
        cwd: Path | None = None,
    ) -> str:
        """Run one stage to completion. Returns captured output. Raises
        StageFailure (after writing diagnostics) when a required stage fails."""
        cmd_str = shlex.join(argv)
        console.print(Panel(f"[bold]{cmd_str}[/bold]", title=f"BEGIN {self.scenario}/{name}", border_style="cyan"))
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout, cwd=cwd or REPO, env=self._env()
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            code = proc.returncode
        except subprocess.TimeoutExpired as exc:
            out = ((exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")) + "\nTIMEOUT"
            code = -1
        elapsed = time.perf_counter() - t0
        log_path = self.logdir / f"{name}.log"
        log_path.write_text(out)
        ok = code == expect_exit
        # an OPTIONAL stage that fails is reported SKIPPED (spec: never a
        # scenario failure, never silently absent)
        status = "pass" if ok else ("fail" if required else "skipped")
        color = {"pass": "green", "fail": "red", "skipped": "yellow"}[status]
        note = "" if ok else f"exit={code} (expected {expect_exit})" + ("" if required else " - optional, non-blocking")
        console.print(
            Panel(
                f"[{color}]{status.upper()}[/{color}] exit={code} (expected {expect_exit}) · {elapsed:.1f}s · log={log_path}",
                title=f"END {self.scenario}/{name}",
                border_style=color,
            )
        )
        self.stages.append(Stage(name, cmd_str, status, elapsed, note=note, log=str(log_path)))
        if not ok and required:
            tail = "\n".join(out.splitlines()[-50:])
            console.print(Panel(tail or "(no output)", title=f"FAILURE TAIL — {name}", border_style="red"))
            probe = self._doctor_probe()
            console.print(Panel(probe, title="env probe", border_style="red"))
            console.print(Panel(f"cd {REPO} && {cmd_str}", title="repro command", border_style="red"))
            self.stages[-1].note = f"exit={code}, expected {expect_exit}; see {log_path}"
            raise StageFailure(name)
        return out

    def _doctor_probe(self) -> str:
        try:
            proc = subprocess.run(
                CLI + ["doctor", "--json"], capture_output=True, text=True, timeout=120, cwd=REPO, env=self._env()
            )
            return proc.stdout[-1500:] or proc.stderr[-500:]
        except Exception as exc:  # noqa: BLE001 - probe is best-effort by design
            return f"mgr doctor unavailable: {exc}; python={sys.version.split()[0]} cwd={REPO}"

    def assert_true(self, name: str, cond: bool, detail: str) -> None:
        """A pure-python check recorded as a stage (artifact contracts etc.)."""
        status = "pass" if cond else "fail"
        color = "green" if cond else "red"
        console.print(
            Panel(f"[{color}]{status.upper()}[/{color}]: {detail}", title=f"check: {name}", border_style=color)
        )
        self.stages.append(Stage(name, f"(check) {detail}", status, 0.0))
        if not cond:
            raise StageFailure(name)

    def report(self) -> bool:
        """Write e2e_report.{md,json}; return True when no required stage failed."""
        rdir = self.workdir / "reports" / self.scenario
        rdir.mkdir(parents=True, exist_ok=True)
        inventory = sorted(
            f"{p.relative_to(self.workdir)} ({p.stat().st_size:,}B)" for p in self.workdir.rglob("summary.json")
        )
        payload = {
            "scenario": self.scenario,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "workdir": str(self.workdir),
            "stages": [vars(s) for s in self.stages],
            "artifact_inventory": inventory,
            "ok": all(s.status != "fail" for s in self.stages),
        }
        (rdir / "e2e_report.json").write_text(json.dumps(payload, indent=2) + "\n")
        rows = "\n".join(
            f"| {s.name} | {s.status} | {s.elapsed_s:.1f}s | {s.note or s.cmd[:90]} |" for s in self.stages
        )
        (rdir / "e2e_report.md").write_text(
            f"# e2e {self.scenario} — {payload['generated_at']}\n\n"
            f"workdir: `{self.workdir}`\n\n| stage | status | elapsed | detail |\n|---|---|---|---|\n{rows}\n\n"
            "## artifact inventory\n" + "\n".join(f"- {line}" for line in inventory) + "\n"
        )
        table = Table(title=f"e2e {self.scenario} — stage matrix", border_style="cyan")
        for col in ("stage", "status", "elapsed"):
            table.add_column(col)
        for s in self.stages:
            style = {"pass": "green", "fail": "red", "skipped": "yellow"}[s.status]
            table.add_row(s.name, f"[{style}]{s.status}[/{style}]", f"{s.elapsed_s:.1f}s")
        console.print(table)
        console.print(f"[bold]report →[/bold] {rdir}")
        return payload["ok"]


# ---------------------------------------------------------------------------
# shared helpers


def _train_argv(
    work: Path, run_id: str, mech: str, *, max_steps: int, ckpt_interval: int, data_dir: Path, seed: int = 7
) -> list[str]:
    return (
        TRAIN
        + TINY_MODEL
        + [
            "--attention-type",
            mech,
            "--data-dir",
            str(data_dir),
            "--max-steps",
            str(max_steps),
            "--checkpoint-interval",
            str(ckpt_interval),
            "--seed",
            str(seed),
            "--artifacts-dir",
            str(work / "artifacts"),
            "--artifacts-kind",
            "e2e",
            "--artifacts-topic",
            "loop",
            "--run-id",
            run_id,
        ]
    )


def _run_dir(work: Path, run_id: str) -> Path:
    return work / "artifacts" / "e2e" / "loop" / run_id


def _metric_lines(run_dir: Path) -> list[dict[str, Any]]:
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        return []
    out = []
    for line in metrics.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _metric_steps(run_dir: Path) -> list[int]:
    return [r["step"] for r in _metric_lines(run_dir) if isinstance(r.get("step"), int)]


def _ckpt_steps(run_dir: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in (run_dir / "checkpoints").glob("meta_*.json"))


def _kill_train_mid_run(
    r: Runner, name: str, argv: list[str], run_dir: Path, kill_at_step: int, timeout: int = 600
) -> None:
    """Start a training run, SIGKILL it once metrics.jsonl reaches kill_at_step."""
    cmd_str = shlex.join(argv)
    console.print(
        Panel(
            f"[bold]{cmd_str}[/bold]\nSIGKILL at checkpoint-on-disk >= step {kill_at_step}",
            title=f"BEGIN {r.scenario}/{name}",
            border_style="cyan",
        )
    )
    t0 = time.perf_counter()
    log_path = r.logdir / f"{name}.log"
    with log_path.open("w") as log_fh:
        proc = subprocess.Popen(argv, stdout=log_fh, stderr=subprocess.STDOUT, cwd=REPO, env=r._env())
        killed = False
        while time.perf_counter() - t0 < timeout:
            if proc.poll() is not None:
                break  # finished before we could kill it - widen the window
            # trigger on checkpoints (flushed atomically at the interval), not
            # on metrics.jsonl - step records buffer for up to flush_every
            # steps so the metrics file lags actual progress
            ckpts = _ckpt_steps(run_dir) if (run_dir / "checkpoints").exists() else []
            if ckpts and max(ckpts) >= kill_at_step:
                os.kill(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
                killed = True
                break
            time.sleep(0.05)
    elapsed = time.perf_counter() - t0
    interrupted = killed and not (run_dir / "summary.json").exists()
    note = (
        f"killed at checkpoint>={kill_at_step}"
        if interrupted
        else (
            "kill landed after training completed (summary.json exists) - widen the window"
            if killed
            else "run finished before kill"
        )
    )
    status = "pass" if interrupted else "fail"
    r.stages.append(Stage(name, cmd_str, status, elapsed, note=note))
    console.print(
        Panel(
            f"{'[green]PASS' if interrupted else '[red]FAIL'}[/] · {elapsed:.1f}s · {note}",
            title=f"END {r.scenario}/{name}",
            border_style="green" if interrupted else "red",
        )
    )
    if not interrupted:
        raise StageFailure(name)


def _resume_and_verify(
    r: Runner, name: str, work: Path, run_id: str, mech: str, *, max_steps: int, ckpt_interval: int, data_dir: Path
) -> None:
    run_dir = _run_dir(work, run_id)
    argv = _train_argv(work, run_id, mech, max_steps=max_steps, ckpt_interval=ckpt_interval, data_dir=data_dir) + [
        "--resume-from",
        "latest",
        "--checkpoint-dir",
        str(run_dir / "checkpoints"),
    ]
    r.run(name, argv)
    # Trajectory continuity across the SIGKILL splice. Honest contract:
    #  - the pre-kill segment survives (the original "header" is still there;
    #    a SIGKILL may lose up to flush_every buffered step records, so the
    #    first segment's TAIL can legitimately be missing),
    #  - the splice is marked by a "resume_header" record,
    #  - the resumed segment is contiguous and reaches the planned final step.
    lines = _metric_lines(run_dir)
    headers = [i for i, rec in enumerate(lines) if rec.get("type") == "header"]
    resumes = [i for i, rec in enumerate(lines) if rec.get("type") == "resume_header"]
    r.assert_true(
        f"{name}-splice",
        len(headers) == 1 and len(resumes) >= 1,
        f"history preserved across resume: headers={len(headers)} resume_headers={len(resumes)}",
    )
    tail = [rec["step"] for rec in lines[resumes[-1] :] if isinstance(rec.get("step"), int)]
    contiguous = all(b == a + 1 for a, b in zip(tail, tail[1:], strict=False))
    r.assert_true(
        f"{name}-continuity",
        bool(tail) and contiguous and tail[-1] == max_steps - 1,
        f"resumed segment {tail[:3]}..{tail[-3:] if len(tail) > 3 else tail} contiguous={contiguous} "
        f"final={tail[-1] if tail else None} (planned {max_steps - 1})",
    )


# ---------------------------------------------------------------------------
# scenarios


def scenario_full_loop(work: Path) -> bool:
    r = Runner(work, "full-loop")
    steps_total, interval = 30, 10
    data_dir = work / "diag" / "arith"
    try:
        r.run("doctor", CLI + ["doctor", "--json"], required=False, timeout=180)
        r.run(
            "gen-tasks",
            CLI + ["gen-tasks", "--task", "arith", "--out", str(work / "diag"), "--size", "300", "--seed", "7"],
        )
        # train the standard arm and SIGKILL it mid-run; the chain then proves
        # resume completes it to the planned budget
        argv_std = _train_argv(
            work, "e2e-standard", "standard", max_steps=steps_total, ckpt_interval=interval, data_dir=data_dir
        )
        _kill_train_mid_run(
            r, "train-standard-kill", argv_std, _run_dir(work, "e2e-standard"), kill_at_step=interval - 1
        )
        _resume_and_verify(
            r,
            "train-standard-resume",
            work,
            "e2e-standard",
            "standard",
            max_steps=steps_total,
            ckpt_interval=interval,
            data_dir=data_dir,
        )
        r.run(
            "train-tropical",
            _train_argv(
                work, "e2e-tropical", "tropical", max_steps=steps_total, ckpt_interval=interval, data_dir=data_dir
            ),
        )
        r.run(
            "certify",
            CLI
            + [
                "certify",
                "-m",
                "standard",
                "-m",
                "tropical",
                "--artifacts-dir",
                str(work / "artifacts"),
                "--run-id",
                "e2e-cert",
            ],
            timeout=1800,
        )
        for arm in ("standard", "tropical"):
            r.run(
                f"eval-{arm}",
                CLI
                + [
                    "eval-tasks",
                    "--checkpoint",
                    str(_run_dir(work, f"e2e-{arm}") / "checkpoints"),
                    "--task",
                    "arith",
                    "--seeds",
                    "0",
                    "--examples",
                    "6",
                    "--artifacts-dir",
                    str(work / "artifacts"),
                    "--run-id",
                    f"e2e-eval-{arm}",
                ],
            )
            summary = work / "artifacts" / "evals" / "tasks" / f"e2e-eval-{arm}" / "summary.json"
            rec = json.loads(summary.read_text())
            r.assert_true(
                f"eval-{arm}-contract",
                rec["schema_version"] == "mgr.evaltasks.v4"
                and "answer_prior" in rec["tasks"]["arith"]
                and (summary.parent / "generations.jsonl").exists(),
                f"{summary.name}: v2 schema + answer_prior + receipts present",
            )
        out = r.run(
            "sample",
            CLI
            + [
                "sample",
                "--checkpoint",
                str(_run_dir(work, "e2e-standard") / "checkpoints"),
                "--prompt",
                PROMPT,
                "--max-tokens",
                "8",
                "--json",
            ],
        )
        r.assert_true(
            "sample-contract",
            '"results"' in out and '"tokens_per_s"' in out,
            "sample --json emits the results contract",
        )
        r.run(
            "regressions",
            CLI
            + [
                "regressions",
                "--baseline",
                str(_run_dir(work, "e2e-standard")),
                "--candidate",
                str(_run_dir(work, "e2e-tropical")),
                "--artifacts-dir",
                str(work / "artifacts"),
                "--run-id",
                "e2e-reg",
                "--no-html",
                "--no-fail-on-regression",
            ],
        )
        # G2 maintenance-contract stage: the verdict engine must REFUSE this
        # under-seeded, single-seed evidence - blocked with machine-readable
        # reasons is the PASSING outcome here
        out = r.run(
            "adjudicate-dry-run",
            CLI
            + [
                "adjudicate",
                "--all",
                "--dry-run",
                "--artifacts",
                str(work / "artifacts"),
                "--artifacts-dir",
                str(work / "artifacts"),
                "--run-id",
                "e2e",
            ],
        )
        verdicts = json.loads((work / "artifacts" / "adjudications" / "e2e" / "verdicts.json").read_text())
        bad = [v for v in verdicts["verdicts"] if v["verdict"] == "blocked" and not v.get("reason_code")]
        ruled = [v for v in verdicts["verdicts"] if v["verdict"] not in ("blocked",)]
        r.assert_true(
            "adjudicate-refusal-integrity",
            not bad and not ruled,
            f"engine refused all {len(verdicts['verdicts'])} hypotheses with machine-readable reasons "
            f"(ruled={len(ruled)}, missing-reason={len(bad)})",
        )
    except StageFailure:
        pass
    return r.report()


def scenario_resume(work: Path, seed: int | None) -> bool:
    r = Runner(work, "resume")
    rng_seed = seed if seed is not None else random.SystemRandom().randint(0, 10_000)
    rng = random.Random(rng_seed)
    steps_total, interval = 48, 6
    # kill triggers on a checkpoint hitting disk, so pick a random checkpoint
    # boundary well short of the end (leave >= 2 intervals of runway)
    kill_at = rng.randrange(interval - 1, steps_total - 2 * interval, interval)
    console.print(
        Panel(
            f"randomized kill: seed={rng_seed} kill_at_step={kill_at}",
            title="resume scenario parameters",
            border_style="magenta",
        )
    )
    data_dir = work / "diag" / "arith"
    try:
        if not data_dir.exists():
            r.run(
                "gen-tasks",
                CLI + ["gen-tasks", "--task", "arith", "--out", str(work / "diag"), "--size", "300", "--seed", "7"],
            )
        argv = _train_argv(
            work, "e2e-resume", "standard", max_steps=steps_total, ckpt_interval=interval, data_dir=data_dir
        )
        _kill_train_mid_run(r, "train-kill", argv, _run_dir(work, "e2e-resume"), kill_at_step=kill_at)
        _resume_and_verify(
            r,
            "train-resume",
            work,
            "e2e-resume",
            "standard",
            max_steps=steps_total,
            ckpt_interval=interval,
            data_dir=data_dir,
        )
    except StageFailure:
        pass
    return r.report()


DETERMINISM_MECHS = [
    "standard",
    "ultrametric",
    "simplicial",
    "quaternion",
    "braid",
    "fractal",
    "surreal",
    "tropical",
    "reversible",
    "gauge",
]


def scenario_determinism(work: Path) -> bool:
    r = Runner(work, "determinism")
    data_dir = work / "diag" / "arith"
    rates: list[tuple[str, float]] = []
    try:
        if not data_dir.exists():
            r.run(
                "gen-tasks",
                CLI + ["gen-tasks", "--task", "arith", "--out", str(work / "diag"), "--size", "300", "--seed", "7"],
            )
        r.skip("octonion", "excluded until 7b0.6 vectorizes the octonion loop (slow at any size)")
        for mech in DETERMINISM_MECHS:
            r.run(
                f"train-{mech}",
                _train_argv(work, f"e2e-det-{mech}", mech, max_steps=12, ckpt_interval=12, data_dir=data_dir),
                timeout=900,
            )
            texts = []
            for rep in (1, 2):
                out = r.run(
                    f"sample-{mech}-{rep}",
                    CLI
                    + [
                        "sample",
                        "--checkpoint",
                        str(_run_dir(work, f"e2e-det-{mech}") / "checkpoints"),
                        "--prompt",
                        PROMPT,
                        "--max-tokens",
                        "8",
                        "--seed",
                        "3",
                        "--no-stop-at-separator",
                        "--json",
                    ],
                )
                payload = json.loads(out[out.index("{") :])
                texts.append(payload["results"][0]["text"])
                if rep == 2:
                    rates.append((mech, payload["results"][0]["tokens_per_s"]))
            r.assert_true(
                f"determinism-{mech}", texts[0] == texts[1], f"{mech}: greedy same-seed outputs byte-identical"
            )
        table = Table(title="greedy decode tokens/s by mechanism", border_style="cyan")
        table.add_column("mechanism")
        table.add_column("tokens/s", justify="right")
        for mech, tps in rates:
            table.add_row(mech, f"{tps:.1f}")
        console.print(table)
    except StageFailure:
        pass
    return r.report()


def scenario_word_problem(work: Path) -> bool:
    """Bead u55.3 acceptance stage: the word-problem mini-eval composes
    gen-tasks group -> train (rmatrix + standard arms) -> eval-tasks with
    per-length slope fits. S_3-scale lengths (dial length=4) keep every
    held-out doc inside the tiny model's 64-token rotary cache on CPU."""
    r = Runner(work, "word-problem")
    data_dir = work / "diag-group" / "group"
    try:
        r.run(
            "gen-tasks",
            CLI
            + [
                "gen-tasks",
                "--task",
                "group",
                "--out",
                str(work / "diag-group"),
                "--size",
                "300",
                "--seed",
                "11",
                "--dial",
                "length=4",
            ],
        )
        argv_rmx = _train_argv(work, "e2e-wp-rmatrix", "braid", max_steps=20, ckpt_interval=10, data_dir=data_dir) + [
            "--braid-crossing-law",
            "rmatrix",
            "--braid-verify",
        ]
        r.run("train-rmatrix", argv_rmx)
        r.run(
            "train-standard",
            _train_argv(work, "e2e-wp-standard", "standard", max_steps=20, ckpt_interval=10, data_dir=data_dir),
        )
        # conserved-charge telemetry must land in the D2 metrics stream for the
        # rmatrix arm: Q1 (mass partition) at fp32 noise, Q2 (braid residual)
        # at fp64 noise - the integrable-vs-heuristic separation observable
        charge_recs = [
            rec
            for rec in _metric_lines(_run_dir(work, "e2e-wp-rmatrix"))
            if rec.get("type") == "step" and "braid_q1_mass_defect_max" in rec
        ]
        r.assert_true(
            "rmatrix-charge-telemetry",
            bool(charge_recs)
            and all(rec["braid_q1_mass_defect_max"] < 1e-5 for rec in charge_recs)
            and all(rec["braid_q2_braid_residual_max"] < 1e-10 for rec in charge_recs),
            f"{len(charge_recs)} step records carry Q1<1e-5 and Q2<1e-10 charge readings",
        )
        for arm in ("rmatrix", "standard"):
            r.run(
                f"eval-{arm}",
                CLI
                + [
                    "eval-tasks",
                    "--checkpoint",
                    str(_run_dir(work, f"e2e-wp-{arm}") / "checkpoints"),
                    "--task",
                    "group",
                    "--dial",
                    "length=4",
                    "--seeds",
                    "0",
                    "--examples",
                    "12",
                    "--artifacts-dir",
                    str(work / "artifacts"),
                    "--run-id",
                    f"e2e-wp-eval-{arm}",
                ],
            )
            summary = work / "artifacts" / "evals" / "tasks" / f"e2e-wp-eval-{arm}" / "summary.json"
            rec = json.loads(summary.read_text())
            slope = rec["tasks"]["group"].get("length_slope")
            held = (slope or {}).get("held_out")
            r.assert_true(
                f"eval-{arm}-slope-contract",
                isinstance(slope, dict)
                and isinstance(held, dict)
                and "slope" in held
                and "ci95" in held
                and held["n_docs"] >= 3
                and isinstance(slope.get("by_category"), dict),
                f"{arm}: length_slope.held_out fit present (n={held.get('n_docs') if held else 0}) "
                "with per-group by_category breakdown",
            )
            run_md = (summary.parent / "run.md").read_text()
            r.assert_true(
                f"eval-{arm}-slope-table", "slope held-out" in run_md, f"{arm}: run.md renders the slope column"
            )
    except StageFailure:
        pass
    return r.report()


def scenario_symplectic(work: Path) -> bool:
    """Bead u55.5 acceptance stage: a symplectic-TIED no-norm mini-run (depth
    8, tiny width, CPU) with shadow-energy/activation-norm telemetry and
    drift assertions. The across-LAYER energy band within each step is the
    conservation observable (tied = the same symplectic map iterated, so the
    coercive H-tilde is conserved across depth by the shadow theorem)."""
    r = Runner(work, "symplectic")
    data_dir = work / "diag-hier" / "hier"
    try:
        r.run(
            "gen-tasks",
            CLI + ["gen-tasks", "--task", "hier", "--out", str(work / "diag-hier"), "--size", "300", "--seed", "11"],
        )
        argv = _train_argv(work, "e2e-symplectic", "reversible", max_steps=24, ckpt_interval=12, data_dir=data_dir) + [
            "--n-layer",
            "8",
            "--n-head",
            "4",
            "--n-kv-head",
            "2",  # depth 8; reversible halves heads
            "--reversible-mode",
            "symplectic",
            "--reversible-tied",
            "--reversible-lambda-min",
            "0.05",
            "--reversible-record-energy",
        ]
        r.run("train-symplectic-tied", argv)
        recs = [
            rec
            for rec in _metric_lines(_run_dir(work, "e2e-symplectic"))
            if rec.get("type") == "step" and "symplectic_shadow_energy_mean" in rec
        ]
        r.assert_true(
            "energy-telemetry-present", len(recs) >= 24, f"{len(recs)} step records carry shadow-energy telemetry"
        )
        finite = all(
            rec["symplectic_shadow_energy_mean"] == rec["symplectic_shadow_energy_mean"]
            and rec["symplectic_x_norm_mean"] == rec["symplectic_x_norm_mean"]
            for rec in recs
        )
        r.assert_true("energy-telemetry-finite", finite, "all shadow-energy/norm readings finite")
        # conservation-across-depth: the per-step across-layer band stays small
        # relative to the energy scale (loose: training moves the potentials
        # BETWEEN steps; the theorem speaks within each forward)
        bands_ok = all(
            rec["symplectic_energy_band_layers"] <= 0.10 * (1.0 + abs(rec["symplectic_shadow_energy_mean"]))
            for rec in recs
        )
        worst = max(
            rec["symplectic_energy_band_layers"] / (1.0 + abs(rec["symplectic_shadow_energy_mean"])) for rec in recs
        )
        r.assert_true(
            "across-layer-energy-band",
            bands_ok,
            f"across-layer shadow-energy band <= 10% of scale at every step (worst {worst:.4f})",
        )
        norms = [rec["symplectic_x_norm_mean"] for rec in recs]
        r.assert_true(
            "activation-norms-bounded",
            max(norms) <= 50.0 * max(norms[0], 1e-9),
            f"activation norms bounded: first {norms[0]:.4f}, max {max(norms):.4f}",
        )
        summary = json.loads((_run_dir(work, "e2e-symplectic") / "summary.json").read_text())
        res = summary.get("results", {})
        r.assert_true(
            "summary-energy-aggregates",
            all(k in res for k in ("symplectic_energy_first", "symplectic_energy_final", "symplectic_x_norm_final")),
            "summary records symplectic energy/norm aggregates",
        )
    except StageFailure:
        pass
    return r.report()


def scenario_regression_gate(work: Path) -> bool:
    r = Runner(work, "regression-gate")
    try:
        r.run(
            "bench",
            CLI
            + [
                "bench-fixed-flops",
                "--run-id",
                "e2e-bench",
                "--device",
                "cpu",
                "--target-flops",
                "2e9",
                "-a",
                "standard",
                "-a",
                "tropical",
                "--n-layer",
                "1",
                "--n-head",
                "2",
                "--n-kv-head",
                "2",
                "--n-embd",
                "32",
                "--sequence-len",
                "64",
                "--batch-size",
                "4",
                "--artifacts-dir",
                str(work / "artifacts"),
            ],
            timeout=1800,
        )
        bench_dirs = sorted((work / "artifacts").rglob("e2e-bench*/summary.json"))
        r.assert_true("bench-artifacts", bool(bench_dirs), f"bench summaries found: {len(bench_dirs)}")
        base = bench_dirs[0].parent
        # suite summaries hold one record per attention_type: variant selectors
        # pick the arm, and --fail-on-missing forbids a vacuous pass where
        # every metric reads 'missing' (the trap this stage originally fell in)
        r.run(
            "gate-self",
            CLI
            + [
                "regressions",
                "--baseline",
                str(base),
                "--baseline-variant",
                "standard",
                "--candidate",
                str(base),
                "--candidate-variant",
                "standard",
                "--artifacts-dir",
                str(work / "artifacts"),
                "--run-id",
                "e2e-gate-self",
                "--no-html",
                "--fail-on-regression",
            ],
        )
        # vacuity guard without --fail-on-missing (peak memory is legitimately
        # null on CPU): the report must show the loss actually compared
        gate_report = json.loads((work / "artifacts" / "regressions" / "e2e-gate-self" / "summary.json").read_text())
        gate_text = json.dumps(gate_report)
        r.assert_true(
            "gate-self-not-vacuous",
            '"final_loss"' in gate_text
            and '"missing"'
            not in json.dumps([m for m in gate_report.get("metrics", []) if m.get("key") == "final_loss"]),
            "gate-self compared final_loss (a run of all-missing metrics would pass vacuously)",
        )
        # deliberately degraded fixture must TRIP the gate (exit 1): the
        # candidate variant's loss is bumped 1.5x and throughput halved
        degraded = work / "degraded"
        degraded.mkdir(exist_ok=True)
        rec = json.loads((base / "summary.json").read_text())
        for run in rec.get("runs", []):
            if run.get("attention_type") == "tropical":
                for key in ("final_loss", "score", "perplexity_est"):
                    if isinstance(run.get(key), (int, float)):
                        run[key] = run[key] * 1.5
                for key in ("tokens_per_second", "tflops_per_second_est"):
                    if isinstance(run.get(key), (int, float)):
                        run[key] = run[key] * 0.5
        (degraded / "summary.json").write_text(json.dumps(rec))
        r.run(
            "gate-trips-on-degraded",
            CLI
            + [
                "regressions",
                "--baseline",
                str(base),
                "--baseline-variant",
                "tropical",
                "--candidate",
                str(degraded),
                "--candidate-variant",
                "tropical",
                "--artifacts-dir",
                str(work / "artifacts"),
                "--run-id",
                "e2e-gate-trip",
                "--no-html",
                "--fail-on-regression",
            ],
            expect_exit=1,
        )
    except StageFailure:
        pass
    return r.report()


def scenario_scorecard(work: Path) -> bool:
    """vdc.4 maintenance contract: one real train/eval cell through C4."""
    r = Runner(work, "scorecard")
    try:
        r.run(
            "scorecard-smoke",
            CLI
            + [
                "scorecard",
                "--mechanism",
                "standard",
                "--task",
                "placebo",
                "--seeds",
                "1",
                "--budget",
                "1e6",
                "--dataset-size",
                "30",
                "--examples",
                "1",
                "--n-layer",
                "1",
                "--n-head",
                "2",
                "--n-kv-head",
                "2",
                "--n-embd",
                "32",
                "--sequence-len",
                "64",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--artifacts-dir",
                str(work / "artifacts"),
                "--run-id",
                "e2e-scorecard",
            ],
            timeout=1200,
        )
        suite = work / "artifacts" / "scorecards" / "e2e-scorecard"
        manifest = json.loads((suite / "manifest.json").read_text())
        summary = json.loads((suite / "summary.json").read_text())
        r.assert_true(
            "scorecard-contract",
            manifest["schema_version"] == "mgr.scorecard.v3"
            and len(manifest["cells"]) == 1
            and manifest["cells"][0]["status"] == "done"
            and manifest["cells"][0]["evidence_qualified"] is False
            and summary["schema_version"] == "mgr.scorecard.v3"
            and summary["adjudications"]["artifact_count"] == 0
            and summary["adjudications"]["quarantined_artifact_count"] == 2
            and (suite / "report.md").exists()
            and (suite / "report.html").exists(),
            "one-cell scorecard produced reports and quarantined its one-step smoke artifacts",
        )
    except StageFailure:
        pass
    return r.report()


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scenario",
        choices=[
            "full-loop",
            "resume",
            "determinism",
            "regression-gate",
            "scorecard",
            "word-problem",
            "symplectic",
            "all",
        ],
        action="append",
        default=None,
        help="Scenario to run (repeatable; default all). A repeated flag used to keep only the last value.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Working directory (default: a fresh temp dir OUTSIDE the repo; kept on failure)",
    )
    parser.add_argument("--keep", action="store_true", help="Keep the workdir even on success")
    parser.add_argument(
        "--resume-kill-seed",
        type=int,
        default=None,
        help="Pin the resume scenario's randomized kill step (logged either way)",
    )
    args = parser.parse_args()

    work = args.workdir or Path(tempfile.mkdtemp(prefix="mgr-e2e-"))
    work.mkdir(parents=True, exist_ok=True)
    console.print(Panel(f"workdir: [bold]{work}[/bold] · repo: {REPO}", title="mgr e2e pipeline", border_style="blue"))

    runners = {
        "full-loop": lambda: scenario_full_loop(work),
        "resume": lambda: scenario_resume(work, args.resume_kill_seed),
        "determinism": lambda: scenario_determinism(work),
        "regression-gate": lambda: scenario_regression_gate(work),
        "scorecard": lambda: scenario_scorecard(work),
        "word-problem": lambda: scenario_word_problem(work),
        "symplectic": lambda: scenario_symplectic(work),
    }
    chosen = list(args.scenario or ["all"])
    wanted = list(runners) if "all" in chosen else list(dict.fromkeys(chosen))
    results = {name: runners[name]() for name in wanted}

    table = Table(title="e2e scenarios", border_style="blue")
    table.add_column("scenario")
    table.add_column("result")
    for name, ok in results.items():
        table.add_row(name, "[green]PASS[/green]" if ok else "[red]FAIL[/red]")
    console.print(table)
    all_ok = all(results.values())
    if all_ok and not args.keep and args.workdir is None:
        console.print(
            f"[dim]workdir retained at {work} (pass --keep to silence this note; "
            "temp dirs are never auto-deleted by this script)[/dim]"
        )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
