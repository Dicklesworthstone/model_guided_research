from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from cmaes import CMA

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_script(mod_name: str, filename: str):
    """Load a scripts/*.py module by path (scripts/ is not a package).

    Registers it in sys.modules BEFORE exec so module-level @dataclass(frozen=True)
    can resolve its own __module__ during class creation.
    """
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPTS / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_parquet_shard(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pydict({"text": texts})
    pq.write_table(table, path, row_group_size=8)


def test_cmaes_rosenbrock_sanity() -> None:
    # A tiny CMA-ES smoke test to catch ask/tell regressions.
    # Rosenbrock has a well-known minimum at (1, 1) with f=0.
    def rosenbrock(x: np.ndarray) -> float:
        a = 1.0
        b = 100.0
        return float((a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2)

    bounds = np.array([(-2.0, 2.0), (-1.0, 3.0)], dtype=np.float64)
    opt = CMA(
        mean=np.array([0.0, 0.0], dtype=np.float64),
        sigma=0.5,
        bounds=bounds,
        seed=0,
        population_size=8,
    )

    best = float("inf")
    for _gen in range(25):
        solutions: list[tuple[np.ndarray, float]] = []
        for _ in range(opt.population_size):
            x = opt.ask()
            fx = rosenbrock(x)
            best = min(best, fx)
            solutions.append((x, fx))
        opt.tell(solutions)

    assert best < 1.0


def test_nanochat_train_objective_deterministic_cpu(tmp_path: Path) -> None:
    """
    End-to-end objective smoke test:
    - Create a tiny local parquet dataset (2 shards) under a temp NANOCHAT_BASE_DIR.
    - Run nanochat.train twice with the same seed and fixed-FLOPs budget.
    - Assert losses match (within tight tolerance) and are sane/finite.
    """
    nanochat_base = tmp_path / "nanochat_base"
    data_dir = nanochat_base / "base_data"

    # Keep docs short but token-dense so even tiny B/T can fill buffers quickly.
    docs = [("hello world " * 200).strip() for _ in range(64)]
    _write_parquet_shard(data_dir / "shard_00000.parquet", docs)
    _write_parquet_shard(data_dir / "shard_00001.parquet", docs)

    artifacts_dir = tmp_path / "artifacts"

    def _run_once(run_id: str) -> dict:
        env = os.environ.copy()
        env["NANOCHAT_BASE_DIR"] = str(nanochat_base)

        cmd = [
            sys.executable,
            "-m",
            "nanochat.train",
            "--device",
            "cpu",
            "--seed",
            "123",
            "--batch-size",
            "1",
            "--sequence-len",
            "16",
            "--n-layer",
            "1",
            "--n-head",
            "1",
            "--n-kv-head",
            "1",
            "--n-embd",
            "32",
            "--optimizer-type",
            "adamw",
            "--attention-type",
            "standard",
            "--target-flops",
            "5e8",
            "--warmup-steps",
            "0",
            "--log-interval",
            "1",
            "--artifacts-dir",
            str(artifacts_dir),
            "--artifacts-kind",
            "tests",
            "--artifacts-topic",
            "cma_obj",
            "--run-id",
            run_id,
            "--min-parquet-files",
            "2",
        ]

        proc = subprocess.run(  # nosec B603
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=240,
            check=False,
        )
        if proc.returncode != 0:
            raise AssertionError(
                "nanochat.train failed:\n"
                f"cmd={cmd!r}\n"
                f"stdout_tail={proc.stdout[-2000:]}\n"
                f"stderr_tail={proc.stderr[-2000:]}\n"
            )

        summary_path = artifacts_dir / "tests" / "cma_obj" / run_id / "summary.json"
        assert summary_path.exists()
        return json.loads(summary_path.read_text(encoding="utf-8"))

    s1 = _run_once("run1")
    s2 = _run_once("run2")

    losses1 = [float(x) for x in s1["results"]["losses"]]
    losses2 = [float(x) for x in s2["results"]["losses"]]
    assert len(losses1) >= 1
    assert len(losses2) == len(losses1)

    # Determinism should be exact on CPU, but allow a tiny tolerance to avoid
    # flaky failures if any underlying numeric kernel changes.
    max_abs = max(abs(a - b) for a, b in zip(losses1, losses2, strict=True))
    assert max_abs < 1e-7

    # Sanity range: cross-entropy should be finite and not wildly out of range.
    assert all(np.isfinite(losses1))
    assert 0.0 < float(losses1[-1]) < 50.0


def test_ca_initializer_variance_sanity() -> None:
    """
    Unit sanity check for the CA initializer:
    - Pure CA init should hit the target std derived from fan-in/out scaling.
    - Same config/seed should be deterministic.
    """
    from nanochat.gpt import GPT, GPTConfig

    cfg = GPTConfig(
        sequence_len=8,
        vocab_size=64,
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        n_embd=16,
        attention_type="standard",
        ca_init_rule="rule30",
        ca_init_alpha=1.0,
        ca_init_seed=123,
    )

    m1 = GPT(cfg)
    m1.init_weights()
    w1 = m1.transformer.h[0].attn.c_q.weight.detach().cpu().float()

    m2 = GPT(cfg)
    m2.init_weights()
    w2 = m2.transformer.h[0].attn.c_q.weight.detach().cpu().float()

    assert torch.equal(w1, w2)

    fan_out, fan_in = w1.shape
    target_std = 1.0 / (fan_in**0.5) * min(1.0, (fan_out / fan_in) ** 0.5)
    actual_std = float(w1.std(unbiased=False).item())
    actual_mean = float(w1.mean().item())

    assert abs(actual_mean) < 1e-3
    assert abs(actual_std - target_std) / target_std < 1e-3


def test_ca_initializer_stats_across_shapes_rules_and_dtypes() -> None:
    """
    Broader CA init safety checks:
    - rule30 + rule116 hit target std across representative shapes
    - skew/kurtosis are not pathological (very loose bounds)
    - works when model weights are bf16 on CPU (mixing path)
    """
    from nanochat.gpt import GPT, GPTConfig, _ca_values_for_weight

    def moments(x: torch.Tensor) -> tuple[float, float, float, float]:
        x = x.detach().cpu().float()
        mean = float(x.mean().item())
        std = float(x.std(unbiased=False).item())
        centered = x - mean
        eps = 1e-12
        skew = float((centered**3).mean().item() / (std**3 + eps))
        kurt = float((centered**4).mean().item() / (std**4 + eps))
        return mean, std, skew, kurt

    shapes = [(64, 64), (127, 33), (33, 127), (8, 512)]
    for rule in (30, 116):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        for fan_out, fan_in in shapes:
            target_std = 1.0 / (fan_in**0.5) * min(1.0, (fan_out / fan_in) ** 0.5)
            w = _ca_values_for_weight(rule=rule, shape=(fan_out, fan_in), target_std=target_std, generator=gen)
            mean, std, skew, kurt = moments(w)

            assert abs(mean) < 5e-3
            assert abs(std - target_std) / target_std < 5e-3
            assert abs(skew) < 2.0
            assert 0.0 < kurt < 25.0

    # Mixed precision (bf16) integration smoke: ensure no exception and finite weights.
    cfg = GPTConfig(
        sequence_len=8,
        vocab_size=64,
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        n_embd=16,
        attention_type="standard",
        ca_init_rule="rule116",
        ca_init_alpha=0.3,
        ca_init_seed=123,
    )
    model = GPT(cfg).to(dtype=torch.bfloat16)
    model.init_weights()
    w = model.transformer.h[0].attn.c_q.weight.detach().cpu().float()
    assert torch.isfinite(w).all()
    assert float(w.std(unbiased=False).item()) > 0.0


# --------------------------------------------------------------------------- #
# scripts/cmaes_phase1.py robustness (beads 2mj/a3u/q8f/wiz)
# --------------------------------------------------------------------------- #
def _cm():
    return _load_script("cmaes_phase1_under_test", "cmaes_phase1.py")


def _cell(seed: int, status: str, score: float):
    cm = _cm()
    return cm.CellEval(seed=seed, status=status, score=score, duration_s=1.0,
                       command="", returncode=0, train_summary_path=None, losses_tail=[score])


def test_cmaes_seed_aggregation() -> None:
    cm = _cm()
    evals = [_cell(0, "ok", 2.0), _cell(1, "ok", 4.0)]
    assert cm._aggregate_seed_scores(evals, how="mean", lam=1.0) == 3.0
    assert cm._aggregate_seed_scores(evals, how="worst", lam=1.0) == 4.0
    # mean + lam*std(ddof=1); std([2,4]) = sqrt(2)
    assert abs(cm._aggregate_seed_scores(evals, how="mean_std", lam=1.0) - (3.0 + 2.0**0.5)) < 1e-9
    # a failed seed keeps the penalty, dominating any aggregation
    mixed = [_cell(0, "error", cm.PENALTY_SCORE), _cell(1, "ok", 3.0)]
    assert cm._aggregate_seed_scores(mixed, how="mean", lam=1.0) > 1e8


def test_cmaes_budget_guards() -> None:
    cm = _cm()
    import argparse
    a = argparse.Namespace(max_evals=8, max_wall_seconds=100.0, patience=2,
                           max_crash_rate=0.5, population_size=4)
    assert cm._check_budget(cm.SearchState(1, 8, 0, 0.0, 0, None), a, 1.0) is not None  # max_evals
    assert cm._check_budget(cm.SearchState(1, 4, 3, 0.0, 0, None), a, 1.0) is not None  # crash rate 75%
    assert cm._check_budget(cm.SearchState(3, 4, 0, 0.0, 2, None), a, 1.0) is not None  # patience
    assert cm._check_budget(cm.SearchState(1, 2, 0, 95.0, 0, None), a, 6.0) is not None  # wall 101>=100
    assert cm._check_budget(cm.SearchState(1, 2, 0, 0.0, 0, None), a, 1.0) is None  # no trip


def test_cmaes_checkpoint_roundtrip(tmp_path: Path) -> None:
    cm = _cm()
    opt = CMA(mean=np.zeros(10), sigma=0.3, seed=0, population_size=4)
    opt.tell([(opt.ask(), 1.0) for _ in range(4)])
    state = cm.SearchState(generation=2, eval_count=8, crash_count=1,
                           wall_accum_s=42.0, no_improve_streak=1, best={"score": 1.23})
    cm._save_checkpoint(tmp_path, opt, state)
    opt2, st = cm._load_checkpoint(tmp_path)
    assert opt2.generation == opt.generation
    assert (st.generation, st.eval_count, st.crash_count, st.no_improve_streak) == (2, 8, 1, 1)
    assert st.wall_accum_s == 42.0 and st.best == {"score": 1.23}


def test_cmaes_resume_arg_restore() -> None:
    cm = _cm()
    import argparse
    prev = {
        "cmaes": {"population_size": 6, "sigma": 0.4, "search_seed": 2},
        "objective": {"target_flops": 5e9, "eval_seeds": [0, 1], "seed_agg": "mean_std",
                      "seed_agg_lambda": 2.0, "score_tail": 5, "device": "cpu",
                      "train_args": {"n_layer": 6, "n_embd": 256, "batch_size": 16}},
        "budget": {"generations": 10, "max_evals": 100, "max_wall_seconds": None,
                   "patience": 3, "max_crash_rate": 0.1},
        "dataset": {"data_dir": None},
    }
    a = argparse.Namespace(population_size=4, sigma=0.3, search_seed=0, target_flops=1e10,
                           eval_seeds=[123], seed_agg="mean", seed_agg_lambda=1.0, score_tail=3,
                           device="cpu", data_dir=None, n_layer=4, n_embd=128, batch_size=8,
                           sequence_len=256, vocab_size=50304, n_head=4, n_kv_head=4,
                           learning_rate=6e-4, warmup_steps=1, log_interval=1,
                           generations=2, max_evals=999, max_wall_seconds=None,
                           patience=None, max_crash_rate=None)
    # resume command explicitly extends --max-evals only
    cm._restore_args_from_run_json(a, prev, ["scripts/cmaes_phase1.py", "--resume", "--max-evals", "999"])
    assert a.population_size == 6 and a.eval_seeds == [0, 1] and a.seed_agg == "mean_std"
    assert a.target_flops == 5e9 and a.n_layer == 6 and a.n_embd == 256 and a.batch_size == 16
    assert a.generations == 10 and a.patience == 3  # restored from run.json
    assert a.max_evals == 999  # explicit argv override preserved


def test_cmaes_dataset_fingerprint(tmp_path: Path) -> None:
    cm = _cm()
    d = tmp_path / "corpus"
    _write_parquet_shard(d / "shard_00000.parquet", ["a b c"] * 8)
    _write_parquet_shard(d / "shard_00001.parquet", ["d e f"] * 8)
    fp1 = cm._dataset_fingerprint(str(d))
    fp2 = cm._dataset_fingerprint(str(d))
    assert fp1["resolved"] and fp1["n_files"] == 2
    assert fp1["digest"] == fp2["digest"]  # deterministic
    _write_parquet_shard(d / "shard_00002.parquet", ["g h i"] * 8)
    fp3 = cm._dataset_fingerprint(str(d))
    assert fp3["digest"] != fp1["digest"]  # detects corpus change


# --------------------------------------------------------------------------- #
# scripts/cmaes_analyze.py — flat objective must not produce spurious rankings
# --------------------------------------------------------------------------- #
def _write_fake_cmaes_run(run_dir: Path, *, scores: list[float], param_name: str = "tau_c") -> None:
    """Fabricate the eval/ tree the analyzer reads: each candidate gets a
    synaptic_config.json and a seed_123/summary.json whose final loss == score.
    The varying param is set proportional to the score so a non-flat run has a
    detectable correlation.
    """
    (run_dir).mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({
        "param_space": {"specs": [{"name": param_name, "kind": "linear"},
                                  {"name": "alpha_c", "kind": "linear"}]}
    }), encoding="utf-8")
    for i, sc in enumerate(scores):
        cand = run_dir / "eval" / "gen_0000" / f"cand_{i:04d}"
        cand.mkdir(parents=True, exist_ok=True)
        (cand / "synaptic_config.json").write_text(
            json.dumps({param_name: float(sc), "alpha_c": 0.5}), encoding="utf-8")
        sd = cand / "seed_123"
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "summary.json").write_text(
            json.dumps({"results": {"losses": [sc + 0.5, sc + 0.2, sc]}}), encoding="utf-8")


def test_cmaes_analyze_flat_suppresses_correlations(tmp_path: Path) -> None:
    an = _load_script("cmaes_analyze_under_test", "cmaes_analyze.py")
    run_dir = tmp_path / "flat"
    # flat: all candidates score ~10.85 with only numerical-noise spread
    _write_fake_cmaes_run(run_dir, scores=[10.850000, 10.850001, 10.850002, 10.850003])
    specs = an._load_param_specs(run_dir)
    points = an._collect_points(run_dir, score_tail=3)
    assert len(points) == 4
    scores = np.array([p["score"] for p in points])
    has_signal = float(np.std(scores)) > 1e-3
    assert not has_signal
    rows = an._sensitivity(points, specs, enabled=has_signal)
    # flat -> every correlation suppressed (None), never a spurious high value
    assert all(r["spearman"] is None for r in rows)


def test_cmaes_analyze_signal_recovers_correlation(tmp_path: Path) -> None:
    an = _load_script("cmaes_analyze_under_test", "cmaes_analyze.py")
    run_dir = tmp_path / "signal"
    # signal: tau_c set == score, so they are perfectly rank-correlated
    _write_fake_cmaes_run(run_dir, scores=[2.0, 3.0, 4.0, 5.0, 6.0])
    specs = an._load_param_specs(run_dir)
    points = an._collect_points(run_dir, score_tail=3)
    scores = np.array([p["score"] for p in points])
    assert float(np.std(scores)) > 1e-3
    rows = an._sensitivity(points, specs, enabled=True)
    tau = next(r for r in rows if r["name"] == "tau_c")
    assert tau["spearman"] is not None and tau["spearman"] > 0.99


# --------------------------------------------------------------------------- #
# scripts/ca_init_bench.py — init-time activation probe (bead m32)
# --------------------------------------------------------------------------- #
def test_ca_init_bench_activation_probe() -> None:
    cb = _load_script("ca_init_bench_under_test", "ca_init_bench.py")
    arch = cb.ArchConfig(name="t", n_layer=2, n_head=2, n_kv_head=2, n_embd=32,
                         sequence_len=16, batch_size=2, note="tiny")
    std = cb.INIT_VARIANTS["standard"]
    ca = cb.INIT_VARIANTS["ca_rule30"]
    a_std = cb._probe_init_activations(arch, std, seed=0, device="cpu", vocab_size=256)
    a_ca = cb._probe_init_activations(arch, ca, seed=0, device="cpu", vocab_size=256)
    for a in (a_std, a_ca):
        assert a["all_finite"]
        assert a["input_proj_rms_mean"] is not None
        assert a["weight_std_mean"] is not None
    # standard init is calibrated to ~unit-RMS activations; CA (spatially
    # correlated weights at matched std) inflates the input-projection RMS.
    assert a_std["input_proj_rms_mean"] < 1.25
    assert a_ca["input_proj_rms_mean"] > a_std["input_proj_rms_mean"]
