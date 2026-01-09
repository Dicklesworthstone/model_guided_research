from __future__ import annotations

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
