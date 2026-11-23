"""Golden loss-trajectory parity harness for the AttentionCore refactor.

Bead: model_guided_research-7b0.1.

Each fixture under tests/fixtures/attention_goldens/ records a fixed-seed,
single-threaded CPU, 50-step AdamW training trajectory for one attention
mechanism, captured BEFORE the scaffolding extraction. After each mechanism
migrates onto the shared scaffold, this suite re-runs the identical
trajectory and demands bitwise-equal losses: the refactor must be a pure
reorganization, not a numerical change.

Capture / recapture (writes fixtures, then passes):

    MGR_CAPTURE_ATTENTION_GOLDENS=1 uv run pytest tests/test_attention_core_goldens.py

The CUDA suite uses the same CPU initialization and batches, then trains on
the GPU. Set MGR_ATTENTION_GOLDEN_DEVICE=cuda and CUBLAS_WORKSPACE_CONFIG=:4096:8
for capture and replay. Its fixtures live in attention_goldens/cuda/.

A golden that cannot be re-derived is not a golden: every fixture embeds the
full config, seed, step count, torch version, and host fingerprint needed to
reproduce it. Trajectories chaotically amplify any ulp-level difference over
50 optimizer steps, so cross-machine or cross-torch comparison is
meaningless; on a mismatched host or torch version the tests skip with a
recapture instruction instead of failing.
"""

import json
import math
import os
import platform
import socket
from pathlib import Path
from typing import TypedDict

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "attention_goldens"
CAPTURE_ENV = "MGR_CAPTURE_ATTENTION_GOLDENS"
SCHEMA_VERSION = 1
BEAD = "model_guided_research-7b0.1"

SEED = 20260610
STEPS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-3


# House tiny-model convention (mirrors tests/test_demos.py): GQA exercised
# via n_kv_head < n_head; head_dim = 16 satisfies quaternion (%4) and
# octonion (%8) divisibility. Gauge forbids GQA, so it pins n_kv_head=n_head.
class _BaseConfig(TypedDict):
    sequence_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int


class _ConfigOverride(TypedDict, total=False):
    sequence_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int


class _ConfigKwargs(_BaseConfig):
    attention_type: str


BASE_CONFIG: _BaseConfig = {
    "sequence_len": 64,
    "vocab_size": 128,
    "n_layer": 2,
    "n_head": 4,
    "n_kv_head": 2,
    "n_embd": 64,
}

MECHANISM_CONFIG_OVERRIDES: dict[str, _ConfigOverride] = {
    "standard": {},
    "hyperbolic": {},
    "tropical": {},
    "ultrametric": {},
    "simplicial": {},
    "quaternion": {},
    "braid": {},
    "fractal": {},
    "octonion": {},
    "surreal": {},
    "reversible": {},
    "gauge": {"n_kv_head": 4},
}

MECHANISMS = sorted(MECHANISM_CONFIG_OVERRIDES)


def _config_for(attention_type: str) -> GPTConfig:
    kwargs: _ConfigKwargs = {
        **BASE_CONFIG,
        **MECHANISM_CONFIG_OVERRIDES[attention_type],
        "attention_type": attention_type,
    }
    return GPTConfig(**kwargs)


def _host_fingerprint(device: str) -> dict[str, str]:
    host = {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
    }
    if device == "cuda":
        host.update(
            gpu=torch.cuda.get_device_name(),
            capability=str(torch.cuda.get_device_capability()),
            cuda=str(torch.version.cuda),
            cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        )
    return host


def _run_trajectory(attention_type: str, device: str) -> list[float]:
    """Run 50 optimizer steps and return each loss.

    Single-threaded so BLAS reduction order is fixed; model init draws from
    the globally seeded RNG (construction order == draw order), data from an
    independent generator so adding RNG consumers to init cannot silently
    shift the batches.
    """
    config = _config_for(attention_type)
    saved_threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    try:
        if device == "cuda":
            torch.use_deterministic_algorithms(True)
        torch.manual_seed(SEED)
        model = GPT(config)
        model.init_weights()
        model.to(device)
        model.train()

        data_gen = torch.Generator(device="cpu").manual_seed(SEED + 1)
        vocab = config.vocab_size
        seq = config.sequence_len
        batches = [
            torch.randint(0, vocab, (BATCH_SIZE, seq + 1), generator=data_gen, dtype=torch.long) for _ in range(STEPS)
        ]

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8)
        losses: list[float] = []
        for tokens in batches:
            tokens = tokens.to(device)
            x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
            loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        return losses
    finally:
        if device == "cuda":
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(saved_threads)


def _write_fixture(path: Path, attention_type: str, device: str, losses: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = _config_for(attention_type)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "bead": BEAD,
        "mechanism": attention_type,
        "seed": SEED,
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "torch.optim.AdamW(betas=(0.9, 0.95), eps=1e-8)",
        "host": _host_fingerprint(device),
        "config": dict(vars(config)),
        "losses": losses,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize("attention_type", MECHANISMS)
def test_attention_golden_trajectory(attention_type: str) -> None:
    capture = os.environ.get(CAPTURE_ENV, "").strip() not in {"", "0"}
    device = os.environ.get("MGR_ATTENTION_GOLDEN_DEVICE", "cpu")
    assert device in {"cpu", "cuda"}, f"Unknown trajectory device {device}"
    if device == "cuda":
        assert torch.cuda.is_available(), "CUDA trajectories require an accessible GPU"
    directory = FIXTURE_DIR / "cuda" if device == "cuda" else FIXTURE_DIR
    path = directory / f"{attention_type}.json"

    if capture:
        losses = _run_trajectory(attention_type, device)
        assert all(map(math.isfinite, losses)), f"Non-finite loss during capture for {attention_type}"
        _write_fixture(path, attention_type, device, losses)
        return

    if not path.exists():
        pytest.fail(
            f"Missing golden fixture {path}. Capture it at a known-good commit with "
            f"{CAPTURE_ENV}=1 uv run pytest {Path(__file__).name}"
        )

    golden = json.loads(path.read_text())
    host = _host_fingerprint(device)
    recorded = golden["host"]
    if recorded != host:
        pytest.skip(
            "Golden fixtures are pinned to the capturing host/torch build "
            f"(recorded {recorded}, running {host}); 50-step trajectories amplify "
            f"ulp differences chaotically. Recapture locally with {CAPTURE_ENV}=1."
        )

    expected_config = dict(vars(_config_for(attention_type)))
    assert golden["config"] == expected_config, (
        f"Golden config drift for {attention_type}: fixture was captured with "
        f"{golden['config']}, harness now builds {expected_config}. If the config "
        "change is intentional, recapture the fixture at a known-good commit."
    )
    assert golden["seed"] == SEED and golden["steps"] == STEPS and golden["batch_size"] == BATCH_SIZE

    losses = _run_trajectory(attention_type, device)
    expected = golden["losses"]
    assert len(losses) == len(expected)
    mismatches = [(i, e, a) for i, (e, a) in enumerate(zip(expected, losses, strict=True)) if e != a]
    assert not mismatches, (
        f"{attention_type}: loss trajectory diverged from pre-refactor golden at "
        f"{len(mismatches)}/{len(expected)} steps; first divergence "
        f"step={mismatches[0][0]} expected={mismatches[0][1]!r} actual={mismatches[0][2]!r}. "
        "The refactor changed numerics — this must be a pure reorganization."
    )
