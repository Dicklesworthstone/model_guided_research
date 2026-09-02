"""Acceptance tests for the diagnostic task suite generator (bead vdc.1).

Covers, per the bead spec: determinism (same seed -> byte-identical parquet),
split disjointness, brute-force label correctness, held-out difficulty axes,
dial monotonicity (brute-force statistics, not the profiler), per-task
tokenizer safety (delimiters must not merge across boundaries), manifest
checksums + version-bump enforcement, dataloader compatibility (the generated
parquet loads through nanochat/dataloader.py untouched), and the robustness
probe spec (eps=0 byte-identical no-op; measured sup-norm matches requested).

Tokenizer-dependent tests skip cleanly when the GPT-2 tokenizer is not cached
locally (CI without network).
"""

import hashlib
import json
import statistics
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from nanochat.diagnostics_data import (
    DEFAULT_TASKS,
    GENERATOR_VERSIONS,
    TASKS,
    apply_embedding_perturbation,
    check_hier,
    generate_task,
    generate_texts,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "diagnostics_generator_hashes.json"
PIN_SEED = 20260610
PIN_SIZE = 30


def _tokenizer_or_skip():
    try:
        from nanochat.tokenizer import get_tokenizer

        return get_tokenizer()
    except Exception as exc:  # no cached tokenizer and no network
        pytest.skip(f"tokenizer unavailable: {exc}")


# ---------------------------------------------------------------------------
# determinism + manifests


@pytest.mark.parametrize("task", sorted(DEFAULT_TASKS))
def test_generation_deterministic_byte_identical(task, tmp_path):
    m1 = generate_task(task, out_dir=tmp_path / "a", size=24, seed=11)
    m2 = generate_task(task, out_dir=tmp_path / "b", size=24, seed=11)
    assert m1["sha256"] == m2["sha256"], f"{task}: same (seed, dials) must produce byte-identical parquet"
    m3 = generate_task(task, out_dir=tmp_path / "c", size=24, seed=12)
    assert m3["sha256"] != m1["sha256"], f"{task}: different seed should change content"


def test_manifest_hashes_match_files(tmp_path):
    manifest = generate_task("hier", out_dir=tmp_path, size=24, seed=5)
    task_dir = tmp_path / "hier"
    for rel, recorded in manifest["sha256"].items():
        actual = hashlib.sha256((task_dir / rel).read_bytes()).hexdigest()
        assert actual == recorded, f"manifest hash mismatch for {rel}"
    on_disk = json.loads((task_dir / "manifest.json").read_text())
    assert on_disk["sha256"] == manifest["sha256"]
    assert on_disk["generator_version"] == GENERATOR_VERSIONS["hier"]


def test_generator_version_bump_enforced(tmp_path):
    """Changing generator output for a fixed (seed, dials) REQUIRES a version
    bump: regenerate the pinned configuration and compare against the
    committed fixture. If this fails after an intentional generator change,
    bump GENERATOR_VERSIONS[task] and recapture the fixture (instructions in
    the assert message)."""
    if not FIXTURE_PATH.exists():
        pytest.fail(
            f"Missing fixture {FIXTURE_PATH}. Capture it with: "
            'uv run python -c "from tests.test_diagnostics_data import capture_hash_fixture; capture_hash_fixture()"'
        )
    fixture = json.loads(FIXTURE_PATH.read_text())
    # Compare the serializer-independent CONTENT hash, not the parquet file
    # bytes: the file hash embeds the parquet writer's version string, so a
    # pyarrow upgrade (23.0.0 -> 25.0.1 on 2026-08-27) changed every task's
    # file hash while the generated documents were byte-identical. Collect
    # every drifted task before failing so the message names all of them.
    drifted: list[str] = []
    for task in sorted(DEFAULT_TASKS):
        manifest = generate_task(task, out_dir=tmp_path / task, size=PIN_SIZE, seed=PIN_SEED)
        pinned = fixture[task]
        if manifest["content_sha256"]["train_000.parquet"] != pinned["train_content_sha256"]:
            if manifest["generator_version"] == pinned["generator_version"]:
                drifted.append(f"{task} (GENERATOR_VERSIONS[{task!r}] still {pinned['generator_version']})")
    assert not drifted, (
        f"generator output changed for pinned (seed={PIN_SEED}, size={PIN_SIZE}) without a version bump: "
        + "; ".join(drifted)
        + ". Bump the version, then recapture the fixture with capture_hash_fixture()."
    )


def capture_hash_fixture() -> None:
    """Recapture the pinned hashes (run after an intentional generator change)."""
    import tempfile

    out: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for task in sorted(DEFAULT_TASKS):
            manifest = generate_task(task, out_dir=Path(tmp) / task, size=PIN_SIZE, seed=PIN_SEED)
            out[task] = {
                "generator_version": manifest["generator_version"],
                "train_content_sha256": manifest["content_sha256"]["train_000.parquet"],
            }
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {FIXTURE_PATH}")


# ---------------------------------------------------------------------------
# correctness + splits


@pytest.mark.parametrize("task", sorted(TASKS))
def test_labels_verified_by_brute_force_checker(task):
    spec = TASKS[task]
    splits = spec.generate(60, 3, spec.resolve_dials(None))
    for split, docs in splits.items():
        for doc in docs:
            result = spec.checker(doc)
            assert result is not False, f"{task}/{split}: checker rejected generated doc: {doc[:160]}"


@pytest.mark.parametrize("task", sorted(DEFAULT_TASKS))
def test_splits_are_disjoint(task):
    spec = TASKS[task]
    splits = spec.generate(90, 4, spec.resolve_dials(None))
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert not (train & val), f"{task}: train/val overlap"
    assert not (train & test), f"{task}: train/test overlap"
    assert not (val & test), f"{task}: val/test overlap"


def _max_dyck_depth(doc: str) -> int:
    depth = best = 0
    for tok in doc.split():
        if tok in "([{":
            depth += 1
            best = max(best, depth)
        elif tok in ")]}":
            depth = max(0, depth - 1)
    return best


def _seq_len(doc: str, start_marker: str, end_marker: str) -> int:
    parts = doc.split()
    return parts.index(end_marker) - parts.index(start_marker) - 1


def test_heldout_difficulty_actually_held_out():
    """The test split must exceed the training regime on its difficulty axis."""
    splits = TASKS["dyck"].generate(120, 9, {"max_depth": 3})
    train_max = max(_max_dyck_depth(d) for d in splits["train"])
    test_max = max(_max_dyck_depth(d) for d in splits["test"])
    assert test_max > train_max, f"dyck: test depth {test_max} must exceed train {train_max}"

    splits = TASKS["copyops"].generate(120, 9, {"length": 6})
    train_max = max(_seq_len(d, "SEQ", "OUT") for d in splits["train"])
    test_min = min(_seq_len(d, "SEQ", "OUT") for d in splits["test"])
    assert train_max <= 6 and test_min > 6, f"copyops: train<= 6 < test ({train_max}, {test_min})"

    splits = TASKS["group"].generate(120, 9, {"length": 4})
    train_max = max(_seq_len(d, "SEQ", "OUT") for d in splits["train"])
    test_min = min(_seq_len(d, "SEQ", "OUT") for d in splits["test"])
    # generator contract: train words in [2, 4]; held-out words in [2*4, 8*4]
    assert train_max <= 4, f"group: train word length {train_max} exceeds the dial"
    assert test_min >= 8, f"group: held-out lengths must start at 2x the dial (got {test_min})"


# ---------------------------------------------------------------------------
# dial monotonicity (brute-force statistics; the profiler cross-check is 77l.3)


def _mono_increasing(values: Sequence[float], label: str) -> None:
    assert all(b > a for a, b in zip(values, values[1:])), f"{label} not strictly increasing: {values}"


def test_dial_monotonicity_depth_and_length_dials():
    stats = []
    for depth in (2, 4, 8):
        docs = generate_texts("dyck", size=120, seed=21, dial_overrides={"max_depth": depth})
        stats.append(max(_max_dyck_depth(d) for d in docs))
    _mono_increasing(stats, "dyck max_depth -> observed nesting depth (seed=21)")

    stats = []
    for length in (6, 12, 24):
        docs = generate_texts("copyops", size=120, seed=21, dial_overrides={"length": length})
        stats.append(max(_seq_len(d, "SEQ", "OUT") for d in docs))
    _mono_increasing(stats, "copyops length -> observed max length (seed=21)")

    stats = []
    for spread in (2, 5, 10):
        docs = generate_texts("arith", size=150, seed=21, dial_overrides={"spread_decades": spread})
        exps = [abs(int(d.split()[3].split("e")[1])) for d in docs]
        stats.append(max(exps))
    _mono_increasing(stats, "arith spread_decades -> observed exponent range (seed=21)")

    stats = []
    for words in (64, 128, 512):
        docs = generate_texts("needle", size=40, seed=21, dial_overrides={"context_words": words})
        stats.append(statistics.mean(len(d.split()) for d in docs))
    _mono_increasing(stats, "needle context_words -> doc length (seed=21)")

    stats = []
    for n_ops in (6, 12, 24):
        docs = generate_texts("bag", size=60, seed=21, dial_overrides={"n_ops": n_ops})
        stats.append(statistics.mean(d.split().count(";") for d in docs))
    _mono_increasing(stats, "bag n_ops -> op separator count (seed=21)")

    stats = []
    for n_regimes in (2, 4, 8):
        docs = generate_texts("regime", size=40, seed=21, dial_overrides={"n_regimes": n_regimes})
        stats.append(statistics.mean(d.split().count("SEG") for d in docs))
    _mono_increasing(stats, "regime n_regimes -> SEG count (seed=21)")


def test_dial_monotonicity_zipf_imbalance():
    """Higher zipf_alpha concentrates subtree MASS on the rank-0 branch: the
    imbalance statistic is the rank-0 top-level branch's share of the tree
    serialization, averaged over documents."""

    def rank0_share(docs: list[str]) -> float:
        shares = []
        for doc in docs:
            parts = doc.split()
            tree_toks = parts[parts.index("TREE") + 1 : parts.index("PATH")]
            group_sizes: list[int] = []
            depth = 0
            current = 0
            for tok in tree_toks:
                current += 1
                if tok == "(":
                    depth += 1
                elif tok == ")":
                    depth -= 1
                    if depth == 0:
                        group_sizes.append(current)
                        current = 0
            if group_sizes:
                shares.append(group_sizes[0] / sum(group_sizes))
        return statistics.mean(shares)

    stats = []
    for alpha in (0.0, 1.5, 3.0):
        docs = generate_texts("hier", size=200, seed=33, dial_overrides={"zipf_alpha": alpha})
        stats.append(rank0_share(docs))
    _mono_increasing(stats, "hier zipf_alpha -> rank-0 branch mass share (seed=33)")


def test_dial_monotonicity_placebo_structure():
    def structure_score(docs: list[str]) -> float:
        scores = []
        for doc in docs:
            parts = doc.split()[2:]
            hits = sum(1 for a, b in zip(parts, parts[1:]) if a == "(" and (b.startswith("k") or b == "("))
            opens = max(1, parts.count("("))
            scores.append(hits / opens)
        return statistics.mean(scores)

    stats = []
    for structure in (0.0, 0.5, 1.0):
        docs = generate_texts("placebo", size=120, seed=27, dial_overrides={"structure": structure})
        stats.append(structure_score(docs))
    _mono_increasing(stats, "placebo structure -> bracket-key adjacency (seed=27)")


def test_placebo_preserves_unigram_statistics():
    """The shuffle destroys structure but preserves each doc's token multiset."""
    random_docs = generate_texts("placebo", size=40, seed=8, dial_overrides={"structure": 0.0})
    intact_docs = generate_texts("placebo", size=40, seed=8, dial_overrides={"structure": 1.0})
    assert len(random_docs) == len(intact_docs)
    for rd, sd in zip(random_docs, intact_docs):
        assert sorted(rd.split()[2:]) == sorted(sd.split()[2:]), "placebo shuffle must preserve unigram stats"
    assert random_docs != intact_docs, "structure=0 must actually shuffle"


def test_dial_out_of_range_rejected():
    with pytest.raises(ValueError, match="outside documented range"):
        generate_texts("dyck", size=10, seed=1, dial_overrides={"max_depth": 99})
    with pytest.raises(ValueError, match="no dial"):
        generate_texts("dyck", size=10, seed=1, dial_overrides={"bogus": 1})


# ---------------------------------------------------------------------------
# checker edge cases


def test_hier_checker_missing_and_wrong_paths():
    doc = "TASK hier TREE ( a ( b v1 ) ) ( c v2 ) PATH a b OUT v1"
    assert check_hier(doc) is True
    assert check_hier("TASK hier TREE ( a ( b v1 ) ) PATH a z OUT v1") is False  # missing path
    assert check_hier("TASK hier TREE ( a ( b v1 ) ) PATH a b OUT v9") is False  # wrong value


def test_group_task_covers_solvable_and_nonsolvable():
    docs = generate_texts("group", size=200, seed=13)
    groups = {d.split()[3] for d in docs}
    assert groups == {"s5", "a5", "z60", "s3"}, f"group coverage incomplete: {groups}"


def test_rot_octahedral_group_is_exact():
    from nanochat.diagnostics_data import _OCTAHEDRAL, check_rot

    assert len(_OCTAHEDRAL) == 24
    # X90 four times is the identity; the identity has a stable index
    identity_idx = _OCTAHEDRAL.index(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    assert check_rot(f"TASK rot SEQ X90 X90 X90 X90 OUT o{identity_idx}") is True


# ---------------------------------------------------------------------------
# tokenizer safety (per task, explicit)


def test_rustbpe_missing_error_names_installable_extra(monkeypatch):
    from nanochat import tokenizer

    monkeypatch.setattr(tokenizer, "rustbpe", None)
    with pytest.raises(ImportError, match="uv sync --extra tokenizer-train"):
        tokenizer.RustBPETokenizer.train_from_iterator(["small training corpus"], vocab_size=270)


def test_rustbpe_training_round_trip_when_extra_is_installed():
    pytest.importorskip("rustbpe")
    from nanochat.tokenizer import RustBPETokenizer

    text = "small training corpus with repeated words " * 32
    tokenizer = RustBPETokenizer.train_from_iterator([text], vocab_size=270)

    assert tokenizer.decode(tokenizer.encode(text)) == text


def _assert_delimiters_unmerged(tok, doc: str, delimiters: tuple[str, ...], task: str) -> None:
    ids = tok.encode(doc)
    pieces = [tok.decode([i]) for i in ids]
    assert "".join(pieces) == doc, f"{task}: tokenizer round-trip changed the document"
    # Walk the piece boundaries: no piece may span a delimiter boundary, i.e.
    # contain non-space material from BOTH sides of a delimiter edge.
    offsets = []
    pos = 0
    for piece in pieces:
        offsets.append((pos, pos + len(piece)))
        pos += len(piece)
    for delim in delimiters:
        start = 0
        while True:
            s = doc.find(f" {delim} ", start)
            if s < 0:
                break
            ds, de = s + 1, s + 1 + len(delim)  # delimiter span in chars
            for ps, pe in offsets:
                inside = max(ps, ds) < min(pe, de)
                if not inside:
                    continue
                left_spill = doc[ps:ds].strip()
                right_spill = doc[de:pe].strip()
                assert not left_spill and not right_spill, (
                    f"{task}: delimiter {delim!r} merged with neighbors in piece {doc[ps:pe]!r} of doc {doc[:120]!r}"
                )
            start = de


@pytest.mark.parametrize("task", sorted(t for t in TASKS if TASKS[t].delimiters))
def test_tokenizer_safety_per_task(task):
    tok = _tokenizer_or_skip()
    spec = TASKS[task]
    docs = generate_texts(task, size=12, seed=99)
    for doc in docs[:6]:
        _assert_delimiters_unmerged(tok, doc, spec.delimiters, task)


# ---------------------------------------------------------------------------
# dataloader compatibility (the acceptance criterion)


def test_generated_parquet_loads_through_existing_dataloader(monkeypatch, tmp_path):
    tok = _tokenizer_or_skip()
    del tok
    generate_task("copyops", out_dir=tmp_path, size=60, seed=42)
    task_dir = tmp_path / "copyops"
    files = sorted(str(p) for p in task_dir.glob("*.parquet"))
    assert [Path(f).name for f in files] == ["train_000.parquet", "val_000.parquet"], (
        "train must sort before val so the dataloader's last-file-is-val convention holds"
    )

    import nanochat.dataloader as dl_mod

    monkeypatch.setattr(dl_mod, "list_parquet_files", lambda data_dir=None: files)
    loader = dl_mod.tokenizing_distributed_data_loader_with_state(B=2, T=64, split="train", device="cpu")
    for _ in range(2):
        inputs, targets, state = next(loader)
        assert inputs.shape == (2, 64) and targets.shape == (2, 64)
        assert int(inputs.max()) < 50257
        assert set(state) == {"pq_idx", "rg_idx"}


# ---------------------------------------------------------------------------
# robustness probe spec


def test_probe_eps_zero_is_byte_identical_noop():
    x = torch.randn(2, 8, 16)
    gen = torch.Generator().manual_seed(0)
    out = apply_embedding_perturbation(x, 0.0, gen)
    assert out is x, "eps=0 must return the identical tensor object (byte-identical no-op)"


def test_probe_magnitude_matches_requested_sup_norm():
    x = torch.zeros(4, 64, 128)
    gen = torch.Generator().manual_seed(7)
    eps = 0.25
    out = apply_embedding_perturbation(x, eps, gen)
    delta = (out - x).abs()
    assert float(delta.max()) <= eps + 1e-7, "perturbation exceeded the sup-norm bound"
    assert float(delta.max()) > eps * 0.98, "max perturbation should approach the bound on a large tensor"
    gen2 = torch.Generator().manual_seed(7)
    out2 = apply_embedding_perturbation(x, eps, gen2)
    assert torch.equal(out, out2), "probe must be deterministic given the generator seed"
    with pytest.raises(ValueError):
        apply_embedding_perturbation(x, -0.1, gen)


# ---------------------------------------------------------------------------
# CLI


def test_cli_gen_tasks_and_profile_task(tmp_path):
    from typer.testing import CliRunner

    import cli as mgr_cli

    runner = CliRunner()
    result = runner.invoke(mgr_cli.app, ["gen-tasks", "--list"])
    assert result.exit_code == 0 and "placebo" in result.output

    result = runner.invoke(
        mgr_cli.app,
        ["gen-tasks", "--task", "dyck", "--out", str(tmp_path), "--size", "30", "--seed", "5", "--dial", "max_depth=6"],
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "dyck" / "manifest.json").read_text())
    assert manifest["dials"]["max_depth"] == 6.0

    result = runner.invoke(mgr_cli.app, ["gen-tasks", "--task", "nonexistent"])
    assert result.exit_code == 2

    result = runner.invoke(mgr_cli.app, ["gen-tasks", "--task", "all", "--dial", "max_depth=6"])
    assert result.exit_code == 2, "--dial with --task all must be rejected"

    result = runner.invoke(mgr_cli.app, ["gen-tasks", "--task", "dyck", "--dial", "max_depth=abc"])
    assert result.exit_code == 2, "non-numeric dial value must exit 2, not traceback"

    result = runner.invoke(mgr_cli.app, ["profile-data", "--task", "hier:depth=abc"])
    assert result.exit_code == 2, "non-numeric profile-data dial must exit 2, not traceback"

    result = runner.invoke(
        mgr_cli.app,
        ["profile-data", "--task", "hier:depth=4", "--sample", "24", "--points", "16", "--json"],
    )
    assert result.exit_code == 0, result.output
    profile = json.loads(result.output)
    assert profile["corpus"] == "task:hier:depth=4"
