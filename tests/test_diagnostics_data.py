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
from collections import Counter
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
    check_regime,
    generate_task,
    generate_texts,
    hier_bind_answer,
    realhier_module_report,
    realhier_sources,
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


def test_paired_hierarchy_separates_trees_and_matches_distinct_answers():
    spec = TASKS["hier_paired"]
    dials = spec.resolve_dials(None)
    splits = spec.generate(200, 42, dials)
    assert spec.generate(200, 42, dials) == splits
    assert "hier_paired" not in DEFAULT_TASKS
    assert {key: len(docs) for key, docs in splits.items()} == {"train": 160, "val": 20, "test": 20}
    trees = set()
    for split, docs in splits.items():
        for left, right in zip(docs[::2], docs[1::2], strict=True):
            tree, query = left.split(" PATH ", 1)
            other_tree, other_query = right.split(" PATH ", 1)
            path, answer = query.split(" OUT ")
            other_path, other_answer = other_query.split(" OUT ")
            assert tree == other_tree and tree not in trees
            trees.add(tree)
            assert path != other_path and answer != other_answer
            assert len(path.split()) == len(other_path.split()) == (3 if split == "test" else 2)
            assert check_hier(left) and check_hier(right)
            assert check_hier(f"{tree} PATH {other_path} OUT {answer}") is False
    with pytest.raises(ValueError, match="even document count"):
        spec.generate(13, 42, dials)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_hierarchy_binding_pairs_change_one_visited_scope(depth):
    spec = TASKS["hier_bind"]
    dials = spec.resolve_dials({"depth": depth})
    splits = spec.generate(240, 42, dials)
    assert spec.paired_queries and "hier_bind" not in DEFAULT_TASKS
    assert splits == spec.generate(240, 42, dials)
    assert {split: len(docs) for split, docs in splits.items()} == {"train": 192, "val": 24, "test": 24}
    tree_identities = set()
    for split, docs in splits.items():
        interventions = Counter()
        for first, second in zip(docs[::2], docs[1::2], strict=True):
            left, right = first.split(" OUT ")[0], second.split(" OUT ")[0]
            base, codes = left.split(" BIND ")
            other_base, other_codes = right.split(" BIND ")
            table, query = codes.split(" PATH ")
            other_table, other_query = other_codes.split(" PATH ")
            assert base == other_base and query == other_query
            assert Counter(left.split()) == Counter(right.split())
            assert len(query.split()) == depth + (split == "test")
            assert set(query.split()) <= {"a0", "a1"}

            # Recover an order-independent tree identity directly from the document.
            labels = {}
            keys = []
            words = iter(base.split()[3:])
            for word in words:
                if word == "(":
                    keys.append(next(words))
                elif word == ")":
                    keys.pop()
                else:
                    labels[tuple(keys)] = word
            identity = tuple(sorted(labels.items()))
            assert identity not in tree_identities and len(set(labels.values())) == 2 ** len(query.split())
            tree_identities.add(identity)

            rows = [line.split() for line in table.split(" ; ")]
            other_rows = [line.split() for line in other_table.split(" ; ")]
            (changed,) = [(row, other) for row, other in zip(rows, other_rows, strict=True) if row != other]
            row, other = changed
            assert other == [row[0], row[1], row[4], row[3], row[2]]
            plan = {parent: {a: k, b: l} for parent, a, k, b, l in rows}
            scope = "r"
            visited = []
            for alias in query.split():
                visited.append(scope)
                scope += plan[scope][alias][-1]
            interventions[visited.index(row[0])] += 1

            expected = first.split(" OUT ")[1]
            assert expected != second.split(" OUT ")[1]
            assert hier_bind_answer(left + " OUT") == expected
            # A prediction held fixed across the pair can get at most one right.
            assert spec.checker(first) and spec.checker(second)
            assert spec.checker(f"{right} OUT {expected}") is False
        assert set(interventions) == set(range(depth + (split == "test")))
        assert max(interventions.values()) - min(interventions.values()) <= 1


def test_hierarchy_binding_oracle_uses_local_tables_and_rejects_malformed_inputs():
    spec = TASKS["hier_bind"]
    tree = "TASK hier_bind TREE ( k0 ( k0 v7 ) ( k1 v8 ) ) ( k1 ( k0 v9 ) ( k1 v10 ) )"
    prompt = f"{tree} BIND r0 a0 k0 a1 k1 ; r a0 k1 a1 k0 ; r1 a1 k0 a0 k1 PATH a0 a0 OUT"
    assert hier_bind_answer(prompt) == "v10"
    assert spec.checker(f"{prompt} v10") is True
    assert spec.checker(f"{prompt} v8") is False
    for malformed in (
        prompt.replace("r1 a1 k0 a0 k1", "r0 a1 k0 a0 k1"),
        prompt.replace("r1 a1 k0 a0 k1", "r1 a1 k0 a0 k0"),
        prompt.replace("r1 a1 k0 a0 k1", "r1 a1 k0 a1 k1"),
        prompt.replace("v7 )", "v7"),
        prompt.replace("PATH a0 a0", "PATH k0 a0"),
        prompt.replace(" BIND ", " "),
        prompt.replace(" OUT", ""),
    ):
        assert hier_bind_answer(malformed) is None
        assert spec.checker(f"{malformed} v10") is False
    assert spec.checker("TASK hier TREE") is None


def test_realhier_separates_source_modules_and_exhausts_unique_test_paths(tmp_path):
    import pyarrow.parquet as pq

    manifest = generate_task("realhier", out_dir=tmp_path, size=1200, seed=42)
    provenance = manifest["provenance"]
    assert len(set(provenance["module_source_sha256"].values())) == 12
    assert len(set(provenance["module_tree_sha256"].values())) == 12
    sources = realhier_sources()
    assert set(provenance["module_splits"]["test"]) == {"contextlib", "enum", "textwrap"}
    eligible = {
        f"TASK hier TREE {source['tree']} PATH {' '.join(path)} OUT {answer}"
        for source in sources.values()
        if source["split"] == "test"
        for path, answer in source["paths"]
    }
    heldout = pq.read_table(tmp_path / "realhier/heldout/test_000.parquet")["text"].to_pylist()
    assert len(heldout) == len(set(heldout)) == min(120, len(eligible))
    assert set(heldout) <= eligible
    assert all(check_hier(doc) for doc in heldout)


def test_realhier_rejects_duplicate_tree_even_if_source_bytes_differ(monkeypatch):
    import nanochat.diagnostics_data as diagnostics

    monkeypatch.setattr(
        diagnostics, "_module_tree", lambda name: ({"alias": "fn"}, hashlib.sha256(name.encode()).hexdigest())
    )
    with pytest.raises(ValueError, match="duplicate source or definition tree"):
        realhier_sources()


def test_realhier_macro_weights_modules_and_counts_overlength_errors():
    class WordTokenizer:
        def get_bos_token_id(self):
            return 0

        def encode(self, text, prepend=None):
            return ([prepend] if prepend is not None else []) + text.split()

    sources = {
        "small": {"tree": "A", "paths": [(("a",), "fn")]},
        "large": {"tree": "B", "paths": [(("a",), "fn"), (("a", "b"), "cls"), (("c",), "fn")]},
    }
    for source in sources.values():
        source.update(split="test", source_sha256="source", tree_sha256="tree")
    outcomes = {
        "TASK hier TREE A PATH a OUT fn": True,
        "TASK hier TREE B PATH a OUT fn": False,
        "TASK hier TREE B PATH a b OUT cls": True,
        "TASK hier TREE B PATH c OUT fn": True,
    }
    report = realhier_module_report(sources, WordTokenizer(), 9, outcomes)
    assert report["modules"]["large"]["fits"] == 2
    assert report["modules"]["large"]["majority_fraction"] == pytest.approx(2 / 3)
    assert report["macro_exact_match"] == pytest.approx((1 + 1 / 3) / 2)
    outcomes.pop("TASK hier TREE A PATH a OUT fn")
    assert realhier_module_report(sources, WordTokenizer(), 9, outcomes)["macro_exact_match"] is None


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

    splits = TASKS["regime"].generate(120, 9, TASKS["regime"].resolve_dials({"n_regimes": 3}))
    train_max = max(d.split().count("SEG") for d in splits["train"])
    test_min = min(d.split().count("SEG") for d in splits["test"])
    assert train_max == 3 and test_min == 6, f"regime: held-out must double the shift count ({train_max}, {test_min})"


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


def test_regime_checker_scores_the_continuation():
    """regime v2 (bead w76r): the OUT block must continue the LAST regime's
    rule; a checker that accepted any continuation could not fail."""
    doc = "TASK regime SEG m2 a1 n1 n3 n5 SEG m5 a90 n90 n95 n3 OUT n8 n13"
    assert check_regime(doc) is True
    assert check_regime("TASK regime SEG m2 a1 n1 n3 n5 SEG m5 a90 n90 n95 n3 OUT n8 n14") is False  # wrong value
    assert check_regime("TASK regime SEG m2 a1 n1 n3 n5 SEG m5 a90 n90 n95 n3 OUT n5 n7") is False  # first regime
    assert check_regime("TASK regime SEG m2 a1 n1 n4 n5 SEG m5 a90 n90 n95 n3 OUT n8 n13") is False  # broken stream
    assert check_regime("TASK regime SEG m2 a1 n1 n3 n5 OUT") is False  # empty answer
    assert check_regime("TASK regime SEG m2 a1 n1 n3 n5") is None  # v1 stream without an answer block
    spec = TASKS["regime"]
    for doc in spec.generate(30, 4, spec.resolve_dials({"answer_len": 3}))["test"]:
        split = spec.split_prompt(doc)
        assert split is not None and len(split[1].split()) == 3
        assert doc.split().count("SEG") == 6  # held-out doubles the default shift count (3)


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
        assert set(state) == {"pq_idx", "rg_idx", "epoch"}  # epoch: per-epoch shuffle state (2026-09-02)


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


def test_group_substitutions_need_only_equal_endpoint_names():
    from nanochat.group_acquisition import record
    from scripts.group_rewrite_probe import Rewriter, oracle, verify_rewrite

    # In S3 a transposition squared is the identity; insertion preserves any context.
    labels = {"11": "x", "0011": "x", "00": "y", "01": "z"}
    rules = Rewriter(labels)
    word = "100110"
    compact, proof = rules.reduce(word)
    assert (compact, proof) == ("1110", [(1, "0011", "11")])
    assert Rewriter({w: y + "!" for w, y in labels.items()}).reduce(word) == (compact, proof)
    verify_rewrite("s3", word, compact)
    assert oracle("s3", "01") != oracle("s3", "10")
    with pytest.raises(ValueError, match="changed"):
        verify_rewrite("s3", "01", "10")
    with pytest.raises(ValueError, match="Inconsistent"):
        record(labels, "11", "y")


def test_active_group_table_queries_only_missing_edges():
    from itertools import product

    from scripts.group_rewrite_probe import complete_action, follow, oracle, transitions

    def teacher(word):
        return f"opaque-{oracle('s3', word)}"

    labels = {a + b: teacher(a + b) for a in "01" for b in "01"}
    before = transitions(labels)
    calls = []

    def counted(word):
        assert word not in labels and word not in calls
        calls.append(word)
        return teacher(word)

    edges, reps, queries = complete_action(labels, counted)
    assert calls == [q[0] for q in queries]
    assert len(reps) == 6 and len(edges) == 12 and len(calls) == 12 - len(before)
    assert follow(labels, before, "0101") is None
    for letters in product("01", repeat=7):
        word = "".join(letters)
        assert follow(labels, edges, word) == teacher(word)


@pytest.mark.parametrize("last", range(3))
def test_prepend_append_rectangle_and_its_inverse(last):
    from nanochat.group_acquisition import Actions

    actions = Actions(set("abcdef"))
    known = [("L0", "a", "b"), ("R1", "a", "c"), ("R1", "b", "d")]
    for index, fact in enumerate(known):
        if index != last:
            actions.bind(*fact)
            actions.close(squares=True)
    actions.bind(*known[last])
    actions.close(squares=False)
    assert "c" not in actions.maps["L0"]
    actions.close(squares=True)
    assert actions.maps["L0"]["c"] == "d" and actions.maps["l0"]["d"] == "c"
    with pytest.raises(ValueError, match="Inconsistent"):
        actions.bind("L0", "c", "e")
    # Two prefixes act on the same side. They have no commuting-square constraint.
    actions = Actions(set("abcdef"))
    actions.bind("L0", "a", "b")
    actions.bind("L1", "a", "c")
    actions.bind("L1", "b", "d")
    actions.close(squares=True)
    assert "c" not in actions.maps["L0"]


def test_incremental_square_closure_keeps_reporting_a_conflict():
    from nanochat.group_acquisition import Actions

    actions = Actions(set("abcdef"))
    for fact in [("L0", "a", "b"), ("R1", "a", "c"), ("R1", "b", "d"), ("L0", "c", "e")]:
        actions.bind(*fact)
    for _ in range(2):
        with pytest.raises(ValueError, match="Inconsistent permutation edge"):
            actions.close(squares=True)


@pytest.mark.parametrize("mode", ["square", "tiered", "compiled", "bundled", "frontier"])
def test_exact_endpoint_acquisition_with_opaque_labels(mode):
    from nanochat.group_acquisition import acquire, compile_policy, load_bundled_policy, load_catalogue

    labels = [f"state {i:02x} / α" for i in range(32)]

    def step(state, mark):
        rotation, reflection = divmod(state, 2)
        if mark == "1":
            return state ^ 1
        return 2 * ((rotation + (-1 if reflection else 1)) % 16) + reflection

    words = []

    def teacher(word):
        assert len(word) >= 2 and word not in words
        words.append(word)
        state = 0
        for mark in word:
            state = step(state, mark)
        return labels[state]

    path = FIXTURE_PATH.parents[2] / "artifacts/probes/sizing/group-solver-baseline/anchor-results.zip"
    if mode == "bundled":
        policy = load_bundled_policy(len(labels))
    else:
        index = None if mode == "square" else load_catalogue(path, len(labels))
        policy = (
            compile_policy(index, exhaustive_frontier=mode == "frontier")
            if mode in ("compiled", "frontier") and index is not None
            else index
        )
    action = acquire(list(reversed(labels)), teacher, catalogue=policy)
    assert words[:4] == ["00", "01", "10", "11"]
    assert action.root == labels[0]
    assert action.terminal == ("square" if mode == "square" else "unique")
    assert action.transitions == {label: (labels[step(i, "0")], labels[step(i, "1")]) for i, label in enumerate(labels)}


@pytest.mark.parametrize("size", [4, 6, 8])
def test_exhaustive_frontier_matches_unrestricted_word_enumeration(size):
    from itertools import product

    from nanochat.group_acquisition import ANCHORS, _prepare, _VersionSpace

    def rooted(edges):
        reached = [0]
        canonical = []
        for state in reached:
            for target in edges[state]:
                if target not in reached:
                    reached.append(target)
            canonical.append(tuple(reached.index(target) for target in edges[state]))
        return tuple(canonical)

    shapes = set()
    for a, b in product(range(size), repeat=2):
        for group in (
            lambda x, y: (x + y) % size,
            lambda x, y: ((x // 2 + y // 2) % (size // 2)) * 2 + ((x ^ y) & 1),
        ):
            shape = rooted([(group(i, a), group(i, b)) for i in range(size)])
            if len(shape) == size:
                shapes.add(shape)
    shapes = sorted(shapes)

    def follow(graph, word):
        state = 0
        for mark in word:
            state = graph[state][int(mark)]
        return state

    index = _prepare(shapes)
    checked = 0
    for truth in shapes:
        version = _VersionSpace(index, size)
        for word in ANCHORS:
            version.learn(word, str(follow(truth, word)))
        while len(version.live) > 1:
            selected = version.select_frontier()
            observed = []
            for graph in shapes:
                labels = {follow(graph, word): label for word, label in version.facts.items()}
                if len(labels) == len(set(version.facts.values())) and all(
                    labels[follow(graph, word)] == label for word, label in version.facts.items()
                ):
                    observed.append((graph, labels))
            assert len(observed) == len(version.live)
            unused = max(1, size - len(set(version.facts.values())))
            choices = []
            for depth in range(2, len(selected) + 1):
                for marks in product("01", repeat=depth):
                    word = "".join(marks)
                    buckets = Counter(labels.get(follow(graph, word)) for graph, labels in observed)
                    score = max(buckets.pop(None, 0), unused * max(buckets.values(), default=0))
                    if score < unused * len(observed):
                        choices.append((depth, score, word))
            assert min(choices)[2] == selected
            checked += 1
            version.learn(selected, str(follow(truth, selected)))
    assert checked > 0


def test_compiled_acquisition_rejects_an_impossible_equality():
    from nanochat.group_acquisition import acquire, compile_policy, load_catalogue

    path = FIXTURE_PATH.parents[2] / "artifacts/probes/sizing/group-solver-baseline/anchor-results.zip"
    policy = compile_policy(load_catalogue(path, 32))
    labels = [f"state-{i}" for i in range(32)]
    observations = dict(zip(("00", "01", "10"), (labels[0], labels[0], labels[1]), strict=True))
    # Equality of the first two answers implies the two marks coincide.
    with pytest.raises(ValueError, match="no action for the observed endpoints"):
        acquire(labels, observations.__getitem__, catalogue=policy)


def test_bundled_policy_rejects_a_changed_resource(tmp_path, monkeypatch):
    from nanochat import group_acquisition

    resources = tmp_path / "research_data"
    resources.mkdir()
    (resources / "group_policies.zip").write_bytes(b"corrupt compiled policy")
    monkeypatch.setattr(group_acquisition, "files", lambda _: tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        group_acquisition.load_bundled_policy(32)


@pytest.mark.parametrize(
    ("reply", "event", "message"),
    [
        ("", "eof", "closed its input"),
        ("xyz\n", "invalid_reply", "Expected a JSON reply"),
        ('{"label":"home","label":"home"}\n', "invalid_reply", "Expected one string label"),
        ('{"label":"unknown"}\n', "reply", "outside the roster"),
    ],
)
def test_group_acquire_retains_failed_oracle_interaction(tmp_path, reply, event, message):
    from typer.testing import CliRunner

    from cli import app

    roster = tmp_path / "roster.json"
    roster.write_text('["home"]', encoding="utf-8")
    transcript = tmp_path / "session.jsonl"
    result = CliRunner().invoke(app, ["group-acquire", str(roster), "--transcript", str(transcript)], input=reply)
    assert result.exit_code == 2
    assert json.loads(result.stdout) == {"word": "00"}
    assert message in result.stderr
    rows = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["start", "request", event, "error"]
    assert rows[2]["roundtrip_ns"] >= 0


def test_group_acquire_validates_inputs_before_querying(tmp_path):
    from typer.testing import CliRunner

    from cli import app

    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps([str(i) for i in range(32)]), encoding="utf-8")
    catalogue = tmp_path / "catalogue.zip"
    catalogue.write_bytes(b"unverified inventory")
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("existing evidence\n", encoding="utf-8")
    runner = CliRunner()

    invalid = runner.invoke(app, ["group-acquire", str(roster), "--catalogue", str(catalogue)])
    assert invalid.exit_code == 2 and not invalid.stdout
    assert "Catalogue checksum mismatch" in invalid.stderr

    conflict = runner.invoke(app, ["group-acquire", str(roster), "--compiled", "--catalogue", str(catalogue)])
    assert conflict.exit_code == 2 and not conflict.stdout
    assert "Choose either --compiled or --catalogue" in conflict.stderr

    occupied = runner.invoke(app, ["group-acquire", str(roster), "--transcript", str(transcript)])
    assert occupied.exit_code == 2 and not occupied.stdout
    assert transcript.read_text(encoding="utf-8") == "existing evidence\n"

    roster.write_text('["home"]', encoding="utf-8")
    unsupported = runner.invoke(app, ["group-acquire", str(roster), "--compiled"])
    assert unsupported.exit_code == 2 and not unsupported.stdout
    assert "covers orders 32, 64, 128" in unsupported.stderr

    roster.write_text('["same","same"]', encoding="utf-8")
    bad_roster = runner.invoke(app, ["group-acquire", str(roster)])
    assert bad_roster.exit_code == 2 and not bad_roster.stdout
    assert "Roster labels must be distinct" in bad_roster.stderr


def test_equality_only_action_recovers_every_edge_with_a_wrong_ranker():
    from scripts.group_student_probe import acquire_equalities, cached_equality

    def action(word):
        # The second generator jumps two states and the first jumps one.
        return (word.count("0") + 2 * word.count("1")) % 5

    def equal(a, b):
        return action(a) == action(b)

    ordinary = acquire_equalities(5, equal, lambda *_: 0.0)
    misleading = acquire_equalities(5, equal, lambda a, b: float(not equal(a, b)))
    assert ordinary["representatives"] == misleading["representatives"]
    reps = misleading["representatives"]
    assert len(reps) == len({action(w) for w in reps}) == 5
    assert misleading["edges"] == [[reps.index(next(r for r in reps if equal(w + g, r))) for g in "01"] for w in reps]
    assert misleading["counts"]["oracle_different"] >= ordinary["counts"]["oracle_different"]
    for size in (4, 6):
        with pytest.raises(ValueError, match="size|reachable"):
            acquire_equalities(size, equal, lambda *_: 0.0)
    short = {"00": "c", "000": "d", "001": "e"}
    assert cached_equality(short, "", "0") is False
    assert cached_equality(short, "00", "00") is True
    assert cached_equality(short, "00", "1111") is None


def test_equivalence_cache_keys_use_only_group_identities():
    from scripts.group_student_probe import acquire_with_memo, cancelled_key, pair_key, relator_key

    assert cancelled_key("0100", "0110") == ("0", "1")
    assert cancelled_key("001", "") != cancelled_key("010", "")
    assert relator_key("001", "") == relator_key("010", "")
    assert relator_key("0100", "0110") == relator_key("1", "0")
    assert relator_key("0011", "") != relator_key("0101", "")
    assert relator_key("010", "010") == ()

    def equal(a, b):
        def eval_word(word):
            p = (0, 1, 2)
            generators = ((1, 0, 2), (1, 2, 0))
            for g in word:
                p = tuple(generators[int(g)][i] for i in p)
            return p

        return eval_word(a) == eval_word(b)

    raw, cancelled, cyclic = [acquire_with_memo(6, equal, key) for key in (pair_key, cancelled_key, relator_key)]
    order_free = acquire_with_memo(None, equal, cancelled_key)
    assert raw["edges"] == cancelled["edges"] == cyclic["edges"] == order_free["edges"]
    assert order_free["representatives"] == cancelled["representatives"]
    assert "forced" not in order_free["counts"]
    assert cyclic["counts"]["cache_different"] > 0


def test_order_free_acquisition_checks_both_generators_of_the_trivial_action():
    from scripts.group_student_probe import acquire_with_memo, cancelled_key

    calls = []

    def equal(word, representative):
        calls.append((word, representative))
        return True

    result = acquire_with_memo(None, equal, cancelled_key)
    assert calls == [("0", ""), ("1", "")]
    assert result["representatives"] == [""]
    assert result["edges"] == [[0, 0]]
    assert result["counts"] == {"oracle_equal": 2}


@pytest.mark.parametrize("frontier_finishes", [False, True])
def test_bounded_proposal_shares_reversed_answers_and_prefers_a_completed_frontier(frontier_finishes):
    from scripts.group_student_probe import run_bounded_proposal

    def frontier():
        assert (yield "0", "") is True
        if not frontier_finishes:
            yield "1", ""
        return {"edges": [[0, 0]]}

    def proposal():
        assert (yield "", "0") is True
        return {"edges": [[0, 0]]}

    def equal(left, right):
        assert (left, right) == ("0", "")
        return True

    result = run_bounded_proposal(frontier(), proposal(), equal)
    assert result["winner"] == ("frontier" if frontier_finishes else "proposal")
    assert result["counts"] == [1, 0]
    assert result["paid"] == [("frontier", "0", "", True)]


@pytest.mark.parametrize("accepts", [False, True])
def test_bounded_proposal_earns_budget_and_keeps_the_frontier_after_rejection(accepts):
    from scripts.group_student_probe import run_bounded_proposal

    def frontier():
        for word in ("0", "00", "000", "0000", "00000", "000000"):
            assert (yield word, "") is True
        return {"edges": [[0, 0]]}

    def proposal():
        answer = yield "1", ""
        return {"edges": [[0, 0]]} if answer else None

    result = run_bounded_proposal(frontier(), proposal(), lambda left, _right: left != "1" or accepts)
    assert result["winner"] == ("proposal" if accepts else "frontier")
    assert result["counts"] == ([4, 1] if accepts else [6, 1])
    assert [owner for owner, *_ in result["paid"][:5]] == ["frontier"] * 4 + ["proposal"]


def test_congruence_combines_shortlex_rules_and_preserves_the_action():
    from scripts.group_student_probe import ShortlexRelations, acquire_with_memo, cancelled_key

    rules = ShortlexRelations()
    assert rules.normalize("1000") == ("1000", ())
    rules.record("000", "1")
    rules.record("11", "")
    assert rules.normalize("1000") == ("", ((1, "000", "1"), (0, "11", "")))
    with pytest.raises(ValueError, match="shortlex"):
        rules.record("0", "000")

    def equality(a, b):
        def exponent(word):
            return (word.count("0") + 3 * word.count("1")) % 6

        return exponent(a) == exponent(b)

    baseline = acquire_with_memo(6, equality, cancelled_key)
    learned = ShortlexRelations()
    treatment = acquire_with_memo(6, equality, learned.comparison_key, relations=learned)
    assert treatment["edges"] == baseline["edges"]
    assert treatment["counts"]["rewritten"] > 0


def test_known_family_uses_a_balanced_separating_query():
    from scripts.group_student_probe import identify_known_family, walk_action

    target = ((1, 0), (2, 1), (0, 2))
    alternate = ((0, 1), (1, 2), (2, 0))
    result = identify_known_family(
        {"first": alternate, "second": target}, lambda a, b: walk_action(a, target) == walk_action(b, target)
    )
    assert result["marked_class"] == "second"
    (question,) = result["queries"]
    assert question["candidates"] == 2 and question["predicted_yes"] == 1
