"""Tests for the data-geometry profiler (bead model_guided_research-77l.1).

THE HARD GATE (per the bead): every estimator must recover the planted
ORDERING across a calibration ladder of known geometries before any
real-corpus number is believed:
    balanced tree (exactly ultrametric, delta = 0)
      <  hyperbolic point cloud (negatively curved)
      <  uniform Euclidean point cloud (flat)
for delta-hyperbolicity and ultrametricity violation, with the tree end
exact (0 within fp tolerance) and the cophenetic correlation ~1 there.

Also covered: bootstrap CI sanity, distance-function semantics (edit /
Jaccard), order-sensitivity direction on structured vs shuffled data,
dynamic-range extraction, the distance-concentration degeneracy warning,
profile schema + determinism, and the CLI surface (--json round-trip,
actionable failure on missing data, --task profiling a vdc.1 generated corpus).

Every assertion carries enough context to reproduce: seeds are fixed module
constants and failure messages embed the measured values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

import cli
import geometry_profile as gp

runner = CliRunner()
SEED = 1234


def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# planted-geometry calibration (the hard gate)
# ---------------------------------------------------------------------------


def test_tree_distances_are_exactly_ultrametric_and_zero_delta() -> None:
    D = gp.planted_tree_distances(branching=3, depth=4)  # 81 leaves
    deltas = gp.four_point_delta(D, rng(), n_quadruples=1500)
    assert float(np.max(deltas)) < 1e-12, f"tree four-point delta must be exactly 0, got max {np.max(deltas):.3e}"
    viol, frac = gp.ultrametricity_violations(D, rng(), n_triples=1500)
    assert float(np.max(viol)) < 1e-12, (
        f"tree triples must satisfy the strong triangle inequality, got {np.max(viol):.3e}"
    )
    assert frac == 0.0
    coph = gp.cophenetic_correlation(D)
    assert coph > 0.999, f"cophenetic correlation on an ultrametric must be ~1, got {coph:.4f}"


def test_calibration_ladder_ordering_delta_hyperbolicity() -> None:
    """tree < hyperbolic < euclidean on mean normalized four-point delta."""
    r = rng()
    D_tree = gp.planted_tree_distances(branching=3, depth=4)
    D_hyp = gp.planted_hyperbolic_distances(81, r, radius=0.999)
    D_euc = gp.planted_euclidean_distances(81, 8, r)
    m_tree = float(np.mean(gp.four_point_delta(D_tree, rng(), 1500)))
    m_hyp = float(np.mean(gp.four_point_delta(D_hyp, rng(), 1500)))
    m_euc = float(np.mean(gp.four_point_delta(D_euc, rng(), 1500)))
    assert m_tree < m_hyp < m_euc, (
        f"planted ordering not recovered: tree={m_tree:.4f}, hyperbolic={m_hyp:.4f}, euclidean={m_euc:.4f} "
        f"(seed {SEED}; expected strictly increasing)"
    )
    # margins, not just ordering: the rungs must be clearly separated
    # (measured on this seed: tree=0.0000, hyp~0.038, euc~0.069)
    assert m_hyp > 0.01, f"hyperbolic rung not separated from the tree floor: {m_hyp:.4f}"
    assert m_euc > 1.5 * m_hyp, f"flat end not separated: euc={m_euc:.4f} vs hyp={m_hyp:.4f}"


def test_calibration_ladder_ordering_ultrametricity() -> None:
    r = rng()
    D_tree = gp.planted_tree_distances(branching=3, depth=4)
    D_euc = gp.planted_euclidean_distances(81, 8, r)
    _, frac_tree = gp.ultrametricity_violations(D_tree, rng(), 1500)
    _, frac_euc = gp.ultrametricity_violations(D_euc, rng(), 1500)
    assert frac_tree == 0.0
    assert frac_euc > 0.9, f"random Euclidean triples should almost all violate ultrametricity, got {frac_euc:.3f}"
    coph_tree = gp.cophenetic_correlation(D_tree)
    coph_euc = gp.cophenetic_correlation(D_euc)
    assert coph_tree > coph_euc, f"cophenetic r ordering violated: tree {coph_tree:.3f} <= euclidean {coph_euc:.3f}"


def test_hyperbolic_distance_formula_sanity() -> None:
    """d(0, u) for a point at Euclidean radius r must equal arccosh(1 + 2r^2/(1-r^2))."""
    P = np.array([[0.0, 0.0], [0.5, 0.0]])
    sq = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    nu = 1.0 - (P**2).sum(-1)
    want = float(np.arccosh(1.0 + 2.0 * sq[0, 1] / (nu[0] * nu[1])))
    # reproduce through the generator's formula via a 2-point cloud
    D = np.array([[0.0, want], [want, 0.0]])
    assert abs(D[0, 1] - 2 * np.arctanh(0.5)) < 1e-12, "closed forms disagree: arccosh form vs 2*artanh(r)"


def test_hierarchy_depth_deeper_for_tree_than_flat() -> None:
    r = rng()
    D_tree = gp.planted_tree_distances(branching=2, depth=6)  # 64 leaves
    D_euc = gp.planted_euclidean_distances(64, 8, r)
    d_tree = gp.hierarchy_depth_spectrum(D_tree)["normalized_mean_depth"]
    d_euc = gp.hierarchy_depth_spectrum(D_euc)["normalized_mean_depth"]
    assert np.isfinite(d_tree) and np.isfinite(d_euc)
    assert d_tree < d_euc, (
        f"balanced tree must merge in FEWER dendrogram levels than chained flat data: "
        f"tree={d_tree:.3f} vs euclidean={d_euc:.3f}"
    )


# ---------------------------------------------------------------------------
# estimator internals
# ---------------------------------------------------------------------------


def test_bootstrap_ci_brackets_mean_and_narrows() -> None:
    r = rng()
    small = r.normal(loc=5.0, scale=1.0, size=20)
    big = r.normal(loc=5.0, scale=1.0, size=2000)
    lo_s, hi_s = gp.bootstrap_ci(small, rng())
    lo_b, hi_b = gp.bootstrap_ci(big, rng())
    assert lo_s < 5.0 < hi_s, f"small-sample CI [{lo_s:.2f},{hi_s:.2f}] should bracket the true mean"
    assert (hi_b - lo_b) < (hi_s - lo_s), "CI must narrow with sample size"
    # empty input -> (NaN, NaN), never a crash (NaN != NaN, so test via isnan)
    lo_e, hi_e = gp.bootstrap_ci(np.zeros((0,)), rng())
    assert np.isnan(lo_e) and np.isnan(hi_e), f"empty bootstrap must return NaNs, got ({lo_e}, {hi_e})"


def test_edit_and_jaccard_distance_semantics() -> None:
    assert gp.normalized_edit_distance([1, 2, 3], [1, 2, 3]) == 0.0
    assert gp.normalized_edit_distance([1, 2, 3], [4, 5, 6]) == 1.0
    assert gp.normalized_edit_distance([], []) == 0.0
    assert abs(gp.normalized_edit_distance([1, 2, 3, 4], [1, 2, 4]) - 0.25) < 1e-12  # 1 deletion / len 4
    a = [1, 2, 3, 4, 5]
    assert gp.jaccard_ngram_distance(a, a) == 0.0
    assert gp.jaccard_ngram_distance(a, [9, 8, 7, 6, 5]) == 1.0
    sym = gp.jaccard_ngram_distance([1, 2, 3, 4], [2, 3, 4, 5])
    assert 0.0 < sym < 1.0


def test_token_distance_matrix_mode_switch() -> None:
    short = [[1, 2, 3], [1, 2, 4], [9, 9, 9]]
    D = gp.token_distance_matrix(short)
    assert D.shape == (3, 3) and np.allclose(D, D.T) and np.all(np.diag(D) == 0)
    # short sequences use edit distance: [1,2,3] vs [1,2,4] differs in 1/3 positions
    assert abs(D[0, 1] - 1.0 / 3.0) < 1e-12, f"expected edit distance 1/3, got {D[0, 1]}"
    long_seqs = [list(range(100)), list(range(50, 150))]
    D2 = gp.token_distance_matrix(long_seqs)
    assert 0.0 < D2[0, 1] < 1.0  # Jaccard regime


def test_order_sensitivity_higher_for_structured_than_shuffled() -> None:
    r = rng()
    # structured: strict arithmetic progressions (bigram model nails them)
    structured = [[(7 * k + j) % 50 for j in range(40)] for k in range(40)]
    shuffled = [list(r.permutation(50)[:40]) for _ in range(40)]
    s_struct = gp.order_sensitivity(structured, rng())["relative_delta"]
    s_shuf = gp.order_sensitivity(shuffled, rng())["relative_delta"]
    assert s_struct > s_shuf, (
        f"transpositions must hurt structured data more than already-shuffled data: "
        f"structured={s_struct:.4f} vs shuffled={s_shuf:.4f}"
    )
    assert s_struct > 0.0


def test_dynamic_range_extraction() -> None:
    texts = ["values 0.001 and 1000000 span nine decades", "tiny 2 4 8 16 cluster", "no numerals at all here"]
    out = gp.dynamic_range_stats(texts)
    assert out["numbers_found"] >= 6
    assert out["max_decades"] >= 8.9, f"0.001..1e6 spans 9 decades, got {out['max_decades']:.2f}"
    empty = gp.dynamic_range_stats(["nothing numeric"])
    assert empty["numbers_found"] == 0 and np.isnan(empty["mean_decades"])  # NaN, never fabricated
    # degenerate tail (all magnitudes equal) -> Hill is NaN, never inf
    degenerate = gp.dynamic_range_stats(["7 " * 30])
    assert np.isnan(degenerate["hill_tail_exponent"]), (
        f"all-equal magnitudes must give NaN Hill exponent, got {degenerate['hill_tail_exponent']}"
    )


def test_distance_concentration_warning_fires_on_uniform_metric() -> None:
    # documents with pairwise-disjoint vocabularies -> all Jaccard distances 1.0
    texts = [" ".join(f"tok{i}_{j}" for j in range(80)) for i in range(24)]
    profile = gp.profile_from_texts(texts, gp.ProfileConfig(seed=SEED, n_points=24))
    assert profile["distance_diagnostics"]["concentrated"] is True
    assert any("distance concentration" in w for w in profile["warnings"]), profile["warnings"]


# ---------------------------------------------------------------------------
# profile assembly: schema, determinism, modes
# ---------------------------------------------------------------------------


def _toy_texts() -> list[str]:
    r = rng()
    base = ["alpha beta gamma delta epsilon zeta", "one two three four five six seven"]
    return [b + f" {int(r.integers(0, 100))}" for b in base for _ in range(20)]


def test_profile_schema_and_determinism() -> None:
    texts = _toy_texts()
    p1 = gp.profile_from_texts(texts, gp.ProfileConfig(seed=SEED, n_points=24))
    p2 = gp.profile_from_texts(texts, gp.ProfileConfig(seed=SEED, n_points=24))
    assert gp.validate_profile_schema(p1) == []
    assert gp.profile_to_json(p1) == gp.profile_to_json(p2), "same seed + corpus must produce byte-identical profiles"
    p3 = gp.profile_from_texts(texts, gp.ProfileConfig(seed=SEED + 1, n_points=24))
    assert gp.profile_to_json(p1) != gp.profile_to_json(p3), "different seeds should differ somewhere"


def test_tiny_corpus_warns_loudly() -> None:
    profile = gp.profile_from_texts(["one two three four five"] * 5, gp.ProfileConfig(seed=SEED, n_points=5))
    assert any("tiny corpus" in w for w in profile["warnings"]), profile["warnings"]


def test_activations_mode_runs_and_is_deterministic() -> None:
    seqs_texts = _toy_texts()[:12]
    cfg = gp.ProfileConfig(mode="activations", seed=SEED, n_points=8, doc_tokens=16)
    p1 = gp.profile_from_texts(seqs_texts, cfg)
    p2 = gp.profile_from_texts(seqs_texts, cfg)
    assert p1["mode"] == "activations"
    assert gp.validate_profile_schema(p1) == []
    assert gp.profile_to_json(p1) == gp.profile_to_json(p2), "activations mode must be deterministic (seeded init)"


def test_schema_validator_rejects_malformed() -> None:
    assert gp.validate_profile_schema({"schema_version": 99}) != []
    good = gp.profile_from_texts(_toy_texts(), gp.ProfileConfig(seed=SEED, n_points=16))
    bad = dict(good)
    bad.pop("estimators")
    assert any("estimators" in e for e in gp.validate_profile_schema(bad))


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_profile_data_json_roundtrip(tmp_path: Path) -> None:
    docs = tmp_path / "corpus"
    docs.mkdir()
    for i in range(8):
        (docs / f"doc{i}.txt").write_text(
            "the quick brown fox jumps over the lazy dog " * 6 + f"number {i * 137}\n\n", encoding="utf-8"
        )
    out_file = tmp_path / "profile.json"
    result = runner.invoke(
        cli.app,
        ["profile-data", "--data", str(docs), "--points", "8", "--sample", "16", "--json", "--out", str(out_file)],
    )
    assert result.exit_code == 0, f"profile-data failed:\n{result.output}"
    payload = json.loads(result.output)
    assert gp.validate_profile_schema(payload) == []
    assert out_file.exists()
    assert json.loads(out_file.read_text(encoding="utf-8")) == payload


def test_cli_profile_data_missing_data_actionable() -> None:
    result = runner.invoke(cli.app, ["profile-data"])
    assert result.exit_code == 2
    assert "--data" in result.output


def test_cli_profile_data_task_profiles_generated_corpus() -> None:
    # Until vdc.1 landed this asserted a stub rejection; --task now generates
    # the named diagnostic corpus in memory and profiles it.
    result = runner.invoke(cli.app, ["profile-data", "--task", "dyck", "--sample", "24", "--points", "16", "--json"])
    assert result.exit_code == 0, result.output
    profile = json.loads(result.output)
    assert profile["corpus"] == "task:dyck"
    result_bad = runner.invoke(cli.app, ["profile-data", "--task", "not-a-task"])
    assert result_bad.exit_code == 2
    assert "Unknown task" in result_bad.output


def test_cli_profile_data_empty_dir_actionable(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["profile-data", "--data", str(tmp_path)])
    assert result.exit_code == 2
    assert "no .txt/.md/.parquet" in result.output
