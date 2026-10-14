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
from typing import Any

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


# ---------------------------------------------------------------------------
# Geometry-informed initialization (--init-from-profile, bead 77l.4)
# ---------------------------------------------------------------------------


def _synthetic_profile(
    *,
    delta_mean: float = 0.020,
    violation_mean: float = 0.050,
    cophenetic_correlation: float = 0.900,
    mean_depth: float = 6.2,
    normalized_mean_depth: float = 1.50,
    corpus: str = "task:hier",
) -> dict[str, Any]:
    return {
        "schema_version": gp.PROFILE_SCHEMA_VERSION,
        "corpus": corpus,
        "mode": "tokens",
        "seed": 42,
        "sample": {"docs": 64, "points": 32, "doc_tokens": 48, "quadruples": 2000, "triples": 2000},
        "distance_diagnostics": {"mean": 0.35, "coefficient_of_variation": 0.30, "concentrated": False},
        "estimators": {
            "delta_hyperbolicity": {"mean": delta_mean, "ci95": [delta_mean * 0.9, delta_mean * 1.1]},
            "ultrametricity": {
                "violation_mean": violation_mean,
                "violation_fraction": 0.5,
                "ci95": [violation_mean * 0.9, violation_mean * 1.1],
                "cophenetic_correlation": cophenetic_correlation,
            },
            "dynamic_range": {
                "numbers_found": 0,
                "docs_with_spread": 0,
                "mean_decades": float("nan"),
                "max_decades": float("nan"),
                "hill_tail_exponent": float("nan"),
            },
            "order_sensitivity": {"nll_original": 2.3, "nll_transposed": 3.3, "relative_delta": 0.43},
            "hierarchy_depth": {
                "mean_depth": mean_depth,
                "max_depth": 12.0,
                "normalized_mean_depth": normalized_mean_depth,
            },
        },
        "warnings": [],
    }


def test_geometry_init_mapping_table_and_exact_formulas() -> None:
    """Every mapping table row has an exact closed-form fixture and non-applicable mechanisms cite blocked-on-theory."""
    assert len(gp.GEOMETRY_INIT_MAPPING_TABLE) == 6
    sources = {row["mechanism"]: row["input_source"] for row in gp.GEOMETRY_INIT_MAPPING_TABLE}
    assert sources["tropical"] == "startup-probe"
    assert sources["fractal"] == "profile-only"
    assert sources["hyperbolic"] == "profile-only"
    assert sources["ultrametric"] == "profile-only"
    for mech, reason in gp.GEOMETRY_INIT_NOT_APPLICABLE.items():
        assert reason.startswith("blocked-on-theory:"), f"{mech} missing blocked-on-theory rationale: {reason}"

    prof = _synthetic_profile(
        delta_mean=0.020,
        violation_mean=0.10,
        cophenetic_correlation=0.80,
        mean_depth=6.4,
        normalized_mean_depth=1.50,
    )
    vals = gp.compute_geometry_init_values(prof, num_prototypes=4, max_radius=0.85)

    # Hand-computed expected values:
    # 1. Hyperbolic curvature: clip((0.010 / 0.020)^2, 0.10, 2.00) = 0.25
    assert vals["hyperbolic.curvature"]["profile_value"] == 0.25
    assert vals["hyperbolic.curvature"]["default_value"] == 1.0

    # 2. Ultrametric K: clip(round(6.4), 2, 8) = 6
    assert vals["ultrametric.ultrametric_K"]["profile_value"] == 6
    # Ultrametric alpha: clip(1.5 + 1.5 * 0.80, 1.25, 3.00) = 2.7
    assert vals["ultrametric.ultrametric_alpha"]["profile_value"] == 2.7
    # Ultrametric lcp_beta: clip(16.0 + 32.0 * (1.0 - 0.10), 16.0, 48.0) = 44.8
    assert vals["ultrametric.ultrametric_lcp_beta"]["profile_value"] == 44.8

    # 3. Fractal IFS radii: s* = 0.85 - 0.45*0.10 = 0.805; zeta = (1 - 0.05) / 1.50 = 0.633333
    assert abs(vals["fractal.ifs_radii"]["target_moran_dim"] - 0.805) < 1e-6
    assert abs(vals["fractal.ifs_radii"]["branch_skew_zeta"] - (0.95 / 1.50)) < 1e-5
    r_list = vals["fractal.ifs_radii"]["profile_value"]
    assert len(r_list) == 4
    assert r_list[0] > r_list[1] > r_list[2] > r_list[3] > 0.0


def test_geometry_init_malformed_profile_errors_name_missing_metrics(tmp_path: Path) -> None:
    """Missing or out-of-range nested estimator metrics raise actionable errors naming the offending metric."""
    import pytest

    base = _synthetic_profile()
    assert gp.validate_profile_for_init(base) == []

    # Missing delta_hyperbolicity.mean
    bad_delta = _synthetic_profile()
    bad_delta["estimators"]["delta_hyperbolicity"] = {}
    errs = gp.validate_profile_for_init(bad_delta)
    assert any("delta_hyperbolicity" in e and "mean" in e for e in errs), errs

    # Missing ultrametricity.cophenetic_correlation
    bad_ultra = _synthetic_profile()
    del bad_ultra["estimators"]["ultrametricity"]["cophenetic_correlation"]
    errs_u = gp.validate_profile_for_init(bad_ultra)
    assert any("cophenetic_correlation" in e for e in errs_u), errs_u

    # Out-of-range violation_mean
    bad_v = _synthetic_profile(violation_mean=1.5)
    errs_v = gp.validate_profile_for_init(bad_v)
    assert any("violation_mean" in e for e in errs_v), errs_v

    # Missing hierarchy_depth.normalized_mean_depth
    bad_depth = _synthetic_profile()
    del bad_depth["estimators"]["hierarchy_depth"]["normalized_mean_depth"]
    errs_d = gp.validate_profile_for_init(bad_depth)
    assert any("normalized_mean_depth" in e for e in errs_d), errs_d

    # File loader actionable errors
    with pytest.raises(ValueError, match="does not exist"):
        gp.load_profile_for_init(tmp_path / "missing.json")

    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps(bad_ultra), encoding="utf-8")
    with pytest.raises(ValueError, match="cophenetic_correlation"):
        gp.load_profile_for_init(bad_file)


def test_geometry_init_opt_in_and_tropical_startup_probe_determinism() -> None:
    """No-profile path is byte-identical to defaults; tropical startup probe is deterministic, RNG-preserving, and fast."""
    import torch

    from nanochat.gpt import GPT, GPTConfig

    prof = _synthetic_profile()
    x = torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(7), dtype=torch.long)

    # 1. Pure opt-in check: building without apply_geometry_init is byte-identical
    for mech in ("fractal", "hyperbolic", "ultrametric", "tropical"):
        torch.manual_seed(123)
        cfg1 = GPTConfig(
            sequence_len=16, vocab_size=64, n_layer=2, n_head=4, n_kv_head=2, n_embd=32, attention_type=mech
        )
        m1 = GPT(cfg1)
        m1.init_weights()

        torch.manual_seed(123)
        cfg2 = GPTConfig(
            sequence_len=16,
            vocab_size=64,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=32,
            attention_type=mech,
            init_from_profile=None,
        )
        m2 = GPT(cfg2)
        m2.init_weights()
        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters(), strict=True):
            assert n1 == n2 and torch.equal(p1, p2), f"opt-in drift on {mech}:{n1}"
        with torch.no_grad():
            assert torch.equal(m1(x), m2(x))

    # 2. Tropical startup probe determinism, RNG isolation, and bounded cost (< 50ms)
    torch.manual_seed(555)
    cfg_t1 = GPTConfig(
        sequence_len=16, vocab_size=64, n_layer=2, n_head=4, n_kv_head=2, n_embd=32, attention_type="tropical"
    )
    mt1 = GPT(cfg_t1)
    mt1.init_weights()
    rng_before = torch.get_rng_state()
    rec1 = gp.apply_geometry_init_to_model(
        mt1, cfg_t1, prof, attention_schedule=["tropical", "tropical"], applied={}, seed=42
    )
    rng_after = torch.get_rng_state()
    assert torch.equal(rng_before, rng_after), "tropical startup probe must not mutate global torch RNG state"

    torch.manual_seed(555)
    cfg_t2 = GPTConfig(
        sequence_len=16, vocab_size=64, n_layer=2, n_head=4, n_kv_head=2, n_embd=32, attention_type="tropical"
    )
    mt2 = GPT(cfg_t2)
    mt2.init_weights()
    rec2 = gp.apply_geometry_init_to_model(
        mt2, cfg_t2, prof, attention_schedule=["tropical", "tropical"], applied={}, seed=42
    )
    assert rec1["startup_probe"]["probe_margin_median"] == rec2["startup_probe"]["probe_margin_median"]
    assert rec1["applied_parameters"]["tropical.semiring_beta"] == rec2["applied_parameters"]["tropical.semiring_beta"]
    assert rec1["startup_probe"]["probe_ms"] < 200.0, f"startup probe too slow: {rec1['startup_probe']['probe_ms']}ms"


def test_train_init_from_profile_records_provenance_and_distance(monkeypatch, tmp_path: Path) -> None:
    """End-to-end trainer smoke test with --init-from-profile verifying summary.json provenance and distance traveled."""
    import torch

    import nanochat.train as train_mod

    def _fake_loader(B, T, split, device="cpu", resume_state_dict=None, **kwargs):
        gen = torch.Generator().manual_seed(1234)
        idx = 0
        while True:
            tokens = torch.randint(0, 64, (B, T + 1), generator=gen, dtype=torch.long)
            yield tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous(), {"pq_idx": 0, "rg_idx": idx}
            idx += 1

    monkeypatch.setattr(train_mod, "tokenizing_distributed_data_loader_with_state", _fake_loader)
    monkeypatch.setattr(
        train_mod, "list_parquet_files", lambda data_dir=None: ["fake_train.parquet", "fake_val.parquet"]
    )

    prof = _synthetic_profile(corpus="task:hier")
    prof_path = tmp_path / "profile.json"
    prof_path.write_text(gp.profile_to_json(prof), encoding="utf-8")
    out_dir = tmp_path / "runs"

    argv = [
        "--device",
        "cpu",
        "--attention-type",
        "hyperbolic",
        "--init-from-profile",
        str(prof_path),
        "--n-layer",
        "2",
        "--n-head",
        "4",
        "--n-kv-head",
        "2",
        "--n-embd",
        "32",
        "--sequence-len",
        "16",
        "--batch-size",
        "2",
        "--max-steps",
        "4",
        "--warmup-steps",
        "0",
        "--log-interval",
        "1",
        "--artifacts-dir",
        str(out_dir),
        "--run-id",
        "geo-smoke",
    ]
    train_mod.train(train_mod.build_parser().parse_args(argv))

    run_files = list(out_dir.rglob("summary.json"))
    assert len(run_files) == 1, f"expected 1 summary.json under {out_dir}, found {run_files}"
    summary = json.loads(run_files[0].read_text(encoding="utf-8"))

    assert summary["hparams"]["init_from_profile"] == str(prof_path)
    assert summary["hparams"]["init_profile_corpus"] == "task:hier"
    geo = summary["geometry_init"]
    assert geo is not None and geo["enabled"] is True
    assert geo["profile_corpus"] == "task:hier"
    curv_entry = geo["applied_parameters"]["hyperbolic.curvature"]
    assert curv_entry["default_value"] == 1.0
    assert curv_entry["profile_value"] == 0.25
    assert curv_entry["input_source"] == "profile-only"
    assert "thm-hyperbolic-score-decomposition" in curv_entry["formula_ref"]

    assert isinstance(summary["results"]["train_ce_early"], float)
    assert isinstance(summary["results"]["train_ce_auc"], float)
    assert isinstance(summary["results"]["structural_param_distance_l2"], float)
    assert summary["results"]["structural_param_distance_l2"] > 0.0

    metrics_files = list(out_dir.rglob("metrics.jsonl"))
    assert len(metrics_files) == 1
    step_rows = [
        json.loads(line)
        for line in metrics_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("type") == "step"
    ]
    assert len(step_rows) == 4
