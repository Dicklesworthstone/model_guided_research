"""Model-free data-geometry estimators with planted-geometry calibration.

Bead: model_guided_research-77l.1 (EPIC GEO, model_guided_research-77l).

WHAT THIS MODULE IS
    The measurement instrument behind ``mgr profile-data``: estimators of the
    geometric properties the theory epics condition on, computed on a corpus
    BEFORE any training:

      1. DELTA-HYPERBOLICITY  — Gromov four-point condition on sampled
         quadruples, normalized by quadruple diameter (tree-likeness; low
         values predict the {ultrametric, fractal, hyperbolic} family).
      2. ULTRAMETRICITY       — fraction/magnitude of strong-triangle-
         inequality violations (sorted triple distances d1<=d2<=d3 are
         ultrametric iff d2 == d3) plus cophenetic correlation against the
         single-linkage dendrogram (the subdominant ultrametric).
      3. DYNAMIC RANGE        — decades spanned by numeric literals within
         documents + a Hill tail-exponent estimate (predicts surreal).
      4. ORDER SENSITIVITY    — NLL delta of a bigram reference scorer under
         k adjacent-token transpositions (the data-side shadow of
         non-commutativity; predicts braid/integrable/gauge).
      5. HIERARCHY DEPTH      — single-linkage merge-tree depth statistics
         (explicit depth readout complementing 1-2; normalized ~1 = balanced
         hierarchy, >> 1 = flat data chaining — see the function docstring,
         the direction is easy to invert).

    Estimator cores operate on DISTANCE MATRICES or token sequences, so the
    calibration suite can feed them planted geometries directly; the corpus
    layer (load -> tokenize -> represent -> distances) is separable.

REPRESENTATION MODES (report both when possible; agreement = robustness):
    tokens      — embedding-free: normalized edit distance over token ids for
                  short sequences (<= EDIT_DISTANCE_MAX_LEN), Jaccard distance
                  over token n-gram sets for long ones (defaults per the bead
                  refinement).
    activations — cosine distance over mean-pooled hidden states of a FROZEN,
                  SEEDED, RANDOM-INIT tiny GPT (a deterministic random-feature
                  encoder; no trained checkpoint required). torch is imported
                  lazily so tokens mode stays light.

HONESTY CONTRACT (rendered into every profile):
    Absolute values are representation-relative. The registered claims
    (77l.2) are about ORDERINGS across corpora, never absolute thresholds.
    Tiny samples produce wide bootstrap CIs and a loud warning, never a
    silently confident number.

CALIBRATION (the hard gate, enforced by tests/test_geometry_profile.py):
    planted_tree_distances / planted_euclidean_distances /
    planted_hyperbolic_distances generate point sets with KNOWN geometry;
    every estimator must recover the planted ORDERING across its calibration
    ladder before any real-corpus number is believed.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROFILE_SCHEMA_VERSION = 1

# tokens-mode defaults (bead refinement: edit distance short, Jaccard long)
EDIT_DISTANCE_MAX_LEN = 64
JACCARD_NGRAM = 3

# sampling defaults (an estimator, not a census)
DEFAULT_SAMPLE_DOCS = 256
DEFAULT_POINTS = 96
DEFAULT_DOC_TOKENS = 256
DEFAULT_QUADRUPLES = 2000
DEFAULT_TRIPLES = 2000
DEFAULT_BOOTSTRAP = 200
SMALL_SAMPLE_WARN = 16


# ===========================================================================
# distance-matrix estimator cores (pure; calibration feeds these directly)
# ===========================================================================


def four_point_delta(D: np.ndarray, rng: np.random.Generator, n_quadruples: int = DEFAULT_QUADRUPLES) -> np.ndarray:
    """Per-quadruple normalized Gromov four-point delta.

    For points (w, x, y, z) form the three pairings S1 = d(w,x)+d(y,z),
    S2 = d(w,y)+d(x,z), S3 = d(w,z)+d(x,y); hyperbolicity of the quadruple is
    (largest - middle)/2, normalized here by the quadruple diameter so values
    are scale-free in [0, 1]. Trees give exactly 0.
    """
    n = D.shape[0]
    if n < 4:
        return np.zeros((0,), dtype=np.float64)
    deltas = np.empty((n_quadruples,), dtype=np.float64)
    for i in range(n_quadruples):
        w, x, y, z = rng.choice(n, size=4, replace=False)
        s = sorted((D[w, x] + D[y, z], D[w, y] + D[x, z], D[w, z] + D[x, y]))
        diam = max(D[w, x], D[w, y], D[w, z], D[x, y], D[x, z], D[y, z])
        deltas[i] = 0.0 if diam <= 0 else (s[2] - s[1]) / (2.0 * diam)
    return deltas


def ultrametricity_violations(
    D: np.ndarray, rng: np.random.Generator, n_triples: int = DEFAULT_TRIPLES, tol: float = 1e-9
) -> tuple[np.ndarray, float]:
    """Per-triple relative ultrametricity violation + violating fraction.

    Sorted triple distances d1 <= d2 <= d3 satisfy the strong triangle
    inequality iff d2 == d3; the relative violation is (d3 - d2)/d3 in [0, 1]
    (0 for ultrametric triples).
    """
    n = D.shape[0]
    if n < 3:
        return np.zeros((0,), dtype=np.float64), 0.0
    viol = np.empty((n_triples,), dtype=np.float64)
    for i in range(n_triples):
        x, y, z = rng.choice(n, size=3, replace=False)
        d = sorted((D[x, y], D[x, z], D[y, z]))
        viol[i] = 0.0 if d[2] <= 0 else (d[2] - d[1]) / d[2]
    frac = float(np.mean(viol > tol))
    return viol, frac


def single_linkage_merge_heights(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-linkage clustering via Kruskal/union-find.

    Returns (cophenetic, depths): the condensed cophenetic distance matrix
    entries (merge height joining each pair — the subdominant ultrametric)
    and the per-leaf dendrogram depth (number of merge events on the path
    from the leaf to the root).
    """
    n = D.shape[0]
    parent = list(range(n))
    # depth bookkeeping: per current cluster root, merges seen so far
    merge_count = [0] * n
    members: list[list[int]] = [[i] for i in range(n)]
    coph = np.zeros((n, n), dtype=np.float64)
    leaf_depth = np.zeros((n,), dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    iu, ju = np.triu_indices(n, k=1)
    order = np.argsort(D[iu, ju], kind="stable")
    merges_done = 0
    for e in order:
        a, b = find(int(iu[e])), find(int(ju[e]))
        if a == b:
            continue
        h = float(D[iu[e], ju[e]])
        for p in members[a]:
            for q in members[b]:
                coph[p, q] = coph[q, p] = h
        for p in members[a] + members[b]:
            leaf_depth[p] += 1
        parent[b] = a
        members[a].extend(members[b])
        members[b] = []
        merge_count[a] = merge_count[a] + merge_count[b] + 1
        merges_done += 1
        if merges_done == n - 1:
            break
    return coph[iu, ju], leaf_depth


def cophenetic_correlation(D: np.ndarray) -> float:
    """Pearson correlation between original distances and the single-linkage
    cophenetic (subdominant ultrametric) distances. ~1 for ultrametric data."""
    n = D.shape[0]
    if n < 3:
        return float("nan")
    iu, ju = np.triu_indices(n, k=1)
    orig = D[iu, ju]
    coph, _ = single_linkage_merge_heights(D)
    if np.std(orig) <= 0 or np.std(coph) <= 0:
        return float("nan")
    return float(np.corrcoef(orig, coph)[0, 1])


def hierarchy_depth_spectrum(D: np.ndarray) -> dict[str, float]:
    """Depth statistics of the single-linkage dendrogram, normalized by the
    balanced-tree depth log2(n).

    DIRECTION (verified on planted geometries; the easy mistake is inverting
    it): balanced nested structure merges in ~log2(n) levels, so
    normalized_mean_depth ~ 1; FLAT/uniform geometries CHAIN under single
    linkage, producing much DEEPER merge paths (planted Euclidean measures
    ~3.3). Values near 1 mean clean hierarchy; large values mean flat/chained.
    """
    n = D.shape[0]
    if n < 3:
        return {"mean_depth": float("nan"), "max_depth": float("nan"), "normalized_mean_depth": float("nan")}
    _, depths = single_linkage_merge_heights(D)
    base = math.log2(n) if n > 1 else 1.0
    return {
        "mean_depth": float(np.mean(depths)),
        "max_depth": float(np.max(depths)),
        "normalized_mean_depth": float(np.mean(depths) / base),
    }


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = DEFAULT_BOOTSTRAP) -> tuple[float, float]:
    """Percentile bootstrap CI (2.5/97.5) for the mean of `values`."""
    if values.size == 0:
        return (float("nan"), float("nan"))
    means = np.empty((n_boot,), dtype=np.float64)
    for b in range(n_boot):
        means[b] = float(np.mean(values[rng.integers(0, values.size, size=values.size)]))
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# ===========================================================================
# sequence-level estimators
# ===========================================================================

_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w.])")


def dynamic_range_stats(texts: list[str]) -> dict[str, Any]:
    """Decades spanned by numeric literals per document + Hill tail exponent.

    Returns NaNs (with numbers_found=0..1) rather than fabricating structure
    when the corpus carries no numeric content.
    """
    decades: list[float] = []
    magnitudes: list[float] = []
    for text in texts:
        vals = [abs(float(m)) for m in _NUMBER_RE.findall(text)]
        vals = [v for v in vals if v > 0 and math.isfinite(v)]
        magnitudes.extend(vals)
        if len(vals) >= 2:
            decades.append(math.log10(max(vals)) - math.log10(min(vals)))
    hill = float("nan")
    if len(magnitudes) >= 20:
        # Hill estimator over the top-10% order statistics; if the tail is
        # degenerate (all equal magnitudes -> mean log ratio 0) report NaN
        # rather than letting 1/0 produce inf in the profile
        mags = np.sort(np.asarray(magnitudes, dtype=np.float64))
        k = max(5, len(mags) // 10)
        tail = mags[-k:]
        if tail[0] > 0:
            mean_log = float(np.mean(np.log(tail / tail[0])))
            hill = 1.0 / mean_log if mean_log > 0 else float("nan")
    return {
        "numbers_found": len(magnitudes),
        "docs_with_spread": len(decades),
        "mean_decades": float(np.mean(decades)) if decades else float("nan"),
        "max_decades": float(np.max(decades)) if decades else float("nan"),
        "hill_tail_exponent": hill,
    }


def _bigram_nll(seqs: list[list[int]], probe: list[list[int]]) -> float:
    """Mean per-token NLL of `probe` under an add-one-smoothed bigram model
    fit on `seqs` (the cheap, embedding-free reference scorer)."""
    from collections import Counter, defaultdict

    vocab: set[int] = set()
    bigram: dict[int, Counter[int]] = defaultdict(Counter)
    unigram: Counter[int] = Counter()
    for s in seqs:
        vocab.update(s)
        unigram.update(s)
        for a, b in zip(s[:-1], s[1:], strict=False):
            bigram[a][b] += 1
    V = max(len(vocab), 1)
    total_nll, total_tok = 0.0, 0
    for s in probe:
        for a, b in zip(s[:-1], s[1:], strict=False):
            num = bigram[a][b] + 1.0
            den = unigram[a] + V
            total_nll += -math.log(num / den)
            total_tok += 1
    return total_nll / max(total_tok, 1)


def order_sensitivity(seqs: list[list[int]], rng: np.random.Generator, k_swaps: int = 8) -> dict[str, float]:
    """Relative NLL increase under k adjacent-token transpositions per
    sequence, scored by a bigram model fit on the originals. High values mean
    local order carries information (non-commutative dependencies)."""
    if not seqs:
        return {"nll_original": float("nan"), "nll_transposed": float("nan"), "relative_delta": float("nan")}
    transposed: list[list[int]] = []
    for s in seqs:
        t = list(s)
        for _ in range(min(k_swaps, max(len(t) - 1, 0))):
            i = int(rng.integers(0, max(len(t) - 1, 1)))
            t[i], t[i + 1] = t[i + 1], t[i]
        transposed.append(t)
    nll_o = _bigram_nll(seqs, seqs)
    nll_t = _bigram_nll(seqs, transposed)
    rel = (nll_t - nll_o) / nll_o if nll_o > 0 else float("nan")
    return {"nll_original": nll_o, "nll_transposed": nll_t, "relative_delta": rel}


# ===========================================================================
# tokens-mode distances (embedding-free, per bead refinement)
# ===========================================================================


def normalized_edit_distance(a: list[int], b: list[int]) -> float:
    """Levenshtein distance / max length, in [0, 1]."""
    if not a and not b:
        return 0.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        prev = cur
    return prev[lb] / max(la, lb)


def jaccard_ngram_distance(a: list[int], b: list[int], n: int = JACCARD_NGRAM) -> float:
    """1 - Jaccard similarity over token n-gram sets, in [0, 1]."""
    sa = {tuple(a[i : i + n]) for i in range(max(len(a) - n + 1, 0))}
    sb = {tuple(b[i : i + n]) for i in range(max(len(b) - n + 1, 0))}
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return 1.0 - inter / union if union else 0.0


def token_distance_matrix(seqs: list[list[int]]) -> np.ndarray:
    """Pairwise tokens-mode distances: edit distance when every sequence is
    short (<= EDIT_DISTANCE_MAX_LEN), Jaccard n-gram distance otherwise."""
    n = len(seqs)
    use_edit = all(len(s) <= EDIT_DISTANCE_MAX_LEN for s in seqs)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = normalized_edit_distance(seqs[i], seqs[j]) if use_edit else jaccard_ngram_distance(seqs[i], seqs[j])
            D[i, j] = D[j, i] = d
    return D


def activation_distance_matrix(seqs: list[list[int]], seed: int) -> np.ndarray:
    """Cosine distances over mean-pooled hidden states of a FROZEN, SEEDED,
    RANDOM-INIT tiny GPT (deterministic random-feature encoder). Lazy torch
    import keeps tokens mode dependency-light."""
    import torch

    from nanochat.gpt import GPT, GPTConfig

    torch.manual_seed(seed)
    cfg = GPTConfig(
        sequence_len=max(max((len(s) for s in seqs), default=8), 8),
        vocab_size=50304,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
    )
    with torch.no_grad():
        model = GPT(cfg).eval()
        feats = []
        for s in seqs:
            ids = torch.tensor([t % cfg.vocab_size for t in s], dtype=torch.long).unsqueeze(0)
            # feature = mean-pooled LOGITS of the seeded random-init model:
            # a deterministic random-feature encoding that needs no hooks and
            # no trained checkpoint (random projections preserve geometry
            # well enough for ORDERING comparisons, which is all we claim)
            logits = model(ids)
            feats.append(logits.float().mean(dim=1).squeeze(0))
        X = torch.stack(feats)
        Xn = torch.nn.functional.normalize(X, p=2, dim=-1)
        sim = Xn @ Xn.T
        D = (1.0 - sim).clamp_min_(0)
    return D.cpu().numpy().astype(np.float64)


# ===========================================================================
# planted-geometry generators (the calibration ladder)
# ===========================================================================


def planted_tree_distances(branching: int = 3, depth: int = 5) -> np.ndarray:
    """Leaf-to-leaf ULTRAMETRIC distances on a balanced tree: d(x, y) =
    depth - LCP(x, y) (normalized). Exactly ultrametric; four-point delta 0."""
    leaves = branching**depth
    digits = np.zeros((leaves, depth), dtype=np.int64)
    for i in range(leaves):
        v = i
        for d in range(depth - 1, -1, -1):
            digits[i, d] = v % branching
            v //= branching
    D = np.zeros((leaves, leaves), dtype=np.float64)
    for i in range(leaves):
        for j in range(i + 1, leaves):
            lcp = 0
            while lcp < depth and digits[i, lcp] == digits[j, lcp]:
                lcp += 1
            D[i, j] = D[j, i] = (depth - lcp) / depth
    return D


def planted_euclidean_distances(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Pairwise distances of uniform random points in [0,1]^dim (flat geometry:
    high delta, high ultrametricity violation)."""
    X = rng.uniform(size=(n, dim))
    diff = X[:, None, :] - X[None, :, :]
    return np.asarray(np.sqrt((diff**2).sum(-1)), dtype=np.float64)


def planted_hyperbolic_distances(n: int, rng: np.random.Generator, radius: float = 0.95) -> np.ndarray:
    """Pairwise hyperbolic distances of points sampled in the Poincare disk
    (area-uniform up to `radius`): d(u,v) = arccosh(1 + 2|u-v|^2 /
    ((1-|u|^2)(1-|v|^2))). Negatively curved: deltas between tree and flat."""
    r = np.sqrt(rng.uniform(size=n)) * radius
    th = rng.uniform(0, 2 * np.pi, size=n)
    P = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    sq = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    nu = 1.0 - (P**2).sum(-1)
    denom = nu[:, None] * nu[None, :]
    arg = 1.0 + 2.0 * sq / np.maximum(denom, 1e-12)
    D = np.asarray(np.arccosh(np.maximum(arg, 1.0)), dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    return D


# ===========================================================================
# corpus layer + profile assembly
# ===========================================================================

_WORD_RE = re.compile(r"\w+|[^\w\s]")


def simple_tokenize(text: str) -> list[int]:
    """Deterministic, model-free tokenization: word/punct split, then stable
    hashing to ids. Documented choice for tokens mode (no tokenizer dep)."""
    return [hash_token(w) for w in _WORD_RE.findall(text)]


def hash_token(w: str) -> int:
    # stable across processes (unlike builtin hash with PYTHONHASHSEED)
    h = 2166136261
    for ch in w.encode("utf-8"):
        h = (h ^ ch) * 16777619 & 0xFFFFFFFF
    return h


def load_corpus_texts(data_path: Path, max_docs: int, rng: np.random.Generator) -> list[str]:
    """Load documents from a directory (or single file) of .txt/.md/.parquet.

    Parquet files are read via pyarrow expecting a `text` column (the FineWeb
    convention used by nanochat/dataset.py).
    """
    texts: list[str] = []
    if data_path.is_file():
        files = [data_path]
    else:
        files = sorted([*data_path.glob("*.txt"), *data_path.glob("*.md"), *data_path.glob("*.parquet")])
    if not files:
        raise FileNotFoundError(f"no .txt/.md/.parquet documents under {data_path}")
    for f in files:
        if f.suffix == ".parquet":
            import pyarrow.parquet as pq

            try:
                table = pq.read_table(f, columns=["text"])
            except Exception as exc:  # pyarrow raises several types here
                raise ValueError(
                    f"could not read parquet {f.name} (expected a 'text' column, the FineWeb convention): {exc}"
                ) from exc
            texts.extend(str(x) for x in table.column("text").to_pylist())
        else:
            # split markdown/plaintext into paragraph-ish documents
            raw = f.read_text(encoding="utf-8", errors="replace")
            texts.extend(p for p in (s.strip() for s in raw.split("\n\n")) if len(p) > 40)
        if len(texts) >= max_docs * 4:
            break
    if not texts:
        raise ValueError(f"documents found under {data_path} but none usable (empty/too short)")
    if len(texts) > max_docs:
        idx = rng.choice(len(texts), size=max_docs, replace=False)
        texts = [texts[int(i)] for i in idx]
    return texts


@dataclass
class ProfileConfig:
    mode: str = "tokens"  # tokens | activations
    sample_docs: int = DEFAULT_SAMPLE_DOCS
    n_points: int = DEFAULT_POINTS
    doc_tokens: int = DEFAULT_DOC_TOKENS
    n_quadruples: int = DEFAULT_QUADRUPLES
    n_triples: int = DEFAULT_TRIPLES
    n_boot: int = DEFAULT_BOOTSTRAP
    seed: int = 42
    corpus_label: str = ""
    warnings: list[str] = field(default_factory=list)


def profile_from_texts(texts: list[str], cfg: ProfileConfig) -> dict[str, Any]:
    """Assemble the full profile from raw document texts (pure; no I/O)."""
    rng = np.random.default_rng(cfg.seed)
    warnings = list(cfg.warnings)
    if len(texts) < SMALL_SAMPLE_WARN:
        warnings.append(
            f"tiny corpus ({len(texts)} docs): bootstrap CIs will be wide; treat every value as indicative only"
        )
    seqs_full = [simple_tokenize(t)[: cfg.doc_tokens] for t in texts]
    seqs_full = [s for s in seqs_full if len(s) >= 2]
    if not seqs_full:
        raise ValueError("no tokenizable documents in corpus sample")
    pts = min(cfg.n_points, len(seqs_full))
    pick = rng.choice(len(seqs_full), size=pts, replace=False)
    seqs = [seqs_full[int(i)] for i in pick]

    if cfg.mode == "tokens":
        D = token_distance_matrix(seqs)
    elif cfg.mode == "activations":
        D = activation_distance_matrix(seqs, seed=cfg.seed)
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}; expected tokens|activations")

    # DISTANCE-CONCENTRATION DIAGNOSTIC: a near-uniform metric (all pairwise
    # distances ~equal, e.g. Jaccard saturation on near-disjoint documents) is
    # DEGENERATELY tree-like (a star tree): delta ~ 0 and ultrametricity ~ 0
    # then mean "the representation cannot discriminate", not "the data is
    # hierarchical". Surface this loudly instead of letting the degenerate
    # geometry masquerade as structure.
    iu, ju = np.triu_indices(D.shape[0], k=1)
    pair_d = D[iu, ju]
    mean_d = float(np.mean(pair_d)) if pair_d.size else float("nan")
    cv = float(np.std(pair_d) / mean_d) if pair_d.size and mean_d > 0 else float("nan")
    concentrated = bool(np.isfinite(cv) and cv < 0.05)
    if concentrated:
        warnings.append(
            f"distance concentration: coefficient of variation {cv:.4f} < 0.05 - pairwise distances are "
            "nearly uniform (star-tree degeneracy), so low delta-hyperbolicity/ultrametricity values here "
            "mean 'representation does not discriminate', NOT 'data is hierarchical'; try the other mode "
            "or shorter documents"
        )

    deltas = four_point_delta(D, rng, cfg.n_quadruples)
    d_lo, d_hi = bootstrap_ci(deltas, rng, cfg.n_boot)
    viol, viol_frac = ultrametricity_violations(D, rng, cfg.n_triples)
    v_lo, v_hi = bootstrap_ci(viol, rng, cfg.n_boot)
    coph = cophenetic_correlation(D)
    depth = hierarchy_depth_spectrum(D)
    dyn = dynamic_range_stats(texts)
    order = order_sensitivity(seqs_full, rng)

    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "corpus": cfg.corpus_label,
        "mode": cfg.mode,
        "seed": cfg.seed,
        "sample": {
            "docs": len(texts),
            "points": pts,
            "doc_tokens": cfg.doc_tokens,
            "quadruples": int(deltas.size),
            "triples": int(viol.size),
        },
        "distance_diagnostics": {"mean": mean_d, "coefficient_of_variation": cv, "concentrated": concentrated},
        "estimators": {
            "delta_hyperbolicity": {
                "mean": float(np.mean(deltas)) if deltas.size else float("nan"),
                "ci95": [d_lo, d_hi],
            },
            "ultrametricity": {
                "violation_mean": float(np.mean(viol)) if viol.size else float("nan"),
                "violation_fraction": viol_frac,
                "ci95": [v_lo, v_hi],
                "cophenetic_correlation": coph,
            },
            "dynamic_range": dyn,
            "order_sensitivity": order,
            "hierarchy_depth": depth,
        },
        "interpretation_note": (
            "values are representation-relative; registered claims (77l.2) are about "
            "ORDERINGS across corpora profiled with identical settings, never absolute thresholds"
        ),
        "warnings": warnings,
    }


def validate_profile_schema(profile: dict[str, Any]) -> list[str]:
    """Structural validation of a profile dict; returns error strings."""
    errors: list[str] = []
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PROFILE_SCHEMA_VERSION}")
    if profile.get("mode") not in ("tokens", "activations"):
        errors.append("mode must be tokens|activations")
    est = profile.get("estimators")
    if not isinstance(est, dict):
        errors.append("estimators must be a mapping")
        return errors
    for key in ("delta_hyperbolicity", "ultrametricity", "dynamic_range", "order_sensitivity", "hierarchy_depth"):
        if key not in est:
            errors.append(f"estimators missing {key}")
    if not isinstance(profile.get("warnings"), list):
        errors.append("warnings must be a list")
    return errors


def profile_to_json(profile: dict[str, Any]) -> str:
    return json.dumps(profile, indent=2, allow_nan=True)
