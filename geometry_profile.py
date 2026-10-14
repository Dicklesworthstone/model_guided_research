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


# ===========================================================================
# Geometry-informed initialization (--init-from-profile, bead 77l.4)
# ===========================================================================

GEOMETRY_INIT_MAPPING_TABLE: tuple[dict[str, str], ...] = (
    {
        "mechanism": "fractal",
        "parameter": "ifs_raw_radii",
        "source_metric": "estimators.ultrametricity.violation_mean + estimators.hierarchy_depth.normalized_mean_depth",
        "input_source": "profile-only",
        "formula_ref": "8gk.5 / thm-moran-capacity: s* = clip(0.85 - 0.45*v, 0.35, 0.85), p_m ~ (m+1)^(-zeta), r_m = min(p_m^(1/s*), r_max)",
    },
    {
        "mechanism": "hyperbolic",
        "parameter": "raw_curvature",
        "source_metric": "estimators.delta_hyperbolicity.mean",
        "input_source": "profile-only",
        "formula_ref": "8gk.6 / thm-hyperbolic-score-decomposition: c_init = clip((0.010 / max(delta, 0.005))^2, 0.10, 2.00)",
    },
    {
        "mechanism": "ultrametric",
        "parameter": "ultrametric_K",
        "source_metric": "estimators.hierarchy_depth.mean_depth",
        "input_source": "profile-only",
        "formula_ref": "77l.1 / thm-lcp-is-padic-valuation: K_init = clip(round(mean_depth), 2, 8)",
    },
    {
        "mechanism": "ultrametric",
        "parameter": "ultrametric_alpha",
        "source_metric": "estimators.ultrametricity.cophenetic_correlation",
        "input_source": "profile-only",
        "formula_ref": "77l.1 / thm-lcp-is-padic-valuation: alpha_init = clip(1.5 + 1.5*max(r_coph, 0.0), 1.25, 3.00)",
    },
    {
        "mechanism": "ultrametric",
        "parameter": "ultrametric_lcp_beta",
        "source_metric": "estimators.ultrametricity.violation_mean",
        "input_source": "profile-only",
        "formula_ref": "77l.1 / thm-lcp-is-padic-valuation: lcp_beta_init = clip(16.0 + 32.0*(1.0 - v), 16.0, 48.0)",
    },
    {
        "mechanism": "tropical",
        "parameter": "semiring_beta",
        "source_metric": "startup_probe.tropical_margin_median",
        "input_source": "startup-probe",
        "formula_ref": "8gk.1 / thm-tropical-margin-stability: beta_0 = clip((ln(head_dim) + ln(max(T_probe/2, 2))) / max(gamma_med, 0.10), 0.50, 16.0)",
    },
)

GEOMETRY_INIT_NOT_APPLICABLE: dict[str, str] = {
    "standard": "blocked-on-theory: dense dot-product attention has no geometric structural parameter",
    "reversible": "blocked-on-theory: symplectic leapfrog step size is governed by stability threshold, not corpus metric",
    "gauge": "blocked-on-theory: Givens transport angles start at identity frame (zero curvature prior)",
    "quaternion": "blocked-on-theory: Hamilton rotor projections are norm-preserving at any isotropic init",
    "octonion": "blocked-on-theory: Cayley-Dickson projections are norm-preserving at any isotropic init",
    "braid": "blocked-on-theory: Yang-Baxter deformation eta and spectral rapidities have no closed-form bigram-NLL mapping yet",
    "simplicial": "blocked-on-theory: 2-simplex window and wedge projections have no closed-form homology-to-init formula yet",
    "surreal": "blocked-on-theory: Levi-Civita valuation scale weights start at zero (exact standard reduction)",
}


def validate_profile_for_init(profile: dict[str, Any]) -> list[str]:
    """Validate both top-level schema and nested estimator metrics required by --init-from-profile."""
    errors = validate_profile_schema(profile)
    if errors:
        return errors
    est = profile["estimators"]

    delta_block = est.get("delta_hyperbolicity")
    if not isinstance(delta_block, dict) or "mean" not in delta_block:
        errors.append("estimators.delta_hyperbolicity missing required metric 'mean'")
    else:
        val = delta_block.get("mean")
        if not isinstance(val, int | float) or not math.isfinite(float(val)) or float(val) < 0.0:
            errors.append(f"estimators.delta_hyperbolicity.mean must be a finite float >= 0, got {val!r}")

    ultra_block = est.get("ultrametricity")
    if not isinstance(ultra_block, dict):
        errors.append("estimators.ultrametricity must be a mapping")
    else:
        for req_key in ("violation_mean", "cophenetic_correlation"):
            if req_key not in ultra_block:
                errors.append(f"estimators.ultrametricity missing required metric {req_key!r}")
        v_mean = ultra_block.get("violation_mean")
        if v_mean is not None and (
            not isinstance(v_mean, int | float) or not math.isfinite(float(v_mean)) or not (0.0 <= float(v_mean) <= 1.0)
        ):
            errors.append(f"estimators.ultrametricity.violation_mean must be in [0, 1], got {v_mean!r}")

    depth_block = est.get("hierarchy_depth")
    if not isinstance(depth_block, dict):
        errors.append("estimators.hierarchy_depth must be a mapping")
    else:
        for req_key in ("mean_depth", "normalized_mean_depth"):
            if req_key not in depth_block:
                errors.append(f"estimators.hierarchy_depth missing required metric {req_key!r}")
            else:
                d_val = depth_block.get(req_key)
                if not isinstance(d_val, int | float) or not math.isfinite(float(d_val)) or float(d_val) <= 0.0:
                    errors.append(f"estimators.hierarchy_depth.{req_key} must be a finite float > 0, got {d_val!r}")

    return errors


def load_profile_for_init(path: str | Path) -> dict[str, Any]:
    """Load and validate a profile.json file for --init-from-profile; raises ValueError with actionable detail."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"--init-from-profile file does not exist: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"--init-from-profile could not parse JSON from {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"--init-from-profile expected a JSON object in {p}, got {type(raw).__name__}")
    errs = validate_profile_for_init(raw)
    if errs:
        raise ValueError(f"--init-from-profile invalid profile {p}: " + "; ".join(errs))
    return raw


def compute_geometry_init_values(
    profile: dict[str, Any],
    *,
    num_prototypes: int = 4,
    max_radius: float = 0.85,
) -> dict[str, dict[str, Any]]:
    """Pure formula evaluator: map a validated profile dict to target parameter values for all supported mechanisms."""
    errs = validate_profile_for_init(profile)
    if errs:
        raise ValueError("Invalid profile for geometry init: " + "; ".join(errs))

    est = profile["estimators"]
    delta_mean = float(est["delta_hyperbolicity"]["mean"])
    v_mean = float(est["ultrametricity"]["violation_mean"])
    coph_raw = est["ultrametricity"]["cophenetic_correlation"]
    coph = float(coph_raw) if isinstance(coph_raw, int | float) and math.isfinite(float(coph_raw)) else 0.5
    mean_depth = float(est["hierarchy_depth"]["mean_depth"])
    norm_depth = float(est["hierarchy_depth"]["normalized_mean_depth"])

    # 1. Fractal IFS per-branch contraction radii (8gk.5 / thm-moran-capacity)
    s_star = min(max(0.85 - 0.45 * v_mean, 0.35), 0.85)
    zeta = min(max((1.0 - 0.5 * v_mean) / max(norm_depth, 0.5), 0.15), 1.50)
    raw_weights = [float((m + 1) ** (-zeta)) for m in range(num_prototypes)]
    w_sum = sum(raw_weights)
    probs = [w / w_sum for w in raw_weights]
    radii = [min(p ** (1.0 / s_star), max_radius) for p in probs]

    # 2. Hyperbolic curvature c_init (8gk.6 / thm-hyperbolic-horosphere-limit)
    c_init = min(max((0.010 / max(delta_mean, 0.005)) ** 2, 0.10), 2.00)

    # 3. Ultrametric digit depth K, alpha, and lcp_beta (77l.1 / thm-lcp-is-padic-valuation)
    k_init = int(min(max(round(mean_depth), 2), 8))
    alpha_init = min(max(1.5 + 1.5 * max(coph, 0.0), 1.25), 3.00)
    lcp_beta_init = min(max(16.0 + 32.0 * (1.0 - v_mean), 16.0), 48.0)

    return {
        "fractal.ifs_radii": {
            "mechanism": "fractal",
            "default_value": [round(0.5 * max_radius, 6)] * num_prototypes,
            "profile_value": [round(r, 6) for r in radii],
            "target_moran_dim": round(s_star, 6),
            "branch_skew_zeta": round(zeta, 6),
            "source_metric": "estimators.ultrametricity.violation_mean + estimators.hierarchy_depth.normalized_mean_depth",
            "source_value": {"violation_mean": v_mean, "normalized_mean_depth": norm_depth},
            "input_source": "profile-only",
            "formula_ref": "8gk.5 / thm-moran-capacity: s* = clip(0.85 - 0.45*v, 0.35, 0.85), p_m ~ (m+1)^(-zeta), r_m = min(p_m^(1/s*), r_max)",
        },
        "hyperbolic.curvature": {
            "mechanism": "hyperbolic",
            "default_value": 1.0,
            "profile_value": round(c_init, 6),
            "source_metric": "estimators.delta_hyperbolicity.mean",
            "source_value": delta_mean,
            "input_source": "profile-only",
            "formula_ref": "8gk.6 / thm-hyperbolic-score-decomposition: c_init = clip((0.010 / max(delta, 0.005))^2, 0.10, 2.00)",
        },
        "ultrametric.ultrametric_K": {
            "mechanism": "ultrametric",
            "default_value": 8,
            "profile_value": k_init,
            "source_metric": "estimators.hierarchy_depth.mean_depth",
            "source_value": mean_depth,
            "input_source": "profile-only",
            "formula_ref": "77l.1 / thm-lcp-is-padic-valuation: K_init = clip(round(mean_depth), 2, 8)",
        },
        "ultrametric.ultrametric_alpha": {
            "mechanism": "ultrametric",
            "default_value": 2.0,
            "profile_value": round(alpha_init, 6),
            "source_metric": "estimators.ultrametricity.cophenetic_correlation",
            "source_value": coph,
            "input_source": "profile-only",
            "formula_ref": "77l.1 / thm-lcp-is-padic-valuation: alpha_init = clip(1.5 + 1.5*max(r_coph, 0.0), 1.25, 3.00)",
        },
        "ultrametric.ultrametric_lcp_beta": {
            "mechanism": "ultrametric",
            "default_value": 32.0,
            "profile_value": round(lcp_beta_init, 6),
            "source_metric": "estimators.ultrametricity.violation_mean",
            "source_value": v_mean,
            "input_source": "profile-only",
            "formula_ref": "77l.1 / thm-lcp-is-padic-valuation: lcp_beta_init = clip(16.0 + 32.0*(1.0 - v), 16.0, 48.0)",
        },
    }


def apply_geometry_init_to_config(
    config: Any,
    profile: dict[str, Any],
    *,
    attention_schedule: list[str],
    cli_overrides: dict[str, bool] | None = None,
) -> dict[str, dict[str, Any]]:
    """Apply pre-construction config parameters (ultrametric_K/alpha/lcp_beta, fractal_ifs) from profile."""
    overrides = cli_overrides or {}
    computed = compute_geometry_init_values(
        profile,
        num_prototypes=int(getattr(config, "num_prototypes", 4)),
        max_radius=float(getattr(config, "ifs_max_radius", 0.85)),
    )
    applied: dict[str, dict[str, Any]] = {}

    if "ultrametric" in attention_schedule:
        if not overrides.get("ultrametric_K", False):
            entry = dict(computed["ultrametric.ultrametric_K"])
            entry["default_value"] = int(getattr(config, "ultrametric_K", 8))
            k_val = int(entry["profile_value"])
            # Respect balltree int64 sort-key bound if balltree mode is active
            if str(getattr(config, "ultrametric_mode", "kernel")).lower() == "balltree":
                p = int(getattr(config, "ultrametric_p", 2))
                max_t = int(getattr(config, "sequence_len", 2048)) * 10
                while k_val > 2 and ((p + 1) ** k_val) * 2 * max_t * 2 >= 2**62:
                    k_val -= 1
                entry["profile_value"] = k_val
            config.ultrametric_K = k_val
            applied["ultrametric.ultrametric_K"] = entry
        if not overrides.get("ultrametric_alpha", False):
            entry = dict(computed["ultrametric.ultrametric_alpha"])
            entry["default_value"] = float(getattr(config, "ultrametric_alpha", 2.0))
            config.ultrametric_alpha = float(entry["profile_value"])
            applied["ultrametric.ultrametric_alpha"] = entry
        if not overrides.get("ultrametric_lcp_beta", False):
            entry = dict(computed["ultrametric.ultrametric_lcp_beta"])
            entry["default_value"] = float(getattr(config, "ultrametric_lcp_beta", 32.0))
            config.ultrametric_lcp_beta = float(entry["profile_value"])
            applied["ultrametric.ultrametric_lcp_beta"] = entry

    if "fractal" in attention_schedule:
        if not overrides.get("fractal_ifs", False):
            config.fractal_ifs = True
        applied["fractal.ifs_radii"] = dict(computed["fractal.ifs_radii"])

    if "hyperbolic" in attention_schedule:
        applied["hyperbolic.curvature"] = dict(computed["hyperbolic.curvature"])

    return applied


def apply_geometry_init_to_model(
    model: Any,
    config: Any,
    profile: dict[str, Any],
    *,
    attention_schedule: list[str],
    applied: dict[str, dict[str, Any]],
    seed: int = 42,
    cli_overrides: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Warm-start module Parameters (fractal ifs_raw_radii, hyperbolic raw_curvature) and run tropical beta_0 startup probe."""
    import time as _time

    import torch

    from nanochat.fractal_attention_torch import FractalCausalSelfAttention, optimal_ifs_radii
    from nanochat.hyperbolic_attention_torch import CURVATURE_MIN, HyperbolicCausalSelfAttention
    from nanochat.tropical_attention_torch import TropicalCausalSelfAttention, set_semiring_beta

    overrides = cli_overrides or {}
    est = profile["estimators"]
    v_mean = float(est["ultrametricity"]["violation_mean"])
    norm_depth = float(est["hierarchy_depth"]["normalized_mean_depth"])
    delta_mean = float(est["delta_hyperbolicity"]["mean"])

    # 1. Apply FractalCausalSelfAttention per-branch radii via optimal_ifs_radii (8gk.5)
    if "fractal" in attention_schedule:
        s_star = min(max(0.85 - 0.45 * v_mean, 0.35), 0.85)
        zeta = min(max((1.0 - 0.5 * v_mean) / max(norm_depth, 0.5), 0.15), 1.50)
        for mod in model.modules():
            if isinstance(mod, FractalCausalSelfAttention):
                p_ref = next(mod.parameters())
                mod._ensure_ifs_parameters(p_ref.device, p_ref.dtype)
                assert mod.ifs_raw_radii is not None
                m_branches = int(mod.m)
                r_max = float(mod.r_max)
                probs = torch.arange(1, m_branches + 1, dtype=torch.float32, device=mod.ifs_raw_radii.device).pow(
                    -zeta
                )
                probs = probs / probs.sum()
                target_r = optimal_ifs_radii(probs, target_moran_dim=s_star, r_max=r_max)
                ratio = (target_r / r_max).clamp(1e-4, 1.0 - 1e-4)
                raw_logits = torch.log(ratio / (1.0 - ratio))
                with torch.no_grad():
                    mod.ifs_raw_radii.copy_(raw_logits.view_as(mod.ifs_raw_radii))

    # 2. Apply HyperbolicCausalSelfAttention per-head curvature (8gk.6)
    if "hyperbolic" in attention_schedule:
        c_init = min(max((0.010 / max(delta_mean, 0.005)) ** 2, 0.10), 2.00)
        raw_c = math.log(math.expm1(max(c_init - CURVATURE_MIN, 1e-4)))
        for mod in model.modules():
            if isinstance(mod, HyperbolicCausalSelfAttention):
                with torch.no_grad():
                    mod.raw_curvature.fill_(raw_c)

    # 3. Startup probe for TropicalCausalSelfAttention beta_0 (8gk.1 + 77l.4 refinement)
    probe_meta: dict[str, Any] | None = None
    if "tropical" in attention_schedule and not overrides.get("semiring_beta", False):
        probe_seed = int(seed) + 7714
        t0 = _time.perf_counter()
        dev = next(model.parameters()).device
        t_probe = min(int(getattr(config, "sequence_len", 64)), 64)
        vocab = int(getattr(config, "vocab_size", 256))
        gen = torch.Generator(device="cpu").manual_seed(probe_seed)
        probe_idx = torch.randint(0, vocab, (2, t_probe), generator=gen, dtype=torch.long).to(dev)

        trop_mods = [m for m in model.modules() if isinstance(m, TropicalCausalSelfAttention)]
        prev_record = [m.tropical_record_margins for m in trop_mods]
        for m in trop_mods:
            m.tropical_record_margins = True

        # Preserve global RNG states so the probe is side-effect-free on training RNG
        with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
            with torch.no_grad():
                model(probe_idx)

        margins: list[float] = []
        for m, prev in zip(trop_mods, prev_record, strict=True):
            gm = m.tropical_gamma_head_mean.detach().float().cpu()
            finite = gm[torch.isfinite(gm)]
            if finite.numel() > 0:
                margins.extend(float(x) for x in finite.tolist())
            m.tropical_record_margins = prev
            if not prev:
                m.tropical_gamma_head_mean.fill_(float("nan"))
                m.tropical_gamma_head_min.fill_(float("nan"))
                m.tropical_gamma_min.fill_(float("nan"))

        probe_ms = (_time.perf_counter() - t0) * 1000.0
        margins_sorted = sorted(margins)
        gamma_med = margins_sorted[len(margins_sorted) // 2] if margins_sorted else 0.5
        head_dim = int(getattr(config, "n_embd", 64)) // max(int(getattr(config, "n_head", 4)), 1)
        m_vis = max(t_probe / 2.0, 2.0)
        beta_0 = min(max((math.log(max(head_dim, 1)) + math.log(m_vis)) / max(gamma_med, 0.10), 0.50), 16.0)
        beta_0_round = round(beta_0, 6)
        default_beta = getattr(config, "semiring_beta", None)
        config.semiring_beta = beta_0_round
        set_semiring_beta(model, beta_0_round)
        applied["tropical.semiring_beta"] = {
            "mechanism": "tropical",
            "default_value": default_beta,
            "profile_value": beta_0_round,
            "source_metric": "startup_probe.tropical_margin_median",
            "source_value": round(gamma_med, 6),
            "input_source": "startup-probe",
            "formula_ref": "8gk.1 / thm-tropical-margin-stability: beta_0 = clip((ln(head_dim) + ln(max(T_probe/2, 2))) / max(gamma_med, 0.10), 0.50, 16.0)",
        }
        probe_meta = {
            "probe_seed": probe_seed,
            "probe_batch_shape": [2, t_probe],
            "probe_margin_median": round(gamma_med, 6),
            "probe_ms": round(probe_ms, 3),
        }

    not_applicable = {
        mech: GEOMETRY_INIT_NOT_APPLICABLE[mech]
        for mech in sorted(set(attention_schedule))
        if mech in GEOMETRY_INIT_NOT_APPLICABLE
    }
    return {
        "enabled": True,
        "profile_corpus": profile.get("corpus", ""),
        "profile_mode": profile.get("mode", "tokens"),
        "profile_schema_version": profile.get("schema_version", PROFILE_SCHEMA_VERSION),
        "startup_probe": probe_meta,
        "applied_parameters": applied,
        "not_applicable_mechanisms": not_applicable,
    }


def _extract_structural_tensors(model: Any) -> dict[str, Any]:
    """Extract effective geometric structural parameter tensors from fractal, hyperbolic, and ultrametric modules."""
    import torch

    from nanochat.fractal_attention_torch import FractalCausalSelfAttention
    from nanochat.hyperbolic_attention_torch import HyperbolicCausalSelfAttention
    from nanochat.ultrametric_attention_torch import UltrametricCausalSelfAttention

    snap: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            prefix = f"{name}." if name else ""
            if isinstance(mod, FractalCausalSelfAttention) and mod.ifs_raw_radii is not None:
                snap[f"{prefix}ifs_radii"] = mod.branch_radii().detach().float().cpu().clone()
            elif isinstance(mod, HyperbolicCausalSelfAttention):
                snap[f"{prefix}curvature"] = mod._curv().detach().float().cpu().clone()
            elif isinstance(mod, UltrametricCausalSelfAttention):
                snap[f"{prefix}to_digits_q.weight"] = mod.to_digits_q.weight.detach().float().cpu().clone()
                snap[f"{prefix}to_digits_k.weight"] = mod.to_digits_k.weight.detach().float().cpu().clone()
    return snap


def snapshot_structural_params(model: Any) -> dict[str, Any]:
    """Snapshot initial structural parameters on fractal, hyperbolic, and ultrametric modules for distance-traveled tracking."""
    return _extract_structural_tensors(model)


def compute_structural_param_distance(model: Any, init_snapshot: dict[str, Any]) -> dict[str, Any] | None:
    """Compute L2 distance traveled by structural parameters from step 0 to final step (bead 77l.4 prediction b)."""
    import torch

    if not init_snapshot:
        return None
    current_map = _extract_structural_tensors(model)
    per_param: dict[str, float] = {}
    sq_sum = 0.0
    for name, init_t in init_snapshot.items():
        current = current_map.get(name)
        if current is None or current.shape != init_t.shape:
            continue
        diff = current - init_t
        dist_sq = float(torch.sum(diff * diff).item())
        sq_sum += dist_sq
        per_param[name] = math.sqrt(dist_sq)
    if not per_param:
        return None
    return {
        "total_l2": math.sqrt(sq_sum),
        "per_parameter_l2": per_param,
    }

