"""
Synthetic diagnostic task suite generator (bead model_guided_research-vdc.1).

WHY: to falsify the README's mechanism-advantage claims we need tasks DESIGNED
so each mechanism's theoretical edge should manifest as a measurable gap.
FineWeb-Edu LM loss cannot do this. Each task family below documents its
target mechanism(s) and the hypothesis its design elicits; that text seeds the
preregistered predictions (C3 / hij.1).

OUTPUT CONTRACT (consumed by nanochat/dataloader.py UNCHANGED):
    <out>/<task>/train_000.parquet      # text column, FineWeb schema
    <out>/<task>/val_000.parquet        # sorts LAST -> the dataloader's val split
    <out>/<task>/heldout/test_000.parquet   # held-out DIFFICULTY (longer/deeper);
                                        # NOT visible to the train-time dataloader,
                                        # read directly by the eval harness (vdc.2)
    <out>/<task>/manifest.json          # task, seed, dials, split sizes, sha256 per
                                        # parquet, generator version, provenance

Text format: every document is a single line of SPACE-SEPARATED symbols. With
the GPT-2 BPE this guarantees task delimiters tokenize as standalone tokens
(" (", " ]", " ;", " OUT" ... are single tokens or clean piece boundaries) -
the tokenizer-safety tests assert this per task, because BPE merging across
delimiters would silently destroy the structure the tasks are probing.

Determinism: byte-identical parquet from the same (seed, dials) pair - the
RNG is derived from a stable string hash per (task, split, seed), never from
global state. GENERATOR_VERSIONS must be bumped when a generator's output
changes; a fixture test enforces hash-change => version-change.

Splits: train and val draw from the IN-RANGE difficulty regime (disjoint RNG
streams); test draws from HELD-OUT difficulty (longer sequences, deeper trees,
wider magnitude spreads). Extrapolation is the point - in-distribution
accuracy alone is uninformative.

ROBUSTNESS PROBE SPEC (single source of truth for vdc.3 and 8gk.7): the
eps-bounded perturbation probe is `apply_embedding_perturbation`: a sup-norm
(L-infinity) bounded perturbation injected at the TOKEN-EMBEDDING OUTPUT
(after wte + the embedding norm, before the first block), drawn uniformly
from [-eps, +eps] per coordinate with a caller-supplied torch.Generator.
All consumers must measure robustness through this function so magnitudes
compare across studies. eps=0 is a byte-identical no-op by construction.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

# Bump a task's version whenever its generator output changes for a fixed
# (seed, dials): the manifest-hash fixture test enforces this.
GENERATOR_VERSIONS: dict[str, int] = {
    "dyck": 1,
    "copyops": 1,
    "hier": 1,
    "rel": 1,
    "rot": 1,
    "arith": 1,
    "regime": 1,
    "needle": 1,
    "group": 1,
    "bag": 1,
    "placebo": 1,
    "realhier": 1,
}

PARQUET_ROW_GROUP_SIZE = 256  # small row groups so DDP iteration works on tiny datasets


# ---------------------------------------------------------------------------
# infrastructure


@dataclass(frozen=True)
class Dial:
    """A structural difficulty knob, monotone in its intended geometric property."""

    name: str
    default: float
    description: str
    lo: float
    hi: float


@dataclass
class TaskSpec:
    name: str
    target_mechanisms: tuple[str, ...]
    hypothesis: str
    dials: tuple[Dial, ...]
    generate: Callable[[int, int, dict[str, float]], dict[str, list[str]]]
    # checker(doc_text) -> True (label/answer in doc is correct) | False (it is
    # not) | None (LM-only task; no per-doc label). Brute force by design:
    # these are the ground truth for the eval harness metrics (vdc.2).
    checker: Callable[[str], bool | None]
    delimiters: tuple[str, ...] = field(default_factory=tuple)
    # Eval-harness knowledge (vdc.2): the token that separates prompt from
    # answer (None = LM-only, perplexity is the only metric), and the
    # difficulty axis for extrapolation curves (accuracy-vs-difficulty with
    # in-range vs held-out buckets marked).
    answer_marker: str | None = None
    difficulty_axis: str | None = None
    difficulty: Callable[[str], float] | None = None
    # Optional per-doc category extractor (bead u55.3): tasks that mix sub-
    # populations with different theoretical status (the group task mixes
    # non-solvable S5/A5 with solvable Z60/S3 controls) expose the label so the
    # eval harness can fit per-category extrapolation slopes - the preregistered
    # mechanism-specificity predictions adjudicate on exactly that breakdown.
    category: Callable[[str], str | None] | None = None

    def split_prompt(self, doc: str) -> tuple[str, str] | None:
        """(prompt-including-marker, expected answer) or None for LM-only docs."""
        if self.answer_marker is None:
            return None
        marker = f" {self.answer_marker} "
        pos = doc.rfind(marker)
        if pos < 0:
            return None
        return doc[: pos + len(marker) - 1], doc[pos + len(marker) :]

    def resolve_dials(self, overrides: dict[str, float] | None) -> dict[str, float]:
        values = {d.name: d.default for d in self.dials}
        for key, value in (overrides or {}).items():
            if key not in values:
                raise ValueError(f"task {self.name!r} has no dial {key!r} (dials: {sorted(values)})")
            dial = next(d for d in self.dials if d.name == key)
            if not (dial.lo <= float(value) <= dial.hi):
                raise ValueError(f"dial {key}={value} outside documented range [{dial.lo}, {dial.hi}]")
            values[key] = float(value)
        return values


def _rng(task: str, split: str, seed: int) -> random.Random:
    """Deterministic per-(task, split, seed) RNG, independent of global state."""
    digest = hashlib.sha256(f"{task}:{split}:{seed}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _split_sizes(size: int) -> dict[str, int]:
    val = max(1, size // 10)
    test = max(1, size // 10)
    return {"train": max(1, size - val - test), "val": val, "test": test}


# ---------------------------------------------------------------------------
# task 1: Dyck-k balanced brackets — braid / topology
# Hypothesis: braid attention's crossing structure tracks nesting topology, so
# validity classification should extrapolate to held-out nesting depths where
# pairwise-similarity attention degrades (precedent: the braid JAX demo).

_DYCK_PAIRS = [("(", ")"), ("[", "]"), ("{", "}")]
_DYCK_OPEN = {o: c for o, c in _DYCK_PAIRS}
_DYCK_CLOSE = {c: o for o, c in _DYCK_PAIRS}


def _dyck_sequence(rng: random.Random, depth_budget: int, length_budget: int) -> list[str]:
    out: list[str] = []
    stack: list[str] = []
    while length_budget > 0:
        can_open = len(stack) < depth_budget and length_budget >= len(stack) + 2
        if stack and (not can_open or rng.random() < 0.45):
            out.append(_DYCK_OPEN[stack.pop()])
        elif can_open:
            o, _c = _DYCK_PAIRS[rng.randrange(len(_DYCK_PAIRS))]
            out.append(o)
            stack.append(o)
        else:
            break
        length_budget -= 1
    while stack:
        out.append(_DYCK_OPEN[stack.pop()])
    return out


def check_dyck(text: str) -> bool | None:
    parts = text.split()
    if parts[:2] != ["TASK", "dyck"] or "LABEL" not in parts:
        return None
    li = parts.index("LABEL")
    seq, label = parts[3:li], parts[li + 1]
    stack: list[str] = []
    valid = True
    for tok in seq:
        if tok in _DYCK_OPEN:
            stack.append(tok)
        elif tok in _DYCK_CLOSE:
            if not stack or stack.pop() != _DYCK_CLOSE[tok]:
                valid = False
                break
        else:
            valid = False
            break
    valid = valid and not stack
    return label == ("valid" if valid else "invalid")


def _gen_dyck(size: int, seed: int, dials: dict[str, float]) -> dict[str, list[str]]:
    max_depth = int(dials["max_depth"])
    splits: dict[str, list[str]] = {}
    for split, n in _split_sizes(size).items():
        rng = _rng("dyck", split, seed)
        depth = max_depth * 2 if split == "test" else max_depth  # held-out DEPTH
        docs = []
        for _ in range(n):
            seq = _dyck_sequence(rng, depth_budget=depth, length_budget=rng.randint(8, 8 + 4 * depth))
            label = "valid"
            if rng.random() < 0.5:  # corrupt into a genuinely invalid sequence
                seq = list(seq)
                i = rng.randrange(len(seq))
                choices = [b for pair in _DYCK_PAIRS for b in pair if b != seq[i]]
                seq[i] = rng.choice(choices)
                # corruption may accidentally stay valid; recheck via the checker
                probe = f"TASK dyck SEQ {' '.join(seq)} LABEL valid"
                label = "valid" if check_dyck(probe) else "invalid"
            docs.append(f"TASK dyck SEQ {' '.join(seq)} LABEL {label}")
        splits[split] = docs
    return splits


# ---------------------------------------------------------------------------
# task 2: copy / reverse / rotate — braid (permutations), reversible (information preservation)
# Hypothesis: mechanisms with explicit permutation/inversion structure
# extrapolate these length-indexed bijections to held-out lengths.

_LETTERS = list("abcdefghijklmnopqrstuvwxyz")


def check_copyops(text: str) -> bool | None:
    parts = text.split()
    if parts[:2] != ["TASK", "copyops"] or "OUT" not in parts:
        return None
    oi = parts.index("OUT")
    op = parts[3]
    seq = parts[parts.index("SEQ") + 1 : oi]
    got = parts[oi + 1 :]
    if op == "COPY":
        want = seq
    elif op == "REV":
        want = seq[::-1]
    elif op.startswith("ROT"):
        k = int(op[3:]) % max(1, len(seq))
        want = seq[k:] + seq[:k]
    else:
        return None
    return got == want


def _gen_copyops(size: int, seed: int, dials: dict[str, float]) -> dict[str, list[str]]:
    base_len = int(dials["length"])
    splits: dict[str, list[str]] = {}
    for split, n in _split_sizes(size).items():
        rng = _rng("copyops", split, seed)
        lo, hi = (base_len + 1, 2 * base_len) if split == "test" else (3, base_len)  # held-out LENGTH
        docs = []
        for _ in range(n):
            length = rng.randint(lo, hi)
            seq = [rng.choice(_LETTERS) for _ in range(length)]
            op = rng.choice(["COPY", "REV", f"ROT{rng.randint(1, max(1, length - 1))}"])
            if op == "COPY":
                out = seq
            elif op == "REV":
                out = seq[::-1]
            else:
                k = int(op[3:]) % length
                out = seq[k:] + seq[:k]
            docs.append(f"TASK copyops OP {op} SEQ {' '.join(seq)} OUT {' '.join(out)}")
        splits[split] = docs
    return splits


# ---------------------------------------------------------------------------
# task 3: hierarchical retrieval — ultrametric / fractal (later hyperbolic)
# Hypothesis: LCP-style routing resolves nested key paths in sub-quadratic
# depth, so accuracy at HELD-OUT DEPTHS is the discriminating signal; the
# branch-imbalance dial (Zipf over branch mass) controls how tree-like vs
# list-like the geometry is (consumed by the GEO dose-response study, 77l.3).


def _hier_tree(rng: random.Random, depth: int, branching: int, zipf_alpha: float) -> dict:
    """Nested dict; leaves are value symbols. Branch sizes follow a Zipf mass."""

    def build(level: int) -> Any:
        if level >= depth:
            return f"v{rng.randrange(100)}"
        weights = [1.0 / (i + 1) ** zipf_alpha for i in range(branching)]
        total = sum(weights)
        children = {}
        for i in range(branching):
            # Zipf imbalance: low-rank branches get subtrees, high-rank may
            # flatten to leaves. The probability saturates at 1 for low-rank
            # branches at high alpha - that saturation IS the imbalance
            # mechanism (rank 0 always deep, the rest rarely). min() makes the
            # cap explicit; it does not change behavior (random() < p for
            # p >= 1 is always true), so generator hashes are unchanged.
            go_deep = rng.random() < min(1.0, (weights[i] / total) * branching * 0.8)
            children[f"k{level}x{i}"] = build(level + 1) if go_deep else f"v{rng.randrange(100)}"
        return children

    root = build(0)
    if not isinstance(root, dict):  # depth 0 degenerate guard
        root = {"k0x0": root}
    return root


def _hier_serialize(node: Any) -> str:
    if isinstance(node, str):
        return node
    inner = " ".join(f"( {key} {_hier_serialize(child)} )" for key, child in node.items())
    return inner


def _hier_paths(node: Any, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str]]:
    if isinstance(node, str):
        return [(prefix, node)]
    out = []
    for key, child in node.items():
        out.extend(_hier_paths(child, prefix + (key,)))
    return out


def check_hier(text: str) -> bool | None:
    parts = text.split()
    if parts[:2] != ["TASK", "hier"] or "OUT" not in parts:
        return None
    ti, pi, oi = parts.index("TREE"), parts.index("PATH"), parts.index("OUT")
    tree_toks, path, want = parts[ti + 1 : pi], parts[pi + 1 : oi], parts[oi + 1]

    def parse(toks: list[str], i: int) -> tuple[dict, int]:
        node: dict[str, Any] = {}
        while i < len(toks) and toks[i] == "(":
            key = toks[i + 1]
            if toks[i + 2] == "(":
                child, i2 = parse(toks, i + 2)
                node[key] = child
                i = i2 + 1  # closing paren of this entry
            else:
                node[key] = toks[i + 2]
                i = i + 4
        return node, i

    tree, _ = parse(tree_toks, 0)
    cur: Any = tree
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return False
        cur = cur[key]
    return bool(cur == want)


def _gen_hier(size: int, seed: int, dials: dict[str, float]) -> dict[str, list[str]]:
    depth = int(dials["depth"])
    branching = int(dials["branching"])
    zipf_alpha = float(dials["zipf_alpha"])
    splits: dict[str, list[str]] = {}
    for split, n in _split_sizes(size).items():
        rng = _rng("hier", split, seed)
        d = depth + 2 if split == "test" else depth  # held-out DEPTH
        docs = []
        for _ in range(n):
            tree = _hier_tree(rng, depth=d, branching=branching, zipf_alpha=zipf_alpha)
            paths = _hier_paths(tree)
            # prefer the deepest available path so the test split actually queries deep
            paths.sort(key=lambda pv: len(pv[0]))
            path, value = paths[-1] if rng.random() < 0.5 else rng.choice(paths)
            docs.append(f"TASK hier TREE {_hier_serialize(tree)} PATH {' '.join(path)} OUT {value}")
        splits[split] = docs
    return splits


# ---------------------------------------------------------------------------
# task 4: multi-entity relation queries — simplicial
# Hypothesis: 2-hop composition (who is connected to X via Y?) is exactly the
# triangle (2-simplex) aggregation simplicial attention adds over pairwise
# attention; the gap should grow with the number of distractor facts.


def check_rel(text: str) -> bool | None:
    parts = text.split()
    if parts[:2] != ["TASK", "rel"] or "OUT" not in parts:
        return None
    qi, oi = parts.index("QUERY"), parts.index("OUT")
    facts: dict[tuple[str, str], str] = {}
    i = 2
    while i < qi:
        if parts[i] == "FACT":
            a, r, b = parts[i + 1], parts[i + 2], parts[i + 3]
            facts[(a, r)] = b
            i += 4
        else:
            i += 1  # the ';' separators
    start, r1, r2 = parts[qi + 1], parts[qi + 2], parts[qi + 3]
    mid = facts.get((start, r1))
    end = facts.get((mid, r2)) if mid is not None else None
    return end == parts[oi + 1]


def _gen_rel(size: int, seed: int, dials: dict[str, float]) -> dict[str, list[str]]:
    n_facts = int(dials["n_facts"])
    relations = ["likes", "knows", "owns", "fears"]
    splits: dict[str, list[str]] = {}
    for split, n in _split_sizes(size).items():
        rng = _rng("rel", split, seed)
        nf = n_facts * 2 if split == "test" else n_facts  # held-out CONTEXT SIZE
        docs = []
        for _ in range(n):
            entities = [f"e{i}" for i in range(nf + 3)]
            rng.shuffle(entities)
            facts: dict[tuple[str, str], str] = {}
            # plant a guaranteed 2-hop chain a -r1-> b -r2-> c
            a, b, c = entities[0], entities[1], entities[2]
            r1, r2 = rng.choice(relations), rng.choice(relations)
            facts[(a, r1)] = b
            facts[(b, r2)] = c
            while len(facts) < nf:
                x, y = rng.sample(entities, 2)
                r = rng.choice(relations)
                facts.setdefault((x, r), y)
            items = list(facts.items())
            rng.shuffle(items)
            ftxt = " ; ".join(f"FACT {x} {r} {y}" for (x, r), y in items)
            docs.append(f"TASK rel {ftxt} QUERY {a} {r1} {r2} OUT {facts[(facts[(a, r1)], r2)]}")
        splits[split] = docs
    return splits


# ---------------------------------------------------------------------------
# task 5: 3D rotation composition — quaternion / octonion (later clifford)
# Hypothesis: rotor-structured value mixing composes orientation algebra
# natively, so predicting the composed cube orientation extrapolates to
# held-out sequence lengths. The 24-element octahedral group keeps the
# checker exact (integer matrices; no fp ambiguity).

_ROT_GENERATORS: dict[str, tuple[tuple[int, ...], ...]] = {
    "X90": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
    "Y90": ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
    "Z90": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
}


def _mat_mul(a: tuple[tuple[int, ...], ...], b: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )


def _octahedral_elements() -> list[tuple[tuple[int, ...], ...]]:
    identity: tuple[tuple[int, ...], ...] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    seen = {identity}
    frontier: list[tuple[tuple[int, ...], ...]] = [identity]
    while frontier:
        cur = frontier.pop()
        for g in _ROT_GENERATORS.values():
            nxt = _mat_mul(g, cur)
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return sorted(seen)


_OCTAHEDRAL = _octahedral_elements()
_OCTA_INDEX = {m: i for i, m in enumerate(_OCTAHEDRAL)}
assert len(_OCTAHEDRAL) == 24


def check_rot(text: str) -> bool | None:
    parts = text.split()
    if parts[:2] != ["TASK", "rot"] or "OUT" not in parts:
        return None
    oi = parts.index("OUT")
    seq = parts[parts.index("SEQ") + 1 : oi]
    cur: tuple[tuple[int, ...], ...] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    for tok in seq:
        axis, deg = tok[0], int(tok[1:])
        g = _ROT_GENERATORS[f"{axis}90"]
        for _ in range((deg // 90) % 4):
            cur = _mat_mul(g, cur)
    return parts[oi + 1] == f"o{_OCTA_INDEX[cur]}"


def _gen_rot(size: int, seed: int, dials: dict[str, float]) -> dict[str, list[str]]:
    base_len = int(dials["length"])
    splits: dict[str, list[str]] = {}
    for split, n in _split_sizes(size).items():
        rng = _rng("rot", split, seed)
        lo, hi = (base_len + 1, 2 * base_len) if split == "test" else (2, base_len)  # held-out LENGTH
        docs = []
        for _ in range(n):
            length = rng.randint(lo, hi)
            seq = [f"{rng.choice('XYZ')}{rng.choice([90, 180, 270])}" for _ in range(length)]
            cur: tuple[tuple[int, ...], ...] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
            for tok in seq:
                g = _ROT_GENERATORS[f"{tok[0]}90"]
                for _ in range((int(tok[1:]) // 90) % 4):
                    cur = _mat_mul(g, cur)
            docs.append(f"TASK rot SEQ {' '.join(seq)} OUT o{_OCTA_INDEX[cur]}")
        splits[split] = docs
    return splits


# ---------------------------------------------------------------------------
# task 6: wide-dynamic-range arithmetic — surreal
# Hypothesis: magnitude-direction separation (w = exp(s) * v) gives surreal
# layers native scale arithmetic, so comparisons spanning many decades should
# degrade more slowly than for unstructured projections as the spread grows.


def check_arith(text: str) -> bool | None:
    parts = text.split()
    if parts[:2] != ["TASK", "arith"] or "OUT" not in parts:
        return None
    oi = parts.index("OUT")
    x, y = float(parts[3]), float(parts[4])
    want = "lt" if x < y else "gt"
    return parts[oi + 1] == want


def _gen_arith(size: int, seed: int, dials: dict[str, float]) -> dict[str, list[str]]:
    spread = int(dials["spread_decades"])
    splits: dict[str, list[str]] = {}
    for split, n in _split_sizes(size).items():
        rng = _rng("arith", split, seed)
        decades = spread * 2 if split == "test" else spread  # held-out SPREAD
        docs = []
        for _ in range(n):
            e1 = rng.randint(-decades, decades)
            e2 = rng.randint(-decades, decades)
            while e2 == e1:  # distinct exponents: comparisons never hinge on fp ties
                e2 = rng.randint(-decades, decades)
            m1 = rng.randint(100, 999) / 100.0
            m2 = rng.randint(100, 999) / 100.0
            x, y = f"{m1:.2f}e{e1:+03d}", f"{m2:.2f}e{e2:+03d}"
            want = "lt" if float(x) < float(y) else "gt"
            docs.append(f"TASK arith CMP {x} {y} OUT {want}")
        splits[split] = docs
    return splits


# ---------------------------------------------------------------------------
# task 7: regime-shift streams — ordinal scheduler / HOSS
# Hypothesis: distribution flips at known boundaries reward principled
# restart/anneal behavior (ordinal) and curvature-aware steps (HOSS); LM loss
# recovery time after each SHIFT marker is the measured signal (consumed by D4).


def check_regime(text: str) -> bool | None:
    parts = text.split()
    if parts[:2] != ["TASK", "regime"]:
        return None
    # LM-style stream: validate every segment continues its arithmetic rule.
    i = 2
    ok = True
    while i < len(parts):
        if parts[i] != "SEG":
            return False
        m, a = int(parts[i + 1][1:]), int(parts[i + 2][1:])
        j = i + 3
        expected = a
        while j < len(parts) and parts[j] not in {"SEG"}:
            if parts[j] != f"n{expected}":
                ok = False
            expected = (expected + m) % 97
            j += 1
        i = j
    return ok


def _gen_regime(size: int, seed: int, dials: dict[str, float]) -> dict[str, list[str]]:
    seg_len = int(dials["segment_len"])
    n_regimes = int(dials["n_regimes"])
    splits: dict[str, list[str]] = {}
    for split, n in _split_sizes(size).items():
        rng = _rng("regime", split, seed)
        regimes = n_regimes * 2 if split == "test" else n_regimes  # held-out SHIFT COUNT
        docs = []
        for _ in range(n):
            chunks = []
            for _ in range(regimes):
                m, a = rng.randint(2, 9), rng.randint(0, 96)
                vals = [(a + k * m) % 97 for k in range(seg_len)]
                chunks.append(f"SEG m{m} a{a} " + " ".join(f"n{v}" for v in vals))
            docs.append("TASK regime " + " ".join(chunks))
        splits[split] = docs
    return splits


# ---------------------------------------------------------------------------
# task 8: needle-in-haystack — ultrametric (sub-quadratic + hierarchy), long context
# Hypothesis: hierarchical addressing retrieves planted pairs at context
# lengths where dense attention's effective resolution dilutes; the dial pair
# (context length, distractor density) separates capacity from interference.


def check_needle(text: str) -> bool | None:
    parts = text.split()
    if parts[:2] != ["TASK", "needle"] or "OUT" not in parts:
        return None
    qi, oi = parts.index("QUERY"), parts.index("OUT")
    key = parts[qi + 1]
    pairs: dict[str, str] = {}
    i = 2
    while i < qi:
        if parts[i] == "PAIR":
            pairs[parts[i + 1]] = parts[i + 2]
            i += 3
        else:
            i += 1
    return pairs.get(key) == parts[oi + 1]


def _gen_needle(size: int, seed: int, dials: dict[str, float]) -> dict[str, list[str]]:
    ctx_words = int(dials["context_words"])
    density = float(dials["distractor_density"])
    splits: dict[str, list[str]] = {}
    for split, n in _split_sizes(size).items():
        rng = _rng("needle", split, seed)
        words = ctx_words * 4 if split == "test" else ctx_words  # held-out LENGTH
        docs = []
        for _ in range(n):
            n_pairs = max(1, int(words * density / 3))
            keys = rng.sample(range(10 * n_pairs), n_pairs)
            pairs = [(f"k{k}", f"v{rng.randrange(100)}") for k in keys]
            filler_count = max(0, words - 3 * n_pairs)
            tokens: list[str] = []
            for key, val in pairs:
                tokens.append(f"PAIR {key} {val}")
            tokens.extend(rng.choice(_LETTERS) for _ in range(filler_count))
            rng.shuffle(tokens)
            qk, qv = pairs[rng.randrange(len(pairs))]
            docs.append(f"TASK needle {' '.join(tokens)} QUERY {qk} OUT {qv}")
        splits[split] = docs
    return splits


# ---------------------------------------------------------------------------
# task 9 (theory program): group word problems — integrable attention (u55.3)
# Hypothesis: composing words in NON-SOLVABLE groups (S5, A5) is the
# NC1-complete state-tracking regime where fixed-depth transformers hit the
# known TC0 barrier; integrable/R-matrix structure is predicted to leapfrog.
# The SOLVABLE controls (Z60, S3) are the mechanism-specificity control: the
# advantage is predicted to SHRINK there.

_GROUP_GENERATORS: dict[str, list[tuple[int, ...]]] = {
    # permutation tuples in one-line notation; standard generating sets
    "s5": [(1, 0, 2, 3, 4), (1, 2, 3, 4, 0)],  # (12), (12345)
    "a5": [(1, 2, 0, 3, 4), (1, 2, 3, 4, 0)],  # (123), (12345)
    "s3": [(1, 0, 2), (1, 2, 0)],  # (12), (123)
    "z60": [(1,), (7,)],  # additive shifts; coprime pair generates Z60
}


def _perm_mul(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    # (p o q)(i) = p(q(i))
    return tuple(p[q[i]] for i in range(len(p)))


def _group_elements(group: str) -> list[tuple[int, ...]]:
    if group == "z60":
        return [(i,) for i in range(60)]
    degree = len(_GROUP_GENERATORS[group][0])
    if group in {"s5", "s3"}:
        return sorted(itertools.permutations(range(degree)))

    def parity(p: tuple[int, ...]) -> int:
        seen = [False] * len(p)
        par = 0
        for i in range(len(p)):
            if not seen[i]:
                j, clen = i, 0
                while not seen[j]:
                    seen[j] = True
                    j = p[j]
                    clen += 1
                par ^= (clen - 1) % 2
        return par

    return sorted(p for p in itertools.permutations(range(degree)) if parity(p) == 0)


_GROUP_ELEMENT_INDEX: dict[str, dict[tuple[int, ...], int]] = {
    g: {p: i for i, p in enumerate(_group_elements(g))} for g in _GROUP_GENERATORS
}


def _group_compose(group: str, word: list[int]) -> tuple[int, ...]:
    if group == "z60":
        total = 0
        for gi in word:
            total = (total + _GROUP_GENERATORS["z60"][gi][0]) % 60
        return (total,)
    degree = len(_GROUP_GENERATORS[group][0])
    cur = tuple(range(degree))
    for gi in word:
        cur = _perm_mul(_GROUP_GENERATORS[group][gi], cur)
    return cur


def check_group(text: str) -> bool | None:
    parts = text.split()
    if parts[:2] != ["TASK", "group"] or "OUT" not in parts:
        return None
    oi = parts.index("OUT")
    group = parts[3]
    word = [int(t[1:]) for t in parts[parts.index("SEQ") + 1 : oi]]
    result = _group_compose(group, word)
    return parts[oi + 1] == f"e{_GROUP_ELEMENT_INDEX[group][result]}"


def _category_group(text: str) -> str | None:
    parts = text.split()
    if parts[:2] != ["TASK", "group"] or len(parts) < 4 or parts[2] != "G":
        return None
    return parts[3]


def _gen_group(size: int, seed: int, dials: dict[str, float]) -> dict[str, list[str]]:
    base_len = int(dials["length"])
    groups = ["s5", "a5", "z60", "s3"]
    splits: dict[str, list[str]] = {}
    for split, n in _split_sizes(size).items():
        rng = _rng("group", split, seed)
        # held-out lengths 2x-8x training length per the theory-program spec
        lo, hi = (2 * base_len, 8 * base_len) if split == "test" else (2, base_len)
        docs = []
        for _ in range(n):
            group = rng.choice(groups)
            length = rng.randint(lo, hi)
            n_gen = len(_GROUP_GENERATORS[group])
            word = [rng.randrange(n_gen) for _ in range(length)]
            result = _group_compose(group, word)
            idx = _GROUP_ELEMENT_INDEX[group][result]
            seq = " ".join(f"g{g}" for g in word)
            docs.append(f"TASK group G {group} SEQ {seq} OUT e{idx}")
        splits[split] = docs
    return splits


# ---------------------------------------------------------------------------
# task 10 (theory program): multiset tracking — symplectic/Noether channels (u55.5)
# Hypothesis: conserved-charge channels maintain bag counts across distractor
# spans natively (the conserved multiset memory prediction); count queries at
# held-out distances stress exactly that invariant.


def check_bag(text: str) -> bool | None:
    parts = text.split()
    if parts[:2] != ["TASK", "bag"] or "OUT" not in parts:
        return None
    qi, oi = parts.index("QUERY"), parts.index("OUT")
    counts: dict[str, int] = {}
    i = 2
    while i < qi:
        if parts[i] == "INS":
            counts[parts[i + 1]] = counts.get(parts[i + 1], 0) + 1
            i += 2
        elif parts[i] == "DEL":
            counts[parts[i + 1]] = max(0, counts.get(parts[i + 1], 0) - 1)
            i += 2
        else:
            i += 1  # NOP distractors and ';'
    return parts[oi + 1] == f"c{counts.get(parts[qi + 1], 0)}"


def _gen_bag(size: int, seed: int, dials: dict[str, float]) -> dict[str, list[str]]:
    n_ops = int(dials["n_ops"])
    distractor_frac = float(dials["distractor_frac"])
    splits: dict[str, list[str]] = {}
    for split, n in _split_sizes(size).items():
        rng = _rng("bag", split, seed)
        ops_n = n_ops * 2 if split == "test" else n_ops  # held-out OP-COUNT
        docs = []
        for _ in range(n):
            items = rng.sample(_LETTERS, 5)
            counts = dict.fromkeys(items, 0)
            chunks = []
            for _ in range(ops_n):
                if rng.random() < distractor_frac:
                    chunks.append(f"NOP {rng.choice(_LETTERS)}")
                    continue
                item = rng.choice(items)
                if counts[item] > 0 and rng.random() < 0.4:
                    chunks.append(f"DEL {item}")
                    counts[item] -= 1
                elif counts[item] < 9:  # single-token count answers c0..c9
                    chunks.append(f"INS {item}")
                    counts[item] += 1
            q = rng.choice(items)
            docs.append(f"TASK bag {' ; '.join(chunks)} QUERY {q} OUT c{counts[q]}")
        splits[split] = docs
    return splits


# ---------------------------------------------------------------------------
# task 11: placebo control — harness fairness (no mechanism should win)
# PURPOSE: a task on which NO mechanism holds a theoretical advantage. A
# statistically significant placebo gap means either (1) a harness fairness
# bug (budget mismatch, tokenizer artifact, LR favoritism) or (2) a genuine
# pure-optimization-rate effect - and either way it blocks publication of the
# affected comparisons until root-caused (re-run at 2x budget: a harness bug
# persists, an optimization-rate effect shrinks toward the entropy floor).
# The scorecard (vdc.4) MUST include the placebo row.
# The `structure` dial interpolates shuffled-hier (1.0 = intact hierarchy)
# down to fully random (0.0), doubling as a geometry dial whose profiler
# reading should be FLAT at the random end (cross-validates 77l.1).


def check_placebo(text: str) -> bool | None:
    parts = text.split()
    if parts[:2] != ["TASK", "placebo"]:
        return None
    return None  # LM-only: no per-document label


def _gen_placebo(size: int, seed: int, dials: dict[str, float]) -> dict[str, list[str]]:
    structure = float(dials["structure"])
    hier_dials = {"depth": 3.0, "branching": 3.0, "zipf_alpha": 1.0}
    hier_docs = _gen_hier(size, seed, hier_dials)
    splits: dict[str, list[str]] = {}
    for split, docs in hier_docs.items():
        rng = _rng("placebo", split, seed)
        out = []
        for doc in docs:
            body = doc.split()[2:]  # strip "TASK hier"; keep unigram statistics
            n_shuffle = int(len(body) * (1.0 - structure))
            idx = list(range(len(body)))
            shuffle_positions = rng.sample(idx, n_shuffle) if n_shuffle else []
            vals = [body[i] for i in shuffle_positions]
            rng.shuffle(vals)
            shuffled = list(body)
            for pos, val in zip(shuffle_positions, vals):
                shuffled[pos] = val
            out.append("TASK placebo " + " ".join(shuffled))
        splits[split] = out
    return splits


# ---------------------------------------------------------------------------
# task 12: real-data hierarchical retrieval — external validity (ultrametric et al.)
# One non-synthetic task: nested-structure retrieval over REAL data. Design
# decision (documented; supersedes the bead's network-fetch sketch): the
# corpus is the Python STANDARD LIBRARY's own AST structure - permissively
# licensed (PSF), ships with the interpreter, needs NO network fetch, and its
# provenance is content-hashed into the manifest. Cross-machine byte-identity
# holds only per Python version (the manifest records both); the synthetic
# tasks carry the unconditional determinism guarantee. Rationale: synthetic
# tasks prove a mechanism CAN win when geometry is planted; this one asks
# whether the geometry exists in the wild at exploitable strength.

_REALHIER_MODULES = ("json", "ast", "dataclasses", "pathlib", "argparse")


def _module_tree(module_name: str) -> tuple[dict, str]:
    import ast as ast_mod
    import importlib.util

    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
        raise RuntimeError(f"cannot locate pure-python source for module {module_name!r}")
    source = Path(spec.origin).read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode()).hexdigest()
    tree = ast_mod.parse(source)

    def build(node) -> dict:
        out: dict[str, Any] = {}
        for child in ast_mod.iter_child_nodes(node):
            if isinstance(child, (ast_mod.ClassDef, ast_mod.FunctionDef, ast_mod.AsyncFunctionDef)):
                sub = build(child)
                kind = "cls" if isinstance(child, ast_mod.ClassDef) else "fn"
                out[child.name] = sub if sub else kind
        return out

    return build(tree), digest


def _gen_realhier(size: int, seed: int, dials: dict[str, float]) -> dict[str, list[str]]:
    del dials
    trees = {}
    for mod in _REALHIER_MODULES:
        try:
            tree, _ = _module_tree(mod)
        except (RuntimeError, OSError):
            continue
        if _hier_paths(tree):  # e.g. pathlib is a re-export shim on 3.13: empty AST
            trees[mod] = tree
    if not trees:
        raise RuntimeError("realhier: no stdlib module sources available")
    splits: dict[str, list[str]] = {}
    for split, n in _split_sizes(size).items():
        rng = _rng("realhier", split, seed)
        docs = []
        for _ in range(n):
            mod = rng.choice(sorted(trees))
            tree = trees[mod]
            paths = _hier_paths(tree)
            if split == "test":  # held-out DEPTH: deepest quartile
                paths.sort(key=lambda pv: len(pv[0]))
                paths = paths[-max(1, len(paths) // 4) :]
            path, value = rng.choice(paths)
            docs.append(f"TASK hier TREE {_hier_serialize(tree)} PATH {' '.join(path)} OUT {value}")
        splits[split] = docs
    return splits


def realhier_provenance() -> dict[str, str]:
    """Source hashes for the manifest: {module: sha256-of-source}."""
    out = {}
    for mod in _REALHIER_MODULES:
        try:
            _, digest = _module_tree(mod)
            out[mod] = digest
        except (RuntimeError, OSError):
            out[mod] = "unavailable"
    return out


# ---------------------------------------------------------------------------
# difficulty axes for the extrapolation curves (vdc.2)


def _difficulty_nesting_depth(doc: str) -> float:
    depth = best = 0
    for tok in doc.split():
        if tok in "([{":
            depth += 1
            best = max(best, depth)
        elif tok in ")]}":
            depth = max(0, depth - 1)
    return float(best)


def _difficulty_span(start: str, end: str) -> Callable[[str], float]:
    def measure(doc: str) -> float:
        parts = doc.split()
        return float(parts.index(end) - parts.index(start) - 1)

    return measure


def _difficulty_token_count(token: str) -> Callable[[str], float]:
    def measure(doc: str) -> float:
        return float(doc.split().count(token))

    return measure


def _difficulty_exponent_spread(doc: str) -> float:
    parts = doc.split()
    exps = [abs(int(parts[i].split("e")[1])) for i in (3, 4)]
    return float(max(exps))


def _difficulty_doc_words(doc: str) -> float:
    return float(len(doc.split()))


# ---------------------------------------------------------------------------
# registry

TASKS: dict[str, TaskSpec] = {
    spec.name: spec
    for spec in [
        TaskSpec(
            name="dyck",
            target_mechanisms=("braid",),
            hypothesis="crossing structure tracks nesting topology; validity extrapolates to held-out depths",
            dials=(Dial("max_depth", 4, "maximum nesting depth (train range; test uses 2x)", 1, 16),),
            generate=_gen_dyck,
            checker=check_dyck,
            delimiters=("(", ")", "[", "]", "{", "}", "LABEL"),
            answer_marker="LABEL",
            difficulty_axis="nesting_depth",
            difficulty=_difficulty_nesting_depth,
        ),
        TaskSpec(
            name="copyops",
            target_mechanisms=("braid", "reversible"),
            hypothesis="explicit permutation/inversion structure extrapolates length-indexed bijections",
            dials=(Dial("length", 8, "max train sequence length (test uses (L, 2L])", 4, 64),),
            generate=_gen_copyops,
            checker=check_copyops,
            delimiters=("OP", "SEQ", "OUT"),
            answer_marker="OUT",
            difficulty_axis="sequence_length",
            difficulty=_difficulty_span("SEQ", "OUT"),
        ),
        TaskSpec(
            name="hier",
            target_mechanisms=("ultrametric", "fractal", "hyperbolic"),
            hypothesis="discrete, recursive, and continuous-curvature routing resolve nested key paths; held-out DEPTH is the discriminating axis",
            dials=(
                Dial("depth", 3, "tree depth (train; test uses depth+2)", 1, 8),
                Dial("branching", 3, "children per node", 2, 6),
                Dial("zipf_alpha", 1.0, "branch-mass imbalance (0 = uniform, larger = more imbalanced)", 0.0, 3.0),
            ),
            generate=_gen_hier,
            checker=check_hier,
            delimiters=("(", ")", "TREE", "PATH", "OUT"),
            answer_marker="OUT",
            difficulty_axis="path_depth",
            difficulty=_difficulty_span("PATH", "OUT"),
        ),
        TaskSpec(
            name="rel",
            target_mechanisms=("simplicial",),
            hypothesis="2-hop composition is triangle aggregation; gap grows with distractor fact count",
            dials=(Dial("n_facts", 8, "facts per document (train; test uses 2x)", 3, 64),),
            generate=_gen_rel,
            checker=check_rel,
            delimiters=("FACT", ";", "QUERY", "OUT"),
            answer_marker="OUT",
            difficulty_axis="fact_count",
            difficulty=_difficulty_token_count("FACT"),
        ),
        TaskSpec(
            name="rot",
            target_mechanisms=("quaternion", "octonion"),
            hypothesis="rotor value mixing composes orientation algebra; exact 24-element octahedral checker",
            dials=(Dial("length", 6, "max train rotation count (test uses (L, 2L])", 2, 32),),
            generate=_gen_rot,
            checker=check_rot,
            delimiters=("SEQ", "OUT"),
            answer_marker="OUT",
            difficulty_axis="rotation_count",
            difficulty=_difficulty_span("SEQ", "OUT"),
        ),
        TaskSpec(
            name="arith",
            target_mechanisms=("surreal",),
            hypothesis="magnitude-direction separation gives native scale arithmetic across decades",
            dials=(Dial("spread_decades", 6, "exponent range +-N decades (train; test uses 2x)", 1, 12),),
            generate=_gen_arith,
            checker=check_arith,
            delimiters=("CMP", "OUT"),
            answer_marker="OUT",
            difficulty_axis="exponent_spread",
            difficulty=_difficulty_exponent_spread,
        ),
        TaskSpec(
            name="regime",
            target_mechanisms=("ordinal", "hoss"),
            hypothesis="distribution flips reward principled restart/anneal; recovery time is the signal",
            dials=(
                Dial("segment_len", 24, "tokens per regime segment", 8, 128),
                Dial("n_regimes", 4, "regimes per document (train; test uses 2x)", 2, 16),
            ),
            generate=_gen_regime,
            checker=check_regime,
            delimiters=("SEG",),
        ),
        TaskSpec(
            name="needle",
            target_mechanisms=("ultrametric",),
            hypothesis="hierarchical addressing retrieves planted pairs at lengths where dense attention dilutes",
            dials=(
                Dial("context_words", 128, "approx words per context (train; test uses 4x)", 32, 2048),
                Dial("distractor_density", 0.2, "fraction of context budget spent on key-value pairs", 0.05, 0.9),
            ),
            generate=_gen_needle,
            checker=check_needle,
            delimiters=("PAIR", "QUERY", "OUT"),
            answer_marker="OUT",
            difficulty_axis="context_words",
            difficulty=_difficulty_doc_words,
        ),
        TaskSpec(
            name="group",
            target_mechanisms=("braid", "gauge"),
            hypothesis=(
                "non-solvable word problems (S5/A5) are the NC1-complete state-tracking barrier; "
                "solvable controls (Z60/S3) are the specificity control where the gap should shrink"
            ),
            dials=(Dial("length", 8, "max train word length (test uses 2x-8x)", 2, 32),),
            generate=_gen_group,
            checker=check_group,
            delimiters=("G", "SEQ", "OUT"),
            answer_marker="OUT",
            difficulty_axis="word_length",
            difficulty=_difficulty_span("SEQ", "OUT"),
            category=_category_group,
        ),
        TaskSpec(
            name="bag",
            target_mechanisms=("gauge", "reversible"),
            hypothesis="conserved-charge channels maintain multiset counts across distractor spans",
            dials=(
                Dial("n_ops", 12, "bag operations per document (train; test uses 2x)", 4, 64),
                Dial("distractor_frac", 0.3, "fraction of ops that are NOP distractors", 0.0, 0.9),
            ),
            generate=_gen_bag,
            checker=check_bag,
            delimiters=("INS", "DEL", "NOP", ";", "QUERY", "OUT"),
            answer_marker="OUT",
            difficulty_axis="op_count",
            difficulty=_difficulty_token_count(";"),
        ),
        TaskSpec(
            name="placebo",
            target_mechanisms=(),
            hypothesis=(
                "NO mechanism should win here; a significant gap is a harness fairness bug or a "
                "pure-optimization effect and blocks publication of affected comparisons until root-caused"
            ),
            dials=(Dial("structure", 0.0, "residual structure: 0 = fully shuffled, 1 = intact hierarchy", 0.0, 1.0),),
            generate=_gen_placebo,
            checker=check_placebo,
            delimiters=(),
        ),
        TaskSpec(
            name="realhier",
            target_mechanisms=("ultrametric", "fractal"),
            hypothesis="does hierarchical geometry exist in the wild (stdlib ASTs) at exploitable strength?",
            dials=(),
            generate=_gen_realhier,
            checker=check_hier,  # same TREE/PATH/OUT format as the synthetic task
            delimiters=("(", ")", "TREE", "PATH", "OUT"),
            answer_marker="OUT",
            difficulty_axis="path_depth",
            difficulty=_difficulty_span("PATH", "OUT"),
        ),
    ]
}

# Tasks built by `--task all`. realhier is opt-in (--include-real): it is the
# one task whose byte-identity is conditional on the Python version.
DEFAULT_TASKS: tuple[str, ...] = tuple(name for name in TASKS if name != "realhier")


# ---------------------------------------------------------------------------
# parquet + manifest plumbing


def _write_parquet(path: Path, docs: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"text": pa.array(docs, type=pa.string())})
    pq.write_table(table, path, row_group_size=PARQUET_ROW_GROUP_SIZE)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_task(
    task: str,
    *,
    out_dir: Path,
    size: int,
    seed: int,
    dial_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Generate one task's splits + manifest; returns the manifest dict."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; available: {sorted(TASKS)}")
    if size < 3:
        raise ValueError("size must be >= 3 (one document per split minimum)")
    spec = TASKS[task]
    dials = spec.resolve_dials(dial_overrides)
    splits = spec.generate(size, seed, dials)

    task_dir = out_dir / task
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for split, docs in splits.items():
        if split == "test":
            path = task_dir / "heldout" / "test_000.parquet"
        else:
            path = task_dir / f"{split}_000.parquet"  # 'train_*' sorts before 'val_*': val is LAST
        hashes[str(path.relative_to(task_dir))] = _write_parquet(path, docs)
        sizes[split] = len(docs)

    manifest: dict[str, Any] = {
        "task": task,
        "generator_version": GENERATOR_VERSIONS[task],
        "seed": seed,
        "size": size,
        "dials": dials,
        "split_sizes": sizes,
        "sha256": hashes,
        "target_mechanisms": list(spec.target_mechanisms),
        "hypothesis": spec.hypothesis,
    }
    if task == "realhier":
        import sys as _sys

        manifest["provenance"] = {
            "source": "python stdlib AST structure (PSF license)",
            "python_version": _sys.version,
            "module_source_sha256": realhier_provenance(),
            "byte_identity": "conditional on python version (synthetic tasks carry the unconditional guarantee)",
        }
    (task_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def generate_texts(task: str, *, size: int, seed: int, dial_overrides: dict[str, float] | None = None) -> list[str]:
    """In-memory train-split corpus for profiling (mgr profile-data --task)."""
    spec = TASKS[task]
    return spec.generate(size, seed, spec.resolve_dials(dial_overrides))["train"]


# ---------------------------------------------------------------------------
# robustness probe (the spec; see module docstring)


def apply_embedding_perturbation(x, eps: float, generator) -> Any:
    """Sup-norm bounded perturbation at the token-embedding output.

    x: (B, T, n_embd) embedding activations (post-wte, post-norm, pre-block-0).
    eps: L-infinity bound; each coordinate is perturbed by U(-eps, +eps).
    generator: torch.Generator pinning the draw (determinism is part of the spec).

    eps=0 returns x UNCHANGED (the identical tensor object - a byte-identical
    no-op by construction, asserted by the probe-spec test).
    """
    import torch

    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps}")
    if eps == 0.0:
        return x
    delta = (torch.rand(x.shape, generator=generator, dtype=torch.float32, device=x.device) * 2.0 - 1.0) * eps
    return x + delta.to(x.dtype)
