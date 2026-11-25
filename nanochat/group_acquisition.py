"""Exact learning of a finite, regularly acting group from resettable word queries."""

import hashlib
import json
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from importlib.resources import files
from io import BytesIO
from itertools import groupby, product
from pathlib import Path
from time import perf_counter_ns
from typing import Literal, TextIO
from zipfile import ZipFile

import numpy as np

SYMBOLS = "01"
ANCHORS = ("00", "01", "10", "11")
SHORT = tuple("".join(word) for length in range(2, 5) for word in product(SYMBOLS, repeat=length))
CATALOGUE_SHA256 = "75c7df5cda251d1d28d63c300e5fd2587d6b64d33229635757e2eb45dfefc70f"
CATALOGUE_ORDERS = (32, 64, 128)
BUNDLED_POLICY_SHA256 = "4645a5343796fabf365eb93f96ee145b9d7f2c32d8d54f19766f4f50371fd96b"

type Shape = tuple[tuple[int, int], ...]
type Query = Callable[[str], str]
type Terminal = Literal["square", "unique", "cap"]


def record(labels: dict[str, str], word: str, label: str) -> None:
    if labels.setdefault(word, label) != label:
        raise ValueError(f"Inconsistent endpoint for {word}")


def shortest(labels: dict[str, str]) -> dict[str, str]:
    representatives = {}
    for word in sorted(labels, key=lambda w: (len(w), w)):
        representatives.setdefault(labels[word], word)
    return representatives


class Actions:
    """A map and its inverse for each generator on each side of the word."""

    def __init__(self, vocabulary: set[str]):
        self.vocabulary = vocabulary
        self.maps: dict[str, dict[str, str]] = {side + a: {} for side in "RrLl" for a in SYMBOLS}
        self._pending: deque[tuple[str, str, str]] = deque()

    def bind(self, edge: str, source: str, target: str) -> bool:
        if source not in self.vocabulary or target not in self.vocabulary:
            raise ValueError("Unknown endpoint")
        forward, backward = self.maps[edge], self.maps[edge.swapcase()]
        if source in forward and forward[source] != target or target in backward and backward[target] != source:
            raise ValueError("Inconsistent permutation edge")
        if source in forward:
            return False
        forward[source] = target
        backward[target] = source
        self._pending.append((edge, source, target))
        self._pending.append((edge.swapcase(), target, source))
        return True

    def _complete_column(self, edge: str) -> None:
        mapping, inverse = self.maps[edge], self.maps[edge.swapcase()]
        if len(mapping) == len(self.vocabulary) - 1:
            (source,) = self.vocabulary.difference(mapping)
            (target,) = self.vocabulary.difference(inverse)
            self.bind(edge, source, target)

    def close(self, squares: bool) -> None:
        for edge in ("R0", "R1", "L0", "L1"):
            self._complete_column(edge)
        if not squares:
            return

        # Both orientations are queued. If the new edge is on the far
        # side of a square, the inverse of the crossing edge lets us
        # examine the same square from the new edge's source.
        while self._pending:
            edge, source, target = self._pending[0]
            self._complete_column(edge)
            mapping = self.maps[edge]
            opposite = ("R0", "R1", "r0", "r1") if edge[0] in "Ll" else ("L0", "L1", "l0", "l1")
            for other in opposite:
                crossing = self.maps[other]
                start = crossing.get(source)
                if start is None:
                    continue
                end = crossing.get(target)
                if end is not None:
                    self.bind(edge, start, end)
                elif (end := mapping.get(start)) is not None:
                    self.bind(other, target, end)
            self._pending.popleft()

    def read_right(self, observations: dict[str, str], word: str) -> str | None:
        if word in observations:
            return observations[word]
        state = observations.get(word[:2])
        for g in word[2:]:
            if state is None:
                return None
            state = self.maps["R" + g].get(state)
        return state

    def representatives(self, observations: dict[str, str]) -> dict[str, str]:
        reps = shortest(observations)
        queue = [(len(w), w, y) for y, w in reps.items()]
        heapify(queue)
        while queue:
            _, w, source = heappop(queue)
            if w != reps[source]:
                continue
            for a in SYMBOLS:
                target = self.maps["R" + a].get(source)
                if target is None:
                    continue
                word = w + a
                if target not in reps or (len(word), word) < (len(reps[target]), reps[target]):
                    reps[target] = word
                    heappush(queue, (len(word), word, target))
        return reps


def _extensions(word: str, target: str) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    return ("R" + word[-1], word[:-1], target), ("L" + word[0], word[1:], target)


def _propagate_extensions(
    actions: Actions, observations: dict[str, str], pending: list[tuple[str, str, str]], *, squares: bool
) -> list[tuple[str, str, str]]:
    while True:
        before = sum(len(m) for m in actions.maps.values())
        unresolved = []
        for edge, part, target in pending:
            start = actions.read_right(observations, part)
            if start is None:
                unresolved.append((edge, part, target))
            else:
                actions.bind(edge, start, target)
        actions.close(squares)
        if not unresolved or before == sum(len(m) for m in actions.maps.values()):
            return unresolved
        pending = unresolved


def infer_actions(actions: Actions, observations: dict[str, str], *, squares: bool) -> None:
    """Propagate observed prefix and suffix extensions to a fixed point."""
    pending = [extension for word, target in observations.items() for extension in _extensions(word, target)]
    _propagate_extensions(actions, observations, pending, squares=squares)


def teach_panel_case(initial: dict[str, str], vocabulary: set[str], ask: Query, *, squares: bool):
    observations = initial.copy()
    actions = Actions(vocabulary)
    pending = [extension for word, target in observations.items() for extension in _extensions(word, target)]
    queries: list[tuple[str, str]] = []
    while True:
        pending = _propagate_extensions(actions, observations, pending, squares=squares)
        reps = actions.representatives(observations)
        missing = (
            (len(w) + 1, w + a, source, a)
            for source, w in reps.items()
            for a in SYMBOLS
            if source not in actions.maps["R" + a]
        )
        choice = min(missing, default=None)
        if choice is None:
            if any(len(actions.maps["R" + a]) != len(vocabulary) for a in SYMBOLS):
                raise ValueError("Learner stalled before covering every advertised state")
            return actions, queries
        _, word, source, symbol = choice
        if word in observations:
            raise ValueError("Requested an already labeled word")
        target = ask(word)
        known = actions.read_right(observations, word)
        if known is not None and known != target:
            raise ValueError(f"Inconsistent endpoint for {word}")
        queries.append((word, target))
        record(observations, word, target)
        pending.extend(_extensions(word, target))
        actions.bind("R" + symbol, source, target)


def _endpoint(shape: Shape, word: str) -> int:
    state = 0
    for mark in word:
        state = shape[state][int(mark)]
    return state


def _access(shape: Shape) -> list[tuple[str, int]]:
    seen: set[int] = set()
    queue: list[tuple[str, int]] = []
    for word in ANCHORS:
        state = _endpoint(shape, word)
        if state not in seen:
            seen.add(state)
            queue.append((word, state))
    for word, state in queue:
        for mark, target in zip(SYMBOLS, shape[state], strict=True):
            if target not in seen:
                seen.add(target)
                queue.append((word + mark, target))
    return queue


@dataclass(frozen=True)
class Catalogue:
    shapes: list[Shape]
    edges: np.ndarray
    word_rows: dict[str, int]
    endpoints: np.ndarray
    tiers: list[list[tuple[str, ...]]]
    naming: list[list[tuple[str, int]]]


def _prepare(shapes: list[Shape]) -> Catalogue:
    edges = np.asarray(shapes, dtype=np.uint8)
    naming = [_access(shape) for shape in shapes]
    words = sorted({*SHORT, *(word for plan in naming for word, _ in plan)})
    rows = np.arange(len(shapes))
    prefixes = {word[:i] for word in words for i in range(1, len(word) + 1)}
    columns = {"": np.zeros(len(shapes), dtype=np.uint8)}
    for word in sorted(prefixes, key=lambda w: (len(w), w)):
        columns[word] = edges[rows, columns[word[:-1]], int(word[-1])]
    tiers = []
    for plan in naming:
        shortlex = sorted(set(SHORT).union(word for word, _ in plan), key=lambda w: (len(w), w))
        tiers.append([tuple(tier) for _, tier in groupby(shortlex, len)])
    endpoints = np.stack([columns[word] for word in words])
    return Catalogue(shapes, edges, {word: row for row, word in enumerate(words)}, endpoints, tiers, naming)


def load_catalogue(path: Path, order: int) -> Catalogue:
    """Index one order from the audited complete anchor-study archive."""
    if order not in CATALOGUE_ORDERS:
        raise ValueError(f"The audited catalogue covers orders {', '.join(map(str, CATALOGUE_ORDERS))}")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != CATALOGUE_SHA256:
        raise ValueError("Catalogue checksum mismatch. Use the audited anchor-results.zip")
    with ZipFile(BytesIO(content)) as archive:
        data = json.loads(archive.read("scored.json"))
    order_entry = next(row for row in data["orders"] if row["order"] == order)
    shapes = []
    for profile in order_entry["profiles"].values():
        for packed in profile["full"]:
            edges = bytes.fromhex(packed)
            shapes.append(tuple(zip(edges[::2], edges[1::2], strict=True)))
    return _prepare(sorted(shapes))


def _column(index: Catalogue, live: np.ndarray, word: str) -> np.ndarray:
    if word in index.word_rows:
        return index.endpoints[index.word_rows[word], live]
    state = np.zeros(len(live), dtype=np.uint8)
    for mark in word:
        state = index.edges[live, state, int(mark)]
    return state


def _bounded_difference(
    shapes: tuple[Shape, Shape], completions: tuple[list[int], list[int]], limit: int | None
) -> str | None:
    left, right = shapes
    left_names, right_names = completions
    pending = deque((word, _endpoint(left, word), _endpoint(right, word)) for word in ANCHORS)
    seen = set()
    while pending:
        word, x, y = pending.popleft()
        if left_names[x] != right_names[y]:
            return word
        if (x, y) not in seen and (limit is None or len(word) < limit):
            seen.add((x, y))
            pending.extend((word + mark, left[x][i], right[y][i]) for i, mark in enumerate(SYMBOLS))
    if limit is None:
        raise ValueError("Different rooted actions gave the same completed observations")
    return None


def _worst(labels: np.ndarray, size: int, free: int) -> np.ndarray:
    stride = size + 1
    offsets = stride * np.arange(len(labels), dtype=np.intp)[:, None]
    counts = np.bincount((labels + 1 + offsets).ravel(), minlength=len(labels) * stride).reshape(-1, stride)
    return np.maximum(counts[:, 0], free * counts[:, 1:].max(axis=1))


class _VersionSpace:
    def __init__(self, index: Catalogue, size: int):
        self.index = index
        self.size = size
        self.live = np.arange(len(index.shapes))
        self.names = np.full((len(index.shapes), size), -1, dtype=np.int16)
        self.states = np.full_like(self.names, -1)
        self.facts: dict[str, str] = {}

    def learn(self, word: str, label: str) -> None:
        name = int(label)
        state = _column(self.index, self.live, word)
        known_name, known_state = self.names[self.live, state], self.states[self.live, name]
        keep = ((known_name == -1) | (known_name == name)) & ((known_state == -1) | (known_state == state))
        self.live, state = self.live[keep], state[keep]
        if not len(self.live):
            raise ValueError("The exact catalogue contains no action for the observed endpoints")
        self.names[self.live, state] = name
        self.states[self.live, name] = state
        self.facts[word] = label

    def completion(self, row: int) -> list[int]:
        names = self.names[row].copy()
        names[names < 0] = np.flatnonzero(self.states[row] < 0)
        return names.tolist()

    def select(self) -> str:
        free = max(1, self.size - len(set(self.facts.values())))
        ceiling = free * len(self.live)
        best = None
        left, right = int(self.live[0]), int(self.live[1])
        for tier in self.index.tiers[left]:
            words = [word for word in tier if word not in self.facts]
            if not words:
                continue
            rows = np.asarray([self.index.word_rows[word] for word in words], dtype=np.intp)
            endpoints = self.index.endpoints[rows[:, None], self.live]
            worst = _worst(self.names[self.live, endpoints], self.size, free)
            winner = int(worst.argmin())
            if worst[winner] < ceiling:
                best = (len(words[winner]), int(worst[winner]), words[winner])
                break
        difference = _bounded_difference(
            (self.index.shapes[left], self.index.shapes[right]),
            (self.completion(left), self.completion(right)),
            None if best is None else best[0],
        )
        if difference is not None and difference not in self.facts:
            endpoints = _column(self.index, self.live, difference)
            worst = _worst(self.names[self.live, endpoints][None, :], self.size, free)
            candidate = (len(difference), int(worst[0]), difference)
            if candidate[1] < ceiling and (best is None or candidate < best):
                best = candidate
        if best is None:
            raise ValueError("No informative probe while multiple rooted actions remain")
        return best[2]

    def select_frontier(self) -> str:
        """Find the globally shortest informative word, then minimize its worst bucket."""
        free = max(1, self.size - len(set(self.facts.values())))
        words = list(ANCHORS)
        endpoints = np.stack([_column(self.index, self.live, word) for word in words])
        seen: set[int] = set()
        while words:
            labels = self.names[self.live, endpoints]
            uniform = np.all(labels == labels[:, :1], axis=1)
            forced = uniform & ((labels[:, 0] >= 0) | (free == 1))
            informative = np.flatnonzero(~forced)
            if len(informative):
                worst = _worst(labels[informative], self.size, free)
                return words[int(informative[int(worst.argmin())])]
            extensions = []
            destinations = []
            # A forced label identifies the same endpoint vector in every
            # surviving shape. Its earliest shortlex prefix covers all suffixes.
            for word, states, raw_label in zip(words, endpoints, labels[:, 0], strict=True):
                label = int(raw_label)
                if label in seen:
                    continue
                seen.add(label)
                for mark, symbol in enumerate(SYMBOLS):
                    extensions.append(word + symbol)
                    destinations.append(self.index.edges[self.live, states, mark])
            words = extensions
            if destinations:
                endpoints = np.stack(destinations)
        raise ValueError("No informative probe while multiple rooted actions remain")


@dataclass(frozen=True, slots=True)
class _Probe:
    word: str
    branches: dict[int, "_PolicyNode"]


@dataclass(frozen=True, slots=True)
class _Determined:
    row: int
    observed_states: tuple[int, ...]


type _PolicyNode = _Probe | _Determined | None


@dataclass(frozen=True)
class CompiledPolicy:
    shapes: list[Shape]
    naming: list[list[tuple[str, int]]]
    _root: _PolicyNode

    def _fit(self, ask: Query) -> tuple[int, list[str]] | None:
        ranks: dict[str, int] = {}
        node = self._root
        while isinstance(node, _Probe):
            label = ask(node.word)
            rank = ranks.setdefault(label, len(ranks))
            if rank not in node.branches:
                raise ValueError("The exact catalogue contains no action for the observed endpoints")
            node = node.branches[rank]
        if node is None:
            return None
        names = [""] * len(self.shapes[node.row])
        for label, state in zip(ranks, node.observed_states, strict=True):
            names[state] = label
        return node.row, names


def compile_policy(catalogue: Catalogue, *, exhaustive_frontier: bool = False) -> CompiledPolicy:
    """Compile the shortlist or exhaustive shortest-word selector for later acquisitions."""
    size = catalogue.edges.shape[1]
    version = _VersionSpace(catalogue, size)
    select = version.select_frontier if exhaustive_frontier else version.select

    def visit(next_name: int) -> _PolicyNode:
        depth = len(version.facts)
        if depth < len(ANCHORS):
            word = ANCHORS[depth]
        elif len(version.live) == 1:
            row = int(version.live[0])
            states = tuple(int(state) for state in version.states[row, :next_name])
            return _Determined(row, states)
        elif depth >= size + 24:
            return None
        else:
            word = select()
        live = version.live
        endpoints = _column(catalogue, live, word)
        names = version.names[live, endpoints]
        branches = {}
        # Branches partition the live rows. Each child writes only its own rows,
        # so siblings need neither copies nor a rollback of the naming arrays.
        for outcome in np.unique(names):
            is_new = outcome == -1
            rank = next_name if is_new else int(outcome)
            version.live = live[names == outcome]
            version.learn(word, str(rank))
            branches[rank] = visit(next_name + int(is_new))
            del version.facts[word]
        return _Probe(word, branches)

    return CompiledPolicy(catalogue.shapes, catalogue.naming, visit(0))


def _decode_policy_node(value: list | None) -> _PolicyNode:
    if value is None:
        return None
    head, data = value
    if isinstance(head, str):
        return _Probe(head, {int(rank): _decode_policy_node(child) for rank, child in data})
    return _Determined(head, tuple(data))


def load_bundled_policy(order: int) -> CompiledPolicy:
    """Load a compiled policy from the release's audited catalogue."""
    if order not in CATALOGUE_ORDERS:
        raise ValueError(f"The bundled policy covers orders {', '.join(map(str, CATALOGUE_ORDERS))}")
    content = files("nanochat").joinpath("research_data", "group_policies.zip").read_bytes()
    if hashlib.sha256(content).hexdigest() != BUNDLED_POLICY_SHA256:
        raise ValueError("The bundled policy checksum does not match this release")
    with ZipFile(BytesIO(content)) as archive:
        payload = json.loads(archive.read(f"{order}.json"))
    if payload["format"] != "mgr.group-policy.v1" or payload["catalogue"] != CATALOGUE_SHA256:
        raise ValueError("The bundled policy has an incompatible format or catalogue")
    shapes = []
    for packed in payload["shapes"]:
        edges = bytes.fromhex(packed)
        shapes.append(tuple(zip(edges[::2], edges[1::2], strict=True)))
    return CompiledPolicy(shapes, [_access(shape) for shape in shapes], _decode_policy_node(payload["tree"]))


def _fit_catalogue(index: Catalogue, initial: dict[str, str], ask: Query) -> tuple[int, list[str]] | None:
    version = _VersionSpace(index, index.edges.shape[1])
    for word, label in initial.items():
        version.learn(word, label)
    while len(version.live) > 1 and len(version.facts) < version.size + 24:
        word = version.select()
        version.learn(word, ask(word))
    if len(version.live) > 1:
        return None
    row = int(version.live[0])
    names = [str(int(name)) if name >= 0 else "" for name in version.names[row]]
    return row, names


def _name_action(
    index: Catalogue | CompiledPolicy, fit: tuple[int, list[str]], roster: set[str], ask: Query
) -> tuple[str, dict[str, tuple[str, str]]]:
    row, names = fit
    missing = [(word, state) for word, state in index.naming[row] if not names[state]]
    available = roster.difference(names)
    for word, state in missing[:-1]:
        answer = ask(word)
        if answer not in available:
            raise ValueError("The oracle reused a name in the determined action")
        available.remove(answer)
        names[state] = answer
    if missing:
        (last,) = available
        names[missing[-1][1]] = last
    rows = {names[state]: (names[a], names[b]) for state, (a, b) in enumerate(index.shapes[row])}
    return names[0], rows


def _square(initial: dict[str, str], roster: set[str], ask: Query) -> tuple[str, dict[str, tuple[str, str]]]:
    actions, _ = teach_panel_case(initial, roster, ask, squares=True)
    inverse = actions.maps["r0"]
    root = inverse[inverse[initial["00"]]]
    rows = {state: (actions.maps["R0"][state], actions.maps["R1"][state]) for state in roster}
    return root, rows


def _verify_action(root: str, rows: dict[str, tuple[str, str]], tape: dict[str, str]) -> None:
    if any(set(column) != rows.keys() for column in zip(*rows.values(), strict=True)):
        raise ValueError("The learned marks are not permutations of the roster")
    # The two left translations must extend consistently along every right edge.
    # They commute with the right action and together reach the entire roster.
    for start in rows[root]:
        translation = {root: start}
        queue = [root]
        for state in queue:
            for target, image in zip(rows[state], rows[translation[state]], strict=True):
                if target not in translation:
                    translation[target] = image
                    queue.append(target)
                elif translation[target] != image:
                    raise ValueError("The learned action is inconsistent with a regular group")
        if len(translation) != len(rows) or len(set(translation.values())) != len(rows):
            raise ValueError("The learned action is inconsistent with a regular group")
    for word, label in tape.items():
        state = root
        for mark in word:
            state = rows[state][int(mark)]
        if state != label:
            raise ValueError(f"The learned action disagrees with the paid endpoint for {word}")


def validate_roster(value: object) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(label, str) or not label for label in value):
        raise ValueError("Roster must be a nonempty JSON array of nonempty strings")
    if len(set(value)) != len(value):
        raise ValueError("Roster labels must be distinct")
    return value


@dataclass(frozen=True)
class AcquiredAction:
    root: str
    transitions: dict[str, tuple[str, str]]
    terminal: Terminal


def acquire(roster: list[str], ask: Query, *, catalogue: Catalogue | CompiledPolicy | None = None) -> AcquiredAction:
    """Learn a regular action from stable endpoint labels and a public complete roster."""
    validate_roster(roster)
    if catalogue is not None and len(catalogue.shapes[0]) != len(roster):
        raise ValueError("Catalogue order differs from the roster size")
    encode = {label: str(i) for i, label in enumerate(roster)}
    identifiers = set(encode.values())
    tape: dict[str, str] = {}

    def query(word: str) -> str:
        if word in tape:
            raise ValueError(f"Repeated paid query for {word}")
        label = ask(word)
        if label not in encode:
            raise ValueError(f"Oracle endpoint for {word} is outside the roster")
        answer = encode[label]
        tape[word] = answer
        return answer

    if isinstance(catalogue, CompiledPolicy):
        fit = catalogue._fit(query)
    else:
        initial = {word: query(word) for word in ANCHORS}
        fit = None if catalogue is None else _fit_catalogue(catalogue, initial, query)
    terminal: Terminal
    if catalogue is None or fit is None:
        root, rows = _square(tape, identifiers, query)
        terminal = "square" if catalogue is None else "cap"
    else:
        root, rows = _name_action(catalogue, fit, identifiers, query)
        terminal = "unique"
    _verify_action(root, rows, tape)
    transitions = {
        label: (roster[int(rows[str(i)][0])], roster[int(rows[str(i)][1])]) for i, label in enumerate(roster)
    }
    return AcquiredAction(roster[int(root)], transitions, terminal)


def _write(stream: TextIO, value: object) -> None:
    stream.write(json.dumps(value, separators=(",", ":")) + "\n")
    stream.flush()


def run_stream(
    roster: list[str],
    stdin: TextIO,
    stdout: TextIO,
    *,
    catalogue: Catalogue | CompiledPolicy | None,
    transcript: TextIO | None,
    setup_ns: int,
) -> None:
    """Serve one acquisition over JSON lines, with an optional per-event journal."""

    def log(event: str, **fields: object) -> None:
        if transcript is not None:
            _write(transcript, {"event": event, **fields})

    def ask(word: str) -> str:
        log("request", word=word)
        started = perf_counter_ns()
        _write(stdout, {"word": word})
        raw = stdin.readline()
        elapsed = perf_counter_ns() - started
        if not raw:
            log("eof", word=word, roundtrip_ns=elapsed)
            raise ValueError(f"Oracle closed its input before answering {word}")
        try:
            reply = json.loads(raw, object_pairs_hook=tuple)
        except ValueError as error:
            log("invalid_reply", word=word, roundtrip_ns=elapsed)
            raise ValueError(f"Expected a JSON reply for {word}") from error
        if (
            not isinstance(reply, tuple)
            or len(reply) != 1
            or reply[0][0] != "label"
            or not isinstance(reply[0][1], str)
        ):
            log("invalid_reply", word=word, roundtrip_ns=elapsed)
            raise ValueError(f"Expected one string label in reply to {word}")
        label = reply[0][1]
        log("reply", word=word, label=label, roundtrip_ns=elapsed)
        return label

    mode = "square"
    if isinstance(catalogue, CompiledPolicy):
        mode = "compiled"
    elif catalogue is not None:
        mode = "tiered"
    log("start", roster=roster, mode=mode, setup_ns=setup_ns)
    try:
        action = acquire(roster, ask, catalogue=catalogue)
    except (ValueError, OSError) as error:
        log("error", message=str(error))
        raise
    payload = {"action": {"root": action.root, "transitions": action.transitions}, "terminal": action.terminal}
    log("complete", **payload)
    _write(stdout, payload)
