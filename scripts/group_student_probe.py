"""Train and score the independent paired group-composition study in bead qj9v."""

import argparse
import hashlib
import hmac
import json
import math
import random
import time
from collections import Counter, deque
from collections.abc import Callable, Generator
from dataclasses import dataclass
from functools import cache
from itertools import combinations, groupby, permutations, product
from pathlib import Path
from typing import cast
from zipfile import ZipFile

import torch
from scipy import stats

from nanochat.diagnostics_data import _GROUP_GENERATORS, _group_elements, _perm_mul
from nanochat.gpt import GPT, GPTConfig
from nanochat.report import get_git_info
from scripts.group_rewrite_probe import Actions, infer_actions, rooted_graph, save, teach_panel_case

ARMS = ("active", "random", "short_only")
STEPS = 900
BATCH = 64
PAD, BOS, ZERO, ONE, OUT = range(120, 125)
VOCAB = 128
TRAIN_WIDTH = 25
OLD_PANEL = Path("artifacts/probes/sizing/group-query-completion/panel.json")
EQUALITY_ARCHIVE = Path("artifacts/probes/sizing/group-equality/verification.zip")
EQUALITY_VIEWS = (("", ""), ("0", ""), ("1", ""), ("", "0"), ("", "1"), ("0", "1"), ("1", "0"))
EQUALITY_POLICIES = ("breadth_first", "symbol_counts", "one_view", "seven_views", "training_cache")


type EqualityPlan[T] = Generator[tuple[str, str], bool, T]


def _resume_equality_plan[T](
    plan: EqualityPlan[T], answer: bool | None = None
) -> tuple[tuple[str, str] | None, T | None]:
    try:
        return (next(plan) if answer is None else plan.send(answer)), None
    except StopIteration as finished:
        return None, finished.value


def _run_equality_plan[T](plan: EqualityPlan[T], equal: Callable[[str, str], bool]) -> T:
    question, result = _resume_equality_plan(plan)
    while question is not None:
        question, result = _resume_equality_plan(plan, equal(*question))
    return cast(T, result)


def config() -> GPTConfig:
    return GPTConfig(sequence_len=72, vocab_size=VOCAB, n_layer=4, n_embd=96, n_head=4, n_kv_head=4)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def graph_digest(graph) -> str:
    return hashlib.sha256(json.dumps(graph).encode()).hexdigest()


def worddraw(rng: random.Random, n: int) -> str:
    return format(rng.getrandbits(n), f"0{n}b")


def distinct_words(rng: random.Random, reserved: set[str], count: int, length: int) -> list[str]:
    result = []
    while len(result) < count:
        word = worddraw(rng, length)
        if word not in reserved:
            result.append(word)
            reserved.add(word)
    return result


def make_case(index: int, group: str, graph, names: list[str]) -> dict:
    rng = random.Random(93001 + index)

    def teacher(word: str) -> str:
        state = 0
        for symbol in word:
            state = graph[state][int(symbol)]
        return names[state]

    initial_words = ("".join(letters) for n in range(2, 9) for letters in product("01", repeat=n))
    initial = {word: teacher(word) for word in initial_words}
    vocabulary = set(names)
    active_table, active_paid = teach_panel_case(initial, vocabulary, teacher, squares=True)
    if any(len(word) > 23 for word, _ in active_paid):
        raise ValueError("Active query exceeded the registered training context")

    # A uniformly drawn endpoint query has only its length in common with the
    # active query. The random arm can buy a word that the active arm also buys.
    reserved = set(initial)
    random_paid = []
    for word, _ in active_paid:
        (chosen,) = distinct_words(rng, reserved, 1, len(word))
        random_paid.append((chosen, teacher(chosen)))
    reserved.update(w for w, _ in active_paid)

    pool: list[str] = []
    while len(pool) < 8192:
        candidate = worddraw(rng, rng.randint(9, 23))
        if candidate not in reserved:
            pool.append(candidate)
            reserved.add(candidate)

    arms = {}
    used = set()
    budget = len(active_paid)
    total = len(initial) + budget + 1536
    for arm, paid in (("active", active_paid), ("random", random_paid), ("short_only", [])):
        observed = dict(initial)
        observed.update(paid)
        additions = []
        if arm != "short_only":
            action = active_table if arm == "active" else Actions(vocabulary)
            infer_actions(action, observed, squares=True)
            for word in pool:
                target = action.read_right(observed, word)
                if target is not None:
                    additions.append((word, target))
                if len(additions) == 1536:
                    break
            if any(teacher(word) != target for word, target in additions):
                raise ValueError("An inferred training label disagrees with the independent teacher")
        originals = list(observed.items())
        rows = originals + additions
        repeat = total - len(rows)
        rows.extend(originals[i % len(originals)] for i in range(repeat))
        arms[arm] = dict(paid=paid, inferred=len(additions), repeated=repeat, rows=rows)
        used.update(word for word, _ in rows)

    primary = [w for length in (32, 48, 64) for w in distinct_words(rng, used, 32, length)]
    secondary = []
    while len(secondary) < 128:
        candidate = worddraw(rng, rng.randint(9, 15))
        if candidate not in used:
            secondary.append(candidate)
            used.add(candidate)
    majority = Counter(initial.values()).most_common(1)[0][0]
    return dict(
        index=index,
        group=group,
        train_seed=91001 + index,
        graph_sha=graph_digest(graph),
        label_count=len(names),
        arms=arms,
        primary=[(w, teacher(w), initial.get(w, majority)) for w in primary],
        secondary=[(w, teacher(w), initial.get(w, majority)) for w in secondary],
    )


def prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(2026120901)
    excluded = {case["graph_digest"] for case in json.loads(OLD_PANEL.read_bytes())["cases"]}
    for group in ("s5", "a5"):
        excluded.add(graph_digest(rooted_graph(group, tuple(_GROUP_GENERATORS[group]))))
    index = 0
    for group in ("s5", "a5"):
        elements = _group_elements(group)
        retained = 0
        while retained < 12:
            generators = rng.choice(elements), rng.choice(elements)
            graph = rooted_graph(group, generators)
            fingerprint = graph_digest(graph)
            if len(graph) != len(elements) or fingerprint in excluded:
                continue
            excluded.add(fingerprint)
            names = [str(i) for i in range(len(graph))]
            rng.shuffle(names)
            case = make_case(index, group, graph, names)
            case["generator_pair"] = generators
            case["opaque_names"] = names
            save(output / f"{index:02}.json", case)
            print(group, index, {arm: (len(a["paid"]), a["inferred"]) for arm, a in case["arms"].items()}, flush=True)
            retained += 1
            index += 1


def tokens(word: str, width: int) -> tuple[list[int], int]:
    prompt = [BOS, *(ONE if s == "1" else ZERO for s in word), OUT]
    if len(prompt) > width:
        raise ValueError("Word does not fit its fixed-width prompt")
    return prompt + [PAD] * (width - len(prompt)), len(prompt) - 1


def encode(rows, width: int, device: torch.device):
    prompts, offsets = zip(*(tokens(word, width) for word, *_ in rows), strict=True)
    ids = torch.tensor(prompts, dtype=torch.long, device=device)
    places = torch.tensor(offsets, dtype=torch.long, device=device)
    y = torch.tensor([int(row[1]) for row in rows], device=device)
    return ids, places, y


def schedule(step: int) -> float:
    if step < 50:
        return 0.001 * (step + 1) / 50
    return 0.0001 + 0.00045 * (1 + math.cos(math.pi * (step - 50) / (STEPS - 51)))


def state_digest(model: GPT) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def train_case(case_path: Path, output: Path, device: torch.device, *, cuda_graph: bool = False) -> None:
    case = json.loads(case_path.read_bytes())
    case_out = output / case_path.stem
    case_out.mkdir(parents=True, exist_ok=True)
    row_counts = {len(value["rows"]) for value in case["arms"].values()}
    (row_count,) = row_counts
    sampler = torch.Generator().manual_seed(case["train_seed"] + 100000)
    batches = torch.randint(row_count, (STEPS, BATCH), generator=sampler).to(device)
    for arm in ARMS:
        result = case_out / f"{arm}.json"
        if result.exists():
            continue
        torch.manual_seed(case["train_seed"])
        model = GPT(config())
        model.init_weights()
        # Initialize on CPU so the repo's CUDA initializer cannot cast the
        # embedding weights before the registered FP32-master training run.
        model.to(device)
        initial = state_digest(model)
        x, where, y = encode(case["arms"][arm]["rows"], TRAIN_WIDTH, device)
        labels = torch.full_like(x, -1)
        labels[torch.arange(row_count, device=device), where] = y
        optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, fused=True)
        train_losses = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.Stream() if cuda_graph else torch.cuda.current_stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph() if cuda_graph else None
        if graph is not None:
            static_x = torch.empty_like(x[:BATCH])
            static_labels = torch.empty_like(labels[:BATCH])
        with torch.cuda.stream(stream):
            start.record()
            for step, batch in enumerate(batches):
                optim.param_groups[0]["lr"] = schedule(step)
                if graph is None or step < 3:
                    optim.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = model(x[batch], targets=labels[batch])
                    loss.backward()
                else:
                    static_x.copy_(x[batch])
                    static_labels.copy_(labels[batch])
                    if step == 3:
                        # Keep the eager optimizer and scheduler, including all
                        # warmup updates. Only dispatch forward and backward as
                        # one graph. Its private gradient buffers are overwritten.
                        del loss
                        optim.zero_grad(set_to_none=True)
                        with torch.cuda.graph(graph, stream=stream):
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                graph_loss = model(static_x, targets=static_labels)
                            graph_loss.backward()
                    graph.replay()
                    loss = graph_loss
                optim.step()
                if step % 100 == 0 or step == STEPS - 1:
                    loss_value = float(loss.detach())
                    if not math.isfinite(loss_value):
                        raise ValueError(f"Nonfinite training loss for {case['index']} {arm}")
                    train_losses.append([step + 1, loss_value])
            end.record()
        end.synchronize()
        weights = case_out / f"{arm}-{time.time_ns()}.pt"
        torch.save(model.state_dict(), weights)
        save(
            result,
            dict(
                arm=arm,
                data_sha=digest(case_path),
                init_sha=initial,
                weights=weights.name,
                weights_sha=digest(weights),
                steps=STEPS,
                batch=BATCH,
                dense_positions=STEPS * BATCH * TRAIN_WIDTH,
                n_params=sum(p.numel() for p in model.parameters()),
                cuda_ms=start.elapsed_time(end),
                backward_mode="cuda_graph" if cuda_graph else "eager",
                losses=train_losses,
            ),
        )
        print(case_path.stem, arm, train_losses[-1], round(start.elapsed_time(end)), flush=True)


@torch.inference_mode()
def predict(model: GPT, rows, device: torch.device) -> list[int]:
    predictions: list[int] = []
    for i in range(0, len(rows), BATCH):
        x, where, _ = encode(rows[i : i + BATCH], 66, device)
        # Read logits from the only evaluated position in each raw, unrewritten word.
        logits = model(x)[torch.arange(len(x), device=device), where]
        predictions.extend(logits.argmax(dim=-1).tolist())
    return predictions


def score(data: Path, runs: Path, destination: Path, device: torch.device) -> None:
    results = []
    for index in range(24):
        path = data / f"{index:02}.json"
        case = json.loads(path.read_bytes())
        entries = {}
        for arm in ARMS:
            run = runs / path.stem
            info = json.loads((run / f"{arm}.json").read_bytes())
            weights = run / info["weights"]
            if not hmac.compare_digest(info["data_sha"], digest(path)) or not hmac.compare_digest(
                info["weights_sha"], digest(weights)
            ):
                raise ValueError("Data or trained weights do not match their training receipts")
            if info["steps"] != STEPS or info["dense_positions"] != STEPS * BATCH * TRAIN_WIDTH:
                raise ValueError("A student's update or token budget differs")
            model = GPT(config()).to(device)
            model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
            model.eval()
            answer = {}
            for split in ("primary", "secondary"):
                rows = case[split]
                guessed = predict(model, rows, device)
                answer[split] = {"correct": sum(y == int(row[1]) for y, row in zip(guessed, rows)), "answers": guessed}
            entries[arm] = dict(training=info, splits=answer)
        if len({e["training"]["init_sha"] for e in entries.values()}) != 1:
            raise ValueError("Matched arms did not start from identical parameters")
        primary = case["primary"]
        prior = Counter(row[1] for row in primary).most_common(1)[0][1]
        results.append(
            dict(
                index=index,
                group=case["group"],
                graph_sha=case["graph_sha"],
                n=len(primary),
                prior=prior,
                lookup=sum(target == fallback for _, target, fallback in primary),
                symbolic=len(primary),
                arms=entries,
            )
        )
    if len({r["graph_sha"] for r in results}) != 24:
        raise ValueError("Repeated presentations in the independent cohort")
    save(destination, {"results": results, **decision(results)})


def bounds(observations: list[float]) -> dict[str, float]:
    mean = sum(observations) / len(observations)
    padding = float(stats.t.ppf(0.95, len(observations) - 1) * stats.sem(observations))
    return dict(mean=mean, lower=mean - padding, upper=mean + padding)


def decision(results: list[dict]) -> dict:
    def difference(case, other):
        arms = case["arms"]
        baseline = case["prior"] if other == "prior" else arms[other]["splits"]["primary"]["correct"]
        candidate = arms["active" if other == "random" else "random"]["splits"]["primary"]["correct"]
        return (candidate - baseline) / case["n"]

    contrast = bounds([difference(r, "random") for r in results])
    floor = bounds([difference(r, "prior") for r in results])
    verdict = "inconclusive"
    if floor["lower"] > 0 and contrast["lower"] > 0.03:
        verdict = "supported"
    elif floor["lower"] > 0 and contrast["upper"] < 0.03:
        verdict = "refuted"
    return dict(verdict=verdict, delta=contrast, baseline_above_floor=floor)


class ShortlexRelations:
    """Streaming, proof-producing rewriting from certified action edges."""

    def __init__(self) -> None:
        self.rules: dict[str, str] = {}
        self.widths: tuple[int, ...] = ()
        self.known = {""}
        self.normal_forms: dict[str, tuple[str, tuple[tuple[int, str, str], ...]]] = {}

    def record(self, word: str, representative: str) -> None:
        self.known.add(representative)
        if word == representative:
            return
        if (len(word), word) <= (len(representative), representative):
            raise ValueError("An action equality does not orient by shortlex decrease")
        if self.rules.setdefault(word, representative) != representative:
            raise ValueError("Conflicting certified representatives")
        self.widths = tuple(sorted({len(side) for side in self.rules}, reverse=True))
        self.normal_forms.clear()

    def normalize(self, word: str) -> tuple[str, tuple[tuple[int, str, str], ...]]:
        if word not in self.normal_forms:
            current = word
            proof = []
            while True:
                match = next(
                    (
                        (start, lhs, self.rules[lhs])
                        for start in range(len(current))
                        for width in self.widths
                        if start + width <= len(current) and (lhs := current[start : start + width]) in self.rules
                    ),
                    None,
                )
                if match is None:
                    break
                start, lhs, rhs = match
                proof.append(match)
                current = current[:start] + rhs + current[start + len(lhs) :]
            self.normal_forms[word] = (current, tuple(proof))
        return self.normal_forms[word]

    def comparison_key(self, word: str, ref: str) -> tuple[str, ...]:
        current = cancelled_key(word, ref)
        while True:
            reduced = cancelled_key(*(self.normalize(side)[0] for side in current))
            if reduced == current:
                return current
            current = reduced


def acquire_equalities(
    size: int | None,
    equal: Callable[[str, str], bool],
    rank: Callable[[str, str], float],
    cached: Callable[[str, str], bool | None] | None = None,
    *,
    relations: ShortlexRelations | None = None,
) -> dict:
    """Recover a finite rooted action using equality and bijective generators.

    The ranker can change the order of queries. It cannot decide a merge or
    alter the candidate set. New representatives keep breadth-first order.
    With no known size, finish when both outgoing edges of every discovered
    representative have been established. Size-based forcing requires a size.
    """
    return _run_equality_plan(_plan_equalities(size, rank, cached, relations=relations), equal)


def _plan_equalities(
    size: int | None,
    rank: Callable[[str, str], float],
    cached: Callable[[str, str], bool | None] | None = None,
    *,
    relations: ShortlexRelations | None = None,
    remember: Callable[[str, str, bool], None] | None = None,
) -> EqualityPlan[dict]:
    representatives = [""]
    indices = {"": 0}
    edges = []
    used: tuple[set[int], set[int]] = (set(), set())
    trace = []
    counts: Counter = Counter()
    demanded = set()
    for source in representatives:
        row = []
        for index, generator in enumerate("01"):
            word = source + generator
            candidates = [i for i in range(len(representatives)) if i not in used[index]]
            trials = []
            event: dict = dict(word=word, trials=trials)
            normal, steps = relations.normalize(word) if relations is not None else (word, ())
            if relations is not None and normal in indices:
                target = indices[normal]
                if target not in candidates:
                    raise ValueError("A normal form conflicts with generator bijectivity")
                event["normal_form"] = normal
                event["substitutions"] = steps
                counts["rewritten"] += 1
            elif len(representatives) == size and len(candidates) == 1:
                target = candidates[0]
                counts["forced"] += 1
            else:
                if candidates:
                    demanded.add(word)
                    demanded.update(representatives[i] for i in candidates)
                ordered = sorted(candidates, key=lambda i: (-rank(word, representatives[i]), i))
                for target in ordered:
                    ref = representatives[target]
                    result = cached(word, ref) if cached else None
                    method = "cache"
                    if result is None:
                        result = yield word, ref
                        if remember is not None:
                            remember(word, ref, result)
                        method = "oracle"
                    counts[f"{method}_{'equal' if result else 'different'}"] += 1
                    trials.append((target, method, result))
                    if result:
                        break
                else:
                    if len(representatives) == size:
                        raise ValueError("Equality oracle is inconsistent with the advertised action size")
                    target = len(representatives)
                    representatives.append(word)
                    indices[word] = target
                    counts["new"] += 1
            used[index].add(target)
            row.append(target)
            event["target"] = target
            trace.append(event)
            if relations is not None:
                relations.record(word, representatives[target])
        edges.append(row)
    if size is not None and len(representatives) != size:
        raise ValueError("The advertised states are not all reachable")
    return dict(
        representatives=representatives, edges=edges, counts=dict(counts), trace=trace, ranked_words=sorted(demanded)
    )


def symbol_count_rank(word: str, ref: str) -> float:
    changes = [abs(word.count(g) - ref.count(g)) for g in "01"]
    parity = sum(change % 2 for change in changes)
    # The full domain uses words shorter than the model's context of 72.
    return -float(144 * parity + sum(changes))


def cached_equality(short: dict[str, str], word: str, ref: str) -> bool | None:
    # A common prefix is cancellable. This admits the one-letter and empty
    # canonical representatives when the padded pair fits the retained data.
    prefix = "0" * max(0, 2 - min(len(word), len(ref)))
    lhs, rhs = short.get(prefix + word), short.get(prefix + ref)
    return lhs == rhs if lhs is not None and rhs is not None else None


def walk_action(word: str, edges, *, start: int = 0) -> int:
    state = start
    for symbol in word:
        state = edges[state][int(symbol)]
    return state


@torch.inference_mode()
def equality_views(model: GPT, words: list[str], size: int, device: torch.device):
    padded = sorted({left + word + right for word in words for left, right in EQUALITY_VIEWS})
    width = max(map(len, padded)) + 2
    vectors = {}
    for start in range(0, len(padded), BATCH):
        batch = padded[start : start + BATCH]
        ids, positions, _ = encode([(word, "0") for word in batch], width, device)
        logits = model(ids)[torch.arange(len(ids), device=device), positions, :size]
        roots = logits.float().softmax(dim=-1).sqrt().cpu().numpy()
        vectors.update(zip(batch, roots, strict=True))
    profiles = {word: [vectors[left + word + right] for left, right in EQUALITY_VIEWS] for word in words}
    return profiles, len(padded)


def equality_case(case_path: Path, run: Path, device: torch.device) -> dict:
    case = json.loads(case_path.read_bytes())
    known_count = case["label_count"]
    graph = rooted_graph(case["group"], tuple(tuple(p) for p in case["generator_pair"]))
    if graph_digest(graph) != case["graph_sha"]:
        raise ValueError("The loaded presentation differs from the frozen group")

    def equal(word: str, ref: str) -> bool:
        return walk_action(word, graph) == walk_action(ref, graph)

    strategies = {"breadth_first": acquire_equalities(known_count, equal, lambda *_: 0.0)}
    strategies["symbol_counts"] = acquire_equalities(known_count, equal, symbol_count_rank)
    short = dict(case["arms"]["short_only"]["rows"][:508])
    strategies["training_cache"] = acquire_equalities(
        known_count, equal, symbol_count_rank, cached=lambda a, b: cached_equality(short, a, b)
    )
    words = strategies["breadth_first"]["ranked_words"]
    receipt = json.loads((run / "short_only.json").read_bytes())
    path = run / receipt["weights"]
    if not hmac.compare_digest(receipt["weights_sha"], digest(path)) or not hmac.compare_digest(
        receipt["data_sha"], digest(case_path)
    ):
        raise ValueError("Frozen checkpoint or training data fingerprint differs")
    model = GPT(config()).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    started = time.perf_counter()
    profiles, forward_prompts = equality_views(model, words, known_count, device)
    forward_seconds = time.perf_counter() - started

    def rank(word: str, ref: str, views: int) -> float:
        from math import fsum

        return fsum(float(a @ b) for a, b in zip(profiles[word][:views], profiles[ref][:views], strict=True)) / views

    strategies["one_view"] = acquire_equalities(known_count, equal, lambda a, b: rank(a, b, 1))
    strategies["seven_views"] = acquire_equalities(known_count, equal, lambda a, b: rank(a, b, len(EQUALITY_VIEWS)))
    for strategy in strategies.values():
        if strategy["edges"] != [list(row) for row in graph]:
            raise ValueError("An acquisition policy changed the canonical action")
        primary = case["primary"]
        if any(case["opaque_names"][walk_action(word, strategy["edges"])] != label for word, label, _ in primary):
            raise ValueError("A reconstructed table disagrees on a held-out word")
        strategy["verified_words"] = len(primary)
    return dict(
        index=case["index"],
        group=case["group"],
        graph_sha=case["graph_sha"],
        checkpoint=receipt["weights_sha"],
        forward_prompts=forward_prompts,
        forward_seconds=forward_seconds,
        strategies=strategies,
    )


def equality_panel(data: Path, runs: Path, output: Path, device: torch.device) -> None:
    results = []
    totals = {group: dict.fromkeys(EQUALITY_POLICIES, 0) for group in ("s5", "a5")}
    for index in range(24):
        case = equality_case(data / f"{index:02}.json", runs / f"{index:02}", device)
        for name, strategy in case["strategies"].items():
            counts = strategy["counts"]
            totals[case["group"]][name] += counts.get("oracle_equal", 0) + counts.get("oracle_different", 0)
        results.append(case)
        print(case["group"], index, totals[case["group"]], flush=True)
    pooled = {name: sum(row[name] for row in totals.values()) for name in EQUALITY_POLICIES}
    success = all(
        10 * pooled["seven_views"] <= 9 * pooled[control] for control in ("breadth_first", "symbol_counts", "one_view")
    ) and all(row["seven_views"] <= row["one_view"] for row in totals.values())
    save(output, dict(totals=totals, pooled=pooled, continuation_gate=success, results=results))


def pair_key(a: str, b: str) -> tuple[str, ...]:
    return tuple(sorted((a, b)))


def cancelled_key(a: str, b: str) -> tuple[str, ...]:
    start = 0
    while start < min(len(a), len(b)) and a[start] == b[start]:
        start += 1
    a, b = a[start:], b[start:]
    suffix = 0
    while suffix < min(len(a), len(b)) and a[-suffix - 1] == b[-suffix - 1]:
        suffix += 1
    if suffix:
        a, b = a[:-suffix], b[:-suffix]
    return pair_key(a, b)


def relator_key(a: str, b: str) -> tuple[int, ...]:
    """Canonical free-group conjugacy class of a b^-1 or its inverse."""
    signed = [int(s) + 1 for s in a] + [-int(s) - 1 for s in reversed(b)]
    reduced: list[int] = []
    for letter in signed:
        if reduced and reduced[-1] == -letter:
            reduced.pop()
        else:
            reduced.append(letter)
    start, stop = 0, len(reduced)
    while stop - start > 1 and reduced[start] == -reduced[stop - 1]:
        start += 1
        stop -= 1
    cyclic = tuple(reduced[start:stop])
    inverse = tuple(-letter for letter in reversed(cyclic))
    return min((word[i:] + word[:i] for word in (cyclic, inverse) for i in range(len(word))), default=())


def acquire_with_memo(
    size: int | None,
    equal: Callable[[str, str], bool],
    key: Callable[[str, str], tuple],
    *,
    relations: ShortlexRelations | None = None,
) -> dict:
    return _run_equality_plan(plan_acquire_with_memo(size, key, relations=relations), equal)


def plan_acquire_with_memo(
    size: int | None,
    key: Callable[[str, str], tuple],
    *,
    relations: ShortlexRelations | None = None,
) -> EqualityPlan[dict]:
    """Yield cache misses and return the complete action after the frontier closes."""
    memo = {key("", ""): True}

    def remember(word: str, ref: str, result: bool) -> None:
        memo[key(word, ref)] = result

    def cached(word: str, ref: str) -> bool | None:
        normalized = key(word, ref)
        if relations is not None:
            a, b = normalized
            if a == b:
                return True
            if a in relations.known and b in relations.known:
                return False
        return memo.get(normalized)

    started = time.perf_counter()
    acquired = yield from _plan_equalities(
        size, symbol_count_rank, cached=cached, relations=relations, remember=remember
    )
    acquired["seconds"] = time.perf_counter() - started
    acquired["memo_entries"] = len(memo)
    acquired["memo_symbols"] = sum(len(k) if not k or isinstance(k[0], int) else sum(map(len, k)) for k in memo)
    if relations is not None:
        acquired["rule_count"] = len(relations.rules)
        acquired["rule_symbols"] = sum(len(a) + len(b) for a, b in relations.rules.items())
        acquired["cached_normal_forms"] = len(relations.normal_forms)
    return acquired


def fresh_marked_actions(group: str, elements, exclusions: set[str], rng: random.Random):
    retained = []
    for _ in range(10000):
        generators = (rng.choice(elements), rng.choice(elements))
        graph = rooted_graph(group, generators)
        fingerprint = graph_digest(graph)
        if len(graph) != len(elements) or fingerprint in exclusions:
            continue
        exclusions.add(fingerprint)
        retained.append((generators, graph, fingerprint))
        if len(retained) == 6:
            return retained
    raise ValueError(f"Found only {len(retained)} eligible marked actions for {group}")


def marked_exclusions(data: Path) -> set[str]:
    return {
        *[case["graph_digest"] for case in json.loads(OLD_PANEL.read_bytes())["cases"]],
        *[json.loads((data / f"{i:02}.json").read_bytes())["graph_sha"] for i in range(24)],
        *(graph_digest(rooted_graph(g, tuple(gens))) for g, gens in _GROUP_GENERATORS.items()),
    }


def panel_families():
    a6 = [p for p in permutations(range(6)) if sum(x > y for i, x in enumerate(p) for y in p[i + 1 :]) % 2 == 0]
    return (("s5", _group_elements("s5")), ("a6", a6), ("z60", _group_elements("z60")))


def congruence_coverage(data: Path, output: Path) -> None:
    exclusions = marked_exclusions(data)
    with ZipFile(EQUALITY_ARCHIVE) as archive:
        previous = json.loads(archive.read("relators.json"))
    exclusions.update(previous["excluded_incomplete_s5"])
    exclusions.update(case["graph_sha"] for case in previous["results"])
    spent = sorted(exclusions)
    rng = random.Random(2026112311)
    presentations = [
        dict(group=group, generators=generators, size=len(graph), graph_sha=fingerprint)
        for group, elements in panel_families()
        for generators, graph, fingerprint in fresh_marked_actions(group, elements, exclusions, rng)
    ]
    save(
        output, dict(seed=2026112311, spent=spent, presentations=presentations, archive_sha256=digest(EQUALITY_ARCHIVE))
    )


def relator_panel(data: Path, output: Path) -> None:
    exclusions = marked_exclusions(data)
    leaked_s5 = fresh_marked_actions("s5", _group_elements("s5"), exclusions, random.Random(2026112303))
    rng = random.Random(2026112307)
    fresh = {group: fresh_marked_actions(group, elements, exclusions, rng) for group, elements in panel_families()}
    results = []
    for group, cases in fresh.items():
        for generators, graph, fingerprint in cases:
            case_index = len(results)
            eval_rng = random.Random(93700 + case_index)
            visited: set[str] = set()
            samples = [
                (w, walk_action(w, graph)) for n in (16, 32, 64, 128) for w in distinct_words(eval_rng, visited, 48, n)
            ]
            strategies = {}
            for name, key in (("raw", pair_key), ("cancelled", cancelled_key), ("relator", relator_key)):
                table = acquire_with_memo(len(graph), lambda a, b: walk_action(a, graph) == walk_action(b, graph), key)
                if table["edges"] != [list(row) for row in graph]:
                    raise ValueError("An algebraic cache changed the recovered action")
                if any(walk_action(w, table["edges"]) != y for w, y in samples):
                    raise ValueError("A new evaluation word changed under replay")
                strategies[name] = table
            results.append(
                dict(group=group, generators=generators, graph_sha=fingerprint, samples=samples, strategies=strategies)
            )
            print(group, case_index, {k: t["counts"] for k, t in strategies.items()}, flush=True)
    save(output, {"seed": 2026112307, "excluded_incomplete_s5": [row[2] for row in leaked_s5], "results": results})


def _conjugacy_representatives(elements):
    """Yield one ordered pair per orbit under simultaneous S5 conjugation."""
    conjugates = []
    for generator in _GROUP_GENERATORS["s5"]:
        inverse = tuple(generator.index(i) for i in range(len(generator)))
        conjugates.append({g: _perm_mul(_perm_mul(generator, g), inverse) for g in elements})

    seen = set()
    for pair in product(elements, repeat=2):
        if pair in seen:
            continue
        # All simultaneous conjugates give the same rooted marked graph.
        seen.add(pair)
        frontier = [pair]
        while frontier:
            first, second = frontier.pop()
            for conjugate in conjugates:
                moved = conjugate[first], conjugate[second]
                if moved not in seen:
                    seen.add(moved)
                    frontier.append(moved)
        yield pair


def _s5_marked_catalog():
    elements = _group_elements("s5")
    graphs = {}
    for pair in _conjugacy_representatives(elements):
        graph = rooted_graph("s5", pair)
        if len(graph) == len(elements):
            graphs[graph_digest(graph)] = graph, pair
    return dict(sorted(graphs.items()))


def s5_catalog():
    return {identifier: graph for identifier, (graph, _) in _s5_marked_catalog().items()}


def _graph_words(graph) -> list[str]:
    reps: list[str | None] = [None] * len(graph)
    reps[0] = ""
    for source, row in enumerate(graph):
        prefix = reps[source]
        if prefix is None:
            raise ValueError("Graph is not in reachable BFS order")
        for digit, dest in enumerate(row):
            if reps[dest] is None:
                reps[dest] = prefix + str(digit)
    words = [word for word in reps if word is not None]
    if len(words) != len(graph):
        raise ValueError("Graph contains a state without a representative")
    return words


def graph_equations(graph) -> list[tuple[str, ...]]:
    """Return transition equations of a canonical BFS-numbered action."""
    words = _graph_words(graph)
    return sorted(
        {
            pair_key(words[source] + str(digit), words[dest])
            for source, row in enumerate(graph)
            for digit, dest in enumerate(row)
            if words[source] + str(digit) != words[dest]
        }
    )


def _permutation_words(generators):
    identity = tuple(range(len(generators[0])))
    words = {identity: ""}
    frontier = deque([identity])
    while frontier:
        state = frontier.popleft()
        for mark, generator in enumerate(generators):
            target = _perm_mul(generator, state)
            if target not in words:
                words[target] = words[state] + str(mark)
                frontier.append(target)
    return words


def s5_presentation_certificates(marked_catalog):
    """Compile Moore's five relators and two marking equations into positive words."""
    transposition, cycle = _GROUP_GENERATORS["s5"]
    standard_words = _permutation_words((transposition, cycle))
    charts = []
    for g in standard_words:
        inverse = tuple(g.index(i) for i in range(len(g)))
        coordinates = {_perm_mul(_perm_mul(g, h), inverse): word for h, word in standard_words.items()}
        charts.append((_perm_mul(_perm_mul(g, transposition), inverse), _perm_mul(_perm_mul(g, cycle), inverse),
                       coordinates))

    certificates = {}
    for identifier, (_, marks) in marked_catalog.items():
        words = _permutation_words(marks)
        best = None
        for x, y, standard in charts:
            a, b = words[x], words[y]
            basis = (a, b)
            ties = [(str(i), "".join(basis[int(j)] for j in standard[h])) for i, h in enumerate(marks)]
            # The preceding b^5 equation licenses replacing inverse powers of b.
            relators = [
                (a * 2, ""), (b * 5, ""), ((a + b) * 4, ""),
                ((a + b * 4 + a + b) * 3, ""), ((a + b * 3 + a + b * 2) * 2, ""),
                *ties,
            ]
            questions = list(dict.fromkeys(cancelled_key(left, right) for left, right in relators if left != right))
            score = (len(questions), sum(len(word) for pair in questions for word in pair), questions)
            if best is None or score < best:
                best = score
        assert best is not None
        certificates[identifier] = best[-1]
    return certificates


def _a5c2_marked_catalog():
    """Encode the central bit by the action on points 5 and 6."""
    catalog = {}
    for first, second in _conjugacy_representatives(_group_elements("a5")):
        if len(_permutation_words((first, second))) != 60:
            continue
        for bits in ((0, 1), (1, 0), (1, 1)):
            marks = (first + (5 + bits[0], 6 - bits[0]), second + (5 + bits[1], 6 - bits[1]))
            graph = _a5c2_marked_graph(marks)
            catalog[graph_digest(graph)] = graph, marks
    return dict(sorted(catalog.items()))


def _a5c2_marked_graph(marks):
    indices = {state: index for index, state in enumerate(_permutation_words(marks))}
    if len(indices) != 120:
        raise ValueError("The lifted marks do not generate A5 x C2")
    return tuple(tuple(indices[_perm_mul(mark, state)] for mark in marks) for state in indices)


def _a5c2_certificate_questions(basis, bindings):
    a, b, c = basis
    relators = [
        (a * 2, ""), (b * 3, ""), ((a + b) * 5, ""), (c * 2, ""),
        (a + c, c + a), (b + c, c + b), ("0", bindings[0]), ("1", bindings[1]),
    ]
    return list(dict.fromkeys(cancelled_key(left, right) for left, right in relators if left != right))


def a5c2_presentation_certificates(marked_catalog):
    """Transport the icosahedral triangle relations and a central involution."""
    x = (1, 0, 3, 2, 4, 5, 6)
    y = (2, 1, 4, 3, 0, 5, 6)
    z = (0, 1, 2, 3, 4, 6, 5)
    standard = _permutation_words((x, y, z))
    charts = []
    for g in _group_elements("s5"):
        lift = g + (5, 6)
        inverse = tuple(lift.index(i) for i in range(7))
        coordinates = {_perm_mul(_perm_mul(lift, h), inverse): word for h, word in standard.items()}
        auxiliary = tuple(_perm_mul(_perm_mul(lift, h), inverse) for h in (x, y, z))
        charts.append((auxiliary, coordinates))

    certificates = {}
    for identifier, (_, marks) in marked_catalog.items():
        words = _permutation_words(marks)
        best = None
        for auxiliary, coordinates in charts:
            basis = tuple(words[h] for h in auxiliary)
            bindings = tuple("".join(basis[int(j)] for j in coordinates[h]) for h in marks)
            questions = _a5c2_certificate_questions(basis, bindings)
            score = (len(questions), sum(len(word) for pair in questions for word in pair), questions, basis, bindings)
            if best is None or score < best:
                best = score
        assert best is not None
        _, _, questions, basis, bindings = best
        certificates[identifier] = dict(basis=basis, bindings=bindings, questions=questions)
    return certificates


def identify_known_family(catalog, equal: Callable[[str, str], bool]) -> dict:
    return _run_equality_plan(plan_known_family(catalog), equal)


def plan_known_family(catalog) -> EqualityPlan[dict]:
    """Yield separating questions in the same order as the blocking selector."""
    survivors = list(catalog)
    trace = []
    while len(survivors) > 1:
        size = len(survivors)
        best = None
        for question in graph_equations(catalog[survivors[0]]):
            left, right = question
            yes = [g for g in survivors if walk_action(left, catalog[g]) == walk_action(right, catalog[g])]
            if 0 < len(yes) < size:
                proposal = (abs(2 * len(yes) - size), question, yes)
                if best is None or proposal[:2] < best[:2]:
                    best = proposal
        if best is None:
            raise ValueError("Distinct action tables failed to yield a separating question")
        _, (left, right), matched = best
        answer = yield left, right
        positive = set(matched)
        survivors = [g for g in survivors if (g in positive) == answer]
        trace.append(dict(pair=[left, right], candidates=size, predicted_yes=len(positive), answer=answer))
    return dict(marked_class=survivors[0], queries=trace)


type FamilyTree = str | FamilySplit


@dataclass(frozen=True)
class FamilySplit:
    question: tuple[str, str]
    size: int
    yes_size: int
    yes: FamilyTree
    no: FamilyTree


def _binary_path_length(size: int) -> int:
    """Minimum sum of leaf depths for a binary tree with this many leaves."""
    depth = size.bit_length() - 1
    return (depth + 2) * size - (1 << (depth + 1))


def compile_known_family(catalog, *, max_depth: int | None = None) -> FamilyTree:
    """Compile a bounded tree, requiring the binary path-length bound by default."""
    identifiers = list(catalog)
    size = len(identifiers)
    if size == 0:
        raise ValueError("The known family is empty")
    if size == 1:
        return identifiers[0]
    depth_limit = (size - 1).bit_length() if max_depth is None else max_depth

    graphs = [catalog[identifier] for identifier in identifiers]
    equations = {(left, right) for graph in graphs for left, right in graph_equations(graph)}
    words = sorted({word for pair in equations for word in pair}, key=lambda w: (len(w), w))
    endpoints = {word: tuple(walk_action(word, graph) for graph in graphs) for word in words}
    all_candidates = (1 << size) - 1

    def order_key(question):
        return sum(map(len, question)), question

    def find_tree(questions):
        partitions: dict[int, tuple[int, tuple[str, str]]] = {}
        for left, right in questions:
            yes = sum(1 << i for i, (a, b) in enumerate(zip(endpoints[left], endpoints[right], strict=True)) if a == b)
            if yes in (0, all_candidates):
                continue
            key = min(yes, all_candidates ^ yes)
            question = (left, right)
            if key not in partitions or order_key(question) < order_key(partitions[key][1]):
                partitions[key] = yes, question
        splits = sorted(partitions.values(), key=lambda entry: order_key(entry[1]))

        @cache
        def search(live: int, depth: int) -> FamilyTree | None:
            count = live.bit_count()
            if count == 1:
                return identifiers[live.bit_length() - 1]
            if count > 1 << depth:
                return None

            child_budget = _binary_path_length(count) - count
            choices = []
            seen = set()
            for mask, question in splits:
                yes = live & mask
                no = live ^ yes
                yes_size, no_size = yes.bit_count(), no.bit_count()
                key = min(yes, no)
                if key == 0 or key in seen or max(yes_size, no_size) > 1 << (depth - 1):
                    continue
                if max_depth is None and _binary_path_length(yes_size) + _binary_path_length(no_size) != child_budget:
                    continue
                seen.add(key)
                choices.append((abs(yes_size - no_size), order_key(question), yes, no))
            for _, (_, question), yes, no in sorted(choices):
                yes_tree = search(yes, depth - 1)
                if yes_tree is None:
                    continue
                no_tree = search(no, depth - 1)
                if no_tree is not None:
                    return FamilySplit(question, count, yes.bit_count(), yes_tree, no_tree)
            return None

        return search(all_candidates, depth_limit)

    tree = find_tree(equations)
    if tree is None:
        # Comparisons of words with identical endpoint vectors are redundant.
        shortest = {}
        for word in words:
            shortest.setdefault(endpoints[word], word)
        tree = find_tree(combinations(shortest.values(), 2))
    if tree is None:
        target = "the binary bound" if max_depth is None else f"depth {max_depth}"
        raise ValueError(f"The available comparisons could not attain {target}")
    return tree


def plan_compiled_family(tree: FamilyTree) -> EqualityPlan[dict]:
    trace = []
    while isinstance(tree, FamilySplit):
        answer = yield tree.question
        trace.append(dict(pair=list(tree.question), candidates=tree.size, predicted_yes=tree.yes_size, answer=answer))
        tree = tree.yes if answer else tree.no
    return dict(marked_class=tree, queries=trace)


_ALTERNATING_POLICY_SCHEMA = "mgr.group.alternating-policy.v1"


def _compile_alternating_policy():
    marked = _a5c2_marked_catalog()
    graphs = {identifier: graph for identifier, (graph, _) in marked.items()}
    return compile_known_family(graphs, max_depth=7), marked, a5c2_presentation_certificates(marked)


def _pack_family_tree(tree: FamilyTree):
    if isinstance(tree, str):
        return tree
    return [*tree.question, _pack_family_tree(tree.yes), _pack_family_tree(tree.no)]


def export_alternating_policy(output: Path) -> None:
    tree, marked, witnesses = _compile_alternating_policy()
    entries = {
        identifier: dict(marks=marks, basis=witnesses[identifier]["basis"], bindings=witnesses[identifier]["bindings"])
        for identifier, (_, marks) in marked.items()
    }
    save(output, dict(schema=_ALTERNATING_POLICY_SCHEMA, entries=entries, tree=_pack_family_tree(tree)))


def _is_policy_words(value, count: int) -> bool:
    return isinstance(value, list) and len(value) == count and all(
        isinstance(word, str) and set(word) <= {"0", "1"} for word in value
    )


def _validate_policy_tree(node, graphs, depth: int) -> FamilyTree:
    if isinstance(node, str) and list(graphs) == [node]:
        return node
    if not depth or not isinstance(node, list) or len(node) != 4 or not _is_policy_words(node[:2], 2):
        raise ValueError("Invalid selector node or incorrect leaf")
    left, right, yes_node, no_node = node
    yes, no = {}, {}
    for identifier, graph in graphs.items():
        branch = yes if walk_action(left, graph) == walk_action(right, graph) else no
        branch[identifier] = graph
    if not yes or not no:
        raise ValueError("The selector question does not split its candidates")
    return FamilySplit(
        (left, right), len(graphs), len(yes),
        _validate_policy_tree(yes_node, yes, depth - 1), _validate_policy_tree(no_node, no, depth - 1),
    )


def load_alternating_policy(path: Path):
    """Verify the marked actions, presentation witnesses, and selector before use."""
    try:
        policy = json.loads(path.read_bytes())
        if not isinstance(policy, dict) or policy.get("schema") != _ALTERNATING_POLICY_SCHEMA:
            raise ValueError("Unknown alternating policy schema")
        entries = policy["entries"]
        if not isinstance(entries, dict) or len(entries) != 57:
            raise ValueError("An alternating policy must contain 57 distinct markings")
        even = set(_group_elements("a5"))
        graphs, certificates = {}, {}
        for identifier, entry in entries.items():
            marks = entry["marks"]
            if not isinstance(marks, list) or len(marks) != 2 or any(
                not isinstance(mark, list) or len(mark) != 7 or any(type(point) is not int for point in mark)
                or tuple(mark[:5]) not in even or mark[5:] not in ([5, 6], [6, 5]) for mark in marks
            ):
                raise ValueError("Invalid lifted permutation marks")
            graph = _a5c2_marked_graph(tuple(map(tuple, marks)))
            if graph_digest(graph) != identifier:
                raise ValueError("The marked graph digest does not match")
            basis, bindings = entry["basis"], entry["bindings"]
            if not _is_policy_words(basis, 3) or not _is_policy_words(bindings, 2):
                raise ValueError("Invalid presentation words")
            questions = _a5c2_certificate_questions(basis, bindings)
            if not 1 <= len(questions) <= 7 or any(walk_action(a, graph) != walk_action(b, graph) for a, b in questions):
                raise ValueError("The presentation does not certify its marked graph")
            graphs[identifier] = graph
            certificates[identifier] = dict(basis=basis, bindings=bindings, questions=questions)
        tree = _validate_policy_tree(policy["tree"], graphs, 7)
        return tree, graphs, certificates
    except (OSError, TypeError, KeyError, ValueError, RecursionError) as error:
        raise ValueError(f"Invalid compiled alternating policy {path}: {error}") from error


def run_bounded_proposal(
    frontier: EqualityPlan[dict],
    proposal: EqualityPlan[dict | None],
    equal: Callable[[str, str], bool],
) -> dict:
    """Resolve either exact plan, paying for at most one proposal per four frontier queries.

    A rejected proposal returns None. Both plans must derive their answer from
    exact oracle results. The scheduler shares only identical unordered pairs.
    """
    plans = (frontier, proposal)
    pending = [_resume_equality_plan(plan) for plan in plans]
    shared: dict[tuple[str, ...], bool] = {}
    counts = [0, 0]
    paid = []
    names = ("frontier", "proposal")
    while True:
        for i, (question, result) in enumerate(pending):
            while question is not None and (key := pair_key(*question)) in shared:
                question, result = _resume_equality_plan(plans[i], shared[key])
            pending[i] = question, result
            if question is None and result is not None:
                return dict(winner=names[i], result=result, paid=paid, counts=counts)

        owner = 1 if pending[1][0] is not None and 4 * (counts[1] + 1) <= counts[0] else 0
        question = pending[owner][0]
        assert question is not None
        answer = equal(*question)
        shared[pair_key(*question)] = answer
        counts[owner] += 1
        paid.append((names[owner], *question, answer))
        pending[owner] = _resume_equality_plan(plans[owner], answer)


def congruence_panel(coverage: Path, output: Path, *, known_family_policy: str = "greedy") -> None:
    manifest = json.loads(coverage.read_bytes())
    if manifest["seed"] != 2026112311 or not hmac.compare_digest(manifest["archive_sha256"], digest(EQUALITY_ARCHIVE)):
        raise ValueError("The coverage manifest does not name the frozen source population")
    presentations = manifest["presentations"]
    if Counter(row["group"] for row in presentations) != {"s5": 6, "a6": 6, "z60": 6}:
        raise ValueError("The coverage manifest has an incomplete family")
    catalog_start = time.perf_counter()
    catalog = s5_catalog()
    catalog_seconds = time.perf_counter() - catalog_start
    compile_start = time.perf_counter()
    tree = compile_known_family(catalog) if known_family_policy == "optimal" else None
    compile_seconds = time.perf_counter() - compile_start
    results = []
    excluded = set(manifest["spent"])
    for i, row in enumerate(presentations):
        group = row["group"]
        graph = rooted_graph(group, tuple(tuple(g) for g in row["generators"]))
        fingerprint = graph_digest(graph)
        if fingerprint in excluded or fingerprint != row["graph_sha"] or len(graph) != row["size"]:
            raise ValueError("The manifest contains an ineligible graph")
        excluded.add(fingerprint)
        rng = random.Random(97510 + i)
        seen: set[str] = set()
        samples = [
            (word, walk_action(word, graph)) for n in (16, 32, 64, 128) for word in distinct_words(rng, seen, 48, n)
        ]
        oracle_seconds = 0.0

        def oracle(left: str, right: str) -> bool:
            nonlocal oracle_seconds
            start = time.perf_counter()
            same = walk_action(left, graph) == walk_action(right, graph)
            oracle_seconds += time.perf_counter() - start
            return same

        strategies = {}
        relations = ShortlexRelations()
        for name, key, rules in (
            ("cancelled", cancelled_key, None),
            ("congruence", relations.comparison_key, relations),
        ):
            oracle_seconds = 0.0
            table = acquire_with_memo(len(graph), oracle, key, relations=rules)
            table["oracle_seconds"] = oracle_seconds
            if table["edges"] != [list(r) for r in graph]:
                raise ValueError("An arm produced the wrong canonical action")
            if any(walk_action(word, table["edges"]) != answer for word, answer in samples):
                raise ValueError("The acquired action changed a fresh held-out answer")
            strategies[name] = table
        family = None
        if group == "s5":
            oracle_seconds = 0.0
            start = time.perf_counter()
            plan = plan_known_family(catalog) if tree is None else plan_compiled_family(tree)
            family = _run_equality_plan(plan, oracle)
            family.update(seconds=time.perf_counter() - start, oracle_seconds=oracle_seconds)
            if family["marked_class"] != fingerprint:
                raise ValueError("The known-family search chose the wrong marked group")
        results.append(
            dict(index=i, graph_sha=fingerprint, group=group, samples=samples, arms=strategies, family=family)
        )
        print(group, i, {k: table["counts"] for k, table in strategies.items()}, flush=True)
    report = dict(
        coverage_sha256=digest(coverage),
        catalog_size=len(catalog),
        catalog_seconds=catalog_seconds,
        results=results,
    )
    if tree is not None:
        report.update(known_family_policy=known_family_policy, family_compile_seconds=compile_seconds)
    save(output, report)


def _regular_proposal_graph(rows):
    graph = tuple(tuple(row) for row in rows)
    if len(graph) != 120 or any(len(row) != 2 for row in graph):
        raise ValueError("The proposal panel requires a two-generator action of order 120")
    if any(type(target) is not int for row in graph for target in row) or any(
        sorted(column) != list(range(120)) for column in zip(*graph, strict=True)
    ):
        raise ValueError("The proposed oracle actions must be permutations")
    words = _graph_words(graph)
    for start in graph[0]:
        left = [walk_action(word, graph, start=start) for word in words]
        if sorted(left) != list(range(120)) or any(
            left[graph[i][mark]] != graph[left[i]][mark] for i in range(120) for mark in range(2)
        ):
            raise ValueError("The proposal oracle is not a regular group action")
    return graph


def _check_proposal(tree, certificates, equal):
    proposal = _run_equality_plan(plan_compiled_family(tree), equal)
    identifier = proposal["marked_class"]
    observed = {pair_key(*step["pair"]): step["answer"] for step in proposal["queries"]}
    checks = []
    for pair in certificates[identifier]:
        reused = pair in observed
        answer = observed[pair] if reused else equal(*pair)
        checks.append(dict(pair=pair, answer=answer, reused=reused))
        if not answer:
            break
    if not checks:
        raise ValueError("Missing certificate for the proposed action")
    return dict(proposal=proposal, checks=checks, accepted=checks[-1]["answer"])


def _acquire_checked_proposal(
    tree, catalog, certificates, equal, *, alternate=None, commutator_route: bool = False, klein_route: bool = False,
    fifth_power_route: bool = False, involution_route: bool = False, central_involution_route: bool = False,
    order_fifteen_route: bool = False,
):
    result = _check_proposal(tree, certificates, equal)
    if result["accepted"]:
        return dict(right=catalog[result["proposal"]["marked_class"]], **result)
    if alternate is not None:
        next_tree, next_catalog, next_certificates = alternate
        attempt = _check_proposal(next_tree, next_certificates, equal)
        result["alternating"] = attempt
        if attempt["accepted"]:
            return dict(right=next_catalog[attempt["proposal"]["marked_class"]], **result)
    central_involution_route = central_involution_route or order_fifteen_route
    involution_route = involution_route or central_involution_route
    fifth_power_route = fifth_power_route or involution_route
    if commutator_route or klein_route or fifth_power_route:
        graph, witness = _try_order_three_commutator(120, equal)
        result["commutator"] = witness
        if graph is not None:
            return dict(right=graph, **result)
        if klein_route or fifth_power_route:
            graph, lift = _try_klein_four_quotient(120, equal, witness["orders"])
            result["klein"] = lift
            if graph is not None:
                return dict(right=graph, **result)
        if fifth_power_route:
            graph, lift = _try_order_five_commutator(120, equal, witness["orders"], witness["word"])
            result["fifth_power"] = lift
            if graph is not None:
                return dict(right=graph, **result)
        if involution_route:
            graph, lift = _try_klein_four_quotient(120, equal, witness["orders"], commutator=witness["word"])
            result["involution"] = lift
            if graph is not None:
                return dict(right=graph, **result)
            if central_involution_route and lift.get("rejected") == "central_involution" and lift.get("involution") == witness["word"]:
                graph, central = _recover_central_involution(120, equal, witness["word"])
                return dict(right=graph, central_involution=central, **result)
        if (order_fifteen_route and witness.get("rejected") == "commutator_order"
                and result["fifth_power"].get("rejected") == "commutator_order"):
            graph, lift = _try_order_fifteen_commutator(120, equal, witness["orders"], witness["word"])
            result["order_fifteen"] = lift
            if graph is not None:
                return dict(right=graph, **result)
    return dict(right=acquire_with_memo(120, equal, cancelled_key)["edges"], **result)


def _prime_divisors(size: int) -> list[int]:
    primes = []
    divisor = 2
    while divisor * divisor <= size:
        if size % divisor == 0:
            primes.append(divisor)
            while size % divisor == 0:
                size //= divisor
        divisor += 1
    if size > 1:
        primes.append(size)
    return primes[::-1]


def _lattice_action(width: int, shift: int, height: int) -> tuple[tuple[int, int], ...]:
    states = {(0, 0): 0}
    queue = deque([(0, 0)])
    edges = []
    while queue:
        x, y = queue.popleft()
        first = ((x + 1) % width, y)
        second = ((x - shift) % width, 0) if y + 1 == height else (x, y + 1)
        for target in (first, second):
            if target not in states:
                states[target] = len(states)
                queue.append(target)
        edges.append((states[first], states[second]))
    return tuple(edges)


def _element_order(mark: str, size: int, primes: list[int], equal: Callable[[str, str], bool]) -> int:
    order = size
    for prime in primes:
        while order % prime == 0 and equal(mark * (order // prime), ""):
            order //= prime
    return order


def _acquire_commuting_lattice(size: int, equal: Callable[[str, str], bool]) -> tuple[int, int, int]:
    primes = _prime_divisors(size)
    width = _element_order("0", size, primes, equal)
    height = size // width
    modulus, shift = 1, 0
    for prime in primes:
        while width % (modulus * prime) == 0:
            new_modulus = modulus * prime
            multiplier = width // new_modulus
            second = "1" * (height * multiplier)
            choices = []
            for digit in range(prime):
                residue = shift + digit * modulus
                exponent = residue * multiplier
                inverse = (-exponent) % width
                # Both forms express a^exponent b^(height*multiplier) = 1.
                left, right = ("0" * exponent + second, "") if exponent <= inverse else (second, "0" * inverse)
                choices.append((len(left) + len(right), residue, left, right))
            choices.sort()
            _, shift, _, _ = choices[-1]
            for _, residue, left, right in choices[:-1]:
                if equal(left, right):
                    shift = residue
                    break
            modulus = new_modulus
    return width, shift, height


def _acquire_commuting_action(size: int, equal: Callable[[str, str], bool]) -> dict:
    """Recover the marked lattice after an exact positive commutation reply."""
    lattice = _acquire_commuting_lattice(size, equal)
    return dict(right=_lattice_action(*lattice), lattice=lattice)


def _cyclic_extension_action(modulus: int, lattice, actions, powers) -> tuple[tuple[int, int], ...]:
    """Build the regular action in normal forms c^k a^x b^y."""
    width, shift, height = lattice
    alpha, beta = actions
    first_power, second_power = powers
    crossings = [0]
    for _ in range(1, height):
        # b^y a = c^d a b^y, with d_{y+1} = beta*d_y - 1.
        crossings.append((beta * crossings[-1] - 1) % modulus)
    root = (0, 0, 0)
    indices = {root: 0}
    queue = deque([root])
    edges = []
    while queue:
        k, x, y = queue.popleft()
        wraps, first_x = divmod(x + 1, width)
        first = ((k + pow(alpha, x, modulus) * crossings[y] + wraps * first_power) % modulus, first_x, y)
        second = (k, x, y + 1)
        if y + 1 == height:
            # a^s b^h = c^t, so conjugate c^t by a^(x-s).
            shifted = x - shift
            wraps, second_x = divmod(shifted, width)
            second_k = k + pow(alpha, shifted, modulus) * second_power + wraps * first_power
            second = (second_k % modulus, second_x, 0)
        for target in (first, second):
            if target not in indices:
                indices[target] = len(indices)
                queue.append(target)
        edges.append((indices[first], indices[second]))
    return tuple(edges)


def _memoized_noncommuting_equality(equal: Callable[[str, str], bool]) -> Callable[[str, str], bool]:
    """Cache exact comparisons after the caller has established noncommutation."""
    observed: dict[tuple[str, ...], bool] = {("01", "10"): False}

    def ask(left: str, right: str) -> bool:
        if left == right:
            return True
        pair = cancelled_key(left, right)
        if pair not in observed:
            observed[pair] = equal(*pair)
        return observed[pair]
    return ask


def _try_order_three_commutator(size: int, equal: Callable[[str, str], bool]):
    """Recognize and recover a group with derived subgroup C3 after a paid noncommutation reply."""
    if size % 3:
        return None, dict(rejected="group_order")
    ask = _memoized_noncommuting_equality(equal)
    primes = _prime_divisors(size)
    order_a = _element_order("0", size, primes, ask)
    order_b = _element_order("1", size, primes, ask)
    commutator = "01" + "0" * (order_a - 1) + "1" * (order_b - 1)
    witness = dict(word=commutator, orders=(order_a, order_b))
    if not ask(commutator * 3, ""):
        return None, dict(rejected="commutator_order", **witness)
    signs = []
    for mark in "01":
        if ask(mark + commutator, commutator + mark):
            signs.append(1)
        elif ask(mark + commutator, commutator * 2 + mark):
            signs.append(-1)
        else:
            return None, dict(rejected=f"normalization_{mark}", **witness)

    def quotient_equal(left: str, right: str) -> bool:
        return any(ask(left, commutator * power + right) for power in range(3))

    lattice = _acquire_commuting_lattice(size // 3, quotient_equal)
    width, shift, height = lattice
    # Both powers are known to belong to <c> from the quotient calculation.
    powers = []
    for word in ("0" * width, "0" * shift + "1" * height):
        powers.append(next((power for power in range(2) if ask(word, commutator * power)), 2))
    return _cyclic_extension_action(3, lattice, signs, powers), dict(lattice=lattice, signs=signs, powers=powers, **witness)


def _positive_inverse(word: str, orders: tuple[int, int]) -> str:
    return "".join(mark * (-sum(1 for _ in run) % orders[int(mark)]) for mark, run in groupby(reversed(word)))


def _shortest_positive_difference(left: str, right: str, orders: tuple[int, int]) -> str:
    first, second = cancelled_key(left, right)
    candidates = (first + _positive_inverse(second, orders), second + _positive_inverse(first, orders))
    return min(candidates, key=lambda word: (len(word), word))


def _recover_central_involution(size: int, equal: Callable[[str, str], bool], commutator: str):
    """Recover the action after the caller has verified a central commutator of order two."""
    ask = _memoized_noncommuting_equality(equal)
    comparisons = []

    def quotient_equal(left: str, right: str) -> bool:
        answer = ask(left, right) or ask(left, commutator + right)
        comparisons.append((left, right, answer))
        return answer

    lattice = _acquire_commuting_lattice(size // 2, quotient_equal)
    width, shift, height = lattice
    # Each known lift in <c> is either the identity or c.
    powers = (int(not ask("0" * width, "")), int(not ask("0" * shift + "1" * height, "")))
    return _cyclic_extension_action(2, lattice, (1, 1), powers), dict(lattice=lattice, powers=powers, quotient_queries=comparisons)


def _try_order_five_commutator(size: int, equal: Callable[[str, str], bool], orders: tuple[int, int], commutator: str):
    """Recover a derived C5 using the paid generator orders and commutator word."""
    if size % 5:
        return None, dict(rejected="group_order")
    ask = _memoized_noncommuting_equality(equal)
    if not ask(commutator * 5, ""):
        return None, dict(rejected="commutator_order")

    def compare_offset(left: str, right: str, power: int) -> bool:
        if power < 0:
            return ask(commutator * -power + left, right)
        return ask(left, commutator * power + right)

    actions = []
    for mark in "01":
        exponent = next((power for power in (1, -1, 2, -2) if compare_offset(mark + commutator, mark, power)), None)
        if exponent is None:
            return None, dict(rejected=f"normalization_{mark}")
        actions.append(exponent)

    coprime = size % 25 != 0
    comparisons = []

    def quotient_equal(left: str, right: str) -> bool:
        if coprime:
            answer = ask(_shortest_positive_difference(left, right, orders) * 5, "")
        else:
            answer = any(compare_offset(left, right, power) for power in (0, 1, -1, 2, -2))
        comparisons.append((left, right, answer))
        return answer

    lattice = _acquire_commuting_lattice(size // 5, quotient_equal)
    width, shift, height = lattice
    powers = []
    for word in ("0" * width, "0" * shift + "1" * height):
        power = next((value for value in (0, 1, -1, 2) if compare_offset(word, "", value)), -2)
        powers.append(power % 5)
    witness = dict(lattice=lattice, actions=actions, powers=powers, quotient_oracle="fifth_power" if coprime else "cosets",
                   quotient_queries=comparisons)
    return _cyclic_extension_action(5, lattice, actions, powers), witness


def _try_order_fifteen_commutator(size: int, equal: Callable[[str, str], bool], orders: tuple[int, int], commutator: str):
    """Use the already verified negative c^3 and c^5 comparisons to test and recover a derived C15."""
    if size % 15:
        return None, dict(rejected="group_order")
    ask = _memoized_noncommuting_equality(equal)
    if not ask(commutator * 15, ""):
        return None, dict(rejected="commutator_order")

    def compare_offset(left: str, right: str, base: str, exponent: int) -> bool:
        if exponent < 0:
            return ask(base * -exponent + left, right)
        return ask(left, base * exponent + right)

    components = ((5, (0, 1, -1)), (3, (0, 1, -1, 2, -2)))

    def recombine(residues: list[int]) -> int:
        return (10 * residues[0] + 6 * residues[1]) % 15

    actions = []
    modes = []
    for mark in "01":
        direct = next((value for value in (1, -1) if compare_offset(mark + commutator, mark, commutator, value)), None)
        if direct is not None:
            actions.append(direct % 15)
            modes.append("direct")
            continue
        residues = []
        for power, options in components:
            base = commutator * power
            residue = next((value for value in options[1:] if compare_offset(mark + base, mark, base, value)), None)
            if residue is None:
                return None, dict(rejected=f"normalization_{mark}")
            residues.append(residue)
        actions.append(recombine(residues))
        modes.append("prime_projections")

    quotient_size = size // 15
    coprime = math.gcd(quotient_size, 15) == 1
    comparisons = []

    def quotient_equal(left: str, right: str) -> bool:
        if coprime:
            answer = ask(_shortest_positive_difference(left, right, orders) * 15, "")
        else:
            answer = ask(left, right) or any(
                compare_offset(left, right, commutator, value) for magnitude in range(1, 8) for value in (magnitude, -magnitude)
            )
        comparisons.append((left, right, answer))
        return answer

    lattice = _acquire_commuting_lattice(quotient_size, quotient_equal)
    width, shift, height = lattice
    powers = []
    for word in ("0" * width, "0" * shift + "1" * height):
        if ask(word, ""):
            powers.append(0)
            continue
        residues = []
        for power, options in components:
            base = commutator * power
            residue = next((value for value in options[:-1] if compare_offset(word * power, "", base, value)), options[-1])
            residues.append(residue)
        powers.append(recombine(residues))
    witness = dict(lattice=lattice, actions=actions, conjugation_modes=modes, powers=powers,
                   quotient_oracle="fifteenth_power" if coprime else "cosets", quotient_queries=comparisons)
    return _cyclic_extension_action(15, lattice, actions, powers), witness


def _klein_extension_action(quotient, corrections) -> tuple[tuple[int, int], ...]:
    root = (0, 0)
    indices = {root: 0}
    queue = deque([root])
    graph = []
    while queue:
        kernel, coset = queue.popleft()
        row = []
        for target, correction in zip(quotient[coset], corrections[coset], strict=True):
            state = (kernel ^ correction, target)
            if state not in indices:
                indices[state] = len(indices)
                queue.append(state)
            row.append(indices[state])
        graph.append((row[0], row[1]))
    return tuple(graph)


def _try_klein_four_quotient(
    size: int, equal: Callable[[str, str], bool], orders: tuple[int, int], *, commutator: str | None = None,
):
    """Recover a detected normal Klein-four extension of a supported quotient."""
    if size % 4:
        return None, dict(rejected="group_order")
    ask = _memoized_noncommuting_equality(equal)
    if commutator is not None:
        # The already established noncommutation excludes the identity.
        if not ask(commutator * 2, ""):
            return None, dict(rejected="commutator_order")
        d = commutator
    else:
        involutions = [mark * (order // 2) for mark, order in zip("01", orders, strict=True) if order % 4 == 0]
        if involutions:
            d = min(involutions, key=lambda word: (len(word), word))
        else:
            order_ab = _element_order("01", size, _prime_divisors(size), ask)
            if order_ab % 4:
                return None, dict(rejected="no_order_four_power")
            d = "01" * (order_ab // 2)

    for order, mark in sorted(zip(orders, "01", strict=True)):
        if not ask(mark + d, d + mark):
            e = mark + d + mark * (order - 1)
            break
    else:
        return None, dict(rejected="central_involution", involution=d)
    if not ask(d + e, e + d):
        return None, dict(rejected="noncommuting_involutions", basis=(d, e))
    kernel = ("", d, e, d + e)
    if any(not any(ask(mark + word, other + mark) for other in kernel[1:]) for mark in "01" for word in (d, e)):
        return None, dict(rejected="nonnormal_kernel", basis=(d, e))

    quotient_size = size // 4
    use_square = commutator is not None and quotient_size % 2 == 1
    comparisons = []

    def quotient_equal(left: str, right: str) -> bool:
        if use_square:
            answer = ask(_shortest_positive_difference(left, right, orders) * 2, "")
        else:
            answer = any(ask(left, word + right) for word in kernel)
        if commutator is not None:
            comparisons.append((left, right, answer))
        return answer

    if quotient_equal("01", "10"):
        quotient = _acquire_commuting_action(quotient_size, quotient_equal)["right"]
    else:
        quotient, _ = _try_order_three_commutator(quotient_size, quotient_equal)
        if quotient is None:
            return None, dict(rejected="unsupported_quotient", basis=(d, e))

    reps = _graph_words(quotient)
    corrections = []
    for rep, targets in zip(reps, quotient, strict=True):
        row = []
        for mark, target in zip("01", targets, strict=True):
            left, right = rep + mark, reps[target]
            row.append(next((i for i, word in enumerate(kernel[:3]) if ask(left, word + right)), 3))
        corrections.append(tuple(row))
    witness = dict(basis=(d, e), quotient=quotient, corrections=corrections)
    if commutator is not None:
        witness.update(quotient_oracle="square" if use_square else "cosets", quotient_queries=comparisons)
    return _klein_extension_action(quotient, corrections), witness


def proposal_panel(
    coverage: Path, output: Path, *, abelian_route: bool = False, alternating_route: bool = False,
    alternating_policy: Path | None = None, commutator_route: bool = False, klein_route: bool = False,
    fifth_power_route: bool = False, involution_route: bool = False, central_involution_route: bool = False,
    order_fifteen_route: bool = False,
) -> None:
    try:
        cases = json.loads(coverage.read_bytes())["cases"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid proposal manifest {coverage}") from error
    if not isinstance(cases, list) or not cases:
        raise ValueError("The proposal manifest needs at least one case")
    started = time.process_time_ns()
    marked = _s5_marked_catalog()
    catalog = {identifier: graph for identifier, (graph, _) in marked.items()}
    tree = compile_known_family(catalog)
    edge = {identifier: graph_equations(graph) for identifier, graph in catalog.items()}
    common_cpu_ns = time.process_time_ns() - started
    started = time.process_time_ns()
    short = s5_presentation_certificates(marked)
    short_cpu_ns = time.process_time_ns() - started
    extra = {}
    alternate = None
    order = ("fallback", "edge", "short")
    central_involution_route = central_involution_route or order_fifteen_route
    involution_route = involution_route or central_involution_route
    fifth_power_route = fifth_power_route or involution_route
    has_alternating = alternating_route or alternating_policy is not None or commutator_route or klein_route or fifth_power_route
    if abelian_route or has_alternating:
        order += ("routed",)
    if has_alternating:
        started = time.process_time_ns()
        if alternating_policy is None:
            alternating_tree, alternating_marked, alternating_certificates = _compile_alternating_policy()
            alternating_graphs = {identifier: graph for identifier, (graph, _) in alternating_marked.items()}
        else:
            alternating_tree, alternating_graphs, alternating_certificates = load_alternating_policy(alternating_policy)
        alternate = (
            alternating_tree, alternating_graphs,
            {identifier: witness["questions"] for identifier, witness in alternating_certificates.items()},
        )
        order += ("extended",)
        extra = dict(alternating_cpu_ns=time.process_time_ns() - started, alternating_certificates=alternating_certificates)
    if commutator_route or klein_route or fifth_power_route:
        order += ("commutator",)
    if klein_route or fifth_power_route:
        order += ("klein",)
    if fifth_power_route:
        order += ("fifth_power",)
    if involution_route:
        order += ("involution",)
    if central_involution_route:
        order += ("central_involution",)
    if order_fifteen_route:
        order += ("order_fifteen",)
    results = []
    seen = set()
    for index, case in enumerate(cases):
        graph = _regular_proposal_graph(case["graph"])
        identifier = graph_digest(graph)
        if identifier != case["graph_sha"] or identifier in seen:
            raise ValueError("The proposal manifest changed or repeated an action")
        seen.add(identifier)
        arms = {}
        offset = index % len(order)
        for arm in order[offset:] + order[:offset]:
            paid = []

            def equal(left, right, tape=paid):
                answer = walk_action(left, graph) == walk_action(right, graph)
                tape.append((left, right, answer))
                return answer

            started = time.process_time_ns()
            if arm == "fallback":
                result = dict(right=acquire_with_memo(120, equal, cancelled_key)["edges"])
            elif arm not in {"edge", "short"} and equal("01", "10"):
                result = _acquire_commuting_action(120, equal)
            else:
                result = _acquire_checked_proposal(
                    tree, catalog, edge if arm == "edge" else short, equal,
                    alternate=alternate if arm not in {"edge", "short", "routed"} else None,
                    commutator_route=arm == "commutator", klein_route=arm == "klein",
                    fifth_power_route=arm == "fifth_power",
                    involution_route=arm == "involution",
                    central_involution_route=arm == "central_involution",
                    order_fifteen_route=arm == "order_fifteen",
                )
            cpu_ns = time.process_time_ns() - started
            if tuple(map(tuple, result["right"])) != graph:
                raise ValueError("Certificate or fallback produced an incorrect marked action")
            result.update(paid=paid, cpu_ns=cpu_ns)
            arms[arm] = result
        results.append(dict(graph_sha=identifier, family=case["family"], arms=arms))
    save(output, dict(
        coverage_sha256=digest(coverage), common_cpu_ns=common_cpu_ns, short_cpu_ns=short_cpu_ns,
        certificates=short, results=results, **extra,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=(
            "prepare", "train", "score", "equality", "relator", "congruence-cover", "congruence", "proposal", "compile-alternating-policy",
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=24)
    parser.add_argument("--cuda-graph", action="store_true", help="Capture forward and backward during training")
    parser.add_argument("--equality-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--abelian-route", action="store_true", help="Add exact commuting-case recovery to the proposal panel")
    parser.add_argument("--alternating-route", action="store_true", help="Also compare certified A5 x C2 proposals")
    parser.add_argument("--alternating-policy", type=Path, help="Use a validated policy from compile-alternating-policy")
    parser.add_argument("--commutator-route", action="store_true", help="Try direct recovery with a normal commutator of order three")
    parser.add_argument("--klein-route", action="store_true", help="Try direct recovery through a normal Klein-four quotient")
    parser.add_argument("--fifth-power-route", action="store_true", help="Try direct recovery with a normal commutator of order five")
    parser.add_argument("--involution-route", action="store_true", help="Try Klein-four recovery seeded by an order-two commutator")
    parser.add_argument("--central-involution-route", action="store_true", help="Recover from a certified central order-two commutator")
    parser.add_argument("--order-fifteen-route", action="store_true", help="Recover a cyclic order-fifteen derived subgroup by prime projections")
    parser.add_argument(
        "--known-family-policy", choices=("greedy", "optimal"), default="greedy",
        help="Equality policy for the known S5 comparison in the congruence panel",
    )
    args = parser.parse_args()
    if args.phase != "congruence" and args.known_family_policy != "greedy":
        parser.error("--known-family-policy optimal requires the congruence phase")
    if args.abelian_route and args.phase != "proposal":
        parser.error("--abelian-route requires the proposal phase")
    if args.alternating_route and args.phase != "proposal":
        parser.error("--alternating-route requires the proposal phase")
    if args.alternating_policy is not None and args.phase != "proposal":
        parser.error("--alternating-policy requires the proposal phase")
    if args.commutator_route and args.phase != "proposal":
        parser.error("--commutator-route requires the proposal phase")
    if args.klein_route and args.phase != "proposal":
        parser.error("--klein-route requires the proposal phase")
    if args.fifth_power_route and args.phase != "proposal":
        parser.error("--fifth-power-route requires the proposal phase")
    if args.involution_route and args.phase != "proposal":
        parser.error("--involution-route requires the proposal phase")
    if args.central_involution_route and args.phase != "proposal":
        parser.error("--central-involution-route requires the proposal phase")
    if args.order_fifteen_route and args.phase != "proposal":
        parser.error("--order-fifteen-route requires the proposal phase")
    if args.phase == "prepare":
        prepare(args.data)
    elif args.phase == "equality":
        if args.runs is None or args.output is None:
            parser.error("--runs and --output are required for the equality pilot")
        torch.set_num_threads(2)
        equality_panel(args.data, args.runs, args.output, torch.device(args.equality_device))
    elif args.phase == "relator":
        if args.output is None:
            parser.error("--output is required for the relator panel")
        relator_panel(args.data, args.output)
    elif args.phase == "congruence-cover":
        if args.output is None:
            parser.error("--output is required for the congruence coverage check")
        congruence_coverage(args.data, args.output)
    elif args.phase == "congruence":
        if args.coverage is None or args.output is None:
            parser.error("--coverage and --output are required for the congruence panel")
        congruence_panel(args.coverage, args.output, known_family_policy=args.known_family_policy)
    elif args.phase == "proposal":
        if args.coverage is None or args.output is None:
            parser.error("--coverage and --output are required for the proposal panel")
        proposal_panel(
            args.coverage, args.output, abelian_route=args.abelian_route, alternating_route=args.alternating_route,
            alternating_policy=args.alternating_policy, commutator_route=args.commutator_route, klein_route=args.klein_route,
            fifth_power_route=args.fifth_power_route, involution_route=args.involution_route,
            central_involution_route=args.central_involution_route,
            order_fifteen_route=args.order_fifteen_route,
        )
    elif args.phase == "compile-alternating-policy":
        if args.output is None:
            parser.error("--output is required to compile the alternating policy")
        export_alternating_policy(args.output)
    else:
        if args.runs is None:
            parser.error("--runs is required for training and scoring")
        device = torch.device("cuda")
        print(
            "Source",
            get_git_info()["commit_full"],
            "torch",
            torch.__version__,
            "GPU",
            torch.cuda.get_device_name(device),
            flush=True,
        )
        if args.phase == "train":
            for index in range(args.first, args.last):
                train_case(args.data / f"{index:02}.json", args.runs, device, cuda_graph=args.cuda_graph)
        else:
            if args.output is None:
                parser.error("--output is required for scoring")
            score(args.data, args.runs, args.output, device)
