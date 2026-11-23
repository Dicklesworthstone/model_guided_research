"""Train and score the independent paired group-composition study in bead qj9v."""

import argparse
import hashlib
import hmac
import json
import math
import random
import time
from collections import Counter
from collections.abc import Callable
from itertools import permutations, product
from pathlib import Path
from zipfile import ZipFile

import torch
from scipy import stats

from nanochat.diagnostics_data import _GROUP_GENERATORS, _group_elements
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
    size: int,
    equal: Callable[[str, str], bool],
    rank: Callable[[str, str], float],
    cached: Callable[[str, str], bool | None] | None = None,
    *,
    relations: ShortlexRelations | None = None,
) -> dict:
    """Recover a rooted action using only equality and the size and bijection promises.

    The ranker can change the order of queries. It cannot decide a merge or
    alter the candidate set. New representatives keep breadth-first order.
    """
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
                        result = equal(word, ref)
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
    if len(representatives) != size:
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


def walk_action(word: str, edges) -> int:
    state = 0
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
    size: int,
    equal: Callable[[str, str], bool],
    key: Callable[[str, str], tuple],
    *,
    relations: ShortlexRelations | None = None,
) -> dict:
    memo = {key("", ""): True}

    def ask(word: str, ref: str) -> bool:
        result = equal(word, ref)
        memo[key(word, ref)] = result
        return result

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
    acquired = acquire_equalities(size, ask, symbol_count_rank, cached=cached, relations=relations)
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


def s5_catalog():
    elements = _group_elements("s5")
    graphs = {}
    for pair in product(elements, repeat=2):
        graph = rooted_graph("s5", pair)
        if len(graph) == len(elements):
            graphs[graph_digest(graph)] = graph
    return dict(sorted(graphs.items()))


def graph_equations(graph) -> list[tuple[str, ...]]:
    """Return transition equations of a canonical BFS-numbered action."""
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
    return sorted(
        {
            pair_key(words[source] + str(digit), words[dest])
            for source, row in enumerate(graph)
            for digit, dest in enumerate(row)
            if words[source] + str(digit) != words[dest]
        }
    )


def identify_known_family(catalog, equal: Callable[[str, str], bool]) -> dict:
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
        answer = equal(left, right)
        positive = set(matched)
        survivors = [g for g in survivors if (g in positive) == answer]
        trace.append(dict(pair=[left, right], candidates=size, predicted_yes=len(positive), answer=answer))
    return dict(marked_class=survivors[0], queries=trace)


def congruence_panel(coverage: Path, output: Path) -> None:
    manifest = json.loads(coverage.read_bytes())
    if manifest["seed"] != 2026112311 or not hmac.compare_digest(manifest["archive_sha256"], digest(EQUALITY_ARCHIVE)):
        raise ValueError("The coverage manifest does not name the frozen source population")
    presentations = manifest["presentations"]
    if Counter(row["group"] for row in presentations) != {"s5": 6, "a6": 6, "z60": 6}:
        raise ValueError("The coverage manifest has an incomplete family")
    catalog_start = time.perf_counter()
    catalog = s5_catalog()
    catalog_seconds = time.perf_counter() - catalog_start
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
            family = identify_known_family(catalog, oracle)
            family.update(seconds=time.perf_counter() - start, oracle_seconds=oracle_seconds)
            if family["marked_class"] != fingerprint:
                raise ValueError("The known-family search chose the wrong marked group")
        results.append(
            dict(index=i, graph_sha=fingerprint, group=group, samples=samples, arms=strategies, family=family)
        )
        print(group, i, {k: table["counts"] for k, table in strategies.items()}, flush=True)
    save(
        output,
        dict(
            coverage_sha256=digest(coverage),
            catalog_size=len(catalog),
            catalog_seconds=catalog_seconds,
            results=results,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("prepare", "train", "score", "equality", "relator", "congruence-cover", "congruence")
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=24)
    parser.add_argument("--cuda-graph", action="store_true", help="Capture forward and backward during training")
    parser.add_argument("--equality-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--coverage", type=Path)
    args = parser.parse_args()
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
        congruence_panel(args.coverage, args.output)
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
