"""Train and score the independent paired group-composition study in bead qj9v."""

import argparse
import hashlib
import hmac
import json
import math
import random
import time
from collections import Counter
from itertools import product
from pathlib import Path

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "train", "score"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=24)
    parser.add_argument("--cuda-graph", action="store_true", help="Capture forward and backward during training")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.data)
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
