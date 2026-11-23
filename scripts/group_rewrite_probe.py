"""Learn short-word substitutions from endpoints and intervene on frozen group-task models.

Run with uv run python -m scripts.group_rewrite_probe --help. See the dated
protocol in bead hx3w. Outcomes belong under artifacts/probes/sizing.
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Callable
from functools import cache
from pathlib import Path
from zipfile import ZipFile

import pyarrow.parquet as pq
import torch
from rich.console import Console

from cli import _eval_score_doc, _eval_split_examples, _load_eval_checkpoint
from nanochat.diagnostics_data import _GROUP_ELEMENT_INDEX, TASKS, _group_compose, content_sha256
from nanochat.tokenizer import checkpoint_tokenizer

SUITE = "vdc4-group-tasktok-shuf-b30-s256-rung"
SPEC = TASKS["group"]
GROUPS = ("s5", "a5", "s3", "z60")
SYMBOLS = "01"


def parse_doc(doc: str) -> tuple[str, str, str]:
    tokens = doc.split()
    if tokens[:3] != ["TASK", "group", "G"] or tokens[4] != "SEQ" or tokens[-2] != "OUT":
        raise ValueError(f"Invalid group document: {doc!r}")
    letters = tokens[5:-2]
    if any(t not in {"g0", "g1"} for t in letters):
        raise ValueError(f"Invalid generator in {doc!r}")
    return tokens[3], "".join(t[1] for t in letters), tokens[-1]


def record(labels: dict[str, str], word: str, label: str) -> None:
    if labels.setdefault(word, label) != label:
        raise ValueError(f"Inconsistent endpoint for {word}")


def shortest(labels: dict[str, str]) -> dict[str, str]:
    representatives = {}
    for word in sorted(labels, key=lambda w: (len(w), w)):
        representatives.setdefault(labels[word], word)
    return representatives


class Rewriter:
    """A rule is licensed by two training examples with the same endpoint."""

    def __init__(self, labels: dict[str, str]):
        reps = shortest(labels)
        self.rules = {w: reps[y] for w, y in labels.items() if len(reps[y]) < len(w)}
        self.widths = sorted({len(w) for w in self.rules})

    def reduce(self, word: str) -> tuple[str, list[tuple[int, str, str]]]:
        steps = []
        while True:
            choices = (
                (len(replacement) - width, i, replacement, width)
                for width in self.widths
                for i in range(len(word) - width + 1)
                if (replacement := self.rules.get(word[i : i + width])) is not None
            )
            best = min(choices, default=None)
            if best is None:
                return word, steps
            _, i, replacement, width = best
            steps.append((i, word[i : i + width], replacement))
            word = word[:i] + replacement + word[i + width :]


def oracle(group: str, word: str) -> str:
    value = _group_compose(group, [int(letter) for letter in word])
    return f"e{_GROUP_ELEMENT_INDEX[group][value]}"


def verify_rewrite(group: str, word: str, result: str) -> None:
    if oracle(group, result) != oracle(group, word):
        raise ValueError(f"A supervised rewrite changed the {group} product: {word} -> {result}")


def transitions(labels: dict[str, str]) -> dict[tuple[str, str], str]:
    edges: dict[tuple[str, str], str] = {}
    for word, label in labels.items():
        prefix = labels.get(word[:-1])
        if prefix is not None and edges.setdefault((prefix, word[-1]), label) != label:
            raise ValueError("Endpoint labels do not form a deterministic right action")
    return edges


def complete_action(labels: dict[str, str], ask: Callable[[str], str]):
    """Learn an opaque, reachable transition table. Only ask() can see the teacher."""
    reps = shortest(labels)
    edges = transitions(labels)
    queries = []
    while True:
        missing = ((len(word) + 1, word + g, y, g) for y, word in reps.items() for g in SYMBOLS if (y, g) not in edges)
        choice = min(missing, default=None)
        if choice is None:
            return edges, reps, queries
        _, word, source, letter = choice
        target = ask(word)
        queries.append((word, source, target))
        edges[source, letter] = target
        if target not in reps or (len(word), word) < (len(reps[target]), reps[target]):
            reps[target] = word


def follow(labels: dict[str, str], edges: dict[tuple[str, str], str], word: str) -> str | None:
    state = labels.get(word[:2])
    for letter in word[2:]:
        if state is None:
            return None
        state = edges.get((state, letter))
    return state


def heldout(seeds: range):
    for seed in seeds:
        _, docs, _ = _eval_split_examples(SPEC, n=48, seed=seed)
        if len(docs) != 48:
            raise ValueError("Incomplete held-out split")
        for index, doc in enumerate(docs):
            yield seed, index, *parse_doc(doc)


def shuffle_word(word: str, seed: int, index: int) -> str:
    letters = list(word)
    random.Random(1000 * seed + index).shuffle(letters)
    return "".join(letters)


def frozen_reader(directory: Path):
    device = torch.device("cpu")
    model, meta, step = _load_eval_checkpoint(directory, 4730, device)
    if step != 4730 or meta["model_config"]["attention_type"] != "standard":
        raise ValueError("Unexpected model or step")
    tokenizer = checkpoint_tokenizer(str(directory), meta)

    @cache
    def predict(group: str, word: str) -> str:
        # Fixed dummy answer keeps every call blind to both target and target token count.
        tokens = " ".join(f"g{letter}" for letter in word)
        doc = f"TASK group G {group} SEQ {tokens} OUT e0"
        with torch.inference_mode():
            scored = _eval_score_doc(model, tokenizer, SPEC, doc, device=device, temperature=0.0, seed=0)
        if scored is None or scored[0] is None:
            raise ValueError("Frozen evaluator skipped a prompt")
        return scored[2]

    return predict


def first_answer(text: str) -> str:
    pieces = text.split()
    return pieces[0] if pieces else ""


def evaluate(predict, docs, rules, labels, policy: str):
    rows = []
    for seed, index, group, raw, target in docs:
        compact, steps = rules[group].reduce(raw)
        if policy == "supervised":
            verify_rewrite(group, raw, compact)
        sham = shuffle_word(compact, seed, index)
        outputs = {
            arm: first_answer(predict(group, w)) for arm, w in (("raw", raw), ("rewrite", compact), ("sham", sham))
        }
        rows.append(
            dict(
                seed=seed,
                index=index,
                group=group,
                word=raw,
                compact=compact,
                sham=sham,
                steps=steps,
                expected=target,
                compact_endpoint=oracle(group, compact),
                sham_endpoint=oracle(group, sham),
                lookup=labels[group].get(compact),
                target_seen=target in labels[group].values(),
                outputs=outputs,
            )
        )
    return rows


def summarize(rows):
    def counts(subset):
        def correct(r, arm):
            return r["outputs"][arm] == r["expected"]

        summary = {
            "n": len(subset),
            "preserved": sum(r["compact_endpoint"] == r["expected"] for r in subset),
            "sham_changed": sum(r["sham_endpoint"] != r["expected"] for r in subset),
            "lookup_covered": sum(r["lookup"] is not None for r in subset),
            "lookup_correct": sum(r["lookup"] == r["expected"] for r in subset),
            "seen_endpoint": sum(r["target_seen"] for r in subset),
            "raw_symbols": sum(len(r["word"]) for r in subset),
            "rewrite_symbols": sum(len(r["compact"]) for r in subset),
        }
        for arm in ("raw", "rewrite", "sham"):
            summary[arm] = sum(correct(r, arm) for r in subset)
        for arm in ("raw", "sham"):
            summary[f"wins_vs_{arm}"] = sum(correct(r, "rewrite") and not correct(r, arm) for r in subset)
            summary[f"losses_vs_{arm}"] = sum(not correct(r, "rewrite") and correct(r, arm) for r in subset)
        return summary

    primary = [r for r in rows if r["group"] in ("s5", "a5")]
    return {
        "all": counts(rows),
        "s5_a5": counts(primary),
        "changed_sham_s5_a5": counts([r for r in primary if r["expected"] != r["sham_endpoint"]]),
        "outside_dictionary_s5_a5": counts([r for r in primary if r["lookup"] is None]),
        **{g: counts([r for r in rows if r["group"] == g]) for g in GROUPS},
    }


def passes(summary, *, blind: bool) -> bool:
    row = summary["s5_a5"]
    effect = all(10 * (row["rewrite"] - row[arm]) >= row["n"] for arm in ("raw", "sham"))
    if blind:
        return effect and 100 * row["preserved"] >= 95 * row["n"]
    whole = summary["all"]
    return effect and whole["n"] == whole["preserved"] and all(summary[g]["sham_changed"] for g in ("s5", "a5"))


def save(path: Path, data) -> None:
    with path.open("x") as output:
        json.dump(data, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")


def run(inputs: Path, reference: Path, output: Path) -> None:
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    train_hash = json.loads((reference / "data/group/manifest.json").read_text())["content_sha256"]["train_000.parquet"]
    docs = pq.read_table(inputs / "data/group/train_000.parquet", columns=["text"]).column("text").to_pylist()
    if len(docs) != 80000 or content_sha256(docs) != train_hash:
        raise ValueError("Training split differs from the original manifest")
    by_group: dict[str, dict[str, str]] = {g: {} for g in GROUPS}
    for doc in docs:
        group, word, y = parse_doc(doc)
        record(by_group[group], word, y)
    rules = {g: Rewriter(labels) for g, labels in by_group.items()}
    probe = list(heldout(range(601, 604)))
    support = {}
    for g, labels in by_group.items():
        before = transitions(labels)
        after, reps, queries = complete_action(labels, lambda w: oracle(g, w))
        if any(oracle(g, w + a) != after[y, a] for y, w in reps.items() for a in SYMBOLS):
            raise ValueError("Full right-action certificate failed")
        group_docs = [(w, target) for _, _, group, w, target in probe if group == g]
        if not all(follow(labels, after, w) == target for w, target in group_docs):
            raise ValueError("Completed table failed a fresh prompt")
        support[g] = dict(
            short_words=len(labels),
            seen_states=len(set(labels.values())),
            rules=len(rules[g].rules),
            states=len(reps),
            known_edges=len(before),
            complete_edges=len(after),
            queries=queries,
            before_correct=sum(follow(labels, before, w) == target for w, target in group_docs),
            after_correct=len(group_docs),
            n=len(group_docs),
        )
    save(output / "teaching.json", support)
    short_by_model = {}
    primary = {}
    transfer = {}
    digests = {}
    historical = list(heldout(range(3)))
    with ZipFile(reference / "generations.zip") as archive:
        for train_seed in range(3):
            directory = inputs / f"runs/budget_0/group/standard/seed_{train_seed}/checkpoints"
            digests[train_seed] = {
                str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    directory / "model_004730.pt",
                    directory / "meta_004730.json",
                    directory / "tokenizer/tokenizer.json",
                )
            }
            predict = frozen_reader(directory)
            archive_path = f"evals/tasks/eval-b0-group-standard-s{train_seed}/generations.jsonl"
            raw_rows = [json.loads(line) for line in archive.read(archive_path).splitlines()]
            original = {
                (r["eval_seed"], r["doc_index"]): r
                for r in raw_rows
                if r["region"] == "held_out" and r["mode"] == "greedy"
            }
            if len(original) != len(historical):
                raise ValueError("Incomplete historical reference")
            for seed, index, g, w, target in historical:
                r = original[seed, index]
                if r["expected"] != target or predict(g, w) != r["got"]:
                    raise ValueError(f"Historical replay mismatch at {(train_seed, seed, index)}")
            rows = evaluate(predict, probe, rules, by_group, "supervised")
            save(output / f"primary-{train_seed}.json", rows)
            primary[train_seed] = summarize(rows)
            Console().print(f"Primary complete for training seed {train_seed}", style="cyan")
    if len({d["model_004730.pt"] for d in digests.values()}) != 3:
        raise ValueError("Repeated checkpoint bytes")
    advance = all(passes(s, blind=False) for s in primary.values())
    if advance:
        transfer_docs = list(heldout(range(701, 704)))
        for train_seed in range(3):
            directory = inputs / f"runs/budget_0/group/standard/seed_{train_seed}/checkpoints"
            predict = frozen_reader(directory)
            pseudo: dict[str, dict[str, str]] = defaultdict(dict)
            for g in GROUPS:
                for w in sorted(by_group[g]):
                    guess = first_answer(predict(g, w))
                    if guess.startswith("e") and guess[1:].isascii() and guess[1:].isdigit():
                        pseudo[g][w] = guess
            short_by_model[train_seed] = {g: len(pseudo[g]) for g in GROUPS}
            save(output / f"self-labels-{train_seed}.json", pseudo)
            blind_rules = {g: Rewriter(pseudo[g]) for g in GROUPS}
            blind_rows = evaluate(predict, transfer_docs, blind_rules, pseudo, "blind")
            gold_rows = evaluate(predict, transfer_docs, rules, by_group, "supervised")
            save(output / f"transfer-blind-{train_seed}.json", blind_rows)
            save(output / f"transfer-supervised-{train_seed}.json", gold_rows)
            transfer[train_seed] = {"blind": summarize(blind_rows), "supervised": summarize(gold_rows)}
            Console().print(f"Blind transfer complete for training seed {train_seed}", style="cyan")
    summary = dict(
        train_sha256=train_hash,
        fingerprints=digests,
        historical_exact_strings=3 * len(historical),
        primary=primary,
        primary_pass=advance,
        transfer=transfer,
        valid_self_labels=short_by_model,
        transfer_pass=all(passes(s["blind"], blind=True) for s in transfer.values()) if advance else None,
    )
    save(output / "summary.json", summary)
    Console().print(f"Historical replay 432/432. Primary {advance}. Blind transfer {summary['transfer_pass']}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", type=Path, required=True, help="Original scorecard with weights and training parquet"
    )
    parser.add_argument("--reference", type=Path, default=Path("artifacts/probes/sizing/scorecards") / SUITE)
    parser.add_argument("--output", type=Path, required=True, help="A new directory for raw paired results")
    args = parser.parse_args()
    run(args.inputs, args.reference, args.output)
