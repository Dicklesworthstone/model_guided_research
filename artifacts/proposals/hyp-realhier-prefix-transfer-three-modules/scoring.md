# Can ultrametric attention transfer to unseen Python modules?

This record precedes the feasibility measurements and model predictions. The
project agent asked one Codex research agent for a proposal and a second for
an independent score on November 22, 2026. Neither agent inspected the target
sources or results. The research agent who set the protocol made the design
decisions below. The project owner authorized exploration and did not choose
these cutoffs or supply a score.

## Original proposer prompt

> Independent prospective proposal for jida.42 and jida.43. Do not inspect generated examples, results, realhier held-out sources, or training data. Based only on the existing project question (ultrametric prefix geometry vs standard attention on nested Python AST path retrieval), propose one direction with an explicit held-out transfer metric, comparator, threshold, failure test, and a rejected alternative. Bound the claim to three unseen stdlib source modules, 256 task-token context, equal model compute, and a maximum of 12 training seeds per arm. Prespecify a quantitative feasibility stop for high majority-answer prior, poor input coverage, or inadequate power before any holdout outcomes. State what would count as a real negative. Reply once with the original proposal in 250 words or fewer; no file edits or tools required.

## Original proposal

> I predict ultrametric prefix geometry improves greedy exact-match accuracy on nested Python AST path retrieval by **at least 5 percentage points** over standard attention. Freeze three unseen stdlib modules and their Python revision before generating examples. Exclude those modules from training, tuning, and checkpoint selection. Score each module separately, then weight the three equally. The claim applies only to those modules.
>
> Use a 256-task-token context and 12 paired training seeds per arm. Within each pair, match training data order, optimizer, checkpoint rule, and total training FLOPs. The experimental unit is the seed pair. Report the mean paired accuracy difference and its two-sided 95% paired-*t* interval.
>
> **Stop before holdout inference** if any module’s majority-answer fraction exceeds 50%, or if fewer than 80% of its eligible queries retain the complete query and required AST evidence within 256 tokens. Compute those checks without model predictions. Using only disjoint development modules, estimate paired-seed variability. Stop if 12 pairs provide less than 80% projected power for the interval’s lower endpoint to exceed +5 points when the true gain is +10 points.
>
> A lower endpoint above +5 confirms the prediction. An upper endpoint below +5 is a **real negative**, including when ultrametric attention has a small positive mean advantage. An interval crossing the threshold is inconclusive. I reject pooled per-query scoring because one large module could conceal a transfer failure on the other two.

## Original scorer prompt

> Independently score this proposal before any realhier source scan or model results. Use the repo rubric: 0–100 each for Theoretical Novelty, Practical Feasibility, Potential Impact, Mathematical Rigor, Implementation Clarity, with 2–3 sentences per dimension. Composite 0–1000 weights 25%, 15%, 25%, 20%, 15% in that order. Proposal: ultrametric prefix geometry should beat standard attention by at least 5 percentage points in equal-weight macro greedy exact match across three unseen source-disjoint stdlib AST modules. Freeze CPython/module sources, 256 task-token context, equal model FLOPs and at most 12 paired training seeds. Stop before holdout inference if <80% of queries retain both complete input and answer in 256 tokens in any module, if its majority fraction exceeds 50%, or if projected power for the lower CI endpoint to exceed +5 points under a true +10-point gain is <80% using disjoint development modules. Report two-sided paired 95% CI, a real negative if its upper end is below +5. Critique the majority gate given only two answer labels; distinguish what must be preregistered from what is actually measured. State one principal failure mode. No tool calls, data inspection or file changes. Reply with your actual scoring transcript.

## Original score

> **Independent pre-evidence score: 689/1000.** I assessed the proposal before inspecting the `realhier` sources or any model results.
>
> | Dimension | Score | Rationale |
> |---|---:|---|
> | Theoretical novelty | 58/100 | Testing whether prefix geometry transfers across source-disjoint AST modules gives the mechanism a concrete prediction. The proposal gives little reason to expect that the distinction between the two answer labels depends on prefix geometry strongly enough to produce a five-point gain. |
> | Practical feasibility | 65/100 | Frozen stdlib sources, greedy decoding and a 12-pair ceiling make execution tractable. The exact-balance requirement could stop the study before inference, and the attainable power depends on paired variation measured on development modules. |
> | Potential impact | 71/100 | A lower confidence bound above five points at matched FLOPs would provide evidence that the attention choice matters on these three unseen modules. An interval narrow enough to rule out five points would also inform the project’s next investment. |
> | Mathematical rigor | 76/100 | Pairing training seeds and weighting modules equally give the comparison a clear estimand. The interval should use one macro-averaged difference per seed pair and describe uncertainty conditional on the three fixed modules. |
> | Implementation clarity | 78/100 | The context limit, effect threshold and stopping rules are explicit. The protocol still needs to specify the eligibility denominator, whether truncated queries count as errors or leave the scoring set, and the exact interval and power procedures. |
>
> **The majority gate requires exact balance.** With two labels, the majority fraction is always at least 50%. A module split 51–50 fails, as does every module with an odd number of eligible queries. If exact balance is intentional, preregister it as such. If the purpose is to exclude tasks with a dominant answer, choose a ceiling above 50% before inspecting the sources.
>
> **Preregister** the module identities and revisions, source-disjoint split, query extraction and scoring rules, tokenizer, FLOP accounting, seed pairing, and development-only power calculation. **Measure** per-module retention and label counts before holdout inference. Report a failed gate as a stopped study. If the gates pass, report the paired macro difference and two-sided 95% interval. A lower endpoint above +5 points supports the proposed gain. An upper endpoint below +5 points rules it out on the frozen comparison, even if the estimated gain is positive. An interval touching or crossing +5 is inconclusive.
>
> **Principal scientific failure mode.** Local syntax may determine the answer well enough that standard attention transfers across modules just as effectively, leaving little advantage for prefix geometry.

## Registered decision

Freeze CPython 3.13.0 at the source hashes in [source-pins.json](source-pins.json).
Train on `json`, `ast`, `dataclasses`, `argparse`, `logging` and `unittest.case`.
Reserve `fractions`, `configparser` and `statistics` for development.
Hold out `contextlib`, `enum` and `textwrap`. Each source and its serialized
definition tree must belong to exactly one split. Repeating an identical
module or tree under another name invalidates the split.

For each held-out module take its deepest quartile of leaf paths under the
existing generator's stable ordering, with a minimum of one path. Count each
distinct path once. Freeze a 512-token byte-level BPE vocabulary trained on
the training modules alone, with dataset seed 42 and 1,200 documents requested.
Count the leading BOS, full serialized tree, query, and complete answer in
each sequence. Report coverage for all eligible queries in each module.
Stop before model inference if coverage is below 80% in any module or if the
most frequent answer exceeds 90% in any module. The 90% limit preserves ten
points of headroom above that module’s constant-answer score. This replaces
the proposed exact-balance requirement before observing any label counts.

If inputs qualify, count an overlength query as an error in the primary
metric. Weight each module equally. Use width 64, depth 2, 4 query heads,
2 KV heads, a 256-token window, batch 4 and AdamW at 0.0006 with no warmup.
Compare the default ultrametric mechanism to standard at two trillion model
FLOPs per arm with matched data order and the final scheduled checkpoint.
First run four development-only pilot seed pairs, 30–33. Require standard to
exceed the development macro constant-answer floor on all four seeds. Use the
paired difference standard deviation to project power with a noncentral
t distribution, 11 degrees of freedom and the two-sided 95% critical value.
Stop if power is below 80% to put the lower endpoint above +0.05 under a true
gain of +0.10. Otherwise run the twelve confirmation pairs 3–14 once.
There is no sample extension or alternative holdout. Report a directional
result only if the paired interval and the registered verdict engine agree.
A failed preflight is a feasibility result and supplies zero model outcomes.

## Outcome at the registered input gate

Registration `e063ee2` preceded measurement. The public generator at
`edc4bf9` matched all 12 pinned source hashes, and the independent
implementation at the registration revision found the same 12 distinct
trees. Both trained the 512-token vocabulary exclusively on the 960
training documents. Both obtained the following counts.

| Held-out module | Distinct queries | Full inputs within 256 | Full sequence in tokens | Constant `fn` predictions correct |
|---|---:|---:|---:|---:|
| contextlib | 15 | 0 | 813–843 | 15 |
| enum | 25 | 0 | 1,086–1,097 | 25 |
| textwrap | 3 | 3 | 152–158 | 3 |

All three modules exceeded the registered 90% majority ceiling. Two also
missed the 80% input-coverage minimum. The constant answer already reaches
100% on every module before context restrictions, so this data cannot test a
five-point structural advantage over that control. The study stopped before
the development pilot, with zero training runs, zero model predictions and
no estimate of paired variability or power. The public adjudicator returned
`blocked / metric_missing`. The registered model prediction stays untested.

The [generator manifest](feasibility.json) and [independent measurement](independent-feasibility.json)
record the complete per-module counts. The [producer receipt](producer-dsr.json),
[independent receipt](independent-dsr.json) and [adjudicator response](refusal.json)
identify the revisions, commands and disposition.
