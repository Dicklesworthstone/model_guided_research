# Can a transformer retain what an exact group teacher learned?

The student receives only a raw word and predicts its endpoint. A different
opaque label map is used for every presentation. Neither symbolic substitution
nor table lookup is available to the student at evaluation. The comparison
asks whether an active query policy improves long-word neural predictions over
a random query policy with the same query count and lengths.

## Results

The preregistered neural verdict is **inconclusive**. The random-query
students answered 36 of 2,304 long words correctly, below the empirical
majority floor of 110. The exact matched difference between active and random
was −0.48 percentage points. Its one-sided 95% lower and upper bounds were
−0.83 and −0.12 percentage points. Random minus the empirical floor averaged
−3.21 percentage points with a lower bound of −3.61. The floor gate therefore
prevents a ruling about the registered +3 percentage-point effect.

| Method | Raw words of length 32, 48 or 64 | Raw words of length 9–15 |
|---|---:|---:|
| Active queries | 25/2,304 | 158/3,072 |
| Random queries | 36/2,304 | 161/3,072 |
| Short labels only | 26/2,304 | 60/3,072 |
| Training-only dictionary with majority fallback | 32/2,304 | |
| Empirical majority per presentation | 110/2,304 | |
| Acquired symbolic table | 2,304/2,304 | |

Active won one presentation, tied 16, and lost seven against random on the
primary endpoint. Both paid curricula yielded more correct secondary answers
than the short-only control. Each acquired symbolic table reproduced its
held-out endpoints when replayed from only the initial examples and the
recorded paid labels. The transformer training experiment stopped after the
fixed evaluation. [Predictions and training receipts](student-score.json)

## Method

The protocol was fixed in Bead `qj9v` at `057b34d`. The sample consists of
12 new S5 presentations and 12 new A5 presentations. Each presentation supplied
all 508 words of lengths two through eight. The two paid arms each purchased
184 additional endpoints across the 24 cases. Both used the same algebraic
inference and supplied 1,536 additional inferred words per case. Seven
presentations required no paid labels and yielded identical active and random
training inputs. The short-only control repeated the initial labels to reach
the same row count. An independent permutation verifier checked every supplied
label and the disjoint train and evaluation words.

The four-layer standard Nanochat GPT used width 96 and four attention heads.
The training objective scored only the endpoint after the answer marker.
Within each presentation all arms shared initial weights, sample indices,
architecture and 900 AdamW updates with a batch size of 64 and 25 positions
per sequence. Each arm processed 1,440,000 sequence positions and 57,600
supervised endpoint targets. GPU elapsed times and data and checkpoint hashes
are recorded separately. Symbolic acquisitions, inferred labels, and independent
answer checking are logged by the study. Validation labels do not enter the
student input or guide optimization.

Each primary evaluation used 32 unseen words at each of lengths 32, 48 and 64.
An additional 128 disjoint words at lengths 9–15 provided a secondary diagnostic.
The study reports one paired accuracy difference per group presentation. The
registered effect is greater than three percentage points, with a one-sided
95% lower confidence bound on random minus the empirical majority prior greater
than zero before ruling in either direction. The fixed design assumes paired
standard deviation 0.14 and a true effect of 0.12 for power planning. The
experiment also records a training-only dictionary with majority fallback and
the exact symbolic reference.

## Reproduction

The dataset was generated at `d37e507`. The trained source revision is
`5cd11996eabe3c298552e36e68968c2700d0e4f8`. The GPU workers used CUDA graph
dispatch for the forward and backward passes after checking six complete
eager and captured runs. All sampled losses and final parameter bytes
matched. The learning-rate schedule and AdamW updates ran eagerly in both
paths. The comparison used two RTX 4090 devices and Torch 2.13.0+cu130.
The archives contain all training and test documents, per-example
predictions, checkpoint fingerprints, parity logs and audit scripts.
Checkpoint binaries are retained on the GPU worker.

At the trained revision, Linux DSR passed Ruff lint and formatting, full
type checking, and pytest with 923 passes and 16 skips. A full ledger dry
run reproduced all 76 other verdict records byte for byte after the
symbolic collector landed. [Formal symbolic adjudication](../../adjudications/group-query-completion/report.md)

Supply the following commands to DSR from a clean checkout, using new paths.
Training and scoring use CUDA. Both training slices must finish before scoring.

```sh
uv run --locked python -m scripts.group_student_probe prepare --data /path/to/cohort
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m scripts.group_student_probe train --cuda-graph --data /path/to/cohort --runs /path/to/runs --first 0 --last 12
CUDA_VISIBLE_DEVICES=1 uv run --locked python -m scripts.group_student_probe train --cuda-graph --data /path/to/cohort --runs /path/to/runs --first 12 --last 24
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m scripts.group_student_probe score --data /path/to/cohort --runs /path/to/runs --output /path/to/results.json
```

[Frozen cohort archive](registered-cohort.tgz) · [Audit scripts and machine receipts](verification.zip)

## What a classical catalogue recovers from the same short labels

A later finite replay gave both classical arms all 508 unique short labels per presentation and the public roster. It used all 24 historical cases. The catalogue considered 723,600 ordered pairs in the complete SmallGrp inventory at orders 60 and 120, without receiving the hidden family or the marked generators. A second implementation verified the 2,880 compatible generating pairs against the same observations and reconstructed the historical actions from pointwise permutations. [SmallGrp inventory and completeness](https://gap-packages.github.io/smallgrp/)

| Method using the short labels | Exact named actions with zero new reads | Correct primary answers with zero new reads |
| --- | ---: | ---: |
| Archived short-only transformer | — | 26/2,304 |
| Live square closure | 7/24 | 1,389/2,304 |
| Complete classical catalogue | 13/24 | 1,248/2,304 |

The catalogue missed its frozen 20/24 target. Each of the 11 unresolved cases had at least two unseen names. Swapping two such names preserves all observations and changes the rooted labeled action. The catalogue abstained on every held-out word in an unresolved case. Square closure could use partial tables there and returned more correct answers under this zero-new-read rule. The transformer count comes from the historical trained checkpoint scores.

### Recovering the missing names at the exact query floor

In the follow-up, all compatible pairs for each case gave the same rooted action up to renaming. The catalogue learner asked a shortest positive access word for each unseen state except one farthest from the root, then used the remaining roster name for that state. An exact learner must distinguish every remaining swap of two unseen names. Each newly named state requires a word at least as long as its distance from the root. The checker independently confirmed that the learner attained both lower bounds in every case.

| Order | Cases | Square closure additional labels (symbols) | Catalogue additional labels (symbols) |
| --- | ---: | ---: | ---: |
| 60 | 12 | 41 (391) | 36 (340) |
| 120 | 12 | 143 (1,347) | 129 (1,211) |
| **Total** | **24** | **184 (1,738)** | **165 (1,551)** |

Both arms recovered every named table against the same live in-process endpoint callback, and the catalogue reproduced all 5,376 held-out answers. It saved 10.33% of additional reads and 10.76% of additional symbols. Both arms also incurred the original 12,192 short labels and 86,016 symbols. Including the initial observations yields 12,376 versus 12,357 total labels.

Over two local runs, catalogue matching took 7.79–8.00 seconds of process CPU across the cohort. The GAP export separately took 2.22–2.30 seconds elapsed. After matching, structure deduplication and minimum-cost naming took 66.14–70.66 ms of CPU. The live square arm took 50.92–51.33 ms. The extra work saved queries but increased local computation. The comparison assumes a regular group action of known order and a known roster. It measures the additional value of those structural assumptions on this historical cohort. Physical latency and generalization remain unmeasured.

The first replay and the follow-up were frozen at `c7fb082` and `4f5e105`. Both DSR qualifications passed 2/2 checks, and the diagnostics suite passed 89 tests with one skipped. The independent audit verified 8,640 transition entries across the two full-label arms and checked the ambiguity and contradictory-observation controls. The archived neural ranker result used runtime comparisons and did not supply the retained labels to its generic ordering baselines. The same-label catalogue results provide the stronger conventional reference for this cohort. [Neural ranking comparison](../../probes/sizing/group-equality/README.md)

The naming check's console summary displays a null totals field because it looks for a nested key. The verified counters are top-level fields in `floor-verified.json` in the results archive.

[Catalogue source and protocol](catalogue-replay-source.zip) · [Naming follow-up source and protocol](catalogue-replay-floor-source.zip) · [Tapes, audits and DSR receipts](catalogue-replay-results.zip)

### Batched exact catalogue matching

The batched matcher evaluates the first 12 words for every ordered pair in a group together. It packs each word's earliest equivalent position into a four-bit field. The resulting key preserves the complete prefix partition. The remaining comparisons require repeated labels to reach their earlier state and new labels to reach unused states. A compatible pair whose words reach more than half the group must generate it. The matcher explicitly checks generation for the other pairs.

The naming step constructs one rooted action and evaluates its shortest access words in every candidate at once. It checks both marked edges under every resulting transport. Since the transports preserve the root and the candidates generate groups of equal order, the maps are bijections. If any edge disagrees, the learner computes the exact scalar set of rooted actions and checks ambiguity before paying for names.

The two executions used the same 24 historical cases and complete order-60 and order-120 tables, reversed arm order, and used hash seeds 0 and 991. They timed the original scalar code live in both runs.

| Live arm | Matching CPU | Naming or closure CPU | Additional reads (symbols) |
| --- | ---: | ---: | ---: |
| Scalar catalogue | 7.674–7.715 s | 59.95–61.04 ms | 165 (1,551) |
| Batched catalogue | 281.00–301.53 ms | 13.71–15.19 ms | 165 (1,551) |
| Square closure | — | 46.75–49.52 ms | 184 (1,738) |

Matching improved **25.59–27.31×** against the frozen 8× criterion. Matching plus naming took **294.70–316.72 ms** for the batched arm. Parsing the shared catalogue cost 22.93–23.00 ms outside the matching timers. The live square arm remained faster with the in-process oracle. Relative to square closure, the batched arm spent 247.95–267.20 ms more CPU for 19 fewer reads and 187 fewer input symbols. If input movement is free and CPU time translates directly to elapsed overhead, that corresponds to a prepared-catalogue break-even of 13.05–14.06 ms per read. The earlier GAP export and real device costs are outside that calculation.

All 2,880 compatible pairs and full named-action sets matched exactly. The separate audit checked 12,960 table entries across the three arms, every paid reply, and 10,752 held-out answers across the two catalogue arms. Both matchers reproduced the survivor sets for six other groups and rejected contradictory and subgroup-only profiles. The naming control refused two distinct rooted actions before any oracle call. DSR passed both checks, including 89 diagnostics tests with one skipped.

The producer is `2b20e3f`. Replay with the source archive extracted next to a checkout of that revision. Use `catalogue.json` from the original catalogue source, and extract `scored.json` and `floor-scored.json` from the original results as `prior.json` and `prior-floor.json`. Extract the registered cohort into the run directory. The frozen manifest checks these inputs before the archived DSR target runs. [Batched source and protocol](indexed-catalogue-source.zip) · [Timings, exact audit and DSR receipt](indexed-catalogue-results.zip)

### Matching the observed partial action

The 508 short labels determine four anchors at length two and a partial right action on the observed labels. Assign one shortest representative per label, beginning at the anchors. Each assignment uses its previously assigned prefix. Then require distinct image states for distinct labels and check every observed edge outside this assignment forest. Induction from the anchors recovers every short word, so these checks enforce the same full partition as the earlier matchers.

Before doing that transport, use the periods visible in the observed graph. Right multiplication by a fixed group element has the same cycle length at every state. Any closed cycle in the observed part gives that exact period, and a path through \(k\) distinct vertices requires a period of at least \(k\). This also holds for partial compositions of the two right actions. The matcher applies it to the two marks and the words 01, 001 and 011. Across 24 cases, 117 of the 120 period tests found exact orders. They reduced 8,683,200 case–pair combinations to 4,452 before checking the remaining action constraints.

| CPU across 24 historical cases | Live word-partition matcher | Partial-action matcher |
| --- | ---: | ---: |
| Catalogue matching | 290.19–300.16 ms | 58.15–65.60 ms |
| Matching plus unchanged naming | 304.16–316.19 ms | 73.60–79.79 ms |
| Live square closure in the same run | 45.61–53.17 ms | 46.18–53.53 ms |

Matching was **4.42–5.16× faster** against the frozen 2× criterion. Both catalogue arms found all 2,880 compatible pairs, reproduced the exact 165-read, 1,551-symbol tape and answered all 5,376 held-out prompts per arm. Square closure used 184 reads and 1,738 symbols. The independent catalogue recheck and endpoint audit verified 17,280 table entries across the two catalogue and two square executions. Hash-seed and arm-order changes reproduced all non-timing output.

The nine outside controls retained the same exact version spaces, including the cyclic case in which all five observed period constraints were lower bounds. On these controls matching rose from 199.08–200.45 ms to 212.02–221.20 ms. The improvement is specific to the cycle-rich historical cohort. Against square closure, the faster catalogue's remaining CPU overhead on that cohort was 20.07–33.61 ms. With a parsed catalogue, free movement and CPU time as an estimate of elapsed overhead, that is **1.06–1.77 ms per saved read**. Charging the one-time table parse gives a 2.29–2.97 ms range. None of these estimates supplies a real device measurement.

The producer is `9fb5b16`. DSR passed both checks, including 89 diagnostics tests with one skipped. Use the same `catalogue.json` and `prior.json` as above, and extract `scored.json` from the batched results as `prior-batched.json`. The source contains the frozen manifest and DSR configuration. [Partial-action source and protocol](partial-action-catalogue-source.zip) · [Raw timings, audit and receipt](partial-action-catalogue-results.zip)
