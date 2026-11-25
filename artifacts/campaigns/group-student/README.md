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

### Reusing complete labelled actions

When all but at most one roster name is observed, each surviving pair already gives a complete naming. The matcher forms the root and both named generator columns for all such pairs in integer arrays and deduplicates those rows before creating Python tables. The last unseen element and its label follow by elimination. When the resulting named-action set is a singleton, acquisition uses that table directly. This avoided a second pass over the witnesses in all 13 cases with complete naming.

| CPU across 24 historical cases | Live scalar table construction | Batched table construction |
| --- | ---: | ---: |
| Catalogue matching | 63.76–72.79 ms | 37.33–45.29 ms |
| Final naming | 13.31–14.89 ms | 7.71–8.38 ms |
| Matching plus naming | 77.08–87.67 ms | 45.71–53.01 ms |

The combined reduction was **31.23% and 47.86%**, above the frozen 20% criterion in both runs. Compared with live square closure in the candidate's run, the catalogue was **5.89 ms slower** in one execution and **4.88 ms faster** in the other. The shared catalogue parse cost 22.19–26.02 ms and is outside the table. Charging that parse to catalogue acquisition, the CPU-based break-even estimate is **0.91–1.68 ms per saved read** with free movement. The estimate uses the measured host overhead on these cases.

Both catalogue arms kept every compatible pair and used **165 reads and 1,551 symbols**, versus 184 reads and 1,738 symbols for square closure. The audit checked 2,880 compatible pairs, 17,280 transition entries and 10,752 held-out answers. The nine outside controls remained exact. Their matching CPU improved by 8.31 ms in one run and increased by 2.41 ms in the other. A separate three-element control preserved two roots with identical marked columns and completed one unseen name.

The producer is `4a28ebb`. Both DSR checks passed and the diagnostics suite passed 89 tests with one skipped. Reproduce with the source archive, the earlier `catalogue.json` and `prior.json`, and `scored.json` from the partial-action results saved as `prior-partial.json`. The source includes the unchanged incumbent and the frozen DSR configuration. [Source and protocol](batch-named-catalogue-source.zip) · [Exact tapes, timing and audit](batch-named-catalogue-results.zip)

### Acquiring the action from four paid anchors

Each live learner starts with the four paid endpoints for 00, 01, 10 and 11, a public order and the roster. The complete catalogue contains 376 rooted shapes of order 60 and 1,498 of order 120. Each paid reply updates a partial naming bijection for every surviving shape. The selector adapts the established small-group minimax rule to this larger catalogue. If \(f\) names remain unused, a known-label reply with \(m\) matching shapes retains \(m f!\) possible tables, and a particular unused label retains \(m(f-1)!\). The chooser minimizes the largest bucket over its shortlist. The two catalogue arms use an exhaustive union of anchored shortest access words or a shorter probe list. Both include a word that separates two consistent complete namings of the first two fitting shapes. Once the structure is unique, the learner uses shortest access words for unseen names and infers the last name.

| 24 historical cases | Total paid reads | Reads above the floor | Input symbols |
| --- | ---: | ---: | ---: |
| Four-anchor square closure | 2,287 | 151 | 13,223 |
| Full catalogue shortlist | 2,179 | 43 | 14,064 |
| Short catalogue shortlist | 2,177 | 41 | 13,780 |

Both frozen gates passed. The full shortlist used **82.37% fewer reads** than the archived 12,357-read workflow that began with 508 labels per case. Four-anchor square closure alone used 81.49% fewer reads than that archived workflow. Against the live control with the same four anchors, the full shortlist saved **108 reads**, reduced surplus above the naming floor by **71.52%**, and sent **841 more symbols**. Its maximum surplus was two reads per case. The short shortlist saved 110 reads and sent 557 more symbols than square closure. The six outside controls, whose groups and markings were fixed by the previous study, used 572, 544 and 542 reads and 3,755, 3,961 and 3,734 symbols respectively. Neither catalogue arm needed its cap fallback on the 30 cases. A separately forced cap used the identical live square tape and table.

Over the 24 historical cases, live square closure took 103–106 ms of CPU, the full shortlist took 4.01 seconds, and the short shortlist took 358–370 ms. The six outside cases took 24 ms, 1.20–1.26 seconds, and 116–125 ms respectively. Preparing the catalogue took 5.96–6.01 seconds of CPU and 6.20–6.34 seconds elapsed, including construction of the endpoint index. For a prepared catalogue, the short shortlist incurred 257–308 ms more elapsed time in the immediate-response harness than live square closure. Under a serial cost model with free input movement, break-even would be 2.34–2.80 ms per saved read. The extra symbols also need pricing on a real device.

The independent checker rebuilt the entire catalogue, verified the feasible shape count and bucket score for every structural probe, checked every paid reply and shortest-word naming suffix, and audited 16,200 table entries and 16,128 held-out observations across the three arms. Both arm orders and hash seeds reproduced all non-timing output. DSR passed both checks, including 89 diagnostics tests with one skipped.

The producer is `397fba3`. Extract the registered cohort and the earlier `catalogue.json`, then extract `scored.json` from the partial-action results as `prior-partial.json` and `verified.json` from the batched named results as `prior-cost.json`. The source archive includes the older catalogue helpers and the DSR replay configuration. [Frozen four-anchor source and protocol](four-anchor-source.zip) · [Paid tapes, complete audit and timings](four-anchor-results.zip)

### Reducing movement in the four-anchor acquisition

For each true action, take the shortest admissible word length to each state whose name was absent from the anchors. These distances come from a multi-source traversal with all anchor endpoints at distance two. The sum of the unseen distances minus the largest is a movement lower bound after the anchors. At least all but one name must be observed, and the roster can supply the last. Including the four anchor words gives a floor of **12,350 symbols** across the 24 historical cases.

The movement-credit policy applies the same bound inside each compatible shape. A newly observed state of distance \(d\) lowers the remaining bound by the lesser of \(d\) and the second-largest distance still unseen. This follows by subtracting the new sum and maximum from the old sum and maximum. Among probes with a worst-case bucket at most twice the best strict minimax value, the selector minimizes word length minus this credit averaged over compatible shapes. Those shapes have equal numbers of residual naming bijections. The length-first policy uses the shortest strictly informative probe, breaking ties by worst bucket and then lexicographic order. Both use the same shortlist as the live short policy and share the exact naming suffix after structure selection.

| Live policy, 24 historical cases | Reads | Input symbols | Symbols above floor | Online CPU |
| --- | ---: | ---: | ---: | ---: |
| Square closure | 2,287 | 13,223 | 873 | 104–105 ms |
| Short minimax catalogue | 2,177 | 13,780 | 1,430 | 388–404 ms |
| Movement credit | 2,175 | 13,148 | 798 | 797–833 ms |
| Length-first | 2,178 | 12,521 | 171 | 307–329 ms |

The preregistered credit policy reduced symbol surplus by **44.20%** against the 35% threshold and used two fewer reads than short minimax. The fixed length-first ablation reduced symbol surplus by **88.04%** at one extra read. Against live square closure it saved 109 reads, 392 zero symbols and 310 one symbols. It took 210–231 ms more elapsed time with a prepared catalogue, so the measured host overhead would be covered if \(109q+392m_0+310m_1\) exceeds that range, where \(q\) prices a reset/read and \(m_0,m_1\) price the two marks. Common catalogue and endpoint preparation took 6.26–6.27 seconds of CPU. Credit's distance preparation added 21–23 ms. Device pricing remains open.

On the six outside controls, square, short minimax, credit and length-first used respectively 572, 542, 544 and 554 reads, and 3,755, 3,734, 3,667 and 3,616 symbols. All recovered exactly. Credit cost 259–269 ms of outside CPU, length-first 107–127 ms, short minimax 115–132 ms and square 25–28 ms. No catalogue needed its cap in the ordinary cohorts. The forced cap recovered with the original live square tape.

The independent checker reconstructed the full catalogue, all candidate choices among the complete shortlists, both cost floors, every paid reply and the exact naming suffixes. It audited 21,600 table entries and 21,504 held-out answers across four arms. Both hashes reproduced the same non-timing data, and the live incumbents reproduced their archived traces. The producer is `f8e486e`. DSR passed both checks, including 89 diagnostics tests with one skipped. Reproduce with the same cohort, catalogue and `prior-partial.json`. Extract the scored tape from the four-anchor results as `prior-four.json`. [Frozen movement source and protocol](movement-credit-source.zip) · [Raw runs, audit and DSR receipt](movement-credit-results.zip)

### Reducing cold catalogue construction

For nonabelian groups, choose the least first mark in each inner conjugacy class. Then choose the least second mark under the subgroup fixing the first mark. This gives one representative per simultaneous-conjugation orbit. Inner conjugation preserves the rooted marked action and whether the pair generates, so only the representative needs traversal. Commuting pairs generate abelian subgroups and can be skipped in these tables.

For abelian groups, each marked action corresponds to the kernel lattice of the two commuting generators. Enumerating the lattices generated by \((w,0)\) and \((s,h)\) with \(wh=n\) and \(0\le s<w\) constructs the complete rooted abelian part directly. This uses the existing lattice-action constructor and avoids pair traversals in the five abelian tables.

| Live builder | Table traversals | Direct lattice constructions | Shape build CPU |
| --- | ---: | ---: | ---: |
| Exhaustive | 723,600 | 0 | 5.71–5.94 s |
| Conjugacy with exhaustive abelian enumeration | 77,242 | 0 | 734–751 ms |
| Conjugacy and direct lattices | 26,842 | 528 | 326–367 ms |

The combined builder was **15.58–18.20× faster** than the live exhaustive builder, passing the frozen 3× criterion in both arm orders. Including a shared parse charged to each pipeline, fresh endpoint indexing and all 30 live acquisitions, the measured stages totaled **0.89–0.94 s CPU** versus **6.30–6.52 s**. The comparison starts from the exported multiplication tables. It charges local construction and excludes production of those tables.

Every builder produced the same 376 order-60 shapes and 1,498 order-120 shapes. The independent checker expanded every noncommuting orbit, accounting for 478,140 noncommuting pairs and 195,060 commuting pairs in nonabelian groups. It recovered 168 and 360 lattice coordinates and rejected an omitted orbit. Both cold pipelines reproduced the prior length-first tapes on all 30 actions, including 2,178 reads and 12,521 symbols on the historical cases. The audit checked 10,800 transition entries and 10,752 held-out answers across the two complete pipelines. Both hash seeds reproduced the same non-timing output.

The producer is `c2d591b`. DSR passed both checks, including 89 diagnostics tests with one skipped. Replay with the same `catalogue.json`, `prior-partial.json` and registered cohort, plus the scored tape from the movement results as `prior-movement.json`. The archived source includes the frozen manifest and DSR configuration. [Frozen cold-build source and protocol](cold-catalogue-source.zip) · [Complete timings, tapes and audit](cold-catalogue-results.zip)

### Complete order-60 and order-120 acquisition census

The two live arms used every distinct rooted marked action in the complete catalogue, once per shape, and paid the same four anchors. The replay changed every roster permutation, reversed per-case arm order and used a second hash seed. After decoding names, the paid words, replies and structural traces agreed on all 1,874 actions.

| Order | Marked shapes | Square reads | Length-first reads | Square symbols | Length-first symbols |
| --- | ---: | ---: | ---: | ---: | ---: |
| 60 | 376 | 24,529 | 23,121 | 169,705 | 160,824 |
| 120 | 1,498 | 188,738 | 182,741 | 1,610,769 | 1,588,412 |
| **Total** | **1,874** | **213,267** | **205,862** | **1,780,474** | **1,749,236** |

Both orders passed the frozen requirement to reduce total reads and total symbols. Length-first paid no more reads on any shape. Compared in reads and total symbols, it improved one or both costs without increasing the other on **1,743** shapes, gave mixed results on **130**, and lost to square closure on **one**. That action used 123 reads in both arms and 964 symbols with length-first versus 947 with square closure. Comparing reads and the two marks separately gave 1,674 length-first wins and 200 mixed results. Excluding the prior 30 acquisitions leaves 1,713 wins, 130 mixed results and the same one loss in the two-cost comparison. Neither run reached the cap.

The largest movement increase was on order-120 abelian slot 277. Length-first used 121 reads and 2,098 symbols versus 125 and 1,266 for square closure. Across all 360 abelian order-120 cases, length-first saved 1,299 reads and used 15,629 more symbols. The four-anchor partition captures many of these cases because equal endpoints for 01 and 10 certify that the generating marks commute. The audit also reports each anchor partition and both primitive totals.

Over both runs, online CPU after the anchors totaled 44.54–45.60 seconds for length-first versus 10.17–10.30 seconds for live square. Shared cold construction and indexing took 0.53–0.57 seconds of CPU. On the whole census length-first saved 7,405 reads and 79,010 zero symbols while using 47,772 more one symbols. In a serial prepared cost model the measured elapsed overhead is covered when \(7405q+79010m_0-47772m_1\) exceeds 35.90–37.14 seconds. Here \(q\) prices a reset/read, and \(m_0,m_1\) price the marks. The census weights each distinct shape once. Physical action frequencies and device costs remain unmeasured.

The separate checker independently rebuilt the complete catalogue, replayed every paid oracle reply and audited 1,618,560 transition entries across the two arms and naming schemes. It checked every structural handoff and shortest naming suffix in the primary run. The producer is `e27ec06`. DSR passed both checks, including 89 diagnostics tests with one skipped. Reproduce using the pinned `catalogue.json` and the scored movement tape as `prior-movement.json`. [Census source and protocol](acquisition-census-source.zip) · [Full census, audit and receipt](acquisition-census-results.zip)

### Scoring only the shortest informative tier

Length-first acquisition now scores the shortest informative word tier in one batch. The search for the disagreement word between the first two completed candidates stops at that length when a fixed candidate already qualifies. A longer word cannot win under length-first ordering. After a unique shape remains, the learner assigns names directly in shortest-access order. It rejects a reused name and still infers the last one from the roster.

| Live acquisition over 1,874 shapes | Online CPU | Online elapsed | Paid reads | Input symbols |
| --- | ---: | ---: | ---: | ---: |
| Square closure | 9.61–10.10 s | 9.89–10.32 s | 213,267 | 1,780,474 |
| Released length-first | 41.04–43.84 s | 41.95–44.73 s | 205,862 | 1,749,236 |
| Shortest-tier length-first | 7.37–7.89 s | 7.55–8.07 s | 205,862 | 1,749,236 |

Online CPU improved by **5.56–5.57×** against the released learner and fell **21.8–23.3%** against live square closure. Building the catalogue and endpoint index from the fixed exported tables took 0.53–0.55 seconds of CPU. Tier preparation added 57–64 ms. Charging all three stages, the optimized learner took **7.96–8.51 seconds of CPU and 8.29–8.69 seconds elapsed** in total. It still finished 1.60–1.63 seconds sooner than square closure on this host. The shared anchor observations are charged in the paid totals. GAP table export and physical command, reset and read times are outside these timings.

The audit checked **2,427,840 transition entries** across the three live arms and two name permutations. The optimized learner reproduced every paid word, reply, selected score and exact table from released length-first in both runs. The checker independently recomputed all 113,068 structural scores in the primary run and verified every uniqueness boundary and shortest naming suffix there. Both runs reproduced the forced square fallback and rejected the duplicate-name control. The optimized trace logs the number of words it scores. The released trace retains the full-shortlist minimax statistic, which length-first does not use to choose a word.

On the equally weighted census, the optimized learner saves **7,405 reads and 31,238 total symbols** against square closure. That is 79,010 fewer `0` symbols and 47,772 more `1` symbols. The order-120 abelian subset still uses 15,629 more symbols. These are finite-census software measurements. Device timings and real action frequencies remain open.

The producer is `ed9cda5`. Both DSR checks passed, including 89 diagnostics tests with one skipped. To replay, unpack the source into `run/` beside a clean `work/` checkout at the producer revision. Extract `catalogue.json` from the catalogue replay source and `scored.json` from the complete census results as `prior-census.json`. Set `MGR_TIER_WORK`, `MGR_TIER_RUN` and `MGR_TIER_SHA` for the archived DSR configuration. [Frozen source and protocol](tiered-acquisition-source.zip) · [Raw tapes, timings, audit and DSR receipt](tiered-acquisition-results.zip)

### Transfer to orders 32, 64 and 128

The frozen shortest-tier learner saved reads and symbols against live square closure at **all three held-out orders**. The joint preregistered prediction failed because CPU savings held only at order 128. Orders 32 and 64 were already slower with a prepared catalogue. Each row covers the full set of distinct rooted marked regular actions of that order, counted once, with the same four paid observations supplied to each arm. [Prior inventory and completeness checks](../../probes/sizing/group-solver-baseline/anchor-report.md) · [SmallGrp 1.7.0](https://gap-packages.github.io/smallgrp/)

| Order | Shapes | Reads saved | Symbols saved | Square CPU | Tiered online CPU | Tiered CPU incl. setup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 151 | 451 | 2,766 | 0.107–0.113 s | 0.121–0.131 s | 0.133–0.146 s |
| 64 | 503 | 1,752 | 12,458 | 1.063–1.064 s | 1.124–1.130 s | 1.241–1.250 s |
| 128 | 1,799 | 8,462 | 53,244 | 12.05–12.34 s | 10.07–10.41 s | 11.63–12.01 s |

At order 128, the learner saved **1.92–1.97 seconds of online CPU** and **0.33–0.42 seconds of CPU with setup charged** across the census. Setup cost 1.56–1.59 seconds of CPU once per order. The measured elapsed savings with setup were 0.35–0.57 seconds across 1,799 actions on this host. The timing includes parsing the fixed exported tables, rebuilding the catalogue, indexing endpoints and preparing tiers. The square arm uses the four observations and the public roster. GAP export and physical action costs are outside these timings.

Reads fell on 2,450 shapes and tied on three. Symbols fell on 2,368 shapes, tied on two and increased on 83. At order 128 the aggregate reduction was **59,493 fewer `0` symbols and 6,249 more `1` symbols**. The 255 abelian shapes at that order saved 976 reads and used 695 more symbols. Their movement mix accounts for the aggregate increase in `1` commands. Device qualification would need to price the two commands independently and account for the action mix and how often the catalogue is reused.

The verifier matched the independently enumerated prior inventory, replayed every paid endpoint and checked **2,138,368 transition entries** across both arms and relabelings. It recomputed all **175,585 selected structural scores** in the primary run and rejected a forged score. Both runs passed the forced fallback and invalid-name controls. DSR passed both checks with 89 diagnostics tests passed and one skipped. The producer is `a6a9fe8`.

To replay, unpack the source into `run/` beside a clean `work/` checkout at the producer revision. At `run/`, extract `group-tables.json` from the prior [anchor source](../../probes/sizing/group-solver-baseline/anchor-source.zip) and extract `scored.json` from the [anchor results](../../probes/sizing/group-solver-baseline/anchor-results.zip) as `prior-anchor-shapes.json`. Create the exact per-order input streams and set `MGR_TRANSFER_WORK`, `MGR_TRANSFER_RUN` and `MGR_TRANSFER_SHA` for the archived DSR configuration.

```sh
set -C
for order in 32 64 128; do
  jq -c --argjson order "$order" \
    '[.[0],.[1], (.[3]|map(select(.[0]==$order) | [.[0],.[1],.[4]]))]' \
    group-tables.json > "tables-$order.json"
done
```

[Frozen source and protocol](transfer-source.zip) · [Raw runs, complete audit and DSR receipts](transfer-results.zip)
