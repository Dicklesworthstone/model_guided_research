# Exact group acquisition from equality queries

The learner receives two generator names, the number of reachable states, and an oracle that answers whether two positive words denote the same element. We ran two experiments on disjoint panels of rooted, marked actions. The first tested whether locally certified rewrite rules improve search. The second tested whether an untrusted catalog can suggest an action and still preserve exact recovery outside the catalog.

## Streaming shortlex rewriting

We froze six unseen `S5`, six `A6`, and six `Z60` actions before any comparison. Both learners construct the same canonical action by breadth-first traversal. The rewrite arm adds equations only after establishing an edge. It orients them toward shorter words or, at equal length, lexicographically smaller words. It then uses these rules to resolve comparisons or find shorter keys for the equality cache.

| Family | Cancellation checks | With rewriting | Checks on new-state edges, each arm |
| --- | ---: | ---: | ---: |
| `S5` | 10,785 | 9,219 | 7,700 |
| `A6` | 76,132 | 60,179 | 52,197 |
| **Primary total** | **86,917** | **69,398** | **59,897** |
| `Z60` control | 766 | 670 | 634 |

The primary gate required 15% fewer calls overall and 10% fewer on new-state edges. Rewriting saved 20.16% overall and 0% on new-state edges. On transitions to known states, the query count fell from 27,020 to 9,501. A substring of a globally shortlex-minimal representative is itself minimal because replacing it with any smaller equivalent substring would decrease the whole word. This explains why these rules cannot directly collapse a new representative. The identical costs for ruling out all prior states are an observation on this panel.

The independent auditor checked 12,960 edges across both arms, replayed each query and substitution witness, checked 6,912 held-out word evaluations, and rejected corrupted oracle and substitution records. In a separate control given the correct `S5` family, an exhaustive library of 57 canonical marked actions let a deterministic splitter identify the six presentations in 7, 9, 9, 5, 7, and 6 checks.

## Checking untrusted catalog proposals

We froze a new panel of 18 order-120 actions. The sampler initially requested six distinct dihedral actions. There are three classes of ordered generating pairs up to automorphism, coming from the two orders of a rotation and a reflection, and from two reflections. Before any oracle outcomes, we changed the six dihedral slots to three dihedral and three `A5 × C2` slots. The amended panel kept six `S5` cases and 12 outside the library. In each case the method received the same cardinality and the same 57-action `S5` library.

The splitter proposes a table. The oracle checks its edge equations, reusing identical comparisons from the splitter. At the first failed equation the algorithm discards the proposal and runs cancellation from scratch. If all equations pass, evaluating each representative in the hidden action gives a root-preserving map that commutes with both transitions. Every hidden state is reachable, so the map is surjective. Both actions have 120 states, so the map is bijective. The representative tree supplies 119 identities among the 240 edges, leaving at most 121 oracle equations. Correctness uses the exact oracle, reachability, and the known cardinality.

| Family | Certified | Cancellation | Proposal selection | New certificate checks | Fallback | Total checks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `S5` | 6/6 | 7,992 | 37 | 717 | 0 | 754 |
| `C120` | 0/6 | 1,617 | 36 | 6 | 1,617 | 1,659 |
| Dihedral order 120 | 0/3 | 607 | 16 | 8 | 607 | 631 |
| `A5 × C2` | 0/3 | 5,163 | 18 | 12 | 5,163 | 5,193 |
| **Total** | **6/18** | **15,379** | **107** | **743** | **7,387** | **8,237** |

The preregistered gate passed. All six fresh `S5` cases certified with 90.57% fewer queries on aggregate, and all 12 other cases recovered their exact action after rejection. They incurred 96 extra calls, or 1.30% above cancellation. Query savings over this particular panel were 46.44%, which depends on the share and difficulty of the catalog hits. The independent auditor reconstructed the library and all hidden actions using explicit permutations, replayed the comparisons and certificates, checked all 4,320 final edges and 3,456 held-out words, and detected corrupted acceptance and rejection evidence.

The oracle here is an in-process table walk. On macOS, cold catalog enumeration took 0.785 s. Over the 18 cases, cancellation took 0.060 s including 0.007 s in the oracle. The checked proposal with fallback took 0.198 s after enumeration, including 0.004 s in the oracle. Query savings therefore describe a resource tradeoff for an expensive oracle. This experiment measures a static library on a fixed panel, with no external latency benchmark.

## Information-optimal selection with a known S5 family

The existing congruence command can compile a decision tree using exact word equalities across the complete 57-action S5 catalogue.

```sh
uv run --locked python -m scripts.group_student_probe congruence \
  --data . \
  --coverage artifacts/probes/sizing/group-congruence/coverage.json \
  --known-family-policy optimal \
  --output "$PANEL_DIR/optimal-family.json"
```

| Policy | Targets identified | Maximum equality calls | Calls across all targets | Queried symbols across all targets |
| --- | ---: | ---: | ---: | ---: |
| Frozen greedy | 57/57 | 9 | 376 | 3,971 |
| Compiled tree | 57/57 | 6 | 335 | 2,656 |

The compiled tree attains both binary information lower bounds. Distinguishing 57 targets requires at least six calls in the worst case. The minimum total leaf depth is 335 under equal target weights, attained by seven five-call and 50 six-call paths. Each split uses a legal endpoint equality, and the auditor independently replayed every split against the live set of actions. The new policy used fewer calls on 29 targets, the same on 18, and more on ten. Its symbol savings were measured. Global symbol optimality was outside this experiment. [Binary code lower bounds](https://web.stanford.edu/class/ee398a/handouts/papers/Huffman%20-%20Min%20Redundancy%20Codes%20-%20IRE52.pdf)

The catalogue builder now visits one representative per simultaneous-conjugation orbit. It preserves the original sorted graph dictionary exactly. In two live paired runs, the old builder took 875–889 ms of CPU and the orbit builder took about 20 ms. Tree compilation added 58–62 ms. The repeated online phase took 632–645 ms of CPU for greedy and under 1 ms for the compiled policy across all 57 targets on this Mac.

The ordinary CLI reproduced all 18 original non-timing rows. With the compiled policy, the six S5 rows went from 43 to 36 calls combined. All other arms reproduced their non-timing fields. The full catalogue replay matched at a second Python hash seed, and the auditor rejected a flipped answer. DSR passed 2/2 checks, including 89 diagnostics tests passed and one skipped. The compiled route assumes known S5 membership. The separate untrusted-catalogue experiment retains its certificate and fallback.

The scored source was frozen at `9d9c2c8`. [Frozen comparison](known-family-source.zip) · [Tapes, CLI runs, and DSR receipts](known-family-results.zip)

## Short certificates for untrusted proposals

The `proposal` command compares the two certificate methods and cancellation alone on the same regular group actions of order 120.

```sh
uv run --locked python -m scripts.group_student_probe proposal \
  --data . \
  --coverage artifacts/probes/sizing/group-congruence/short-certificate-coverage.json \
  --output "$PANEL_DIR/short-certificates.json"
```

Both certificate arms use the same selector, which takes at most six calls. The short arm verifies Moore's five S5 relators and two equations that bind the proposed presentation to the original marks. Once the equations hold, the resulting homomorphism covers both hidden generators. The known order makes it an isomorphism. The benchmark rejects a failed certificate at its first false answer and runs the same exact cancellation fallback. [Moore presentation in Bray, Conder, Leedham-Green and O'Brien](https://www.math.auckland.ac.nz/~conder/preprints/SymGpPresentations.pdf)

| Six cases per family | Cancellation calls | Edge certificate calls | Short certificate calls |
| --- | ---: | ---: | ---: |
| S5 | 9,095 | 755 | 70 |
| C120 | 1,543 | 1,588 | 1,585 |
| A5 × C2 | 7,561 | 7,623 | 7,607 |
| S3 × C20 | 3,032 | 3,074 | 3,074 |

On the six S5 cases, the short arm paid 10–13 calls including selection. It reduced calls by 90.7% and queried symbols from 10,125 to 1,605. Both certificate arms accepted all six S5 cases. They rejected all 18 other actions and recovered each through fallback. The short path paid seven to nine extra calls on rejection, totaling 130 beyond cancellation's 12,136. On S3 × C20 it queried 14 more symbols than the edge arm. All costs charge both queried words and the entire fallback.

In the scored run, shared catalogue, selector and edge-table setup took 90.1 ms of CPU. Compiling the short certificates added 56.7 ms. Total online CPU for the new path was 68.3 ms versus 62.4 ms for the edge path across the 24 table oracles. This comparison measures saved oracle traffic and keeps the higher measured host overhead visible.

The auditor reconstructed 17,280 transitions across the three arms. It exercised every compiled certificate on its own S5 marking, rejected all 3,192 wrong S5 pairings and all 1,026 out-of-family pairings, and rejected a flipped paid answer. Both normal CLI runs preserved every non-timing field. The original congruence command also reproduced its 18 prior rows. DSR passed 2/2 checks, with 89 diagnostics tests passed and one skipped. The six S5 cases were fresh for certification. The prior known-family selector census had already exercised every S5 marking.

The scored source was frozen at `75840b7`. [Frozen protocol](short-certificate-source.zip) · [Coverage](short-certificate-coverage.json) · [Oracle tapes, audit and receipts](short-certificate-results.zip)

## Direct recovery after a paid commutation check

The proposal panel can run a fourth arm that first asks whether `01` and `10` have the same endpoint. In a regular action, a positive answer means the two generators commute. The learner finds the order \(w\) of the first generator by testing divisors of 120. The quotient by the first generator's cyclic subgroup has order \(h = 120/w\) and is generated by the second mark's image. The learner then finds the unique residue \(s\) modulo \(w\) with \(a^s b^h = 1\) by testing prime-power digits. The recovered relations \((w,0)\) and \((s,h)\) span a lattice of index \(wh = 120\), so they generate the entire kernel. The direct learner constructs the marked graph from this lattice without a catalogue. On a negative commutation reply, it continues through the short S5 certificate and its exact fallback.

```sh
uv run --locked python -m scripts.group_student_probe proposal \
  --data . \
  --coverage artifacts/probes/sizing/group-congruence/abelian-route-coverage.json \
  --abelian-route \
  --output "$PANEL_DIR/abelian-route.json"
```

The frozen panel used six distinct acquisition cases per family. Each entry shows paid calls, with literal symbols in parentheses. Every comparison charges both operands, including commutation and fallback.

| Family | Cancellation | Edge certificate | Short certificate | Routed |
| --- | ---: | ---: | ---: | ---: |
| C120 | 1,695 (22,684) | 1,737 (22,977) | 1,737 (22,992) | 62 (1,431) |
| C60 × C2 | 1,263 (17,552) | 1,304 (17,842) | 1,304 (17,852) | 60 (1,326) |
| S5 | 8,125 (99,889) | 755 (10,705) | 69 (1,683) | 75 (1,707) |
| A5 × C2 | 9,863 (111,503) | 9,913 (111,891) | 9,911 (111,912) | 9,917 (111,936) |

The 12 commuting cases took 8–12 calls each. They used 122 calls versus 3,041 for the live short-certificate method, a 95.99% reduction against an 80% gate. Symbols fell from 40,844 to 2,757, a 93.25% reduction. Each of the 12 noncommuting cases paid exactly one extra call and four extra symbols before reproducing the short arm's entire paid continuation. Across all 24 cases, calls fell from 13,021 to 10,114.

Shared catalogue, selector and edge setup cost 78.2–94.1 ms of CPU across two hash-seed runs. Short-certificate preparation added 40.4–46.8 ms. Online CPU across the commuting cases was 0.78–0.79 ms for the routed arm and 24.3–24.7 ms for the short arm. Over the entire panel it was 36.7–41.6 ms for the routed arm and 64.6–67.7 ms for the short arm. These are observations from in-process table oracles on one Mac. The comparison uses the same setup and gives a classical equality-only baseline for future model-guided methods. Classical generic abelian-group algorithms address a related problem with different oracle operations and cost units. [Sutherland, Structure computation and discrete logarithms in finite abelian p-groups](https://arxiv.org/abs/0809.3413)

The independent checker replayed the paid answers and compared 23,040 output transitions with separately constructed actions. It verified that the paid tapes uniquely identify the correct lattice among all 360 possible abelian markings. In an exhaustive run of the direct learner across those markings, the maximum was 13 calls and 449 symbols, including commutation. The two binding equations rejected all 129,240 wrong lattice pairings, and the audit rejected a forged commutation reply. All non-timing fields matched across hash seeds. The default CLI reproduced all three original arms. DSR passed 2/2 checks, including 89 diagnostics tests passed and one skipped.

The producer was frozen at `db9ede5` before scoring. [Frozen protocol and auditor](abelian-route-source.zip) · [Coverage](abelian-route-coverage.json) · [Oracle tapes, audit and receipts](abelian-route-results.zip)

## Reproduction

The [first coverage manifest](coverage.json) was committed at `3d8e745` and the matched learner ran at `9696164`. The [second manifest](proposal-coverage.json) and [frozen producer and auditor](proposal-source.zip) were committed at `5becc54`. The [verification archive](verification.zip) includes both complete transcripts on each platform, the prior synthetic cohort used to check exclusions, audit source, and the DSR commands and receipts. The summaries expose the per-case counts for [rewriting](summary.json) and [checked proposals](proposal-summary.json).

macOS and Linux emitted byte-identical transcripts after removing only the timing fields and sorting JSON object keys. The Linux regression run reported 927 passed and 16 skipped with `tests/test_runner.py` and `tests/test_runner_enhanced.py` outside that invocation. The macOS diagnostics suite reported 72 passed and one skipped. The archive retains the failed original coverage preflight and a Linux attempt that referenced a nonexistent environment before the successful replay used the qualified Python 3.13 environment.
