# Exact group acquisition from equality queries

The learner receives two generator names, the number of reachable states, and an oracle that answers whether two positive words denote the same element. These experiments test local rewriting, exact certificates for untrusted catalog proposals, and direct recovery from verified group structure. Each comparison specifies which marked actions were fresh and which served as replay controls.

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

## Alternating-product proposals after S5 rejection

The `--alternating-route` flag adds a fifth arm. After a negative commutation reply and a failed short S5 certificate, it selects a proposed \(A_5\times C_2\) marking by paid equalities. It tests the \((2,3,5)\) triangle relations for two auxiliary words, the order-two and commutation equations for a central word, and two equations binding the original marks to those three words. The binding equations make the resulting homomorphism onto. The known order 120 makes it bijective. After cancellation and removal of duplicate or tautological equations, each of the 57 compiled certificates needed four to seven equality checks. The first false reply triggers the existing cancellation fallback. [Bray's presentations of A5 and A5 × 2](https://webspace.maths.qmul.ac.uk/j.n.bray/web/Pres/A5.html)

```sh
uv run --locked python -m scripts.group_student_probe proposal \
  --data . \
  --coverage artifacts/probes/sizing/group-congruence/alternating-route-coverage.json \
  --alternating-route \
  --output "$PANEL_DIR/alternating-route.json"
```

**Seven calls are necessary and sufficient in the worst case when the family is known.** The six-call binary information bound cannot be attained with equality queries here. The 57 actions form 19 triples, one per marked A5 projection, with the three nonzero central bit assignments. A question with equal letter-count parity on both sides preserves each intact triple. With different parity, at most one of its three lifts can answer yes. A six-call tree must start with a 27/30 split of intact triples, since a parity-changing query leaves at least 38 actions on the no branch. To fit the 30 candidates into five remaining calls, the next split must be 15/15 and preserve the triples. At either 15-candidate node, a parity-preserving split has both sizes divisible by three and therefore cannot fit both children within eight. A parity-changing query leaves at least ten on the no branch. The compiled tree identified 13 markings in five calls, 32 in six, and 12 in seven. This bound concerns selection given the family. The measured acquisition arms pay for dispatch and certification as well.

| Family | Cases | Prior routed calls (symbols) | Extended calls (symbols) |
| --- | ---: | ---: | ---: |
| A5 × C2 | 6 fresh | 8,814 (104,518) | 119 (1,207) |
| S3 × C20 | 6 fresh | 3,481 (37,634) | 3,525 (37,946) |
| C120 | 4 replayed | 40 (919) | 40 (919) |
| S5 | 4 replayed | 49 (832) | 49 (832) |

Calls on the fresh target cases fell 98.65%, clearing the frozen 90% gate. The outside cases each paid seven or eight extra calls and then produced the identical cancellation tape. The eight replay cases reproduced their complete prior routed outputs after removing timings. Across the whole matched panel, the extension saved 8,651 calls and 102,999 symbols.

Across two runs on one Mac, the new catalogue, selector, and certificate compilation cost 126.8–131.3 ms of CPU beyond the shared setup. Summed online CPU on the complete panel was 19.8–22.8 ms for the extension and 54.2–61.5 ms for the incumbent. Including the incremental cold setup, this panel cost about 90–95 ms more CPU. The included oracle walks in-process tables. The measured query savings establish a stronger classical baseline for model-guided acquisition, with external latency and amortization unmeasured.

The separate checker independently enumerated the 57 marked tables and all 19 triples, replayed all 57 selector paths, and rejected all 7,011 wrong certificate pairings against the two full catalogues and ten negative controls. It audited all 24,000 output transitions and every paid reply in the five-arm panel, rejected a flipped reply, and checked the identical non-timing fields across two hash seeds and the ordinary incumbent CLI. DSR passed 2/2 main checks, including 89 diagnostics tests passed and one skipped.

Source and coverage were frozen at `02e1a0a`. The protocol records the failed six-call feasibility check and the seven-call successor, with the same 14-call rejection limit. [Frozen source and protocol](alternating-route-source.zip) · [Coverage](alternating-route-coverage.json) · [Tapes, audit, and receipts](alternating-route-results.zip)

## Reusing the compiled alternating policy

Pass `--alternating-policy` to use the included policy. The loader reconstructs all 57 graphs from the supplied permutations, regenerates and checks the presentation equations, and checks the complete selector against the reconstructed graphs before making an oracle call. The artifact contains the markings, witness words, and split tree. The CLI rejects an invalid bundle before writing a benchmark result. [Ready-to-load policy](compiled-policy-a5xc2.json)

```sh
uv run --locked python -m scripts.group_student_probe proposal \
  --data . \
  --coverage artifacts/probes/sizing/group-congruence/alternating-route-coverage.json \
  --alternating-policy artifacts/probes/sizing/group-congruence/compiled-policy-a5xc2.json \
  --output "$PANEL_DIR/reused-policy-run.json"
```

To generate a policy at a new path, run `uv run --locked python -m scripts.group_student_probe compile-alternating-policy --data . --output "$PANEL_DIR/a5xc2-policy.json"`. The command refuses to overwrite an existing file. `--alternating-route` still compiles the policy in the benchmark process.

We paired these paths seven times in one process on the same 20-case regression panel, alternating order. Median alternate setup CPU fell from **130.34 ms to 15.09 ms**, an 88.43% reduction against the frozen 60% gate. One-time compilation and export took 129.52 ms of CPU and produced a 15,340-byte policy. With the saved artifact, the extended path used 19.84 ms less variable CPU at the median than the live routed arm, including validation and both arms' online work. The saved path was faster in all seven pairs on that measure. Shared S5 setup was common to both arms. This is a per-invocation CPU measurement on one Mac with in-process table oracles.

All 14 runs reproduced the original five arms' non-timing fields. The loaded extension still paid 3,733 calls and 40,904 symbols versus 12,384 calls and 143,903 symbols for the incumbent. The checker independently reconstructed all 57 marked actions and replayed every selector path. It checked 24,000 output transitions and all paid replies in a separate CLI run, reproduced the default three-arm invocation, confirmed byte-identical policy exports across hash seeds, and rejected six invalid or corrupted inputs and an overwrite attempt. DSR passed 2/2 checks, with 89 diagnostics tests passed and one skipped.

The producer was frozen at `464c6d0`. [Source and preregistration](compiled-policy-source.zip) · [Paired runs, audit, and receipts](compiled-policy-results.zip)

## Direct recovery from an order-three commutator

The `--commutator-route` flag adds a sixth arm after the two catalogue certificates reject. The learner has already paid for a negative commutation reply. It finds the two generator orders and uses them to write \(c=aba^{-1}b^{-1}\) as a positive word. It checks that \(c^3=1\) and that each mark conjugates \(c\) to either \(c\) or \(c^{-1}\). The negative commutation reply makes \(c\) nontrivial. These checks establish that \(\langle c\rangle\) is normal and that the quotient is abelian, so \(\langle c\rangle\) is the derived subgroup. Conversely, in a noncommuting two-generator group with derived subgroup of order three, the same commutator generates the subgroup and passes the checks.

The existing abelian decoder gets equality in this quotient by comparing words up to each of the three powers of \(c\). It recovers the quotient lattice \((w,s,h)\) at order \(n/3\). Two lifts, \(a^w\) and \(a^s b^h\), are powers of \(c\). Together with the conjugations and \(ab=cba\), these equations give the two transitions on the unique forms \(c^k a^x b^y\) with \(k<3\), \(x<w\), and \(y<h\). The learner canonicalizes the result by breadth-first traversal. Each call to the underlying equality oracle charges both positive words. Power and conjugation relations are standard tools for collecting normal forms in polycyclic groups. [GAP reference on power-conjugate presentations](https://docs.gap-system.org/doc/ref/chap46_mj.html)

```sh
uv run --locked python -m scripts.group_student_probe proposal \
  --data . \
  --coverage artifacts/probes/sizing/group-congruence/commutator-route-coverage.json \
  --alternating-policy artifacts/probes/sizing/group-congruence/compiled-policy-a5xc2.json \
  --commutator-route \
  --output "$PANEL_DIR/commutator-route.json"
```

| Family | Cases | Prior extended calls (symbols) | Commutator arm calls (symbols) |
| --- | ---: | ---: | ---: |
| S3 × C20 | 6 fresh | 3,006 (29,696) | 267 (10,300) |
| C3 ⋊ C40 | 6 fresh | 3,604 (38,019) | 268 (14,864) |
| S4 × C5 | 6 fresh | 7,061 (79,260) | 7,137 (82,036) |
| C120 | 2 replayed | 18 (382) | 18 (382) |
| S5 | 2 replayed | 23 (248) | 23 (248) |
| A5 × C2 | 2 replayed | 44 (409) | 44 (409) |

Calls on the 12 targets fell 91.91%, clearing the frozen 80% gate. Queried symbols fell 62.84%. After the shared prefix, direct recovery took 19–36 branch calls per target. All six S4 × C5 cases had a commutator of order three and failed the normalization check. They paid 11–14 additional calls each and reproduced the entire original fallback tape. Over all 24 cases the arm saved 5,999 calls and 39,775 symbols.

Summed online CPU on the target cases was 2.04–2.16 ms for the new arm and 36.11–36.37 ms for the incumbent across two local runs. On the outside cases the measurements were 24.71–27.07 ms and 78.99–93.30 ms respectively, despite identical cancellation tapes after the extra checks. Shared S5 setup took 72.6–74.2 ms, short-certificate compilation took 37.0–50.2 ms, and the shared alternating-policy load and validation took 11.4–11.7 ms. These are process CPU observations on one Mac with in-process table oracles. The test did not isolate causes of timing variation on the identical fallback path.

The separate checker independently reconstructed 34,560 output transitions, replayed the paid tapes, verified the lift relations and normal-form bijections, and rejected a forged lift. It checked exact recovery on all 972 ordered generating pairs across four groups of orders 12 and 27, including groups with a central commutator. All 96 generating pairs of the order-12 A4 control rejected. Both hash seeds reproduced the same non-timing output, and removing the new flag reproduced the existing five arms. DSR passed 2/2 checks, with 89 diagnostics tests passed and one skipped.

The producer and manifest were frozen at `12774ef`. [Frozen source and protocol](commutator-route-source.zip) · [Coverage](commutator-route-coverage.json) · [Tapes, audit, and receipts](commutator-route-results.zip)

## Lifting a certified Klein-four quotient

The `--klein-route` flag adds an arm after the paid commutator test rejects. The learner reuses the orders of the two marks and tests the order of their product if needed. It takes the half-order power of a word of order divisible by four, obtaining an involution \(d\). A mark that changes \(d\) under conjugation supplies another involution \(e\). Paid checks establish that \(d\) and \(e\) commute and that both marks normalize their four-element subgroup \(K\). An equality in the quotient by \(K\) can then be checked against the four possible kernel offsets. The learner attempts the existing abelian or order-three-commutator decoder on that quotient.

Choose a breadth-first representative \(u\) in each quotient coset. Each marked quotient edge \(u\to v\) has a unique correction \(k'\in K\) satisfying \(um=k'v\) for its mark \(m\). The lift sends \((k,u)\) to \((kk',v)\). The kernel coordinate stays on the left, so this works for noncentral kernels. Tree edges have the identity correction by construction. For a quotient of order 30, only 31 edges remain, each requiring at most three equality calls to identify its correction. The reconstruction is a voltage graph lift with the corrections learned through paid equalities. [Voltage graph construction and spanning-tree normalization](https://arxiv.org/abs/2501.17135)

```sh
uv run --locked python -m scripts.group_student_probe proposal \
  --data . \
  --coverage artifacts/probes/sizing/group-congruence/klein-route-coverage.json \
  --alternating-policy artifacts/probes/sizing/group-congruence/compiled-policy-a5xc2.json \
  --klein-route \
  --output "$PANEL_DIR/klein-route.json"
```

| Family | Cases | Prior commutator calls (symbols) | Klein arm calls (symbols) |
| --- | ---: | ---: | ---: |
| S4 × C5 | 8 fresh | 9,765 (105,568) | 1,223 (35,343) |
| D10 × C12 | 6 fresh | 3,659 (41,867) | 3,665 (41,935) |
| Five previously solved families | 10 replayed | 279 (12,966) | 279 (12,966) |

On the eight targets, calls fell 87.48% against a frozen 70% gate, and queried symbols fell 66.52%. The additional branch took 84–170 calls per target after the shared prefix. Each dihedral control yielded a central involution, paid one extra call, and reproduced the entire fallback tape. Across the full panel the new arm saved 8,536 calls and 70,157 symbols. The ten historical cases reproduced their prior non-timing fields.

Across two local runs, summed online CPU on the eight targets was 3.26–3.33 ms for the new arm and 38.14–43.49 ms for the incumbent. On the six dihedral controls, it was 20.37–22.53 ms and 20.78–21.56 ms respectively. S5 setup, short-certificate preparation, and alternating-policy validation took 68.0–79.7 ms, 40.2–48.4 ms, and 11.2–13.8 ms respectively, shared across the two arms. The branch requires no additional catalogue setup. These are measurements of an in-process oracle on one Mac.

The separate checker replayed the paid replies, reconstructed all 40,320 transitions, built the quotient from concrete cosets, and rejected a forged edge correction. It verified exact recovery on all 216 ordered generating pairs of S4 and all 432 of S4 × C2. All 96 generating pairs of A4 refused at the initial search. A4 has a normal Klein subgroup, and this extractor needs a word whose order is divisible by four. Both hash seeds reproduced the same non-timing output, and the incumbent flag reproduced its six arms. DSR passed 2/2 checks, with 89 diagnostics tests passed and one skipped.

The producer and manifest were frozen at `000d318`. [Frozen source and protocol](klein-route-source.zip) · [Coverage](klein-route-coverage.json) · [Tapes, audit, and receipts](klein-route-results.zip)

## Fifth-power tests for exact quotient equality

The `--fifth-power-route` flag adds an arm after the existing subgroup searches reject. It reuses the paid generator orders and the positive word for the commutator \(c=aba^{-1}b^{-1}\). The earlier negative commutation reply makes \(c\) nontrivial. The new arm checks \(c^5=1\) and that both marks conjugate \(c\) to a unit power of \(c\). This proves \(K=\langle c\rangle\) has order five and is normal. The quotient is abelian because \(ab=cba\), so \(K\) is the derived subgroup.

When five is coprime to the quotient order, \(uK=vK\) holds precisely when \((uv^{-1})^5=1\). In one direction, \(uv^{-1}\) belongs to \(K\), and in the other its quotient image has order dividing five and must be trivial. The learner can therefore answer each quotient comparison with at most one paid call. It uses the shorter positive representation of \(uv^{-1}\) and \(vu^{-1}\) after cancellation, with exact generator orders supplying the inverses. When five divides the quotient order, the learner tests membership against powers of \(c\) explicitly. It then uses the existing abelian lattice decoder and two power lifts to recover the action in the normal forms \(c^k a^x b^y\). The same collection arithmetic handles both the order-three and order-five routes. [GAP reference on power-conjugate presentations](https://docs.gap-system.org/doc/ref/chap46_mj.html)

```sh
uv run --locked python -m scripts.group_student_probe proposal \
  --data . \
  --coverage artifacts/probes/sizing/group-congruence/fifth-power-route-coverage.json \
  --alternating-policy artifacts/probes/sizing/group-congruence/compiled-policy-a5xc2.json \
  --fifth-power-route \
  --output "$PANEL_DIR/fifth-power-route.json"
```

| Family | Cases | Prior Klein calls (symbols) | Fifth-power arm calls (symbols) |
| --- | ---: | ---: | ---: |
| D10 × C12 | 6 fresh | 3,305 (39,096) | 228 (4,997) |
| C5 ⋊ C24 | 6 fresh | 5,968 (65,374) | 244 (6,959) |
| A4 × C10 | 6 fresh | 4,105 (47,840) | 4,111 (48,860) |
| Six previously solved families | 12 replayed | 656 (25,619) | 656 (25,619) |

On the 12 targets, calls fell 94.91% against a frozen 85% gate, and queried symbols fell 88.56%. The new branch took 8–18 calls per target after the shared prefix, with 71 logical quotient equality tests in total. The six outside cases each paid one extra call, adding 1,020 symbols across the six, and reproduced the full cancellation fallback. All historical controls retained their exact tapes and tables. Over the entire panel the new arm saved 8,795 calls and 91,494 symbols.

Over two local runs, summed online CPU on the target cases was 1.99–2.13 ms for the new arm and 95.77–103.12 ms for the incumbent. For the outside cases it was 17.21–19.35 ms and 18.63–21.42 ms respectively, with identical fallback tapes after the added query. Shared S5 setup took 72.4–78.0 ms, short-certificate preparation took 45.7–46.8 ms, and alternating-policy validation took 13.2–14.6 ms. There was no catalogue setup for the new branch. These measurements use an in-process oracle on one Mac.

The separate checker reconstructed all 57,600 transitions, replayed every paid reply, verified the quotient answers by explicit coset arithmetic, and rejected a forged lift. All 300 ordered generating pairs of D10 and C5 ⋊ C4 recovered exactly, exercising all four unit conjugation exponents modulo five. The 12 sampled generating pairs of the order-125 Heisenberg group recovered exactly with explicit coset checks. Every Heisenberg pair exhibited a false equality under an unguarded fifth-power test, since the first mark has fifth power one while its coset is nontrivial. Both hash seeds reproduced the same non-timing output, and the previous flag reproduced its seven arms. DSR passed 2/2 checks, with 89 diagnostics tests passed and one skipped.

The producer and manifest were frozen at `9c598c4`. [Frozen source and protocol](fifth-power-route-source.zip) · [Coverage](fifth-power-route-coverage.json) · [Tapes, audit, and receipts](fifth-power-route-results.zip)

## Seeding the Klein-four lift with a commutator

The `--involution-route` flag tries a second way to find a normal Klein-four subgroup after the prior routes reject. It reuses the paid generator orders and the positive word for the commutator. The initial negative commutation reply makes that word nontrivial. One equality test establishes whether it squares to the identity. If it does, the learner uses it as the first involution and retains the existing checks for a distinct conjugate, commutation, and normalization by both marks. Successful checks feed the existing quotient decoder and its edge corrections. This recovers \(A_4\) actions even though \(A_4\) has no element of order divisible by four. [Graph lift construction](https://arxiv.org/abs/2501.17135)

For this seed, if the quotient order is odd, two words have the same coset exactly when the square of their difference is the identity. Elements of the verified kernel square to the identity, and an odd-order quotient cannot contain a nonidentity element whose order divides two. The implementation reuses the positive-word difference from the fifth-power route. Even-order quotients use explicit kernel offsets.

```sh
uv run --locked python -m scripts.group_student_probe proposal \
  --data . \
  --coverage artifacts/probes/sizing/group-congruence/involution-route-coverage.json \
  --alternating-policy artifacts/probes/sizing/group-congruence/compiled-policy-a5xc2.json \
  --involution-route \
  --output "$PANEL_DIR/involution-route.json"
```

| Family | Cases | Prior fifth-power calls (symbols) | Involution arm calls (symbols) |
| --- | ---: | ---: | ---: |
| A4 × C10 | 8 fresh | 6,093 (68,987) | 888 (41,492) |
| D8 × C15 | 6 fresh | 2,450 (27,145) | 2,468 (28,809) |
| Eight previously solved families | 16 replayed | 815 (30,651) | 815 (30,651) |

The eight targets recovered through explicit coset comparisons in a quotient of order 30. Calls fell 85.43% against the frozen 70% gate, and queried symbols fell 39.86%. The extra branch took 55–101 calls per target after the shared prefix. Each dihedral control produced a central commutator and failed to supply a second involution. Those controls paid three extra calls each and 1,664 additional symbols in total, then ran the identical cancellation fallback. Across the entire panel the new arm saved 5,187 calls and 25,831 symbols.

Across two local runs, summed online CPU for the eight targets was 3.44–3.56 ms for the new arm and 33.09–35.29 ms for the incumbent. On the six dihedral controls it was 16.86–17.67 ms and 17.10–18.12 ms respectively. S5 setup took 76.4–77.9 ms, short-certificate preparation took 39.5–48.2 ms, and alternating-policy validation took 16.1–16.4 ms, shared across the two arms. The new arm adds no catalogue setup. These measurements use an in-process oracle on one Mac.

The separate checker reconstructed 64,800 output transitions, checked every paid reply, independently built the quotient from concrete cosets, and rejected a forged edge correction. The transfer census recovered all 96 ordered generating pairs of A4 with the odd-order square shortcut and all 288 of A4 × C2 with explicit cosets. Every A4 × C2 pair exhibited a false equality under an unguarded square test. All 24 generating pairs of the dihedral group of order eight rejected the central commutator seed. Both hash seeds reproduced the same non-timing fields, and the previous flag reproduced its eight arms. DSR passed 2/2 checks, with 89 diagnostics tests passed and one skipped.

The producer and manifest were frozen at `008b75b`. [Frozen source and protocol](involution-route-source.zip) · [Coverage](involution-route-coverage.json) · [Tapes, audit, and receipts](involution-route-results.zip)

## Recovering a central order-two extension

The `--central-involution-route` flag reuses a result from the commutator-seeded Klein attempt. When that attempt finds a central involution, it has already established that its commutator word \(c\) is nontrivial, \(c^2=1\), and both marks commute with \(c\). The new route uses that proof when the reported involution is the same word \(c\). Since \(ab=cba\), the quotient by \(\langle c\rangle\) is abelian and the derived subgroup is exactly \(\langle c\rangle\). This handoff requires no additional recognition queries.

For quotient equality, compare \(u\) with \(v\) and, if needed, with \(cv\). The existing lattice learner recovers the quotient of order \(n/2\). The lifts of the two lattice relations each lie in \(\{1,c\}\), so one identity comparison identifies each lift. The regular action follows from the unique normal forms \(c^k a^x b^y\), using the same power and conjugation arithmetic as the earlier cyclic-commutator routes. [GAP reference on power-conjugate presentations](https://docs.gap-system.org/doc/ref/chap46_mj.html)

```sh
uv run --locked python -m scripts.group_student_probe proposal \
  --data . \
  --coverage artifacts/probes/sizing/group-congruence/central-route-coverage.json \
  --alternating-policy artifacts/probes/sizing/group-congruence/compiled-policy-a5xc2.json \
  --central-involution-route \
  --output "$PANEL_DIR/central-route.json"
```

| Family | Cases | Prior involution calls (symbols) | Central arm calls (symbols) |
| --- | ---: | ---: | ---: |
| D8 × C15 | 6 fresh | 2,642 (34,251) | 298 (11,251) |
| Q8 × C15 | 6 fresh | 2,523 (36,026) | 292 (16,348) |
| D30 × C4 | 6 fresh | 2,625 (54,761) | 2,625 (54,761) |
| Nine previously solved families | 18 replayed | 1,039 (40,898) | 1,039 (40,898) |

The 12 targets recovered exactly with 88.58% fewer calls against a frozen 80% gate, and 60.73% fewer queried symbols. After the shared prefix, the direct recovery took 9–22 calls per target. The six D30 × C4 controls and the 18 historical cases reproduced the complete incumbent tapes and output fields. The full panel saved 4,575 calls and 42,678 symbols.

Across two local runs, summed online CPU on the target cases was 2.18–2.54 ms for the new arm and 27.63–34.93 ms for the incumbent. On the outside cases, it was 11.00–12.91 ms and 12.17–13.90 ms respectively despite identical paid tapes. S5 setup took 66.0–80.5 ms, short-certificate preparation took 39.6–43.8 ms, and alternating-policy validation took 14.3–14.6 ms, shared across the two arms. These measurements use an in-process oracle on one Mac. The comparison does not isolate the cause of timing variation on the outside controls.

The separate checker reconstructed 86,400 output transitions, replayed every paid reply, verified that every qualified case already paid for the centrality proof, and rejected a forged lift. It recovered all 144 ordered generating pairs of D8, Q8, and the order-16 semidirect product with conjugation \(x\mapsto x^5\). The census exercised all four lift-bit combinations. All 96 generating pairs of the order-16 dihedral control refused on commutator order. The hash-seed replay and previous CLI reproduced the original non-timing fields. DSR passed 2/2 checks, with 89 diagnostics tests passed and one skipped.

The producer and manifest were frozen at `1800539`. [Frozen source and protocol](central-route-source.zip) · [Coverage](central-route-coverage.json) · [Tapes, audit, and receipts](central-route-results.zip)

## Reproduction

The [first coverage manifest](coverage.json) was committed at `3d8e745` and the matched learner ran at `9696164`. The [second manifest](proposal-coverage.json) and [frozen producer and auditor](proposal-source.zip) were committed at `5becc54`. The [verification archive](verification.zip) includes both complete transcripts on each platform, the prior synthetic cohort used to check exclusions, audit source, and the DSR commands and receipts. The summaries expose the per-case counts for [rewriting](summary.json) and [checked proposals](proposal-summary.json).

macOS and Linux emitted byte-identical transcripts after removing only the timing fields and sorting JSON object keys. The Linux regression run reported 927 passed and 16 skipped with `tests/test_runner.py` and `tests/test_runner_enhanced.py` outside that invocation. The macOS diagnostics suite reported 72 passed and one skipped. The archive retains the failed original coverage preflight and a Linux attempt that referenced a nonexistent environment before the successful replay used the qualified Python 3.13 environment.
