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

## Reproduction

The [first coverage manifest](coverage.json) was committed at `3d8e745` and the matched learner ran at `9696164`. The [second manifest](proposal-coverage.json) and [frozen producer and auditor](proposal-source.zip) were committed at `5becc54`. The [verification archive](verification.zip) includes both complete transcripts on each platform, the prior synthetic cohort used to check exclusions, audit source, and the DSR commands and receipts. The summaries expose the per-case counts for [rewriting](summary.json) and [checked proposals](proposal-summary.json).

macOS and Linux emitted byte-identical transcripts after removing only the timing fields and sorting JSON object keys. The Linux regression run reported 927 passed and 16 skipped with `tests/test_runner.py` and `tests/test_runner_enhanced.py` outside that invocation. The macOS diagnostics suite reported 72 passed and one skipped. The archive retains the failed original coverage preflight and a Linux attempt that referenced a nonexistent environment before the successful replay used the qualified Python 3.13 environment.
