# Live bounded scheduling for equality-only group acquisition

The frontier learner and catalog selector now yield their next equality comparisons as resumable plans. Their blocking callback APIs run the same plans. The bounded scheduler runs the frontier and a checked, size-free S₅ proposal against one live equality callback. It shares answers to identical unordered pairs and permits one paid proposal comparison per four paid frontier comparisons. A rejected proposal drops out while the original frontier keeps its state. If both finish together, the frontier takes precedence.

This policy caps paid comparisons at B + floor(B/4), where B is the isolated frontier's charge count on the same oracle. The shared exact answers can only reduce the number of queries charged to that frontier. If an isolated proposal certifies in P comparisons, the combined run needs at most 5P. These bounds are on comparison count. Input symbol costs differ by query.

## Live conformance

All 28 cases produced the exact rooted action. The 18 reused cases reproduced the previous study's blocking comparisons and scheduling counts. An independent checker re-evaluated the words as explicit permutations, reconstructed each paid comparison and its owner, and checked both the local frontier and the compact certificate. It inspected 5,126 transitions in the live results and rejected a transcript that charged the wrong planner.

| Panel | Family | Cases | Certified by proposal | Frontier calls | Proposal-first calls | Live bounded calls | Frontier symbols | Live bounded symbols |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reused | S₅ | 6 | 6 | 8,186 | 110 | 540 | 103,289 | 4,361 |
| Reused | S₄ | 4 | 0 | 326 | 357 | 353 | 2,174 | 2,527 |
| Reused | A₆ | 4 | 0 | 54,692 | 54,725 | 54,717 | 870,301 | 870,579 |
| Reused | Trivial | 1 | 0 | 2 | 17 | 2 | 2 | 2 |
| Reused | C₂ | 3 | 0 | 11 | 51 | 12 | 16 | 32 |
| Conformance | S₅ | 2 | 2 | 2,563 | 40 | 195 | 32,101 | 1,550 |
| Conformance | A₄ | 4 | 0 | 131 | 164 | 159 | 585 | 931 |
| Conformance | C₃ | 4 | 0 | 23 | 53 | 25 | 44 | 76 |

The conformance panel used both remaining unseen S₅ markings, all four rooted marked A₄ classes and all four C₃ classes. Both held-out S₅ certificates completed and every other new case closed the frontier. All 57 marked S₅ classes have now appeared in this series of experiments.

## Local elapsed time and allocation

Timing used one process with an in-process equality oracle. The cached proof library took 0.036 s to load and check on macOS and 0.106 s on Linux. For the reused panel, frontier and live bounded runs took 0.255 s and 0.477 s on macOS and 0.759 s and 1.249 s on Linux. On the conformance panel the corresponding times were 0.013 s and 0.075 s on macOS and 0.036 s and 0.270 s on Linux.

Separate allocation tracing used the first case of each family per panel and included the returned traces. On macOS the reused A₆ sample rose from 5,444,278 to 7,838,791 peak allocated bytes with the live scheduler. The reused S₅ sample fell from 668,500 to 36,541 bytes because the frontier stopped before completing the table. The archived allocation measurements describe Python allocations under this harness. The call bound does not imply a latency or allocation bound.

## Reproduction

The protocol and planner change are at `24b8b6c`. The [coverage manifest](coverage.json) and [experiment source](source.zip) were frozen at `90048dd`. The [audited summary](summary.json) records every case and charge split. The [verification archive](verification.zip) contains the full live callback tapes, independent audit output, allocation samples, parity proof and DSR receipts from both platforms.

macOS and Linux produced byte-identical scientific transcripts after removing timing fields. The Linux DSR run passed all five checks, including the full repository test suite with 932 passed and 16 skipped. The macOS diagnostics suite reported 77 passed and one skipped.
