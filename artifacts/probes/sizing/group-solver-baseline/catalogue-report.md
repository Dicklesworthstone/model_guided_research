# Complete small-group catalogue versus solver reuse

A classical catalogue recovered exact labeled tables without fallback on
**8/8** fresh marked actions. Solver reuse reached uniqueness on **3/8**.
The catalogue retained those three completions and used **20 fewer reads
and 76 fewer primitive moves** in aggregate. The preregistered completion
and oracle-cost criterion passed. All three arms recovered every table
when fallback was included. [Frozen protocol and producer](catalogue-source.zip)

| Arm | Completions without fallback | Reads / resets | Turns | Advances | Online CPU | Online elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Square closure | 8/8 | 164 | 271 | 431 | 3.43 ms | 3.65 ms |
| Solver reuse | 3/8 | 156 | 271 | 402 | 1.69 s | 2.43 s |
| Catalogue | 8/8 | 136 | 246 | 351 | 42.35 ms | 44.68 ms |

Read counts include the four anchor words and all fallback calls. Each
read includes a reset. Timings are aggregate observations from the first
run on the same Mac and include witness extraction in the solver arm.
The catalogue's online column uses a prepared index. Constructing it from
the exported tables took 30.54 and 30.58 ms of CPU in the two runs. The
GAP export and its inventory and hash checks took about 2.4 s as a
separate one-time operation. [Raw runs and DSR receipts](catalogue-results.zip)

| Action | Read floor | Square reads | Reuse reads | Catalogue reads | Reuse native result |
| --- | ---: | ---: | ---: | ---: | --- |
| 2×7 torus | 14 | 19 | 19 | 16 | Initial UNKNOWN |
| Decagon mirrors | 20 | 25 | 25 | 21 | Initial UNKNOWN |
| Dodecagon turn first | 23 | 28 | 28 | 25 | Initial UNKNOWN |
| 2×3 torus | 6 | 11 | 9 | 7 | Complete |
| 2×5 torus | 10 | 15 | 12 | 11 | Complete |
| 2×6 torus | 12 | 17 | 14 | 14 | Complete |
| 4×4 torus | 16 | 21 | 21 | 17 | Initial UNKNOWN |
| S4, 4-cycle then transposition | 23 | 28 | 28 | 25 | Initial UNKNOWN |

The catalogue finished one or two reads above the naming lower bound
on every case. It made one extra primitive move on S4 relative to each
other arm. At panel level it saved 28 reads, 25 turns and 80 advances
relative to square closure. For a prepared index, the observed wall
time overhead versus square was 41–43 ms across the two host runs.
If the oracle is serial with constant costs per reset/read ($q$),
turn ($t$) and advance ($f$), the savings would cover that overhead when
$28q+25t+80f$ exceeds 41–43 ms. The prices and real elapsed-time
benefit still need measurement.

## Coverage and verification

The input is GAP SmallGrp 1.7.0's 45 multiplication tables at orders 6,
10, 12, 14, 16, 20 and 24. Enumerating all ordered generating pairs
and quotienting by rooted marked isomorphism gave 368 distinct shapes.
Each acquisition arm saw the same opaque roster, four anchors and paid
endpoint oracle. The catalogue used the public roster size to select
the full inventory for that order. Its uniqueness test relies on the
catalogue's completeness at those orders. [SmallGrp library and completeness statement](https://gap-packages.github.io/smallgrp/)

The checker tested the group axioms on the exported tables,
independently rebuilt the 368 shapes and tested freshness against 32
rooted marked actions in earlier archives. It verified 252 labeled
right transitions per arm, checked strict reductions in the catalogue
version space, validated solver witnesses and scopes, and rejected an
omitted group and a premature uniqueness claim. The GAP export
reproduced byte for byte. The acquisition replays agreed on all
non-timing fields across Python hash seeds.

The panel was frozen at `38dbdfc`. The first run saved both raw tapes
and stopped when the verifier tried to use set subtraction on a
dictionary values view for the last opaque label. The result bundle
contains the original failure and the corrected one-line set
difference operation. DSR passed 2/2 checks against the original raw
tapes after that repair. [Replay bundle and corrected verifier](catalogue-results.zip)

The catalogue establishes a stronger conventional baseline at these
tested orders. This experiment supplies no evidence that solver reuse
beats it there. Scaling when a complete catalogue is impractical
remains a separate question. `jqc4.6` records the finite comparison,
and `jqc4.5` tracks matched physical measurements.
