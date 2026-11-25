# Unseen-label conjugation on held-out group actions

Swapping two unseen labels reduced global alternative-table solver checks
from **22 to 6**. The original and modified solvers completed the same
three of six held-out actions. The completion-improvement criterion
failed. All three remaining fallbacks arose during a base check for a
fitting model. [Frozen protocol and code](conjugation-source.zip)

## Cost and exactness

The number of paid words includes the four initial words per case.
Each word incurs a reset and endpoint read. Primitive motion counts
expand the two command symbols as specified in the protocol.

| Arm | Reads | Turns | Advances | Base checks | Alternative checks | CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Visible exploration | 138 | 296 | 246 | 0 | 0 | 0.30 ms |
| Square closure | 90 | 172 | 141 | 0 | 0 | 2.03 ms |
| Frozen native solver | 80 | 148 | 124 | 25 | 22 | 1.60 s |
| Conjugation then solver | 80 | 148 | 124 | 24 | 6 | 1.40 s |

The constructive arm used **15** transpositions on the ordinary cases.
It passed the work-count criterion and the read-and-movement criterion
relative to square closure. The counts include fallback effort and
exclude only the stated exhaustion control. CPU times are single-host
observations that include model extraction. The simulation supplies
no measured hardware prices.

| Action | Original native solver | Conjugation arm | Constructive queries | Reads in conjugation arm |
| --- | --- | --- | ---: | ---: |
| Square mirrors | Complete | Complete | 4 | 9 |
| Pentagon mirrors | Initial base UNKNOWN | Initial base UNKNOWN | 0 | 15 |
| Hexagon turn first | Later base UNKNOWN | Later base UNKNOWN | 1 | 16 |
| Octagon turn first | Initial base UNKNOWN | Initial base UNKNOWN | 0 | 20 |
| Quaternion units | Complete | Complete | 4 | 9 |
| Two-axis torus | Complete | Complete | 6 | 11 |

The checker verified **142 transitions per arm** across these cases
and the one-unit exhaustion control. It reconstructed the actions,
validated every paid answer and every saved model, and brute-forced
the shortest claimed query words. The second Python hash seed preserved
all untimed fields. The exhaustion control preserved the square
fallback tape in both solver arms, and the checker rejected a planted
attempt to swap an exposed label. DSR passed **2/2** checks against
registration `56c42ca`. [Raw observations and receipts](conjugation-results.zip)

## A fitting model can survive an oracle reply

On the saved trace, a witness already fit the new reply or could be
conjugated to fit it after **17 of 18** solver-selected queries. The
transposition case requires both the predicted name and observed reply
to have been unexposed before the query. Conjugating by their swap fixes
the earlier responses and maps the new prediction to the observed one.

For the hexagon, after the oracle's answer to `000`, the transported
model produced a second valid distinguishing query, `001`. The saved
witnesses predicted `opaque09` and `opaque00`. The diagnostic performed
zero extra solver checks and zero extra oracle reads. DSR passed **1/1**
on this diagnostic. Full acquisition with model reuse and its effect on
the initial base failures remain open. [Saved witnesses](conjugation-results.zip)

The work-count result closes `jqc4.2`. The subsequent model-reuse and
initial-solve tests, plus measured oracle pricing, remain part of
`jqc4`.
