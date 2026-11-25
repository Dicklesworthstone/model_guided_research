# Reusing a fitting model during group acquisition

Model reuse raised solver-only completion from **2/7 to 4/7** on held-out
marked actions. It retained the two original completions and finished the
pentagon and 3×4 torus. Base checks fell from **32 to 12**, passing the
preregistered work criterion. Temporary rank ordering rescued **0/2** initial
failures. It changed the search path enough to finish the pentagon, but it
lost the 3×4 torus completion when combined with reuse. The standalone reuse
arm saved the most paid observations. [Frozen protocol and source](base-solver-source.zip)

| Arm | Solver completions | Base checks | Alternative checks | Reads / resets | Turns | Advances | CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Square closure | — | 0 | 0 | 119 | 247 | 227 | 2.50 ms |
| Frozen conjugation | 2/7 | 32 | 5 | 112 | 229 | 208 | 3.06 s |
| Reuse | 4/7 | 12 | 4 | 103 | 222 | 182 | 1.68 s |
| Rank ordering | 3/7 | 36 | 7 | 107 | 220 | 200 | 2.65 s |
| Reuse and rank ordering | 3/7 | 12 | 7 | 107 | 222 | 200 | 2.08 s |

Totals charge the same four anchor words per case and include fallback calls.
They exclude the one-unit control. CPU includes Z3 and witness extraction.
Of the 25 reused witnesses, three kept the first candidate, six kept the
second, and 16 transported a candidate by swapping two unseen names.

| Action | Frozen conjugation | Reuse | Rank ordering | Both | Read floor | Reuse reads |
| --- | --- | --- | --- | --- | ---: | ---: |
| Pentagon turn first | Alternative UNKNOWN | Complete | Complete | Complete | 9 | 10 |
| Hexagon mirrors | Complete | Complete | Complete | Complete | 12 | 13 |
| Octagon mirrors | Initial UNKNOWN | Initial UNKNOWN | Initial UNKNOWN | Initial UNKNOWN | 16 | 21 |
| Nonagon turn first | Initial UNKNOWN | Initial UNKNOWN | Initial UNKNOWN | Initial UNKNOWN | 17 | 22 |
| 2×4 torus | Complete | Complete | Complete | Complete | 8 | 9 |
| 3×4 torus | Later base UNKNOWN | Complete | Alternative UNKNOWN | Alternative UNKNOWN | 12 | 13 |
| Tetrahedral rotations | Later base UNKNOWN | Later base UNKNOWN | Initial UNKNOWN | Initial UNKNOWN | 11 | 15 |

Every UNKNOWN in this run carried Z3's `canceled` reason. The independent
checker verified **192 right transitions per arm**, including fallback and
the forced-exhaustion control. It reconstructed the actions separately,
verified the paid queries and both fitting witnesses, checked scope and
reuse provenance, and rejected a missing pop and a forged reuse method.
The separate hash-seed execution reproduced every non-timing field. The
one-unit check exhausted all four solver arms before paying another label.
All four continued with the same tape as square closure. Uniqueness uses
Z3's replayed UNSAT result after the rank scope has been popped.
[Raw tapes, verifier output and receipts](base-solver-results.zip)

## Minimum paid observations

With a known roster of size $n$, a sound algorithm must observe at least
$n-1$ distinct endpoints to recover exact named transitions. Suppose two
names remain unseen. Transposing them everywhere, including at the root,
fixes all observed replies. It changes at least one right transition. If
the transposition commuted with both right generators, then fixing any
observed vertex would force it to fix every vertex reachable from that
vertex. The finite permutations give transitive reachability from every
vertex, so the transposition would be the identity. This contradiction
gives an indistinguishable alternative table. With $u$ distinct names in
the four mandatory replies, the total paid-read floor is $4+(n-1-u)$ on
this panel. It sums to **85**, leaving reuse 18 reads above the bound and
square closure 34 above. If the rooted, unlabeled action were supplied,
one chosen word would label each queried vertex, and the roster would
determine the final label by elimination. Structural ambiguity in the
actual task can require additional queries. The preregistration did not
include this post-run analysis.

## When the saved queries could pay for the solver

Reuse saved 16 reset-and-read pairs, 25 turns and 45 advances relative to
square closure. Across the two Mac executions it added 1.84 to 1.95 seconds
of aggregate wall time on the immediate-response simulator. In a serial
oracle with constant costs $q$ per reset-and-read, $t$ per turn and $f$
per advance, the saved time would be $16q+25t+45f$. If movements were
free, the observed break-even range would be about 115 to 122 ms per
reset-and-read. This calculation uses host timings and hypothetical
physical costs. Actual latency distributions and any state-dependent
costs still need measurements.

Published product-table recovery can directly ask for multiplication of
two named elements. Rivest and Schapire's historical inference setting
allows continuous exploration without resetting. The cost calculation
here applies to fixed-root generator words that return a full opaque
endpoint label. Neither paper provides a direct measured comparator for
that interface. [Product-table recovery](https://www.math.uzh.ch/en/fileadmin/user/rosen/publikation/zu08p.pdf)
[Diversity-based inference](https://people.csail.mit.edu/rivest/pubs/RS94a.pdf)

The first DSR attempt at `d99ae72` reached all seven ordinary cases and
failed when the one-unit control canceled scope setup. The repair at
`6aca3e2` applied the one-unit limit to the scoped check and kept the
ordinary command paths fixed. The full DSR run passed **2/2** checks,
with the failed receipt retained. This closes the two synthetic probes
`jqc4.3` and `jqc4.4`. The parent study `jqc4` still needs measured
oracle pricing. [Evidence bundle](base-solver-results.zip)
