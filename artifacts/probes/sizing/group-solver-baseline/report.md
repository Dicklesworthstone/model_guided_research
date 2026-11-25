# Exact group acquisition costs on four new families

**Decision.** Square closure used 28.65% fewer total endpoint labels than
classical visible-state exploration, at 22.65× the measured CPU time.
That misses the preregistered 1.25× CPU ceiling for an overall win.
The Z3 comparison was inconclusive under the preregistered replay and
solver-unknown gates. The exact tables and paid labels for all three
learners reproduced. [Protocol](protocol.md) · [Outputs and receipts](results.zip)

## Measured cost

Every learner started with the same 448 paid labels across 16 distinct
ordered group actions. The independently implemented verifier checked all
448 endpoints, every proposed response, 64 length-two anchors and 630 right
transitions per learner. The anchors and transitions determine answers for
all words of length at least two in each of these actions.

| Learner | Additional labels | Total labels | Total input symbols | CPU | Elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Visible-state exploration | 278 | 726 | 3,040 | 1.404 ms | 1.680 ms |
| Square closure | 70 | 518 | 1,907 | 31.804 ms | 35.681 ms |
| Square closure with incremental Z3 | 52 | 500 | 1,806 | 27.074 s | 43.021 s |

| Family | Actions | Visible additional | Square additional | Z3 additional | Z3 unknowns in scored run |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dihedral, order 14 | 3 | 24 | 8 | 4 | 0 |
| Generalized quaternion, order 16 | 3 | 36 | 10 | 5 | 0 |
| Nonabelian semidirect product, order 21 | 5 | 90 | 10 | 6 | 1 |
| Symmetric group, order 24 | 5 | 128 | 42 | 37 | 4 |

The visible-state arm adapts classical permutation-automata exploration
to a reset-word endpoint oracle. It exploits bijectivity, seeds its
transition cache with the common inputs and follows its own shortest
access words. Square closure saves labels in every family. At a uniform
price per label, the observed break-even is **0.146 ms of CPU per avoided
label**, calculated as 30.4 ms of extra CPU divided by 208 saved labels.
This cost example uses one M5 Ultra host and one timing observation per
arm and case. The archive keeps elapsed time and input symbols separately.
[Rivest and Schapire](https://people.csail.mit.edu/rivest/pubs/RS93a.pdf)

Z3 used the identical square-closure proposal tape and waived 18 calls.
That saves 3.47% of total labels, below the registered 10% margin.
Its measured additional CPU was about 1.50 seconds per label avoided.
The conventional solver could use its own proposal strategy in a different
experiment. The present setup isolates whether stronger inference saves
labels on the same pending words.

## What failed and what survived

DSR ran the producer and audit from clean registration revision
`16889a67d51da752ea14eec1c25ce1d5f3706716`. The prescore checks passed
**1/1**, including three Lean statements, three rejected forged artifacts,
nine paid queries when the solver ran out of resources and two saturated
controls that launched zero solvers and asked zero questions. The proof
checks establish the local square rule, the implication from a valid
unsatisfiability result and extension by induction. The UNSAT judgments
themselves remain solver-reported.

The scored DSR run passed **1/2**. In case 11, the nonabelian group of
order 21, the search for a second endpoint of `00110` was SAT in the
scored run and UNKNOWN in the audit. Both branches paid for the same
oracle answer. The cap reason was not recorded, so the source of the
solver outcome change is unresolved. The full nontiming gate rejected
the result. No arm was rerun or retuned after that failure.

The separate DSR diagnostic passed **1/1** on the two already saved
transcripts. It verified each run's tables and labels independently,
checked every saved audit SAT witness, and identified exactly one
solver-event mismatch. The classical and square-closure arms reproduced
their full nontiming tapes on 16/16 cases. So did every final table and
paid-label tape across the three arms. This diagnostic leaves the original
failure in force.

## Scope

This is a conditional label-efficiency improvement over a
literature-derived method on a finite synthetic benchmark. The opaque
state names are enumerated in advance, the oracle supplies trusted labels,
and both generators act as permutations. Population-level performance,
transformer benefit, and the value of model guidance relative to
conventional research remain open. A useful next comparison
needs a named workload with a meaningful label cost and a solver with an
independent query policy. Freeze deterministic resource accounting and
check it on separate development inputs first.

The Beads graph tracks the prescore comparison, implementation, controls,
scored run and final disposition under `ucsf`. The NTM comparator review
arrived through direct pane dispatch after native pipeline delivery failed.
An independent refuter corrected the oversubscribed group-family counts
before freezing and independently reconstructed the scored actions afterward.
[Source and comparator review](source.zip)
