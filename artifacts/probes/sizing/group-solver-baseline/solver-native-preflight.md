# Solver-selected queries on resettable rings

The solver selected and verified its own queries on the 8- and 10-state
rings. It exhausted its deterministic budget before choosing a query on the
12-state ring. The frozen execution and replay check passed. The requirement
that the solver finish all three ordinary cases failed. [Frozen protocol and source](solver-native-source.zip)

## Observed cost

All arms were charged for the same four initial words. Each word incurs one
reset and endpoint read. Each symbol incurs one movement command.

| Ring states | Visible exploration reads / moves | Square closure reads / moves | Active solver reads / moves | Solver outcome |
| ---: | ---: | ---: | ---: | --- |
| 8 | 18 / 56 | 13 / 36 | 8 / 21 | Unique table |
| 10 | 22 / 72 | 15 / 44 | 9 / 23 | Unique table |
| 12 | 26 / 90 | 16 / 48 | 16 / 48 | Budget exhausted, square continuation |

The active arm used about 920 ms of aggregate CPU on the three ordinary
cases, including Z3 model extraction. The square arm used about 0.95 ms.
The test uses a simulated reset and fully observable position and heading.
Its historical inspiration used partial senses, and the simulator provides
no measured price for real resets, endpoint observations or turns.
[Rivest and Schapire, 1994](https://www.schapire.net/papers/diversity-based-inference.pdf)

The independent checker rebuilt **80 transitions per arm** across the three
ordinary cases and one deliberate exhaustion control. It checked both SAT
witnesses and brute-forced the shortest distinguishing word for all nine
solver-selected queries. The one-unit control paid the same tape as square
closure. The checker also rejected a forged completion. Changing the Python hash seed
preserved every query, table, solver status and recorded resource count.
The result relies on Z3's reported UNSAT for uniqueness in the two completed
cases. [Raw runs, verifier outputs and receipts](solver-native-results.zip)

## The exhausted check had a constructive alternative

The 12-state case had a valid first model. Z3 returned `canceled` when its
next check tried to find a different right table within the resource limit.
In a separate diagnostic, swapping two unseen names in that first model
preserved the four public labels and produced a different fitting table.
The two verified candidates predicted `opaque00` and `opaque01` for the
same three-move word `000`. This required **zero additional solver checks
and zero oracle reads**.

Conjugating every transition by a permutation of unseen names preserves
bijections, reachability, commutation and generator agreement when the
root is renamed as well. An observed word's endpoint is fixed by the
permutation. Any word leading to one of two swapped names therefore
provides a paid query that would eliminate at least one of the witnesses.
This proof supplies a constructive alternative in that round. Complete
acquisition with the shortcut and its cost on unseen actions remain open.

The development was frozen at `20cf02c`. DSR recorded 2/2 checks for the
preflight and 1/1 for the saved-model diagnostic. The result belongs to
`jqc4.1`. The parent `jqc4` still requires a held-out study and defensible
oracle pricing. [Evidence bundle](solver-native-results.zip)
