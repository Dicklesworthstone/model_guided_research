# What four endpoint reads tell us before building a catalogue

Four of the five possible equality patterns admit a direct construction
of every compatible marked action. Three patterns give a quotient of
Z². The noncommuting equal-squares pattern gives a quotient of the
Klein bottle group, with a cyclic normal subgroup and inversion action.
The first four paid replies determine which construction applies. The
all-distinct pattern remains the hard case. [Frozen scope, proofs and code](anchor-source.zip)

The construction matched the full catalogue in all 45 order-by-pattern
cells, including 30 impossible cells, across orders 32, 64 and 128.
There were 2,453 distinct rooted marked graphs. The independent verifier
checked every graph's edge range, bijectivity, canonical labeling and
anchor pattern. It rejected an omitted action and an action filed under
a wrong pattern. The direct equal-squares construction had exactly 6,
8 and 10 shapes, with the expected parameters recovered independently
from each graph. Each order-32 integration case recovered all 64 right
transitions and used the same 33 reads under every applicable route.
[Raw graph sets, tapes and receipts](anchor-results.zip)

## Cold construction costs

The incumbent constructs all marked graphs of the order and can reuse
its resulting index. The timed early filter builds one pattern from
the same already loaded multiplication tables. The direct builders
use the order and pattern alone. All times are process CPU in the
first execution. The measurements exclude reading the export and
include canonical sorting. The replay reversed build order. Exact
graph sets, traversal counts, paid tapes and tables reproduced.

| Order | Anchor pattern | Shapes | All-pair traversals | Filtered traversals | Full build ms | Filtered ms | Direct ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0000 | 1 | 20,480 | 96 | 61.04 | 0.167 | 0.161 |
| 32 | 0110 | 2 | 20,480 | 224 | 61.04 | 0.511 | 0.159 |
| 32 | 0112 | 60 | 20,480 | 2,752 | 61.04 | 10.467 | 0.638 |
| 32 | 0120 | 6 | 20,480 | 1,568 | 61.04 | 5.323 | 0.079 |
| 32 | 0123 | 82 | 20,480 | 7,840 | 61.04 | 32.570 | — |
| 64 | 0000 | 1 | 221,184 | 256 | 1,131.87 | 1.310 | 0.665 |
| 64 | 0110 | 2 | 221,184 | 640 | 1,131.87 | 2.989 | 0.691 |
| 64 | 0112 | 124 | 221,184 | 15,488 | 1,131.87 | 96.173 | 2.097 |
| 64 | 0120 | 8 | 221,184 | 12,736 | 1,131.87 | 54.659 | 0.157 |
| 64 | 0123 | 368 | 221,184 | 118,208 | 1,131.87 | 807.859 | — |
| 128 | 0000 | 1 | 2,670,592 | 512 | 25,539.50 | 3.623 | 2.282 |
| 128 | 0110 | 2 | 2,670,592 | 1,280 | 25,539.50 | 10.932 | 2.288 |
| 128 | 0112 | 252 | 2,670,592 | 63,744 | 25,539.50 | 774.942 | 7.575 |
| 128 | 0120 | 10 | 2,670,592 | 106,880 | 25,539.50 | 657.694 | 0.289 |
| 128 | 0123 | 1,534 | 2,670,592 | 1,780,864 | 25,539.50 | 22,441.966 | — |

The original cold criterion used the abelian direct route for commuting
patterns and the early filter otherwise. It passed with median factors
of 1,089× and 1,131× over the ten order-64 and order-128 cells. Equal
weighting gives six of those cells to the abelian route. For the
all-distinct pattern at order 128, filtering shortened the build from
25.54 seconds to 22.44 seconds. That pattern held 1,534 of the 1,799
shapes. No distribution over physical actions or warm-index benefit
was measured.

The 237 input group tables come from the complete rank-one and rank-two
SmallGrp inventories at the chosen orders. The Frattini quotient
characterizes the generator rank of finite p-groups. The DSR input
export reproduced byte for byte. The sizing run also counted 541
eligible groups at order 256 and 2,044 at order 512. Neither size
underwent a builder benchmark. [SmallGrp inventory](https://docs.gap-system.org/pkg/smallgrp/doc/chap1_mj.html)

The baseline was frozen at `8b4381b`, and the separate equal-squares
test was added before scoring at `731f0b5`. The main DSR run passed
2/2 checks. These data close `jqc4.7` and `jqc4.8`. End-to-end
acquisition was tested at order 32. Physical timing remains open under
`jqc4.5`.

## Can one extra read make the all-distinct case simpler?

If the endpoint of `0101` equals that of `00`, the alternate marks
$(a,ab)$ have equal squares. They generate the same group, and the
original b action is recovered by applying the inverse of a followed
by ab. The mirrored test uses `1010` and `11`. Each success reduces
the problem to the exact inversion construction.

| Order | All-distinct shapes | 0101 succeeds | 1010 succeeds | Either succeeds |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 82 | 6 | 6 | 12 |
| 64 | 368 | 8 | 8 | 16 |
| 128 | 1,534 | 10 | 10 | 20 |

The two success classes were disjoint. The diagnostic converted every
passing graph to an equal-squares graph and restored the original
marks exactly. Each query was a bijection onto the full equal-squares
class on its success set. At order 128 both probes together covered
1.30% of distinct marked shapes, which does not estimate their
frequency on physical oracles. This was a separate exact census of the
frozen graphs. No new paid tape or timing advantage was measured. DSR
passed 2/2 checks and reproduced the counts. This closes `jqc4.9`.
[Diagnostic source and proof](anchor-remarking-source.zip)
