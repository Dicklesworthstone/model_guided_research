# Canonicalize one pair per automorphism orbit

At order 128, the orbit method built exactly the same 1,534
all-distinct marked actions. Cold build CPU fell from **23.13 to
3.55 seconds** against the frozen early filter and from **18.02 to
3.55 seconds** against the Frattini generation control. The replay
gave **23.18, 17.94, and 3.41 seconds**, respectively. Both preregistered
timing criteria passed. The orbit arm lost to both baselines at
orders 32 and 64. [Frozen protocol and source](orbit-source.zip)

| Order | Early filter traversals | Frattini traversals | Orbit traversals | Early CPU | Frattini CPU | Orbit CPU | GAP component of orbit CPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 7,840 | 5,984 | 82 | 31.23 ms | 30.36 ms | 1.028 s | 1.004 s |
| 64 | 118,208 | 74,432 | 368 | 0.844 s | 0.672 s | 1.264 s | 1.244 s |
| 128 | 1,780,864 | 967,040 | 1,534 | 23.132 s | 18.019 s | 3.545 s | 3.467 s |

The table reports the first run. The orbit timings include startup
of a fresh GAP process per order, reconstruction of the provider's
groups, calculation of automorphisms and pair orbits, parsing,
and construction of the canonical graphs. The other arms start
with the common tables in memory. The score and the measured replay
reverse arm order. The order-128 orbit wall times were 5.04 and 4.36
seconds. The live early baseline took 26.42 and 48.41 seconds of wall
time. The CPU criterion was set before either execution.
[Raw timing and provider receipts](orbit-results.zip)

An ordered generating pair determines a rooted marked Cayley graph.
Two pairs from the same group determine isomorphic graphs exactly when
an automorphism sends one pair to the other. Fixing both generators
forces the automorphism to be the identity. The anchor equalities are
invariant, so each eligible orbit contributes one graph. The provider
first takes orbits on the first generator and then takes stabilizer
orbits on compatible second generators. [GAP automorphism reference](https://docs.gap-system.org/doc/ref/chap40.html)
[AutPGrp 1.13.0](https://gap-packages.github.io/autpgrp/)

The independent checker validated **2,039 automorphism generators**
on the multiplication tables. It checked the Frattini quotient maps,
expanded disjoint pair orbits to cover **1,047,456 eligible generating
pairs** at the three orders, and independently reconstructed all
**1,984 canonical graphs**. Every final set matched the earlier archive
and both live controls. The checker rejected a corrupted permutation
and a missing orbit. The replay preserved every non-timing field.

The DSR provider preflight passed 1/1 checks. The frozen comparison at
`1cccb9a` passed 2/2. The package archive and native GAP binary
fingerprints are in the evidence bundle. For reproduction, take
`group-tables.json` from [the anchor source](anchor-source.zip) and
`scored.json` from [the anchor evidence](anchor-results.zip) as
`historical-shapes.json`. The orbit source manifest pins both bytestrings.

This closes `jqc4.10`. The result concerns the exact initial version
space for the all-distinct profile at orders 32, 64 and 128. It provides
no new online acquisition or physical timing measurements. The hardware
comparison remains open under `jqc4.5`.
