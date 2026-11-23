# Compact certificates for equality-only group acquisition

We replaced a 121-equation transition certificate with at most 12 positive-word equations. On six fresh marked S₅ actions, paid equality calls fell from 754 to 109 and oracle input symbols fell from 9,832 to 1,852. All 12 same-order negative controls rejected the proposed action and recovered exactly through cancellation. One further comparison removes the need to know the hidden group order when certifying an S₅ proposal.

## Why the certificate works

Let x and y be the two hidden generators. For each candidate, the compiler expresses four adjacent transpositions as positive words u₀, …, u₃ in x and y. It supplies four involution equations, three adjacent braid equations, and three distant commutation equations. Two more equations express x and y as positive words in the uᵢ. The compiler chooses among 60 oriented orderings of the five points by expanded symbol count, equation count, lexicographic equation order, and then point order. It drops only syntactic identities and duplicate equations.

The first ten equations are the type A₄ Coxeter relations. The verifier checks an explicit order bound using 40 local word rewrites. At rank r, a normal form is a word from rank r − 1 followed by one of r + 1 descending tails, including the empty tail. Each tail times a generator rewrites to a lower-rank generator or the identity followed by another allowed tail. This closes the normal forms under multiplication and gives an upper bound of 2 × 3 × 4 × 5 = 120. The two recovery equations make the four images generate the entire hidden group. When that group is known to have order 120, the resulting surjection from S₅ is an isomorphism, and the recovered words identify the same two marks. The verifier checks the rewrites and both transports for every catalog entry before any hidden-oracle query. [Coxeter's original presentation and classification](https://sites.math.washington.edu/~billey/classes/reflection.groups/references/Coxeter.1935.pdf)

## Preregistered matched comparison

Both arms used the same 57-action catalog, the same proposal queries, identical-answer reuse, and the original cancellation fallback. The [coverage manifest](coverage.json) freezes the 18 previously unseen actions. Totals include selection, paid certificate checks, and fallback. The full breakdown by phase is in the [audited summary](summary.json).

| Hidden family | Cases | Accepted in each arm | Edge calls | Compact calls | Edge symbols | Compact symbols |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `S5` | 6 | 6 | 754 | 109 | 9,832 | 1,852 |
| `C120` | 4 | 0 | 816 | 816 | 17,969 | 18,001 |
| `A5 × C2` | 4 | 0 | 4,857 | 4,854 | 64,110 | 64,142 |
| `S4 × C5` | 4 | 0 | 5,685 | 5,685 | 62,356 | 62,373 |

The gate passed. The catalog had 57 of 57 checked witnesses, each with 11 or 12 equations. The six `S5` cases used 40 identical selection calls and 714 edge or 69 compact certificate calls, a total reduction of 85.54%. Their symbol count fell by 81.16%. Outside the catalog the compact arm asked three fewer questions and sent 81 more symbols. The independent replay checked every paid Boolean answer, 8,640 final transition entries across both arms, and 3,456 distinct-per-case held-out words against the exact actions.

| Platform | Catalog build | Compact witness search | Offline proof checks | Edge online | Compact online |
| --- | ---: | ---: | ---: | ---: | ---: |
| macOS | 0.782 s | 0.048 s | 0.009 s | 0.211 s | 0.210 s |
| Linux | 2.587 s | 0.169 s | 0.032 s | 0.765 s | 0.778 s |

The elapsed times are single runs with an in-process oracle. The gain is in oracle calls and input size. Query latency in a remote or human-supplied oracle was outside this experiment.

## One query removes the order assumption for acceptance

If only the positive equations hold, the hidden group may be any quotient of S₅. All 57 certificates' positive equations hold in both the sign quotient and the trivial quotient. Ask whether u₀u₁ = u₁u₀ and require a negative answer. Their commutator is a 3-cycle. For each of the 119 nonidentity permutations g, the finite proof checks that a fixed such commutator is a product of at most two conjugates of g. Any nontrivial kernel is normal and therefore kills the commutator. A negative answer forces the kernel to be trivial. The two recovery equations make the map onto, so this certifies the marked S₅ without an input cardinality.

Across 171 exact candidate-and-quotient tests, the added query retained all 57 S₅ cases and rejected all 114 proper-quotient controls. The checker also rejected a truncated conjugation witness. This extension certifies an existing proposal using at most 13 equality checks. Proposal selection and fallback without a known group order were outside the scope of this proof and experiment.

## Reproduction

The matched protocol was committed at `801019e` before the panel ran. Source and coverage were frozen at `63b788d` in the [source archive](source.zip) and [coverage manifest](coverage.json). The additional kernel test was registered at `f7fbb80`. The [verification archive](verification.zip) contains the candidate transports, both complete transcripts, kernel witnesses and checker, the cross-platform comparison, and DSR command and test receipts. The independent audit rejects an invalid local rewrite, an omitted equation, and a corrupted recovery word.
