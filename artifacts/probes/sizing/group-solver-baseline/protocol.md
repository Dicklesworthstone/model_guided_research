# Exact group acquisition against visible-state exploration and Z3

EXPLORATION ends when this protocol, the producer and the verifier are
committed. EXPLOITATION runs the proof, control and fixed-panel checks at that
source revision. See Beads `ucsf` through `ucsf.5`.

## Question and competing explanations

Can the shipped square-closure learner recover exact actions on new group
families at materially lower cost than conventional visible-state
exploration? How much further does incremental constraint solving go?

1. Local inference may materially reduce compute at equal or lower label
   cost. Falsifier: a total-label penalty, failure on any case, or less than
   a fivefold aggregate CPU saving when the solver completes its checks.
2. Global inference may need fewer labels. Falsifier: it fails to save 10%
   of total label cost under the decision rule below. Local inference cannot
   use fewer labels under the shared query selector.
3. The methods may trade oracle labels for compute or tie within the gates.
   A qualified directional win under either gate rules out this outcome.

The application is finite symbolic exact identification. Each arm recovers a
transition table for every label and both ordered symbols. Four length-two
anchors and the total right action suffice for every word of length at least
two. The separately checked Lean statements cover one commuting-square
completion, the logic of forced edge inference, and extension by induction.
They make no complexity, novelty or hardware claim.

## Incumbent and information budget

The primary literature baseline adapts visible permutation automata
exploration from [Rivest and Schapire, §6](https://people.csail.mit.edu/rivest/pubs/RS93a.pdf)
to the reset-word oracle. Starting with the same observed edges, it
extends the shortest accessible representatives in breadth-first order.
It exploits permutation bijectivity to fill the last missing edge of a
generator column. It has its own query tape and uses no left transitions.

Incremental Z3 follows the standard constraint-based tradition of
[Heule and Verwer](https://www.cs.cmu.edu/~mheule/publications/DFA_ICGI.pdf).
Their DFA sample protocol does not use our endpoint-only oracle. Angluin's
L* assumes access to equivalence queries. This comparison uses neither that
oracle nor a group presentation or catalog. It tests a new exact encoding
using [stock Z3](https://z3prover.github.io/papers/programmingz3.html) as the
solver engine. This arm isolates the value of stronger inference behind
the same policy. Neither baseline is a published result on this panel.

All three arms receive the complete set of opaque names, two ordered symbols,
and exact endpoints of all 28 words of lengths two through four. Charge all
28 labels and their input symbols to each arm. They can buy one trusted word
endpoint per call. The candidate is the shipped square-closure learner.
The Z3 arm runs the same query chooser and square closure and asks Z3
whether the proposed word already has a unique endpoint. Z3 models finite
left and right maps as permutations, requires left/right commutation for
each pair, anchors each generator to the same endpoint on either side of
the identity, requires every label be reachable from the root via a ranked
predecessor witness, and constrains the observed words. Its identity label is
unknown. It checks satisfiability and then excludes the model's proposed
endpoint for that word. A different satisfiable endpoint requires a paid
query. Unsatisfiability of the exclusion alone licenses an inferred answer.
A solver unknown requires a paid query and suppresses a directional win
claim for the solver comparison. The local and Z3 arms must emit exactly
the same proposals and receive exactly the same proposal answers. Z3's paid tape
must be a subsequence of the local arm's. Any divergence is a producer
failure. A solver with its own query policy could outperform both arms on
labels and is outside this comparison.

Every real group action is a model of these constraints. An encoding that
admits additional models can defer a sound inference but cannot thereby
force a false endpoint. The independent verifier checks actual endpoints
and transitions from the fixture's group law and verifies recorded SAT
assignment witnesses against the encoding's stated semantics. A Lean proof
of the inference rule does not certify the external solver's UNSAT results.

## Frozen panel, controls and decision

Use 3, 3, 5 and 5 ordered generating pairs for D14, Q16, the nonabelian
group of order 21, and S4, respectively. Here D14 has 14 elements and Q16
is generalized quaternion of order 16. The first two have only three
distinct marked-action classes each. Pick pairs from the exact ordered
element enumeration with Python 3.13 PRNG seeds 2026112501 through
2026112504. Reject pairs that do not generate and reject repeated
identity-rooted directed graphs with ordered edges. Shuffle endpoint names
independently with the per-family stream after accepting each pair. The
learner receives neither family nor the naming permutation. The primary
unit is a distinct marked action and the entire result is finite-panel.
These are new families relative to the twelve original S5/A5 presentations.

Use disjoint cyclic-5 and S3 order-six development cases for code checks,
with only the length-two anchors to exercise paid and inferred answers.
Run them again with depth-eight labels for the saturated control and with a
one-unit solver budget for the unknown control. Require zero eager solver
startups and zero paid queries on the saturated inputs and no accepted
unknown inference. Reject a corrupted reply, a corrupted table and a
forged witness. No estimation of power is needed for an exhaustive check
of a fixed collection of exact outcomes. There is no population inference,
adaptive sampling, extra confirmation seed, or finer rung selection.

Pin Z3 4.15.4 and Python 3.13. Run one thread and at most one fresh solver
process per solver case, launched at the first unresolved word. Rotate
the arm order by panel index. Include process
startup, Python inference, and child CPU in the charge. Five seconds and
ten million solver resource units per check are maximums. Record paid
labels, input symbols, per-kind solver counts, CPU and elapsed time for
each case. Compute time on this CPU includes Python-to-Z3 translation and
subprocess overhead. Collect full SAT witnesses only on a separately
executed replay, excluding audit overhead from the timed comparison.
Verify that replay matches the scored proposal, decision, endpoint and
action tapes and check every witness independently.

The exactness gate is all 16 cases and all 630 right transitions per arm,
four anchors per case, every teacher interaction and every SAT witness.
Require zero solver unknowns for a directional result against Z3.

- Against visible-state exploration, a material overall saving requires
  at least 10% fewer total labels at no more than 1.25× the other arm's
  aggregate CPU time. Apply the gate to either direction. Report all
  input symbol counts as a distinct resource.

- In the matched-policy comparison, a material label saving requires
  at least 10% fewer total labels.
  Report additional compute for those savings separately.
- A material compute saving requires at least 5× lower total CPU time
  with the same total labels. In this design equal label counts imply
  identical charged words, and hence identical input symbol counts.
- Include per-family results and the observed break-even CPU time per
  additional label, when defined. Classify all other exact
  outcomes as a tie or trade-off. An inexact run, unknown or failed
  independent check cannot support a win.

Freeze this protocol before scoring. The one scored panel is reused only
for replay, never for changing the implementation or selecting a stronger
claim. If a defect prevents scoring, archive the refusal.
