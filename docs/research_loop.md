# The AI-Guided Research Loop: an Executable Protocol

The project's thesis (README "Genesis") is a meta-cognitive loop: an AI
proposes mathematical research directions, scores them, helps implement them,
and then — the part this document operationalizes — **submits them to an
adjudication engine that can prove it wrong**. This page turns the
origin-story prose into the protocol future sessions actually run, so the
registry grows instead of the README accreting untracked claims.

Everything below is enforced by tooling that exists today: the hypothesis
registry (`hypotheses/registry.yaml`, append-only, validated against git
HEAD), the verdict engine (`mgr adjudicate`, policy `ci-v9`), the campaign
launcher (`scripts/run_campaign.py`, frozen-worktree provenance), and the
beads tracker (`br`).

**Two companion documents make this loop reliable, and you should use them:**

- `campaign_preregistration_template.md` — the form to fill (before evidence
  exists) for any mechanism-vs-baseline comparison. It forces the
  rung-finding probe, the sample-efficiency-vs-asymptotic split, the
  power-derived seed count, and a single pre-registered stopping rule. **Use
  it for every comparison** — it is the direct fix for the floored-win trap
  that the braid–Dyck arc exposed (§7, worked example).
- `new_mechanism_checklist.md` — the merge gate every new attention mechanism
  passes before its numbers are trusted (exact reduction to a known
  mechanism as a certify check, placebo control, parameterization
  coordinate-check, validate-before-write, numerics policy, interpretability
  observable, goldens recapture).

## The loop, step by step

```
propose → score → register (PRE-EVIDENCE) → bead → implement → campaign
   ↑                                                              ↓
   └────────── beliefs update ←── ledger append ←── adjudicate ←──┘
```

### 1. Propose

Proposals come from humans or models, usually as a falsifiable consequence of
one of the theory notes in `markdown_documentation/`. A proposal is not an
idea ("braid attention is good") but a *measurable contrast* ("braid beats
standard on held-out Dyck EM by ≥ +0.05 at equal FLOPs").

### 2. Score (the reusable rubric prompt)

The original GPT-5 Pro sessions scored proposals 0–1000. The reusable prompt
template, verbatim dimensions from the README:

> Score this research proposal on five dimensions, 0–100 each:
> **Theoretical Novelty** (how innovative is the mathematical approach?),
> **Practical Feasibility** (can this be implemented efficiently?),
> **Potential Impact** (could this revolutionize AI?), **Mathematical Rigor**
> (how solid is the theoretical foundation?), **Implementation Clarity** (how
> clear is the path to implementation?). Then give a composite 0–1000 as a
> weighted sum, weighting theoretical novelty and potential impact most
> heavily. Justify each dimension score in 2–3 sentences before giving the
> number. State the single most likely way the proposal FAILS.

Archive scoring transcripts under `artifacts/proposals/<hypothesis-id>/`
(markdown or JSON) — that is the provenance trail for `source.kind: model`
registry entries, which must name where the claim came from.

### 3. Register — BEFORE evidence exists

```bash
uv run mgr hypotheses add --id hyp-<mechanism>-<claim-slug> \
    --statement "..." --mechanism braid \
    --source-kind model --provenance "artifacts/proposals/<id>/scoring.md" \
    --metric-path "evaltasks:tasks.dyck.exact_match.greedy.held_out.mean" \
    --comparator ">=" --threshold 0.05 --threshold-kind absolute_delta \
    --baseline-mechanism standard --min-seeds 3 \
    --baseline-floor 0.512 --floor-source "population majority-answer prior ..."
uv run mgr hypotheses validate   # append-only governance vs git HEAD
```

The registry validator checks schema and append-only fields. The verdict
engine verifies the committed operational definition and the producer history
before scoring evidence.

- **Pre-evidence registration.** The report's `registration_commit` identifies
  the commit introducing the current statement, population and executable
  prediction. A change to a threshold or selector starts a new eligibility
  window, as does promotion from a null prediction. Descriptive source notes
  and `scale_caveats` can be revised without changing that commit. A reviewer
  must still check substantive constraints recorded only in prose.
  A successor has its own registration. The adjudicator follows merged Git
  history and requires a clean producer revision on a path from registration
  to the checkout's HEAD. A run launched from the registration commit itself
  may qualify. An identical artifact already committed there at its recorded
  path belongs to the discovery set. The report lists excluded measurements
  in `excluded_evidence` with paths and machine-readable causes.
- **Falsification criteria in the prediction**: metric path, comparator,
  threshold, baseline arm, `min_seeds`. Exact-match claims register a
  `validity.baseline_floor` (the answer-prior floor; see the worked example
  for why).
- **min_seeds should be power-derived** once pilot variance exists:
  `mgr hypotheses power -H <id>` reports achieved power for the registered
  effect and the seed count 80% needs.
- A claim that cannot be operationalized yet registers
  `prediction: null` + `operationalization_note` and status `blocked` —
  visible debt, never silent omission.

### 4. Bead

Every registered hypothesis gets a bead (`br create`), and follow-on work
links provenance with `--deps discovered-from:<parent-bead>`. Conventions:

- **Hypothesis ↔ bead cross-reference**: the bead description names the
  hypothesis id; the registry entry's campaign artifacts
  (`artifacts/adjudications/<bead>/campaign.md`) name the bead.
- Campaign protocols (arms, rung, seeds, *and any adaptive rule*) are
  written into the bead description **before launch** — the bead is the
  preregistration document for experiment-design choices the registry
  schema does not capture.

### 5. Implement & run the campaign

**Before launching, fill `campaign_preregistration_template.md` into the
bead.** A comparison is only meaningful at a rung where the baseline clears
the answer-prior floor, and "does X beat the baseline?" is really two claims
(sample-efficiency at a floored rung vs asymptotic at an off-floor rung) that
can have opposite answers. Phase 0 of the template is a quarantined
**rung-finding sizing probe**; skipping it is how a floored win gets mistaken
for a structural one (§7).

```bash
uv run python scripts/run_campaign.py \
    --combo dyck:braid --combo dyck:standard --seeds 30,31,32,33 \
    --target-flops 3e14 --topic e2-dyck
```

The launcher freezes a **detached worktree at a clean SHA** to isolate a
campaign from edits in the main checkout. The collector excludes artifacts
flagged as tainted, and task evaluations propagate recorded training taint.
For task evaluations and charge probes, record the training run ID, checkpoint
step and clean training and evaluation SHAs. The training commit must be at
or after registration and reachable from the evaluation commit. A later
evaluation of earlier weights, an unknown SHA or a resumed run with unverified
parent weights cannot qualify. Derived benchmarks with unverified source
inputs receive `missing_input_lineage`. [Original audit](bridge_plan.md#november-2026-assessment)

Rules that have each cost a session to learn:

- `--checkpoint-interval` must be > 0 or no final checkpoint is saved.
- Evals run from a worktree at **≥ the checkpoint producer's SHA** (config
  fields added later break older loaders).
- **Sizing probes are not evidence.** Runs used to *choose* a rung or budget
  go under `artifacts/probes/sizing/` — the collector refuses that path by
  construction, because a run that selected the experiment cannot also
  adjudicate it (selection bias). `artifacts/probes/charges/` stays
  evidence: chargeprobe instruments measure, they don't select.

### 6. Adjudicate

```bash
uv run mgr adjudicate -H hyp-braid-dyck-depth-extrapolation --dry-run  # inspect
uv run mgr adjudicate -H hyp-braid-dyck-depth-extrapolation           # append
```

The engine (policy `ci-v9`) is the only writer of verdicts. It refuses weak
evidence (BLOCKED with machine-readable reasons) rather than soft-ruling; it
counts **one observation per trained checkpoint** (eval seeds are repeated
measurements); it groups evidence into equal-FLOPs **budget cohorts** so
bigger-budget campaigns supersede smaller ones by appending, never editing;
it downgrades refutations of floored baselines to INCONCLUSIVE
(`floor_effect`); and it stamps every arm with achieved **power**, a
one-sided **p-value**, and — for **SUPPORTED** arms — the **UNDERPOWERED**
qualifier when a clean-looking confirmation came from a test that couldn't
have detected the registered effect. A **REFUTED** arm instead records
**`refutation_margin`** = |effect − threshold| / CI-half-width (how decisively
the CI excludes the threshold; > 1 by construction), because the
power-to-*confirm* lens is the wrong adequacy measure for a refutation
(ci-v6, bead 4b82). Run-level reports add Benjamini–Hochberg q-values: the
headline is always "N supported, of which M survive FDR at q=0.10".

Prefer `-H <id>` over `--all` unless you intend a full-ledger re-adjudication:
`--all` recomputes *every* hypothesis against the grown evidence pool, which
can move verdicts you weren't looking at.

Training budgets count model forward and backward work. In a paired CPU step
at width 32 and depth 2, AdamW and HOSS both received an estimate of 11,796,480
FLOPs and started at loss 4.8520303. PyTorch counted 39 matrix multiplies and
12 batched matrix multiplies for AdamW, versus 825 and 252 for HOSS. The extra
work comes from differentiating gradients during the optimizer step. These
operator counts measure this particular configuration. Since `ci-v8` the engine refuses
mixed HOSS and first-order comparisons from training or task-eval summaries
with `optimizer_compute_unmeasured` when both arms carry the registered
metric. Bead `sequ` tracks calibration across ranks and architectures.

Policy `ci-v9` selects the registered evaluation task before choosing among
repeat measurements. Scorecard cells reuse names such as `seed_0` across
tasks. Selecting the newest file first could replace a regime measurement
with a later placebo measurement. Optimizer and scheduler comparisons require
checkpoint metadata identifying the relevant axis on both arms. Each new
evaluation carries the training optimizer and scheduler forward.

#### November 2026 historical replay

Two DSR dry runs used the same 75 registry entries and 526 tracked artifact
summaries from `4db6657`. The rows show the `ci-v6` results. The columns show
the `ci-v7` results after the registration and checkpoint checks.

| Previous verdict | Supported | Refuted | Inconclusive | Blocked | Total |
|---|---:|---:|---:|---:|---:|
| Supported | 10 | 0 | 0 | 1 | 11 |
| Refuted | 0 | 1 | 0 | 2 | 3 |
| Inconclusive | 1 | 0 | 6 | 4 | 11 |
| Blocked | 0 | 0 | 0 | 50 | 50 |
| **Total** | **11** | **1** | **6** | **57** | **75** |

There were 214 exclusions across 147 distinct files. Of those
hypothesis–file pairs, 101 predated registration and 113 lacked training
lineage. For `hyp-surreal-optimization-rate`, six summaries predated its
registration at `1c6abc5` and seven lacked training lineage. The remaining
evidence could not meet the registered five-training-run minimum per arm at a
common budget. The placebo registry prediction moved from INCONCLUSIVE to
SUPPORTED on 43 selected summaries. Its separate fixed-budget scorecard and
publication decision still require their own evidence and owner disposition.
The path audit includes each hypothesis, its registration commit where
applicable, both decisions and the files each policy selected or excluded.
[Machine-readable replay](../artifacts/adjudications/jida36-eligibility-replay/eligibility-audit.json)

The append-only ledger still records 14 SUPPORTED, 4 REFUTED, 14 INCONCLUSIVE,
7 BLOCKED and 36 OPEN. Interpret earlier verdicts under their stamped policy.
Both runs respected the four owner holds on `hyp-braid-dyck-depth-extrapolation`,
`hyp-braid-dyck-directional`, `hyp-symplectic-nonorm-depth-tied` and
`hyp-symplectic-nonorm-depth-untied`. The directional braid hold therefore
has no new ruling in this replay. The prior audit found that 18 of its 50
cited evaluations predated registration. A later replication needs its own
evidence and an explicit owner decision on the held report.

### 7. Beliefs update → next proposals

`mgr report --artifacts <dir>` rolls a campaign into arm tables (clean,
lineage-deduped, the engine's own semantics).

**The load-bearing lesson — a floored win is not a structural win.** The
braid–Dyck claim was SUPPORTED at E1 (+0.144) and then REFUTED at E2 (−0.245,
the ledger's first supersession with a status flip) once a larger budget let
the *standard* baseline actually learn the task. The E1 win was real but it
measured **sample efficiency** (braid reaches competence where standard is
floored), not the **asymptotic** superiority the hypothesis claimed. Before
believing any comparison: confirm at an off-floor rung, and register the two
claims separately (`campaign_preregistration_template.md` §1). Verdicts then
feed the next round:

- **SUPPORTED** → scale rung up, or sharpen into a mechanism question
  (score-half vs aggregate-half, dose-response, ablations).
- **REFUTED, powered** → retire, or register a *revised successor* under a
  new id when the refutation exposed a protocol artifact rather than a dead
  mechanism (precedent: `hyp-padic-truncation-graceful` REFUTED on a
  matched-memory window that structurally could not see the float cliff →
  successor `hyp-padic-truncation-graceful-k16` registered fresh,
  pre-evidence, with the asymmetric knowledge disclosed — then SUPPORTED).
  Never morph the refuted entry; the ledger keeps both.
- **INCONCLUSIVE / UNDERPOWERED** → `mgr hypotheses power` says how many
  seeds the claim needs; if that is absurd, diagnose *which arm* is wide
  (`campaign_preregistration_template.md` §4): a floored, bimodal baseline
  means move the rung (not more seeds); a wide *candidate* arm with a tight
  off-floor baseline means the mechanism is unreliable at this task — a
  finding in itself, and if the effect is large the verdict can still be
  decisive (braid–Dyck E2).

## Worked example: the braid–Dyck claim, end to end

The most instructive arc in the ledger, because it exhibits the loop *and*
its honest failure-mode corrections. All artifacts are in-repo.

1. **Propose/score**: braid attention's crossing structure should track
   Dyck-bracket stack topology (theory note
   `markdown_documentation/knot_theoretic_programs_and_braid_based_attention.md`;
   README braid scoring).
2. **Register**: `hyp-braid-dyck-depth-extrapolation` — held-out Dyck EM,
   `>= +0.05` vs standard at equal FLOPs, min_seeds 3, floor 0.512.
3. **First contact (pilot1)**: every EM hypothesis "refuted" — by a
   degenerate baseline stuck at the answer prior. The *policy* was wrong,
   not the claims: ci-v2 added the floor gate, and pilot1's refutations were
   superseded to INCONCLUSIVE by appending (`artifacts/adjudications/pilot1-civ2/`).
4. **E1 campaign (hqwi)**: SUPPORTED, +0.144 [+0.093, +0.195], n=25/arm —
   but the baseline sat at the floor (best standard seed 0.604 vs 0.625
   prior), so this proves braid *learns where standard cannot*, not the
   sharper off-floor contrast.
5. **Seed expansion (sm47)**: more seeds at E1 could NOT decide the +0.05
   claim — baseline bimodality at the floor drives the variance. ci-v4's
   power instrument now quantifies this in one line:
   `mgr hypotheses power` reads 51% power at n=25/arm, needs ~50. The right
   move was never more seeds.
6. **Sizing probe (83r6)**: found the rung where standard learns Dyck —
   d128/L4 @ 3e14 (EM 0.917/0.906 vs 0.625 prior). Probe runs quarantined
   under `artifacts/probes/sizing/` — they chose the rung, so they cannot
   adjudicate it.
7. **Decisive campaign (8h0e)**: preregistered in the bead before launch —
   arms, seeds 30–33, *and the adaptive rule* (if wave-1 power < 80%, add
   seeds 34–37 before adjudicating; adjudicate exactly once). Outcome:
   **REFUTED, −0.245 [−0.326, −0.164], n=8/8 at 3e14** — the 3e14 cohort
   superseded the floored E1 SUPPORTED via the budget-cohort rule (the
   ledger's first supersession with a status flip). At the off-floor rung
   standard wins decisively; three braid seeds even fell *below* the 0.625
   prior. The E1 win was real but it measured **sample efficiency**, not the
   **asymptotic** superiority the hypothesis claimed — which is exactly why
   the two are now registered as separate claims (§7,
   `campaign_preregistration_template.md` §1). At adjudication time (ci-v5)
   the verdict carried a misleading REFUTED-UNDERPOWERED stamp — power is
   measured against the registered +0.05 regardless of the observed sign, so
   a decisively-negative result read as 29% power. Bead 4b82 fixed exactly
   this: under **ci-v6** the UNDERPOWERED qualifier fires on SUPPORTED arms
   only, and a refutation instead records **`refutation_margin = 3.6×`** (how
   many CI-half-widths the interval sits past the threshold) — the decisive
   refutation is no longer mislabeled weak.

Compact second example (`9qk3`, dequantization annealing): theory note →
operationalized prediction with **variant selectors** (both arms tropical,
distinguished by recorded `semiring_beta_spec`) → frozen-worktree campaign →
SUPPORTED 0.9997 [0.9991, 1.0] — annealing to the certified tropical endpoint
costs nothing (`artifacts/adjudications/9qk3/campaign.md`).

## Conventions summary

| Concern | Convention |
|---|---|
| Hypothesis ids | `hyp-<mechanism>-<claim-slug>` |
| Bead ↔ hypothesis | bead description names the hypothesis id; campaign.md names the bead |
| Scoring transcripts | `artifacts/proposals/<hypothesis-id>/` |
| Campaign artifacts | `artifacts/campaigns/<topic>/<task>-<mech>-s<seed>/` |
| Eval evidence | `artifacts/evals/tasks/<eval-run-id>/` |
| Sizing probes | `artifacts/probes/sizing/` — engine-refused by path |
| Verdict reports | `artifacts/adjudications/<run-id>/` (verdicts.json, report.md, campaign.md) |
| Refuted → successor | new id, registered pre-evidence, asymmetric knowledge disclosed; never edit the refuted entry |
| Registry edits | only via `mgr hypotheses add` / engine appends; `mgr hypotheses validate` gates every change |
