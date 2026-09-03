# Campaign Pre-Registration Template

**Use this before running any mechanism-vs-baseline comparison campaign.** It
exists because of the braid–Dyck arc: a mechanism that was SUPPORTED at one
budget (+0.144) was REFUTED at a larger budget (−0.245) — the first result was
real but measured against a *floored* baseline, so it answered a different
question than the hypothesis claimed. This template makes the rung explicit,
splits the two questions a comparison actually asks, and pre-commits the
stopping rule so the verdict is trustworthy whichever way it lands.

Read `research_loop.md` first for the overall loop and `new_mechanism_checklist.md`
if the mechanism itself is new. Copy the form in §7 into the campaign bead's
description and fill it **before any evidence exists** (the registration commit
must predate every qualifying artifact — `mgr hypotheses validate` enforces
append-only governance, and the bead description is the pre-registration
record for the experiment-design choices the registry schema does not capture).

## 1. The core principle: one comparison is two claims

"Does mechanism X beat the baseline?" is two questions that can have **opposite
answers**:

| claim | tested at | the win means |
|---|---|---|
| **Sample-efficiency** | a **floored** rung (baseline stuck at the answer prior) | X reaches competence at lower budget than the baseline |
| **Asymptotic** | an **off-floor** rung (baseline reliably clears the prior, tight variance) | X is better once *both* arms are competent |

Braid–Dyck was **sample-efficiency = yes** (+0.144 where standard was floored)
and **asymptotic = no** (−0.245 where standard learns the task). Both are
real, interesting, and registrable — but a single hypothesis tested at one
arbitrary rung silently conflates them, which is how a floored win gets
mistaken for a structural one. **Register the claim(s) you actually mean, name
the rung in `scale_caveats`, and never let a floored rung answer an asymptotic
question.**

## 2. Phase 0 — find the rung *before* you believe anything (mandatory)

A comparison is only meaningful once you know where the baseline clears the
floor. Run a **sizing probe**: a small budget ladder on the *standard*
baseline for the target task, reporting the budget at which its held-out
exact-match reliably exceeds the answer prior.

**Clear the *recorded* `answer_prior`, not the registered fallback — they
differ, and the recorded one is usually higher (stricter).** The engine prefers
the per-eval `answer_prior` an artifact records (the best constant-answer score
on the exact docs scored) over the registered `validity.baseline_floor`
fallback. For Dyck the gap is the whole lesson: registered fallback **0.512**
vs recorded **0.625**, and standard's best E1 seed was **0.604** — *above* the
fallback, *below* the recorded prior, i.e. **still floored** even though the
registered number says otherwise. Read the recorded `answer_prior` straight
from the probe's `summary.json` and clear *that*. (Registered fallbacks, for
orientation only: hier 0.014, arith 0.521, dyck 0.512, rel 0.059, rot 0.049,
bag 0.336, needle 0.014, copyops 0.001.)

- Route probe runs to **`artifacts/probes/sizing/`** — the verdict engine
  refuses that path by construction (bead dzor), because a run that *selects*
  a rung must never *adjudicate* it (selection bias). `scripts/run_campaign.py`
  always writes to `artifacts/campaigns/<topic>/` (it does not expose an
  artifacts-dir flag), so either `git mv` the probe output into
  `artifacts/probes/sizing/` afterward (what dzor did), or run the probe with
  `python -m nanochat.train … --artifacts-dir artifacts/probes/sizing` (and
  `mgr eval-tasks … --artifacts-dir artifacts/probes/sizing`) directly.
- Read the per-seed EM **distribution shape**, not just the mean:
  - *unimodal, tight, above the prior* → off-floor rung found; good for the
    asymptotic claim.
  - *bimodal near the prior* (some seeds learn, some sit at the floor) → still
    floored; this is the variance trap of §4 — go up a rung, do **not** just
    add seeds.
- Record the chosen rung and the probe evidence in the bead. The probe stays
  out of the evidence pool forever.

(`77l.3` is the standing dose-response instrument for this; it produces the
budget ladder and the per-rung floor-clearance readout.)

## 3. Phase 1 — register the hypotheses (pre-evidence)

```bash
uv run mgr hypotheses add --id hyp-<mech>-<task>-<claim> \
    --statement "..." --mechanism <mech> \
    --source-kind model --provenance "artifacts/proposals/<id>/scoring.md" \
    --metric-path "evaltasks:tasks.<task>.exact_match.greedy.held_out.mean" \
    --comparator ">=" --threshold 0.05 --threshold-kind absolute_delta \
    --baseline-mechanism standard --min-seeds <fixed floor, never edited; §4> \
    --baseline-floor <task floor> --floor-source "<how computed>" \
    --scale-caveats "asymptotic claim: rung d128/L4 @ <off-floor budget> from probe <bead>; never adjudicate at a floored rung"
uv run mgr hypotheses validate
```

- **Asymptotic claim**: register at the off-floor rung. Comparator `>=`,
  threshold the registered effect size, baseline `standard`, floor set.
- **Sample-efficiency claim** (optional second entry, distinct id): register at
  the floored rung; here a positive EM delta *with the baseline below floor* is
  the point, so the floor gate is informational rather than disqualifying —
  state in `scale_caveats` that the floored baseline is intended.
- **Non-inferiority** variant (when the claim is "X does not lose"): register
  a negative `--threshold` with `absolute_delta` ("loses by no more than ε").

## 4. Phase 2 — size the campaign by power; diagnose *which arm* drives the variance

`min_seeds: 3` is convention, not a power analysis. Two distinct knobs are
easy to conflate:

- **`min_seeds` (registry)** — a *fixed floor* the engine refuses to
  adjudicate below, registered once and **never edited** (the governance
  spirit: "claims should be retired and re-registered, not morphed"; 8h0e
  kept `min_seeds: 3` throughout while actually running n=25 then n=8). Set it
  pre-evidence to a sensible floor; you can always run *more*.
- **How many seeds you actually run** — governed by the bead's pre-registered
  adaptive rule (Phase 3), informed by the wave-1 power gate. *This* is what
  power-sizing drives, not the registry field.

The power gate cannot see the quarantined probe — `mgr hypotheses power` runs
the verdict engine's collector, which *skips* `artifacts/probes/sizing/`. So
the two-step is: (1) eyeball the probe's per-seed variance from its
`summary.json` to pick a sane `min_seeds` floor at registration; (2) once
wave-1 of the *non-quarantined* campaign lands, run the formal gate to drive
the adaptive rule:

```bash
uv run mgr hypotheses power -H hyp-<mech>-<task>-<claim>   # reads indexed evidence only
```

It reports achieved power for the **registered effect size** at the observed
per-arm variance, and the per-arm seed count for 80% — the trigger for the
Phase-3 adaptive rule.

**A large required-n means high variance in *some* arm — diagnose which before
reacting** (look at each arm's per-seed spread, not just the headline n):

- **Baseline near the floor, bimodal** (some seeds learn, some sit at the
  prior) → the *rung* is wrong. Move up; the variance collapses; do not grind
  seeds. This is the `sm47` lesson (seed expansion could not resolve the +0.05
  Dyck claim while the standard baseline was floored).
- **Candidate arm wide while the baseline is tight and off-floor** → the
  *mechanism* is unreliable at this task, and a rung change will **not** fix
  it. That is itself a finding. Braid–Dyck E2 is exactly this: standard
  sd 0.027 (tight, off-floor at 0.908), braid sd 0.096 (three seeds below the
  0.625 prior), required n≈32 — driven by the candidate, not the baseline. The
  right move was *not* more seeds and *not* a rung change: the effect was so
  large (−0.245, CI excludes the threshold by >3×) that the refutation was
  decisive at n=8 despite the "29% power" headline.

Power is computed for the registered effect *regardless of observed sign*. The
UNDERPOWERED qualifier (power-to-*confirm* below the floor) is therefore the
adequacy lens for a **SUPPORTED** verdict only. A **REFUTED** verdict instead
records **`refutation_margin`** = |effect − threshold| / CI-half-width (how
decisively the CI excludes the threshold; > 1 by construction) — so a decisive
refutation is never mislabeled "underpowered" (ci-v6, bead 4b82, which fixed
exactly the 8h0e braid–Dyck case: REFUTED-UNDERPOWERED@29% under ci-v5 →
`refuted, refutation_margin=3.6×` under ci-v6).

## 5. Phase 3 — pre-register the adaptive / stopping rule (the only stopping rule)

Optional-stopping is the easiest way to manufacture a false positive. Write the
**entire** stopping rule into the bead before launch. The `8h0e` pattern:

> After wave-1 evals, run `mgr hypotheses power`. If achieved power for the
> registered effect < 80%, train seeds N+4…N+7 per arm **once**, then
> adjudicate exactly once. The n-after-extension power reading does **not**
> trigger a second extension — the rule names the single extension, not an
> iterate-to-80% loop.

Anything not in the rule is forbidden: no peeking-and-extending, no
post-hoc rung changes, no re-adjudicating until the verdict looks clean.

## 6. Phase 4 — the placebo arm (mandatory for "captures structure X" claims)

If the claim is that X exploits some structure (hierarchy, brackets,
composition), add a **structure-free placebo corpus** arm at equal budget. If X
"wins" on placebo data too, the win is not about that structure. (This exposed
the surreal optimization-rate confound; the hyperbolic predictions bake in a
curvature-collapses-on-placebo control.)

## 7. Phase 5 — run, eval, adjudicate (once)

```bash
# train both arms from a frozen worktree (clean provenance, immune to other agents' dirty trees)
uv run python scripts/run_campaign.py \
    --combo <task>:<mech> --combo <task>:standard \
    --seeds <list> --target-flops <off-floor budget> --topic <topic> --sha <clean-sha>
# eval each checkpoint from the SAME frozen worktree (>= producer SHA)
uv run mgr eval-tasks --checkpoint <ckpt> --task <task> --seeds 0,1,2 --run-id <id>
# adjudicate EXACTLY ONCE, -H only (never --all: it moves verdicts you weren't looking at)
uv run mgr adjudicate -H hyp-<mech>-<task>-<claim> --dry-run   # inspect
uv run mgr adjudicate -H hyp-<mech>-<task>-<claim>             # append, on a CLEAN tree
```

Pre-flight invariants (each has cost a session): `--checkpoint-interval > 0`
(or no final checkpoint is saved); eval worktree SHA ≥ checkpoint producer's
SHA; equal `--target-flops` (5% tolerance), arms differ in exactly one axis;
clean, in-sync tree before the ledger append (don't write the ledger over
another agent's uncommitted registry/engine edits).

## 8. The fill-in form (copy into the campaign bead, before evidence)

```
CAMPAIGN PRE-REGISTRATION — bead <id>
Mechanism: <X>        Baseline: standard        Task: <task>
Claim type: [ ] sample-efficiency (floored rung)  [ ] asymptotic (off-floor rung)
Hypothesis id(s): <hyp-...>
Metric: evaltasks:tasks.<task>.exact_match.greedy.held_out.mean   threshold: <ε>  comparator: >=
Answer-prior floor: <task floor>   (engine prefers artifact-recorded answer_prior)
Window: sequence_len <N>; document tokens in-range <n> / held-out <m> (measured, task tokenizer);
        held-out overflow is [ ] the test (extrapolation claim)  [ ] a defect (choose N >= m)

PHASE 0 — rung: probe <bead/path in probes/sizing/>; standard clears floor at <budget>;
          per-seed shape at chosen rung: <unimodal-tight | bimodal-near-floor>; chosen rung: <budget>
PHASE 2 — pilot variance: <sd>; mgr hypotheses power → <power>% at n=<pilot>, needs n=<N> for 80%;
          min_seeds set to <N>   (if N absurd: rung moved to <budget> — record old/new)
PHASE 3 — stopping rule (verbatim, the ONLY rule): <one-extension spec or fixed-n>
PHASE 4 — placebo arm: [ ] yes corpus=<...>  [ ] N/A (claim is not structure-capture, justify)
PHASE 5 — frozen SHA: <sha>; topic: <topic>; adjudicate: -H <hyp> once
Pre-registration commit: <sha/date, BEFORE any artifact>
```

## 9. Anti-patterns (each one has actually happened or nearly happened)

- **Floored-win misread** — believing a comparison run at a rung where the
  baseline is at the prior. (braid–Dyck E1.) → Phase 0.
- **Seed-grinding at a bad rung** — adding seeds when the variance is floor
  bimodality. (sm47.) → Phase 4, move the rung.
- **Optional stopping** — peeking at power/effect and extending until it looks
  good. → Phase 3, the rule is fixed a priori.
- **Post-hoc crossover** — after a refutation, finding the intermediate rung
  where X wins and registering *that* on the existing data. → a crossover
  claim must be registered for a genuinely *untested* rung, pre-evidence.
- **`--all` adjudication** — recomputing every hypothesis against a grown pool
  and moving verdicts you weren't testing. → `-H` only.
- **Probe contamination** — sizing runs in the default evidence pool. →
  `probes/sizing/` quarantine (dzor).
- **Dirty-tree ledger write** — appending a verdict over another agent's
  uncommitted registry/engine edits. → clean, in-sync tree first.
