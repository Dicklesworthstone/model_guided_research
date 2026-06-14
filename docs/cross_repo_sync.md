# Cross-Repo Change Watch & Sync Cadence

Bead: `model_guided_research-qha`

A lightweight, manual process for keeping `model_guided_research` aware of
improvements landing in the sibling `bio_inspired_nanochat`
(`/data/projects/bio_inspired_nanochat`), and deciding what to port. Pairs with
the catalogue in `docs/cross_repo_ideas.md` (bead `23x`).

## What to watch (in priority order)

| Area | bio paths to diff |
|---|---|
| Training loop / scheduling | `scripts/base_train.py`, `scripts/mid_train.py` |
| Optimizers | `bio_inspired_nanochat/{adamw.py, muon.py}` |
| Checkpoint / resume / RNG | `bio_inspired_nanochat/checkpoint_manager.py` |
| Stability | `bio_inspired_nanochat/divergence_guard.py` |
| FlexAttention | `bio_inspired_nanochat/flex_synaptic.py`, `scripts/{verify_flex_correctness,benchmark_flex}.py` |
| Dataloader / cache | `bio_inspired_nanochat/dataloader.py`, dataset code |
| Telemetry | `bio_inspired_nanochat/run_logging.py` |
| Kernels | `bio_inspired_nanochat/kernels/` |
| Memory tooling | `scripts/{scale_memory,param_census}.py` |

## Cadence

- **Trigger-based, not calendar-based.** Re-sync when (a) starting a GPU
  session, (b) starting perf/optimizer work, or (c) a bio change is announced.
  A monthly glance is a reasonable floor if nothing else triggers it.
- **Owner:** whoever picks up a perf/infra bead does the sync first as context.

## How to summarize a delta

Recommended one-liner to surface what changed in the watched areas since a known
commit (run from the bio repo):

```bash
git -C /data/projects/bio_inspired_nanochat log --oneline --since="<date>" -- \
  scripts/base_train.py bio_inspired_nanochat/{adamw.py,muon.py,checkpoint_manager.py,divergence_guard.py,flex_synaptic.py,run_logging.py}
```

For a structural diff of a specific file pair, compare directly:

```bash
diff <(sed -n '1,400p' /data/projects/bio_inspired_nanochat/bio_inspired_nanochat/checkpoint_manager.py) \
     <(sed -n '1,400p' /data/projects/model_guided_research/nanochat/checkpoint_manager.py)
```

(No auto-codemod — diffs inform a human/agent decision; ports are made manually,
gated by the goldens harness and quality gates.)

## How to record a decision

- **Port it:** file a `br` bead (`discovered-from:` the relevant idea), link the
  bio source ref, and update the row in `docs/cross_repo_ideas.md`.
- **Skip it:** add a one-line rationale to the idea-bank row (e.g. "GPU-only,
  deferred" or "superseded by our certify loop") so it isn't re-evaluated cold.

## First sync summary (2026-06-14)

Baseline established by the `23x` mining pass: 14 techniques catalogued, top-3
CPU quick wins identified (RNG-state resume, atomic JSON meta-write,
`expandable_segments`). bio is ahead on GPU/perf/robustness plumbing; this repo
is ahead on the math mechanisms, the certify/adjudication research loop, and the
fixed-FLOPs benchmark + regression-gate infra. Next sync trigger: the next GPU
session or perf-bead pickup.
