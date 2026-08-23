# Scaling sweep harness (`mgr scaling-sweep`)

Bead `model_guided_research-w94.1`. One command trains a single attention
mechanism across a resumable model-size ladder and records everything needed
for later scaling fits (consumed by E2 / w94.2 / lab.2).

## Why it exists

Scaling-law comparisons need many runs per mechanism across a parameter
ladder; individual rungs can take hours. The sweep is therefore resumable at
**two** levels:

1. **mid-sweep** - `manifest.json` tracks per-(rung, seed) status;
   re-invoking with the same `--run-id` skips completed work;
2. **mid-rung** - when checkpoints were enabled (default), an interrupted
   seed continues via nanochat D1 resume (`--resume-from latest`), which also
   restores the ORIGINAL step budget from checkpoint meta regardless of the
   retry's command line.

## Ladder design (documented rationale)

| invariant | value | why |
|---|---|---|
| head_dim | 32 on every rung | widest power-of-two that keeps the narrowest rung (n_embd=64) multi-head; %8 keeps quaternion (%4) and octonion (%8) feasible everywhere |
| n_head | n_embd // 32, always even | reversible requires even n_head |
| aspect ratio | n_embd / n_layer = 32 | depth grows in lockstep with width (deep-narrow ladder shape) |
| vocab | GPTConfig default 50304 | the FineWeb tokenizer id range requires it; embeddings therefore dominate small rungs - param counts are exact measured TOTALS, never estimated |
| GQA | n_kv_head = n_head, except reversible = n_head // 2 | reversible halves attention heads and requires n_kv_head to divide n_head // 2 |

Presets:

- `smoke` - same shapes as the two smallest research rungs but pinned tiny
  step counts (25/40). CPU plumbing verification in minutes; NOT science.
- `small` - ~6.5M / ~13.7M / ~32.0M params.
- `full` - adds ~59.9M and ~101.8M (~16x span).

Token budgets: research ladders spend `tokens ~= token_multiplier x params`
(default 20x, Chinchilla heuristic), expressed to `nanochat.train` as a
derived per-rung `--target-flops`. The feasibility table instantiates every
rung once up front (exact param counts + `estimate_flops`) and emits a
per-mechanism feasibility table; infeasible rows record the validator reason
instead of crashing. Mechanism notes (e.g. octonion wall-clock until 7b0.6,
gauge cached-eval until A5) ride along as SOFT gates - nothing is blocked.

## Artifact layout

```
artifacts/scaling/<mechanism>/<run_id>/
├── manifest.json                  # sweep-level state machine (schema below)
├── report.md                      # human-readable ladder + run tables
├── logs/
│   ├── <rung>_seed<k>.stdout.txt
│   └── <rung>_seed<k>.stderr.txt
└── rung_<i>/seed_<k>/             # one nanochat train run dir
    ├── summary.json               # train summary (results.losses, val_ce_final, ...)
    ├── metrics.jsonl              # D2 per-step metric stream (rz8.2 provenance)
    └── checkpoints/               # when --checkpoint-interval > 0 (D1 resume)
```

## Manifest schema (v1)

```jsonc
{
  "schema_version": 1,
  "suite": "scaling_sweep",
  "created_at": "2026-08-23T17:00:00Z",
  "updated_at": "...",
  "mechanism": "tropical",           // nanochat attention type
  "ladder": "small",                 // smoke | small | full
  "git": {...},                       // _get_git_info() provenance
  "python": {"executable": "...", "version": "..."},
  "argv": ["..."],                    // exact invocation
  "sweep_config": {                   // compatibility contract for resume
    "device": "cpu",
    "batch_size": 8,
    "sequence_len": 256,
    "token_multiplier": 20.0,
    "learning_rate": 0.0006,
    "optimizer_type": "adamw",
    "warmup_steps": 0,
    "val_interval": 0,
    "checkpoint_interval": 500,
    "checkpoint_keep": 1
  },
  "dataset_fingerprint": {            // corpus pinning (bead wiz scheme)
    "resolved": true,
    "data_dir": null,                 // null = FineWeb cache
    "n_files": 3,
    "train_files": 2,
    "val_files": 1,
    "files": [{"name": "shard_00000.parquet", "size_bytes": 123, "mtime_ns": 456}, ...],
    "digest": "<sha256 of sorted metadata json>",
    "method": "metadata-sha256(name,size,mtime_ns)"
  },
  "rungs": [
    {
      "index": 0,
      "name": "6M",
      "n_layer": 2, "n_embd": 64, "n_head": 2, "n_kv_head": 2,
      "max_steps": null,              // pinned steps (smoke) or null
      "feasible": true,
      "infeasible_reason": null,
      "notes": "soft-gate mechanism note or null",
      "param_count": 6537216,         // EXACT total incl. embeddings
      "flops_per_token_est": 20299776,
      "token_budget": 130744320,      // multiplier x params (research ladders)
      "planned_max_steps": 63840,     // ceil(tokens / (batch*seq)) or pinned
      "target_flops_est": 2654080409272320,
      "status": "pending",            // pending | running | done | failed | infeasible
      "runs": [
        {
          "seed": 0,
          "status": "done",           // pending | running | done | failed
          "summary_path": "artifacts/scaling/.../summary.json",
          "wall_seconds": 41231.2,
          "returncode": 0,
          "metrics": {"final_loss": 3.21, "val_ce_final": null, "tokens_per_second": 91011.2}
        }
      ]
    }
  ]
}
```

Status derivation: rung status is always recomputed from its seed runs -
`done` iff all seeds done; `failed` if any seed failed; `infeasible` only
from the feasibility gate. This keeps resumed sweeps honest (a stale `done`
can never mask a partially-failed rung).

## Resume semantics

Re-invoking with the SAME `--run-id` (and `--resume-sweep`, the default):

- hard-refuses (exit 2) when `mechanism`, `ladder`, or any sweep_config key
  above differs from the stored manifest - silently mixing sweeps would
  corrupt the scaling comparison; ladder-definition drift is likewise refused;
- WARNS when the dataset fingerprint digest changed since generation 1
  (completed rungs trained on different data);
- skips `(rung, seed)` pairs already `done`; retries `failed`/`running`
  pairs, adding `--resume-from latest --checkpoint-dir ...` when checkpoints
  exist so D1 continues mid-rung instead of restarting;
- `--fresh-sweep` ignores stored statuses entirely and retrains everything.

Exit codes: `0` all feasible work done; `1` any training failure OR zero
feasible rungs; `2` usage/refusal errors.

## Data

`--data-dir` accepts any parquet corpus following repo convention (sorted
files, LAST file = val split) - e.g. an `mgr gen-tasks` output directory for
fast CPU smoke runs. Default is the FineWeb cache with
`--auto-download-data` passthrough. Whatever corpus was actually read is
pinned in the manifest fingerprint.

## Quickstart

```bash
# plumbing check on CPU (point at a synthetic corpus; minutes)
mgr gen-tasks --task arith --out /tmp/corpus --size 300
mgr scaling-sweep --mechanism tropical --ladder smoke \
    --data-dir /tmp/corpus --no-auto-download-data \
    --batch-size 4 --sequence-len 64

# research ladder on GPU (Chinchilla-ish 20x token budget)
mgr scaling-sweep --mechanism gauge --ladder small --device cuda \
    --seeds 3 --val-interval 500

# inspect / fit later
jq '.rungs[] | {name, status, param_count}' artifacts/scaling/gauge/<run_id>/manifest.json
```

## Downstream consumer: `mgr scaling-report` (E2, bead w94.2)

The report command re-derives the science from these manifests ALONE (no
retraining):

```bash
mgr scaling-report --runs artifacts/scaling/<mechanism>/<run_id> [...] --out DIR
```

- reads `manifest.json` + each seed's `metrics.jsonl` (fallback:
  `summary.json` losses), reducing noise via a tail mean
  (`--tail-fraction`, default 0.1);
- fits L(C) = a*C^-b + c per mechanism (robust least squares; plain power
  law alongside), bootstraps exponent CIs over rungs with a deterministic
  seed (`--bootstrap-seed`, default 1729) so regeneration is byte-stable;
- runs pairwise bootstrap-overlap tests on exponents and renders an HONEST
  headline: when nothing separates, it says "NO pairwise exponent
  differences are significant" rather than manufacturing a winner;
- refuses saturating fits on <3 rungs (underdetermined theater);
- emits a `mgr.scaling.v1` JSON block (in `fits.json` and embedded in the
  markdown) for future G1 predictions / `mgr adjudicate`, plus a log-log
  overlay + per-mechanism panel PNG.

A committed example from the tropical smoke ladder lives in
`docs/scaling_report_smoke_tropical.md`.
