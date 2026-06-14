# Cross-Repo Idea Bank — `bio_inspired_nanochat` → `model_guided_research`

Bead: `model_guided_research-23x`

`bio_inspired_nanochat` (`/data/projects/bio_inspired_nanochat`) is a more
mature, GPU-oriented sibling. This is a living catalogue of concrete techniques
there that could transfer here, each with a source ref, current status in this
repo, effort, and a proposed next action. The deeper FlexAttention-porting
analysis lives in `docs/gpu_flex_diff.md`; this doc is the broader sweep.

> Line numbers are as-of the mining pass (2026-06-14) and may drift; treat the
> file as the anchor and re-locate the symbol if a line ref is stale.

## Quick wins (low effort, high value, CPU-feasible)

These three are the recommended first ports — all CPU-doable and reproducibility/
robustness wins, independent of the GPU work.

1. **RNG state capture/restore for bit-comparable resume** — `bio/checkpoint_manager.py:119-174`
   (`capture_rng_state` / `restore_rng_state`, saving torch CPU+CUDA, python, numpy
   RNG with `weights_only=True`). This repo's `nanochat/checkpoint_manager.py` has
   no RNG capture, so resumed runs are not bit-identical. **Next action:** port both
   functions and wire into `train.py` resume; gate on the attention goldens harness.
2. **Atomic JSON meta write as the commit point** — `bio/checkpoint_manager.py:106-116`
   (`_atomic_write_json`). This repo atomically writes the torch checkpoint but not
   the JSON meta; making the meta `os.replace` the *final* step prevents a crash from
   leaving a checkpoint whose meta disagrees with its weights. **Next action:** wrap
   meta write in an atomic helper and order it last. (We already use this pattern in
   the new `scripts/cmaes_phase1.py` `state/` checkpoints — reuse it.)
3. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — `bio/scripts/base_train.py:16`.
   One env line set before CUDA init; reduces fragmentation on long GPU runs. Missing
   here. **Next action:** set it at the top of `train.py` (no-op on CPU), document it.

## Full catalogue

| # | Technique | bio source | here | effort / GPU | next action |
|---|---|---|---|---|---|
| 1 | RNG state capture/restore | `checkpoint_manager.py:119-174` | missing | low / no | port + wire into resume (quick win) |
| 2 | Atomic JSON meta write | `checkpoint_manager.py:106-116` | partial | low / no | make meta the atomic commit point (quick win) |
| 3 | `expandable_segments` alloc flag | `scripts/base_train.py:16` | missing | low / no | set env at train.py top (quick win) |
| 4 | `torch.compile(dynamic=False)` | `scripts/base_train.py:298-300` | partial (`gpt.py:911`) | med / GPU | verify static shapes; enable in train loop |
| 5 | Fused AdamW + DDP-aware fallback | `gpt.py:245-246` | partial (`gpt.py:327`) | low / GPU | add DDP/CUDA guard to optimizer factory |
| 6 | DistAdamW (ZeRO-2 sharded state) | `adamw.py:1-171` | missing | med / GPU+DDP | port; test under DDP |
| 7 | Muon (Newton–Schulz) validation | `muon.py` | partial (`muon.py`) | low / GPU-opt | validate across the 11 mechanisms |
| 8 | Warmup+warmdown LR schedule | `scripts/base_train.py:435-444` | partial (`ordinal_scheduler.py`) | low / no | add warmdown to the scheduler |
| 9 | KV-cache mask centralization | `gpt.py:79-96` (`_autoregressive_keep_mask`) | partial (inline per-mech) | low / no | extract a shared mask util |
| 10 | Divergence guard (spike/NaN → skip/backoff/rollback) | `divergence_guard.py:1-209` | missing | med / no | port; wire `guard.check(loss,model,step)` into loop |
| 11 | Pre-run VRAM estimator | `scripts/scale_memory.py` | missing | med / no | port; extend for the 11 attention footprints |
| 12 | Structured JSONL telemetry + provenance | `run_logging.py` | partial (`report.py`) | med / no | align metrics.jsonl schema; stamp git/torch/cfg |
| 13 | SynapticFlexAttention (O(N) score_mod) | `flex_synaptic.py:1-102` | missing | high / GPU+torch≥2.5 | port (see `gpu_flex_diff.md`) |
| 14 | GQA scaling experiments | `gpt.py:32-33,69-96` | partial (`gpt.py` GQA) | low / GPU-opt | scaling-law sweep with `n_kv_head<n_head` |

## Notes on prioritization

- **CPU-first (this box):** items 1, 2, 3, 8, 9, 10, 11, 12 need no GPU and are the
  natural near-term backlog. 10 (divergence guard) is especially relevant to the
  exotic mechanisms, several of which can produce NaN/Inf under aggressive LR.
- **GPU-gated:** items 4, 5, 6, 7, 13, 14 want a GPU host; defer to a GPU session.
  13 (Synaptic flex) is the single largest port and already has a dedicated plan.
- **Already strong here:** the math attention mechanisms, the certify/adjudication
  research loop, and the fixed-FLOPs benchmark+regression-gate infra are *ahead* of
  bio — transfer is not one-directional, but those are out of scope for this bead.

See `docs/cross_repo_sync.md` (bead `qha`) for the cadence/process to keep this
catalogue current as bio_inspired_nanochat evolves.
