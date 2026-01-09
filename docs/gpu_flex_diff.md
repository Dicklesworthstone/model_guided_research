# GPU / FlexAttention Stack Diff (bio_inspired_nanochat → model_guided_research)

Bead: `model_guided_research-6l7`

This note captures the current “known-good” GPU + `torch.compile` + FlexAttention integration pattern from
`/data/projects/bio_inspired_nanochat` and contrasts it with this repo (`/data/projects/model_guided_research`).

Scope: documentation only (no code changes in this bead).

## TL;DR (What To Copy First)

From `bio_inspired_nanochat`:
- GPU stack expectations are explicit: Python 3.14 and CUDA 12.4+ (dual RTX 4090 recommended):
  `/data/projects/bio_inspired_nanochat/README.md:403` and `/data/projects/bio_inspired_nanochat/README.md:405`.
- GPU scripts set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:
  `/data/projects/bio_inspired_nanochat/scripts/base_train.py:16`,
  `/data/projects/bio_inspired_nanochat/scripts/mid_train.py:14`,
  `/data/projects/bio_inspired_nanochat/scripts/chat_sft.py:13`.
- The FlexAttention integration pattern is **Synaptic-only** (not global SDPA), wired via a config flag
  (`use_flex_attention`) and guarded by a torch>=2.5 availability check. See:
  - `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/synaptic.py:653`
  - `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/flex_synaptic.py:1`
- FlexAttention is treated as **compile-required** (they `torch.compile(model)` unconditionally in the flex scripts).
  See `/data/projects/bio_inspired_nanochat/scripts/benchmark_flex.py:45` and
  `/data/projects/bio_inspired_nanochat/scripts/verify_flex_correctness.py:40`.
- For GPU stability, the flex scripts strongly prefer **fp16 autocast**, not bf16:
  `/data/projects/bio_inspired_nanochat/scripts/benchmark_flex.py:34` and
  `/data/projects/bio_inspired_nanochat/scripts/verify_flex_correctness.py:13`.
- Activation checkpointing is listed as “in progress” (not obviously wired yet): `/data/projects/bio_inspired_nanochat/README.md:290`.
- Fused-kernel “backends” are a first-class perf narrative: `/data/projects/bio_inspired_nanochat/README.md:268`
  (Triton) and `/data/projects/bio_inspired_nanochat/README.md:274` (Rust).

In this repo (today):
- We target Python 3.13+: `/data/projects/model_guided_research/README.md:114`, and currently suggest `cu118` torch wheels
  as a generic CUDA “fix”: `/data/projects/model_guided_research/README.md:973`.
- We have no FlexAttention path; “standard” attention is SDPA (`F.scaled_dot_product_attention`) and the custom math
  attentions are all non-flex. See `/data/projects/model_guided_research/nanochat/gpt.py:91`.
- Our `nanochat/train.py` uses bf16 autocast on CUDA by default; that diverges from the bio_inspired flex scripts:
  `/data/projects/model_guided_research/nanochat/train.py:84`.
- We do not set `PYTORCH_CUDA_ALLOC_CONF` anywhere (no matches in-tree).
- Activation checkpointing is discussed in docs, but not implemented in `nanochat/` today:
  `/data/projects/model_guided_research/markdown_documentation/reversible_computation_and_measure_preserving_learning.md:498`.

## Repo Locations (This Environment)

The bead text references `/tmp/bio_inspired_nanochat`, but in this environment the repo is at:

- `bio_inspired_nanochat`: `/data/projects/bio_inspired_nanochat`
- `model_guided_research`: `/data/projects/model_guided_research`

## GPU stack / CUDA version expectations

`bio_inspired_nanochat` explicitly targets a newer GPU stack:
- Python 3.14: `/data/projects/bio_inspired_nanochat/README.md:403`
- NVIDIA + CUDA 12.4+: `/data/projects/bio_inspired_nanochat/README.md:405`
- Flex module asserts torch>=2.5: `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/flex_synaptic.py:3`

This repo is less specific (today):
- Python 3.13+: `/data/projects/model_guided_research/README.md:114`
- Troubleshooting suggests `cu118` wheels for “PyTorch CUDA issues”: `/data/projects/model_guided_research/README.md:973`

## Environment flags (allocator / stability)

`bio_inspired_nanochat` standardizes allocator behavior by forcing:
- `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` in:
  - `/data/projects/bio_inspired_nanochat/scripts/base_train.py:16`
  - `/data/projects/bio_inspired_nanochat/scripts/mid_train.py:14`
  - `/data/projects/bio_inspired_nanochat/scripts/chat_sft.py:13`

This repo does not set or document `PYTORCH_CUDA_ALLOC_CONF`.

## torch.compile usage patterns

`bio_inspired_nanochat`:
- Compiles the training model by default with `dynamic=False`:
  - `/data/projects/bio_inspired_nanochat/scripts/base_train.py:236`
  - `/data/projects/bio_inspired_nanochat/scripts/base_train.py:237`
  - `/data/projects/bio_inspired_nanochat/scripts/mid_train.py:77`
- Compiles explicitly in Flex correctness/bench scripts:
  - `/data/projects/bio_inspired_nanochat/scripts/verify_flex_correctness.py:40`
  - `/data/projects/bio_inspired_nanochat/scripts/benchmark_flex.py:47`
- Notes that `dynamic=True` is awkward for variable-length SFT:
  - `/data/projects/bio_inspired_nanochat/scripts/chat_sft.py:81`

This repo:
- Has no `torch.compile(...)` calls under `nanochat/` (as of this note).

## Optimizer + scheduler differences

Optimizers:
- Bio uses `partial(torch.optim.AdamW, fused=use_fused)` when not DDP and on CUDA:
  `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/gpt.py:246`
- This repo enables fused AdamW only when the kwarg exists in the local torch build:
  `/data/projects/model_guided_research/nanochat/gpt.py:327`

Schedulers:
- Bio’s base training script defines a warmup/warmdown LR multiplier function:
  `/data/projects/bio_inspired_nanochat/scripts/base_train.py:362`
- This repo’s minimal trainer supports the ordinal scheduler:
  `/data/projects/model_guided_research/nanochat/train.py:68`

## Activation checkpointing status

Bio:
- Lists activation checkpointing as an “in progress” memory optimization item:
  `/data/projects/bio_inspired_nanochat/README.md:290`

This repo:
- Discusses an activation-checkpoint baseline in reversible documentation:
  `/data/projects/model_guided_research/markdown_documentation/reversible_computation_and_measure_preserving_learning.md:498`
- Does not implement activation checkpointing in `nanochat/` today.

## Fused kernels (Triton / Rust)

Bio:
- Positions Triton kernels as the primary GPU backend:
  `/data/projects/bio_inspired_nanochat/README.md:268`
- Example Triton kernel module imports triton directly:
  `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/kernels/presyn_fused.py:4`
- Positions Rust kernels + PyO3 as the primary CPU backend:
  `/data/projects/bio_inspired_nanochat/README.md:274`
  and Rust implementation exists at `/data/projects/bio_inspired_nanochat/rust_src/src/presyn.rs:1`.
- Triton “metrics fused” path is actually wired behind `BIO_FUSED_METRICS`:
  `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/neuroscore.py:20`,
  `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/neuroscore.py:91`.

This repo:
- Includes the same-style Triton kernels:
  `/data/projects/model_guided_research/nanochat/kernels/presyn_fused.py:4`
- Wires “metrics fused” behind `BIO_FUSED_METRICS`:
  `/data/projects/model_guided_research/nanochat/neuroscore.py:66`,
  `/data/projects/model_guided_research/nanochat/neuroscore.py:91`
- Does not have a Rust/PyO3 kernel backend.

## FlexAttention Prereqs + Behavioral Assumptions (bio_inspired_nanochat)

### PyTorch Version / Import Surface

FlexAttention is gated on availability of `torch.nn.attention.flex_attention`:
- Implementation import: `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/flex_synaptic.py:16`
- Guard + error message: `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/synaptic.py:653`

This implies:
- **torch>=2.5** is required in practice (their docs/comments say so).
- Flex is treated as a **hard error if enabled but unavailable**.

### torch.compile is Treated as Required

Their flex-focused scripts compile the model:
- `/data/projects/bio_inspired_nanochat/scripts/verify_flex_correctness.py:40`
- `/data/projects/bio_inspired_nanochat/scripts/benchmark_flex.py:47`

Their main training entrypoint compiles too (even for non-flex):
- `/data/projects/bio_inspired_nanochat/scripts/base_train.py:236`

### DType: fp16 favored over bf16 (for flex)

The flex scripts explicitly choose fp16 and cite bf16 issues:
- `/data/projects/bio_inspired_nanochat/scripts/benchmark_flex.py:34`
- `/data/projects/bio_inspired_nanochat/scripts/verify_flex_correctness.py:13`

## FlexAttention Integration Pattern (bio_inspired_nanochat)

### 1) A dedicated module: `SynapticFlexAttention`

File: `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/flex_synaptic.py`

Key mechanics:
- Precompute O(N) biological factors from presynaptic state (`key_factor`, `qamp`):
  `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/flex_synaptic.py:25`
- Provide a `score_mod(score, b, h, q_idx, kv_idx)` closure that:
  - scales by `1/sqrt(D)` (`flex_synaptic.py:80`)
  - adds a log biological bias term (`flex_synaptic.py:92`)
  - subtracts a distance barrier (`flex_synaptic.py:95`)
- Call `flex_attention(q, k, v, score_mod=..., block_mask=...)`:
  `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/flex_synaptic.py:104`

### 2) Wiring via config flag + fallback to standard synaptic attention

Config flag:
- `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/synaptic.py:154` (`use_flex_attention`)

Initialization:
- When `cfg.use_flex_attention` is true, the attention module creates `self.flex`:
  `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/synaptic.py:653`

Forward path structure:
- They still compute dense logits to update state (correctness-first), then call flex for output:
  `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/synaptic.py:716`
- They build a causal `BlockMask` via `create_block_mask`:
  `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/synaptic.py:752`
- Then call `y = self.flex(q, k, v, presyn_state, block_mask=block_mask)`:
  `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/synaptic.py:764`

Important nuance (potential perf pitfall):
- The “correctness-first” state update still materializes `(B,H,T,T)` (`dots`) on the flex path:
  `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/synaptic.py:727`
  so memory isn’t truly O(N) until that state update is also fused/streamed.

## Verification + Bench Scripts (bio_inspired_nanochat)

### Correctness smoke test

`/data/projects/bio_inspired_nanochat/scripts/verify_flex_correctness.py`
- Builds a small model with `use_flex_attention=True`: `verify_flex_correctness.py:20`
- Compiles it: `verify_flex_correctness.py:40`
- Runs forward+backward under fp16 autocast and checks NaNs: `verify_flex_correctness.py:49`

### Throughput + VRAM benchmark

`/data/projects/bio_inspired_nanochat/scripts/benchmark_flex.py`
- Creates `GPTSynaptic` with `use_flex_attention` toggled: `benchmark_flex.py:14`
- Compiles it: `benchmark_flex.py:45`
- Measures tokens/sec + peak VRAM: `benchmark_flex.py:79`

## Parity / Diffs vs model_guided_research (this repo)

### Attention stack

This repo’s baseline GPT uses SDPA:
- `/data/projects/model_guided_research/nanochat/gpt.py:91`

We do have a synaptic implementation here, but **no FlexAttention toggle or module**:
- `SynapticConfig` has no `use_flex_attention`: `/data/projects/model_guided_research/nanochat/synaptic.py:66`
- `SynapticCausalSelfAttention` always computes dense logits + scatter augmentation:
  `/data/projects/model_guided_research/nanochat/synaptic.py:645`

### Training defaults (dtype + compile)

This repo’s minimal trainer:
- Uses bf16 autocast on CUDA: `/data/projects/model_guided_research/nanochat/train.py:84`
- Does not use `torch.compile` at all (`rg torch.compile nanochat/` only hits a comment):
  `/data/projects/model_guided_research/nanochat/adamw.py:19`

### Dataset / dataloader parity

These appear effectively identical between repos (same parquet sharding logic, same token buffer strategy):
- `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/dataloader.py:10`
- `/data/projects/model_guided_research/nanochat/dataloader.py:10`
- `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/dataset.py:22`
- `/data/projects/model_guided_research/nanochat/dataset.py:22`

### Optimizer wiring

Bio uses fused AdamW via a `partial(..., fused=use_fused)` pattern:
- `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/gpt.py:245`

This repo enables fused AdamW only when the kwarg exists in the local torch build:
- `/data/projects/model_guided_research/nanochat/gpt.py:327`

## Actionable Deltas (Inputs to Other Beads)

The table below is a “handoff map”: concrete differences discovered, where they likely belong, and the next action.

| Delta | Owner (bead) | Next step |
| --- | --- | --- |
| FlexAttention implementation + flag | `model_guided_research-2e5` | Port `SynapticFlexAttention` + `use_flex_attention` gate and add correctness/bench scripts mirroring bio. |
| Flex masking/caching edge cases | `model_guided_research-6vw` | Enumerate prefill/decode/chunked decode + GQA/MQA + dtype interactions; add tests. |
| CUDA / torch version guidance + env var docs | `model_guided_research-5td` | Document “known-good” torch/CUDA combos for flex + allocator env vars; align this repo’s CUDA install guidance with the Flex stack. |
| Flex compatibility matrix | `model_guided_research-b5l` | Record at least 1 working and 1 failing torch/CUDA/GPU combo + mitigation flags. |
| Activation checkpointing strategy | `model_guided_research-1ra` | Decide checkpointing approach; document interactions with Flex + reversible. |
| Config parity (dtype, compile defaults) | `model_guided_research-wyf` | Decide fp16 vs bf16 defaults and when to compile (`dynamic=False` vs `dynamic=True`). |
| Fixed-FLOPs baseline harness | `model_guided_research-gjm` / `model_guided_research-5bo` | Port the `target_flops` approach from bio `base_train.py` into this repo. |

### For `model_guided_research-2e5` (Integrate FlexAttention path)

Porting recommendation (high-level):
1. Mirror the `bio_inspired_nanochat` structure:
   - a dedicated `SynapticFlexAttention` wrapper (their `flex_synaptic.py`)
   - a `SynapticConfig.use_flex_attention` toggle + availability guard
2. Keep the “correctness-first” shape, but explicitly track that true O(N) won’t land until presyn-state updates avoid
   the dense `(B,H,T,T)` logits (`bio synaptic.py:727`).
3. Treat fp16 vs bf16 as a first-class configuration choice; don’t assume bf16 is safe for flex on all stacks
   (`benchmark_flex.py:34`).

### For `model_guided_research-6vw` (FlexAttention masking/caching edge cases)

bio_inspired_nanochat uses `create_block_mask(causal_mask, ...)`:
- `/data/projects/bio_inspired_nanochat/bio_inspired_nanochat/synaptic.py:752`

Mask/caching matrix to enumerate later:
- causal prefill (Tk==Tq)
- single-token decode (Tq==1)
- chunked decode (Tq>1, Tk>Tq)
- MQA/GQA interactions (n_head != n_kv_head)
- dtype promotion interactions (q/k cast to v dtype in `synaptic.py:759`)

### For `model_guided_research-gjm` / `model_guided_research-5bo` (FLOPs-budget harness + baseline)

bio_inspired_nanochat’s `base_train.py` already implements:
- `torch.compile(dynamic=False)` assumption: `/data/projects/bio_inspired_nanochat/scripts/base_train.py:236`
- `estimate_flops()` + compute iterations from `target_flops`: `/data/projects/bio_inspired_nanochat/scripts/base_train.py:241`

This is a good template for how to structure fixed-FLOPs comparisons in this repo.
