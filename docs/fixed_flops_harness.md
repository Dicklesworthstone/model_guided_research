# Fixed-FLOPs Benchmark Harness

Bead: `model_guided_research-gjm`

This repo standardizes *compute-budgeted* runs so we can do fair comparisons (baseline vs feature, optimizer, etc.)
without accidentally changing total work.

All runs should emit:

- `summary.json` (machine-readable telemetry; see `artifacts/README.md`)
- `run.md` (human-readable summary)

under the unified artifacts layout.

---

## Nanochat FLOPs accounting (reference implementation)

Nanochat uses a simple, explicit accounting model (good enough for consistent comparisons):

1) Estimate per-token FLOPs from the model:

- `f_tok = model.estimate_flops()`  (an *estimate*, but stable across runs for a fixed config)

2) Compute tokens per optimizer step:

- `tokens_per_step_global = batch_size * sequence_len * world_size`

3) Compute per-step FLOPs:

- `flops_per_step = f_tok * tokens_per_step_global`

4) Convert a global compute budget to steps:

- `max_steps = ceil(target_flops / flops_per_step)`

During the run we measure throughput (tokens/s) and report estimated TFLOP/s:

- `tflops_per_second_est = (f_tok * tokens_per_second) / 1e12`

Notes:

- This ignores optimizer/data overhead and any compile-time costs; it’s meant for *comparability*, not exact hardware FLOPs.
- Warmup steps are excluded from throughput measurement.

### Running a fixed-FLOPs nanochat baseline

```bash
uv run python -m nanochat.train \
  --device cpu \
  --auto-download-data \
  --min-parquet-files 2 \
  --attention-type standard \
  --optimizer-type adamw \
  --batch-size 8 \
  --sequence-len 256 \
  --target-flops 1e11 \
  --warmup-steps 2 \
  --artifacts-dir artifacts \
  --artifacts-kind bench \
  --artifacts-topic fixed_flops/nanochat \
  --run-id 20251218_flops_smoke
```

This writes into:

`artifacts/bench/fixed_flops/nanochat/20251218_flops_smoke/`

---

### The tokenizer is part of the FLOPs coordinate

`GPT.estimate_flops` counts the embedding and lm_head matmuls, and at the
widths the CPU campaigns use they dominate: at d=64 the 50,304-row GPT-2
vocabulary is 96.6% of the FLOPs per token (19,955,712 vs 688,128 with a
128-row vocabulary). A fixed-FLOPs budget therefore bought almost no
transformer body, which is how the copyops sizing ladder (5e10 to 8e12)
floored at every rung (bead r7qn, 2026-09-01).

`nanochat.train --tokenizer task` (bead n6y1) trains a byte-level BPE on the
corpus's train split (default 512 tokens, `--tokenizer-vocab-size`), pads the
embedding table to a multiple of 64, saves the tokenizer inside the checkpoint
directory, and records `tokenizer.kind == "task"` in the checkpoint meta and
`summary.json` hparams. `mgr scorecard --tokenizer task` threads it into every
cell and records it in the manifest config. Two rules follow:

- **One tokenizer per comparison.** Cells trained with different tokenizers
  sit at different coordinates; never adjudicate across them.
- **Re-probe the rung.** Budgets found with the GPT-2 tokenizer do not carry
  over: the body sees ~25x more tokens per FLOP at d=64, so the floor-clearing
  rung moves. Preregister the new ladder before spending compute.

### Epochs are part of the coordinate too

A FLOPs budget buys tokens; a corpus fixes how many DISTINCT tokens exist. When
the budget is many epochs of a small corpus, the model learns the corpus, not
the task: the 1e12 task-tokenizer copyops probe (dataset_size 1000, ~93
epochs) reached train loss 0.9 on the stream in corpus order and 4.5 on the
same documents shuffled, with exact match 0.0. Two rules:

- **Scale the corpus to the budget.** Keep every rung at or below about two
  epochs (`mgr scorecard --dataset-size`); at ~18 task-tokenizer tokens per
  copyops document that is 40k documents per 1e12 FLOPs at d=64.
- **Shuffle per epoch.** `nanochat.train --data-shuffle epoch` (the default
  since 2026-09-02) visits each row group's documents in a per-epoch seeded
  permutation, so no run can learn document order; `none` restores file
  order. The evaluator has prefixed prompts and perplexity documents with the
  trainer's `<|bos|>` since the same date (`mgr.evaltasks.v4`); v3 and v4
  artifacts must never share an arm.

### Two controls every comparison needs

An apparatus that has never reported "nothing" has not been shown able to
report anything, and one that has never found a planted effect has not been
shown able to find one. The harness carries both controls:

- **Negative control: the placebo task.** Answers carry no structure, so no
  mechanism may win. `hyp-placebo-no-winner` is adjudicated from placebo
  cells of every mechanism at the campaign budget; the scorecard's
  publication gate stays BLOCKED until it is SUPPORTED.
- **Positive control: the no-context arm.** `nanochat.train
  --control-zero-attention` multiplies every block's attention output by
  zero, so the model is a per-token MLP stack bounded by the answer prior on
  any context-dependent task. `hyp-control-no-context-planted-effect`
  registers "standard beats that arm by 0.20 held-out exact match at equal
  FLOPs on arith"; the engine MUST return SUPPORTED, and any other outcome
  indicts the evaluator, the budget cohorts, the variant selectors or the
  statistics rather than attention. The flag is recorded in `model_config`,
  so the registry's `baseline.variant` / `candidate_variant` selectors tell
  the two arms apart.

### Per-arm variants

`mgr scorecard --mechanism MECHANISM@key=value[,key=value]` trains the same
mechanism with one recorded knob changed and treats it as its own arm:
`braid@braid_crossing_law=rmatrix`, `standard@control_zero_attention=true`,
`reversible@reversible_mode=symplectic,reversible_tied=true`. Extras must be
`GPTConfig` fields with a `nanochat.train` flag, so the trainer records them in
`model_config` and the registry's `baseline.variant` / `candidate_variant`
selectors find the arm at adjudication (a selector equal to the config
default resolves to the plain arm). Model shape keys are campaign-global and
refused as extras. Cells and directories carry the arm label
(`standard+control_zero_attention-true`), the FLOPs estimate and the placebo
coverage are per mechanism.

## JAX demos (current status)

Exact FLOPs for JAX demos is trickier because XLA fusion and compilation obscure a clean “FLOPs per step” number.

Current policy:

- Demos should still be run under *explicit* compute knobs (usually iterations/epochs) and record them in artifacts.
- The fixed-FLOPs harness is *fully implemented for nanochat*; demos will gain FLOPs estimators incrementally.

### Demo run sizes

The JAX demos are fixed-size programs: none of them reads an iteration cap,
so there is no CLI knob that shrinks them (the former `--max-iterations`
option was accepted by `mgr run` but consumed by nothing, and was removed).
A demo run with artifacts is:

```bash
mgr run matrix-gauge --seed 0 --artifacts-dir artifacts --run-id 20251218_demo_smoke
```

