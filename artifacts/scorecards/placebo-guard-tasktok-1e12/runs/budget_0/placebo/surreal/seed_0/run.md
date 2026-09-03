# nanochat run (fixed FLOPs)

- Run ID: `seed_0`
- Generated: 2026-09-02T22:23:20Z
- Artifacts: `runs/budget_0/placebo/surreal/seed_0`
- Commit: 5dce958e2cb119015937e7ca756ebcdde8b347d2
- Attention schedule: `surreal`

## Command

```bash
uv run python -m nanochat.train --device cpu --seed 0 --batch-size 4 --sequence-len 64 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64 --learning-rate 0.0006 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --attention-type surreal --target-flops 1000000000000.0 --checkpoint-interval 1000 --checkpoint-keep 1 --data-dir /data/projects/model_guided_research/artifacts/scorecards/placebo-guard-tasktok-1e12/data/placebo --min-parquet-files 2 --artifacts-dir /data/projects/model_guided_research/artifacts/scorecards/placebo-guard-tasktok-1e12 --artifacts-kind runs --artifacts-topic budget_0/placebo/surreal --run-id seed_0 --check-numerics
```

## Budget

- steps: 4953
- warmup_steps: 0
- tokens/step (global): 256
- FLOPs/token (est): 788,736
- FLOPs/step (est): 201,916,416
- planned_total_FLOPs (est): 1,000,092,008,448

## Compilation

- torch.compile: False
- compile_backend: 'inductor'
- compile_mode: None
- compile_fullgraph: False
- compile_dynamic: None
- compile_flex_attention: False

## Numerics (debug)

- check_numerics: True
- detect_anomaly: False

## Results (measured after warmup)

- measured_steps: 4953
- measured_tokens: 1,267,968
- measured_time_s: 87.104
- tokens/s: 14,557
- TFLOP/s (est): 0.01
- peak_memory_allocated_gb: n/a
- final_train_ce: 1.4355
- val_interval: 0
- val_batches: n/a
- final_val_ce: n/a
- final_val_bpb: n/a (bits per byte; comparable across tokenizers)

See `summary.json` for full details.
