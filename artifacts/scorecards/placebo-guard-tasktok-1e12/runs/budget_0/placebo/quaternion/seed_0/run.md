# nanochat run (fixed FLOPs)

- Run ID: `seed_0`
- Generated: 2026-09-02T21:59:59Z
- Artifacts: `runs/budget_0/placebo/quaternion/seed_0`
- Commit: 5dce958e2cb119015937e7ca756ebcdde8b347d2
- Attention schedule: `quaternion`

## Command

```bash
uv run python -m nanochat.train --device cpu --seed 0 --batch-size 4 --sequence-len 64 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64 --learning-rate 0.0006 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --attention-type quaternion --target-flops 1000000000000.0 --checkpoint-interval 1000 --checkpoint-keep 1 --data-dir /data/projects/model_guided_research/artifacts/scorecards/placebo-guard-tasktok-1e12/data/placebo --min-parquet-files 2 --artifacts-dir /data/projects/model_guided_research/artifacts/scorecards/placebo-guard-tasktok-1e12 --artifacts-kind runs --artifacts-topic budget_0/placebo/quaternion --run-id seed_0 --check-numerics
```

## Budget

- steps: 4968
- warmup_steps: 0
- tokens/step (global): 256
- FLOPs/token (est): 786,432
- FLOPs/step (est): 201,326,592
- planned_total_FLOPs (est): 1,000,190,509,056

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

- measured_steps: 4968
- measured_tokens: 1,271,808
- measured_time_s: 74.673
- tokens/s: 17,032
- TFLOP/s (est): 0.01
- peak_memory_allocated_gb: n/a
- final_train_ce: 1.5126
- val_interval: 0
- val_batches: n/a
- final_val_ce: n/a
- final_val_bpb: n/a (bits per byte; comparable across tokenizers)

See `summary.json` for full details.
