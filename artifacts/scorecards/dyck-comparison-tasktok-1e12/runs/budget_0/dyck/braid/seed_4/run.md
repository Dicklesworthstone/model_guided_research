# nanochat run (fixed FLOPs)

- Run ID: `seed_4`
- Generated: 2026-09-02T21:47:04Z
- Artifacts: `runs/budget_0/dyck/braid/seed_4`
- Commit: 49f59e7454b46b8bd7a048066b63f9aa55546229
- Attention schedule: `braid`

## Command

```bash
uv run python -m nanochat.train --device cpu --seed 4 --batch-size 4 --sequence-len 64 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64 --learning-rate 0.0006 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --attention-type braid --target-flops 1000000000000.0 --checkpoint-interval 1000 --checkpoint-keep 1 --data-dir /data/projects/model_guided_research/artifacts/scorecards/dyck-comparison-tasktok-1e12/data/dyck --min-parquet-files 2 --artifacts-dir /data/projects/model_guided_research/artifacts/scorecards/dyck-comparison-tasktok-1e12 --artifacts-kind runs --artifacts-topic budget_0/dyck/braid --run-id seed_4 --check-numerics
```

## Budget

- steps: 5126
- warmup_steps: 0
- tokens/step (global): 256
- FLOPs/token (est): 762,048
- FLOPs/step (est): 195,084,288
- planned_total_FLOPs (est): 1,000,002,060,288

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

- measured_steps: 5126
- measured_tokens: 1,312,256
- measured_time_s: 75.338
- tokens/s: 17,418
- TFLOP/s (est): 0.01
- peak_memory_allocated_gb: n/a
- final_train_ce: 1.0616
- val_interval: 0
- val_batches: n/a
- final_val_ce: n/a
- final_val_bpb: n/a (bits per byte; comparable across tokenizers)

See `summary.json` for full details.
