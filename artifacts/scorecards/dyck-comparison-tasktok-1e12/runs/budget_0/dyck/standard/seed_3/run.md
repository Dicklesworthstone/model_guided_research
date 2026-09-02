# nanochat run (fixed FLOPs)

- Run ID: `seed_3`
- Generated: 2026-09-02T20:42:52Z
- Artifacts: `runs/budget_0/dyck/standard/seed_3`
- Commit: 49f59e7454b46b8bd7a048066b63f9aa55546229
- Attention schedule: `standard`

## Command

```bash
uv run python -m nanochat.train --device cpu --seed 3 --batch-size 4 --sequence-len 64 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64 --learning-rate 0.0006 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --attention-type standard --target-flops 1000000000000.0 --checkpoint-interval 1000 --checkpoint-keep 1 --data-dir /data/projects/model_guided_research/artifacts/scorecards/dyck-comparison-tasktok-1e12/data/dyck --min-parquet-files 2 --artifacts-dir /data/projects/model_guided_research/artifacts/scorecards/dyck-comparison-tasktok-1e12 --artifacts-kind runs --artifacts-topic budget_0/dyck/standard --run-id seed_3 --check-numerics
```

## Budget

- steps: 5128
- warmup_steps: 0
- tokens/step (global): 256
- FLOPs/token (est): 761,856
- FLOPs/step (est): 195,035,136
- planned_total_FLOPs (est): 1,000,140,177,408

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

- measured_steps: 5128
- measured_tokens: 1,312,768
- measured_time_s: 1197.766
- tokens/s: 1,096
- TFLOP/s (est): 0.00
- peak_memory_allocated_gb: n/a
- final_train_ce: 0.9975
- val_interval: 0
- val_batches: n/a
- final_val_ce: n/a
- final_val_bpb: n/a (bits per byte; comparable across tokenizers)

See `summary.json` for full details.
