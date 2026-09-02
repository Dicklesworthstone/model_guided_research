# nanochat run (fixed FLOPs)

- Run ID: `seed_6`
- Generated: 2026-09-02T22:08:23Z
- Artifacts: `runs/budget_0/placebo/braid/seed_6`
- Commit: 49f59e7454b46b8bd7a048066b63f9aa55546229
- Attention schedule: `braid`

## Command

```bash
uv run python -m nanochat.train --device cpu --seed 6 --batch-size 4 --sequence-len 64 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64 --learning-rate 0.0006 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --attention-type braid --target-flops 1000000000000.0 --checkpoint-interval 1000 --checkpoint-keep 1 --data-dir /data/projects/model_guided_research/artifacts/scorecards/dyck-comparison-tasktok-1e12/data/placebo --min-parquet-files 2 --artifacts-dir /data/projects/model_guided_research/artifacts/scorecards/dyck-comparison-tasktok-1e12 --artifacts-kind runs --artifacts-topic budget_0/placebo/braid --run-id seed_6 --check-numerics
```

## Budget

- steps: 4966
- warmup_steps: 0
- tokens/step (global): 256
- FLOPs/token (est): 786,624
- FLOPs/step (est): 201,375,744
- planned_total_FLOPs (est): 1,000,031,944,704

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

- measured_steps: 4966
- measured_tokens: 1,271,296
- measured_time_s: 258.522
- tokens/s: 4,918
- TFLOP/s (est): 0.00
- peak_memory_allocated_gb: n/a
- final_train_ce: 1.5250
- val_interval: 0
- val_batches: n/a
- final_val_ce: n/a
- final_val_bpb: n/a (bits per byte; comparable across tokenizers)

See `summary.json` for full details.
