# nanochat run (fixed FLOPs)

- Run ID: `seed_1`
- Generated: 2026-09-02T22:20:06Z
- Artifacts: `runs/budget_0/placebo/fractal/seed_1`
- Commit: 5dce958e2cb119015937e7ca756ebcdde8b347d2
- Attention schedule: `fractal`

## Command

```bash
uv run python -m nanochat.train --device cpu --seed 1 --batch-size 4 --sequence-len 64 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64 --learning-rate 0.0006 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --attention-type fractal --target-flops 1000000000000.0 --checkpoint-interval 1000 --checkpoint-keep 1 --data-dir /data/projects/model_guided_research/artifacts/scorecards/placebo-guard-tasktok-1e12/data/placebo --min-parquet-files 2 --artifacts-dir /data/projects/model_guided_research/artifacts/scorecards/placebo-guard-tasktok-1e12 --artifacts-kind runs --artifacts-topic budget_0/placebo/fractal --run-id seed_1 --check-numerics
```

## Budget

- steps: 4948
- warmup_steps: 0
- tokens/step (global): 256
- FLOPs/token (est): 789,504
- FLOPs/step (est): 202,113,024
- planned_total_FLOPs (est): 1,000,055,242,752

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

- measured_steps: 4948
- measured_tokens: 1,266,688
- measured_time_s: 92.302
- tokens/s: 13,723
- TFLOP/s (est): 0.01
- peak_memory_allocated_gb: n/a
- final_train_ce: 1.4530
- val_interval: 0
- val_batches: n/a
- final_val_ce: n/a
- final_val_bpb: n/a (bits per byte; comparable across tokenizers)

See `summary.json` for full details.
