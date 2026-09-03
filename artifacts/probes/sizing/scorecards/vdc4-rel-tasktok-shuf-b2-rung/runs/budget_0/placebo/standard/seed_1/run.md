# nanochat run (fixed FLOPs)

- Run ID: `seed_1`
- Generated: 2026-09-03T03:18:08Z
- Artifacts: `runs/budget_0/placebo/standard/seed_1`
- Commit: 5dce958e2cb119015937e7ca756ebcdde8b347d2
- Attention schedule: `standard`

## Command

```bash
uv run python -m nanochat.train --device cpu --seed 1 --batch-size 4 --sequence-len 64 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64 --learning-rate 0.0006 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --attention-type standard --target-flops 2000000000000.0 --checkpoint-interval 1000 --checkpoint-keep 1 --data-dir /data/projects/model_guided_research/artifacts/probes/sizing/scorecards/vdc4-rel-tasktok-shuf-b2-rung/data/placebo --min-parquet-files 2 --artifacts-dir /data/projects/model_guided_research/artifacts/probes/sizing/scorecards/vdc4-rel-tasktok-shuf-b2-rung --artifacts-kind runs --artifacts-topic budget_0/placebo/standard --run-id seed_1 --check-numerics
```

## Budget

- steps: 9935
- warmup_steps: 0
- tokens/step (global): 256
- FLOPs/token (est): 786,432
- FLOPs/step (est): 201,326,592
- planned_total_FLOPs (est): 2,000,179,691,520

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

- measured_steps: 9935
- measured_tokens: 2,543,360
- measured_time_s: 5754.090
- tokens/s: 442
- TFLOP/s (est): 0.00
- peak_memory_allocated_gb: n/a
- final_train_ce: 1.4304
- val_interval: 0
- val_batches: n/a
- final_val_ce: n/a
- final_val_bpb: n/a (bits per byte; comparable across tokenizers)

See `summary.json` for full details.
