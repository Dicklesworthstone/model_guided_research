# nanochat run (fixed FLOPs)

- Run ID: `seed_1`
- Generated: 2026-09-02T22:52:10Z
- Artifacts: `runs/budget_0/rot/standard/seed_1`
- Commit: 5dce958e2cb119015937e7ca756ebcdde8b347d2
- Attention schedule: `standard`

## Command

```bash
uv run python -m nanochat.train --device cpu --seed 1 --batch-size 4 --sequence-len 64 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64 --learning-rate 0.0006 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --attention-type standard --target-flops 2000000000000.0 --checkpoint-interval 1000 --checkpoint-keep 1 --data-dir /data/projects/model_guided_research/artifacts/probes/sizing/scorecards/vdc4-rot-tasktok-shuf-b2-rung/data/rot --min-parquet-files 2 --artifacts-dir /data/projects/model_guided_research/artifacts/probes/sizing/scorecards/vdc4-rot-tasktok-shuf-b2-rung --artifacts-kind runs --artifacts-topic budget_0/rot/standard --run-id seed_1 --check-numerics
```

## Budget

- steps: 10255
- warmup_steps: 0
- tokens/step (global): 256
- FLOPs/token (est): 761,856
- FLOPs/step (est): 195,035,136
- planned_total_FLOPs (est): 2,000,085,319,680

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

- measured_steps: 10255
- measured_tokens: 2,625,280
- measured_time_s: 145.112
- tokens/s: 18,091
- TFLOP/s (est): 0.01
- peak_memory_allocated_gb: n/a
- final_train_ce: 0.7553
- val_interval: 0
- val_batches: n/a
- final_val_ce: n/a
- final_val_bpb: n/a (bits per byte; comparable across tokenizers)

See `summary.json` for full details.
