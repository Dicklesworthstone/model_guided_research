# nanochat run (fixed FLOPs)

- Run ID: `seed_0`
- Generated: 2026-11-23T05:52:19Z
- Artifacts: `runs/budget_0/copyops/standard/seed_0`
- Commit: 5dce958e2cb119015937e7ca756ebcdde8b347d2
- Attention schedule: `standard`

## Command

```bash
uv run python -m nanochat.train --device cpu --seed 0 --batch-size 4 --sequence-len 64 --n-layer 4 --n-head 8 --n-kv-head 4 --n-embd 128 --learning-rate 0.0006 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --attention-type standard --target-flops 10000000000000.0 --checkpoint-interval 1000 --checkpoint-keep 1 --data-dir /Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing/scorecards/vdc4-copyops-tasktok-shuf-b10-s64-rung/data/copyops --min-parquet-files 2 --artifacts-dir /Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing/scorecards/vdc4-copyops-tasktok-shuf-b10-s64-rung --artifacts-kind runs --artifacts-topic budget_0/copyops/standard --run-id seed_0 --check-numerics
```

## Budget

- steps: 7869
- warmup_steps: 0
- tokens/step (global): 256
- FLOPs/token (est): 4,964,352
- FLOPs/step (est): 1,270,874,112
- planned_total_FLOPs (est): 10,000,508,387,328

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

- measured_steps: 7869
- measured_tokens: 2,014,464
- measured_time_s: 150.565
- tokens/s: 13,379
- TFLOP/s (est): 0.07
- peak_memory_allocated_gb: n/a
- final_train_ce: 1.2454
- val_interval: 0
- val_batches: n/a
- final_val_ce: n/a
- final_val_bpb: n/a (bits per byte; comparable across tokenizers)

See `summary.json` for full details.
