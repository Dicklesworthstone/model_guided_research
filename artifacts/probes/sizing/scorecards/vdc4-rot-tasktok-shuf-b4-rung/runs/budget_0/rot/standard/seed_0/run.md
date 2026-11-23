# nanochat run (fixed FLOPs)

- Run ID: `seed_0`
- Generated: 2026-11-23T05:17:12Z
- Artifacts: `runs/budget_0/rot/standard/seed_0`
- Commit: 5dce958e2cb119015937e7ca756ebcdde8b347d2
- Attention schedule: `standard`

## Command

```bash
uv run python -m nanochat.train --device cpu --seed 0 --batch-size 4 --sequence-len 64 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64 --learning-rate 0.0006 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --attention-type standard --target-flops 4000000000000.0 --checkpoint-interval 1000 --checkpoint-keep 1 --data-dir /Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing/scorecards/vdc4-rot-tasktok-shuf-b4-rung/data/rot --min-parquet-files 2 --artifacts-dir /Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing/scorecards/vdc4-rot-tasktok-shuf-b4-rung --artifacts-kind runs --artifacts-topic budget_0/rot/standard --run-id seed_0 --check-numerics
```

## Budget

- steps: 20510
- warmup_steps: 0
- tokens/step (global): 256
- FLOPs/token (est): 761,856
- FLOPs/step (est): 195,035,136
- planned_total_FLOPs (est): 4,000,170,639,360

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

- measured_steps: 20510
- measured_tokens: 5,250,560
- measured_time_s: 130.890
- tokens/s: 40,114
- TFLOP/s (est): 0.03
- peak_memory_allocated_gb: n/a
- final_train_ce: 0.6810
- val_interval: 0
- val_batches: n/a
- final_val_ce: n/a
- final_val_bpb: n/a (bits per byte; comparable across tokenizers)

See `summary.json` for full details.
