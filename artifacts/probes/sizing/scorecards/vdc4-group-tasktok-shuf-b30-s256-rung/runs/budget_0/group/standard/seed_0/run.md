# nanochat run (fixed FLOPs)

- Run ID: `seed_0`
- Generated: 2026-11-23T02:27:19Z
- Artifacts: `runs/budget_0/group/standard/seed_0`
- Commit: a5abe773b4b5ce1073d63eafe499ef809dc071f3
- Attention schedule: `standard`

## Command

```bash
uv run python -m nanochat.train --device cpu --seed 0 --batch-size 4 --sequence-len 256 --n-layer 4 --n-head 8 --n-kv-head 4 --n-embd 128 --learning-rate 0.0006 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --attention-type standard --target-flops 30000000000000.0 --checkpoint-interval 1000 --checkpoint-keep 1 --data-dir /Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing/scorecards/vdc4-group-tasktok-shuf-b30-s256-rung/data/group --min-parquet-files 2 --artifacts-dir /Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing/scorecards/vdc4-group-tasktok-shuf-b30-s256-rung --artifacts-kind runs --artifacts-topic budget_0/group/standard --run-id seed_0 --check-numerics
```

## Budget

- steps: 4731
- warmup_steps: 0
- tokens/step (global): 1,024
- FLOPs/token (est): 6,193,152
- FLOPs/step (est): 6,341,787,648
- planned_total_FLOPs (est): 30,002,997,362,688

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

- measured_steps: 4731
- measured_tokens: 4,844,544
- measured_time_s: 626.333
- tokens/s: 7,735
- TFLOP/s (est): 0.05
- peak_memory_allocated_gb: n/a
- final_train_ce: 0.3457
- val_interval: 0
- val_batches: n/a
- final_val_ce: n/a
- final_val_bpb: n/a (bits per byte; comparable across tokenizers)

See `summary.json` for full details.
