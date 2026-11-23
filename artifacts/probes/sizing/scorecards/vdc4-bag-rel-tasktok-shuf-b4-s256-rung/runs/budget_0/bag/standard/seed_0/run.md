# nanochat run (fixed FLOPs)

- Run ID: `seed_0`
- Generated: 2026-11-23T02:34:19Z
- Artifacts: `runs/budget_0/bag/standard/seed_0`
- Commit: a5abe773b4b5ce1073d63eafe499ef809dc071f3
- Attention schedule: `standard`

## Command

```bash
uv run python -m nanochat.train --device cpu --seed 0 --batch-size 4 --sequence-len 256 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64 --learning-rate 0.0006 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --attention-type standard --target-flops 4000000000000.0 --checkpoint-interval 1000 --checkpoint-keep 1 --data-dir /Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing/scorecards/vdc4-bag-rel-tasktok-shuf-b4-s256-rung/data/bag --min-parquet-files 2 --artifacts-dir /Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing/scorecards/vdc4-bag-rel-tasktok-shuf-b4-s256-rung --artifacts-kind runs --artifacts-topic budget_0/bag/standard --run-id seed_0 --check-numerics
```

## Budget

- steps: 3697
- warmup_steps: 0
- tokens/step (global): 1,024
- FLOPs/token (est): 1,056,768
- FLOPs/step (est): 1,082,130,432
- planned_total_FLOPs (est): 4,000,636,207,104

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

- measured_steps: 3697
- measured_tokens: 3,785,728
- measured_time_s: 120.679
- tokens/s: 31,370
- TFLOP/s (est): 0.03
- peak_memory_allocated_gb: n/a
- final_train_ce: 1.1964
- val_interval: 0
- val_batches: n/a
- final_val_ce: n/a
- final_val_bpb: n/a (bits per byte; comparable across tokenizers)

See `summary.json` for full details.
