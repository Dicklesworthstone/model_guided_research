# nanochat run (fixed FLOPs)

- Run ID: `seed_7`
- Generated: 2026-08-23T21:25:54Z
- Artifacts: `bench/fixed_flops/nanochat/20260823_172511/quaternion/seed_7`
- Commit: 3d98bcd30624ce03dd4d525153b3d381ac4421da (dirty)
- Attention schedule: `quaternion`

## Command

```bash
uv run python -m nanochat.train --device cpu --seed 7 --batch-size 8 --sequence-len 256 --n-layer 4 --n-head 4 --n-kv-head 4 --n-embd 128 --learning-rate 0.0006 --optimizer-type adamw --attention-type quaternion --target-flops 30000000.0 --warmup-steps 0 --log-interval 1 --artifacts-dir artifacts --artifacts-kind bench --artifacts-topic fixed_flops/nanochat/20260823_172511/quaternion --run-id seed_7 --auto-download-data --min-parquet-files 2
```

## Budget

- steps: 1
- warmup_steps: 0
- tokens/step (global): 2,048
- FLOPs/token (est): 44,924,928
- FLOPs/step (est): 92,006,252,544
- planned_total_FLOPs (est): 92,006,252,544

## Compilation

- torch.compile: False
- compile_backend: 'inductor'
- compile_mode: None
- compile_fullgraph: False
- compile_dynamic: None
- compile_flex_attention: False

## Numerics (debug)

- check_numerics: False
- detect_anomaly: False

## Results (measured after warmup)

- measured_steps: 1
- measured_tokens: 2,048
- measured_time_s: 1.121
- tokens/s: 1,827
- TFLOP/s (est): 0.08
- peak_memory_allocated_gb: n/a
- final_train_ce: 10.8258
- val_interval: 0
- val_batches: n/a
- final_val_ce: n/a

See `summary.json` for full details.
