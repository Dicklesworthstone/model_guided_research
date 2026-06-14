# nanochat run (fixed FLOPs)

- Run ID: `seed0`
- Generated: 2026-06-14T11:01:29Z
- Artifacts: `ca_init/m32/runs/mlp_heavy__ca_rule30/seed0`
- Commit: 504e2f375f1dcbbdb000700d45b3069fd1269a77 (dirty)

## Command

```bash
uv run python -m nanochat.train --device cpu --seed 0 --attention-type standard --optimizer-type adamw --learning-rate 0.0006 --max-steps 120 --warmup-steps 5 --log-interval 1 --vocab-size 50304 --n-layer 6 --n-head 2 --n-kv-head 2 --n-embd 128 --sequence-len 32 --batch-size 8 --ca-init-rule rule30 --ca-init-alpha 1 --artifacts-dir artifacts --artifacts-kind ca_init --artifacts-topic m32/runs/mlp_heavy__ca_rule30 --run-id seed0 --auto-download-data --min-parquet-files 2
```

## Budget

- steps: 120
- warmup_steps: 5
- tokens/step (global): 256
- FLOPs/token (est): 46,006,272
- FLOPs/step (est): 11,777,605,632
- planned_total_FLOPs (est): 1,413,312,675,840

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

- measured_steps: 115
- measured_tokens: 29,440
- measured_time_s: 132.717
- tokens/s: 222
- TFLOP/s (est): 0.01
- peak_memory_allocated_gb: n/a
- final_train_ce: 7.4308
- val_interval: 0
- val_batches: n/a
- final_val_ce: n/a

See `summary.json` for full details.
