# nanochat run (fixed FLOPs)

- Run ID: `seed0`
- Generated: 2026-06-14T10:51:40Z
- Artifacts: `ca_init/m32/runs/attn_heavy__ca_mix0.5/seed0`
- Commit: 504e2f375f1dcbbdb000700d45b3069fd1269a77 (dirty)

## Command

```bash
uv run python -m nanochat.train --device cpu --seed 0 --attention-type standard --optimizer-type adamw --learning-rate 0.0006 --max-steps 120 --warmup-steps 5 --log-interval 1 --vocab-size 50304 --n-layer 4 --n-head 8 --n-kv-head 8 --n-embd 128 --sequence-len 256 --batch-size 4 --ca-init-rule rule30 --ca-init-alpha 0.5 --artifacts-dir artifacts --artifacts-kind ca_init --artifacts-topic m32/runs/attn_heavy__ca_mix0.5 --run-id seed0 --auto-download-data --min-parquet-files 2
```

## Budget

- steps: 120
- warmup_steps: 5
- tokens/step (global): 1,024
- FLOPs/token (est): 44,924,928
- FLOPs/step (est): 46,003,126,272
- planned_total_FLOPs (est): 5,520,375,152,640

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
- measured_tokens: 117,760
- measured_time_s: 139.476
- tokens/s: 844
- TFLOP/s (est): 0.04
- peak_memory_allocated_gb: n/a
- final_train_ce: 7.8547
- val_interval: 0
- val_batches: n/a
- final_val_ce: n/a

See `summary.json` for full details.
