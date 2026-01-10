# nanochat run (fixed FLOPs)

- Run ID: `gjm_smoke_cpu`
- Generated: 2025-12-18T05:59:09Z
- Artifacts: `bench/fixed_flops/nanochat/gjm_smoke_cpu`
- Commit: 95ab717124a53d816acfdc82d21628e443b811df (dirty)

## Command

```bash
uv run python -m nanochat.train --device cpu --auto-download-data --min-parquet-files 2 --attention-type standard --optimizer-type adamw --batch-size 4 --sequence-len 128 --target-flops 5e9 --warmup-steps 1 --log-interval 1 --artifacts-dir artifacts --artifacts-kind bench --artifacts-topic fixed_flops/nanochat --run-id gjm_smoke_cpu
```

## Budget

- steps: 1
- warmup_steps: 1
- tokens/step (global): 512
- FLOPs/token (est): 44,138,496
- FLOPs/step (est): 22,598,909,952
- planned_total_FLOPs (est): 22,598,909,952

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

- measured_steps: 0
- measured_tokens: 0
- measured_time_s: 0.000
- tokens/s: 0
- TFLOP/s (est): 0.00
- peak_memory_allocated_gb: n/a

See `summary.json` for full details.
