# nanochat run (fixed FLOPs)

- Run ID: `gjm_smoke_cpu_w0`
- Generated: 2025-12-18T05:59:32Z
- Artifacts: `bench/fixed_flops/nanochat/gjm_smoke_cpu_w0`
- Commit: 95ab717124a53d816acfdc82d21628e443b811df (dirty)

## Command

```bash
uv run python -m nanochat.train --device cpu --auto-download-data --min-parquet-files 2 --attention-type standard --optimizer-type adamw --batch-size 4 --sequence-len 128 --target-flops 5e9 --warmup-steps 0 --log-interval 1 --artifacts-dir artifacts --artifacts-kind bench --artifacts-topic fixed_flops/nanochat --run-id gjm_smoke_cpu_w0
```

## Budget

- steps: 1
- warmup_steps: 0
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

- measured_steps: 1
- measured_tokens: 512
- measured_time_s: 2.356
- tokens/s: 217
- TFLOP/s (est): 0.01
- peak_memory_allocated_gb: n/a

See `summary.json` for full details.
