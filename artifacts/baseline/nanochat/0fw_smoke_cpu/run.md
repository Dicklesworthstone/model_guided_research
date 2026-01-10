# nanochat baseline (fixed FLOPs)

- Run ID: `0fw_smoke_cpu`
- Generated: 2025-12-18 00:26:19 EST
- Commit: 95ab717124a53d816acfdc82d21628e443b811df (dirty)

## Command

```bash
uv run python -m nanochat.train --device cpu --batch-size 2 --sequence-len 64 --n-layer 2 --n-head 4 --n-kv-head 4 --n-embd 64 --max-steps 2 --warmup-steps 0 --log-interval 1 --check-numerics --run-id 0fw_smoke_cpu
```

## Budget

- steps: 2
- warmup_steps: 0
- tokens/step (global): 128
- FLOPs/token (est): 20,004,864
- FLOPs/step (est): 2,560,622,592
- planned_total_FLOPs (est): 5,121,245,184

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

- measured_steps: 2
- measured_tokens: 256
- measured_time_s: 0.668
- tokens/s: 383
- TFLOP/s (est): 0.01
- peak_memory_allocated_gb: n/a

See `summary.json` for full details.
