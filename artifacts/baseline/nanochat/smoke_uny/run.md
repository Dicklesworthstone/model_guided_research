# nanochat baseline (fixed FLOPs)

- Run ID: `smoke_uny`
- Generated: 2025-12-17 22:32:04 EST
- Commit: 95ab717124a53d816acfdc82d21628e443b811df (dirty)

## Command

```bash
uv run python -m nanochat.train --device cuda --auto-download-data --target-flops 1e11 --log-interval 20 --warmup-steps 2 --run-id smoke_uny
```

## Budget

- steps: 2
- warmup_steps: 2
- tokens/step (global): 2,048
- FLOPs/token (est): 44,924,928
- FLOPs/step (est): 92,006,252,544
- planned_total_FLOPs (est): 184,012,505,088

## Results (measured after warmup)

- measured_steps: 0
- measured_tokens: 0
- measured_time_s: 0.000
- tokens/s: 0
- TFLOP/s (est): 0.00
- peak_memory_allocated_gb: 1.5425324440002441

See `summary.json` for full details.
