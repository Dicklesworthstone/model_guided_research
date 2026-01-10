# nanochat baseline (fixed FLOPs)

- Run ID: `0jk_baseline_flex`
- Generated: 2025-12-17 23:13:53 EST
- Commit: 95ab717124a53d816acfdc82d21628e443b811df (dirty)

## Command

```bash
uv run python -m nanochat.train --device cuda --auto-download-data --target-flops 2e12 --log-interval 10 --warmup-steps 5 --use-flex-attention --run-id 0jk_baseline_flex
```

## Budget

- steps: 22
- warmup_steps: 5
- tokens/step (global): 2,048
- FLOPs/token (est): 44,924,928
- FLOPs/step (est): 92,006,252,544
- planned_total_FLOPs (est): 2,024,137,555,968

## Results (measured after warmup)

- measured_steps: 17
- measured_tokens: 34,816
- measured_time_s: 1.677
- tokens/s: 20,761
- TFLOP/s (est): 0.93
- peak_memory_allocated_gb: 1.542665958404541

See `summary.json` for full details.
