# nanochat baseline (fixed FLOPs)

- Generated: 2025-12-17 22:17:29 EST
- Commit: 95ab717124a53d816acfdc82d21628e443b811df (dirty)

## Command

```bash
/data/projects/model_guided_research/nanochat/train.py --device cuda --auto-download-data --target-flops 2e12 --log-interval 10 --warmup-steps 5
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
- measured_time_s: 0.318
- tokens/s: 109,598
- TFLOP/s (est): 4.92
- peak_memory_allocated_gb: 1.5425324440002441

See `summary.json` for full details.
