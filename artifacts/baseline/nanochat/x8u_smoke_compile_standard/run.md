# nanochat baseline (fixed FLOPs)

- Run ID: `x8u_smoke_compile_standard`
- Generated: 2025-12-17 23:45:01 EST
- Commit: 95ab717124a53d816acfdc82d21628e443b811df (dirty)

## Command

```bash
uv run python -m nanochat.train --device cuda --batch-size 2 --sequence-len 64 --n-layer 2 --n-head 4 --n-kv-head 4 --n-embd 64 --max-steps 3 --warmup-steps 1 --log-interval 1 --run-id x8u_smoke_compile_standard --compile --compile-mode default
```

## Budget

- steps: 3
- warmup_steps: 1
- tokens/step (global): 128
- FLOPs/token (est): 20,004,864
- FLOPs/step (est): 2,560,622,592
- planned_total_FLOPs (est): 7,681,867,776

## Compilation

- torch.compile: True
- compile_backend: 'inductor'
- compile_mode: 'default'
- compile_fullgraph: False
- compile_dynamic: None
- compile_flex_attention: False

## Results (measured after warmup)

- measured_steps: 2
- measured_tokens: 256
- measured_time_s: 0.039
- tokens/s: 6,529
- TFLOP/s (est): 0.13
- peak_memory_allocated_gb: 0.14698028564453125

See `summary.json` for full details.
