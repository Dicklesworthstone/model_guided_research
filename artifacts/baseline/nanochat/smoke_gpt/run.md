# nanochat run (fixed FLOPs)

- Run ID: `smoke_gpt`
- Generated: 2025-12-18T06:33:56Z
- Artifacts: `baseline/nanochat/smoke_gpt`
- Commit: 95ab717124a53d816acfdc82d21628e443b811df (dirty)

## Command

```bash
uv run python -m nanochat.train --model-type gpt --device cpu --target-flops 2e9 --warmup-steps 0 --batch-size 2 --sequence-len 64 --n-layer 2 --n-head 2 --n-kv-head 2 --n-embd 64 --run-id smoke_gpt --artifacts-kind baseline --artifacts-topic nanochat --auto-download-data --min-parquet-files 2
```

## Budget

- steps: 1
- warmup_steps: 0
- tokens/step (global): 128
- FLOPs/token (est): 20,004,864
- FLOPs/step (est): 2,560,622,592
- planned_total_FLOPs (est): 2,560,622,592

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
- measured_tokens: 128
- measured_time_s: 0.600
- tokens/s: 213
- TFLOP/s (est): 0.00
- peak_memory_allocated_gb: n/a

See `summary.json` for full details.
