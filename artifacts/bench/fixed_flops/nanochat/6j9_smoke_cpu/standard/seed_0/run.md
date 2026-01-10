# nanochat run (fixed FLOPs)

- Run ID: `seed_0`
- Generated: 2025-12-18T06:53:27Z
- Artifacts: `bench/fixed_flops/nanochat/6j9_smoke_cpu/standard/seed_0`
- Commit: 95ab717124a53d816acfdc82d21628e443b811df (dirty)

## Command

```bash
uv run python -m nanochat.train --device cpu --seed 0 --batch-size 2 --sequence-len 64 --n-layer 2 --n-head 2 --n-kv-head 2 --n-embd 64 --learning-rate 0.0006 --optimizer-type adamw --attention-type standard --target-flops 2000000000.0 --warmup-steps 0 --log-interval 1 --artifacts-dir artifacts --artifacts-kind bench --artifacts-topic fixed_flops/nanochat/6j9_smoke_cpu/standard --run-id seed_0 --auto-download-data --min-parquet-files 2
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
- measured_time_s: 0.522
- tokens/s: 245
- TFLOP/s (est): 0.00
- peak_memory_allocated_gb: n/a

See `summary.json` for full details.
