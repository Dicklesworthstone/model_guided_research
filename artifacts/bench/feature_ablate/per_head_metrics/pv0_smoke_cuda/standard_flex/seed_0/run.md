# nanochat run (fixed FLOPs)

- Run ID: `seed_0`
- Generated: 2025-12-18T11:44:56Z
- Artifacts: `bench/feature_ablate/per_head_metrics/pv0_smoke_cuda/standard_flex/seed_0`
- Commit: 95ab717124a53d816acfdc82d21628e443b811df (dirty)

## Command

```bash
uv run python -m nanochat.train --device cuda --seed 0 --batch-size 4 --sequence-len 64 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64 --learning-rate 0.0006 --optimizer-type adamw --attention-type standard --target-flops 200000000.0 --warmup-steps 0 --log-interval 1 --artifacts-dir artifacts --artifacts-kind bench --artifacts-topic feature_ablate/per_head_metrics/pv0_smoke_cuda/standard_flex --run-id seed_0 --standard-record-attn-entropy --use-flex-attention --auto-download-data --min-parquet-files 2
```

## Budget

- steps: 1
- warmup_steps: 0
- tokens/step (global): 256
- FLOPs/token (est): 19,955,712
- FLOPs/step (est): 5,108,662,272
- planned_total_FLOPs (est): 5,108,662,272

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
- measured_tokens: 256
- measured_time_s: 0.676
- tokens/s: 379
- TFLOP/s (est): 0.01
- peak_memory_allocated_gb: 0.20455169677734375

See `summary.json` for full details.
