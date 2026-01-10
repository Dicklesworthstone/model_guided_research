# nanochat run (fixed FLOPs)

- Run ID: `seed_123`
- Generated: 2025-12-18T06:32:06Z
- Artifacts: `cmaes/phase1/68v_smoke_cpu/eval/gen_0000/cand_0001/seed_123`
- Commit: 95ab717124a53d816acfdc82d21628e443b811df (dirty)

## Command

```bash
uv run python -m nanochat.train --model-type synaptic --synaptic-config artifacts/cmaes/phase1/68v_smoke_cpu/eval/gen_0000/cand_0001/synaptic_config.json --device cpu --seed 123 --batch-size 8 --sequence-len 256 --vocab-size 50304 --n-layer 4 --n-head 4 --n-kv-head 4 --n-embd 128 --learning-rate 0.0006 --target-flops 2000000000.0 --warmup-steps 0 --log-interval 1 --artifacts-dir artifacts --artifacts-kind cmaes --artifacts-topic phase1/68v_smoke_cpu/eval/gen_0000/cand_0001 --run-id seed_123 --auto-download-data --min-parquet-files 2
```

## Budget

- steps: 1
- warmup_steps: 0
- tokens/step (global): 2,048
- FLOPs/token (est): 1,441,792
- FLOPs/step (est): 2,952,790,016
- planned_total_FLOPs (est): 2,952,790,016

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
- measured_tokens: 2,048
- measured_time_s: 3.651
- tokens/s: 561
- TFLOP/s (est): 0.00
- peak_memory_allocated_gb: n/a

See `summary.json` for full details.
