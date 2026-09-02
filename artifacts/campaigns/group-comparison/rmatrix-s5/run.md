# nanochat run (fixed FLOPs)

- Run ID: `rmatrix-s5`
- Generated: 2026-09-02T21:45:08Z
- Artifacts: `campaigns/group-comparison/rmatrix-s5`
- Commit: 5dce958e2cb119015937e7ca756ebcdde8b347d2
- Attention schedule: `braid`

## Command

```bash
uv run python -m nanochat.train --device cpu --seed 5 --target-flops 1e12 --attention-type braid --braid-crossing-law rmatrix --batch-size 4 --sequence-len 64 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64 --learning-rate 6e-4 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 50 --checkpoint-interval 1000 --checkpoint-keep 1 --data-dir /data/projects/model_guided_research/artifacts/probes/sizing/scorecards/vdc4-group-tasktok-shuf-b1-rung/data/group --min-parquet-files 2 --artifacts-dir /data/projects/model_guided_research/artifacts --artifacts-kind campaigns --artifacts-topic group-comparison --run-id rmatrix-s5
```

## Budget

- steps: 4781
- warmup_steps: 0
- tokens/step (global): 256
- FLOPs/token (est): 817,200
- FLOPs/step (est): 209,203,200
- planned_total_FLOPs (est): 1,000,200,499,200

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

- measured_steps: 4781
- measured_tokens: 1,223,936
- measured_time_s: 62.455
- tokens/s: 19,597
- TFLOP/s (est): 0.02
- peak_memory_allocated_gb: n/a
- final_train_ce: 0.4425
- val_interval: 0
- val_batches: n/a
- final_val_ce: n/a
- final_val_bpb: n/a (bits per byte; comparable across tokenizers)

See `summary.json` for full details.
