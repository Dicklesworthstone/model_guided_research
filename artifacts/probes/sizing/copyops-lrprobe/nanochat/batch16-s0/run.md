# nanochat run (fixed FLOPs)

- Run ID: `batch16-s0`
- Generated: 2026-11-23T05:47:41Z
- Artifacts: `copyops-lrprobe/nanochat/batch16-s0`
- Commit: 5dce958e2cb119015937e7ca756ebcdde8b347d2
- Attention schedule: `standard`

## Command

```bash
uv run python -m nanochat.train --device cpu --seed 0 --batch-size 16 --sequence-len 64 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64 --learning-rate 0.0006 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --attention-type standard --target-flops 1000000000000 --checkpoint-interval 1000 --checkpoint-keep 0 --data-dir /Users/jemanuel/projects/model_guided_research/artifacts/probes/sizing/scorecards/vdc4-copyops-tasktok-shuf-b1-rung/data/copyops --min-parquet-files 2 --artifacts-dir /Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing --artifacts-kind copyops-lrprobe --artifacts-topic nanochat --run-id batch16-s0 --check-numerics
```

## Budget

- steps: 1282
- warmup_steps: 0
- tokens/step (global): 1,024
- FLOPs/token (est): 761,856
- FLOPs/step (est): 780,140,544
- planned_total_FLOPs (est): 1,000,140,177,408

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

- measured_steps: 1282
- measured_tokens: 1,312,768
- measured_time_s: 12.565
- tokens/s: 104,480
- TFLOP/s (est): 0.08
- peak_memory_allocated_gb: n/a
- final_train_ce: 2.2110
- val_interval: 0
- val_batches: n/a
- final_val_ce: n/a
- final_val_bpb: n/a (bits per byte; comparable across tokenizers)

See `summary.json` for full details.
