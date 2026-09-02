# nanochat run (fixed FLOPs)

- Run ID: `seed_2`
- Generated: 2026-09-02T22:24:38Z
- Artifacts: `runs/budget_0/needle/standard/seed_2`
- Commit: e7f5845ddacf19a82d162ea3d846f04aa881935f
- Attention schedule: `standard`

## Command

```bash
uv run python -m nanochat.train --device cpu --seed 2 --batch-size 4 --sequence-len 256 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64 --learning-rate 0.0006 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --attention-type standard --target-flops 2000000000000.0 --checkpoint-interval 1000 --checkpoint-keep 1 --data-dir /data/projects/model_guided_research/artifacts/probes/sizing/scorecards/vdc4-needle-tasktok-shuf-b2-s256-rung/data/needle --min-parquet-files 2 --artifacts-dir /data/projects/model_guided_research/artifacts/probes/sizing/scorecards/vdc4-needle-tasktok-shuf-b2-s256-rung --artifacts-kind runs --artifacts-topic budget_0/needle/standard --run-id seed_2 --check-numerics
```

## Budget

- steps: 1767
- warmup_steps: 0
- tokens/step (global): 1,024
- FLOPs/token (est): 1,105,920
- FLOPs/step (est): 1,132,462,080
- planned_total_FLOPs (est): 2,001,060,495,360

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

- measured_steps: 1767
- measured_tokens: 1,809,408
- measured_time_s: 83.029
- tokens/s: 21,793
- TFLOP/s (est): 0.02
- peak_memory_allocated_gb: n/a
- final_train_ce: 2.9664
- val_interval: 0
- val_batches: n/a
- final_val_ce: n/a
- final_val_bpb: n/a (bits per byte; comparable across tokenizers)

See `summary.json` for full details.
