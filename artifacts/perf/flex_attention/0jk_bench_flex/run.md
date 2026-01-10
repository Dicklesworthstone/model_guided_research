# FlexAttention perf benchmark (SDPA vs Flex)

- Run ID: `0jk_bench_flex`
- Generated: 2025-12-17 23:14:40 EST

## Command

```bash
uv run python scripts/benchmark_flex.py --device cuda --dtype bfloat16 --run-id 0jk_bench_flex
```

## Config

- device: `cuda`
- dtype: `bfloat16`
- batch_size: 8
- sequence_len: 512
- n_layer/n_head/n_kv_head/n_embd: 8/8/4/512
- torch.compile: True

## Results

- ms/iter: sdpa=4.41  flex=10.49
- tokens/s: sdpa=928,928  flex=390,619
- speedup (sdpa/flex): 0.421×
- peak MB: sdpa=1044.4  flex=1817.5

See `summary.json` for full details.
