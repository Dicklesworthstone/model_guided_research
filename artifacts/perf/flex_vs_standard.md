# FlexAttention vs Standard (bead model_guided_research-0jk)

- Generated: 2025-12-17 23:17:25 EST
- Host: `threadripperje`
- GPU: NVIDIA GeForce RTX 4090, NVIDIA GeForce RTX 4090
- CUDA: `12.8`
- NVIDIA driver: `580.95.05`
- Torch: `2.9.1+cu128`
- Python (uv): `3.13.0`
- Git commit: `95ab717124a53d816acfdc82d21628e443b811df` (dirty=True)

## Correctness

### Standard GPT (SDPA vs Flex)

- Run: `artifacts/certs/flex_attention/0jk_standard_correctness_postfix2/`
- Overall: **PASS**

Key checks:

| check | max_abs |
|---|---:|
| mqa/full_forward | 7.153e-07 |
| mqa/kv_decode_last | 3.576e-07 |
| mqa/kv_chunk_decode | 4.768e-07 |
| gqa/full_forward | 4.768e-07 |
| gqa/kv_decode_last | 4.768e-07 |
| gqa/kv_chunk_decode | 4.768e-07 |
| mha/full_forward | 5.960e-07 |
| mha/kv_decode_last | 4.768e-07 |
| mha/kv_chunk_decode | 5.960e-07 |


### Synaptic GPT (non-flex vs Flex)

- Run: `artifacts/certs/synaptic_flex_attention/0jk_synaptic_correctness_postfix3/`
- Overall: **PASS**

| check | max_abs |
|---|---:|
| synaptic/full_forward | 0.000e+00 |
| synaptic/kv_decode_last | 1.192e-07 |
| synaptic/kv_chunk_decode | 2.384e-07 |


**Note:** `torch.compile` currently fails for synaptic FlexAttention in KV-cache paths on this stack (see JSON report for the exact note).

## Performance

### Microbench (GPT forward)

- Run: `artifacts/perf/flex_attention/0jk_bench_flex/`

| mode | ms/iter | tokens/s | peak MB |
|---|---:|---:|---:|
| SDPA | 4.41 | 928928 | 1044.4 |
| Flex | 10.49 | 390619 | 1817.5 |

- Speedup (sdpa/flex): 0.421×

### Training baseline (fixed FLOPs)

- SDPA run: `artifacts/baseline/nanochat/0jk_baseline_sdpa/`
- Flex run: `artifacts/baseline/nanochat/0jk_baseline_flex/`

| mode | tokens/s | TFLOP/s (est) | peak_mem_gb |
|---|---:|---:|---:|
| SDPA | 54950 | 2.47 | 1.54 |
| Flex | 20761 | 0.93 | 1.54 |

**Important:** `nanochat.train` currently does **not** `torch.compile` the model. FlexAttention warns that, without compilation, it may run an unfused implementation.

## Repro commands (exact)

- Standard correctness:
  ```bash
  uv run python scripts/verify_flex_correctness.py --model standard --device cuda --dtype float32 --run-id 0jk_standard_correctness_postfix2 --suite
  ```
- Synaptic correctness:
  ```bash
  uv run python scripts/verify_flex_correctness.py --model synaptic --device cuda --dtype float32 --run-id 0jk_synaptic_correctness_postfix3
  ```
- Microbench:
  ```bash
  uv run python scripts/benchmark_flex.py --device cuda --dtype bfloat16 --run-id 0jk_bench_flex
  ```
- Baseline SDPA:
  ```bash
  uv run python -m nanochat.train --device cuda --auto-download-data --target-flops 2e12 --log-interval 10 --warmup-steps 5 --run-id 0jk_baseline_sdpa
  ```
- Baseline Flex:
  ```bash
  uv run python -m nanochat.train --device cuda --auto-download-data --target-flops 2e12 --log-interval 10 --warmup-steps 5 --use-flex-attention --run-id 0jk_baseline_flex
  ```
