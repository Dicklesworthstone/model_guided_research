# Fixed-FLOPs nanochat benchmark

- Run ID: `6j9_smoke_cpu`
- Baseline: `standard`
- Device: `cpu`
- Target FLOPs/run (est): `2.000e+09`
- Seed: `0`

## Results

| attention_type | status | score | Δ vs baseline | tokens/s | TFLOP/s(est) | peak_mem_gb |
| --- | --- | --- | --- | --- | --- | --- |
| standard | ok | 10.825837 | +0.00% | 245 | 0.00 | n/a |
| tropical | ok | 10.825837 | +0.00% | 239 | 0.00 | n/a |

## Conclusions

- Best (lowest score): `standard` score=`10.825837`
- Baseline `standard` score=`10.825837`; best Δ=`+0.00%`
- Worse than baseline: `tropical` (+0.00%)

## Command

```bash
/data/projects/model_guided_research/.venv/bin/mgr bench-fixed-flops -a standard -a tropical --run-id 6j9_smoke_cpu --device cpu --target-flops 2e9 --warmup-steps 0 --batch-size 2 --sequence-len 64 --n-layer 2 --n-head 2 --n-kv-head 2 --n-embd 64 --timeout-s 900
```
