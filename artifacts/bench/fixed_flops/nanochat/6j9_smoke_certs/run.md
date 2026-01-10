# Fixed-FLOPs nanochat benchmark

- Run ID: `6j9_smoke_certs`
- Baseline: `standard`
- Device: `cpu`
- Target FLOPs/run (est): `2.000e+09`
- Seed: `0`

## Results

| attention_type | status | score | Δ vs baseline | tokens/s | TFLOP/s(est) | peak_mem_gb |
| --- | --- | --- | --- | --- | --- | --- |
| standard | ok | 10.825837 | +0.00% | 122 | 0.01 | n/a |
| reversible | ok | 10.825837 | +0.00% | 185 | 0.01 | n/a |

## Conclusions

- Best (lowest score): `standard` score=`10.825837`
- Baseline `standard` score=`10.825837`; best Δ=`+0.00%`
- Worse than baseline: `reversible` (+0.00%)

## Demo Certificates (Diagnostics)

- `tropical` status=error summary=None
- `reversible` status=error summary=None
- `matrix-gauge` status=error summary=None
- `simplicial` status=error summary=None
- `ultrametric` status=error summary=None

## Command

```bash
/data/projects/model_guided_research/.venv/bin/mgr bench-fixed-flops -a standard -a reversible --run-id 6j9_smoke_certs --device cpu --target-flops 2e9 --warmup-steps 0 --batch-size 2 --sequence-len 64 --n-layer 2 --n-head 4 --n-kv-head 4 --n-embd 128 --include-demo-certs --timeout-s 900
```
