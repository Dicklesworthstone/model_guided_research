# Regression Report

- generated_at: `2025-12-18 05:39:47 EST`
- baseline: `artifacts/baseline/nanochat/x8u_smoke_compile_standard/summary.json`
- candidate: `artifacts/baseline/nanochat/x8u_smoke_compile_flex/summary.json`

## Thresholds

```json
{
  "loss_abs": 0.01,
  "loss_rel": 0.01,
  "memory_rel": 0.05,
  "tflops_rel": 0.05,
  "throughput_rel": 0.05
}
```

## Metrics

| metric | baseline | candidate | delta | delta% | status |
| --- | ---: | ---: | ---: | ---: | --- |
| Final loss | 10.8114 | 10.8114 | +3.8147e-06 | +0.00% | ok |
| Tokens/s | 6529.44 | 4546.93 | -1982.52 | -30.36% | regression |
| TFLOP/s (est) | 0.130621 | 0.0909607 | -0.03966 | -30.36% | regression |
| Peak mem (GB) | 0.14698 | 0.161346 | +0.0143657 | +9.77% | regression |

## Loss sparklines (tail)

- baseline: `▇█▁`
- candidate: `▇█▁`

## Command

```bash
/data/projects/model_guided_research/.venv/bin/mgr regressions -b baseline/nanochat/x8u_smoke_compile_standard -c baseline/nanochat/x8u_smoke_compile_flex --run-id sample_compile_flex_vs_standard --artifacts-dir artifacts --write-artifacts --html
```
