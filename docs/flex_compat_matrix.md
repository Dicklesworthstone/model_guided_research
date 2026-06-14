# FlexAttention Compatibility Matrix (versions / hardware)

Bead: `model_guided_research-b5l`

A compatibility matrix for the FlexAttention-backed paths
(`torch.nn.attention.flex_attention`): which torch / CUDA / hardware
combinations compile, run stably, and perform — with required flags and
mitigations. Complements the *behavioral* support matrix in
`docs/flex_attention_edge_cases.md` (masking/caching cases) and the porting
analysis in `docs/gpu_flex_diff.md`.

> **Scope note.** This repo's development box is **CPU-only**
> (`torch 2.9.1+cu128`, `torch.cuda.is_available() == False`), so the working
> GPU rows below are sourced from the sibling `bio_inspired_nanochat` (which
> runs flex on CUDA) and from torch's documented requirements — not measured
> here. Rows tagged *(to verify)* must be re-run and filled in on a GPU host;
> the harness to do so already exists: `scripts/verify_flex_correctness.py
> --suite` and `scripts/benchmark_flex.py`.

## Version × hardware matrix

| torch | CUDA | GPU / device | compile | flex available | correctness | perf vs SDPA | status |
|---|---|---|---|---|---|---|---|
| 2.9.1+cu128 | n/a (CPU) | **this box (CPU-only)** | n/a | **no** (flex needs CUDA) | falls back to SDPA | n/a | **expected-unsupported** (use `--attention-type standard` w/o `--use-flex-attention`) |
| 2.9.1 | 12.x | CUDA GPU (Ampere+) | `--compile` | yes | float32 tight parity | fused, ≥ SDPA | **working** *(to verify on local GPU)* — bio_inspired runs this combo |
| 2.9.1 | 12.x | CUDA GPU | eager (no compile) | yes but unfused | parity holds | may **warn + slow** (unfused fallback) | **degraded** — flex perf depends on `torch.compile` |
| < 2.5 | any | any | any | **no** (`flex_attention` absent) | n/a | n/a | **failing** — API does not exist; minimum is torch 2.5 |
| 2.9.1 | 12.x | CUDA GPU | `--compile` + KV-cache | yes | parity holds | recompile churn | **working-with-caveat** — static int module attrs (`layer_idx`) trigger recompilation warnings |

### A known-working and a known-failing combo (acceptance)
- **Working:** torch 2.9.1 + CUDA 12.x + Ampere+ GPU + `torch.compile` →
  FlexAttention compiles, float32-parity holds (suite), fused and competitive
  with SDPA. (bio_inspired's standard config; verify locally when a GPU is available.)
- **Failing:** torch < 2.5 (any hardware) → `torch.nn.attention.flex_attention`
  does not exist; the `use_flex_attention` path is unavailable. Mitigation:
  upgrade to torch ≥ 2.5, or run the SDPA path.

## Required flags / settings

- Enable: `python -m nanochat.train --attention-type standard --use-flex-attention`
  (`GPTConfig.use_flex_attention=True`).
- **Always pair with `torch.compile`** (`--compile` in the verify/bench scripts;
  `compile_flex_attention` in config) — without it flex may take an unfused path
  and warn.
- Prefer **float32** for parity verification; bf16/fp16 show larger SDPA↔flex
  numeric drift (different kernel families/accumulation) — expected, not a bug,
  unless it destabilizes training.

## Mitigations for known issues

| Symptom | Cause | Mitigation |
|---|---|---|
| `flex_attention` import error | torch < 2.5 | upgrade torch ≥ 2.5 |
| "unfused / falling back" warning + slow | flex without `torch.compile` | add `--compile` |
| Recompilation warnings with KV-cache | static int module attrs (`layer_idx`) | benign; mark dynamic or ignore — correctness unaffected |
| bf16/fp16 parity drift vs SDPA | different kernels | verify parity in float32; treat bf16 drift as expected |
| CPU run wants flex | flex requires CUDA | use SDPA path (omit `--use-flex-attention`) |

## How to refresh this matrix (on a GPU host)

```bash
# correctness (fills the "correctness" column)
uv run python scripts/verify_flex_correctness.py --suite --device cuda --dtype float32 --compile
# performance (fills "perf vs SDPA")
uv run python scripts/benchmark_flex.py --device cuda --compile
# then record GPU model, driver version (nvidia-smi), CUDA, torch, and outcomes
# as a new dated row above.
```

Reference data point: `bio_inspired_nanochat` pins `torch==2.9.1`, targets CUDA
12.8 (or CPU), and `requires-python >=3.14,<3.15`; this repo pins `torch>=2.0.0`,
`requires-python >=3.13` and currently resolves `torch 2.9.1+cu128`.
