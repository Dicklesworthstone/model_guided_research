# Triton Hotspot Benchmarking Plan

**Bead:** `model_guided_research-c6h`
**Date:** 2026-06-14
**Status:** plan only (no code). Adapts the *methodology* of
`bio_inspired_nanochat/PLAN_TO_OPTIMIZE_KEY_HOTSPOTS_USING_TRITON.md` (kernel
fusion, feature-flagged golden tests) to **this repo's measured hotspots**.

## The hotspot is identified by measurement, not guessed

`docs/flops_validation.md` (bead `bks`) measured effective hardware efficiency
per mechanism (fwd+bwd, CPU, E1 rung):

| Mechanism | effective GFLOP/s | vs standard |
|---|---|---|
| standard / reversible | 62–71 | 1.0× (BLAS-bound, healthy) |
| **tropical (attn)** | **34** | **0.5×** |
| **tropical-hybrid (attn+FFN)** | **11** | **0.17×** |

Tropical (max-plus) is **3–6× less hardware-efficient** than the matmul
mechanisms — the clear #1 Triton target. The root cause is concrete:
`tropical_inner` (`nanochat/tropical_attention_torch.py:21`) computes scores as

```python
torch.max(q.unsqueeze(-2) + k.unsqueeze(-3), dim=-1).values  # (B,H,T,T) from a (B,H,T,T,d) broadcast
```

i.e. it **materializes a `(B,H,T,T,d)` tensor** then reduces — O(T²·d) memory,
memory-bandwidth-bound, no BLAS. The aggregation
(`y_d = max_k(score + v_{k,d})`) is the same pattern. This is exactly the
shape FlashAttention solved for softmax: **tile over K/V in SRAM and never
materialize the T×T(×d) intermediate** — but with `(max, +)` instead of
`(softmax, ·)`.

## Candidate ops (ranked by measured payoff)

1. **Fused tropical max-plus attention** (`tropical_attention_torch.py:21,32`).
   FlashAttention-style streaming: load Q tile + K/V tiles to SRAM, maintain a
   running `max` over the score `max_d(q_d+k_d)` and a running max-plus
   aggregation over V, never materialize `(T,T)` or `(T,T,d)`. Expected: kill
   the O(T²·d) memory traffic; the largest single win (34→ toward BLAS-band).
2. **Fused tropical FFN max-plus layer** (`tropical_maxplus_layer`,
   `tropical_attention_torch.py:35–52`): `out_j = max_d(W_jd + x_d)`. Same
   broadcast-then-max pattern; fuse the `(+)`-then-`amax` (and the Maslov
   `logsumexp(beta·)` soft path) into one kernel. The hybrid's 11 GFLOP/s says
   the FFN matters as much as attention.
3. **Ultrametric LCP kernel** (`ultrametric_attention_torch.py`): kernel-mode is
   O(T²); the trie path is CPU-only. A Triton LCP-weighted aggregation is a
   secondary target (also tracked structurally by bead `a1o`).
4. **Gauge parallel-transport** (`gauge_block_torch.py`): Givens-rotation
   transport is many small element-wise ops — fusion candidate, lower priority
   (gauge was not measured as a bandwidth outlier).

Do **not** port bio's presynaptic/genetics/router kernels — those target its
bio-inspired synaptic model, which this repo does not have.

## Phase 0 — groundwork (prereq for any kernel)

| Goal | Task |
|---|---|
| Layout | new `nanochat/kernels/` package; add `triton>=3.0.0` to `pyproject.toml` (GPU extra) |
| Feature flags | `NANOCHAT_FUSED_TROPICAL=1` env (and a config field) to toggle PyTorch reference vs Triton; default OFF |
| Parity gate | reuse the **attention goldens harness** (`tests/test_attention_core_goldens.py`) — the Triton kernel must reproduce the bitwise/within-tol trajectory of the PyTorch reference before it is trusted (the `new_mechanism_checklist.md` "goldens recapture" + "exact reduction to known" discipline) |
| FLOPs accounting | the kernel changes throughput, **not** arithmetic FLOPs, so `estimate_flops` is unchanged; fixed-FLOPs budgets stay valid (cf. `docs/flops_validation.md`) |

## Microbench checklist (before/after any kernel)

Per candidate op, isolated from the training loop:

- **Shapes:** sweep `B ∈ {4,8}`, `T ∈ {256,512,1024,2048}`, `head_dim ∈ {32,64}`,
  `n_head ∈ {4,8}` (T is the axis where the O(T²·d) materialization bites).
- **Dtypes:** fp32 + bf16 (the training dtype, `train.py:1028`).
- **Metrics:** latency (median of N after warmup), achieved GB/s and GFLOP/s,
  PyTorch-reference vs Triton speedup, peak memory (the materialized tensor is
  the thing being eliminated — expect the largest delta here).
- **Correctness:** max-abs / rel error vs the PyTorch reference at each shape
  (max-plus is exact in fp32; bf16 has a tolerance), plus the goldens gate.
- **Profiler:** Nsight Compute / `torch.profiler` to confirm kernel-count
  collapse and SRAM residency (ties to bead `b1l`).
- **Artifacts:** write timing tables to `artifacts/microbench/triton/` with the
  exact commands (the `2cg` per-op microbench harness is the natural home).

## Prerequisites / constraints

- **GPU + Triton required** — this audit's box is CPU-only, so the kernels and
  their GPU benchmarks are deferred; the *target selection* above is the
  CPU-measurable deliverable and is done.
- Land the **FlexAttention** path first for `standard` (already present,
  `--use-flex-attention`) so the fused-kernel pattern (SRAM tiling) has a
  reference; tropical is the same idea with a different semiring.
- Each kernel ships behind its flag with a passing golden + microbench delta;
  one op per PR.

## Success metrics

- Tropical attention effective GFLOP/s from ~34 toward the matmul band (~60+),
  i.e. **≥1.5–2× tropical-arm throughput**, with **bitwise/within-tol golden
  parity** and **unchanged `estimate_flops`**.
- Peak memory for tropical attention drops from O(T²·d) to O(T·d) tiles.

## Follow-up

- This plan unblocks the Triton-kernel implementation bead (`7b0.7` per the
  dependency graph). File the Phase-1 fused-tropical-attention kernel as its own
  bead `--deps discovered-from:model_guided_research-c6h` when a GPU box is
  available. The companion Rust/PyO3 study (`2xa`) should defer to this: for
  high-frequency on-device attention ops, Triton (on-GPU, no H2D copy) beats
  PyO3 — reserve Rust for CPU-side or host-bound work.
</content>
