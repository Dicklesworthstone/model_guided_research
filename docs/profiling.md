# Profiling & instrumentation (`nanochat.profiling`)

Bead **b1l**. Optional, default-off hooks for measuring where time and memory go
in the attention / optimizer hot paths — `torch.profiler` for a kernel/memory
breakdown, NVTX ranges for nsys / Nsight Systems timelines.

**Overhead when off is negligible by construction**: a disabled
`torch_profiler` constructs no profiler (it just yields `None`), and `nvtx_range`
is a true no-op without CUDA (one `torch.cuda.is_available()` check on entry).

## The reusable hooks

```python
from nanochat.profiling import nvtx_range, torch_profiler, ProfileConfig, summarize_profile

# NVTX marker — free on CPU / when not profiling; shows as a range in nsys.
with nvtx_range("attention"):
    out = attn(x, cos_sin, kv_cache)

# A profiling session (CPU + CUDA activities, shapes, memory). None when off.
cfg = ProfileConfig.from_env()            # NANOCHAT_PROFILE=1 enables
with torch_profiler(cfg) as prof:
    train_step()
if prof is not None:
    print(summarize_profile(prof, row_limit=15))
```

### Toggles

CLI flags on the microbench (below), or environment variables anywhere:

| env var | meaning | default |
| --- | --- | --- |
| `NANOCHAT_PROFILE` | enable the session (`1/true/on`) | off |
| `NANOCHAT_PROFILE_DIR` | directory for the exported Chrome trace | none |
| `NANOCHAT_PROFILE_MEMORY` | record per-op memory | on |
| `NANOCHAT_PROFILE_STACK` | record python stacks (heavier) | off |
| `NANOCHAT_PROFILE_ROWS` | rows in the summary table | 15 |

## Standalone microbench (sample traces, Flex on/off)

Profiles a mechanism's forward(+backward) hot path on a small seeded model,
prints a kernel/memory breakdown, and exports a Chrome trace (`trace.json` —
open in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev)).

```bash
# One mechanism (fwd+bwd by default).
python -m nanochat.profiling bench --attention standard --out artifacts/profiles/std
python -m nanochat.profiling bench --attention tropical --steps 8

# FlexAttention ON vs OFF for standard attention (skips cleanly if Flex is
# unavailable). On GPU pass --device cuda; combine with torch.compile in the
# trainer for the fused path — the eager CPU Flex path is intentionally slower.
python -m nanochat.profiling bench --attention standard --compare-flex --device cpu

# Forward-only (inference path), bigger batch:
python -m nanochat.profiling bench --attention standard --forward-only --batch-size 32
```

Knobs: `--device {cpu,cuda}`, `--steps`, `--warmup` (kept out of the trace),
`--n-layer/--n-head/--n-kv-head/--n-embd`, `--seq-len`, `--batch-size`,
`--row-limit`, `--seed`, `--out`.

### Reading the breakdown

The table ranks ops by **self time** on the active device (CUDA when present,
else CPU), with CPU-total and per-op memory:

- **`aten::mm` / `aten::addmm` / `aten::bmm`** dominate a healthy transformer —
  the QKV / output / MLP GEMMs. If a non-matmul op (gather, copy, softmax)
  rivals them, that mechanism has an un-fused hot spot worth a kernel.
- **`FlexAttentionAutogradOp*`** appears on the Flex path; compare its self time
  to SDPA on the `--compare-flex` roll-up. Eager (un-compiled) Flex is slower —
  the fused win needs `torch.compile`.
- **memory** column flags allocation-heavy ops (materialized score matrices,
  intermediate copies) — candidates for the reversible / O(1)-memory paths.

`summary.json` carries the full per-op records + a `meta` block (config, params,
peak device memory, trace path) for machine consumption / regression tracking.

## Notes

- **Device-agnostic** — on a CPU box the CUDA columns are zero and the table
  reports CPU time/memory; the same code yields CUDA kernel times on a GPU.
- **Trainer integration (planned)** — these hooks are built to drop into the
  training loop: wrap the attention/optimizer steps in `nvtx_range` and gate a
  `torch_profiler` session on `ProfileConfig.from_env()`, so a run emits a trace
  with zero overhead when off. Until that wiring lands, the standalone microbench
  above is the entry point and profiles the same forward/backward hot paths.
</content>
