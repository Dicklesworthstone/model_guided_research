# Dataloader Performance Tuning

**Bead:** `model_guided_research-2kz`
**Date:** 2026-06-14
**Measured on:** 64-core CPU box, `fineweb-edu-100b-shuffle` shard 0, rustbpe
tokenizer (vocab 50257), `nanochat/dataloader.py` at `B=8, T=256`.

**Bottom line:** the streaming dataloader is **buffer-bound and very fast at
steady state (~7.0M tokens/s)** but has a **bimodal latency**: every ~57
batches it pays a **synchronous ~200 ms tokenization refill** that blocks the
training step (~7,800 tokens/s during that step). On CPU this is invisible (the
model step dwarfs it); **on GPU it would periodically starve the device.** The
high-leverage fix is **async prefetch (overlap tokenization with compute)**,
not tweaking thread counts. The loader is identical in `bio_inspired_nanochat`,
so the same conclusion applies to both.

## Measurement

Per-batch latency over 400 batches (`tokenizer_threads=4, tokenizer_batch_size=128`):

| metric | value |
|---|---|
| median | 0.292 ms → **7.0M tokens/s** |
| mean | 4.10 ms (dragged up by stalls) |
| p99 | 204 ms |
| max | 261 ms → **7,800 tokens/s** during a refill |
| refill spikes (>5× median) | 7 in 400 batches (~1 per 57) |

Throughput sweeps (steady-state, buffer-served):

| knob | result |
|---|---|
| `tokenizer_threads ∈ {1,2,4,8}` | 6.74M → 6.95M tok/s (**~3 % spread**; ~no effect) |
| `tokenizer_batch_size ∈ {128,256,512}` | 6.89M → 6.98M tok/s (~no effect) |
| `tokenizer_batch_size = 64` | catches a refill in-window (slow); use ≥128 |

## Why bimodal (the mechanics)

`tokenizing_distributed_data_loader` (`dataloader.py:77`) keeps a `deque` token
buffer. It serves a `B·T+1`-token batch by popping from the buffer
(microseconds). When the buffer drops below `needed_tokens`, it **synchronously**
calls `tokenizer.encode(doc_batch, num_threads=…)` on the next
`tokenizer_batch_size` documents (`dataloader.py:79–83`) — a ~200 ms burst for
128 FineWeb-edu docs — and refills. So:

- **steady state** = deque pops → ~7M tok/s, independent of thread/batch knobs.
- **refill** = one synchronous `encode()` of `tokenizer_batch_size` docs blocks
  the calling (training) step.
- **cadence** ≈ `tokens_per_encode / (B·T)` batches between stalls. 128 docs ≈
  ~117K tokens ≈ 57 batches at `B·T=2048`, matching the measured ~1-per-57.

There is **no** background prefetch, **no** `torch.utils.data.DataLoader`
worker pool, and **no** double-buffering — tokenization runs inline in the
training loop.

## Recommendations

### 1. Async prefetch / double-buffering — delivered (`--dataloader-prefetch N`, bead atkp)
Overlap the `encode()` refill with compute: a daemon thread runs the
document→token tokenization ahead of the consumer into a bounded queue, so the
training step never blocks on a refill. The stall is fully hidden iff
`encode_time < buffer_drain_time = (tokens_per_encode / (B·T)) · step_time`.
At GPU step times this is the difference between a fed and a periodically
starved device. Default `0` keeps the synchronous loop; the yielded batches
and the recorded resume positions are identical in both modes
(`tests/test_dataloader_prefetch.py`). A producer error is raised at the
consumer, and the thread exits when the loader generator is closed.

### 2. `tokenizer_threads` — set to physical cores **on GPU**
Irrelevant at steady state (amortized), but it parallelizes each refill burst,
shortening the stall. Default 4 is fine on CPU (model-bound); on GPU set it to
the physical core count to minimize each synchronous stall (until prefetch
lands, this is the cheapest mitigation).

### 3. `tokenizer_batch_size` — keep ≥128 (256 is fine)
Controls refill granularity, not total work: larger ⇒ less frequent but larger
stalls. 128–256 balances stall size vs frequency. Avoid 64 (more frequent
stalls, and it skews short benchmarks).

### 4. `pin_memory` / `non_blocking` — already optimal
Enabled automatically iff `device.type=="cuda"` (`dataloader.py:87–95`); the
H2D copy already overlaps. No change needed. On CPU they are correctly off.

### 5. "num_workers / prefetch" — N/A as written
This is a **custom streaming generator**, not `torch.utils.data.DataLoader`, so
there is no `num_workers`/`prefetch_factor` knob. The equivalent lever is the
prefetch thread in (1). Don't try to bolt a DataLoader worker pool onto a
parquet-row-group stream; add the prefetch thread instead.

## GPU-deferred items

This box is CPU-only (`torch.cuda.is_available()==False`), so the numbers above
are CPU. To complete the GPU side, on a CUDA box capture: (a) GPU step time at
the target rung, (b) confirm the ~200 ms refill stalls show up as GPU idle gaps
in `torch.profiler` (ties to bead `b1l`), (c) verify prefetch (item 1) removes
them. The decision rule is the inequality in (1); the CPU encode time (~200 ms /
128 docs / 4 threads) is the input to it.

## Parity note

`bio_inspired_nanochat/dataloader.py` is the same algorithm (see
`docs/data_parity.md`) with the same synchronous-refill behaviour and no
prefetch — so this tuning (and the prefetch follow-up) applies equally there.
No correctness change is proposed; this is a throughput/overlap optimization.
</content>
