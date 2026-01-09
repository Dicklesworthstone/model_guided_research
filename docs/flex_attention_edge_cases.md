# FlexAttention Masking/Caching Edge Cases (Nanochat)

This note captures the **current support matrix** and **known pitfalls** for the FlexAttention-backed paths in this repo.

## What We Have

- **Standard GPT** (`nanochat/gpt.py`): `GPTConfig.use_flex_attention=True` (and `python -m nanochat.train --use-flex-attention`).
- **Synaptic GPT** (`nanochat/gpt_synaptic.py` + `nanochat/synaptic.py`): `SynapticConfig.use_flex_attention=True` (experimental).

## Scripted Parity Checks (Standard GPT)

Run the built-in masking/caching suite (covers **MQA/GQA/MHA** and **KV-cache decode + chunk decode**):

```bash
uv run python scripts/verify_flex_correctness.py --suite
```

Common useful variants:

```bash
# CUDA float32 parity (tight)
uv run python scripts/verify_flex_correctness.py --suite --device cuda --dtype float32 --compile

# CPU-only quick check
uv run python scripts/verify_flex_correctness.py --suite --device cpu --dtype float32
```

## Performance Microbench (Standard GPT)

```bash
uv run python scripts/benchmark_flex.py --device cuda --compile
```

## Masking/Caching Matrix

Legend:
- **PASS**: covered by scripts/tests and behaves as expected.
- **N/A**: not applicable to the current model/API surface.
- **UNSUPPORTED**: would require new code/API changes.

| Case | Standard GPT (SDPA) | Standard GPT (Flex) | Synaptic (Manual) | Synaptic (Flex) | Notes |
|------|----------------------|---------------------|-------------------|-----------------|-------|
| Causal full-seq (`kv_cache=None`) | PASS | PASS | PASS | PASS | Causal-only transformer trunk. |
| KV decode (`Tq==1`) | PASS | PASS | PASS | PASS | Decode uses effectively non-causal access to the prefix. |
| KV chunk decode (`Tk>Tq>1`) | PASS | PASS | PASS | PASS | Critical for streaming decode; verified for standard GPT via suite + for synaptic via `tests/test_demos.py`. |
| MHA (`n_kv_head == n_head`) | PASS | PASS | PASS | PASS | Covered by `--suite`. |
| GQA (`n_kv_head < n_head`) | PASS | PASS | PASS | PASS | Covered by `--suite`. |
| MQA (`n_kv_head == 1`) | PASS | PASS | PASS | PASS | Covered by `--suite`. |
| Padding mask (ragged batches) | N/A | N/A | N/A | N/A | Current nanochat forward API is dense-token only (no padding mask). |
| Bidirectional attention | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | The model is GPT-style causal by design. |
| Attention dropout | N/A | N/A | PARTIAL | PARTIAL | Standard GPT does not use attention dropout; synaptic has `attn_drop` but FlexAttention path does not currently apply equivalent dropout. |

## Dtype / Numerical Notes

- **Parity is tight in float32**, especially for standard GPT.
- In **bf16/fp16**, expect **larger numeric drift** between SDPA kernels and FlexAttention kernels (different kernel families / accumulation).
  - Treat this as expected unless it destabilizes training; prefer verifying **algorithmic parity** in float32.

## Torch Compile Notes

- FlexAttention performance depends on `torch.compile` (otherwise it may fall back to an unfused path and warn).
- KV-cache + `torch.compile` can trigger recompilation warnings due to static integer module attributes (e.g. `layer_idx`); this is mostly a compiler ergonomics issue, not a correctness issue.

