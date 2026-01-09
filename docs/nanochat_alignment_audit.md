# Nanochat Alignment Audit (Docs ↔ Implementations)

This report cross-checks the **nanochat** implementations against the corresponding theoretical writeups in `markdown_documentation/` (and, where helpful, the root JAX demos). It focuses on **semantic alignment** and **KV-cache correctness**.

Last updated: **2025-12-18**

## What Was Fixed In This Pass (Bead: `model_guided_research-3bx`)

### Tropical attention (PyTorch + JAX): remove softmax; implement full max-plus aggregation

- **PyTorch** `nanochat/tropical_attention_torch.py:44` now implements:
  - score: `max_d(q_d + k_d)` and
  - aggregation: `y_d = max_k(score(q,k) + v_{k,d})` (tropical matmul).
- **JAX** `nanochat/tropical_attention.py:125` mirrors the same semantics and now correctly handles **boolean causal masks** vs **additive (0 / -inf) cache masks**.

Docs reference:
- `markdown_documentation/tropical_geometry_and_idempotent_algebra.md` (tropical attention: hard routing / max-plus)

### Ultrametric attention (PyTorch + JAX): remove dot-product/softmax placeholder; implement LCP-kernel attention

- **PyTorch** `nanochat/ultrametric_attention_torch.py:51` now implements an **LCP-kernel** attention:
  - learn K digit projections per head,
  - compute a differentiable proxy for `LCP(q,k)` via prefix products,
  - use `w(q,k) ∝ alpha^{LCP(q,k)}` to produce a **causal weighted average** (no dot-product, no softmax).
- **JAX** `nanochat/ultrametric_attention.py:93` was previously internally inconsistent (dim-mismatch leftovers + dot-product/softmax). It now:
  - handles KV cache like `nanochat/tropical_attention.py`,
  - supports boolean/additive masks,
  - implements the same LCP-kernel attention in a clean, shape-correct way.

Docs reference:
- `markdown_documentation/ultrametric_worlds_and_p_adic_computation.md` (LCP as geometry; similarity kernels and LCP-based routing)

### Simplicial attention (PyTorch): KV-cache parity for 2-hop diffusion

Previously, simplicial attention used a **2-hop path only when `Tq == Tk`**, and fell back to `y2 = y1` during KV-cache decode/chunked decode. This was a real inference/training semantic mismatch.

- **KVCache** now supports an optional per-layer cache of 1-hop outputs:
  - `nanochat/engine.py:109` (state)
  - `nanochat/engine.py:250` (init/insert)
- **Simplicial attention** now uses the cache to compute 2-hop consistently:
  - `nanochat/simplicial_attention_torch.py:41` (captures write position)
  - `nanochat/simplicial_attention_torch.py:64` (cache-backed `y2 = A @ y1_all`)
- **Tests**: `tests/test_demos.py:76` now includes `simplicial` in the “KV-cache last-token matches full forward” test.

Docs reference:
- `markdown_documentation/simplicial_complexes_and_higher_order_attention.md`

### Quaternion + octonion attention (PyTorch): normalize query “rotors”

Both modules already normalized keys but not queries; that makes the “rotor” multiplication **not norm-preserving** on the query side.

- Quaternion: `nanochat/quaternion_attention_torch.py:96`
- Octonion: `nanochat/octonion_attention_torch.py:101`

Docs reference:
- `markdown_documentation/octonionic_quaternionic_signal_flow.md` (unit rotors and norm control)

## Remaining Gaps / Follow-ups

These are places where nanochat still doesn’t fully realize what the docs describe; they should become beads (or map to existing ones).

1) **Ultrametric sublinear retrieval / packed trie layout**
   - Current nanochat ultrametric attention is **O(T²)** (kernel form), not trie-based sublinear lookup.
   - Related bead: `model_guided_research-a1o` (packed trie layout).

2) **Tropical gauge-fixing + certificates**
   - Tropical docs emphasize gauge fixing (centering/row anchoring) and margin-style certificates.
   - Current nanochat tropical attention implements max-plus similarity + aggregation, but does not emit or track cert/margins.

3) **Braid attention: discrete decoder + (optional) YBE-valid crossing law**
   - `nanochat/braid_attention_torch.py` remains a “soft braid” approximation rather than a verified braid-word decoder.
   - Related bead: `model_guided_research-k2y` (YBE-satisfying crossing option).

4) **Synaptic alignment**
   - `nanochat/synaptic.py` contains an explicitly “simplified consolidation” path with shape-dependent fallbacks (e.g., `nanochat/synaptic.py:432`).
   - Needs a focused pass to reconcile the intended math/spec with the implementation and remove silent no-op updates.

## Quick Per-Module Checklist (Nanochat)

Legend: ✅ aligned, ⚠️ partially aligned / missing key aspects, ❌ broken/mismatched.

- Tropical attention
  - ✅ `nanochat/tropical_attention_torch.py` (full max-plus; KV-cache correct)
  - ✅ `nanochat/tropical_attention.py` (full max-plus; mask handling fixed)
- Ultrametric attention
  - ✅ `nanochat/ultrametric_attention_torch.py` (LCP-kernel; KV-cache correct)
  - ✅ `nanochat/ultrametric_attention.py` (LCP-kernel; KV-cache + mask handling fixed)
  - ⚠️ Missing trie-based scaling/caches (see `model_guided_research-a1o`)
- Simplicial attention
  - ✅ `nanochat/simplicial_attention_torch.py` (KV-cache consistent 2-hop)
- Quaternion / Octonion
  - ✅ Rotor normalization added for queries (`nanochat/quaternion_attention_torch.py`, `nanochat/octonion_attention_torch.py`)
  - ⚠️ Still uses scalar dot-product logits + softmax (acceptable if we keep “weights decide how much, rotors decide how” as the scope)
- Braid
  - ⚠️ Soft approximation only (`nanochat/braid_attention_torch.py`)
- Ordinal scheduler
  - ✅ Basic implementation exists (`nanochat/ordinal_scheduler.py`)
- Synaptic
  - ⚠️ Needs doc/spec reconciliation (`nanochat/synaptic.py`)

