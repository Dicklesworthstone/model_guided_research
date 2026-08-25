"""Differential JAX<->torch kernel parity suite (bead model_guided_research-5ki.6).

Every framework exists twice in this repo: a JAX demo and a torch production
module. The law-based tests (tests/test_algebraic_properties.py) prove each
side obeys the AXIOMS; this suite proves both sides compute the SAME FUNCTION.
A silent convention divergence (a sign flip, a different blade order, a
transposed table) can satisfy every axiom while invalidating the project's
core bridge claim ("the demo validates the theory, the torch port runs the
benchmark"). Inputs are generated ONCE in NumPy (the neutral source) from
seeded RNGs and pushed through both implementations.

SCOPE / TIERS
  Tier A — direct two-implementation parity (same mathematical function,
           public on both sides):
    quaternion : torch qmul/qconj/qnormalize  <->  demo qmul/qconj/qnormalize
    octonion   : torch omul/oconj             <->  demo octonion_multiply/_conjugate
    tropical   : torch tropical_inner / max,+  <->  demo tmm / tropical_add,_multiply
    gauge      : torch GaugeBlock._apply_rotations <-> demo apply_givens_stage
                 (identical even/odd pairing and rotation-sign conventions)
  Tier B — shared-oracle parity (the two sides expose DIFFERENT APIs over the
           same mathematics; both are anchored to one brute-force NumPy oracle):
    ultrametric: torch _PackedPrefixTrie alpha-weighted read  <-> oracle
                 demo HeadTrie.deepest_occupied               <-> oracle argmax-LCP

DOCUMENTED EXCLUSIONS (no shared kernel exists today — re-scope when one appears):
  - surreal: the torch SurrealLayer decomposes activations into
    direction/scale; the JAX demo does dominance asymptotics on transseries —
    different mathematical objects, nothing to diff.
  - ultrametric soft/kernel mode (exp(-beta diff^2) match probabilities) is a
    torch-only construct with no demo twin.
  - torch _digits_hard_int (real -> base-p digits) has no demo twin; the demo
    consumes integer digits directly.

TOLERANCE TABLE
  exact (==)        : qconj/oconj (sign flips), tropical max/+ ops and
                      tropical_inner vs tmm (max/+ are exact float ops),
                      -inf identity handling, demo deepest_occupied depths
  atol 1e-6 (fp32)  : qmul/omul products, qnormalize, Givens rotations
  atol 1e-12 (fp64) : qmul/omul products in float64 (table agreement to
                      accumulation noise)
  atol 1e-5 (fp32)  : trie alpha-weighted reads vs oracle (fp32 sum order)

MUTATION CHECK (performed during development of this suite, then reverted):
  flipping the sign of the az*bx term in torch qmul's oy component made
  test_qmul_parity fail with max|diff| ~ O(1); flipping the Cayley-Dickson
  'ac - d_conj*b' sign in torch omul made test_omul_parity fail (both dtypes).
  The suite has teeth. (Seed 20260610; recorded in the bead close notes.)
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch


def _x64_if(dtype: type):
    """JAX downcasts float64 to float32 unless x64 is enabled; scope it to the
    test so the global fp32 default (which the demos rely on) is untouched."""
    return jax.enable_x64(True) if dtype == np.float64 else contextlib.nullcontext()


import octonionic_quaternionic_signal_flow as oq_demo
import tropical_geometry_and_idempotent_algebra as trop_demo
import ultrametric_worlds_and_p_adic_computation as ultra_demo
from matrix_exponential_gauge_learning import apply_givens_stage, even_odd_pairs
from nanochat.gauge_block_torch import GaugeBlock
from nanochat.octonion_attention_torch import OctonionCausalSelfAttention, oconj, omul, onormalize
from nanochat.quaternion_attention_torch import qconj, qmul, qnormalize
from nanochat.tropical_attention_torch import tropical_inner
from nanochat.ultrametric_attention_torch import _PackedPrefixTrie

SEED = 20260610
ATOL: dict[type, float] = {np.float32: 1e-6, np.float64: 1e-12}


def _rng(salt: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED + salt)


def _report(name: str, a: np.ndarray, b: np.ndarray) -> str:
    diff = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
    return f"{name}: max|torch - jax| = {diff:.3e} over shape {a.shape}"


# ---------------------------------------------------------------------------
# Tier A — quaternion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_qmul_parity(dtype: type) -> None:
    rng = _rng(1)
    a = rng.normal(size=(64, 4)).astype(dtype)
    b = rng.normal(size=(64, 4)).astype(dtype)
    out_t = qmul(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    with _x64_if(dtype):
        out_j = np.asarray(oq_demo.qmul(jnp.asarray(a), jnp.asarray(b)))
    assert np.allclose(out_t, out_j, atol=ATOL[dtype], rtol=0), _report("qmul", out_t, out_j)


def test_qconj_parity_exact() -> None:
    rng = _rng(2)
    q = rng.normal(size=(32, 4)).astype(np.float32)
    out_t = qconj(torch.from_numpy(q)).numpy()
    out_j = np.asarray(oq_demo.qconj(jnp.asarray(q)))
    assert np.array_equal(out_t, out_j), _report("qconj", out_t, out_j)


def test_qnormalize_parity() -> None:
    rng = _rng(3)
    q = rng.normal(size=(32, 4)).astype(np.float32) * 3.0
    out_t = qnormalize(torch.from_numpy(q)).numpy()
    out_j = np.asarray(oq_demo.qnormalize(jnp.asarray(q)))
    # both guard the zero-norm case with an epsilon; compare on generic inputs
    assert np.allclose(out_t, out_j, atol=1e-6, rtol=0), _report("qnormalize", out_t, out_j)


def test_qmul_batched_shapes_parity() -> None:
    rng = _rng(4)
    a = rng.normal(size=(2, 3, 5, 4)).astype(np.float32)
    b = rng.normal(size=(2, 3, 5, 4)).astype(np.float32)
    out_t = qmul(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    out_j = np.asarray(oq_demo.qmul(jnp.asarray(a), jnp.asarray(b)))
    assert out_t.shape == out_j.shape == (2, 3, 5, 4)
    assert np.allclose(out_t, out_j, atol=1e-6, rtol=0), _report("qmul batched", out_t, out_j)


# ---------------------------------------------------------------------------
# Tier A — octonion (Cayley-Dickson; parenthesization conventions must agree)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_omul_parity(dtype: type) -> None:
    rng = _rng(5)
    x = rng.normal(size=(64, 8)).astype(dtype)
    y = rng.normal(size=(64, 8)).astype(dtype)
    out_t = omul(torch.from_numpy(x), torch.from_numpy(y)).numpy()
    with _x64_if(dtype):
        out_j = np.asarray(oq_demo.octonion_multiply(jnp.asarray(x), jnp.asarray(y)))
    assert np.allclose(out_t, out_j, atol=ATOL[dtype], rtol=0), _report("omul", out_t, out_j)


def test_oconj_parity_exact() -> None:
    rng = _rng(6)
    o = rng.normal(size=(32, 8)).astype(np.float32)
    out_t = oconj(torch.from_numpy(o)).numpy()
    out_j = np.asarray(oq_demo.octonion_conjugate(jnp.asarray(o)))
    # torch: (qconj(a), -b); demo: flip signs of components 1..7 - identical maps
    assert np.array_equal(out_t, out_j), _report("oconj", out_t, out_j)


def test_octonion_quaternion_subalgebra_parity() -> None:
    """Octonions with zero second half must multiply exactly like quaternions
    on BOTH sides (cross-checks blade ordering between the two kernels)."""
    rng = _rng(7)
    a4 = rng.normal(size=(16, 4)).astype(np.float64)
    b4 = rng.normal(size=(16, 4)).astype(np.float64)
    zero = np.zeros_like(a4)
    a8 = np.concatenate([a4, zero], axis=-1)
    b8 = np.concatenate([b4, zero], axis=-1)
    q_t = qmul(torch.from_numpy(a4), torch.from_numpy(b4)).numpy()
    o_t = omul(torch.from_numpy(a8), torch.from_numpy(b8)).numpy()
    with _x64_if(np.float64):
        o_j = np.asarray(oq_demo.octonion_multiply(jnp.asarray(a8), jnp.asarray(b8)))
    assert np.allclose(o_t[:, :4], q_t, atol=1e-12, rtol=0), "torch omul does not restrict to torch qmul"
    assert np.allclose(o_t[:, 4:], 0.0, atol=1e-12, rtol=0), "subalgebra leaked into the epsilon half"
    assert np.allclose(o_t, o_j, atol=1e-12, rtol=0), _report("subalgebra omul", o_t, o_j)


# ---------------------------------------------------------------------------
# Tier A — tropical (max/+ are exact ops: parity must be EXACT)
# ---------------------------------------------------------------------------


def test_tropical_inner_vs_tmm_exact() -> None:
    rng = _rng(8)
    Q = rng.normal(size=(7, 5)).astype(np.float32)
    K = rng.normal(size=(9, 5)).astype(np.float32)
    out_t = tropical_inner(torch.from_numpy(Q), torch.from_numpy(K)).numpy()
    out_j = np.asarray(trop_demo.tmm(jnp.asarray(Q), jnp.asarray(K).T))
    assert out_t.shape == out_j.shape == (7, 9)
    assert np.array_equal(out_t, out_j), _report("tropical_inner vs tmm", out_t, out_j)


def test_tropical_inner_batched_vs_tmm_exact() -> None:
    rng = _rng(9)
    Q = rng.normal(size=(2, 3, 4, 6)).astype(np.float32)
    K = rng.normal(size=(2, 3, 5, 6)).astype(np.float32)
    out_t = tropical_inner(torch.from_numpy(Q), torch.from_numpy(K)).numpy()
    out_j = np.asarray(trop_demo.tmm(jnp.asarray(Q), jnp.asarray(np.swapaxes(K, -1, -2))))
    assert np.array_equal(out_t, out_j), _report("batched tropical_inner vs tmm", out_t, out_j)


def test_tropical_semiring_ops_exact() -> None:
    rng = _rng(10)
    a = rng.normal(size=(64,)).astype(np.float32)
    b = rng.normal(size=(64,)).astype(np.float32)
    add_t = torch.maximum(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    add_j = np.asarray(trop_demo.tropical_add(jnp.asarray(a), jnp.asarray(b)))
    mul_t = (torch.from_numpy(a) + torch.from_numpy(b)).numpy()
    mul_j = np.asarray(trop_demo.tropical_multiply(jnp.asarray(a), jnp.asarray(b)))
    assert np.array_equal(add_t, add_j), _report("tropical_add", add_t, add_j)
    assert np.array_equal(mul_t, mul_j), _report("tropical_multiply", mul_t, mul_j)


def test_tropical_neg_inf_identity_parity() -> None:
    """-inf is the additive identity of max-plus; both sides must treat it
    identically (the masking path depends on this)."""
    a = np.array([1.5, -np.inf, 0.0, -np.inf], dtype=np.float32)
    ninf = np.full_like(a, -np.inf)
    add_t = torch.maximum(torch.from_numpy(a), torch.from_numpy(ninf)).numpy()
    add_j = np.asarray(trop_demo.tropical_add(jnp.asarray(a), jnp.asarray(ninf)))
    assert np.array_equal(add_t, a), "max(x, -inf) must be x on the torch side"
    assert np.array_equal(add_j, a), "max(x, -inf) must be x on the JAX side"
    # the all--inf row: both sides must propagate -inf, never NaN
    both_t = torch.maximum(torch.from_numpy(ninf), torch.from_numpy(ninf)).numpy()
    both_j = np.asarray(trop_demo.tropical_add(jnp.asarray(ninf), jnp.asarray(ninf)))
    assert np.all(np.isneginf(both_t)) and np.all(np.isneginf(both_j))


# ---------------------------------------------------------------------------
# Tier A — gauge Givens rotations (identical pairing + sign conventions)
# ---------------------------------------------------------------------------


def test_givens_rotation_parity() -> None:
    rng = _rng(11)
    B, T, D = 2, 3, 8
    x = rng.normal(size=(B, T, D)).astype(np.float32)
    thetas = (rng.normal(size=(B, T, D // 2)) * np.pi).astype(np.float32)
    out_t = GaugeBlock._apply_rotations(torch.from_numpy(x), torch.from_numpy(thetas)).numpy()
    pairs = even_odd_pairs(D)
    out_j = np.asarray(apply_givens_stage(jnp.asarray(x), jnp.asarray(thetas), pairs))
    assert np.allclose(out_t, out_j, atol=1e-6, rtol=0), _report("givens forward", out_t, out_j)


def test_givens_rotation_inverse_parity() -> None:
    rng = _rng(12)
    B, T, D = 1, 4, 6
    x = rng.normal(size=(B, T, D)).astype(np.float32)
    thetas = (rng.normal(size=(B, T, D // 2)) * np.pi).astype(np.float32)
    out_t = GaugeBlock._apply_rotations(torch.from_numpy(x), torch.from_numpy(thetas), inverse=True).numpy()
    pairs = even_odd_pairs(D)
    out_j = np.asarray(apply_givens_stage(jnp.asarray(x), jnp.asarray(-thetas), pairs))
    assert np.allclose(out_t, out_j, atol=1e-6, rtol=0), _report("givens inverse", out_t, out_j)


# ---------------------------------------------------------------------------
# Tier B — ultrametric (shared brute-force oracle anchors both sides)
# ---------------------------------------------------------------------------


def _lcp_len(a: np.ndarray, b: np.ndarray) -> int:
    n = 0
    for x, y in zip(a, b, strict=True):
        if x != y:
            break
        n += 1
    return n


def test_torch_packed_trie_matches_bruteforce_oracle() -> None:
    """The packed trie's alpha-weighted read must equal the brute-force
    LCP-weighted average sum_i alpha^LCP(q, k_i) v_i / sum_i alpha^LCP(q, k_i)."""
    rng = _rng(13)
    p, K, head_dim, n_keys, n_queries, alpha = 3, 5, 4, 24, 8, 2.0
    keys = rng.integers(0, p, size=(n_keys, K))
    vals = rng.normal(size=(n_keys, head_dim)).astype(np.float32)
    trie = _PackedPrefixTrie(p=p, K=K, head_dim=head_dim, device=torch.device("cpu"))
    for i in range(n_keys):
        trie.insert(torch.from_numpy(keys[i]).to(torch.int64), torch.from_numpy(vals[i]))
    queries = rng.integers(0, p, size=(n_queries, K))
    for qi in range(n_queries):
        got = trie.query(torch.from_numpy(queries[qi]).to(torch.int64), alpha=alpha).numpy()
        w = np.array([alpha ** _lcp_len(queries[qi], keys[i]) for i in range(n_keys)], dtype=np.float64)
        want = (w[:, None] * vals.astype(np.float64)).sum(axis=0) / w.sum()
        assert np.allclose(got, want, atol=1e-5, rtol=0), (
            f"query {qi}: trie read disagrees with brute-force oracle; max|diff|={np.max(np.abs(got - want)):.3e}"
        )


def test_demo_headtrie_deepest_occupied_matches_oracle() -> None:
    """The demo trie's deepest_occupied must report exactly the brute-force
    max-LCP depth (0-indexed deepest occupied level = max LCP length - 1)."""
    rng = _rng(14)
    p, K, m = 3, 5, 2
    trie = ultra_demo.HeadTrie(p=p, K=K, m=m, r=1)
    keys = rng.integers(0, p, size=(12, K))
    for key in keys:
        for d, res in enumerate(trie.path_residues([int(x) for x in key])):
            trie.ensure(d, res)
    queries = rng.integers(0, p, size=(8, K))
    for q in queries:
        d_got, _ = trie.deepest_occupied([int(x) for x in q])
        best_lcp = max(_lcp_len(q, k) for k in keys)
        assert d_got == best_lcp - 1, (
            f"demo deepest_occupied={d_got} but oracle max-LCP depth={best_lcp - 1} (query={q.tolist()})"
        )


def test_both_tries_agree_on_lcp_structure() -> None:
    """Cross-implementation agreement at the level both sides share: with one
    key inserted, the torch trie's read weight is alpha^LCP (recoverable from
    the read being exactly the key's value regardless of LCP), and the demo
    trie's deepest depth equals LCP-1; both must agree with the SAME oracle
    LCP for the same (key, query) pairs."""
    rng = _rng(15)
    p, K, head_dim, alpha = 2, 6, 3, 2.0
    key = rng.integers(0, p, size=(K,))
    val = rng.normal(size=(head_dim,)).astype(np.float32)
    t_trie = _PackedPrefixTrie(p=p, K=K, head_dim=head_dim, device=torch.device("cpu"))
    t_trie.insert(torch.from_numpy(key).to(torch.int64), torch.from_numpy(val))
    j_trie = ultra_demo.HeadTrie(p=p, K=K, m=2, r=1)
    for d, res in enumerate(j_trie.path_residues([int(x) for x in key])):
        j_trie.ensure(d, res)
    for _ in range(6):
        q = rng.integers(0, p, size=(K,))
        oracle = _lcp_len(q, key)
        d_demo, _ = j_trie.deepest_occupied([int(x) for x in q])
        assert d_demo == oracle - 1, f"demo LCP {d_demo + 1} != oracle {oracle}"
        got = t_trie.query(torch.from_numpy(q).to(torch.int64), alpha=alpha).numpy()
        assert np.allclose(got, val, atol=1e-6), (
            "single-key trie read must return the key's value exactly "
            f"(weights cancel); max|diff|={np.max(np.abs(got - val)):.3e}"
        )


# ---------------------------------------------------------------------------
# Torch-internal parity — vectorized vs per-query reference (bead 7b0.6)
# ---------------------------------------------------------------------------


def _octonion_aggregate_per_query_reference(weights, v, q, k, *, n_head, head_dim):
    """Pre-7b0.6 OctonionCausalSelfAttention.aggregate, kept verbatim as the
    numerical oracle: one Python iteration per query, explicit
    ((Q_i * conj(K_j)) * V_j) parenthesization."""
    B = q.size(0)
    Tq = q.size(2)
    N = head_dim // 8
    q_o = onormalize(q.reshape(B, n_head, Tq, N, 8))
    k_o = onormalize(k.reshape(B, n_head, -1, N, 8))
    v_o = v.reshape(B, n_head, -1, N, 8)
    k_conj = oconj(k_o)
    y_list = []
    for i in range(Tq):
        q_i = q_o[:, :, i : i + 1]
        r_i = omul(q_i, k_conj)
        term = omul(r_i, v_o)
        p_i = weights[:, :, i].unsqueeze(-1).unsqueeze(-1)
        y_list.append((term * p_i).sum(dim=2).unsqueeze(2))
    return torch.cat(y_list, dim=2).reshape(B, n_head, Tq, head_dim)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tile_budget_mb", [None, "0.003", "3"])
@pytest.mark.parametrize("n_kv_head", [4, 2])  # 4 = GQA off, 2 = GQA on
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("tq", [1, 7, 256])
def test_octonion_tiled_aggregate_matches_per_query_reference(
    batch, tq, n_kv_head, tile_budget_mb, dtype, monkeypatch
) -> None:
    """The chunked-vectorized aggregate must reproduce the historical
    per-query loop within fp reduction noise across the bead's shape grid:
    Tq in {1, 7, 256}, GQA on/off, B in {1, 4}; forced tiny tile budgets
    exercise ragged multi-tile boundaries (c=1 and c=3 against Tq=7/256)."""
    from nanochat.gpt import GPTConfig
    from nanochat.model_utils import causal_attn_mask

    if tile_budget_mb is None:
        monkeypatch.delenv("NANOCHAT_OCTONION_TILE_BUDGET_MB", raising=False)
    else:
        monkeypatch.setenv("NANOCHAT_OCTONION_TILE_BUDGET_MB", tile_budget_mb)

    torch.manual_seed(SEED + batch * 1000 + tq * 10 + n_kv_head + len(tile_budget_mb or ""))

    config = GPTConfig(
        n_layer=1,
        n_head=4,
        n_kv_head=n_kv_head,
        n_embd=64,
        sequence_len=max(tq, 2),
        vocab_size=64,
        attention_type="octonion",
    )
    attn = OctonionCausalSelfAttention(config, layer_idx=0).to(dtype).eval()
    n_head, head_dim = attn.n_head, attn.head_dim
    q = torch.randn(batch, n_head, tq, head_dim, dtype=dtype)
    k = torch.randn(batch, n_head, tq, head_dim, dtype=dtype)
    v = torch.randn(batch, n_head, tq, head_dim, dtype=dtype)

    # Reproduce the scaffold's score -> causal-mask -> softmax pipeline so
    # BOTH paths consume identical weights.
    scores = attn.score(q, k)
    scores = scores.masked_fill(~causal_attn_mask(tq, tq, device=q.device), float("-inf"))
    weights = torch.softmax(scores, dim=-1)

    got = attn.aggregate(weights, v, q=q, k=k, kv_cache=None, pos0=None)
    want = _octonion_aggregate_per_query_reference(weights, v, q, k, n_head=n_head, head_dim=head_dim)
    assert got.shape == want.shape == (batch, n_head, tq, head_dim)
    assert torch.isfinite(got).all(), "vectorized aggregate produced non-finite output"
    atol = 1e-6 if dtype == torch.float32 else 5e-12
    max_diff = float((got - want).abs().max())
    assert max_diff <= atol, (
        f"tiled aggregate diverged from per-query reference "
        f"(B={batch}, Tq={tq}, n_kv_head={n_kv_head}, budget={tile_budget_mb}, {dtype}): "
        f"max|diff|={max_diff:.3e} > {atol}"
    )
