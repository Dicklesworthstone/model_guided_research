"""FLOPs-per-token accounting for GPT.estimate_flops (bead 7lba).

The canonical 6*N rule assumes exactly one forward + one backward per parameter.
Symplectic-reversible blocks (u55.5) violate that: each exact-gradient kick
takes a first backward to form grad(phi) DURING the model forward
(create_graph=True) and a SECOND backward (autograd-of-autograd) at train time,
so a kick costs ~3x a standard module in forward-equivalent units. Before the
fix the estimator saw only the half-width block params and rated a symplectic
arm CHEAPER than standard, inflating its step count under equal-target-FLOPs
budgeting (the z4xx confound: symp-tied ran 3236 steps vs standard 2355 at the
same 3e14 target). These tests pin the corrected accounting and guard the
standard/additive paths against silent drift.
"""

from nanochat.gpt import GPT, GPTConfig

# The z4xx symp-tied rung (depth-16, half-width reversible, tied), used as a
# concrete regression anchor so the exact pre/post numbers stay documented.
_Z4XX = dict(n_layer=16, n_head=4, n_kv_head=2, n_embd=128, sequence_len=256, vocab_size=50304)


def _flops(**overrides):
    return GPT(GPTConfig(**{**_Z4XX, **overrides})).estimate_flops()


def _naive_6n(cfg: GPTConfig) -> int:
    """The pre-fix formula: 6*(N - N_emb) + 12*L*H*Q*T (one fwd + one bwd)."""
    m = GPT(cfg)
    nparams = sum(p.numel() for p in m.parameters())
    nemb = m.transformer.wte.weight.numel()
    l, h, q, t = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.sequence_len
    return 6 * (nparams - nemb) + 12 * l * h * q * t


def test_standard_matches_canonical_6n():
    cfg = GPTConfig(attention_type="standard", **_Z4XX)
    assert GPT(cfg).estimate_flops() == _naive_6n(cfg) == 62_226_432


def test_additive_reversible_uses_canonical_6n():
    # Additive reversible runs the eager fwd+bwd path (the memory-saving
    # ReversibleFunction recompute is unused), so the correction must NOT apply.
    cfg = GPTConfig(attention_type="reversible", reversible_mode="additive", **_Z4XX)
    assert GPT(cfg).estimate_flops() == _naive_6n(cfg)


def test_symplectic_counts_the_double_backward():
    # Documented contract: 6*nonblock + 18*block + 3*attn.
    m = GPT(GPTConfig(attention_type="reversible", reversible_mode="symplectic", reversible_tied=True, **_Z4XX))
    nparams = sum(p.numel() for p in m.parameters())
    nemb = m.transformer.wte.weight.numel()
    nblock = sum(p.numel() for p in m.transformer.h.parameters())
    nonblock = (nparams - nemb) - nblock
    l, h, q, t = m.config.n_layer, m.config.n_head, m.config.n_embd // m.config.n_head, m.config.sequence_len
    attn = 12 * l * h * q * t
    assert m.estimate_flops() == 6 * nonblock + 18 * nblock + 3 * attn


def test_mixed_symplectic_schedule_counts_only_reversible_layers():
    # Mixed stacks should charge the double-backward correction only to the
    # symplectic reversible layers, while ordinary layers keep canonical 6N.
    m = GPT(
        GPTConfig(
            attention_type="standard,reversible",
            reversible_mode="symplectic",
            reversible_tied=False,
            **_Z4XX,
        )
    )
    nparams = sum(p.numel() for p in m.parameters())
    nemb = m.transformer.wte.weight.numel()
    symplectic_blocks = [
        block for block in m.transformer.h if getattr(block, "attention_type", None) == "reversible"
    ]
    nblock = sum(p.numel() for block in symplectic_blocks for p in block.parameters())
    nonblock = (nparams - nemb) - nblock
    h, q, t = m.config.n_head, m.config.n_embd // m.config.n_head, m.config.sequence_len
    attn_per_layer = 12 * h * q * t
    attn = attn_per_layer * (m.config.n_layer + 2 * len(symplectic_blocks))
    assert 0 < len(symplectic_blocks) < m.config.n_layer
    assert m.estimate_flops() == 6 * nonblock + 18 * nblock + attn


def test_symplectic_anchor_numbers():
    # Pre-fix: 45,269,772 (the artifact). Post-fix: 58,542,372.
    assert _flops(attention_type="reversible", reversible_mode="symplectic", reversible_tied=True) == 58_542_372


def test_symplectic_estimate_strictly_exceeds_buggy_undercount():
    # The correction must strictly INCREASE the per-token cost vs the naive 6N
    # the symplectic arm used to get — that is what removes the step inflation.
    cfg = GPTConfig(attention_type="reversible", reversible_mode="symplectic", reversible_tied=True, **_Z4XX)
    assert GPT(cfg).estimate_flops() > _naive_6n(cfg)


def test_untied_symplectic_is_the_most_expensive_arm():
    # When the block is non-trivial (untied -> 16 distinct blocks) the
    # double-backward dominates and symplectic becomes dearer than standard,
    # exactly the bead's "MOST expensive" claim.
    std = _flops(attention_type="standard")
    symp_untied = _flops(attention_type="reversible", reversible_mode="symplectic", reversible_tied=False)
    assert symp_untied > std


def test_equal_flops_budget_no_longer_inflates_symplectic_steps():
    # At a fixed target, steps = ceil(target / (flops_per_token * tokens_per_step)).
    # Pre-fix the symp/std flops-per-token ratio was 0.728 (=> +37% steps); the
    # fix lifts it to ~0.94, cutting the inflation to single digits.
    import math

    std = _flops(attention_type="standard")
    symp = _flops(attention_type="reversible", reversible_mode="symplectic", reversible_tied=True)
    ratio = symp / std
    assert ratio > 0.90, f"symplectic still badly undercounted: ratio={ratio:.3f}"

    target, tokens_per_step = 3e14, 8 * 256
    std_steps = math.ceil(target / (std * tokens_per_step))
    symp_steps = math.ceil(target / (symp * tokens_per_step))
    step_inflation = symp_steps / std_steps - 1.0
    assert step_inflation < 0.10, f"symplectic step inflation still {step_inflation:.1%} (was ~37%)"
