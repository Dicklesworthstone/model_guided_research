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

from dataclasses import replace

from nanochat.gpt import GPT, GPTConfig

# The z4xx symp-tied rung (depth-16, half-width reversible, tied), used as a
# concrete regression anchor so the exact pre/post numbers stay documented.
_Z4XX = GPTConfig(
    n_layer=16,
    n_head=4,
    n_kv_head=2,
    n_embd=128,
    sequence_len=256,
    vocab_size=50304,
)


def _config(
    *,
    attention_type: str | list[str] = "standard",
    reversible_mode: str = "additive",
    reversible_tied: bool = False,
    activation_ckpt: str = "none",
    activation_ckpt_every_k: int = 1,
) -> GPTConfig:
    return replace(
        _Z4XX,
        attention_type=attention_type,
        reversible_mode=reversible_mode,
        reversible_tied=reversible_tied,
        activation_ckpt=activation_ckpt,
        activation_ckpt_every_k=activation_ckpt_every_k,
    )


def _flops(
    *,
    attention_type: str | list[str] = "standard",
    reversible_mode: str = "additive",
    reversible_tied: bool = False,
    activation_ckpt: str = "none",
    activation_ckpt_every_k: int = 1,
) -> int:
    config = _config(
        attention_type=attention_type,
        reversible_mode=reversible_mode,
        reversible_tied=reversible_tied,
        activation_ckpt=activation_ckpt,
        activation_ckpt_every_k=activation_ckpt_every_k,
    )
    return GPT(config).estimate_flops()


def _naive_6n(cfg: GPTConfig) -> int:
    """The pre-fix formula: 6*(N - N_emb) + 12*L*H*Q*T (one fwd + one bwd)."""
    m = GPT(cfg)
    nparams = sum(p.numel() for p in m.parameters())
    nemb = m.transformer.wte.weight.numel()
    l, h, q, t = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.sequence_len
    return 6 * (nparams - nemb) + 12 * l * h * q * t


def test_standard_matches_canonical_6n():
    cfg = _config(attention_type="standard")
    assert GPT(cfg).estimate_flops() == _naive_6n(cfg) == 62_226_432


def test_additive_reversible_charges_wired_recompute():
    # a6k3 wired the memory-saving ReversibleFunction into the additive
    # TRAINING path: backward recomputes activations, so a block pair costs
    # 9 forward-equivalent units per param (vs canonical 6) and its attention
    # matmuls run on every pass (16 H Q T vs 12).
    cfg = _config(attention_type="reversible", reversible_mode="additive")
    m = GPT(cfg)
    nparams = sum(p.numel() for p in m.parameters())
    nemb = m.transformer.wte.weight.numel()
    nblock = sum(p.numel() for p in m.transformer.h.parameters())
    h, q, t = cfg.n_head, cfg.n_embd // cfg.n_head, cfg.sequence_len
    expected = (
        6 * (nparams - nemb - nblock)
        + 9 * nblock
        + 12 * h * q * t * cfg.n_layer
        + 4 * h * q * t * cfg.n_layer
    )
    assert GPT(cfg).estimate_flops() == expected


def test_additive_reversible_strictly_exceeds_canonical_6n():
    # The wired recompute must cost MORE than the naive single-backward rule,
    # mirroring the symplectic guard below.
    cfg = _config(attention_type="reversible", reversible_mode="additive")
    assert GPT(cfg).estimate_flops() > _naive_6n(cfg)


def test_symplectic_counts_the_double_backward():
    # Documented contract: 6*nonblock + 18*block + 3*attn.
    m = GPT(_config(attention_type="reversible", reversible_mode="symplectic", reversible_tied=True))
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
        _config(
            attention_type="standard,reversible",
            reversible_mode="symplectic",
            reversible_tied=False,
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
    cfg = _config(attention_type="reversible", reversible_mode="symplectic", reversible_tied=True)
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


def test_activation_checkpointing_surcharge_accounting():
    """saew: checkpointed blocks re-run their forward inside backward, so the
    estimator must charge +2 FLOPs/param for params in checkpointed blocks and
    +4 H Q T per checkpointed layer. Ordering: none < every-k(2) < full."""
    none = _flops(attention_type="standard", activation_ckpt="none")
    full = _flops(attention_type="standard", activation_ckpt="full")
    ek2 = _flops(attention_type="standard", activation_ckpt="every-k", activation_ckpt_every_k=2)
    assert full > ek2 > none > 0
    # exact surcharge: L=16 layers -> full checkpoints 16; every-2 checkpoints 8
    m = GPT(_config(attention_type="standard"))
    nblock = sum(p.numel() for p in m.transformer.h.parameters())
    per_layer_params = nblock // 16
    h, q, t = 4, 32, 256
    expected_full = none + 2 * 16 * per_layer_params + 4 * h * q * t * 16
    assert full == expected_full
    assert ek2 == none + 2 * 8 * per_layer_params + 4 * h * q * t * 8
    # default is inert
    assert _flops(attention_type="standard") == none
