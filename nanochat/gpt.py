"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
"""

import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.adamw import DistAdamW
from nanochat.braid_attention_torch import BraidCausalSelfAttention
from nanochat.clifford_attention_torch import CliffordCausalSelfAttention
from nanochat.common import get_dist_info
from nanochat.fractal_attention_torch import FractalCausalSelfAttention
from nanochat.gauge_block_torch import GaugeBlock
from nanochat.hoss_opt_torch import HOSS
from nanochat.model_utils import AttentionCore, causal_attn_mask, norm, sdpa_causal_attend
from nanochat.muon import DistMuon, Muon
from nanochat.octonion_attention_torch import OctonionCausalSelfAttention
from nanochat.quaternion_attention_torch import QuaternionCausalSelfAttention
from nanochat.reversible_block_torch import ReversibleBlock
from nanochat.simplicial_attention_torch import SimplicialCausalSelfAttention
from nanochat.surreal_torch import SurrealCausalSelfAttention
from nanochat.tropical_attention_torch import TropicalCausalSelfAttention, TropicalMLP
from nanochat.ultrametric_attention_torch import UltrametricCausalSelfAttention

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except Exception:  # pragma: no cover - depends on torch build
    create_block_mask = None
    flex_attention = None
    _HAS_FLEX = False
else:
    _HAS_FLEX = True

_COMPILED_FLEX_ATTENTION: dict[tuple[str, str | None, bool, bool | None], Callable[..., Any]] = {}

_CA_RULES: dict[str, int] = {
    "rule30": 30,
    "rule116": 116,
}

SUPPORTED_ATTENTION_TYPES: tuple[str, ...] = (
    "standard",
    "clifford",
    "tropical",
    "ultrametric",
    "simplicial",
    "quaternion",
    "braid",
    "fractal",
    "octonion",
    "surreal",
    "reversible",
    "gauge",
)


def _split_attention_spec(spec: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(spec, str):
        parts = [part.strip() for part in spec.split(",")]
    elif isinstance(spec, list | tuple):
        parts = []
        for item in spec:
            if not isinstance(item, str):
                raise TypeError(f"attention schedule entries must be strings, got {type(item).__name__}")
            parts.extend(part.strip() for part in item.split(","))
    else:
        raise TypeError(f"attention_type must be a string or list of strings, got {type(spec).__name__}")
    return parts


def resolve_attention_schedule(config: "GPTConfig") -> list[str]:
    """Expand GPTConfig.attention_type into one mechanism name per layer.

    A single name keeps the historical homogeneous model. A comma-separated
    pattern or list repeats across the depth only when it divides n_layer
    exactly, so every layer's mechanism is auditable from the short spec.
    """

    n_layer = int(getattr(config, "n_layer", 0))
    if n_layer <= 0:
        raise ValueError("n_layer must be positive before resolving the attention schedule")
    pattern = _split_attention_spec(getattr(config, "attention_type", "standard"))
    if not pattern:
        raise ValueError("attention schedule must name at least one mechanism")
    if not any(pattern):
        raise ValueError("attention schedule must name at least one mechanism")
    for idx, name in enumerate(pattern):
        if not name:
            raise ValueError(f"attention schedule entry {idx} is empty; remove the extra comma or name a mechanism")
        if name not in SUPPORTED_ATTENTION_TYPES:
            raise ValueError(
                f"attention schedule entry {idx} has unknown mechanism {name!r}; "
                f"expected one of: {', '.join(SUPPORTED_ATTENTION_TYPES)}"
            )
    if len(pattern) > n_layer:
        raise ValueError(
            f"attention schedule length {len(pattern)} exceeds n_layer={n_layer}; "
            "provide exactly n_layer entries or a shorter repeating pattern"
        )
    if len(pattern) != n_layer and n_layer % len(pattern) != 0:
        raise ValueError(
            f"attention schedule length {len(pattern)} does not evenly divide n_layer={n_layer}; "
            "provide exactly n_layer entries or a shorter repeating pattern"
        )
    return [pattern[i % len(pattern)] for i in range(n_layer)]


def attention_schedule_contains(config: "GPTConfig", mechanism: str) -> bool:
    return mechanism in resolve_attention_schedule(config)


def attention_schedule_label(config: "GPTConfig") -> str:
    schedule = resolve_attention_schedule(config)
    return schedule[0] if all(name == schedule[0] for name in schedule) else ",".join(schedule)


def _ca_bitfield(*, rule: int, length: int, generator: torch.Generator) -> torch.Tensor:
    """
    Generate a 1D {0,1} bitfield using a 1D, radius-1 cellular automaton.

    Determinism: controlled entirely by `generator` (used for the initial state).
    """
    if length <= 0:
        raise ValueError("CA bitfield length must be positive")
    if not (0 <= rule <= 255):
        raise ValueError("CA rule must be in [0, 255]")

    # Trade off interesting structure vs loop count: cap width so big tensors don't
    # require thousands of CA steps.
    width = int(math.ceil(math.sqrt(length)))
    width = max(8, min(4096, width))
    steps = int(math.ceil(length / width))

    # Initial state: random bits (deterministic via generator).
    state = torch.randint(0, 2, (width,), dtype=torch.int64, generator=generator)

    # LUT for neighborhoods encoded as (L<<2 | C<<1 | R), with Wolfram numbering:
    # output_bit = (rule >> neighborhood) & 1.
    lut = torch.tensor([(rule >> i) & 1 for i in range(8)], dtype=torch.int64)

    out = torch.empty((steps * width,), dtype=torch.int64)
    write = 0
    for _ in range(steps):
        out[write : write + width] = state
        write += width

        left = torch.zeros_like(state)
        right = torch.zeros_like(state)
        left[1:] = state[:-1]
        right[:-1] = state[1:]
        neighborhood = (left << 2) | (state << 1) | right
        state = lut[neighborhood]

    return out[:length]


def _ca_values_for_weight(
    *,
    rule: int,
    shape: tuple[int, ...],
    target_std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Return CA-derived values shaped like `shape`, scaled to mean≈0 and std≈target_std.

    Note: generates on CPU in float32; caller is responsible for casting/moving.
    """
    if target_std <= 0.0 or not math.isfinite(target_std):
        raise ValueError(f"target_std must be finite and positive, got {target_std}")
    numel = int(math.prod(shape))
    bits = _ca_bitfield(rule=rule, length=numel, generator=generator)
    vals = bits.to(torch.float32).mul(2.0).sub(1.0)
    mean = float(vals.mean().item())
    std = float(vals.std(unbiased=False).item())
    if std <= 0.0 or not math.isfinite(std):
        raise RuntimeError("CA initializer produced degenerate variance; cannot rescale.")
    vals = vals.sub(mean).div(std).mul(float(target_std))
    return vals.reshape(shape)


def _get_compiled_flex_attention(
    *,
    backend: str,
    mode: str | None,
    fullgraph: bool,
    dynamic: bool | None,
) -> Callable[..., Any] | None:
    if not hasattr(torch, "compile") or flex_attention is None:
        return flex_attention
    key = (backend, mode, fullgraph, dynamic)
    fn = _COMPILED_FLEX_ATTENTION.get(key)
    if fn is None:
        fn = torch.compile(
            flex_attention,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        _COMPILED_FLEX_ATTENTION[key] = fn
    return fn


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    attention_type: str | list[str] = "standard"
    use_flex_attention: bool = False
    compile_flex_attention: bool = False
    compile_backend: str = "inductor"
    compile_mode: str | None = None
    compile_fullgraph: bool = False
    compile_dynamic: bool | None = None
    # Standard attention diagnostics.
    standard_record_attn_entropy: bool = False
    optimizer_type: str = "adamw"
    ca_init_rule: str | None = None  # "rule30" | "rule116"
    ca_init_alpha: float = 1.0  # alpha*CA + (1-alpha)*standard
    ca_init_seed: int | None = None  # defaults to train --seed when unset
    # Tropical-specific diagnostics/stabilization.
    tropical_gauge_fix: bool = True
    tropical_score_center: bool = True
    tropical_record_margins: bool = False
    # Maslov smoothing for tropical ATTENTION (8gk.1): None = exact tropical
    # endpoint (bitwise-identical to the pre-8gk.1 path); finite beta>0
    # selects the (+)_beta semiring family. Annealing schedules update live
    # modules per step via set_semiring_beta; this is the initial value.
    semiring_beta: float | None = None
    # FFN structure (bead 8gk.8): the semiring design axis extended past attention.
    # "standard" = ReLU^2 MLP; "tropical" = pure max-plus stack (1-Lipschitz,
    # closes the certified chain's MLP hole); "tropical-rational" = difference
    # of two pure stacks (all piecewise-linear maps, 2-Lipschitz declared).
    ffn_type: str = "standard"
    # Maslov smoothing for the tropical FFN: None = exact max (tropical
    # endpoint); finite beta>0 = (+)_beta semiring (network-wide annealing
    # alongside 8gk.1's attention schedule when that lands).
    ffn_beta: float | None = None
    # Width-scaling parameterization arm (beads lab.1/bp08): "current" keeps
    # today's rules; "nsa" applies the nonstandard-analysis per-stage
    # corrections from WIDTH_SCALING_TABLE (nanochat/parameterization.py):
    # tropical gets exact finite-N E[max] location constants (MLP bias init +
    # attention first-stage score shift); quaternion/octonion rotor-LR split
    # is approximated by the existing Muon/AdamW grouping (derivation in the
    # table notes). Default "current" until E1 adopts nsa deliberately.
    parameterization: str = "current"
    # Ultrametric-specific options (see nanochat.ultrametric_attention_torch).
    ultrametric_mode: str = "kernel"  # "kernel" | "trie" | "balltree" (exact O(K T log T), bead 33dd)
    # Eval-time digit-precision truncation (bead 8gk.4): use only the first k
    # of the K digit channels - quantization as valuation truncation, with the
    # flat-error lemma governing the p-adic-native path. None = full precision.
    ultrametric_digits_k: int | None = None
    # Eval-time float weight quantization (bead tcuy): symmetric uniform
    # per-tensor fake-quantization of Linear/Embedding weights to b bits,
    # applied once at checkpoint load by the eval loader (the model itself
    # never reads this field). The archimedean baseline the digit-truncation
    # curves are preregistered against. None = full precision.
    eval_weight_quant_bits: int | None = None
    ultrametric_hard_digits: bool = False
    ultrametric_K: int = 8
    ultrametric_p: int = 2
    ultrametric_alpha: float = 2.0
    ultrametric_lcp_beta: float = 32.0
    # Standard-attention falsification arm for the symplectic no-norm program
    # (bead z4xx): strip the per-layer norms from the standard block - the
    # expected-failure control of hyp-symplectic-nonorm-depth-tied/-untied.
    # Standard attention only (validated); trunk-boundary norms retained.
    disable_block_norms: bool = False
    # Reversible-specific options (see nanochat.reversible_block_torch; u55.5).
    # symplectic = kick-kick coupling with exact-gradient kicks (corrected
    # sign: conserves the coercive H = phi_G + phi_F; theory note
    # markdown_documentation/symplectic_transformer.md).
    reversible_mode: str = "additive"  # "additive" | "symplectic"
    # Layer-tied potentials: ONE block shared across all layers (the regime
    # where shadow conservation is theory-backed). Generation/kv-cache is
    # unsupported when tied (per-layer cache slots vs shared layer_idx).
    reversible_tied: bool = False
    # Coercivity floor for the learnable confinement lambda; 0.0 is the
    # registered falsification control (unconfined potentials).
    reversible_lambda_min: float = 0.05
    reversible_record_energy: bool = False  # shadow-energy/norm telemetry
    # Braid-specific options (see nanochat.braid_attention_torch).
    braid_mode: str = "soft"  # "soft" | "discrete"
    braid_tau: float = 0.0
    braid_crossing_law: str = "restricted"  # "restricted" | "ybe" | "rmatrix" (integrable, u55.3)
    braid_record_schedule: bool = False
    braid_verify: bool = False
    braid_rmatrix_probes: int = 0  # rmatrix only: spectral multi-view probe sweeps (0i1v; 0 = off)


class CausalSelfAttention(AttentionCore):
    # GQA is handled inside SDPA/flex via enable_gqa; no materialized repeat.
    gqa_via_repeat = False

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.record_attn_entropy = bool(getattr(config, "standard_record_attn_entropy", False))
        self.use_flex_attention = bool(getattr(config, "use_flex_attention", False))
        self.compile_flex_attention = bool(getattr(config, "compile_flex_attention", False))
        self.compile_backend = str(getattr(config, "compile_backend", "inductor"))
        self.compile_mode = getattr(config, "compile_mode", None)
        self.compile_fullgraph = bool(getattr(config, "compile_fullgraph", False))
        self.compile_dynamic = getattr(config, "compile_dynamic", None)
        if self.use_flex_attention and not _HAS_FLEX:
            raise ImportError(
                "GPTConfig.use_flex_attention=True but FlexAttention is unavailable "
                "(requires torch>=2.5 and torch.nn.attention.flex_attention)."
            )
        self.register_buffer(
            "attn_entropy_head_mean",
            torch.full((self.n_head,), float("nan"), dtype=torch.float32),
            persistent=False,
        )
        self._flex_attention = flex_attention
        if self.use_flex_attention and self.compile_flex_attention:
            self._flex_attention = _get_compiled_flex_attention(
                backend=self.compile_backend,
                mode=self.compile_mode,
                fullgraph=self.compile_fullgraph,
                dynamic=self.compile_dynamic,
            )

    def attend(self, q, k, v, *, kv_cache, pos0):
        Tq = q.size(2)  # number of queries in this forward pass
        Tk = k.size(2)  # number of keys/values in total (in the cache + current forward pass)
        enable_gqa = (
            self.n_head != self.n_kv_head
        )  # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        if self.record_attn_entropy:
            with torch.no_grad():
                k_entropy = k
                if enable_gqa:
                    repeat = self.n_head // self.n_kv_head
                    k_entropy = k.repeat_interleave(repeat, dim=1)
                mask = causal_attn_mask(Tq, Tk, device=q.device)
                scale = 1.0 / math.sqrt(float(self.head_dim))
                scores = torch.matmul(q.detach().float(), k_entropy.detach().float().transpose(-2, -1)).mul(scale)
                scores = scores.masked_fill(~mask, float("-inf"))
                p = torch.softmax(scores, dim=-1)
                safe_scores = scores.masked_fill(~mask, 0.0)
                exp_score = (p * safe_scores).sum(dim=-1)
                log_z = torch.logsumexp(scores, dim=-1)
                entropy = log_z - exp_score
                self.attn_entropy_head_mean.copy_(entropy.mean(dim=(0, 2)))
        if self.use_flex_attention:
            if not _HAS_FLEX or create_block_mask is None or flex_attention is None:
                raise RuntimeError("FlexAttention requested but unavailable at runtime.")

            B = q.size(0)
            prefix_len = Tk - Tq

            def causal_mask(b, h, q_idx, kv_idx):
                return kv_idx <= (prefix_len + q_idx)

            block_mask = create_block_mask(causal_mask, B, self.n_head, Tq, Tk, device=q.device)
            return self._flex_attention(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)
        return sdpa_causal_attend(q, k, v, kv_cache=kv_cache, enable_gqa=enable_gqa)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


def _build_ffn(config) -> nn.Module:
    """FFN dispatch (bead 8gk.8): standard ReLU^2 MLP, or the max-plus
    TropicalMLP (pure 1-Lipschitz / rational 2-Lipschitz, optional Maslov
    beta-smoothing). Validation of ffn_type happens in GPT._validate_config."""
    if getattr(config, "ffn_type", "standard") in ("tropical", "tropical-rational"):
        return TropicalMLP(config)
    return MLP(config)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = resolve_attention_schedule(config)[layer_idx]
        # Special Block Types that replace the standard Attention+MLP structure
        if self.attention_type == "gauge":
            # The gauge block owns its residual skeleton and MLP slot (1fr6);
            # the MLP is built here so ffn_type dispatch applies to gauge too.
            self.special_block = GaugeBlock(config, layer_idx, _build_ffn(config))
            return
        if self.attention_type == "reversible":
            # Reversible blocks split channels in half: x = [x1, x2].
            # We keep the RoPE head_dim constant by halving the number of query heads.
            # IMPORTANT: KV cache is allocated from the top-level config, so we keep n_kv_head unchanged.
            sub_config = GPTConfig(**config.__dict__)
            sub_config.attention_type = "standard"
            sub_config.n_embd = config.n_embd // 2
            sub_config.n_head = config.n_head // 2
            sub_config.n_kv_head = config.n_kv_head

            self.special_block = ReversibleBlock(
                config,
                layer_idx,
                CausalSelfAttention(sub_config, layer_idx),
                # tropical FFN inside additive coupling is allowed: ANY G
                # preserves invertibility (bead 8gk.8 interaction rule b)
                _build_ffn(sub_config),
            )
            return

        if self.attention_type == "tropical":
            self.attn = TropicalCausalSelfAttention(config, layer_idx)
        elif self.attention_type == "ultrametric":
            self.attn = UltrametricCausalSelfAttention(config, layer_idx)
        elif self.attention_type == "simplicial":
            self.attn = SimplicialCausalSelfAttention(config, layer_idx)
        elif self.attention_type == "quaternion":
            self.attn = QuaternionCausalSelfAttention(config, layer_idx)
        elif self.attention_type == "braid":
            self.attn = BraidCausalSelfAttention(config, layer_idx)
        elif self.attention_type == "fractal":
            self.attn = FractalCausalSelfAttention(config, layer_idx)
        elif self.attention_type == "clifford":
            self.attn = CliffordCausalSelfAttention(config, layer_idx)
        elif self.attention_type == "octonion":
            self.attn = OctonionCausalSelfAttention(config, layer_idx)
        elif self.attention_type == "surreal":
            self.attn = SurrealCausalSelfAttention(config, layer_idx)
        else:
            self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = _build_ffn(config)

    def forward(self, x, cos_sin, kv_cache):
        if self.attention_type == "gauge":
            return self.special_block(x, cos_sin, kv_cache)
        if self.attention_type == "reversible":
            return self.special_block(x, cos_sin, kv_cache)

        if getattr(self.config, "disable_block_norms", False):
            # the no-norm falsification arm (z4xx): identical block, norms
            # stripped - drift is expected; that is the point of the control
            x = x + self.attn(x, cos_sin, kv_cache)
            x = x + self.mlp(x)
            return x
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._validate_config()
        # Construction order is RNG draw order (golden-trajectory contract):
        # wte FIRST, then blocks - exactly the original dict-literal order.
        wte = nn.Embedding(config.vocab_size, config.n_embd)
        if getattr(config, "reversible_tied", False):
            # Layer-tied symplectic potentials (u55.5): ONE block iterated
            # n_layer times - the regime where shadow conservation is the
            # theorem (the same symplectic map composed with itself).
            # Parameters dedupe by identity in .parameters()/named_parameters().
            tied_block = Block(config, 0)
            blocks = nn.ModuleList([tied_block for _ in range(config.n_layer)])
        else:
            blocks = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)])
        self.transformer = nn.ModuleDict(
            {
                "wte": wte,
                "h": blocks,
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10  # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)  # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def _validate_config(self) -> None:
        if self.config.sequence_len <= 0:
            raise ValueError("sequence_len must be positive")
        if self.config.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.config.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.config.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.config.n_kv_head <= 0:
            raise ValueError("n_kv_head must be positive")
        if self.config.n_embd <= 0:
            raise ValueError("n_embd must be positive")

        if self.config.n_embd % self.config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not (self.config.n_kv_head <= self.config.n_head and self.config.n_head % self.config.n_kv_head == 0):
            raise ValueError("n_kv_head must divide n_head and be <= n_head")

        head_dim = self.config.n_embd // self.config.n_head
        if head_dim % 2 != 0:
            raise ValueError("head_dim (= n_embd // n_head) must be even for RoPE")

        attention_schedule = resolve_attention_schedule(self.config)
        has_reversible = "reversible" in attention_schedule
        all_reversible = all(name == "reversible" for name in attention_schedule)

        for layer_idx, attention_type in enumerate(attention_schedule):
            if attention_type == "quaternion" and head_dim % 4 != 0:
                raise ValueError(
                    f"attention schedule layer {layer_idx} uses quaternion, which requires "
                    f"head_dim % 4 == 0 (got head_dim={head_dim})"
                )
            if attention_type == "octonion" and head_dim % 8 != 0:
                raise ValueError(
                    f"attention schedule layer {layer_idx} uses octonion, which requires "
                    f"head_dim % 8 == 0 (got head_dim={head_dim})"
                )
            if attention_type == "gauge" and self.config.n_embd % 2 != 0:
                raise ValueError(
                    f"attention schedule layer {layer_idx} uses gauge, which requires even n_embd "
                    f"(got n_embd={self.config.n_embd})"
                )
            if attention_type == "clifford" and head_dim % 8 != 0:
                raise ValueError(
                    f"attention schedule layer {layer_idx} uses clifford, which requires "
                    f"head_dim % 8 == 0 (got head_dim={head_dim})"
                )
            if attention_type == "reversible":
                if self.config.n_embd % 2 != 0:
                    raise ValueError(
                        f"attention schedule layer {layer_idx} uses reversible, which requires even n_embd "
                        f"(got n_embd={self.config.n_embd})"
                    )
                if self.config.n_head % 2 != 0:
                    raise ValueError(
                        f"attention schedule layer {layer_idx} uses reversible, which requires even n_head "
                        f"(got n_head={self.config.n_head})"
                    )
                sub_n_head = self.config.n_head // 2
                if not (self.config.n_kv_head <= sub_n_head and sub_n_head % self.config.n_kv_head == 0):
                    raise ValueError(
                        f"attention schedule layer {layer_idx} uses reversible, which requires n_kv_head to "
                        f"divide n_head // 2 and be <= n_head // 2 "
                        f"(got n_kv_head={self.config.n_kv_head}, n_head//2={sub_n_head})"
                    )

        if has_reversible:
            if self.config.n_head % 2 != 0:
                raise ValueError("reversible attention requires n_head to be even (to keep head_dim constant)")
            sub_n_head = self.config.n_head // 2
            if not (self.config.n_kv_head <= sub_n_head and sub_n_head % self.config.n_kv_head == 0):
                raise ValueError(
                    "reversible attention requires n_kv_head to divide (n_head // 2) and be <= (n_head // 2)"
                )
            rev_mode = getattr(self.config, "reversible_mode", "additive")
            if rev_mode not in ("additive", "symplectic"):
                raise ValueError(f"reversible_mode must be additive | symplectic, got {rev_mode!r}")
            if float(getattr(self.config, "reversible_lambda_min", 0.05)) < 0.0:
                raise ValueError("reversible_lambda_min must be >= 0 (0 = unconfined falsification control)")
            if getattr(self.config, "reversible_tied", False) and rev_mode != "symplectic":
                raise ValueError("reversible_tied requires reversible_mode=symplectic (the tied shadow regime)")
            if getattr(self.config, "reversible_tied", False) and not all_reversible:
                raise ValueError("reversible_tied requires every attention schedule layer to be reversible")
        elif getattr(self.config, "reversible_mode", "additive") != "additive" or getattr(
            self.config, "reversible_tied", False
        ):
            raise ValueError("reversible_mode/reversible_tied require at least one reversible attention layer")
        if getattr(self.config, "disable_block_norms", False) and not all(
            name == "standard" for name in attention_schedule
        ):
            bad_idx, bad_name = next((i, n) for i, n in enumerate(attention_schedule) if n != "standard")
            raise ValueError(
                "disable_block_norms is the standard-attention falsification arm only (z4xx); "
                f"attention schedule layer {bad_idx} is {bad_name!r}"
            )
        ffn_type = getattr(self.config, "ffn_type", "standard")
        if ffn_type not in ("standard", "tropical", "tropical-rational"):
            raise ValueError(f"ffn_type must be standard | tropical | tropical-rational, got {ffn_type!r}")
        ffn_beta = getattr(self.config, "ffn_beta", None)
        if ffn_beta is not None and not (float(ffn_beta) > 0):
            raise ValueError(f"ffn_beta must be None or > 0, got {ffn_beta!r}")
        semiring_beta = getattr(self.config, "semiring_beta", None)
        if semiring_beta is not None and not (float(semiring_beta) > 0):
            raise ValueError(f"semiring_beta must be None or > 0, got {semiring_beta!r}")
        if semiring_beta is not None and "tropical" not in attention_schedule:
            raise ValueError("semiring_beta applies to the tropical attention path only (8gk.1)")

        parameterization = getattr(self.config, "parameterization", "current")
        if parameterization not in ("current", "nsa"):
            raise ValueError(f"parameterization must be current | nsa, got {parameterization!r}")
        if parameterization == "nsa" and "tropical" not in attention_schedule and not any(
            m in attention_schedule for m in ("quaternion", "octonion")
        ):
            raise ValueError(
                "parameterization='nsa' currently changes only the tropical (E[max] constants) "
                "and quaternion/octonion (rotor LR derivation) rules; refusing a silent no-op arm"
            )

        ca_rule = getattr(self.config, "ca_init_rule", None)
        if isinstance(ca_rule, str):
            ca_rule = ca_rule.strip().lower()
            if ca_rule in {"", "none", "off"}:
                ca_rule = None
        if ca_rule is not None and ca_rule not in _CA_RULES:
            raise ValueError(f"ca_init_rule must be one of {sorted(_CA_RULES)} (or unset), got {ca_rule!r}")
        ca_alpha = float(getattr(self.config, "ca_init_alpha", 1.0))
        if not (0.0 <= ca_alpha <= 1.0):
            raise ValueError(f"ca_init_alpha must be in [0, 1], got {ca_alpha}")
        ca_seed = getattr(self.config, "ca_init_seed", None)
        if ca_seed is not None and int(ca_seed) < 0:
            raise ValueError(f"ca_init_seed must be non-negative, got {ca_seed}")

    def init_weights(self):
        ca_rule = getattr(self.config, "ca_init_rule", None)
        if isinstance(ca_rule, str):
            ca_rule = ca_rule.strip().lower()
            if ca_rule in {"", "none", "off"}:
                ca_rule = None
        ca_alpha = float(getattr(self.config, "ca_init_alpha", 1.0))

        self._ca_init_generator: torch.Generator | None = None
        self._ca_init_rule_number: int | None = None
        if ca_rule is not None and ca_alpha > 0.0:
            ca_seed = getattr(self.config, "ca_init_seed", None)
            if ca_seed is None:
                ca_seed = 0
            self._ca_init_generator = torch.Generator(device="cpu")
            self._ca_init_generator.manual_seed(int(ca_seed))
            self._ca_init_rule_number = _CA_RULES[ca_rule]

        try:
            self.apply(self._init_weights)
        finally:
            self._ca_init_generator = None
            self._ca_init_rule_number = None
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)

        def _zero_proj_weight(module: nn.Module) -> None:
            # SurrealLayer projections have no .weight Parameter (w is recomposed
            # from weight_s/weight_v every forward, and exp(s)*normalize(v) cannot
            # represent zero with stable gradients) - keep their construction init.
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor):
                torch.nn.init.zeros_(weight)

        # zero out c_proj weights in all blocks (where present)
        for block in self.transformer.h:
            if hasattr(block, "mlp") and hasattr(block.mlp, "c_proj"):
                _zero_proj_weight(block.mlp.c_proj)
            if hasattr(block, "attn") and hasattr(block.attn, "c_proj"):
                _zero_proj_weight(block.attn.c_proj)
            if hasattr(block, "special_block"):
                sb = block.special_block
                if hasattr(sb, "c_proj"):
                    _zero_proj_weight(sb.c_proj)
                if hasattr(sb, "mlp") and hasattr(sb.mlp, "c_proj"):
                    _zero_proj_weight(sb.mlp.c_proj)
                if hasattr(sb, "f_block") and hasattr(sb.f_block, "c_proj"):
                    _zero_proj_weight(sb.f_block.c_proj)
                if hasattr(sb, "g_block") and hasattr(sb.g_block, "c_proj"):
                    _zero_proj_weight(sb.g_block.c_proj)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        ca_rule_number = getattr(self, "_ca_init_rule_number", None)
        ca_gen = getattr(self, "_ca_init_generator", None)
        ca_alpha = float(getattr(self.config, "ca_init_alpha", 1.0))
        use_ca = ca_rule_number is not None and ca_gen is not None and ca_alpha > 0.0
        pure_ca = use_ca and ca_alpha >= 1.0

        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            if pure_ca:
                try:
                    ca_vals = _ca_values_for_weight(
                        rule=int(ca_rule_number),
                        shape=tuple(module.weight.shape),
                        target_std=float(std),
                        generator=ca_gen,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"CA init failed for Linear weight (shape={tuple(module.weight.shape)}, rule={ca_rule_number})"
                    ) from exc
                ca_vals = ca_vals.to(device=module.weight.device)
                with torch.no_grad():
                    module.weight.copy_(ca_vals.to(dtype=module.weight.dtype))
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if use_ca:
                    with torch.no_grad():
                        try:
                            ca_vals = _ca_values_for_weight(
                                rule=int(ca_rule_number),
                                shape=tuple(module.weight.shape),
                                target_std=float(std),
                                generator=ca_gen,
                            )
                        except Exception as exc:
                            raise RuntimeError(
                                f"CA init failed for Linear weight (shape={tuple(module.weight.shape)}, rule={ca_rule_number})"
                            ) from exc
                        ca_vals = ca_vals.to(device=module.weight.device)
                        if module.weight.dtype in {torch.float16, torch.bfloat16}:
                            blended = module.weight.detach().float().mul(1.0 - ca_alpha).add_(ca_vals, alpha=ca_alpha)
                            module.weight.copy_(blended.to(dtype=module.weight.dtype))
                        else:
                            module.weight.mul_(1.0 - ca_alpha).add_(
                                ca_vals.to(dtype=module.weight.dtype), alpha=ca_alpha
                            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            emb_std = 1.0
            if pure_ca:
                try:
                    ca_vals = _ca_values_for_weight(
                        rule=int(ca_rule_number),
                        shape=tuple(module.weight.shape),
                        target_std=float(emb_std),
                        generator=ca_gen,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"CA init failed for Embedding weight (shape={tuple(module.weight.shape)}, rule={ca_rule_number})"
                    ) from exc
                ca_vals = ca_vals.to(device=module.weight.device)
                with torch.no_grad():
                    module.weight.copy_(ca_vals.to(dtype=module.weight.dtype))
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=emb_std)
                if use_ca:
                    with torch.no_grad():
                        try:
                            ca_vals = _ca_values_for_weight(
                                rule=int(ca_rule_number),
                                shape=tuple(module.weight.shape),
                                target_std=float(emb_std),
                                generator=ca_gen,
                            )
                        except Exception as exc:
                            raise RuntimeError(
                                f"CA init failed for Embedding weight (shape={tuple(module.weight.shape)}, rule={ca_rule_number})"
                            ) from exc
                        ca_vals = ca_vals.to(device=module.weight.device)
                        if module.weight.dtype in {torch.float16, torch.bfloat16}:
                            blended = module.weight.detach().float().mul(1.0 - ca_alpha).add_(ca_vals, alpha=ca_alpha)
                            module.weight.copy_(blended.to(dtype=module.weight.dtype))
                        else:
                            module.weight.mul_(1.0 - ca_alpha).add_(
                                ca_vals.to(dtype=module.weight.dtype), alpha=ca_alpha
                            )

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()  # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]  # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311

        The 6*N rule charges every matmul parameter 2 FLOPs in the forward and 4
        in a SINGLE backward (6 = 2 + 4), plus 12*L*H*Q*T for the attention
        score/value matmuls (their own fwd+bwd). Symplectic-reversible blocks
        (u55.5) break the single-backward assumption: each exact-gradient kick
        (SymplecticKick) evaluates a scalar potential phi, takes a FIRST backward
        to form grad(phi) DURING the model forward (create_graph=True), then a
        SECOND backward (autograd-of-autograd) at training time. In
        forward-equivalent units (fwd=1, bwd=2) a kick costs phi-fwd(1) +
        grad-phi(2) + double-bwd(~2*3=6) = 9 units versus a standard module's 3
        -> a 3x multiplier on the block compute (the kick params AND the
        attention living inside phi_F). Without this correction the estimator
        rates a symplectic arm CHEAPER than standard (it sees only the half-width
        block params), so equal-target-FLOPs budgeting silently trains it ~37%
        more steps and confounds every equal-FLOPs A/B that contains a symplectic
        arm (bead 7lba; the z4xx confound). Additive-reversible training now
        routes through ReversibleFunction (bead a6k3): backward RECOMPUTES the
        activations, so a block pair costs 9 forward-equivalent units per
        param (vs the canonical 6) and its attention matmuls execute on every
        pass (16 H Q T vs 12); both corrections are charged below. This is a
        leading-order analytic estimate;
        for the strictest symplectic equal-compute contract, budget by matched
        STEPS (train --max-steps without --target-flops) rather than FLOPs.
        """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = (
            self.config.n_layer,
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.sequence_len,
        )
        attn_flops_per_token = 12 * l * h * q * t
        attn_per_layer = 12 * h * q * t
        schedule = resolve_attention_schedule(self.config)
        symplectic_layers = [
            i
            for i, name in enumerate(schedule)
            if name == "reversible" and str(getattr(self.config, "reversible_mode", "additive")) == "symplectic"
        ]
        if symplectic_layers:
            # Block params live inside the kicks (the double-backward path);
            # lm_head and any other non-block, non-embedding params take the
            # ordinary 6x. A tied block dedupes by identity in .parameters(),
            # matching the global nparams count.
            seen_blocks: set[int] = set()
            symplectic_blocks: list[nn.Module] = []
            for idx in symplectic_layers:
                block = self.transformer.h[idx]
                if id(block) in seen_blocks:
                    continue
                seen_blocks.add(id(block))
                symplectic_blocks.append(block)
            nparams_block = sum(p.numel() for block in symplectic_blocks for p in block.parameters())
            nparams_nonblock = (nparams - nparams_embedding) - nparams_block
            attn_flops = attn_per_layer * (len(schedule) + 2 * len(symplectic_layers))
            return 6 * nparams_nonblock + 18 * nparams_block + attn_flops

        additive_layers = [
            i
            for i, name in enumerate(schedule)
            if name == "reversible" and str(getattr(self.config, "reversible_mode", "additive")) == "additive"
        ]
        if additive_layers:
            # Block params pay the wired recompute (bead a6k3): forward F+G
            # under no_grad (2 units) + backward G-reconstruct (1) + G re-run
            # with grad (3) + F re-run with grad (3) = 9 units vs the
            # canonical 6. The attention matmuls ride inside f_block and
            # execute on every pass: fwd(no_grad) + fwd(graph) + bwd =
            # 16 H Q T vs 12. A tied block dedupes by identity, matching the
            # symplectic branch above.
            seen_blocks: set[int] = set()
            additive_blocks: list[nn.Module] = []
            for idx in additive_layers:
                block = self.transformer.h[idx]
                if id(block) in seen_blocks:
                    continue
                seen_blocks.add(id(block))
                additive_blocks.append(block)
            nparams_block = sum(p.numel() for block in additive_blocks for p in block.parameters())
            nparams_nonblock = (nparams - nparams_embedding) - nparams_block
            attn_flops = attn_per_layer * len(schedule) + 4 * h * q * t * len(additive_layers)
            return 6 * nparams_nonblock + 9 * nparams_block + attn_flops

        num_flops_per_token = 6 * (nparams - nparams_embedding) + attn_flops_per_token
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        if self.config.optimizer_type == "hoss":
            print("Using HOSS optimizer")
            return [HOSS([p for p in self.parameters() if p.requires_grad], lr=matrix_lr)]

        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head).
        # Muon's Newton-Schulz orthogonalization requires ndim >= 2, but some
        # mechanisms carry sub-2D block parameters (simplicial mix_1/mix_2
        # scalars, TropicalMLP bias vectors) - those route to AdamW at the
        # matrix LR instead of crashing Muon. Frozen parameters (e.g. the dead
        # q/k projections of the purely-positional braid rmatrix law) never
        # enter an optimizer, and 2D parameters whose geometry is a per-position
        # scalar field rather than a matmul weight (the rmatrix rapidity table)
        # route to AdamW: Newton-Schulz orthogonalization of a rapidity profile
        # is meaningless.
        named_block = [(n, p) for n, p in self.transformer.h.named_parameters() if p.requires_grad]

        def _muon_exempt(name: str) -> bool:
            # Every rmatrix_* parameter is a positional scalar field (rapidity
            # increments, probe offsets, probe gates), never a matmul weight -
            # Newton-Schulz orthogonalization has no meaning for any of them.
            return ".rmatrix_" in name or name.startswith("rmatrix_")

        matrix_params = [p for n, p in named_block if p.ndim >= 2 and not _muon_exempt(n)]
        lowdim_block_params = [p for n, p in named_block if p.ndim < 2 or _muon_exempt(n)]
        embedding_params = [p for p in self.transformer.wte.parameters() if p.requires_grad]
        lm_head_params = [p for p in self.lm_head.parameters() if p.requires_grad]
        expected = len(list(self.transformer.h.parameters())) + len(list(self.transformer.wte.parameters())) + len(
            list(self.lm_head.parameters())
        )
        if len(list(self.parameters())) != expected:
            raise RuntimeError("Parameter count mismatch between blocks, embeddings, and lm_head")
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        if lowdim_block_params:
            adam_groups.append(dict(params=lowdim_block_params, lr=matrix_lr))
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else torch.optim.AdamW
        if (
            (not ddp)
            and next(self.parameters()).is_cuda
            and ("fused" in inspect.signature(torch.optim.AdamW).parameters)
        ):
            adamw_kwargs["fused"] = True
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        if T > self.cos.size(1):
            raise ValueError(f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}")
        if idx.device != self.cos.device:
            raise ValueError(f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}")
        if self.cos.dtype != torch.bfloat16:
            raise TypeError("Rotary embeddings must be in bfloat16")
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        if kv_cache is not None and getattr(self.config, "reversible_tied", False):
            # the tied block owns ONE attention instance (layer_idx 0), so
            # per-layer cache slots would alias - teacher-forced forward and
            # prefill-free scoring are the supported tied paths (u55.5)
            raise ValueError("reversible_tied does not support kv-cache generation (per-layer cache aliasing)")
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0 : T0 + T], self.sin[:, T0 : T0 + T]  # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15
        if targets is not None:
            # training mode: compute and return the loss
            # TODO: experiment with Liger Kernels / chunked cross-entropy etc.
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)  # logits softcap
            logits = logits.float()  # use tf32/fp32 for logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction
            )
            return loss
        else:
            # inference mode: compute and return the logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)  # logits softcap
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        if not isinstance(tokens, list):
            raise TypeError("tokens must be a list")
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
