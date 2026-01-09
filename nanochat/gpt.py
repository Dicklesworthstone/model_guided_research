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

from nanochat.common import get_dist_info
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
from nanochat.model_utils import norm, apply_rotary_emb, causal_attn_mask
from nanochat.tropical_attention_torch import TropicalCausalSelfAttention
from nanochat.ultrametric_attention_torch import UltrametricCausalSelfAttention
from nanochat.simplicial_attention_torch import SimplicialCausalSelfAttention
from nanochat.quaternion_attention_torch import QuaternionCausalSelfAttention
from nanochat.braid_attention_torch import BraidCausalSelfAttention
from nanochat.fractal_attention_torch import FractalCausalSelfAttention
from nanochat.octonion_attention_torch import OctonionCausalSelfAttention
from nanochat.surreal_torch import SurrealCausalSelfAttention
from nanochat.reversible_block_torch import ReversibleBlock
from nanochat.gauge_block_torch import GaugeBlock
from nanochat.hoss_opt_torch import HOSS

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
    attention_type: str = "standard"
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
    # Ultrametric-specific options (see nanochat.ultrametric_attention_torch).
    ultrametric_mode: str = "kernel"  # "kernel" | "trie"
    ultrametric_hard_digits: bool = False
    ultrametric_K: int = 8
    ultrametric_p: int = 2
    ultrametric_alpha: float = 2.0
    ultrametric_lcp_beta: float = 32.0
    # Braid-specific options (see nanochat.braid_attention_torch).
    braid_mode: str = "soft"  # "soft" | "discrete"
    braid_tau: float = 0.0
    braid_crossing_law: str = "restricted"  # "restricted" | "ybe"
    braid_record_schedule: bool = False
    braid_verify: bool = False

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
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
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not (self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0):
            raise ValueError("n_kv_head must divide n_head and be <= n_head")
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
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

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = self.n_head != self.n_kv_head # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
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

            prefix_len = Tk - Tq

            def causal_mask(b, h, q_idx, kv_idx):
                return kv_idx <= (prefix_len + q_idx)

            block_mask = create_block_mask(causal_mask, B, self.n_head, Tq, Tk, device=q.device)
            y = self._flex_attention(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)
        else:
            if kv_cache is None or Tq == Tk:
                # During training (no KV cache), attend as usual with causal attention
                # And even if there is KV cache, we can still use this simple version when Tq == Tk
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
            elif Tq == 1:
                # During inference but with a single query in this forward pass:
                # The query has to attend to all the keys/values in the cache
                y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
            else:
                # During inference AND we have a chunk of queries in this forward pass:
                # First, each query attends to all the cached keys/values (i.e. full prefix)
                attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
                prefix_len = Tk - Tq
                if prefix_len > 0: # can't be negative but could be zero
                    attn_mask[:, :prefix_len] = True
                # Then, causal attention within this chunk
                attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


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


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        # Special Block Types that replace the standard Attention+MLP structure
        if config.attention_type == "gauge":
            self.special_block = GaugeBlock(config, layer_idx)
            return
        if config.attention_type == "reversible":
            # Reversible blocks split channels in half: x = [x1, x2].
            # We keep the RoPE head_dim constant by halving the number of query heads.
            # IMPORTANT: KV cache is allocated from the top-level config, so we keep n_kv_head unchanged.
            sub_config = GPTConfig(**config.__dict__)
            sub_config.n_embd = config.n_embd // 2
            sub_config.n_head = config.n_head // 2
            sub_config.n_kv_head = config.n_kv_head

            self.special_block = ReversibleBlock(
                config,
                layer_idx,
                CausalSelfAttention(sub_config, layer_idx),
                MLP(sub_config),
            )
            return
            
        if config.attention_type == "tropical":
            self.attn = TropicalCausalSelfAttention(config, layer_idx)
        elif config.attention_type == "ultrametric":
            self.attn = UltrametricCausalSelfAttention(config, layer_idx)
        elif config.attention_type == "simplicial":
            self.attn = SimplicialCausalSelfAttention(config, layer_idx)
        elif config.attention_type == "quaternion":
            self.attn = QuaternionCausalSelfAttention(config, layer_idx)
        elif config.attention_type == "braid":
            self.attn = BraidCausalSelfAttention(config, layer_idx)
        elif config.attention_type == "fractal":
            self.attn = FractalCausalSelfAttention(config, layer_idx)
        elif config.attention_type == "octonion":
            self.attn = OctonionCausalSelfAttention(config, layer_idx)
        elif config.attention_type == "surreal":
            self.attn = SurrealCausalSelfAttention(config, layer_idx)
        else:
            self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        if self.config.attention_type == "gauge":
            return self.special_block(x, cos_sin, kv_cache)
        if self.config.attention_type == "reversible":
            return self.special_block(x, cos_sin, kv_cache)
            
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._validate_config()
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
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

        if self.config.attention_type == "reversible":
            if self.config.n_head % 2 != 0:
                raise ValueError("reversible attention requires n_head to be even (to keep head_dim constant)")
            sub_n_head = self.config.n_head // 2
            if not (self.config.n_kv_head <= sub_n_head and sub_n_head % self.config.n_kv_head == 0):
                raise ValueError(
                    "reversible attention requires n_kv_head to divide (n_head // 2) and be <= (n_head // 2)"
                )
        if self.config.attention_type == "gauge":
            if self.config.n_kv_head != self.config.n_head:
                raise ValueError("gauge attention requires n_kv_head == n_head (GQA not supported)")

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
        # zero out c_proj weights in all blocks (where present)
        for block in self.transformer.h:
            if hasattr(block, "mlp") and hasattr(block.mlp, "c_proj"):
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if hasattr(block, "attn") and hasattr(block.attn, "c_proj"):
                torch.nn.init.zeros_(block.attn.c_proj.weight)
            if hasattr(block, "special_block"):
                sb = block.special_block
                if hasattr(sb, "c_proj"):
                    torch.nn.init.zeros_(sb.c_proj.weight)
                if hasattr(sb, "f_block") and hasattr(sb.f_block, "c_proj"):
                    torch.nn.init.zeros_(sb.f_block.c_proj.weight)
                if hasattr(sb, "g_block") and hasattr(sb.g_block, "c_proj"):
                    torch.nn.init.zeros_(sb.g_block.c_proj.weight)
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
                            module.weight.mul_(1.0 - ca_alpha).add_(ca_vals.to(dtype=module.weight.dtype), alpha=ca_alpha)
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
                            module.weight.mul_(1.0 - ca_alpha).add_(ca_vals.to(dtype=module.weight.dtype), alpha=ca_alpha)

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
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        if self.config.optimizer_type == "hoss":
            print("Using HOSS optimizer")
            return [HOSS(self.parameters(), lr=matrix_lr)]

        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        expected = len(matrix_params) + len(embedding_params) + len(lm_head_params)
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
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else torch.optim.AdamW
        if (not ddp) and next(self.parameters()).is_cuda and ("fused" in inspect.signature(torch.optim.AdamW).parameters):
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

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        if T > self.cos.size(1):
            raise ValueError(f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}")
        if idx.device != self.cos.device:
            raise ValueError(f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}")
        if self.cos.dtype != torch.bfloat16:
            raise TypeError("Rotary embeddings must be in bfloat16")
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

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
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference mode: compute and return the logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
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
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
