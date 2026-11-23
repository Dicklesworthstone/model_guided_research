"""Public API for nanochat mathematical attention mechanisms (bead model_guided_research-swh.1).

Every attention mechanism and block-level transform can be constructed directly
from dimensions via ``MechanismClass.from_dims(n_embd, n_head, **mechanism_kwargs)``
without instantiating ``GPTConfig``:

    import torch
    from nanochat import TropicalCausalSelfAttention

    attn = TropicalCausalSelfAttention.from_dims(n_embd=64, n_head=4, semiring_beta=2.0)
    x = torch.randn(2, 16, 64)
    y = attn(x)  # (2, 16, 64)
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanochat.braid_attention_torch import BraidCausalSelfAttention
    from nanochat.clifford_attention_torch import CliffordCausalSelfAttention
    from nanochat.fractal_attention_torch import FractalCausalSelfAttention
    from nanochat.gauge_block_torch import GaugeBlock
    from nanochat.gpt import GPT, Block, CausalSelfAttention, GPTConfig
    from nanochat.hyperbolic_attention_torch import HyperbolicCausalSelfAttention
    from nanochat.model_utils import AttentionCore, apply_rotary_emb, build_rotary_embeddings, causal_attn_mask, norm
    from nanochat.octonion_attention_torch import OctonionCausalSelfAttention
    from nanochat.quaternion_attention_torch import QuaternionCausalSelfAttention
    from nanochat.reversible_block_torch import ReversibleBlock
    from nanochat.simplicial_attention_torch import SimplicialCausalSelfAttention
    from nanochat.surreal_torch import SurrealCausalSelfAttention
    from nanochat.tropical_attention_torch import TropicalCausalSelfAttention, TropicalMLP
    from nanochat.ultrametric_attention_torch import UltrametricCausalSelfAttention

    MECHANISM_CLASSES: dict[str, type]


# A spawned interpreter importing nanochat.execution needs only the standard
# library. Import tensor libraries when a caller first requests a model.
_MECHANISMS = {
    "standard": ("gpt", "CausalSelfAttention"),
    "tropical": ("tropical_attention_torch", "TropicalCausalSelfAttention"),
    "ultrametric": ("ultrametric_attention_torch", "UltrametricCausalSelfAttention"),
    "simplicial": ("simplicial_attention_torch", "SimplicialCausalSelfAttention"),
    "quaternion": ("quaternion_attention_torch", "QuaternionCausalSelfAttention"),
    "braid": ("braid_attention_torch", "BraidCausalSelfAttention"),
    "fractal": ("fractal_attention_torch", "FractalCausalSelfAttention"),
    "octonion": ("octonion_attention_torch", "OctonionCausalSelfAttention"),
    "surreal": ("surreal_torch", "SurrealCausalSelfAttention"),
    "clifford": ("clifford_attention_torch", "CliffordCausalSelfAttention"),
    "hyperbolic": ("hyperbolic_attention_torch", "HyperbolicCausalSelfAttention"),
    "gauge": ("gauge_block_torch", "GaugeBlock"),
    "reversible": ("reversible_block_torch", "ReversibleBlock"),
}
_MODULES = {
    **{name: module for module, name in _MECHANISMS.values()},
    **dict.fromkeys(("GPT", "GPTConfig", "Block"), "gpt"),
    **dict.fromkeys(
        ("AttentionCore", "apply_rotary_emb", "build_rotary_embeddings", "causal_attn_mask", "norm"), "model_utils"
    ),
    "TropicalMLP": "tropical_attention_torch",
}


def __getattr__(name: str) -> Any:
    if name == "MECHANISM_CLASSES":
        value = {
            key: getattr(import_module(f".{module}", __name__), symbol) for key, (module, symbol) in _MECHANISMS.items()
        }
    elif name in _MODULES:
        value = getattr(import_module(f".{_MODULES[name]}", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return globals().setdefault(name, value)


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))


__all__ = [
    "AttentionCore",
    "Block",
    "BraidCausalSelfAttention",
    "CausalSelfAttention",
    "CliffordCausalSelfAttention",
    "FractalCausalSelfAttention",
    "GPT",
    "GPTConfig",
    "GaugeBlock",
    "HyperbolicCausalSelfAttention",
    "MECHANISM_CLASSES",
    "OctonionCausalSelfAttention",
    "QuaternionCausalSelfAttention",
    "ReversibleBlock",
    "SimplicialCausalSelfAttention",
    "SurrealCausalSelfAttention",
    "TropicalCausalSelfAttention",
    "TropicalMLP",
    "UltrametricCausalSelfAttention",
    "apply_rotary_emb",
    "build_rotary_embeddings",
    "causal_attn_mask",
    "norm",
]
