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

from nanochat.braid_attention_torch import BraidCausalSelfAttention
from nanochat.clifford_attention_torch import CliffordCausalSelfAttention
from nanochat.fractal_attention_torch import FractalCausalSelfAttention
from nanochat.gauge_block_torch import GaugeBlock
from nanochat.gpt import GPT, Block, CausalSelfAttention, GPTConfig
from nanochat.hyperbolic_attention_torch import HyperbolicCausalSelfAttention
from nanochat.model_utils import (
    AttentionCore,
    apply_rotary_emb,
    build_rotary_embeddings,
    causal_attn_mask,
    norm,
)
from nanochat.octonion_attention_torch import OctonionCausalSelfAttention
from nanochat.quaternion_attention_torch import QuaternionCausalSelfAttention
from nanochat.reversible_block_torch import ReversibleBlock
from nanochat.simplicial_attention_torch import SimplicialCausalSelfAttention
from nanochat.surreal_torch import SurrealCausalSelfAttention
from nanochat.tropical_attention_torch import TropicalCausalSelfAttention, TropicalMLP
from nanochat.ultrametric_attention_torch import UltrametricCausalSelfAttention

MECHANISM_CLASSES: dict[str, type] = {
    "standard": CausalSelfAttention,
    "tropical": TropicalCausalSelfAttention,
    "ultrametric": UltrametricCausalSelfAttention,
    "simplicial": SimplicialCausalSelfAttention,
    "quaternion": QuaternionCausalSelfAttention,
    "braid": BraidCausalSelfAttention,
    "fractal": FractalCausalSelfAttention,
    "octonion": OctonionCausalSelfAttention,
    "surreal": SurrealCausalSelfAttention,
    "clifford": CliffordCausalSelfAttention,
    "hyperbolic": HyperbolicCausalSelfAttention,
    "gauge": GaugeBlock,
    "reversible": ReversibleBlock,
}

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
