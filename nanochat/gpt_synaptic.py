# nanochat/gpt_synaptic.py
# pylint: disable=too-many-instance-attributes
from __future__ import annotations
# GPT with Synaptic Attention/MLP and optional Synaptic MoE + structural hooks

from nanochat.torch_imports import torch, nn, F, Tensor
from dataclasses import dataclass, field
from typing import Optional

from .synaptic import (
    SynapticCausalSelfAttention,
    SynapticMLP,
    SynapticConfig,
    SynapticMoE,
)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class GPTSynapticConfig:
    sequence_len: int = 2048
    vocab_size: int = 65536
    n_layer: int = 20
    n_head: int = 10
    n_kv_head: int = 10
    n_embd: int = 1280
    rope_base: float = 10000.0
    synapses: bool = True
    syn_cfg: SynapticConfig = field(default_factory=SynapticConfig)
    dropout: float = 0.0
    # MoE & structural options
    use_moe: bool = False
    num_experts: int = 8
    moe_top_k: int = 2
    moe_hidden_mult: int = 4
    moe_balance_loss: float = 0.01
    structural_every: int = 0  # 0 → off; >0 → run hooks every N blocks


# -----------------------------------------------------------------------------
# Blocks
# -----------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, n_embd: int, syn_cfg: SynapticConfig, dropout: float = 0.0):
        super().__init__()
        self.mlp = SynapticMLP(n_embd, syn_cfg, dropout)

    def forward(self, x):
        return self.mlp(x)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: int,
        rope_cos: Tensor,
        rope_sin: Tensor,
        syn_cfg: SynapticConfig,
        attn_drop=0.0,
        resid_drop=0.0,
    ):
        super().__init__()
        self.attn = SynapticCausalSelfAttention(
            n_embd,
            n_head,
            n_kv_head,
            rope_cos,
            rope_sin,
            syn_cfg,
            attn_drop,
            resid_drop,
        )

    def forward(self, x, kv_cache=None, presyn_state=None, train_mode=True):
        y, st = self.attn(x, kv_cache, presyn_state, train_mode)
        return y, st


class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: int,
        rope_cos: Tensor,
        rope_sin: Tensor,
        syn_cfg: SynapticConfig,
        dropout: float = 0.0,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        hidden_mult: int = 4,
        balance_loss: float = 0.01,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd,
            n_head,
            n_kv_head,
            rope_cos,
            rope_sin,
            syn_cfg,
            attn_drop=dropout,
            resid_drop=dropout,
        )
        self.norm2 = nn.LayerNorm(n_embd)
        self.use_moe: bool = use_moe
        self.balance_loss: float = float(balance_loss)
        self.last_aux_loss: Tensor = torch.tensor(0.0)
        self.mlp = (
            SynapticMoE(n_embd, num_experts, top_k, hidden_mult, syn_cfg, dropout)
            if use_moe
            else MLP(n_embd, syn_cfg, dropout)
        )

    def forward(self, x, kv_cache=None, presyn_state=None, train_mode=True):
        a, st = self.attn(self.norm1(x), kv_cache, presyn_state, train_mode)
        x = x + a
        if self.use_moe:
            y, aux = self.mlp(self.norm2(x))
            self.last_aux_loss = self.balance_loss * aux
        else:
            y = self.mlp(self.norm2(x))
            self.last_aux_loss = torch.tensor(0.0, device=x.device)
        x = x + y
        return x, st


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class GPTSynaptic(nn.Module):
    def __init__(self, config: GPTSynapticConfig):
        super().__init__()
        c = config
        self.config: GPTSynapticConfig = c
        self.wte: nn.Embedding = nn.Embedding(c.vocab_size, c.n_embd)
        self.h: nn.ModuleList[Block] = nn.ModuleList()
        self.transformer = nn.ModuleDict(dict(wte=self.wte, h=self.h))
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.drop = nn.Dropout(c.dropout)
        nn.init.trunc_normal_(self.lm_head.weight, std=0.02)
        T = c.sequence_len
        hd = c.n_embd // c.n_head
        base = c.rope_base
        inv_freq: Tensor = 1.0 / (
            base ** (torch.arange(0, hd // 2, dtype=torch.float32) / (hd // 2))
        )
        t: Tensor = torch.arange(0, T * 8, dtype=torch.float32)
        freqs: Tensor = torch.outer(t, inv_freq)
        self.cos: Tensor
        self.register_buffer(
            "cos", torch.cos(freqs).unsqueeze(0).to(torch.bfloat16), persistent=False
        )
        self.sin: Tensor
        self.register_buffer(
            "sin", torch.sin(freqs).unsqueeze(0).to(torch.bfloat16), persistent=False
        )
        for _ in range(c.n_layer):
            self.h.append(
                Block(
                    c.n_embd,
                    c.n_head,
                    c.n_kv_head,
                    self.cos,
                    self.sin,
                    c.syn_cfg,
                    dropout=c.dropout,
                    use_moe=c.use_moe,
                    num_experts=c.num_experts,
                    top_k=c.moe_top_k,
                    hidden_mult=c.moe_hidden_mult,
                    balance_loss=c.moe_balance_loss,
                )
            )

    def estimate_flops(self):
        L = self.config.n_layer
        N = self.config.n_embd
        H = self.config.n_head
        return 6 * L * N * N + 4 * L * N * H * 128

    def forward(
        self,
        idx: Tensor,
        targets: Optional[Tensor] = None,
        kv_cache=None,
        train_mode=True,
    ):
        B, T = idx.size()
        assert T <= self.config.sequence_len
        tok = self.wte(idx)
        x = self.drop(tok.to(dtype=torch.bfloat16))
        presyn_state = None
        for li, block in enumerate(self.h):
            x, presyn_state = block(x, kv_cache, presyn_state, train_mode)
            if self.config.structural_every and targets is not None:
                if (li + 1) % self.config.structural_every == 0 and hasattr(
                    block.mlp, "experts"
                ):
                    # Hook point for split/merge (kept as a callable point on purpose)
                    pass
        logits = self.lm_head(x.to(dtype=self.lm_head.weight.dtype))
        if targets is None:
            return logits, None
        aux = sum(
            (
                getattr(b, "last_aux_loss", torch.tensor(0.0, device=logits.device))
                for b in self.h
            )
        )
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="mean"
        )
        loss = ce + aux
        return logits, loss

    def setup_optimizers(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        lr=None,
        wd=None,
    ):
        # Support both old GPT-style signature and new simple signature
        # If GPT-style args are provided (defaults), use GPT-style optimizer setup
        # Otherwise use simple single optimizer
        if lr is not None and wd is not None:
            # New simple signature (explicit lr/wd provided)
            no_decay, set_decay = set(), set()
            for n, p in self.named_parameters():
                if p.ndim < 2 or "lm_head" in n or "wte" in n:
                    no_decay.add(n)
                else:
                    set_decay.add(n)
            optim_groups = [
                {
                    "params": [p for n, p in self.named_parameters() if n in set_decay],
                    "weight_decay": wd,
                },
                {
                    "params": [p for n, p in self.named_parameters() if n in no_decay],
                    "weight_decay": 0.0,
                },
            ]
            opt = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
            return [opt]  # Return as list for compatibility
        else:
            # GPT-style signature (for compatibility with existing training scripts)
            from nanochat.common import get_dist_info
            from nanochat.muon import Muon, DistMuon
            from nanochat.adamw import DistAdamW
            from functools import partial

            model_dim = self.config.n_embd
            ddp, rank, local_rank, world_size = get_dist_info()
            
            # Separate matrix params (2D) for Muon from other params (1D/0D) for AdamW
            matrix_params = []
            other_params = []
            
            # Collect params from transformer blocks
            for p in self.h.parameters():
                if p.ndim >= 2:
                    matrix_params.append(p)
                else:
                    other_params.append(p)
            
            embedding_params = list(self.wte.parameters())
            lm_head_params = list(self.lm_head.parameters())
            
            dmodel_lr_scale = (model_dim / 768) ** -0.5
            if rank == 0:
                print(
                    f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}"
                )
            
            # AdamW gets embedding, lm_head, and all 1D/0D params from blocks (biases, layernorms)
            adam_groups = [
                dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
                dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
                dict(params=other_params, lr=embedding_lr * dmodel_lr_scale), # Use embedding LR scale for other params? Or maybe just matrix_lr? Usually AdamW params get higher LR.
            ]
            adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
            AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
            adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
            
            muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
            MuonFactory = DistMuon if ddp else Muon
            muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
            optimizers = [adamw_optimizer, muon_optimizer]
            for opt in optimizers:
                for group in opt.param_groups:
                    group["initial_lr"] = group["lr"]
            return optimizers

    def init_weights(self):
        """Initialize weights (needed for checkpoint loading compatibility)."""
        # RoPE buffers are already initialized in __init__
        pass

    def get_device(self):
        return self.wte.weight.device
