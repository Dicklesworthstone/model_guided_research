"""
Ultrametric Attention Module (PyTorch).

Modes (select via GPTConfig.ultrametric_mode):
- ``kernel``: continuous LCP-kernel attention (baseline; supports training/prefill).
- ``trie``: packed prefix-trie lookup for KV-cache decode (currently CPU-only and only used when Tq==1).
"""

import math
import os
import weakref
from dataclasses import dataclass

import torch
import torch.nn as nn
from nanochat.model_utils import norm, apply_rotary_emb, causal_attn_mask, repeat_kv_heads


@dataclass
class _TrieCacheState:
    """Per-(kv_cache, layer) trie state, kept on CPU for fast Python-side updates."""

    tries: list[list["_PackedPrefixTrie"]]  # [B][H]
    seen_Tk: int


class _PackedPrefixTrie:
    """Packed p-ary prefix trie storing subtree sums and counts.

    Nodes store:
    - child indices (int32) for digits in [0, p)
    - subtree sum of values (float32)
    - subtree count (int32)
    """

    def __init__(self, *, p: int, K: int, head_dim: int, device: torch.device):
        if p <= 1:
            raise ValueError("p must be >= 2")
        if K <= 0:
            raise ValueError("K must be positive")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if device.type != "cpu":
            raise ValueError("Trie mode currently supports CPU only")

        self.p = int(p)
        self.K = int(K)
        self.head_dim = int(head_dim)
        self.device = device

        self._cap = 256
        # Node 0 is root.
        self._size = 1
        self._child = torch.full((self._cap, self.p), -1, dtype=torch.int32, device=self.device)
        self._sum_v = torch.zeros((self._cap, self.head_dim), dtype=torch.float32, device=self.device)
        self._count = torch.zeros((self._cap,), dtype=torch.int32, device=self.device)

    def reset(self) -> None:
        self._size = 1
        self._child.fill_(-1)
        self._sum_v.zero_()
        self._count.zero_()

    def _grow(self) -> None:
        new_cap = int(self._cap) * 2
        child = torch.full((new_cap, self.p), -1, dtype=torch.int32, device=self.device)
        sum_v = torch.zeros((new_cap, self.head_dim), dtype=torch.float32, device=self.device)
        count = torch.zeros((new_cap,), dtype=torch.int32, device=self.device)

        child[: self._size] = self._child[: self._size]
        sum_v[: self._size] = self._sum_v[: self._size]
        count[: self._size] = self._count[: self._size]

        self._child = child
        self._sum_v = sum_v
        self._count = count
        self._cap = new_cap

    def _alloc(self) -> int:
        if self._size >= self._cap:
            self._grow()
        idx = int(self._size)
        self._size += 1
        return idx

    def insert(self, digits: torch.Tensor, v: torch.Tensor) -> None:
        if digits.ndim != 1 or digits.numel() != self.K:
            raise ValueError(f"insert expects digits shape ({self.K},), got {tuple(digits.shape)}")
        if v.ndim != 1 or v.numel() != self.head_dim:
            raise ValueError(f"insert expects v shape ({self.head_dim},), got {tuple(v.shape)}")
        if digits.device.type != "cpu" or v.device.type != "cpu":
            raise ValueError("Trie insert expects CPU tensors")

        node = 0
        v32 = v.to(dtype=torch.float32)
        self._sum_v[node] += v32
        self._count[node] += 1
        for d in range(self.K):
            a = int(digits[d].item())
            nxt = int(self._child[node, a].item())
            if nxt < 0:
                nxt = self._alloc()
                self._child[node, a] = nxt
            node = nxt
            self._sum_v[node] += v32
            self._count[node] += 1

    def query(self, digits: torch.Tensor, *, alpha: float) -> torch.Tensor:
        if digits.ndim != 1 or digits.numel() != self.K:
            raise ValueError(f"query expects digits shape ({self.K},), got {tuple(digits.shape)}")
        if digits.device.type != "cpu":
            raise ValueError("Trie query expects CPU digits")
        if not (alpha > 1.0 and math.isfinite(alpha)):
            raise ValueError("alpha must be finite and > 1")

        # Collect subtree aggregates along the query path: S[0] is root, S[l] is prefix length l.
        sums: list[torch.Tensor] = [self._sum_v[0]]
        counts: list[torch.Tensor] = [self._count[0]]
        node = 0
        for d in range(self.K):
            a = int(digits[d].item())
            nxt = int(self._child[node, a].item())
            if nxt < 0:
                break
            node = nxt
            sums.append(self._sum_v[node])
            counts.append(self._count[node])

        # Pad to length K+1 with zeros so exact buckets are well-defined.
        while len(sums) < self.K + 1:
            sums.append(torch.zeros((self.head_dim,), dtype=torch.float32, device=self.device))
            counts.append(torch.zeros((), dtype=torch.int32, device=self.device))

        num = torch.zeros((self.head_dim,), dtype=torch.float32, device=self.device)
        den = torch.zeros((), dtype=torch.float32, device=self.device)
        for l in range(self.K):
            exact_sum = sums[l] - sums[l + 1]
            exact_count = (counts[l] - counts[l + 1]).to(dtype=torch.float32)
            w = alpha**l
            num += exact_sum * w
            den += exact_count * w
        # l = K bucket (exact match to depth K).
        num += sums[self.K] * (alpha**self.K)
        den += counts[self.K].to(dtype=torch.float32) * (alpha**self.K)
        return num / den.clamp_min(1e-9)

class UltrametricCausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # Ultrametric hyperparameters + mode selection.
        self.K = int(getattr(config, "ultrametric_K", 8))
        self.p = int(getattr(config, "ultrametric_p", 2))
        self.alpha = float(getattr(config, "ultrametric_alpha", 2.0))
        self.lcp_beta = float(getattr(config, "ultrametric_lcp_beta", 32.0))
        mode = str(getattr(config, "ultrametric_mode", os.environ.get("NANOCHAT_ULTRAMETRIC_MODE", "kernel"))).strip().lower()
        self.ultrametric_mode = mode
        self.ultrametric_hard_digits = bool(getattr(config, "ultrametric_hard_digits", False))

        if self.K <= 0:
            raise ValueError(f"ultrametric_K must be positive, got {self.K}")
        if self.p < 2:
            raise ValueError(f"ultrametric_p must be >= 2, got {self.p}")
        if not (self.alpha > 1.0 and math.isfinite(self.alpha)):
            raise ValueError(f"ultrametric_alpha must be finite and > 1, got {self.alpha}")
        if not (self.lcp_beta > 0.0 and math.isfinite(self.lcp_beta)):
            raise ValueError(f"ultrametric_lcp_beta must be finite and > 0, got {self.lcp_beta}")
        if self.ultrametric_mode not in {"kernel", "trie"}:
            raise ValueError(f"ultrametric_mode must be 'kernel' or 'trie', got {self.ultrametric_mode!r}")
        self._log_alpha = math.log(self.alpha)

        # Trie cache keyed by KVCache object (decode-only; CPU).
        self._trie_cache: weakref.WeakKeyDictionary[object, _TrieCacheState] = weakref.WeakKeyDictionary()
        # Digits are derived per-head, so the projection is over head_dim.
        self.to_digits_q = nn.Linear(self.head_dim, self.K, bias=False)
        self.to_digits_k = nn.Linear(self.head_dim, self.K, bias=False)

    def _digits_soft(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(raw) * (self.p - 1)

    def _digits_hard_int(self, raw: torch.Tensor) -> torch.Tensor:
        digits = torch.round(self._digits_soft(raw)).to(dtype=torch.int64)
        return digits.clamp_(0, self.p - 1)

    def _get_trie_state(self, kv_cache: object, *, B: int, H: int, device: torch.device) -> _TrieCacheState:
        state = self._trie_cache.get(kv_cache)
        if state is not None:
            if len(state.tries) == B and (B == 0 or len(state.tries[0]) == H):
                return state
        tries = [
            [_PackedPrefixTrie(p=self.p, K=self.K, head_dim=self.head_dim, device=device) for _ in range(H)] for _ in range(B)
        ]
        state = _TrieCacheState(tries=tries, seen_Tk=0)
        self._trie_cache[kv_cache] = state
        return state

    def _reset_trie_state(self, state: _TrieCacheState) -> None:
        for row in state.tries:
            for trie in row:
                trie.reset()
        state.seen_Tk = 0

    def _update_trie_from_kv(self, state: _TrieCacheState, k: torch.Tensor, v: torch.Tensor) -> None:
        Tk = int(k.size(2))
        if Tk < state.seen_Tk:
            self._reset_trie_state(state)
        if state.seen_Tk >= Tk:
            return

        k_new = k[:, :, state.seen_Tk:Tk]
        v_new = v[:, :, state.seen_Tk:Tk]
        digits = self._digits_hard_int(self.to_digits_k(k_new))  # (B, H, Tnew, K)
        B = int(k.size(0))
        H = int(k.size(1))
        Tnew = int(k_new.size(2))
        for b in range(B):
            for h in range(H):
                trie = state.tries[b][h]
                for t in range(Tnew):
                    trie.insert(digits[b, h, t], v_new[b, h, t])
        state.seen_Tk = Tk

    def _trie_decode(self, state: _TrieCacheState, q: torch.Tensor, *, out_dtype: torch.dtype) -> torch.Tensor:
        # q: (B, H, 1, D)
        q_digits = self._digits_hard_int(self.to_digits_q(q))  # (B, H, 1, K)
        B = int(q.size(0))
        H = int(q.size(1))
        y = torch.empty((B, H, self.head_dim), dtype=torch.float32, device=q.device)
        for b in range(B):
            for h in range(H):
                y[b, h] = state.tries[b][h].query(q_digits[b, h, 0], alpha=self.alpha)
        return y.to(dtype=out_dtype).unsqueeze(2)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        if self.n_kv_head != self.n_head:
            k, v = repeat_kv_heads(k, v, n_head=self.n_head)
        Tq = q.size(2)
        Tk = k.size(2)
        mode = str(self.ultrametric_mode).strip().lower()
        if mode == "trie" and kv_cache is not None and q.device.type == "cpu":
            state = self._get_trie_state(kv_cache, B=int(B), H=int(k.size(1)), device=q.device)
            self._update_trie_from_kv(state, k, v)
            if Tq == 1:
                y = self._trie_decode(state, q, out_dtype=v.dtype)  # (B, H, 1, D)
                y = y.transpose(1, 2).contiguous().view(B, T, -1)
                return self.c_proj(y)

        # LCP-kernel ultrametric attention (continuous relaxation).
        #
        # We map queries/keys to K "digits" in base p, then compute a differentiable proxy
        # for the longest-common-prefix (LCP) depth. Attention weights are derived from
        # the ultrametric similarity kernel w(q,k) ∝ alpha^{LCP(q,k)} (alpha > 1).
        q_dig_raw = self.to_digits_q(q)  # (B, H, Tq, K)
        k_dig_raw = self.to_digits_k(k)  # (B, H, Tk, K)

        if self.ultrametric_hard_digits:
            q_dig = self._digits_hard_int(q_dig_raw).to(dtype=torch.float32)
            k_dig = self._digits_hard_int(k_dig_raw).to(dtype=torch.float32)
        else:
            q_dig = self._digits_soft(q_dig_raw)
            k_dig = self._digits_soft(k_dig_raw)

        # Per-depth match probability and expected LCP depth via prefix products.
        diff = (q_dig.unsqueeze(3) - k_dig.unsqueeze(2)).abs()  # (B, H, Tq, Tk, K)
        match_prob = torch.exp(-self.lcp_beta * diff.square())
        prefix_prob = torch.cumprod(match_prob, dim=-1)
        lcp = prefix_prob.sum(dim=-1)  # (B, H, Tq, Tk) in [0, K]

        causal_mask = causal_attn_mask(Tq, Tk, device=q.device)  # (Tq, Tk)

        weights = torch.exp(lcp.to(torch.float32) * self._log_alpha)
        weights = weights.masked_fill(~causal_mask, 0.0)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        attn = (weights / denom).to(dtype=v.dtype)

        y = attn @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


def perf_sanity_ultrametric_trie_decode(
    *,
    context_lengths: tuple[int, ...] = (256, 1024, 4096),
    repeats: int = 5,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Return a tiny CPU timing comparison for kernel vs trie decode (Tq==1).

    Notes / limits:
    - Intended as a quick sanity check, not a rigorous benchmark.
    - Trie mode currently runs on CPU only and is optimized for KV-cache decode.
    """
    import time
    from types import SimpleNamespace

    from nanochat.engine import KVCache

    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    device = torch.device("cpu")
    B = 1
    n_head = 4
    head_dim = 16
    n_embd = n_head * head_dim

    cfg = SimpleNamespace(
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=n_embd,
        ultrametric_K=8,
        ultrametric_p=2,
        ultrametric_alpha=2.0,
        ultrametric_lcp_beta=32.0,
        ultrametric_hard_digits=True,
        ultrametric_mode="kernel",
    )

    results: list[dict[str, float]] = []
    g = torch.Generator(device="cpu").manual_seed(int(seed))

    x = torch.randn((B, 1, n_embd), generator=g, device=device)
    cos = torch.ones((1, 1, head_dim // 2), device=device)
    sin = torch.zeros((1, 1, head_dim // 2), device=device)
    cos_sin = (cos, sin)

    for Tk in context_lengths:
        if Tk < 2:
            raise ValueError("context_lengths must be >= 2 (need a non-empty prefix)")

        k_pref = torch.randn((B, n_head, Tk - 1, head_dim), generator=g, device=device)
        v_pref = torch.randn((B, n_head, Tk - 1, head_dim), generator=g, device=device)

        def _time_one(mode: str) -> float:
            cfg.ultrametric_mode = mode
            torch.manual_seed(int(seed))
            attn = UltrametricCausalSelfAttention(cfg, layer_idx=0).train(False)

            best = float("inf")
            for _ in range(repeats):
                kv = KVCache(batch_size=B, num_heads=n_head, seq_len=Tk, head_dim=head_dim, num_layers=1)
                _ = kv.insert_kv(0, k_pref, v_pref)  # sets pos = Tk-1 (single-layer cache)

                if mode == "trie":
                    state = attn._get_trie_state(kv, B=B, H=n_head, device=device)
                    k_view = kv.kv_cache[0, 0, :, :, : kv.pos]
                    v_view = kv.kv_cache[0, 1, :, :, : kv.pos]
                    attn._update_trie_from_kv(state, k_view, v_view)

                t0 = time.perf_counter()
                with torch.inference_mode():
                    _ = attn(x, cos_sin, kv)
                dt = time.perf_counter() - t0
                best = min(best, dt)
            return float(best)

        kernel_s = _time_one("kernel")
        trie_s = _time_one("trie")
        results.append(
            {
                "Tk": float(Tk),
                "kernel_s": kernel_s,
                "trie_s": trie_s,
                "speedup": (kernel_s / trie_s) if trie_s > 0 else float("inf"),
            }
        )

    return results
