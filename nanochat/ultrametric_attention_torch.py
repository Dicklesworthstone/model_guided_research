"""
Ultrametric Attention Module (PyTorch).

Modes (select via GPTConfig.ultrametric_mode):
- ``kernel``: continuous LCP-kernel attention (baseline; supports training/prefill).
- ``trie``: packed prefix-trie lookup for KV-cache decode (currently CPU-only and only used when Tq==1).
- ``balltree``: exact hard-digit ball-tree attention (bead 33dd, the torch port
  of thm-balltree-exact-attention): per depth d, keys and queries become a
  merged event stream sorted by (prefix code, time, kind); a segmented
  inclusive cumsum gives every query the causal sum of its depth-d ball, and
  the telescoped shell decomposition
      out = [S_ge0 + (alpha-1) sum_{d>=1} alpha^(d-1) S_ged] / (same in counts)
  reproduces alpha^lcp attention EXACTLY in O(K * T log T) (vs the kernel's
  O(T^2 K)), batched over (B, H), any device, training/prefill/decode alike.
  Digits are hard by construction (the exactness contract); gradients flow
  through v exactly as in hard-digit kernel mode.
"""

import math
import os
import weakref
from dataclasses import dataclass

import torch
import torch.nn as nn

from nanochat.model_utils import AttentionCore, causal_attn_mask


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


class UltrametricCausalSelfAttention(AttentionCore):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)

        # Ultrametric hyperparameters + mode selection.
        self.K = int(getattr(config, "ultrametric_K", 8))
        self.p = int(getattr(config, "ultrametric_p", 2))
        self.alpha = float(getattr(config, "ultrametric_alpha", 2.0))
        self.lcp_beta = float(getattr(config, "ultrametric_lcp_beta", 32.0))
        # Digit-precision truncation (bead 8gk.4): evaluate with only the
        # first k of the K digit channels - quantization as valuation
        # truncation. The flat-error lemma (thm-flat-error) governs the
        # p-adic-native path: truncation errors of valuation >= k cannot
        # accumulate under the digit machinery's sums/products; the float V
        # aggregation is the archimedean interface where guarantees end.
        # None (the default) leaves every path bitwise unchanged.
        digits_k = getattr(config, "ultrametric_digits_k", None)
        self.digits_k: int | None = None if digits_k is None else int(digits_k)
        if self.digits_k is not None and not (1 <= self.digits_k <= self.K):
            raise ValueError(f"ultrametric_digits_k must be in [1, K={self.K}] or None, got {self.digits_k}")
        mode = (
            str(getattr(config, "ultrametric_mode", os.environ.get("NANOCHAT_ULTRAMETRIC_MODE", "kernel")))
            .strip()
            .lower()
        )
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
        if self.ultrametric_mode not in {"kernel", "trie", "balltree"}:
            raise ValueError(f"ultrametric_mode must be 'kernel', 'trie', or 'balltree', got {self.ultrametric_mode!r}")
        self._log_alpha = math.log(self.alpha)
        if self.ultrametric_mode == "balltree":
            # int64 sort keys: ((p+1)^K * 2T) * 2 must not overflow; guard at
            # init with the rotary-cache bound (10x sequence_len, see gpt.py).
            max_t = int(getattr(config, "sequence_len", 2048)) * 10
            if ((self.p + 1) ** self.K) * 2 * max_t * 2 >= 2**62:
                raise ValueError(
                    f"balltree sort keys overflow int64 for p={self.p}, K={self.K}, T<={max_t}; reduce K or p"
                )

        # Trie cache keyed by KVCache object (decode-only; CPU).
        self._trie_cache: weakref.WeakKeyDictionary[object, _TrieCacheState] = weakref.WeakKeyDictionary()
        # Digits are derived per-head, so the projection is over head_dim.
        self.to_digits_q = nn.Linear(self.head_dim, self.K, bias=False)
        self.to_digits_k = nn.Linear(self.head_dim, self.K, bias=False)
        if self.ultrametric_mode == "balltree" or self.ultrametric_hard_digits:
            # Hard digits sever the graph at the int64 cast, so the q/k path
            # (c_q, c_k, to_digits_q, to_digits_k) is gradient-dead by math:
            # backward returns grad=None and Muon refuses parameters without
            # gradients. Freeze the dead path explicitly (same remedy as the
            # braid rmatrix law's frozen projections); values/c_v/c_proj stay
            # fully trainable.
            self.c_q.weight.requires_grad_(False)
            self.c_k.weight.requires_grad_(False)
            self.to_digits_q.weight.requires_grad_(False)
            self.to_digits_k.weight.requires_grad_(False)

    @property
    def K_eff(self) -> int:
        """Active digit depth: ultrametric_digits_k when set, else K (8gk.4)."""
        return self.digits_k if self.digits_k is not None else self.K

    def _truncate_digits(self, raw: torch.Tensor) -> torch.Tensor:
        """Valuation truncation: keep the first K_eff digit channels. The
        slice is the identity at full precision, so every path is bitwise
        unchanged when the knob is off."""
        return raw if self.digits_k is None else raw[..., : self.digits_k]

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
            [_PackedPrefixTrie(p=self.p, K=self.K_eff, head_dim=self.head_dim, device=device) for _ in range(H)]
            for _ in range(B)
        ]
        state = _TrieCacheState(tries=tries, seen_Tk=0)
        self._trie_cache[kv_cache] = state
        return state

    def _reset_trie_state(self, state: _TrieCacheState) -> None:
        for row in state.tries:
            for trie in row:
                trie.reset()
        state.seen_Tk = 0

    def _update_trie_from_kv(
        self, state: _TrieCacheState, k: torch.Tensor, v: torch.Tensor, *, pos0: int | None = None
    ) -> None:
        Tk = int(k.size(2))
        # A cache rewound past what the trie has ingested (reset() followed by
        # a re-prefill, or a shorter prefill into a reused cache) invalidates
        # every stored key. Comparing Tk alone missed the equal-length case -
        # a new prompt of the same length as the previous one was served the
        # previous prompt's subtree sums - so the cache's own write position
        # (pos0, the keys present BEFORE this forward) is the authority.
        if Tk < state.seen_Tk or (pos0 is not None and pos0 < state.seen_Tk):
            self._reset_trie_state(state)
        if state.seen_Tk >= Tk:
            return

        k_new = k[:, :, state.seen_Tk : Tk]
        v_new = v[:, :, state.seen_Tk : Tk]
        digits = self._digits_hard_int(self._truncate_digits(self.to_digits_k(k_new)))  # (B, H, Tnew, K_eff)
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
        q_digits = self._digits_hard_int(self._truncate_digits(self.to_digits_q(q)))  # (B, H, 1, K_eff)
        B = int(q.size(0))
        H = int(q.size(1))
        y = torch.empty((B, H, self.head_dim), dtype=torch.float32, device=q.device)
        for b in range(B):
            for h in range(H):
                y[b, h] = state.tries[b][h].query(q_digits[b, h, 0], alpha=self.alpha)
        return y.to(dtype=out_dtype).unsqueeze(2)

    def _balltree_attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exact hard-digit alpha^lcp attention in O(K * T log T) (bead 33dd).

        Correctness is thm-balltree-exact-attention's shell partition: balls
        are prefix-code groups, nested by the strong triangle inequality, so
        a per-depth segmented causal cumsum over the merged (key, query) event
        stream gives every query its exact ball sums - numerator and
        normalizer alike. Queries at absolute positions Tk-Tq..Tk-1 (decode
        and chunked prefill included) see exactly their causal prefix: key
        events sort before query events at equal (code, time).
        """
        B, H, Tq, _ = q.shape
        Tk = k.size(2)
        D = v.size(-1)
        G = B * H
        device = q.device
        K_eff = self.K_eff
        q_dig = self._digits_hard_int(self._truncate_digits(self.to_digits_q(q))).reshape(G, Tq, K_eff)
        k_dig = self._digits_hard_int(self._truncate_digits(self.to_digits_k(k))).reshape(G, Tk, K_eff)
        vv = v.reshape(G, Tk, D).to(torch.float32)

        n_ev = Tk + Tq
        k_time = torch.arange(Tk, device=device)
        q_time = (Tk - Tq) + torch.arange(Tq, device=device)  # absolute positions
        ev_time = torch.cat([k_time, q_time])  # (n_ev,)
        # kind tiebreak: keys (0) before queries (1) at equal (code, time)
        ev_kind = torch.cat(
            [torch.zeros(Tk, dtype=torch.int64, device=device), torch.ones(Tq, dtype=torch.int64, device=device)]
        )
        # event weights: zero rows for query events (no masking needed post-sort)
        ev_w = torch.cat([vv, torch.zeros(G, Tq, D, device=device)], dim=1)
        ev_is_key = torch.cat([torch.ones(Tk, device=device), torch.zeros(Tq, device=device)])
        positions = torch.arange(n_ev, device=device).expand(G, -1)

        qc = torch.zeros(G, Tq, dtype=torch.int64, device=device)
        kc = torch.zeros(G, Tk, dtype=torch.int64, device=device)
        S_total = torch.zeros(G, Tq, D, device=device)
        C_total = torch.zeros(G, Tq, device=device)
        time_kind = ev_time * 2 + ev_kind  # constant across depths
        for d in range(K_eff + 1):
            if d > 0:
                # radix prefix codes, +1-shifted so depth-d codes are unique
                qc = qc * (self.p + 1) + q_dig[:, :, d - 1] + 1
                kc = kc * (self.p + 1) + k_dig[:, :, d - 1] + 1
            ev_code = torch.cat([kc, qc], dim=1)  # (G, n_ev)
            order = torch.argsort(ev_code * (2 * n_ev) + time_kind, dim=1)
            inv = torch.empty_like(order)
            inv.scatter_(1, order, positions)
            s_code = torch.gather(ev_code, 1, order)
            s_w = torch.gather(ev_w, 1, order.unsqueeze(-1).expand(-1, -1, D))
            s_cnt = torch.gather(ev_is_key.expand(G, -1), 1, order)
            cw = torch.cumsum(s_w, dim=1)
            cc = torch.cumsum(s_cnt, dim=1)
            # segment bases: cumsum value just before each code group starts
            seg_start = torch.ones(G, n_ev, dtype=torch.bool, device=device)
            seg_start[:, 1:] = s_code[:, 1:] != s_code[:, :-1]
            start_idx = torch.cummax(torch.where(seg_start, positions, torch.zeros_like(positions)), dim=1).values
            gather_idx = (start_idx - 1).clamp(min=0)
            zero_mask = start_idx == 0
            base_w = torch.gather(cw, 1, gather_idx.unsqueeze(-1).expand(-1, -1, D))
            base_c = torch.gather(cc, 1, gather_idx)
            seg_w = cw - base_w.masked_fill(zero_mask.unsqueeze(-1), 0.0)
            seg_c = cc - base_c.masked_fill(zero_mask, 0.0)
            # read each query's running ball sum at its sorted position
            q_pos = inv[:, Tk:]  # (G, Tq)
            S_d = torch.gather(seg_w, 1, q_pos.unsqueeze(-1).expand(-1, -1, D))
            C_d = torch.gather(seg_c, 1, q_pos)
            coeff = 1.0 if d == 0 else (self.alpha - 1.0) * (self.alpha ** (d - 1))
            S_total = S_total + coeff * S_d
            C_total = C_total + coeff * C_d
        out = S_total / C_total.clamp(min=1e-9).unsqueeze(-1)
        return out.reshape(B, H, Tq, D).to(v.dtype)

    def attend(self, q, k, v, *, kv_cache, pos0):
        B = q.size(0)
        Tq = q.size(2)
        Tk = k.size(2)
        mode = str(self.ultrametric_mode).strip().lower()
        if mode == "balltree":
            return self._balltree_attend(q, k, v)
        if mode == "trie" and kv_cache is not None and q.device.type == "cpu":
            # The trie ingests new keys on EVERY cached forward (prefill
            # chunks included) so single-token decode can read it; only the
            # Tq == 1 read path short-circuits the kernel computation.
            state = self._get_trie_state(kv_cache, B=int(B), H=int(k.size(1)), device=q.device)
            self._update_trie_from_kv(state, k, v, pos0=pos0)
            if Tq == 1:
                return self._trie_decode(state, q, out_dtype=v.dtype)  # (B, H, 1, D)

        # LCP-kernel ultrametric attention (continuous relaxation).
        #
        # We map queries/keys to K "digits" in base p, then compute a differentiable proxy
        # for the longest-common-prefix (LCP) depth. Attention weights are derived from
        # the ultrametric similarity kernel w(q,k) ∝ alpha^{LCP(q,k)} (alpha > 1).
        q_dig_raw = self._truncate_digits(self.to_digits_q(q))  # (B, H, Tq, K_eff)
        k_dig_raw = self._truncate_digits(self.to_digits_k(k))  # (B, H, Tk, K_eff)

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

        # Masking is multiplicative (weights live in the kernel's [0, inf)
        # similarity scale, not a log scale), so the default attend's
        # -inf/softmax pipeline does not apply here.
        causal_mask = causal_attn_mask(Tq, Tk, device=q.device)  # (Tq, Tk)

        weights = torch.exp(lcp.to(torch.float32) * self._log_alpha)
        weights = weights.masked_fill(~causal_mask, 0.0)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        attn = (weights / denom).to(dtype=v.dtype)

        return attn @ v


def bench_ultrametric_paths(
    *,
    context_lengths: tuple[int, ...] = (1024, 4096),
    repeats: int = 3,
    seed: int = 0,
) -> dict:
    """Wall-clock comparison of the ultrametric paths (bead 33dd, CPU).

    - forward: kernel (O(T^2 K)) vs balltree (O(K T log T)) full prefill at T.
    - decode: kernel vs trie per-token decode (the existing sanity probe).

    Identical projections (same seed/init) and hard digits in both arms: the
    paths compute the same function (thm-balltree-exact-attention), so the
    quality side is a correctness check and wall-clock is the entire claim.
    Best-of-`repeats` per point. This feeds `mgr bench-ultrametric`'s
    versioned artifact (mgr.bench.ultrametric_paths.v1).
    """
    import time
    from types import SimpleNamespace

    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    device = torch.device("cpu")
    B, n_head, head_dim = 1, 4, 16
    n_embd = n_head * head_dim

    forward_rows: list[dict[str, float]] = []
    for T in context_lengths:
        cfg = SimpleNamespace(
            n_head=n_head,
            n_kv_head=n_head,
            n_embd=n_embd,
            sequence_len=max(int(T) // 10 + 1, 64),  # balltree overflow guard reads 10x this
            ultrametric_K=8,
            ultrametric_p=2,
            ultrametric_alpha=2.0,
            ultrametric_lcp_beta=32.0,
            ultrametric_hard_digits=True,
            ultrametric_mode="kernel",
        )
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        x = torch.randn((B, int(T), n_embd), generator=g, device=device)
        cos = torch.ones((1, int(T), 1, head_dim // 2), device=device)
        sin = torch.zeros((1, int(T), 1, head_dim // 2), device=device)

        def _time_forward(mode: str, T=T, cfg=cfg, x=x, cos=cos, sin=sin) -> float:
            cfg.ultrametric_mode = mode
            torch.manual_seed(int(seed))
            attn = UltrametricCausalSelfAttention(cfg, layer_idx=0).train(False)
            best = float("inf")
            for _ in range(repeats):
                t0 = time.perf_counter()
                with torch.inference_mode():
                    _ = attn(x, (cos, sin), None)
                best = min(best, time.perf_counter() - t0)
            return float(best)

        kernel_s = _time_forward("kernel")
        balltree_s = _time_forward("balltree")
        forward_rows.append(
            {
                "T": float(T),
                "kernel_s": kernel_s,
                "balltree_s": balltree_s,
                "speedup": (kernel_s / balltree_s) if balltree_s > 0 else float("inf"),
            }
        )

    decode_rows = perf_sanity_ultrametric_trie_decode(
        context_lengths=tuple(int(t) for t in context_lengths), repeats=repeats, seed=seed
    )
    out: dict = {"forward": forward_rows, "decode": decode_rows}
    for row in forward_rows:
        out[f"speedup_forward_t{int(row['T'])}"] = row["speedup"]
    for row in decode_rows:
        out[f"speedup_decode_t{int(row['Tk'])}"] = row["speedup"]
    return out


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
                    cache = kv.kv_cache
                    if cache is None:
                        raise RuntimeError("KVCache insert_kv did not initialize the cache tensor")
                    k_view = cache[0, 0, :, :, : kv.pos]
                    v_view = cache[0, 1, :, :, : kv.pos]
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
