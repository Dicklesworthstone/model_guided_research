"""
Braid Attention Module (PyTorch)
Implements "Braid Attention": permutation + limited crossings with invariant-aware aggregation.

Program model (from JAX reference):
- Score each token (and an 'aggregator' token).
- Output a permutation pi.
- Execute 'braid words' w = sigma_1^k (aggregator swaps with neighbors k times).
- Crossing algebra: (x_a, y_a), (x_b, y_b) -> (x_a + y_b, y_a), (x_b + y_a, y_b).
- This restricted crossing accumulates payloads {y} onto the aggregator {x}.

PyTorch Implementation:
- "Prioritized Accumulation": We simulate the permutation and accumulation process
  using a soft scoring mechanism that respects the additive invariant.
- Instead of hard sorting and scanning (hard to autograd/GPU-optimize in PyTorch without custom kernels),
  we use a "Soft Braid" approximation:
  - Learn priority scores s_i.
  - Probability of crossing (swap) P(i crosses j) ~ Sigmoid(s_i - s_j).
  - Accumulate: x_i += sum_j P(j crosses i) * y_j.

Discrete mode (opt-in):
- Replace sigmoid weights with a hard threshold on the braid score matrix.
- Optionally record a discrete "braid schedule" (a per-head permutation + prefix length)
  for KV-cache decode (Tq==1) and verify local invariants.
- Optional crossing law: 'restricted' (fast, non-YBE) vs 'ybe' (swap-output, YBE-valid).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nanochat.model_utils import norm, apply_rotary_emb, causal_attn_mask, repeat_kv_heads

class BraidCausalSelfAttention(nn.Module):
    _ybe_checked: bool = False

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
        
        # Braid Scoring Network
        # JAX: "sigmoid(w*[tag, value] + b)"
        # Here we learn a scalar score from the head dimension.
        self.braid_score = nn.Linear(self.head_dim, 1, bias=False)

        self.braid_mode = str(getattr(config, "braid_mode", "soft")).strip().lower()
        self.braid_tau = float(getattr(config, "braid_tau", 0.0))
        self.braid_crossing_law = str(getattr(config, "braid_crossing_law", "restricted")).strip().lower()
        self.braid_record_schedule = bool(getattr(config, "braid_record_schedule", False))
        self.braid_verify = bool(getattr(config, "braid_verify", False))
        if self.braid_mode not in {"soft", "discrete"}:
            raise ValueError(f"braid_mode must be 'soft' or 'discrete', got {self.braid_mode!r}")
        if self.braid_crossing_law not in {"restricted", "ybe"}:
            raise ValueError(f"braid_crossing_law must be 'restricted' or 'ybe', got {self.braid_crossing_law!r}")
        self.last_braid_debug: dict[str, object] | None = None

        if self.braid_verify and self.braid_crossing_law == "ybe":
            self._verify_ybe_crossing_law_once()

    @staticmethod
    def _crossing_update_restricted(ax: torch.Tensor, ay: torch.Tensor, bx: torch.Tensor, by: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return ax + by, ay, bx + ay, by

    @staticmethod
    def _crossing_update_ybe(ax: torch.Tensor, ay: torch.Tensor, bx: torch.Tensor, by: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Swap-output version of the restricted crossing law.
        return bx + ay, by, ax + by, ay

    @classmethod
    def _verify_ybe_crossing_law_once(cls) -> None:
        if cls._ybe_checked:
            return
        cls._ybe_checked = True

        # Quick set-theoretic YBE (R3) sanity check on random tensors.
        torch.manual_seed(0)
        n = 64
        d = 8
        ax = torch.randn(n, d)
        ay = torch.randn(n, d)
        bx = torch.randn(n, d)
        by = torch.randn(n, d)
        cx = torch.randn(n, d)
        cy = torch.randn(n, d)

        def apply12(ax, ay, bx, by, cx, cy):
            nax, nay, nbx, nby = cls._crossing_update_ybe(ax, ay, bx, by)
            return nax, nay, nbx, nby, cx, cy

        def apply23(ax, ay, bx, by, cx, cy):
            nbx, nby, ncx, ncy = cls._crossing_update_ybe(bx, by, cx, cy)
            return ax, ay, nbx, nby, ncx, ncy

        lhs = apply12(*apply23(*apply12(ax, ay, bx, by, cx, cy)))
        rhs = apply23(*apply12(*apply23(ax, ay, bx, by, cx, cy)))
        err = torch.max(torch.abs(torch.stack(lhs, dim=-1) - torch.stack(rhs, dim=-1))).item()
        if err > 1e-5:
            raise RuntimeError(f"YBE crossing law check failed: max |lhs-rhs| = {err:.3e}")

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
            
        # Braid Attention Logic
        # 1. Compute priority scores for Q (Aggregator) and K (Tokens).
        # In Braid model, aggregator is just a designated strand.
        # Here, every query i acts as a potential aggregator for its past.
        
        # Score(q): (B, H, Tq, 1)
        s_q = self.braid_score(q)
        # Score(k): (B, H, Tk, 1)
        s_k = self.braid_score(k)
        
        # Crossing Condition: Aggregator i interacts with Token j if i "sorts" past j?
        # Or if they are "compatible".
        # JAX: "Allowed set A = { j : p_j > tau }".
        # This implies interaction depends purely on the token j's score, relative to a threshold.
        # But in Attention, we need Q-dependence.
        # Let's interpret: Q sets the threshold/context?
        # "Score = s_q + s_k" (additive interaction) or "s_q * s_k" (multiplicative).
        # The previous "s_q + s_k.T" gave a pairwise matrix.
        
        scores = s_q + s_k.transpose(-2, -1) # (B, H, Tq, Tk)
        
        # Masking (Causal)
        Tq = q.size(2)
        Tk = k.size(2)
        if kv_cache is None or Tq > 1:
            mask = causal_attn_mask(Tq, Tk, device=q.device)
            scores.masked_fill_(~mask, float("-inf"))
        
        # Interaction Strength
        # Soft mode: sigmoid(scores) in [0,1].
        # Discrete mode: (scores > tau) in {0,1}.
        # Since braid crossing is x += y, we don't normalize to sum=1.
        # We sum raw values weighted by crossing probability (or hard selection).

        # Accumulation: x += sum(p * y)
        # Note: Standard attention is convex combination (sum p = 1).
        # Braid is additive accumulation (sum p can be anything).
        # This can lead to explosion.
        # We add a scaling factor 1/sqrt(T) or similar, or rely on LayerNorm.
        # JAX code uses "MSE(pred_soft - gt)" where GT is sum.
        # So it expects to learn the scale.
        # We'll divide by sqrt(Tk) to keep variance stable at init.

        if self.braid_mode == "soft":
            attn_weights = torch.sigmoid(scores)
        else:
            # Hard gating by threshold: produces a discrete braid schedule (prefix after sort).
            attn_weights = (scores > self.braid_tau).to(dtype=v.dtype)

            if self.braid_record_schedule and Tq == 1:
                score_vec = scores.squeeze(2)  # (B, H, Tk)
                order = torch.argsort(score_vec, dim=-1, descending=True)  # (B, H, Tk)
                selected = score_vec > self.braid_tau  # (B, H, Tk)
                k = selected.sum(dim=-1)  # (B, H)
                self.last_braid_debug = {
                    "tau": float(self.braid_tau),
                    "crossing_law": str(self.braid_crossing_law),
                    "scores": score_vec.detach(),
                    "order": order.detach(),
                    "selected": selected.detach(),
                    "k": k.detach(),
                }

                if self.braid_verify:
                    ordered_scores = score_vec.gather(-1, order)
                    pos = torch.arange(Tk, device=score_vec.device)
                    prefix = pos.view(1, 1, Tk) < k.unsqueeze(-1)
                    # Sorted scores imply a prefix property for threshold selection.
                    ordered_selected = ordered_scores > self.braid_tau
                    if torch.any(ordered_selected != prefix):
                        raise RuntimeError("Discrete braid decode failed prefix verification (threshold/sort mismatch).")

                    # Invariant check: prefix-sum over permuted values equals mask-sum.
                    v_perm = v.gather(2, order.unsqueeze(-1).expand(-1, -1, -1, v.size(-1)))
                    sum_perm = (prefix.to(v.dtype).unsqueeze(-1) * v_perm).sum(dim=2)
                    sum_mask = (selected.to(v.dtype).unsqueeze(-1) * v).sum(dim=2)
                    max_err = torch.max(torch.abs(sum_perm - sum_mask)).item()
                    if max_err > 1e-4:
                        raise RuntimeError(f"Discrete braid invariant check failed: max |perm-sum - mask-sum| = {max_err:.3e}")

        y = attn_weights @ v # [B, H, Tq, D]
        y = y / (Tk ** 0.5 + 1e-6)
        
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
