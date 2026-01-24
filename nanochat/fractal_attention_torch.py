"""
Fractal Memory / Attention Module (PyTorch)
Implements attention over a "Fractal" memory structure (IFS).

Mathematical core (from JAX reference):
  - Values live as fixed points x*_w of composed contractions F_w.
  - Keys are depth-k paths w = (i_1, ..., i_k).
  - c_w = sum_{j=1..k} A^{k-j} t_{j, i_j}.
  - Read v_hat = x*_w = (c_w + u_w) / (1 - s^k).

PyTorch Implementation:
  - We simulate the "Path Matching" aspect via Hierarchical Soft-Routing.
  - Q defines a target path. K defines a storage path.
  - Attention weight ~ Probability(Path(Q) == Path(K)).
"""

import torch.nn as nn
import torch.nn.functional as F

from nanochat.model_utils import apply_rotary_emb, causal_attn_mask, norm, repeat_kv_heads


class FractalCausalSelfAttention(nn.Module):
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

        # IFS Router
        # "A learned router (k independent m-way classifiers) maps query q->w"
        self.m = 4  # Branching factor
        self.depth = 4  # Depth of IFS
        self.router = nn.Linear(self.head_dim, self.depth * self.m, bias=False)

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
            # Ensure K/V head count matches Q before routing (router assumes n_head).
            k, v = repeat_kv_heads(k, v, n_head=self.n_head)

        # Fractal Addressing / Similarity
        # We compute the "Soft Path" for Q and K.
        # Path is a sequence of categorical distributions over m branches at each depth d.
        # q_route: (B, H, Tq, Depth, m)
        q_route = self.router(q).view(B, self.n_head, -1, self.depth, self.m)
        k_route = self.router(k).view(B, self.n_head, -1, self.depth, self.m)

        # Softmax over m to get branch probabilities
        q_prob = F.softmax(q_route, dim=-1)
        k_prob = F.softmax(k_route, dim=-1)

        # Similarity = Probability that Q and K took the SAME path.
        # P(overlap) = prod_{d=1..Depth} (sum_{i=1..m} q_{d,i} * k_{d,i})
        # Log-Sim = sum_{d} log(q_d . k_d)

        # Implementation:
        # 1. Flatten (Depth, m) -> (Depth*m).
        # 2. This is just creating a "Fractal Embedding" of size Depth*m.
        # 3. Standard dot product approximates the similarity if normalized correctly.

        # Better approximation:
        # Compute per-depth overlap, then sum/prod.
        # q_prob: (..., D, m)
        # Overlap at depth d: O_d = (q_d * k_d).sum(m)
        # Total Sim = prod(O_d) or sum(O_d) or sum(log O_d).
        # We use dot product of flattened probs as a proxy for "Total path overlap mass".
        # Score = (Q_flat . K_flat) / sqrt(Depth)

        q_flat = q_prob.view(B, self.n_head, -1, self.depth * self.m)
        k_flat = k_prob.view(B, self.n_head, -1, self.depth * self.m)

        # We treat these probability vectors as the keys/queries for attention
        scores = (q_flat @ k_flat.transpose(-2, -1)) * (1.0 / (self.depth**0.5))

        # Masking
        Tq = q.size(2)
        Tk = k.size(2)
        if kv_cache is None or Tq > 1:
            mask = causal_attn_mask(Tq, Tk, device=q.device)
            scores.masked_fill_(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        y = attn_weights @ v

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
