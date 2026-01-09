
import torch
import torch.nn.functional as F

def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))

def causal_attn_mask(Tq: int, Tk: int, *, device: torch.device) -> torch.Tensor:
    """
    Build a boolean attention mask of shape (Tq, Tk) where True means "keep".

    Matches the KV-cache semantics used in `nanochat.gpt.CausalSelfAttention`:
    - Training / prefill (Tk == Tq): standard causal (lower-triangular) mask
    - Decode (Tq == 1): allow attending to all cached keys (no masking)
    - Chunked decode (Tk > Tq > 1): allow full prefix + causal within the chunk
    """
    if Tq <= 0 or Tk <= 0:
        raise ValueError("Tq and Tk must be positive")
    if Tq == 1:
        return torch.ones((Tq, Tk), dtype=torch.bool, device=device)
    if Tk == Tq:
        return torch.tril(torch.ones((Tq, Tk), dtype=torch.bool, device=device))
    if Tk < Tq:
        raise ValueError(f"Expected Tk >= Tq for causal attention, got Tk={Tk}, Tq={Tq}")
    prefix_len = Tk - Tq
    mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=device)
    if prefix_len > 0:
        mask[:, :prefix_len] = True
    mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=device))
    return mask

def repeat_kv_heads(k: torch.Tensor, v: torch.Tensor, *, n_head: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat KV heads for Group-Query Attention (GQA) to match `n_head` query heads."""
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("repeat_kv_heads expects k/v of shape (B, H, T, D)")
    if k.shape != v.shape:
        raise ValueError(f"repeat_kv_heads expects k and v to have the same shape, got k={k.shape}, v={v.shape}")
    n_kv_head = k.size(1)
    if n_kv_head == n_head:
        return k, v
    if n_kv_head <= 0 or n_head <= 0:
        raise ValueError("n_head and n_kv_head must be positive")
    if n_head % n_kv_head != 0:
        raise ValueError(f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})")
    repeat = n_head // n_kv_head
    return k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)

def apply_rotary_emb(x, cos, sin):
    if x.ndim != 4:
        raise ValueError("apply_rotary_emb expects tensor of shape (B, T, H, D)")
    if x.shape[3] % 2 != 0:
        raise ValueError("apply_rotary_emb requires an even head dimension D (pairs of channels)")
    d = x.shape[3] // 2
    if cos.shape[-1] != d or sin.shape[-1] != d:
        raise ValueError(
            f"apply_rotary_emb expects cos/sin last dim == D/2 ({d}), got cos={cos.shape}, sin={sin.shape}"
        )
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out
