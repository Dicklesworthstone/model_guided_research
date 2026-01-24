from typing import Any, cast

import triton
import triton.language as tl

from nanochat.torch_imports import torch


@triton.jit
def metrics_scatter_add_kernel(
    Gates_ptr,
    Indices_ptr,
    Usage_ptr,
    Contrib_ptr,
    stride_g_b,
    stride_g_t,
    stride_g_k,
    stride_i_b,
    stride_i_t,
    stride_i_k,
    total_tokens,
    T,
    K,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_tokens
    t_idx = offsets % T
    b_idx = offsets // T

    base_g = Gates_ptr + b_idx * stride_g_b + t_idx * stride_g_t
    base_i = Indices_ptr + b_idx * stride_i_b + t_idx * stride_i_t

    for k in range(K):
        gate = tl.load(base_g + k * stride_g_k, mask=mask, other=0.0)
        idx = tl.load(base_i + k * stride_i_k, mask=mask, other=-1).to(tl.int32)
        valid = mask & (idx >= 0)
        tl.atomic_add(Usage_ptr + idx, 1.0, mask=valid)
        tl.atomic_add(Contrib_ptr + idx, gate, mask=valid)


def _ensure_gpu_stat(state, key, device):
    gpu_key = f"{key}_gpu"
    if gpu_key not in state:
        state[gpu_key] = state[key].to(device)
    elif state[gpu_key].device != device:
        state[gpu_key] = state[gpu_key].to(device)
    if state[key].device != torch.device("cpu"):
        state[key] = state[key].cpu()
    return state[gpu_key]


def _sync_back(state, key):
    gpu_key = f"{key}_gpu"
    if gpu_key in state:
        state[key].copy_(state[gpu_key].detach().cpu())


def update_metrics_fused(indices, gates, energy, state, cfg):
    if not gates.is_cuda:
        return False

    device = gates.device
    B, T, K = gates.shape
    energy.shape[0]
    total_tokens = B * T

    loss_contrib = _ensure_gpu_stat(state, "loss_contrib", device)
    routing_freq = _ensure_gpu_stat(state, "routing_freq", device)
    resilience = _ensure_gpu_stat(state, "resilience", device)
    prev_contrib = _ensure_gpu_stat(state, "prev_contrib", device)
    efficiency = _ensure_gpu_stat(state, "efficiency", device)

    contrib_step = torch.zeros_like(loss_contrib)
    freq_step = torch.zeros_like(loss_contrib)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_tokens, BLOCK_SIZE),)

    metrics_scatter_add_kernel[grid](
        gates,
        indices,
        freq_step,
        contrib_step,
        gates.stride(0),
        gates.stride(1),
        gates.stride(2),
        indices.stride(0),
        indices.stride(1),
        indices.stride(2),
        total_tokens,
        T,
        K,
        BLOCK_SIZE=cast(Any, BLOCK_SIZE),
    )

    decay = cfg.decay
    norm = float(total_tokens)
    freq_step /= norm
    contrib_step /= norm

    routing_freq.mul_(decay).add_(freq_step * (1 - decay))
    loss_contrib.mul_(decay).add_(contrib_step * (1 - decay))

    energy_gpu = energy.to(device, dtype=loss_contrib.dtype, copy=False)
    efficiency.copy_(loss_contrib / (energy_gpu + 1e-6))

    diff = (loss_contrib - prev_contrib).abs()
    resilience.mul_(decay).add_((1.0 / (diff + 1e-6)) * (1 - decay))
    prev_contrib.copy_(loss_contrib)

    _sync_back(state, "loss_contrib")
    _sync_back(state, "routing_freq")
    _sync_back(state, "efficiency")
    _sync_back(state, "resilience")
    _sync_back(state, "prev_contrib")

    return True

