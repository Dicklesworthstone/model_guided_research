from typing import Any, cast

import triton
import triton.language as tl

from nanochat.torch_imports import torch


@triton.jit
def router_stats_kernel(
    Gates_ptr,
    Indices_ptr,
    Count_ptr,
    GateSum_ptr,
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
        tl.atomic_add(Count_ptr + idx, 1.0, mask=valid)
        tl.atomic_add(GateSum_ptr + idx, gate, mask=valid)


def accumulate_router_stats(indices, gates, num_experts):
    B, T, K = gates.shape
    device = gates.device
    total_tokens = B * T

    counts = torch.zeros(num_experts, device=device, dtype=torch.float32)
    gate_sums = torch.zeros_like(counts)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_tokens, BLOCK_SIZE),)

    router_stats_kernel[grid](
        gates,
        indices,
        counts,
        gate_sums,
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

    return counts, gate_sums


@triton.jit
def metabolism_kernel(
    Fatigue_ptr,
    Energy_ptr,
    AlphaFat_ptr,
    AlphaEn_ptr,
    Util_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    fatigue = tl.load(Fatigue_ptr + offsets, mask=mask, other=0.0)
    energy = tl.load(Energy_ptr + offsets, mask=mask, other=0.0)
    alpha_f = tl.load(AlphaFat_ptr + offsets, mask=mask, other=0.0)
    alpha_e = tl.load(AlphaEn_ptr + offsets, mask=mask, other=0.0)
    util = tl.load(Util_ptr + offsets, mask=mask, other=0.0)

    fatigue = (1.0 - alpha_f) * fatigue + alpha_f * util
    energy = (1.0 - alpha_e) * energy + alpha_e * (1.0 - util)

    tl.store(Fatigue_ptr + offsets, fatigue, mask=mask)
    tl.store(Energy_ptr + offsets, energy, mask=mask)


def update_metabolism_fused(fatigue, energy, alpha_fatigue, alpha_energy, util):
    N = fatigue.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    metabolism_kernel[grid](
        fatigue,
        energy,
        alpha_fatigue,
        alpha_energy,
        util,
        N,
        BLOCK_SIZE=cast(Any, BLOCK_SIZE),
    )
