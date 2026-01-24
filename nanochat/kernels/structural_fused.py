from typing import Any, cast

import triton
import triton.language as tl

from nanochat.torch_imports import torch


@triton.jit
def mix_rows_kernel(
    Mat_ptr,
    stride_r, stride_c,
    idx1, idx2,
    alpha,
    noise_scale,
    seed,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # Pointers to the two rows
    ptr1 = Mat_ptr + idx1 * stride_r + cols * stride_c
    ptr2 = Mat_ptr + idx2 * stride_r + cols * stride_c

    val1 = tl.load(ptr1, mask=mask, other=0.0)
    val2 = tl.load(ptr2, mask=mask, other=0.0)

    # Merge: val1 = alpha * val1 + (1-alpha) * val2
    merged = alpha * val1 + (1.0 - alpha) * val2

    # Clone: val2 = merged + noise
    # Simple uniform noise approximation
    rng_offset = idx2 * n_cols + cols
    r = tl.rand(seed, rng_offset)
    noise = (r - 0.5) * 2.0 * noise_scale

    cloned = merged + noise

    tl.store(ptr1, merged, mask=mask)
    tl.store(ptr2, cloned, mask=mask)

def mix_and_shift_rows(mat, idx1, idx2, alpha, noise_scale):
    """
    Merges row[idx2] into row[idx1] with weight alpha,
    then sets row[idx2] to row[idx1] + noise.
    """
    if not mat.is_cuda:
        # Fallback for CPU
        row1 = mat[idx1].clone()
        row2 = mat[idx2].clone()
        merged = alpha * row1 + (1.0 - alpha) * row2
        mat[idx1] = merged
        mat[idx2] = merged + (torch.rand_like(merged) - 0.5) * 2.0 * noise_scale
        return

    n_rows, n_cols = mat.shape
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_cols, BLOCK_SIZE),)
    seed = torch.randint(0, 2**31, (1,)).item()

    mix_rows_kernel[grid](
        mat,
        mat.stride(0), mat.stride(1),
        idx1, idx2,
        alpha,
        noise_scale,
        seed,
        n_cols,
        BLOCK_SIZE=cast(Any, BLOCK_SIZE)
    )

@triton.jit
def mix_tensors_kernel(
    T1_ptr, T2_ptr,
    alpha,
    noise_scale,
    seed,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    ptr1 = T1_ptr + offsets
    ptr2 = T2_ptr + offsets

    val1 = tl.load(ptr1, mask=mask, other=0.0)
    val2 = tl.load(ptr2, mask=mask, other=0.0)

    merged = alpha * val1 + (1.0 - alpha) * val2

    r = tl.rand(seed, offsets)
    noise = (r - 0.5) * 2.0 * noise_scale

    cloned = merged + noise

    tl.store(ptr1, merged, mask=mask)
    tl.store(ptr2, cloned, mask=mask)

def mix_and_shift_tensors(t1, t2, alpha, noise_scale):
    """
    Merges t2 into t1 with weight alpha,
    then sets t2 to t1 + noise.
    """
    if not t1.is_cuda:
        # Fallback
        merged = alpha * t1 + (1.0 - alpha) * t2
        t1.copy_(merged)
        t2.copy_(merged + (torch.rand_like(merged) - 0.5) * 2.0 * noise_scale)
        return

    n_elements = t1.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    seed = torch.randint(0, 2**31, (1,)).item()

    # Ensure contiguous for simple pointer arithmetic
    # (In-place modification requires we don't change data_ptr, so we assume they are contiguous or handle strides)
    # For simplicity, we assume contiguous or flatten.
    # If not contiguous, we can't easily use single pointer.
    # But model params are usually contiguous.
    if not t1.is_contiguous() or not t2.is_contiguous():
        t1_c = t1.contiguous()
        t2_c = t2.contiguous()
        mix_and_shift_tensors(t1_c, t2_c, alpha, noise_scale)
        t1.copy_(t1_c)
        t2.copy_(t2_c)
        return

    mix_tensors_kernel[grid](
        t1, t2,
        alpha,
        noise_scale,
        seed,
        n_elements,
        BLOCK_SIZE=cast(Any, BLOCK_SIZE)
    )

