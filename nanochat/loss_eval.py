"""
Validation loss in two units: cross-entropy per token and bits per byte.

Cross-entropy per token depends on the tokenizer: a coarser vocabulary packs
more bytes into each token and reports a HIGHER per-token loss for the same
model quality, so runs trained with different vocabularies (the shared GPT-2
tokenizer vs a task-scoped one, `nanochat.train --tokenizer task`) cannot be
compared on it. Bits per byte divides the summed loss by the number of BYTES
the target tokens spell instead, which is the same quantity for any
tokenizer; nanochat.train reports both (`val_ce_final`, `val_bpb_final`).
"""

import math
from typing import Any, cast

import torch.distributed as torch_dist

from nanochat.torch_imports import torch

dist = cast(Any, torch_dist)


@torch.no_grad()
def evaluate_loss_and_bpb(model, batches, steps, token_bytes) -> tuple[float, float]:
    """Mean cross-entropy per target token and bits per byte over ``steps``
    batches from ``batches`` (an iterator of ``(inputs, targets)``).

    ``token_bytes`` is a 1D int64 tensor of shape ``(vocab_size,)`` holding the
    byte length of each token id, 0 for tokens that must not count (special
    tokens such as ``<|bos|>``; see ``HuggingFaceTokenizer.token_bytes``).
    The model is called with ``loss_reduction="none"`` and must return the
    per-token loss ``(B, T)``.

    Rules, so the two numbers mean what they say:
    1) the cross-entropy is the mean over every target that is not
       ``ignore_index`` (-1): the same population as the model's own mean loss;
    2) bits per byte counts only tokens with a positive byte length - special
       tokens contribute neither nats nor bytes;
    3) ignored targets (-1) contribute nothing to either.
    Both are summed across ranks under torch.distributed, so every rank gets
    the global figures. Returns ``(ce, bpb)``; ``bpb`` is ``inf`` when no byte
    was counted.
    """
    device = model.get_device()
    # [nats over valid targets, valid targets, nats over byte tokens, bytes]
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    batch_iter = iter(batches)
    for _ in range(steps):
        try:
            x, y = next(batch_iter)
        except StopIteration:
            break
        loss2d = model(x, y, loss_reduction="none").view(-1).to(torch.float64)
        y = y.view(-1)
        valid = y >= 0
        # never index token_bytes with ignore_index values
        y_safe = torch.where(valid, y, torch.zeros_like(y))
        num_bytes = torch.where(valid, token_bytes[y_safe], torch.zeros_like(y, dtype=token_bytes.dtype))
        counted = num_bytes > 0
        totals[0] += (loss2d * valid).sum()
        totals[1] += valid.sum()
        totals[2] += (loss2d * counted).sum()
        totals[3] += num_bytes.sum()
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    nats_all, n_valid, nats_bytes, n_bytes = totals.tolist()
    ce = nats_all / n_valid if n_valid > 0 else float("nan")
    bpb = nats_bytes / (math.log(2) * n_bytes) if n_bytes > 0 else float("inf")
    return ce, bpb
