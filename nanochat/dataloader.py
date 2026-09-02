import threading
from collections import deque

import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer
from nanochat.torch_imports import torch


def tokenizing_distributed_data_loader_with_state(
    B,
    T,
    split,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
    device="cuda",
    resume_state_dict=None,
    data_dir=None,
    prefetch_chunks=0,
    tokenizer=None,
    shuffle_seed=None,
):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    This implementation became a bit more complex because we wish to support approximate resume training.
    Instead of turning this into a Class, we opt to return the state_dict with every batch,
    and then the caller can pass in a state_dict to resume training from a desired point.
    Note that this resumption is atm only *approximate* for simplicity.
    We won't repeat the same documents but we might skip a few.
    The state_dict that is returned can be later passed into this function via `resume_state_dict` to approximately resume.

    Perfect state resumption is possible but would be a lot more bloated, probably not worth it atm.

    Prefetch (bead atkp): with ``prefetch_chunks > 0`` a daemon thread runs the
    document iterator + tokenizer encode ahead of the consumer, hiding the
    ~200ms synchronous refill stalls behind training steps. The thread only
    ever APPENDS fully-tokenized (chunk, position) items to a bounded queue;
    the recorded state is the position of the last chunk POURED into the
    yielded batch — identical to the synchronous path's semantics, so exact
    resume replay stays correct. Default 0 = the original synchronous loop,
    byte-for-byte unchanged.

    Shuffle (bead r7qn): with ``shuffle_seed`` set, the documents of every
    parquet row group are visited in a permutation seeded by
    ``(shuffle_seed, epoch, pq_idx, rg_idx)``, so each epoch presents the
    corpus in a different order. Without it a small corpus replays in the SAME
    order every epoch, and a multi-epoch run learns which document follows
    which instead of the task: the 1e12 copyops probe reached train loss 0.9
    on the stream in corpus order and 4.5 on the same documents shuffled. The
    permutation is a pure function of the state the loader already records
    (plus ``epoch``, now part of the state dict), so exact and approximate
    resume replay it identically. ``None`` = file order (the FineWeb stream).
    """
    if split not in ["train", "val"]:
        raise ValueError("split must be 'train' or 'val'")

    # infinite iterator over document batches (list of text strings)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    def document_batches():
        # data_dir=None -> the FineWeb cache; a path -> any parquet corpus
        # following the same convention (sorted; LAST file is the val split),
        # e.g. an mgr gen-tasks output directory (bead kbj2).
        parquet_paths = list_parquet_files(data_dir)
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        epoch = int(resume_state_dict.get("epoch", 0)) if resume_state_dict is not None else 0
        pq_idx = resume_pq_idx  # we kick off parquet files at the resume index (or by default just 0)
        while True:  # iterate infinitely (multi-epoch)
            if not parquet_paths:
                raise RuntimeError("No parquet files found for split: " + split)
            while pq_idx < len(parquet_paths):  # iterate over all parquet files
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                # Start from resume point if resuming on same file, otherwise from DDP rank
                # I know this state resumption is a little bit tricky and a little bit hacky... sigh.
                if resume_rg_idx is not None:
                    base_idx = resume_rg_idx // ddp_world_size  # in units of ddp_world_size
                    base_idx += 1  # advance by 1 so that we definitely don't repeat data after resuming
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    resume_rg_idx = None  # set to None as we only want to do this a single time
                else:
                    rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column("text").to_pylist()  # each batch is a parquet group, e.g. 1024 rows
                    if shuffle_seed is not None and len(batch) > 1:
                        # per-epoch permutation of this row group's documents,
                        # a pure function of the recorded position (see docstring)
                        gen = torch.Generator().manual_seed(
                            (int(shuffle_seed) * 1_000_003 + epoch * 10_007 + pq_idx * 101 + rg_idx) % (2**63 - 1)
                        )
                        order = torch.randperm(len(batch), generator=gen).tolist()
                        batch = [batch[j] for j in order]
                    # the tokenizer encode might want to go in even smaller batches, e.g. 128 rows
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i : i + tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                    rg_idx += ddp_world_size  # advance to the next row group (in DDP)
                pq_idx += 1  # advance to the next parquet file
            # Reset for next epoch
            pq_idx = 0
            rg_idx = ddp_rank  # Reset row group index for new epoch
            epoch += 1

    batches = document_batches()

    # Now emit batches of tokens.
    needed_tokens = B * T + 1  # +1 is because we also need the target at the last token
    # the tokenizer (a task-scoped one when the trainer built one) and its bos token
    tokenizer = tokenizer if tokenizer is not None else get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one iteration
    token_buffer = deque()  # we stream tokens on the right and pop from the left

    def _emit(tokens, state_dict):
        # CUDA supports memory pinning for asynchronous transfers between CPU and GPU
        device_type = torch.device(device).type
        use_cuda_optimizations = device_type == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)  # long=int64
        inputs_cpu = scratch[:-1]  # drop the last token: it is only the target of the previous step
        targets_cpu = scratch[1:]
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        return inputs, targets, state_dict

    if prefetch_chunks and prefetch_chunks > 0:
        # Prefetch mode (bead atkp): a daemon thread tokenizes ahead of the
        # consumer into a bounded pending-queue of (token_list, position)
        # chunks. The recorded state is the position of the last chunk POURED
        # into the yielded batch — consumer-accurate, so exact-resume replay
        # semantics are identical to the synchronous path.
        buf_cond = threading.Condition()
        pending: deque = deque()  # (token_list, (pq_idx, rg_idx, epoch)) not yet poured
        latest_pos = {"pq_idx": 0, "rg_idx": 0, "epoch": 0}
        stop = threading.Event()  # set when the consumer generator is closed
        failure: list[BaseException] = []  # the producer's exception, if it died

        def _refill():
            # Any error in the producer (unreadable shard, tokenizer failure)
            # is handed to the consumer, which re-raises it at its next pour.
            # A silently dead producer left the consumer spinning forever on an
            # empty queue with no message - a training run that never stepped
            # and never failed.
            try:
                while not stop.is_set():
                    doc_batch, pos = next(batches)
                    token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
                    with buf_cond:
                        while len(pending) >= prefetch_chunks and not stop.is_set():
                            buf_cond.wait(timeout=0.05)
                        for tl in token_lists:
                            pending.append((tl, pos))
                        buf_cond.notify_all()
            except Exception as exc:  # noqa: BLE001 - deliberately forwarded, never swallowed
                with buf_cond:
                    failure.append(exc)
                    buf_cond.notify_all()

        threading.Thread(target=_refill, daemon=True, name="dataloader-prefetch").start()

        try:
            while True:
                with buf_cond:
                    while len(token_buffer) < needed_tokens:
                        if pending:
                            tl, pos = pending.popleft()
                            token_buffer.extend(tl)
                            latest_pos["pq_idx"], latest_pos["rg_idx"], latest_pos["epoch"] = pos
                            buf_cond.notify_all()
                        elif failure:
                            raise RuntimeError("dataloader prefetch thread failed while tokenizing") from failure[0]
                        else:
                            buf_cond.wait(timeout=0.02)
                    tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
                    state_dict = dict(latest_pos)
                    buf_cond.notify_all()
                yield _emit(tokens, state_dict)
        finally:
            # Generator closed (resume rebuilt the loader, training ended):
            # release the producer instead of leaking a thread that tokenizes
            # ahead into a queue nobody drains.
            stop.set()
            with buf_cond:
                buf_cond.notify_all()
    else:
        while True:
            # Accumulate enough tokens for one iteration before yielding.
            while len(token_buffer) < needed_tokens:
                doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
                token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
                for tokens in token_lists:
                    token_buffer.extend(tokens)
            # Move tokens from the deque into the scratch buffer
            tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
            state_dict = {
                "pq_idx": pq_idx,
                "rg_idx": rg_idx,
                "epoch": epoch,
            }  # we need this in case we wish to approximately resume training
            yield _emit(tokens, state_dict)


def tokenizing_distributed_data_loader(*args, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
