"""Tests for the dataloader's opt-in prefetch thread (bead atkp).

Contract: with prefetch enabled the yielded token batches are IDENTICAL to
the synchronous loader's (same corpus, same iteration order), and the
recorded resume state stays consumer-accurate — the position of the last
chunk actually poured into the batch, never the producer's run-ahead
position — so exact-resume replay semantics are unchanged.
"""

import pytest

from nanochat.dataloader import tokenizing_distributed_data_loader_with_state

N_DOCS = 48
DOC = "The quick brown fox jumps over the lazy dog again and again. "


def _make_corpus(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    tmp_path.mkdir(parents=True, exist_ok=True)
    texts = [DOC * (1 + i % 3) for i in range(N_DOCS)]
    table = pa.table({"text": pa.array(texts)})
    for name in ("shard_a.parquet", "shard_b.parquet"):
        pq.write_table(table, tmp_path / name)
    return tmp_path


def _collect(data_dir, batches, **kw):
    out = []
    loader = tokenizing_distributed_data_loader_with_state(
        B=2, T=8, split="train", device="cpu", data_dir=str(data_dir), **kw
    )
    for _ in range(batches):
        inputs, targets, state = next(loader)
        assert inputs.shape == (2, 8) and targets.shape == (2, 8)
        assert bool((inputs[:, 1:] == targets[:, :-1]).all()), "targets must be inputs shifted by one"
        out.append((inputs.clone(), state["pq_idx"], state["rg_idx"]))
    return out


@pytest.mark.parametrize("prefetch_chunks", [0, 4])
def test_prefetch_matches_sync_token_stream(tmp_path, prefetch_chunks):
    """Both modes yield byte-identical token batches: collect under the given
    mode and compare against an independent synchronous reference."""
    data_dir = _make_corpus(tmp_path / f"corpus_{prefetch_chunks}")
    ref_dir = _make_corpus(tmp_path / f"ref_{prefetch_chunks}")
    got = _collect(data_dir, 6, prefetch_chunks=prefetch_chunks)
    ref = _collect(ref_dir, 6, prefetch_chunks=0)
    for (inp_a, _, _), (inp_b, _, _) in zip(got, ref, strict=True):
        assert bool((inp_a == inp_b).all())


def test_prefetch_state_is_consumer_accurate(tmp_path):
    """Recorded positions equal the synchronous path's at every yield: the
    producer's run-ahead position must NOT leak into the resume state."""
    data_dir = _make_corpus(tmp_path / "state_corpus")
    sync = _collect(data_dir, 6, prefetch_chunks=0)
    pre = _collect(data_dir, 6, prefetch_chunks=4)
    for (_, pq_s, rg_s), (_, pq_p, rg_p) in zip(sync, pre, strict=True):
        assert (pq_p, rg_p) == (pq_s, rg_s), "resume state drifted ahead of consumption"


def test_prefetch_states_are_monotonic(tmp_path):
    """Resume positions never move backwards across yields."""
    data_dir = _make_corpus(tmp_path / "mono_corpus")
    batches = _collect(data_dir, 6, prefetch_chunks=4)
    keys = [(pq, rg) for _, pq, rg in batches]
    assert keys == sorted(keys)
