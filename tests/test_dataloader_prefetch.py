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


def _next_with_deadline(loader, seconds: float):
    """Pull one batch on a helper thread; returns ("ok", batch) | ("error", exc) | ("hang", None)."""
    import threading

    box: dict[str, object] = {}

    def run():
        try:
            box["value"] = next(loader)
        except BaseException as exc:  # noqa: BLE001 - the test inspects whatever escaped
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=seconds)
    if worker.is_alive():
        return "hang", None
    if "error" in box:
        return "error", box["error"]
    return "ok", box["value"]


def test_prefetch_producer_failure_reaches_the_consumer(tmp_path):
    """An unreadable shard must surface as an exception at the consumer within
    seconds. Before the fix the producer thread died silently and the consumer
    spun forever on an empty queue: a training run that never stepped and
    never failed."""
    data_dir = tmp_path / "broken_corpus"
    data_dir.mkdir()
    (data_dir / "shard_a.parquet").write_bytes(b"this is not a parquet file")  # train split
    _make_corpus(tmp_path / "good")
    (data_dir / "shard_z.parquet").write_bytes((tmp_path / "good" / "shard_a.parquet").read_bytes())  # val
    loader = tokenizing_distributed_data_loader_with_state(
        B=2, T=8, split="train", device="cpu", data_dir=str(data_dir), prefetch_chunks=4
    )
    status, payload = _next_with_deadline(loader, seconds=30.0)
    assert status == "error", f"consumer must raise, got {status}"
    assert isinstance(payload, RuntimeError) and "prefetch thread failed" in str(payload)
    assert payload.__cause__ is not None, "the producer's original exception must be chained"


def test_prefetch_thread_stops_when_the_generator_is_closed(tmp_path):
    """Closing the consumer generator releases the producer thread instead of
    leaking a daemon that tokenizes ahead into a queue nobody drains."""
    import threading
    import time

    data_dir = _make_corpus(tmp_path / "close_corpus")
    before = {t.name for t in threading.enumerate()}
    loader = tokenizing_distributed_data_loader_with_state(
        B=2, T=8, split="train", device="cpu", data_dir=str(data_dir), prefetch_chunks=2
    )
    next(loader)
    started = [t for t in threading.enumerate() if t.name == "dataloader-prefetch" and t.name not in before]
    assert started, "prefetch thread should be running after the first batch"
    loader.close()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and any(t.is_alive() for t in started):
        time.sleep(0.05)
    assert not any(t.is_alive() for t in started), "prefetch thread must exit after the generator is closed"


def test_shuffle_seed_permutes_documents_per_epoch_deterministically(tmp_path):
    """bead r7qn: with ``shuffle_seed`` the documents of each row group are
    visited in a seeded per-epoch permutation - the same stream on every
    instantiation (resume-safe), different from file order, and the recorded
    state carries the epoch the permutation depends on."""
    data_dir = _make_corpus(tmp_path / "shuffle_corpus")

    def stream(n, **kw):
        loader = tokenizing_distributed_data_loader_with_state(
            B=2, T=8, split="train", device="cpu", data_dir=str(data_dir), **kw
        )
        out = []
        for _ in range(n):
            inputs, _targets, state = next(loader)
            out.append((inputs.clone(), dict(state)))
        return out

    plain = stream(160)
    shuffled = stream(160, shuffle_seed=3)
    again = stream(160, shuffle_seed=3)
    other_seed = stream(160, shuffle_seed=4)
    assert all(bool((a == b).all()) for (a, _), (b, _) in zip(shuffled, again, strict=True)), "not deterministic"
    assert any(not bool((a == b).all()) for (a, _), (b, _) in zip(plain, shuffled, strict=True)), "file order kept"
    assert any(not bool((a == b).all()) for (a, _), (b, _) in zip(shuffled, other_seed, strict=True))
    assert all(state["epoch"] == 0 for _, state in plain[:5])
    # the train split is 48 short documents (~1500 tokens): 160 batches of 16
    # tokens cross into the next epoch, and the state must say so (the
    # permutation is keyed on it)
    assert any(state["epoch"] >= 1 for _, state in shuffled)
    for (_, s_prev), (_, s_next) in zip(shuffled, shuffled[1:], strict=False):
        assert s_next["epoch"] >= s_prev["epoch"], "epoch counter must be monotonic"
