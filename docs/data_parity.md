# Tokenizer / Dataset / Dataloader Parity vs `bio_inspired_nanochat`

**Bead:** `model_guided_research-6fl`
**Audit date:** 2026-06-14
**Compared trees:**
- `model_guided_research` @ `main` (this repo)
- `bio_inspired_nanochat` @ `e09ad46` (`/data/projects/bio_inspired_nanochat`)

**Bottom line for benchmarking:** the two data stacks are forks of the same
karpathy-`nanochat` base and produce **byte-identical token streams and
batch tensors** given the same dataset, vocab size, `B`, and `T`. There is
**no padding, no attention masking, and no per-document boundary reset in
either repo** — both pack a contiguous BOS-separated token stream into `B×T`
windows. The differences are all operational (download integrity, resume
helpers, smoke-run conventions, defensive error handling) and **do not
affect a fair fixed-FLOPs comparison** as long as both sides use ≥2 shards
of the same `fineweb-edu-100b-shuffle` corpus and the same `(B, T)`.

The only items a benchmark author must consciously match are listed in
[§5 Action items](#5-action-items-for-fair-benchmarks).

---

## 1. Tokenizer — semantic parity, ours is hardened

Both files implement the same GPT-4-style BPE tokenizer.

| Aspect | This repo (`nanochat/tokenizer.py`) | bio_inspired (`tokenizer.py`) | Verdict |
|---|---|---|---|
| Split regex | `SPLIT_PATTERN` (`:32`) | identical (`:46`) | ✅ identical |
| Special tokens | `SPECIAL_TOKENS` (`:15`) — bos/user/assistant/python/output | identical list | ✅ identical |
| BPE model | `BPE(byte_fallback=True, unk_token=None, fuse_unk=False)` | identical | ✅ identical |
| Pre-tokenizer | `Split(isolated)` → `ByteLevel(add_prefix_space=False, use_regex=False)` | identical | ✅ identical |
| Vocab accounting | `vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)`, min 256 (`:218`) | identical | ✅ identical |
| Training backend | Explicit choice: optional `rustbpe` extra or core HF `tokenizers` | identical implementations | ✅ identical |

A whitespace-insensitive diff is **199 lines**, but every substantive hunk is
defensive hardening that this repo added on top of the shared base; none
changes the produced token ids:

- `encode_special()` raises `ValueError` on an unknown special token
  (`nanochat/tokenizer.py` `encode_special`), where bio returns `None`
  silently (`tokenizer.py:` `token_to_id` passthrough). Same ids for known
  tokens; ours just fails loudly on a typo.
- `get_bos_token_id()` prefers `<|bos|>` but **falls back to
  `<|endoftext|>`** if absent; bio assumes `<|bos|>` exists. Same id when
  `<|bos|>` is present (the standard case).
- `encode()` input guard raises `TypeError` for non-`str` input via the
  `_ensure(...)` helper; bio raises `TypeError` inline. Equivalent.
- Import ordering / `Sequence([...])` formatting differences are pure ruff
  cosmetics.

**Conclusion:** given the same trained vocab artifact (or the same
`vocab_size` re-trained on the same shards), the two tokenizers emit
identical token sequences. Token parity holds.

> Caveat worth checking before a cross-repo run: confirm both sides load the
> *same* tokenizer artifact (`get_tokenizer()` resolves a cached vocab). If
> one repo re-trains BPE on a different shard sample, the vocabs diverge even
> though the *code* is identical. For in-repo A/B (the actual use case), this
> is a non-issue — one tokenizer is shared across all attention mechanisms.

---

## 2. Dataloader — effectively identical

`nanochat/dataloader.py:11` vs `bio_inspired_nanochat/dataloader.py:10`:
`tokenizing_distributed_data_loader_with_state(B, T, split, …)`.

Both implement the **exact same algorithm**, line-for-line:

1. `document_batches()` streams parquet **row groups** (≈1024 rows each),
   DDP-sharded by `rg_idx += ddp_world_size` starting at `ddp_rank`
   (`dataloader.py:56–62` here / `:47–53` bio).
2. Documents are tokenized in `tokenizer_batch_size=128` chunks with
   `prepend=bos_token` and `num_threads=tokenizer_threads=4`
   (`:81` / `:71`).
3. A `deque` token buffer accumulates `needed_tokens = B*T + 1` tokens, then
   pops them into a `torch.long` scratch tensor (`:71,85` / `:61,75`).
4. `inputs = scratch[:-1].view(B, T)`, `targets = scratch[1:].view(B, T)` —
   **next-token targets, contiguous packing, no padding** (`:91–95` / `:81–85`).
5. `pin_memory` + `non_blocking` transfer enabled iff `device.type == "cuda"`
   (`:87–95` / `:76–85`) — identical CUDA-path behaviour; both are CPU-safe.
6. Approximate resume via a returned `state_dict={"pq_idx","rg_idx"}`
   (`:96–100` / `:86`).

**The single code difference:** this repo's loader takes an extra
`data_dir=None` argument (`nanochat/dataloader.py:12`) so it can stream **any**
parquet corpus that follows the FineWeb convention (e.g. an `mgr gen-tasks`
output dir, bead `kbj2`), whereas bio hardcodes the FineWeb cache via
`parquet_paths_for_split(split)`. With `data_dir=None` the two are identical.

This repo additionally exposes an **exact** (bit-identical) resume mode in
`train.py:1010` that replays the deterministic stream and discards already-
consumed batches — bio only has the approximate resume. Exact resume is a
superset; it does not change the token stream a fresh run sees.

---

## 3. Dataset layer — same corpus, ours adds a programmatic guard, theirs adds integrity checks

Both target the **same corpus**: `karpathy/fineweb-edu-100b-shuffle`,
`MAX_SHARD = 1822`, `shard_{i:05d}.parquet`, `text` column, and the
**"last sorted shard is the val split"** convention
(`nanochat/dataset.py:26,36,45,59` vs `bio_inspired_nanochat/dataset.py:35,36,49,68`).

| Capability | This repo | bio_inspired |
|---|---|---|
| `list_parquet_files(data_dir=None)` | ✅ `dataset.py:42` | ✅ `dataset.py:46` |
| Split selection | inline `[:-1]`/`[-1:]` in loader (`dataloader.py:37`) | `parquet_paths_for_split()` helper (`dataset.py:57`) |
| **1-shard smoke convention** | ❌ requires ≥2 shards (raises) | ✅ 1 shard used for **both** train & val (`dataset.py:66–68`) |
| Programmatic min-shard ensure | ✅ `ensure_min_parquet_files()` + `FileLock` (`dataset.py:69`) | ❌ (CLI download only) |
| Download integrity (size/sha256) | ❌ | ✅ `--verify-size` / `--checksum-file` / `--verify-existing` (`dataset.py:142–199`) |
| Atomic finalize | `os.rename` (`dataset.py:140`) | `os.replace` + partial-file cleanup (`dataset.py:221,228–237`) |

Neither difference changes the *content* of a shard, so neither changes
training data given the same shards on disk. They diverge only in:

- **Smallest runnable dataset.** This repo's `train.py:847` sets
  `required_parquet_files = max(2, --min-parquet-files)` and raises a clear
  error below that (`train.py:865`). bio can train+eval on a single shard.
  → For a tiny CPU smoke benchmark, **download ≥2 shards here** (or it won't
  start); bio will run on 1. Use ≥2 on both for any comparison that reports a
  val number, so the val split is genuinely held out.
- **Supply-chain hygiene.** bio can verify shard checksums; this repo trusts
  the cache. See [§5](#5-action-items-for-fair-benchmarks) and bead
  `model_guided_research-wgd` (dataset licensing/integrity).

---

## 4. Batching / masking / shuffle / seq-len / cache — the benchmark-relevant axes

| Axis | Behaviour (both repos) | Fairness impact |
|---|---|---|
| **Batching** | Contiguous token stream reshaped to `B×T`; `B`=`--batch-size`, `T`=`sequence_len`. No gradient accumulation in the loader — `tokens/step = B·T·world_size` (`train.py:1071`). | Match `B`, `T`, and DDP world size. |
| **Padding** | **None.** Tokens are packed densely; every position is a real token. | No pad-token dilution to control for. |
| **Attention masking** | **Causal only**, applied in the model — not the loader. Documents are concatenated with a leading BOS per doc; there is **no intra-document mask**, so cross-document attention within a `B×T` window is permitted (standard nanochat behaviour). Identical in both. | Same effective context contamination on both sides; nothing to equalize. |
| **Shuffle** | No in-loader shuffle. Relies on the pre-shuffled `…-shuffle` corpus; order is deterministic given shards + DDP rank/world_size. | Reproducible; same shard set ⇒ same order. |
| **Sequence length** | `T = config.sequence_len`; same window for train (`train.py:1004`) and val (`:1325`). | Match `sequence_len`. |
| **Caching** | Parquet row-group streaming; OS page cache only. No tokenized-token cache on disk in either. KV-cache is an *inference* concern (`nanochat/engine.py`), not a dataloader one. | None at train time. |
| **Val split** | Same loader with `split="val"` over the last shard; `val_interval`/`val_batches` flags here (`train.py:1319–1331`). | Match `val_batches` when comparing `val_ce`. |

There is **no measured difference in what the model sees** between the two
loaders given matched `(corpus, vocab, B, T, world_size)`.

---

## 5. Action items for fair benchmarks

For any cross-repo or in-repo fixed-FLOPs A/B over this data stack:

1. **Use the same shard set** (≥2 shards of `fineweb-edu-100b-shuffle`) and
   the **same tokenizer artifact** on both sides. In-repo A/B already shares
   one tokenizer across all 11 mechanisms — nothing to do.
2. **Match `(B, T, ddp_world_size)`** so `tokens/step` is equal; remember
   this repo has no loader-level grad-accum, so `--batch-size` *is* the
   micro-batch.
3. **Download ≥2 shards here before tiny runs** (the ≥2-shard guard at
   `train.py:865`); bio's 1-shard smoke shortcut is not replicated and is not
   needed for benchmarking.
4. **Match `val_batches`/`val_interval`** when comparing `val_ce` numbers.
5. Optional supply-chain hardening (not required for fairness): port bio's
   size/sha256 verification (`bio_inspired_nanochat/dataset.py:142–199`) into
   `nanochat/dataset.py` if/when we pin a shard manifest. Tracked separately
   under bead `model_guided_research-wgd`.

**No code changes are warranted by this audit.** The data path is already at
parity for the project's benchmarking needs; the only divergences are an
operational convenience (bio's 1-shard smoke) and a supply-chain feature
(bio's checksum verification) that is independently tracked.

---

## Appendix — files compared

| Role | This repo | bio_inspired |
|---|---|---|
| Dataloader | `nanochat/dataloader.py` | `bio_inspired_nanochat/dataloader.py` |
| Dataset/download | `nanochat/dataset.py` | `bio_inspired_nanochat/dataset.py` |
| Tokenizer | `nanochat/tokenizer.py` | `bio_inspired_nanochat/tokenizer.py` |
| Train consumption | `nanochat/train.py:1002,1319` | `bio_inspired_nanochat/` train script |
</content>
