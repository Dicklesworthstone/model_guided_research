# `configs/` — fair-comparison training templates

**Bead:** `model_guided_research-2bx`

Ready-to-run, **FLOPs-annotated** `nanochat` training templates for *fair*
mechanism A/B. Every template fixes the model dims, `(B, T)`, seed, and the
fairness flags, and records the model's own `estimate_flops()`/token so an
equal-budget comparison is reproducible from one place.

- **Machine-readable:** [`fair_comparison.json`](fair_comparison.json) — six
  templates, each validated to parse *and* to match the live
  `GPT.estimate_flops()` (see [validation](#validation)).
- **Canonical rung:** the campaign default (E1) — `L4 / H4 / KV2 / D128 /
  seq256 / bs8` at `target_flops = 1e14` — encoded in
  `scripts/run_campaign.py` and mirrored here as `baseline_standard`.

> `nanochat/train.py` is argparse-driven (it does **not** read a config file —
> `nanochat/configurator.py` is the unused legacy karpathy loader). So a
> "template" here is a fully-specified, copy-pasteable flag set plus its
> precomputed FLOPs accounting, not a file the trainer ingests. Each template's
> `run_train` field is the exact command.

## Templates

| Template | Mechanism | flops/token | tokens/step | steps @1e14 | Horizon |
|---|---|---|---|---|---|
| `baseline_standard` | standard (reference) | 4.4532e7 | 2048 | ~1097 | target_flops 1e14 |
| `flex_off` | standard, dense SDPA | 4.4532e7 | 2048 | ~1097 | target_flops 1e14 |
| `flex_on` | standard, FlexAttention | 4.4532e7 | 2048 | ~1097 | target_flops 1e14 |
| `math_hybrid_tropical` | tropical attn + tropical FFN | 4.4547e7 | 2048 | ~1096 | target_flops 1e14 |
| `math_reversible_symplectic` | reversible symplectic | 4.7490e7 | 2048 | ~1028 | **match max_steps** |
| `cpu_smoke` | standard | 4.4532e7 | 2048 | n/a (20 steps) | max_steps 20 |

## The fairness flags (from the parity audits)

Baked into every template; see `docs/config_parity.md` (wyf) and
`docs/data_parity.md` (6fl) for the derivations:

1. **`--grad-clip-norm 1.0`** — ours defaults to *off*; `bio_inspired`
   defaults to 1.0. Set it on every arm (HOSS ignores it by design).
2. **Match `--compile`** across arms (all or none). It changes throughput,
   not loss. `bio_inspired` always compiles `dynamic=False`; ours is opt-in.
3. **Match `(B, T, ddp_world_size)`** — `tokens_per_step = B·T·world_size`.
   There is no loader-level grad-accum, so `--batch-size` *is* the micro-batch.
4. **LR schedule caveat** — `--scheduler-type none` (default) holds LR flat
   unless `--lr-warmdown-ratio` is passed (bead `kj8s` landed). Within-repo
   A/B is fair either way; for an absolute `val_ce` comparable to
   `bio_inspired`, pass the same warmdown ratio bio uses.
5. **Use ≥2 dataset shards** (`train.py:865`) or `--data-dir <corpus>`.

## The FLOPs lesson (why annotate at all)

`est_steps = round(target_flops / (flops_per_token · B · T · world_size))`.

**Matched model dims do *not* give matched FLOPs/token across mechanisms.** At
the E1 rung:

- standard / tropical-hybrid: ~4.45e7 → ~1097 steps at 1e14 (within 0.1 %).
- reversible **additive**: 4.14e7 (cheaper/token) → ~1180 steps at 1e14.
- reversible **symplectic**: 4.75e7 (dearer/token, the 3× double-backward
  charge from bead `7lba`) → ~1028 steps at 1e14.

`--target-flops` auto-compensates by changing the step count, which is exactly
what you want for an equal-*compute* contract. But for the strict symplectic
case (`gpt.py:686`), prefer matching `--max-steps` so the two arms see the same
number of optimizer updates — hence `math_reversible_symplectic` uses a
`max_steps` horizon.

## Running

Single run (copy the template's `run_train`):

```bash
python -m nanochat.train --device cpu --attention-type standard \
    --n-layer 4 --n-head 4 --n-kv-head 2 --n-embd 128 --sequence-len 256 \
    --batch-size 8 --grad-clip-norm 1.0 --target-flops 1e14 \
    --checkpoint-interval 100000 --seed 42 --run-id baseline_standard
```

A/B suite (the campaign harness, frozen-worktree provenance):

```bash
uv run python scripts/run_campaign.py \
    --combo dyck:standard --combo dyck:tropical --seeds 0,1,2 \
    --target-flops 1e14 --topic fair_demo \
    --extra-args "--grad-clip-norm 1.0"
```

Fixed-FLOPs A/B report:

```bash
mgr bench-fixed-flops --run-id flops_suite_cpu --device cpu --target-flops 1e14 \
    -a standard -a tropical -a reversible
```

## Validation

The numbers in `fair_comparison.json` are not hand-typed guesses — each was
produced by instantiating the model and calling `estimate_flops()`. To
re-verify after a model change, load each template, rebuild the `GPTConfig`,
and assert `params`/`flops_per_token` match (the authoring check confirmed all
six: params exact, FLOPs within 1e-5 relative). If a mechanism's accounting
changes, regenerate this file rather than editing the numbers by hand.
</content>
