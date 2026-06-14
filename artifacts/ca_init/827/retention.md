# CA-init structure retention — `827`

Bead model_guided_research-827. After training, how much of the init's weight structure survives? Cosine similarity between the (deterministically reconstructed) init weights and the trained checkpoint, over the CA-initialized tensors (c_q/c_k/c_v/c_fc + wte). Higher cosine = more structure retained.

- Device `cpu` · steps `200` · lr `0.0006` · seeds `[0, 1]`

Bias axis: `ca_rule30` (alpha=1.0 pure CA) → `ca_mix0.5` → `ca_mix0.25` (more bias-like) → `standard` (random-init baseline).

| config | variant | alpha | ok | retention cosine | rel L2 drift | final loss |
|---|---|---|---|---|---|---|
| mlp_heavy | standard | 0.00 | 2/2 | 0.9993 | 0.0433 | 7.3802 |
| mlp_heavy | ca_rule30 | 1.00 | 2/2 | 0.9991 | 0.0467 | 7.4074 |
| mlp_heavy | ca_mix0.5 | 0.50 | 2/2 | 0.9983 | 0.0721 | 7.5370 |
| mlp_heavy | ca_mix0.25 | 0.25 | 2/2 | 0.9988 | 0.0593 | 7.4767 |

## Reading it

- **Cosine → 1.0**: the trained weights still point the same way as init → CA structure is retained; SGD only nudged magnitudes.
- **Cosine ≪ 1.0**: training reoriented the weights → the init structure washed out (CA-init then acts as a fancy random seed, not a lasting prior).
- Compare across the alpha axis: if lower-alpha (more bias-like) variants retain MORE structure at equal/again-better loss, the bias framing helps; if not, pure CA is as good as any blend.

The freeze-channel arm (freeze CA channels for N warmup steps) is left as a follow-up: it needs a `train.py --freeze-init-steps` feature and is out of scope for a no-default-change experiment.

## Reproduction
```
uv run python scripts/ca_init_bench.py --mode retention --run-id 827 --configs mlp_heavy --max-steps 200 --warmup-steps 5 --seeds 0 1 --learning-rate 6e-4
```
