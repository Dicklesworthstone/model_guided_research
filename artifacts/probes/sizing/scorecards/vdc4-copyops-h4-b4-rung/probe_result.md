# vdc.4 Phase-0d exact-H4 copyops sizing result

This is a quarantined sizing artifact. It is not eligible as evidence for any
mechanism hypothesis.

- Bead: `model_guided_research-vdc.4`
- Producer: clean detached commit `035903690e433f8a89a36ee3bab523938b011fce`
- Runtime: 2,897.4 seconds
- Completion: 6/6 cells done, 0 failed
- Coordinate: standard attention, L2/H4/KV2/d64, T64, B4
- Target: 4e12 FLOPs, 783 steps, 4.000082558976e12 planned FLOPs
- Training seeds: 0, 1, 2
- Eval seeds: 0, 1, 2; 48 examples per split
- Dataset: 1,000 documents, seed 42
- Full provenance: [manifest.json](manifest.json)

## Preregistered selection rule

The rung qualifies only if all three standard copyops checkpoints exceed their
own artifact-recorded held-out answer prior and the per-seed held-out
distribution is non-pathological and unimodal rather than floor-bimodal. If
the rung fails, preregister one fresh budget-only successor and change no
model, optimizer, corpus, task, seed, or evaluation knob.

## Result

| Train seed | Final loss | Held-out exact match | In-range exact match | Recorded prior | Held-out PPL | In-range PPL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.8309 | 0 | 0 | 0.020833 | 15.4522 | 6.8674 |
| 1 | 1.8088 | 0 | 0 | 0.020833 | 16.5630 | 6.7863 |
| 2 | 1.8162 | 0 | 0 | 0.020833 | 16.3033 | 6.8340 |

No checkpoint qualified. All 864 scored copyops examples completed with zero
skips, but every checkpoint produced zero exact matches in both regions. The
distribution is uniformly floored, not bimodal. The qualitative failure mode
changed: the 2e12 `e OUT` loops gave way to mostly one-token or otherwise
prematurely terminated outputs. In fact, 530 of 864 generations contained a
single output token. In-range perplexity improved while held-out perplexity
worsened, so greater exposure still did not produce the stateful COPY/REV/ROT
operation.

The quarantined placebo cells also completed cleanly. Their held-out
perplexities were 50.6009, 50.4340, and 36.6946; their in-range perplexities
were 4.1222, 4.1213, and 4.1227, with zero skipped examples.

## Decision

The preregistered 4e12 rung fails. The fresh successor is 8e12 FLOPs at the
same exact coordinate, giving 1,566 steps and 8.000165117952e12 planned FLOPs.
Target FLOPs is the only changed knob. This is the final automatic budget
doubling: if it fails, stop the ladder and open focused diagnostic work before
changing any non-budget knob. The successor remains quarantined and cannot
contribute claim evidence. Checkpoints, raw logs, datasets, and generation
receipts remain local and untouched under this directory.
