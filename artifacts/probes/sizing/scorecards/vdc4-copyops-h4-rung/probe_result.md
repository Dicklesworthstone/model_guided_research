# vdc.4 Phase-0b exact-H4 copyops sizing result

This is a quarantined sizing artifact. It is not eligible as evidence for any
mechanism hypothesis.

- Bead: `model_guided_research-vdc.4`
- Producer: clean detached commit `b61aa95977801e6d4c5a2a0d80e152c252abed93`
- Runtime: 2,681.2 seconds
- Completion: 6/6 cells done, 0 failed
- Coordinate: standard attention, L2/H4/KV2/d64, T64, B4
- Target: 1e12 FLOPs, 196 steps, 1.001295805312e12 planned FLOPs
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
| 0 | 1.8497 | 0 | 0 | 0.020833 | 13.7785 | 7.2521 |
| 1 | 1.8569 | 0 | 0 | 0.020833 | 14.1617 | 7.3148 |
| 2 | 1.8530 | 0 | 0 | 0.020833 | 13.8088 | 7.2217 |

No checkpoint qualified. All 864 scored copyops examples completed with zero
skips, but every checkpoint produced zero exact matches in both regions. The
distribution is uniformly floored, not bimodal. Read-only receipt inspection
found prompt-blind local-grammar loops rather than a prompt-boundary, corpus,
checkpoint, or evaluator fault. Low cross-entropy is therefore not evidence
that the model learned the stateful COPY/REV/ROT operation.

The quarantined placebo cells also completed cleanly. Their held-out
perplexities were 24.1336, 17.8245, and 20.3520; their in-range perplexities
were 5.1242, 5.0919, and 5.2272, with zero skipped examples.

## Decision

The preregistered 1e12 rung fails. The fresh successor is 2e12 FLOPs at the
same exact coordinate, giving 392 steps and 2.002595610624e12 planned FLOPs.
Target FLOPs is the only changed knob. That successor remains quarantined and
cannot contribute claim evidence. Checkpoints, raw logs, datasets, and
generation receipts remain local and untouched under this directory.
