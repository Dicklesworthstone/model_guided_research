# Can a transformer retain what an exact group teacher learned?

The student receives only a raw word and predicts its endpoint. A different
opaque label map is used for every presentation. Neither symbolic substitution
nor table lookup is available to the student at evaluation. The comparison
asks whether an active query policy improves long-word neural predictions over
a random query policy with the same query count and lengths.

## Results

The preregistered neural verdict is **inconclusive**. The random-query
students answered 36 of 2,304 long words correctly, below the empirical
majority floor of 110. The exact matched difference between active and random
was −0.48 percentage points. Its one-sided 95% lower and upper bounds were
−0.83 and −0.12 percentage points. Random minus the empirical floor averaged
−3.21 percentage points with a lower bound of −3.61. The floor gate therefore
prevents a ruling about the registered +3 percentage-point effect.

| Method | Raw words of length 32, 48 or 64 | Raw words of length 9–15 |
|---|---:|---:|
| Active queries | 25/2,304 | 158/3,072 |
| Random queries | 36/2,304 | 161/3,072 |
| Short labels only | 26/2,304 | 60/3,072 |
| Training-only dictionary with majority fallback | 32/2,304 | |
| Empirical majority per presentation | 110/2,304 | |
| Acquired symbolic table | 2,304/2,304 | |

Active won one presentation, tied 16, and lost seven against random on the
primary endpoint. Both paid curricula yielded more correct secondary answers
than the short-only control. Each acquired symbolic table reproduced its
held-out endpoints when replayed from only the initial examples and the
recorded paid labels. The transformer training experiment stopped after the
fixed evaluation. [Predictions and training receipts](student-score.json)

## Method

The protocol was fixed in Bead `qj9v` at `057b34d`. The sample consists of
12 new S5 presentations and 12 new A5 presentations. Each presentation supplied
all 508 words of lengths two through eight. The two paid arms each purchased
184 additional endpoints across the 24 cases. Both used the same algebraic
inference and supplied 1,536 additional inferred words per case. Seven
presentations required no paid labels and yielded identical active and random
training inputs. The short-only control repeated the initial labels to reach
the same row count. An independent permutation verifier checked every supplied
label and the disjoint train and evaluation words.

The four-layer standard Nanochat GPT used width 96 and four attention heads.
The training objective scored only the endpoint after the answer marker.
Within each presentation all arms shared initial weights, sample indices,
architecture and 900 AdamW updates with a batch size of 64 and 25 positions
per sequence. Each arm processed 1,440,000 sequence positions and 57,600
supervised endpoint targets. GPU elapsed times and data and checkpoint hashes
are recorded separately. Symbolic acquisitions, inferred labels, and independent
answer checking are logged by the study. Validation labels do not enter the
student input or guide optimization.

Each primary evaluation used 32 unseen words at each of lengths 32, 48 and 64.
An additional 128 disjoint words at lengths 9–15 provided a secondary diagnostic.
The study reports one paired accuracy difference per group presentation. The
registered effect is greater than three percentage points, with a one-sided
95% lower confidence bound on random minus the empirical majority prior greater
than zero before ruling in either direction. The fixed design assumes paired
standard deviation 0.14 and a true effect of 0.12 for power planning. The
experiment also records a training-only dictionary with majority fallback and
the exact symbolic reference.

## Reproduction

The dataset was generated at `d37e507`. The trained source revision is
`5cd11996eabe3c298552e36e68968c2700d0e4f8`. The GPU workers used CUDA graph
dispatch for the forward and backward passes after checking six complete
eager and captured runs. All sampled losses and final parameter bytes
matched. The learning-rate schedule and AdamW updates ran eagerly in both
paths. The comparison used two RTX 4090 devices and Torch 2.13.0+cu130.
The archives contain all training and test documents, per-example
predictions, checkpoint fingerprints, parity logs and audit scripts.
Checkpoint binaries are retained on the GPU worker.

At the trained revision, Linux DSR passed Ruff lint and formatting, full
type checking, and pytest with 923 passes and 16 skips. A full ledger dry
run reproduced all 76 other verdict records byte for byte after the
symbolic collector landed. [Formal symbolic adjudication](../../adjudications/group-query-completion/report.md)

Supply the following commands to DSR from a clean checkout, using new paths.
Training and scoring use CUDA. Both training slices must finish before scoring.

```sh
uv run --locked python -m scripts.group_student_probe prepare --data /path/to/cohort
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m scripts.group_student_probe train --cuda-graph --data /path/to/cohort --runs /path/to/runs --first 0 --last 12
CUDA_VISIBLE_DEVICES=1 uv run --locked python -m scripts.group_student_probe train --cuda-graph --data /path/to/cohort --runs /path/to/runs --first 12 --last 24
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m scripts.group_student_probe score --data /path/to/cohort --runs /path/to/runs --output /path/to/results.json
```

[Frozen cohort archive](registered-cohort.tgz) · [Audit scripts and machine receipts](verification.zip)
