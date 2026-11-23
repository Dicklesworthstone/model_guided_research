# Can a frozen transformer compose its short-word answers?

Three standard transformers trained on the group benchmark at 30 trillion estimated model FLOPs answered almost all familiar short prompts correctly in the original evaluation. Their long-word scores were 4, 5 and 4 out of 144. This study held the checkpoints fixed and intervened on the words. Its protocol and conditional transfer test were committed in `e4513fa` before evaluating new prompts. The old scores motivated selection of these three runs. [Original study](../scorecards/vdc4-group-tasktok-shuf-b30-s256-rung/report.md)

Within each group, two short words with the same product can replace one another inside any longer word. The preprocessor learned these substitutions from the 80,000 archived training documents. It repeatedly applied the substitution that saved the most symbols, breaking ties deterministically. It received only the input word at inference. A shuffled control preserved the exact symbol counts and length of each shortened prompt. The model saw a BOS-prefixed prompt and a fixed decoding budget independent of the correct answer.

### Verified training labels

The first panel had 144 new prompts across four groups. The primary contrast used its 73 S5/A5 prompts. Every supervised rewrite preserved its product. The table reports correct answers on the same 73 words per checkpoint.

| Training seed | Raw | Shortened | Shuffled | Exact training-dictionary lookup after shortening |
|---:|---:|---:|---:|---:|
| 0 | 2 | 54 | 17 | 57 |
| 1 | 0 | 50 | 16 | 57 |
| 2 | 0 | 49 | 17 | 57 |

For 56 words the shuffle actually changed the correct answer. The shortened prompts yielded 37, 34 and 32 correct there. The controls yielded zero for each model. On the 16 shortened words outside the training dictionary, the models scored 1, 0 and 0. The strongest result was on A5, where the dictionary covered all 43 examples. These results locate the gain in verified substitution plus familiar-word recognition. They give little evidence for extrapolation beyond dictionary coverage.

### Labels supplied by the frozen model

The primary result triggered the precommitted transfer experiment on fresh seeds 701–703. Each network supplied answers for its own dictionary before seeing the new prompts. On the 1,016 distinct short A5/S5 words the three networks correctly labeled 840, 803 and 773. Local errors then compounded under repeated substitution.

| Training seed | Raw | Verified substitutions | Self-labeled substitutions | Self-labeled shuffle | Products preserved by self-labeled substitutions |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 47 | 9 | 1 | 10/73 |
| 1 | 1 | 44 | 3 | 1 | 3/73 |
| 2 | 1 | 44 | 4 | 1 | 4/73 |

The prespecified 95% preservation requirement failed for every model. We stopped the self-labeling study. A separate pointwise-permutation verifier checked the endpoint, rewrite trace and sham of all 1,296 stored rows. The same prompts recur across models and intervention arms. The independent training count is three. Earlier work documents multiple learned algorithms and partial group-state tracking, which motivated testing input interventions directly. [Li et al.](https://arxiv.org/abs/2503.02854) · [Tracking States or Tracking Cosets?](https://arxiv.org/abs/2609.29951)

### How much extra supervision sufficed?

A reference learner tracked transitions between opaque answer labels, asking for the endpoint of a representative word extended by one symbol whenever it lacked that transition. Its completed tables agreed with the task on every state and generator. This direct policy used 108 new labels for S5, 6 for A5, 0 for S3 and 54 for Z60. Both the policy and its query transcript are available. [Teaching results](teaching.json)

### Replay

The executable revision is `4e55e282dafedee9a4b5e1421ecbdd696d9cdc17`. Extract `frozen-inputs.zip` into an empty directory. From a checkout containing this script and the original scorecard, the following command is the check to supply to DSR. The output path must be new.

```sh
uv run --locked python -m scripts.group_rewrite_probe --inputs /path/to/inputs --output /path/to/results
```

Both macOS and Linux reproduced all 432 archived greedy responses, including their exact decoded text. Their 14 output files matched byte for byte. The inputs archive has SHA-256 `ea58a726846ef9842caa493ae53b996664101ff9c628900528b5a663f757fbd8`. These are quarantined diagnostic interventions on existing checkpoints. [Model and data hashes, paired outcomes and denominators](summary.json) · [Weights, tokenizers and original training split](frozen-inputs.zip) · [Per-prompt observations, independent verifier and DSR receipts](observations.zip)
