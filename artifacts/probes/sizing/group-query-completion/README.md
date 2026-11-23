# Learning exact group transitions from short words and paid labels

A learner received the opaque endpoint labels of all 508 binary words of lengths two through eight. To answer arbitrarily long words, it could buy one more endpoint at a time. On twelve previously unseen S5/A5 generating pairs, one algebraic inference reduced the total from **360 to 89 additional queries**. Both learners recovered every transition. The registered target was a reduction of at least 50%. [Registration](../../../../hypotheses/registry.yaml) · [Frozen study and query logs](panel.json)

### Where the savings come from

Take any word `w`. Prepending `a` and appending `b` gives the same word whichever operation is performed first. Once three sides of the resulting square of endpoint labels are known, its fourth side follows. This works even when `ab` and `ba` have different products. For a group, each generator is a permutation, so the same inference applies when an edge is traversed backwards.

Both learners observed left and right extensions, used inverses and filled the last unknown edge of a bijection by elimination. Both queried the shortest accessible representative extended by one symbol, with lexical tie-breaking. One also propagated commuting squares. Neither learner received a multiplication table, group identity, group name, permutation representation, or test answers. Each received the complete set of opaque output names, the two input symbols, its own transcript of paid answers, and the same initial word labels.

| Fresh presentations | Cases | Queries without squares | Queries with squares | Label-naming lower bound | Right transitions checked per learner |
|---|---:|---:|---:|---:|---:|
| S5 | 6 | 274 | 64 | 57 | 1,440 |
| A5 | 6 | 86 | 25 | 22 | 720 |
| S3 | 2 | 0 | 0 | 0 | 24 |
| Z60 | 6 | 344 | 148 | 135 | 720 |

On the twelve primary cases, ten required fewer queries with squares and two were solved without querying in either arm. The case with the most unseen S5 labels had 32. Its query counts were 112 and 34. Some Z60 presentations required more queries than an A5 presentation, even though addition in Z60 commutes. The short-word labels expose different parts of the state graph under different generators.

### What is certified?

For each of 20 marked groups, the verifier compared every recovered right-transition edge with the hidden action and checked all four two-symbol starting words. This covers **2,904 edges per learner**. Induction on the remaining symbols proves the recovered table gives the correct answer for every word of length at least two on that presentation. The verifier renamed every opaque label and reran all 40 acquisitions against only the recorded answer tapes. Each learner made exactly the same queries, in order, and recovered the corresponding renamed table. The panel excludes the original four generating pairs and duplicates up to isomorphism preserving the ordered generators and identity. It samples from these four group families.

For a presentation with *U* unseen output names, a learner guaranteeing exact answers for every opaque labeling needs at least `max(0, U − 1)` extra endpoint labels. If two names remain unseen, swapping them preserves every observed answer and changes some reachable word's answer. Summing this bound over the primary cases gives **79**. The policy used ten queries above that bound. The bound counts naming ambiguity alone. The minimum needed to identify the transition structure may be higher.

The comparison was frozen in `b313900`. Its finite-panel threshold was met and the independent verifier reproduced it. The formal registry entry is open because the current adjudicator has no collector for this experimental unit. These counts measure label efficiency on a finite symbolic benchmark. The transfer to a neural network answering raw prompts is an open experiment. Active automata learning and group-word rewriting are established methods. [Angluin, 1987](https://people.eecs.berkeley.edu/~dawnsong/teaching/s10/papers/angluin87.pdf) · [GAP reference manual](https://gap-system.github.io/gap/doc/ref/chap52.html) · [Sequential Group Composition](https://proceedings.mlr.press/v306/marchetti26a.html)

### Replay

Use source revision `24819abed491ac698741d989a749f0cb05c35831` and an unused output filename. Supply the following command to DSR.

```sh
uv run --locked python -m scripts.group_rewrite_probe --query-panel --output /path/to/panel.json
```

macOS and Linux produced byte-identical JSON with SHA-256 `ce479b50cc86a80eaee8749ef3ef151d88d041dd3347c6d424ab6dd1fb50c26a`. The archive includes the independent verifier and DSR logs. [Queries and full recovered tables](panel.json) · [Verification archive](verification.zip) · [Frozen transformer intervention](../group-rewrite/README.md)
