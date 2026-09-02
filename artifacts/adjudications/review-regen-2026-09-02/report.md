# Adjudication — 2026-09-01

- policy: `ci-v6`
- artifacts indexed: 263 from ['artifacts']
- ledger entries appended: 3
- verdicts: inconclusive: 1 · supported: 2
- **2 supported, of which 2 survive FDR at q=0.1 (family: 3 adjudicated - PARTIAL run, not the whole ledger)**
- ledger note: latest recorded verdicts span policies ['ci-v2', 'ci-v3', 'ci-v4', 'ci-v5']; q-values here are computed fresh under ci-v6 for this run's family only.

| hypothesis | verdict | q | detail |
|---|---|---|---|
| hyp-rmatrix-charge-drift-separation | supported | 2.92e-06 | braid: effect=6.855e-07 ci95=[4.009e-07,9.7e-07] (n=4/0) power=100% |
| hyp-ultrametric-trie-decode-speedup | supported-underpowered | 0.000299 | ultrametric: effect=71.78 ci95=[65.7,77.87] (n=3/0) power=11% UNDERPOWERED(need n≈48) |
| hyp-balltree-valued-attention-speedup | inconclusive | 0.967 | ultrametric: effect=3.387 ci95=[2.671,4.103] (n=3/0) power=100% |

BLOCKED rows are refusals, not adjudications: the engine declines to rule on weak, mismatched, or tainted evidence. UNDERPOWERED verdicts cleared their threshold at a test with under 50% power to detect the registered effect - an asterisk, not a clean verdict. See `verdicts.json` for machine-readable reasons, p-values, and q-values.
