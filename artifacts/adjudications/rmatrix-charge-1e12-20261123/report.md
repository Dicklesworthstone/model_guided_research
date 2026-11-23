# Adjudication — 2026-11-23

- policy: `ci-v10`
- artifacts indexed: 535 from ['artifacts']
- ledger entries appended: 1
- verdicts: refuted: 1
- **0 supported, of which 0 survive FDR at q=0.1 (family: 1 adjudicated - PARTIAL run, not the whole ledger)**
- ledger note: latest recorded verdicts span policies ['ci-v2', 'ci-v3', 'ci-v4', 'ci-v5', 'ci-v6']; q-values here are computed fresh under ci-v10 for this run's family only.

| hypothesis | verdict | q | detail |
|---|---|---|---|
| hyp-rmatrix-charge-decodability | refuted | 0.999 | braid: effect=0.2778 ci95=[0.03874,0.5168] (n=3/0) power=100% refutation_margin=7.2× |

BLOCKED rows are refusals, not adjudications: the engine declines to rule on weak, mismatched, or tainted evidence. UNDERPOWERED verdicts cleared their threshold at a test with under 50% power to detect the registered effect - an asterisk, not a clean verdict. See `verdicts.json` for machine-readable reasons, p-values, and q-values.
