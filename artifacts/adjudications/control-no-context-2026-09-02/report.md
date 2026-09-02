# Adjudication — 2026-09-02

- policy: `ci-v6`
- artifacts indexed: 276 from ['artifacts']
- ledger entries appended: 1
- verdicts: supported: 1
- **1 supported, of which 1 survive FDR at q=0.1 (family: 1 adjudicated - PARTIAL run, not the whole ledger)**
- ledger note: latest recorded verdicts span policies ['ci-v2', 'ci-v3', 'ci-v4', 'ci-v5', 'ci-v6']; q-values here are computed fresh under ci-v6 for this run's family only.

| hypothesis | verdict | q | detail |
|---|---|---|---|
| hyp-control-no-context-planted-effect | supported | 0.00972 | standard: effect=0.3472 ci95=[0.2576,0.4369] (n=3/3) power=100% |

BLOCKED rows are refusals, not adjudications: the engine declines to rule on weak, mismatched, or tainted evidence. UNDERPOWERED verdicts cleared their threshold at a test with under 50% power to detect the registered effect - an asterisk, not a clean verdict. See `verdicts.json` for machine-readable reasons, p-values, and q-values.
