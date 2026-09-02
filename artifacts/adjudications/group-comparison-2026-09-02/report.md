# Adjudication — 2026-09-02

- policy: `ci-v6`
- artifacts indexed: 328 from ['artifacts']
- ledger entries appended: 2
- verdicts: inconclusive: 2
- **0 supported, of which 0 survive FDR at q=0.1 (family: 2 adjudicated - PARTIAL run, not the whole ledger)**
- ledger note: latest recorded verdicts span policies ['ci-v2', 'ci-v3', 'ci-v4', 'ci-v5', 'ci-v6']; q-values here are computed fresh under ci-v6 for this run's family only.

| hypothesis | verdict | q | detail |
|---|---|---|---|
| hyp-rmatrix-s5-length-slope | inconclusive | 0.378 | braid: effect=-0.3704 ci95=[-0.7454,122] (n=3/3) power=3% |
| hyp-rmatrix-solvable-control-specificity | inconclusive | 1 | braid: effect=-0 ci95=[0,-0] (n=3/3) power=100% FLOOR(em 0.02315 <= prior 0.09722, slopes vacuous) |

BLOCKED rows are refusals, not adjudications: the engine declines to rule on weak, mismatched, or tainted evidence. UNDERPOWERED verdicts cleared their threshold at a test with under 50% power to detect the registered effect - an asterisk, not a clean verdict. See `verdicts.json` for machine-readable reasons, p-values, and q-values.
