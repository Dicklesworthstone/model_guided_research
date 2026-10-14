# Adjudication — 2026-10-14

- policy: `ci-v6`
- artifacts indexed: 525 from ['artifacts']
- ledger entries appended: 1
- verdicts: inconclusive: 1
- **0 supported, of which 0 survive FDR at q=0.1 (family: 1 adjudicated - PARTIAL run, not the whole ledger)**
- ledger note: latest recorded verdicts span policies ['ci-v2', 'ci-v3', 'ci-v4', 'ci-v5', 'ci-v6']; q-values here are computed fresh under ci-v6 for this run's family only.

| hypothesis | verdict | q | detail |
|---|---|---|---|
| hyp-cert-telemetry-leads-divergence | inconclusive | 0.0889 | gauge: effect=2.4 ci95=[1.72,3.08] (n=5/5) power=100% |

BLOCKED rows are refusals, not adjudications: the engine declines to rule on weak, mismatched, or tainted evidence. UNDERPOWERED verdicts cleared their threshold at a test with under 50% power to detect the registered effect - an asterisk, not a clean verdict. See `verdicts.json` for machine-readable reasons, p-values, and q-values.
