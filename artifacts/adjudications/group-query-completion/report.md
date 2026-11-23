# Adjudication — 2026-11-23

- policy: `ci-v11`
- artifacts indexed: 1 from ['artifacts/campaigns/group-query-completion']
- ledger entries appended: 1
- verdicts: supported: 1
- **0 supported, of which 0 survive FDR at q=0.1 (family: 0 adjudicated - PARTIAL run, not the whole ledger). Exact fixed-panel verdicts outside the statistical family: 1.**
- ledger note: latest recorded verdicts span policies ['ci-v10', 'ci-v2', 'ci-v3', 'ci-v4', 'ci-v5', 'ci-v6']; q-values here are computed fresh under ci-v11 for this run's family only.

| hypothesis | verdict | q | detail |
|---|---|---|---|
| hyp-group-completion-trusted-queries | supported |  | 89/360 paid queries (ratio 0.2472, 12 primary cases, 20 verified, fixed panel) |

BLOCKED rows are refusals, not adjudications: the engine declines to rule on weak, mismatched, or tainted evidence. UNDERPOWERED verdicts cleared their threshold at a test with under 50% power to detect the registered effect - an asterisk, not a clean verdict. See `verdicts.json` for machine-readable reasons, p-values, and q-values.
