# Adjudication — 2026-09-02

- policy: `ci-v6`
- artifacts indexed: 308 from ['artifacts']
- ledger entries appended: 2
- verdicts: refuted: 1 · supported: 1
- **1 supported, of which 1 survive FDR at q=0.1 (family: 2 adjudicated - PARTIAL run, not the whole ledger)**
- ledger note: latest recorded verdicts span policies ['ci-v2', 'ci-v3', 'ci-v4', 'ci-v5', 'ci-v6']; q-values here are computed fresh under ci-v6 for this run's family only.

| hypothesis | verdict | q | detail |
|---|---|---|---|
| hyp-tropical-evt-miscoupling | refuted | 1 | tropical: effect=0.01943 ci95=[-0.008588,0.04744] (n=6/6) power=98% refutation_margin=2.5× |
| hyp-coordcheck-clt-flat | supported | 1.24e-05 | standard: effect=0.001196 ci95=[0.0004646,0.001927] (n=3/0) power=100%; reversible: effect=0.0005632 ci95=[-0.0001867,0.001313] (n=3/0) power=100% |

BLOCKED rows are refusals, not adjudications: the engine declines to rule on weak, mismatched, or tainted evidence. UNDERPOWERED verdicts cleared their threshold at a test with under 50% power to detect the registered effect - an asterisk, not a clean verdict. See `verdicts.json` for machine-readable reasons, p-values, and q-values.
