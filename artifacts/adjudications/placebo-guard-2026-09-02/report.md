# Adjudication — 2026-09-02

- policy: `ci-v6`
- artifacts indexed: 427 from ['artifacts']
- ledger entries appended: 1
- verdicts: inconclusive: 1
- **0 supported, of which 0 survive FDR at q=0.1 (family: 1 adjudicated - PARTIAL run, not the whole ledger)**
- ledger note: latest recorded verdicts span policies ['ci-v2', 'ci-v3', 'ci-v4', 'ci-v5', 'ci-v6']; q-values here are computed fresh under ci-v6 for this run's family only.

| hypothesis | verdict | q | detail |
|---|---|---|---|
| hyp-placebo-no-winner | inconclusive | 0.691 | tropical: effect=1.006 ci95=[1.004,1.008] (n=3/8) power=100%; ultrametric: effect=1.012 ci95=[1.01,1.014] (n=3/6) power=100%; simplicial: effect=0.9964 ci95=[0.9937,0.9991] (n=3/6) power=100%; quaternion: effect=1.008 ci95=[1.006,1.011] (n=3/6) power=100%; braid: effect=1.002 ci95=[1,1.003] (n=3/6) power=100%; fractal: effect=1.021 ci95=[1.015,1.032] (n=3/6) power=100%; octonion: effect=1.003 ci95=[1.001,1.004] (n=3/8) power=100%; surreal: effect=0.9892 ci95=[0.9872,0.9912] (n=7/6) power=100%; reversible: effect=0.9995 ci95=[0.9978,1.001] (n=3/8) power=100%; gauge: effect=1.005 ci95=[1.003,1.006] (n=3/8) power=100% |

BLOCKED rows are refusals, not adjudications: the engine declines to rule on weak, mismatched, or tainted evidence. UNDERPOWERED verdicts cleared their threshold at a test with under 50% power to detect the registered effect - an asterisk, not a clean verdict. See `verdicts.json` for machine-readable reasons, p-values, and q-values.
