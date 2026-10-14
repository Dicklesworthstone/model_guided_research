# Adjudication — 2026-10-14

- policy: `ci-v6`
- artifacts indexed: 513 from ['artifacts']
- ledger entries appended: 3
- verdicts: inconclusive: 2 · supported: 1
- **1 supported, of which 1 survive FDR at q=0.1 (family: 3 adjudicated - PARTIAL run, not the whole ledger)**
- ledger note: latest recorded verdicts span policies ['ci-v2', 'ci-v3', 'ci-v4', 'ci-v5', 'ci-v6']; q-values here are computed fresh under ci-v6 for this run's family only.

| hypothesis | verdict | q | detail |
|---|---|---|---|
| hyp-geo-init-mismatched-safety | supported | 0.0003 | ultrametric: effect=1.004 ci95=[0.991,1.018] (n=3/3) power=100%; fractal: effect=0.9868 ci95=[0.9649,1.008] (n=3/3) power=99%; hyperbolic: effect=0.9918 ci95=[0.969,1.015] (n=3/3) power=99% |
| hyp-geo-init-early-advantage | inconclusive | 0.187 | ultrametric: effect=0.9992 ci95=[0.9625,1.037] (n=3/3) power=18%; fractal: effect=0.9647 ci95=[0.9552,0.9727] (n=3/3) power=99%; hyperbolic: effect=1.001 ci95=[0.9852,1.025] (n=3/3) power=47% |
| hyp-geo-init-less-distance-traveled | inconclusive-underpowered | 0.764 | ultrametric: effect=1.001 ci95=[0.9678,1.053] (n=3/3) power=4%; fractal: effect=1.073 ci95=[0.8911,1.325] (n=3/3) power=3%; hyperbolic: effect=0.6494 ci95=[0.4394,0.8482] (n=3/3) power=3% UNDERPOWERED(need n≈10203) |

BLOCKED rows are refusals, not adjudications: the engine declines to rule on weak, mismatched, or tainted evidence. UNDERPOWERED verdicts cleared their threshold at a test with under 50% power to detect the registered effect - an asterisk, not a clean verdict. See `verdicts.json` for machine-readable reasons, p-values, and q-values.
