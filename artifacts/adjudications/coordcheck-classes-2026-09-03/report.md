# Adjudication — 2026-09-02

- policy: `ci-v6`
- artifacts indexed: 495 from ['artifacts']
- ledger entries appended: 3
- verdicts: supported: 3
- **3 supported, of which 3 survive FDR at q=0.1 (family: 3 adjudicated - PARTIAL run, not the whole ledger)**
- ledger note: latest recorded verdicts span policies ['ci-v2', 'ci-v3', 'ci-v4', 'ci-v5', 'ci-v6']; q-values here are computed fresh under ci-v6 for this run's family only.

| hypothesis | verdict | q | detail |
|---|---|---|---|
| hyp-coordcheck-clt-assumed-flat | supported | 6.81e-10 | gauge: effect=0.0004763 ci95=[-0.000277,0.001229] (n=6/0) power=100%; braid: effect=0.001058 ci95=[0.0005901,0.001525] (n=6/0) power=100%; simplicial: effect=0.0008987 ci95=[0.0003264,0.001471] (n=6/0) power=100%; fractal: effect=0.001129 ci95=[0.0004069,0.00185] (n=6/0) power=100%; surreal: effect=0.001168 ci95=[-6.937e-06,0.002343] (n=6/0) power=100% |
| hyp-coordcheck-isometry-flat | supported | 8.7e-11 | quaternion: effect=0.000759 ci95=[0.0001289,0.001389] (n=6/0) power=100%; octonion: effect=0.0005259 ci95=[-9.911e-05,0.001151] (n=6/0) power=100% |
| hyp-coordcheck-branching-radial-flat | supported | 1.83e-10 | ultrametric: effect=0.001225 ci95=[0.000393,0.002057] (n=6/0) power=100%; hyperbolic: effect=0.0007567 ci95=[0.0002231,0.00129] (n=6/0) power=100% |

BLOCKED rows are refusals, not adjudications: the engine declines to rule on weak, mismatched, or tainted evidence. UNDERPOWERED verdicts cleared their threshold at a test with under 50% power to detect the registered effect - an asterisk, not a clean verdict. See `verdicts.json` for machine-readable reasons, p-values, and q-values.
