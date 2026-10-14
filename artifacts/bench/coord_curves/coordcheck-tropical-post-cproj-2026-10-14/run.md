# coord-check — coordcheck-tropical-post-cproj-2026-10-14

- widths [64, 128, 256, 512, 1024, 2048] · seeds [6, 7, 8, 9, 10, 11] · device cpu · schema mgr.bench.coord_curves.v1
- flat (|slope| <= 0.05) means correctly parameterized in width; one artifact per (mechanism, arm, seed)

| mechanism | arm | ffn | seed | slope | abs slope | R2 | class | rms first→last |
|---|---|---|---|---|---|---|---|---|
| tropical | current | tropical | 6 | +0.0950 | 0.0950 | 0.968 | EVT (Gumbel max) | 2.806→3.932 |
| tropical | current | tropical | 7 | +0.1187 | 0.1187 | 0.946 | EVT (Gumbel max) | 2.610→3.934 |
| tropical | current | tropical | 8 | +0.1014 | 0.1014 | 0.923 | EVT (Gumbel max) | 2.970→4.100 |
| tropical | current | tropical | 9 | +0.0834 | 0.0834 | 0.874 | EVT (Gumbel max) | 2.874→3.861 |
| tropical | current | tropical | 10 | +0.0865 | 0.0865 | 0.935 | EVT (Gumbel max) | 2.865→4.023 |
| tropical | current | tropical | 11 | +0.1307 | 0.1307 | 0.943 | EVT (Gumbel max) | 2.684→4.149 |
| tropical | nsa | tropical | 6 | -0.0077 | 0.0077 | 0.473 | EVT (Gumbel max) | 1.173→1.139 |
| tropical | nsa | tropical | 7 | -0.0112 | 0.0112 | 0.883 | EVT (Gumbel max) | 1.176→1.129 |
| tropical | nsa | tropical | 8 | -0.0036 | 0.0036 | 0.440 | EVT (Gumbel max) | 1.145→1.129 |
| tropical | nsa | tropical | 9 | -0.0094 | 0.0094 | 0.713 | EVT (Gumbel max) | 1.173→1.135 |
| tropical | nsa | tropical | 10 | -0.0069 | 0.0069 | 0.498 | EVT (Gumbel max) | 1.156→1.139 |
| tropical | nsa | tropical | 11 | -0.0100 | 0.0100 | 0.630 | EVT (Gumbel max) | 1.157→1.117 |
