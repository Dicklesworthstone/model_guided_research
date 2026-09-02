# coord-check — coordcheck-2026-09-02-b

- widths [64, 128, 256, 512, 1024, 2048] · seeds [3, 4, 5] · device cpu · schema mgr.bench.coord_curves.v1
- flat (|slope| <= 0.05) means correctly parameterized in width; one artifact per (mechanism, arm, seed)

| mechanism | arm | ffn | seed | slope | abs slope | R2 | class | rms first→last |
|---|---|---|---|---|---|---|---|---|
| tropical | current | tropical | 3 | +0.1130 | 0.1130 | 0.946 | EVT (Gumbel max) | 2.719→3.862 |
| tropical | current | tropical | 4 | +0.1095 | 0.1095 | 0.968 | EVT (Gumbel max) | 2.673→3.892 |
| tropical | current | tropical | 5 | +0.1070 | 0.1070 | 0.948 | EVT (Gumbel max) | 2.618→3.959 |
| tropical | nsa | tropical | 3 | +0.1338 | 0.1338 | 0.949 | EVT (Gumbel max) | 2.259→3.429 |
| tropical | nsa | tropical | 4 | +0.1307 | 0.1307 | 0.970 | EVT (Gumbel max) | 2.206→3.458 |
| tropical | nsa | tropical | 5 | +0.1272 | 0.1272 | 0.951 | EVT (Gumbel max) | 2.162→3.525 |
