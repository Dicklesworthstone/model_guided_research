# coord-check — coordcheck-2026-09-02

- widths [64, 128, 256, 512, 1024, 2048] · seeds [0, 1, 2] · device cpu · schema mgr.bench.coord_curves.v1
- flat (|slope| <= 0.05) means correctly parameterized in width; one artifact per (mechanism, arm, seed)

| mechanism | arm | ffn | seed | slope | abs slope | R2 | class | rms first→last |
|---|---|---|---|---|---|---|---|---|
| standard | current | standard | 0 | +0.0015 | 0.0015 | 0.559 | CLT (Gaussian sum) | 1.031→1.038 |
| standard | current | standard | 1 | +0.0011 | 0.0011 | 0.536 | CLT (Gaussian sum) | 1.036→1.038 |
| standard | current | standard | 2 | +0.0009 | 0.0009 | 0.235 | CLT (Gaussian sum) | 1.035→1.038 |
| reversible | current | standard | 0 | -0.0005 | 0.0005 | 0.532 | CLT (volume-preserving) | 1.021→1.020 |
| reversible | current | standard | 1 | -0.0009 | 0.0009 | 0.357 | CLT (volume-preserving) | 1.023→1.021 |
| reversible | current | standard | 2 | -0.0003 | 0.0003 | 0.131 | CLT (volume-preserving) | 1.020→1.019 |
| tropical | current | tropical | 0 | +0.0995 | 0.0995 | 0.908 | EVT (Gumbel max) | 2.711→4.046 |
| tropical | current | tropical | 1 | +0.1178 | 0.1178 | 0.941 | EVT (Gumbel max) | 2.633→3.937 |
| tropical | current | tropical | 2 | +0.0620 | 0.0620 | 0.868 | EVT (Gumbel max) | 3.019→3.932 |
| tropical | nsa | tropical | 0 | +0.1179 | 0.1179 | 0.913 | EVT (Gumbel max) | 2.253→3.611 |
| tropical | nsa | tropical | 1 | +0.1396 | 0.1396 | 0.943 | EVT (Gumbel max) | 2.175→3.502 |
| tropical | nsa | tropical | 2 | +0.0762 | 0.0762 | 0.883 | EVT (Gumbel max) | 2.539→3.498 |

skipped no-op arms: standard@nsa, reversible@nsa
