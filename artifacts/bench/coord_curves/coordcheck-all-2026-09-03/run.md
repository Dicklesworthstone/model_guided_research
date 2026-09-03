# coord-check — coordcheck-all-2026-09-03

- widths [64, 128, 256, 512, 1024, 2048] · seeds [0, 1, 2] · device cpu · schema mgr.bench.coord_curves.v1
- flat (|slope| <= 0.05) means correctly parameterized in width; one artifact per (mechanism, arm, seed)

| mechanism | arm | ffn | seed | slope | abs slope | R2 | class | rms first→last |
|---|---|---|---|---|---|---|---|---|
| gauge | current | standard | 0 | +0.0001 | 0.0001 | 0.019 | CLT (assumed) | 1.034→1.035 |
| gauge | current | standard | 1 | -0.0001 | 0.0001 | 0.002 | CLT (assumed) | 1.038→1.035 |
| gauge | current | standard | 2 | +0.0019 | 0.0019 | 0.532 | CLT (assumed) | 1.027→1.034 |
| braid | current | standard | 0 | +0.0016 | 0.0016 | 0.469 | CLT (assumed) | 1.036→1.043 |
| braid | current | standard | 1 | -0.0006 | 0.0006 | 0.140 | CLT (assumed) | 1.046→1.042 |
| braid | current | standard | 2 | +0.0009 | 0.0009 | 0.334 | CLT (assumed) | 1.040→1.045 |
| ultrametric | current | standard | 0 | +0.0018 | 0.0018 | 0.482 | branching / geometric (LCP depth) | 1.028→1.037 |
| ultrametric | current | standard | 1 | +0.0024 | 0.0024 | 0.568 | branching / geometric (LCP depth) | 1.026→1.037 |
| ultrametric | current | standard | 2 | +0.0014 | 0.0014 | 0.359 | branching / geometric (LCP depth) | 1.029→1.036 |
| hyperbolic | current | standard | 0 | +0.0013 | 0.0013 | 0.569 | radial / curvature-gated Lorentz energy | 1.026→1.032 |
| hyperbolic | current | standard | 1 | +0.0013 | 0.0013 | 0.677 | radial / curvature-gated Lorentz energy | 1.029→1.032 |
| hyperbolic | current | standard | 2 | +0.0009 | 0.0009 | 0.261 | radial / curvature-gated Lorentz energy | 1.029→1.032 |
| quaternion | current | standard | 0 | +0.0008 | 0.0008 | 0.585 | isometry (normed algebra) | 1.034→1.037 |
| quaternion | current | standard | 1 | +0.0019 | 0.0019 | 0.773 | isometry (normed algebra) | 1.034→1.038 |
| quaternion | current | standard | 2 | +0.0008 | 0.0008 | 0.206 | isometry (normed algebra) | 1.037→1.038 |
| quaternion | nsa | standard | 0 | +0.0008 | 0.0008 | 0.585 | isometry (normed algebra) | 1.034→1.037 |
| quaternion | nsa | standard | 1 | +0.0019 | 0.0019 | 0.773 | isometry (normed algebra) | 1.034→1.038 |
| quaternion | nsa | standard | 2 | +0.0008 | 0.0008 | 0.206 | isometry (normed algebra) | 1.037→1.038 |
| octonion | current | standard | 0 | +0.0008 | 0.0008 | 0.615 | isometry (normed algebra) | 1.035→1.037 |
| octonion | current | standard | 1 | +0.0015 | 0.0015 | 0.637 | isometry (normed algebra) | 1.035→1.038 |
| octonion | current | standard | 2 | +0.0006 | 0.0006 | 0.176 | isometry (normed algebra) | 1.037→1.038 |
| octonion | nsa | standard | 0 | +0.0008 | 0.0008 | 0.615 | isometry (normed algebra) | 1.035→1.037 |
| octonion | nsa | standard | 1 | +0.0015 | 0.0015 | 0.637 | isometry (normed algebra) | 1.035→1.038 |
| octonion | nsa | standard | 2 | +0.0006 | 0.0006 | 0.176 | isometry (normed algebra) | 1.037→1.038 |
| simplicial | current | standard | 0 | +0.0016 | 0.0016 | 0.417 | CLT (assumed) | 1.041→1.048 |
| simplicial | current | standard | 1 | +0.0008 | 0.0008 | 0.452 | CLT (assumed) | 1.048→1.049 |
| simplicial | current | standard | 2 | +0.0013 | 0.0013 | 0.258 | CLT (assumed) | 1.045→1.049 |
| fractal | current | standard | 0 | +0.0017 | 0.0017 | 0.383 | CLT (assumed) | 1.026→1.034 |
| fractal | current | standard | 1 | +0.0022 | 0.0022 | 0.546 | CLT (assumed) | 1.025→1.035 |
| fractal | current | standard | 2 | +0.0012 | 0.0012 | 0.300 | CLT (assumed) | 1.027→1.034 |
| surreal | current | standard | 0 | +0.0001 | 0.0001 | 0.021 | CLT (assumed) | 1.120→1.120 |
| surreal | current | standard | 1 | +0.0026 | 0.0026 | 0.576 | CLT (assumed) | 1.110→1.122 |
| surreal | current | standard | 2 | +0.0023 | 0.0023 | 0.427 | CLT (assumed) | 1.110→1.121 |

skipped no-op arms: gauge@nsa, braid@nsa, ultrametric@nsa, hyperbolic@nsa, simplicial@nsa, fractal@nsa, surreal@nsa
