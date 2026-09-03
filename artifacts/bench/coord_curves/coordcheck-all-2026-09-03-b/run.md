# coord-check — coordcheck-all-2026-09-03-b

- widths [64, 128, 256, 512, 1024, 2048] · seeds [3, 4, 5] · device cpu · schema mgr.bench.coord_curves.v1
- flat (|slope| <= 0.05) means correctly parameterized in width; one artifact per (mechanism, arm, seed)

| mechanism | arm | ffn | seed | slope | abs slope | R2 | class | rms first→last |
|---|---|---|---|---|---|---|---|---|
| gauge | current | standard | 3 | +0.0002 | 0.0002 | 0.298 | CLT (assumed) | 1.034→1.035 |
| gauge | current | standard | 4 | +0.0005 | 0.0005 | 0.040 | CLT (assumed) | 1.037→1.035 |
| gauge | current | standard | 5 | -0.0001 | 0.0001 | 0.003 | CLT (assumed) | 1.033→1.034 |
| braid | current | standard | 3 | -0.0015 | 0.0015 | 0.482 | CLT (assumed) | 1.046→1.042 |
| braid | current | standard | 4 | -0.0006 | 0.0006 | 0.101 | CLT (assumed) | 1.047→1.043 |
| braid | current | standard | 5 | +0.0011 | 0.0011 | 0.488 | CLT (assumed) | 1.038→1.042 |
| ultrametric | current | standard | 3 | -0.0004 | 0.0004 | 0.219 | branching / geometric (LCP depth) | 1.040→1.038 |
| ultrametric | current | standard | 4 | +0.0005 | 0.0005 | 0.237 | branching / geometric (LCP depth) | 1.034→1.036 |
| ultrametric | current | standard | 5 | +0.0008 | 0.0008 | 0.689 | branching / geometric (LCP depth) | 1.033→1.036 |
| hyperbolic | current | standard | 3 | +0.0002 | 0.0002 | 0.093 | radial / curvature-gated Lorentz energy | 1.032→1.033 |
| hyperbolic | current | standard | 4 | -0.0002 | 0.0002 | 0.035 | radial / curvature-gated Lorentz energy | 1.031→1.032 |
| hyperbolic | current | standard | 5 | +0.0005 | 0.0005 | 0.338 | radial / curvature-gated Lorentz energy | 1.031→1.033 |
| quaternion | current | standard | 3 | +0.0002 | 0.0002 | 0.151 | isometry (normed algebra) | 1.038→1.039 |
| quaternion | current | standard | 4 | -0.0003 | 0.0003 | 0.126 | isometry (normed algebra) | 1.039→1.038 |
| quaternion | current | standard | 5 | +0.0006 | 0.0006 | 0.353 | isometry (normed algebra) | 1.036→1.039 |
| octonion | current | standard | 3 | -0.0001 | 0.0001 | 0.019 | isometry (normed algebra) | 1.039→1.039 |
| octonion | current | standard | 4 | +0.0000 | 0.0000 | 0.006 | isometry (normed algebra) | 1.038→1.038 |
| octonion | current | standard | 5 | -0.0000 | 0.0000 | 0.001 | isometry (normed algebra) | 1.039→1.039 |
| simplicial | current | standard | 3 | -0.0003 | 0.0003 | 0.080 | CLT (assumed) | 1.052→1.050 |
| simplicial | current | standard | 4 | -0.0002 | 0.0002 | 0.047 | CLT (assumed) | 1.049→1.050 |
| simplicial | current | standard | 5 | +0.0011 | 0.0011 | 0.644 | CLT (assumed) | 1.046→1.051 |
| fractal | current | standard | 3 | -0.0005 | 0.0005 | 0.285 | CLT (assumed) | 1.037→1.035 |
| fractal | current | standard | 4 | +0.0005 | 0.0005 | 0.200 | CLT (assumed) | 1.031→1.033 |
| fractal | current | standard | 5 | +0.0008 | 0.0008 | 0.738 | CLT (assumed) | 1.031→1.034 |
| surreal | current | standard | 3 | -0.0004 | 0.0004 | 0.043 | CLT (assumed) | 1.128→1.123 |
| surreal | current | standard | 4 | -0.0013 | 0.0013 | 0.216 | CLT (assumed) | 1.123→1.123 |
| surreal | current | standard | 5 | -0.0002 | 0.0002 | 0.008 | CLT (assumed) | 1.120→1.124 |
