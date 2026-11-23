# probe-charges — charge-1e12-train4-fit0

- checkpoint: `/Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/charge-measurement/artifacts/campaigns/charges-group-1e12/rmatrix-s4/checkpoints` @ step 4780
- task: group · examples: 240 · seed: 0
- features: per-head charge observables `<v, t(theta_k) v>` of the final braid layer

| category | n | classes | floor | linear test | mlp test | decodable@2x |
|---|---|---|---|---|---|---|
| a5 | 43 | 27 | 0.125 | 0.375 | 0.375 | True |
| s3 | 53 | 6 | 0.200 | 0.000 | 0.300 | False |
| s5 | 54 | 28 | 0.300 | 0.400 | 0.500 | False |
| z60 | 47 | 21 | 0.222 | 0.111 | 0.111 | False |

Both halves of the protected-memory story are required: conservation (certify: braid.rmatrix_mass_partition_charge_conserved) AND decodability (this probe). If conservation holds but decodability fails, the conclusion is that the integrable structure conserves the wrong quantities for this task - a result, not a bug.
