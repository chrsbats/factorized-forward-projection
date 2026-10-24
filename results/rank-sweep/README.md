# Reduced-rank sweep (archived)

Five paired folds; dense FP and ranks 64/128/256/512, plus a dense AdamW benchmark.

| Arm | Test accuracy | Parameters | Estimated FLOPs/image | Complete s | Median paired ratio |
|---|---:|---:|---:|---:|---:|
| Dense | 86.280% | 2,797,010 | 5,594,010 | 0.434 | 27.25× |
| Rank64 | 86.522% | 383,186 | 766,362 | 1.109 | 10.39× |
| Rank128 | 86.696% | 753,362 | 1,506,714 | 1.127 | 10.13× |
| Rank256 | 86.504% | 1,493,714 | 2,987,418 | 1.096 | 9.75× |
| Rank512 | 86.374% | 2,974,418 | 5,948,826 | 1.200 | 9.04× |

Dense AdamW selected accuracy: 88.042% (CE 0.34116), full campaign 34.302 s.
`comparison.json` retains per-fold trajectories, fit components, prefix diagnostics
and inference profiling. `preflight.json` is the independent numerical check of the
reduced-rank solver. `accuracy-*.svg` shows the curves.

The reduced-rank fits are slower than dense FP because the exact whitened SVD
dominates. Lower rank reduces parameters and estimated FLOPs, not measured latency or
activation memory.
