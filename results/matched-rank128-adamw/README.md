# Matched rank-128 FP versus AdamW (archived)

Both arms use the same 753,362-parameter rank-128 graph.

| Arm | Validation accuracy | Time |
|---|---:|---:|
| Analytical FP | 87.531% ± 0.335 | 1.127 s complete |
| Random factors + AdamW, first equal accuracy | — | 26.368 ± 7.223 s |
| Random factors + AdamW, CE-selected | 88.225% ± 0.841 | 38.542 s to selected |

Median paired observed-time ratio **20.76×**; mean **23.92 ± 8.47×**; all five folds
reach the FP target. AdamW's full campaign (five bad full epochs) is 55.277 ± 20.469 s.

`comparison.json` also contains dense controls: dense AdamW first records the same
rank-128-FP targets in 13.030 ± 2.126 s (median ratio 11.78×), so the factorized model
is *not* faster to those targets than a dense AdamW run. `validation-*.svg` shows raw
and CE-selected curves. `verification.json` records checkpoint replay.
