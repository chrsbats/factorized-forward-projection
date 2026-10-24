# Dense FP replication (archived)

Five paired folds of the paper's dense 784→1000→1000→1000→10 Fashion-MNIST MLP.

- FP before fine-tuning: **86.28% ± 0.11** test accuracy (paper 86.3% ± 0.1).
- Random + Adam, validation-selected: **88.20% ± 0.16**.
- FP + Adam, validation-selected: **87.54% ± 0.36**.
- FP fit time: 0.268 ± 0.209 s; complete initial checkpoint 0.487 s.
- Median paired first-observed speedup at 86% accuracy: **22.10×**; 1.17× at 87%,
  0.86× at 88%.

`comparison.json` contains all five folds and the observed threshold brackets.
`verification.json` records the saved-snapshot and normal-equation checks.
`accuracy-*.svg` and `ce-*.svg` are the learning-curve plots.

Source run: RTX 4070 Ti SUPER, CUDA, FP32, TF32 disabled, one CPU BLAS thread.
