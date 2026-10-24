# Archived measurements

These JSON and SVG files are the compact evidence from the original single-machine
runs that produced the README tables. They are kept for inspection and are not needed
to run the code.

| Directory | Content |
|---|---|
| `dense-fp/` | Five-fold dense FP replication, random Adam and FP+Adam controls; comparison, verification, suite metadata and accuracy/CE plots. |
| `rank-sweep/` | Dense and rank-64/128/256/512 analytical sweep, dense AdamW benchmark, numerical preflight, verification and plots. |
| `matched-rank128-adamw/` | Matched rank-128 analytical FP versus random-factor AdamW, validation crossings, verification and learning-curve plots. |

The files retain the original run's naming and absolute-path-independent metadata.
They include five-fold comparisons, per-layer fit components, inference/cost
accounting and saved-checkpoint verification summaries. Large model checkpoints and
the Fashion-MNIST archives are intentionally excluded.

Environment: NVIDIA RTX 4070 Ti SUPER, CUDA, FP32, TF32 disabled, one CPU
BLAS/OpenMP thread, `torch.linalg` on the GPU. Accuracy and parameter counts are
platform-independent; wall times are not.

The reproduction scripts write new timestamped directories under `results/`; they do
not overwrite these archives.
