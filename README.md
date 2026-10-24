# Factorized Forward Projection

**A replication of the Fashion-MNIST Forward Projection MLP, extended with jointly fitted reduced-rank weight matrices.**

Forward Projection (FP) is a feedback-free, closed-form construction for supervised
networks. It was introduced by O'Shea and Rajendran in
[*Closed-form feedback-free learning with forward projection*](https://doi.org/10.1038/s41467-026-69161-1),
*Nature Communications* **17**, 2414 (2026). This repository:

1. **replicates the paper's dense Fashion-MNIST MLP result** (three 1000-unit ReLU
   hidden layers), and
2. **extends each hidden map with a rank-constrained factorization**, fitted directly
   rather than compressing a dense solution afterward.

No gradients, backpropagation or iterative optimizers are used for the analytical
models. The reduced-rank fit is the exact local optimum of the same regularized
squared-error objective under the rank constraint.

## Headline results

Five paired folds, RTX 4070 Ti SUPER, FP32, identical splits and seeds.

| Model | Test accuracy | Parameters | Estimated inference FLOPs/image | Complete construction time |
|---|---:|---:|---:|---:|
| Dense FP (replication) | **86.28%** | 2,797,010 | 5,594,010 | 0.434 s |
| **Rank-128 FP (extension)** | **86.70%** | **753,362** | **1,506,714** | 1.127 s |
| Rank-64 FP | 86.52% | 383,186 | 766,362 | 1.109 s |
| Rank-256 FP | 86.50% | 1,493,714 | 2,987,418 | 1.096 s |
| Rank-512 FP | 86.37% | 2,974,418 | 5,948,826 | 1.200 s |

The published dense FP accuracy is **86.3% ± 0.1**; our replication gives
**86.28% ± 0.11**. Rank-128 removes **73.07%** of the parameters and estimated
inference FLOPs while raising test accuracy by **0.42 percentage points** over our
dense replication (paired fold delta; descriptive variation, not an independent-test
confidence interval).

The matched-model learning control is the more striking result. Both rank-128 arms use
**exactly the same 753,362-parameter graph**:

| Rank-128 arm | Validation accuracy | Time to first record FP's accuracy |
|---|---:|---:|
| Analytical FP | 87.53% | **1.13 s** |
| Random factors + AdamW | — | **26.37 s** mean |
| Random factors + AdamW, validation-CE-selected | **88.23%** | 38.54 s to selected checkpoint |

Median paired time-to-FP-accuracy ratio: **20.76×** (mean 23.92 ± 8.47×, five folds).
The advantage is **time to the accuracy FP actually reaches**, not a faster route to
AdamW's higher final accuracy.

## What is replicated, what is new

### Replication

The dense construction follows the paper's released MLP:

- 784 → 1000 → 1000 → 1000 → 10, ReLU hidden layers, unrestricted affine head.
- Fixed Gaussian input and label projections; hidden targets
  `sign(A_aug Q) + sign(Y R)`; final head target `2Y − 1`.
- Raw summed ridge with penalty 10 in hidden layers and 1 at the head; the constant
  bias coordinate is penalized.
- Actual fitted ReLU activations are propagated between layers.
- Predictions use learned weights only; labels and projections are absent at inference.

The dense replication also includes ordinary Adam baselines (random initialization and
FP-initialized fine-tuning). See [the replication table](#dense-replication) below.

### Extension

For each hidden map we solve

\[
\min_{\operatorname{rank}(W)\le r,\;b}
\frac{\|AW + \mathbf 1 b - T\|_F^2 + \lambda(\|W\|_F^2 + \|b\|_2^2)}{N},
\qquad W = UV.
\]

Eliminating the penalized bias gives a Schur-complement covariance. We factor it,
whiten the cross-covariance, take the leading rank-\(r\) singular triplets, and
recover \(U,V,b\). This is the **global optimum of the local regularized squared-error
problem** under the rank constraint, up to numerical precision. It is not the ordinary
SVD of an unrestricted dense solution, and it is not a global optimum of the nonlinear
network. A direct numerical check against L-BFGS-B on independent factorized problems
is included in the test suite.

## Reproduction

Python ≥ 3.11. Install (CPU or CUDA PyTorch):

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e .
pip install pytest
pytest -q
```

Fashion-MNIST is downloaded and MD5-verified automatically into `data/fashion-mnist`.

Run the three reported experiments (CUDA recommended; each writes a fresh timestamped
directory under `results/`):

```sh
# 1. Dense FP replication + ordinary Adam controls (five folds)
python experiments/replicate_dense.py --device cuda:0

# 2. Dense and reduced-rank analytical sweep + dense AdamW benchmark
python experiments/rank_sweep.py --device cuda:0

# 3. Matched rank-128 analytical FP versus random-factor AdamW (headline speedup)
python experiments/matched_rank128_adamw.py --device cuda:0
```

For a quick CPU pipeline check (tiny width, one epoch, one fold, not research evidence):

```sh
python experiments/replicate_dense.py --smoke --device cpu --folds 0
python experiments/rank_sweep.py --smoke --device cpu --folds 0
python experiments/matched_rank128_adamw.py --smoke --device cpu --folds 0
```

The archived measurements behind the tables are in
[`results/`](results/README.md), including machine-readable comparisons, verification
records and learning-curve plots. Absolute timings are platform- and I/O-specific;
accuracy and parameter counts are the stable quantities.

## Dense replication

| Checkpoint | Our test accuracy | Paper |
|---|---:|---:|
| FP before fine-tuning | **86.28% ± 0.11** | **86.3% ± 0.1** |
| Random + Adam, validation-selected | 88.20% ± 0.16 | BP 88.4% ± 0.3 |
| FP + Adam, validation-selected | 87.54% ± 0.36 | not reported |

Median paired first-observed speedup for FP over Adam is **22.10×** at 86% accuracy,
**1.17×** at 87%, and **0.86×** at 88%. FP provides a large early-accuracy head start
but does not retain an advantage at the higher thresholds under this matched schedule.
The complete archived numbers are in
[`results/dense-fp/comparison.json`](results/dense-fp/comparison.json).

## Rank sweep

The rank sweep compares analytical FP at ranks 64/128/256/512 with the dense FP
replication and a dense AdamW benchmark. All 25 paired equal-accuracy targets are
reached; the median observed ratios are 10.39×/10.13×/9.75×/9.04× for ranks
64/128/256/512. Dense FP's median is 27.25×, but its complete checkpoint includes a
much cheaper fitting pass; reduced-rank fitting is slower because the exact whitened
SVD dominates.

Lower rank reduces **stored parameters and estimated FLOPs**, not measured inference
latency: the factorized eager implementation adds matrix multiplications and kernel
launches, and short wall-time measurements are noisy. Hidden width remains 1000, so
activation memory is not reduced. See
[`results/rank-sweep/comparison.json`](results/rank-sweep/comparison.json).

## Limitations

- **AdamW eventually reaches higher accuracy.** On the dense model, 88.04% vs FP's
  86.28%; on the matched rank-128 model, 88.23% vs FP's 87.53%. FP's case is
  low-cost useful accuracy, not better convergence quality.
- **Reduced-rank fitting is slower than dense FP** (roughly 0.79–0.86 s vs 0.21 s).
  The exact dense whitened SVD dominates; lower rank does not establish a fitting-time
  saving here.
- **FLOPs are an arithmetic estimate**, not a measured latency win. Inference
  memory is driven by the unchanged hidden width.
- **Rank-512 is not a size win**: it has 6.34% more parameters and FLOPs than dense.
- **The local optimum is not a global network optimum.** Each layer is fitted greedily
  against fixed targets; the final classifier is fitted after the hidden stack.
- **Five folds share one official test set.** Fold variation is sample standard
  deviation, not an independent-test confidence interval.
- Results are for Fashion-MNIST only. They do not establish text learning, larger-network
  scaling, or a general nonlinear-network optimum.

## Future directions

We also experimented with causal convolutional networks and transformer-based language
models. Those adaptations exposed a practical difficulty in extending local closed-form
learning: constructing intermediate targets that are both predictable from a layer's
own inputs and useful for the representations later layers need. In our adaptations,
locally fitted targets did not consistently create the compositional structure required
for multi-step sequence prediction, and the choice of residual scale could suppress the
fitted signal. This is evidence about those adaptations, not a claim that Forward
Projection cannot support convolutional or attention-based architectures. The original
work includes convolutional and vision-transformer demonstrations.

Natural next steps are architecture-aware target construction, iterative or
multi-pass fitting, better initialization/scale alignment between local objectives and
downstream prediction, and larger-scale comparisons that separate target quality from
architecture capacity. See [`docs/future-directions.md`](docs/future-directions.md) for
the measured observations behind this discussion.

## Repository layout

```text
src/factorized_fp/
  fp.py          dense Forward Projection, data loading, evaluation
  rank.py        factorized maps, exact reduced-rank solver, cost accounting
  training.py    ordinary Adam/AdamW controls and random factor initialization
  analysis.py    observed-checkpoint crossings and summaries
  preflight.py   independent numerical verification of the local solver
experiments/     three reproduction entry points
tests/           numerical, architectural and training tests
results/         archived comparisons, verification records and plots
docs/            method notes and future directions
```

## Citation

If you use this code, please cite the original Forward Projection paper and this
repository (see [`CITATION.cff`](CITATION.cff)):

> Chris Bates. *Factorized Forward Projection: Fashion-MNIST replication and
> reduced-rank extension*. Software.
> https://github.com/chrsbats/factorized-forward-projection.

> Robert O'Shea and Bipin Rajendran. “Closed-form feedback-free learning with forward
> projection.” *Nature Communications* **17**, 2414 (2026).
> DOI: 10.1038/s41467-026-69161-1.

The original software release is
`ForwardProjectionDec2025`, commit `29ea819e1661db8d3e4e683fe0beec5ffb321fc7`.

## License

Our code is released under the [MIT License](LICENSE). The original paper is CC-BY-4.0;
its software deposit declares CC-BY-4.0. This repository is an independent
reimplementation and does not redistribute the original code.
