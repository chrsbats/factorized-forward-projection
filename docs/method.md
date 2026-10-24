# Method notes

## Forward Projection in one layer

Let `A` be the actual input activations (with a constant bias column), `Y` the
one-hot labels, and `Q`, `R` fixed Gaussian projections. The hidden target is

```text
T = sign(A Q) + sign(Y R).
```

The inference weights solve the raw ridge problem

```text
min_W,b  ||A W + 1 b - T||_F^2 + lambda (||W||_F^2 + ||b||_2^2).
```

The paper uses `lambda=10` for hidden layers. The final head uses `2Y-1` and
`lambda=1`. Fitting is sequential: after each hidden solve, the actual ReLU
activations of the fitted prefix become the next layer's inputs.

The only random objects are `Q`, `R` and the initial factor placeholders for the
ordinary optimizer control. Hidden inference weights are learned by the closed-form
solve, not random features.

## Reduced-rank extension

Impose `W = U V` with `rank(W) <= r`. The external bias is separate and penalized.
Eliminate it by the Schur complement. With `k = N + lambda` and augmented raw
statistics `S = A_aug^T A_aug`, `D = A_aug^T T`:

```text
G = (S_hh + lambda I - S_h1 S_1h / k) / N
C = (D_h - S_h1 D_1 / k) / N
```

Factor `G = L L^T` and whiten `B = L^{-1} C`. Retain the leading `r` singular
triplets `P_r, Sigma_r, Q_r` of `B`, then

```text
U = L^{-T} P_r sqrt(Sigma_r)
V = sqrt(Sigma_r) Q_r^T
b = (D_1 - S_1h U V) / k.
```

This is the exact global optimum of the local regularized squared-error problem
under the rank constraint, up to numerical precision. It is **not** the truncation of
an unrestricted dense solution, because the whitening metric changes which directions
are optimal. It is also not a global optimum of the nonlinear network: each map is
fitted greedily against fixed targets.

The implementation uses an exact thin SVD. The dense spectrum cost is independent of
`r`; this is why reduced-rank fitting is not faster than dense FP at these sizes.

## Timing protocol

- Device clock is synchronized around timed sections on CUDA.
- An analytical arm's complete time starts before model construction and includes
  fitting, checkpoint serialization and validation/test evaluation.
- An optimizer arm's time starts before model construction and includes every update,
  validation/test evaluation and snapshot.
- Validation cross-entropy selects checkpoints and controls the five-bad-full-epoch
  stopping rule. Test observations never choose settings or stopping.
- The primary rank-128 comparison uses each fold's paired FP **validation** accuracy
  as the target and the first recorded AdamW validation observation that reaches it.
  No accuracy interpolation is used.
- Two 128-token/fold crossing checks use the paper's validation-best checkpoint rather
  than the terminal state where they differ.

## Parameter and FLOP accounting

For a hidden map with input width `d`, output width `h` and rank `r`:

```text
factorized parameters = r(d + h) + h
dense parameters      = d h + h
matrix FLOPs          = 2 r (d + h)   (factorized)
                        2 d h         (dense)
```

Bias and ReLU add one FLOP per output; these are included in the reported estimates.
Multiply-add is counted as two FLOPs. The estimates are arithmetic, not latency
predictions.
