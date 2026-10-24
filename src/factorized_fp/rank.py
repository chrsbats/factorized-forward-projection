"""Jointly fitted reduced-rank Forward Projection.

Each hidden map is parameterized as ``ReLU((H U) V + b)`` with rank r. The
rank constraint applies to ``W = U V``; the external bias is ridge-penalized.
Eliminating that bias gives a Schur-complement covariance. The exact thin SVD is
truncated in the whitened metric, which is the global optimum of the local
regularized squared-error problem (up to numerical precision).
"""

import torch
from torch import nn
from torch.nn import functional as F

from .fp import augment, clock, fp_targets


class FactorizedLinear(nn.Module):
    """H U V + b with no activation between the factors."""

    def __init__(self, inputs, outputs, rank):
        super().__init__()
        if not 1 <= rank <= min(inputs, outputs):
            raise ValueError("rank must be in 1..min(inputs, outputs)")
        self.in_features, self.out_features, self.rank = inputs, outputs, rank
        self.u = nn.Parameter(torch.zeros(inputs, rank))
        self.v = nn.Parameter(torch.zeros(rank, outputs))
        self.bias = nn.Parameter(torch.zeros(outputs))

    def forward(self, inputs):
        return (inputs @ self.u) @ self.v + self.bias


class ReducedRankMLP(nn.Module):
    """Three factorized ReLU hidden layers and an unrestricted affine head."""

    def __init__(self, rank, inputs=784, width=1000, hidden_layers=3, classes=10):
        super().__init__()
        dimensions = [inputs] + [width] * hidden_layers
        self.layers = nn.ModuleList([
            FactorizedLinear(a, b, rank) for a, b in zip(dimensions, dimensions[1:])
        ] + [nn.Linear(dimensions[-1], classes)])

    def forward(self, inputs):
        return self.layers[-1](prefix(self, inputs, len(self.layers) - 1))


def prefix(model, inputs, depth):
    """Actual installed ReLU activations of a fitted prefix, for one batch."""
    for layer in model.layers[:depth]:
        inputs = F.relu(layer(inputs))
    return inputs


@torch.no_grad()
def accumulate(model, inputs, labels, depth, projections=None, *, batch_size=1024):
    """Raw augmented sufficient statistics without retaining hidden features.

    With projections=None this fits the ordinary 2Y-1 classification readout at
    ``depth``. Returned tensors belong to the caller; the model is not modified.
    """
    if batch_size < 1 or len(inputs) == 0 or len(labels) != len(inputs):
        raise ValueError("nonempty paired inputs and positive batch size required")
    device, dtype = inputs.device, inputs.dtype
    dimension = model.layers[depth].in_features
    classes = model.layers[-1].out_features
    outputs = classes if projections is None else projections["label"].shape[1]
    gram = torch.zeros(dimension + 1, dimension + 1, device=device, dtype=dtype)
    cross = torch.zeros(dimension + 1, outputs, device=device, dtype=dtype)
    target_square = torch.zeros((), device=device, dtype=dtype)
    feature_seconds, statistics_seconds, batches = 0., 0., 0
    for start in range(0, len(inputs), batch_size):
        begin = clock(device)
        a = augment(prefix(model, inputs[start:start + batch_size], depth))
        y = labels[start:start + batch_size]
        target = (2 * F.one_hot(y, classes).to(dtype) - 1 if projections is None else
                  fp_targets(a, y, projections["data"], projections["label"]))
        feature_seconds += clock(device) - begin
        begin = clock(device)
        gram.addmm_(a.T, a)
        cross.addmm_(a.T, target)
        target_square += target.square().sum()
        statistics_seconds += clock(device) - begin
        batches += 1
        del a, target, y
    return {"gram": gram, "cross": cross, "target_square": target_square,
            "count": len(inputs)}, {
                "feature_seconds": feature_seconds, "statistics_seconds": statistics_seconds,
                "batches": batches, "max_batch_examples": min(batch_size, len(inputs)),
                "statistics_bytes": (gram.numel() + cross.numel() + 1) * gram.element_size(),
                "prefix_layer_examples": depth * len(inputs)}


def bias_eliminated(statistics, penalty):
    """Mean-normalized covariance/cross-covariance for a free, penalized bias."""
    gram, cross, count = statistics["gram"], statistics["cross"], statistics["count"]
    if penalty <= 0 or count < 1:
        raise ValueError("positive raw ridge penalty and sample count required")
    k = count + penalty
    g = gram[:-1, :-1].clone()
    g.diagonal().add_(penalty)
    g.sub_(torch.outer(gram[:-1, -1], gram[-1, :-1]) / k).div_(count)
    c = (cross[:-1] - torch.outer(gram[:-1, -1], cross[-1]) / k) / count
    return g, c


@torch.no_grad()
def solve_reduced_rank(statistics, rank, *, penalty=10.):
    """Globally optimal local rank-constrained ridge, including the penalized bias."""
    device = statistics["gram"].device
    begin = clock(device)
    g, c = bias_eliminated(statistics, penalty)
    if not 1 <= rank <= min(c.shape):
        raise ValueError("rank exceeds permissible weight rank")
    lower = torch.linalg.cholesky(g)
    whitened = torch.linalg.solve_triangular(lower, c, upper=False)
    whitening_seconds = clock(device) - begin
    begin = clock(device)
    kwargs = {"driver": "gesvd"} if device.type == "cuda" else {}
    p, singular, qt = torch.linalg.svd(whitened, full_matrices=False, **kwargs)
    svd_seconds = clock(device) - begin
    begin = clock(device)
    root = singular[:rank].sqrt()
    u = torch.linalg.solve_triangular(lower.T, p[:, :rank] * root, upper=True)
    v = root[:, None] * qt[:rank]
    gram, cross = statistics["gram"], statistics["cross"]
    bias = (cross[-1] - (gram[-1, :-1] @ u) @ v) / (statistics["count"] + penalty)
    recovery_seconds = clock(device) - begin
    begin = clock(device)
    fitted_white = (lower.T @ u) @ v
    expected_white = (p[:, :rank] * singular[:rank]) @ qt[:rank]
    relative = float((fitted_white - expected_white).norm() / whitened.norm().clamp_min(1e-20))
    tail = singular[rank:].square().sum()
    spectrum_energy = singular.square().sum()
    excess = float(((fitted_white - whitened).square().sum() - tail).abs()
                   / spectrum_energy.clamp_min(1e-20))
    finite = all(bool(torch.isfinite(t).all()) for t in (u, v, bias, singular))
    if not finite or relative > 1e-3 or excess > 1e-3:
        raise ArithmeticError(f"reduced-rank numerical gate failed: reconstruction={relative}, gap={excess}")
    record = {"whitening_seconds": whitening_seconds, "svd_seconds": svd_seconds,
              "recovery_seconds": recovery_seconds, "checks_seconds": clock(device) - begin,
              "raw_penalty": penalty, "mean_penalty": penalty / statistics["count"],
              "rank": rank, "whitened_reconstruction_relative": relative,
              "spectral_optimality_relative": excess,
              "discarded_energy_fraction": float(tail / spectrum_energy.clamp_min(1e-20)),
              "largest_singular": float(singular[0]), "smallest_singular": float(singular[-1]),
              "boundary_singular": float(singular[rank - 1]),
              "cholesky_min_diagonal": float(lower.diagonal().min()),
              "cholesky_max_diagonal": float(lower.diagonal().max()),
              "finite": finite,
              "svd": "exact thin gesvd, leading-r retained" if kwargs else "exact thin LAPACK, leading-r retained"}
    return (u, v, bias), record


@torch.no_grad()
def solve_head(statistics, *, penalty=1.):
    """Unrestricted augmented ridge head with the same raw penalty as dense FP."""
    begin = clock(statistics["gram"].device)
    gram = statistics["gram"].clone()
    gram.diagonal().add_(penalty)
    weights = torch.linalg.solve(gram, statistics["cross"])
    relative = float((gram @ weights - statistics["cross"]).norm()
                     / statistics["cross"].norm().clamp_min(1e-20))
    if not bool(torch.isfinite(weights).all()) or relative > 1e-3:
        raise ArithmeticError(f"output ridge numerical gate failed: {relative}")
    return weights, {"solve_seconds": clock(gram.device) - begin,
                     "stationarity_relative": relative, "raw_penalty": penalty,
                     "mean_penalty": penalty / statistics["count"]}


@torch.no_grad()
def fit_forward(model, inputs, labels, *, seed=1000, batch_size=1024):
    """Sequential analytical fitting; replay raw inputs through the fitted factors.

    Offline artifacts are CPU sufficient statistics and projection matrices, never
    retained hidden-feature datasets. Inference uses only model weights.
    """
    device, dtype = inputs.device, inputs.dtype
    begin = clock(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    records, artifacts = [], []
    for depth, layer in enumerate(model.layers):
        started = clock(device)
        output = depth == len(model.layers) - 1
        projections = None if output else {
            "data": torch.randn(layer.in_features + 1, layer.out_features,
                                generator=generator, device=device, dtype=dtype),
            "label": torch.randn(model.layers[-1].out_features, layer.out_features,
                                 generator=generator, device=device, dtype=dtype)}
        projection_seconds = clock(device) - started
        statistics, record = accumulate(model, inputs, labels, depth, projections, batch_size=batch_size)
        if output:
            weights, solve = solve_head(statistics)
            started = clock(device)
            layer.weight.copy_(weights[:-1].T)
            layer.bias.copy_(weights[-1])
            del weights
        else:
            factors, solve = solve_reduced_rank(statistics, layer.rank)
            started = clock(device)
            for parameter, fitted in zip((layer.u, layer.v, layer.bias), factors):
                parameter.copy_(fitted)
            del factors
        installation_seconds = clock(device) - started
        started = clock(device)
        artifacts.append({"statistics": {k: v.cpu() if isinstance(v, torch.Tensor) else v
                                         for k, v in statistics.items()},
                          "projections": None if output else {k: v.cpu() for k, v in projections.items()}})
        records.append({"layer": depth + 1, "projection_seconds": projection_seconds,
                        **record, **solve, "installation_seconds": installation_seconds,
                        "artifact_transfer_seconds": clock(device) - started})
        del statistics, projections
    return {"seconds": clock(device) - begin, "layers": records,
            "device": str(device), "dtype": str(dtype), "backprop_steps": 0}, artifacts


def inference_cost(model):
    """Exact stored scalars and estimated per-image FLOPs, FMA=2; bias/ReLU=1."""
    matrix_flops, bias_flops, relu_flops = 0, 0, 0
    for index, layer in enumerate(model.layers):
        matrix_flops += 2 * (layer.rank * (layer.in_features + layer.out_features)
                             if isinstance(layer, FactorizedLinear)
                             else layer.in_features * layer.out_features)
        bias_flops += layer.out_features
        if index < len(model.layers) - 1:
            relu_flops += layer.out_features
    parameters = sum(p.numel() for p in model.parameters())
    return {"parameters": parameters, "model_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "matrix_flops_per_image": matrix_flops, "bias_flops_per_image": bias_flops,
            "relu_flops_per_image": relu_flops,
            "estimated_flops_per_image": matrix_flops + bias_flops + relu_flops}
