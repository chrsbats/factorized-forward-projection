"""Dense Fashion-MNIST Forward Projection (FP) replication.

This reconstructs the MLP experiment of O'Shea and Rajendran, "Closed-form
feedback-free learning with forward projection", Nature Communications 17, 2414
(2026), DOI 10.1038/s41467-026-69161-1. The implementation follows the released
MLP construction: raw ridge on signed input/label projections, a penalized bias,
ReLU hidden layers, and a final ridge head fitted to 2Y-1.
"""

import gzip
import hashlib
from pathlib import Path
import struct
from time import perf_counter
from urllib.request import Request, urlopen

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
DATA_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"


def clock(device):
    """Wall clock, synchronized on CUDA."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return perf_counter()


def parse_idx(raw):
    """Decode an unsigned-byte IDX array; reject inconsistent headers/payloads."""
    if len(raw) < 4:
        raise ValueError("truncated IDX header")
    zero, kind, dimensions = struct.unpack(">HBB", raw[:4])
    header = 4 + 4 * dimensions
    if zero != 0 or kind != 8 or dimensions == 0 or len(raw) < header:
        raise ValueError("invalid unsigned-byte IDX header")
    shape = struct.unpack(">" + "I" * dimensions, raw[4:header])
    if any(size == 0 for size in shape) or len(raw) - header != int(np.prod(shape)):
        raise ValueError("IDX dimensions and payload differ")
    return np.frombuffer(raw, dtype=np.uint8, offset=header).copy().reshape(shape)


def load_fashion(directory):
    """Download/verify official Fashion-MNIST and return float pixel tensors."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays, manifest = {}, {}
    for name, expected in FILES.items():
        path = directory / name
        if not path.exists():
            request = Request(DATA_URL + name, headers={"User-Agent": "factorized-forward-projection"})
            with urlopen(request, timeout=180) as response:
                payload = response.read()
            if hashlib.md5(payload).hexdigest() != expected:
                raise ValueError(f"download checksum differs: {name}")
            path.write_bytes(payload)
        payload = path.read_bytes()
        if hashlib.md5(payload).hexdigest() != expected:
            raise ValueError(f"cached Fashion-MNIST checksum differs: {name}")
        manifest[name] = {"md5": expected, "sha256": hashlib.sha256(payload).hexdigest(),
                          "bytes": len(payload), "url": DATA_URL + name}
        arrays[name] = parse_idx(gzip.decompress(payload))
    data = {}
    for split, prefix, count in (("train", "train", 60000), ("test", "t10k", 10000)):
        images = arrays[prefix + "-images-idx3-ubyte.gz"]
        labels = arrays[prefix + "-labels-idx1-ubyte.gz"]
        if images.shape != (count, 28, 28) or labels.shape != (count,) or labels.max() >= 10:
            raise ValueError("official Fashion-MNIST dimensions/labels differ")
        data[split + "_x"] = torch.from_numpy(images).reshape(count, 784).float().div_(255.)
        data[split + "_y"] = torch.from_numpy(labels).long()
    return data, manifest


class FPMLP(nn.Module):
    """Dense ReLU MLP with the paper's Fashion-MNIST shape by default."""

    def __init__(self, inputs=784, width=1000, hidden_layers=3, classes=10):
        super().__init__()
        dimensions = [inputs] + [width] * hidden_layers + [classes]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dimensions, dimensions[1:]))

    def forward(self, inputs):
        for layer in self.layers[:-1]:
            inputs = F.relu(layer(inputs))
        return self.layers[-1](inputs)


def augment(inputs):
    """Append the constant feature whose coefficient is the bias."""
    return torch.cat((inputs, torch.ones_like(inputs[:, :1])), dim=1)


def fp_targets(inputs, labels, data_projection, label_projection):
    """Exact sign(A_aug Q) + sign(Y R); one-hot Y is gathered, never materialized."""
    return torch.sign(inputs @ data_projection) + torch.sign(label_projection[labels])


@torch.no_grad()
def fit_ridge(inputs, targets, penalty):
    """Raw summed ridge; the constant bias coordinate is penalized too."""
    device = inputs.device
    begin = clock(device)
    gram, cross = inputs.T @ inputs, inputs.T @ targets
    gram.diagonal().add_(penalty)
    statistics_seconds = clock(device) - begin
    begin = clock(device)
    weights = torch.linalg.solve(gram, cross)
    solve_seconds = clock(device) - begin
    relative = float((gram @ weights - cross).norm() / cross.norm().clamp_min(1e-20))
    if not bool(torch.isfinite(weights).all()) or relative > 1e-3:
        raise ArithmeticError(f"ridge numerical gate failed: {relative}")
    return weights, {"statistics_seconds": statistics_seconds, "solve_seconds": solve_seconds,
                     "stationarity_relative": relative}


@torch.no_grad()
def fit_forward(model, inputs, labels, *, seed=0):
    """Fit all maps sequentially, propagating only actual current activations."""
    device, dtype = inputs.device, inputs.dtype
    begin = clock(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    hidden, records, projections = inputs, [], []
    classes = model.layers[-1].out_features
    for index, layer in enumerate(model.layers):
        started = clock(device)
        augmented = augment(hidden)
        output = index == len(model.layers) - 1
        if output:
            target = 2 * F.one_hot(labels, classes).to(dtype) - 1
        else:
            q = torch.randn(augmented.shape[1], layer.out_features, device=device,
                            dtype=dtype, generator=generator)
            r = torch.randn(classes, layer.out_features, device=device, dtype=dtype, generator=generator)
            target = fp_targets(augmented, labels, q, r)
            projections.append({"data": q, "label": r})
        target_seconds = clock(device) - started
        weights, record = fit_ridge(augmented, target, 1. if output else 10.)
        started = clock(device)
        layer.weight.copy_(weights[:-1].T)
        layer.bias.copy_(weights[-1])
        parity = float((layer(hidden[:256]) - augmented[:256] @ weights).abs().max())
        if parity > 2e-4:
            raise ArithmeticError(f"affine installation differs: {parity}")
        if not output:
            hidden = F.relu(layer(hidden))
        records.append({"layer": index + 1, "penalty": 1. if output else 10.,
                        "target_seconds": target_seconds, **record,
                        "installation_max_abs": parity,
                        "installation_propagation_seconds": clock(device) - started})
        del target, augmented, weights
    return {"seconds": clock(device) - begin, "layers": records, "backprop_steps": 0}, projections


@torch.no_grad()
def evaluate(model, inputs, labels, batch_size=4096):
    device = inputs.device
    total = torch.zeros((), device=device, dtype=torch.float64)
    classes = model.layers[-1].out_features
    class_correct = torch.zeros(classes, dtype=torch.long, device=device)
    for start in range(0, len(inputs), batch_size):
        x, y = inputs[start:start + batch_size], labels[start:start + batch_size]
        logits = model(x)
        total += F.cross_entropy(logits, y, reduction="sum").double()
        correct = logits.argmax(-1) == y
        class_correct += torch.bincount(y[correct], minlength=classes)
    counts = torch.bincount(labels, minlength=classes)
    return {"ce": float(total / len(inputs)), "accuracy": float(class_correct.sum().double() / len(inputs)),
            "macro_accuracy": float((class_correct.double() / counts.clamp_min(1)).mean()),
            "examples": len(inputs)}


def inference_cost_dense(model):
    """Stored parameters and estimated per-image FLOPs (FMA=2, bias/ReLU=1)."""
    matrix_flops = sum(2 * layer.in_features * layer.out_features for layer in model.layers)
    bias_flops = sum(layer.out_features for layer in model.layers)
    relu_flops = sum(layer.out_features for layer in model.layers[:-1])
    parameters = sum(p.numel() for p in model.parameters())
    return {"parameters": parameters, "model_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "estimated_flops_per_image": matrix_flops + bias_flops + relu_flops}
