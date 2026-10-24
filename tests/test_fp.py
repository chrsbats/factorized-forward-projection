"""Dense Forward Projection targets, ridge solve, propagation and cost."""

import struct

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from factorized_fp.fp import (
    FPMLP, augment, evaluate, fit_forward, fit_ridge, fp_targets, inference_cost_dense, parse_idx,
)


def test_idx_unsigned_byte_decoding_and_validation():
    values = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    raw = struct.pack(">HBBIII", 0, 8, 3, 2, 2, 3) + values.tobytes()
    assert np.array_equal(parse_idx(raw), values)
    with pytest.raises(ValueError):
        parse_idx(raw[:-1])
    with pytest.raises(ValueError):
        parse_idx(struct.pack(">HBB", 0, 9, 1) + raw[4:])


def test_targets_match_separate_signed_gaussian_projections():
    x = torch.tensor([[2., -1., 1.], [-3., 4., 1.]], dtype=torch.float64)
    q = torch.tensor([[1., -1.], [2., 3.], [-.5, .25]], dtype=torch.float64)
    r = torch.tensor([[-2., 1.], [4., -3.]], dtype=torch.float64)
    labels = torch.tensor([1, 0])
    expected = torch.sign(x @ q) + torch.sign(F.one_hot(labels, 2).double() @ r)
    assert torch.equal(fp_targets(x, labels, q, r), expected)
    assert not torch.equal(expected, torch.sign(x @ q + r[labels]))


def test_raw_ridge_penalizes_bias_and_matches_dense_reference():
    rng = np.random.default_rng(4)
    x = np.column_stack((rng.normal(size=(61, 5)) * np.array([.1, 2, 30, 1, 4]), np.ones(61)))
    target = rng.normal(size=(61, 3)) + 2.
    weights, diagnostics = fit_ridge(torch.from_numpy(x), torch.from_numpy(target), 10.)
    expected = np.linalg.solve(x.T @ x + 10 * np.eye(6), x.T @ target)
    assert np.allclose(weights, expected, atol=1e-11)
    assert diagnostics["stationarity_relative"] < 1e-12


def test_sequential_fp_matches_independent_numpy_reference():
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(96, 4, generator=generator, dtype=torch.float64)
    labels = torch.randint(0, 3, (96,), generator=generator)
    model = FPMLP(inputs=4, width=7, classes=3).double()
    record, projections = fit_forward(model, x, labels, seed=3)
    hidden = x.numpy()
    onehot = np.eye(3)[labels.numpy()]
    for index, layer in enumerate(model.layers):
        a = np.column_stack((hidden, np.ones(len(hidden))))
        if index < 3:
            pair = projections[index]
            target = np.sign(a @ pair["data"].numpy()) + np.sign(onehot @ pair["label"].numpy())
            penalty = 10.
        else:
            target, penalty = 2 * onehot - 1, 1.
        expected = np.linalg.solve(a.T @ a + penalty * np.eye(a.shape[1]), a.T @ target)
        assert np.allclose(layer.weight.detach().numpy(), expected[:-1].T, atol=1e-11)
        assert np.allclose(layer.bias.detach().numpy(), expected[-1], atol=1e-11)
        hidden = a @ expected
        if index < 3:
            hidden = np.maximum(hidden, 0)
    assert np.allclose(model(x).detach(), hidden, atol=1e-11)
    assert [r["penalty"] for r in record["layers"]] == [10., 10., 10., 1.]
    assert record["backprop_steps"] == 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_dense_parameter_and_flop_accounting():
    model = FPMLP()
    assert sum(p.numel() for p in model.parameters()) == 2797010
    assert inference_cost_dense(model)["estimated_flops_per_image"] == 2 * 2794000 + 3010 + 3000


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_fitting_and_metrics():
    x = torch.randn(128, 5, device="cuda")
    y = torch.arange(128, device="cuda") % 3
    model = FPMLP(inputs=5, width=12, classes=3).cuda()
    records, _ = fit_forward(model, x, y, seed=2)
    assert max(r["stationarity_relative"] for r in records["layers"]) < 1e-3
    result = evaluate(model, x, y, batch_size=17)
    assert result["examples"] == 128 and np.isfinite(result["ce"])
