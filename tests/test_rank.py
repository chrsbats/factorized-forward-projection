"""Reduced-rank optimality, factorization and inference cost."""

import pytest
import torch
from torch.nn import functional as F

from factorized_fp.fp import FPMLP, augment, evaluate, fp_targets
from factorized_fp.preflight import preflight
from factorized_fp.rank import (
    FactorizedLinear, ReducedRankMLP, accumulate, bias_eliminated, fit_forward,
    inference_cost, prefix, solve_reduced_rank,
)


def test_direct_numerical_optimization_and_full_rank_parity():
    report = preflight()
    assert report["pass"] and len(report["fixtures"]) == 3


def test_whitened_truncation_is_optimal_and_penalty_is_raw_over_n():
    torch.manual_seed(4)
    h = torch.randn(70, 6, dtype=torch.float64) * torch.arange(1., 7., dtype=torch.float64)
    t = torch.randn(70, 5, dtype=torch.float64) + 3
    a = augment(h)
    statistics = {"gram": a.T @ a, "cross": a.T @ t, "count": len(h)}
    (u, v, b), record = solve_reduced_rank(statistics, 2)
    g, c = bias_eliminated(statistics, 10.)
    eigen, rotation = torch.linalg.eigh(g)
    root = (rotation * eigen.sqrt()) @ rotation.T
    inverse_root = (rotation / eigen.sqrt()) @ rotation.T
    p, s, qt = torch.linalg.svd(inverse_root @ c, full_matrices=False)
    expected = inverse_root @ ((p[:, :2] * s[:2]) @ qt[:2])
    assert torch.allclose(u @ v, expected, atol=1e-11)
    assert torch.allclose(root @ expected, root @ (u @ v), atol=1e-11)
    assert record["mean_penalty"] == 10. / len(h)
    normal = (a.T @ a + 10 * torch.eye(7)) @ torch.cat((u @ v, b[None])) - a.T @ t
    assert torch.allclose(normal[-1], torch.zeros(5, dtype=torch.float64), atol=1e-11)
    residual = g @ (u @ v) - c
    assert residual.norm() > 1e-3
    assert (residual @ v.T).norm() < 1e-11 and (u.T @ residual).norm() < 1e-11
    with pytest.raises(ValueError):
        solve_reduced_rank(statistics, 7)


def test_factorized_relu_is_after_both_factors_and_forward_is_label_free():
    model = ReducedRankMLP(1, inputs=2, width=2, hidden_layers=1, classes=2).double()
    layer = model.layers[0]
    with torch.no_grad():
        layer.u.copy_(torch.tensor([[1.], [-1.]]))
        layer.v.copy_(torch.tensor([[-2., 3.]]))
        layer.bias.copy_(torch.tensor([.5, -.5]))
        model.layers[-1].weight.copy_(torch.eye(2))
        model.layers[-1].bias.zero_()
    x = torch.tensor([[-1., 1.], [1., -1.]], dtype=torch.float64)
    expected = F.relu((x @ layer.u) @ layer.v + layer.bias)
    assert torch.equal(model(x), expected)
    assert not torch.equal(expected, F.relu(F.relu(x @ layer.u) @ layer.v + layer.bias))
    assert set(model.state_dict()) == {"layers.0.u", "layers.0.v", "layers.0.bias",
                                       "layers.1.weight", "layers.1.bias"}


def test_streamed_fit_replays_actual_fitted_prefix_and_is_deterministic():
    torch.manual_seed(7)
    x = torch.randn(103, 5, dtype=torch.float64)
    y = torch.arange(103) % 3
    model = ReducedRankMLP(3, inputs=5, width=8, classes=3).double()
    record, artifacts = fit_forward(model, x, y, seed=8, batch_size=17)
    hidden = x
    for depth, layer in enumerate(model.layers):
        a = augment(hidden)
        pair, saved = artifacts[depth]["projections"], artifacts[depth]["statistics"]
        target = (2 * F.one_hot(y, 3).double() - 1 if pair is None else
                  fp_targets(a, y, pair["data"], pair["label"]))
        assert torch.allclose(saved["gram"], a.T @ a, atol=1e-10)
        assert torch.allclose(saved["cross"], a.T @ target, atol=1e-10)
        assert record["layers"][depth]["max_batch_examples"] == 17
        hidden = layer(hidden)
        if pair is not None:
            hidden = F.relu(hidden)
    assert torch.allclose(model(x), hidden)
    assert all(p.grad is None for p in model.parameters()) and record["backprop_steps"] == 0
    again = ReducedRankMLP(3, inputs=5, width=8, classes=3).double()
    _, again_artifacts = fit_forward(again, x, y, seed=8, batch_size=17)
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), again.parameters()))
    before = model(x).clone()
    artifacts[0]["projections"]["label"].fill_(1000.)
    assert torch.equal(before, model(x))


def test_conditioning_with_duplicate_features_and_finite_rank_deficient_targets():
    x = torch.arange(30., dtype=torch.float64)[:, None] / 30
    x = torch.cat((x, x, torch.ones_like(x), torch.zeros_like(x)), 1)
    target = torch.cat((x, 2 * x), 1)
    a = augment(x)
    statistics = {"gram": a.T @ a, "cross": a.T @ target, "count": len(x)}
    factors, record = solve_reduced_rank(statistics, 4, penalty=.1)
    weights = torch.cat((factors[0] @ factors[1], factors[2][None]))
    expected = torch.linalg.solve(a.T @ a + .1 * torch.eye(5), a.T @ target)
    assert torch.allclose(weights, expected, atol=1e-11)
    assert record["finite"] and record["cholesky_min_diagonal"] > 0


def test_parameter_bytes_and_flops_include_bias_and_dense_head():
    dense = inference_cost(FPMLP())
    assert dense["parameters"] == 2797010
    assert dense["estimated_flops_per_image"] == 2 * 2794000 + 3010 + 3000
    for rank in (64, 128, 256, 512):
        cost = inference_cost(ReducedRankMLP(rank))
        expected = rank * (784 + 1000 + 4 * 1000) + 3000 + 10010
        assert cost["parameters"] == expected and cost["model_bytes"] == 4 * expected
        assert cost["matrix_flops_per_image"] == 2 * (expected - 3010)
    assert inference_cost(ReducedRankMLP(512))["parameters"] > dense["parameters"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_exact_solver_and_streamed_inference():
    assert preflight("cuda:0", torch.float32)["pass"]
    x = torch.randn(131, 5, device="cuda")
    y = torch.arange(131, device="cuda") % 3
    model = ReducedRankMLP(3, inputs=5, width=11, classes=3).cuda()
    record, _ = fit_forward(model, x, y, batch_size=19)
    assert record["device"] == "cuda:0"
    assert evaluate(model, x, y)["examples"] == 131
