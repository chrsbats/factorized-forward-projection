"""Random factor initialization, Adam training, selection and crossings."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from factorized_fp.analysis import crossing
from factorized_fp.fp import evaluate
from factorized_fp.rank import ReducedRankMLP, inference_cost
from factorized_fp.training import make_optimizer, random_reduced_rank, train


def test_random_control_is_the_same_rank_graph_with_deterministic_nonzero_factors():
    model, again = random_reduced_rank(128, seed=0), random_reduced_rank(128, seed=0)
    assert type(model) is ReducedRankMLP
    assert inference_cost(model)["parameters"] == 753362
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), again.parameters()))
    assert not torch.equal(model.layers[0].u, random_reduced_rank(128, seed=1).layers[0].u)
    assert all(p.requires_grad and bool(torch.isfinite(p).all()) for p in model.parameters())
    x = torch.randn(4, 784)
    hidden = x
    for layer in model.layers[:-1]:
        assert layer.rank == 128 and layer.u.norm() > 0 and layer.v.norm() > 0
        hidden = F.relu((hidden @ layer.u) @ layer.v + layer.bias)
    assert torch.equal(model(x), model.layers[-1](hidden))


def test_training_saves_history_selects_by_validation_and_restores_terminal(tmp_path):
    torch.manual_seed(9)
    x, y = torch.randn(37, 784), torch.arange(37) % 10
    splits = ((x, y), (x, y), (x, y))
    model = random_reduced_rank(4, width=16, seed=0)
    optimizer = make_optimizer(model, "adamw", lr=.001)
    result = train(model, optimizer, splits, batch_size=10, max_epochs=1, patience=5,
                   checkpoint_dir=tmp_path)
    assert len(result["history"]) == 5
    assert result["terminal"]["training_examples"] == 37
    assert result["selected"]["validation"]["ce"] == min(
        row["validation"]["ce"] for row in result["history"])
    terminal = torch.load(tmp_path / result["terminal"]["checkpoint"], weights_only=True)
    restored = random_reduced_rank(4, width=16, seed=1)
    restored.load_state_dict(terminal)
    assert abs(evaluate(restored, x, y, 16)["ce"] - result["terminal"]["test"]["ce"]) < 1e-6


def test_crossing_uses_recorded_validation_and_keeps_preceding_observation():
    history = [
        {"cumulative_seconds": 1., "validation": {"accuracy": .86, "ce": .5}, "checkpoint": "initial"},
        {"cumulative_seconds": 2., "validation": {"accuracy": .85, "ce": .4}, "checkpoint": "lower-ce"},
        {"cumulative_seconds": 3., "validation": {"accuracy": .88, "ce": .45}, "checkpoint": "higher"},
    ]
    hit = crossing(history, "accuracy", .88, higher_is_better=True)
    assert hit["observed"]["checkpoint"] == "higher"
    assert hit["previous"]["checkpoint"] == "lower-ce"
    assert crossing(history, "accuracy", .90, higher_is_better=True) is None
    first = crossing(history, "accuracy", .86, higher_is_better=True)
    assert first["previous"] is None
    assert "interpolated_seconds" not in hit
