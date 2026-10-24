"""Fashion-MNIST Forward Projection replication and reduced-rank extension."""

from .fp import FPMLP, evaluate, fit_forward as fit_dense, load_fashion
from .rank import ReducedRankMLP, fit_forward as fit_reduced_rank, inference_cost

__all__ = [
    "FPMLP", "ReducedRankMLP", "evaluate", "fit_dense", "fit_reduced_rank",
    "inference_cost", "load_fashion",
]
