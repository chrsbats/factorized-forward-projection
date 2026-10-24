"""Independent numerical verification of the reduced-rank local solver."""

import argparse
import json

import numpy as np
from scipy.optimize import minimize
import torch

from .rank import solve_reduced_rank


def preflight(device="cpu", dtype=torch.float64):
    """Compare the exact whitened truncation with direct factor optimization."""
    rng = np.random.default_rng(22)
    reports = []
    for seed in range(3):
        h = rng.normal(size=(43, 6)) * np.array([.1, 1, 3, .5, 2, 1]) + .7
        t = rng.normal(size=(43, 5)) + np.arange(5)
        a = np.column_stack((h, np.ones(len(h))))
        statistics = {"gram": torch.tensor(a.T @ a, device=device, dtype=dtype),
                      "cross": torch.tensor(a.T @ t, device=device, dtype=dtype), "count": len(h)}
        factors, info = solve_reduced_rank(statistics, 2, penalty=10.)
        u, v, b = (value.cpu().double().numpy() for value in factors)

        def objective(flat):
            u0, v0, b0 = flat[:12].reshape(6, 2), flat[12:22].reshape(2, 5), flat[22:]
            w = u0 @ v0
            residual = h @ w + b0 - t
            loss = (np.sum(residual**2) + 10 * (np.sum(w**2) + np.sum(b0**2))) / len(h)
            gradient = 2 * (h.T @ residual + 10 * w) / len(h)
            jac = np.concatenate(((gradient @ v0.T).ravel(), (u0.T @ gradient).ravel(),
                                  2 * (residual.sum(0) + 10 * b0) / len(h)))
            return loss, jac

        exact = objective(np.concatenate((u.ravel(), v.ravel(), b)))[0]
        optimized = minimize(objective, rng.normal(scale=.2, size=27), jac=True,
                             method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-14, "gtol": 1e-9})
        relative = abs(optimized.fun - exact) / max(1., exact)
        full, _ = solve_reduced_rank(statistics, 5, penalty=10.)
        weights = np.vstack(((full[0] @ full[1]).cpu().double().numpy(),
                             full[2].cpu().double().numpy()))
        dense = np.linalg.solve(a.T @ a + 10 * np.eye(7), a.T @ t)
        full_error = float(np.linalg.norm(weights - dense) / np.linalg.norm(dense))
        normal = float(np.linalg.norm((a.T @ a + 10 * np.eye(7)) @ weights - a.T @ t)
                       / np.linalg.norm(a.T @ t))
        bias_residual = float(np.linalg.norm((h @ (u @ v) + b - t).sum(0) + 10 * b)
                              / np.linalg.norm(a.T @ t))
        tolerance = 1e-9 if dtype == torch.float64 else 2e-5
        if not optimized.success or max(relative, full_error, normal, bias_residual) > tolerance:
            raise AssertionError({"optimization": str(optimized.message),
                                  "objective_relative": relative, "full_rank_relative": full_error,
                                  "normal": normal, "bias": bias_residual})
        reports.append({"fixture": seed, "objective": exact,
                        "direct_optimized_objective": float(optimized.fun),
                        "optimizer_success": bool(optimized.success),
                        "optimizer_iterations": optimized.nit,
                        "objective_relative_difference": relative,
                        "full_rank_relative": full_error,
                        "full_rank_normal_equation_relative": normal,
                        "bias_stationarity_relative": bias_residual,
                        "whitened_reconstruction_relative": info["whitened_reconstruction_relative"],
                        "spectral_optimality_relative": info["spectral_optimality_relative"],
                        "tolerance": tolerance})
    return {"pass": True, "device": device, "dtype": str(dtype), "fixtures": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(preflight(args.device,
        torch.float32 if args.device.startswith("cuda") else torch.float64), indent=2))
