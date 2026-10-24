"""Shared device, data, fold, and JSON helpers for the reproduction scripts."""

import json
import os
from pathlib import Path
import platform

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
import torch

from factorized_fp.fp import load_fashion


ROOT = Path(__file__).resolve().parents[1]


def setup(device_name, *, allow_cpu=False):
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --device cpu only for a smoke run")
    if device.type != "cuda" and not allow_cpu:
        raise SystemExit("measured runs require CUDA; use --smoke --device cpu for a pipeline check")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return device


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def environment(device):
    result = {"python": platform.python_version(), "torch": torch.__version__,
              "numpy": np.__version__, "device": str(device),
              "threads": torch.get_num_threads(),
              "float32_matmul_precision": torch.get_float32_matmul_precision(),
              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else None}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(cuda_runtime=torch.version.cuda, gpu=properties.name,
                      gpu_memory_bytes=properties.total_memory,
                      compute_capability=[properties.major, properties.minor])
    return result


def folds(data, *, seed=0, device="cpu"):
    """Deterministic five-fold story: one saved assignment shared by every arm."""
    assignments = torch.randint(0, 5, (60000,), generator=torch.Generator().manual_seed(seed))
    train = {key: value.to(device) if isinstance(value, torch.Tensor) else value
             for key, value in data.items()}
    return assignments, train


def split(data, assignments, fold, device):
    training = (assignments != fold).to(device)
    return (((data["train_x"][training], data["train_y"][training]),
             (data["train_x"][~training], data["train_y"][~training]),
             (data["test_x"], data["test_y"])))


def load_data(data_dir, device):
    data, manifest = load_fashion(data_dir)
    data = {key: value.to(device) for key, value in data.items()}
    return data, manifest


def run_directory(root, name, smoke=False):
    from datetime import datetime, timezone
    import uuid
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return Path(root) / "results" / name / (("smoke-" if smoke else "") + stamp + "-" + suffix)
