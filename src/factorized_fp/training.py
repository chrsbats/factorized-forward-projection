"""Ordinary Adam/AdamW training controls for the Fashion-MNIST models."""

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .fp import clock, evaluate
from .rank import ReducedRankMLP


def random_reduced_rank(rank, *, width=1000, seed=0, hidden_layers=3):
    """Same rank graph with ordinary random factor initialization for AdamW."""
    torch.manual_seed(seed)
    model = ReducedRankMLP(rank, width=width, hidden_layers=hidden_layers)
    for layer in model.layers[:-1]:
        nn.init.kaiming_uniform_(layer.u, a=math.sqrt(5))
        nn.init.kaiming_uniform_(layer.v, a=math.sqrt(5))
        bound = 1/math.sqrt(layer.rank)
        nn.init.uniform_(layer.bias, -bound, bound)
    return model


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train(model, optimizer, splits, *, fold=0, seed=0, batch_size=50, eval_batch=4096,
          max_epochs=100, patience=5, checkpoint_dir=None):
    """Train with ordinary cross-entropy; select and stop by validation CE.

    The cadence is epoch 0 plus quarter epochs in epochs 1-2, then full epochs,
    matching the archived protocol. Times are cumulative from before training.
    """
    (x, y), (vx, vy), (tx, ty) = splits
    device = x.device
    begin = clock(device)
    directory = None if checkpoint_dir is None else Path(checkpoint_dir)
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    history, best_index, examples, steps = [], 0, 0, 0

    def record(epoch, full_epoch, training_ce=None, save=True):
        nonlocal best_index
        model.eval()
        validation = evaluate(model, vx, vy, eval_batch)
        test = evaluate(model, tx, ty, eval_batch)
        checkpoint = f"step-{steps:06d}.pt"
        if directory is not None and save:
            torch.save(cpu_state(model), directory / checkpoint)
        row = {"epoch": epoch, "steps": steps, "training_examples": examples,
               "full_epoch": full_epoch, "training_ce": training_ce,
               "validation": validation, "test": test,
               "cumulative_seconds": clock(device) - begin}
        if directory is not None and save:
            row["checkpoint"] = checkpoint
        history.append(row)
        if validation["ce"] < history[best_index]["validation"]["ce"]:
            best_index = len(history) - 1
        row["selected_index"] = best_index
        model.train()
        return row

    initial = record(0., False)
    batches = (len(x) + batch_size - 1) // batch_size
    bad, best_full_ce, stop, epoch = 0, initial["validation"]["ce"], "maximum", 0
    for epoch in range(1, max_epochs + 1):
        generator = torch.Generator(device=device).manual_seed(10000 + seed + fold * 100 + epoch)
        order = torch.randperm(len(x), generator=generator, device=device)
        checkpoints = ({max(1, round(batches * fraction)) for fraction in (.25, .5, .75, 1.)}
                       if epoch <= 2 else {batches})
        training_sum = torch.zeros((), dtype=torch.float64, device=device)
        epoch_examples = 0
        current = None
        for batch, start in enumerate(range(0, len(x), batch_size), 1):
            ids = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[ids]), y[ids])
            loss.backward()
            optimizer.step()
            training_sum += loss.detach().double() * len(ids)
            epoch_examples += len(ids)
            examples += len(ids)
            steps += 1
            if batch in checkpoints:
                current = record(epoch - 1 + batch / batches, batch == batches,
                                 float(training_sum / epoch_examples))
        if not bool(torch.isfinite(training_sum)):
            raise ArithmeticError("non-finite training loss")
        if current["validation"]["ce"] < best_full_ce:
            best_full_ce, bad = current["validation"]["ce"], 0
        else:
            bad += 1
        if bad >= patience:
            stop = "validation-patience"
            break
    selected = history[best_index]
    return {"initial": initial, "history": history, "selected": selected,
            "terminal": history[-1], "stop": stop, "epochs": epoch,
            "total_seconds": clock(device) - begin,
            "optimizer": {"name": type(optimizer).__name__,
                          "lr": optimizer.param_groups[0]["lr"],
                          "weight_decay": optimizer.param_groups[0]["weight_decay"]}}


def make_optimizer(model, name="adamw", *, lr=.001, weight_decay=.01):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                 betas=(.9, .999), eps=1e-8)
    raise ValueError("optimizer must be 'adam' or 'adamw'")


__all__ = ["random_reduced_rank", "cpu_state", "train", "make_optimizer"]
