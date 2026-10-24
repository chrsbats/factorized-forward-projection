"""Matched rank-128 analytical FP versus random-factor AdamW.

Both arms use exactly the same ReducedRankMLP graph and parameter count. The
comparison asks how long AdamW takes to first record the analytical model's
validation accuracy; it is not a matched-final-quality speedup.
"""

import argparse
from pathlib import Path

import torch

from _common import ROOT, dump, environment, folds, load_data, run_directory, setup, split
from factorized_fp.analysis import crossing, summary
from factorized_fp.fp import clock, evaluate
from factorized_fp.rank import ReducedRankMLP, fit_forward, inference_cost
from factorized_fp.training import make_optimizer, random_reduced_rank, train


def analytical(splits, args, fold, directory):
    (x, y), (vx, vy), (tx, ty) = splits
    device = x.device
    begin = clock(device)
    model = ReducedRankMLP(args.rank, width=args.width).to(device)
    fitted, _ = fit_forward(model, x, y, seed=1000 + args.seed + fold, batch_size=args.fit_batch)
    fit_seconds = clock(device) - begin
    if args.save_checkpoints:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, directory / "fp.pt")
    validation, test = evaluate(model, vx, vy), evaluate(model, tx, ty)
    return {"fit": fitted, "fit_seconds": fit_seconds, "validation": validation, "test": test,
            "complete_seconds": clock(device) - begin, "cost": inference_cost(model)}, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/fashion-mnist")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--fit-batch", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.width, args.max_epochs, args.rank = 16, 1, 4
    args.device = str(setup(args.device, allow_cpu=args.smoke))
    args.device = torch.device(args.device)
    output = args.output or run_directory(ROOT, "matched-rank128-adamw", args.smoke)
    output.mkdir(parents=True, exist_ok=False)
    data, manifest = load_data(args.data, args.device)
    assignments, data = folds(data, seed=args.seed, device=args.device)
    torch.save(assignments, output / "fold-assignments.pt")
    dump(output / "data-manifest.json", manifest)
    dump(output / "environment.json", environment(args.device))
    fold_rows, ratios = {}, []
    for fold in [int(value) for value in args.folds.split(",")]:
        splits = split(data, assignments, fold, args.device)
        directory = output / f"fold-{fold}"
        fp, _ = analytical(splits, args, fold, directory / "fp")
        begin = clock(args.device)
        model = random_reduced_rank(args.rank, width=args.width, seed=args.seed + fold).to(args.device)
        optimizer = make_optimizer(model, "adamw", lr=args.lr, weight_decay=args.weight_decay)
        adamw = train(model, optimizer, splits, fold=fold, seed=args.seed,
                      batch_size=args.batch_size, max_epochs=args.max_epochs,
                      patience=args.patience, checkpoint_dir=directory / "adamw")
        adamw["complete_seconds"] = clock(args.device) - begin
        adamw["cost"] = inference_cost(model)
        hit = crossing(adamw["history"], "accuracy", fp["validation"]["accuracy"],
                       higher_is_better=True)
        ratio = None if hit is None else hit["observed_seconds"]/fp["complete_seconds"]
        if ratio is not None:
            ratios.append(ratio)
        fold_rows[str(fold)] = {"fp": fp, "adamw": adamw,
                                "target_validation_accuracy": fp["validation"]["accuracy"],
                                "first_equal_accuracy": hit, "paired_ratio": ratio}
        dump(output / "summary.json", {"folds": fold_rows})
        if not args.smoke:
            print(f"fold{fold}: FP {fp['test']['accuracy']:.4f} in {fp['complete_seconds']:.2f}s; "
                  f"AdamW selected {adamw['selected']['test']['accuracy']:.4f}; "
                  f"ratio {ratio:.2f}x" if ratio else "not crossed", flush=True)
    aggregate = {
        "fp_validation_accuracy": summary(row["fp"]["validation"]["accuracy"] for row in fold_rows.values()),
        "fp_test_accuracy": summary(row["fp"]["test"]["accuracy"] for row in fold_rows.values()),
        "fp_complete_seconds": summary(row["fp"]["complete_seconds"] for row in fold_rows.values()),
        "adamw_selected_test_accuracy": summary(
            row["adamw"]["selected"]["test"]["accuracy"] for row in fold_rows.values()),
        "adamw_selected_validation_accuracy": summary(
            row["adamw"]["selected"]["validation"]["accuracy"] for row in fold_rows.values()),
        "parameter_count": next(iter(fold_rows.values()))["fp"]["cost"]["parameters"],
        "paired_observed_ratio": summary(ratios),
        "crossed_folds": len(ratios),
    }
    dump(output / "summary.json", {"folds": fold_rows, "aggregate": aggregate})
    print(f"Summary: {output}", flush=True)


if __name__ == "__main__":
    main()
