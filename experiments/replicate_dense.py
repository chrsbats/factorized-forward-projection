"""Five-fold dense Forward Projection replication with an Adam control.

Reproduces the Fashion-MNIST MLP result of O'Shea and Rajendran (2026): a
sequential analytical FP fit followed by a matched ordinary Adam baseline.
"""

import argparse
import copy
from pathlib import Path

import torch

from _common import ROOT, dump, environment, folds, load_data, run_directory, setup, split
from factorized_fp.analysis import crossing, summary
from factorized_fp.fp import FPMLP, clock, evaluate, fit_forward
from factorized_fp.training import make_optimizer, train


def arm_fp(splits, args, fold):
    (x, y), (vx, vy), (tx, ty) = splits
    device = x.device
    begin = clock(device)
    model = FPMLP(width=args.width).to(device)
    fitted, _ = fit_forward(model, x, y, seed=1000 + args.seed + fold)
    validation, test = evaluate(model, vx, vy), evaluate(model, tx, ty)
    if args.save_checkpoints:
        (args.directory / "fp.pt").parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.directory / "fp.pt")
    return {"fit": fitted, "validation": validation, "test": test,
            "complete_seconds": clock(device) - begin,
            "parameters": sum(p.numel() for p in model.parameters())}, model


def arm_adam(splits, args, fold, directory, name, model=None):
    (x, y), (vx, vy), (tx, ty) = splits
    device = x.device
    begin = clock(device)
    if model is None:
        torch.manual_seed(args.seed + fold)
        model = FPMLP(width=args.width).to(device)
    optimizer = make_optimizer(model, "adam", lr=args.lr)
    result = train(model, optimizer, splits, fold=fold, seed=args.seed,
                   batch_size=args.batch_size, max_epochs=args.max_epochs,
                   patience=args.patience, checkpoint_dir=directory)
    result["complete_seconds"] = clock(device) - begin
    result["parameters"] = sum(p.numel() for p in model.parameters())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/fashion-mnist")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.width, args.max_epochs = 16, 1
    args.device = str(setup(args.device, allow_cpu=args.smoke))
    args.device = torch.device(args.device)
    fold_list = [int(value) for value in args.folds.split(",")]
    output = args.output or run_directory(ROOT, "dense-fp", args.smoke)
    output.mkdir(parents=True, exist_ok=False)
    data, manifest = load_data(args.data, args.device)
    assignments, data = folds(data, seed=args.seed, device=args.device)
    torch.save(assignments, output / "fold-assignments.pt")
    dump(output / "data-manifest.json", manifest)
    dump(output / "environment.json", environment(args.device))
    summary_rows, fold_rows = [], {}
    for fold in fold_list:
        splits = split(data, assignments, fold, args.device)
        directory = output / f"fold-{fold}"
        args.directory = directory
        fp, fp_model = arm_fp(splits, args, fold)
        adam = arm_adam(splits, args, fold, directory / "adam", "adam")
        fp_adam = arm_adam(splits, args, fold, directory / "fp-adam", "fp-adam",
                           model=copy.deepcopy(fp_model))
        # FP accuracy target on validation, as in the archived crossing protocol.
        target = fp["validation"]["accuracy"]
        hit = crossing(adam["history"], "accuracy", target, higher_is_better=True)
        fp_hit = crossing(fp_adam["history"], "accuracy", target, higher_is_better=True)
        fold_rows[str(fold)] = {"fp": fp, "adam": adam, "fp_adam": fp_adam,
                                "adam_first_equal_accuracy": hit,
                                "fp_adam_first_equal_accuracy": fp_hit,
                                "target_validation_accuracy": target}
        if not args.smoke:
            print(f"fold{fold}: FP {fp['test']['accuracy']:.4f} | Adam selected "
                  f"{adam['selected']['test']['accuracy']:.4f} | ratio "
                  f"{(hit['observed_seconds']/fp['complete_seconds']) if hit else float('nan'):.2f}x",
                  flush=True)
        dump(output / "summary.json", {"folds": fold_rows})
    valid = [row for row in fold_rows.values() if row["adam_first_equal_accuracy"]]
    summary_rows = {
        "fp_validation_accuracy": summary(row["fp"]["validation"]["accuracy"] for row in fold_rows.values()),
        "fp_test_accuracy": summary(row["fp"]["test"]["accuracy"] for row in fold_rows.values()),
        "fp_complete_seconds": summary(row["fp"]["complete_seconds"] for row in fold_rows.values()),
        "adam_selected_test_accuracy": summary(row["adam"]["selected"]["test"]["accuracy"] for row in fold_rows.values()),
        "fp_adam_selected_test_accuracy": summary(row["fp_adam"]["selected"]["test"]["accuracy"] for row in fold_rows.values()),
        "paired_observed_ratio": summary(
            row["adam_first_equal_accuracy"]["observed_seconds"]/row["fp"]["complete_seconds"]
            for row in valid),
        "crossed_folds": len(valid),
    }
    dump(output / "summary.json", {"folds": fold_rows, "aggregate": summary_rows})
    print(f"Summary: {output}", flush=True)


if __name__ == "__main__":
    main()
