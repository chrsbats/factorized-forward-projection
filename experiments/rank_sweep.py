"""Five-fold reduced-rank Forward Projection sweep with a dense AdamW control."""

import argparse
from pathlib import Path

import torch

from _common import ROOT, dump, environment, folds, load_data, run_directory, setup, split
from factorized_fp.analysis import crossing, summary
from factorized_fp.fp import FPMLP, clock, evaluate, fit_forward as fit_dense
from factorized_fp.rank import ReducedRankMLP, fit_forward as fit_rank, inference_cost
from factorized_fp.training import make_optimizer, train


def fit_analytical(model, splits, rank, args, fold, directory):
    (x, y), (vx, vy), (tx, ty) = splits
    device = x.device
    begin = clock(device)
    fitted, _ = (fit_dense(model, x, y, seed=1000 + args.seed + fold) if rank is None else
                 fit_rank(model, x, y, seed=1000 + args.seed + fold, batch_size=args.fit_batch))
    fit_seconds = clock(device) - begin
    if args.save_checkpoints:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, directory / "model.pt")
    validation, test = evaluate(model, vx, vy), evaluate(model, tx, ty)
    cost = inference_cost(model)
    return {"rank": rank, "fit": fitted, "fit_seconds": fit_seconds,
            "complete_seconds": clock(device) - begin, "validation": validation,
            "test": test, "cost": cost}, model


def run_adamw(splits, args, fold, directory):
    (x, y), (_, _), (_, _) = splits
    device = x.device
    begin = clock(device)
    torch.manual_seed(args.seed + fold)
    model = FPMLP(width=args.width).to(device)
    optimizer = make_optimizer(model, "adamw", lr=args.lr)
    result = train(model, optimizer, splits, fold=fold, seed=args.seed,
                   batch_size=args.batch_size, max_epochs=args.max_epochs,
                   patience=args.patience, checkpoint_dir=directory)
    result["complete_seconds"] = clock(device) - begin
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/fashion-mnist")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--ranks", default="64,128,256,512")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--fit-batch", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.width, args.ranks, args.max_epochs = 16, "2,4", 1
    args.ranks = [int(value) for value in args.ranks.split(",")]
    args.device = str(setup(args.device, allow_cpu=args.smoke))
    args.device = torch.device(args.device)
    output = args.output or run_directory(ROOT, "rank-sweep", args.smoke)
    output.mkdir(parents=True, exist_ok=False)
    data, manifest = load_data(args.data, args.device)
    assignments, data = folds(data, seed=args.seed, device=args.device)
    torch.save(assignments, output / "fold-assignments.pt")
    dump(output / "data-manifest.json", manifest)
    dump(output / "environment.json", environment(args.device))
    folds_selected = [int(value) for value in args.folds.split(",")]
    rows, fold_rows = [], {}
    for fold in folds_selected:
        splits = split(data, assignments, fold, args.device)
        fold_results = {}
        directory = output / f"fold-{fold}"
        for rank in [None, *args.ranks]:
            name = "dense" if rank is None else f"rank-{rank}"
            model = (FPMLP(width=args.width) if rank is None else
                     ReducedRankMLP(rank, width=args.width)).to(args.device)
            result, fitted_model = fit_analytical(model, splits, rank, args, fold, directory / name)
            fold_results[name] = result
            del model, fitted_model
        adamw = run_adamw(splits, args, fold, directory / "adamw")
        fold_results["adamw"] = adamw
        for name, result in fold_results.items():
            if name == "adamw":
                continue
            hit = crossing(adamw["history"], "accuracy", result["validation"]["accuracy"],
                           higher_is_better=True)
            result["adamw_first_equal_accuracy"] = hit
            if hit:
                rows.append({"fold": fold, "arm": name,
                             "ratio": hit["observed_seconds"]/result["complete_seconds"]})
        fold_rows[str(fold)] = fold_results
        dump(output / "summary.json", {"folds": fold_rows})
        if not args.smoke:
            print(f"fold{fold}: " + ", ".join(
                f"{name} {result['test']['accuracy']:.4f}" for name, result in fold_results.items()
                if name != "adamw"), flush=True)
    aggregate = {}
    for name in [*[f"rank-{r}" for r in args.ranks]]:
        arm_rows = [fold_rows[str(f)][name] for f in folds_selected if name in fold_rows[str(f)]]
        ratios = [row["ratio"] for row in rows if row["arm"] == name]
        aggregate[name] = {
            "test_accuracy": summary(row["test"]["accuracy"] for row in arm_rows),
            "validation_accuracy": summary(row["validation"]["accuracy"] for row in arm_rows),
            "complete_seconds": summary(row["complete_seconds"] for row in arm_rows),
            "parameters": arm_rows[0]["cost"]["parameters"],
            "estimated_flops_per_image": arm_rows[0]["cost"]["estimated_flops_per_image"],
            "paired_observed_ratio": summary(ratios),
        }
    dump(output / "summary.json", {"folds": fold_rows, "aggregate": aggregate})
    print(f"Summary: {output}", flush=True)


if __name__ == "__main__":
    main()
