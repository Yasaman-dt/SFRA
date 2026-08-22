"""Plot the SFRA forget/retain trade-off for one forgotten class.

For every unlearned checkpoint, this script generates one fixed synthetic
dataset, trains only the released classifier head, and evaluates the head at
every epoch.  It then computes the constrained envelope

    max A_f(t)  subject to  A_r(t) >= A_r(0) - delta,

where ``delta`` is an allowed retain-accuracy drop in percentage points.
Consequently, changing ``delta`` does not change the probes or retrain the
head; it only changes which saved points on the same trajectory are eligible.
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ablation.synthesis_strategy_ablation import (  # noqa: E402
    Strategy,
    accuracy,
    build_synthetic_sets_for_strategies,
)
from plots_tables.tsne_real_gaussian_probes import (  # noqa: E402
    checkpoint_for,
    extract_features,
    get_final_linear,
    load_checkpoint_model,
    make_feature_extractor,
)
from utils import get_dataset, get_transforms  # noqa: E402


DEFAULT_METHOD_LRS = {
    "salun": 0.001,
    "gradient_ascent": 5e-5,
    "delete": 0.001,
    "bad_teacher": 0.001,
    "random_label": 1e-7,
}

DISPLAY_NAMES = {
    "salun": "SalUn",
    "gradient_ascent": "Negative Gradient",
    "delete": "DELETE",
    "bad_teacher": "Bad Teacher",
    "random_label": "Random Label",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure maximum SFRA forget accuracy at retain-drop constraints."
    )
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    parser.add_argument("--model", "--model_name", dest="model_name", default="resnet18")
    parser.add_argument("--forget_class", type=int, default=9)
    parser.add_argument(
        "--methods", nargs="+", default=list(DEFAULT_METHOD_LRS),
        choices=list(DEFAULT_METHOD_LRS),
    )
    parser.add_argument(
        "--method_lr", action="append", default=[], metavar="METHOD=LR",
        help="Override a checkpoint LR, for example --method_lr salun=0.001.",
    )
    parser.add_argument(
        "--allowed_retain_drops", type=float, nargs="+",
        default=[0, 0.5, 1, 2, 3, 5, 7.5, 10],
        help="Allowed drops from unlearned retain accuracy, in percentage points.",
    )
    parser.add_argument("--generated_per_class", type=int, default=500000)
    parser.add_argument("--retain_per_class", type=int, default=500)
    parser.add_argument("--forget_per_class", type=int, default=500)
    parser.add_argument("--sample_batch_size", type=int, default=4096)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--head_lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--ckpt_dir",
        default="/export/livia/home/vision/Zdehghani/classification/exps",
    )
    parser.add_argument("--data_dir", default=str(Path("~/data").expanduser()))
    parser.add_argument(
        "--output_dir", default="results_retain_forget_tradeoff/cifar10_resnet18_fg9"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot_only", action="store_true")
    return parser.parse_args()


def method_lrs(args):
    values = dict(DEFAULT_METHOD_LRS)
    for item in args.method_lr:
        if "=" not in item:
            raise ValueError(f"Expected METHOD=LR, got {item!r}")
        method, value = item.split("=", 1)
        if method not in DEFAULT_METHOD_LRS:
            raise ValueError(f"Unknown method in --method_lr: {method}")
        values[method] = float(value)
    return values


def train_trajectory(classifier, synthetic, real, args, device, seed):
    retain_x, retain_y, forget_x, forget_y, _ = synthetic
    train_x = torch.cat([retain_x, forget_x])
    train_y = torch.cat([retain_y, forget_y])
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 991),
    )
    head = copy.deepcopy(classifier).to(device)
    optimizer = torch.optim.Adam(
        head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    test_retain_x, test_retain_y, test_forget_x, test_forget_y = real

    def evaluate(epoch):
        return {
            "epoch": epoch,
            "retain_accuracy": accuracy(
                head, test_retain_x, test_retain_y, device, args.eval_batch_size
            ),
            "forget_accuracy": accuracy(
                head, test_forget_x, test_forget_y, device, args.eval_batch_size
            ),
        }

    rows = [evaluate(0)]
    for epoch in range(1, args.epochs + 1):
        head.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        rows.append(evaluate(epoch))
    return pd.DataFrame(rows)


def envelope(trajectory, allowed_drops):
    baseline_retain = float(trajectory.loc[trajectory["epoch"].eq(0), "retain_accuracy"].iloc[0])
    rows = []
    for allowed_drop in sorted(set(allowed_drops)):
        floor = baseline_retain - allowed_drop
        eligible = trajectory[trajectory["retain_accuracy"] >= floor]
        best = eligible.loc[eligible["forget_accuracy"].idxmax()]
        rows.append({
            "allowed_retain_drop_pp": allowed_drop,
            "retain_floor": floor,
            "max_forget_accuracy": best["forget_accuracy"],
            "retain_accuracy_at_selected_epoch": best["retain_accuracy"],
            "selected_epoch": int(best["epoch"]),
            "actual_retain_drop_pp": max(0.0, baseline_retain - best["retain_accuracy"]),
        })
    return pd.DataFrame(rows)


def plot_results(all_tradeoffs, output_dir):
    frame = pd.concat(all_tradeoffs, ignore_index=True)
    frame.to_csv(output_dir / "retain_forget_tradeoff_all_methods.csv", index=False)
    summary = (
        frame.groupby(["method", "allowed_retain_drop_pp"], sort=False)
        ["max_forget_accuracy"].agg(["mean", "std", "count"]).reset_index()
    )
    summary.to_csv(output_dir / "retain_forget_tradeoff_summary.csv", index=False)

    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, axis = plt.subplots(figsize=(6.6, 4.4))
    for method in frame["method"].drop_duplicates():
        part = summary[summary["method"].eq(method)].sort_values("allowed_retain_drop_pp")
        yerr = part["std"].fillna(0.0)
        axis.errorbar(
            part["allowed_retain_drop_pp"], part["mean"], yerr=yerr,
            marker="o", linewidth=1.8, capsize=2.5, label=DISPLAY_NAMES.get(method, method),
        )
    axis.set_xlabel("Maximum allowed retain-accuracy drop (percentage points)")
    axis.set_ylabel("Maximum forget accuracy (%)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "retain_forget_tradeoff.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "retain_forget_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.generated_per_class < args.retain_per_class + args.forget_per_class:
        raise ValueError("generated_per_class must cover both disjoint selected subsets.")
    if any(value < 0 for value in args.allowed_retain_drops):
        raise ValueError("Allowed retain drops must be nonnegative.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lrs = method_lrs(args)

    if not args.plot_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, test_transform = get_transforms(args.dataset, args.model_name, wo_dataaug=False)
        _, test_dataset = get_dataset(
            args.dataset, test_transform, test_transform,
            path=Path(args.data_dir).expanduser(),
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
        )
        strategy = Strategy("gaussian", "low_confidence", "high_confidence", "softmax")

        for method in args.methods:
            checkpoint = checkpoint_for(
                method, args.dataset, args.model_name, args.forget_class,
                lrs[method], args.ckpt_dir,
            )
            if not Path(checkpoint).exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            model = load_checkpoint_model(
                checkpoint, args.model_name, args.dataset, 10, device
            )
            feature_model = make_feature_extractor(model, 10, device)
            _, classifier = get_final_linear(model, 10)
            real_x, real_y = extract_features(feature_model, test_loader, device)
            mask = real_y.eq(args.forget_class)
            real = (
                real_x[~mask].float(), real_y[~mask],
                real_x[mask].float(), real_y[mask],
            )
            for seed in args.seeds:
                method_dir = output_dir / method
                method_dir.mkdir(parents=True, exist_ok=True)
                trajectory_path = method_dir / f"trajectory_seed{seed}.csv"
                tradeoff_path = method_dir / f"tradeoff_seed{seed}.csv"
                if tradeoff_path.exists() and not args.force:
                    print(f"[skip] {method} seed={seed}")
                    continue
                print(f"[run] method={method} seed={seed} checkpoint={checkpoint}")
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                synthetic, _ = build_synthetic_sets_for_strategies(
                    classifier=classifier, num_classes=10,
                    forget_class=args.forget_class, strategies=[strategy],
                    generated_per_class=args.generated_per_class,
                    retain_per_class=args.retain_per_class,
                    forget_per_class=args.forget_per_class,
                    sample_batch_size=args.sample_batch_size, device=device, seed=seed,
                )
                trajectory = train_trajectory(
                    classifier, synthetic[strategy], real, args, device, seed
                )
                trajectory.insert(0, "seed", seed)
                trajectory.insert(0, "method", method)
                trajectory.to_csv(trajectory_path, index=False)
                result = envelope(trajectory, args.allowed_retain_drops)
                result.insert(0, "seed", seed)
                result.insert(0, "method", method)
                result.to_csv(tradeoff_path, index=False)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del model, feature_model, classifier, real_x, real_y, real
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        (output_dir / "metadata.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    tradeoffs = [pd.read_csv(path) for path in output_dir.glob("*/tradeoff_seed*.csv")]
    if not tradeoffs:
        raise FileNotFoundError(f"No trade-off CSVs found below {output_dir}")
    plot_results(tradeoffs, output_dir)
    print(f"[saved] {output_dir.resolve()}")


if __name__ == "__main__":
    main()
