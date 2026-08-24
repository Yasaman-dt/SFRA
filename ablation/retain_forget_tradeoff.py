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
from plots_tables.method_colors import get_method_color  # noqa: E402


METHOD_ORDER = [
    "bad_teacher", "delete", "gradient_ascent", "random_label", "salun",
    "retrained", "finetune", "neggrad_plus", "l2ul_adv",
]

DEFAULT_METHOD_LRS = {
    ("cifar10", "resnet18"): {
        "bad_teacher": 0.001,
        "delete": 0.001,
        "gradient_ascent": 5e-5,
        "random_label": 1e-7,
        "salun": 0.001,
        "retrained": 0,
        "finetune": 0.02,
        "neggrad_plus": 0.5,
        "l2ul_adv": 1e-5,
    },
    ("cifar100", "resnet18"): {
        "bad_teacher": 0.001,
        "delete": 0.001,
        "gradient_ascent": 0.01,
        "random_label": 1e-7,
        "salun": 0.1,
        "retrained": 0,
        "finetune": 0.02,
        "neggrad_plus": 0.5,
        "l2ul_adv": 0.002,
    },
    ("tiny_imagenet", "resnet18"): {
        "bad_teacher": 0.002,
        "delete": 0.001,
        "gradient_ascent": 5e-5,
        "random_label": 1e-4,
        "salun": 0.001,
        "retrained": 0,
        "finetune": 0.02,
        "neggrad_plus": 0.5,
        "l2ul_adv": 1e-5,
    },
}

DISPLAY_NAMES = {
    "salun": "SalUn",
    "gradient_ascent": "Negative Gradient",
    "delete": "DELETE",
    "bad_teacher": "Bad Teacher",
    "random_label": "Random Label",
    "retrained": "Retrained",
    "finetune": "Finetune",
    "neggrad_plus": "Negative Gradient+",
    "l2ul_adv": "Learn to Unlearn",
}

METHOD_STYLES = {
    "bad_teacher": {
        "color": get_method_color("bad_teacher"), "marker": "o", "linestyle": ":",
    },
    "delete": {
        "color": get_method_color("delete"), "marker": "s", "linestyle": "--",
    },
    "gradient_ascent": {
        "color": get_method_color("gradient_ascent"), "marker": "^", "linestyle": "-.",
    },
    "random_label": {
        "color": get_method_color("random_label"), "marker": "D", "linestyle": "--",
    },
    "salun": {
        "color": get_method_color("salun"), "marker": "X", "linestyle": ":",
    },
    "retrained": {
        "color": get_method_color("retrained"), "marker": "P", "linestyle": "-.",
    },
    "finetune": {
        "color": get_method_color("finetune"), "marker": "v", "linestyle": "--",
    },
    "neggrad_plus": {
        "color": get_method_color("neggrad_plus"), "marker": "*", "linestyle": ":",
    },
    "l2ul_adv": {
        "color": get_method_color("l2ul_adv"), "marker": "h", "linestyle": "-.",
    },
}

SATURATION_EPSILON_PP = 1.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure maximum SFRA forget accuracy at retain-drop constraints."
    )
    parser.add_argument(
        "--dataset", default="cifar10",
        choices=["cifar10", "cifar100", "tiny_imagenet"],
    )
    parser.add_argument("--model", "--model_name", dest="model_name", default="resnet18")
    parser.add_argument("--forget_class", type=int, default=9)
    parser.add_argument(
        "--methods", nargs="+", default=METHOD_ORDER,
        choices=METHOD_ORDER,
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
    protocol = (args.dataset, args.model_name)
    if protocol not in DEFAULT_METHOD_LRS:
        raise ValueError(
            f"No default checkpoint learning rates for "
            f"dataset={args.dataset}, model={args.model_name}. "
            "Provide a supported protocol or extend DEFAULT_METHOD_LRS."
        )
    values = dict(DEFAULT_METHOD_LRS[protocol])
    for item in args.method_lr:
        if "=" not in item:
            raise ValueError(f"Expected METHOD=LR, got {item!r}")
        method, value = item.split("=", 1)
        if method not in METHOD_ORDER:
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
    baseline_forget = float(trajectory.loc[trajectory["epoch"].eq(0), "forget_accuracy"].iloc[0])
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
            "baseline_forget_accuracy": baseline_forget,
            "forget_accuracy_gain_pp": float(best["forget_accuracy"] - baseline_forget),
        })
    return pd.DataFrame(rows)


def accuracy_pareto_frontier(points):
    """Return points not dominated when both retain and forget accuracy are maximized."""
    ordered = points.sort_values(
        ["retain_accuracy", "forget_accuracy"], ascending=[False, False]
    )
    keep = []
    best_forget = float("-inf")
    for index, row in ordered.iterrows():
        if float(row["forget_accuracy"]) > best_forget + 1e-12:
            keep.append(index)
            best_forget = float(row["forget_accuracy"])
    return points.loc[keep].sort_values("retain_accuracy")


def trim_after_forget_saturation(frontier, epsilon_pp=SATURATION_EPSILON_PP):
    """Drop lower-retain points after forget accuracy is effectively saturated."""
    if frontier.empty:
        return frontier
    saturation_level = float(frontier["forget_accuracy"].max()) - float(epsilon_pp)
    saturated = frontier[frontier["forget_accuracy"] >= saturation_level]
    if saturated.empty:
        return frontier
    highest_retain_at_saturation = float(saturated["retain_accuracy"].max())
    return frontier[frontier["retain_accuracy"] >= highest_retain_at_saturation].copy()


def plot_results(all_tradeoffs, all_trajectories, output_dir, expected_seeds=None):
    frame = pd.concat(all_tradeoffs, ignore_index=True)
    trajectories = pd.concat(all_trajectories, ignore_index=True)

    if expected_seeds:
        expected_seeds = set(expected_seeds)
        tradeoff_seeds = frame.groupby("method")["seed"].agg(
            lambda values: set(values.astype(int))
        )
        trajectory_seeds = trajectories.groupby("method")["seed"].agg(
            lambda values: set(values.astype(int))
        )
        complete_methods = {
            method
            for method in set(tradeoff_seeds.index) & set(trajectory_seeds.index)
            if expected_seeds.issubset(tradeoff_seeds[method])
            and expected_seeds.issubset(trajectory_seeds[method])
        }
        incomplete_methods = sorted(
            (set(frame["method"]) | set(trajectories["method"])) - complete_methods
        )
        for method in incomplete_methods:
            available = sorted(
                set(tradeoff_seeds.get(method, set()))
                & set(trajectory_seeds.get(method, set()))
            )
            missing = sorted(expected_seeds - set(available))
            print(
                f"[warn] Excluding incomplete method {method}; "
                f"missing seeds {missing}."
            )
        frame = frame[frame["method"].isin(complete_methods)].copy()
        trajectories = trajectories[
            trajectories["method"].isin(complete_methods)
        ].copy()
        if frame.empty or trajectories.empty:
            raise ValueError(
                f"No methods contain all requested seeds {sorted(expected_seeds)}."
            )

    frame.to_csv(output_dir / "retain_forget_tradeoff_all_methods.csv", index=False)
    summary = frame.groupby(
        ["method", "allowed_retain_drop_pp"], sort=False
    ).agg(
        actual_retain_drop_mean=("actual_retain_drop_pp", "mean"),
        actual_retain_drop_std=("actual_retain_drop_pp", "std"),
        forget_gain_mean=("forget_accuracy_gain_pp", "mean"),
        forget_gain_std=("forget_accuracy_gain_pp", "std"),
        retain_accuracy_mean=("retain_accuracy_at_selected_epoch", "mean"),
        retain_accuracy_std=("retain_accuracy_at_selected_epoch", "std"),
        forget_accuracy_mean=("max_forget_accuracy", "mean"),
        forget_accuracy_std=("max_forget_accuracy", "std"),
        count=("forget_accuracy_gain_pp", "count"),
    ).reset_index()
    summary.to_csv(output_dir / "retain_forget_tradeoff_summary.csv", index=False)

    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 24,
        "font.size": 15,
        "legend.fontsize": 18,
        "legend.title_fontsize": 18,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "text.color": "black",
        "legend.labelcolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.edgecolor": "black",
    })
    fig, axis = plt.subplots(figsize=(6.6, 4.4))
    methods_present = list(trajectories["method"].drop_duplicates())
    plot_order = [method for method in METHOD_STYLES if method in methods_present]
    plot_order.extend(method for method in methods_present if method not in plot_order)
    for method in plot_order:
        style = METHOD_STYLES.get(method, {})
        method_runs = trajectories[trajectories["method"].eq(method)]
        epoch_summary = method_runs.groupby("epoch", as_index=False).agg(
            retain_accuracy=("retain_accuracy", "mean"),
            forget_accuracy=("forget_accuracy", "mean"),
            forget_std=("forget_accuracy", "std"),
        )
        frontier = accuracy_pareto_frontier(epoch_summary)
        frontier = trim_after_forget_saturation(frontier)
        x = frontier["retain_accuracy"].to_numpy(dtype=float)
        mean = frontier["forget_accuracy"].to_numpy(dtype=float)
        std = frontier["forget_std"].fillna(0.0).to_numpy(dtype=float)
        axis.fill_between(
            x, mean - std, mean + std,
            color=style.get("color"), alpha=0.08, linewidth=0,
        )
        axis.plot(
            x, mean, linewidth=2.2, markersize=6,
            markevery=max(1, len(x) // 7),
            label=DISPLAY_NAMES.get(method, method), **style,
        )

        # The Pareto and saturation filters may remove the unlearned
        # checkpoint. Show its epoch-0 operating point explicitly.
        epoch_zero = method_runs[method_runs["epoch"].eq(0)]
        if not epoch_zero.empty:
            axis.scatter(
                epoch_zero["retain_accuracy"].mean(),
                epoch_zero["forget_accuracy"].mean(),
                color=style.get("color"),
                marker=style.get("marker", "o"),
                s=48,
                edgecolors="black",
                linewidths=0.65,
                zorder=5,
            )
    axis.set_xlabel(r"$\mathcal{A}_r^t(\%)$")
    axis.set_ylabel(r"$\mathcal{A}_f^t(\%)$")
    axis.grid(alpha=0.25)
    legend = axis.legend(
        frameon=True, fancybox=False, framealpha=1.0,
        facecolor="white", edgecolor="0.65",
        ncol=1, fontsize=18, markerscale=0.6,
        handlelength=2.4, handletextpad=0.6,
        borderpad=0.35, labelspacing=0.25,
        loc="upper left",
    )
    legend.get_frame().set_linewidth(0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "retain_forget_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def load_tradeoff_with_gains(path):
    """Load a trade-off CSV and derive gain columns for legacy saved runs."""
    tradeoff = pd.read_csv(path)
    if "forget_accuracy_gain_pp" in tradeoff.columns:
        return tradeoff

    seed_text = path.stem.removeprefix("tradeoff_seed")
    trajectory_path = path.with_name(f"trajectory_seed{seed_text}.csv")
    if not trajectory_path.exists():
        raise FileNotFoundError(
            f"Cannot derive forget-accuracy gain without {trajectory_path}"
        )
    trajectory = pd.read_csv(trajectory_path)
    baseline_rows = trajectory[trajectory["epoch"].eq(0)]
    if baseline_rows.empty:
        raise ValueError(f"No epoch-0 baseline in {trajectory_path}")
    baseline_forget = float(baseline_rows["forget_accuracy"].iloc[0])
    tradeoff["baseline_forget_accuracy"] = baseline_forget
    tradeoff["forget_accuracy_gain_pp"] = (
        tradeoff["max_forget_accuracy"] - baseline_forget
    )
    return tradeoff


def main():
    args = parse_args()
    num_classes = {
        "cifar10": 10,
        "cifar100": 100,
        "tiny_imagenet": 200,
    }[args.dataset]
    if not 0 <= args.forget_class < num_classes:
        raise ValueError(
            f"forget_class must be in [0, {num_classes - 1}] for {args.dataset}."
        )
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
                checkpoint, args.model_name, args.dataset, num_classes, device
            )
            feature_model = make_feature_extractor(model, num_classes, device)
            _, classifier = get_final_linear(model, num_classes)
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
                    classifier=classifier, num_classes=num_classes,
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

    tradeoffs = [
        load_tradeoff_with_gains(path)
        for path in output_dir.glob("*/tradeoff_seed*.csv")
    ]
    if not tradeoffs:
        raise FileNotFoundError(f"No trade-off CSVs found below {output_dir}")
    trajectories = [
        pd.read_csv(path) for path in output_dir.glob("*/trajectory_seed*.csv")
    ]
    if not trajectories:
        raise FileNotFoundError(f"No trajectory CSVs found below {output_dir}")
    plot_results(tradeoffs, trajectories, output_dir, expected_seeds=args.seeds)
    print(f"[saved] {output_dir.resolve()}")


if __name__ == "__main__":
    main()
