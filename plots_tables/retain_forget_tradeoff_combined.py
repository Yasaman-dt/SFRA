"""Combine the CIFAR-10 and CIFAR-100 retain/forget trade-off plots."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ablation.retain_forget_tradeoff import (
    DISPLAY_NAMES,
    METHOD_STYLES,
    accuracy_pareto_frontier,
    trim_after_forget_saturation,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cifar10_dir",
        type=Path,
        default=Path("results_retain_forget_tradeoff/cifar10_resnet18_fg7"),
    )
    parser.add_argument(
        "--cifar100_dir",
        type=Path,
        default=Path("results_retain_forget_tradeoff/cifar100_resnet18_fg0"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results_retain_forget_tradeoff/retain_forget_tradeoff_cifar10_cifar100.png"
        ),
    )
    return parser.parse_args()


def load_complete_trajectories(root: Path, expected_seeds):
    frames = [pd.read_csv(path) for path in root.glob("*/trajectory_seed*.csv")]
    if not frames:
        raise FileNotFoundError(f"No trajectory CSVs found below {root}")

    trajectories = pd.concat(frames, ignore_index=True)
    expected_seeds = set(expected_seeds)
    seed_sets = trajectories.groupby("method")["seed"].agg(
        lambda values: set(values.astype(int))
    )
    complete = {
        method for method, seeds in seed_sets.items()
        if expected_seeds.issubset(seeds)
    }
    for method in sorted(set(trajectories["method"]) - complete):
        missing = sorted(expected_seeds - seed_sets[method])
        print(f"[warn] Excluding incomplete method {method}; missing seeds {missing}.")
    return trajectories[trajectories["method"].isin(complete)].copy()


def plot_panel(axis, trajectories, title, show_legend=False):
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
        frontier = trim_after_forget_saturation(
            accuracy_pareto_frontier(epoch_summary)
        )
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

        epoch_zero = method_runs[method_runs["epoch"].eq(0)]
        if not epoch_zero.empty:
            axis.scatter(
                epoch_zero["retain_accuracy"].mean(),
                epoch_zero["forget_accuracy"].mean(),
                color=style.get("color"), marker=style.get("marker", "o"),
                s=48, edgecolors="black", linewidths=0.65, zorder=5,
            )

    axis.set_title(title, fontweight="bold")
    axis.set_xlabel(r"$\mathcal{A}_r^t(\%)$")
    axis.set_ylabel(r"$\mathcal{A}_f^t(\%)$")
    axis.grid(alpha=0.25)
    if show_legend:
        legend = axis.legend(
            loc="upper left", frameon=True, fancybox=False, framealpha=1.0,
            facecolor="white", edgecolor="0.65", fontsize=18,
            markerscale=0.6, handlelength=2.4, handletextpad=0.6,
            borderpad=0.35, labelspacing=0.25,
        )
        legend.get_frame().set_linewidth(0.6)


def main():
    args = parse_args()
    cifar10 = load_complete_trajectories(args.cifar10_dir, args.seeds)
    cifar100 = load_complete_trajectories(args.cifar100_dir, args.seeds)

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
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), sharey=True)
    plot_panel(axes[0], cifar10, "CIFAR-10", show_legend=False)
    plot_panel(axes[1], cifar100, "CIFAR-100", show_legend=False)
    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels,
        title="Unlearning Method",
        loc="center left", bbox_to_anchor=(1.0, 0.5),
        frameon=True, fancybox=False, framealpha=1.0,
        facecolor="white", edgecolor="0.65", fontsize=18,
        markerscale=0.6, handlelength=2.4, handletextpad=0.6,
        borderpad=0.35, labelspacing=0.25,
    )
    legend.get_frame().set_linewidth(0.6)
    legend.get_title().set_fontweight("bold")
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {args.output.resolve()}")


if __name__ == "__main__":
    main()
