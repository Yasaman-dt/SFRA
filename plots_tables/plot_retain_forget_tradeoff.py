"""Combine the CIFAR-10 and CIFAR-100 retain/forget trade-off plots."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_rs_vs_accuracy_change import (
    DISPLAY_NAMES,
    METHOD_STYLES,
    accuracy_pareto_frontier,
    load_complete_trajectories,
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
    parser.add_argument("--include_vit", action="store_true")
    parser.add_argument(
        "--vit_cifar10_dir",
        type=Path,
        default=Path("results_retain_forget_tradeoff/cifar10_vit-b-16_fg7"),
    )
    parser.add_argument(
        "--vit_cifar100_dir",
        type=Path,
        default=Path("results_retain_forget_tradeoff/cifar100_vit-b-16_fg0"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--legend",
        choices=["none", "first-upper-left", "second-lower-left"],
        default="first-upper-left",
        help="Choose whether and where to draw the shared legend.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results_retain_forget_tradeoff/retain_forget_tradeoff_resnet18_cifar10_cifar100.png"
        ),
    )
    return parser.parse_args()


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
        epoch_zero = epoch_summary[epoch_summary["epoch"].eq(0)]
        relearning_summary = epoch_summary[epoch_summary["epoch"].gt(0)]
        frontier = trim_after_forget_saturation(
            accuracy_pareto_frontier(relearning_summary)
        )
        # Start at the unlearned checkpoint (epoch 0), then connect it to the
        # highest-retain point on the SFRA relearning frontier.
        frontier = frontier.sort_values("retain_accuracy", ascending=False)
        plotted = pd.concat([epoch_zero, frontier], ignore_index=True)
        x = plotted["retain_accuracy"].to_numpy(dtype=float)
        mean = plotted["forget_accuracy"].to_numpy(dtype=float)
        std = plotted["forget_std"].fillna(0.0).to_numpy(dtype=float)
        axis.fill_between(
            x, mean - std, mean + std,
            color=style.get("color"), alpha=0.08, linewidth=0,
        )
        axis.plot(
            x, mean, linewidth=2.2, markersize=6,
            markevery=max(1, len(x) // 7),
            label=DISPLAY_NAMES.get(method, method), **style,
        )

        if not epoch_zero.empty:
            axis.scatter(
                epoch_zero["retain_accuracy"].mean(),
                epoch_zero["forget_accuracy"].mean(),
                color=style.get("color"), marker=style.get("marker", "o"),
                s=48, edgecolors="black", linewidths=0.65, zorder=5,
            )

    axis.set_title(title, fontsize=18, fontweight="bold")
    axis.set_xlabel(r"$\mathcal{A}_r^t\,(\%)$")
    axis.set_ylabel(r"$\mathcal{A}_f^t\,(\%)$")
    # Use a common range across all architecture figures.  The small margin
    # below zero prevents epoch-zero markers from being clipped, while ticks
    # begin at zero so no negative accuracy label is shown.
    axis.set_ylim(-4, 105)
    axis.set_yticks([0, 20, 40, 60, 80, 100])
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
        "axes.labelsize": 18,
        "font.size": 12,
        "legend.fontsize": 18,
        "legend.title_fontsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "text.color": "black",
        "legend.labelcolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.edgecolor": "black",
    })
    if args.include_vit:
        vit_cifar10 = load_complete_trajectories(args.vit_cifar10_dir, args.seeds)
        vit_cifar100 = load_complete_trajectories(args.vit_cifar100_dir, args.seeds)
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4), sharey=True)
        plot_panel(axes[0, 0], cifar10, "CIFAR-10", show_legend=False)
        plot_panel(axes[0, 1], cifar100, "CIFAR-100", show_legend=False)
        plot_panel(axes[1, 0], vit_cifar10, "", show_legend=False)
        plot_panel(axes[1, 1], vit_cifar100, "", show_legend=False)
        axes[0, 1].set_ylabel("")
        axes[1, 1].set_ylabel("")
        axes[0, 0].annotate(
            "ResNet-18", xy=(-0.19, 0.5), xycoords="axes fraction",
            rotation=90, ha="center", va="center", fontsize=15,
            fontweight="bold",
        )
        axes[1, 0].annotate(
            "ViT-B/16", xy=(-0.19, 0.5), xycoords="axes fraction",
            rotation=90, ha="center", va="center", fontsize=15,
            fontweight="bold",
        )
        first_axis = axes[0, 0]
        second_axis = axes[0, 1]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), sharey=True)
        plot_panel(axes[0], cifar10, "CIFAR-10", show_legend=False)
        plot_panel(axes[1], cifar100, "CIFAR-100", show_legend=False)
        axes[1].set_ylabel("")
        first_axis = axes[0]
        second_axis = axes[1]

    if args.legend != "none":
        legend_axis = (
            second_axis if args.legend == "second-lower-left" else first_axis
        )
        legend_location = (
            "lower left" if args.legend == "second-lower-left" else "upper left"
        )
        handles, labels = first_axis.get_legend_handles_labels()
        legend = legend_axis.legend(
            handles, labels,
            title="Unlearning Method",
            loc=legend_location,
            frameon=True, fancybox=False, framealpha=1.0,
            facecolor="white", edgecolor="0.65", fontsize=10,
            title_fontsize=11,
            markerscale=0.7, handlelength=2.3, handletextpad=0.6,
            borderpad=0.4, labelspacing=0.4,
        )
        legend.get_frame().set_linewidth(0.6)
        legend.get_title().set_fontweight("bold")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {args.output.resolve()}")

if __name__ == "__main__":
    main()
