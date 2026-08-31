"""Plot SFRA RS against retain- or forget-accuracy change over relearning."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from method_colors import get_method_color


REPO_ROOT = Path(__file__).resolve().parents[1]
DISPLAY_NAMES = {
    "bad_teacher": "Bad Teacher", "delete": "DELETE",
    "gradient_ascent": "Negative Gradient", "random_label": "Random Label",
    "salun": "SalUn", "retrained": "Retrained", "finetune": "Finetune",
    "neggrad_plus": "Negative Gradient+", "l2ul_adv": "Learn to Unlearn",
}
METHOD_STYLES = {
    "bad_teacher": {"color": get_method_color("bad_teacher"), "marker": "o", "linestyle": ":"},
    "delete": {"color": get_method_color("delete"), "marker": "s", "linestyle": "--"},
    "gradient_ascent": {"color": get_method_color("gradient_ascent"), "marker": "^", "linestyle": "-."},
    "random_label": {"color": get_method_color("random_label"), "marker": "D", "linestyle": "--"},
    "salun": {"color": get_method_color("salun"), "marker": "*", "linestyle": ":"},
    "retrained": {"color": get_method_color("retrained"), "marker": "P", "linestyle": "-."},
    "finetune": {"color": get_method_color("finetune"), "marker": "v", "linestyle": "--"},
    "neggrad_plus": {"color": get_method_color("neggrad_plus"), "marker": "X", "linestyle": ":"},
    "l2ul_adv": {"color": get_method_color("l2ul_adv"), "marker": "h", "linestyle": "-."},
}


def load_complete_trajectories(root: Path, expected_seeds) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in root.glob("*/trajectory_seed*.csv")]
    if not frames:
        raise FileNotFoundError(f"No trajectory CSVs found below {root}")
    trajectories = pd.concat(frames, ignore_index=True)
    expected = set(expected_seeds)
    seed_sets = trajectories.groupby("method")["seed"].agg(
        lambda values: set(values.astype(int))
    )
    complete = {method for method, seeds in seed_sets.items() if expected.issubset(seeds)}
    for method in sorted(set(trajectories["method"]) - complete):
        print(f"[warn] Excluding incomplete method {method}; missing seeds {sorted(expected - seed_sets[method])}.")
    return trajectories.loc[trajectories["method"].isin(complete)].copy()


def accuracy_pareto_frontier(points: pd.DataFrame) -> pd.DataFrame:
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


def trim_after_forget_saturation(frontier: pd.DataFrame, epsilon_pp: float = 1.0) -> pd.DataFrame:
    if frontier.empty:
        return frontier
    level = float(frontier["forget_accuracy"].max()) - epsilon_pp
    saturated = frontier.loc[frontier["forget_accuracy"] >= level]
    if saturated.empty:
        return frontier
    cutoff = float(saturated["retain_accuracy"].max())
    return frontier.loc[frontier["retain_accuracy"] >= cutoff].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifar10-dir", type=Path, required=True)
    parser.add_argument("--cifar100-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--legend",
        choices=["none", "first-upper-left", "second-lower-left"],
        default="none",
    )
    parser.add_argument(
        "--x-metric",
        choices=["retain", "forget"],
        default="retain",
        help=(
            "Accuracy change shown on the x-axis, measured relative to the "
            "corresponding unlearned checkpoint at epoch zero."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def add_scores(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Compute per-seed Delta retain accuracy and RS relative to epoch zero."""
    frame = trajectories.copy()
    baseline = (
        frame.loc[frame["epoch"].eq(0),
                  ["method", "seed", "retain_accuracy", "forget_accuracy"]]
        .rename(
            columns={
                "retain_accuracy": "retain_accuracy_unlearned",
                "forget_accuracy": "forget_accuracy_unlearned",
            }
        )
    )
    frame = frame.merge(baseline, on=["method", "seed"], how="left", validate="many_to_one")
    if frame[["retain_accuracy_unlearned", "forget_accuracy_unlearned"]].isna().any(axis=None):
        raise ValueError("At least one method/seed trajectory has no epoch-zero row.")

    frame["delta_retain_accuracy"] = (
        frame["retain_accuracy"] - frame["retain_accuracy_unlearned"]
    )
    frame["delta_forget_accuracy"] = (
        frame["forget_accuracy"] - frame["forget_accuracy_unlearned"]
    )
    retain_score = 1.0 - (
        (frame["retain_accuracy_unlearned"] - frame["retain_accuracy"])
        .clip(lower=0.0) / 100.0
    )
    forget_score = (
        (frame["forget_accuracy"] - frame["forget_accuracy_unlearned"])
        .clip(lower=0.0) / 100.0
    )
    denominator = retain_score + forget_score
    frame["RS"] = np.where(
        denominator.gt(0.0),
        2.0 * retain_score * forget_score / denominator,
        0.0,
    )
    frame["RS"] = frame["RS"].clip(0.0, 1.0)
    return frame


def summarize_method(method_runs: pd.DataFrame) -> pd.DataFrame:
    summary = method_runs.groupby("epoch", as_index=False).agg(
        retain_accuracy=("retain_accuracy", "mean"),
        forget_accuracy=("forget_accuracy", "mean"),
        delta_retain_accuracy=("delta_retain_accuracy", "mean"),
        delta_forget_accuracy=("delta_forget_accuracy", "mean"),
        RS=("RS", "mean"),
        RS_std=("RS", "std"),
    )
    epoch_zero = summary.loc[summary["epoch"].eq(0)]
    relearning = summary.loc[summary["epoch"].gt(0)]
    frontier = trim_after_forget_saturation(accuracy_pareto_frontier(relearning))
    frontier = frontier.sort_values("retain_accuracy", ascending=False)
    return pd.concat([epoch_zero, frontier], ignore_index=True)


def plot_panel(
    axis: plt.Axes,
    trajectories: pd.DataFrame,
    title: str,
    x_metric: str,
) -> None:
    methods_present = list(trajectories["method"].drop_duplicates())
    order = [method for method in METHOD_STYLES if method in methods_present]
    order.extend(method for method in methods_present if method not in order)

    for method in order:
        plotted = summarize_method(trajectories.loc[trajectories["method"].eq(method)])
        style = METHOD_STYLES.get(method, {})
        x_column = (
            "delta_retain_accuracy"
            if x_metric == "retain"
            else "delta_forget_accuracy"
        )
        x = plotted[x_column].to_numpy(float)
        mean = plotted["RS"].to_numpy(float)
        std = plotted["RS_std"].fillna(0.0).to_numpy(float)
        marker_step = max(1, len(x) // 7)
        # Epoch zero is the same conceptual unlearned starting location for
        # every method. Keep it in the connected trajectory, but do not stack
        # method-specific markers at that common point.
        marker_indices = list(range(marker_step, len(x), marker_step))
        axis.fill_between(
            x,
            np.clip(mean - std, 0.0, 1.0),
            np.clip(mean + std, 0.0, 1.0),
            color=style.get("color"),
            alpha=0.08,
            linewidth=0,
        )
        axis.plot(
            x,
            mean,
            linewidth=2.2,
            markersize=8 if method == "salun" else 6,
            markevery=marker_indices,
            label=DISPLAY_NAMES.get(method, method),
            **style,
        )

    axis.axvline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    axis.set_title(title, fontsize=22, fontweight="bold")
    axis.set_xlabel(
        r"$\Delta\mathcal{A}_r^t\,(\%)$"
        if x_metric == "retain"
        else r"$\Delta\mathcal{A}_f^t\,(\%)$"
    )
    axis.set_ylabel(r"$\mathrm{RS}$")
    axis.set_ylim(-0.025, 1.025)
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.grid(alpha=0.25)


def add_legend(args: argparse.Namespace, axes) -> None:
    if args.legend == "none":
        return
    axis = axes[1] if args.legend == "second-lower-left" else axes[0]
    location = "lower left" if args.legend == "second-lower-left" else "upper left"
    handles, labels = axes[0].get_legend_handles_labels()
    legend = axis.legend(
        handles,
        labels,
        title="Unlearning Method",
        loc=location,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="0.65",
        fontsize=12.5,
        title_fontsize=13.5,
        markerscale=0.85,
        handlelength=2.3,
        handletextpad=0.6,
        borderpad=0.4,
        labelspacing=0.4,
    )
    legend.get_frame().set_linewidth(0.6)
    legend.get_title().set_fontweight("bold")


def main() -> None:
    args = parse_args()
    cifar10 = add_scores(load_complete_trajectories(args.cifar10_dir, args.seeds))
    cifar100 = add_scores(load_complete_trajectories(args.cifar100_dir, args.seeds))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman No9 L",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "axes.labelsize": 22,
            "font.size": 14,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.edgecolor": "black",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), sharey=True)
    plot_panel(axes[0], cifar10, "CIFAR-10", args.x_metric)
    plot_panel(axes[1], cifar100, "CIFAR-100", args.x_metric)
    axes[1].set_ylabel("")
    add_legend(args, axes)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {args.output.resolve()}")


if __name__ == "__main__":
    main()
