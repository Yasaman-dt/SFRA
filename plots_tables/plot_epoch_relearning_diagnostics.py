"""Plot epoch diagnostics directly from completed trajectory CSV files."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--max_epoch", type=int, default=None)
    args = parser.parse_args()

    paths = sorted(args.result_dir.glob("*/trajectory_seed*.csv"))
    if not paths:
        raise FileNotFoundError(f"No trajectory CSVs below {args.result_dir}")
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    # Read older result files while writing and plotting the shorter val_* names.
    for old_name, new_name in (
        ("validation_retain_accuracy", "val_retain_accuracy"),
        ("validation_forget_accuracy", "val_forget_accuracy"),
        ("validation_RS", "val_RS"),
    ):
        if new_name not in frame and old_name in frame:
            frame[new_name] = frame[old_name]
    if args.max_epoch is not None:
        frame = frame[frame["epoch"].le(args.max_epoch)]

    colors = {"bad_teacher": "#1f77b4", "delete": "#ff7f0e"}
    names = {"bad_teacher": "Bad Teacher", "delete": "DELETE"}
    panels = [
        ("Synthetic accuracy", "Accuracy (%)",
         [("synthetic_retain_accuracy", "Retain"),
          ("synthetic_forget_accuracy", "Forget")]),
        ("Real accuracy", "Accuracy (%)",
         [("val_retain_accuracy", "Val retain"),
          ("val_forget_accuracy", "Val forget"),
          ("test_retain_accuracy", "Test retain"),
          ("test_forget_accuracy", "Test forget")]),
        ("Relearning score", "RS",
         [("synthetic_RS", "Synthetic"),
          ("val_RS", "Val"), ("test_RS", "Test")]),
        ("Forget-class confidence", "Mean confidence (%)",
         [("synthetic_forget_confidence", "Synthetic"),
          ("test_forget_confidence", "Test")]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for axis, (title, ylabel, metrics) in zip(axes.flat, panels):
        for method, method_rows in frame.groupby("method", sort=False):
            for index, (metric, metric_name) in enumerate(metrics):
                summary = method_rows.groupby("epoch")[metric].agg(["mean", "std"])
                x = summary.index.to_numpy()
                mean = summary["mean"].to_numpy()
                std = summary["std"].fillna(0).to_numpy()
                color = colors.get(method)
                axis.plot(x, mean, color=color, linestyle=("-" if index == 0 else "--"),
                          linewidth=1.8, label=f"{names.get(method, method)}: {metric_name}")
                axis.fill_between(x, mean - std, mean + std,
                                  color=color, alpha=0.08, linewidth=0)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9, ncol=2)
    for axis in axes[-1]:
        axis.set_xlabel("Epoch")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(args.result_dir / f"epoch_relearning_diagnostics.{extension}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

