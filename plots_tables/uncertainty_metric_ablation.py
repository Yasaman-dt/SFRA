"""Aggregate MSP/entropy/energy relearning runs with matched retrained controls."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["msp", "entropy", "energy"]
METHOD_LABELS = {
    "bad_teacher": "Bad Teacher", "delete": "DELETE", "scrub": "SCRUB",
    "neggrad_plus": "Negative Gradient+", "finetune": "Finetune",
    "retrained": "Retrained",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results_synthesis_ablation"))
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--model_name", default="resnet18")
    parser.add_argument("--methods", nargs="+", default=[
        "bad_teacher", "delete", "scrub", "neggrad_plus", "finetune",
    ])
    parser.add_argument("--classes", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--out_dir", type=Path, default=Path("results_uncertainty_ablation"))
    return parser.parse_args()


def canonical_metric(value):
    return "msp" if str(value) == "softmax" else str(value)


def load_runs(args, method):
    frames = []
    agreements = []
    for forget_class in args.classes:
        pattern = f"{args.dataset}_{args.model_name}_lr*_fg{forget_class}"
        folders = sorted((args.root / method).glob(pattern))
        if not folders:
            continue
        # A method should have one paper checkpoint LR per forget class.
        if len(folders) > 1:
            raise RuntimeError(f"Multiple runs match {method}/fg{forget_class}: {folders}")
        run_csv = folders[0] / "runs.csv"
        if run_csv.is_file():
            frame = pd.read_csv(run_csv)
            if "uncertainty_score" not in frame:
                frame["uncertainty_score"] = "softmax"
            else:
                frame["uncertainty_score"] = frame["uncertainty_score"].fillna("softmax")
            frame["selection_metric"] = frame["uncertainty_score"].map(canonical_metric)
            frame = frame[
                frame.distribution.eq("gaussian")
                & frame.forget_selection.eq("low_confidence")
                & frame.retain_selection.eq("high_confidence")
                & frame.selection_metric.isin(METRICS)
            ].copy()
            frames.append(frame)
        agreement_csv = folders[0] / "ranking_agreement.csv"
        if agreement_csv.is_file():
            agreements.append(pd.read_csv(agreement_csv))
    runs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    agreement = pd.concat(agreements, ignore_index=True) if agreements else pd.DataFrame()
    return runs, agreement


def build_outputs(args):
    controls, control_agreement = load_runs(args, "retrained")
    if controls.empty:
        raise FileNotFoundError("Matched retrained uncertainty runs are required.")
    control = controls.groupby(
        ["forget_class", "seed", "selection_metric"], as_index=False
    )["RS"].mean().rename(columns={"RS": "retrained_RS"})
    per_class_frames = []
    agreement_frames = [control_agreement] if not control_agreement.empty else []
    for method in args.methods:
        runs, agreement = load_runs(args, method)
        if runs.empty:
            print(f"[warning] no runs found for {method}")
            continue
        merged = runs.merge(
            control, on=["forget_class", "seed", "selection_metric"], how="left",
            validate="many_to_one",
        )
        if merged.retrained_RS.isna().any():
            missing = merged.loc[merged.retrained_RS.isna(), [
                "forget_class", "seed", "selection_metric"
            ]].drop_duplicates()
            raise RuntimeError(f"Missing matched retrained controls for {method}:\n{missing}")
        merged["delta_RS"] = merged.RS - merged.retrained_RS
        merged["method"] = method
        metrics = [
            "baseline_retain_acc", "baseline_forget_acc",
            "relearned_retain_acc", "relearned_forget_acc",
            "RS", "retrained_RS", "delta_RS",
        ]
        per_class = merged.groupby(
            ["method", "forget_class", "selection_metric"], as_index=False
        )[metrics].agg(["mean", "std"]).reset_index()
        per_class.columns = [
            "_".join(part for part in column if part).rstrip("_")
            if isinstance(column, tuple) else column
            for column in per_class.columns
        ]
        per_class_frames.append(per_class)
        if not agreement.empty:
            agreement_frames.append(agreement)
    if not per_class_frames:
        raise RuntimeError("No method runs were found.")
    per_class = pd.concat(per_class_frames, ignore_index=True)
    mean_columns = [column for column in per_class if column.endswith("_mean")]
    aggregate = per_class.groupby(
        ["method", "selection_metric"], as_index=False
    )[mean_columns].agg(["mean", "std"]).reset_index()
    aggregate.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple) else column
        for column in aggregate.columns
    ]
    agreement = pd.concat(agreement_frames, ignore_index=True) if agreement_frames else pd.DataFrame()
    if not agreement.empty:
        agreement["pair"] = agreement.metric_a + " vs " + agreement.metric_b
        agreement_summary = agreement.groupby("pair", as_index=False).overlap_at_m.agg(["mean", "std"]).reset_index()
    else:
        agreement_summary = pd.DataFrame(columns=["pair", "mean", "std"])
    return per_class, aggregate, agreement, agreement_summary


def write_table(frame, path):
    def f(value, digits=2, signed=False):
        if pd.isna(value):
            return "--"
        return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Uncertainty-score ablation for source-free relearning on CIFAR-10/ResNet-18. Values are means across forget classes. Each $\Delta$RS uses the retrained control evaluated with the same selection metric. Energy is $-\log\sum_c\exp(f_c)$ at $T=1$, with larger values treated as more uncertain.}",
        r"\label{tab:uncertainty_metric_ablation}",
        r"\begin{tabular}{l|l|rrrr}", r"\toprule",
        r"Method & Selection Metric & Forget Acc. & Retain Acc. & RS & $\Delta$RS \\",
        r"\midrule",
    ]
    for method_index, (method, rows) in enumerate(frame.groupby("method", sort=False)):
        for row_index, (_, row) in enumerate(rows.iterrows()):
            lines.append(" & ".join([
                METHOD_LABELS.get(method, method) if row_index == 0 else "",
                str(row.selection_metric).upper() if row.selection_metric == "msp" else str(row.selection_metric).title(),
                f(row.relearned_forget_acc_mean_mean), f(row.relearned_retain_acc_mean_mean),
                f(row.RS_mean_mean, 3), f(row.delta_RS_mean_mean, 3, True),
            ]) + r" \\")
        if method_index < frame.method.nunique() - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n")


def write_summary(frame, agreement, path):
    lines = []
    for method, rows in frame.groupby("method", sort=False):
        best = rows.loc[rows.RS_mean_mean.idxmax()]
        spread = rows.RS_mean_mean.max() - rows.RS_mean_mean.min()
        lines.append(
            f"{METHOD_LABELS.get(method, method)}: highest mean RS is "
            f"{best.selection_metric} ({best.RS_mean_mean:.3f}); across-metric RS spread is {spread:.3f}."
        )
    for row in agreement.itertuples():
        lines.append(f"Selected-probe {row.pair} overlap@M: {row.mean:.3f} ± {row.std:.3f}.")
    path.write_text("\n".join(lines) + "\n")


def write_plot(frame, path):
    methods = list(dict.fromkeys(frame.method))
    x = np.arange(len(methods)); width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for index, metric in enumerate(METRICS):
        rows = frame[frame.selection_metric.eq(metric)].set_index("method")
        axes[0].bar(x + (index - 1) * width, [rows.RS_mean_mean.get(m, np.nan) for m in methods], width, label=metric.upper())
        axes[1].bar(x + (index - 1) * width, [rows.delta_RS_mean_mean.get(m, np.nan) for m in methods], width, label=metric.upper())
    for axis, ylabel in zip(axes, ["RS", r"$\Delta$RS"]):
        axis.set_xticks(x, [METHOD_LABELS.get(m, m) for m in methods], rotation=30, ha="right")
        axis.set_ylabel(ylabel); axis.grid(axis="y", alpha=.2); axis.legend()
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)


def main():
    args = parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    per_class, aggregate, agreement, agreement_summary = build_outputs(args)
    per_class.to_csv(args.out_dir / "uncertainty_ablation_per_class.csv", index=False)
    aggregate.to_csv(args.out_dir / "uncertainty_ablation_aggregated.csv", index=False)
    agreement.to_csv(args.out_dir / "uncertainty_ranking_agreement_raw.csv", index=False)
    agreement_summary.to_csv(args.out_dir / "uncertainty_ranking_agreement.csv", index=False)
    write_table(aggregate, args.out_dir / "uncertainty_ablation_table.tex")
    write_summary(aggregate, agreement_summary, args.out_dir / "uncertainty_ablation_summary.txt")
    write_plot(aggregate, args.out_dir / "uncertainty_ablation_rs.png")
    print(f"[saved] {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
