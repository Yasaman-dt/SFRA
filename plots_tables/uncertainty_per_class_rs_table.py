"""Generate per-forget-class uncertainty-score RS tables.

The input is the directory tree written by
``ablation.synthesis_strategy_ablation``.  For every method, architecture,
forget class, and uncertainty score, this script reports RS mean +/- standard
deviation across random seeds.  Only the matched Gaussian / low-confidence
forget / high-confidence retain strategy is included.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = [
    "retrained",
    "finetune",
    "gradient_ascent",
    "neggrad_plus",
    "random_label",
    "boundary_shrink",
    "l2ul_adv",
    "scrub",
    "bad_teacher",
    "salun",
    "delete",
]

METHOD_LABELS = {
    "retrained": "Retrained",
    "finetune": r"Finetune \cite{golatkar2020eternal}",
    "gradient_ascent": r"Negative Gradient \cite{golatkar2020eternal}",
    "neggrad_plus": r"Negative Gradient+ \cite{kurmanji2023towards}",
    "random_label": r"Random Label \cite{hayase2020selective}",
    "boundary_shrink": r"Boundary Shrink \cite{chen2023boundary}",
    "l2ul_adv": r"Learn to Unlearn \cite{cha2024learning}",
    "scrub": r"SCRUB \cite{kurmanji2023towards}",
    "bad_teacher": r"Bad Teacher \cite{chundawat2023can}",
    "salun": r"SalUn \cite{fan2023salun}",
    "delete": r"DELETE \cite{zhou2025decoupled}",
}

MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "vit-b-16": "ViT-B/16",
    "swin-t": "Swin-T",
}

SCORES = ["softmax", "entropy", "energy"]
METHODS_EXCLUDED_BY_MODEL = {
    "vit-b-16": {"boundary_shrink"},
    "swin-t": {"boundary_shrink"},
}
SCORE_LABELS = {
    "softmax": "Softmax",
    "entropy": "Entropy",
    "energy": "Energy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results_uncertainty_ablation_cifar10_all_classes_500ep"),
    )
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument(
        "--model", "--model_name", dest="model_name", default="resnet18"
    )
    parser.add_argument("--classes", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=3,
        help="Show a cell only when at least this many distinct seeds exist.",
    )
    parser.add_argument("--precision", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--caption", default=None)
    parser.add_argument("--label", default=None)
    return parser.parse_args()


def parse_run_directory(name: str, dataset: str, model: str) -> tuple[float, int] | None:
    match = re.fullmatch(
        rf"{re.escape(dataset)}_{re.escape(model)}_lr([-+0-9.eE]+)_fg(\d+)",
        name,
    )
    if match is None:
        return None
    return float(match.group(1)), int(match.group(2))


def load_runs(args: argparse.Namespace) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    wanted_classes = set(args.classes)
    for method in args.methods:
        method_root = args.root / method
        if not method_root.is_dir():
            continue
        for path in sorted(method_root.glob("*/runs.csv")):
            parsed = parse_run_directory(path.parent.name, args.dataset, args.model_name)
            if parsed is None:
                continue
            unlearn_lr, forget_class = parsed
            if forget_class not in wanted_classes:
                continue
            frame = pd.read_csv(path)
            if "uncertainty_score" not in frame:
                frame["uncertainty_score"] = "softmax"
            frame["uncertainty_score"] = (
                frame["uncertainty_score"].fillna("softmax").replace({"msp": "softmax"})
            )
            mask = (
                frame["distribution"].astype(str).eq("gaussian")
                & frame["forget_selection"].astype(str).eq("low_confidence")
                & frame["retain_selection"].astype(str).eq("high_confidence")
                & frame["uncertainty_score"].isin(SCORES)
            )
            frame = frame.loc[mask, ["seed", "uncertainty_score", "RS"]].copy()
            frame["RS"] = pd.to_numeric(frame["RS"], errors="coerce")
            frame["method"] = method
            frame["forget_class"] = forget_class
            frame["unlearn_lr"] = unlearn_lr
            frames.append(frame.dropna(subset=["RS"]))
    if not frames:
        return pd.DataFrame(
            columns=["method", "forget_class", "seed", "uncertainty_score", "RS"]
        )
    combined = pd.concat(frames, ignore_index=True)
    # A resumed run should have one row per method/class/score/seed. Refuse to
    # average accidental duplicate runs (for example, two checkpoint LRs).
    keys = ["method", "forget_class", "uncertainty_score", "seed"]
    duplicates = combined.duplicated(keys, keep=False)
    if duplicates.any():
        rows = combined.loc[duplicates, keys + ["unlearn_lr"]].sort_values(keys)
        raise ValueError(
            "Duplicate method/class/score/seed rows were found. Remove the stale "
            "run directory or use a clean result root before generating the table:\n"
            + rows.to_string(index=False)
        )
    return combined


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(["method", "uncertainty_score", "forget_class"], as_index=False)
        .agg(rs_mean=("RS", "mean"), rs_std=("RS", "std"), seeds=("seed", "nunique"))
    )


def summarize_across_classes(
    frame: pd.DataFrame, classes: list[int]
) -> pd.DataFrame:
    """Average classes within each seed, then summarize across audit seeds."""
    if frame.empty:
        return pd.DataFrame()
    wanted = set(classes)
    per_seed = (
        frame[frame["forget_class"].isin(wanted)]
        .groupby(["method", "uncertainty_score", "seed"], as_index=False)
        .agg(rs_class_average=("RS", "mean"), classes=("forget_class", "nunique"))
    )
    # Do not report an average based on an incomplete set of forget classes.
    per_seed = per_seed[per_seed["classes"].eq(len(wanted))]
    return (
        per_seed.groupby(["method", "uncertainty_score"], as_index=False)
        .agg(
            rs_mean=("rs_class_average", "mean"),
            rs_std=("rs_class_average", "std"),
            seeds=("seed", "nunique"),
        )
    )


def format_cell(
    mean: float,
    std: float,
    seeds: int,
    args: argparse.Namespace,
    bold: bool = False,
    show_std: bool = True,
) -> str:
    if seeds < args.min_seeds or pd.isna(mean):
        return "-"
    if not show_std or pd.isna(std):
        cell = rf"${mean:.{args.precision}f}$"
        return rf"\textbf{{{cell}}}" if bold else cell
    cell = (
        rf"${mean:.{args.precision}f}$"
        rf"{{\tiny $\,\pm\,{std:.{args.precision}f}$}}"
    )
    return rf"\textbf{{{cell}}}" if bold else cell


def make_latex(
    summary: pd.DataFrame,
    average_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    model_label = MODEL_LABELS.get(args.model_name, args.model_name)
    caption = args.caption or (
        "Uncertainty-score ablation on CIFAR-10 using a "
        f"{model_label} backbone. Each class column corresponds to a separate "
        r"unlearned checkpoint with the indicated forget class and reports "
        r"$\mathrm{RS}$ as the mean $\pm$ standard deviation across three "
        r"independent audit seeds, while Avg. gives the mean $\mathrm{RS}$ "
        r"across all forget classes and seeds."
    )
    label = args.label or f"tab:uncertainty_per_class_{args.model_name.replace('-', '_')}"
    columns = "l|l|" + "c" * len(args.classes) + "|c"
    lines = [
        r"\begin{table*}[p]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\vspace{-0.15cm}",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{0.88}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        rf"\multirow{{2}}{{*}}{{Unlearning Method}} & "
        rf"\multirow{{2}}{{*}}{{Uncertainty}} & "
        rf"\multicolumn{{{len(args.classes)}}}{{c|}}{{Forget Class}} & "
        rf"\multirow{{2}}{{*}}{{Avg.}} \\",
        " & & " + " & ".join(str(value) for value in args.classes) + r" & \\",
        r"\midrule",
    ]

    lookup = {}
    if not summary.empty:
        lookup = {
            (row.method, row.uncertainty_score, int(row.forget_class)): row
            for row in summary.itertuples(index=False)
        }
    average_lookup = {}
    if not average_summary.empty:
        average_lookup = {
            (row.method, row.uncertainty_score): row
            for row in average_summary.itertuples(index=False)
        }
    for method_index, method in enumerate(args.methods):
        if method_index:
            lines.append(r"\midrule")
        best_by_class = {}
        for forget_class in args.classes:
            eligible = [
                lookup[(method, score, forget_class)].rs_mean
                for score in SCORES
                if (method, score, forget_class) in lookup
                and int(lookup[(method, score, forget_class)].seeds) >= args.min_seeds
            ]
            best_by_class[forget_class] = max(eligible) if eligible else np.nan
        eligible_averages = [
            average_lookup[(method, score)].rs_mean
            for score in SCORES
            if (method, score) in average_lookup
            and int(average_lookup[(method, score)].seeds) >= args.min_seeds
        ]
        best_average = max(eligible_averages) if eligible_averages else np.nan
        for score_index, score in enumerate(SCORES):
            method_cell = (
                rf"\multirow{{{len(SCORES)}}}{{*}}{{{METHOD_LABELS.get(method, method)}}}"
                if score_index == 0
                else ""
            )
            cells = [method_cell, SCORE_LABELS[score]]
            for forget_class in args.classes:
                row = lookup.get((method, score, forget_class))
                cells.append(
                    "-"
                    if row is None
                    else format_cell(
                        row.rs_mean,
                        row.rs_std,
                        int(row.seeds),
                        args,
                        bold=(
                            pd.notna(best_by_class[forget_class])
                            and np.isclose(row.rs_mean, best_by_class[forget_class])
                        ),
                    )
                )
            average_row = average_lookup.get((method, score))
            cells.append(
                "-"
                if average_row is None
                else format_cell(
                    average_row.rs_mean,
                    average_row.rs_std,
                    int(average_row.seeds),
                    args,
                    bold=(
                        pd.notna(best_average)
                        and np.isclose(average_row.rs_mean, best_average)
                    ),
                    show_std=False,
                )
            )
            lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    excluded = METHODS_EXCLUDED_BY_MODEL.get(args.model_name, set())
    args.methods = [method for method in args.methods if method not in excluded]
    frame = load_runs(args)
    summary = summarize(frame)
    average_summary = summarize_across_classes(frame, args.classes)
    output = args.out or (
        args.root / f"cifar10_{args.model_name}_uncertainty_per_class_rs_table.tex"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(make_latex(summary, average_summary, args), encoding="utf-8")
    average_for_csv = average_summary.copy()
    if not average_for_csv.empty:
        average_for_csv["forget_class"] = "Avg."
    csv_summary = pd.concat(
        [summary, average_for_csv], ignore_index=True, sort=False
    )
    csv_summary.to_csv(output.with_suffix(".csv"), index=False)
    print(f"[OK] wrote {output}")
    print(f"[OK] wrote {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
