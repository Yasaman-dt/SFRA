"""Create a forget-class-column table from synthesis-ablation summaries.

The script scans directories produced by
``ablation.synthesis_strategy_ablation`` and pivots one dataset/backbone with
one or more unlearning methods into a table whose columns are forgotten classes.

Example:
    python plots_tables/synthesis_ablation_class_table.py \
      --root results_synthesis_ablation/results_synthesis_ablation_resnet18 \
      --method bad_teacher --dataset cifar10 --model_name resnet18 \
      --unlearn_lr 0.001 --classes 0 1 2 3 4 5 6 7 8 9
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METRICS = ("RS",)
METRIC_LABELS = {
    "baseline_forget_acc": r"$A_f^{\mathrm{un}}$",
    "baseline_retain_acc": r"$A_r^{\mathrm{un}}$",
    "relearned_forget_acc": r"$A_f^{\mathrm{re}}$",
    "relearned_retain_acc": r"$A_r^{\mathrm{re}}$",
    "forget_gain": r"$\Delta A_f$",
    "retain_drop": r"$\Delta A_r$",
    "RS": "RS",
    "best_epoch": "Epoch",
}
STRATEGY_LABELS = {
    "dist=gaussian__forget=low_confidence__retain=high_confidence":
        "Gaussian + Low-conf. Forget + High-conf. Retain",
    "dist=uniform__forget=low_confidence__retain=high_confidence":
        "Uniform + Low-conf. Forget + High-conf. Retain",
    "dist=gaussian__forget=high_confidence__retain=high_confidence":
        "Gaussian + High-conf. Forget + High-conf. Retain",
    "dist=gaussian__forget=random__retain=high_confidence":
        "Gaussian + Random Forget + High-conf. Retain",
    "dist=gaussian__forget=low_confidence__retain=random":
        "Gaussian + Low-conf. Forget + Random Retain",
}
STRATEGY_ORDER = list(STRATEGY_LABELS)

# Change the order here to control how methods appear in the table.
METHOD_ORDER = [
    "original",
    "retrained",
    "finetune",
    "gradient_ascent",
    "neggrad_plus",
    "random_label",
    "boundary_shrink",
    "boundary_expand",
    "l2ul_adv",
    "l2ul_imp",
    "fisher",
    "wood_fisher",
    "scrub",
    "bad_teacher",
    "salun",
    "delete",
]


METHOD_NAME_AND_REF = {
    "original": r"Original",
    "retrained": r"Retrained",
    "finetune": r"Finetune \cite{golatkar2020eternal}",
    "gradient_ascent": r"Negative Gradient \cite{golatkar2020eternal}",
    "neggrad_plus": r"Negative Gradient+ \cite{kurmanji2023towards}",
    "random_label": r"Random Label \cite{hayase2020selective}",
    "boundary_shrink": r"Boundary Shrink \cite{chen2023boundary}",
    "boundary_expand": r"Boundary Expand \cite{chen2023boundary}",
    "l2ul_adv": r"Learn to Unlearn \cite{cha2024learning}",
    "l2ul_imp": r"Learn to Unlearn Adv+IMP \cite{cha2024learning}",
    "fisher": r"Fisher",
    "wood_fisher": r"WoodFisher",
    "scrub": r"SCRUB \cite{kurmanji2023towards}",
    "bad_teacher": r"Bad Teacher \cite{chundawat2023can}",
    "salun": r"SalUn \cite{fan2023salun}",
    "delete": r"DELETE \cite{zhou2025decoupled}",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pivot synthesis-ablation results across forget classes."
    )
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Architecture-specific synthesis-results root. By default, this "
            "is derived from --model_name."
        ),
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Single unlearning method (backward-compatible shorthand).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Multiple unlearning methods to place in the same table.",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", "--model_name", dest="model_name", required=True)
    parser.add_argument(
        "--unlearn_lr",
        type=float,
        default=None,
        help="Filter checkpoint learning rate. Required if a class has multiple rates.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Forget-class columns. Defaults to all discovered classes.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        choices=["RS"],
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Optional exact strategy identifiers. Defaults to all discovered strategies.",
    )
    parser.add_argument(
        "--exclude_strategies",
        nargs="+",
        default=None,
        help="Exact strategy identifiers to omit from the generated CSV and LaTeX tables.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Audit seed used for the selection-strategy comparison. The "
            "default is 0 because all selection controls were run with seed 0."
        ),
    )
    parser.add_argument(
        "--accuracy_as_fraction",
        action="store_true",
        help="Divide accuracy and gain/drop metrics by 100.",
    )
    parser.add_argument(
        "--bold_best",
        action="store_true",
        help="Bold the largest value within each metric/class column.",
    )
    parser.add_argument(
        "--bold_scope",
        choices=["method", "global"],
        default="method",
        help="Compute bold best values within each method or across the full table.",
    )
    parser.add_argument(
        "--output_prefix",
        default=None,
        help="Output path without extension.",
    )
    return parser.parse_args()


def parse_folder_name(name, dataset, model_name):
    pattern = re.compile(
        rf"^{re.escape(dataset)}_{re.escape(model_name)}_lr"
        rf"(?P<lr>[-+0-9.eE]+)_fg(?P<class_id>\d+)$"
    )
    match = pattern.match(name)
    if not match:
        return None
    return float(match.group("lr")), int(match.group("class_id"))


def discover_summaries(args, method):
    method_root = Path(args.root) / method
    if not method_root.exists():
        raise FileNotFoundError(f"Method directory not found: {method_root}")

    records = []
    # Read the append-only run records rather than summary.csv. Later ablation
    # invocations may overwrite summary.csv with only their active strategies,
    # whereas runs.csv retains the complete selection-strategy experiment.
    for summary_path in sorted(method_root.glob("*/runs.csv")):
        parsed = parse_folder_name(
            summary_path.parent.name, args.dataset, args.model_name
        )
        if parsed is None:
            continue
        learning_rate, class_id = parsed
        if args.unlearn_lr is not None and not np.isclose(
            learning_rate, args.unlearn_lr
        ):
            continue
        records.append(
            {
                "class_id": class_id,
                "unlearn_lr": learning_rate,
                "path": summary_path,
            }
        )

    if not records:
        raise FileNotFoundError(
            "No matching summary.csv files were found for "
            f"method={method}, dataset={args.dataset}, "
            f"model={args.model_name}, lr={args.unlearn_lr}."
        )

    by_class = {}
    for record in records:
        by_class.setdefault(record["class_id"], []).append(record)
    duplicates = {
        class_id: values for class_id, values in by_class.items() if len(values) > 1
    }
    if duplicates:
        details = "; ".join(
            f"class {class_id}: {[item['unlearn_lr'] for item in values]}"
            for class_id, values in duplicates.items()
        )
        raise ValueError(
            "Multiple learning rates match the same forget class. "
            f"Pass --unlearn_lr explicitly. {details}"
        )
    return records


def load_long_frame(records, method, seed=None):
    frames = []
    for record in records:
        frame = pd.read_csv(record["path"])
        if seed is not None and "seed" in frame.columns:
            frame = frame[
                pd.to_numeric(frame["seed"], errors="coerce").eq(seed)
            ]
        if "RS_mean" not in frame.columns and "RS" in frame.columns:
            dedupe_columns = [
                column for column in ["strategy", "seed"]
                if column in frame.columns
            ]
            if dedupe_columns:
                frame = frame.drop_duplicates(
                    subset=dedupe_columns, keep="last"
                )
            group_columns = [
                column for column in [
                    "strategy", "distribution", "forget_selection",
                    "retain_selection", "uncertainty_score",
                ]
                if column in frame.columns
            ]
            frame = (
                frame.groupby(group_columns, as_index=False, dropna=False)["RS"]
                .agg(["mean", "std"])
                .reset_index()
                .rename(columns={"mean": "RS_mean", "std": "RS_std"})
            )
        frame["forget_class"] = record["class_id"]
        frame["unlearn_lr"] = record["unlearn_lr"]
        frame["unlearning_method"] = method
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def strategy_order(frame, methods, requested):
    available = list(
        dict.fromkeys(
            zip(frame["unlearning_method"].tolist(), frame["strategy"].tolist())
        )
    )
    if requested:
        requested_keys = [
            (method, strategy)
            for method in methods
            for strategy in requested
            if (method, strategy) in available
        ]
        missing = [
            strategy
            for strategy in requested
            if not any((method, strategy) in available for method in methods)
        ]
        if missing:
            raise ValueError(f"Requested strategies not found: {missing}")
        return requested_keys
    ordered = []
    for method in methods:
        method_available = [
            strategy for available_method, strategy in available
            if available_method == method
        ]
        ordered.extend(
            (method, strategy)
            for strategy in STRATEGY_ORDER
            if strategy in method_available
        )
        ordered.extend(
            (method, strategy)
            for strategy in method_available
            if strategy not in STRATEGY_ORDER
        )
    return ordered


def maybe_scale(metric, value, accuracy_as_fraction):
    if not accuracy_as_fraction:
        return value
    if metric in {
        "baseline_forget_acc",
        "baseline_retain_acc",
        "relearned_forget_acc",
        "relearned_retain_acc",
        "forget_gain",
        "retain_drop",
    }:
        return value / 100.0
    return value


def numeric_tables(frame, strategies, metrics, classes, accuracy_as_fraction):
    means = {}
    stds = {}
    for strategy_key in strategies:
        method, strategy = strategy_key
        for metric in metrics:
            key = (strategy_key, metric)
            means[key], stds[key] = {}, {}
            for class_id in classes:
                rows = frame[
                    (frame["unlearning_method"] == method)
                    & (frame["strategy"] == strategy)
                    & (frame["forget_class"] == class_id)
                ]
                mean_column = f"{metric}_mean"
                std_column = f"{metric}_std"
                if rows.empty or mean_column not in rows:
                    means[key][class_id] = np.nan
                    stds[key][class_id] = np.nan
                    continue
                means[key][class_id] = maybe_scale(
                    metric, float(rows.iloc[0][mean_column]), accuracy_as_fraction
                )
                std_value = (
                    float(rows.iloc[0][std_column])
                    if std_column in rows and pd.notna(rows.iloc[0][std_column])
                    else np.nan
                )
                stds[key][class_id] = maybe_scale(
                    metric, std_value, accuracy_as_fraction
                )
    return means, stds


def format_value(mean, std, metric, precision):
    if pd.isna(mean):
        return "--"
    metric_precision = 2 if metric == "RS" else 1
    if pd.isna(std):
        return f"{mean:.{metric_precision}f}"
    return (
        rf"${mean:.{metric_precision}f}"
        rf"\pm{std:.{metric_precision}f}$"
    )


def latex_escape(text):
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
    )


def write_latex(
    path,
    means,
    stds,
    strategies,
    metrics,
    classes,
    precision,
    bold_best,
    bold_scope,
    frame_lookup,
    caption,
    label,
    compact=False,
):
    best = {}
    if bold_best:
        method_names = list(dict.fromkeys(key[0] for key in strategies))
        scopes = method_names if bold_scope == "method" else ["__global__"]
        for scope in scopes:
            scoped_strategies = (
                [key for key in strategies if key[0] == scope]
                if bold_scope == "method"
                else strategies
            )
            for metric in metrics:
                for class_id in classes:
                    values = [
                        means[(strategy, metric)][class_id]
                        for strategy in scoped_strategies
                    ]
                    finite = [value for value in values if not pd.isna(value)]
                    best[(scope, metric, class_id)] = (
                        max(finite) if finite else np.nan
                    )

    # Vertical lines are placed after Unlearning Method, Forget Probes,
    # and Retain Probes. Sampling is intentionally omitted from the LaTeX table.
    class_columns = r"@{\hspace{5pt}}".join("c" for _ in classes)
    column_spec = r"l|l|l|" + class_columns
    last_class_column = 3 + len(classes)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\renewcommand{{\arraystretch}}{{{'1.35' if compact else '1'}}}",
        r"\scriptsize" if compact else r"\normalsize",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        r"\multirow{2}{*}{Unlearning Method} "
        r"& \multirow{2}{*}{\shortstack{Forget\\Probes}} "
        r"& \multirow{2}{*}{\shortstack{Retain\\Probes}} "
        rf"& \multicolumn{{{len(classes)}}}{{c}}{{Forget Class}} \\",
        rf"\cline{{4-{last_class_column}}}",
        " & & & "
        + " & ".join(str(class_id) for class_id in classes)
        + r" \\",
        r"\hline",
        r"\hline",
    ]
    if compact:
        lines.pop()

    method_order = list(dict.fromkeys(strategy[0] for strategy in strategies))
    for method_index, method in enumerate(method_order):
        method_strategies = [
            strategy for strategy in strategies if strategy[0] == method
        ]
        method_row_count = len(method_strategies)
        first_method_row = True
        for strategy_index, strategy_key in enumerate(method_strategies):
            _, strategy = strategy_key
            strategy_rows = frame_lookup[strategy_key]
            distribution = strategy_rows["distribution"]
            forget_selection = strategy_rows["forget_selection"]
            retain_selection = strategy_rows["retain_selection"]
            selection_labels = {
                "low_confidence": "Low Conf.",
                "high_confidence": "High Conf.",
                "random": "Random",
            }
            forget_label = selection_labels.get(
                forget_selection, forget_selection.replace("_", " ").title()
            )
            retain_label = selection_labels.get(
                retain_selection, retain_selection.replace("_", " ").title()
            )
            cells = []
            for class_id in classes:
                for metric in metrics:
                    mean = means[(strategy_key, metric)][class_id]
                    std = stds[(strategy_key, metric)][class_id]
                    cell = format_value(mean, std, metric, precision)
                    if (
                        bold_best
                        and not pd.isna(mean)
                        and np.isclose(
                            mean,
                            best[
                                (
                                    method if bold_scope == "method" else "__global__",
                                    metric,
                                    class_id,
                                )
                            ],
                        )
                    ):
                        cell = rf"\textbf{{{cell}}}"
                    # Split tables keep compact numeric cells. The combined
                    # table applies one consistent font size to all content.
                    if not compact:
                        cell = rf"{{\scriptsize {cell}}}"
                    cells.append(cell)

            # Values from METHOD_NAME_AND_REF intentionally contain LaTeX commands
            # such as \cite{...}, so do not pass the mapped label to latex_escape().
            method_label = METHOD_NAME_AND_REF.get(
                method, latex_escape(method.replace("_", " ").title())
            )
            method_cell = (
                rf"\multirow{{{method_row_count}}}{{*}}{{{method_label}}}"
                if first_method_row
                else ""
            )
            lines.append(
                f"{method_cell} & {latex_escape(forget_label)} & "
                f"{latex_escape(retain_label)} & "
                + " & ".join(cells)
                + r" \\"
            )
            first_method_row = False
            if strategy_index != len(method_strategies) - 1:
                lines.append(
                    r"\cline{2-" + str(3 + len(classes)) + "}"
                )
        if method_index != len(method_order) - 1:
            lines.append(r"\hline")
            if not compact:
                lines.append(r"\hline")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_csv(path, means, stds, strategies, metrics, classes, precision, frame_lookup):
    rows = []
    for strategy_key in strategies:
        method, strategy = strategy_key
        strategy_rows = frame_lookup[strategy_key]
        row = {
            "unlearning_method": method,
            "sampling": strategy_rows["distribution"],
            "forget_probes": strategy_rows["forget_selection"],
            "retain_probes": strategy_rows["retain_selection"],
            "strategy": strategy,
        }
        for class_id in classes:
            metric = "RS"
            row[str(class_id)] = format_value(
                means[(strategy_key, metric)][class_id],
                stds[(strategy_key, metric)][class_id],
                metric,
                precision,
            ).replace("$", "")
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    args = parse_args()
    if args.root is None:
        args.root = str(
            Path("results_synthesis_ablation")
            / f"results_synthesis_ablation_{args.model_name}"
        )
    # This table reports only RS; class IDs are the column names.
    args.metrics = ["RS"]
    if args.methods is not None:
        methods = list(dict.fromkeys(args.methods))
        if args.method is not None:
            methods = list(dict.fromkeys([args.method] + methods))
    elif args.method is not None:
        methods = [args.method]
    else:
        raise ValueError("Provide --method or --methods.")

    # The table follows METHOD_ORDER. Methods not listed there are placed last.
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    methods = sorted(
        methods,
        key=lambda method: (method_rank.get(method, len(METHOD_ORDER)), method),
    )

    frames = []
    for method in methods:
        records = discover_summaries(args, method)
        frames.append(load_long_frame(records, method, args.seed))
    frame = pd.concat(frames, ignore_index=True)
    classes = (
        args.classes
        if args.classes is not None
        else sorted(frame["forget_class"].unique().tolist())
    )
    strategies = strategy_order(frame, methods, args.strategies)
    if args.exclude_strategies:
        excluded = set(args.exclude_strategies)
        strategies = [
            strategy_key
            for strategy_key in strategies
            if strategy_key[1] not in excluded
        ]
        if not strategies:
            raise ValueError("No strategies remain after applying --exclude_strategies.")
    frame_lookup = {}
    for strategy_key in strategies:
        method, strategy = strategy_key
        row = frame[
            (frame["unlearning_method"] == method)
            & (frame["strategy"] == strategy)
        ].iloc[0]
        frame_lookup[strategy_key] = {
            "distribution": row["distribution"],
            "forget_selection": row["forget_selection"],
            "retain_selection": row["retain_selection"],
        }
    means, stds = numeric_tables(
        frame,
        strategies,
        args.metrics,
        classes,
        args.accuracy_as_fraction,
    )

    if args.output_prefix:
        prefix = Path(args.output_prefix)
    else:
        lr_tag = "all_lr" if args.unlearn_lr is None else f"lr{args.unlearn_lr:g}"
        prefix = (
            Path(args.root)
            / f"{args.dataset}_{args.model_name}_{lr_tag}_class_table"
        )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(
        prefix.with_suffix(".csv"),
        means,
        stds,
        strategies,
        args.metrics,
        classes,
        args.precision,
        frame_lookup,
    )
    # Also write one table containing every requested unlearning method.
    combined_latex_path = prefix.with_suffix(".tex")
    write_latex(
        combined_latex_path,
        means,
        stds,
        strategies,
        args.metrics,
        classes,
        args.precision,
        args.bold_best,
        args.bold_scope,
        frame_lookup,
        caption=(
            "Synthesis-strategy ablation on CIFAR-10 using a ResNet-18 "
            "backbone for all unlearning methods. "
            r"Each entry reports $\mathrm{RS}$."
        ),
        label="tab:synthesis_ablation_all_methods",
        compact=True,
    )
    print(f"[saved] {prefix.with_suffix('.csv').resolve()}")
    print(f"[saved] {combined_latex_path.resolve()}")


if __name__ == "__main__":
    main()
