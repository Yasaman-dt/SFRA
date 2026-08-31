"""Build an appendix table for probe-generation runtime.

This script formats the CSV files produced by
``analysis/benchmark_probe_generation.py`` into one compact LaTeX table grouped
by backbone and dataset. It is meant to answer the reviewer concern about the
cost of generating many Gaussian probes, especially for large-class datasets.

Typical workflow:

  1. Run ``benchmark_probe_generation.py`` once per architecture, e.g. with
     ``--forget_class 0`` and ``--full_probe_construction``.
  2. Run this script to combine the timing CSVs into a paper table.

By default, the script looks for:

  tables/probe_generation_full_timing_resnet18_fg0.csv
  tables/probe_generation_full_timing_swin-t_fg0.csv
  tables/probe_generation_full_timing_vit-b-16_fg0.csv

You can override this with ``--inputs``.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DATASETS = ["cifar10", "cifar100", "tiny_imagenet"]
DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "tiny_imagenet": "TinyImageNet",
}

MODELS = ["resnet18", "swin-t", "vit-b-16"]
MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "swin-t": "Swin-T",
    "vit-b-16": "ViT-B/16",
}

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

METHOD_LABELS = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine probe-generation timing CSVs into a LaTeX table."
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Timing CSVs from benchmark_probe_generation.py. If omitted, the script "
            "uses tables/probe_generation_full_timing_{model}_fg{forget_class}.csv."
        ),
    )
    parser.add_argument(
        "--tables_dir",
        type=Path,
        default=Path("tables"),
        help="Directory used for default input discovery and output.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="Architectures to include, in order.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="Datasets to include, in order.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional method subset. Default uses available methods in main-paper order.",
    )
    parser.add_argument(
        "--forget_class",
        type=int,
        default=0,
        help="Forget class used for default input discovery and caption.",
    )
    parser.add_argument(
        "--metric",
        choices=["full", "per_class", "both"],
        default="full",
        help=(
            "Which timing value to report. 'full' uses extrapolated_full_seconds, "
            "'per_class' uses mean_seconds_per_class, and 'both' prints both."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output prefix. Default: tables/probe_generation_timing_architectures_fg{forget_class}.",
    )
    return parser.parse_args()


def to_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def fmt_time(seconds: float) -> str:
    if math.isnan(seconds):
        return "--"
    if seconds < 1:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def fmt_count(value: float) -> str:
    if math.isnan(value):
        return "--"
    value = int(value)
    if value % 1_000_000 == 0:
        return f"{value // 1_000_000}M"
    if value % 1_000 == 0:
        return f"{value // 1_000}K"
    return f"{value:,}"


def method_rank(method: str) -> tuple[int, str]:
    if method in METHOD_ORDER:
        return METHOD_ORDER.index(method), method
    return len(METHOD_ORDER), method


def default_inputs(tables_dir: Path, models: list[str], forget_class: int) -> list[Path]:
    return [
        tables_dir / f"probe_generation_full_timing_{model}_fg{forget_class}.csv"
        for model in models
    ]


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            print(f"[skip] missing timing CSV: {path}")
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(row)
    if not rows:
        raise FileNotFoundError(
            "No timing rows were loaded. Run benchmark_probe_generation.py first or pass --inputs."
        )
    return rows


def build_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    index: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        model = row.get("model_name", "")
        dataset = row.get("dataset", "")
        method = row.get("method", "")
        if not model or not dataset or not method:
            continue
        index[(model, dataset, method)] = row
    return index


def dataset_probe_setting(rows: list[dict[str, str]], dataset: str) -> str | None:
    dataset_rows = [row for row in rows if row.get("dataset") == dataset]
    if not dataset_rows:
        return None
    n_values = {
        to_float(row.get("accepted_per_class"))
        for row in dataset_rows
        if not math.isnan(to_float(row.get("accepted_per_class")))
    }
    m_values = {
        to_float(row.get("per_retain_for_forget") or row.get("retain_top_k"))
        for row in dataset_rows
        if not math.isnan(to_float(row.get("per_retain_for_forget") or row.get("retain_top_k")))
    }
    if len(n_values) == 1 and len(m_values) == 1:
        n = fmt_count(next(iter(n_values)))
        m = fmt_count(next(iter(m_values)))
        label = DATASET_LABELS.get(dataset, dataset)
        return rf"{label}: $N={n}$ and $M={m}$"
    return None


def available_methods(
    rows: list[dict[str, str]],
    requested: list[str] | None,
) -> list[str]:
    present = {row.get("method", "") for row in rows if row.get("method", "")}
    if requested is not None:
        return [method for method in requested if method in present]
    return sorted(present, key=method_rank)


def metric_cell(row: dict[str, str] | None, metric: str) -> str:
    if row is None:
        return "--"
    per_class = to_float(row.get("mean_seconds_per_class"))
    full = to_float(row.get("extrapolated_full_seconds"))
    if metric == "per_class":
        return fmt_time(per_class)
    if metric == "both":
        return rf"{fmt_time(per_class)} / {fmt_time(full)}"
    return fmt_time(full)


def should_skip_method_for_model(method: str, model: str) -> bool:
    return method == "boundary_shrink" and model in {"swin-t", "vit-b-16"}


def write_latex(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    out_tex: Path,
) -> None:
    index = build_index(rows)
    methods = available_methods(rows, args.methods)
    if not methods:
        raise RuntimeError("No requested methods are present in the timing CSVs.")

    column_spec = "l|l|" + "c" * len(args.datasets)
    dataset_header = " & ".join(
        rf"\multicolumn{{1}}{{c}}{{{DATASET_LABELS.get(dataset, dataset)}}}"
        for dataset in args.datasets
    )
    probe_settings = [
        setting
        for dataset in args.datasets
        if (setting := dataset_probe_setting(rows, dataset)) is not None
    ]
    if len(probe_settings) > 1:
        formatted_probe_settings = (
            ", ".join(probe_settings[:-1]) + ", and " + probe_settings[-1]
        )
    elif probe_settings:
        formatted_probe_settings = probe_settings[0]
    else:
        formatted_probe_settings = ""
    settings_phrase = (
        " The probe settings are " + formatted_probe_settings + "."
        if formatted_probe_settings
        else ""
    )
    if args.metric == "both":
        timing_phrase = "per-retain-class time / full extrapolated time"
    elif args.metric == "per_class":
        timing_phrase = "mean time per retain class"
    else:
        timing_phrase = "full extrapolated probe-construction time"

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        (
            r"\caption{Probe-generation runtime for our proposed SFRA using "
            rf"single-class unlearning checkpoints with forget class {args.forget_class}. "
            rf"Each entry reports the {timing_phrase} for constructing Gaussian "
            rf"feature-space probes.{settings_phrase}}}"
        ),
        r"\label{tab:probe_generation_timing_architectures}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\resizebox{\columnwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        rf"Backbone & Unlearning Method & {dataset_header} \\",
        r"\midrule",
    ]

    first_model = True
    for model in args.models:
        model_methods = [
            method
            for method in methods
            if not should_skip_method_for_model(method, model)
            if any((model, dataset, method) in index for dataset in args.datasets)
        ]
        if not model_methods:
            continue
        if not first_model:
            lines.append(r"\midrule")
        first_model = False
        for method_idx, method in enumerate(model_methods):
            backbone_cell = (
                rf"\multirow{{{len(model_methods)}}}{{*}}{{{MODEL_LABELS.get(model, model)}}}"
                if method_idx == 0
                else ""
            )
            method_cell = METHOD_LABELS.get(method, method.replace("_", r"\_"))
            value_cells = [
                metric_cell(index.get((model, dataset, method)), args.metric)
                for dataset in args.datasets
            ]
            lines.append(
                " & ".join([backbone_cell, method_cell, *value_cells]) + r" \\"
            )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_compact_csv(rows: list[dict[str, str]], args: argparse.Namespace, out_csv: Path) -> None:
    index = build_index(rows)
    methods = available_methods(rows, args.methods)
    fieldnames = ["model_name", "method", *args.datasets]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model in args.models:
            for method in methods:
                if should_skip_method_for_model(method, model):
                    continue
                if not any((model, dataset, method) in index for dataset in args.datasets):
                    continue
                writer.writerow({
                    "model_name": model,
                    "method": method,
                    **{
                        dataset: metric_cell(index.get((model, dataset, method)), args.metric)
                        for dataset in args.datasets
                    },
                })


def main() -> None:
    args = parse_args()
    inputs = args.inputs
    if inputs is None or len(inputs) == 0:
        inputs = default_inputs(args.tables_dir, args.models, args.forget_class)
    rows = load_rows(inputs)

    out_prefix = args.out
    if out_prefix is None:
        out_prefix = args.tables_dir / f"probe_generation_timing_architectures_fg{args.forget_class}"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_tex = out_prefix.with_suffix(".tex")
    out_csv = out_prefix.with_suffix(".csv")

    # The all-results artifact must never be silently replaced by a partial
    # dataset discovery. Require at least one raw timing row for every requested
    # dataset before overwriting that combined table.
    if "all_results" in out_prefix.name:
        present_datasets = {row.get("dataset", "") for row in rows}
        missing_datasets = [
            dataset for dataset in args.datasets if dataset not in present_datasets
        ]
        if missing_datasets:
            raise RuntimeError(
                "Refusing to overwrite the all-results timing table with partial "
                f"inputs; missing datasets: {missing_datasets}"
            )

    write_latex(rows, args, out_tex)
    write_compact_csv(rows, args, out_csv)
    print(f"[saved] {out_tex.resolve()}")
    print(f"[saved] {out_csv.resolve()}")


if __name__ == "__main__":
    main()
