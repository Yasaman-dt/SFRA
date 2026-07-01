"""Make a compact Gaussian-vs-Uniform RS table across forget classes.

This table combines:
  * Gaussian source-free relearning results from results_single_class;
  * Uniform synthesis-ablation results from results_synthesis_ablation.

The intended use is the appendix ablation where the only displayed strategy
factor is the embedding sampling distribution.

Example:
    python plots_tables/sampling_distribution_rs_table.py \
        --dataset cifar10 \
        --model_name resnet18 \
        --classes 0 1 2 3 4 5 6 7 8 9 \
        --bold_best
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


# Change the order here to control how methods appear in the table.
METHOD_ORDER = [
    "original",
    "retrained",
    "random_label",
    "finetune",
    "gradient_ascent",
    "neggrad_plus",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Gaussian-vs-Uniform RS table across forget classes."
    )
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--model_name", "--model", dest="model_name", default="resnet18")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Forget classes to include as columns.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional method subset. Default uses METHOD_ORDER and available data.",
    )
    parser.add_argument(
        "--single_root",
        type=Path,
        default=REPO_ROOT / "results_single_class",
        help="Root containing standardized single-class Gaussian results.",
    )
    parser.add_argument(
        "--ablation_root",
        type=Path,
        default=REPO_ROOT / "results_synthesis_ablation",
        help="Root containing synthesis-ablation results.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "results_synthesis_ablation",
    )
    parser.add_argument(
        "--uniform_forget_selection",
        default="low_confidence",
        help="Uniform ablation forget-probe selection to report.",
    )
    parser.add_argument(
        "--uniform_retain_selection",
        default="high_confidence",
        help="Uniform ablation retain-probe selection to report.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Number of decimals for RS values.",
    )
    parser.add_argument(
        "--bold_best",
        action="store_true",
        help="Bold the best RS between Gaussian and Uniform within each method/class.",
    )
    parser.add_argument(
        "--include_missing_methods",
        action="store_true",
        help="Keep methods even if both Gaussian and Uniform rows are missing.",
    )
    parser.add_argument(
        "--caption",
        default=None,
        help="Optional LaTeX caption.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional LaTeX label.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]+", "_", str(value)).strip("_")


def method_label(method: str) -> str:
    return METHOD_NAME_AND_REF.get(method, method.replace("_", r"\_"))


def latex_dataset_name(dataset: str) -> str:
    mapping = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "tiny_imagenet": "TinyImageNet",
    }
    return mapping.get(dataset, dataset.replace("_", " ").title())


def latex_model_name(model_name: str) -> str:
    mapping = {
        "resnet18": "ResNet-18",
        "vit-s-16": "ViT-S/16",
        "vit-b-16": "ViT-B/16",
        "swin-t": "Swin-T",
        "vgg16": "VGG-16",
    }
    return mapping.get(model_name, model_name)


def fmt_rs(value: float, precision: int) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{precision}f}"


def fmt_rs_latex(value: float, precision: int, bold: bool = False) -> str:
    text = fmt_rs(value, precision)
    if text == "-":
        return text
    if bold:
        return rf"\textbf{{{text}}}"
    return text


def normalize_forget_class(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def load_gaussian_rs(
    single_root: Path,
    dataset: str,
    model_name: str,
    classes: Iterable[int],
) -> pd.DataFrame:
    """Load Gaussian RS2 from standardized single-class revival results."""
    candidates = [
        single_root / f"z_standardized_revival_all_methods_{dataset}_{model_name}.csv",
        single_root / f"z_standardized_selected_all_methods_{dataset}_{model_name}.csv",
        single_root / "z_standardized_revival_all_methods.csv",
        single_root / "z_standardized_selected_all_methods.csv",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            "Could not find standardized single-class results. Tried:\n  "
            + "\n  ".join(str(p) for p in candidates)
        )

    df = pd.read_csv(path)
    required = {"method", "dataset", "model", "forget_class", "RS2"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Gaussian file missing columns {sorted(missing)}: {path}")

    df = df.copy()
    if "phase" in df.columns:
        df = df[df["phase"].astype(str).eq("revival")]
    df = df[
        df["dataset"].astype(str).eq(dataset)
        & df["model"].astype(str).eq(model_name)
    ].copy()
    df["forget_class"] = normalize_forget_class(df["forget_class"])
    df["RS"] = pd.to_numeric(df["RS2"], errors="coerce")
    df = df[df["forget_class"].isin(list(classes))]
    df = (
        df.groupby(["method", "forget_class"], as_index=False)["RS"]
        .mean()
        .assign(sampling="Gaussian")
    )
    return df


def parse_forget_class_from_dir(path: Path) -> Optional[int]:
    match = re.search(r"_fg(\d+)(?:$|[_/])", str(path))
    return int(match.group(1)) if match else None


def load_uniform_rs(
    ablation_root: Path,
    dataset: str,
    model_name: str,
    classes: Iterable[int],
    forget_selection: str,
    retain_selection: str,
) -> pd.DataFrame:
    """Load Uniform RS from synthesis-ablation summary.csv files."""
    class_set = set(classes)
    rows = []

    for method_dir in sorted(p for p in ablation_root.iterdir() if p.is_dir()):
        method = method_dir.name
        for summary_path in method_dir.glob(
            f"{dataset}_{model_name}_lr*_fg*/summary.csv"
        ):
            forget_class = parse_forget_class_from_dir(summary_path.parent)
            if forget_class is None or forget_class not in class_set:
                continue

            summary = pd.read_csv(summary_path)
            required = {
                "distribution",
                "forget_selection",
                "retain_selection",
                "RS_mean",
            }
            missing = required - set(summary.columns)
            if missing:
                raise KeyError(
                    f"Uniform summary missing columns {sorted(missing)}: {summary_path}"
                )

            matched = summary[
                summary["distribution"].astype(str).eq("uniform")
                & summary["forget_selection"].astype(str).eq(forget_selection)
                & summary["retain_selection"].astype(str).eq(retain_selection)
            ].copy()
            if matched.empty:
                continue

            rows.append(
                {
                    "method": method,
                    "forget_class": forget_class,
                    "sampling": "Uniform",
                    "RS": pd.to_numeric(matched["RS_mean"], errors="coerce").mean(),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["method", "forget_class", "sampling", "RS"])

    return (
        pd.DataFrame(rows)
        .groupby(["method", "forget_class", "sampling"], as_index=False)["RS"]
        .mean()
    )


def build_records(
    gaussian_df: pd.DataFrame,
    uniform_df: pd.DataFrame,
    methods: List[str],
    classes: List[int],
    include_missing_methods: bool,
) -> pd.DataFrame:
    data = pd.concat([gaussian_df, uniform_df], ignore_index=True)
    rows = []
    for method in methods:
        method_data = data[data["method"].eq(method)]
        if method_data.empty and not include_missing_methods:
            continue
        for sampling in ["Gaussian", "Uniform"]:
            row = {"method": method, "sampling": sampling}
            sampling_data = method_data[method_data["sampling"].eq(sampling)]
            for class_id in classes:
                value = sampling_data.loc[
                    sampling_data["forget_class"].astype("Int64").eq(class_id),
                    "RS",
                ]
                row[class_id] = float(value.mean()) if not value.empty else np.nan
            if any(pd.notna(row[class_id]) for class_id in classes) or include_missing_methods:
                rows.append(row)
    return pd.DataFrame(rows)


def apply_bold_best(
    table: pd.DataFrame,
    classes: List[int],
    precision: int,
    bold_best: bool,
) -> pd.DataFrame:
    out = table.copy()
    for class_id in classes:
        out[class_id] = out[class_id].astype(object)
    for method, block in table.groupby("method", sort=False):
        for class_id in classes:
            values = pd.to_numeric(block[class_id], errors="coerce")
            best = values.max(skipna=True)
            for idx, value in values.items():
                should_bold = (
                    bold_best
                    and pd.notna(value)
                    and pd.notna(best)
                    and abs(float(value) - float(best)) < 1e-12
                    and values.notna().sum() > 1
                )
                out.at[idx, class_id] = fmt_rs_latex(value, precision, should_bold)
    for class_id in classes:
        out[class_id] = out[class_id].apply(
            lambda x: x if isinstance(x, str) else fmt_rs_latex(x, precision)
        )
    return out


def csv_table(table: pd.DataFrame, classes: List[int]) -> pd.DataFrame:
    out = table.copy()
    out["method"] = out["method"].map(lambda m: METHOD_NAME_AND_REF.get(m, m))
    return out[["method", "sampling"] + classes].rename(
        columns={
            "method": "Unlearning Method",
            "sampling": "Embedding Distribution",
        }
    )


def latex_escape_text(value: str) -> str:
    # Method labels intentionally contain LaTeX citations, so do not escape them.
    return value


def make_latex_table(
    table: pd.DataFrame,
    classes: List[int],
    dataset: str,
    model_name: str,
    caption: Optional[str],
    label: Optional[str],
) -> str:
    if caption is None:
        caption = (
            f"Synthesis-distribution ablation on {latex_dataset_name(dataset)} "
            f"using a {latex_model_name(model_name)} backbone. Each entry reports "
            "the relearning score (RS). Gaussian denotes the original source-free "
            "relearning results, while Uniform uses the same low-confidence "
            "forget-probe and high-confidence retain-probe selection with uniform "
            "embedding sampling."
        )
    if label is None:
        label = f"tab:{slugify(dataset)}_{slugify(model_name)}_sampling_distribution_rs"

    lines = []
    colspec = "l|l|" + "c" * len(classes)
    total_cols = 2 + len(classes)

    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{Unlearning Method} & "
        r"\multirow{2}{*}{Embedding Distribution} & "
        rf"\multicolumn{{{len(classes)}}}{{c}}{{Forget Class}} \\"
    )
    lines.append(
        " & & "
        + " & ".join(str(class_id) for class_id in classes)
        + r" \\"
    )
    lines.append(r"\midrule")

    for method, block in table.groupby("method", sort=False):
        label_text = method_label(method)
        block = block.reset_index(drop=True)
        n_rows = len(block)
        for row_idx, row in block.iterrows():
            method_cell = (
                rf"\multirow{{{n_rows}}}{{*}}{{{label_text}}}"
                if row_idx == 0
                else ""
            )
            cells = [method_cell, row["sampling"]]
            cells.extend(str(row[class_id]) for class_id in classes)
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    classes = list(args.classes)
    requested_methods = args.methods if args.methods is not None else METHOD_ORDER
    methods = [m for m in METHOD_ORDER if m in requested_methods]
    extras = [m for m in requested_methods if m not in methods]
    methods.extend(extras)

    gaussian = load_gaussian_rs(
        args.single_root,
        args.dataset,
        args.model_name,
        classes,
    )
    uniform = load_uniform_rs(
        args.ablation_root,
        args.dataset,
        args.model_name,
        classes,
        args.uniform_forget_selection,
        args.uniform_retain_selection,
    )

    table_numeric = build_records(
        gaussian,
        uniform,
        methods,
        classes,
        args.include_missing_methods,
    )
    if table_numeric.empty:
        raise RuntimeError("No rows found for the requested dataset/model/classes.")

    table_latex_values = apply_bold_best(
        table_numeric,
        classes,
        precision=args.precision,
        bold_best=args.bold_best,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.dataset}_{args.model_name}_gaussian_vs_uniform_rs_table"
    )

    csv_out = args.out_dir / f"{stem}.csv"
    tex_out = args.out_dir / f"{stem}.tex"

    csv_df = csv_table(table_numeric, classes)
    for class_id in classes:
        csv_df[class_id] = pd.to_numeric(csv_df[class_id], errors="coerce").round(
            args.precision
        )
    csv_df.to_csv(csv_out, index=False)

    tex = make_latex_table(
        table_latex_values,
        classes,
        args.dataset,
        args.model_name,
        args.caption,
        args.label,
    )
    tex_out.write_text(tex, encoding="utf-8")

    print(f"[OK] wrote {csv_out}")
    print(f"[OK] wrote {tex_out}")
    print(
        "[info] Uniform row uses "
        f"forget_selection={args.uniform_forget_selection}, "
        f"retain_selection={args.uniform_retain_selection}"
    )


if __name__ == "__main__":
    main()
