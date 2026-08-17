"""Aggregate confidence-comparison CSVs across forget classes.

This script reads the per-forget-class files produced by
retain_forget_assigned_confidence.py, e.g.

  results/bad_teacher/confidence_tables/
    cifar10_resnet18_bad_teacher_lr0.001_fg0_correct_retain_vs_forget_assigned.csv
    ...
    cifar10_resnet18_bad_teacher_lr0.001_fg9_correct_retain_vs_forget_assigned.csv

and makes a compact table with one row per forgotten class/checkpoint.

For each forgotten class, it reports:
  * weighted average confidence over correctly classified retain samples;
  * weighted average confidence over forget samples assigned to retain classes;
  * the confidence gap.

The weighting is by the corresponding sample count, so classes with more
assigned samples contribute proportionally more to the average.

Example:
    python plots_tables/aggregate_confidence_by_forget_class.py \
      --root results/bad_teacher/confidence_tables \
      --method bad_teacher \
      --dataset cifar10 \
      --model_name resnet18 \
      --lr 0.001 \
      --classes 0 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


METHOD_LABELS = {
    "bad_teacher": r"Bad Teacher",
    "neggrad_plus": r"Negative Gradient+",
    "gradient_ascent": r"Negative Gradient",
    "random_label": r"Random Label",
    "finetune": r"Finetune",
    "boundary_shrink": r"Boundary Shrink",
    "delete": r"DELETE",
    "salun": r"SalUn",
    "scrub": r"SCRUB",
}


DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "tiny_imagenet": "TinyImageNet",
}


MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "vit-s-16": "ViT-S/16",
    "vit-b-16": "ViT-B/16",
    "swin-t": "Swin-T",
    "vgg16": "VGG-16",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate retain/forget confidence tables across forget classes."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/bad_teacher/confidence_tables"),
        help="Directory containing *_correct_retain_vs_forget_assigned.csv files.",
    )
    parser.add_argument("--method", default="bad_teacher")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--model_name", "--model", dest="model_name", default="resnet18")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Forgotten classes/checkpoints to include.",
    )
    parser.add_argument("--precision", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--caption", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--allow_any_lr",
        action="store_true",
        help=(
            "If the exact --lr file is missing for a forget class, use any "
            "matching lr file for that method/dataset/model/forget class."
        ),
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip missing forget-class CSVs instead of raising an error.",
    )
    parser.add_argument(
        "--include_counts",
        action="store_true",
        help="Also include the total retain/forget counts used in the weighted averages.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]+", "_", str(value)).strip("_")


def fmt_lr(lr: float) -> str:
    return f"{lr:g}"


def fmt_float(value: float, precision: int) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{precision}f}"


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def table_average(summary: pd.DataFrame, include_counts: bool) -> dict[str, float]:
    """Compute the final average row shown at the bottom of the table."""
    if include_counts:
        retain_count = pd.to_numeric(
            summary["retain_correct_count"], errors="coerce"
        ).fillna(0.0)
        forget_count = pd.to_numeric(
            summary["forget_assigned_count"], errors="coerce"
        ).fillna(0.0)
        retain_conf = pd.to_numeric(
            summary["retain_correct_weighted_confidence"], errors="coerce"
        )
        forget_conf = pd.to_numeric(
            summary["forget_assigned_weighted_confidence"], errors="coerce"
        )
        retain_avg = weighted_average(retain_conf, retain_count)
        forget_avg = weighted_average(forget_conf, forget_count)
        return {
            "retain_correct_count": retain_count.sum(),
            "retain_correct_weighted_confidence": retain_avg,
            "forget_assigned_count": forget_count.sum(),
            "forget_assigned_weighted_confidence": forget_avg,
            "confidence_gap": retain_avg - forget_avg,
        }

    retain_avg = pd.to_numeric(
        summary["retain_correct_weighted_confidence"], errors="coerce"
    ).mean()
    forget_avg = pd.to_numeric(
        summary["forget_assigned_weighted_confidence"], errors="coerce"
    ).mean()
    return {
        "retain_correct_weighted_confidence": retain_avg,
        "forget_assigned_weighted_confidence": forget_avg,
        "confidence_gap": retain_avg - forget_avg,
    }


def csv_path_for(
    root: Path,
    dataset: str,
    model_name: str,
    method: str,
    lr: float,
    forget_class: int,
) -> Path:
    return root / (
        f"{dataset}_{model_name}_{method}_lr{fmt_lr(lr)}_fg{forget_class}"
        "_correct_retain_vs_forget_assigned.csv"
    )


def resolve_csv_path(
    root: Path,
    dataset: str,
    model_name: str,
    method: str,
    lr: float,
    forget_class: int,
    allow_any_lr: bool,
) -> Path | None:
    exact = csv_path_for(root, dataset, model_name, method, lr, forget_class)
    if exact.is_file() or not allow_any_lr:
        return exact

    matches = sorted(
        root.glob(
            f"{dataset}_{model_name}_{method}_lr*_fg{forget_class}"
            "_correct_retain_vs_forget_assigned.csv"
        )
    )
    return matches[0] if matches else exact


def load_and_aggregate(
    path: Path,
    forget_class: int,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing confidence CSV for fg{forget_class}: {path}")

    df = pd.read_csv(path)
    required = {
        "retain_correct_count",
        "retain_correct_confidence",
        "forget_assigned_count",
        "forget_assigned_confidence",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns {sorted(missing)} in {path}")

    retain_count = int(pd.to_numeric(df["retain_correct_count"], errors="coerce").sum())
    forget_count = int(pd.to_numeric(df["forget_assigned_count"], errors="coerce").sum())
    retain_conf = weighted_average(
        df["retain_correct_confidence"],
        df["retain_correct_count"],
    )
    forget_conf = weighted_average(
        df["forget_assigned_confidence"],
        df["forget_assigned_count"],
    )

    return {
        "forget_class": forget_class,
        "source_file": path.name,
        "retain_correct_count": retain_count,
        "retain_correct_weighted_confidence": retain_conf,
        "forget_assigned_count": forget_count,
        "forget_assigned_weighted_confidence": forget_conf,
        "confidence_gap": retain_conf - forget_conf,
    }


def make_latex(
    summary: pd.DataFrame,
    method: str,
    dataset: str,
    model_name: str,
    precision: int,
    caption: str | None,
    label: str | None,
    include_counts: bool,
) -> str:
    method_label = METHOD_LABELS.get(method, method.replace("_", " ").title())
    dataset_label = DATASET_LABELS.get(dataset, dataset)
    model_label = MODEL_LABELS.get(model_name, model_name)

    if caption is None:
        caption = (
            f"Average confidence comparison for {method_label} on "
            f"{dataset_label} for {model_label} backbone. For each forget "
            "class, we report the weighted average confidence of correctly "
            "classified retain samples and forget-class samples assigned to "
            "retain classes."
        )
    if label is None:
        label = (
            f"tab:{slugify(method)}_{slugify(dataset)}_{slugify(model_name)}"
            "_weighted_confidence"
        )

    if include_counts:
        colspec = "c|cc|cc|c"
        header1 = (
            r"\multirow{2}{*}{Forgotten Class} & "
            r"\multicolumn{2}{c|}{Correct Retain} & "
            r"\multicolumn{2}{c|}{Forget Assigned to Retain} & "
            r"\multirow{2}{*}{Gap} \\"
        )
        header2 = r" & Count & Avg. Conf. & Count & Avg. Conf. & \\"
    else:
        colspec = "c|ccc"
        header1 = (
            r"Forget Class & \shortstack{Correct Retain\\Conf.} & "
            r"\shortstack{Forget Assigned\\Conf.} & Gap \\"
        )
        header2 = None

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        header1,
    ]
    if header2 is not None:
        lines.append(header2)
    lines.append(r"\midrule")

    if include_counts:
        for _, row in summary.iterrows():
            cells = [
                str(int(row["forget_class"])),
                str(int(row["retain_correct_count"])),
                fmt_float(row["retain_correct_weighted_confidence"], precision),
                str(int(row["forget_assigned_count"])),
                fmt_float(row["forget_assigned_weighted_confidence"], precision),
                fmt_float(row["confidence_gap"], precision),
            ]
            lines.append(" & ".join(cells) + r" \\")
        avg = table_average(summary, include_counts=True)
        lines.append(r"\midrule")
        cells = [
            r"\textbf{Average}",
            str(int(avg["retain_correct_count"])),
            fmt_float(avg["retain_correct_weighted_confidence"], precision),
            str(int(avg["forget_assigned_count"])),
            fmt_float(avg["forget_assigned_weighted_confidence"], precision),
            fmt_float(avg["confidence_gap"], precision),
        ]
        lines.append(" & ".join(cells) + r" \\")
    else:
        for _, row in summary.iterrows():
            cells = [
                str(int(row["forget_class"])),
                fmt_float(row["retain_correct_weighted_confidence"], precision),
                fmt_float(row["forget_assigned_weighted_confidence"], precision),
                fmt_float(row["confidence_gap"], precision),
            ]
            lines.append(" & ".join(cells) + r" \\")
        avg = table_average(summary, include_counts=False)
        lines.append(r"\midrule")
        cells = [
            r"\textbf{Average}",
            fmt_float(avg["retain_correct_weighted_confidence"], precision),
            fmt_float(avg["forget_assigned_weighted_confidence"], precision),
            fmt_float(avg["confidence_gap"], precision),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = []
    for forget_class in args.classes:
        path = resolve_csv_path(
            args.root,
            args.dataset,
            args.model_name,
            args.method,
            args.lr,
            forget_class,
            args.allow_any_lr,
        )
        if not path.is_file() and args.skip_missing:
            print(f"[WARN] skipping missing fg{forget_class}: {path}")
            continue
        rows.append(load_and_aggregate(path, forget_class))

    summary = pd.DataFrame(rows).sort_values("forget_class")
    out_base = args.out
    if out_base is None:
        out_base = args.root / (
            f"{args.dataset}_{args.model_name}_{args.method}_lr{fmt_lr(args.lr)}"
            "_weighted_confidence_by_forget_class"
        )
    if out_base.suffix in {".csv", ".tex"}:
        csv_out = out_base.with_suffix(".csv")
        tex_out = out_base.with_suffix(".tex")
    else:
        csv_out = Path(str(out_base) + ".csv")
        tex_out = Path(str(out_base) + ".tex")

    summary.to_csv(csv_out, index=False)
    latex = make_latex(
        summary,
        args.method,
        args.dataset,
        args.model_name,
        args.precision,
        args.caption,
        args.label,
        args.include_counts,
    )
    tex_out.write_text(latex, encoding="utf-8")
    print(f"[OK] wrote {csv_out}")
    print(f"[OK] wrote {tex_out}")


if __name__ == "__main__":
    main()
