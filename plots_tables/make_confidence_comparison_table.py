"""Convert retain/forget assigned-confidence CSV into a LaTeX table.

Example:
    python plots_tables/make_confidence_comparison_table.py \
      --csv results/bad_teacher/confidence_tables/cifar10_resnet18_bad_teacher_lr0.001_fg0_correct_retain_vs_forget_assigned.csv \
      --method "Bad Teacher" \
      --dataset "CIFAR-10" \
      --model_name "ResNet-18" \
      --forget_class 0
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make a LaTeX confidence-comparison table from a CSV."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--method", default="Bad Teacher")
    parser.add_argument("--dataset", default="CIFAR-10")
    parser.add_argument("--model_name", default="ResNet-18")
    parser.add_argument("--forget_class", type=int, default=None)
    parser.add_argument("--precision", type=int, default=3)
    parser.add_argument(
        "--caption",
        default=None,
        help="Optional custom caption.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional custom LaTeX label.",
    )
    parser.add_argument(
        "--table_star",
        action="store_true",
        help="Use table* instead of table.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]+", "_", str(value)).strip("_")


def fmt(value, precision: int) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{precision}f}"


def infer_forget_class(path: Path) -> int | None:
    match = re.search(r"_fg(\d+)_", path.name)
    return int(match.group(1)) if match else None


def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "retain_class",
        "retain_correct_count",
        "retain_correct_confidence",
        "forget_assigned_count",
        "forget_assigned_confidence",
        "confidence_gap",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns {sorted(missing)} in {path}")

    df = df.sort_values("retain_class").copy()
    return df


def to_latex(
    df: pd.DataFrame,
    out_path: Path,
    method: str,
    dataset: str,
    model_name: str,
    forget_class: int | None,
    precision: int,
    caption: str | None,
    label: str | None,
    table_star: bool,
) -> str:
    env = "table*" if table_star else "table"
    if caption is None:
        if forget_class is None:
            caption = (
                f"Confidence assigned to retain classes by the {method} "
                f"unlearned {model_name} model on {dataset}."
            )
        else:
            caption = (
                f"Confidence assigned to retain classes by the {method} "
                f"unlearned {model_name} model on {dataset} when class "
                f"{forget_class} is forgotten. For each retain class, we compare "
                "correctly classified retain samples with forget-class samples "
                "that are assigned to that retain class."
            )
    if label is None:
        fg = f"fg{forget_class}" if forget_class is not None else "fg"
        label = f"tab:{slugify(method)}_{slugify(dataset)}_{slugify(model_name)}_{fg}_confidence"

    lines = []
    lines.append(rf"\begin{{{env}}}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(r"\begin{tabular}{c|cc|cc|c}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{Retain Class} & "
        r"\multicolumn{2}{c|}{Correct Retain Samples} & "
        r"\multicolumn{2}{c|}{Forget Samples Assigned to Class} & "
        r"\multirow{2}{*}{Gap} \\"
    )
    lines.append(
        r" & Count & Avg. Conf. & Count & Avg. Conf. & \\"
    )
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        cells = [
            str(int(row["retain_class"])),
            str(int(row["retain_correct_count"])),
            fmt(row["retain_correct_confidence"], precision),
            str(int(row["forget_assigned_count"])),
            fmt(row["forget_assigned_confidence"], precision),
            fmt(row["confidence_gap"], precision),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\end{{{env}}}")
    latex = "\n".join(lines) + "\n"
    out_path.write_text(latex, encoding="utf-8")
    return latex


def main() -> None:
    args = parse_args()
    df = load_table(args.csv)
    forget_class = args.forget_class
    if forget_class is None:
        forget_class = infer_forget_class(args.csv)

    out = args.out or args.csv.with_suffix(".paper_table.tex")
    to_latex(
        df=df,
        out_path=out,
        method=args.method,
        dataset=args.dataset,
        model_name=args.model_name,
        forget_class=forget_class,
        precision=args.precision,
        caption=args.caption,
        label=args.label,
        table_star=args.table_star,
    )
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
