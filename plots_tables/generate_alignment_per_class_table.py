"""Create a per-forget-class table from combined synthetic/real alignment results."""

import argparse
from pathlib import Path

import pandas as pd


METHOD_ORDER = [
    "retrained", "finetune", "gradient_ascent", "neggrad_plus", "random_label",
    "boundary_shrink", "l2ul_adv", "scrub", "bad_teacher", "salun", "delete",
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
METRICS = [
    ("RS", r"$\mathrm{RS}$", False),
    ("alignment_inner_product", r"$A_{\mathrm{IP}}$", True),
    ("alignment_cosine", r"$A_{\mathrm{cos}}$", True),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path(
            "results_single_class/analysis/alignment_analysis/combined/"
            "alignment_per_class_seed_combined.csv"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "results_single_class/analysis/alignment_analysis/combined/"
            "alignment_per_class_table_combined.tex"
        ),
    )
    parser.add_argument("--classes", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--precision", type=int, default=2)
    return parser.parse_args()


def format_cell(mean, std, precision, signed):
    if pd.isna(mean):
        return "--"
    mean_text = f"{mean:+.{precision}f}" if signed else f"{mean:.{precision}f}"
    if pd.isna(std):
        return rf"${mean_text}$"
    return rf"${mean_text}{{\scriptstyle\,\pm {std:.{precision}f}}}$"


def build_table(data, classes, precision):
    data = data[data.forget_class.isin(classes)].copy()
    grouped = data.groupby(["method", "forget_class"])[
        [metric for metric, _, _ in METRICS]
    ].agg(["mean", "std"])
    present = set(data.method.astype(str))
    methods = [method for method in METHOD_ORDER if method in present]
    methods += sorted(present - set(methods))
    class_header = " & ".join(str(value) for value in classes)
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Per-forget-class synthetic--real alignment and $\mathrm{RS}$ values on CIFAR-10 using ResNet-18. Each forget class column corresponds to a separate unlearned checkpoint in which that class is designated for forgetting. $A_{\mathrm{IP}}$ and $A_{\mathrm{cos}}$ denote inner-product and cosine alignment, respectively.}",
        r"\label{tab:synthetic_real_alignment_per_class}",
        r"\scriptsize", r"\setlength{\tabcolsep}{2pt}", r"\renewcommand{\arraystretch}{0.82}",
        r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{l|l|" + "c" * len(classes) + "}",
        r"\toprule",
        rf"\multirow{{2}}{{*}}{{Unlearning Method}} & \multirow{{2}}{{*}}{{Metric}} & \multicolumn{{{len(classes)}}}{{c}}{{Forget Class}} \\",
        rf" & & {class_header} \\", r"\midrule",
    ]
    for method_index, method in enumerate(methods):
        for metric_index, (metric, metric_label, signed) in enumerate(METRICS):
            cells = []
            for forget_class in classes:
                key = (method, forget_class)
                if key not in grouped.index:
                    cells.append("--")
                    continue
                row = grouped.loc[key]
                cells.append(format_cell(row[(metric, "mean")], row[(metric, "std")], precision, signed))
            method_cell = rf"\multirow{{{len(METRICS)}}}{{*}}{{{METHOD_LABELS.get(method, method)}}}" if metric_index == 0 else ""
            lines.append(" & ".join([method_cell, metric_label, *cells]) + r" \\")
        if method_index < len(methods) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    data = pd.read_csv(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_table(data, args.classes, args.precision))
    print(f"[saved] {args.output.resolve()}")


if __name__ == "__main__":
    main()
