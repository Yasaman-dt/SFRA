"""Generate a table comparing unlearned and randomized forget-class rows."""

import argparse
from pathlib import Path

import pandas as pd


METHOD_ORDER = [
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

METHOD_NAMES = {
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

METRICS = {
    "retain": "test_retain",
    "forget": "test_fgt",
    "rs": "RS",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("results_forget_row_ablation_cifar10_fg7"),
    )
    parser.add_argument(
        "--output_tex",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def load_results(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("seed_*/*/*.csv")):
        randomized = "_random_forget_row_" in path.name
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        row = frame.iloc[-1]
        seed_name = path.parents[1].name
        rows.append({
            "method": path.parent.name,
            "seed": int(seed_name.removeprefix("seed_")),
            "row_state": "randomized" if randomized else "unlearned",
            "retain": pd.to_numeric(row.get(METRICS["retain"]), errors="coerce"),
            "forget": pd.to_numeric(row.get(METRICS["forget"]), errors="coerce"),
            "rs": pd.to_numeric(row.get(METRICS["rs"]), errors="coerce"),
        })
    if not rows:
        raise FileNotFoundError(f"No result CSVs found below {root}")
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.groupby(["method", "row_state"], as_index=False).agg(
        retain_mean=("retain", "mean"),
        retain_std=("retain", "std"),
        forget_mean=("forget", "mean"),
        forget_std=("forget", "std"),
        rs_mean=("rs", "mean"),
        rs_std=("rs", "std"),
        seeds=("seed", "nunique"),
    )
    return summary


def fmt(mean, std, precision):
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"


def lookup(summary, method, state):
    match = summary[
        summary["method"].eq(method) & summary["row_state"].eq(state)
    ]
    return None if match.empty else match.iloc[0]


def make_rows(summary):
    methods_present = set(summary["method"])
    methods = [method for method in METHOD_ORDER if method in methods_present]
    methods.extend(sorted(methods_present - set(methods)))
    rows = []
    for method in methods:
        unlearned = lookup(summary, method, "unlearned")
        randomized = lookup(summary, method, "randomized")

        def value(record, metric, precision):
            if record is None:
                return "--"
            return fmt(record[f"{metric}_mean"], record[f"{metric}_std"], precision)

        rows.append({
            "Unlearning Method": METHOD_NAMES.get(method, method),
            "Checkpoint retain": value(unlearned, "retain", 2),
            "Random-init retain": value(randomized, "retain", 2),
            "Checkpoint forget": value(unlearned, "forget", 2),
            "Random-init forget": value(randomized, "forget", 2),
            "Checkpoint RS": value(unlearned, "rs", 2),
            "Random-init RS": value(randomized, "rs", 2),
        })
    return pd.DataFrame(rows)


def make_latex(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Robustness of SFRA to removal of the forget class output row on CIFAR-10 with ResNet-18 and forget class~7. Here, $w_f$ denotes the forget class output-row parameters: Unlearned $w_f$ uses the row from the released unlearned checkpoint, whereas Random $w_f$ restores a missing row using random initialization before probe generation and relearning. All other audit settings are fixed.}",
        r"\label{tab:forget_row_ablation_cifar10_resnet18_fg7}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l|cc|cc|cc}",
        r"\toprule",
        r"\multirow{2}{*}{Unlearning Method} & \multicolumn{2}{c|}{$\mathcal{A}_r^t(\%)$} & \multicolumn{2}{c|}{$\mathcal{A}_f^t(\%)$} & \multicolumn{2}{c}{$\mathrm{RS}$} \\",
        r" & Unlearned $w_f$ & Random $w_f$ & Unlearned $w_f$ & Random $w_f$ & Unlearned $w_f$ & Random $w_f$ \\",
        r"\midrule",
    ]
    for row in table.to_dict("records"):
        values = [str(value).replace("_", r"\_") for value in row.values()]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    output_tex = args.output_tex or args.input_dir / "forget_row_ablation_table.tex"
    output_csv = args.output_csv or args.input_dir / "forget_row_ablation_table.csv"
    summary = summarize(load_results(args.input_dir))
    table = make_rows(summary)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(make_latex(table))
    table.to_csv(output_csv, index=False)
    print(f"[saved] {output_tex.resolve()}")
    print(f"[saved] {output_csv.resolve()}")


if __name__ == "__main__":
    main()
