"""Create a Softmax-vs-entropy-vs-energy uncertainty ablation table.

This script reads summaries produced by:

    python -m ablation.synthesis_strategy_ablation \
      --distributions gaussian \
      --forget_selections low_confidence \
      --retain_selections high_confidence \
      --uncertainty_scores softmax energy

and reports RS for all three ranking scores. Deltas are paired by seed and
defined relative to Softmax.

    Delta RS = RS_energy - RS_softmax

Example:
    python plots_tables/softmax_energy_ablation_table.py \
      --root results_synthesis_ablation \
      --methods bad_teacher neggrad_plus delete \
      --dataset cifar10 \
      --model_name resnet18 \
      --classes 7
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


METHOD_LABELS = {
    "retrained": r"Retrained",
    "bad_teacher": r"Bad Teacher \cite{chundawat2023can}",
    "neggrad_plus": r"Negative Gradient+ \cite{kurmanji2023towards}",
    "delete": r"DELETE \cite{zhou2025decoupled}",
    "random_label": r"Random Label \cite{hayase2020selective}",
    "finetune": r"Finetune \cite{golatkar2020eternal}",
    "gradient_ascent": r"Negative Gradient \cite{golatkar2020eternal}",
    "boundary_shrink": r"Boundary Shrink \cite{chen2023boundary}",
    "l2ul_adv": r"Learn to Unlearn \cite{cha2024learning}",
    "scrub": r"SCRUB \cite{kurmanji2023towards}",
    "salun": r"SalUn \cite{fan2023salun}",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build softmax-vs-energy uncertainty ablation table."
    )
    parser.add_argument("--root", default="results_synthesis_ablation")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bad_teacher", "neggrad_plus", "delete"],
    )
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--model", "--model_name", dest="model_name", default="resnet18")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[7],
        help="Forget classes to aggregate/report.",
    )
    parser.add_argument(
        "--unlearn_lr",
        type=float,
        default=None,
        help="Optional learning-rate filter when multiple runs exist for a class.",
    )
    parser.add_argument("--precision", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--caption", default=None)
    parser.add_argument("--label", default=None)
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]+", "_", str(value)).strip("_")


SCORES = ("softmax", "entropy", "energy")


def fmt(value: float, precision: int, signed: bool = False) -> str:
    if pd.isna(value):
        return "-"
    spec = f"+.{precision}f" if signed else f".{precision}f"
    return format(float(value), spec)


def fmt_mean_std(mean: float, std: float, precision: int, bold: bool = False) -> str:
    if pd.isna(mean):
        return "-"
    text = (
        rf"${fmt(mean, precision)}$"
        if pd.isna(std)
        else (
            rf"${fmt(mean, precision)}"
            rf"{{\scriptstyle\,\pm\,{fmt(std, precision)}}}$"
        )
    )
    return rf"\textbf{{{text}}}" if bold else text


def fmt_delta(mean: float, std: float, precision: int) -> str:
    if pd.isna(mean):
        return "-"
    if pd.isna(std):
        return rf"${fmt(mean, precision, signed=True)}$"
    return rf"${fmt(mean, precision, signed=True)} \pm {fmt(std, precision)}$"


def parse_folder(folder_name: str, dataset: str, model_name: str) -> Optional[tuple[float, int]]:
    pattern = re.compile(
        rf"^{re.escape(dataset)}_{re.escape(model_name)}_lr"
        rf"(?P<lr>[-+0-9.eE]+)_fg(?P<fg>\d+)$"
    )
    match = pattern.match(folder_name)
    if not match:
        return None
    return float(match.group("lr")), int(match.group("fg"))


def read_method_runs(args: argparse.Namespace, method: str) -> pd.DataFrame:
    method_root = Path(args.root) / method
    if not method_root.is_dir():
        raise FileNotFoundError(f"Method directory not found: {method_root}")

    frames = []
    wanted_classes = set(args.classes)
    for runs_path in sorted(method_root.glob("*/runs.csv")):
        parsed = parse_folder(runs_path.parent.name, args.dataset, args.model_name)
        if parsed is None:
            continue
        lr, forget_class = parsed
        if forget_class not in wanted_classes:
            continue
        if args.unlearn_lr is not None and not np.isclose(lr, args.unlearn_lr):
            continue

        runs = pd.read_csv(runs_path)
        if "uncertainty_score" not in runs.columns:
            runs["uncertainty_score"] = "softmax"
        runs["uncertainty_score"] = runs["uncertainty_score"].fillna("softmax")

        needed = {
            "distribution",
            "forget_selection",
            "retain_selection",
            "uncertainty_score",
            "seed",
            "RS",
        }
        missing = needed - set(runs.columns)
        if missing:
            raise KeyError(f"Missing columns {sorted(missing)} in {runs_path}")

        subset = runs[
            runs["distribution"].astype(str).eq("gaussian")
            & runs["forget_selection"].astype(str).eq("low_confidence")
            & runs["retain_selection"].astype(str).eq("high_confidence")
            & runs["uncertainty_score"].astype(str).isin(SCORES)
        ]
        if subset.empty:
            continue
        subset = subset.copy()
        subset["forget_class"] = forget_class
        subset["RS"] = pd.to_numeric(subset["RS"], errors="coerce")
        frames.append(subset[["forget_class", "seed", "uncertainty_score", "RS"]])
    if not frames:
        return pd.DataFrame(columns=["forget_class", "seed", "uncertainty_score", "RS"])
    return pd.concat(frames, ignore_index=True)


def build_table(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for method in args.methods:
        runs = read_method_runs(args, method)
        # First average across requested classes within each seed, preserving
        # seeds as the independent replicates used for mean/std and paired deltas.
        per_seed = runs.groupby(["seed", "uncertainty_score"], as_index=False)["RS"].mean()
        pivot = per_seed.pivot(index="seed", columns="uncertainty_score", values="RS")
        row = {"method": method, "num_seeds": int(len(pivot))}
        for score in SCORES:
            values = pivot[score].dropna() if score in pivot else pd.Series(dtype=float)
            row[f"{score}_n"] = int(len(values))
            row[f"{score}_rs_mean"] = values.mean() if len(values) else np.nan
            row[f"{score}_rs_std"] = values.std(ddof=1) if len(values) > 1 else np.nan
        for score in ("entropy", "energy"):
            if "softmax" in pivot and score in pivot:
                paired = (pivot[score] - pivot["softmax"]).dropna()
            else:
                paired = pd.Series(dtype=float)
            row[f"{score}_delta_mean"] = paired.mean() if len(paired) else np.nan
            row[f"{score}_delta_std"] = paired.std(ddof=1) if len(paired) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_latex(
    table: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    caption = args.caption
    if caption is None:
        caption = (
            "Comparison of Softmax confidence, predictive entropy, and energy "
            "for ranking synthetic probes on CIFAR-10/ResNet-18. "
            "All settings other than the uncertainty score are fixed. Results "
            r"are RS mean $\pm$ standard deviation when multiple seeds are "
            "available."
        )
    label = args.label or "tab:softmax_energy_uncertainty_ablation"
    lines = [
        r"\begin{table}[h]",
        r"\color{red}",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l|ccc}",
        r"\toprule",
        r"Unlearning Method & Softmax RS & Entropy RS & Energy RS \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        method = METHOD_LABELS.get(row["method"], str(row["method"]).replace("_", r"\_"))
        means = {score: row[f"{score}_rs_mean"] for score in SCORES}
        available = [value for value in means.values() if pd.notna(value)]
        best = max(available) if available else np.nan
        cells = [
            method,
            fmt_mean_std(row["softmax_rs_mean"], row["softmax_rs_std"], args.precision,
                         pd.notna(best) and np.isclose(row["softmax_rs_mean"], best)),
            fmt_mean_std(row["entropy_rs_mean"], row["entropy_rs_std"], args.precision,
                         pd.notna(best) and np.isclose(row["entropy_rs_mean"], best)),
            fmt_mean_std(row["energy_rs_mean"], row["energy_rs_std"], args.precision,
                         pd.notna(best) and np.isclose(row["energy_rs_mean"], best)),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    table = build_table(args)

    out_base = args.out
    if out_base is None:
        class_tag = "_".join(str(c) for c in args.classes)
        out_base = (
            Path(args.root)
            / f"{args.dataset}_{args.model_name}_softmax_vs_energy_fg{class_tag}"
        )
    if out_base.suffix in {".csv", ".tex"}:
        csv_out = out_base.with_suffix(".csv")
        tex_out = out_base.with_suffix(".tex")
    else:
        csv_out = Path(str(out_base) + ".csv")
        tex_out = Path(str(out_base) + ".tex")

    rs_columns = ["method"]
    for score in SCORES:
        rs_columns.extend([f"{score}_rs_mean", f"{score}_rs_std"])
    table[rs_columns].to_csv(csv_out, index=False)
    tex_out.write_text(make_latex(table, args), encoding="utf-8")
    print(f"[OK] wrote {csv_out}")
    print(f"[OK] wrote {tex_out}")


if __name__ == "__main__":
    main()
