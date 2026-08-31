"""Create a fixed dataset/backbone table with forget classes as columns.

Rows are grouped by unlearning method and metric:

  Method | Metric | Variant | forget class 0 | forget class 1 | ...

For each method, the displayed variants are:
  * Unlearned
  * Retrained
  * PRA
  * SFRA (ours)

For retain and forget accuracy, all four variants are shown. For RS and
Delta RS, only PRA and our source-free relearning audit are shown because the
unlearned checkpoint and the retrained-from-scratch checkpoint are controls.
Delta RS is matched to the retrained-from-scratch control for the same forget
class and the same audit variant. For unlearning methods, we also report a
linear-probe forget-accuracy gap relative to the matched retrained control.

Example:
    python plots_tables/generate_single_class_variant_table.py \
        --dataset cifar10 \
        --model_name resnet18
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "tiny_imagenet": "TinyImageNet",
}

MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "vit-b-16": "ViT-B/16",
    "swin-t": "Swin-T",
    "vgg16": "VGG-16",
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

DEFAULT_CLASSES = {
    "cifar10": list(range(10)),
    "cifar100": [0, 20, 40, 60, 80],
    "tiny_imagenet": [0, 40, 80, 120, 160],
}

RESNET18_EXTENDED_CLASSES = {
    "cifar100": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
    "tiny_imagenet": [0, 20, 40, 60, 80, 100, 120, 140, 160, 180],
}

# Architecture-specific overrides. Other TinyImageNet architectures continue
# to use the five-class DEFAULT_CLASSES subset unless listed here.
MODEL_DATASET_CLASS_OVERRIDES = {
    ("resnet18", "cifar100"): RESNET18_EXTENDED_CLASSES["cifar100"],
    ("resnet18", "tiny_imagenet"): RESNET18_EXTENDED_CLASSES["tiny_imagenet"],
    ("vit-b-16", "cifar100"): [
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90
    ],
    ("vit-b-16", "tiny_imagenet"): [
        0, 20, 40, 60, 80, 100, 120, 140, 160, 180
    ],
    ("swin-t", "cifar100"): [
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90
    ],
    ("swin-t", "tiny_imagenet"): [
        0, 20, 40, 60, 80, 100, 120, 140, 160, 180
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed dataset/backbone table with forget classes as columns."
    )
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--model_name", "--model", dest="model_name", default="resnet18")
    parser.add_argument("--classes", type=int, nargs="+", default=None)
    parser.add_argument("--sourcefree_root", type=Path, default=Path("results_single_class"))
    parser.add_argument("--pra_root", type=Path, default=Path("pra_single"))
    parser.add_argument("--linear_probe_root", type=Path, default=Path("linear_probe_single"))
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results_single_class/tables"),
    )
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Number of decimal places for retain, forget, and linear-probe accuracies.",
    )
    parser.add_argument(
        "--rs_precision",
        type=int,
        default=2,
        help="Number of decimal places for RS and Delta RS values.",
    )
    parser.add_argument(
        "--num_parts",
        type=int,
        default=2,
        help="Also write this many method-split LaTeX tables. Use 1 to disable split tables.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]+", "_", str(value)).strip("_")


def class_key(value) -> str:
    if pd.isna(value):
        return ""
    try:
        value_float = float(value)
        if value_float.is_integer():
            return str(int(value_float))
    except (TypeError, ValueError):
        pass
    return str(value)


def class_list(args: argparse.Namespace) -> list[int]:
    if args.classes is not None:
        return args.classes
    override = MODEL_DATASET_CLASS_OVERRIDES.get((args.model_name, args.dataset))
    if override is not None:
        return override
    return DEFAULT_CLASSES.get(args.dataset, [])


def sourcefree_candidates(root: Path, dataset: str, model_name: str) -> list[Path]:
    csv_root = root / "csvs"
    return [
        csv_root / f"z_standardized_selected_all_methods_{dataset}_{model_name}.csv",
        csv_root / f"z_standardized_revival_all_methods_{dataset}_{model_name}.csv",
        csv_root / "z_standardized_selected_all_methods.csv",
        csv_root / "z_standardized_revival_all_methods.csv",
    ]


def load_sourcefree(root: Path, dataset: str, model_name: str, classes: list[int]) -> pd.DataFrame:
    path = next((p for p in sourcefree_candidates(root, dataset, model_name) if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"No source-free summary found for {dataset}/{model_name}.")
    df = pd.read_csv(path)
    df = df[
        df["dataset"].astype(str).eq(dataset)
        & df["model"].astype(str).eq(model_name)
    ].copy()
    df["forget_class_int"] = pd.to_numeric(df["forget_class"], errors="coerce").astype("Int64")
    df = df[df["forget_class_int"].isin(classes)].copy()
    for col in ["test_retain_acc", "test_forget_acc", "RS2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["forget_key"] = df["forget_class_int"].astype(str)
    return df


def harmonic_rs(retain_un, forget_un, retain_after, forget_after) -> float:
    vals = [retain_un, forget_un, retain_after, forget_after]
    if any(pd.isna(v) for v in vals):
        return np.nan
    scale = 100.0 if max(abs(float(v)) for v in vals) > 1.5 else 1.0
    ar_un = float(retain_un) / scale
    af_un = float(forget_un) / scale
    ar_after = float(retain_after) / scale
    af_after = float(forget_after) / scale
    retain_score = 1.0 - max(0.0, ar_un - ar_after)
    forget_score = max(0.0, af_after - af_un)
    retain_score = float(np.clip(retain_score, 0.0, 1.0))
    forget_score = float(np.clip(forget_score, 0.0, 1.0))
    denominator = retain_score + forget_score
    if denominator <= 0:
        return 0.0
    return float(2.0 * retain_score * forget_score / denominator)


def load_pra_method(root: Path, dataset: str, model_name: str, method: str, classes: list[int]) -> pd.DataFrame:
    path = root / method / f"{dataset}_{model_name}.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {
        "forget_class",
        "baseline_acc_r_test",
        "baseline_acc_f_test",
        "pra_acc_r_test",
        "pra_acc_f_test",
    }
    if not required.issubset(df.columns):
        return pd.DataFrame()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "support_seed" not in df.columns:
        df["support_seed"] = 0
    df = df.drop_duplicates(subset=["forget_class", "support_seed"], keep="last")
    df["forget_class_int"] = pd.to_numeric(df["forget_class"], errors="coerce").astype("Int64")
    df = df[df["forget_class_int"].isin(classes)].copy()
    if df.empty:
        return df
    df["RS"] = [
        harmonic_rs(row.baseline_acc_r_test, row.baseline_acc_f_test, row.pra_acc_r_test, row.pra_acc_f_test)
        for row in df.itertuples(index=False)
    ]
    out = (
        df.groupby("forget_class_int", as_index=False)
        .agg(
            test_retain_acc=("pra_acc_r_test", "mean"),
            test_forget_acc=("pra_acc_f_test", "mean"),
            RS=("RS", "mean"),
        )
    )
    out["forget_key"] = out["forget_class_int"].astype(str)
    out["method"] = method
    return out


def load_linear_probe_method(root: Path, dataset: str, model_name: str, method: str, classes: list[int]) -> pd.DataFrame:
    path = root / method / f"{dataset}_{model_name}.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"forget_classes", "linear_probe_acc_f_test"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["forget_class_int"] = pd.to_numeric(df["forget_classes"], errors="coerce").astype("Int64")
    df = df[df["forget_class_int"].isin(classes)].copy()
    if df.empty:
        return df
    out = (
        df.groupby("forget_class_int", as_index=False)
        .agg(linear_probe_acc_f_test=("linear_probe_acc_f_test", "mean"))
    )
    out["forget_key"] = out["forget_class_int"].astype(str)
    out["method"] = method
    return out


def method_list(df: pd.DataFrame, requested: list[str] | None) -> list[str]:
    present = set(df["method"].astype(str).unique())
    if requested is not None:
        return [m for m in requested if m in present]
    ordered = [m for m in METHOD_ORDER if m in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def build_metric_maps(args: argparse.Namespace) -> tuple[list[int], list[str], dict]:
    classes = class_list(args)
    df = load_sourcefree(args.sourcefree_root, args.dataset, args.model_name, classes)
    methods = method_list(df, args.methods)

    sourcefree_retrained = df[
        df["method"].astype(str).eq("retrained")
        & df["phase"].astype(str).eq("revival")
    ]
    sourcefree_retrained_rs = {
        row.forget_key: float(row.RS2)
        for row in sourcefree_retrained[["forget_key", "RS2"]].dropna().itertuples(index=False)
    }

    pra_by_method = {
        method: load_pra_method(args.pra_root, args.dataset, args.model_name, method, classes)
        for method in methods
    }
    linear_probe_by_method = {
        method: load_linear_probe_method(args.linear_probe_root, args.dataset, args.model_name, method, classes)
        for method in methods
    }
    pra_retrained = load_pra_method(args.pra_root, args.dataset, args.model_name, "retrained", classes)
    pra_retrained_rs = {
        row.forget_key: float(row.RS)
        for row in pra_retrained[["forget_key", "RS"]].dropna().itertuples(index=False)
    } if not pra_retrained.empty else {}

    data = {}
    for method in methods:
        # Initial/source checkpoint accuracy. For the original model the
        # summary uses phase="original"; for all unlearning/retrained controls
        # the corresponding starting point is phase="unlearned".
        initial_variant = "Original" if method == "original" else "Unlearned"
        initial_phase = "original" if method == "original" else "unlearned"
        initial = df[
            df["method"].astype(str).eq(method)
            & df["phase"].astype(str).eq(initial_phase)
        ]
        data[(method, initial_variant)] = {
            "retain": dict(zip(initial["forget_key"], initial["test_retain_acc"])),
            "forget": dict(zip(initial["forget_key"], initial["test_forget_acc"])),
            "rs": {},
            "delta": {},
        }

        # PRA.
        pra = pra_by_method.get(method, pd.DataFrame())
        if not pra.empty:
            rs = dict(zip(pra["forget_key"], pra["RS"]))
            delta = {
                key: value - pra_retrained_rs[key]
                for key, value in rs.items()
                if key in pra_retrained_rs and method != "retrained"
            }
            data[(method, r"PRA \cite{ha2025unlearning}")] = {
                "retain": dict(zip(pra["forget_key"], pra["test_retain_acc"])),
                "forget": dict(zip(pra["forget_key"], pra["test_forget_acc"])),
                "rs": rs,
                "delta": delta,
            }
        else:
            data[(method, r"PRA \cite{ha2025unlearning}")] = {
                "retain": {},
                "forget": {},
                "rs": {},
                "delta": {},
            }

        # Ours.
        revival = df[
            df["method"].astype(str).eq(method)
            & df["phase"].astype(str).eq("revival")
        ]
        rs = dict(zip(revival["forget_key"], revival["RS2"]))
        delta = {
            key: value - sourcefree_retrained_rs[key]
            for key, value in rs.items()
            if key in sourcefree_retrained_rs and method != "retrained"
        }
        data[(method, "SFRA (ours)")] = {
            "retain": dict(zip(revival["forget_key"], revival["test_retain_acc"])),
            "forget": dict(zip(revival["forget_key"], revival["test_forget_acc"])),
            "rs": rs,
            "delta": delta,
        }

        # Frozen-encoder linear-probe accuracy on the forget class. Original has
        # no forget-set-specific linear-probe audit; all other methods, including
        # Retrained, report the raw accuracy rather than a matched-control gap.
        linear_probe = linear_probe_by_method.get(method, pd.DataFrame())
        if method != "original" and not linear_probe.empty:
            lp_forget = dict(zip(linear_probe["forget_key"], linear_probe["linear_probe_acc_f_test"]))
        else:
            lp_forget = {}
        data[(method, "Linear Probe")] = {
            "retain": {},
            "forget": {},
            "rs": {},
            "delta": {},
            "lp_forget": lp_forget,
        }

    return classes, methods, data


def math_number(text: str, compact: bool, scale: float = 0.78) -> str:
    if compact:
        return rf"\scalebox{{{scale:.2f}}}{{${text}$}}"
    return rf"${text}$"


def format_acc(value, precision: int, compact: bool = False, scale: float = 0.78) -> str:
    if pd.isna(value):
        return "-"
    return math_number(f"{float(value):.{precision}f}", compact, scale)


def format_rs(value, precision: int, compact: bool = False, scale: float = 0.78) -> str:
    if pd.isna(value):
        return "-"
    return math_number(f"{float(value):.{precision}f}", compact, scale)


def format_delta(value, precision: int, compact: bool = False, scale: float = 0.78) -> str:
    if pd.isna(value):
        return "-"
    return math_number(f"{float(value):+.{precision}f}", compact, scale)


def metric_value(
    metric_map: dict,
    metric: str,
    forget_class: int,
    precision: int,
    rs_precision: int,
    compact: bool = False,
    scale: float = 0.78,
) -> str:
    key = str(forget_class)
    value = metric_map[metric].get(key, np.nan)
    if metric in {"retain", "forget"}:
        return format_acc(value, precision, compact, scale)
    if metric == "rs":
        return format_rs(value, rs_precision, compact, scale)
    if metric == "lp_forget":
        return format_acc(value, precision, compact, scale)
    return format_delta(value, rs_precision, compact, scale)


def metric_groups_for_method(method: str) -> list[tuple[str, str, list[str]]]:
    if method == "original":
        accuracy_variants = ["Original"]
        rs_variants: list[str] = []
        delta_variants: list[str] = []
    elif method == "retrained":
        accuracy_variants = ["Unlearned", r"PRA \cite{ha2025unlearning}", "SFRA (ours)"]
        rs_variants = [r"PRA \cite{ha2025unlearning}", "SFRA (ours)"]
        delta_variants = []
    else:
        accuracy_variants = ["Unlearned", r"PRA \cite{ha2025unlearning}", "SFRA (ours)"]
        rs_variants = [r"PRA \cite{ha2025unlearning}", "SFRA (ours)"]
        delta_variants = [r"PRA \cite{ha2025unlearning}", "SFRA (ours)"]

    groups = [
        ("retain", r"$\mathcal{A}^{t}_{r}(\%)$", accuracy_variants),
        ("forget", r"$\mathcal{A}^{t}_{f}(\%)$", accuracy_variants),
    ]
    if method != "original":
        groups.append(("lp_forget", r"$\mathcal{A}^{LP}_{f}(\%)$", ["Linear Probe"]))
    if rs_variants:
        groups.append(("rs", r"$\mathrm{RS}$", rs_variants))
    if delta_variants:
        groups.append(("delta", r"$\Delta\mathrm{RS}$", delta_variants))
    return groups


def split_method_list(methods: list[str], num_parts: int) -> list[list[str]]:
    if num_parts <= 1 or len(methods) <= 1:
        return [methods]
    num_parts = min(num_parts, len(methods))
    return [list(part) for part in np.array_split(np.array(methods, dtype=object), num_parts) if len(part) > 0]


def write_latex_one(
    classes: list[int],
    methods: list[str],
    data: dict,
    args: argparse.Namespace,
    out_path: Path,
    label_suffix: str = "",
    caption_suffix: str = "",
    append: bool = False,
    embedded: bool = False,
) -> Path:
    dataset_label = DATASET_LABELS.get(args.dataset, args.dataset)
    model_label = MODEL_LABELS.get(args.model_name, args.model_name)
    col_format = "l|l|l|" + ("c" * len(classes))
    class_header = " & ".join(str(c) for c in classes)
    caption_extra = f" {caption_suffix}" if caption_suffix else ""
    if embedded:
        lines = [r"\begin{minipage}[t]{0.49\textwidth}", r"\centering"]
    else:
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            (
                r"\caption{Single-class unlearning and relearning results for each "
                rf"forget class on {dataset_label} using a {model_label} backbone. "
                r"For every unlearning method, we report the unlearned model, the "
                r"source-dependent PRA baseline, our proposed SFRA, and a "
                r"linear-probe diagnostic of representation "
                r"separability. Each forget class column corresponds to a "
                r"separate unlearned checkpoint in which that class is "
                r"designated for forgetting."
                rf"{caption_extra}}}"
            ),
            rf"\label{{tab:{slugify(args.dataset)}_{slugify(args.model_name)}_variant_by_forget{label_suffix}}}",
        ]
    font_size = r"\scriptsize"
    tabcolsep = "2pt"
    arraystretch = "0.80"
    compact_numbers = False
    number_scale = 1.0
    resize_width = r"\columnwidth"
    lines.extend([
        font_size,
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
        rf"\renewcommand{{\arraystretch}}{{{arraystretch}}}",
    ])
    lines.append(rf"\resizebox{{{resize_width}}}{{!}}{{%")
    lines.extend([
        rf"\begin{{tabular}}{{{col_format}}}",
        r"\toprule",
        rf"\multirow{{2}}{{*}}{{Unlearning Method}} & \multirow{{2}}{{*}}{{Metric}} & \multirow{{2}}{{*}}{{Variant}} & \multicolumn{{{len(classes)}}}{{c}}{{Forget Class}} \\",
        rf" & & & {class_header} \\",
        r"\midrule",
    ])

    for method_index, method in enumerate(methods):
        metric_groups = metric_groups_for_method(method)
        method_label = METHOD_LABELS.get(method, method.replace("_", r"\_"))
        total_rows = sum(len(variants) for _, _, variants in metric_groups)
        first_method_row = True
        for metric_group_index, (metric_key, metric_label, variants) in enumerate(metric_groups):
            for variant_index, variant in enumerate(variants):
                metric_map = data.get((method, variant), {"retain": {}, "forget": {}, "rs": {}, "delta": {}})
                values = [
                    metric_value(
                        metric_map,
                        metric_key,
                        c,
                        args.precision,
                        args.rs_precision,
                        compact=compact_numbers,
                        scale=number_scale,
                    )
                    for c in classes
                ]
                method_cell = rf"\multirow{{{total_rows}}}{{*}}{{{method_label}}}" if first_method_row else ""
                metric_cell = rf"\multirow{{{len(variants)}}}{{*}}{{{metric_label}}}" if variant_index == 0 else ""
                lines.append(
                    " & ".join([method_cell, metric_cell, variant, *values]) + r" \\"
                )
                first_method_row = False
            if metric_group_index != len(metric_groups) - 1:
                lines.append(rf"\cmidrule(lr){{2-{len(classes) + 3}}}")
        if method_index != len(methods) - 1:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    lines.append(r"}")
    lines.append(r"\end{minipage}%" if embedded else r"\end{table}")
    mode = "a" if append else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        if append and not embedded:
            handle.write("\n")
        handle.write("\n".join(lines) + "\n")
    return out_path


def write_latex(classes: list[int], methods: list[str], data: dict, args: argparse.Namespace) -> list[Path]:
    base = f"{slugify(args.dataset)}_{slugify(args.model_name)}_single_class_variant_by_forget_table"
    parts = split_method_list(methods, args.num_parts)
    out_path = args.out_dir / f"{base}.tex"
    side_by_side = len(parts) == 2
    if side_by_side:
        dataset_label = DATASET_LABELS.get(args.dataset, args.dataset)
        model_label = MODEL_LABELS.get(args.model_name, args.model_name)
        out_path.write_text(
            "\n".join([
                r"\begin{table*}[t]",
                r"\centering",
                (
                    r"\caption{Per-forget-class single-class unlearning and "
                    rf"relearning results on {dataset_label} using {model_label}. We report "
                    r"the unlearned checkpoint, the source-dependent PRA "
                    r"baseline, our proposed SFRA, and frozen-encoder linear "
                    r"probing. Each forget class column corresponds to a "
                    r"separate unlearned checkpoint in which that class is "
                    r"designated for forgetting.}"
                ),
                rf"\label{{tab:{slugify(args.dataset)}_{slugify(args.model_name)}_variant_by_forget}}",
            ]) + "\n",
            encoding="utf-8",
        )
    for part_index, part_methods in enumerate(parts, start=1):
        if side_by_side and part_index == 2:
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write("\\hfill\n")
        write_latex_one(
            classes,
            part_methods,
            data,
            args,
            out_path,
            label_suffix=f"_part{part_index}",
            caption_suffix=f"Part {part_index} of {len(parts)}.",
            append=side_by_side or part_index > 1,
            embedded=side_by_side,
        )
    if side_by_side:
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write("\n\\end{table*}\n")
    return [out_path]


def write_csv(classes: list[int], methods: list[str], data: dict, args: argparse.Namespace) -> Path:
    rows = []
    for method in methods:
        for metric, _, variants in metric_groups_for_method(method):
            for variant in variants:
                metric_map = data.get((method, variant), {"retain": {}, "forget": {}, "rs": {}, "delta": {}})
                row = {"method": method, "variant": variant, "metric": metric}
                for c in classes:
                    row[str(c)] = metric_map[metric].get(str(c), np.nan)
                rows.append(row)
    out_path = args.out_dir / f"{slugify(args.dataset)}_{slugify(args.model_name)}_single_class_variant_by_forget_table.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    classes, methods, data = build_metric_maps(args)
    tex_paths = write_latex(classes, methods, data, args)
    for tex_path in tex_paths:
        print(f"[saved] {tex_path.resolve()}")


if __name__ == "__main__":
    main()
