"""Build the multi-class ResNet-18 variant table with a linear-probe delta audit."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tables/linear_probe_multi_forget_separability_resnet18.tex"
LP_CSV = ROOT / "linear_probe_multi/linear_probe_multi_forget_separability_resnet18.csv"
SETTINGS = [
    ("cifar10_2", "CIFAR-10", "2-Class", "cifar10", "1,6", "forget2"),
    ("cifar100_2", "CIFAR-100", "2-Class", "cifar100", "25,58", "forget2"),
    ("cifar100_5", "CIFAR-100", "5-Class", "cifar100", "23,25,38,58,96", "forget5"),
    ("cifar100_10", "CIFAR-100", "10-Class", "cifar100", "23,25,38,49,51,54,58,66,96,98", "forget10"),
]
METHODS = ["retrained", "finetune", "gradient_ascent", "neggrad_plus", "random_label",
           "l2ul_adv", "scrub", "bad_teacher", "salun", "delete"]
LABELS = {
    "retrained": "Retrained",
    "finetune": r"Finetune \cite{golatkar2020eternal}",
    "gradient_ascent": r"Negative Gradient \cite{golatkar2020eternal}",
    "neggrad_plus": r"Negative Gradient+ \cite{kurmanji2023towards}",
    "random_label": r"Random Label \cite{hayase2020selective}",
    "l2ul_adv": r"Learn to Unlearn \cite{cha2024learning}",
    "scrub": r"SCRUB \cite{kurmanji2023towards}",
    "bad_teacher": r"Bad Teacher \cite{chundawat2023can}",
    "salun": r"SalUn \cite{fan2023salun}",
    "delete": r"DELETE \cite{zhou2025decoupled}",
}


def harmonic_rs(retain_un, forget_un, retain_after, forget_after):
    retain_score = np.clip(1 - max(0, (retain_un - retain_after) / 100), 0, 1)
    forget_score = np.clip((forget_after - forget_un) / 100, 0, 1)
    return 0.0 if retain_score + forget_score == 0 else 2 * retain_score * forget_score / (retain_score + forget_score)


def fmt(value, signed=False, precision=2):
    if value is None or pd.isna(value):
        return "--"
    return f"${value:+.{precision}f}$" if signed else f"${value:.{precision}f}$"


def source_row(frame, method, dataset, forget_classes, setting, phase):
    rows = frame[(frame.method == method) & (frame.dataset == dataset) & (frame.phase == phase)]
    if "setting" in rows.columns and setting != "forget2":
        rows = rows[rows.setting == setting]
    else:
        rows = rows[rows.forget_class.astype(str) == forget_classes]
    return None if rows.empty else rows.iloc[-1]


def build_data():
    source2 = pd.read_csv(ROOT / "results_multi_class_2/z_standardized_selected_all_methods.csv")
    source510 = pd.read_csv(ROOT / "results_multi_class_5_10/z_merged_with_setting_all.csv")
    lp = pd.read_csv(LP_CSV).set_index("method")
    data = {}
    for key, _, _, dataset, forget_classes, setting in SETTINGS:
        source = source2 if setting == "forget2" else source510
        for method in METHODS:
            un = source_row(source, method, dataset, forget_classes, setting, "unlearned")
            revival = source_row(source, method, dataset, forget_classes, setting, "revival")
            pra_path = ROOT / "pra_multi" / method / f"{dataset}_resnet18.csv"
            pra = pd.read_csv(pra_path)
            pra = pra[pra.forget_classes.astype(str) == forget_classes]
            pra = None if pra.empty else pra.iloc[-1]
            data[(key, method)] = {
                "un_r": None if un is None else un.test_retain_acc,
                "un_f": None if un is None else un.test_forget_acc,
                "pra_r": None if pra is None else pra.pra_acc_r_test,
                "pra_f": None if pra is None else pra.pra_acc_f_test,
                "our_r": None if revival is None else revival.test_retain_acc,
                "our_f": None if revival is None else revival.test_forget_acc,
                "pra_rs": None if pra is None else harmonic_rs(
                    pra.baseline_acc_r_test, pra.baseline_acc_f_test,
                    pra.pra_acc_r_test, pra.pra_acc_f_test),
                "our_rs": None if revival is None else revival.RS2,
                "lp_delta": lp.loc[method, f"{key}_gap"],
                "lp_acc": lp.loc[method, f"{key}_lp_forget"],
            }
        control = data[(key, "retrained")]
        for method in METHODS:
            values = data[(key, method)]
            values["pra_delta"] = values["pra_rs"] - control["pra_rs"]
            values["our_delta"] = values["our_rs"] - control["our_rs"]
    return data


def value_row(data, methods, method, metric, variant, signed=False, precision=2):
    values = [fmt(data[(key, method)][metric], signed, precision) for key, *_ in SETTINGS]
    return " & ".join(["", "", variant, *values]) + r" \\" 


def table_part(data, methods, part, total):
    headers = " & ".join(rf"\shortstack{{\textbf{{{dataset}}}\\\textbf{{{count}}}}}" for _, dataset, count, *_ in SETTINGS)
    caption_suffix = rf" Part {part} of {total}." if total > 1 else ""
    label_suffix = rf"_part{part}" if total > 1 else ""
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{Multi-class unlearning and relearning results using ResNet-18. Retain and forget accuracy compare the unlearned checkpoint, PRA, and SFRA (ours); RS compares PRA and SFRA (ours), and $\Delta$RS is relative to the matched retrained control. $\mathcal{{A}}_f^{{\mathrm{{LP}}}}$ is the frozen-encoder linear-probe accuracy on the forgotten classes.{caption_suffix}}}",
             rf"\label{{tab:linear_probe_multi_forget_separability_resnet18{label_suffix}}}",
             r"\color{red}", r"\fontsize{5.5}{5.8}\selectfont", r"\setlength{\tabcolsep}{2pt}",
             r"\renewcommand{\arraystretch}{0.72}", r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{l|l|l|cccc}", r"\toprule",
             r"\textbf{Unlearning Method} & \textbf{Metric} & \textbf{Variant} & \multicolumn{4}{c}{\textbf{Dataset / Number of Forgotten Classes}} \\",
             " & & & " + headers + r" \\", r"\midrule"]
    groups = [
        (r"$\mathcal{A}^{t}_{r}(\%)$", [("un_r", "Unlearned", False, 2), ("pra_r", r"PRA \cite{ha2025unlearning}", False, 2), ("our_r", "SFRA (ours)", False, 2)]),
        (r"$\mathcal{A}^{t}_{f}(\%)$", [("un_f", "Unlearned", False, 2), ("pra_f", r"PRA \cite{ha2025unlearning}", False, 2), ("our_f", "SFRA (ours)", False, 2)]),
        (r"$\mathcal{A}^{\mathrm{LP}}_{f}(\%)$", [("lp_acc", "Linear Probe", False, 2)]),
        ("RS", [("pra_rs", r"PRA \cite{ha2025unlearning}", False, 3), ("our_rs", "SFRA (ours)", False, 3)]),
        (r"$\Delta$RS", [("pra_delta", r"PRA \cite{ha2025unlearning}", True, 3), ("our_delta", "SFRA (ours)", True, 3)]),
    ]
    for mi, method in enumerate(methods):
        method_groups = [group for group in groups if method != "retrained" or group[0] != r"$\Delta$RS"]
        method_rows = sum(len(variants) for _, variants in method_groups)
        first = True
        for gi, (metric_label, variants) in enumerate(method_groups):
            for vi, (metric, variant, signed, precision) in enumerate(variants):
                row = value_row(data, methods, method, metric, variant, signed, precision).split(" & ")
                row[0] = rf"\multirow{{{method_rows}}}{{*}}{{{LABELS[method]}}}" if first else ""
                row[1] = rf"\multirow{{{len(variants)}}}{{*}}{{{metric_label}}}" if vi == 0 else ""
                lines.append(" & ".join(row))
                first = False
            if gi < len(method_groups) - 1:
                lines.append(r"\cmidrule(lr){2-7}")
        if mi < len(methods) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    return "\n".join(lines)


def main():
    data = build_data()
    parts = [METHODS[:5], METHODS[5:]]
    OUT.write_text("\n\n".join(table_part(data, methods, i, len(parts)) for i, methods in enumerate(parts, 1)) + "\n")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
