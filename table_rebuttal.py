import pandas as pd
from pathlib import Path

# ========= CONFIG =========
BASE_DIR = Path("/projets/Zdehghani/Source_Free_Class_Revival/results/proto_attack")

# proto attack files from Ref 19
PROTO_FILES = {
    "retrained":       BASE_DIR / "cifar10_resnet18_retrained_proto_attack_test_metrics.csv",
    "finetune":        BASE_DIR / "cifar10_resnet18_finetune_proto_attack_test_metrics.csv",
    "gradient_ascent": BASE_DIR / "cifar10_resnet18_gradient_ascent_proto_attack_test_metrics.csv",
    "neggrad_plus": BASE_DIR / "cifar10_resnet18_neggrad_plus_proto_attack_test_metrics.csv",
    "random_label":    BASE_DIR / "cifar10_resnet18_random_label_proto_attack_test_metrics.csv",
    "boundary_shrink": BASE_DIR / "cifar10_resnet18_boundary_shrink_proto_attack_test_metrics.csv",
    "l2ul_adv":        BASE_DIR / "cifar10_resnet18_l2ul_adv_proto_attack_test_metrics.csv",
    "salun":           BASE_DIR / "cifar10_resnet18_salun_proto_attack_test_metrics.csv",
    "delete":          BASE_DIR / "cifar10_resnet18_delete_proto_attack_test_metrics.csv",
}

# your revival files
OURS_FILES = {
    "retrained":       BASE_DIR / "cifar10_resnet18_unlearned_retrained_revival_by_forget_class.csv",
    "finetune":        BASE_DIR / "cifar10_resnet18_unlearned_finetune_revival_by_forget_class_lr0.02.csv",
    "gradient_ascent": BASE_DIR / "cifar10_resnet18_unlearned_gradient_ascent_revival_by_forget_class_lr5e-05.csv",
    "neggrad_plus":    BASE_DIR / "cifar10_resnet18_unlearned_neggrad_plus_revival_by_forget_class_lr0.5.csv",
    "random_label":    BASE_DIR / "cifar10_resnet18_unlearned_random_label_revival_by_forget_class_lr1e-07.csv",
    "boundary_shrink": BASE_DIR / "cifar10_resnet18_unlearned_boundary_shrink_revival_by_forget_class_lr1e-08.csv",
    "l2ul_adv":        BASE_DIR / "cifar10_resnet18_unlearned_l2ul_adv_revival_by_forget_class_lr1e-05.csv",
    "salun":           BASE_DIR / "cifar10_resnet18_unlearned_salun_revival_by_forget_class_lr0.001.csv",
    "delete":          BASE_DIR / "cifar10_resnet18_unlearned_delete_revival_by_forget_class_lr0.001.csv",
}

ROW_LABELS = {
    "retrained": "Retrained",
    "finetune": "FT",
    "random_label": "RL",
    "gradient_ascent": "NG",
    "neggrad_plus": "NG+",
    "boundary_shrink": "BS",
    "l2ul_adv": "L2UL",
    "salun": "SalUn",
    "delete": "DELETE",
}

OUT_CSV = BASE_DIR / "summary_table.csv"
OUT_XLSX = BASE_DIR / "summary_table.xlsx"
OUT_TEX = BASE_DIR / "summary_table.tex"

OUT_CSV = BASE_DIR / "summary_full_rows.csv"


# ========= HELPERS =========
def fmt_triplet(series: pd.Series, digits: int = 2) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "-"
    return f"({s.min():.{digits}f}, {s.mean():.{digits}f}, {s.max():.{digits}f})"


def get_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(dtype=float)


# ========= BUILD FULL CSV ROWS =========
rows = []

for method_key, short_name in ROW_LABELS.items():
    proto_path = PROTO_FILES.get(method_key)
    ours_path = OURS_FILES.get(method_key)

    df_proto = pd.read_csv(proto_path) if proto_path is not None and proto_path.exists() else None
    df_ours = pd.read_csv(ours_path) if ours_path is not None and ours_path.exists() else None

    # Shared unlearning part comes from proto baseline file
    if df_proto is not None:
        unlearn_train_Af = fmt_triplet(get_col(df_proto, ["baseline_train_fgt"]))
        unlearn_train_Ar = fmt_triplet(get_col(df_proto, ["baseline_train_retain"]))
        unlearn_test_Af  = fmt_triplet(get_col(df_proto, ["baseline_test_fgt"]))
        unlearn_test_Ar  = fmt_triplet(get_col(df_proto, ["baseline_test_retain"]))
    else:
        unlearn_train_Af = "-"
        unlearn_train_Ar = "-"
        unlearn_test_Af = "-"
        unlearn_test_Ar = "-"

    # [19] relearning row
    if df_proto is not None:
        relearn_train_Af_19 = fmt_triplet(get_col(df_proto, ["pra_train_fgt"]))
        relearn_train_Ar_19 = fmt_triplet(get_col(df_proto, ["pra_train_retain"]))
        relearn_test_Af_19  = fmt_triplet(get_col(df_proto, ["pra_test_fgt"]))
        relearn_test_Ar_19  = fmt_triplet(get_col(df_proto, ["pra_test_retain"]))
    else:
        relearn_train_Af_19 = "-"
        relearn_train_Ar_19 = "-"
        relearn_test_Af_19 = "-"
        relearn_test_Ar_19 = "-"

    rows.append({
        "row_name": f"{short_name} [19]",
        "unlearning train Af": unlearn_train_Af,
        "unlearning train Ar": unlearn_train_Ar,
        "unlearning test Af": unlearn_test_Af,
        "unlearning test Ar": unlearn_test_Ar,
        "relearning train Af": relearn_train_Af_19,
        "relearning train Ar": relearn_train_Ar_19,
        "relearning test Af": relearn_test_Af_19,
        "relearning test Ar": relearn_test_Ar_19,
    })

    # Ours relearning row
    if df_ours is not None:
        relearn_train_Af_ours = fmt_triplet(get_col(df_ours, ["train_fgt", "pra_train_fgt"]))
        relearn_train_Ar_ours = fmt_triplet(get_col(df_ours, ["train_retain", "pra_train_retain"]))
        relearn_test_Af_ours  = fmt_triplet(get_col(df_ours, ["test_fgt", "pra_test_fgt"]))
        relearn_test_Ar_ours  = fmt_triplet(get_col(df_ours, ["test_retain", "pra_test_retain"]))
    else:
        relearn_train_Af_ours = "-"
        relearn_train_Ar_ours = "-"
        relearn_test_Af_ours = "-"
        relearn_test_Ar_ours = "-"

    rows.append({
        "row_name": f"{short_name} (ours)",
        "unlearning train Af": unlearn_train_Af,
        "unlearning train Ar": unlearn_train_Ar,
        "unlearning test Af": unlearn_test_Af,
        "unlearning test Ar": unlearn_test_Ar,
        "relearning train Af": relearn_train_Af_ours,
        "relearning train Ar": relearn_train_Ar_ours,
        "relearning test Af": relearn_test_Af_ours,
        "relearning test Ar": relearn_test_Ar_ours,
    })

summary_df = pd.DataFrame(rows)

summary_df.to_csv(OUT_CSV, index=False)

print(summary_df.to_string(index=False))
print(f"\nSaved CSV to: {OUT_CSV}")


# ========= HELPERS =========
def get_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").dropna()
    return pd.Series(dtype=float)


def fmt_minmax(series: pd.Series, digits: int = 2) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "-"
    return f"({s.min():.{digits}f}, {s.max():.{digits}f})"


def fmt_meanstd(series: pd.Series, digits: int = 2) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "-"
    std = s.std(ddof=1) if len(s) > 1 else 0.0

    # no spaces around \pm, smaller std
    return f"{s.mean():.{digits}f}\\!\\pm\\!{{\\scriptscriptstyle {std:.{digits}f}}}"


# ========= BUILD LATEX ROWS =========
latex_df_rows = []

for method in ROW_LABELS.keys():
    proto_path = PROTO_FILES.get(method)
    ours_path = OURS_FILES.get(method)

    baseline_test_Af = "-"
    baseline_test_Ar = "-"
    ours_test_Af = "-"
    ours_test_Ar = "-"

    if proto_path is not None and proto_path.exists():
        df_proto = pd.read_csv(proto_path)

        # Ref [19]: forget accuracy as min/max, retain as mean±std
        baseline_test_Af = fmt_minmax(get_col(df_proto, ["pra_test_fgt"]))
        baseline_test_Ar = fmt_meanstd(get_col(df_proto, ["pra_test_retain"]))

    if ours_path is not None and ours_path.exists():
        df_ours = pd.read_csv(ours_path)

        # Ours: forget accuracy as min/max, retain as mean±std
        ours_test_Af = fmt_minmax(get_col(df_ours, ["test_fgt", "pra_test_fgt"]))
        ours_test_Ar = fmt_meanstd(get_col(df_ours, ["test_retain", "pra_test_retain"]))

    latex_df_rows.append({
        "Method": ROW_LABELS[method],
        "Baseline_Af": baseline_test_Af,
        "Baseline_Ar": baseline_test_Ar,
        "Ours_Af": ours_test_Af,
        "Ours_Ar": ours_test_Ar,
    })

latex_df = pd.DataFrame(latex_df_rows)

# ========= MAKE LATEX TABLE =========
latex_lines = []
latex_lines.append(r"\begin{table}[h]")
latex_lines.append(r"\centering")
latex_lines.append(r"\scriptsize")
latex_lines.append(r"\setlength{\tabcolsep}{3pt}")
latex_lines.append(r"\renewcommand{\arraystretch}{0.95}")
latex_lines.append(
    r"\caption{Comparison with the relearning attck baseline with Ref.~[19] on CIFAR-10 under the same ResNet-18 setting. Forget accuracy ($\mathcal{A}_f$) is reported as (min,max) across forget classes, while retain accuracy ($\mathcal{A}_r$) is reported as mean$\pm$std across forget classes. }"
)
latex_lines.append(r"\label{tab:cifar10_compare_test_only}")
latex_lines.append(r"\begin{tabular}{lcccc}")
latex_lines.append(r"\toprule")
latex_lines.append(
    r"Method & $\mathcal{A}_f$ ([19]) & $\mathcal{A}_r$ ([19]) & $\mathcal{A}_f$ (Ours) & $\mathcal{A}_r$ (Ours) \\"
)
latex_lines.append(r"\midrule")

for _, row in latex_df.iterrows():
    latex_lines.append(
        f"{row['Method']} & "
        f"${row['Baseline_Af']}$ & ${row['Baseline_Ar']}$ & "
        f"${row['Ours_Af']}$ & ${row['Ours_Ar']}$ \\\\"
    )

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table}")

latex_table = "\n".join(latex_lines)

with open(OUT_TEX, "w", encoding="utf-8") as f:
    f.write(latex_table)

print(latex_table)
print(f"\nSaved TEX to: {OUT_TEX}")