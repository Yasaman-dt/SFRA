import pandas as pd
import numpy as np
import re
from pathlib import Path

# ============================================================
# 1. PATHS
# ============================================================

ROOT_2 = Path(
    r"/projets/Zdehghani/"
    r"Source_Free_Class_Revival/results_multi_class_2/"
)

ROOT_510 = Path(
    r"/projets/Zdehghani/"
    r"Source_Free_Class_Revival/results_multi_class_5_10/"
)

CSV_2 = ROOT_2 / "z_standardized_selected_all_methods.csv"
CSV_510 = ROOT_510 / "z_merged_with_setting_all.csv"

OUT_DIR = ROOT_510
MODEL_TO_RENDER = "resnet18"


# ============================================================
# 2. TABLE SETTINGS
# ============================================================

OUT_METRIC_COLS = ["test_retain_acc", "test_forget_acc"]
RS_LABEL = "RS"

METHOD_ORDER = [
    "original",
    "retrained",
    "finetune",
    "gradient_ascent",
    "neggrad_plus",
    "random_label",
    "l2ul_adv",
    "scrub",
    "bad_teacher",
    "salun",
    "delete",
]

method_name_and_ref = {
    "original": r"Original",
    "retrained": r"Retrained",
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

GROUPS = [
    {
        "key": "cifar10_forget2",
        "dataset": "cifar10",
        "setting": "forget2",
        "label": r"\shortstack{\textbf{CIFAR-10}\\\textbf{(2-Classes)}}",
    },
    {
        "key": "cifar100_forget2",
        "dataset": "cifar100",
        "setting": "forget2",
        "label": r"\shortstack{\textbf{CIFAR-100}\\\textbf{(2-Classes)}}",
    },
    {
        "key": "cifar100_forget5",
        "dataset": "cifar100",
        "setting": "forget5",
        "label": r"\shortstack{\textbf{CIFAR-100}\\\textbf{(5-Classes)}}",
    },
    {
        "key": "cifar100_forget10",
        "dataset": "cifar100",
        "setting": "forget10",
        "label": r"\shortstack{\textbf{CIFAR-100}\\\textbf{(10-Classes)}}",
    },
]


# ============================================================
# 3. SMALL HELPERS
# ============================================================

def slugify(s):
    return re.sub(r"[^A-Za-z0-9\-]+", "_", str(s))


def normalize_setting(x):
    s = str(x).lower().strip()
    if "10" in s:
        return "forget10"
    if "5" in s:
        return "forget5"
    if "2" in s:
        return "forget2"
    return s


def normalize_key(s):
    return str(s).replace("-", "_").lower().strip()


def method_label(m):
    return method_name_and_ref.get(normalize_key(m), str(m))


def fmt_mu(x):
    if pd.isna(x):
        return "-"
    return rf"${float(x):.2f}$"


def rank_styles(values):
    vals = sorted({float(v) for v in values if pd.notna(v)}, reverse=True)
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], None
    return vals[0], vals[1]


def style_rs(v, max_v, second_v):
    if pd.isna(v):
        return "-"
    val = f"{float(v):.3f}"
    if max_v is not None and abs(float(v) - max_v) < 1e-12:
        return rf"\textbf{{\boldmath ${val}$}}"
    if second_v is not None and abs(float(v) - second_v) < 1e-12:
        return rf"\underline{{${val}$}}"
    return rf"${val}$"


# ============================================================
# 4. LOAD AND MERGE THE TWO RESULT FILES
# ============================================================

df2 = pd.read_csv(CSV_2)
df2["setting"] = "forget2"

df510 = pd.read_csv(CSV_510)

df_all = pd.concat([df2, df510], ignore_index=True)

for col in ["dataset", "model", "method", "phase", "setting"]:
    if col not in df_all.columns:
        raise KeyError(f"Missing column: {col}")
    df_all[col] = df_all[col].astype(str).str.lower().str.strip()

df_all["setting"] = df_all["setting"].apply(normalize_setting)

for col in OUT_METRIC_COLS + ["RS2"]:
    if col not in df_all.columns:
        df_all[col] = np.nan
    df_all[col] = pd.to_numeric(df_all[col], errors="coerce")


# ============================================================
# 5. ATTACH FOUR TABLE GROUPS
# ============================================================

def build_grouped_df(df_all):
    parts = []

    for g in GROUPS:
        key = g["key"]
        ds = g["dataset"]
        setting = g["setting"]

        # Non-original rows must match both dataset and setting
        part = df_all[
            (df_all["dataset"] == ds)
            & (df_all["setting"] == setting)
            & (df_all["phase"] != "original")
        ].copy()
        part["group"] = key
        parts.append(part)

        # Original row should appear under each group
        orig = df_all[
            (df_all["dataset"] == ds)
            & (df_all["phase"] == "original")
        ].copy()

        if not orig.empty:
            orig = orig.drop_duplicates(
                subset=["dataset", "model", "method", "phase"],
                keep="last",
            )
            orig["setting"] = setting
            orig["group"] = key
            parts.append(orig)

    return pd.concat(parts, ignore_index=True)


df_table = build_grouped_df(df_all)


# ============================================================
# 6. RENDER ONE BIG LATEX TABLE
# ============================================================

def render_four_group_table(mdl, df_src, out_dir):
    df = df_src[df_src["model"] == mdl].copy()

    if df.empty:
        print(f"[WARN] No rows for model={mdl}")
        return None

    group_keys = [g["key"] for g in GROUPS]
    group_labels = {g["key"]: g["label"] for g in GROUPS}

    agg_cols = OUT_METRIC_COLS + ["RS2"]
    g = (
        df.groupby(["group", "method", "phase"], dropna=False)[agg_cols]
        .mean(numeric_only=True)
    )

    present = [m for m in METHOD_ORDER if m in df["method"].unique()]
    extras = sorted(set(df["method"].unique()) - set(present))
    method_list = present + extras

    # RS ranking independently inside each group
    rs_per_group = {key: {} for key in group_keys}

    for key in group_keys:
        for m in method_list:
            if m == "original":
                rs_per_group[key][m] = np.nan
            elif (key, m, "revival") in g.index:
                rs_per_group[key][m] = g.loc[(key, m, "revival"), "RS2"]
            else:
                rs_per_group[key][m] = np.nan

    rs_thresholds = {
        key: rank_styles(list(rs_per_group[key].values()))
        for key in group_keys
    }

    def metric_cells(key, method, phase):
        idx = (key, method, phase)
        if idx not in g.index:
            return ["-", "-"]

        ar = g.loc[idx, "test_retain_acc"]
        af = g.loc[idx, "test_forget_acc"]
        return [fmt_mu(ar), fmt_mu(af)]

    body = []

    # ---------------- Original row ----------------
    original_cells = [r"Original", r"Original"]
    for key in group_keys:
        original_cells += metric_cells(key, "original", "original")
        original_cells += ["-"]

    body.append(" & ".join(original_cells) + r" \\")
    body.append(r"\midrule")
    body.append(r"\midrule")

    # ---------------- Other methods ----------------
    for m in method_list:
        if m == "original":
            continue

        has_any = any(
            ((key, m, "unlearned") in g.index) or ((key, m, "revival") in g.index)
            for key in group_keys
        )
        if not has_any:
            continue

        pretty = method_label(m)

        row_un = [rf"\multirow{{2}}{{*}}{{{pretty}}}", "Unlearned"]
        row_re = ["", "Revival"]

        for key in group_keys:
            row_un += metric_cells(key, m, "unlearned")

            raw_rs = rs_per_group[key].get(m, np.nan)
            max_v, second_v = rs_thresholds[key]
            rs_text = style_rs(raw_rs, max_v, second_v)
            row_un += [rf"\multirow{{2}}{{*}}{{{rs_text}}}"]

            row_re += metric_cells(key, m, "revival")
            row_re += [""]

        body.append(" & ".join(row_un) + r" \\")
        body.append(" & ".join(row_re) + r" \\")

        if m == "retrained":
            body.append(r"\midrule")
            body.append(r"\midrule")
        else:
            body.append(r"\midrule")

    # ---------------- Header ----------------
    group_header = []
    for i, key in enumerate(group_keys):
        align = "c|" if i < len(group_keys) - 1 else "c"
        group_header.append(
            rf"\multicolumn{{3}}{{{align}}}{{{group_labels[key]}}}"
        )

    header1 = (
        r"\multirow{2}{*}{Unlearning Method} & "
        r"\multirow{2}{*}{Model Variant} & "
        + " & ".join(group_header)
        + r" \\"
    )

    metric_header = []
    for _ in group_keys:
        metric_header += [
            r"$\mathcal{A}^{t}_{r}$",
            r"$\mathcal{A}^{t}_{f}$",
            r"RS",
        ]

    header2 = r" & & " + " & ".join(metric_header) + r" \\"

    col_format = "c|c|" + "|".join(["ccc"] * len(group_keys))

    tabular = "\n".join([
        rf"\begin{{tabular}}{{{col_format}}}",
        r"\toprule",
        r"\toprule",
        header1,
        header2,
        r"\midrule",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
    ])

    caption = (
        r"The results of the class revival applied to multi-class unlearned models "
        r"on CIFAR-10 and CIFAR-100 for ResNet-18. For each dataset, "
        r"\textbf{bold} indicates the highest RS and \underline{underlined} "
        r"indicates the second-highest RS."
    )

    latex = (
        r"\begin{table*}[t]" + "\n"
        r"\centering" + "\n"
        r"\small" + "\n"
        rf"\caption{{{caption}}}" + "\n"
        rf"\label{{tab:class_revival_{slugify(mdl)}_cifar10_cifar100}}" + "\n"
        r"\vspace{-0.2cm}" + "\n"
        r"\resizebox{\textwidth}{!}{%" + "\n"
        + tabular + "\n"
        r"}" + "\n"
        r"\end{table*}" + "\n"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"latex_table_{slugify(mdl)}_multi_class.tex"
    out_path.write_text(latex, encoding="utf-8")

    print(f"[OK] wrote: {out_path}")
    return out_path


render_four_group_table(MODEL_TO_RENDER, df_table, OUT_DIR)
