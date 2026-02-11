import pandas as pd
import re
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from itertools import product

# ===================== CONFIG =====================

# Root directory that contains the two settings (5-class / 10-class) OR everything mixed.
ROOT_DIR = Path(r"C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/class_unlearning/results_multi_class_5_10/")

# --- IMPORTANT: Set these to YOUR actual folders (best & safest) ---
# Each should contain method subfolders (finetune/, scrub/, ...) and optionally original/
SETTING_DIRS: Dict[str, Path] = {
    "5-class":  ROOT_DIR / "5",     # <-- change if needed
    "10-class": ROOT_DIR / "10",    # <-- change if needed
}

# If you don't have separate folders "5" and "10", set them to ROOT_DIR and rely on autodetect,
# OR edit infer_setting_from_path() below to match your filename conventions.

DATASET_TARGET = "cifar100"  # <-- what you want for the joint table (same dataset)

DATASETS = ["cifar10", "cifar100", "tiny_imagenet"]
MODELS   = ["resnet18", "vit-s-16", "vit-b-16", "swin-t", "vgg16"]

# include ORIGINAL as a method
methods = [
    "original", "retrained",
    "random_label", "finetune", "gradient_ascent", "neggrad_plus",
    "boundary_shrink", "boundary_expand",
    "l2ul_adv", "l2ul_imp", "fisher", "wood_fisher",
    "scrub", "bad_teacher", "salun", "delete",
]
KNOWN_METHODS = set(methods)

# ---- Table output choices ----
SHOW_TRAIN_METRICS = False
ALL_METRIC_COLS = ["train_retain_acc","train_forget_acc","test_retain_acc","test_forget_acc"]
OUT_METRIC_COLS = ALL_METRIC_COLS if SHOW_TRAIN_METRICS else ["test_retain_acc","test_forget_acc"]

COL_LABELS = {
    "train_retain_acc": r"$\mathcal{A}^{\text{train}}_{r}(\%)$",
    "train_forget_acc": r"$\mathcal{A}^{\text{train}}_{f}(\%)$",
    "test_retain_acc":  r"$\mathcal{A}^{t}_{r}(\%)$",
    "test_forget_acc":  r"$\mathcal{A}^{t}_{f}(\%)$",
}
RS_LABEL = r"RS"

# Keep only these forget classes for TinyImageNet
FORGET_CLASS_FILTERS = {"tiny_imagenet": {40, 80, 120, 160}}

# Preferred order in tables
METHOD_ORDER = [
    "original", "retrained", "finetune",
    "gradient_ascent", "neggrad_plus", "random_label",
    "boundary_shrink", "boundary_expand",
    "l2ul_adv", "l2ul_imp",
    "fisher", "wood_fisher",
    "scrub", "bad_teacher", "salun", "delete",
]

method_name_and_ref = {
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
    "scrub": r"SCRUB \cite{kurmanji2023towards}",
    "bad_teacher": r"Bad Teacher \cite{chundawat2023can}",
    "salun": r"SalUn \cite{fan2023salun}",
    "delete": r"DELETE \cite{zhou2025decoupled}",
}


SETTING_DISPLAY = {
    "forget5": "5-Classes",
    "forget10": "10-Classes",
}


# ===================== HELPERS =====================

def slugify(s: Optional[str]) -> str:
    if s is None:
        return "unknown"
    return re.sub(r'[^A-Za-z0-9\-]+', "_", str(s))

def _normalize_key(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s)
    return s.replace("-", "_").lower().strip()

def _method_label(raw: str) -> str:
    v = method_name_and_ref.get(_normalize_key(raw))
    return raw if v is None else (v[0] if isinstance(v, tuple) else v)

def _latex_model_name(mdl: str) -> str:
    mapping = {
        "resnet18": "ResNet-18",
        "vit-s-16": "ViT-S/16",
        "vit-b-16": "ViT-B/16",
        "swin-t":   "Swin-T",
        "vgg16":    "VGG-16",
    }
    return mapping.get(mdl, mdl)

def _latex_dataset_name(ds: str) -> str:
    key = ds.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    mapping = {
        "cifar10":     "CIFAR-10",
        "cifar100":    "CIFAR-100",
        "tinyimagenet":"TinyImageNet",
    }
    if key in mapping:
        return mapping[key]
    m = re.match(r"^([a-z]+)(\d+)$", key)
    if m:
        return f"{m.group(1).upper()}-{m.group(2)}"
    return ds.replace("_", " ").title()

def fmt_mu(mu):
    if pd.isna(mu):
        return "-"
    return rf"${float(mu):.2f}$"

def _rank_styles(values):
    vals = sorted({float(v) for v in values if pd.notna(v)}, reverse=True)
    if not vals:
        return (None, None)
    if len(vals) == 1:
        return (vals[0], None)
    return (vals[0], vals[1])

def _style_rs(v, max_v, second_v):
    if pd.isna(v):
        return "-"
    val = f"{float(v):.3f}"
    if (max_v is not None) and (abs(v - max_v) < 1e-12):
        return rf"\textbf{{\boldmath ${val}$}}"
    if (second_v is not None) and (abs(v - second_v) < 1e-12):
        return rf"\underline{{$ {val} $}}"
    return rf"${val}$"

def wrap_with_resizebox(latex_src: str, caption: str, label: str, width: str = r"\columnwidth") -> str:
    return (
        f"\\begin{{table}}[h]\n"
        f"\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\resizebox{{{width}}}{{!}}{{%\n{latex_src}\n}}\n"
        f"\\end{{table}}\n"
    )

def add_midrules_between_methods(latex_src: str) -> str:
    """
    - double \toprule
    - double \midrule after header
    - double \midrule after Original row
    - double \midrule after Retrained *block* (after its Revival row)
    - \midrule after each method block (after Revival row)
    """
    lines = latex_src.splitlines()
    out = []
    duplicated_toprule = False
    duplicated_header_sep = False

    in_retrained_block = False

    for line in lines:
        stripped = line.strip()
        out.append(line)

        # double top rule
        if stripped.startswith(r"\toprule") and not duplicated_toprule:
            out.append(r"\toprule")
            duplicated_toprule = True
            continue

        # double header separator (first \midrule after header)
        if stripped.startswith(r"\midrule") and not duplicated_header_sep:
            out.append(r"\midrule")
            duplicated_header_sep = True
            continue

        # detect start of retrained block (unlearned row has the multirow label)
        if r"\multirow{2}{*}{Retrained" in line:
            in_retrained_block = True

        # body separators (after rows)
        if line.rstrip().endswith(r"\\"):
            # double rule after Original row
            if re.search(r"^\s*Original\s*&", line):
                out.append(r"\midrule")
                out.append(r"\midrule")
                continue

            # after each method block (after Revival row)
            if " & Revival &" in line:
                if in_retrained_block:
                    out.append(r"\midrule")
                    out.append(r"\midrule")
                    in_retrained_block = False
                else:
                    out.append(r"\midrule")

    return "\n".join(out)

def add_group_vertical_bars_settings(latex_src: str, group_labels: List[str]) -> str:
    """
    Ensure vertical rules continue through the setting \multicolumn headers.
    The first group should be 'c' (no leading bar inside the multicolumn cell),
    the second group (and onwards) should be '|c' so the divider appears.
    """
    out = latex_src
    ncols = len(OUT_METRIC_COLS) + 1  # metrics + RS per group

    for i, g in enumerate(group_labels):
        spec = "c" if i == 0 else "|c"
        pat = rf"(\\multicolumn\{{{ncols}\}}\{{)(?:\|?c\|?)(\}}\{{[^}}]*{re.escape(g)}[^}}]*\}})"
        out = re.sub(pat, rf"\1{spec}\2", out)

    return out

def apply_multirow(table_df: pd.DataFrame) -> pd.DataFrame:
    df = table_df.copy()
    for label, block in df.groupby("Method", sort=False):
        idxs = list(block.index)
        if len(idxs) >= 2:
            df.at[idxs[0], "Method"] = rf"\multirow{{{len(idxs)}}}{{*}}{{{label}}}"
            df.loc[idxs[1:], "Method"] = ""
    return df

def _apply_forget_filter(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    allowed = FORGET_CLASS_FILTERS.get(dataset)
    if not allowed or "forget_class" not in df.columns:
        return df
    fc_num = pd.to_numeric(df["forget_class"], errors="coerce").astype("Int64")
    return df[fc_num.isin(list(allowed))].copy()

def infer_setting_from_filename(path: Path) -> str:
    """
    Detect forget-K setting from filename. Examples:
      ..._forget5_...  -> "forget5"
      ..._forget10_... -> "forget10"
    """
    m = re.search(r"_forget(\d+)", path.stem.lower())
    if m:
        return f"forget{m.group(1)}"
    return "unknown"

def infer_setting_from_df(d: pd.DataFrame) -> str:
    # 1) if there is an explicit column
    for col in ["forget", "forget_k", "num_forget", "forget_classes"]:
        if col in d.columns:
            try:
                return f"forget{int(pd.to_numeric(d[col].iloc[0], errors='coerce'))}"
            except:
                pass

    # 2) fallback: infer K from "forget_class" string like "23,25,38,..." (your revival_multi file)
    if "forget_class" in d.columns:
        s = str(d["forget_class"].iloc[0])
        if "," in s:
            k = len([x for x in s.split(",") if str(x).strip() != ""])
            if k > 0:
                return f"forget{k}"

    return "unknown"

def infer_setting_from_forget_class_value(x) -> str:
    if pd.isna(x):
        return "unknown"
    s = str(x).strip()

    # case: "23,25,38,49,51"
    if "," in s:
        k = len([t for t in s.split(",") if t.strip() != ""])
        return f"forget{k}" if k > 0 else "unknown"

    # case: single class "23"
    try:
        int(float(s))
        return "forget1"
    except:
        return "unknown"
    
    
def _pick(value, hint, what: str, path: Path):
    """
    Prefer `value` (parsed from filename). If missing, fallback to `hint`.
    Otherwise raise a clear error.
    """
    if value not in (None, "", np.nan):
        return value
    if hint not in (None, "", np.nan):
        return hint
    raise ValueError(
        f"Cannot infer {what} from filename '{path.name}', and no {what} hint was provided."
    )
    
# ===================== RS COMPUTATION =====================

def compute_rs_for_revival1(df_rev: pd.DataFrame, df_un: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df_rev is None or df_rev.empty or df_un is None or df_un.empty:
        out = df_rev.copy()
        out["RS1"] = pd.NA
        return out

    un_keep = ["forget_class","dataset","model","method","setting","test_retain_acc","test_forget_acc"]
    keys = ["forget_class","dataset","model","method","setting"]

    un = df_un[un_keep].copy().rename(columns={
        "test_retain_acc": "test_retain_acc_un",
        "test_forget_acc": "test_forget_acc_un",
    })
    m = df_rev.merge(un, on=keys, how="left")

    A_re_t_r = pd.to_numeric(m["test_retain_acc"], errors="coerce")
    A_re_t_f = pd.to_numeric(m["test_forget_acc"], errors="coerce")
    A_un_t_r = pd.to_numeric(m["test_retain_acc_un"], errors="coerce")
    A_un_t_f = pd.to_numeric(m["test_forget_acc_un"], errors="coerce")

    num = (A_un_t_r - A_re_t_r).abs()
    den = (A_un_t_f - A_re_t_f).abs()

    den_zero = (den == 0) | den.isna()
    rs = np.where(den_zero & (num.fillna(0) == 0), 1.0,
          np.where(den_zero, np.nan, 1 - (num / den)))

    m["RS1"] = rs
    return m

def compute_rs_for_revival2(df_rev: pd.DataFrame, df_un: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df_rev is None or df_rev.empty:
        return df_rev
    if df_un is None or df_un.empty:
        out = df_rev.copy()
        out["RS2"] = pd.NA
        return out

    keys = ["forget_class", "dataset", "model", "method", "setting"]
    need_un = keys + ["test_retain_acc", "test_forget_acc"]

    out = df_rev.drop(columns=["test_retain_acc_un", "test_forget_acc_un"], errors="ignore")
    un_small = df_un[need_un].rename(columns={
        "test_retain_acc": "test_retain_acc_un",
        "test_forget_acc": "test_forget_acc_un"
    })
    out = out.merge(un_small, on=keys, how="left")

    req_cols = ["test_retain_acc", "test_forget_acc",
                "test_retain_acc_un", "test_forget_acc_un"]
    for c in req_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    max_val = pd.concat([out[c] for c in req_cols], axis=0).max(skipna=True)
    S = 100.0 if (pd.notna(max_val) and max_val > 1.5) else 1.0

    Ar_un = out["test_retain_acc_un"] / S
    Ar_re = out["test_retain_acc"]    / S
    Af_un = out["test_forget_acc_un"] / S
    Af_re = out["test_forget_acc"]    / S

    retain_drop = (Ar_un - Ar_re).clip(lower=0.0)
    forget_gain = (Af_re - Af_un).clip(lower=0.0)

    out["RS2"] = (1.0 - retain_drop) * forget_gain
    return out

def get_k_from_setting(s):
    m = re.search(r"forget(\d+)", str(s))
    return m.group(1) if m else None

# ===================== FILE DISCOVERY + LOADING =====================

def parse_filename(path: Path):
    name = path.stem
    toks = name.split("_")
    ds  = toks[0] if len(toks) > 0 else None
    mdl = toks[1] if len(toks) > 1 else None

    mth = None
    for t in toks:
        if t in KNOWN_METHODS:
            mth = t
            break
    if mth is None and len(toks) > 3 and toks[3] in KNOWN_METHODS:
        mth = toks[3]

    if "revival" in name:
        phase = "revival"
    elif "forget" in name:
        phase = "unlearned"
    elif ("original" in toks) or (mth == "original"):
        phase = "original"
    else:
        phase = "unknown"
    return ds, mdl, mth, phase

def find_all(dirpath: Path, patterns: List[str]) -> List[Path]:
    candidates = []
    for pat in patterns:
        candidates.extend(dirpath.glob(pat))
    # unique + sorted (optional but nice)
    uniq = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    return uniq

def find_one_across(dirs: List[Path], patterns: List[str]) -> Optional[Path]:
    best = None
    best_mtime = -1
    for d in dirs:
        for pat in patterns:
            for p in d.glob(pat):
                mt = p.stat().st_mtime
                if mt > best_mtime:
                    best, best_mtime = p, mt
    return best

def standardize_cols(df: pd.DataFrame, ds: str, mdl: str, mth: str, setting: str) -> pd.DataFrame:
    if "dataset" not in df.columns:
        df["dataset"] = ds
    if "model" not in df.columns:
        df["model"] = mdl
    if "method" not in df.columns:
        df["method"] = mth
    if "setting" not in df.columns:
        df["setting"] = setting

    rename_map = {
        "train_retain": "train_retain_acc",
        "train_fgt":    "train_forget_acc",
        "test_retain":  "test_retain_acc",
        "test_fgt":     "test_forget_acc",
        "retain_test_acc": "test_retain_acc",
        "forget_test_acc": "test_forget_acc",
    }
    df = df.rename(columns=rename_map)

    for col in ALL_METRIC_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    for col in ALL_METRIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def load_forget_csv(path: Path, ds_hint=None, mdl_hint=None, mth_hint=None) -> pd.DataFrame:
    ds_p, mdl_p, mth_p, _ = parse_filename(path)
    ds  = _pick(ds_p,  ds_hint,  "dataset", path)
    mdl = _pick(mdl_p, mdl_hint, "model",   path)
    mth = mth_p or mth_hint or "unknown"

    setting = infer_setting_from_filename(path)

    d = pd.read_csv(path)
    d = standardize_cols(d, ds, mdl, mth, setting)
    if "forget_class" not in d.columns:
        d["forget_class"] = pd.NA

    keep_cols = ["forget_class","dataset","model","method","setting",
                 "train_retain_acc","train_forget_acc","test_retain_acc","test_forget_acc"]
    return d[keep_cols].copy()

def load_revival_csv(path: Path, ds_hint=None, mdl_hint=None, mth_hint=None) -> pd.DataFrame:
    ds_p, mdl_p, mth_p, _ = parse_filename(path)
    ds  = _pick(ds_p,  ds_hint,  "dataset", path)
    mdl = _pick(mdl_p, mdl_hint, "model",   path)
    mth = mth_p or mth_hint or "unknown"

    d = pd.read_csv(path)

    # ✅ infer setting from file name IF possible, otherwise from forget_class per-row
    setting = infer_setting_from_filename(path)
    if setting == "unknown":
        d["setting"] = d["forget_class"].apply(infer_setting_from_forget_class_value)
    else:
        d["setting"] = setting

    if "epoch" in d.columns:
        d = d.sort_values(["forget_class","epoch"]).groupby("forget_class", as_index=False).tail(1)

    d = standardize_cols(d, ds, mdl, mth, setting="unknown")  # won't overwrite d["setting"]

    keep_cols = ["forget_class","dataset","model","method","setting",
                 "train_retain_acc","train_forget_acc","test_retain_acc","test_forget_acc"]
    return d[keep_cols].copy()


def load_original_csv(path: Path, ds_hint=None, mdl_hint=None) -> pd.DataFrame:
    ds_p, mdl_p, _, _ = parse_filename(path)
    ds  = _pick(ds_p,  ds_hint,  "dataset", path)
    mdl = _pick(mdl_p, mdl_hint, "model",   path)

    setting = "original"

    d = pd.read_csv(path)
    d = standardize_cols(d, ds, mdl, "original", setting)
    if "forget_class" not in d.columns:
        d["forget_class"] = -1

    keep_cols = ["forget_class","dataset","model","method","setting",
                 "train_retain_acc","train_forget_acc","test_retain_acc","test_forget_acc"]
    return d[keep_cols].copy()
# ===================== TABLE RENDER: JOINT (SAME DATASET, TWO SETTINGS) =====================

def center_method_phase_headers_settings(latex_src: str, group_labels: List[str]) -> str:
    """
    Rewrites the FIRST header row to show grouped setting names centered,
    and blanks the first two cells of the SECOND header row.
    Assumes first two columns are: Unlearning Method, Model Variant.
    """
    lines = latex_src.splitlines()
    if not lines:
        return latex_src

    # Find first header row (the one containing \multicolumn{...})
    h1 = next((i for i, L in enumerate(lines) if r"\multicolumn" in L), None)
    if h1 is None:
        return latex_src

    # Find second header row (next line with '&' and ending with '\\')
    h2 = None
    for j in range(h1 + 1, min(h1 + 10, len(lines))):
        Lj = lines[j].strip()
        if "&" in Lj and Lj.endswith(r"\\"):
            h2 = j
            break
    if h2 is None:
        return latex_src

    # number of columns per group = metrics + RS
    ncols = len(OUT_METRIC_COLS) + 1

    group_bits = [rf"\multicolumn{{{ncols}}}{{c}}{{\textbf{{{g}}}}}" for g in group_labels]
    lines[h1] = (
        r"\multirow{2}{*}{Unlearning Method} & "
        r"\multirow{2}{*}{Model Variant} & "
        + " & ".join(group_bits)
        + r" \\"
    )

    # --- helper to split/join latex header row cells ---
    def _split_cells(line: str):
        trail = ""
        if line.rstrip().endswith(r"\\"):
            k = line.rfind(r"\\")
            body, trail = line[:k], line[k:]
        else:
            body, trail = line, ""
        cells = [c.strip() for c in body.split("&")]
        return cells, trail

    def _join_cells(cells, trail):
        return " & ".join(cells) + trail

    # Blank first 2 cells of second header row
    h2_cells, h2_trail = _split_cells(lines[h2])
    while len(h2_cells) < 2:
        h2_cells.append("")
    h2_cells[0] = ""
    h2_cells[1] = ""
    lines[h2] = _join_cells(h2_cells, h2_trail)

    return "\n".join(lines)

def wrap_with_wraptable(
    latex_src: str,
    caption: str,
    label: str,
    width: str = r"0.5\columnwidth",
    placement: str = "r",          # r = right, l = left
    vspace_top: str = r"-0.5\baselineskip",
    vspace_bottom: str = r"-0.5\baselineskip",
) -> str:
    """
    Wraps a tabular in a wraptable + resizebox(width) + \\small.
    NOTE: wrapfig package must be loaded in LaTeX: \\usepackage{wrapfig}
    """
    top = f"\\vspace{{{vspace_top}}}\n" if vspace_top else ""
    bot = f"\\vspace{{{vspace_bottom}}}\n" if vspace_bottom else ""

    return (
        f"\\begin{{wraptable}}{{{placement}}}{{{width}}}\n"
        f"{top}"
        f"\\centering\n"
        f"\\small\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\resizebox{{{width}}}{{!}}{{%\n{latex_src}\n}}\n"
        f"{bot}"
        f"\\end{{wraptable}}\n"
    )

def add_cmidrules_for_groups(latex_src: str, n_groups: int, group_width: int, start_col: int = 3) -> str:
    """
    Insert \cmidrule lines RIGHT BEFORE the first \midrule (header separator).
    start_col=3 because columns 1-2 are method + variant.
    """
    lines = latex_src.splitlines()
    mid_idx = next((i for i, L in enumerate(lines) if L.strip().startswith(r"\midrule")), None)
    if mid_idx is None:
        return latex_src

    rules = []
    for g in range(n_groups):
        a = start_col + g * group_width
        b = start_col + (g + 1) * group_width - 1
        rules.append(rf"\cmidrule(lr){{{a}-{b}}}")

    lines = lines[:mid_idx] + rules + lines[mid_idx:]
    return "\n".join(lines)

def render_joint_table_same_dataset(mdl: str, df_src: pd.DataFrame, dataset: str, settings: List[str], out_dir: Path) -> Optional[Path]:
    df = df_src[(df_src["model"] == mdl) & (df_src["dataset"] == dataset) & (df_src["setting"].isin(settings))].copy()
    if df.empty:
        print(f"[WARN] No rows for model={mdl}, dataset={dataset}, settings={settings}")
        return None

    for c in ALL_METRIC_COLS:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    if "RS2" in df.columns:
        df["RS2"] = pd.to_numeric(df["RS2"], errors="coerce")

    agg_cols = ALL_METRIC_COLS.copy()
    if "RS2" in df.columns:
        agg_cols.append("RS2")

    g = df.groupby(["setting", "method", "phase"], dropna=False)[agg_cols].agg(["mean"])

    present = [m for m in METHOD_ORDER if m in df["method"].unique()]
    extras  = sorted(set(df["method"].unique()) - set(present))
    method_list = present + extras

    # RS ranking per setting
    rs_per_setting = { s: {} for s in settings }
    for s in settings:
        for m in method_list:
            if m == "original":
                rs_per_setting[s][m] = np.nan
                continue
            if ("RS2" in g.columns.get_level_values(0).unique()) and ((s, m, "revival") in g.index):
                rs_mu = g.loc[(s, m, "revival"), ("RS2", "mean")]
                rs_per_setting[s][m] = float(rs_mu) if pd.notna(rs_mu) else np.nan
            else:
                rs_per_setting[s][m] = np.nan

    rs_rank_thresholds = { s: _rank_styles(list(rs_per_setting[s].values())) for s in settings }

    rows = []
    for m in method_list:
        pretty = _method_label(m)

        if m == "original":
            row = {"Method": pretty, "Phase": "Original"}
            for s in settings:
                row[(s, RS_LABEL)] = "-"
                if (s, m, "original") in g.index:
                    for col in OUT_METRIC_COLS:
                        mu = g.loc[(s, m, "original"), (col, "mean")]
                        row[(s, COL_LABELS[col])] = fmt_mu(mu)
                else:
                    for col in OUT_METRIC_COLS:
                        row[(s, COL_LABELS[col])] = "-"
            rows.append(row)
            continue

        # Unlearned row (RS shown here)
        row_un = {"Method": rf"\multirow{{2}}{{*}}{{{pretty}}}", "Phase": "Unlearned"}
        for s in settings:
            raw_rs = rs_per_setting[s].get(m, np.nan)
            max_v, second_v = rs_rank_thresholds[s]
            rs_text_styled = _style_rs(raw_rs, max_v, second_v) if pd.notna(raw_rs) else "-"
            row_un[(s, RS_LABEL)] = rf"\multirow{{2}}{{*}}{{{rs_text_styled}}}"

            if (s, m, "unlearned") in g.index:
                for col in OUT_METRIC_COLS:
                    mu = g.loc[(s, m, "unlearned"), (col, "mean")]
                    row_un[(s, COL_LABELS[col])] = fmt_mu(mu)
            else:
                for col in OUT_METRIC_COLS:
                    row_un[(s, COL_LABELS[col])] = "-"
        rows.append(row_un)

        # Revival row
        row_rev = {"Method": "", "Phase": "Revival"}
        for s in settings:
            row_rev[(s, RS_LABEL)] = ""
            if (s, m, "revival") in g.index:
                for col in OUT_METRIC_COLS:
                    mu = g.loc[(s, m, "revival"), (col, "mean")]
                    row_rev[(s, COL_LABELS[col])] = fmt_mu(mu)
            else:
                for col in OUT_METRIC_COLS:
                    row_rev[(s, COL_LABELS[col])] = "-"
        rows.append(row_rev)

    # Column order: per setting (metrics then RS)
    ordered_cols = [("","Unlearning Method"), ("","Model Variant")]
    for s in settings:
        for c in OUT_METRIC_COLS:
            ordered_cols.append((s, COL_LABELS[c]))
        ordered_cols.append((s, RS_LABEL))

    table_df = pd.DataFrame(rows)
    table_df = table_df.rename(columns={"Method": ("","Unlearning Method"), "Phase": ("","Model Variant")})
    table_df.columns = pd.MultiIndex.from_tuples(table_df.columns)
    table_df = table_df[ordered_cols]

    for col in [("","Unlearning Method"), ("","Model Variant")]:
        table_df[col] = (table_df[col].astype(str)
                         .str.replace("_", r"\_", regex=False)
                         .str.replace("&", r"\&", regex=False)
                         .str.replace("%", r"\%", regex=False))

    cols_per_setting = 1 + len(OUT_METRIC_COLS)  # RS + metrics
    # e.g., c|c|cccc|cccc   (single bar)  -> we want c|c|cccc||cccc (double bar between groups)
    blocks = [("c" * cols_per_setting) for _ in settings]
    column_format = "c|c|" + ("|".join(blocks))
    

    latex = table_df.to_latex(index=False, escape=False, multicolumn=True,
                              multicolumn_format="c", column_format=column_format)
    
    # 1) rename group headers to "5-Classes / 10-Classes"
    display_groups = [SETTING_DISPLAY.get(s, s) for s in settings]
    latex = center_method_phase_headers_settings(latex, display_groups)
    latex = add_group_vertical_bars_settings(latex, display_groups)
    
    # 2) add cmidrules under the two group headers
    group_width = len(OUT_METRIC_COLS) + 1  # metrics + RS
    #latex = add_cmidrules_for_groups(latex, n_groups=len(settings), group_width=group_width)
    
    # 3) add method separators (without duplicating header lines)
    latex = add_midrules_between_methods(latex)

    mdl_latex = _latex_model_name(mdl)
    ds_latex  = _latex_dataset_name(dataset)
    caption = (rf"The results of the proposed class revival applied to multi-class "
               rf"(5-Classes and 10-Classes) unlearned models on CIFAR-100 for ResNet-18. "
               rf"For each dataset, \textbf{{bold}} indicates the highest RS and "
               rf"\underline{{underlined}} indicates the second-highest RS.")
    label = f"tab:{slugify(ds_latex)}_{slugify(mdl_latex)}_5v10"
    #latex = wrap_with_resizebox(latex, caption, label, width=r"\columnwidth")

    latex = wrap_with_wraptable(
        latex, caption, label,
        width=r"0.5\columnwidth",
        placement="r",          # "l" if you want it on the left
        vspace_top=r"-1.2cm",
        vspace_bottom=r"-1cm",
    )
    


    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"latex_table_{slugify(dataset)}_{slugify(mdl)}_5v10.tex"
    out.write_text(latex, encoding="utf-8")
    print(f"[OK] wrote: {out}")
    return out

# ===================== MAIN PIPELINE =====================
def make_forget_id_from_forget_class(x, setting):
    s = "" if pd.isna(x) else str(x)
    # if it's an explicit list of forgotten classes, keep it
    if "," in s:
        return s
    # otherwise fall back to the setting (forget5/forget10)
    return setting

def autodetect_setting_dirs(root: Path) -> Dict[str, Path]:
    # if user didn't set valid dirs, try to find subfolders containing 5/10
    subs = [p for p in root.iterdir() if p.is_dir()]
    pick5 = next((p for p in subs if re.search(r"(^|[^0-9])5([^0-9]|$)", p.name.lower()) or "5class" in p.name.lower()), None)
    pick10 = next((p for p in subs if re.search(r"(^|[^0-9])10([^0-9]|$)", p.name.lower()) or "10class" in p.name.lower()), None)
    out = {}
    if pick5: out["5-class"] = pick5
    if pick10: out["10-class"] = pick10
    return out

def normalize_retrained_keys(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    df = df.copy()

    # 1) Ensure setting exists for revival_multi (it has list strings)
    if "setting" not in df.columns or (df["setting"] == "unknown").any():
        if "forget_class" in df.columns:
            df["setting"] = df["forget_class"].apply(infer_setting_from_forget_class_value)

    # 2) For retrained revival: convert list-string -> K (5/10) so it matches unlearned
    if phase == "revival" and "forget_class" in df.columns:
        k = get_k_from_setting(df["setting"].iloc[0])  # "5" or "10"
        if k is not None:
            df["forget_class"] = str(k)

    # 3) For retrained unlearned: force forget_class also to "5"/"10" (string)
    if phase == "unlearned":
        if "forget" in df.columns and "forget_class" in df.columns:
            df["forget_class"] = df["forget"].astype(str)
        else:
            df["forget_class"] = df["forget_class"].astype(str)

    # Always string-ify for merge stability
    df["forget_class"] = df["forget_class"].astype(str)

    return df

def collect_all_rows(setting_name: str, base_dir: Path) -> List[pd.DataFrame]:
    all_rows = []

    for ds in DATASETS:
        for mdl in MODELS:
            for mth in methods:
                df_f = None

                # ---------- ORIGINAL ----------
                if mth == "original":
                    original_search_dirs = [ROOT_DIR / "original", ROOT_DIR]
                    original_patterns = [
                        f"results_original_{ds}_{mdl}.csv",
                        f"{ds}_{mdl}_original*_metrics*.csv",
                        f"{ds}_{mdl}_original*.csv",
                    ]
                    original_path = find_one_across(original_search_dirs, original_patterns)
                    if original_path is None:
                        continue
                
                    try:
                        d = pd.read_csv(original_path)
                
                        # IMPORTANT: your original file has a "forget" column (5 or 10)
                        if "forget" in d.columns:
                            d["setting"] = d["forget"].apply(lambda x: f"forget{int(x)}")
                        else:
                            # fallback: single original row -> duplicate if needed
                            d["setting"] = "original"
                
                        d = standardize_cols(d, ds, mdl, "original", d["setting"])
                        d["phase"] = "original"
                        d["forget_class"] = "-1"
                
                        # Keep only the settings you want in the joint table
                        d = d[d["setting"].isin(["forget5", "forget10"])].copy()
                
                        keep_cols = ["forget_class","dataset","model","method","setting",
                                     "train_retain_acc","train_forget_acc","test_retain_acc","test_forget_acc","phase"]
                        all_rows.append(d[keep_cols])
                
                    except Exception as e:
                        print(f"[ERROR] original failed {setting_name} {ds}/{mdl}: {original_path}\n{e}")
                    continue
                

                # ---------- NON-ORIGINAL ----------
                method_dir = base_dir / mth

                forget_patterns = [
                    f"{ds}_{mdl}_unlearned_{mth}_forget*_model_metrics_lr*.csv",
                    f"{ds}_{mdl}_unlearned_{mth}_forget*_metrics_lr*.csv",
                    f"{ds}_{mdl}_unlearned_{mth}_forget*.csv",
                ]
                revival_patterns = [
                    f"{ds}_{mdl}_unlearned_{mth}_revival*forget*class*_lr*.csv",
                    f"{ds}_{mdl}_unlearned_{mth}_revival*.csv",
                ]
                
                forget_paths  = find_all(method_dir, forget_patterns)
                revival_paths = find_all(method_dir, revival_patterns)
                
                if not forget_paths and not revival_paths:
                    continue
                
                # ---------- FORGET (load ALL: forget5 + forget10) ----------
                df_f_all = []  # keep all unlearned dfs so revival can match by "setting"
                for forget_path in forget_paths:
                    try:
                        df_f = load_forget_csv(forget_path, ds_hint=ds, mdl_hint=mdl, mth_hint=mth)
                        df_f["phase"] = "unlearned"
                    
                        
                        df_f["forget_class"] = df_f["forget_class"].astype(str)
                        df_f = _apply_forget_filter(df_f, ds)
                        all_rows.append(df_f)
                        df_f_all.append(df_f)
                    except Exception as e:
                        print(f"[ERROR] forget failed {setting_name} {mth} {ds}/{mdl}: {forget_path}\n{e}")
                
                df_f_all = pd.concat(df_f_all, ignore_index=True) if df_f_all else None
                
                # ---------- REVIVAL (load ALL; compute RS per matching setting) ----------
                for revival_path in revival_paths:
                    try:
                        df_r = load_revival_csv(revival_path, ds_hint=ds, mdl_hint=mdl, mth_hint=mth)
                        df_r["phase"] = "revival"
                        df_r["forget_class"] = df_r["forget_class"].astype(str)

                                        
                                        
                        if mth == "retrained":
                            if (df_r["setting"] == "unknown").any():
                                df_r["setting"] = df_r["forget_class"].apply(infer_setting_from_forget_class_value)
                        
                            # ✅ FIX: per-row forget_class (not iloc[0])
                            df_r["forget_class"] = df_r["setting"].apply(get_k_from_setting).astype(str)
                            
                        

                        if df_f_all is None or df_f_all.empty:
                            df_r["RS1"] = pd.NA
                            df_r["RS2"] = pd.NA
                        else:
                            # match RS using BOTH setting + forget_class + dataset + model + method
                            df_r = compute_rs_for_revival1(df_r, df_f_all)
                            df_r = compute_rs_for_revival2(df_r, df_f_all)
                
                        # --- Fix key mismatch for retrained (do NOT change RS functions) ---
                        if mth == "retrained":
                            # ensure setting is known (if needed)
                            if (df_r["setting"] == "unknown").any():
                                df_r["setting"] = df_r["forget_class"].apply(infer_setting_from_forget_class_value)
                
                            k = get_k_from_setting(df_r["setting"].iloc[0])
                            if k is not None:
                                df_r["forget_class"] = str(k)
                        # ---------------------------------------------------------------                
                
                
                
                        df_r = _apply_forget_filter(df_r, ds)
                        all_rows.append(df_r)
                    except Exception as e:
                        print(f"[ERROR] revival failed {setting_name} {mth} {ds}/{mdl}: {revival_path}\n{e}")

    return all_rows

def main():
    # Validate / autodetect setting dirs
    good = {k: v for k, v in SETTING_DIRS.items() if v.exists()}
    if len(good) < 2:
        auto = autodetect_setting_dirs(ROOT_DIR)
        if len(auto) >= 2:
            print("[INFO] Autodetected setting dirs:", auto)
            good = auto
        else:
            print("[WARN] Could not find two separate setting folders. Falling back to ROOT_DIR for both settings.")
            good = {"5-class": ROOT_DIR, "10-class": ROOT_DIR}

    # Collect rows for each setting
    all_rows = []
    for setting_name, base_dir in good.items():
        print(f"\n[INFO] Collecting setting={setting_name} from {base_dir}")
        all_rows.extend(collect_all_rows(setting_name, base_dir))

    if not all_rows:
        print("[WARN] No data collected. Check patterns/paths.")
        return

    merged = pd.concat(all_rows, ignore_index=True)

    # Save merged
    out_merged = ROOT_DIR / "z_merged_with_setting_all.csv"
    merged.to_csv(out_merged, index=False)
    print(f"\n[OK] Saved merged CSV: {out_merged}")

    # Load back for table generation
    df_all = pd.read_csv(out_merged)

    # Ensure numeric
    for c in ALL_METRIC_COLS:
        df_all[c] = pd.to_numeric(df_all.get(c), errors="coerce")
    if "RS2" in df_all.columns:
        df_all["RS2"] = pd.to_numeric(df_all["RS2"], errors="coerce")

    # Joint table: SAME DATASET comparing 5 vs 10
    out_dir = ROOT_DIR  
    settings = ["forget5", "forget10"]

    for mdl in MODELS:
        render_joint_table_same_dataset(mdl, df_all, dataset=DATASET_TARGET, settings=settings, out_dir=out_dir)

if __name__ == "__main__":
    main()