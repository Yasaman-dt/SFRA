import pandas as pd
import re
import numpy as np
from pathlib import Path
from typing import Optional, List

# ----------------- Config -----------------
base_dir = Path("C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/class_unlearning/results_multi_class/")

DATASETS = ["cifar10", "cifar100", "tiny_imagenet"]

# include ORIGINAL as a method
methods = [
    "original", "retrained",
    "random_label", "finetune", "gradient_ascent", "neggrad_plus",
    "boundary_shrink", "boundary_expand",
    "l2ul_adv", "l2ul_imp", "fisher", "wood_fisher", 
    "scrub", "bad_teacher", "salun", "delete",
]



MODELS   = ["resnet18", "vit-s-16", "vit-b-16", "swin-t", "vgg16"]

KNOWN_METHODS = set(methods)

# ---- Table output choices ----
SHOW_TRAIN_METRICS = False  # set True to keep train_* columns, False to drop them

# Columns to compute from CSV regardless (we still need them for aggregation)
ALL_METRIC_COLS = ["train_retain_acc","train_forget_acc","test_retain_acc","test_forget_acc"]

# Columns to actually display in LaTeX
if SHOW_TRAIN_METRICS:
    OUT_METRIC_COLS = ["train_retain_acc","train_forget_acc","test_retain_acc","test_forget_acc"]
else:
    OUT_METRIC_COLS = ["test_retain_acc","test_forget_acc"]

# Pretty labels for displayed columns only
COL_LABELS = {
    "train_retain_acc": r"$\mathcal{A}^{\text{train}}_{r}(\%)$",
    "train_forget_acc": r"$\mathcal{A}^{\text{train}}_{f}(\%)$",
    "test_retain_acc":  r"$\mathcal{A}^{t}_{r}(\%)$",
    "test_forget_acc":  r"$\mathcal{A}^{t}_{f}(\%)$",
}

RS_LABEL = r"$\mathrm{RS}$"

# Keep only these forget classes for TinyImageNet
FORGET_CLASS_FILTERS = {
    "tiny_imagenet": {40, 80, 120, 160}
}

def _apply_forget_filter(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    If a filter is defined for this dataset, keep only rows whose forget_class
    is in the allowed set. Works whether forget_class is str or numeric.
    """
    allowed = FORGET_CLASS_FILTERS.get(dataset)
    if not allowed or "forget_class" not in df.columns:
        return df

    # Robust to strings like "40" or numbers; ignores NaN gracefully.
    fc_num = pd.to_numeric(df["forget_class"], errors="coerce").astype("Int64")
    return df[fc_num.isin(list(allowed))].copy()


def compute_rs_for_revival1(df_rev: pd.DataFrame, df_un: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Adds RS to revival rows by merging in unlearned (forget) test metrics.

    RS = 1 - |A^un_t_r - A^re_t_r| / |A^un_t_f - A^re_t_f|

    A^un_t_r, A^un_t_f come from the 'unlearned' (forget) file
    A^re_t_r, A^re_t_f come from the 'revival' file
    """
    if df_rev is None or df_rev.empty or df_un is None or df_un.empty:
        if df_rev is None:
            return df_rev
        out = df_rev.copy()
        out["RS1"] = pd.NA
        return out

    # keep only keys + needed metrics from unlearned
    un_keep = ["forget_class","dataset","model","method","test_retain_acc","test_forget_acc"]
    un = df_un[un_keep].copy().rename(columns={
        "test_retain_acc": "test_retain_acc_un",
        "test_forget_acc": "test_forget_acc_un",
    })

    keys = ["forget_class","dataset","model","method"]
    m = df_rev.merge(un, on=keys, how="left")

    # cast to numeric (works whether your accs are 0–1 or 0–100; ratio cancels scale)
    A_re_t_r = pd.to_numeric(m["test_retain_acc"], errors="coerce")
    A_re_t_f = pd.to_numeric(m["test_forget_acc"], errors="coerce")
    A_un_t_r = pd.to_numeric(m["test_retain_acc_un"], errors="coerce")
    A_un_t_f = pd.to_numeric(m["test_forget_acc_un"], errors="coerce")

    num = (A_un_t_r - A_re_t_r).abs()
    den = (A_un_t_f - A_re_t_f).abs()

    # Safe division:
    # - If den==0 and num==0 -> RS = 1 (identical changes)
    # - If den==0 and num>0  -> RS = NaN (undefined)
    # else RS = 1 - num/den
    den_zero = (den == 0) | den.isna()
    rs = np.where(den_zero & (num.fillna(0) == 0), 1.0,
          np.where(den_zero, np.nan, 1 - (num / den)))

    m["RS1"] = rs
    return m

from typing import Optional
import pandas as pd
import numpy as np
def slugify(s: Optional[str]) -> str:
    """Safe tag for filenames/labels: keep letters/numbers/dashes; replace the rest with '_'."""
    if s is None:
        return "unknown"
    return re.sub(r'[^A-Za-z0-9\-]+', "_", str(s))

def compute_rs_for_revival2(df_rev: pd.DataFrame, df_un: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df_rev is None or df_rev.empty:
        return df_rev
    if df_un is None or df_un.empty:
        out = df_rev.copy()
        out["RS2"] = pd.NA
        return out

    keys = ["forget_class", "dataset", "model", "method"]
    need_un = keys + ["test_retain_acc", "test_forget_acc"]
    missing_un = [c for c in need_un if c not in df_un.columns]
    if missing_un:
        raise KeyError(f"Unlearned dataframe missing columns: {missing_un}")

    # If RS1 already added *_un columns, reuse them; otherwise merge safely
    have_un_cols = {"test_retain_acc_un", "test_forget_acc_un"}.issubset(df_rev.columns)
    if have_un_cols:
        out = df_rev.copy()
    else:
        # avoid _x/_y by dropping any stale *_un first, then merging
        out = df_rev.drop(columns=["test_retain_acc_un", "test_forget_acc_un"], errors="ignore")
        un_small = df_un[need_un].rename(columns={
            "test_retain_acc": "test_retain_acc_un",
            "test_forget_acc": "test_forget_acc_un"
        })
        out = out.merge(un_small, on=keys, how="left")

    req_cols = ["test_retain_acc", "test_forget_acc",
                "test_retain_acc_un", "test_forget_acc_un"]
    missing_after = [c for c in req_cols if c not in out.columns]
    if missing_after:
        raise KeyError(f"After merge, missing required columns: {missing_after}")

    for c in req_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    max_val = pd.concat([out[c] for c in req_cols], axis=0).max(skipna=True)
    S = 100.0 if (pd.notna(max_val) and max_val > 1.5) else 1.0

    d_retain = (out["test_retain_acc_un"] - out["test_retain_acc"]).abs() / S
    d_forget = (out["test_forget_acc_un"] - out["test_forget_acc"]).abs() / S
    out["RS2"] = ((1.0 - d_retain) * d_forget).clip(lower=0.0, upper=1.0)
    return out



def parse_filename(path: Path):
    """
    Derive (dataset, model, method, phase) from filename.
    Works with both '..._unlearned_<method>_...' and baseline '..._original_...'.
    """
    name = path.stem
    toks = name.split("_")

    ds  = toks[0] if len(toks) > 0 else None
    mdl = toks[1] if len(toks) > 1 else None

    # Method detection by token membership
    mth = None
    for t in toks:
        if t in KNOWN_METHODS:
            mth = t
            break

    # Fallback to old heuristic (4th token) if still None
    if mth is None and len(toks) > 3 and toks[3] in KNOWN_METHODS:
        mth = toks[3]

    # Phase detection
    if "revival" in name:
        phase = "revival"
    elif "forget" in name:
        phase = "unlearned"
    elif ("original" in toks) or (mth == "original"):
        phase = "original"
    else:
        phase = "unknown"

    return ds, mdl, mth, phase


def find_one(dirpath: Path, patterns: List[str]) -> Optional[Path]:
    """Return the most recently modified match among given glob patterns in a single dir."""
    candidates = []
    for pat in patterns:
        candidates.extend(dirpath.glob(pat))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_one_across(dirs: List[Path], patterns: List[str]) -> Optional[Path]:
    """Search multiple directories; return the most recently modified match."""
    best = None
    best_mtime = -1
    for d in dirs:
        for pat in patterns:
            for p in d.glob(pat):
                mt = p.stat().st_mtime
                if mt > best_mtime:
                    best, best_mtime = p, mt
    return best


def standardize_cols(df: pd.DataFrame, ds: str, mdl: str, mth: str) -> pd.DataFrame:
    """Ensure key meta columns and metric names exist."""
    if "dataset" not in df.columns:
        df["dataset"] = ds
    if "model" not in df.columns:
        df["model"] = mdl
    if "method" not in df.columns:
        df["method"] = mth

    # unify metric names if needed
    rename_map = {
        "train_retain": "train_retain_acc",
        "train_fgt":    "train_forget_acc",
        "test_retain":  "test_retain_acc",
        "test_fgt":     "test_forget_acc",
        "retain_test_acc": "test_retain_acc",
        "forget_test_acc": "test_forget_acc",
    }
    df = df.rename(columns=rename_map)

    # Some files may use names without _acc; fill if so.
    for src, tgt in [
        ("train_retain", "train_retain_acc"),
        ("train_forget", "train_forget_acc"),
        ("test_retain",  "test_retain_acc"),
        ("test_forget",  "test_forget_acc"),
    ]:
        if tgt not in df.columns and src in df.columns:
            df[tgt] = df[src]

    # Ensure the 4 target columns exist
    for col in ["train_retain_acc","train_forget_acc","test_retain_acc","test_forget_acc"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Coerce to numeric
    for col in ["train_retain_acc","train_forget_acc","test_retain_acc","test_forget_acc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _pick(value, hint, what: str, path: Path):
    if value not in (None, "", np.nan):
        return value
    if hint not in (None, "", np.nan):
        return hint
    raise ValueError(f"Cannot infer {what} from filename '{path.name}', and no {what} hint was provided.")

def load_forget_csv(path: Path, ds_hint: Optional[str]=None, mdl_hint: Optional[str]=None, mth_hint: Optional[str]=None) -> pd.DataFrame:
    ds_p, mdl_p, mth_p, _ = parse_filename(path)
    ds  = _pick(ds_p,  ds_hint,  "dataset", path)
    mdl = _pick(mdl_p, mdl_hint, "model",   path)
    mth = mth_p or mth_hint or "unknown"

    d = pd.read_csv(path)
    d = standardize_cols(d, ds, mdl, mth)
    if "forget_class" not in d.columns:
        d["forget_class"] = pd.NA
    keep_cols = ["forget_class","dataset","model","method",
                 "train_retain_acc","train_forget_acc","test_retain_acc","test_forget_acc"]
    return d[keep_cols].copy()

def load_revival_csv(path: Path, ds_hint: Optional[str]=None, mdl_hint: Optional[str]=None, mth_hint: Optional[str]=None) -> pd.DataFrame:
    ds_p, mdl_p, mth_p, _ = parse_filename(path)
    ds  = _pick(ds_p,  ds_hint,  "dataset", path)
    mdl = _pick(mdl_p, mdl_hint, "model",   path)
    mth = mth_p or mth_hint or "unknown"

    d = pd.read_csv(path)
    if "epoch" in d.columns:
        d = d.sort_values(["forget_class","epoch"]).groupby("forget_class", as_index=False).tail(1)
    d = standardize_cols(d, ds, mdl, mth)
    if "forget_class" not in d.columns:
        d["forget_class"] = pd.NA
    keep_cols = ["forget_class","dataset","model","method",
                 "train_retain_acc","train_forget_acc","test_retain_acc","test_forget_acc"]
    return d[keep_cols].copy()

def load_original_csv(path: Path, ds_hint: Optional[str]=None, mdl_hint: Optional[str]=None) -> pd.DataFrame:
    ds_p, mdl_p, _, _ = parse_filename(path)
    ds  = _pick(ds_p,  ds_hint,  "dataset", path)
    mdl = _pick(mdl_p, mdl_hint, "model",   path)

    d = pd.read_csv(path)
    d = standardize_cols(d, ds, mdl, "original")
    if "forget_class" not in d.columns:
        d["forget_class"] = -1
    keep_cols = ["forget_class","dataset","model","method",
                 "train_retain_acc","train_forget_acc","test_retain_acc","test_forget_acc"]
    return d[keep_cols].copy()

def _rank_styles(values):
    """
    Given a list of numeric RS values (may contain NaN), return two thresholds:
    max_val and second_max (next distinct less-than-max). If there's no second
    distinct value, second_max is None.
    """
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

all_rows = []
per_method_info = []

for ds in DATASETS:
    
    for mdl in MODELS:
        for mth in methods:
            df_f = None
            if mth == "original":
                # Search in /original then base_dir for this dataset+model
                original_search_dirs = [base_dir / "original", base_dir]
                original_patterns = [
                    f"results_original_{ds}_{mdl}.csv",
                    f"{ds}_{mdl}_original*_metrics*.csv",
                    f"{ds}_{mdl}_original*.csv",
                ]
                original_path = find_one_across(original_search_dirs, original_patterns)

                if original_path is None:
                    print(f"[WARN] No 'original' CSV for {ds}/{mdl} in {original_search_dirs}")
                    per_method_info.append({"method": "original", "dataset": ds, "model": mdl, "original_file": None})
                else:
                    try:
                        df_o = load_original_csv(original_path, ds_hint=ds, mdl_hint=mdl)
                        df_o["phase"] = "original"
                        # ensure meta columns are correct
                        df_o["dataset"] = ds
                        df_o["model"]   = mdl
                        all_rows.append(df_o)

                        out_original = original_path.parent / f"z_standardized_original_{slugify(ds)}_{slugify(mdl)}.csv"
                        df_o.to_csv(out_original, index=False)

                        per_method_info.append({
                            "method": "original", "dataset": ds, "model": mdl,
                            "original_file": str(original_path)
                        })
                    except Exception as e:
                        print(f"[ERROR] Failed reading original for {ds}/{mdl}: {original_path}\n{e}")

                continue  # next method

            # ---------- non-original methods ----------
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

            forget_path  = find_one(method_dir, forget_patterns)
            revival_path = find_one(method_dir, revival_patterns)

            if forget_path is None and revival_path is None:
                print(f"[WARN] No files for {mth} {ds}/{mdl} in {method_dir}")
                per_method_info.append({
                    "method": mth, "dataset": ds, "model": mdl,
                    "forget_file": None, "revival_file": None,
                })
                continue

            if forget_path is not None:
                try:
                    df_f = load_forget_csv(forget_path,  ds_hint=ds, mdl_hint=mdl, mth_hint=mth)
                    df_f["phase"] = "unlearned"
                    # normalize meta
                    df_f["dataset"] = ds
                    df_f["model"]   = mdl
                    df_f["method"]  = mth
                    df_f["forget_class"] = df_f["forget_class"].astype(str)

                    df_f = _apply_forget_filter(df_f, ds)
                    
                    all_rows.append(df_f)

                    out_forget = method_dir / f"z_standardized_forget_selected_{mth}_{slugify(ds)}_{slugify(mdl)}.csv"
                    df_f.to_csv(out_forget, index=False)
                except Exception as e:
                    print(f"[ERROR] Failed to read forget CSV for {mth} {ds}/{mdl}: {forget_path}\n{e}")

            if revival_path is not None:
                try:
                    df_r = load_revival_csv(revival_path, ds_hint=ds, mdl_hint=mdl, mth_hint=mth)
                    df_r["phase"] = "revival"
                    # normalize meta
                    df_r["dataset"] = ds
                    df_r["model"]   = mdl
                    df_r["method"]  = mth

                    df_r["forget_class"] = df_r["forget_class"].astype(str)

                    if df_f is None or df_f.empty:
                        df_r["RS1"] = pd.NA
                        df_r["RS2"] = pd.NA
                    else:
                        df_r = compute_rs_for_revival1(df_r, df_f)
                        df_r = compute_rs_for_revival2(df_r, df_f)
                        

                    df_r = _apply_forget_filter(df_r, ds)

                    if "RS2" not in df_r.columns:
                        raise RuntimeError("RS2 not found in df_r columns. Columns are: " + ", ".join(df_r.columns))

                    all_rows.append(df_r)

                    out_revival = method_dir / f"z_standardized_revival_selected_{mth}_{slugify(ds)}_{slugify(mdl)}.csv"
                    df_r.to_csv(out_revival, index=False)
                except Exception as e:
                    print(f"[ERROR] Failed to read revival CSV for {mth} {ds}/{mdl}: {revival_path}\n{e}")

            per_method_info.append({
                "method": mth, "dataset": ds, "model": mdl,
                "forget_file":  str(forget_path)  if forget_path  else None,
                "revival_file": str(revival_path) if revival_path else None,
            })


# ------------- Write merged outputs -------------
if all_rows:
    merged = pd.concat(all_rows, ignore_index=True)

    # (A) Global merged (all datasets/models)
    global_merged = base_dir / "z_standardized_selected_all_methods.csv"
    merged.to_csv(global_merged, index=False)

    (merged[merged["phase"] == "unlearned"]
        .to_csv(base_dir / "z_standardized_forget_all_methods.csv", index=False))
    (merged[merged["phase"] == "revival"]
        .to_csv(base_dir / "z_standardized_revival_all_methods.csv", index=False))
    (merged[merged["phase"] == "original"]
        .to_csv(base_dir / "z_standardized_original_all_methods.csv", index=False))

    # (B) Per-(dataset, model) files
    for (ds_i, mdl_i), df_i in merged.groupby(["dataset", "model"], dropna=False):
        ds_tag  = slugify(ds_i)
        mdl_tag = slugify(mdl_i)

        out_merged = base_dir / f"z_standardized_selected_all_methods_{ds_tag}_{mdl_tag}.csv"
        df_i.to_csv(out_merged, index=False)

        for ph in ["unlearned", "revival", "original"]:
            df_ph = df_i[df_i["phase"] == ph]
            if not df_ph.empty:
                out_ph = base_dir / f"z_standardized_{ph}_all_methods_{ds_tag}_{mdl_tag}.csv"
                df_ph.to_csv(out_ph, index=False)

    print("Saved global:", global_merged)
    print(
        merged.groupby(["dataset","model","method","phase"])
              .size()
              .rename("rows")
              .reset_index()
    )
else:
    print("[WARN] No data collected. Check your patterns/paths.")




# ==== Per-model LaTeX tables ====

merged_path = base_dir / "z_standardized_selected_all_methods.csv"
df_all = pd.read_csv(merged_path)

for c in ALL_METRIC_COLS:
    df_all[c] = pd.to_numeric(df_all.get(c), errors="coerce")
    
if "RS2" in df_all.columns:
    df_all["RS2"] = pd.to_numeric(df_all["RS2"], errors="coerce")
    
# Order you prefer; any extra/unknown methods present in the CSV will be appended at the end.
METHOD_ORDER = [
    "original", "retrained", "finetune", 
    "gradient_ascent", "neggrad_plus", "random_label", 
    "boundary_shrink", "boundary_expand",
    "l2ul_adv", "l2ul_imp",
    "fisher", "wood_fisher",
    "scrub", "bad_teacher", "salun", "delete",
]
PHASE_ORDER = ["unlearned", "revival"]  # non-original methods get these two rows

def fmt_mu_sigma(mu, sigma):
    if pd.isna(mu) and pd.isna(sigma):
        return "-"
    mu = float(mu) if not pd.isna(mu) else np.nan
    sigma = 0.0 if pd.isna(sigma) else float(sigma)
    if pd.isna(mu):
        return "-"
    # mean in normal size, ±std smaller (in parentheses)
    return rf"${mu:.2f}\,\text{{\scriptsize\,±\,{sigma:.2f}}}$"



def add_midrules_between_methods(latex_src: str) -> str:
    """
    Post-process the pandas LaTeX table to:
      1) add a double rule at the very top,
      2) add a double rule between header and the first data row,
      3) insert \midrule after each method block (after Revival rows),
         with a double rule after the 'Retrained & Revival' row,
      4) add a double rule right after the 'Original' row.
    Requires \\usepackage{booktabs}.
    """
    lines = latex_src.splitlines()
    out = []

    duplicated_toprule = False
    duplicated_header_sep = False

    for line in lines:
        stripped = line.strip()
        out.append(line)

        # 1) Double rule at very top of the tabular
        if stripped.startswith(r"\toprule") and not duplicated_toprule:
            out.append(r"\toprule")        # If you prefer thin lines, use r"\midrule" instead.
            duplicated_toprule = True
            continue

        # 2) Double rule between header and data (pandas emits the first \midrule here)
        if stripped.startswith(r"\midrule") and not duplicated_header_sep:
            out.append(r"\midrule")
            duplicated_header_sep = True
            continue

        # 3) & 4) Within the body, add separators after specific rows
        if line.rstrip().endswith(r"\\"):

            if re.search(r"^\s*Original\s*&", line):
                out.append(r"\midrule")
                out.append(r"\midrule")
                continue
                        

            if " & Revival &" in line:
                # Double rule for Retrained's revival row
                if "Retrained" in line or re.search(r"\\multirow\{2\}\{\*\}\{Retrained\}", "\n".join(out[-3:])):
                    out.append(r"\midrule"); out.append(r"\midrule")
                else:
                    out.append(r"\midrule")

    return "\n".join(out)


def apply_multirow(table_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse repeated 'Method' cells into a single \multirow spanning its block.
    Assumes rows for each method are already consecutive (as in your builder).
    """
    df = table_df.copy()
    for label, block in df.groupby("Method", sort=False):
        idxs = list(block.index)
        if len(idxs) >= 2:
            df.at[idxs[0], "Method"] = rf"\multirow{{{len(idxs)}}}{{*}}{{{label}}}"
            df.loc[idxs[1:], "Method"] = ""   # blank the following cells
    return df


def fmt_rs(mu, sigma):
    if pd.isna(mu):
        return "-"
    sigma = 0.0 if pd.isna(sigma) else float(sigma)
    return rf"${float(mu):.3f}\,\text{{\scriptsize\,±\,{sigma:.3f}}}$"


def inject_rs2_multicolumn(latex_src: str, n_metric_cols: int) -> str:
    """
    Find lines with '& RS2 &' and replace all metric cells with one
    \multicolumn{n}{c}{<RS2 text>} cell.
    Assumes the RS2 text was placed in the *first* metric column and the
    other metric columns contain '-' or ''.
    """
    out_lines = []
    for line in latex_src.splitlines():
        if "& RS2 &" in line and r"\\" in line:
            # split cells (keep trailing \\)
            body, trail = line.rsplit(r"\\", 1)
            cells = [c.strip() for c in body.split("&")]
            # cells: [Method, Phase, m1, m2, ... mK]
            if len(cells) >= 2 + n_metric_cols:
                method_cell = cells[0]
                phase_cell  = cells[1]
                rs_text     = cells[2]  # first metric cell carries RS text
                new_line = f"{method_cell} & {phase_cell} & " \
                           f"\\multicolumn{{{n_metric_cols}}}{{c}}{{{rs_text}}} \\\\{trail}"
                out_lines.append(new_line)
                continue
        out_lines.append(line)
    return "\n".join(out_lines)


def wrap_with_resizebox(latex_src: str, caption: str, label: str,
                        star: bool = True, width: str = r"\columnwidth") -> str:
    env = "table"
    return (
        f"\\begin{{{env}}}[h]\n"
        f"\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\resizebox{{{width}}}{{!}}{{%\n{latex_src}\n}}\n"
        f"\\end{{{env}}}\n"
    )


method_name_and_ref = {
    # core set you mentioned (accept strings or tuples)
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
    # "fisher": r"Fisher",
    # "wood_fisher": r"WoodFisher",
    "scrub": r"SCRUB \cite{kurmanji2023towards}",
    "bad_teacher" : r"Bad Teacher \cite{chundawat2023can}",
    "salun" : r"Saliency Unlearn \cite{fan2023salun}",
    "delete": r"DELETE \cite{zhou2025decoupled}",
}


def _latex_model_name(mdl: str) -> str:
    mapping = {
        "resnet18": "ResNet-18",
        "vit-s-16": "ViT-S-16",
        "vit-b-16": "ViT-B-16",
        "swin-t":   "Swin-T",
        "vgg16":    "VGGNet-16",
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
    m = re.match(r"^([a-z]+)(\d+)$", key)  # e.g., foo123 -> FOO-123
    if m:
        return f"{m.group(1).upper()}-{m.group(2)}"
    return ds.replace("_", " ").title()

def _normalize_key(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s)
    return s.replace("-", "_").lower().strip()


def _method_label(raw: str) -> str:
    v = method_name_and_ref.get(_normalize_key(raw))
    if v is None:
        return raw  # fallback: leave as-is
    if isinstance(v, tuple):
        return v[0]  # take first element if tuple provided
    return v

def fmt_mu(mu):
    if pd.isna(mu):
        return "-"
    return rf"${float(mu):.2f}$"

def fmt_rs(mu):
    if pd.isna(mu):
        return "-"
    return rf"${float(mu):.3f}$"

def render_table_for(ds: str, mdl: str, df_src: pd.DataFrame):
    df = df_src[(df_src["dataset"] == ds) & (df_src["model"] == mdl)].copy()
    if df.empty:
        print(f"[WARN] No rows for dataset={ds}, model={mdl}")
        return None

    # aggregate ALL metrics; we'll only display OUT_METRIC_COLS
    agg_cols = ALL_METRIC_COLS.copy()
    has_rs = "RS2" in df.columns
    if has_rs:
        agg_cols = agg_cols + ["RS2"]
    g = df.groupby(["method","phase"], dropna=False)[agg_cols].agg(["mean"])
    
    # -------- NEW: collect RS means per method (from revival) and rank them --------
    rs_by_method = {}
    if has_rs:
        for m in df["method"].unique():
            if (m, "revival") in g.index:
                rs_mu = g.loc[(m, "revival"), ("RS2", "mean")]
                rs_by_method[m] = float(rs_mu) if pd.notna(rs_mu) else np.nan
            else:
                rs_by_method[m] = np.nan
    max_v, second_v = _rank_styles(list(rs_by_method.values()))
    # -------------------------------------------------------------------------------

    # method ordering (yours + extras found)
    present = [m for m in METHOD_ORDER if m in df["method"].unique()]
    extras  = sorted(set(df["method"].unique()) - set(present))
    method_list = present + extras

    rows = []
    for m in method_list:
        if m == "original":
            phase = "original"
            row = {"Method": m, "Phase": phase.title()}
            for col in OUT_METRIC_COLS:
                if (m, phase) in g.index:
                    mu = g.loc[(m, phase), (col, "mean")]
                else:
                    mu= np.nan
                row[col] = fmt_mu(mu)
            # RS column: "-" for Original
            row[RS_LABEL] = "-"
            rows.append(row)
        else:
            # --- Unlearned row (RS shown here) ---
            row_un = {"Method": m, "Phase": "Unlearned"}
            if (m, "unlearned") in g.index:
                for col in OUT_METRIC_COLS:
                    mu = g.loc[(m, "unlearned"), (col, "mean")]
                    row_un[col] = fmt_mu(mu)
            else:
                for col in OUT_METRIC_COLS:
                    row_un[col] = "-"

            # -------- CHANGED: style the RS mean using max/second thresholds --------
            if has_rs:
                rs_mu = rs_by_method.get(m, np.nan)
                rs_text_styled = _style_rs(rs_mu, max_v, second_v)
            else:
                rs_text_styled = "-"
            row_un[RS_LABEL] = rf"\multirow{{2}}{{*}}{{{rs_text_styled}}}"
            # ------------------------------------------------------------------------

            rows.append(row_un)

            # --- Revival row (RS cell blank to keep multirow) ---
            row_re = {"Method": m, "Phase": "Revival"}
            if (m, "revival") in g.index:
                for col in OUT_METRIC_COLS:
                    mu = g.loc[(m, "revival"), (col, "mean")]
                    row_re[col] = fmt_mu(mu)
            else:
                for col in OUT_METRIC_COLS:
                    row_re[col] = "-"
            row_re[RS_LABEL] = ""  # continue multirow
            rows.append(row_re)


    table_df = pd.DataFrame(rows, columns=["Method","Phase"] + OUT_METRIC_COLS + [RS_LABEL]) \
                 .rename(columns={c: COL_LABELS[c] for c in OUT_METRIC_COLS})


    table_df["Method"] = table_df["Method"].map(_method_label)

    table_df = apply_multirow(table_df)


    # escape LaTeX in text columns
    for col in ["Method","Phase"]:
        table_df[col] = (table_df[col].astype(str)
                         .str.replace("_", r"\_", regex=False)
                         .str.replace("&", r"\&", regex=False)
                         .str.replace("%", r"\%", regex=False))

    # dynamic column format: Method | Phase | (one 'c' per displayed metric)
    column_format = "c|c|" + ("c" * (len(OUT_METRIC_COLS) + 1))  # +1 for RS
    latex = table_df.to_latex(index=False, escape=False, column_format=column_format)
    #latex = inject_rs2_multicolumn(latex, n_metric_cols=len(OUT_METRIC_COLS))

    latex = add_midrules_between_methods(latex)

    mdl_latex = _latex_model_name(mdl)
    ds_latex  = _latex_dataset_name(ds)

    if SHOW_TRAIN_METRICS:
        cap = (f"Revival results of multi-class unlearning on {ds_latex} for {mdl_latex} ")
    else:
        cap = (f"Revival results of multi-class unlearning on {ds_latex} for {mdl_latex} ")

    lab = f"tab:{slugify(ds_latex)}_{slugify(mdl_latex)}_ALL"
    latex = wrap_with_resizebox(latex, cap, lab, star=True, width=r"\columnwidth")

    safe_mdl = re.sub(r'[^A-Za-z0-9\-]+', "_", mdl)
    safe_ds  = slugify(ds_latex)
    out = base_dir / f"latex_table_{safe_ds}_{safe_mdl}_multi_class.tex"
    with open(out, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"[OK] wrote: {out}")
    return out


# ---- choose the models you want, and the dataset (here: cifar10) ----
   
from itertools import product
for ds, mdl in product(["cifar10", "cifar100", "tiny_imagenet"], MODELS):
    render_table_for(ds, mdl, df_all)
    
    
def add_group_vertical_bars(latex_src: str, datasets: List[str]) -> str:
    """
    Ensure vertical rules continue through the dataset \\multicolumn headers.
    Works even if the header text is wrapped. (Joint table: RS + metrics per dataset)
    """
    out = latex_src
    ncols = 1 + len(OUT_METRIC_COLS)  # RS + metrics  <<< CHANGED
    for i, ds in enumerate(datasets):
        dsl = _latex_dataset_name(ds)
        pat = rf"(\\multicolumn\{{{ncols}\}}\{{)(?:\|?c\|?)(\}}\{{[^}}]*{re.escape(dsl)}[^}}]*\}})"
        if i == 0:
            spec = "c"
        else:
            spec = "|c"
        out = re.sub(pat, rf"\1{spec}\2", out)
    return out


def center_method_phase_headers(latex_src: str, dataset_labels: List[str]) -> str:
    lines = latex_src.splitlines()
    if not lines:
        return latex_src
    h1 = next((i for i, L in enumerate(lines) if r"\multicolumn{" in L), None)
    if h1 is None:
        return latex_src
    h2 = None
    for j in range(h1 + 1, min(h1 + 8, len(lines))):
        Lj = lines[j].strip()
        if "&" in Lj and Lj.endswith(r"\\"):
            h2 = j
            break
    if h2 is None:
        return latex_src

    ncols = 1 + len(OUT_METRIC_COLS)  # RS + metrics
    group_bits = [rf"\multicolumn{{{ncols}}}{{c}}{{\textbf{{{ds}}}}}" for ds in dataset_labels]
    lines[h1] = r"\multirow{2}{*}{Method} & \multirow{2}{*}{Phase} & " + " & ".join(group_bits) + r" \\"

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

    h2_cells, h2_trail = _split_cells(lines[h2])
    while len(h2_cells) < 2:
        h2_cells.append("")
    h2_cells[0] = ""
    h2_cells[1] = ""
    lines[h2] = _join_cells(h2_cells, h2_trail)
    return "\n".join(lines)


    
def render_joint_table_for_model(mdl: str, df_src: pd.DataFrame, datasets: List[str]) -> Optional[Path]:
    df = df_src[df_src["model"] == mdl].copy()
    if df.empty:
        print(f"[WARN] No rows for model={mdl}")
        return None

    df = df[df["dataset"].isin(datasets)].copy()

    if "RS2" in df.columns:
        df["RS2"] = pd.to_numeric(df["RS2"], errors="coerce")

    agg_cols = ALL_METRIC_COLS.copy()
    if "RS2" in df.columns:
        agg_cols.append("RS2")

    g = df.groupby(["dataset", "method", "phase"], dropna=False)[agg_cols].agg(["mean", "std", "min", "max"])

    present = [m for m in METHOD_ORDER if m in df["method"].unique()]
    extras  = sorted(set(df["method"].unique()) - set(present))
    method_list = present + extras

    # ---------- NEW: Pass 1 — collect RS means per dataset & method and rank ----------
    rs_per_ds = { _latex_dataset_name(ds): {} for ds in datasets }
    for ds in datasets:
        dsl = _latex_dataset_name(ds)
        for m in method_list:
            if m == "original":
                rs_per_ds[dsl][m] = np.nan
                continue
            if "RS2" in g.columns.get_level_values(0).unique() and ((ds, m, "revival") in g.index):
                rs_mu = g.loc[(ds, m, "revival"), ("RS2", "mean")]
                rs_per_ds[dsl][m] = float(rs_mu) if pd.notna(rs_mu) else np.nan
            else:
                rs_per_ds[dsl][m] = np.nan

    rs_rank_thresholds = {}
    for ds in datasets:
        dsl = _latex_dataset_name(ds)
        max_v, second_v = _rank_styles(list(rs_per_ds[dsl].values()))
        rs_rank_thresholds[dsl] = (max_v, second_v)
    # -------------------------------------------------------------------------------

    rows = []
    for m in method_list:
        pretty = _method_label(m)

        if m == "original":
            row = {"Method": pretty, "Phase": "Original"}
            for ds in datasets:
                dsl = _latex_dataset_name(ds)
                row[(dsl, RS_LABEL)] = "-"
                if (ds, m, "original") in g.index:
                    for col in OUT_METRIC_COLS:
                        mu = g.loc[(ds, m, "original"), (col, "mean")]
                        row[(dsl, COL_LABELS[col])] = fmt_mu(mu)
                else:
                    for col in OUT_METRIC_COLS:
                        row[(dsl, COL_LABELS[col])] = "-"
            rows.append(row)
            continue

        # -------- Unlearned row (styled RS shown here for each dataset) --------
        row_un = {"Method": rf"\multirow{{2}}{{*}}{{{pretty}}}", "Phase": "Unlearned"}
        for ds in datasets:
            dsl = _latex_dataset_name(ds)

            # -------- CHANGED: style RS using per-dataset thresholds --------
            raw_rs = rs_per_ds[dsl].get(m, np.nan)
            max_v, second_v = rs_rank_thresholds[dsl]
            rs_text_styled = _style_rs(raw_rs, max_v, second_v) if pd.notna(raw_rs) else "-"
            row_un[(dsl, RS_LABEL)] = rf"\multirow{{2}}{{*}}{{{rs_text_styled}}}"
            # ------------------------------------------------------------------

            if (ds, m, "unlearned") in g.index:
                for col in OUT_METRIC_COLS:
                    mu = g.loc[(ds, m, "unlearned"), (col, "mean")]
                    row_un[(dsl, COL_LABELS[col])] = fmt_mu(mu)
            else:
                for col in OUT_METRIC_COLS:
                    row_un[(dsl, COL_LABELS[col])] = "-"

        rows.append(row_un)

        # -------- Revival row (no RS value here; metrics only) --------
        row_rev = {"Method": "", "Phase": "Revival"}
        for ds in datasets:
            dsl = _latex_dataset_name(ds)
            row_rev[(dsl, RS_LABEL)] = ""  # continues multirow
            if (ds, m, "revival") in g.index:
                for col in OUT_METRIC_COLS:
                    mu = g.loc[(ds, m, "revival"), (col, "mean")]
                    row_rev[(dsl, COL_LABELS[col])] = fmt_mu(mu)
            else:
                for col in OUT_METRIC_COLS:
                    row_rev[(dsl, COL_LABELS[col])] = "-"
        rows.append(row_rev)


    # ---------- MultiIndex columns in the order of 'datasets' ----------
    dataset_labels = [_latex_dataset_name(d) for d in datasets]
    ordered_cols = [("","Method"), ("","Phase")]
    for dsl in dataset_labels:
        for c in OUT_METRIC_COLS:
            ordered_cols.append((dsl, COL_LABELS[c]))
        ordered_cols.append((dsl, RS_LABEL))  # <-- use RS_LABEL here


    table_df = pd.DataFrame(rows)
    table_df["Method"] = table_df["Method"].map(_method_label)
    table_df = table_df.rename(columns={"Method": ("","Method"), "Phase": ("","Phase")})
    table_df.columns = pd.MultiIndex.from_tuples(table_df.columns)
    table_df = table_df[ordered_cols]

    for col in [("","Method"), ("","Phase")]:
        table_df[col] = (table_df[col].astype(str)
                         .str.replace("_", r"\_", regex=False)
                         .str.replace("&", r"\&", regex=False)
                         .str.replace("%", r"\%", regex=False))

    cols_per_dataset = 1 + len(OUT_METRIC_COLS)  # RS + metrics
    column_format = "c|c|" + ("{}|".format("c"*cols_per_dataset) * len(datasets)).rstrip("|")

    latex = table_df.to_latex(index=False, escape=False, multicolumn=True,
                              multicolumn_format="c", column_format=column_format)

    latex = center_method_phase_headers(latex, dataset_labels)
    latex = add_group_vertical_bars(latex, datasets)
    latex = add_midrules_between_methods(latex)

    mdl_latex = _latex_model_name(mdl)

    # ---- NEW: dataset suffix for label/filename (e.g., c10_c100) ----
    ds_short = "_".join(["c10" if d=="cifar10" else "c100" if d=="cifar100" else slugify(d) for d in datasets])

    if len(datasets) == 1:
        ds_names = _latex_dataset_name(datasets[0])
    else:
        ds_names = ", ".join(_latex_dataset_name(d) for d in datasets[:-1]) + f", and {_latex_dataset_name(datasets[-1])}"

    caption = (f"Class revival results of multi-class unlearning on {ds_names} for {mdl_latex} ")

    # ---- CHANGED: unique label per dataset selection ----
    label   = f"tab:{slugify(mdl_latex)}_multi_class"

    latex = wrap_with_resizebox(latex, caption, label, star=True, width=r"\columnwidth")

    # ---- CHANGED: unique filename per dataset selection ----
    out = base_dir / f"latex_table_{slugify(mdl)}_multi_class.tex"
    with open(out, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"[OK] wrote: {out}")
    return out

PAIR_DATASETS = ["cifar10", "cifar100"]

for mdl in MODELS:
    render_joint_table_for_model(mdl, df_all, PAIR_DATASETS)

