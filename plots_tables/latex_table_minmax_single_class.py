import pandas as pd
import re
import csv
from io import StringIO
import numpy as np
from pathlib import Path
from typing import Optional, List
from itertools import product

# ----------------- Config -----------------
# Resolve the results directory relative to this repo so the script works on
# Linux/macOS/Windows without editing the path by hand.
base_dir = Path(__file__).resolve().parents[1] / "results_single_class"

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
SHOW_LINEAR_PROBE = False   # set True to include Linear Probe rows in the joint tables

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

# Keep only these forget classes for TinyImageNet
FORGET_CLASS_FILTERS = {
    "cifar100": {0, 10, 20, 30, 40, 50, 60, 70, 80, 90},
    "tiny_imagenet": {0, 20, 40, 60, 80, 100, 120, 140, 160, 180}
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

    #d_retain = (out["test_retain_acc_un"] - out["test_retain_acc"]).abs() / S
    #d_forget = (out["test_forget_acc_un"] - out["test_forget_acc"]).abs() / S
    #out["RS2"] = ((1.0 - d_retain) * d_forget).clip(lower=0.0, upper=1.0)
    
    # normalize to [0,1]
    Ar_un = out["test_retain_acc_un"] / S
    Ar_re = out["test_retain_acc"]    / S
    Af_un = out["test_forget_acc_un"] / S
    Af_re = out["test_forget_acc"]    / S

    # NEW RS2:
    # retain term penalizes only retain drops: max(0, Ar_un - Ar_re)
    retain_drop = (Ar_un - Ar_re).clip(lower=0.0)

    # forget term rewards only forget improvement: max(0, Af_re - Af_un)
    forget_gain = (Af_re - Af_un).clip(lower=0.0)

    #out["RS2"] = (1.0 - retain_drop) * forget_gain
    
    retain_score = 1.0 - retain_drop
    forget_score = forget_gain

    denominator = retain_score + forget_score

    out["RS2"] = np.where(
        denominator > 0,
        2.0 * retain_score * forget_score / denominator,
        0.0
    )

    out["RS2"] = out["RS2"].clip(lower=0.0, upper=1.0)
        
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

def read_csv_keep_well_formed_rows(path: Path) -> pd.DataFrame:
    """
    Read a CSV while tolerating rows with extra trailing fields.

    This is useful for old result files where extra per-class accuracy columns
    were appended to only some rows. If a row has more fields than the header,
    keep the first header-width fields, which preserves the main aggregate
    metrics and forget-class identifier. Rows with too few fields are dropped.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = list(csv.reader(StringIO(text)))
    if not rows:
        return pd.DataFrame()

    header = rows[0]
    width = len(header)
    good_rows = [header]
    truncated_count = 0
    dropped_count = 0
    for row in rows[1:]:
        if len(row) == width:
            good_rows.append(row)
        elif len(row) > width:
            good_rows.append(row[:width])
            truncated_count += 1
        else:
            dropped_count += 1

    if truncated_count or dropped_count:
        print(
            f"[WARN] While reading {path}, truncated {truncated_count} rows "
            f"with extra fields and dropped {dropped_count} short rows; "
            f"expected {width} fields."
        )

    buf = StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerows(good_rows)
    buf.seek(0)
    return pd.read_csv(buf)

def load_forget_csv(path: Path, ds_hint: Optional[str]=None, mdl_hint: Optional[str]=None, mth_hint: Optional[str]=None) -> pd.DataFrame:
    ds_p, mdl_p, mth_p, _ = parse_filename(path)
    ds  = _pick(ds_p,  ds_hint,  "dataset", path)
    mdl = _pick(mdl_p, mdl_hint, "model",   path)
    mth = mth_p or mth_hint or "unknown"

    d = read_csv_keep_well_formed_rows(path)
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

    d = read_csv_keep_well_formed_rows(path)
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

    d = read_csv_keep_well_formed_rows(path)
    d = standardize_cols(d, ds, mdl, "original")
    if "forget_class" not in d.columns:
        d["forget_class"] = -1
    keep_cols = ["forget_class","dataset","model","method",
                 "train_retain_acc","train_forget_acc","test_retain_acc","test_forget_acc"]
    return d[keep_cols].copy()


def load_pra_csv(
    path: Path,
    ds_hint: Optional[str] = None,
    mdl_hint: Optional[str] = None,
    mth_hint: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load PRA results produced by run_pra_on_checkpoint_auto_alpha.py.

    Supports both:
      New format:
        pra_acc_r_test, pra_acc_f_test,
        baseline_acc_r_test, baseline_acc_f_test,
        selected_alpha, support_seed

      Legacy format:
        acc_r, acc_f

    PRA_RS uses the same RS definition as the proposed method:
        (1 - retain_drop) * forget_gain
    with accuracies normalized to [0, 1].
    """
    ds_p, mdl_p, _, _ = parse_filename(path)

    # Prefer explicit hints. This is required for names such as
    # tiny_imagenet_resnet18.csv, which cannot be parsed correctly by
    # simply splitting on underscores.
    ds = ds_hint or ds_p
    mdl = mdl_hint or mdl_p
    if ds is None:
        raise ValueError(f"Cannot infer dataset for PRA file: {path}")
    if mdl is None:
        raise ValueError(f"Cannot infer model for PRA file: {path}")

    mth = mth_hint or "pra_single"
    d = pd.read_csv(path)

    # New auto-alpha output.
    new_format = {
        "pra_acc_r_test",
        "pra_acc_f_test",
    }.issubset(d.columns)

    if new_format:
        d = d.rename(columns={
            "pra_acc_r_test": "test_retain_acc",
            "pra_acc_f_test": "test_forget_acc",
            "baseline_acc_r_test": "test_retain_acc_un",
            "baseline_acc_f_test": "test_forget_acc_un",
        })
    else:
        # Backward compatibility with the older PRA CSV.
        d = d.rename(columns={
            "acc_r": "test_retain_acc",
            "acc_f": "test_forget_acc",
        })

    numeric_cols = [
        "forget_class",
        "support_seed",
        "selected_alpha",
        "test_retain_acc",
        "test_forget_acc",
        "test_retain_acc_un",
        "test_forget_acc_un",
        "selection_retain_drop",
    ]
    for col in numeric_cols:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    if "forget_class" not in d.columns:
        d["forget_class"] = pd.NA
    if "support_seed" not in d.columns:
        d["support_seed"] = 0
    if "selected_alpha" not in d.columns:
        d["selected_alpha"] = pd.NA

    # Appended runs can create duplicates when a command is rerun.
    # Keep the latest row for each forget-class/seed pair.
    dedup_keys = [
        col for col in ["forget_class", "support_seed"]
        if col in d.columns
    ]
    if dedup_keys:
        d = d.drop_duplicates(subset=dedup_keys, keep="last")

    # Compute PRA RS only when the before-PRA accuracies are available.
    required_for_rs = {
        "test_retain_acc",
        "test_forget_acc",
        "test_retain_acc_un",
        "test_forget_acc_un",
    }
    if required_for_rs.issubset(d.columns):
        vals = [
            pd.to_numeric(d[col], errors="coerce")
            for col in required_for_rs
        ]
        max_val = pd.concat(vals, axis=0).max(skipna=True)
        scale = 100.0 if pd.notna(max_val) and max_val > 1.5 else 1.0

        ar_un = d["test_retain_acc_un"] / scale
        ar_pra = d["test_retain_acc"] / scale
        af_un = d["test_forget_acc_un"] / scale
        af_pra = d["test_forget_acc"] / scale

        retain_drop = (ar_un - ar_pra).clip(lower=0.0)
        forget_gain = (af_pra - af_un).clip(lower=0.0)
        #d["PRA_RS2"] = (1.0 - retain_drop) * forget_gain
        
        retain_score = 1.0 - retain_drop
        forget_score = forget_gain

        denominator = retain_score + forget_score

        d["PRA_RS2"] = np.where(
            denominator > 0,
            2.0 * retain_score * forget_score / denominator,
            0.0
        )

        d["PRA_RS2"] = d["PRA_RS2"].clip(lower=0.0, upper=1.0)
        
        
    else:
        d["PRA_RS2"] = pd.NA

    d["dataset"] = ds
    d["model"] = mdl
    d["method"] = mth

    keep_cols = [
        "forget_class",
        "support_seed",
        "selected_alpha",
        "dataset",
        "model",
        "method",
        "test_retain_acc_un",
        "test_forget_acc_un",
        "test_retain_acc",
        "test_forget_acc",
        "PRA_RS2",
    ]
    for col in keep_cols:
        if col not in d.columns:
            d[col] = pd.NA

    return d[keep_cols].copy()


def summarize_pra_results(
    df_pra: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    """
    Average repeated support-seed runs within each forget class first.

    This matches the intended reporting order:
      1) average the repeated PRA trials for each forget class;
      2) aggregate across forget classes for the table.
    """
    if df_pra is None or df_pra.empty:
        return pd.DataFrame()

    d = _apply_forget_filter(df_pra.copy(), dataset)

    for col in [
        "test_retain_acc",
        "test_forget_acc",
        "PRA_RS2",
        "selected_alpha",
    ]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    return (
        d.groupby("forget_class", as_index=False, dropna=False)
         .agg(
             test_retain_acc=("test_retain_acc", "mean"),
             test_forget_acc=("test_forget_acc", "mean"),
             PRA_RS2=("PRA_RS2", "mean"),
             selected_alpha=("selected_alpha", "mean"),
             num_support_runs=("support_seed", "nunique"),
         )
    )


def load_linear_probe_results(path: Path, dataset: str) -> pd.DataFrame:
    """Load single-class linear-probe results and compute the common RS."""
    if not path.is_file():
        return pd.DataFrame()
    d = pd.read_csv(path)
    required = {
        "forget_classes",
        "output_acc_r_test",
        "output_acc_f_test",
        "linear_probe_acc_r_test",
        "linear_probe_acc_f_test",
    }
    if not required.issubset(d.columns):
        raise KeyError(
            f"Linear-probe CSV missing {sorted(required - set(d.columns))}: "
            f"{path}"
        )
    d["forget_class"] = pd.to_numeric(
        d["forget_classes"], errors="coerce"
    )
    d = _apply_forget_filter(d, dataset)
    d = d.drop_duplicates(
        subset=["forget_classes", "seed"],
        keep="last",
    )
    for col in required - {"forget_classes"}:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    ar_un = d["output_acc_r_test"] / 100.0
    ar_lp = d["linear_probe_acc_r_test"] / 100.0
    af_un = d["output_acc_f_test"] / 100.0
    af_lp = d["linear_probe_acc_f_test"] / 100.0
    retain_score = (
        1.0 - (ar_un - ar_lp).clip(lower=0.0)
    ).clip(lower=0.0, upper=1.0)
    forget_score = (af_lp - af_un).clip(lower=0.0, upper=1.0)
    denominator = retain_score + forget_score
    d["LP_RS2"] = np.where(
        denominator > 0,
        2.0 * retain_score * forget_score / denominator,
        0.0,
    )
    return d


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
    return rf"${mu:.2f}{{\scriptstyle\,\pm\,{sigma:.2f}}}$"


def fmt_min_mean_max(vmin, mean, vmax):
    if pd.isna(vmin) and pd.isna(mean) and pd.isna(vmax):
        return "-"
    if pd.isna(vmin) or pd.isna(mean) or pd.isna(vmax):
        return "-"
    return rf"$({float(vmin):.2f},{float(mean):.2f},{float(vmax):.2f})$"


def fmt_forget_series(values: pd.Series) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "-"
    return fmt_min_mean_max(values.min(), values.mean(), values.max())



def add_midrules_between_methods(latex_src: str) -> str:
    """
    Formatting:
      - double line at the top
      - double line below the header
      - double line after Original
      - one line between unlearning-method blocks
      - after the final method: one midrule + pandas bottomrule
    """
    lines = latex_src.splitlines()
    out = []

    duplicated_toprule = False
    duplicated_header_sep = False

    for line in lines:
        stripped = line.strip()
        out.append(line)

        # Double line at the top
        if stripped.startswith(r"\toprule") and not duplicated_toprule:
            out.append(r"\toprule")
            duplicated_toprule = True
            continue

        # Double line below the table header
        if stripped.startswith(r"\midrule") and not duplicated_header_sep:
            out.append(r"\midrule")
            duplicated_header_sep = True
            continue

        if not line.rstrip().endswith(r"\\"):
            continue

        # Double separator after Original
        if re.search(r"^\s*Original\s*&", line):
            out.append(r"\midrule")
            out.append(r"\midrule")
            continue

        # One separator after every method block.
        # For DELETE, pandas adds \bottomrule afterward,
        # so the bottom of the table contains exactly two lines.
        if re.search(r"&\s*Relearned\s*\(ours\)", line):
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
    return rf"${float(mu):.3f}{{\scriptstyle\,\pm\,{sigma:.3f}}}$"

def fmt_rs_max(v):
    return "-" if pd.isna(v) else rf"${float(v):.3f}$"


DELTA_RS_LABEL = r"$\Delta$RS"


def _style_delta_text(delta, delta_max_v=None, delta_second_v=None):
    if pd.isna(delta):
        return r"\,(-)"
    text = f"({float(delta):+.3f})"
    if (delta_max_v is not None) and (abs(delta - delta_max_v) < 1e-12):
        return rf"\,\mathbf{{({float(delta):+.3f})}}"
    if (delta_second_v is not None) and (abs(delta - delta_second_v) < 1e-12):
        return rf"\,\underline{{({float(delta):+.3f})}}"
    return rf"\,{text}"


def _style_rs_delta(v, delta, max_v, second_v, delta_max_v=None, delta_second_v=None):
    if pd.isna(v):
        return "-"
    val = f"{float(v):.3f}"
    delta_text = _style_delta_text(delta, delta_max_v, delta_second_v)
    if (max_v is not None) and (abs(v - max_v) < 1e-12):
        return rf"\textbf{{\boldmath ${val}{delta_text}$}}"
    if (second_v is not None) and (abs(v - second_v) < 1e-12):
        return rf"$\underline{{{val}}}{delta_text}$"
    return rf"${val}{delta_text}$"


def _style_rs_only(v, max_v, second_v):
    if pd.isna(v):
        return "-"
    val = f"{float(v):.3f}"
    if (max_v is not None) and (abs(v - max_v) < 1e-12):
        return rf"\textbf{{\boldmath ${val}$}}"
    if (second_v is not None) and (abs(v - second_v) < 1e-12):
        return rf"$\underline{{{val}}}$"
    return rf"${val}$"


def _style_delta_only(delta, delta_max_v, delta_second_v):
    if pd.isna(delta):
        return "-"
    val = f"{float(delta):+.3f}"
    if (delta_max_v is not None) and (abs(delta - delta_max_v) < 1e-12):
        return rf"$\mathbf{{{val}}}$"
    if (delta_second_v is not None) and (abs(delta - delta_second_v) < 1e-12):
        return rf"$\underline{{{val}}}$"
    return rf"${val}$"


def _forget_key(value) -> str:
    """Stable key for matching a forget class across source-free/PRA tables."""
    if pd.isna(value):
        return ""
    try:
        as_float = float(value)
        if as_float.is_integer():
            return str(int(as_float))
    except (TypeError, ValueError):
        pass
    return str(value)


def _retrained_rs_by_forget(df: pd.DataFrame, dataset: str) -> dict:
    sub = df[
        (df["dataset"] == dataset)
        & (df["method"] == "retrained")
        & (df["phase"] == "revival")
    ].copy()
    if sub.empty or "forget_class" not in sub.columns or "RS2" not in sub.columns:
        return {}
    sub["RS2"] = pd.to_numeric(sub["RS2"], errors="coerce")
    sub["forget_key"] = sub["forget_class"].map(_forget_key)
    grouped = sub.groupby("forget_key")["RS2"].mean()
    return {
        str(forget_key): float(value)
        for forget_key, value in grouped.items()
        if pd.notna(value)
    }


def _rs_by_forget(frame: pd.DataFrame, rs_col: str) -> dict:
    if frame is None or frame.empty or rs_col not in frame.columns:
        return {}
    tmp = frame.copy()
    tmp[rs_col] = pd.to_numeric(tmp[rs_col], errors="coerce")
    if "forget_class" not in tmp.columns:
        return {}
    tmp["forget_key"] = tmp["forget_class"].map(_forget_key)
    grouped = tmp.groupby("forget_key")[rs_col].mean()
    return {
        str(forget_key): float(value)
        for forget_key, value in grouped.items()
        if pd.notna(value)
    }


def _max_rs_and_delta(
    frame: pd.DataFrame,
    rs_col: str,
    retrained_by_forget: dict,
    select_by: str = "rs",
) -> tuple[float, float]:
    if frame is None or frame.empty or rs_col not in frame.columns:
        return np.nan, np.nan
    tmp = frame.copy()
    tmp[rs_col] = pd.to_numeric(tmp[rs_col], errors="coerce")
    tmp = tmp[tmp[rs_col].notna()]
    if tmp.empty:
        return np.nan, np.nan

    # Compute the matched delta for each forget class first:
    #   DeltaRS(c) = RS(c) - RS_retrained(c)
    # This avoids the invalid operation max_c RS(c) - max_c RS_retrained(c).
    if "forget_class" in tmp.columns:
        tmp["_forget_key"] = tmp["forget_class"].map(_forget_key)
    else:
        tmp["_forget_key"] = ""
    tmp["_matched_retrained_rs"] = tmp["_forget_key"].map(retrained_by_forget)
    tmp["_matched_delta_rs"] = tmp[rs_col] - tmp["_matched_retrained_rs"]

    if select_by == "delta" and tmp["_matched_delta_rs"].notna().any():
        idx = tmp["_matched_delta_rs"].idxmax()
    else:
        idx = tmp[rs_col].idxmax()

    rs = float(tmp.loc[idx, rs_col])
    delta = tmp.loc[idx, "_matched_delta_rs"]
    delta = float(delta) if pd.notna(delta) else np.nan
    return rs, delta


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
                        star: bool = True, width: str = r"\textwidth") -> str:
    env = "table*" if star else "table"
    return (
        f"\\begin{{{env}}}[t]\n"
        f"\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\vspace{{-0.3cm}}\n"
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
    "salun" : r"SalUn \cite{fan2023salun}",
    "delete": r"DELETE \cite{zhou2025decoupled}",
}


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
    m = re.match(r"^([a-z]+)(\d+)$", key)  # e.g., foo123 -> FOO-123
    if m:
        return f"{m.group(1).upper()}-{m.group(2)}"
    return ds.replace("_", " ").title()

def _normalize_key(s: str) -> str:
    return s.replace("-", "_").lower().strip()

def _method_label(raw: str) -> str:
    v = method_name_and_ref.get(_normalize_key(raw))
    if v is None:
        return raw  # fallback: leave as-is
    if isinstance(v, tuple):
        return v[0]  # take first element if tuple provided
    return v

def fmt_min_max(vmin, vmax):
    if pd.isna(vmin) and pd.isna(vmax):
        return "-"
    vmin = float(vmin) if not pd.isna(vmin) else np.nan
    vmax = float(vmax) if not pd.isna(vmax) else np.nan
    if pd.isna(vmin) or pd.isna(vmax):
        return "-"
    # prints as (min,max)
    return rf"$({vmin:.2f},{vmax:.2f})$"

def render_table_for(ds: str, mdl: str, df_src: pd.DataFrame):
    df = df_src[(df_src["dataset"] == ds) & (df_src["model"] == mdl)].copy()
    if df.empty:
        print(f"[WARN] No rows for dataset={ds}, model={mdl}")
        return None

    # aggregate ALL metrics; we'll only display OUT_METRIC_COLS
    g = df.groupby(["method","phase"], dropna=False)[ALL_METRIC_COLS].agg(["mean","std","min","max"])

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
                    sd = g.loc[(m, phase), (col, "std")]
                else:
                    mu, sd = np.nan, np.nan
                row[col] = fmt_mu_sigma(mu, sd)
            rows.append(row)
            # Special Baseline row for ResNet-18: show average retain (mean±std)
            # and forget accuracy as a (min,max) range, similar to revival rows.
            if mdl == "resnet18":
                baseline_phase = "unlearned"
                if ("retrained", baseline_phase) in g.index:
                    # compute mean±std for retain, min/max for forget
                    mu_ret = g.loc[("retrained", baseline_phase), ("test_retain_acc", "mean")]
                    sd_ret = g.loc[("retrained", baseline_phase), ("test_retain_acc", "std")]
                    min_fgt = g.loc[("retrained", baseline_phase), ("test_forget_acc", "min")]
                    max_fgt = g.loc[("retrained", baseline_phase), ("test_forget_acc", "max")]
                    baseline_row = {"Method": "Baseline", "Phase": "Baseline"}
                    baseline_row["test_retain_acc"] = fmt_mu_sigma(mu_ret, sd_ret)
                    baseline_row["test_forget_acc"] = fmt_min_max(min_fgt, max_fgt)
                    rows.append(baseline_row)
        else:
            # -----------------------------------------------------
            # 1) Unlearned model
            # -----------------------------------------------------
            phase = "unlearned"
            row = {"Method": m, "Phase": "Unlearned"}
            if (m, phase) in g.index:
                for col in OUT_METRIC_COLS:
                    mu = g.loc[(m, phase), (col, "mean")]
                    sd = g.loc[(m, phase), (col, "std")]
                    row[col] = fmt_mu_sigma(mu, sd)
            else:
                for col in OUT_METRIC_COLS:
                    row[col] = "-"
            rows.append(row)

            # -----------------------------------------------------
            # 2) PRA baseline, when available
            # -----------------------------------------------------
            pra_path = (
                base_dir.parent
                / "pra_single"
                / m
                / f"{ds}_{mdl}.csv"
            )
            if pra_path.is_file():
                try:
                    df_pra = load_pra_csv(
                        pra_path,
                        ds_hint=ds,
                        mdl_hint=mdl,
                        mth_hint=m,
                    )
                    pra_summary = summarize_pra_results(
                        df_pra,
                        dataset=ds,
                    )

                    if not pra_summary.empty:
                        pra_row = {
                            "Method": m,
                            "Phase": r"PRA \cite{ha2025unlearning}",
                        }

                        for col in OUT_METRIC_COLS:
                            # PRA CSV contains test metrics only.
                            if col == "test_retain_acc":
                                values = pd.to_numeric(
                                    pra_summary["test_retain_acc"],
                                    errors="coerce",
                                )
                                pra_row[col] = fmt_mu_sigma(
                                    values.mean(),
                                    values.std(),
                                )
                            elif col == "test_forget_acc":
                                values = pd.to_numeric(
                                    pra_summary["test_forget_acc"],
                                    errors="coerce",
                                )
                                pra_row[col] = fmt_min_max(
                                    values.min(),
                                    values.max(),
                                )
                            else:
                                pra_row[col] = "-"

                        rows.append(pra_row)

                except Exception as exc:
                    print(
                        f"[WARN] Failed to add PRA row for "
                        f"{m} {ds}/{mdl}: {exc}"
                    )

            # -----------------------------------------------------
            # 3) Our source-free revival
            # -----------------------------------------------------
            phase = "revival"
            row = {"Method": m, "Phase": "Relearned (ours)"}
            if (m, phase) in g.index:
                for col in OUT_METRIC_COLS:
                    if col in (
                        "train_forget_acc",
                        "test_forget_acc",
                    ):
                        vmin = g.loc[(m, phase), (col, "min")]
                        vmax = g.loc[(m, phase), (col, "max")]
                        row[col] = fmt_min_max(vmin, vmax)
                    else:
                        mu = g.loc[(m, phase), (col, "mean")]
                        sd = g.loc[(m, phase), (col, "std")]
                        row[col] = fmt_mu_sigma(mu, sd)
            else:
                for col in OUT_METRIC_COLS:
                    row[col] = "-"
            rows.append(row)

    table_df = pd.DataFrame(rows, columns=["Method","Phase"] + OUT_METRIC_COLS) \
                 .rename(columns={c: COL_LABELS[c] for c in OUT_METRIC_COLS})

    table_df["Method"] = table_df["Method"].map(_method_label)

    table_df = table_df.rename(columns={"Phase": "Model state"})


    table_df = apply_multirow(table_df)


    # escape LaTeX in text columns
    for col in ["Method","Model state"]:
        table_df[col] = (table_df[col].astype(str)
                         .str.replace("_", r"\_", regex=False)
                         .str.replace("&", r"\&", regex=False)
                         .str.replace("%", r"\%", regex=False))

    # dynamic column format: Method | Phase | (one 'c' per displayed metric)
    column_format = "c|c|" + ("c" * len(OUT_METRIC_COLS))
    latex = table_df.to_latex(index=False, escape=False, column_format=column_format)
    latex = add_midrules_between_methods(latex)

    mdl_latex = _latex_model_name(mdl)
    ds_latex  = _latex_dataset_name(ds)

    if SHOW_TRAIN_METRICS:
        cap = (
            f"{ds_latex} / {mdl_latex} — Comparison of PRA and "
            f"the proposed source-free relearning audit. PRA uses five real "
            f"forget-class samples and retain-constrained $\\alpha$ selection. "
            f"Retain accuracy is mean$\\pm$std and relearning "
            f"forget accuracy is reported as $({{\\min}},{{\\max}})$."
        )
    else:
        cap = (
            f"{ds_latex} / {mdl_latex} — Comparison of PRA and "
            f"the proposed source-free relearning audit. PRA uses five real "
            f"forget-class samples and retain-constrained $\\alpha$ selection. "
            f"Retain accuracy is mean$\\pm$std and relearning "
            f"forget accuracy is reported as $({{\\min}},{{\\max}})$."
        )

    lab = f"tab:{slugify(ds_latex)}_{slugify(mdl_latex)}_ALL"
    latex = wrap_with_resizebox(latex, cap, lab, star=True, width=r"\textwidth")

    safe_mdl = re.sub(r'[^A-Za-z0-9\-]+', "_", mdl)
    safe_ds  = slugify(ds_latex)
    out = base_dir / f"latex_table_{safe_ds}_{safe_mdl}.tex"
    with open(out, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"[OK] wrote: {out}")
    return out


# ---- choose the models you want, and the dataset (here: cifar10) ----
   
for ds, mdl in product(["cifar10", "cifar100", "tiny_imagenet"], MODELS):
    render_table_for(ds, mdl, df_all)
    
    
def add_group_vertical_bars(latex_src: str, datasets: List[str]) -> str:
    """
    Ensure vertical rules continue through the dataset \\multicolumn headers.
    Works even if the header text is wrapped. (Joint table: RS + metrics per dataset)
    """
    out = latex_src
    ncols = 2 + len(OUT_METRIC_COLS)
    for i, ds in enumerate(datasets):
        dsl = _latex_dataset_name(ds)
        pat = rf"(\\multicolumn\{{{ncols}\}}\{{)(?:\|?c\|?)(\}}\{{[^}}]*{re.escape(dsl)}[^}}]*\}})"
        if i == 0:
            spec = "c"
        elif i == len(datasets) - 1:
            spec = "c"
        else:
            spec = "|c|"
        out = re.sub(pat, rf"\1{spec}\2", out)
    return out


def center_method_phase_headers(latex_src: str, dataset_labels: List[str]) -> str:
    r"""
    Rebuild header row 1 to use dynamic ncols per dataset:
    RS + displayed metrics for each dataset group.
    Then blank the Method/Phase cells on header row 2.
    """
    lines = latex_src.splitlines()
    if not lines:
        return latex_src

    # find first header line with \multicolumn{
    h1 = next((i for i, L in enumerate(lines) if r"\multicolumn{" in L), None)
    if h1 is None:
        return latex_src

    # header2 is the next non-empty line with '& ... \\'
    h2 = None
    for j in range(h1 + 1, min(h1 + 8, len(lines))):
        Lj = lines[j].strip()
        if "&" in Lj and Lj.endswith(r"\\"):
            h2 = j
            break
    if h2 is None:
        return latex_src

    ncols = 2 + len(OUT_METRIC_COLS)  # metrics + RS + DeltaRS
    group_bits = [rf"\multicolumn{{{ncols}}}{{c}}{{\textbf{{{ds}}}}}" for ds in dataset_labels]
    new_h1 = (
        r"\multirow{2}{*}{Unlearning Method} & \multirow{2}{*}{Phase} & "
        + " & ".join(group_bits)
        + r" \\"
    )
    lines[h1] = new_h1

    # blank "Method & Phase" under the multirow on header row 2
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
    h2_cells[0] = ""   # under the multirow
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

    dataset_labels = [_latex_dataset_name(d) for d in datasets]

    # ---------- PASS 1: collect displayed RS values per dataset ----------
    # Rank all displayed RS values together (PRA, optional Linear Probe, and ours).
    rs_per_ds = {dsl: {} for dsl in dataset_labels}   # {dsl: {method -> rs_value or NaN}}
    rs_delta_per_ds = {dsl: {} for dsl in dataset_labels}
    pra_rs_per_ds = {dsl: {} for dsl in dataset_labels}
    pra_delta_per_ds = {dsl: {} for dsl in dataset_labels}
    lp_rs_per_ds = {dsl: {} for dsl in dataset_labels}
    lp_delta_per_ds = {dsl: {} for dsl in dataset_labels}
    retrained_rs_per_ds = {
        _latex_dataset_name(ds): _retrained_rs_by_forget(df, ds)
        for ds in datasets
    }

    pra_summary_cache = {}

    def get_pra_summary(method: str, dataset: str) -> pd.DataFrame:
        key = (method, dataset)
        if key in pra_summary_cache:
            return pra_summary_cache[key]
        pra_path = (
            base_dir.parent / "pra_single" / method
            / f"{dataset}_{mdl}.csv"
        )
        if not pra_path.is_file():
            pra_summary_cache[key] = pd.DataFrame()
            return pra_summary_cache[key]
        df_pra = load_pra_csv(
            pra_path,
            ds_hint=dataset,
            mdl_hint=mdl,
            mth_hint=method,
        )
        pra_summary_cache[key] = summarize_pra_results(
            df_pra,
            dataset=dataset,
        )
        return pra_summary_cache[key]

    pra_retrained_rs_per_ds = {
        _latex_dataset_name(ds): _rs_by_forget(
            get_pra_summary("retrained", ds),
            "PRA_RS2",
        )
        for ds in datasets
    }

    lp_retrained_rs_per_ds = {dsl: {} for dsl in dataset_labels}
    if SHOW_LINEAR_PROBE:
        for ds in datasets:
            dsl = _latex_dataset_name(ds)
            lp_path = (
                base_dir.parent / "linear_probe_single" / "retrained"
                / f"{ds}_{mdl}.csv"
            )
            if lp_path.is_file():
                try:
                    lp_retrained_rs_per_ds[dsl] = _rs_by_forget(
                        load_linear_probe_results(lp_path, ds),
                        "LP_RS2",
                    )
                except Exception:
                    lp_retrained_rs_per_ds[dsl] = {}

    for m in method_list:
        if m == "original":
            continue
        for ds in datasets:
            dsl = _latex_dataset_name(ds)
            sourcefree_rows = df[
                (df["dataset"] == ds)
                & (df["method"] == m)
                & (df["phase"] == "revival")
            ]
            rs_value, rs_delta = _max_rs_and_delta(
                sourcefree_rows,
                "RS2",
                retrained_rs_per_ds[dsl],
                select_by="rs",
            )
            if m == "retrained":
                rs_delta = np.nan
            rs_per_ds[dsl][m] = rs_value
            rs_delta_per_ds[dsl][m] = rs_delta

            pra_rs_per_ds[dsl][m] = np.nan
            pra_delta_per_ds[dsl][m] = np.nan
            try:
                pra_summary = get_pra_summary(m, ds)
                if not pra_summary.empty:
                    pra_rs = pd.to_numeric(
                        pra_summary["PRA_RS2"],
                        errors="coerce",
                    )
                    if pra_rs.notna().any():
                        pra_value, pra_delta = _max_rs_and_delta(
                            pra_summary,
                            "PRA_RS2",
                            pra_retrained_rs_per_ds[dsl],
                            select_by="rs",
                        )
                        if m == "retrained":
                            pra_delta = np.nan
                        pra_rs_per_ds[dsl][m] = pra_value
                        pra_delta_per_ds[dsl][m] = pra_delta
            except Exception:
                pra_rs_per_ds[dsl][m] = np.nan
                pra_delta_per_ds[dsl][m] = np.nan

            lp_rs_per_ds[dsl][m] = np.nan
            lp_delta_per_ds[dsl][m] = np.nan
            if SHOW_LINEAR_PROBE:
                lp_path = (
                    base_dir.parent / "linear_probe_single" / m
                    / f"{ds}_{mdl}.csv"
                )
                if lp_path.is_file():
                    try:
                        df_lp = load_linear_probe_results(lp_path, ds)
                        if not df_lp.empty:
                            lp_rs = pd.to_numeric(
                                df_lp["LP_RS2"],
                                errors="coerce",
                            )
                            if lp_rs.notna().any():
                                lp_value, lp_delta = _max_rs_and_delta(
                                    df_lp,
                                    "LP_RS2",
                                    lp_retrained_rs_per_ds[dsl],
                                )
                                if m == "retrained":
                                    lp_delta = np.nan
                                lp_rs_per_ds[dsl][m] = lp_value
                                lp_delta_per_ds[dsl][m] = lp_delta
                    except Exception:
                        lp_rs_per_ds[dsl][m] = np.nan
                        lp_delta_per_ds[dsl][m] = np.nan

    # Compute (max, second max) thresholds per dataset (distinct values)
    rs_rank_thresholds = {}
    delta_rank_thresholds = {}
    for dsl in dataset_labels:
        rs_vals = (
            list(rs_per_ds[dsl].values())
            + list(pra_rs_per_ds[dsl].values())
            + list(lp_rs_per_ds[dsl].values())
        )
        max_v, second_v = _rank_styles(rs_vals)
        rs_rank_thresholds[dsl] = (max_v, second_v)
        delta_vals = (
            list(rs_delta_per_ds[dsl].values())
            + list(pra_delta_per_ds[dsl].values())
            + list(lp_delta_per_ds[dsl].values())
        )
        delta_rank_thresholds[dsl] = _rank_styles(delta_vals)

    # ---------- PASS 2: build rows with styled RS ----------
    rows = []
    for m in method_list:
        pretty = _method_label(m)

        if m == "original":
            row = {"Method": pretty, "Phase": "Original"}
            for ds in datasets:
                dsl = _latex_dataset_name(ds)
                row[(dsl, "RS")] = "-"
                row[(dsl, DELTA_RS_LABEL)] = "-"
                if (ds, m, "original") in g.index:
                    for col in OUT_METRIC_COLS:
                        if col == "test_forget_acc":
                            vmin = g.loc[(ds, m, "original"), (col, "min")]
                            mu = g.loc[(ds, m, "original"), (col, "mean")]
                            vmax = g.loc[(ds, m, "original"), (col, "max")]
                            row[(dsl, COL_LABELS[col])] = fmt_min_mean_max(vmin, mu, vmax)
                        else:
                            mu = g.loc[(ds, m, "original"), (col, "mean")]
                            sd = g.loc[(ds, m, "original"), (col, "std")]
                            row[(dsl, COL_LABELS[col])] = fmt_mu_sigma(mu, sd)
                else:
                    for col in OUT_METRIC_COLS:
                        row[(dsl, COL_LABELS[col])] = "-"
            rows.append(row)
            continue

        # ---------------------------------------------------------
        # Build the PRA row first so it can be displayed before ours.
        # ---------------------------------------------------------
        pra_row = {"Method": "", "Phase": r"PRA \cite{ha2025unlearning}"}
        has_pra_data = False
        lp_row = {
            "Method": "",
            "Phase": r"Linear Probe \cite{gao2026illusion}",
        }
        has_lp_data = False

        for ds in datasets:
            dsl = _latex_dataset_name(ds)
            pra_path = (
                base_dir.parent / "pra_single" / m
                / f"{ds}_{mdl}.csv"
            )
            lp_path = (
                base_dir.parent / "linear_probe_single" / m
                / f"{ds}_{mdl}.csv"
            )

            # Initialize all PRA cells for this dataset.
            for col in OUT_METRIC_COLS:
                pra_row[(dsl, COL_LABELS[col])] = "-"
            pra_row[(dsl, "RS")] = "-"
            pra_row[(dsl, DELTA_RS_LABEL)] = "-"
            for col in OUT_METRIC_COLS:
                lp_row[(dsl, COL_LABELS[col])] = "-"
            lp_row[(dsl, "RS")] = "-"
            lp_row[(dsl, DELTA_RS_LABEL)] = "-"

            if SHOW_LINEAR_PROBE and lp_path.is_file():
                try:
                    df_lp = load_linear_probe_results(lp_path, ds)
                    if not df_lp.empty:
                        acc_r = df_lp["linear_probe_acc_r_test"]
                        acc_f = df_lp["linear_probe_acc_f_test"]
                        lp_rs = df_lp["LP_RS2"]
                        lp_row[
                            (dsl, COL_LABELS["test_retain_acc"])
                        ] = fmt_mu_sigma(acc_r.mean(), acc_r.std())
                        lp_row[
                            (dsl, COL_LABELS["test_forget_acc"])
                        ] = fmt_forget_series(acc_f)
                        lp_raw_rs = lp_rs_per_ds[dsl].get(m, np.nan)
                        lp_delta = lp_delta_per_ds[dsl].get(m, np.nan)
                        max_v, second_v = rs_rank_thresholds[dsl]
                        delta_max_v, delta_second_v = delta_rank_thresholds[dsl]
                        lp_row[(dsl, "RS")] = _style_rs_only(
                            lp_raw_rs, max_v, second_v
                        )
                        lp_row[(dsl, DELTA_RS_LABEL)] = _style_delta_only(
                            lp_delta, delta_max_v, delta_second_v
                        )
                        has_lp_data = True
                except Exception as e:
                    print(
                        f"[WARN] Failed to load linear-probe results for "
                        f"method={m}, dataset={ds}, model={mdl}: {e}"
                    )

            if pra_path.is_file():
                try:
                    df_pra = load_pra_csv(
                        pra_path,
                        ds_hint=ds,
                        mdl_hint=mdl,
                        mth_hint=m,
                    )

                    pra_summary = summarize_pra_results(
                        df_pra,
                        dataset=ds,
                    )

                    if not pra_summary.empty:
                        acc_r = pd.to_numeric(
                            pra_summary["test_retain_acc"],
                            errors="coerce",
                        )
                        acc_f = pd.to_numeric(
                            pra_summary["test_forget_acc"],
                            errors="coerce",
                        )
                        pra_rs = pd.to_numeric(
                            pra_summary["PRA_RS2"],
                            errors="coerce",
                        )

                        # Match the table convention:
                        # retain = mean±std across forget classes;
                        # forget = min/max across forget classes;
                        # RS = maximum per-class RS.
                        pra_row[
                            (dsl, COL_LABELS["test_retain_acc"])
                        ] = fmt_mu_sigma(acc_r.mean(), acc_r.std())

                        pra_row[
                            (dsl, COL_LABELS["test_forget_acc"])
                        ] = fmt_forget_series(acc_f)

                        pra_raw_rs = pra_rs_per_ds[dsl].get(m, np.nan)
                        pra_delta = pra_delta_per_ds[dsl].get(m, np.nan)
                        max_v, second_v = rs_rank_thresholds[dsl]
                        delta_max_v, delta_second_v = delta_rank_thresholds[dsl]
                        pra_row[(dsl, "RS")] = _style_rs_only(
                            pra_raw_rs, max_v, second_v
                        )
                        pra_row[(dsl, DELTA_RS_LABEL)] = _style_delta_only(
                            pra_delta, delta_max_v, delta_second_v
                        )

                        has_pra_data = True

                except Exception as e:
                    print(
                        f"[WARN] Failed to load PRA results for "
                        f"method={m}, dataset={ds}, model={mdl}: {e}"
                    )

        method_row_span = 2 + int(has_lp_data) + int(has_pra_data)

        # ---------------------------------------------------------
        # 1) Unlearned model
        # ---------------------------------------------------------
        row_un = {
            "Method": rf"\multirow{{{method_row_span}}}{{*}}{{{pretty}}}",
            "Phase": "Unlearned",
        }

        for ds in datasets:
            dsl = _latex_dataset_name(ds)

            # RS is shown on the Revival (ours) row only.
            row_un[(dsl, "RS")] = ""
            row_un[(dsl, DELTA_RS_LABEL)] = ""

            if (ds, m, "unlearned") in g.index:
                for col in OUT_METRIC_COLS:
                    if col == "test_forget_acc":
                        vmin = g.loc[(ds, m, "unlearned"), (col, "min")]
                        mu = g.loc[(ds, m, "unlearned"), (col, "mean")]
                        vmax = g.loc[(ds, m, "unlearned"), (col, "max")]
                        row_un[(dsl, COL_LABELS[col])] = fmt_min_mean_max(vmin, mu, vmax)
                    else:
                        mu = g.loc[(ds, m, "unlearned"), (col, "mean")]
                        sd = g.loc[(ds, m, "unlearned"), (col, "std")]
                        row_un[(dsl, COL_LABELS[col])] = fmt_mu_sigma(mu, sd)
            else:
                for col in OUT_METRIC_COLS:
                    row_un[(dsl, COL_LABELS[col])] = "-"

        rows.append(row_un)

        # 2) Source-dependent linear probe
        if has_lp_data:
            rows.append(lp_row)

        # ---------------------------------------------------------
        # 3) PRA baseline attack
        # ---------------------------------------------------------
        if has_pra_data:
            rows.append(pra_row)

        # ---------------------------------------------------------
        # 4) Our relearning method
        # ---------------------------------------------------------
        row_rev = {"Method": "", "Phase": "Relearned (ours)"}

        for ds in datasets:
            dsl = _latex_dataset_name(ds)
            raw_rs = rs_per_ds[dsl].get(m, np.nan)
            delta_rs = rs_delta_per_ds[dsl].get(m, np.nan)
            max_v, second_v = rs_rank_thresholds[dsl]
            delta_max_v, delta_second_v = delta_rank_thresholds[dsl]

            if pd.notna(raw_rs):
                row_rev[(dsl, "RS")] = _style_rs_only(
                    raw_rs, max_v, second_v
                )
                row_rev[(dsl, DELTA_RS_LABEL)] = _style_delta_only(
                    delta_rs, delta_max_v, delta_second_v
                )
            else:
                row_rev[(dsl, "RS")] = "-"
                row_rev[(dsl, DELTA_RS_LABEL)] = "-"

            if (ds, m, "revival") in g.index:
                for col in OUT_METRIC_COLS:
                    if col == "test_forget_acc":
                        vmin = g.loc[(ds, m, "revival"), (col, "min")]
                        mu = g.loc[(ds, m, "revival"), (col, "mean")]
                        vmax = g.loc[(ds, m, "revival"), (col, "max")]
                        row_rev[(dsl, COL_LABELS[col])] = fmt_min_mean_max(vmin, mu, vmax)
                    else:
                        mu = g.loc[(ds, m, "revival"), (col, "mean")]
                        sd = g.loc[(ds, m, "revival"), (col, "std")]
                        row_rev[(dsl, COL_LABELS[col])] = fmt_mu_sigma(
                            mu, sd
                        )
            else:
                for col in OUT_METRIC_COLS:
                    row_rev[(dsl, COL_LABELS[col])] = "-"

        rows.append(row_rev)

    # ---------- build final DataFrame with MultiIndex columns ----------
    ordered_cols = [("","Unlearning Method"), ("","Phase")]
    for dsl in dataset_labels:
        for c in OUT_METRIC_COLS:
            ordered_cols.append((dsl, COL_LABELS[c]))
        ordered_cols.append((dsl, "RS"))
        ordered_cols.append((dsl, DELTA_RS_LABEL))

    table_df = pd.DataFrame(rows)
    table_df["Method"] = table_df["Method"].map(_method_label)

    table_df = table_df.rename(columns={"Method": ("","Unlearning Method"), "Phase": ("","Phase")})
    table_df.columns = pd.MultiIndex.from_tuples(table_df.columns)
    table_df = table_df[ordered_cols]

    for col in [("","Unlearning Method"), ("","Phase")]:
        table_df[col] = (table_df[col].astype(str)
                         .str.replace("_", r"\_", regex=False)
                         .str.replace("&", r"\&", regex=False)
                         .str.replace("%", r"\%", regex=False))

    cols_per_dataset = 2 + len(OUT_METRIC_COLS)  # metrics + RS + DeltaRS
    block_format = ("c" * len(OUT_METRIC_COLS)) + "cc"
    column_format = "c|c|" + ("{}|".format(block_format) * len(datasets)).rstrip("|")

    latex = table_df.to_latex(index=False, escape=False, multicolumn=True,
                              multicolumn_format="c", column_format=column_format)

    latex = center_method_phase_headers(latex, dataset_labels)
    latex = latex.replace(r"\multirow{2}{*}{Phase}", r"\multirow{2}{*}{Model Variant}")

    latex = add_group_vertical_bars(latex, datasets)
    latex = add_midrules_between_methods(latex)

    mdl_latex = _latex_model_name(mdl)
    if len(datasets) == 1:
        ds_names = _latex_dataset_name(datasets[0])
    else:
        ds_names = ", ".join(_latex_dataset_name(d) for d in datasets[:-1]) + f", and {_latex_dataset_name(datasets[-1])}"

    compared_baselines = (
        "the full-data linear probe, and the source-dependent PRA baseline"
        if SHOW_LINEAR_PROBE
        else "the source-dependent PRA baseline"
    )
    forget_variant_text = (
        "Linear Probe, PRA, and Relearned (ours)"
        if SHOW_LINEAR_PROBE
        else "PRA and Relearned (ours)"
    )
    rs_variant_text = (
        "Linear Probe, PRA, and our method"
        if SHOW_LINEAR_PROBE
        else "PRA and our method"
    )

    caption = (
        f"Comparison of unlearning methods using the proposed source-free "
        f"class-relearning audit and {compared_baselines} on {mdl_latex} models "
        f"under single-class unlearning across three datasets. For all model "
        f"variants, retain accuracy $\\mathcal{{A}}^t_r$ is reported as the "
        f"mean $\\pm$ standard deviation across forget classes, while "
        f"forget-class accuracy $\\mathcal{{A}}^t_f$ is reported as "
        f"$(\\min,\\mathrm{{avg}},\\max)$. For each method, we report "
        f"the maximum RS across forget classes, and $\\Delta$RS is computed "
        f"relative to the matched retrained-from-scratch control for the "
        f"same forget class. "
        f"Within each dataset block, "
        r"\textbf{bold} and \underline{underlined} values denote the highest "
        r"and second-highest displayed values, respectively, for both RS and "
        r"$\Delta$RS across PRA, the proposed audit, and all enabled baselines."
    )



    label   = f"tab:{slugify(mdl_latex)}_single_class_all_datasets"

    latex = wrap_with_resizebox(latex, caption, label, star=True, width=r"\textwidth")

    out = base_dir / f"latex_table_{slugify(mdl)}_single_class.tex"
    with open(out, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"[OK] wrote: {out}")
    return out

def write_pra_alpha_summary(
    models: List[str],
    datasets: List[str],
) -> Optional[Path]:
    """
    Save the selected-alpha statistics separately instead of adding another
    crowded column to the main paper table.
    """
    rows = []

    for method in METHOD_ORDER:
        if method == "original":
            continue

        for dataset in datasets:
            for model in models:
                pra_path = (
                    base_dir.parent
                    / "pra_single"
                    / method
                    / f"{dataset}_{model}.csv"
                )
                if not pra_path.is_file():
                    continue

                try:
                    d = load_pra_csv(
                        pra_path,
                        ds_hint=dataset,
                        mdl_hint=model,
                        mth_hint=method,
                    )
                    summary = summarize_pra_results(d, dataset)
                    if summary.empty:
                        continue

                    alpha = pd.to_numeric(
                        summary["selected_alpha"],
                        errors="coerce",
                    )
                    runs = pd.to_numeric(
                        summary["num_support_runs"],
                        errors="coerce",
                    )

                    rows.append({
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                        "alpha_mean": alpha.mean(),
                        "alpha_std": alpha.std(),
                        "alpha_min": alpha.min(),
                        "alpha_max": alpha.max(),
                        "num_forget_classes": summary["forget_class"].nunique(),
                        "min_support_runs_per_class": runs.min(),
                        "max_support_runs_per_class": runs.max(),
                    })
                except Exception as exc:
                    print(
                        f"[WARN] Could not summarize PRA alpha for "
                        f"{dataset}/{model}/{method}: {exc}"
                    )

    if not rows:
        print("[WARN] No PRA alpha results were found.")
        return None

    out_df = pd.DataFrame(rows)
    out_path = base_dir / "pra_selected_alpha_summary.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[OK] wrote: {out_path}")
    return out_path


write_pra_alpha_summary(MODELS, DATASETS)

for mdl in MODELS:
    render_joint_table_for_model(mdl, df_all, DATASETS)
