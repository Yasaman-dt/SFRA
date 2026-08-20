import pandas as pd
import re
import numpy as np
from pathlib import Path
from typing import Optional, List
from itertools import product

# ----------------- Config -----------------
base_dir = Path("/projets/Zdehghani/Source_Free_Class_Revival/results_multi_class_2/")

DATASETS = ["cifar10", "cifar100", "tiny_imagenet"]

# include ORIGINAL as a method
methods = [
    "original", "retrained",
    "random_label", "finetune", "gradient_ascent", "neggrad_plus",
    "boundary_shrink", "boundary_expand",
    "l2ul_adv", "l2ul_imp", "fisher", "wood_fisher", 
    "scrub", "bad_teacher", "salun", "delete",
]

MODELS   = ["resnet18", "vit-b-16", "swin-t", "vgg16"]

KNOWN_METHODS = set(methods)

# ---- Table output choices ----
SHOW_TRAIN_METRICS = False  # set True to keep train_* columns, False to drop them
SHOW_LINEAR_PROBE = False   # set True to include Linear Probe rows in the tables

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

RS_LABEL = "RS"
DELTA_RS_LABEL = r"$\Delta$RS"

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

    # # NEW RS2:
    # # retain term penalizes only retain drops: max(0, Ar_un - Ar_re)
    # retain_drop = (Ar_un - Ar_re).clip(lower=0.0)

    # # forget term rewards only forget improvement: max(0, Af_re - Af_un)
    # forget_gain = (Af_re - Af_un).clip(lower=0.0)

    # out["RS2"] = (1.0 - retain_drop) * forget_gain
       


    # Retain-preservation term:
    # equals 1 when retain accuracy does not decrease
    retain_drop = (Ar_un - Ar_re).clip(lower=0.0)
    retain_preservation = (1.0 - retain_drop).clip(
        lower=0.0,
        upper=1.0,
    )

    # Forget-recovery term:
    # positive only when forget accuracy improves after revival
    forget_recovery = (Af_re - Af_un).clip(
        lower=0.0,
        upper=1.0,
    )

    # Harmonic mean of retain preservation and forget recovery
    denominator = retain_preservation + forget_recovery

    rs = pd.Series(0.0, index=out.index, dtype=float)

    valid = (
        denominator.gt(0)
        & retain_preservation.notna()
        & forget_recovery.notna()
    )

    rs.loc[valid] = (
        2.0
        * retain_preservation.loc[valid]
        * forget_recovery.loc[valid]
        / denominator.loc[valid]
    )

    # Preserve missing values when any required accuracy is unavailable
    missing = retain_preservation.isna() | forget_recovery.isna()
    rs.loc[missing] = np.nan

    out["RS2"] = rs.clip(lower=0.0, upper=1.0)

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

def load_multi_pra_csv(path: Path) -> pd.DataFrame:
    """Load a multi-class PRA CSV and compute PRA's revival score."""
    d = pd.read_csv(path).rename(columns={
        "baseline_acc_r_test": "test_retain_acc_un",
        "baseline_acc_f_test": "test_forget_acc_un",
        "pra_acc_r_test": "test_retain_acc",
        "pra_acc_f_test": "test_forget_acc",
    })

    required = [
        "test_retain_acc_un",
        "test_forget_acc_un",
        "test_retain_acc",
        "test_forget_acc",
    ]
    missing = [col for col in required if col not in d.columns]
    if missing:
        raise KeyError(f"PRA CSV missing columns {missing}: {path}")

    for col in required + ["selected_alpha", "support_seed"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    if "forget_classes" not in d.columns:
        d["forget_classes"] = pd.NA
    if "support_seed" not in d.columns:
        d["support_seed"] = 0

    # Rerunning PRA appends rows. Keep the latest copy of each trial.
    d = d.drop_duplicates(
        subset=["forget_classes", "support_seed"],
        keep="last",
    )

    max_value = pd.concat([d[col] for col in required]).max(skipna=True)
    scale = 100.0 if pd.notna(max_value) and max_value > 1.5 else 1.0

    ar_un = d["test_retain_acc_un"] / scale
    ar_pra = d["test_retain_acc"] / scale
    af_un = d["test_forget_acc_un"] / scale
    af_pra = d["test_forget_acc"] / scale

    retain_score = (
        1.0 - (ar_un - ar_pra).clip(lower=0.0)
    ).clip(lower=0.0, upper=1.0)
    forget_score = (af_pra - af_un).clip(lower=0.0, upper=1.0)
    denominator = retain_score + forget_score

    d["PRA_RS2"] = np.where(
        denominator > 0,
        2.0 * retain_score * forget_score / denominator,
        0.0,
    )
    d["PRA_RS2"] = d["PRA_RS2"].clip(lower=0.0, upper=1.0)
    return d

def summarize_multi_pra(
    path: Path,
    num_forget_classes: int = 2,
) -> Optional[dict]:
    """Return one PRA row for the requested forget-set cardinality."""
    if not path.is_file():
        return None
    d = load_multi_pra_csv(path)
    forget_count = (
        d["forget_classes"]
        .astype(str)
        .map(lambda value: len([
            item for item in value.split(",") if item.strip()
        ]))
    )
    d = d[forget_count == num_forget_classes].copy()
    if d.empty:
        return None
    return {
        "test_retain_acc": d["test_retain_acc"].mean(),
        "test_forget_acc": d["test_forget_acc"].mean(),
        "PRA_RS2": d["PRA_RS2"].mean(),
    }

def summarize_multi_linear_probe(
    path: Path,
    num_forget_classes: int = 2,
) -> Optional[dict]:
    if not path.is_file():
        return None
    d = pd.read_csv(path)
    required = [
        "forget_classes",
        "output_acc_r_test",
        "output_acc_f_test",
        "linear_probe_acc_r_test",
        "linear_probe_acc_f_test",
    ]
    missing = [col for col in required if col not in d.columns]
    if missing:
        raise KeyError(f"Linear-probe CSV missing columns {missing}: {path}")
    counts = d["forget_classes"].astype(str).map(
        lambda value: len([
            item for item in value.split(",") if item.strip()
        ])
    )
    d = d[counts == num_forget_classes].copy()
    if d.empty:
        return None
    d = d.drop_duplicates(
        subset=["forget_classes", "seed"],
        keep="last",
    )
    for col in required[1:]:
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
    rs = np.where(
        denominator > 0,
        2.0 * retain_score * forget_score / denominator,
        0.0,
    )
    return {
        "test_retain_acc": d["linear_probe_acc_r_test"].mean(),
        "test_forget_acc": d["linear_probe_acc_f_test"].mean(),
        "LP_RS2": float(np.nanmean(rs)),
    }

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


def _style_delta_text(delta, delta_max_v=None, delta_second_v=None):
    if pd.isna(delta):
        return r"\,(-)"
    if (delta_max_v is not None) and (abs(delta - delta_max_v) < 1e-12):
        return rf"\,\mathbf{{({float(delta):+.3f})}}"
    if (delta_second_v is not None) and (abs(delta - delta_second_v) < 1e-12):
        return rf"\,\underline{{({float(delta):+.3f})}}"
    return rf"\,({float(delta):+.3f})"


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



def add_midrules_between_methods(latex_src: str) -> str:
    """
    Post-process the pandas LaTeX table to:
      1) add a double rule at the very top,
      2) add a double rule between header and the first data row,
      3) insert \midrule after each method block (after 'Revival' rows),
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
                        

            if " & SFRA (ours) &" in line:
                # Double rule for Retrained's relearned row
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
    "salun" : r"SalUn \cite{fan2023salun}",
    "delete": r"DELETE \cite{zhou2025decoupled}",
}


def _latex_model_name(mdl: str) -> str:
    mapping = {
        "resnet18": "ResNet-18",
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
    
    # -------- collect displayed RS values and rank them together --------
    # Rank PRA, optional Linear Probe, and Relearned (ours) in the same pool.
    rs_by_method = {}
    pra_rs_by_method = {}
    lp_rs_by_method = {}
    if has_rs:
        for m in df["method"].unique():
            if (m, "revival") in g.index:
                rs_mu = g.loc[(m, "revival"), ("RS2", "mean")]
                rs_by_method[m] = float(rs_mu) if pd.notna(rs_mu) else np.nan
            else:
                rs_by_method[m] = np.nan
            pra_rs_by_method[m] = np.nan
            pra_path = (
                base_dir.parent / "pra_multi" / m / f"{ds}_{mdl}.csv"
            )
            try:
                pra = summarize_multi_pra(pra_path)
                if pra is not None and pd.notna(pra.get("PRA_RS2", np.nan)):
                    pra_rs_by_method[m] = float(pra["PRA_RS2"])
            except Exception:
                pra_rs_by_method[m] = np.nan

            lp_rs_by_method[m] = np.nan
            if SHOW_LINEAR_PROBE:
                lp_path = (
                    base_dir.parent / "linear_probe_multi" / m
                    / f"{ds}_{mdl}.csv"
                )
                try:
                    lp = summarize_multi_linear_probe(lp_path)
                    if lp is not None and pd.notna(lp.get("LP_RS2", np.nan)):
                        lp_rs_by_method[m] = float(lp["LP_RS2"])
                except Exception:
                    lp_rs_by_method[m] = np.nan
    max_v, second_v = _rank_styles(
        list(rs_by_method.values())
        + list(pra_rs_by_method.values())
        + list(lp_rs_by_method.values())
    )
    retrained_rs = rs_by_method.get("retrained", np.nan)
    retrained_pra_rs = pra_rs_by_method.get("retrained", np.nan)
    retrained_lp_rs = lp_rs_by_method.get("retrained", np.nan)

    def delta_from_baseline(method: str, value: float, baseline: float) -> float:
        if method == "retrained" or pd.isna(value) or pd.isna(baseline):
            return np.nan
        return float(value) - float(baseline)

    lp_delta_by_method = {
        m: delta_from_baseline(m, value, retrained_lp_rs)
        for m, value in lp_rs_by_method.items()
    }
    pra_delta_by_method = {
        m: delta_from_baseline(m, value, retrained_pra_rs)
        for m, value in pra_rs_by_method.items()
    }
    rs_delta_by_method = {
        m: delta_from_baseline(m, value, retrained_rs)
        for m, value in rs_by_method.items()
    }
    delta_max_v, delta_second_v = _rank_styles(
        list(lp_delta_by_method.values())
        + list(pra_delta_by_method.values())
        + list(rs_delta_by_method.values())
    )
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
            row[DELTA_RS_LABEL] = "-"
            rows.append(row)
        else:
            # --- Unlearned row ---
            row_un = {"Method": m, "Phase": "Unlearned"}
            if (m, "unlearned") in g.index:
                for col in OUT_METRIC_COLS:
                    mu = g.loc[(m, "unlearned"), (col, "mean")]
                    row_un[col] = fmt_mu(mu)
            else:
                for col in OUT_METRIC_COLS:
                    row_un[col] = "-"

            row_un[RS_LABEL] = "-"
            row_un[DELTA_RS_LABEL] = "-"
            rows.append(row_un)

            lp = None
            if SHOW_LINEAR_PROBE:
                lp_path = (
                    base_dir.parent / "linear_probe_multi" / m
                    / f"{ds}_{mdl}.csv"
                )
                try:
                    lp = summarize_multi_linear_probe(lp_path)
                except Exception as exc:
                    print(
                        f"[WARN] Failed to load linear-probe row from "
                        f"{lp_path}: {exc}"
                    )
                    lp = None
            if lp is not None:
                row_lp = {
                    "Method": m,
                    "Phase": r"Linear Probe \cite{gao2026illusion}",
                }
                for col in OUT_METRIC_COLS:
                    row_lp[col] = fmt_mu(lp[col]) if col in lp else "-"
                raw_lp_rs = lp_rs_by_method.get(m, np.nan)
                row_lp[RS_LABEL] = _style_rs_only(raw_lp_rs, max_v, second_v)
                row_lp[DELTA_RS_LABEL] = _style_delta_only(
                    lp_delta_by_method.get(m, np.nan),
                    delta_max_v,
                    delta_second_v,
                )
                rows.append(row_lp)

            # --- PRA baseline row ---
            pra_path = (
                base_dir.parent / "pra_multi" / m / f"{ds}_{mdl}.csv"
            )
            try:
                pra = summarize_multi_pra(pra_path)
            except Exception as exc:
                print(f"[WARN] Failed to load PRA row from {pra_path}: {exc}")
                pra = None

            if pra is not None:
                row_pra = {
                    "Method": m,
                    "Phase": r"PRA \cite{ha2025unlearning}",
                }
                for col in OUT_METRIC_COLS:
                    row_pra[col] = (
                        fmt_mu(pra[col])
                        if col in pra
                        else "-"
                    )
                raw_pra_rs = pra_rs_by_method.get(m, np.nan)
                row_pra[RS_LABEL] = _style_rs_only(raw_pra_rs, max_v, second_v)
                row_pra[DELTA_RS_LABEL] = _style_delta_only(
                    pra_delta_by_method.get(m, np.nan),
                    delta_max_v,
                    delta_second_v,
                )
                rows.append(row_pra)

            # --- Relearned row (RS cell blank to keep multirow) ---
            row_re = {"Method": m, "Phase": "SFRA (ours)"}
            if (m, "revival") in g.index:
                for col in OUT_METRIC_COLS:
                    mu = g.loc[(m, "revival"), (col, "mean")]
                    row_re[col] = fmt_mu(mu)
            else:
                for col in OUT_METRIC_COLS:
                    row_re[col] = "-"
            if has_rs:
                rs_mu = rs_by_method.get(m, np.nan)
                row_re[RS_LABEL] = _style_rs_only(rs_mu, max_v, second_v)
                row_re[DELTA_RS_LABEL] = _style_delta_only(
                    rs_delta_by_method.get(m, np.nan),
                    delta_max_v,
                    delta_second_v,
                )
            else:
                row_re[RS_LABEL] = "-"
                row_re[DELTA_RS_LABEL] = "-"
            rows.append(row_re)


    table_df = pd.DataFrame(rows, columns=["Method","Phase"] + OUT_METRIC_COLS + [RS_LABEL, DELTA_RS_LABEL]) \
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
    column_format = "c|c|" + ("c" * len(OUT_METRIC_COLS)) + "cc"
    latex = table_df.to_latex(index=False, escape=False, column_format=column_format)
    
    # change "Phase" header to "Model state" (only the first header hit)
    latex = latex.replace(" & Phase & ", " & Model Variant & ", 1)

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
   
for ds, mdl in product(["cifar10", "cifar100", "tiny_imagenet"], MODELS):
    render_table_for(ds, mdl, df_all)
    
    
def add_group_vertical_bars(latex_src: str, datasets: List[str]) -> str:
    """
    Ensure vertical rules continue through the dataset \\multicolumn headers.
    Works even if the header text is wrapped. (Joint table: RS + metrics per dataset)
    """
    out = latex_src
    ncols = 2 + len(OUT_METRIC_COLS)  # metrics + RS + DeltaRS
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

    ncols = 2 + len(OUT_METRIC_COLS)  # metrics + RS + DeltaRS
    group_bits = [rf"\multicolumn{{{ncols}}}{{c}}{{\textbf{{{ds}}}}}" for ds in dataset_labels]
    lines[h1] = r"\multirow{2}{*}{Unlearning Method} & \multirow{2}{*}{Phase} & " + " & ".join(group_bits) + r" \\"

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

def make_table(latex_src: str, caption: str, label: str) -> str:
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\vspace{-3mm}\n"
        "\\fontsize{10.5}{10.5}\\selectfont\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        f"{latex_src}\n"
        "}\n"
        "\\end{table}\n"
    )

    
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

    # ---------- Pass 1 — collect displayed RS values per dataset and rank ----------
    # Rank PRA, optional Linear Probe, and Relearned (ours) in the same pool.
    rs_per_ds = { _latex_dataset_name(ds): {} for ds in datasets }
    pra_rs_per_ds = { _latex_dataset_name(ds): {} for ds in datasets }
    lp_rs_per_ds = { _latex_dataset_name(ds): {} for ds in datasets }
    for ds in datasets:
        dsl = _latex_dataset_name(ds)
        for m in method_list:
            if m == "original":
                rs_per_ds[dsl][m] = np.nan
                pra_rs_per_ds[dsl][m] = np.nan
                lp_rs_per_ds[dsl][m] = np.nan
                continue
            if "RS2" in g.columns.get_level_values(0).unique() and ((ds, m, "revival") in g.index):
                rs_mu = g.loc[(ds, m, "revival"), ("RS2", "mean")]
                rs_per_ds[dsl][m] = float(rs_mu) if pd.notna(rs_mu) else np.nan
            else:
                rs_per_ds[dsl][m] = np.nan

            pra_rs_per_ds[dsl][m] = np.nan
            pra_path = (
                base_dir.parent / "pra_multi" / m / f"{ds}_{mdl}.csv"
            )
            try:
                pra = summarize_multi_pra(pra_path)
                if pra is not None and pd.notna(pra.get("PRA_RS2", np.nan)):
                    pra_rs_per_ds[dsl][m] = float(pra["PRA_RS2"])
            except Exception:
                pra_rs_per_ds[dsl][m] = np.nan

            lp_rs_per_ds[dsl][m] = np.nan
            if SHOW_LINEAR_PROBE:
                lp_path = (
                    base_dir.parent / "linear_probe_multi" / m
                    / f"{ds}_{mdl}.csv"
                )
                try:
                    lp = summarize_multi_linear_probe(lp_path)
                    if lp is not None and pd.notna(lp.get("LP_RS2", np.nan)):
                        lp_rs_per_ds[dsl][m] = float(lp["LP_RS2"])
                except Exception:
                    lp_rs_per_ds[dsl][m] = np.nan

    rs_rank_thresholds = {}
    delta_rank_thresholds = {}
    retrained_rs_per_ds = {}
    retrained_pra_rs_per_ds = {}
    retrained_lp_rs_per_ds = {}
    for ds in datasets:
        dsl = _latex_dataset_name(ds)
        max_v, second_v = _rank_styles(
            list(rs_per_ds[dsl].values())
            + list(pra_rs_per_ds[dsl].values())
            + list(lp_rs_per_ds[dsl].values())
        )
        rs_rank_thresholds[dsl] = (max_v, second_v)
        retrained_rs_per_ds[dsl] = rs_per_ds[dsl].get("retrained", np.nan)
        retrained_pra_rs_per_ds[dsl] = pra_rs_per_ds[dsl].get("retrained", np.nan)
        retrained_lp_rs_per_ds[dsl] = lp_rs_per_ds[dsl].get("retrained", np.nan)

    def delta_from_baseline(dsl: str, method: str, value: float, baseline_by_ds: dict) -> float:
        baseline = baseline_by_ds.get(dsl, np.nan)
        if method == "retrained" or pd.isna(value) or pd.isna(baseline):
            return np.nan
        return float(value) - float(baseline)

    for ds in datasets:
        dsl = _latex_dataset_name(ds)
        delta_values = []
        for method, value in lp_rs_per_ds[dsl].items():
            delta_values.append(
                delta_from_baseline(dsl, method, value, retrained_lp_rs_per_ds)
            )
        for method, value in pra_rs_per_ds[dsl].items():
            delta_values.append(
                delta_from_baseline(dsl, method, value, retrained_pra_rs_per_ds)
            )
        for method, value in rs_per_ds[dsl].items():
            delta_values.append(
                delta_from_baseline(dsl, method, value, retrained_rs_per_ds)
            )
        delta_rank_thresholds[dsl] = _rank_styles(delta_values)
    # -------------------------------------------------------------------------------

    rows = []
    for m in method_list:
        pretty = _method_label(m)

        if m == "original":
            row = {"Method": pretty, "Phase": "Original"}
            for ds in datasets:
                dsl = _latex_dataset_name(ds)
                row[(dsl, RS_LABEL)] = "-"
                row[(dsl, DELTA_RS_LABEL)] = "-"
                if (ds, m, "original") in g.index:
                    for col in OUT_METRIC_COLS:
                        mu = g.loc[(ds, m, "original"), (col, "mean")]
                        row[(dsl, COL_LABELS[col])] = fmt_mu(mu)
                else:
                    for col in OUT_METRIC_COLS:
                        row[(dsl, COL_LABELS[col])] = "-"
            rows.append(row)
            continue

        pra_by_dataset = {}
        lp_by_dataset = {}
        for ds in datasets:
            pra_path = (
                base_dir.parent / "pra_multi" / m / f"{ds}_{mdl}.csv"
            )
            try:
                pra_by_dataset[ds] = summarize_multi_pra(pra_path)
            except Exception as exc:
                print(f"[WARN] Failed to load PRA row from {pra_path}: {exc}")
                pra_by_dataset[ds] = None
            if SHOW_LINEAR_PROBE:
                lp_path = (
                    base_dir.parent / "linear_probe_multi" / m
                    / f"{ds}_{mdl}.csv"
                )
                try:
                    lp_by_dataset[ds] = summarize_multi_linear_probe(lp_path)
                except Exception as exc:
                    print(
                        f"[WARN] Failed to load linear-probe row from "
                        f"{lp_path}: {exc}"
                    )
                    lp_by_dataset[ds] = None
            else:
                lp_by_dataset[ds] = None

        has_pra_data = any(
            value is not None for value in pra_by_dataset.values()
        )
        has_lp_data = any(
            value is not None for value in lp_by_dataset.values()
        )
        method_row_span = 2 + int(has_lp_data) + int(has_pra_data)

        # -------- Unlearned row --------
        row_un = {
            "Method": rf"\multirow{{{method_row_span}}}{{*}}{{{pretty}}}",
            "Phase": "Unlearned",
        }
        for ds in datasets:
            dsl = _latex_dataset_name(ds)
            row_un[(dsl, RS_LABEL)] = "-"
            row_un[(dsl, DELTA_RS_LABEL)] = "-"

            if (ds, m, "unlearned") in g.index:
                for col in OUT_METRIC_COLS:
                    mu = g.loc[(ds, m, "unlearned"), (col, "mean")]
                    row_un[(dsl, COL_LABELS[col])] = fmt_mu(mu)
            else:
                for col in OUT_METRIC_COLS:
                    row_un[(dsl, COL_LABELS[col])] = "-"

        rows.append(row_un)

        if has_lp_data:
            row_lp = {
                "Method": "",
                "Phase": r"Linear Probe \cite{gao2026illusion}",
            }
            for ds in datasets:
                dsl = _latex_dataset_name(ds)
                lp = lp_by_dataset[ds]
                if lp is None:
                    for col in OUT_METRIC_COLS:
                        row_lp[(dsl, COL_LABELS[col])] = "-"
                    row_lp[(dsl, RS_LABEL)] = "-"
                    row_lp[(dsl, DELTA_RS_LABEL)] = "-"
                    continue
                for col in OUT_METRIC_COLS:
                    row_lp[(dsl, COL_LABELS[col])] = (
                        fmt_mu(lp[col]) if col in lp else "-"
                    )
                raw_lp_rs = lp_rs_per_ds[dsl].get(m, np.nan)
                max_v, second_v = rs_rank_thresholds[dsl]
                delta_max_v, delta_second_v = delta_rank_thresholds[dsl]
                row_lp[(dsl, RS_LABEL)] = _style_rs_only(
                    raw_lp_rs, max_v, second_v
                )
                row_lp[(dsl, DELTA_RS_LABEL)] = _style_delta_only(
                    delta_from_baseline(dsl, m, raw_lp_rs, retrained_lp_rs_per_ds),
                    delta_max_v,
                    delta_second_v,
                )
            rows.append(row_lp)

        # -------- One PRA row per method --------
        if has_pra_data:
            row_pra = {
                "Method": "",
                "Phase": r"PRA \cite{ha2025unlearning}",
            }
            for ds in datasets:
                dsl = _latex_dataset_name(ds)
                pra = pra_by_dataset[ds]
                if pra is None:
                    for col in OUT_METRIC_COLS:
                        row_pra[(dsl, COL_LABELS[col])] = "-"
                    row_pra[(dsl, RS_LABEL)] = "-"
                    row_pra[(dsl, DELTA_RS_LABEL)] = "-"
                    continue

                for col in OUT_METRIC_COLS:
                    row_pra[(dsl, COL_LABELS[col])] = (
                        fmt_mu(pra[col])
                        if col in pra
                        else "-"
                    )
                raw_pra_rs = pra_rs_per_ds[dsl].get(m, np.nan)
                max_v, second_v = rs_rank_thresholds[dsl]
                delta_max_v, delta_second_v = delta_rank_thresholds[dsl]
                row_pra[(dsl, RS_LABEL)] = _style_rs_only(
                    raw_pra_rs, max_v, second_v
                )
                row_pra[(dsl, DELTA_RS_LABEL)] = _style_delta_only(
                    delta_from_baseline(dsl, m, raw_pra_rs, retrained_pra_rs_per_ds),
                    delta_max_v,
                    delta_second_v,
                )
            rows.append(row_pra)

        # -------- Revival row --------
        row_rev = {"Method": "", "Phase": "SFRA (ours)"}
        for ds in datasets:
            dsl = _latex_dataset_name(ds)
            raw_rs = rs_per_ds[dsl].get(m, np.nan)
            max_v, second_v = rs_rank_thresholds[dsl]
            delta_max_v, delta_second_v = delta_rank_thresholds[dsl]
            row_rev[(dsl, RS_LABEL)] = _style_rs_only(
                raw_rs, max_v, second_v
            )
            row_rev[(dsl, DELTA_RS_LABEL)] = _style_delta_only(
                delta_from_baseline(dsl, m, raw_rs, retrained_rs_per_ds),
                delta_max_v,
                delta_second_v,
            )
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
    ordered_cols = [("","Unlearning Method"), ("","Phase")]
    for dsl in dataset_labels:
        for c in OUT_METRIC_COLS:
            ordered_cols.append((dsl, COL_LABELS[c]))
        ordered_cols.append((dsl, RS_LABEL))  # <-- use RS_LABEL here
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

    # ---- dataset suffix for label/filename (e.g., c10_c100) ----
    ds_short = "_".join(["c10" if d=="cifar10" else "c100" if d=="cifar100" else slugify(d) for d in datasets])

    if len(datasets) == 1:
        ds_names = _latex_dataset_name(datasets[0])
    else:        ds_names = ", ".join(_latex_dataset_name(d) for d in datasets[:-1]) + f" and {_latex_dataset_name(datasets[-1])}"

    compared_baselines = (
        "linear probing, and the source-dependent PRA baseline"
        if SHOW_LINEAR_PROBE
        else "the source-dependent PRA baseline"
    )
    caption = ( f"Comparison of unlearning methods under the proposed Source-Free "
                f"Relearning Audit (SFRA) and {compared_baselines} for 2-class unlearning "
                f"on CIFAR-10 and CIFAR-100 with ResNet-18. "
               rf"$\Delta$RS denotes the difference from the matched retrained control under the same audit variant. "
               rf"The best and second-best results in each dataset are shown in "
               rf"\textbf{{bold}} and \underline{{underlined}}, respectively.")
    


    # ---- unique label per dataset selection ----
    label   = f"tab:{slugify(mdl_latex)}_multi_class_2"

    #latex = wrap_with_resizebox(latex, caption, label, star=True, width=r"\columnwidth")
    latex = make_table(
        latex,
        caption,
        label
    )
    
    
    # ---- unique filename per dataset selection ----
    out = base_dir / f"latex_table_{slugify(mdl)}_multi_class_2.tex"
    with open(out, "w", encoding="utf-8") as f:
        f.write(latex)
    if mdl == "resnet18":
        paper_out = base_dir.parent / "tables" / out.name
        paper_out.parent.mkdir(parents=True, exist_ok=True)
        paper_out.write_text(latex, encoding="utf-8")
        print(f"[OK] wrote: {paper_out}")
    print(f"[OK] wrote: {out}")
    return out

PAIR_DATASETS = ["cifar10", "cifar100"]

for mdl in MODELS:
    render_joint_table_for_model(mdl, df_all, PAIR_DATASETS)
