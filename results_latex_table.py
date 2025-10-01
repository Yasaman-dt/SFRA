import pandas as pd
import re
import numpy as np
from pathlib import Path
from typing import Optional, List

# ----------------- Config -----------------
base_dir = Path("C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/class_unlearning/results/")

DATASETS = ["cifar10"]  # extend if needed

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


def slugify(s: Optional[str]) -> str:
    if s is None:
        return "unknown"
    return re.sub(r'[^A-Za-z0-9\-]+', "_", s)


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
        phase = "forget"
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

all_rows = []
per_method_info = []

for ds in DATASETS:
    for mdl in MODELS:
        for mth in methods:
            if mth == "original":
                # Search in /original then base_dir for this dataset+model
                original_search_dirs = [base_dir / "original", base_dir]
                original_patterns = [
                    f"results_original_{ds}_{mdl}.csv",
                    f"{ds}_{mdl}_original*_metrics*.csv",
                    f"{ds}_{mdl}_original*.csv",
                    f"*{ds}*{mdl}*original*.csv",
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

                        out_original = original_path.parent / f"standardized_original_{slugify(ds)}_{slugify(mdl)}.csv"
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
                    df_f["phase"] = "forget"
                    # normalize meta
                    df_f["dataset"] = ds
                    df_f["model"]   = mdl
                    df_f["method"]  = mth
                    all_rows.append(df_f)

                    out_forget = method_dir / f"standardized_forget_selected_{mth}_{slugify(ds)}_{slugify(mdl)}.csv"
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
                    all_rows.append(df_r)

                    out_revival = method_dir / f"standardized_revival_selected_{mth}_{slugify(ds)}_{slugify(mdl)}.csv"
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
    global_merged = base_dir / "standardized_selected_all_methods.csv"
    merged.to_csv(global_merged, index=False)

    (merged[merged["phase"] == "forget"]
        .to_csv(base_dir / "standardized_forget_all_methods.csv", index=False))
    (merged[merged["phase"] == "revival"]
        .to_csv(base_dir / "standardized_revival_all_methods.csv", index=False))
    (merged[merged["phase"] == "original"]
        .to_csv(base_dir / "standardized_original_all_methods.csv", index=False))

    # (B) Per-(dataset, model) files
    for (ds_i, mdl_i), df_i in merged.groupby(["dataset", "model"], dropna=False):
        ds_tag  = slugify(ds_i)
        mdl_tag = slugify(mdl_i)

        out_merged = base_dir / f"standardized_selected_all_methods_{ds_tag}_{mdl_tag}.csv"
        df_i.to_csv(out_merged, index=False)

        for ph in ["forget", "revival", "original"]:
            df_ph = df_i[df_i["phase"] == ph]
            if not df_ph.empty:
                out_ph = base_dir / f"standardized_{ph}_all_methods_{ds_tag}_{mdl_tag}.csv"
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

merged_path = base_dir / "standardized_selected_all_methods.csv"
df_all = pd.read_csv(merged_path)

metric_cols = ["train_retain_acc","train_forget_acc","test_retain_acc","test_forget_acc"]
for c in metric_cols:
    df_all[c] = pd.to_numeric(df_all.get(c), errors="coerce")

# Order you prefer; any extra/unknown methods present in the CSV will be appended at the end.
METHOD_ORDER = [
    "original", "retrained", "finetune", 
    "gradient_ascent", "neggrad_plus"
    "random_label", 
    "boundary_shrink", "boundary_expand",
    "l2ul_adv", "l2ul_imp",
    "fisher", "wood_fisher",
    "scrub", "bad_teacher", "salun", "delete",
]
PHASE_ORDER = ["forget", "revival"]  # non-original methods get these two rows

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
    r"""
    Turn pairs (Method, Phase=Forget/Revival) into a single visible Method cell
    using \multirow{2}{*}{...} on the 'Forget' row and an empty cell on 'Revival'.
    Requires \usepackage{multirow} in your LaTeX preamble.
    """
    df = table_df.copy()

    # Work per "display label" (after mapping) so citations, etc., are preserved
    for label in df["Method"].unique():
        sub = df[df["Method"] == label]
        # Only multirow when we truly have both phases
        if len(sub) == 2 and set(sub["Phase"]) == {"Forget", "Revival"}:
            idx_forget  = sub.index[sub["Phase"] == "Forget"][0]
            idx_revival = sub.index[sub["Phase"] == "Revival"][0]
            df.loc[idx_forget,  "Method"] = rf"\multirow{{2}}{{*}}{{{label}}}"
            df.loc[idx_revival, "Method"] = ""  # empty cell under the multirow
    return df



def wrap_with_resizebox(latex_src: str, caption: str, label: str,
                        star: bool = True, width: str = r"\textwidth") -> str:
    env = "table*" if star else "table"
    return (
        f"\\begin{{{env}}}[t]\n"
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
    "l2ul_adv": r"Learn to Unlearn Adv \cite{cha2024learning}",
    "l2ul_imp": r"Learn to Unlearn Adv+IMP \cite{cha2024learning}",
    # "fisher": r"Fisher",
    # "wood_fisher": r"WoodFisher",
    "scrub": r"SCRUB \cite{kurmanji2023towards}",
    "bad_teacher" : r"Bad Teacher \cite{chundawat2023can}",
    "salun" : r"Saliency Unlearn \cite{fan2023salun}",
    "delete": r"Delete \cite{zhou2025decoupled}",
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
    return s.replace("-", "_").lower().strip()

def _method_label(raw: str) -> str:
    v = method_name_and_ref.get(_normalize_key(raw))
    if v is None:
        return raw  # fallback: leave as-is
    if isinstance(v, tuple):
        return v[0]  # take first element if tuple provided
    return v


def render_table_for(ds: str, mdl: str, df_src: pd.DataFrame):
    df = df_src[(df_src["dataset"] == ds) & (df_src["model"] == mdl)].copy()
    if df.empty:
        print(f"[WARN] No rows for dataset={ds}, model={mdl}")
        return None

    g = df.groupby(["method","phase"], dropna=False)[metric_cols].agg(["mean","std"])

    # Keep your preferred ordering, then append any extras found in the data.
    present = [m for m in METHOD_ORDER if m in df["method"].unique()]
    extras  = sorted(set(df["method"].unique()) - set(present))
    method_list = present + extras

    rows = []
    for m in method_list:
        if m == "original":
            phase = "original"
            row = {"Method": m, "Phase": phase.title()}
            for col in metric_cols:
                mu = g.loc[(m, phase), (col, "mean")] if (m, phase) in g.index else np.nan
                sd = g.loc[(m, phase), (col, "std")]  if (m, phase) in g.index else np.nan
                row[col] = fmt_mu_sigma(mu, sd)
            rows.append(row)
        else:
            for phase in PHASE_ORDER:
                row = {"Method": m, "Phase": phase.title()}
                if (m, phase) in g.index:
                    for col in metric_cols:
                        mu = g.loc[(m, phase), (col, "mean")]
                        sd = g.loc[(m, phase), (col, "std")]
                        row[col] = fmt_mu_sigma(mu, sd)
                else:
                    for col in metric_cols:
                        row[col] = "-"
                rows.append(row)

    table_df = pd.DataFrame(
        rows, columns=["Method","Phase"] + metric_cols
        ).rename(columns={
            "train_retain_acc": r"$\mathcal{A}^{\text{train}}_{r}$",
            "train_forget_acc": r"$\mathcal{A}^{\text{train}}_{f}$",
            "test_retain_acc":  r"$\mathcal{A}^{\text{test}}_{r}$",
            "test_forget_acc":  r"$\mathcal{A}^{\text{test}}_{f}$",
        })

            
    table_df["Method"] = table_df["Method"].map(_method_label)

    table_df = apply_multirow(table_df)

    # Escape LaTeX specials in text columns
    for col in ["Method", "Phase"]:
        table_df[col] = (
            table_df[col].astype(str)
            .str.replace("_", r"\_", regex=False)
            .str.replace("&", r"\&", regex=False)
            .str.replace("%", r"\%", regex=False)
        )

    latex = table_df.to_latex(
        index=False, escape=False, column_format="c|c|cccc",
        caption=None,
        label=None,
    )
    latex = add_midrules_between_methods(latex)

    mdl_latex = _latex_model_name(mdl)
    ds_latex  = _latex_dataset_name(ds)

    cap = f"{ds_latex} / {mdl_latex} — Unlearning results (mean$\\pm$std)"
    lab = f"tab:{slugify(ds_latex)}_{slugify(mdl_latex)}_ALL"
    latex = wrap_with_resizebox(latex, cap, lab, star=True, width=r"\textwidth")


    # Make a safe filename (avoid weird chars)
    safe_mdl = re.sub(r'[^A-Za-z0-9\-]+', "_", mdl)
    safe_ds  = slugify(ds_latex)
    out = base_dir / f"latex_table_{safe_ds}_{safe_mdl}.tex"
    with open(out, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"[OK] wrote: {out}")
    return out

# ---- choose the models you want, and the dataset (here: cifar10) ----
models_to_render = MODELS
dataset_to_render = "cifar10"  # change/loop if you have more datasets


for mdl in MODELS:
    render_table_for(dataset_to_render, mdl, df_all)
    
   