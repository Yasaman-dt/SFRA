#!/usr/bin/env python3
import os, re, glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from matplotlib.ticker import LogLocator, NullLocator, NullFormatter, ScalarFormatter, MultipleLocator

# ===================== STYLE =====================
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator, NullFormatter

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # first choice; falls back if missing
    "mathtext.fontset": "stix",         # math text in a Times-like style
})


def _fmt_kmb(x, pos=None):
    # Pretty 1e3→1K, 1e6→1M, 1e9→1B; keep ints for [1, 1000), compact <1
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1e9:
        s = f"{x/1e9:g}B"
    elif ax >= 1e6:
        s = f"{x/1e6:g}M"
    elif ax >= 1e3:
        s = f"{x/1e3:g}K"
    elif ax >= 1:
        s = f"{int(x):,}"
    else:
        s = f"{x:.3g}"
    return s

def pretty_log_x(ax, base=10):
    ax.set_xscale('log', base=base)
    ax.xaxis.set_major_locator(LogLocator(base=base))
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_kmb))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5)
matplotlib.rcParams.update({
    # 'text.usetex': True,
    # 'font.family': 'serif',
    # 'font.serif': ['Computer Modern Roman'],
    'axes.labelsize': 19,
    'font.size': 14,
    'legend.fontsize': 15,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
})
plt.rc('axes', labelcolor='black')
plt.rc('ytick', color='black')
plt.rc('xtick', labelcolor='black')
plt.rc('ytick', labelcolor='black')
plt.rc('axes', edgecolor='black')
plt.rc('legend', labelcolor='black')

# ===================== CONFIG =====================
method_dirs_m = [
    r"diff_M/delete",
    r"diff_M/random_label",
    r"diff_M/bad_teacher",
    r"diff_M/gradient_ascent",
    r"diff_M/salun",
    r"diff_M/boundary_shrink",
]
method_dirs_n = [
    r"diff_N/delete",
    r"diff_N/random_label",
    r"diff_N/bad_teacher",
    r"diff_N/gradient_ascent",
    r"diff_N/salun",
    r"diff_N/boundary_shrink",
]
csv_glob_pattern = "*.csv"

METHOD_MAP = {
    "bad_teacher": "Bad Teacher",
    "boundary_shrink": "Boundary Shrink",
    "delete": "DELETE",
    "random_label": "Random Label",
    "salun": "Saliency Unlearn",
    "gradient_ascent": "Negative Gradient",
}


def tab10(n):
    base = sns.color_palette("tab10", 10)
    if n <= len(base):
        return base[:n]
    extra = sns.color_palette("tab20", 20)
    return (base + extra)[:n]

def plot_with_black_box(plot_obj, filename):
    # Check if `plot_obj` is a Matplotlib Figure
    if hasattr(plot_obj, 'axes') and isinstance(plot_obj, plt.Figure):
        for ax in plot_obj.axes:
            pos = ax.get_position()
            rect = patches.Rectangle(
                (pos.x0, pos.y0), pos.width, pos.height,
                linewidth=1.5, edgecolor='black', facecolor='none',
                transform=plot_obj.transFigure
            )
            plot_obj.add_artist(rect)
    elif hasattr(plot_obj, 'axes'):
        # Handle Seaborn FacetGrid with `.axes.flat`
        for ax in plot_obj.axes.flat:
            pos = ax.get_position()
            rect = patches.Rectangle(
                (pos.x0, pos.y0), pos.width, pos.height,
                linewidth=1, edgecolor='black', facecolor='none',
                transform=plt.gcf().transFigure
            )
            plt.gcf().add_artist(rect)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(f"{filename}.png", dpi=600, bbox_inches='tight')
    plt.show()

# ===================== HELPERS =====================
def _pick_from_columns(df, cands, fallback_contains=None):
    lower_map = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in lower_map:
            return lower_map[c]
    if fallback_contains:
        for lc, orig in lower_map.items():
            if all(sub in lc for sub in fallback_contains):
                return orig
    return None

def _infer_arch(df, csv_path):
    # 1) explicit column if present
    arch_col = _pick_from_columns(
        df,
        ["arch","architecture","model","backbone","net","arch_name","model_name"],
        fallback_contains=["arch"]
    )
    if arch_col is not None:
        val = str(df[arch_col].iloc[0]).strip()
        if val and val.lower() not in ("nan", "none", "unknown"):
            return val
    # 2) infer from filename
    name = os.path.basename(csv_path).lower()
    name_norm = re.sub(r"[^\w]+", "-", name)
    m = re.search(
        r"(resnet\d+|wide[-_]?resnet\d+-\d+|"
        r"vit[-_a-z0-9]*|deit[-_a-z0-9]*|"
        r"mamba[-_a-z0-9]*|"
        r"mala(?:_[tsbl])?|ravlt(?:_[tsbl])?)",
        name_norm
    )
    if m:
        return m.group(1).replace("_", "-")
    return "unknown"

def _list_csvs(dir_path, pattern, prefer_keyword=None):
    files = sorted(glob.glob(os.path.join(dir_path, pattern)))
    if not files:
        raise FileNotFoundError(f"No CSVs matching '{pattern}' in {dir_path}")
    if prefer_keyword:
        files = sorted(files, key=lambda f: (prefer_keyword not in os.path.basename(f).lower(), f))
    return files

def tweak_axes(axs, line_alpha=None, marker_size=None, collections_alpha=None):
    for ax in np.ravel(axs):
        # Lines (seaborn lineplot -> ax.lines)
        if line_alpha is not None:
            for ln in ax.lines:
                ln.set_alpha(line_alpha)
        if marker_size is not None:
            for ln in ax.lines:
                ln.set_markersize(marker_size)
        # Collections (e.g., scatter/ci bands)
        if collections_alpha is not None:
            for coll in ax.collections:
                try:
                    coll.set_alpha(collections_alpha)
                except Exception:
                    pass

def _load_block(method_dirs, x_candidates, x_fallback_contains, label_for_xaxis, arch_whitelist=None):
    """Load ALL CSVs in each method dir into long-form accuracy + RS frames."""
    all_long, all_rs = [], []
    common_x_col = None
    out_root = os.path.commonpath([os.path.abspath(d) for d in method_dirs]) if len(method_dirs) > 1 else method_dirs[0]


    for mdir in method_dirs:
        method = os.path.basename(mdir.rstrip("\\/"))
        csv_paths = _list_csvs(mdir, csv_glob_pattern, prefer_keyword="revival")  # keep all

        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)

            x_col = _pick_from_columns(df, x_candidates, fallback_contains=x_fallback_contains)
            y_forget = _pick_from_columns(
                df, ["test_forget","test_fgt","acc_test_forget","acc_test_fgt","forget_test"],
                fallback_contains=["test","forg"]
            )
            y_retain = _pick_from_columns(
                df, ["test_retain","test_rtn","acc_test_retain","acc_test_rtn","retain_test"],
                fallback_contains=["test","retain"]
            )
            rs_col = _pick_from_columns(
                df, ["rs","revival_score","revival","revivalscore","revival_s"],
                fallback_contains=["revival","rs"]
            )
            if any(v is None for v in [x_col, y_forget, y_retain]):
                continue

            # unify x name across files
            if common_x_col is None:
                common_x_col = x_col
            elif x_col != common_x_col:
                df = df.rename(columns={x_col: common_x_col})
                x_col = common_x_col

            arch = _infer_arch(df, csv_path)
            if arch_whitelist is not None and arch not in arch_whitelist:
                continue

            # accuracy long form
            sdf = df[[x_col, y_forget, y_retain]].dropna()
            sdf = sdf[sdf[x_col] > 0].sort_values(by=x_col)

            all_long.append(pd.DataFrame({
                x_col: sdf[x_col].values,
                "value": sdf[y_forget].values,
                "series": "Forget Test",
                "method_arch": f"{method} · {arch}",
                "x_label": label_for_xaxis
            }))
            all_long.append(pd.DataFrame({
                x_col: sdf[x_col].values,
                "value": sdf[y_retain].values,
                "series": "Retain Test",
                "method_arch": f"{method} · {arch}",
                "x_label": label_for_xaxis
            }))

            # RS if available
            if rs_col is not None:
                srs = df[[x_col, rs_col]].dropna()
                srs = srs[srs[x_col] > 0].sort_values(by=x_col)
                srs = srs.rename(columns={rs_col: "RS"})
                srs["method_arch"] = f"{method} · {arch}"
                srs["x_label"] = label_for_xaxis
                all_rs.append(srs)

    if not all_long:
        raise ValueError(f"No valid CSVs found in: {method_dirs}")

    long_df = pd.concat(all_long, ignore_index=True)
    rs_df = pd.concat(all_rs, ignore_index=True) if all_rs else None
    return long_df, rs_df, common_x_col, out_root

# ===================== LOAD BOTH BLOCKS =====================
m_long, m_rs, m_xcol, out_root_m = _load_block(
    method_dirs_m,
    x_candidates=["retain_per_class","retain-per-class","retain_k_per_class","retain_count_per_class","per_class_retain"],
    x_fallback_contains=["retain","class"],
    label_for_xaxis="M",
    arch_whitelist=None
)
n_long, n_rs, n_xcol, out_root_n = _load_block(
    method_dirs_n,
    x_candidates=["pool_take_per_class","pool-k-per-class","pool_per_class","pool_k_per_class","pool_k","pooltakeperclass"],
    x_fallback_contains=["pool","class"],
    label_for_xaxis="N",
    arch_whitelist=None
)

out_root = os.path.commonpath([os.path.abspath(d) for d in (method_dirs_m + method_dirs_n)]) \
           if (method_dirs_m and method_dirs_n) else (out_root_m if method_dirs_m else out_root_n)
os.makedirs(out_root, exist_ok=True)

# ===================== PER-ARCH HELPERS =====================
def _split_method_arch(df):
    if "method_arch" not in df.columns:
        return df
    parts = df["method_arch"].str.split(" · ", n=1, expand=True)
    if parts.shape[1] == 2:
        df = df.copy()
        df["method"] = parts[0]
        df["arch"]   = parts[1]
    else:
        df = df.copy()
        df["method"] = df["method_arch"]
        df["arch"]   = "unknown"
    return df

def _safe_name(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))

# Add method/arch columns + display names
m_long = _split_method_arch(m_long)
n_long = _split_method_arch(n_long)
if m_rs is not None: m_rs = _split_method_arch(m_rs)
if n_rs is not None: n_rs = _split_method_arch(n_rs)

def _add_method_display(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    df["Method"] = df["method"].map(lambda k: METHOD_MAP.get(k, k))
    return df

m_long = _add_method_display(m_long)
n_long = _add_method_display(n_long)
m_rs   = _add_method_display(m_rs)
n_rs   = _add_method_display(n_rs)

# ---- Global palette (consistent across figures) ----
ALL_METHODS = sorted(pd.unique(
    pd.concat([m_long["Method"], n_long["Method"]], ignore_index=True).dropna()
))
PALETTE_GLOBAL = dict(zip(ALL_METHODS, tab10(len(ALL_METHODS))))
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(
    color=[PALETTE_GLOBAL[m] for m in ALL_METHODS]
)

# Collect arches across frames
arches = set(m_long["arch"].unique()) | set(n_long["arch"].unique())
if m_rs is not None and not m_rs.empty:
    arches |= set(m_rs["arch"].unique())
if n_rs is not None and not n_rs.empty:
    arches |= set(n_rs["arch"].unique())

# series → linestyle for combined plot
STYLE_MAP   = {"Retain Test": "", "Forget Test": (5, 2)}
STYLE_ORDER = ["Retain Test", "Forget Test"]

def build_combined_legend(ax, methods, palette, style_order=STYLE_ORDER, style_map=STYLE_MAP):
    """Legend entries for each method × series (retain=solid, forget=dashed)."""
    handles, labels = [], []
    for m in methods:
        col = palette[m]
        for s in style_order:
            dash = style_map[s]
            h = Line2D([0], [0], color=col, lw=3, linestyle='-' if dash == "" else (0, dash))
            label = f"{m}, {'Retain' if 'Retain' in s else 'Forget'}"
            handles.append(h); labels.append(label)
    ax.legend(handles, labels, title="Method, metric", loc="best", frameon=True)

def _legend_methods_only(ax, methods, palette):
    handles, labels = [], []
    for m in methods:
        col = palette[m]
        h = Line2D([0], [0], color=col, lw=3)
        handles.append(h); labels.append(m)
    ax.legend(handles, labels, title="Method", loc="best", frameon=True)

from matplotlib.lines import Line2D

# --- replace your legends with these helpers ---

def add_method_and_style_legends(ax, methods, palette,
                                 style_order=("Retain Test","Forget Test"),
                                 style_map={"Retain Test":"", "Forget Test":(5,2)}):
    """
    Adds two legends:
      1) Method (colors)
      2) Metric (line styles: solid vs dashed)
    """

    # Legend 1: Method colors (solid line, no dash)
    meth_handles, meth_labels = [], []
    for m in methods:
        h = Line2D([0],[0], lw=3, color=palette[m], linestyle='-',
                   marker='o', markersize=8)
        meth_handles.append(h); meth_labels.append(m)
    leg1 = ax.legend(meth_handles, meth_labels, title="Method",
                     loc="upper left", frameon=True)

    # Legend 2: Metric line styles (use neutral color so only style communicates)
    style_handles, style_labels = [], []
    for s in style_order:
        dash = style_map[s]
        # use black/gray so style is clear and not confused with method colors
        style_handles.append(Line2D([0],[0], lw=3, color='black',
                                    linestyle='-' if dash=="" else (0, dash),
                                    marker='o', markersize=8))
        style_labels.append("Retain" if "Retain" in s else "Forget")

    leg2 = ax.legend(style_handles, style_labels, title="Metric",
                     loc="upper right", frameon=True)

    # keep both legends
    ax.add_artist(leg1)


def add_combined_entries(ax, methods, palette,
                         style_order=("Retain Test","Forget Test"),
                         style_map={"Retain Test":"", "Forget Test":(5,2)}):
    """
    Single legend with color × style per entry (if you prefer one legend).
    """
    handles, labels = [], []
    for m in methods:
        for s in style_order:
            dash = style_map[s]
            handles.append(Line2D([0],[0], lw=3, color=palette[m],
                                  linestyle='-' if dash=="" else (0, dash),
                                  marker='o', markersize=8))
            labels.append(f"{m} — {'Retain' if 'Retain' in s else 'Forget'}")
    ax.legend(handles, labels, title="Method × Metric", loc="best", frameon=True)

import math
import re

# 1) Hard rules per architecture name (case-insensitive substrings or regex)
ARCH_YMIN_RULES = [
    (re.compile(r'\b(vit|deit)\b', re.I), 80),
    (re.compile(r'\bresnet', re.I),       39),
    # add more as needed:
    # (re.compile(r'\bmamba|vmamba\b', re.I), 55),
]

def arch_ymin_from_name(arch: str, fallback: int = 39) -> int:
    a = str(arch or "").lower()
    for pat, ymin in ARCH_YMIN_RULES:
        if pat.search(a):
            return ymin
    return fallback

def set_arch_ylim(ax, arch, y_max: int = 101, fallback_min: int = 39):
    ax.set_ylim(arch_ymin_from_name(arch, fallback=fallback_min), y_max)


OUT_DIR = os.path.join(out_root, "ablation_plots")
os.makedirs(OUT_DIR, exist_ok=True)

def out_png(name: str) -> str:
    return os.path.join(OUT_DIR, name)

# ===================== PER-ARCH FIGURES (COMBINED) =====================
for arch in sorted(arches):
    m_long_a = m_long[m_long["arch"] == arch]
    n_long_a = n_long[n_long["arch"] == arch]

    fig_acc_a, (ax_m_a, ax_n_a) = plt.subplots(1, 2, figsize=(10, 5))
    method_order = [m for m in ALL_METHODS if
                    (m in set(m_long_a["Method"]) or m in set(n_long_a["Method"]))]

    # M panel (linear x)
    if not m_long_a.empty:
        sns.lineplot(
            data=m_long_a, x=m_xcol, y="value",
            hue="Method", hue_order=method_order, palette=PALETTE_GLOBAL,
            style="series", style_order=STYLE_ORDER, dashes=STYLE_MAP,
            marker="o", linewidth=3, ax=ax_m_a, legend=False
        )
        ax_m_a.set_xlabel("M"); ax_m_a.set_ylabel("Accuracy (%)")
        #ax_m_a.minorticks_on()
        ax_m_a.tick_params(which='major', bottom=True, left=True)
        ax_m_a.tick_params(which='minor', bottom=True)
        set_arch_ylim(ax_m_a,      arch)
        ax_m_a.yaxis.set_major_locator(MultipleLocator(10))
        ax_m_a.yaxis.set_major_formatter(ScalarFormatter())
        xticks = sorted(m_long_a[m_xcol].unique())
        want = [300, 400]
        xticks = sorted(set(xticks).union(want))
        ax_m_a.set_xticks(xticks)
        ax_m_a.set_xticklabels([f'{int(x):,}' if x >= 1 else f'{x:.3g}' for x in xticks])
        ax_m_a.grid(True)
    else:
        ax_m_a.set_visible(False)

    # N panel (log x)
    if not n_long_a.empty:
        sns.lineplot(
            data=n_long_a, x=n_xcol, y="value",
            hue="Method", hue_order=method_order, palette=PALETTE_GLOBAL,
            style="series", style_order=STYLE_ORDER, dashes=STYLE_MAP,
            marker="o", linewidth=3, ax=ax_n_a, legend=False
        )
        ax_n_a.set_xlabel("N"); ax_n_a.set_ylabel("Accuracy (%)")
        ax_n_a.set_xscale('log', base=10)
        #ax_n_a.minorticks_on()
        ax_n_a.tick_params(which='major', bottom=True, left=True)
        ax_n_a.tick_params(which='minor', bottom=True)
        set_arch_ylim(ax_n_a,      arch)
        ax_n_a.yaxis.set_major_locator(MultipleLocator(10))
        ax_n_a.yaxis.set_major_formatter(ScalarFormatter())
        pretty_log_x(ax_n_a)
        ax_n_a.grid(True)
    else:
        ax_n_a.set_visible(False)

    host_ax = ax_m_a if ax_m_a.get_visible() else ax_n_a
    if host_ax.get_visible() and method_order:
        build_combined_legend(host_ax, method_order, PALETTE_GLOBAL)

    tweak_axes([ax for ax in (ax_m_a, ax_n_a) if ax.get_visible()],
               line_alpha=1.0, marker_size=8, collections_alpha=0.1)
    plt.tight_layout()

    plot_with_black_box(fig_acc_a, out_png(f"accuracies_M_N_{_safe_name(arch)}"))


# ===================== PER-ARCH FIGURES: SPLIT BY SERIES (M and N) =====================
for arch in sorted(arches):
    m_long_a = m_long[m_long["arch"] == arch]
    n_long_a = n_long[n_long["arch"] == arch]

    # ---------------- M figure: two panels (Forget, Retain) ----------------
    if not m_long_a.empty:
        fig_m, (ax_m_forget, ax_m_retain) = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
        method_order_m = [m for m in ALL_METHODS if m in set(m_long_a["Method"])]

        # LEFT: Forget
        m_forget = m_long_a[m_long_a["series"] == "Forget Test"]
        if not m_forget.empty:
            sns.lineplot(
                data=m_forget, x=m_xcol, y="value", style="Method",
                hue="Method", hue_order=method_order_m, palette=PALETTE_GLOBAL, dashes=True,
                marker="o", linewidth=3, ax=ax_m_forget, legend=False
            )
        #ax_m_forget.set_title("Forget accuracy")
        ax_m_forget.set_xlabel("M")
        ax_m_forget.set_ylabel("Forget Accuracy (%)")
        #ax_m_forget.minorticks_on()
        ax_m_forget.tick_params(which='major', bottom=True, left=True)
        ax_m_forget.tick_params(which='minor', bottom=True)
        set_arch_ylim(ax_m_forget, arch)
        ax_m_forget.yaxis.set_major_locator(MultipleLocator(10))
        ax_m_forget.yaxis.set_major_formatter(ScalarFormatter())
        xticks_m = sorted(m_long_a[m_xcol].unique())
        want = [300, 400]
        xticks_m = sorted(set(xticks_m).union(want))
        ax_m_forget.set_xticks(xticks_m)
        ax_m_forget.set_xticklabels([f'{int(x):,}' if x >= 1 else f'{x:.3g}' for x in xticks_m])
        ax_m_forget.grid(True)

        # RIGHT: Retain
        m_retain = m_long_a[m_long_a["series"] == "Retain Test"]
        if not m_retain.empty:
            sns.lineplot(
                data=m_retain, x=m_xcol, y="value", style="Method",
                hue="Method", hue_order=method_order_m, palette=PALETTE_GLOBAL, dashes=True,
                marker="o", linewidth=3, ax=ax_m_retain, legend=True
            )
        #ax_m_retain.set_title("Retain accuracy")
        leg = ax_m_retain.legend(loc='lower right', title="Method")
        ax_m_retain.set_xlabel("M")
        ax_m_retain.set_ylabel("Retain Accuracy (%)")
        #ax_m_retain.minorticks_on()
        ax_m_retain.tick_params(which='major', bottom=True, left=True)
        ax_m_retain.tick_params(which='minor', bottom=True)
        set_arch_ylim(ax_m_retain, arch)
        ax_m_retain.yaxis.set_major_locator(MultipleLocator(10))
        ax_m_retain.yaxis.set_major_formatter(ScalarFormatter())
        xticks_mr = sorted(m_long_a[m_xcol].unique())
        want = [300, 400]
        xticks_mr = sorted(set(xticks_mr).union(want))
        ax_m_retain.set_xticks(xticks_mr)
        ax_m_retain.set_xticks(xticks_mr)
        ax_m_retain.set_xticklabels([f'{int(x):,}' if x >= 1 else f'{x:.3g}' for x in xticks_mr])
        ax_m_retain.grid(True)


        tweak_axes([ax_m_forget, ax_m_retain], line_alpha=1.0, marker_size=8, collections_alpha=0.1)
        plt.tight_layout()
        plot_with_black_box(fig_m,     out_png(f"accuracies_M_{_safe_name(arch)}"))

    # ---------------- N figure: two panels (Forget, Retain) ----------------
    if not n_long_a.empty:
        fig_n, (ax_n_forget, ax_n_retain) = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
        method_order_n = [m for m in ALL_METHODS if m in set(n_long_a["Method"])]

        # LEFT: Forget (log-x)
        n_forget = n_long_a[n_long_a["series"] == "Forget Test"]
        if not n_forget.empty:
            sns.lineplot(
                data=n_forget, x=n_xcol, y="value", style="Method",
                hue="Method", hue_order=method_order_n, palette=PALETTE_GLOBAL,dashes=True,
                marker="o", linewidth=3, ax=ax_n_forget, legend=False
            )
        #ax_n_forget.set_title("Forget accuracy")
        ax_n_forget.set_xlabel("N")
        ax_n_forget.set_ylabel("Forget Accuracy (%)")
        ax_n_forget.set_xscale('log', base=10)
        pretty_log_x(ax_n_forget)
        #ax_n_forget.minorticks_on()
        ax_n_forget.tick_params(which='major', bottom=True, left=True)
        ax_n_forget.tick_params(which='minor', bottom=True)
        set_arch_ylim(ax_n_forget, arch)
        ax_n_forget.yaxis.set_major_locator(MultipleLocator(10))
        ax_n_forget.yaxis.set_major_formatter(ScalarFormatter())
        ax_n_forget.grid(True)

        # RIGHT: Retain (log-x)
        n_retain = n_long_a[n_long_a["series"] == "Retain Test"]
        if not n_retain.empty:
            sns.lineplot(
                data=n_retain, x=n_xcol, y="value", style="Method",
                hue="Method", hue_order=method_order_n, palette=PALETTE_GLOBAL, dashes=True,
                marker="o", linewidth=3, ax=ax_n_retain, legend=True
            )
        #ax_n_retain.set_title("Retain accuracy")
        leg = ax_n_retain.legend(loc='lower right')
        ax_n_retain.set_xlabel("N")
        ax_n_retain.set_ylabel("Retain Accuracy (%)")
        ax_n_retain.set_xscale('log', base=10)
        pretty_log_x(ax_n_retain)
        #ax_n_retain.minorticks_on()
        ax_n_retain.tick_params(which='major', bottom=True, left=True)
        ax_n_retain.tick_params(which='minor', bottom=True)
        set_arch_ylim(ax_n_retain, arch)
        ax_n_retain.yaxis.set_major_locator(MultipleLocator(10))
        ax_n_retain.yaxis.set_major_formatter(ScalarFormatter())
        ax_n_retain.grid(True)
        tweak_axes([ax_n_forget, ax_n_retain], line_alpha=1.0, marker_size=8, collections_alpha=0.1)
        plt.tight_layout()
        plot_with_black_box(fig_n,     out_png(f"accuracies_N_{_safe_name(arch)}"))


# ===================== ONE 2×2 FIGURE PER ARCH (M top, N bottom) =====================


for arch in sorted(arches):
    m_long_a = m_long[m_long["arch"] == arch]
    n_long_a = n_long[n_long["arch"] == arch]

    # skip if nothing to plot
    if m_long_a.empty and n_long_a.empty:
        continue

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))  # [[M_forget, M_retain],[N_forget, N_retain]]
    ax_m_f, ax_m_r = axes[0, 0], axes[0, 1]
    ax_n_f, ax_n_r = axes[1, 0], axes[1, 1]


    # ---------- Row 1: M (linear x) ----------
    if not m_long_a.empty:
        order_m = [m for m in ALL_METHODS if m in set(m_long_a["Method"])]

        # Forget
        m_forget = m_long_a[m_long_a["series"] == "Forget Test"]
        if not m_forget.empty:
            sns.lineplot(data=m_forget, x=m_xcol, y="value", style="Method",
                         hue="Method", hue_order=order_m, palette=PALETTE_GLOBAL,
                         marker="o", linewidth=3, dashes=True, ax=ax_m_f, legend=False)
        ax_m_f.set_xlabel("M"); ax_m_f.set_ylabel("Forget Accuracy (%)")
        #ax_m_f.minorticks_on()
        ax_m_f.tick_params(which='major', bottom=True, left=True)
        ax_m_f.tick_params(which='minor', bottom=True)
        set_arch_ylim(ax_m_f, arch)
        ax_m_f.yaxis.set_major_locator(MultipleLocator(10))
        ax_m_f.yaxis.set_major_formatter(ScalarFormatter())
        xticks_m_f = sorted(m_long_a[m_xcol].unique())
        want = [300, 400]
        xticks_m_f = sorted(set(xticks_m_f).union(want))
        ax_m_f.set_xticks(xticks_m_f)
        ax_m_f.set_xticklabels([f'{int(x):,}' if x >= 1 else f'{x:.3g}' for x in xticks_m])
        ax_m_f.grid(True)

        # Retain
        m_retain = m_long_a[m_long_a["series"] == "Retain Test"]
        if not m_retain.empty:
            sns.lineplot(data=m_retain, x=m_xcol, y="value", style="Method",
                         hue="Method", hue_order=order_m, palette=PALETTE_GLOBAL,
                         marker="o", linewidth=3, dashes=True, ax=ax_m_r, legend=True)
        ax_m_r.legend(loc="lower right", title="Method")
        ax_m_r.set_xlabel("M"); ax_m_r.set_ylabel("Retain Accuracy (%)")
        #ax_m_r.minorticks_on()
        ax_m_r.tick_params(which='major', bottom=True, left=True)
        ax_m_r.tick_params(which='minor', bottom=True)
        set_arch_ylim(ax_m_r, arch)
        ax_m_r.yaxis.set_major_locator(MultipleLocator(10))
        ax_m_r.yaxis.set_major_formatter(ScalarFormatter())
        xticks_m_r = sorted(m_long_a[m_xcol].unique())
        want = [300, 400]
        xticks_m_r = sorted(set(xticks_m_r).union(want))
        ax_m_r.set_xticks(xticks_m_r)
        ax_m_r.set_xticklabels([f'{int(x):,}' if x >= 1 else f'{x:.3g}' for x in xticks_m])
        ax_m_r.grid(True)
    else:
        ax_m_f.set_visible(False); ax_m_r.set_visible(False)

    # ---------- Row 2: N (log x) ----------
    if not n_long_a.empty:
        order_n = [m for m in ALL_METHODS if m in set(n_long_a["Method"])]

        # Forget
        n_forget = n_long_a[n_long_a["series"] == "Forget Test"]
        if not n_forget.empty:
            sns.lineplot(data=n_forget, x=n_xcol, y="value", style="Method",
                         hue="Method", hue_order=order_n, palette=PALETTE_GLOBAL,
                         marker="o", linewidth=3, dashes=True, ax=ax_n_f, legend=False)
        ax_n_f.set_xlabel("N"); ax_n_f.set_ylabel("Forget Accuracy (%)")
        ax_n_f.set_xscale('log', base=10); pretty_log_x(ax_n_f)
        pretty_log_x(ax_n_f)
        #ax_n_retain.minorticks_on()
        ax_n_f.tick_params(which='major', bottom=True, left=True)
        ax_n_f.tick_params(which='minor', bottom=True)
        set_arch_ylim(ax_n_f, arch)
        ax_n_f.yaxis.set_major_locator(MultipleLocator(10))
        ax_n_f.yaxis.set_major_formatter(ScalarFormatter())
        ax_n_f.grid(True)

        # Retain
        n_retain = n_long_a[n_long_a["series"] == "Retain Test"]
        if not n_retain.empty:
            sns.lineplot(data=n_retain, x=n_xcol, y="value", style="Method",
                         hue="Method", hue_order=order_n, palette=PALETTE_GLOBAL,
                         marker="o", linewidth=3, dashes=True, ax=ax_n_r, legend=False)
        ax_n_r.set_xlabel("N"); ax_n_r.set_ylabel("Retain Accuracy (%)")
        ax_n_r.set_xscale('log', base=10); pretty_log_x(ax_n_r)
        set_arch_ylim(ax_n_r, arch)
        ax_n_r.yaxis.set_major_locator(MultipleLocator(10))
        ax_n_r.yaxis.set_major_formatter(ScalarFormatter())
        pretty_log_x(ax_n_r)
        #ax_n_retain.minorticks_on()
        ax_n_r.tick_params(which='major', bottom=True, left=True)
        ax_n_r.tick_params(which='minor', bottom=True)        
        ax_n_r.grid(True)
        
    else:
        ax_n_f.set_visible(False); ax_n_r.set_visible(False)

    
    tweak_axes([ax for ax in (ax_m_f, ax_m_r, ax_n_f, ax_n_r) if ax.get_visible()],
               line_alpha=1.0, marker_size=8, collections_alpha=0.1)

    plt.tight_layout()
    plot_with_black_box(fig,       out_png(f"Acc_M_N_{_safe_name(arch)}"))

