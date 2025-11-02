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
import numpy as np
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

def pretty_log_x(ax):
    ax.set_xscale('log', base=10)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos: '{:,.0f}'.format(x) if x >= 1 else ('{:.3g}'.format(x))
    ))


sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5)  # choose one; remove the earlier call
matplotlib.rcParams.update({
    #'text.usetex': True,                # Use LaTeX for all text rendering
    #'font.family': 'serif',            # Use serif fonts
    #'font.serif': ['Computer Modern Roman'],  # Matches LaTeX default
    'axes.labelsize': 20,
    'font.size': 17,
    'legend.fontsize': 17,
    'xtick.labelsize': 19,
    'ytick.labelsize': 19,
    'axes.titlesize': 20
})

# If additional precision is needed, manually adjust elements
# plt.rc('axes', titlesize=22)         # Larger title font size if using titles
# plt.rc('axes', labelsize=22)         # Axis labels font size
# plt.rc('xtick', labelsize=20)         # X-tick labels font size
# plt.rc('ytick', labelsize=20)         # Y-tick labels font size
# plt.rc('legend', fontsize=18)         # Legend font size
# plt.rc('font', size=17)              # Base font size
plt.rc('axes', labelcolor='black')        # Makes x and y axis labels black
plt.rc('xtick', color='black')            # Makes x-tick labels black
plt.rc('ytick', color='black')            # Makes y-tick labels black
plt.rc('xtick', labelcolor='black')       # Makes x-tick numbers black
plt.rc('ytick', labelcolor='black')       # Makes y-tick numbers black
plt.rc('axes', edgecolor='black')         # Sets the axis edges to black
plt.rc('legend', labelcolor='black')  # Ensures legend text is black

# ===================== CONFIG =====================
method_dirs_m = [r"diff_M/delete", r"diff_M/random_label", r"diff_M/bad_teacher", r"diff_M/gradient_ascent", r"diff_M/salun"]
method_dirs_n = [r"diff_N/delete", r"diff_N/random_label", r"diff_N/bad_teacher", r"diff_N/gradient_ascent", r"diff_N/salun"]
csv_glob_pattern = "*.csv"

METHOD_MAP = {
    "bad_teacher": "Bad Teacher",
    "boundary_shrink": "Boundary Shrink",
    "delete": "DELETE",
    "random_label": "Random Label",
    "salun": "Saliency Unlearn",
    "gradient_ascent": "Negative Gradient",
}



def tab10_without_orange(n):
    base = sns.color_palette("tab10", 10)
    no_orange = [c for i, c in enumerate(base) if i != 1]  # drop orange (idx 1)
    if n <= len(no_orange):
        return no_orange[:n]
    # need more than 9 colors? extend sensibly (avoid re-introducing orange)
    extra = sns.color_palette("tab20", 20)  # or any other palette you like
    return (no_orange + extra)[:n]

def tab10(n):
    base = sns.color_palette("tab10", 10)  # includes orange at index 1
    if n <= len(base):
        return base[:n]
    # need more than 10? extend with another palette (keeps original tab10 order/colors)
    extra = sns.color_palette("tab20", 20)
    return (base + extra)[:n]


def plot_with_black_box(plot_obj, filename):
    # Check if `plot_obj` is a Seaborn FacetGrid or a Matplotlib Figure
    if hasattr(plot_obj, 'axes') and isinstance(plot_obj, plt.Figure):
        # Handle Matplotlib Figure with individual axes
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
    
    # Save and show the plot
    plt.savefig(f"{filename}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{filename}.pdf", dpi=600, bbox_inches='tight')

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
    # 1) prefer an explicit column if present
    arch_col = _pick_from_columns(
        df,
        ["arch","architecture","model","backbone","net","arch_name","model_name"],
        fallback_contains=["arch"]
    )
    if arch_col is not None:
        val = str(df[arch_col].iloc[0]).strip()
        if val and val.lower() not in ("nan", "none", "unknown"):
            return val

    # 2) infer from filename (robust to -, _ and digits)
    name = os.path.basename(csv_path).lower()
    # normalize separators
    name_norm = re.sub(r"[^\w]+", "-", name)  # turn spaces/.,/() into '-'

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
        # Put files containing the keyword first, but KEEP ALL FILES
        files = sorted(files, key=lambda f: (prefer_keyword not in os.path.basename(f).lower(), f))
    return files


def tweak_axes(axs, line_alpha=None, marker_size=None, collections_alpha=None):
    for ax in np.ravel(axs):
        # Seaborn lineplot -> Line2D objects live in ax.lines
        if line_alpha is not None:
            for ln in ax.lines:
                ln.set_alpha(line_alpha)
        if marker_size is not None:
            for ln in ax.lines:
                ln.set_markersize(marker_size)
        # If seaborn created PathCollections (e.g., scatter/ci bands), they’re in ax.collections
        if collections_alpha is not None:
            for coll in ax.collections:
                coll.set_alpha(collections_alpha)

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
                # Skip files that don't have the required columns
                continue

            # unify x name across files in this block
            if common_x_col is None:
                common_x_col = x_col
            elif x_col != common_x_col:
                df = df.rename(columns={x_col: common_x_col})
                x_col = common_x_col

            arch = _infer_arch(df, csv_path)
            if arch_whitelist is not None and arch not in arch_whitelist:
                continue  # skip unwanted architectures

            # filter/sort; positive x only for log-scale safety
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
    arch_whitelist=None  # or e.g., ["resnet18", "vit-b-16"]
)
n_long, n_rs, n_xcol, out_root_n = _load_block(
    method_dirs_n,
    x_candidates=["pool_take_per_class","pool-k-per-class","pool_per_class","pool_k_per_class","pool_k","pooltakeperclass"],
    x_fallback_contains=["pool","class"],
    label_for_xaxis="N",
    arch_whitelist=None  # or same whitelist as above
)



out_root = os.path.commonpath([os.path.abspath(d) for d in (method_dirs_m + method_dirs_n)]) \
           if (method_dirs_m and method_dirs_n) else (out_root_m if method_dirs_m else out_root_n)
os.makedirs(out_root, exist_ok=True)

sns.set_theme(style="whitegrid")

# ===================== PER-ARCH HELPERS =====================
def _split_method_arch(df):
    if "method_arch" not in df.columns:
        return df
    # Split on the first ' · ' into method and arch; fallback if delimiter missing
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

# Add method/arch columns
m_long = _split_method_arch(m_long)
n_long = _split_method_arch(n_long)
if m_rs is not None:
    m_rs = _split_method_arch(m_rs)
if n_rs is not None:
    n_rs = _split_method_arch(n_rs)

def _add_method_display(df):
    if df is None or df.empty: 
        return df
    df = df.copy()
    df["method_disp"] = df["method"].map(lambda k: METHOD_MAP.get(k, k))
    return df

m_long = _add_method_display(m_long)
n_long = _add_method_display(n_long)
m_rs   = _add_method_display(m_rs)
n_rs   = _add_method_display(n_rs)


# ---- Build global method palette (after method_disp exists) ----
ALL_METHODS = sorted(pd.unique(
    pd.concat([m_long["method_disp"], n_long["method_disp"]], ignore_index=True).dropna()
))

PALETTE_GLOBAL = dict(zip(
    ALL_METHODS,
    tab10(len(ALL_METHODS))
))

# Optional: set default color cycle to this order
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(
    color=[PALETTE_GLOBAL[m] for m in ALL_METHODS]
)


# Collect all arches present across m/n (accuracies) and RS
arches = set(m_long["arch"].unique()) | set(n_long["arch"].unique())
if m_rs is not None and not m_rs.empty:
    arches |= set(m_rs["arch"].unique())
if n_rs is not None and not n_rs.empty:
    arches |= set(n_rs["arch"].unique())


# series → linestyle
STYLE_MAP   = {"Retain Test": "", "Forget Test": (5, 2)}   # <- CHANGED
STYLE_ORDER = ["Retain Test", "Forget Test"]

def build_combined_legend(ax, methods, palette, style_order=STYLE_ORDER, style_map=STYLE_MAP):
    """Single legend: '<method> — retain' (solid) and '<method> — forget' (dashed)."""
    handles, labels = [], []
    for m in methods:
        col = palette[m]
        for s in style_order:
            dash = style_map[s]
            # Make a handle with the right color + linestyle
            if dash == "":  # solid
                h = Line2D([0], [0], color=col, lw=3, linestyle='-')
            else:           # dashed pattern
                h = Line2D([0], [0], color=col, lw=3, linestyle=(0, dash))
            label = f"{m} — {'retain' if 'Retain' in s else 'forget'}"
            handles.append(h); labels.append(label)
    ax.legend(handles, labels, title="Method, metric", loc="best", frameon=True)

# ===================== PER-ARCH FIGURES =====================
for arch in sorted(arches):
    m_long_a = m_long[m_long["arch"] == arch]
    n_long_a = n_long[n_long["arch"] == arch]

    # --- Figure A: Accuracies only (per-arch) ---
    fig_acc_a, (ax_m_a, ax_n_a) = plt.subplots(1, 2, figsize=(10, 5))

    # Build consistent method palette across both panels for this arch
    # Build consistent palette using display names
    # methods present in this figure, but colors come from the global dict
    method_order = [m for m in ALL_METHODS if
                    (m in set(m_long_a["method_disp"]) or m in set(n_long_a["method_disp"]))]

    palette_list = tab10(len(method_order))
    palette = dict(zip(method_order, palette_list))

    # M-accuracies
    if not m_long_a.empty:
        sns.lineplot(
            data=m_long_a, x=m_xcol, y="value",
            hue="method_disp", hue_order=method_order, palette=PALETTE_GLOBAL,
            style="series", style_order=STYLE_ORDER, dashes=STYLE_MAP,
            marker="o", linewidth=3, ax=ax_m_a, legend=False
        )

        ax_m_a.set_xlabel("M")
        ax_m_a.set_ylabel("Accuracy (%)")
        ax_m_a.minorticks_on()
        ax_m_a.tick_params(which='major', bottom=True, left=True)
        ax_m_a.tick_params(which='minor', bottom=True)
        ax_m_a.set_ylim(49, 101)
        ax_m_a.yaxis.set_major_locator(MultipleLocator(10))
        ax_m_a.yaxis.set_major_formatter(ScalarFormatter())
        xticks = sorted(m_long_a[m_xcol].unique())
        ax_m_a.set_xticks(xticks)
        ax_m_a.set_xticklabels([f'{int(x):,}' if x >= 1 else f'{x:.3g}' for x in xticks])
        ax_m_a.grid(True)
    else:
        ax_m_a.set_visible(False)

    # N-accuracies
    if not n_long_a.empty:
        sns.lineplot(
            data=n_long_a, x=n_xcol, y="value",
            hue="method_disp", hue_order=method_order, palette=PALETTE_GLOBAL,
            style="series", style_order=STYLE_ORDER, dashes=STYLE_MAP,
            marker="o", linewidth=3, ax=ax_n_a, legend=False
        )
        ax_n_a.set_xlabel("N")
        ax_n_a.set_ylabel("Accuracy (%)")
        ax_n_a.set_xscale('log', base=10)
        ax_n_a.minorticks_on()
        ax_n_a.tick_params(which='major', bottom=True, left=True)
        ax_n_a.tick_params(which='minor', bottom=True)
        ax_n_a.set_ylim(49, 101)
        ax_n_a.yaxis.set_major_locator(MultipleLocator(10))
        ax_n_a.yaxis.set_major_formatter(ScalarFormatter())
        pretty_log_x(ax_n_a)

        ax_n_a.grid(True)
    else:
        ax_n_a.set_visible(False)

    def build_combined_legend(ax, methods, palette, style_order=STYLE_ORDER, style_map=STYLE_MAP):
        handles, labels = [], []
        for m in methods:  # m is a display name now
            col = palette[m]
            for s in style_order:
                dash = style_map[s]
                h = Line2D([0], [0], color=col, lw=3, linestyle='-' if dash=="" else (0, dash))
                label = f"{m} — {'retain' if 'Retain' in s else 'forget'}"
                handles.append(h); labels.append(label)
        ax.legend(handles, labels, title="Method, metric", loc="best", frameon=True)

    # legend builder can stay the same; just pass method_order and PALETTE_GLOBAL
    host_ax = ax_m_a if ax_m_a.get_visible() else ax_n_a
    if host_ax.get_visible() and method_order:
        build_combined_legend(host_ax, method_order, PALETTE_GLOBAL)


    tweak_axes([ax for ax in (ax_m_a, ax_n_a) if ax.get_visible()],
               line_alpha=1.0, marker_size=8, collections_alpha=0.1)
    plt.tight_layout()
    plot_with_black_box(fig_acc_a, os.path.join(out_root, f"accuracies_M_N_{_safe_name(arch)}"))
