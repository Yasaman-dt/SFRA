import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Style (keep it simple + paper friendly)
# --------------------------
mpl.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 17,
    "axes.titlesize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

# --------------------------
# Config
# --------------------------
base_dir = Path(r"/projets/Zdehghani/Source_Free_Class_Revival/results_single_class/")
merged_path = base_dir / "z_standardized_selected_all_methods.csv"

OUT_DIR = base_dir / "plots_distribution_grid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_PNG_TOO = True

DATASETS = ["cifar10", "cifar100", "tiny_imagenet"]
DATASET_PRETTY = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "tiny_imagenet": "TinyImageNet"}

MODELS = ["resnet18", "vit-s-16", "vit-b-16", "swin-t", "vgg16"]
MODEL_PRETTY = {"resnet18": "ResNet-18", "vit-s-16": "ViT-S/16", "vit-b-16": "ViT-B/16", "swin-t": "Swin-T", "vgg16": "VGG-16"}

PANEL_TAGS = {
    "resnet18": "(a) ResNet-18",
    "vgg16": "(b) VGG-16",
    "vit-b-16": "(c) ViT-B/16",
    "swin-t": "(d) Swin-T",
}

METHOD_ORDER = [
    "retrained", "finetune",
    "gradient_ascent", "neggrad_plus", "random_label",
    "boundary_shrink", 
    "l2ul_adv", 
    "scrub", "bad_teacher", "salun", "delete",
]

METHOD_LABEL_SHORT = {
    "retrained": "Retrained",
    "finetune": "Finetune",
    "gradient_ascent": "Negative Gradient",
    "neggrad_plus": "Negative Gradient+",
    "random_label": "Random Label",
    "boundary_shrink": "Boundary Shrink",
    "l2ul_adv": "Learn to Unlearn",
    "scrub": "SCRUB",
    "bad_teacher": "Bad Teacher",
    "salun": "SalUn",
    "delete": "DELETE",
}

def nice_name(m: str) -> str:
    return METHOD_LABEL_SHORT.get(m, m)

def ordered_methods(present_methods):
    present = [m for m in METHOD_ORDER if m in present_methods]
    extras = sorted(set(present_methods) - set(present))
    return present + extras

# --------------------------
# Colors (consistent with ablation)
# --------------------------
TAB10 = sns.color_palette("tab20", 20)

# Force the 5 ablation methods to match the ablation figure exactly
FIXED_KEY_COLORS = {
    "bad_teacher":      TAB10[0],  # blue
    "delete":           TAB10[1],  # orange
    "gradient_ascent":  TAB10[2],  # green
    "random_label":     TAB10[3],  # red
    "salun":            TAB10[4],  # purple
}


# Fill remaining methods with colors that DON'T collide with the fixed ones
pool = sns.color_palette("tab20", 20) + sns.color_palette("tab10", 10)
used = {tuple(c) for c in FIXED_KEY_COLORS.values()}
pool_it = (c for c in pool if tuple(c) not in used)


# One canonical palette for all plots (pick any base palette you like)
TAB20 = sns.color_palette("tab20", 20)   # for extra distinct colors


METHOD_COLOR = {
    "bad_teacher":      mcolors.to_hex(TAB20[0]),  # blue
    "delete":           mcolors.to_hex(TAB20[2]),  # orange
    "gradient_ascent":  mcolors.to_hex(TAB20[4]),  # green
    "random_label":     mcolors.to_hex(TAB20[6]),  # red
    "salun":            mcolors.to_hex(TAB20[8]),  # purple

    # add the rest ONCE (examples below — choose indices you like and keep fixed)
    "retrained":        mcolors.to_hex(TAB20[10]),
    "finetune":         mcolors.to_hex(TAB20[12]),
    "boundary_shrink":  mcolors.to_hex(TAB20[14]),
    "l2ul_adv":         mcolors.to_hex(TAB20[16]),
    "scrub":            mcolors.to_hex(TAB20[18]),
    "neggrad_plus":     mcolors.to_hex(TAB20[1]),
}

def get_method_color(method_key: str) -> str:
    return METHOD_COLOR.get(method_key, "#555555")


def style_violin(ax, parts, facecolors, alpha=0.28, edge_lw=0.9, edge_alpha=0.95):
    # color each violin body
    for pc, fc in zip(parts["bodies"], facecolors):
        pc.set_facecolor(fc)
        pc.set_edgecolor(mcolors.to_rgba(fc, edge_alpha))
        pc.set_alpha(alpha)
        pc.set_linewidth(edge_lw)

    # make the median line readable
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.1)
        parts["cmedians"].set_alpha(0.9)

# --------------------------
# Load
# --------------------------
df_all = pd.read_csv(merged_path)
if "RS2" not in df_all.columns:
    raise KeyError("RS2 column not found in merged CSV.")
df_all["RS2"] = pd.to_numeric(df_all["RS2"], errors="coerce")

# Keep only what we need
df_all = df_all.dropna(subset=["dataset", "model", "phase", "method", "forget_class", "RS2"])
df_all = df_all[df_all["phase"] == "revival"].copy()

# Reproducible jitter
rng = np.random.default_rng(0)

# --------------------------
# Build one figure per model
# --------------------------
for mdl in MODELS:
    df_m = df_all[df_all["model"] == mdl].copy()
    if df_m.empty:
        continue

    # global method order across datasets (for consistency)
    present_methods_global = set(df_m["method"].unique())
    methods_global = ordered_methods(present_methods_global)

    # shared y-lims (robust)
    y = df_m["RS2"].values
    y_finite = y[np.isfinite(y)]
    if len(y_finite) == 0:
        continue
    ylo = np.percentile(y_finite, 1)
    yhi = np.percentile(y_finite, 99)
    pad = 0.05 * (yhi - ylo + 1e-12)
    ylo, yhi = ylo - pad, yhi + pad

    fig, axes = plt.subplots(
        nrows=1, ncols=len(DATASETS),
        figsize=(4.8 * len(DATASETS), 3.8),
        sharey=True,
        constrained_layout=True
    )

    if len(DATASETS) == 1:
        axes = [axes]

    for j, ds in enumerate(DATASETS):
        ax = axes[j]
        d = df_m[df_m["dataset"] == ds]

        if d.empty:
            ax.set_axis_off()
            continue

        # Methods present in this dataset, ordered by global order
        present = [m for m in methods_global if m in set(d["method"].unique())]

        # Build arrays + labels
        data = []
        labels = []
        methods_used = []
        for m in present:
            vals = d.loc[d["method"] == m, "RS2"].values
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            data.append(vals)
            labels.append(nice_name(m))
            methods_used.append(m)

        if len(data) == 0:
            ax.set_axis_off()
            continue

        facecolors = [get_method_color(m) for m in methods_used]

        parts = ax.violinplot(
            data,
            showmeans=False,
            showmedians=True,
            showextrema=False,
            widths=0.85,
            points=200,
            bw_method="scott",
        )
        style_violin(ax, parts, facecolors=facecolors, alpha=0.7, edge_lw=0.8, edge_alpha=1)

        # Jittered points (match violin color)
        for xi, (vals, fc) in enumerate(zip(data, facecolors), start=1):
            # If too dense, optionally subsample (uncomment)
            # if len(vals) > 1500:
            #     idx = rng.choice(len(vals), size=1500, replace=False)
            #     vals_plot = vals[idx]
            # else:
            #     vals_plot = vals

            vals_plot = vals
            x = xi + 0.10 * (rng.random(len(vals_plot)) - 0.5)
            ax.scatter(
                x, vals_plot,
                s=8,
                c=fc,
                alpha=1,
                linewidths=0,
                rasterized=True  # helps keep PDF size reasonable
            )

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")

        # Paper-friendly grid + order
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.25)

        # y axis formatting (same for all)
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


        # show y tick labels ONLY on the left subplot
        if j == 0:
            ax.tick_params(axis="y", which="both", left=True, labelleft=True)
            ax.set_ylabel("RS")
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        ax.set_title(DATASET_PRETTY.get(ds, ds), fontweight="bold")


        ax.set_xlabel("Unlearning Method")
        

    panel = PANEL_TAGS.get(mdl, f"{MODEL_PRETTY.get(mdl, mdl)}")

    out_pdf = OUT_DIR / f"fig_RS_violin_{mdl}.pdf"
    if SAVE_PNG_TOO:
        fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")

    plt.close(fig)
    print("[OK] saved", out_pdf)


# --------------------------
# Separate single violin plot:
# ResNet-18 + CIFAR-10
# --------------------------

import re

def label_two_lines(name):
    return re.sub(r'\s*\(', r'\n(', name)

METHOD_LABEL_SHORT = {
    "retrained": "Retrained",
    "finetune": "Finetune (CVPR 2020)",
    "gradient_ascent": "Negative Gradient (CVPR 2020)",
    "neggrad_plus": "Negative Gradient+ (NeurIPS 2023)",
    "random_label": "Random Label (AAAI 2021)",
    "boundary_shrink": "Boundary Shrink (CVPR 2023)",
    "l2ul_adv": "Learn to Unlearn (AAAI 2024)",
    "scrub": "SCRUB (NeurIPS 2023)",
    "bad_teacher": "Bad Teacher (AAAI 2023)",
    "salun": "SalUn (ICLR 2024)",
    "delete": "DELETE (CVPR 2025)",
}

mdl = "resnet18"
ds = "cifar10"

d_single = df_all[
    (df_all["model"] == mdl) &
    (df_all["dataset"] == ds)
].copy()

if d_single.empty:
    print(f"[WARN] No data for model={mdl}, dataset={ds}")
else:
    present_methods = set(d_single["method"].unique())
    methods_single = ordered_methods(present_methods)

    data = []
    labels = []
    methods_used = []

    for m in methods_single:
        vals = d_single.loc[d_single["method"] == m, "RS2"].values
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            continue

        data.append(vals)
        labels.append(label_two_lines(nice_name(m)))
        methods_used.append(m)

    if len(data) == 0:
        print(f"[WARN] No valid RS2 values for model={mdl}, dataset={ds}")
    else:
        facecolors = [get_method_color(m) for m in methods_used]

        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(7.5, 3),
            constrained_layout=True
        )

        parts = ax.violinplot(
            data,
            showmeans=False,
            showmedians=True,
            showextrema=False,
            widths=0.85,
            points=200,
            bw_method="scott",
        )

        style_violin(
            ax,
            parts,
            facecolors=facecolors,
            alpha=0.7,
            edge_lw=0.8,
            edge_alpha=1
        )

        # Jittered points
        for xi, (vals, fc) in enumerate(zip(data, facecolors), start=1):
            vals_plot = vals
            x = xi + 0.10 * (rng.random(len(vals_plot)) - 0.5)

            ax.scatter(
                x,
                vals_plot,
                s=8,
                c=fc,
                alpha=1,
                linewidths=0,
                rasterized=True
            )

        
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels([])
        
    for i, lab in enumerate(labels, start=1):
        lines = lab.split("\n")
    
        # --- Method name (BIGGER + BOLD) ---
        ax.text(
            i,
            -0.03,
            lines[0],
            transform=ax.get_xaxis_transform(),
            rotation=45,
            ha="right",
            va="top",
            fontsize=11,          # bigger
            fontweight="bold"
        )
    
        # --- Conference name (SMALLER) ---
        if len(lines) > 1:
            ax.text(
                i,
                -0.15,
                lines[1],
                transform=ax.get_xaxis_transform(),
                rotation=45,
                ha="right",
                va="top",
                fontsize=7           # smaller
            )
                                            
        for t in ax.get_xticklabels():
            t.set_fontsize(10)
            t.set_ha("center")

        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        ax.set_ylabel("RS")
        ax.set_xlabel("Unlearning Method")
        ax.xaxis.set_label_coords(0.5, -0.75)
        

        ax.set_axisbelow(True)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.25)

        out_png = OUT_DIR / f"fig_RS_violin_{mdl}_{ds}_single.png"

        fig.savefig(out_pdf, bbox_inches="tight")

        if SAVE_PNG_TOO:
            fig.savefig(out_png, dpi=600, bbox_inches="tight")

        plt.close(fig)

        print("[OK] saved", out_pdf)
        if SAVE_PNG_TOO:
            print("[OK] saved", out_png)