import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# --------------------------
# Style (keep it simple + paper friendly)
# --------------------------
mpl.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
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
base_dir = Path(r"C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/class_unlearning/results_single_class/")
merged_path = base_dir / "z_standardized_selected_all_methods.csv"

OUT_DIR = base_dir / "plots_distribution_grid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_PNG_TOO = True

DATASETS = ["cifar10", "cifar100", "tiny_imagenet"]
DATASET_PRETTY = {"cifar10":"CIFAR-10", "cifar100":"CIFAR-100", "tiny_imagenet":"TinyImageNet"}

MODELS = ["resnet18", "vit-s-16", "vit-b-16", "swin-t", "vgg16"]
MODEL_PRETTY = {"resnet18":"ResNet-18", "vit-s-16":"ViT-S/16", "vit-b-16":"ViT-B/16", "swin-t":"Swin-T", "vgg16":"VGG-16"}

PANEL_TAGS = {
    "resnet18": "(a) ResNet-18",
    "vgg16": "(b) VGG-16",
    "vit-b-16": "(c) ViT-B/16",
    "swin-t": "(d) Swin-T",
}


METHOD_ORDER = [
    "retrained", "finetune",
    "gradient_ascent", "neggrad_plus", "random_label",
    "boundary_shrink", "boundary_expand",
    "l2ul_adv", "l2ul_imp",
    "fisher", "wood_fisher",
    "scrub", "bad_teacher", "salun", "delete",
]

METHOD_LABEL_SHORT = {
    "retrained": "Retrained",
    "finetune": "Finetune",
    "gradient_ascent": "Negative Gradient",
    "neggrad_plus": "Negative Gradient+",
    "random_label": "Random Label",
    "boundary_shrink": "Boundary Shrink",
    "boundary_expand": "Boundary Expand",
    "l2ul_adv": "Learn to Unlearn",
    "l2ul_imp": "Learn to Unlearn Adv+IMP",
    "fisher": "Fisher",
    "wood_fisher": "WoodFisher",
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

def style_violin(ax, parts, alpha=0.35, lw=0.8):
    # Style bodies
    for pc in parts["bodies"]:
        pc.set_alpha(alpha)
        pc.set_linewidth(lw)

    # If present, style summary lines
    for key in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
        if key in parts:
            parts[key].set_linewidth(lw)

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

        # Build arrays + Ns
        data = []
        labels = []
        ns = []
        for m in present:
            vals = d.loc[d["method"] == m, "RS2"].values
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            data.append(vals)
            labels.append(nice_name(m))
            ns.append(len(vals))

        if len(data) == 0:
            ax.set_axis_off()
            continue

        parts = ax.violinplot(
            data,
            showmeans=False,
            showmedians=True,
            showextrema=False,
            widths=0.85,
            points=200,
            bw_method="scott",
        )
        style_violin(ax, parts, alpha=0.35, lw=0.8)

        # Also plot a thin jittered strip of points (helps reviewers see spread)
        # (Turn off if too dense)
        for xi, vals in enumerate(data, start=1):
            x = xi + 0.08 * (np.random.rand(len(vals)) - 0.5)
            ax.plot(x, vals, marker=".", linestyle="None", markersize=1.8, alpha=0.25)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")


        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
        
        # y axis formatting (same for all)
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        # show y tick labels ONLY on the left subplot
        if j == 0:
            ax.tick_params(axis="y", which="both", left=True, labelleft=True)
            ax.set_ylabel(r"$\mathrm{RS}$")
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)  # hide ticks + labels


        ax.set_title(DATASET_PRETTY.get(ds, ds), fontweight="bold", fontsize=12)

    #fig.suptitle(f"{MODEL_PRETTY.get(mdl, mdl)}", y=1.02, fontsize=13)

    panel = PANEL_TAGS.get(mdl, f"{MODEL_PRETTY.get(mdl, mdl)}")


    out_pdf = OUT_DIR / f"fig_RS_violin_{mdl}.pdf"
    if SAVE_PNG_TOO:
        fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[OK] saved", out_pdf)
