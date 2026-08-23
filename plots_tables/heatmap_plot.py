import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


mpl.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
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
base_dir = Path(r"/projets/Zdehghani/Source_Free_Class_Revival/results_single_class/")
merged_path = base_dir / "csvs" / "z_standardized_selected_all_methods.csv"
OUT_DIR = base_dir / "plots" / "plots_per_class_grid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["cifar10", "cifar100", "tiny_imagenet"]
DATASET_PRETTY = {"cifar10":"CIFAR-10", "cifar100":"CIFAR-100", "tiny_imagenet":"TinyImageNet"}

MODELS = ["resnet18", "vit-b-16", "swin-t", "vgg16"]
MODEL_PRETTY = {"resnet18":"ResNet-18", "vit-b-16":"ViT-B/16", "swin-t":"Swin-T", "vgg16":"VGG-16"}

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
    "l2ul_imp": "L2U Adv+IMP",
    "fisher": "Fisher",
    "wood_fisher": "WoodFisher",
    "scrub": "SCRUB",
    "bad_teacher": "Bad Teacher",
    "salun": "SalUn",
    "delete": "DELETE",
}
def nice_method(m): return METHOD_LABEL_SHORT.get(m, m)

def ordered_methods(present):
    present = [m for m in METHOD_ORDER if m in present]
    extras = sorted(set(present) - set(METHOD_ORDER))
    return present + extras

# --------------------------
# Load merged
# --------------------------
df = pd.read_csv(merged_path)
df["RS2"] = pd.to_numeric(df.get("RS2"), errors="coerce")
df = df.dropna(subset=["dataset","model","method","phase","forget_class","RS2"])
df = df[df["phase"] == "revival"].copy()
df["forget_class_num"] = pd.to_numeric(df["forget_class"], errors="coerce")

for mdl in MODELS:
    df_m = df[df["model"] == mdl].copy()
    if df_m.empty:
        print(f"[WARN] No data for model={mdl}")
        continue

    methods_global = ordered_methods(list(df_m["method"].unique()))

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 5), sharey=True)
    fig.subplots_adjust(left=0.30, right=0.99, top=0.86, bottom=0.18, wspace=0.06)

    for j, ds in enumerate(DATASETS):
        ax = axes[j]
        d = df_m[df_m["dataset"] == ds].copy()
        ax.set_title(DATASET_PRETTY.get(ds, ds), fontweight="bold")

        # Important: no gridlines / no seams
        ax.grid(False)

        if d.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_xticks([])
            ax.set_xlabel("Forget Class")

            # ✅ do NOT ax.set_yticks([]) because sharey=True
            if j != 0:
                ax.tick_params(labelleft=False)
            continue

        pivot = (d.groupby(["method","forget_class_num"])["RS2"].mean().unstack("forget_class_num"))
        pivot = pivot.reindex(methods_global)
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)

        ax.imshow(
            pivot.values,
            aspect="auto",
            vmin=0.0, vmax=1.0, 
            interpolation="nearest",  # crisp blocks, avoids white seams
            cmap="coolwarm"
        )


        # --- annotate cells with values ---
        vals = pivot.values
        nrows, ncols = vals.shape
        
        # choose a font size that scales a bit with matrix size
        fs = 10 if (nrows <= 20 and ncols <= 20) else 7
        
        for i in range(nrows):
            for k in range(ncols):
                v = vals[i, k]
                if np.isnan(v):
                    continue
        
                # pick text color based on the cell brightness
                # (threshold at mid of vmin/vmax; here 0.5 since [0,1])
                txt_color = "white" if v > 0.6 else "black"
        
                ax.text(
                    k, i, f"{v:.2f}",     # 2 decimals
                    ha="center", va="center",
                    color=txt_color,
                    fontsize=fs
                )

        # X ticks
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns], rotation=0)
        ax.set_xlabel("Forget Class")

        # Y ticks: only show labels on left subplot
        if j == 0:
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([nice_method(m) for m in pivot.index])
            ax.set_ylabel("Unlearning Method")
        else:
            ax.tick_params(labelleft=False)  # ✅ hide labels without deleting shared ticks

        # cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0)

    #fig.suptitle(f"{MODEL_PRETTY.get(mdl, mdl)}: per-class revival score ($RS_c$)", fontweight="bold")
    cbar = fig.colorbar(axes[0].images[0], ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("RS")

    out = OUT_DIR / f"heatmap_RS_per_class_grid_{mdl}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] saved", out)



# --------------------------
# Separate single heatmap:
# ResNet-18 + CIFAR-10
# --------------------------
mdl = "resnet18"
ds = "cifar10"

d_single = df[(df["model"] == mdl) & (df["dataset"] == ds)].copy()

if d_single.empty:
    print(f"[WARN] No data for model={mdl}, dataset={ds}")
else:
    methods_single = ordered_methods(list(d_single["method"].unique()))

    pivot = (
        d_single.groupby(["method", "forget_class_num"])["RS2"]
        .mean()
        .unstack("forget_class_num")
    )
    pivot = pivot.reindex(methods_single)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.subplots_adjust(left=0.32, right=0.92, top=0.88, bottom=0.16)

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        vmin=0.0, vmax=1.0,
        interpolation="nearest",
        cmap="coolwarm"
    )

    # Annotate cells
    vals = pivot.values
    nrows, ncols = vals.shape
    fs = 10 if (nrows <= 20 and ncols <= 20) else 7

    for i in range(nrows):
        for k in range(ncols):
            v = vals[i, k]
            if np.isnan(v):
                continue

            txt_color = "white" if v > 0.6 else "black"
            ax.text(
                k, i, f"{v:.2f}",
                ha="center", va="center",
                color=txt_color,
                fontsize=fs
            )

    # X ticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns], rotation=0)
    ax.set_xlabel("Forget Class")

    # Y ticks
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([nice_method(m) for m in pivot.index])
    ax.set_ylabel("Unlearning Method")

    # Remove spines and tick marks
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("RS")

    out = OUT_DIR / f"heatmap_RS_{mdl}_{ds}.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("[OK] saved", out)

