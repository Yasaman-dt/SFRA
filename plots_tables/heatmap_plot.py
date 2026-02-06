import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base_dir = Path(r"C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/class_unlearning/results_single_class/")
merged_path = base_dir / "z_standardized_selected_all_methods.csv"
OUT_DIR = base_dir / "plots_per_class_grid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["cifar10", "cifar100", "tiny_imagenet"]
DATASET_PRETTY = {"cifar10":"CIFAR-10", "cifar100":"CIFAR-100", "tiny_imagenet":"TinyImageNet"}

MODELS = ["resnet18", "vit-s-16", "vit-b-16", "swin-t", "vgg16"]
MODEL_PRETTY = {"resnet18":"ResNet-18", "vit-s-16":"ViT-S/16", "vit-b-16":"ViT-B/16", "swin-t":"Swin-T", "vgg16":"VGG-16"}

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
            ax.set_xlabel("Forget class")

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
            interpolation="nearest"  # crisp blocks, avoids white seams
        )

        # X ticks
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns], rotation=0)
        ax.set_xlabel("Forget class")

        # Y ticks: only show labels on left subplot
        if j == 0:
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([nice_method(m) for m in pivot.index], fontsize=10)
            ax.set_ylabel("Unlearning method")
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
