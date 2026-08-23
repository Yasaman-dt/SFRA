import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
import matplotlib as mpl
import re
import math

mpl.rcParams.update({
    "font.size": 5,        
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})
mpl.rcParams.update({
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "legend.labelcolor": "black",
})


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

# ----------------- USER CONFIG -----------------
base_dir   = Path(r"/projets/Zdehghani/Source_Free_Class_Revival/results_single_class/")
DATASET    = "tiny_imagenet"
MODEL      = "vit-b-16"
FORGET_C   = "160"          # accept "9" or 9
SAVE_PNG   = True
OUT_NAME   = f"radar_{DATASET}_{MODEL}_forget{FORGET_C}.png"

# Keep your preferred order; we'll drop those that are missing
METHOD_ORDER = [
    "finetune", "gradient_ascent", "neggrad_plus", "random_label",
    "boundary_shrink", "l2ul_adv", "l2ul_imp",
    "scrub", "bad_teacher", "salun", "delete",
]

PRETTY = {
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

def break_paren(name: str) -> str:
    return re.sub(r'\s*\(', r'\n(', name)

def label_two_lines(name: str) -> str:
    s = break_paren(name)
    return s if '\n' in s else two_line(s)

# ----------------- HELPERS -----------------
def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)

def _load_df(base: Path, ds: str, mdl: str) -> pd.DataFrame:
    csv_base = base / "csvs"
    per_file = csv_base / f"z_standardized_selected_all_methods_{_slug(ds)}_{_slug(mdl)}.csv"
    global_file = csv_base / "z_standardized_selected_all_methods.csv"
    if per_file.exists():
        df = pd.read_csv(per_file)
    else:
        df = pd.read_csv(global_file)
    # normalize types
    for c in ["dataset","model","method","phase","forget_class"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    for c in ["train_retain_acc","train_forget_acc","test_retain_acc","test_forget_acc","RS2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _aggregate_forget_acc(df: pd.DataFrame, methods: List[str], forget_c: str) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (unlearned, revival) arrays aligned with methods list."""
    un_vals, re_vals = [], []
    for m in methods:
        d_un = df[(df["method"] == m) & (df["phase"] == "unlearned") & (df["forget_class"] == str(forget_c))]
        d_re = df[(df["method"] == m) & (df["phase"] == "revival")   & (df["forget_class"] == str(forget_c))]

        # mean for unlearned if multiple rows
        un_v = np.nan if d_un.empty else float(d_un["test_forget_acc"].mean())
        # max for revival if multiple checkpoints/epochs
        re_v = np.nan if d_re.empty else float(d_re["test_forget_acc"].max())
        un_vals.append(un_v); re_vals.append(re_v)
    return np.array(un_vals, dtype=float), np.array(re_vals, dtype=float)

def _get_original_forget_acc(df: pd.DataFrame) -> float:
    d0 = df[(df["method"] == "original") & (df["phase"] == "original")]
    if d0.empty:
        # fallback: any original row
        d0 = df[(df["method"] == "original")]
    return float(d0["test_forget_acc"].mean()) if not d0.empty else np.nan


def two_line(name: str) -> str:
    """Split a label into ~two equal lines (uses first space if only two words)."""
    words = name.split()
    if len(words) <= 1:
        return name
    if len(words) == 2:
        return "\n".join(words)
    mid = math.ceil(len(words) / 2)           # split near the middle
    return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])

def _move_polar_xticklabels_out(ax, angles, radius, labels, fontsize=9):
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    for t, th in zip(ax.get_xticklabels(), angles):
        t.set_fontsize(fontsize)
        t.set_ha("center"); t.set_va("center")
        t.set_rotation(0)
        # IMPORTANT: in polar axes you can set (theta, r) directly
        t.set_position((th, radius))
        
def put_xticklabels_outside(ax, angles, labels, pad=0.16,
                            fontsize_main=12, fontsize_paren=10,
                            line_gap=0.07, rotation=0,
                            bold_main=True):
    ax.set_xticklabels([])  # we'll draw them manually
    for th, lab in zip(angles, labels):
        p = (pad[labels.index(lab)] if hasattr(pad, "__len__") else pad)
        r0 = 1.0 + p
        lines = lab.split("\n")

        if len(lines) == 2 and lines[1].lstrip().startswith("("):
            # main line (method) — bold
            ax.text(th, r0, lines[0],
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom",
                    rotation=rotation, rotation_mode="anchor",
                    fontsize=fontsize_main, clip_on=False,
                    color="black",
                    fontweight="bold" if bold_main else None)
            # second line (venue/year) — normal
            ax.text(th, r0, lines[1],
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top",
                    rotation=rotation, rotation_mode="anchor",
                    fontsize=fontsize_paren, clip_on=False,
                    color="black",
                    fontweight=None)
        else:
            # generic multi-line: bold only the first line
            for i, line in enumerate(lines):
                is_main = (i == 0)
                fs = fontsize_main if is_main else fontsize_paren
                ax.text(th, r0 + i * line_gap, line,
                        transform=ax.get_xaxis_transform(),
                        ha="center", va="center",
                        rotation=rotation, rotation_mode="anchor",
                        fontsize=fs, clip_on=False,
                        color="black",
                        fontweight="bold" if (bold_main and is_main) else None)



        
        
def _radar_plot(labels, orig, un, re, title="", save_path=None, annotate=True):
    K = len(labels)
    angles = np.linspace(0, 2*np.pi, K, endpoint=False)

    def _close(v): return np.concatenate([v, v[:1]])
    angles_c = np.concatenate([angles, angles[:1]])

    clamp = lambda v: np.clip(v, 0, 100)
    orig_vec = clamp(np.array([orig]*K, dtype=float))
    un = clamp(un.astype(float))
    re = clamp(re.astype(float))

    fig = plt.figure(figsize=(6.8, 6.8))  # slightly larger canvas
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    rmax = 100.0
    ax.set_ylim(-1, rmax)
    radii = [20, 40, 60, 80, 100]
    ax.set_rgrids(radii, labels=[f"${r}\%$" for r in radii], angle=180)


    ax.set_rlabel_position(180)
    ax.set_xticks(angles)
    #ax.set_xticklabels(labels)
    
    # default pad for all labels
    pads = [0.25] * len(labels)
    # # push "Negative Gradient+" a bit further out
    if "neggrad_plus" in methods:
        pads[methods.index("neggrad_plus")] = 0.50  
    if "random_label" in methods:
        pads[methods.index("random_label")] = 0.40 
    if "finetune" in methods:
        pads[methods.index("finetune")] = 0.15
    if "scrub" in methods:
        pads[methods.index("scrub")] = 0.20        
    if "l2ul_adv" in methods:
        pads[methods.index("l2ul_adv")] = 0.20 
    if "bad_teacher" in methods:
        pads[methods.index("bad_teacher")] = 0.35 
    if "delete" in methods:
        pads[methods.index("delete")] = 0.20 
    # if "gradient_ascent" in methods:
    #     pads[methods.index("gradient_ascent")] = 0.25 

    put_xticklabels_outside(ax, angles, labels, pad=pads,  fontsize_main=12, fontsize_paren=9)
    
    
    #plt.subplots_adjust(top=1, bottom=0.05, left=0.05, right=0.7)  # a bit more room
    # --- BIGGER tick labels ---
    # for t in ax.get_xticklabels():
    #     t.set_fontsize(12)
    # for t in ax.get_yticklabels():
    #     t.set_fontsize(12)

    p0, = ax.plot(angles_c, _close(orig_vec), linewidth=2.2, linestyle="-",
                  color="blue", label="Original")
    ax.fill(angles_c, _close(orig_vec), alpha=0.10, color="blue")

    p1, = ax.plot(angles_c, _close(un), linewidth=2.2, linestyle="-",
                  color="red", label="Unlearned")
    ax.fill(angles_c, _close(un), alpha=0.10, color="red")

    p2, = ax.plot(angles_c, _close(re), linewidth=2.2, linestyle="-",
                  color="green", label="Relearned")
    ax.fill(angles_c, _close(re), alpha=0.10, color="green")



    if annotate:
        off0, off2 = rmax*0.020, rmax*0.060
        def _annotate_series(vals, offset, fmt="{:.1f}", color=None):
            for ang, v in zip(angles, vals):
                if not np.isfinite(v): continue
                ax.text(ang, min(v + offset, rmax), fmt.format(v),
                        ha="center", va="center",
                        fontsize=12,  # <-- bigger annotation numbers
                        color=color if color is not None else "black",
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.6))
       # _annotate_series(orig_vec, off0, color="blue")
        #_annotate_series(re,       off2, color="green")

    # --- Bigger legend + actually show the title you computed ---
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.15, 1.15), fontsize=12)
    #leg.get_title().set_fontweight("bold")


    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=1200, bbox_inches="tight")
    plt.show()



# ----------------- MAIN -----------------
if __name__ == "__main__":
    df = _load_df(base_dir, DATASET, MODEL)

    # keep rows for this dataset/model
    df = df[(df["dataset"] == str(DATASET)) & (df["model"] == str(MODEL))].copy()
    if FORGET_C is not None:
        df = df[(df["forget_class"].fillna("").astype(str) == str(FORGET_C)) | (df["phase"] == "original")]

    # pick methods that actually exist for this (ds, mdl, forget_class)
    present = sorted(set(df.loc[df["phase"].isin(["unlearned","revival"]), "method"]))
    methods = [m for m in METHOD_ORDER if m in present]
    if not methods:
        raise SystemExit("No unlearned/revival rows found for the chosen dataset/model/forget_class.")

    # pretty labels for axes
    labels = [label_two_lines(PRETTY.get(m, m)) for m in methods]


    # values
    orig_val = _get_original_forget_acc(df)
    un_vals, re_vals = _aggregate_forget_acc(df, methods, str(FORGET_C))

    title = f"{DATASET.upper()} / {MODEL} — forget class {FORGET_C}"
    
    # build a FILE path, not a directory
    save_dir  = Path(r"/projets/Zdehghani/Source_Free_Class_Revival/Radar_Plots")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / OUT_NAME   # e.g., .../Radar_Plots/radar_cifar10_resnet18_forget8.png
    
    _radar_plot(labels, orig_val, un_vals, re_vals, title=title, save_path=save_path)
