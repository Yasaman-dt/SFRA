import os, argparse, math, random
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.manifold import TSNE
from pathlib import Path

# -------- helpers --------
def _load_real(real_pt_path):
    obj = torch.load(real_pt_path, map_location='cpu')
    # Use the full train set to avoid leakage from augmentations; labels exist
    X = obj["all_train"]["feats"].float().numpy()
    y = obj["all_train"]["labels"].long().numpy()
    meta = obj["meta"]
    return X, y, meta

def _load_synth(synth_pt_path):
    obj = torch.load(synth_pt_path, map_location='cpu')
    # Concatenate retain + forget synthetic
    Xr  = obj["retain_feats"].float().numpy()
    yr  = obj["retain_labels"].long().numpy()
    Xf  = obj["forget_feats"].float().numpy()
    yf  = obj["forget_labels"].long().numpy()
    X   = np.concatenate([Xr, Xf], axis=0)
    y   = np.concatenate([yr, yf], axis=0)
    meta = obj["summary"]
    return X, y, meta

def _per_class_subsample(X, y, max_per_class, seed=0):
    if (max_per_class is None) or (max_per_class <= 0):
        return X, y
    rng = np.random.default_rng(seed)
    keep_idx = []
    by_cls = defaultdict(list)
    for i, c in enumerate(y):
        by_cls[int(c)].append(i)
    for c, idxs in by_cls.items():
        if len(idxs) > max_per_class:
            sel = rng.choice(idxs, size=max_per_class, replace=False)
        else:
            sel = np.array(idxs, dtype=np.int64)
        keep_idx.append(sel)
    keep_idx = np.concatenate(keep_idx) if keep_idx else np.arange(len(y))
    return X[keep_idx], y[keep_idx]

def _make_colors(y, n_colors=20):
    # Map class → color index using modulo (works for CIFAR10/100)
    return (y.astype(int) % n_colors)

from matplotlib.cm import get_cmap

def make_color_lookup(all_labels, max_colors=10, cmap_name="tab10"):
    cmap = get_cmap(cmap_name)
    uniq = np.unique(all_labels.astype(int))
    # wrap by max_colors so it still works for many classes
    lut = {int(c): cmap(int(c) % max_colors) for c in uniq}
    return lut

def map_colors(y, lut):
    return np.array([lut[int(c)] for c in y])

def plot_tsne(real_X2, real_y, synth_X2, synth_y, out_png, title):
    plt.figure(figsize=(9, 8))

    # ONE shared lookup → identical colors per class across sources
    all_y = np.concatenate([real_y, synth_y])
    lut = make_color_lookup(all_y, max_colors=10, cmap_name="tab10")
    color_real  = map_colors(real_y,  lut)
    color_synth = map_colors(synth_y, lut)

    plt.scatter(real_X2[:,0],  real_X2[:,1],  c=color_real,  s=10, marker='o', alpha=0.7,  linewidths=0)
    plt.scatter(synth_X2[:,0], synth_X2[:,1], c=color_synth, s=16, marker='^', alpha=0.85, linewidths=0)

    from matplotlib.lines import Line2D
    ax = plt.gca()

    # --- Shape legend (source) ---
    shape_handles = [
        Line2D([0],[0], marker='o', color='w', label='Real',      markerfacecolor='gray', markersize=7),
        Line2D([0],[0], marker='^', color='w', label='Synthetic', markerfacecolor='gray', markersize=7),
    ]
    shape_legend = ax.legend(handles=shape_handles, loc='upper right', frameon=True, title='Source')
    ax.add_artist(shape_legend)

    # --- Color legend (classes) ---
    MAX_LEGEND_CLASSES = 20
    uniq_classes = np.unique(np.concatenate([real_y, synth_y])).astype(int)
    if len(uniq_classes) > MAX_LEGEND_CLASSES:
        print(f"[legend] {len(uniq_classes)} classes; showing first {MAX_LEGEND_CLASSES} in legend.")
        uniq_classes = uniq_classes[:MAX_LEGEND_CLASSES]

    class_handles = [
        Line2D([0],[0], marker='s', color='w',
               label=f"class {c}", markerfacecolor=map_colors(np.array([c]), lut)[0],
               markersize=7)
        for c in uniq_classes
    ]
    ax.legend(handles=class_handles, loc='lower left', frameon=True, title='Class')

    # Title + save
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser("t-SNE of real vs synthetic embeddings")
    ap.add_argument("--real_pt",   required=True, help="Path to real_fg{C}.pt")
    ap.add_argument("--synth_pt",  required=True, help="Path to synth_fg{C}.pt")
    ap.add_argument("--out",       default="tsne_real_vs_synth.png")
    ap.add_argument("--per_class_real",  type=int, default=300, help="Max real samples per class (0=all)")
    ap.add_argument("--per_class_synth", type=int, default=300, help="Max synthetic samples per class (0=all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tsne_perplexity", type=float, default=30.0)
    ap.add_argument("--tsne_iters",      type=int,   default=1000)
    args = ap.parse_args()

    # Load
    real_X, real_y, real_meta   = _load_real(args.real_pt)
    synth_X, synth_y, synth_meta = _load_synth(args.synth_pt)

    # Optional per-class caps (keeps class color balance sane)
    real_X, real_y   = _per_class_subsample(real_X,  real_y,  args.per_class_real,  seed=args.seed)
    synth_X, synth_y = _per_class_subsample(synth_X, synth_y, args.per_class_synth, seed=args.seed)

    # Stack and run t-SNE jointly so the 2D space is shared
    X_all = np.concatenate([real_X, synth_X], axis=0)
    n_real = real_X.shape[0]
    # t-SNE
    tsne = TSNE(
        n_components=2, perplexity=args.tsne_perplexity, n_iter=args.tsne_iters,
        init="pca", learning_rate="auto", random_state=args.seed, metric="euclidean"
    )
    X2_all = tsne.fit_transform(X_all)
    real_X2  = X2_all[:n_real]
    synth_X2 = X2_all[n_real:]

    # Title
    forget_class = synth_meta.get("forget_class", None)
    title = f"{real_meta.get('dataset','?')} | {real_meta.get('model','?')} | method={real_meta.get('method','?')}"
    if forget_class is not None:
        title += f" | fg={forget_class}"

    # ---- Build output path: results/<method>/<your --out filename> ----
    method_name = str(real_meta.get("method", "unknown"))
    out_dir = Path("results") / method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = os.path.basename(args.out)  # keep your --out name
    out_path = out_dir / out_name

    # Plot (and save)
    plot_tsne(real_X2, real_y, synth_X2, synth_y, str(out_path), title)
    print(f"[tsne] saved to: {out_path}")


if __name__ == "__main__":
    main()
