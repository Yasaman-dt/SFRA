"""
t-SNE of real samples: pre-FC embeddings vs. post-FC probabilities.

Marker encodes subset: retain = 'o', forget = '^'
Color encodes class:
  - retain  -> ground-truth label
  - forget  -> predicted label (argmax)

No per-subset outputs are created (only <split>_tsne_pre_fc.* and <split>_tsne_prob.*).
"""

import os
import copy
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import get_transforms, get_dataset
from trainer import *  # must provide load_model(ckpt, model_name, dataset, num_classes)

NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'tiny_imagenet': 200, 'imagenet': 1000}

def checkpoint_for(method: str, dataset_name: str, model_name: str,
                   forget_class: int, lr: float, base_dir: str) -> str:
    if method == 'original':
        return f"{base_dir}/test_pretrained_model/{dataset_name}_{model_name}_original_model.pth"
    elif method == 'retrained':
        return f"{base_dir}/test_pretrained_model/{dataset_name}_{model_name}_retrain_forgetcls{forget_class}_model.pth"
    else:
        return f"{base_dir}/{dataset_name}_{model_name}_forgetcls{forget_class}/{method}/lr{lr}/ckpt_best_by_aus.pth"

def parse_class_list(s: str, num_classes: int):
    s = (s or "").strip()
    if not s:
        return []
    selected = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if a > b: a, b = b, a
            for c in range(a, b + 1):
                if 0 <= c < num_classes:
                    selected.add(c)
        else:
            c = int(part)
            if 0 <= c < num_classes:
                selected.add(c)
    return sorted(selected)

def make_feature_extractor(net: nn.Module, num_classes: int, device: str) -> nn.Module:
    feat_net = copy.deepcopy(net).eval().to(device)
    if hasattr(feat_net, "fc") and isinstance(feat_net.fc, nn.Linear) and feat_net.fc.out_features == num_classes:
        feat_net.fc = nn.Identity(); return feat_net
    if hasattr(feat_net, "head") and isinstance(feat_net.head, nn.Linear) and feat_net.head.out_features == num_classes:
        feat_net.head = nn.Identity(); return feat_net
    if hasattr(feat_net, "heads") and hasattr(feat_net.heads, "head") and \
       isinstance(feat_net.heads.head, nn.Linear) and feat_net.heads.head.out_features == num_classes:
        feat_net.heads.head = nn.Identity(); return feat_net
    last_linear_name = None
    for name, m in reversed(list(feat_net.named_modules())):
        if isinstance(m, nn.Linear) and m.out_features == num_classes:
            last_linear_name = name; break
    if last_linear_name is None:
        raise RuntimeError("Could not locate final Linear(out=num_classes) layer to strip.")
    def set_module(root, dotted, new):
        parts = dotted.split("."); parent = root
        for p in parts[:-1]: parent = getattr(parent, p)
        setattr(parent, parts[-1], new)
    set_module(feat_net, last_linear_name, nn.Identity())
    return feat_net

@torch.inference_mode()
def collect_real_embeddings_and_probs(model, feat_model, loader, device: str):
    model.eval(); feat_model.eval()
    feats_list, probs_list, labels_list = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = feat_model(x)
        logits = model(x)
        p = F.softmax(logits, dim=1)
        feats_list.append(f.detach().cpu())
        probs_list.append(p.detach().cpu())
        labels_list.append(y.detach().cpu())
    return torch.cat(feats_list), torch.cat(probs_list), torch.cat(labels_list)

def balanced_indices(labels: torch.Tensor, max_per_class: int) -> torch.Tensor:
    N = labels.numel()
    if max_per_class <= 0:
        return torch.arange(N, dtype=torch.long)
    idxs, counts = [], {}
    for i in range(N):
        c = int(labels[i])
        if counts.get(c, 0) < max_per_class:
            idxs.append(i); counts[c] = counts.get(c, 0) + 1
    return torch.tensor(idxs, dtype=torch.long)

def run_tsne(matrix: np.ndarray, perplexity: float, seed: int, pca_dim: int = 50):
    X = matrix
    if X.ndim != 2:
        raise ValueError("Input to t-SNE must be 2D (N,D).")
    if X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000,
                learning_rate="auto", init="pca", random_state=seed,
                metric="euclidean", verbose=1)
    return tsne.fit_transform(X)
from matplotlib.lines import Line2D  # <-- add this import near the top

def scatter_tsne_mixed(
    Z: np.ndarray,
    color_labels: torch.Tensor,   # int per sample (0..K-1)
    is_forget: torch.Tensor,      # bool per sample
    K: int,
    title: str,                   # kept in signature but unused
    out_png: str,
    alpha: float = 0.7,
    s_retain: int = 20,
    s_forget: int = 20,
    class_names: list = None,     # <-- optional, to show names instead of ids
):
    assert Z.shape[0] == color_labels.numel() == is_forget.numel()

    fig, ax = plt.subplots(figsize=(7, 6), dpi=600)
    cmap = plt.get_cmap("tab20" if K > 10 else "tab10")
    def color_for(c): return cmap(c % cmap.N)

    # Plot retain ('o') and forget ('^') samples per class with consistent colors
    for c in range(K):
        idx_c = (color_labels == c)
        if not idx_c.any():
            continue

        idx_r = idx_c & (~is_forget)
        if idx_r.any():
            P = Z[idx_r.numpy()]
            ax.scatter(P[:, 0], P[:, 1], marker='o', s=s_retain,
                       alpha=alpha, linewidths=0, color=color_for(c))

        idx_f = idx_c & is_forget
        if idx_f.any():
            P = Z[idx_f.numpy()]
            ax.scatter(P[:, 0], P[:, 1], marker='^', s=s_forget,
                       alpha=alpha, linewidths=0, color=color_for(c))

    # -------- legends: put them side-by-side at the top --------
    retain_mask = (~is_forget)
    # if retain_mask.any():
    #     retain_classes = torch.unique(color_labels[retain_mask]).tolist()
    #     retain_classes = [int(c) for c in retain_classes]

    #     # Class-color legend (retain classes)
    #     class_handles = [
    #         Line2D([0],[0], marker='o', linestyle='None', markersize=6,
    #                markerfacecolor=color_for(c), markeredgecolor='none',
    #                label=f"Class {c}")
    #         for c in sorted(retain_classes)
    #     ]

    #     # Place the class legend at the top-right INSIDE the axes
    #     leg_classes = ax.legend(
    #         handles=class_handles,
    #         loc='upper right',
    #         bbox_to_anchor=(0.98, 0.98),  # (x, y) in axes fraction
    #         frameon=True, fancybox=True, framealpha=0.85,
    #         borderpad=0.4, labelspacing=0.3,
    #         handlelength=1.0, handletextpad=0.3,
    #         fontsize=10, title_fontsize=9,
    #         ncol=1
    #     )

    #     # Shape legend (retain vs forget) — unfilled triangle for forget
    #     shape_handles = [
    #         Line2D([0],[0], marker='o', linestyle='None', markersize=6,
    #                markerfacecolor='gray', markeredgecolor='gray', label='Retain Class'),
    #         Line2D([0],[0], marker='^', linestyle='None', markersize=6,
    #                markerfacecolor='gray', markeredgecolor='gray', label='Forget Class'),
    #     ]

    #     # Place the shape legend to the LEFT of the class legend
    #     # tweak x=0.60..0.80 depending on how wide your class legend is
    #     leg_shapes = ax.legend(
    #         handles=shape_handles,
    #         loc='upper right',
    #         bbox_to_anchor=(0.84, 0.98),  # <-- sit beside the class legend
    #         frameon=True, fancybox=True, framealpha=0.85,
    #         borderpad=0.4, labelspacing=0.3,
    #         handlelength=1.0, handletextpad=0.3,
    #         fontsize=10,
    #     )

    #     # Keep both
    #     ax.add_artist(leg_classes)

    # -----------------------------------------------

    # Axis labels only (no title)
    ax.set_xticks([])
    ax.set_yticks([])    
    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)




def build_out_dir(args) -> Path:
    name = f"{args.dataset}_{args.model_name}_{args.method}_f{args.forget_class}_2"
    return Path(args.out_root) / name

def main():
    ap = argparse.ArgumentParser("t-SNE of real samples (pre-FC vs probabilities)")
    ap.add_argument("--ckpt", help="Direct path to checkpoint .pth/.pt for load_model")
    ap.add_argument("--base_dir", default="/export/livia/home/vision/Zdehghani/classification/exps")
    ap.add_argument("--method", default="original",
                    choices=['original','retrained','random_label','finetune','gradient_ascent','neggrad_plus',
                             'boundary_shrink','boundary_expand','l2ul_adv','l2ul_imp','fisher','wood_fisher','delete',
                             'bad_teacher','salun','scrub'])
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--forget_class", type=int, default=0)

    ap.add_argument("--dataset", required=True, choices=list(NUM_CLASSES.keys()))
    ap.add_argument("--model_name", default="resnet18")
    ap.add_argument("--split", default="test", choices=["train", "test", "both"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_per_class", type=int, default=500)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out_root", default="tsne_out")
    ap.add_argument("--autoname", action="store_true")
    ap.add_argument("--forget", type=str, default="",
                    help="Classes to treat as 'forget' for marker shape (e.g., '5' or '0,2-4,7').")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = NUM_CLASSES[args.dataset]
    forget_classes_plot = parse_class_list(args.forget, num_classes)

    # checkpoint
    if args.ckpt and len(args.ckpt) > 0:
        ckpt_path = args.ckpt
    else:
        ckpt_path = checkpoint_for(args.method, args.dataset, args.model_name,
                                   int(args.forget_class), args.lr, args.base_dir)
    print(f"[ckpt] Using checkpoint: {ckpt_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    # model + feature extractor
    model = load_model(ckpt_path, args.model_name, args.dataset, num_classes).to(device).eval()
    feat_model = make_feature_extractor(model, num_classes, device=device)

    # data
    wo_dataaug = True
    transform_train, transform_test = get_transforms(args.dataset, args.model_name, wo_dataaug=wo_dataaug)
    trainset, testset = get_dataset(args.dataset, transform_train, transform_test)

    def eval_loader(ds):
        ds_eval = copy.copy(ds); ds_eval.transform = transform_test
        return DataLoader(ds_eval, batch_size=args.batch_size, shuffle=False,
                          drop_last=False, num_workers=args.num_workers,
                          pin_memory=(device == "cuda"))

    train_loader = eval_loader(trainset)
    test_loader  = eval_loader(testset)

    out_dir = build_out_dir(args) if args.autoname else Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[save] outputs -> {out_dir.resolve()}")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    def process_split(name: str, loader: DataLoader):
        print(f"[t-SNE] Collecting features/probabilities for split: {name}")
        feats, probs, labels = collect_real_embeddings_and_probs(model, feat_model, loader, device)

        keep = balanced_indices(labels, args.max_per_class)
        feats = feats.index_select(0, keep)
        probs = probs.index_select(0, keep)
        labels_sub = labels.index_select(0, keep)

        #torch.save({"feats": feats, "probs": probs, "labels": labels_sub}, out_dir / f"{name}_raw.pt")

        # masks and coloring labels
        K = num_classes
        is_forget = torch.zeros_like(labels_sub, dtype=torch.bool)
        for c in forget_classes_plot:
            is_forget |= (labels_sub == c)

        preds = probs.argmax(dim=1)
        color_labels = torch.where(is_forget, preds, labels_sub)

        # ---- pre-FC ----
        Z_feat = run_tsne(feats.numpy(), perplexity=args.perplexity, seed=args.seed, pca_dim=50)
        pd.DataFrame({
            "x": Z_feat[:, 0], "y": Z_feat[:, 1],
            "true_label": labels_sub.numpy().astype(int),
            "pred_label": preds.numpy().astype(int),
            "color_label": color_labels.numpy().astype(int),
            "is_forget": is_forget.numpy().astype(int),
            "space": "pre_fc", "split": name
        }).to_csv(out_dir / f"{name}_tsne_pre_fc.csv", index=False)
        # ---- pre-FC ----
        scatter_tsne_mixed(
            Z_feat,
            color_labels=color_labels,
            is_forget=is_forget,
            K=num_classes,
            title=f"{name} • t-SNE (pre-FC)",
            out_png=str(out_dir / f"{name}_tsne_pre_fc.png"),
            class_names=getattr(trainset, "classes", None),
        )

        # ---- probabilities ----
        Z_prob = run_tsne(probs.numpy(), perplexity=args.perplexity, seed=args.seed, pca_dim=50)
        pd.DataFrame({
            "x": Z_prob[:, 0], "y": Z_prob[:, 1],
            "true_label": labels_sub.numpy().astype(int),
            "pred_label": preds.numpy().astype(int),
            "color_label": color_labels.numpy().astype(int),
            "is_forget": is_forget.numpy().astype(int),
            "space": "prob", "split": name
        }).to_csv(out_dir / f"{name}_tsne_prob.csv", index=False)
        # ---- probabilities ----
        scatter_tsne_mixed(
            Z_prob,
            color_labels=color_labels,
            is_forget=is_forget,
            K=K,
            title=f"{name} • t-SNE (probabilities)",
            out_png=str(out_dir / f"{name}_tsne_prob.png"),
            class_names=getattr(trainset, "classes", None),
        )

        print(f"[t-SNE] Done for {name}. Saved outputs to {out_dir.resolve()}")

    if args.split in ("train", "both"): process_split("train", train_loader)
    if args.split in ("test", "both"):  process_split("test",  test_loader)

if __name__ == "__main__":
    main()
