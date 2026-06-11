"""
t-SNE of real samples: pre-FC embeddings vs. post-FC probabilities.

Two ways to pick a checkpoint:
  A) Direct path: --ckpt /path/to/model.pth
  B) Revival-style construction (same as revival_single_class):
     --base_dir DIR --method {original,retrained,...} --lr 5e-5 --forget_class 0

Outputs are auto-named if --autoname is set:
  <out_root>/<dataset>_<model>_<method>_f<forget_class>/

Also supports --forget "0,2-4,7" to produce forget-only / retain-only plots.
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

# ------------------- constants -------------------

NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'tiny_imagenet': 200, 'imagenet': 1000}

# ------------------- revival-style checkpoint helper -------------------

def checkpoint_for(method: str, dataset_name: str, model_name: str,
                   forget_class: int, lr: float, base_dir: str) -> str:
    """
    Mirrors the path logic from revival_single_class without importing it.
    """
    if method == 'original':
        return f"{base_dir}/test_pretrained_model/{dataset_name}_{model_name}_original_model.pth"
    elif method == 'retrained':
        return f"{base_dir}/test_pretrained_model/{dataset_name}_{model_name}_retrain_forgetcls{forget_class}_model.pth"
    else:
        return f"{base_dir}/{dataset_name}_{model_name}_forgetcls{forget_class}/{method}/lr{lr}/ckpt_best_by_aus.pth"

# ------------------- helpers -------------------

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
            if a > b:
                a, b = b, a
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
    # common heads
    if hasattr(feat_net, "fc") and isinstance(feat_net.fc, nn.Linear) and feat_net.fc.out_features == num_classes:
        feat_net.fc = nn.Identity(); return feat_net
    if hasattr(feat_net, "head") and isinstance(feat_net.head, nn.Linear) and feat_net.head.out_features == num_classes:
        feat_net.head = nn.Identity(); return feat_net
    if hasattr(feat_net, "heads") and hasattr(feat_net.heads, "head") and \
       isinstance(feat_net.heads.head, nn.Linear) and feat_net.heads.head.out_features == num_classes:
        feat_net.heads.head = nn.Identity(); return feat_net
    # fallback: last Linear with out=num_classes
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

import matplotlib.pyplot as plt

def scatter_tsne(Z, labels, title, out_png, alpha=0.7, s=25,
                 hide_axes=True, forget_classes=None, num_classes=None):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=600)
    forget_classes = set(forget_classes or [])
    K = int(num_classes) if num_classes is not None else (
        int(labels.max().item() + 1) if labels.numel() > 0 else 0
    )

    cmap = plt.get_cmap('tab10')
    handles, legend_texts = [], []

    for c in range(K):
        m = (labels == c).numpy()
        is_forget = (c in forget_classes)
        marker = '^' if is_forget else 'o'
        color  = cmap(c % 10)

        if m.sum() > 0:
            h = ax.scatter(Z[m, 0], Z[m, 1], alpha=alpha, s=s,
                           marker=marker, color=color, edgecolors='none')
        else:
            # proxy handle so legend still shows Class c
            import matplotlib.lines as mlines
            h = mlines.Line2D([0], [0], marker=marker, linestyle='None',
                              markerfacecolor=color, markeredgewidth=0, markersize=6)

        handles.append(h)
        legend_texts.append(f"Class {c}" + (" (forget)" if is_forget else ""))

    #ax.set_title(title)
    leg = ax.legend(handles, legend_texts, loc="upper right", ncol=1, fontsize=8,
                    frameon=True, fancybox=True, framealpha=0.95, borderpad=0.6,
                    markerscale=1.6, handlelength=1.4, handletextpad=0.6)
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_edgecolor("0.6")

    #if hide_axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)  
    #ax.set_xlabel(''); ax.set_ylabel('')
        # for spine in ax.spines.values(): spine.set_visible(False)
        # ax.set_frame_on(False)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)





def build_out_dir(args) -> Path:
    # Auto-name: <out_root>/<dataset>_<model>_<method>_f<forget_class>/
    name = f"{args.dataset}_{args.model_name}_{args.method}_f{args.forget_class}_1"
    return Path(args.out_root) / name

# ------------------- main -------------------

def main():
    ap = argparse.ArgumentParser("t-SNE of real samples (pre-FC vs probabilities)")
    # Way A: direct checkpoint
    ap.add_argument("--ckpt", help="Direct path to checkpoint .pth/.pt for load_model")

    # Way B: build like revival_single_class
    ap.add_argument("--base_dir", default="/export/livia/home/vision/Zdehghani/classification/exps",
                    help="Base DIR used by revival_single_class to store checkpoints")
    ap.add_argument("--method", default="original",
                    choices=['original','retrained','random_label','finetune','gradient_ascent','neggrad_plus',
                             'boundary_shrink','boundary_expand','l2ul_adv','l2ul_imp','fisher','wood_fisher','delete',
                             'bad_teacher','salun','scrub'],
                    help="Revival method name (used only if --ckpt is not provided)")
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="LR in the path for most methods (ignored for 'original')")
    ap.add_argument("--forget_class", type=int, default=0,
                    help="Forget class for path construction")

    # Model/data
    ap.add_argument("--dataset", required=True, choices=list(NUM_CLASSES.keys()))
    ap.add_argument("--model_name", default="resnet18")
    ap.add_argument("--split", default="test", choices=["train", "test", "both"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_per_class", type=int, default=500)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)

    # Saving (auto-name; no datetime)
    ap.add_argument("--out_root", default="tsne_out",
                    help="Root folder to save results under.")
    ap.add_argument("--autoname", action="store_true",
                    help="If set, saves to <out_root>/<dataset>_<model>_<method>_f<forget_class>/")

    # Optional plotting subset
    ap.add_argument("--forget", type=str, default="",
                    help="Classes to treat as 'forget' for plotting subsets, e.g. '7' or '0,2-4,7'")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = NUM_CLASSES[args.dataset]
    forget_classes_plot = parse_class_list(args.forget, num_classes)
    if not forget_classes_plot:
        forget_classes_plot = [int(args.forget_class)]
    
    
    # Resolve checkpoint
    if args.ckpt and len(args.ckpt) > 0:
        ckpt_path = args.ckpt
    else:
        ckpt_path = checkpoint_for(
            method=args.method,
            dataset_name=args.dataset,
            model_name=args.model_name,
            forget_class=int(args.forget_class),
            lr=args.lr,
            base_dir=args.base_dir
        )
    print(f"[ckpt] Using checkpoint: {ckpt_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    # Load model & feature extractor
    model = load_model(ckpt_path, args.model_name, args.dataset, num_classes).to(device).eval()
    feat_model = make_feature_extractor(model, num_classes, device=device)

    # Data (no aug)
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

    # Out dir
    out_dir = build_out_dir(args) if args.autoname else Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[save] outputs -> {out_dir.resolve()}")

    # Seeds
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

        Z_feat = run_tsne(feats.numpy(), perplexity=args.perplexity, seed=args.seed, pca_dim=50)
        pd.DataFrame({"x": Z_feat[:,0], "y": Z_feat[:,1],
                      "label": labels_sub.numpy().astype(int),
                      "space": "pre_fc", "split": name}).to_csv(out_dir / f"{name}_tsne_pre_fc.csv", index=False)
        
        scatter_tsne(
            Z_feat, labels_sub, f"",
            str(out_dir / f"{name}_tsne_pre_fc.png"),
            forget_classes=forget_classes_plot
        )

        Z_prob = run_tsne(probs.numpy(), perplexity=args.perplexity, seed=args.seed, pca_dim=50)
        pd.DataFrame({"x": Z_prob[:,0], "y": Z_prob[:,1],
                      "label": labels_sub.numpy().astype(int),
                      "space": "prob", "split": name}).to_csv(out_dir / f"{name}_tsne_prob.csv", index=False)

        scatter_tsne(
            Z_prob, labels_sub, f"",
            str(out_dir / f"{name}_tsne_prob.png"),
            forget_classes=forget_classes_plot
        )


    if args.split in ("train", "both"): process_split("train", train_loader)
    if args.split in ("test", "both"):  process_split("test",  test_loader)

if __name__ == "__main__":
    main()
