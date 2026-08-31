"""
t-SNE of real samples: pre-FC embeddings vs. post-FC probabilities.

Two ways to pick a checkpoint for A (unlearned/current):
  A) Direct path: --ckpt /path/to/model.pth
  B) Revival-style construction (same as revival_single_class):
     --base_dir DIR --method {original,retrained,...} --lr 5e-5 --forget_class 0

Outputs are auto-named if --autoname is set:
  <out_root>/<dataset>_<model>_<method>_f<forget_class>/

Also supports --forget "0,2-4,7" to produce forget-only / retain-only plots.

Model B (revival/compare) is resolved IN-CODE via:
  results/{method}/{dataset}_{architecture}_{method}_fg{forget_Class}_lr{lr}_best.pth
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
import matplotlib as mpl


import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from utils import get_transforms, get_dataset
from project_paths import EXPS_DIR
from trainer import *  # must provide load_model(ckpt, model_name, dataset, num_classes)

# ------------------- constants -------------------

NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'tiny_imagenet': 200, 'imagenet': 1000}

# ------------------- revival-style checkpoint helper (A) -------------------

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

# ------------------- fixed results-layout helper (B) -------------------

RESULTS_ROOT = "results"  # change if your results live elsewhere

def results_checkpoint_for(method: str, dataset: str, architecture: str,
                           forget_class: int, lr: float, root: str = RESULTS_ROOT) -> str:
    # Keep LR string exactly as stored on disk
    lr_str = str(lr)
    return f"{root}/{method}/{dataset}_{architecture}_{method}_fg{forget_class}_lr{lr_str}_best.pth"

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
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=1000,              # <- updated from n_iter
        learning_rate="auto",
        init="pca",
        random_state=seed,
        metric="euclidean",
        verbose=1
    )
    return tsne.fit_transform(X)

def scatter_tsne(
    Z,
    labels,
    title,
    out_png,
    alpha=0.7,
    s=6,
    classnames=None,
    hide_box=True
):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=600)

    K = int(labels.max().item() + 1) if labels.numel() > 0 else 0
    # fallback to numeric names if custom names are missing/misaligned
    if not classnames or len(classnames) < K:
        classnames = [str(i) for i in range(K)]

    for c in range(K):
        m = (labels == c).numpy()
        if m.sum() == 0:
            continue
        ax.scatter(Z[m, 0], Z[m, 1], label=classnames[c], alpha=alpha, s=s)

    # Remove axes spines/ticks/labels for a clean look
    if hide_box:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_tsne_with_shapes_and_colors(
    Z: np.ndarray,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    forget_set: set,
    out_png: str,
    classnames=None,
    alpha: float = 0.7,
    s: int = 10,
    title: str | None = None,
):
    """
    Shapes:
      - retain (true label NOT in forget_set): circle 'o'
      - forget (true label IN forget_set): triangle '^'
    Colors:
      - retain: color by TRUE label
      - forget: color by PRED label
    """
    # tensors -> cpu numpy
    y_true = true_labels.detach().cpu().numpy().astype(int)
    y_pred = pred_labels.detach().cpu().numpy().astype(int)

    K = int(max(y_true.max(), y_pred.max()) + 1) if y_true.size > 0 else 0
    if not classnames or len(classnames) < K:
        classnames = [str(i) for i in range(K)]

    # discrete colormap with K distinct colors
    cmap = mpl.cm.get_cmap('tab10', K)
    def color_from_class(c): return cmap(c)

    is_forget = np.isin(y_true, list(forget_set))
    is_retain = ~is_forget

    # Colors per point
    retain_colors = [color_from_class(c) for c in y_true[is_retain]]   # retain: color by TRUE
    forget_colors = [color_from_class(c) for c in y_pred[is_forget]]   # forget: color by PRED

    fig, ax = plt.subplots(figsize=(6, 6), dpi=600)

    if is_retain.sum() > 0:
        ax.scatter(Z[is_retain, 0], Z[is_retain, 1], s=s, alpha=alpha, marker='o',
                   linewidths=0, c=retain_colors, label="retain (true-color)")
    if is_forget.sum() > 0:
        ax.scatter(Z[is_forget, 0], Z[is_forget, 1], s=max(s, 12), alpha=alpha, marker='^',
                   linewidths=0, c=forget_colors, label="forget (pred-color)")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    if title:
        ax.set_title(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def build_out_dir(args) -> Path:
    # Auto-name: <out_root>/<dataset>_<model>_<method>_f<forget_class>/
    name = f"{args.dataset}_{args.model_name}_{args.method}_f{args.forget_class}_framework"
    return Path(args.out_root) / name

# ------------------- main -------------------
def _get_final_linear_module(model, num_classes):
    # try common names first
    for attr in ("fc", "head"):
        if hasattr(model, attr):
            m = getattr(model, attr)
            if isinstance(m, torch.nn.Linear) and m.out_features == num_classes:
                return m
    # fallback: search the last Linear with matching out_features
    last_lin = None
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == num_classes:
            last_lin = m
    if last_lin is None:
        raise RuntimeError("Could not find final Linear with out_features == num_classes.")
    return last_lin

def main():
    ap = argparse.ArgumentParser("t-SNE of real samples (pre-FC vs probabilities)")
    # Way A: direct checkpoint
    ap.add_argument("--ckpt", help="Direct path to checkpoint .pth/.pt for load_model")

    # Way B: build like revival_single_class
    ap.add_argument("--base_dir", default=str(EXPS_DIR),
                    help="Base DIR used by revival_single_class to store checkpoints")
    ap.add_argument("--method", default="original",
                    choices=['original','retrained','random_label','finetune','gradient_ascent','neggrad_plus',
                             'boundary_shrink','boundary_expand','l2ul_adv','l2ul_imp','fisher',
                             'wood_fisher','delete','bad_teacher','salun','scrub'],
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

    # Toggle to disable revival usage entirely
    ap.add_argument("--no_revival", action="store_true",
                    help="If set, skip loading/using the revival checkpoint and only analyze the unlearned model.")

    # args
    args = ap.parse_args()

    # device / meta
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = NUM_CLASSES[args.dataset]
    forget_classes_plot = parse_class_list(args.forget, num_classes)

    # Resolve checkpoint A (unlearned/current)
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
    print(f"Using unlearned model checkpoint: {ckpt_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Unlearned Model Checkpoint does not exist: {ckpt_path}")

    # Resolve checkpoint B (revival/compare) IN CODE via results/...
    ckpt_path_b = results_checkpoint_for(
        method=args.method,
        dataset=args.dataset,
        architecture=args.model_name,
        forget_class=int(args.forget_class),
        lr=args.lr,
        root=RESULTS_ROOT
    )
    print(f"Using revival checkpoint: {ckpt_path_b}")
    if not args.no_revival:
        if not os.path.isfile(ckpt_path_b):
            raise FileNotFoundError(f"Revival Checkpoint does not exist: {ckpt_path_b}")
    else:
        print("[revival] Skipped by --no_revival; proceeding with unlearned-only analysis.")

    # Load model & feature extractor (A)
    model = load_model(ckpt_path, args.model_name, args.dataset, num_classes).to(device).eval()
    feat_model = make_feature_extractor(model, num_classes, device=device)

    # Build model_b for revival only if enabled
    if not args.no_revival:
        model_b = load_model(ckpt_path, args.model_name, args.dataset, num_classes).to(device).eval()
        rev = torch.load(ckpt_path_b, map_location="cpu")
        if "fc_state_dict" not in rev:
            raise RuntimeError(f"Revival checkpoint {ckpt_path_b} has no 'fc_state_dict' key.")
        fc_module = _get_final_linear_module(model_b, num_classes)
        sd = rev["fc_state_dict"]
        w = sd.get("weight", None)
        if w is not None and hasattr(fc_module, "in_features"):
            if w.shape[-1] != fc_module.in_features or w.shape[0] != fc_module.out_features:
                raise RuntimeError(
                    f"Revival FC shape {tuple(w.shape)} "
                    f"does not match model head ({fc_module.out_features},{fc_module.in_features})."
                )
        fc_module.load_state_dict(sd, strict=True)
        model_b.eval()

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

    @torch.inference_mode()
    def collect_probs_only(model, loader, device: str):
        model.eval()
        probs_list, labels_list = [], []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            p = F.softmax(logits, dim=1)
            probs_list.append(p.detach().cpu())
            labels_list.append(y.detach().cpu())
        return torch.cat(probs_list), torch.cat(labels_list)

    def process_split(name: str, loader: DataLoader):
        print(f"[t-SNE] Collecting features/probabilities for split: {name}")
        feats, probs, labels = collect_real_embeddings_and_probs(model, feat_model, loader, device)

        keep = balanced_indices(labels, args.max_per_class)
        feats = feats.index_select(0, keep)
        probs = probs.index_select(0, keep)
        labels_sub = labels.index_select(0, keep)

        torch.save({"feats": feats, "probs": probs, "labels": labels_sub}, out_dir / f"{name}_raw_pre_fc.pt")

        Z_feat = run_tsne(feats.numpy(), perplexity=args.perplexity, seed=args.seed, pca_dim=50)
        pd.DataFrame({"x": Z_feat[:,0], "y": Z_feat[:,1],
                      "label": labels_sub.numpy().astype(int),
                      "space": "pre_fc", "split": name}).to_csv(out_dir / f"{name}_tsne_pre_fc.csv", index=False)
        scatter_tsne(
            Z_feat, labels_sub,
            f"{name} • t-SNE (pre-FC)",
            str(out_dir / f"{name}_tsne_pre_fc.png"),
        )

        Z_prob = run_tsne(probs.numpy(), perplexity=args.perplexity, seed=args.seed, pca_dim=50)
        pd.DataFrame({"x": Z_prob[:,0], "y": Z_prob[:,1],
                      "label": labels_sub.numpy().astype(int),
                      "space": "prob", "split": name}).to_csv(out_dir / f"{name}_tsne_prob_unlearned.csv", index=False)
        scatter_tsne(
            Z_prob, labels_sub,
            f"{name} • t-SNE (probabilities)",
            str(out_dir / f"{name}_tsne_prob_unlearned.png"),
        )

        # ---- Define forget set once (used throughout this function) ----
        forget_shape_set = set(forget_classes_plot) if len(forget_classes_plot) > 0 else {int(args.forget_class)}

        # predictions for unlearned model on the kept subset
        pred_unlearned = probs.argmax(dim=1)

        # ---- Confidence exports (UNLEARNED ONLY) ----
        is_forget = torch.zeros_like(labels_sub, dtype=torch.bool)
        for c in forget_shape_set:
            is_forget |= (labels_sub == c)
        is_retain = ~is_forget


        # ---- Subset probability dumps (UNLEARNED) ----
        # Save the order of rows in each subset, to align with the .npy arrays
        row_idx = torch.arange(labels_sub.numel())
        idx_forget = row_idx[is_forget].numpy()
        idx_retain = row_idx[is_retain].numpy()







        # Full probability vectors for each subset
        forget_probs_un = probs[is_forget].numpy()   # shape: [n_forget, K]
        retain_probs_un = probs[is_retain].numpy()   # shape: [n_retain, K]

        # Persist as .npy + matching index .csv (so you can join with other tables)
        np.save(out_dir / f"{name}_forget_probs_unlearned.npy", forget_probs_un)
        np.save(out_dir / f"{name}_retain_probs_unlearned.npy", retain_probs_un)

        pd.DataFrame({"row_in_kept_subset": idx_forget}).to_csv(
            out_dir / f"{name}_forget_row_indices_unlearned.csv", index=False
        )
        pd.DataFrame({"row_in_kept_subset": idx_retain}).to_csv(
            out_dir / f"{name}_retain_row_indices_unlearned.csv", index=False
        )

        # (A) Forget samples: confidence of the predicted class
        if is_forget.any():
            conf_forget_pred = probs[is_forget].max(dim=1).values
            df_forget = pd.DataFrame({
                "true_label": labels_sub[is_forget].numpy().astype(int),
                "pred_label": pred_unlearned[is_forget].numpy().astype(int),
                "conf_pred":  conf_forget_pred.numpy()
            })
            out_csv_forget = out_dir / f"{name}_forget_pred_conf_unlearned.csv"
            df_forget.to_csv(out_csv_forget, index=False)
            print(f"[conf] wrote {out_csv_forget} (n={len(df_forget)})")
        else:
            print("[conf] no forget samples in subset.")

        # (B) Retain samples: confidence of the true class
        if is_retain.any():
            true_retain = labels_sub[is_retain]
            conf_retain_true = probs[is_retain].gather(1, true_retain.view(-1,1)).squeeze(1)
            df_retain = pd.DataFrame({
                "true_label": true_retain.numpy().astype(int),
                "pred_label": pred_unlearned[is_retain].numpy().astype(int),
                "conf_true":  conf_retain_true.numpy()
            })
            out_csv_retain = out_dir / f"{name}_retain_true_conf_unlearned.csv"
            df_retain.to_csv(out_csv_retain, index=False)
            print(f"[conf] wrote {out_csv_retain} (n={len(df_retain)})")
        else:
            print("[conf] no retain samples in subset.")

        # ---- Revival comparison (wrapped) ----
        if not args.no_revival:
            probs_b_all, labels_b_all = collect_probs_only(model_b, loader, device)
            probs_b  = probs_b_all.index_select(0, keep)
            labels_b = labels_b_all.index_select(0, keep)

            torch.save({"probs": probs_b, "labels": labels_b}, out_dir / f"{name}_raw_prob_revival.pt")

            Z_prob_b = run_tsne(probs_b.numpy(), perplexity=args.perplexity, seed=args.seed, pca_dim=50)
            pd.DataFrame(
                {"x": Z_prob_b[:,0], "y": Z_prob_b[:,1],
                 "label": labels_b.numpy().astype(int),
                 "space": "prob", "split": name, "model": "revival"}
            ).to_csv(out_dir / f"{name}_tsne_prob_revival.csv", index=False)

            scatter_tsne(
                Z_prob_b, labels_b,
                f"{name} • t-SNE (probabilities • revival)",
                str(out_dir / f"{name}_tsne_prob_revival.png"),
            )

            pred_revival = probs_b.argmax(dim=1)

            # Predicted-class confidence for FORGET samples (both models)
            is_forget_mask = is_forget  # same mask
            if is_forget_mask.any():
                conf_unlearned = probs[is_forget_mask].gather(
                    1, pred_unlearned[is_forget_mask].view(-1, 1)
                ).squeeze(1)
                conf_revival   = probs_b[is_forget_mask].gather(
                    1, pred_revival[is_forget_mask].view(-1, 1)
                ).squeeze(1)

                df_forget_cmp = pd.DataFrame({
                    "true_label":   labels_sub[is_forget_mask].numpy().astype(int),
                    "pred_unlearn": pred_unlearned[is_forget_mask].numpy().astype(int),
                    "conf_unlearn": conf_unlearned.numpy(),
                    "pred_revival": pred_revival[is_forget_mask].numpy().astype(int),
                    "conf_revival": conf_revival.numpy(),
                })
                out_csv = out_dir / f"{name}_forget_pred_conf.csv"
                df_forget_cmp.to_csv(out_csv, index=False)
                print(f"[forget/conf] Saved per-sample confidences -> {out_csv}")

                per_cls = df_forget_cmp.groupby("true_label")[["conf_unlearn","conf_revival"]].mean()
                per_cls_path = out_dir / f"{name}_forget_pred_conf_per_class.csv"
                per_cls.to_csv(per_cls_path)
                print(f"[forget/conf] Per-class means -> {per_cls_path}")
            else:
                print("[forget/conf] No forget samples in this subset; skipped.")

            # shape/color plot with revival preds
            plot_tsne_with_shapes_and_colors(
                Z=Z_feat,
                true_labels=labels_sub,
                pred_labels=pred_revival,
                forget_set=forget_shape_set,
                out_png=str(out_dir / f"{name}_tsne_pre_fc_shapes_revival.png"),
                alpha=0.7, s=25,
            )
        else:
            print("[revival] All revival visualizations skipped by --no_revival.")

        # Unlearned view (always)
        plot_tsne_with_shapes_and_colors(
            Z=Z_feat,
            true_labels=labels_sub,
            pred_labels=pred_unlearned,
            forget_set=forget_shape_set,
            out_png=str(out_dir / f"{name}_tsne_pre_fc_shapes_unlearned.png"),
            alpha=0.7, s=25,
        )

        # Optional per-subset t-SNEs if --forget specified
        if len(forget_classes_plot) > 0:
            is_forget_sub = torch.zeros_like(labels_sub, dtype=torch.bool)
            for c in forget_classes_plot: is_forget_sub |= (labels_sub == c)
            is_retain_sub = ~is_forget_sub

            def _subset(tag, mask):
                n = int(mask.sum().item())
                if n == 0:
                    print(f"[t-SNE] {name}-{tag}: no samples, skipping."); return
                Z = run_tsne(feats[mask].numpy(), perplexity=args.perplexity, seed=args.seed, pca_dim=50)
                pd.DataFrame({"x": Z[:,0], "y": Z[:,1],
                              "label": labels_sub[mask].numpy().astype(int),
                              "space": "pre_fc", "split": name, "subset": tag}).to_csv(
                                  out_dir / f"{name}_tsne_pre_fc_{tag}.csv", index=False)
                scatter_tsne(
                    Z, labels_sub[mask],
                    f"{name} • t-SNE (pre-FC • {tag})",
                    str(out_dir / f"{name}_tsne_pre_fc_{tag}.png"),
                )
                Zp = run_tsne(probs[mask].numpy(), perplexity=args.perplexity, seed=args.seed, pca_dim=50)
                pd.DataFrame({"x": Zp[:,0], "y": Zp[:,1],
                              "label": labels_sub[mask].numpy().astype(int),
                              "space": "prob", "split": name, "subset": tag}).to_csv(
                                  out_dir / f"{name}_tsne_prob_{tag}.csv", index=False)
                scatter_tsne(
                    Zp, labels_sub[mask],
                    f"{name} • t-SNE (prob • {tag})",
                    str(out_dir / f"{name}_tsne_prob_{tag}.png"),
                )

            _subset("forget", is_forget_sub)
            _subset("retain", is_retain_sub)

        print(f"[t-SNE] Done for {name}. Saved outputs to {out_dir.resolve()}")

    if args.split in ("train", "both"): process_split("train", train_loader)
    if args.split in ("test", "both"):  process_split("test",  test_loader)

if __name__ == "__main__":
    main()
