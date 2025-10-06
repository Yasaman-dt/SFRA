import torch
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset, Subset
import os
import pandas as pd
import argparse
import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import (
    get_transforms, get_dataset, get_dataloader, get_unlearn_loader, _top1_and_per_class
)
from trainer import *  # assumes load_model is here
import random
import math
from copy import deepcopy
from method.utils import *
from pathlib import Path

# ------------------ Argparse ------------------
import argparse

parser = argparse.ArgumentParser("Class unlearning revival")
parser.add_argument('--method', type=str, default='original',
                    choices=['original','retrained','random_label','finetune','gradient_ascent','neggrad_plus',
                             'boundary_shrink','boundary_expand','l2ul_adv','l2ul_imp','fisher','wood_fisher','delete',
                             'bad_teacher', 'salun', 'scrub'])
# accept both --model and --model_name, store in model_name
parser.add_argument('--model', '--model_name', dest='model_name', type=str, default='resnet18')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument(
    '--forget',
    type=str,
    default='all',
    help=(
        "Which class(es) to forget. Options:\n"
        "  - 'all' (default) → run 0..num_classes-1\n"
        "  - a single int, e.g. '7'\n"
        "  - a comma list, e.g. '1,3,5'\n"
        "  - a range, e.g. '2-6'\n"
        "You can also combine comma lists and ranges: '0,2-4,7'"
    )
)
parser.add_argument(
    '--tpr', type=int, default=5000,
    help='How many synthetic embeddings to *generate* per class before top-K filtering.'
)
parser.add_argument(
    '--cpr', type=int, default=100)

args = parser.parse_args()

method       = args.method
model_name   = args.model_name
dataset_name = args.dataset
lr           = args.lr
epochs       = args.epochs
total_per_class = args.tpr 
choose_per_retain_class_for_fgt = args.cpr           


# set num_classes from dataset
NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'tinyimagenet': 200, 'imagenet': 1000}
try:
    num_classes = NUM_CLASSES[dataset_name.lower()]
except KeyError:
    raise ValueError(f"Unknown dataset '{dataset_name}'. Add it to NUM_CLASSES.")


# ------------------ Load Pre-Trained ResNet-18 and Run the Function ------------------
DIR = "/export/livia/home/vision/Zdehghani/classification/exps"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed=42

#dataset_name = "cifar10"
#model_name = "resnet18"
#num_classes = 10
#method="boundary_shrink"  #method: random_label, finetune, gradient_ascent, boundary_shrink, boundary_expand, l2ul_adv, l2ul_imp, fisher, wood_fisher, delete
#lr = 5e-05
#forget_class = 0
#epochs = 50

save_dir1 = os.path.join(DIR, "tsne/tsne_embedding")
save_dir2 = os.path.join(DIR, "tsne/tsne_prob")
os.makedirs(save_dir1, exist_ok=True)
os.makedirs(save_dir2, exist_ok=True)

AGG_CSV_DIR = os.path.join("results", method)
AGG_CSV_PATH = os.path.join(
    AGG_CSV_DIR,
    f"{dataset_name}_{model_name}_unlearned_{method}_revival_by_forget_class_lr{lr}.csv"
)
os.makedirs(AGG_CSV_DIR, exist_ok=True)



COLUMNS = [
    "forget_class", "dataset", "model", "method", "lr",
    "epochs_total", "tpr", "cpr",
    "epoch", "syn_train_loss", "syn_train_acc",
    "syn_total", "syn_retain", "syn_forget",
    "all_train", "all_test",
    "train_fgt", "train_retain", "test_fgt", "test_retain",
]

def append_best_for_class(forget_class: int, best_row: dict):
    """
    Append exactly ONE row per class with a fixed schema.
    """
    if best_row is None:
        print(f"[AGG] No best row to append for class {forget_class}.")
        return

    row = {
        "forget_class": int(forget_class),
        "dataset": dataset_name,
        "model": model_name,
        "method": method,
        "lr": lr,
        "epochs_total": epochs,                         
        "tpr": total_per_class,                          
        "cpr": choose_per_retain_class_for_fgt,    
        **best_row,  # expects keys like epoch, syn_* , all_*, train_*, test_*
    }

    # Enforce column order; missing keys become NaN
    df = pd.DataFrame([row], columns=COLUMNS)

    header_needed = not os.path.exists(AGG_CSV_PATH)

    METRIC_COLS = [
        "epochs_total", "tpr", "cpr",
        "syn_train_loss", "syn_train_acc",
        "syn_total", "syn_retain", "syn_forget",
        "all_train", "all_test",
        "train_fgt", "train_retain", "test_fgt", "test_retain",
    ]

    # after df = pd.DataFrame([row], columns=COLUMNS)
    df[METRIC_COLS] = df[METRIC_COLS].apply(pd.to_numeric, errors="coerce").round(3)

    df.to_csv(
        AGG_CSV_PATH,
        mode="a",
        header=header_needed,
        index=False,
        float_format="%.3f"
    )

    print(f"[AGG] Appended best row for class {forget_class} -> {AGG_CSV_PATH}")


def checkpoint_for(method, dataset_name, model_name, forget_class, lr, base_dir):
    if method == 'original':
        return f"{base_dir}/test_pretrained_model/{dataset_name}_{model_name}_original_model.pth"
    elif method == 'retrained':
        return f"{base_dir}/test_pretrained_model/{dataset_name}_{model_name}_retrain_forgetcls{forget_class}_model.pth"
    else:
        return f"{base_dir}/{dataset_name}_{model_name}_forgetcls{forget_class}/{method}/lr{lr}/ckpt_best_by_aus.pth"


def make_feature_extractor(net, num_classes):
    """
    Returns a copy of `net` with the final classification head removed,
    so its forward(x) returns the pre-FC embedding.
    Works for torchvision resnet-style models and most nets that end in nn.Linear.
    """
    feat_net = deepcopy(net).eval().to(device)

    # Common fast-paths
    if hasattr(feat_net, "fc") and isinstance(feat_net.fc, nn.Linear) and feat_net.fc.out_features == num_classes:
        feat_net.fc = nn.Identity()
        return feat_net
    if hasattr(feat_net, "head") and isinstance(feat_net.head, nn.Linear) and feat_net.head.out_features == num_classes:
        feat_net.head = nn.Identity()
        return feat_net
    if hasattr(feat_net, "heads") and hasattr(feat_net.heads, "head") and \
       isinstance(feat_net.heads.head, nn.Linear) and feat_net.heads.head.out_features == num_classes:
        feat_net.heads.head = nn.Identity()
        return feat_net

    # Generic fallback
    last_linear = None
    for name, m in reversed(list(feat_net.named_modules())):
        if isinstance(m, nn.Linear) and m.out_features == num_classes:
            last_linear = name
            break
    if last_linear is None:
        raise RuntimeError("Cannot find final Linear layer to strip for feature extractor.")

    def set_module(root, dotted, new):
        parts = dotted.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new)

    set_module(feat_net, last_linear, nn.Identity())
    return feat_net



# ---- one-time embedding pass + fast embedding loaders ----


def make_emb_loader(x, y, bs=4096, shuffle=False, num_workers=4):
    ds  = TensorDataset(x, y)                 # x,y are CPU tensors now
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle, drop_last=False,
        pin_memory=(device == 'cuda'),        # OK: pin CPU → faster H2D
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=4
    )



@torch.inference_mode()
def eval_accuracy_on_emb_loader(fc, loader, device):
    total, correct = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = fc(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total   += y.numel()
    return 100.0 * correct / max(total, 1)


# ------------------ Synthetic pre-FC embeddings + selection (retain-only forget set) ------------------
def _get_final_linear(model, num_classes):
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear) and module.out_features == num_classes:
            return module
    raise RuntimeError("Could not find final Linear layer with out_features == num_classes.")




@torch.no_grad()
def _sample_predicted_as_class(
    W, b, emb_dim, target_class, per_class_strict=1000, batch=4096, device="cuda"
):
    """
    Rejection sample synthetic embeddings so that the classifier (W,b) predicts them
    as `target_class`. Returns exactly per_class_strict embeddings and their probs
    for the target_class.
    """
    feats_buf = []
    probs_buf = []

    while sum(t.shape[0] for t in feats_buf) < per_class_strict:
        # sample a batch of random pre-FC embeddings
        feats = torch.randn(batch, emb_dim, device=device)
        logits = feats @ W.T + b
        probs  = F.softmax(logits, dim=1)

        preds = probs.argmax(dim=1)
        mask  = (preds == target_class)
        if mask.any():
            feats_buf.append(feats[mask])
            probs_buf.append(probs[mask, target_class])  # confidence for the class

    feats_cat = torch.cat(feats_buf, dim=0)
    probs_cat = torch.cat(probs_buf, dim=0)

    # trim to exactly per_class_strict (keep the first chunk; we’ll sort later anyway)
    if feats_cat.shape[0] > per_class_strict:
        feats_cat = feats_cat[:per_class_strict]
        probs_cat = probs_cat[:per_class_strict]

    return feats_cat, probs_cat


def build_synthetic_embeddings_and_splits(
    model, num_classes, forget_class, device,
    per_class, retain_top_k, per_retain_for_forget, loader_batch_size=256
):
    fc = _get_final_linear(model, num_classes)
    emb_dim = fc.in_features
    W = fc.weight.detach()
    b = fc.bias.detach() if fc.bias is not None else torch.zeros(num_classes, device=W.device)

    retain_classes = [c for c in range(num_classes) if c != forget_class]

    # -------- Retain split: EXACT per_class samples predicted as their class, then top-K by confidence --------
    retain_feats_list, retain_labels_list = [], []
    for c in retain_classes:
        feats_c, probs_c = _sample_predicted_as_class(
            W=W, b=b, emb_dim=emb_dim, target_class=c,
            per_class_strict=per_class, batch=4096, device=device
        )
        # sort by model confidence for class c (desc), then take top-K
        conf_sorted, idx_sorted = torch.sort(probs_c, descending=True)
        topk = idx_sorted[:retain_top_k]
        retain_feats_list.append(feats_c.index_select(0, topk))
        retain_labels_list.append(torch.full((retain_top_k,), c, device=device, dtype=torch.long))

    retain_feats  = torch.cat(retain_feats_list, dim=0)
    retain_labels = torch.cat(retain_labels_list, dim=0)

    # -------- Forget split (unchanged, pick bottom-K of retain confidence, relabel as forget_class) --------
    # If you want forgets to be strictly "not predicted as their own class", keep your old logic.
    # Below: for each retain class, resample the SAME 1000 pool and take the *lowest* confidences.
    forget_feats_list = []
    for c in retain_classes:
        feats_c, probs_c = _sample_predicted_as_class(
            W=W, b=b, emb_dim=emb_dim, target_class=c,
            per_class_strict=per_class, batch=4096, device=device
        )
        # take the smallest confidences for class c
        _, idx_sorted_asc = torch.sort(probs_c, descending=False)
        lowk = idx_sorted_asc[:per_retain_for_forget]
        forget_feats_list.append(feats_c.index_select(0, lowk))

    forget_feats  = torch.cat(forget_feats_list, dim=0)
    forget_labels = torch.full((forget_feats.shape[0],), forget_class, device=device, dtype=torch.long)

    retain_ds = TensorDataset(retain_feats.cpu(), retain_labels.cpu())
    forget_ds  = TensorDataset(forget_feats.cpu(),  forget_labels.cpu())
    retain_loader = DataLoader(retain_ds, batch_size=loader_batch_size, shuffle=True, drop_last=False)
    forget_loader  = DataLoader(forget_ds,  batch_size=loader_batch_size, shuffle=True, drop_last=False)

    summary = {
        "emb_dim": emb_dim,
        "retain_classes": retain_classes,
        "retain_per_class_generated": per_class,
        "retain_top_k": retain_top_k,
        "retain_total": retain_feats.shape[0],
        "forget_class": forget_class,
        "per_retain_for_forget": per_retain_for_forget,
        "forget_total": forget_feats.shape[0],
    }
    return {
        "retain_feats": retain_feats, "retain_labels": retain_labels, "retain_loader": retain_loader,
        "forget_feats": forget_feats, "forget_labels": forget_labels, "forget_loader": forget_loader,
        "summary": summary
    }

@torch.no_grad()
def eval_accuracy_on_loader(model, fc, loader, device):
    """
    Works with loaders that yield either:
    - (images, labels): runs full model forward
    - (pre_fc_embeddings, labels): applies fc directly
    """
    model.eval()
    total, correct = 0, 0
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0].to(device), batch[1].to(device)
        else:
            # Unexpected batch format
            continue

        # Heuristic: if input is 4D (B,C,H,W), treat as images; else treat as pre-FC embeddings
        if x.dim() == 4:
            logits = model(x)                         # full forward to logits
        else:
            logits = fc(x)                            # embeddings -> logits via FC only

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)



# --- determinism ---
seed = 0
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
g = torch.Generator(device="cpu").manual_seed(seed)

def _parse_forget_arg(s: str, num_classes: int):
    s = (s or 'all').strip().lower()
    if s in ('all', '-1'):
        return list(range(num_classes))

    # allow "7", "1,3,5", "2-6", or combos like "0,2-4,7"
    selected = set()
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
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

    if not selected:
        raise ValueError(f"No valid classes parsed from --forget='{s}'. "
                         f"Valid range is 0..{num_classes-1} or 'all'.")
    return sorted(selected)

forget_classes = _parse_forget_arg(args.forget, num_classes)

for forget_class in forget_classes:
    print(f"\n================= FORGET CLASS {forget_class} =================")

    # per-class experiment folder
    experiment_path = Path(f"results/{method}/plots_{model_name}_lr{lr}/forget_class_{forget_class}")
    experiment_path.mkdir(parents=True, exist_ok=True)

    # keep curves from epoch 0 baseline onward
    accs_curves = {
        "train_forget": [],  # %
        "test_forget":  [],
        "train_remain": [],
        "test_remain":  [],
    }

    ckpt = checkpoint_for(method, dataset_name, model_name, forget_class, lr, DIR)
    model = load_model(ckpt, model_name, num_classes).to(device).eval()

    # --- transforms & datasets ---
    wo_dataaug = False
    transform_train, transform_test = get_transforms(dataset_name, model_name, wo_dataaug=wo_dataaug)
    trainset, testset = get_dataset(dataset_name, transform_train, transform_test)

    # --- base loaders (with augmentation on train) ---
    batch_size_real = 256
    num_workers = 8
    all_train_loader, all_test_loader = get_dataloader(
        trainset, testset, batch_size=batch_size_real, num_workers=num_workers
    )

    # --- deterministic train-eval loader (NO AUG, no shuffle) ---
    trainset_eval = copy.copy(trainset)
    trainset_eval.transform = transform_test
    all_train_loader = DataLoader(
        trainset_eval,
        batch_size=batch_size_real, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=True, generator=g
    )


    # --- retain/forget splits for the chosen forget_class ---
    # one-vs-all: pass [forget_class]
    (
        train_fgt_loader, train_retain_loader,
        test_fgt_loader,  test_retain_loader,
        repair_class_loader,
        train_fgt_idx, train_retain_idx,
        test_fgt_idx,  test_retain_idx,
    ) = get_unlearn_loader(
        trainset, testset, [forget_class],
        batch_size=batch_size_real, num_workers=num_workers, num_forget=float("inf")
    )


    feature_model = make_feature_extractor(model, num_classes)
    feature_model.eval()

    @torch.inference_mode()
    def embed_loader_to_tensor(loader):
        feats, labels = [], []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            f = feature_model(x)                   
            feats.append(f.detach().cpu())          
            labels.append(y.cpu())                  
        return torch.cat(feats, dim=0), torch.cat(labels, dim=0)



    # ---- embed each real loader ONCE ----
    all_train_feats, all_train_labels = embed_loader_to_tensor(all_train_loader)
    all_test_feats,  all_test_labels  = embed_loader_to_tensor(all_test_loader)

    train_fgt_feats, train_fgt_labels = embed_loader_to_tensor(train_fgt_loader)
    train_ret_feats, train_ret_labels = embed_loader_to_tensor(train_retain_loader)
    test_fgt_feats,  test_fgt_labels  = embed_loader_to_tensor(test_fgt_loader)
    test_ret_feats,  test_ret_labels  = embed_loader_to_tensor(test_retain_loader)

    # (optional) move long-lived cached embeddings to CPU to free GPU RAM
    # all_train_feats = all_train_feats.cpu(); all_test_feats = all_test_feats.cpu()
    # ... same for the rest

    # ---- wrap as fast embedding loaders (no backbone in the loop) ----
    all_train_emb_loader = make_emb_loader(all_train_feats, all_train_labels)
    all_test_emb_loader  = make_emb_loader(all_test_feats,  all_test_labels)
    train_fgt_emb_loader = make_emb_loader(train_fgt_feats, train_fgt_labels)
    train_ret_emb_loader = make_emb_loader(train_ret_feats, train_ret_labels)
    test_fgt_emb_loader  = make_emb_loader(test_fgt_feats,  test_fgt_labels)
    test_ret_emb_loader  = make_emb_loader(test_ret_feats,  test_ret_labels)


    # ---- run it ----
    synth = build_synthetic_embeddings_and_splits(
        model=model,
        num_classes=num_classes,
        forget_class=forget_class,
        device=device,
        per_class=total_per_class,
        retain_top_k=choose_per_retain_class_for_fgt*9,
        per_retain_for_forget=choose_per_retain_class_for_fgt,   # 10 from each retain class → e.g., 9*10 = 90 total for CIFAR-10
        loader_batch_size=256,
    )

    print("Synthetic selection summary:", synth["summary"])
    # Access: synth["retain_loader"], synth["forget_loader"], etc.

    # === after you build `synth` ===
    syn_retain_eval_loader = DataLoader(TensorDataset(synth["retain_feats"].cpu(),
                                                    synth["retain_labels"].cpu()),
                                        batch_size=1024, shuffle=False, drop_last=False)
    syn_forget_eval_loader = DataLoader(TensorDataset(synth["forget_feats"].cpu(),
                                                    synth["forget_labels"].cpu()),
                                        batch_size=1024, shuffle=False, drop_last=False)

    n_syn_ret = len(syn_retain_eval_loader.dataset)
    n_syn_fgt = len(syn_forget_eval_loader.dataset)

    # (optional) keep a history list
    metrics_history = []




    # 1) Freeze backbone; train only FC
    for p in model.parameters():
        p.requires_grad = False
    fc = _get_final_linear(model, num_classes)
    for p in fc.parameters():
        p.requires_grad = True
    model.eval()  # backbone is frozen; we won't backprop through it

    # 2) Train set from synthetic data (retain + forget)
    train_feats = torch.cat([synth["retain_feats"], synth["forget_feats"]], dim=0)
    train_labels = torch.cat([synth["retain_labels"], synth["forget_labels"]], dim=0)
    train_loader_syn = DataLoader(
        list(zip(train_feats, train_labels)), batch_size=256, shuffle=True, drop_last=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fc.parameters(), lr=1e-2, weight_decay=1e-4)
    # optimizer = optim.SGD(
    #     fc.parameters(),
    #     lr=1e-2,
    #     momentum=0.9,
    #     weight_decay=1e-4,
    #     nesterov=True
    # )

    # ----- Baseline BEFORE any training (epoch 0) -----
    fc.eval()



    # --- keep this once, above the function (replace your current `best = {...}`) ---
    best = {"key": (float("-inf"), float("-inf"), float("-inf"), float("-inf")), "row": None}

    def _safe_get(row, k):
        v = row.get(k, float("-inf"))
        return v if (isinstance(v, (int, float)) and v == v) else float("-inf")

    def _key_from_row(row):
        # Order of priority for “best”: test_fgt, then test_retain, then train_fgt, then train_retain
        return (
            float(_safe_get(row, "test_fgt")),
            float(_safe_get(row, "test_retain")),
            float(_safe_get(row, "train_fgt")),
            float(_safe_get(row, "train_retain")),
        )


    def save_best_revival_result(row_dict):
        """
        Update in-memory 'best' by (test_fgt, test_retain, train_fgt, train_retain).
        We only WRITE TO DISK ONCE per forget_class, after the epoch loop.
        """
        key = (
            float(_safe_get(row_dict, "test_fgt")),
            float(_safe_get(row_dict, "test_retain")),
            float(_safe_get(row_dict, "train_fgt")),
            float(_safe_get(row_dict, "train_retain")),
        )
        if key > best["key"]:  # lexicographic tuple comparison
            best["key"] = key
            best["row"] = row_dict.copy()

    # synthetic eval
    acc_syn_ret   = eval_accuracy_on_loader(model, fc, syn_retain_eval_loader, device)
    acc_syn_fgt   = eval_accuracy_on_loader(model, fc, syn_forget_eval_loader, device)
    acc_syn_total = (acc_syn_ret * n_syn_ret + acc_syn_fgt * n_syn_fgt) / (n_syn_ret + n_syn_fgt)

    # real loaders eval
    with torch.no_grad():
        # acc_all_train    = eval_accuracy_on_loader(model, fc, all_train_loader,    device)
        # acc_all_test     = eval_accuracy_on_loader(model, fc, all_test_loader,     device)
        # acc_train_fgt    = eval_accuracy_on_loader(model, fc, train_fgt_loader,    device)
        # acc_train_retain = eval_accuracy_on_loader(model, fc, train_retain_loader, device)
        # acc_test_fgt     = eval_accuracy_on_loader(model, fc, test_fgt_loader,     device)
        # acc_test_retain  = eval_accuracy_on_loader(model, fc, test_retain_loader,  device)

        acc_all_train    = eval_accuracy_on_emb_loader(fc, all_train_emb_loader,    device)
        acc_all_test     = eval_accuracy_on_emb_loader(fc, all_test_emb_loader,     device)
        acc_train_fgt    = eval_accuracy_on_emb_loader(fc, train_fgt_emb_loader,    device)
        acc_train_retain = eval_accuracy_on_emb_loader(fc, train_ret_emb_loader,    device)
        acc_test_fgt     = eval_accuracy_on_emb_loader(fc, test_fgt_emb_loader,     device)
        acc_test_retain  = eval_accuracy_on_emb_loader(fc, test_ret_emb_loader,     device)


    print(
        f"[Epoch 00] "
        f"syn_train_loss=NA syn_train_acc=NA | "
        f"syn_total={acc_syn_total:.2f}% syn_ret={acc_syn_ret:.2f}% syn_fgt={acc_syn_fgt:.2f}% | "
        f"all_train={acc_all_train:.2f}% all_test={acc_all_test:.2f}% | "
        f"train_fgt={acc_train_fgt:.2f}% train_ret={acc_train_retain:.2f}% | "
        f"test_fgt={acc_test_fgt:.2f}% test_ret={acc_test_retain:.2f}%"
    )

    metrics_history.append({
        "epoch": 0,
        "syn_train_loss": None,
        "syn_train_acc": None,
        "syn_total": float(acc_syn_total),
        "syn_retain": float(acc_syn_ret),
        "syn_forget": float(acc_syn_fgt),
        "all_train": float(acc_all_train),
        "all_test": float(acc_all_test),
        "train_fgt": float(acc_train_fgt),
        "train_retain": float(acc_train_retain),
        "test_fgt": float(acc_test_fgt),
        "test_retain": float(acc_test_retain),
    })

    row0 = {
        "epoch": 0,
        "syn_train_loss": None,
        "syn_train_acc": None,
        "syn_total": float(acc_syn_total),
        "syn_retain": float(acc_syn_ret),
        "syn_forget": float(acc_syn_fgt),
        "all_train": float(acc_all_train),
        "all_test": float(acc_all_test),
        "train_fgt": float(acc_train_fgt),
        "train_retain": float(acc_train_retain),
        "test_fgt": float(acc_test_fgt),
        "test_retain": float(acc_test_retain),
    }
    best["row"] = row0.copy()
    best["key"] = _key_from_row(row0)


    # map your naming: remain == retain
    accs_curves["train_forget"].append(float(acc_train_fgt) / 100.0)
    accs_curves["test_forget"].append(float(acc_test_fgt) / 100.0)
    accs_curves["train_remain"].append(float(acc_train_retain) / 100.0)
    accs_curves["test_remain"].append(float(acc_test_retain) / 100.0)

    epoch_for_plot = len(accs_curves["train_forget"])  
    plot_unlearn_remain_acc_figure(
        epoch=epoch_for_plot,
        accs_dict=accs_curves,
        experiment_path=experiment_path,
        plot_type="plot",
    )
    best_test_fgt = acc_test_fgt
    best_train_fgt = acc_train_fgt
    best_test_retain = acc_test_retain
    best_train_retain = acc_train_retain
    best_row = None
    # 3) Train only the FC for a few epochs; evaluate on real loaders each epoch
    for epoch in range(1, epochs + 1):
        fc.train()
        running_loss, seen, right = 0.0, 0, 0
        for feats, targets in train_loader_syn:
            feats = feats.to(device)
            targets = targets.to(device)

            logits = fc(feats)
            loss = criterion(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * feats.size(0)
            preds = logits.argmax(dim=1)
            right += (preds == targets).sum().item()
            seen += targets.numel()

        train_loss = running_loss / max(seen, 1)
        train_acc  = 100.0 * right / max(seen, 1)

        # ----- Synthetic eval (retain / forget / total) -----
        acc_syn_ret = eval_accuracy_on_loader(model, fc, syn_retain_eval_loader, device)
        acc_syn_fgt = eval_accuracy_on_loader(model, fc, syn_forget_eval_loader, device)
        acc_syn_total = (acc_syn_ret * n_syn_ret + acc_syn_fgt * n_syn_fgt) / (n_syn_ret + n_syn_fgt)


        # ----- Eval on your real loaders (images OR real-embeddings) -----
        with torch.no_grad():
            # acc_all_train   = eval_accuracy_on_loader(model, fc, all_train_loader,   device)
            # acc_all_test    = eval_accuracy_on_loader(model, fc, all_test_loader,    device)
            # acc_train_fgt   = eval_accuracy_on_loader(model, fc, train_fgt_loader,   device)
            # acc_train_retain= eval_accuracy_on_loader(model, fc, train_retain_loader,device)
            # acc_test_fgt    = eval_accuracy_on_loader(model, fc, test_fgt_loader,    device)
            # acc_test_retain = eval_accuracy_on_loader(model, fc, test_retain_loader, device)

            acc_all_train    = eval_accuracy_on_emb_loader(fc, all_train_emb_loader,    device)
            acc_all_test     = eval_accuracy_on_emb_loader(fc, all_test_emb_loader,     device)
            acc_train_fgt    = eval_accuracy_on_emb_loader(fc, train_fgt_emb_loader,    device)
            acc_train_retain = eval_accuracy_on_emb_loader(fc, train_ret_emb_loader,    device)
            acc_test_fgt     = eval_accuracy_on_emb_loader(fc, test_fgt_emb_loader,     device)
            acc_test_retain  = eval_accuracy_on_emb_loader(fc, test_ret_emb_loader,     device)


        print(
            f"[Epoch {epoch:02d}] "
            f"syn_train_loss={train_loss:.4f} syn_train_acc={train_acc:.2f}% | "
            f"syn_total={acc_syn_total:.2f}% syn_ret={acc_syn_ret:.2f}% syn_fgt={acc_syn_fgt:.2f}% | "
            f"all_train={acc_all_train:.2f}% all_test={acc_all_test:.2f}% | "
            f"train_fgt={acc_train_fgt:.2f}% train_ret={acc_train_retain:.2f}% | "
            f"test_fgt={acc_test_fgt:.2f}% test_ret={acc_test_retain:.2f}%"
        )


        metrics_history.append({
            "epoch": epoch,
            "syn_train_loss": float(train_loss),
            "syn_train_acc": float(train_acc),
            "syn_total": float(acc_syn_total),
            "syn_retain": float(acc_syn_ret),
            "syn_forget": float(acc_syn_fgt),
            "all_train": float(acc_all_train),
            "all_test": float(acc_all_test),
            "train_fgt": float(acc_train_fgt),
            "train_retain": float(acc_train_retain),
            "test_fgt": float(acc_test_fgt),
            "test_retain": float(acc_test_retain),
        })


        row = {
            "epoch": epoch,
            "syn_train_loss": float(train_loss),
            "syn_train_acc": float(train_acc),
            "syn_total": float(acc_syn_total),
            "syn_retain": float(acc_syn_ret),
            "syn_forget": float(acc_syn_fgt),
            "all_train": float(acc_all_train),
            "all_test": float(acc_all_test),
            "train_fgt": float(acc_train_fgt),
            "train_retain": float(acc_train_retain),
            "test_fgt": float(acc_test_fgt),
            "test_retain": float(acc_test_retain),
        }
        key = _key_from_row(row)
        if key > best["key"]:
            best["key"] = key
            best["row"] = row.copy()


        # append as FRACTIONS
        accs_curves["train_forget"].append(float(acc_train_fgt) / 100.0)
        accs_curves["test_forget"].append(float(acc_test_fgt) / 100.0)
        accs_curves["train_remain"].append(float(acc_train_retain) / 100.0)
        accs_curves["test_remain"].append(float(acc_test_retain) / 100.0)

        # re-plot; epoch must equal the length of your y-series (1..N)
        epoch_for_plot = len(accs_curves["train_forget"]) 
        plot_unlearn_remain_acc_figure(
            epoch=epoch_for_plot,
            accs_dict=accs_curves,
            experiment_path=experiment_path,
            plot_type="plot",
        )

    # # === AFTER the epoch loop finishes, write ONE best row for this forget_class ===
    # best_csv_dir = os.path.join("results", method)
    # best_csv_path = os.path.join(
    #     best_csv_dir,
    #     f"{dataset_name}_{model_name}_unlearned_{method}_revival.csv"
    # )
    # os.makedirs(best_csv_dir, exist_ok=True)

    if best["row"] is not None:
        append_best_for_class(forget_class, best["row"])
    else:
        print(f"[BEST PER CLASS] No best row found for forget_class={forget_class}.")
