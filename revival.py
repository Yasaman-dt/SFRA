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


# # ------------------ Argparse ------------------
# parser.add_argument('--method', type=str, default='original', help='Unlearning method to evaluate (e.g., original, retrained, FT, etc.)')
# parser.add_argument('--model', type=str, default='resnet18', help='Model name (e.g., resnet18, vit, etc.)')
# parser.add_argument('--dataset', type=str)
# parser.add_argument('--lr', type=float)

# args = parser.parse_args()

# method = args.method
# model_name = args.model
# dataset_name = args.dataset
# lr = args.lr

# ------------------ Load Pre-Trained ResNet-18 and Run the Function ------------------
DIR = "/export/livia/home/vision/Zdehghani/classification/exps"

dataset_name = "cifar10"
model_name = "resnet18"
num_classes = 10
method="retrained"  #method: random_label, finetune, gradient_ascent, boundary_shrink, boundary_expand, l2ul_adv, fisher, wood_fisher, delete
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed=42
lr = 5e-5
forget_class = 0
checkpoint_folder = f"{dataset_name}_{model_name}_forgetcls{forget_class}/{method}"

save_dir1 = os.path.join(DIR, "tsne/tsne_embedding")
save_dir2 = os.path.join(DIR, "tsne/tsne_prob")
os.makedirs(save_dir1, exist_ok=True)
os.makedirs(save_dir2, exist_ok=True)


         
print(f"  - Forget class {forget_class}")

if method == 'original':
    checkpoint_path_model = f"{DIR}/test_pretrained_model/{dataset_name}_{model_name}_original_model.pth"
    
elif method == 'retrained':
    checkpoint_path_model = f"{DIR}/test_pretrained_model/{dataset_name}_{model_name}_retrain_forgetcls{forget_class}_model.pth"

else:
    checkpoint_path_model = f"{DIR}/{dataset_name}_{model_name}_forgetcls{forget_class}/{method}/lr{lr}/ckpt_best_by_aus.pth"


model = load_model(checkpoint_path_model, model_name, num_classes)
model.to(device).eval()



# --- determinism ---
seed = 0
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
g = torch.Generator(device="cpu").manual_seed(seed)

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
    per_class=1000, retain_top_k=90, per_retain_for_forget=10, loader_batch_size=256
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

    retain_ds = TensorDataset(retain_feats, retain_labels)
    forget_ds = TensorDataset(forget_feats, forget_labels)
    retain_loader = DataLoader(retain_ds, batch_size=loader_batch_size, shuffle=True, drop_last=False)
    forget_loader = DataLoader(forget_ds, batch_size=loader_batch_size, shuffle=True, drop_last=False)

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





# ---- run it ----
synth = build_synthetic_embeddings_and_splits(
    model=model,
    num_classes=num_classes,
    forget_class=forget_class,
    device=device,
    per_class=1000,
    retain_top_k=180,
    per_retain_for_forget=20,   # 10 from each retain class → e.g., 9*10 = 90 total for CIFAR-10
    loader_batch_size=256,
)

print("Synthetic selection summary:", synth["summary"])
# Access: synth["retain_loader"], synth["forget_loader"], etc.

# === after you build `synth` ===
syn_retain_eval_loader = DataLoader(
    TensorDataset(synth["retain_feats"], synth["retain_labels"]),
    batch_size=1024, shuffle=False, drop_last=False
)
syn_forget_eval_loader = DataLoader(
    TensorDataset(synth["forget_feats"], synth["forget_labels"]),
    batch_size=1024, shuffle=False, drop_last=False
)
n_syn_ret = len(syn_retain_eval_loader.dataset)
n_syn_fgt = len(syn_forget_eval_loader.dataset)

# (optional) keep a history list
metrics_history = []



def _get_final_linear(model, num_classes):
    for _, m in reversed(list(model.named_modules())):
        if isinstance(m, nn.Linear) and m.out_features == num_classes:
            return m
    raise RuntimeError("No final nn.Linear with out_features == num_classes found.")

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


# ----- Baseline BEFORE any training (epoch 0) -----
fc.eval()


# where to save the best revival metrics
best_csv_dir = os.path.join("results", method)
best_csv_path = os.path.join(
    best_csv_dir,
    f"{dataset_name}_{model_name}_unlearned_{method}_revival.csv"
)
os.makedirs(best_csv_dir, exist_ok=True)

best_test_fgt = -math.inf
best_train_fgt = -math.inf
best_row = None



def save_best_revival_result(row_dict):
    """
    Keep the row with the highest test_fgt.
    Break ties with higher train_fgt.
    Save the single best row to CSV every time it improves.
    """
    global best_test_fgt, best_train_fgt, best_row
    test_fgt = row_dict.get("test_fgt", float("-inf"))
    train_fgt = row_dict.get("train_fgt", float("-inf"))

    is_better = (test_fgt > best_test_fgt) or (
        test_fgt == best_test_fgt and train_fgt > best_train_fgt
    )
    if is_better:
        best_test_fgt = test_fgt
        best_train_fgt = train_fgt
        best_row = row_dict.copy()
        # write single-row CSV
        pd.DataFrame([best_row]).to_csv(best_csv_path, index=False)
        print(f"[BEST UPDATE] Saved best revival to {best_csv_path} | "
              f"test_fgt={test_fgt:.2f} train_fgt={train_fgt:.2f}")



# synthetic eval
acc_syn_ret   = eval_accuracy_on_loader(model, fc, syn_retain_eval_loader, device)
acc_syn_fgt   = eval_accuracy_on_loader(model, fc, syn_forget_eval_loader, device)
acc_syn_total = (acc_syn_ret * n_syn_ret + acc_syn_fgt * n_syn_fgt) / (n_syn_ret + n_syn_fgt)

# real loaders eval
with torch.no_grad():
    acc_all_train    = eval_accuracy_on_loader(model, fc, all_train_loader,    device)
    acc_all_test     = eval_accuracy_on_loader(model, fc, all_test_loader,     device)
    acc_train_fgt    = eval_accuracy_on_loader(model, fc, train_fgt_loader,    device)
    acc_train_retain = eval_accuracy_on_loader(model, fc, train_retain_loader, device)
    acc_test_fgt     = eval_accuracy_on_loader(model, fc, test_fgt_loader,     device)
    acc_test_retain  = eval_accuracy_on_loader(model, fc, test_retain_loader,  device)

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
save_best_revival_result(metrics_history[-1])



# 3) Train only the FC for a few epochs; evaluate on real loaders each epoch
epochs = 50
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
        acc_all_train   = eval_accuracy_on_loader(model, fc, all_train_loader,   device)
        acc_all_test    = eval_accuracy_on_loader(model, fc, all_test_loader,    device)
        acc_train_fgt   = eval_accuracy_on_loader(model, fc, train_fgt_loader,   device)
        acc_train_retain= eval_accuracy_on_loader(model, fc, train_retain_loader,device)
        acc_test_fgt    = eval_accuracy_on_loader(model, fc, test_fgt_loader,    device)
        acc_test_retain = eval_accuracy_on_loader(model, fc, test_retain_loader, device)

    print(
        f"[Epoch {epoch:02d}] "
        f"syn_train_loss={train_loss:.4f} syn_train_acc={train_acc:.2f}% | "
        f"syn_total={acc_syn_total:.2f}% syn_ret={acc_syn_ret:.2f}% syn_fgt={acc_syn_fgt:.2f}% | "
        f"all_train={acc_all_train:.2f}% all_test={acc_all_test:.2f}% | "
        f"train_fgt={acc_train_fgt:.2f}% train_ret={acc_train_retain:.2f}% | "
        f"test_fgt={acc_test_fgt:.2f}% test_ret={acc_test_retain:.2f}%"
    )

    # (optional) store metrics
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
    save_best_revival_result(metrics_history[-1])


# (optional) save metrics
if metrics_history:
    os.makedirs(f"{DIR}/logs", exist_ok=True)
    pd.DataFrame(metrics_history).to_csv(
        f"{DIR}/logs/fc_syn_finetune_metrics_{dataset_name}_{model_name}_forget{forget_class}.csv",
        index=False
    )
