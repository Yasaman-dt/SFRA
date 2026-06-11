import os
import copy
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import get_transforms, get_dataset, get_dataloader, get_unlearn_loader
from trainer import *   # assumes load_model is here


# ------------------ Argparse ------------------
parser = argparse.ArgumentParser("Prototypical Relearning Attack on unlearned checkpoints")

parser.add_argument('--method', type=str, required=True,
                    choices=[
                        'original','retrained','random_label','finetune','gradient_ascent','neggrad_plus',
                        'boundary_shrink','boundary_expand','l2ul_adv','l2ul_imp','fisher',
                        'wood_fisher','delete','bad_teacher','salun','scrub'
                    ])
parser.add_argument('--model', '--model_name', dest='model_name', type=str, default='resnet18')
parser.add_argument('--dataset', type=str, required=True)

parser.add_argument(
    '--forget',
    type=str,
    default='all',
    help=(
        "Which class(es) to attack. Options:\n"
        "  - 'all'\n"
        "  - a single int, e.g. '7'\n"
        "  - a comma list, e.g. '1,3,5'\n"
        "  - a range, e.g. '2-6'\n"
        "  - combos like '0,2-4,7'"
    )
)

parser.add_argument('--lr', type=float, default=5e-5,
                    help='Used only for locating the checkpoint path of unlearning methods.')
parser.add_argument('--attack_k', type=int, default=5,
                    help='Number of real forget samples used to compute the prototype.')
parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'l2'])
parser.add_argument('--retain_drop_tol', type=float, default=1.0,
                    help='Maximum allowed drop in test retain accuracy after attack.')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

method = args.method
model_name = args.model_name
dataset_name = args.dataset
lr = args.lr
attack_k = args.attack_k
metric = args.metric
retain_drop_tol = args.retain_drop_tol
batch_size_real = args.batch_size
seed = args.seed

# Moderate alpha range
ALPHA_GRID = [0.2, 0.4, 0.6, 0.8]

NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'tiny_imagenet': 200, 'imagenet': 1000}
if dataset_name.lower() not in NUM_CLASSES:
    raise ValueError(f"Unknown dataset '{dataset_name}'")
num_classes = NUM_CLASSES[dataset_name.lower()]

DIR = "/export/livia/home/vision/Zdehghani/classification/exps"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ------------------ Helpers ------------------
def _parse_forget_arg(s: str, num_classes: int):
    s = (s or 'all').strip().lower()
    if s in ('all', '-1'):
        return list(range(num_classes))

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
        raise ValueError(f"No valid classes parsed from --forget='{s}'")
    return sorted(selected)


def checkpoint_for(method, dataset_name, model_name, forget_class, lr, base_dir):
    if method == 'original':
        return f"{base_dir}/test_pretrained_model/{dataset_name}_{model_name}_original_model.pth"
    elif method == 'retrained':
        return f"{base_dir}/test_pretrained_model/{dataset_name}_{model_name}_retrain_forgetcls{forget_class}_model.pth"
    else:
        return f"{base_dir}/{dataset_name}_{model_name}_forgetcls{forget_class}/{method}/lr{lr}/ckpt_best_by_aus.pth"


def _get_final_linear(model, num_classes):
    for _, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear) and module.out_features == num_classes:
            return module
    raise RuntimeError("Could not find final Linear layer with out_features == num_classes.")


def make_feature_extractor(net, num_classes):
    feat_net = copy.deepcopy(net).eval().to(device)

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


def make_emb_loader(x, y, bs=4096, shuffle=False, num_workers=4):
    ds = TensorDataset(x, y)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=(device == 'cuda'),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None
    )


@torch.inference_mode()
def embed_loader_to_tensor(feature_model, loader):
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = feature_model(x)
        feats.append(f.detach().cpu())
        labels.append(y.cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


@torch.inference_mode()
def eval_accuracy_on_emb_loader(fc, loader, device):
    total, correct = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = fc(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


@torch.inference_mode()
def sample_k_forget_embeddings(feature_model, forget_loader, k, device):
    xs, ys = [], []
    for x, y in forget_loader:
        xs.append(x)
        ys.append(y)
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)

    if x.size(0) < k:
        raise ValueError(f"Forget loader has only {x.size(0)} samples, but k={k}")

    perm = torch.randperm(x.size(0))[:k]
    xk = x[perm].to(device)
    yk = y[perm].to(device)

    zk = feature_model(xk)
    return zk, yk


@torch.inference_mode()
def apply_prototypical_relearning_attack(
    model,
    feature_model,
    forget_loader,
    forget_class,
    num_classes,
    k=5,
    alpha=0.5,
    metric="cosine",
    device="cuda",
):
    attacked_model = copy.deepcopy(model).to(device).eval()
    attacked_fc = _get_final_linear(attacked_model, num_classes)

    z_forget, _ = sample_k_forget_embeddings(feature_model, forget_loader, k, device)
    proto = z_forget.mean(dim=0)

    w_old = attacked_fc.weight.data[forget_class].clone()
    if attacked_fc.bias is not None:
        b_old = attacked_fc.bias.data[forget_class].clone()
    else:
        b_old = torch.tensor(0.0, device=device)

    if metric == "cosine":
        w_hat = F.normalize(proto.unsqueeze(0), dim=1).squeeze(0)
        b_hat = torch.tensor(0.0, device=device)
    elif metric == "l2":
        w_hat = 2.0 * proto
        b_hat = -torch.norm(proto, p=2).pow(2)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    attacked_fc.weight.data[forget_class] = alpha * w_hat + (1.0 - alpha) * w_old
    if attacked_fc.bias is not None:
        attacked_fc.bias.data[forget_class] = alpha * b_hat + (1.0 - alpha) * b_old

    return attacked_model


# ------------------ Output CSV ------------------
RESULTS_DIR = os.path.join("results", "proto_attack")
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(
    RESULTS_DIR,
    f"{dataset_name}_{model_name}_{method}_proto_attack_test_metrics.csv"
)

COLUMNS = [
    "forget_class", "dataset", "model", "method",
    "attack_k", "metric", "alpha",

    "baseline_train_fgt", "baseline_train_retain",
    "baseline_test_fgt", "baseline_test_retain",

    "pra_train_fgt", "pra_train_retain",
    "pra_test_fgt", "pra_test_retain",

    "delta_train_fgt", "delta_train_retain",
    "delta_test_fgt", "delta_test_retain",
]


def append_row(row):
    df = pd.DataFrame([row], columns=COLUMNS)
    header_needed = not os.path.exists(CSV_PATH)
    df.to_csv(CSV_PATH, mode="a", header=header_needed, index=False, float_format="%.3f")


# ------------------ Main ------------------
forget_classes = _parse_forget_arg(args.forget, num_classes)

for forget_class in forget_classes:
    print(f"\n================= FORGET CLASS {forget_class} =================")

    ckpt = checkpoint_for(method, dataset_name, model_name, forget_class, lr, DIR)
    print(f"Checkpoint: {ckpt}")

    model = load_model(ckpt, model_name, dataset_name, num_classes).to(device).eval()
    feature_model = make_feature_extractor(model, num_classes).eval()

    wo_dataaug = False
    transform_train, transform_test = get_transforms(dataset_name, model_name, wo_dataaug=wo_dataaug)
    trainset, testset = get_dataset(dataset_name, transform_train, transform_test)

    # deterministic eval version of trainset
    trainset_eval = copy.copy(trainset)
    trainset_eval.transform = transform_test

    # loaders
    all_train_loader = DataLoader(
        trainset_eval,
        batch_size=batch_size_real,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
    )
    all_test_loader = DataLoader(
        testset,
        batch_size=batch_size_real,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
    )

    (
        train_fgt_loader, train_retain_loader,
        test_fgt_loader, test_retain_loader,
        repair_class_loader,
        train_fgt_idx, train_retain_idx,
        test_fgt_idx, test_retain_idx,
    ) = get_unlearn_loader(
        trainset, testset, [forget_class],
        batch_size=batch_size_real, num_workers=8, num_forget=float("inf")
    )

    # cache real embeddings once
    all_train_feats, all_train_labels = embed_loader_to_tensor(feature_model, all_train_loader)
    all_test_feats, all_test_labels = embed_loader_to_tensor(feature_model, all_test_loader)

    train_fgt_feats, train_fgt_labels = embed_loader_to_tensor(feature_model, train_fgt_loader)
    train_ret_feats, train_ret_labels = embed_loader_to_tensor(feature_model, train_retain_loader)
    test_fgt_feats, test_fgt_labels = embed_loader_to_tensor(feature_model, test_fgt_loader)
    test_ret_feats, test_ret_labels = embed_loader_to_tensor(feature_model, test_retain_loader)

    all_train_emb_loader = make_emb_loader(all_train_feats, all_train_labels)
    all_test_emb_loader = make_emb_loader(all_test_feats, all_test_labels)
    train_fgt_emb_loader = make_emb_loader(train_fgt_feats, train_fgt_labels)
    train_ret_emb_loader = make_emb_loader(train_ret_feats, train_ret_labels)
    test_fgt_emb_loader = make_emb_loader(test_fgt_feats, test_fgt_labels)
    test_ret_emb_loader = make_emb_loader(test_ret_feats, test_ret_labels)

    # baseline unlearned model metrics
    fc = _get_final_linear(model, num_classes)

    baseline_train_fgt = eval_accuracy_on_emb_loader(fc, train_fgt_emb_loader, device)
    baseline_train_retain = eval_accuracy_on_emb_loader(fc, train_ret_emb_loader, device)

    baseline_test_fgt = eval_accuracy_on_emb_loader(fc, test_fgt_emb_loader, device)
    baseline_test_retain = eval_accuracy_on_emb_loader(fc, test_ret_emb_loader, device)

    print(
        f"[Baseline] "
        f"train_fgt={baseline_train_fgt:.2f} | "
        f"train_retain={baseline_train_retain:.2f} | "
        f"test_fgt={baseline_test_fgt:.2f} | "
        f"test_retain={baseline_test_retain:.2f}"
    )
    # alpha sweep
    best = None
    best_forget = -1.0

    for alpha in ALPHA_GRID:
        attacked_model = apply_prototypical_relearning_attack(
            model=model,
            feature_model=feature_model,
            forget_loader=train_fgt_loader,   # attack uses real forget samples
            forget_class=forget_class,
            num_classes=num_classes,
            k=attack_k,
            alpha=alpha,
            metric=metric,
            device=device,
        )
        attacked_fc = _get_final_linear(attacked_model, num_classes)

        pra_train_fgt = eval_accuracy_on_emb_loader(attacked_fc, train_fgt_emb_loader, device)
        pra_train_retain = eval_accuracy_on_emb_loader(attacked_fc, train_ret_emb_loader, device)

        pra_test_fgt = eval_accuracy_on_emb_loader(attacked_fc, test_fgt_emb_loader, device)
        pra_test_retain = eval_accuracy_on_emb_loader(attacked_fc, test_ret_emb_loader, device)

        delta_train_fgt = pra_train_fgt - baseline_train_fgt
        delta_train_retain = pra_train_retain - baseline_train_retain

        delta_test_fgt = pra_test_fgt - baseline_test_fgt
        delta_test_retain = pra_test_retain - baseline_test_retain
        
        valid = pra_test_retain >= (baseline_test_retain - retain_drop_tol)

        print(
            f"[alpha={alpha:.2f}] "
            f"pra_train_fgt={pra_train_fgt:.2f} | "
            f"pra_train_retain={pra_train_retain:.2f} | "
            f"pra_test_fgt={pra_test_fgt:.2f} | "
            f"pra_test_retain={pra_test_retain:.2f} | "
            f"delta_train_fgt={delta_train_fgt:.2f} | "
            f"delta_train_retain={delta_train_retain:.2f} | "
            f"delta_test_fgt={delta_test_fgt:.2f} | "
            f"delta_test_retain={delta_test_retain:.2f} | "
            f"valid={valid}"
        )

        if valid and pra_test_fgt > best_forget:
            best_forget = pra_test_fgt
            best = {
                "forget_class": int(forget_class),
                "dataset": dataset_name,
                "model": model_name,
                "method": method,
                "attack_k": int(attack_k),
                "metric": metric,
                "alpha": float(alpha),

                "baseline_train_fgt": float(baseline_train_fgt),
                "baseline_train_retain": float(baseline_train_retain),
                "baseline_test_fgt": float(baseline_test_fgt),
                "baseline_test_retain": float(baseline_test_retain),

                "pra_train_fgt": float(pra_train_fgt),
                "pra_train_retain": float(pra_train_retain),
                "pra_test_fgt": float(pra_test_fgt),
                "pra_test_retain": float(pra_test_retain),

                "delta_train_fgt": float(delta_train_fgt),
                "delta_train_retain": float(delta_train_retain),
                "delta_test_fgt": float(delta_test_fgt),
                "delta_test_retain": float(delta_test_retain),
            }

    # fallback if none satisfy retain constraint
    if best is None:
        print("[Warning] No alpha satisfied retain-drop constraint. Saving best forget recovery anyway.")
        fallback_best = None
        fallback_best_forget = -1.0

        for alpha in ALPHA_GRID:
            attacked_model = apply_prototypical_relearning_attack(
                model=model,
                feature_model=feature_model,
                forget_loader=train_fgt_loader,
                forget_class=forget_class,
                num_classes=num_classes,
                k=attack_k,
                alpha=alpha,
                metric=metric,
                device=device,
            )
            attacked_fc = _get_final_linear(attacked_model, num_classes)

            pra_test_fgt = eval_accuracy_on_emb_loader(attacked_fc, test_fgt_emb_loader, device)
            pra_test_retain = eval_accuracy_on_emb_loader(attacked_fc, test_ret_emb_loader, device)

            if pra_test_fgt > fallback_best_forget:
                fallback_best_forget = pra_test_fgt
                fallback_best = {
                    "forget_class": int(forget_class),
                    "dataset": dataset_name,
                    "model": model_name,
                    "method": method,
                    "attack_k": int(attack_k),
                    "metric": metric,
                    "alpha": float(alpha),

                    "baseline_train_fgt": float(baseline_train_fgt),
                    "baseline_train_retain": float(baseline_train_retain),
                    "baseline_test_fgt": float(baseline_test_fgt),
                    "baseline_test_retain": float(baseline_test_retain),

                    "pra_train_fgt": float(pra_train_fgt),
                    "pra_train_retain": float(pra_train_retain),
                    "pra_test_fgt": float(pra_test_fgt),
                    "pra_test_retain": float(pra_test_retain),

                    "delta_train_fgt": float(pra_train_fgt - baseline_train_fgt),
                    "delta_train_retain": float(pra_train_retain - baseline_train_retain),
                    "delta_test_fgt": float(pra_test_fgt - baseline_test_fgt),
                    "delta_test_retain": float(pra_test_retain - baseline_test_retain),
                }
        best = fallback_best

    print(
        f"[BEST] forget_class={best['forget_class']} | "
        f"alpha={best['alpha']:.2f} | "
        f"train_fgt: {best['baseline_train_fgt']:.2f} -> {best['pra_train_fgt']:.2f} | "
        f"train_retain: {best['baseline_train_retain']:.2f} -> {best['pra_train_retain']:.2f} | "
        f"test_fgt: {best['baseline_test_fgt']:.2f} -> {best['pra_test_fgt']:.2f} | "
        f"test_retain: {best['baseline_test_retain']:.2f} -> {best['pra_test_retain']:.2f}"
    )

    append_row(best)

print(f"\nSaved results to: {CSV_PATH}")