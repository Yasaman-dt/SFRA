import os
import math
import copy
import random
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ---- your project utilities (unchanged) ----
from trainer import *  # expects load_model, plot_unlearn_remain_acc_figure, etc.
from method.utils import *
from utils import (
    get_transforms, get_dataset, get_dataloader, get_unlearn_loader, _top1_and_per_class
)

# ------------------ Argparse ------------------
parser = argparse.ArgumentParser("Class unlearning revival (VGG16: cut before conv5_3; train last FC)")
parser.add_argument('--method', type=str, default='original',
                    choices=['original','retrained','random_label','finetune','gradient_ascent','neggrad_plus',
                             'boundary_shrink','boundary_expand','l2ul_adv','l2ul_imp','fisher','wood_fisher','delete',
                             'bad_teacher', 'salun', 'scrub'])
parser.add_argument('--model', '--model_name', dest='model_name', type=str, default='vgg16',
                    choices=['vgg16','vgg16_bn'])
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--lr', type=float, default=5e-5, help="(Unused here; kept for CSV parity)")
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument(
    '--forget',
    type=str,
    default='all',
    help=("Which class(es) to forget. "
          "Examples: 'all' | '7' | '1,3,5' | '2-6' | '0,2-4,7'")
)
parser.add_argument('--tpr', type=int, default=5000, help='Synthetic embeddings to *generate* per class before top-K.')
parser.add_argument('--cpr', type=int, default=100,  help='Low-confidence per retain-class to relabel as forget.')
args = parser.parse_args()

method       = args.method
model_name   = args.model_name
dataset_name = args.dataset
lr_cli       = args.lr
epochs       = args.epochs
total_per_class = args.tpr
choose_per_retain_class_for_fgt = args.cpr

# ------------------ Globals ------------------
NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'tinyimagenet': 200, 'imagenet': 1000}
try:
    num_classes = NUM_CLASSES[dataset_name.lower()]
except KeyError:
    raise ValueError(f"Unknown dataset '{dataset_name}'. Add it to NUM_CLASSES.")

DIR = "/export/livia/home/vision/Zdehghani/classification/exps"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------ CSV aggregation ------------------
AGG_CSV_DIR = os.path.join("results", method)
AGG_CSV_PATH = os.path.join(
    AGG_CSV_DIR,
    f"{dataset_name}_{model_name}_unlearned_{method}_revival_by_forget_class_lr{lr_cli}.csv"
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
    if best_row is None:
        print(f"[AGG] No best row to append for class {forget_class}.")
        return
    row = {
        "forget_class": int(forget_class),
        "dataset": dataset_name,
        "model": model_name,
        "method": method,
        "lr": lr_cli,
        "epochs_total": epochs,
        "tpr": total_per_class,
        "cpr": choose_per_retain_class_for_fgt,
        **best_row,
    }
    df = pd.DataFrame([row], columns=COLUMNS)
    header_needed = not os.path.exists(AGG_CSV_PATH)
    METRIC_COLS = [
        "epochs_total", "tpr", "cpr",
        "syn_train_loss", "syn_train_acc",
        "syn_total", "syn_retain", "syn_forget",
        "all_train", "all_test",
        "train_fgt", "train_retain", "test_fgt", "test_retain",
    ]
    df[METRIC_COLS] = df[METRIC_COLS].apply(pd.to_numeric, errors="coerce").round(3)
    df.to_csv(AGG_CSV_PATH, mode="a", header=header_needed, index=False, float_format="%.3f")
    print(f"[AGG] Appended best row for class {forget_class} -> {AGG_CSV_PATH}")

def checkpoint_for(method, dataset_name, model_name, forget_class, lr, base_dir):
    if method == 'original':
        return f"{base_dir}/test_pretrained_model/{dataset_name}_{model_name}_original_model.pth"
    elif method == 'retrained':
        return f"{base_dir}/test_pretrained_model/{dataset_name}_{model_name}_retrain_forgetcls{forget_class}_model.pth"
    else:
        return f"{base_dir}/{dataset_name}_{model_name}_forgetcls{forget_class}/{method}/lr{lr}/ckpt_best_by_aus.pth"

def _parse_forget_arg(s: str, num_classes: int):
    s = (s or 'all').strip().lower()
    if s in ('all', '-1'):
        return list(range(num_classes))
    selected = set()
    for part in s.split(','):
        part = part.strip()
        if not part: continue
        if '-' in part:
            a, b = part.split('-', 1)
            a, b = int(a), int(b)
            if a > b: a, b = b, a
            for c in range(a, b + 1):
                if 0 <= c < num_classes:
                    selected.add(c)
        else:
            c = int(part)
            if 0 <= c < num_classes:
                selected.add(c)
    if not selected:
        raise ValueError(f"No valid classes parsed from --forget='{s}'. Valid range is 0..{num_classes-1} or 'all'.")
    return sorted(selected)

# =========================================================
#   VGG16 split before last conv of block5 (conv5_3)
# =========================================================
def _split_vgg16_features(vgg: nn.Module):
    feats = list(vgg.features.children())
    conv5_convs = []
    pool5_idx = None
    # collect all convs that output 512 ch (block5)
    for idx, m in enumerate(feats):
        if isinstance(m, nn.Conv2d) and m.out_channels == 512:
            conv5_convs.append(idx)
    if len(conv5_convs) < 3:
        raise RuntimeError("Could not find 3 convs of block5 in VGG features.")
    c53 = conv5_convs[-1]
    # find first MaxPool after conv5_3
    for idx in range(c53 + 1, len(feats)):
        if isinstance(feats[idx], nn.MaxPool2d):
            pool5_idx = idx; break
    if pool5_idx is None:
        raise RuntimeError("Could not find maxpool after block5 in VGG features.")
    return c53, pool5_idx

class FeatNetBeforeLastConvB5_VGG(nn.Module):
    """
    Outputs:
      - conv_last_in: tensor right BEFORE conv5_3 (i.e., after conv5_2 (+bn)+relu), [B,512,H,W]
      - dummy_identity: zeros like conv_last_in, so interface matches (feats, identity)
    """
    def __init__(self, vgg: nn.Module):
        super().__init__()
        self._features_full = copy.deepcopy(vgg.features)
        c53, _ = _split_vgg16_features(vgg)
        # up to (but excluding) conv5_3
        self.until_conv5_2_relu = nn.Sequential(*list(self._features_full.children())[:c53])
        self.out_channels = 512

    def forward(self, x):
        conv_last_in = self.until_conv5_2_relu(x)
        dummy_identity = torch.zeros_like(conv_last_in)
        return conv_last_in, dummy_identity

class HeadFromLastConvB5_VGG(nn.Module):
    """
    Applies conv5_3 (+BN if present) -> ReLU -> MaxPool5 -> AvgPool -> Classifier -> logits
    """
    def __init__(self, vgg: nn.Module, num_classes: int):
        super().__init__()
        feats = list(copy.deepcopy(vgg.features).children())
        c53, p5 = _split_vgg16_features(vgg)

        blocks = []
        idx = c53
        # conv5_3
        blocks.append(feats[idx]); idx += 1
        # optional BN
        if idx < len(feats) and isinstance(feats[idx], nn.BatchNorm2d):
            blocks.append(feats[idx]); idx += 1
        # ReLU
        if idx < len(feats) and isinstance(feats[idx], nn.ReLU):
            blocks.append(feats[idx]); idx += 1
        else:
            blocks.append(nn.ReLU(inplace=True))
        # MaxPool after block5
        if idx < len(feats) and isinstance(feats[idx], nn.MaxPool2d):
            blocks.append(feats[idx])
        else:
            blocks.append(feats[p5])
        self.post_b5 = nn.Sequential(*blocks)

        self.avgpool = copy.deepcopy(getattr(vgg, "avgpool", nn.AdaptiveAvgPool2d((7,7))))
        # clone classifier and fix last layer to num_classes
        self.classifier = copy.deepcopy(vgg.classifier)
        if isinstance(self.classifier, nn.Sequential):
            last = self.classifier[-1]
            if getattr(last, "out_features", None) != num_classes:
                in_f = last.in_features
                self.classifier[-1] = nn.Linear(in_f, num_classes)

    def forward(self, feats_tuple):
        conv_last_in, _ = feats_tuple
        x = self.post_b5(conv_last_in)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

def split_vgg16_before_last_conv_b5(vgg_model: nn.Module, num_classes: int, device="cuda"):
    m = copy.deepcopy(vgg_model).to(device).eval()
    if not hasattr(m, "features"):
        raise RuntimeError("Expected torchvision-style VGG model with .features")
    feat_net = FeatNetBeforeLastConvB5_VGG(m).to(device).eval()
    head     = HeadFromLastConvB5_VGG(m, num_classes).to(device).train()
    return feat_net, head

# =========================================================
#   Generic helpers (pair inputs)
# =========================================================
@torch.inference_mode()
def infer_pre_lastconv_shapes_vgg(feat_net, sample_loader, device="cuda"):
    for xb, _ in sample_loader:
        xb = xb.to(device, non_blocking=True)
        conv_last_in, dummy = feat_net(xb)
        _, C1, H1, W1 = conv_last_in.shape
        _, C2, H2, W2 = dummy.shape
        return (C1, H1, W1), (C2, H2, W2)
    raise RuntimeError("sample_loader is empty.")

@torch.inference_mode()
def embed_loader_to_pre_lastconv_vgg(feat_net, loader, device="cuda"):
    feat_net.eval()
    c_all, d_all, y_all = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        c, d = feat_net(x)
        c_all.append(c.float().contiguous().cpu())
        d_all.append(d.float().contiguous().cpu())
        y_all.append(y.cpu())
    return torch.cat(c_all, 0), torch.cat(d_all, 0), torch.cat(y_all, 0)

def make_lastconv_emb_loader(c, d, y, bs=4096, shuffle=False, num_workers=4):
    ds = TensorDataset(c, d, y)
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle, drop_last=False,
        pin_memory=(device == 'cuda'), num_workers=num_workers,
        persistent_workers=True, prefetch_factor=4
    )

@torch.inference_mode()
def eval_accuracy_on_pre_lastconv_emb_loader(head, loader, device="cuda"):
    total, correct = 0, 0
    head.eval()
    for c, d, labels in loader:
        c = c.to(device, non_blocking=True).float()
        d = d.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True)
        logits = head((c, d))
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / max(total, 1)

@torch.inference_mode()
def sanity_check_equivalence_vgg(model, feat_net, head, sample_loader, device="cuda"):
    model.eval(); feat_net.eval(); head.eval()
    xb, yb = next(iter(sample_loader))
    xb = xb.to(device, non_blocking=True)
    logits_full = model(xb)

    c, d = feat_net(xb)
    logits_split = head((c, d))

    max_abs = (logits_full - logits_split).abs().max().item()
    acc_full  = (logits_full.argmax(1).cpu()  == yb).float().mean().item()*100
    acc_split = (logits_split.argmax(1).cpu() == yb).float().mean().item()*100
    print(f"[SANITY vgg] max|Δ|={max_abs:.6f} | acc full={acc_full:.2f}% | acc split={acc_split:.2f}%")
    return max_abs, acc_full, acc_split

# -------- synthetic streaming (unchanged interface) --------
@torch.inference_mode()
def sample_pre_lastconv_pred_as_class_streaming(
    head,
    conv_shape, identity_shape,
    target_class: int,
    need_top_k:    int = 0,
    need_bottom_k: int = 0,
    batch: int = 256,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    safety_multiplier: float = 1.3,
    max_batches: int = 10_000,
    use_amp: bool = True,
):
    C1, H1, W1 = conv_shape
    C2, H2, W2 = identity_shape

    was_training = head.training
    head.eval()

    top_c, top_d, top_pc = None, None, None
    bot_c, bot_d, bot_pc = None, None, None

    want_top = int(math.ceil(need_top_k    * safety_multiplier))
    want_bot = int(math.ceil(need_bottom_k * safety_multiplier))

    def _insert_pool(c_gpu, d_gpu, score_gpu, select_top, want, c_cpu, d_cpu, score_cpu):
        if want <= 0 or score_gpu.numel() == 0:
            return c_cpu, d_cpu, score_cpu
        c_c    = c_gpu.detach().to("cpu", non_blocking=True)
        d_c    = d_gpu.detach().to("cpu", non_blocking=True)
        score_c = score_gpu.detach().to("cpu", non_blocking=True)
        if score_cpu is None:
            c_cpu, d_cpu, score_cpu = c_c, d_c, score_c
        else:
            c_cpu    = torch.cat([c_cpu,    c_c],    dim=0)
            d_cpu    = torch.cat([d_cpu,    d_c],    dim=0)
            score_cpu = torch.cat([score_cpu, score_c], dim=0)
        if score_cpu.shape[0] > want:
            vals, idx = torch.topk(score_cpu, k=want, largest=select_top, sorted=False)
            c_cpu    = c_cpu.index_select(0, idx)
            d_cpu    = d_cpu.index_select(0, idx)
            score_cpu = vals
        return c_cpu, d_cpu, score_cpu

    have_top = have_bot = 0

    while (have_top < want_top or have_bot < want_bot) and (have_top+have_bot < 10_000_000):
        z_c = torch.randn(batch, C1, H1, W1, device=device, dtype=dtype).contiguous(memory_format=torch.channels_last)
        z_d = torch.randn(batch, C2, H2, W2, device=device, dtype=dtype).contiguous(memory_format=torch.channels_last)
        if use_amp and torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = head((z_c, z_d))
        else:
            logits = head((z_c, z_d))
        probs = torch.softmax(logits, dim=1)
        pc    = probs[:, target_class]
        pred  = probs.argmax(dim=1)

        if need_top_k > 0:
            keep = (pred == target_class)
            if keep.any():
                top_c, top_d, top_pc = _insert_pool(z_c[keep], z_d[keep], pc[keep], True, want_top, top_c, top_d, top_pc)
                have_top = 0 if top_pc is None else top_pc.shape[0]

        if need_bottom_k > 0:
            keep_forget = (pred != target_class)
            if keep_forget.any():
                bot_c, bot_d, bot_pc = _insert_pool(z_c[keep_forget], z_d[keep_forget], pc[keep_forget],
                                                    False, want_bot, bot_c, bot_d, bot_pc)
                have_bot = 0 if bot_pc is None else bot_pc.shape[0]

        del z_c, z_d, logits, probs, pc, pred
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if need_top_k > 0 and top_pc is not None and top_pc.shape[0] >= need_top_k:
        vals, idx = torch.topk(top_pc, k=need_top_k, largest=True, sorted=False)
        top_c, top_d, top_pc = top_c.index_select(0, idx), top_d.index_select(0, idx), vals
    if need_bottom_k > 0 and bot_pc is not None and bot_pc.shape[0] >= need_bottom_k:
        vals, idx = torch.topk(bot_pc, k=need_bottom_k, largest=False, sorted=False)
        bot_c, bot_d, bot_pc = bot_c.index_select(0, idx), bot_d.index_select(0, idx), vals

    if was_training: head.train()
    return (top_c, top_d, top_pc, bot_c, bot_d, bot_pc)

def build_synth_pre_lastconv_vgg_splits(
    head, num_classes, forget_class,
    conv_shape, identity_shape,
    retain_top_k, per_retain_for_forget,
    device="cuda", loader_batch_size=256
):
    retain_classes = [c for c in range(num_classes) if c != forget_class]
    ret_c_list, ret_d_list, ret_y_list = [], [], []
    fgt_c_list, fgt_d_list            = [], []

    for c in retain_classes:
        top_c, top_d, _, bot_c, bot_d, _ = sample_pre_lastconv_pred_as_class_streaming(
            head=head,
            conv_shape=conv_shape, identity_shape=identity_shape,
            target_class=c,
            need_top_k=retain_top_k,
            need_bottom_k=per_retain_for_forget,
            batch=256, device=device, dtype=torch.float16, safety_multiplier=1.2, use_amp=True
        )
        if top_c is None or top_c.shape[0] < retain_top_k:
            raise RuntimeError(f"Insufficient retain feats for class {c}")
        if bot_c is None or bot_c.shape[0] < per_retain_for_forget:
            raise RuntimeError(f"Insufficient forget feats for class {c}")

        ret_c_list.append(top_c); ret_d_list.append(top_d)
        ret_y_list.append(torch.full((retain_top_k,), c, dtype=torch.long))
        fgt_c_list.append(bot_c); fgt_d_list.append(bot_d)

    retain_c = torch.cat(ret_c_list, 0)
    retain_d = torch.cat(ret_d_list, 0)
    retain_y = torch.cat(ret_y_list, 0)
    forget_c = torch.cat(fgt_c_list, 0)
    forget_d = torch.cat(fgt_d_list, 0)
    forget_y = torch.full((forget_c.shape[0],), forget_class, dtype=torch.long)

    retain_loader = DataLoader(TensorDataset(retain_c, retain_d, retain_y),
                               batch_size=loader_batch_size, shuffle=True, drop_last=False)
    forget_loader = DataLoader(TensorDataset(forget_c, forget_d, forget_y),
                               batch_size=loader_batch_size, shuffle=True, drop_last=False)

    return {
        "retain_c2": retain_c, "retain_id": retain_d, "retain_labels": retain_y, "retain_loader": retain_loader,
        "forget_c2": forget_c, "forget_id": forget_d, "forget_labels": forget_y, "forget_loader": forget_loader,
        "summary": {
            "space": "pre_lastconv_vgg_b5",
            "conv_shape": tuple(conv_shape),
            "identity_shape": tuple(identity_shape),
            "retain_top_k": retain_top_k,
            "forget_per_retain": per_retain_for_forget,
            "retain_total": retain_c.shape[0],
            "forget_total": forget_c.shape[0],
            "forget_class": forget_class,
        }
    }

# ------------------ Repro ------------------
seed = 0
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
g = torch.Generator(device="cpu").manual_seed(seed)

# ------------------ Main ------------------
forget_classes = _parse_forget_arg(args.forget, num_classes)

for forget_class in forget_classes:
    print(f"\n================= FORGET CLASS {forget_class} =================")

    experiment_path = Path(f"results/{method}/plots_{model_name}_lr{lr_cli}/forget_class_{forget_class}")
    experiment_path.mkdir(parents=True, exist_ok=True)

    accs_curves = {"train_forget": [], "test_forget": [], "train_remain": [], "test_remain": []}

    # load baseline model
    ckpt  = checkpoint_for(method, dataset_name, model_name, forget_class, lr_cli, DIR)
    model = load_model(ckpt, model_name, num_classes).to(device).eval()  # must return torchvision vgg16/_bn

    # transforms & datasets
    wo_dataaug = False
    transform_train, transform_test = get_transforms(dataset_name, model_name, wo_dataaug=wo_dataaug)
    trainset, testset = get_dataset(dataset_name, transform_train, transform_test)

    # base dataloaders
    batch_size_real = 256
    num_workers = 8
    all_train_loader, all_test_loader = get_dataloader(
        trainset, testset, batch_size=batch_size_real, num_workers=num_workers
    )
    # deterministic eval train loader (test transforms, no shuffle)
    trainset_eval = copy.copy(trainset); trainset_eval.transform = transform_test
    all_train_loader = DataLoader(
        trainset_eval, batch_size=batch_size_real, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=True, generator=g
    )

    # one-vs-all retain/forget
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

    # -------- split BEFORE conv5_3 --------
    feat_net, head = split_vgg16_before_last_conv_b5(model, num_classes, device=device)

    # freeze features
    for p in feat_net.parameters():
        p.requires_grad = False
        
    # freeze head by default
    for _, p in head.named_parameters():
        p.requires_grad = True
        
    # unfreeze final classifier layer only (mirror "only FC" policy)
    if hasattr(head, "classifier") and isinstance(head.classifier, nn.Sequential):
        for p in head.classifier[-1].parameters():
            p.requires_grad = True
    else:
        raise RuntimeError("VGG head lacks classifier[-1]")

    feat_net.eval(); head.train()

    # sanity
    _ = sanity_check_equivalence_vgg(model, feat_net, head, all_test_loader, device)

    # shapes
    conv_shape, identity_shape = infer_pre_lastconv_shapes_vgg(feat_net, all_train_loader, device=device)
    print(f"[shape] conv={conv_shape} | identity={identity_shape}")

    # embed real data
    tr_c, tr_d, tr_y = embed_loader_to_pre_lastconv_vgg(feat_net, all_train_loader, device=device)
    te_c, te_d, te_y = embed_loader_to_pre_lastconv_vgg(feat_net, all_test_loader,  device=device)
    trF_c, trF_d, trF_y = embed_loader_to_pre_lastconv_vgg(feat_net, train_fgt_loader,    device=device)
    trR_c, trR_d, trR_y = embed_loader_to_pre_lastconv_vgg(feat_net, train_retain_loader, device=device)
    teF_c, teF_d, teF_y = embed_loader_to_pre_lastconv_vgg(feat_net, test_fgt_loader,     device=device)
    teR_c, teR_d, teR_y = embed_loader_to_pre_lastconv_vgg(feat_net, test_retain_loader,  device=device)

    # wrap as loaders
    all_train_emb_loader = make_lastconv_emb_loader(tr_c, tr_d, tr_y)
    all_test_emb_loader  = make_lastconv_emb_loader(te_c, te_d, te_y)
    train_fgt_emb_loader = make_lastconv_emb_loader(trF_c, trF_d, trF_y)
    train_ret_emb_loader = make_lastconv_emb_loader(trR_c, trR_d, trR_y)
    test_fgt_emb_loader  = make_lastconv_emb_loader(teF_c, teF_d, teF_y)
    test_ret_emb_loader  = make_lastconv_emb_loader(teR_c, teR_d, teR_y)

    # synthetic splits
    head.eval()
    synth = build_synth_pre_lastconv_vgg_splits(
        head=head,
        num_classes=num_classes,
        forget_class=forget_class,
        conv_shape=conv_shape,
        identity_shape=identity_shape,
        retain_top_k=choose_per_retain_class_for_fgt * (num_classes - 1),
        per_retain_for_forget=choose_per_retain_class_for_fgt,
        device=device, loader_batch_size=256,
    )
    print("Synthetic selection summary:", synth["summary"])

    syn_retain_eval_loader = DataLoader(
        TensorDataset(synth["retain_c2"].cpu(), synth["retain_id"].cpu(), synth["retain_labels"].cpu()),
        batch_size=1024, shuffle=False, drop_last=False
    )
    syn_forget_eval_loader = DataLoader(
        TensorDataset(synth["forget_c2"].cpu(), synth["forget_id"].cpu(), synth["forget_labels"].cpu()),
        batch_size=1024, shuffle=False, drop_last=False
    )

    n_syn_ret = len(syn_retain_eval_loader.dataset)
    n_syn_fgt = len(syn_forget_eval_loader.dataset)
    metrics_history = []

    # ----- Optimizer / Loss -----
    trainable_params = (p for p in head.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    # ----- Baseline BEFORE training -----
    head.eval()
    acc_syn_ret   = eval_accuracy_on_pre_lastconv_emb_loader(head, syn_retain_eval_loader, device)
    acc_syn_fgt   = eval_accuracy_on_pre_lastconv_emb_loader(head, syn_forget_eval_loader, device)
    acc_syn_total = (acc_syn_ret * n_syn_ret + acc_syn_fgt * n_syn_fgt) / (n_syn_ret + n_syn_fgt)
    with torch.no_grad():
        acc_all_train    = eval_accuracy_on_pre_lastconv_emb_loader(head, all_train_emb_loader, device)
        acc_all_test     = eval_accuracy_on_pre_lastconv_emb_loader(head, all_test_emb_loader,  device)
        acc_train_fgt    = eval_accuracy_on_pre_lastconv_emb_loader(head, train_fgt_emb_loader, device)
        acc_train_retain = eval_accuracy_on_pre_lastconv_emb_loader(head, train_ret_emb_loader, device)
        acc_test_fgt     = eval_accuracy_on_pre_lastconv_emb_loader(head, test_fgt_emb_loader,  device)
        acc_test_retain  = eval_accuracy_on_pre_lastconv_emb_loader(head, test_ret_emb_loader,  device)

    print(
        f"[Epoch 00] syn_train_loss=NA syn_train_acc=NA | "
        f"syn_total={acc_syn_total:.2f}% syn_ret={acc_syn_ret:.2f}% syn_fgt={acc_syn_fgt:.2f}% | "
        f"all_train={acc_all_train:.2f}% all_test={acc_all_test:.2f}% | "
        f"train_fgt={acc_train_fgt:.2f}% train_ret={acc_train_retain:.2f}% | "
        f"test_fgt={acc_test_fgt:.2f}% test_ret={acc_test_retain:.2f}%"
    )

    def _safe_get(row, k):
        v = row.get(k, float("-inf"))
        return v if (isinstance(v, (int, float)) and v == v) else float("-inf")
    def _key_from_row(row):
        return (
            float(_safe_get(row, "test_fgt")),
            float(_safe_get(row, "test_retain")),
            float(_safe_get(row, "train_fgt")),
            float(_safe_get(row, "train_retain")),
        )
    best = {"key": (float("-inf"),)*4, "row": None}
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
    metrics_history.append(row0)

    # synthetic train set (pairs)
    train_c  = torch.cat([synth["retain_c2"], synth["forget_c2"]], dim=0)
    train_d  = torch.cat([synth["retain_id"], synth["forget_id"]], dim=0)
    train_y  = torch.cat([synth["retain_labels"], synth["forget_labels"]], dim=0)
    train_loader_syn = DataLoader(list(zip(train_c, train_d, train_y)),
                                  batch_size=128, shuffle=True, drop_last=False)

    # ----- Training -----
    for epoch in range(1, epochs + 1):
        head.train(); feat_net.eval()
        running_loss, right, seen = 0.0, 0, 0
        p_dtype = next(head.parameters()).dtype

        for c, d, targets in train_loader_syn:
            c       = c.to(device, non_blocking=True).to(p_dtype)
            d       = d.to(device, non_blocking=True).to(p_dtype)
            targets = targets.to(device, non_blocking=True)

            logits = head((c, d))
            loss   = criterion(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            right        += (logits.argmax(dim=1) == targets).sum().item()
            seen         += targets.numel()

        train_loss = running_loss / max(seen, 1)
        train_acc  = 100.0 * right / max(seen, 1)

        # eval
        head.eval()
        acc_syn_ret   = eval_accuracy_on_pre_lastconv_emb_loader(head, syn_retain_eval_loader, device)
        acc_syn_fgt   = eval_accuracy_on_pre_lastconv_emb_loader(head, syn_forget_eval_loader, device)
        acc_syn_total = (acc_syn_ret * n_syn_ret + acc_syn_fgt * n_syn_fgt) / (n_syn_ret + n_syn_fgt)
        with torch.no_grad():
            acc_all_train    = eval_accuracy_on_pre_lastconv_emb_loader(head, all_train_emb_loader, device)
            acc_all_test     = eval_accuracy_on_pre_lastconv_emb_loader(head, all_test_emb_loader,  device)
            acc_train_fgt    = eval_accuracy_on_pre_lastconv_emb_loader(head, train_fgt_emb_loader, device)
            acc_train_retain = eval_accuracy_on_pre_lastconv_emb_loader(head, train_ret_emb_loader, device)
            acc_test_fgt     = eval_accuracy_on_pre_lastconv_emb_loader(head, test_fgt_emb_loader,  device)
            acc_test_retain  = eval_accuracy_on_pre_lastconv_emb_loader(head, test_ret_emb_loader,  device)

        print(
            f"[Epoch {epoch:02d}] "
            f"syn_train_loss={train_loss:.4f} syn_train_acc={train_acc:.2f}% | "
            f"syn_total={acc_syn_total:.2f}% syn_ret={acc_syn_ret:.2f}% syn_fgt={acc_syn_fgt:.2f}% | "
            f"all_train={acc_all_train:.2f}% all_test={acc_all_test:.2f}% | "
            f"train_fgt={acc_train_fgt:.2f}% train_ret={acc_train_retain:.2f}% | "
            f"test_fgt={acc_test_fgt:.2f}% test_ret={acc_test_retain:.2f}%"
        )

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
        metrics_history.append(row)

        key = (row["test_fgt"], row["test_retain"], row["train_fgt"], row["train_retain"])
        if key > best["key"]:
            best["key"] = key
            best["row"] = row.copy()

        # curves are fractions
        accs_curves["train_forget"].append(float(acc_train_fgt) / 100.0)
        accs_curves["test_forget"].append(float(acc_test_fgt) / 100.0)
        accs_curves["train_remain"].append(float(acc_train_retain) / 100.0)
        accs_curves["test_remain"].append(float(acc_test_retain) / 100.0)

        plot_unlearn_remain_acc_figure(
            epoch=len(accs_curves["train_forget"]),
            accs_dict=accs_curves,
            experiment_path=experiment_path,
            plot_type="plot",
        )

    if best["row"] is not None:
        append_best_for_class(forget_class, best["row"])
    else:
        print(f"[BEST PER CLASS] No best row found for forget_class={forget_class}.")
