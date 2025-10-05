
import os
import math
import copy
import random
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset, Subset
import argparse
import torch.optim as optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from trainer import *  # assumes load_model is here
from method.utils import *
from pathlib import Path
from utils import (
    get_transforms, get_dataset, get_dataloader, get_unlearn_loader, _top1_and_per_class
)
from trainer import *  # assumes load_model, plot_unlearn_remain_acc_figure, etc.

# ------------------ Argparse ------------------


parser = argparse.ArgumentParser("Class unlearning revival (pre-layer4 cut; train head)")
parser.add_argument('--method', type=str, default='original',
                    choices=['original','retrained','random_label','finetune','gradient_ascent','neggrad_plus',
                             'boundary_shrink','boundary_expand','l2ul_adv','l2ul_imp','fisher','wood_fisher','delete',
                             'bad_teacher', 'salun', 'scrub'])
# accept both --model and --model_name, store in model_name
parser.add_argument('--model', '--model_name', dest='model_name', type=str, default='resnet18')
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
parser.add_argument(
    '--tpr', type=int, default=5000,
    help='How many synthetic embeddings to *generate* per class before top-K filtering.'
)
parser.add_argument(
    '--cpr', type=int, default=100,
    help='How many low-confidence per retain-class to relabel as forget (per_retain_for_forget).'
)

args = parser.parse_args()

method       = args.method
model_name   = args.model_name
dataset_name = args.dataset
lr_cli       = args.lr            # kept for CSV parity (head has its own internal LRs)
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

# ------------------ Aggregation CSV ------------------
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

    df.to_csv(
        AGG_CSV_PATH,
        mode="a",
        header=header_needed,
        index=False,
        float_format="%.3f"
    )
    print(f"[AGG] Appended best row for class {forget_class} -> {AGG_CSV_PATH}")

# ------------------ Helpers ------------------
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

# ------------------ Pre-layer4 split & sampling ------------------
def split_resnet18_before_layer4(resnet18_model: nn.Module, num_classes: int, device="cuda"):
    """
    Returns:
      feat_net: up to (and including) layer3  → outputs a tensor of shape (N, 256, H, W)
      head:     layer4 → avgpool → flatten → fc  → logits (num_classes)
    """
    m = deepcopy(resnet18_model).to(device).eval()
    needed = ["conv1","bn1","relu","maxpool","layer1","layer2","layer3","layer4","avgpool","fc"]
    for n in needed:
        if not hasattr(m, n):
            raise RuntimeError(f"Model missing '{n}'. Expected torchvision-style ResNet-18.")

    class FeatNet(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.conv1   = m.conv1
            self.bn1     = m.bn1
            self.relu    = m.relu
            self.maxpool = m.maxpool
            self.layer1  = m.layer1
            self.layer2  = m.layer2
            self.layer3  = m.layer3

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return x  # (N, 256, H, W)

    class HeadNet(nn.Module):
        def __init__(self, m, num_classes):
            super().__init__()
            self.layer4  = m.layer4
            self.avgpool = m.avgpool
            self.fc      = m.fc
        def forward(self, feats_pre_l4):  # (N,256,H,W)
            x = self.layer4(feats_pre_l4)  # (N,512,H',W')
            x = self.avgpool(x)            # (N,512,1,1)
            x = torch.flatten(x, 1)        # (N,512)
            logits = self.fc(x)            # (N,num_classes)
            return logits

    feat_net = FeatNet(m).to(device).eval()
    head     = HeadNet(m, num_classes).to(device).eval()
    return feat_net, head

@torch.inference_mode()
def infer_pre_layer4_shape(feat_net, sample_loader, device="cuda"):
    for xb, yb in sample_loader:
        xb = xb.to(device, non_blocking=True)
        feats = feat_net(xb)
        _, C, H, W = feats.shape
        return (C, H, W)
    raise RuntimeError("sample_loader is empty.")

@torch.inference_mode()
def sample_pre_layer4_predicted_as_class_streaming(
    head,
    emb_shape,
    target_class: int,
    need_top_k: int = 0,
    need_bottom_k: int = 0,
    batch: int = 256,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,   # halve memory on GPU
    safety_multiplier: float = 1.3,       # cushion so we likely reach both sets
    max_batches: int = 10_000,            # hard stop to avoid infinite loops
    use_amp: bool = True,
    # Optional: minimum margin p_c - p_best_other to accept for retain picks
    # Set to 0.0 if you want no margin filter.
    min_margin: float = 0.02,
):
    """
    Streams random (pre-l4) tensors and keeps:
      - top-K by margin for the target class (only if argmax==target) if need_top_k > 0
      - bottom-K by P[target_class] (only from argmax!=target) if need_bottom_k > 0

    Returns (CPU tensors):
      top_feats_cpu    (need_top_k,  C,H,W) or None if need_top_k == 0
      top_scores_cpu   (need_top_k,)        or None  (scores = margin)
      bottom_feats_cpu (need_bottom_k,C,H,W) or None if need_bottom_k == 0
      bottom_scores_cpu(need_bottom_k,)      or None  (scores = p_target)
    """
    import math
    C, H, W = emb_shape

    # CPU buffers
    top_feats, top_scores = None, None     # scores = margin
    bot_feats, bot_scores = None, None     # scores = p_target

    want_top = int(math.ceil(need_top_k * safety_multiplier))
    want_bot = int(math.ceil(need_bottom_k * safety_multiplier))

    have_top = 0 if want_top == 0 else -1
    have_bot = 0 if want_bot == 0 else -1
    batches_done = 0

    def _insert_pool(feats_gpu, scores_gpu, select_top: bool, want: int, feats_cpu, scores_cpu):
        """Merge candidates (already filtered) into a CPU pool and prune to 'want' by score."""
        if want <= 0 or feats_gpu.numel() == 0:
            return feats_cpu, scores_cpu
        cand_feats  = feats_gpu.detach().to("cpu", non_blocking=True)
        cand_scores = scores_gpu.detach().to("cpu", non_blocking=True)
        if feats_cpu is None:
            feats_cpu, scores_cpu = cand_feats, cand_scores
        else:
            feats_cpu  = torch.cat([feats_cpu,  cand_feats],  dim=0)
            scores_cpu = torch.cat([scores_cpu, cand_scores], dim=0)

        if scores_cpu.shape[0] > want:
            vals, idx = torch.topk(scores_cpu, k=want, largest=select_top, sorted=False)
            feats_cpu  = feats_cpu.index_select(0, idx)
            scores_cpu = vals
        return feats_cpu, scores_cpu

    was_training = head.training
    head.eval()
    try:
        while ((need_top_k > 0 and (top_scores is None or top_scores.shape[0] < want_top)) or
               (need_bottom_k > 0 and (bot_scores is None or bot_scores.shape[0] < want_bot))) and \
              batches_done < max_batches:
            batches_done += 1

            # Sample noise at the cut
            z = torch.randn(batch, C, H, W, device=device, dtype=dtype).contiguous(memory_format=torch.channels_last)

            # Forward under AMP if requested
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = head(z)
            else:
                logits = head(z)

            probs = torch.softmax(logits, dim=1)            # [B, num_classes]
            pc    = probs[:, target_class]                  # [B]
            pred  = probs.argmax(dim=1)                     # [B]

            # margin = p_c - best_other
            top2_vals, top2_idx = probs.topk(k=2, dim=1)    # [B,2]
            best_other = torch.where(
                top2_idx[:, 0] == target_class, top2_vals[:, 1], top2_vals[:, 0]
            )
            margin = pc - best_other                         # [B]

            # --- RETAIN: only correctly predicted as target with sufficient margin
            if need_top_k > 0:
                keep = (pred == target_class) & (margin >= min_margin)
                if keep.any():
                    top_feats, top_scores = _insert_pool(
                        z[keep], margin[keep], True, want_top, top_feats, top_scores
                    )

            # --- FORGET: only those NOT predicted as target (low p_target)
            if need_bottom_k > 0:
                keep = (pred != target_class)
                if keep.any():
                    bot_feats, bot_scores = _insert_pool(
                        z[keep], pc[keep], False, want_bot, bot_feats, bot_scores
                    )

            # cleanup
            del z, logits, probs, pc, pred, top2_vals, top2_idx, best_other, margin
            torch.cuda.empty_cache()

        # Final prune to exact requested sizes
        if need_top_k > 0 and top_scores is not None and top_scores.shape[0] >= need_top_k:
            vals, idx = torch.topk(top_scores, k=need_top_k, largest=True, sorted=False)
            top_feats, top_scores = top_feats.index_select(0, idx), vals

        if need_bottom_k > 0 and bot_scores is not None and bot_scores.shape[0] >= need_bottom_k:
            vals, idx = torch.topk(bot_scores, k=need_bottom_k, largest=False, sorted=False)
            bot_feats, bot_scores = bot_feats.index_select(0, idx), vals

    finally:
        if was_training:
            head.train()

    return top_feats, top_scores, bot_feats, bot_scores



def build_synth_pre_layer4_splits(
    head, num_classes, forget_class, emb_shape,
    per_class, retain_top_k, per_retain_for_forget, device="cuda", loader_batch_size=256
):
    """
    per_class is ignored now (kept for signature parity); we stream to get exactly:
      - retain: retain_top_k per non-forget class
      - forget: per_retain_for_forget per non-forget class
    """
    retain_classes = [c for c in range(num_classes) if c != forget_class]

    retain_feats_list, retain_labels_list = [], []
    forget_feats_list = []

    for c in retain_classes:
        top_feats, _, bot_feats, _ = sample_pre_layer4_predicted_as_class_streaming(
            head=head,
            emb_shape=emb_shape,
            target_class=c,
            need_top_k=retain_top_k,                 # what we actually need
            need_bottom_k=per_retain_for_forget,     # what we actually need
            batch=256,
            device=device,
            dtype=torch.float16,
            safety_multiplier=1.2,                   # small cushion
            use_amp=True,
        )

        if top_feats is None or top_feats.shape[0] < retain_top_k:
            raise RuntimeError(f"Could not collect enough retain feats for class {c}. Got {0 if top_feats is None else top_feats.shape[0]}")
        if bot_feats is None or bot_feats.shape[0] < per_retain_for_forget:
            raise RuntimeError(f"Could not collect enough forget feats for class {c}. Got {0 if bot_feats is None else bot_feats.shape[0]}")

        retain_feats_list.append(top_feats)
        retain_labels_list.append(torch.full((retain_top_k,), c, dtype=torch.long))

        forget_feats_list.append(bot_feats)

    retain_feats  = torch.cat(retain_feats_list, dim=0)          # CPU tensors
    retain_labels = torch.cat(retain_labels_list, dim=0)

    forget_feats  = torch.cat(forget_feats_list, dim=0)
    forget_labels = torch.full((forget_feats.shape[0],), forget_class, dtype=torch.long)

    retain_ds = TensorDataset(retain_feats, retain_labels)
    forget_ds = TensorDataset(forget_feats,  forget_labels)

    retain_loader = DataLoader(retain_ds, batch_size=loader_batch_size, shuffle=True, drop_last=False)
    forget_loader = DataLoader(forget_ds, batch_size=loader_batch_size, shuffle=True, drop_last=False)

    summary = {
        "space": "pre_layer4",
        "emb_shape": tuple(emb_shape),
        "retain_classes": retain_classes,
        "retain_per_class_generated": "streamed",
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


# ------------------ Embedding/eval utilities ------------------
def make_emb_loader(x, y, bs=4096, shuffle=False, num_workers=4):
    ds = TensorDataset(x, y)
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle, drop_last=False,
        pin_memory=(device == 'cuda'),
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=4
    )

@torch.inference_mode()
def eval_accuracy_on_pre_l4_emb_loader(head, loader, device="cuda"):
    p_dtype = next(head.parameters()).dtype
    total, correct = 0, 0
    for feats, labels in loader:
        feats  = feats.to(device, non_blocking=True).float()  # <— ensure dtype match
        labels = labels.to(device, non_blocking=True)
        logits = head(feats)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total   += labels.numel()
    return 100.0 * correct / max(total, 1)


@torch.inference_mode()
def embed_loader_to_pre_l4(feat_net, loader, device="cuda"):
    feat_net.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = feat_net(x).float().contiguous()  # fp32 & contiguous
        feats.append(f.cpu())
        labels.append(y.cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)

# ------------------ Repro ------------------
seed = 0
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
g = torch.Generator(device="cpu").manual_seed(seed)

# ------------------ Main ------------------
forget_classes = _parse_forget_arg(args.forget, num_classes)

for forget_class in forget_classes:
    print(f"\n================= FORGET CLASS {forget_class} =================")

    # per-class experiment folder
    experiment_path = Path(f"results/{method}/plots_{model_name}_lr{lr_cli}/forget_class_{forget_class}")
    experiment_path.mkdir(parents=True, exist_ok=True)

    accs_curves = {
        "train_forget": [],
        "test_forget":  [],
        "train_remain": [],
        "test_remain":  [],
    }

    # load baseline model
    ckpt = checkpoint_for(method, dataset_name, model_name, forget_class, lr_cli, DIR)
    model = load_model(ckpt, model_name, num_classes).to(device).eval()

    # --- transforms & datasets ---
    wo_dataaug = False
    transform_train, transform_test = get_transforms(dataset_name, model_name, wo_dataaug=wo_dataaug)
    trainset, testset = get_dataset(dataset_name, transform_train, transform_test)

    # --- base loaders ---
    batch_size_real = 256
    num_workers = 8
    all_train_loader, all_test_loader = get_dataloader(
        trainset, testset, batch_size=batch_size_real, num_workers=num_workers
    )

    # deterministic evaluation loader (no aug, no shuffle)
    trainset_eval = copy.copy(trainset)
    trainset_eval.transform = transform_test
    all_train_loader = DataLoader(
        trainset_eval,
        batch_size=batch_size_real, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=True, generator=g
    )

    # one-vs-all retain/forget loaders
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

    # -------- cut before layer4 --------
    feat_net, head = split_resnet18_before_layer4(model, num_classes, device=device)

    # freeze ≤ layer3; train whole head
    for p in feat_net.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True
    feat_net.eval()
    head.train()




    @torch.inference_mode()
    def sanity_check_equivalence(model, feat_net, head, sample_loader, device="cuda"):
        model.eval(); feat_net.eval(); head.eval()
        xb, yb = next(iter(sample_loader))              # images+labels
        xb = xb.to(device, non_blocking=True)

        logits_full = model(xb)                         # full model path
        pre_l4      = feat_net(xb)                      # pre-layer4
        logits_split = head(pre_l4)                     # head on pre-layer4

        # numeric comparison
        max_abs = (logits_full - logits_split).abs().max().item()
        acc_full = (logits_full.argmax(1).cpu() == yb).float().mean().item()*100
        acc_split= (logits_split.argmax(1).cpu() == yb).float().mean().item()*100

        print(f"[SANITY] max|Δ| logits = {max_abs:.6f} | acc full = {acc_full:.2f}% | acc split = {acc_split:.2f}%")
        return max_abs, acc_full, acc_split

    # call it once right after you build feat_net/head and your (non-aug) loaders:
    _ = sanity_check_equivalence(model, feat_net, head, all_test_loader, device)


    # Infer (256,H,W)
    emb_shape = infer_pre_layer4_shape(feat_net, all_train_loader, device=device)  # e.g., (256,4,4) on CIFAR
    print(f"[shape] pre-layer4 embedding shape = {emb_shape}")

    # Embed real data once
    all_train_feats, all_train_labels = embed_loader_to_pre_l4(feat_net, all_train_loader, device=device)
    all_test_feats,  all_test_labels  = embed_loader_to_pre_l4(feat_net, all_test_loader,  device=device)

    train_fgt_feats, train_fgt_labels = embed_loader_to_pre_l4(feat_net, train_fgt_loader, device=device)
    train_ret_feats, train_ret_labels = embed_loader_to_pre_l4(feat_net, train_retain_loader, device=device)
    test_fgt_feats,  test_fgt_labels  = embed_loader_to_pre_l4(feat_net, test_fgt_loader,  device=device)
    test_ret_feats,  test_ret_labels  = embed_loader_to_pre_l4(feat_net, test_retain_loader, device=device)

    # Wrap as fast loaders on pre-layer4 tensors
    all_train_emb_loader = make_emb_loader(all_train_feats, all_train_labels)
    all_test_emb_loader  = make_emb_loader(all_test_feats,  all_test_labels)
    train_fgt_emb_loader = make_emb_loader(train_fgt_feats, train_fgt_labels)
    train_ret_emb_loader = make_emb_loader(train_ret_feats, train_ret_labels)
    test_fgt_emb_loader  = make_emb_loader(test_fgt_feats,  test_fgt_labels)
    test_ret_emb_loader  = make_emb_loader(test_ret_feats,  test_ret_labels)

    # Build synthetic pre-layer4 splits
    synth = build_synth_pre_layer4_splits(
        head=head,
        num_classes=num_classes,
        forget_class=forget_class,
        emb_shape=emb_shape,
        per_class=total_per_class,
        retain_top_k=choose_per_retain_class_for_fgt* (num_classes - 1),  # mirrors your 9x logic on CIFAR-10
        per_retain_for_forget=choose_per_retain_class_for_fgt,
        device=device, loader_batch_size=256,
    )
    print("Synthetic selection summary:", synth["summary"])

    syn_retain_eval_loader = DataLoader(TensorDataset(synth["retain_feats"].cpu(),
                                                      synth["retain_labels"].cpu()),
                                        batch_size=1024, shuffle=False, drop_last=False)
    syn_forget_eval_loader = DataLoader(TensorDataset(synth["forget_feats"].cpu(),
                                                      synth["forget_labels"].cpu()),
                                        batch_size=1024, shuffle=False, drop_last=False)

    n_syn_ret = len(syn_retain_eval_loader.dataset)
    n_syn_fgt = len(syn_forget_eval_loader.dataset)
    metrics_history = []

    # ----- Optimizer-----
    lr_all = 1e-3
    wd_all = 1e-4  # set 0.0 if you don't want weight decay anywhere

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr_all, weight_decay=wd_all, betas=(0.9, 0.999))

    criterion = nn.CrossEntropyLoss()

    # ----- Baseline BEFORE training -----
    head.eval()
    acc_syn_ret   = eval_accuracy_on_pre_l4_emb_loader(head, syn_retain_eval_loader, device)
    acc_syn_fgt   = eval_accuracy_on_pre_l4_emb_loader(head, syn_forget_eval_loader, device)
    acc_syn_total = (acc_syn_ret * n_syn_ret + acc_syn_fgt * n_syn_fgt) / (n_syn_ret + n_syn_fgt)
    with torch.no_grad():
        acc_all_train    = eval_accuracy_on_pre_l4_emb_loader(head, all_train_emb_loader, device)
        acc_all_test     = eval_accuracy_on_pre_l4_emb_loader(head, all_test_emb_loader,  device)
        acc_train_fgt    = eval_accuracy_on_pre_l4_emb_loader(head, train_fgt_emb_loader, device)
        acc_train_retain = eval_accuracy_on_pre_l4_emb_loader(head, train_ret_emb_loader, device)
        acc_test_fgt     = eval_accuracy_on_pre_l4_emb_loader(head, test_fgt_emb_loader,  device)
        acc_test_retain  = eval_accuracy_on_pre_l4_emb_loader(head, test_ret_emb_loader,  device)

    print(
        f"[Epoch 00] "
        f"syn_train_loss=NA syn_train_acc=NA | "
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

    # Build synthetic train set (pre-layer4)
    train_feats  = torch.cat([synth["retain_feats"], synth["forget_feats"]], dim=0)
    train_labels = torch.cat([synth["retain_labels"], synth["forget_labels"]], dim=0)
    train_loader_syn = DataLoader(
        list(zip(train_feats, train_labels)), batch_size=128, shuffle=True, drop_last=False
    )

    # ----- Training epochs -----
    for epoch in range(1, epochs + 1):
        head.train()
        feat_net.eval()

        running_loss, right, seen = 0.0, 0, 0
        for feats, targets in train_loader_syn:
            p_dtype = next(head.parameters()).dtype
            feats   = feats.to(device, non_blocking=True).to(p_dtype)   # (B,256,H,W)
            targets = targets.to(device, non_blocking=True)

            logits = head(feats)
            loss   = criterion(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * feats.size(0)
            right += (logits.argmax(dim=1) == targets).sum().item()
            seen  += targets.numel()

        train_loss = running_loss / max(seen, 1)
        train_acc  = 100.0 * right / max(seen, 1)

        # synthetic eval
        head.eval()
        acc_syn_ret   = eval_accuracy_on_pre_l4_emb_loader(head, syn_retain_eval_loader, device)
        acc_syn_fgt   = eval_accuracy_on_pre_l4_emb_loader(head, syn_forget_eval_loader, device)
        acc_syn_total = (acc_syn_ret * n_syn_ret + acc_syn_fgt * n_syn_fgt) / (n_syn_ret + n_syn_fgt)

        # real-embedding eval
        with torch.no_grad():
            acc_all_train    = eval_accuracy_on_pre_l4_emb_loader(head, all_train_emb_loader, device)
            acc_all_test     = eval_accuracy_on_pre_l4_emb_loader(head, all_test_emb_loader,  device)
            acc_train_fgt    = eval_accuracy_on_pre_l4_emb_loader(head, train_fgt_emb_loader, device)
            acc_train_retain = eval_accuracy_on_pre_l4_emb_loader(head, train_ret_emb_loader, device)
            acc_test_fgt     = eval_accuracy_on_pre_l4_emb_loader(head, test_fgt_emb_loader,  device)
            acc_test_retain  = eval_accuracy_on_pre_l4_emb_loader(head, test_ret_emb_loader,  device)

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

        # best-by-lexicographic (test_fgt, test_retain, train_fgt, train_retain)
        key = (
            row["test_fgt"], row["test_retain"], row["train_fgt"], row["train_retain"]
        )
        if key > best["key"]:
            best["key"] = key
            best["row"] = row.copy()

        # curves are fractions
        accs_curves["train_forget"].append(float(acc_train_fgt) / 100.0)
        accs_curves["test_forget"].append(float(acc_test_fgt) / 100.0)
        accs_curves["train_remain"].append(float(acc_train_retain) / 100.0)
        accs_curves["test_remain"].append(float(acc_test_retain) / 100.0)

        # re-plot
        epoch_for_plot = len(accs_curves["train_forget"])
        plot_unlearn_remain_acc_figure(
            epoch=epoch_for_plot,
            accs_dict=accs_curves,
            experiment_path=experiment_path,
            plot_type="plot",
        )

    # write one best row per forget class
    if best["row"] is not None:
        append_best_for_class(forget_class, best["row"])
    else:
        print(f"[BEST PER CLASS] No best row found for forget_class={forget_class}.")


