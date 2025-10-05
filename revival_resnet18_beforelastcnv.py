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

# Your local project utilities
from trainer import *  # expects load_model, plot_unlearn_remain_acc_figure, etc.
from method.utils import *
from utils import (
    get_transforms, get_dataset, get_dataloader, get_unlearn_loader, _top1_and_per_class
)

# ------------------ Argparse ------------------
parser = argparse.ArgumentParser("Class unlearning revival (pre-lastconv layer4 cut; train head)")
parser.add_argument('--method', type=str, default='original',
                    choices=['original','retrained','random_label','finetune','gradient_ascent','neggrad_plus',
                             'boundary_shrink','boundary_expand','l2ul_adv','l2ul_imp','fisher','wood_fisher','delete',
                             'bad_teacher', 'salun', 'scrub'])
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

# ------------------ Checkpoint helper ------------------
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

# =========================================================
#   SPLIT BEFORE THE LAST CONV OF LAYER4 (resnet18)
# =========================================================
def split_resnet18_before_last_conv_l4(resnet18_model: nn.Module, num_classes: int, device="cuda"):
    """
    Split point: just BEFORE layer4[1].conv2.
    feat_net outputs a tuple: (conv2_in, identity_b2)
        - conv2_in: tensor after layer4[1].conv1 → bn1 → relu  (shape ~ [B, 512, H, W])
        - identity_b2: the residual identity for layer4[1]     (shape ~ [B, 512, H, W])
    head applies: conv2 → bn2 → (+ identity_b2) → relu → avgpool → fc
    """
    m = deepcopy(resnet18_model).to(device).eval()
    for n in ["conv1","bn1","relu","maxpool","layer1","layer2","layer3","layer4","avgpool","fc"]:
        if not hasattr(m, n):
            raise RuntimeError(f"Model missing '{n}'. Expected torchvision-style ResNet-18.")

    b4_0 = m.layer4[0]
    b4_1 = m.layer4[1]

    class FeatNetLastConvIn(nn.Module):
        def __init__(self, m, b4_0, b4_1):
            super().__init__()
            self.conv1   = m.conv1
            self.bn1     = m.bn1
            self.relu    = m.relu
            self.maxpool = m.maxpool
            self.layer1  = m.layer1
            self.layer2  = m.layer2
            self.layer3  = m.layer3
            self.b4_0    = b4_0
            # part-A of b4_1
            self.b4_1_conv1 = b4_1.conv1
            self.b4_1_bn1   = b4_1.bn1
            self.b4_1_relu  = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
            x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
            x = self.b4_0(x)
            identity_b2 = x  # input to layer4[1]
            out = self.b4_1_conv1(identity_b2)
            out = self.b4_1_bn1(out)
            conv2_in = self.b4_1_relu(out)
            return conv2_in, identity_b2

    class HeadLastConv(nn.Module):
        def __init__(self, m, b4_1, num_classes):
            super().__init__()
            self.b4_1_conv2 = b4_1.conv2
            self.b4_1_bn2   = b4_1.bn2
            self.b4_1_relu  = nn.ReLU(inplace=True)
            self.avgpool = m.avgpool
            self.fc      = m.fc

        def forward(self, feats_tuple):
            conv2_in, identity_b2 = feats_tuple
            out = self.b4_1_conv2(conv2_in)
            out = self.b4_1_bn2(out)
            out = out + identity_b2
            out = self.b4_1_relu(out)
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            logits = self.fc(out)
            return logits

    feat_net = FeatNetLastConvIn(m, b4_0, b4_1).to(device).eval()
    head     = HeadLastConv(m, b4_1, num_classes).to(device).train()
    return feat_net, head

@torch.inference_mode()
def infer_pre_lastconv_l4_shapes(feat_net, sample_loader, device="cuda"):
    for xb, _ in sample_loader:
        xb = xb.to(device, non_blocking=True)
        conv2_in, identity_b2 = feat_net(xb)
        _, C1, H1, W1 = conv2_in.shape
        _, C2, H2, W2 = identity_b2.shape
        return (C1, H1, W1), (C2, H2, W2)
    raise RuntimeError("sample_loader is empty.")

@torch.inference_mode()
def embed_loader_to_pre_lastconv_l4(feat_net, loader, device="cuda"):
    feat_net.eval()
    conv2_in_all, ident_all, labels_all = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        conv2_in, identity_b2 = feat_net(x)
        conv2_in_all.append(conv2_in.float().contiguous().cpu())
        ident_all.append(identity_b2.float().contiguous().cpu())
        labels_all.append(y.cpu())
    return torch.cat(conv2_in_all, 0), torch.cat(ident_all, 0), torch.cat(labels_all, 0)

def make_lastconv_emb_loader(conv2_in, identity_b2, y, bs=4096, shuffle=False, num_workers=4):
    ds = TensorDataset(conv2_in, identity_b2, y)
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle, drop_last=False,
        pin_memory=(device == 'cuda'),
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=4
    )

@torch.inference_mode()
def eval_accuracy_on_pre_lastconv_emb_loader(head, loader, device="cuda"):
    total, correct = 0, 0
    head.eval()
    for conv2_in, identity_b2, labels in loader:  # <-- three tensors
        conv2_in    = conv2_in.to(device, non_blocking=True).float()
        identity_b2 = identity_b2.to(device, non_blocking=True).float()
        labels      = labels.to(device, non_blocking=True)
        logits      = head((conv2_in, identity_b2))
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += labels.numel()
    return 100.0 * correct / max(total, 1)

@torch.inference_mode()
def sanity_check_equivalence_lastconv(model, feat_net, head, sample_loader, device="cuda"):
    model.eval(); feat_net.eval(); head.eval()
    xb, yb = next(iter(sample_loader))
    xb = xb.to(device, non_blocking=True)

    logits_full = model(xb)

    conv2_in, identity_b2 = feat_net(xb)
    logits_split = head((conv2_in, identity_b2))

    max_abs = (logits_full - logits_split).abs().max().item()
    acc_full  = (logits_full.argmax(1).cpu()  == yb).float().mean().item()*100
    acc_split = (logits_split.argmax(1).cpu() == yb).float().mean().item()*100
    print(f"[SANITY lastconv] max|Δ|={max_abs:.6f} | acc full={acc_full:.2f}% | acc split={acc_split:.2f}%")
    return max_abs, acc_full, acc_split

# =========================================================
#   Synthetic streaming at the cut (pair inputs)
# =========================================================
@torch.inference_mode()
@torch.inference_mode()
def sample_pre_lastconv_l4_predicted_as_class_streaming(
    head,
    conv2_in_shape, identity_shape,
    target_class: int,
    need_top_k:      int = 0,
    need_bottom_k:   int = 0,
    batch: int = 256,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    safety_multiplier: float = 1.3,
    max_batches: int = 10_000,
    use_amp: bool = True,
    min_margin: float = 0.00,   # kept for backward-compat; unused
):
    C1, H1, W1 = conv2_in_shape
    C2, H2, W2 = identity_shape

    was_training = head.training
    head.eval()

    # we'll score by p(target_class)
    top_c2, top_id, top_pc = None, None, None     # largest pc
    bot_c2, bot_id, bot_pc = None, None, None     # smallest pc

    want_top = int(math.ceil(need_top_k    * safety_multiplier))
    want_bot = int(math.ceil(need_bottom_k * safety_multiplier))

    def _insert_pool(c2_gpu, id_gpu, score_gpu, select_top, want, c2_cpu, id_cpu, score_cpu):
        if want <= 0 or score_gpu.numel() == 0:
            return c2_cpu, id_cpu, score_cpu
        c2_c    = c2_gpu.detach().to("cpu", non_blocking=True)
        id_c    = id_gpu.detach().to("cpu", non_blocking=True)
        score_c = score_gpu.detach().to("cpu", non_blocking=True)
        if score_cpu is None:
            c2_cpu, id_cpu, score_cpu = c2_c, id_c, score_c
        else:
            c2_cpu    = torch.cat([c2_cpu,    c2_c],    dim=0)
            id_cpu    = torch.cat([id_cpu,    id_c],    dim=0)
            score_cpu = torch.cat([score_cpu, score_c], dim=0)
        if score_cpu.shape[0] > want:
            vals, idx = torch.topk(score_cpu, k=want, largest=select_top, sorted=False)
            c2_cpu    = c2_cpu.index_select(0, idx)
            id_cpu    = id_cpu.index_select(0, idx)
            score_cpu = vals
        return c2_cpu, id_cpu, score_cpu

    have_top = have_bot = 0
    batches_done = 0
    pc_min = 0.0  # you can raise this (e.g., 0.3~0.5) to improve retain quality

    while (have_top < want_top or have_bot < want_bot) and batches_done < max_batches:
        batches_done += 1

        z_c2 = torch.randn(batch, C1, H1, W1, device=device, dtype=dtype).contiguous(memory_format=torch.channels_last)
        z_id = torch.randn(batch, C2, H2, W2, device=device, dtype=dtype).contiguous(memory_format=torch.channels_last)

        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = head((z_c2, z_id))
        else:
            logits = head((z_c2, z_id))

        probs = torch.softmax(logits, dim=1)
        pc    = probs[:, target_class]
        pred  = probs.argmax(dim=1)

        # RETAIN: largest pc among items the model actually labels as target
        if need_top_k > 0:
            keep = (pred == target_class) & (pc >= pc_min)
            if keep.any():
                top_c2, top_id, top_pc = _insert_pool(
                    z_c2[keep], z_id[keep], pc[keep], True, want_top, top_c2, top_id, top_pc
                )
                have_top = 0 if top_pc is None else top_pc.shape[0]

        # FORGET: smallest pc among items NOT labeled as target
        if need_bottom_k > 0:
            keep_forget = (pred != target_class)
            if keep_forget.any():
                bot_c2, bot_id, bot_pc = _insert_pool(
                    z_c2[keep_forget], z_id[keep_forget], pc[keep_forget],
                    False, want_bot, bot_c2, bot_id, bot_pc
                )
                have_bot = 0 if bot_pc is None else bot_pc.shape[0]

        del z_c2, z_id, logits, probs, pc, pred
        torch.cuda.empty_cache()

    # Final trim to exact K
    if need_top_k > 0 and top_pc is not None and top_pc.shape[0] >= need_top_k:
        vals, idx = torch.topk(top_pc, k=need_top_k, largest=True, sorted=False)
        top_c2, top_id, top_pc = top_c2.index_select(0, idx), top_id.index_select(0, idx), vals
    if need_bottom_k > 0 and bot_pc is not None and bot_pc.shape[0] >= need_bottom_k:
        vals, idx = torch.topk(bot_pc, k=need_bottom_k, largest=False, sorted=False)
        bot_c2, bot_id, bot_pc = bot_c2.index_select(0, idx), bot_id.index_select(0, idx), vals

    if was_training:
        head.train()

    return (top_c2, top_id, top_pc, bot_c2, bot_id, bot_pc)


def build_synth_pre_lastconv_l4_splits(
    head, num_classes, forget_class,
    conv2_in_shape, identity_shape,
    retain_top_k, per_retain_for_forget,
    device="cuda", loader_batch_size=256
):
    retain_classes = [c for c in range(num_classes) if c != forget_class]

    ret_c2_list, ret_id_list, ret_y_list = [], [], []
    fgt_c2_list, fgt_id_list            = [], []

    for c in retain_classes:
        top_c2, top_id, _, bot_c2, bot_id, _ = sample_pre_lastconv_l4_predicted_as_class_streaming(
            head=head,
            conv2_in_shape=conv2_in_shape,
            identity_shape=identity_shape,
            target_class=c,
            need_top_k=retain_top_k,
            need_bottom_k=per_retain_for_forget,
            batch=256, device=device, dtype=torch.float16, safety_multiplier=1.2, use_amp=True
        )
        if top_c2 is None or top_c2.shape[0] < retain_top_k:
            raise RuntimeError(f"Insufficient retain feats for class {c}")
        if bot_c2 is None or bot_c2.shape[0] < per_retain_for_forget:
            raise RuntimeError(f"Insufficient forget feats for class {c}")

        ret_c2_list.append(top_c2); ret_id_list.append(top_id)
        ret_y_list.append(torch.full((retain_top_k,), c, dtype=torch.long))

        fgt_c2_list.append(bot_c2); fgt_id_list.append(bot_id)

    retain_c2  = torch.cat(ret_c2_list, 0)
    retain_id  = torch.cat(ret_id_list, 0)
    retain_y   = torch.cat(ret_y_list, 0)

    forget_c2  = torch.cat(fgt_c2_list, 0)
    forget_id  = torch.cat(fgt_id_list, 0)
    forget_y   = torch.full((forget_c2.shape[0],), forget_class, dtype=torch.long)

    retain_loader = DataLoader(TensorDataset(retain_c2, retain_id, retain_y),
                               batch_size=loader_batch_size, shuffle=True, drop_last=False)
    forget_loader = DataLoader(TensorDataset(forget_c2, forget_id, forget_y),
                               batch_size=loader_batch_size, shuffle=True, drop_last=False)

    return {
        "retain_c2": retain_c2, "retain_id": retain_id, "retain_labels": retain_y, "retain_loader": retain_loader,
        "forget_c2": forget_c2, "forget_id": forget_id, "forget_labels": forget_y, "forget_loader": forget_loader,
        "summary": {
            "space": "pre_lastconv_l4",
            "conv2_in_shape": tuple(conv2_in_shape),
            "identity_shape": tuple(identity_shape),
            "retain_top_k": retain_top_k,
            "forget_per_retain": per_retain_for_forget,
            "retain_total": retain_c2.shape[0],
            "forget_total": forget_c2.shape[0],
            "forget_class": forget_class,
        }
    }

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

    # -------- cut BEFORE last conv in layer4 --------
    feat_net, head = split_resnet18_before_last_conv_l4(model, num_classes, device=device)

    # freeze feats; train head
    for p in feat_net.parameters(): p.requires_grad = False
    for p in head.parameters():     p.requires_grad = True
    feat_net.eval()
    head.train()

    # sanity check
    _ = sanity_check_equivalence_lastconv(model, feat_net, head, all_test_loader, device)

    # Shapes at the cut
    conv2_in_shape, identity_shape = infer_pre_lastconv_l4_shapes(feat_net, all_train_loader, device=device)
    print(f"[shape] conv2_in={conv2_in_shape} | identity={identity_shape}")

    # Embed real data once (pairs)
    tr_c2, tr_id, tr_y = embed_loader_to_pre_lastconv_l4(feat_net, all_train_loader, device=device)
    te_c2, te_id, te_y = embed_loader_to_pre_lastconv_l4(feat_net, all_test_loader,  device=device)

    trF_c2, trF_id, trF_y = embed_loader_to_pre_lastconv_l4(feat_net, train_fgt_loader,    device=device)
    trR_c2, trR_id, trR_y = embed_loader_to_pre_lastconv_l4(feat_net, train_retain_loader, device=device)
    teF_c2, teF_id, teF_y = embed_loader_to_pre_lastconv_l4(feat_net, test_fgt_loader,     device=device)
    teR_c2, teR_id, teR_y = embed_loader_to_pre_lastconv_l4(feat_net, test_retain_loader,  device=device)

    # Wrap as loaders
    all_train_emb_loader = make_lastconv_emb_loader(tr_c2, tr_id, tr_y)
    all_test_emb_loader  = make_lastconv_emb_loader(te_c2, te_id, te_y)
    train_fgt_emb_loader = make_lastconv_emb_loader(trF_c2, trF_id, trF_y)
    train_ret_emb_loader = make_lastconv_emb_loader(trR_c2, trR_id, trR_y)
    test_fgt_emb_loader  = make_lastconv_emb_loader(teF_c2, teF_id, teF_y)
    test_ret_emb_loader  = make_lastconv_emb_loader(teR_c2, teR_id, teR_y)

    head.eval()
    # Build synthetic pre-lastconv splits
    synth = build_synth_pre_lastconv_l4_splits(
        head=head,
        num_classes=num_classes,
        forget_class=forget_class,
        conv2_in_shape=conv2_in_shape,
        identity_shape=identity_shape,
        retain_top_k=choose_per_retain_class_for_fgt * (num_classes - 1),  # mirrors your 9x logic on CIFAR-10
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
    lr_all = 1e-3
    wd_all = 1e-4
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr_all, weight_decay=wd_all, betas=(0.9, 0.999))
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

    # Build synthetic train set (pairs)
    train_c2  = torch.cat([synth["retain_c2"], synth["forget_c2"]], dim=0)
    train_id  = torch.cat([synth["retain_id"], synth["forget_id"]], dim=0)
    train_y   = torch.cat([synth["retain_labels"], synth["forget_labels"]], dim=0)
    train_loader_syn = DataLoader(
        list(zip(train_c2, train_id, train_y)), batch_size=128, shuffle=True, drop_last=False
    )

    # ----- Training epochs -----
    for epoch in range(1, epochs + 1):
        head.train()
        feat_net.eval()

        running_loss, right, seen = 0.0, 0, 0
        p_dtype = next(head.parameters()).dtype

        for c2, ident, targets in train_loader_syn:
            c2     = c2.to(device, non_blocking=True).to(p_dtype)
            ident  = ident.to(device, non_blocking=True).to(p_dtype)
            targets= targets.to(device, non_blocking=True)

            logits = head((c2, ident))
            loss   = criterion(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            right += (logits.argmax(dim=1) == targets).sum().item()
            seen  += targets.numel()

        train_loss = running_loss / max(seen, 1)
        train_acc  = 100.0 * right / max(seen, 1)

        # synthetic eval
        head.eval()
        acc_syn_ret   = eval_accuracy_on_pre_lastconv_emb_loader(head, syn_retain_eval_loader, device)
        acc_syn_fgt   = eval_accuracy_on_pre_lastconv_emb_loader(head, syn_forget_eval_loader, device)
        acc_syn_total = (acc_syn_ret * n_syn_ret + acc_syn_fgt * n_syn_fgt) / (n_syn_ret + n_syn_fgt)

        # real-embedding eval
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
