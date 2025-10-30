#!/usr/bin/env python3
import os, math, argparse, copy, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from method.utils import *

# your helpers
from utils import get_transforms, get_dataset, get_dataloader, get_unlearn_loader
from trainer import *  # assumes load_model is here

def _to01(xs):
    # your hist uses % (0..100). Convert to 0..1 for the plotter.
    return [float(x)/100.0 for x in xs]

def _hist_to_accs_dict(hist: dict) -> dict:
    # map your keys → the plotter’s keys
    return {
        "train_forget": _to01(hist["train_fgt"]),
        "test_forget":  _to01(hist["test_fgt"]),
        "train_remain": _to01(hist["train_retain"]),
        "test_remain":  _to01(hist["test_retain"]),
    }
    

# ---------------- Argparse ----------------
p = argparse.ArgumentParser("Class unlearning revival (from prebuilt pool)")
p.add_argument('--pool_path', type=str, required=True,
               help='Path to synthetic pool .pt created by build_synth_pool.py')
p.add_argument('--pool_take_per_class', type=int, default=None,
               help='Randomly take up to K per class from the pool BEFORE top/bottom filtering (None → take all).')

p.add_argument('--method', type=str, default='original',
              choices=['original','retrained','random_label','finetune','gradient_ascent','neggrad_plus',
                       'boundary_shrink','boundary_expand','l2ul_adv','l2ul_imp','fisher','wood_fisher','delete',
                       'bad_teacher', 'salun', 'scrub'])
p.add_argument('--model', '--model_name', dest='model_name', type=str, default='resnet18')
p.add_argument('--dataset', type=str, required=True)
p.add_argument('--lr', type=float, default=5e-5)
p.add_argument('--epochs', type=int, default=500)

p.add_argument('--forget', type=str, default='all',
               help="Classes to forget: 'all' or like '7' or '1,3,5' or '2-6' or mixed '0,2-4,7'.")

# selection sizes AFTER subsampling
p.add_argument('--retain_per_class', type=int, default=500,
               help='Top-K retain synthetic embeddings per non-forget class (by confidence).')
p.add_argument('--forget_per_class', type=int, default=10,
               help='Bottom-K synthetic embeddings per non-forget class (relabel to forget class).')

# RS / constraints
p.add_argument('--rs_patience', type=int, default=500)
p.add_argument('--rs_directional', action='store_true')
p.add_argument('--retain_floor_frac', type=float, default=0.90,
               help='Min fraction of baseline retain accuracy.')

# infra
p.add_argument('--seed', type=int, default=0)
p.add_argument('--ckpt_dir', default='/export/livia/home/vision/Zdehghani/classification/exps')
args = p.parse_args()

# ---------------- Setup ----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
rng_cpu = torch.Generator(device="cpu").manual_seed(args.seed)

NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'tiny_imagenet': 200, 'imagenet': 1000}
dataset_name = args.dataset
try:
    num_classes = NUM_CLASSES[dataset_name.lower()]
except KeyError:
    raise ValueError(f"Unknown dataset '{dataset_name}'. Add it to NUM_CLASSES.")

model_name = args.model_name
method = args.method
base_dir = args.ckpt_dir
lr = args.lr
epochs = args.epochs


COLUMNS = [
    "forget_class", "dataset", "model", "method", "lr",
    "epochs_total", "tpr", "forget_per_class",
    "pool_take_per_class",
    "retain_per_class", "total_retain", "total_forget",
    "chosen_total",                     
    "epoch", "syn_train_loss", "syn_train_acc",
    "syn_total", "syn_retain", "syn_forget",
    "all_train", "all_test",
    "train_fgt", "train_retain", "test_fgt", "test_retain",
    "RS",
]

# ---------------- Small helpers ----------------
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
            if a > b: a, b = b, a
            for c in range(a, b + 1):
                if 0 <= c < num_classes: selected.add(c)
        else:
            c = int(part)
            if 0 <= c < num_classes: selected.add(c)
    if not selected:
        raise ValueError(f"No valid classes parsed from --forget='{s}'. "
                         f"Valid range is 0..{num_classes-1} or 'all'.")
    return sorted(selected)

def make_feature_extractor(net, num_classes):
    feat = copy.deepcopy(net).eval().to(device)
    if hasattr(feat, "fc") and isinstance(feat.fc, nn.Linear) and feat.fc.out_features == num_classes:
        feat.fc = nn.Identity(); return feat
    if hasattr(feat, "head") and isinstance(feat.head, nn.Linear) and feat.head.out_features == num_classes:
        feat.head = nn.Identity(); return feat
    if hasattr(feat, "heads") and hasattr(feat.heads, "head") and \
       isinstance(feat.heads.head, nn.Linear) and feat.heads.head.out_features == num_classes:
        feat.heads.head = nn.Identity(); return feat
    last_linear = None
    for name, m in reversed(list(feat.named_modules())):
        if isinstance(m, nn.Linear) and m.out_features == num_classes:
            last_linear = name; break
    if last_linear is None:
        raise RuntimeError("Cannot find final Linear layer to strip for feature extractor.")
    def set_module(root, dotted, new):
        parent = root
        for p in dotted.split(".")[:-1]:
            parent = getattr(parent, p)
        setattr(parent, dotted.split(".")[-1], new)
    set_module(feat, last_linear, nn.Identity())
    return feat

def make_emb_loader(x, y, bs=4096, shuffle=False, num_workers=4):
    ds = TensorDataset(x, y)
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle, drop_last=False,
        pin_memory=(device == 'cuda'), num_workers=num_workers,
        persistent_workers=(num_workers > 0), prefetch_factor=(4 if num_workers > 0 else None)
    )

@torch.inference_mode()
def eval_accuracy_on_emb_loader(fc, loader, device):
    total, correct = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits = fc(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total   += y.numel()
    return 100.0 * correct / max(total, 1)

@torch.inference_mode()
def embed_loader_to_tensor(feature_model, loader):
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = feature_model(x)
        feats.append(f.detach().cpu()); labels.append(y.cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)

def revival_score(A_r_tr, A_f_tr, A_r_tu, A_f_tu, directional=False):
    retain_term = 1.0 - abs(A_r_tr - A_r_tu) / 100.0
    forget_delta = (A_f_tr - A_f_tu) / 100.0
    forget_term = max(0.0, forget_delta) if directional else abs(forget_delta)
    retain_term = float(np.clip(retain_term, 0.0, 1.0))
    forget_term = float(np.clip(forget_term, 0.0, 1.0))
    return retain_term * forget_term

# 2) In append_best_for_class(), compute and include it
def append_best_for_class(
    forget_class: int, best_row: dict,
    retain_per_class: int, total_retain: int, total_forget: int,
    pool_take_per_class: int,   # <-- NEW PARAM
):
    if best_row is None:
        print(f"[AGG] No best row to append for class {forget_class}."); return
    row = {
        "forget_class": int(forget_class),
        "dataset": dataset_name, "model": model_name,
        "method": method, "lr": lr, "epochs_total": epochs,
        "tpr": -1,
        "forget_per_class": int(args.forget_per_class),
        "pool_take_per_class": int(pool_take_per_class),   # <-- WRITE IT
        "retain_per_class": int(retain_per_class),
        "total_retain": int(total_retain),
        "total_forget": int(total_forget),
        "chosen_total": int(total_retain + total_forget),
        **best_row,
    }
    df = pd.DataFrame([row], columns=COLUMNS)
    header_needed = not os.path.exists(AGG_CSV_PATH)
    METRIC_COLS = [
        "epochs_total", "tpr", "forget_per_class",
        "pool_take_per_class",                 # <-- ADD HERE
        "retain_per_class",
        "total_retain", "total_forget", "chosen_total",
        "syn_train_loss", "syn_train_acc",
        "syn_total", "syn_retain", "syn_forget", "all_train", "all_test",
        "train_fgt", "train_retain", "test_fgt", "test_retain", "RS",
    ]
    df[METRIC_COLS] = df[METRIC_COLS].apply(pd.to_numeric, errors="coerce").round(3)
    df.to_csv(AGG_CSV_PATH, mode="a", header=header_needed, index=False, float_format="%.3f")
    print(f"[AGG] Appended best row for class {forget_class} -> {AGG_CSV_PATH}")



# ---------------- Pool selection ----------------
def load_synth_pool(path: str):
    print(f"[pool] loading {path}")
    obj = torch.load(path, map_location='cpu')
    assert 'meta' in obj and 'classes' in obj, "Invalid pool file structure"
    return obj

def select_from_pool_for_class(pool, class_id: int, take_k: int, rng: torch.Generator):
    entry = pool['classes'][int(class_id)]
    feats = entry['feats']  # (N, emb) fp16 CPU
    probs = entry['probs']  # (N,)     fp16 CPU
    N = feats.shape[0]
    if (take_k is None) or (take_k <= 0) or (take_k >= N):
        idx = torch.arange(N)
    else:
        idx = torch.randperm(N, generator=rng)[:take_k]
    return feats[idx], probs[idx]

def build_synth_from_pool(
    pool, num_classes, forget_class, retain_top_k, per_retain_for_forget,
    take_per_class: int, rng: torch.Generator, loader_batch_size: int = 256
):
    retain_classes = [c for c in range(num_classes) if c != forget_class]
    emb_dim = int(pool['meta']['emb_dim'])

    retain_feats_list, retain_labels_list = [], []
    forget_feats_list = []

    for c in retain_classes:
        feats16, probs16 = select_from_pool_for_class(pool, c, take_per_class, rng)
        # cast to fp32 for ordering & training stability
        probs32 = probs16.to(torch.float32)

        # retain: top-K by confidence
        _, idx_desc = torch.sort(probs32, descending=True)
        top_idx = idx_desc[:retain_top_k]
        retain_feats_list.append(feats16[top_idx].to(torch.float32))
        retain_labels_list.append(torch.full((top_idx.numel(),), c, dtype=torch.long))

        # forget: bottom-K by confidence
        _, idx_asc = torch.sort(probs32, descending=False)
        low_idx = idx_asc[:per_retain_for_forget]
        forget_feats_list.append(feats16[low_idx].to(torch.float32))

    retain_feats  = torch.cat(retain_feats_list, dim=0) if retain_feats_list else torch.empty(0, emb_dim)
    retain_labels = torch.cat(retain_labels_list, dim=0) if retain_labels_list else torch.empty(0, dtype=torch.long)
    forget_feats  = torch.cat(forget_feats_list, dim=0) if forget_feats_list else torch.empty(0, emb_dim)
    forget_labels = torch.full((forget_feats.shape[0],), forget_class, dtype=torch.long)

    retain_loader = DataLoader(TensorDataset(retain_feats, retain_labels),
                               batch_size=loader_batch_size, shuffle=True, drop_last=False)
    forget_loader = DataLoader(TensorDataset(forget_feats, forget_labels),
                               batch_size=loader_batch_size, shuffle=True, drop_last=False)

    summary = {
        "emb_dim": emb_dim,
        "retain_classes": retain_classes,
        "retain_top_k_per_class": int(retain_top_k),
        "retain_total": int(retain_feats.shape[0]),
        "forget_class": int(forget_class),
        "per_retain_for_forget": int(per_retain_for_forget),
        "forget_total": int(forget_feats.shape[0]),
        "take_per_class": int(take_per_class if take_per_class is not None else -1),
        "pool_path": args.pool_path,
    }
    return {
        "retain_feats": retain_feats, "retain_labels": retain_labels, "retain_loader": retain_loader,
        "forget_feats": forget_feats, "forget_labels": forget_labels, "forget_loader": forget_loader,
        "summary": summary
    }

# ---------------- Main loop per forget class ----------------
forget_classes = _parse_forget_arg(args.forget, num_classes)
sel_take = args.pool_take_per_class if args.pool_take_per_class is not None else -1
PLOTS_TAG = f"ret{args.retain_per_class}_fg{args.forget_per_class}_take{sel_take}_seed{args.seed}"
# root for plots (per method + config); include pool tag too if you want
pool_tag = Path(args.pool_path).stem
PLOTS_DIR = os.path.join("results", method, "plots", f"{pool_tag}_{PLOTS_TAG}")
os.makedirs(PLOTS_DIR, exist_ok=True)
AGG_CSV_DIR = os.path.join("results", method)
os.makedirs(AGG_CSV_DIR, exist_ok=True)
AGG_CSV_PATH = os.path.join(
    AGG_CSV_DIR,
    f"{dataset_name}_{model_name}_{method}_revival_by_forget_class_lr{lr}.csv"
)
pool = load_synth_pool(args.pool_path)
take_per_class = args.pool_take_per_class

for forget_class in forget_classes:
    print(f"\n================= FORGET CLASS {forget_class} =================")
    # ckpt + model
    run_plots_dir = os.path.join(PLOTS_DIR, f"fg{forget_class}")
    os.makedirs(run_plots_dir, exist_ok=True)
    ckpt = checkpoint_for(method, dataset_name, model_name, forget_class, lr, base_dir)
    model = load_model(ckpt, model_name, dataset_name, num_classes).to(device).eval()

    # --- init plot histories + per-epoch CSV path ---
    hist = {
        "train_fgt":   [],
        "test_fgt":    [],
        "train_retain":[],
        "test_retain": [],
        "RS":          [],
    }
    
    per_epoch_csv = os.path.join(run_plots_dir, "metrics_by_epoch.csv")
    if not os.path.exists(per_epoch_csv):
        pd.DataFrame(columns=[
            "epoch","syn_train_loss","syn_train_acc","syn_total","syn_retain","syn_forget",
            "all_train","all_test","train_fgt","train_retain","test_fgt","test_retain","RS"
        ]).to_csv(per_epoch_csv, index=False)


    # datasets & loaders (real images)
    wo_dataaug = False
    transform_train, transform_test = get_transforms(dataset_name, model_name, wo_dataaug=wo_dataaug)
    trainset, testset = get_dataset(dataset_name, transform_train, transform_test)
    batch_size_real = 256; num_workers = 8
    all_train_loader, all_test_loader = get_dataloader(
        trainset, testset, batch_size=batch_size_real, num_workers=num_workers
    )
    # deterministic train-eval loader (no aug, no shuffle) for measuring baseline train stats
    trainset_eval = copy.copy(trainset); trainset_eval.transform = transform_test
    all_train_loader = DataLoader(trainset_eval, batch_size=batch_size_real, shuffle=False,
                                  drop_last=False, num_workers=num_workers, pin_memory=True)

    # one-vs-all splits (real images)
    (
        train_fgt_loader, train_retain_loader,
        test_fgt_loader,  test_retain_loader,
        _repair_loader, _train_fgt_idx, _train_retain_idx,
        _test_fgt_idx,  _test_retain_idx,
    ) = get_unlearn_loader(
        trainset, testset, [forget_class],
        batch_size=batch_size_real, num_workers=num_workers, num_forget=float("inf")
    )

    # embed once (real) to speed eval by using FC only
    feature_model = make_feature_extractor(model, num_classes).eval()
    all_train_feats, all_train_labels = embed_loader_to_tensor(feature_model, all_train_loader)
    all_test_feats,  all_test_labels  = embed_loader_to_tensor(feature_model, all_test_loader)
    train_fgt_feats, train_fgt_labels = embed_loader_to_tensor(feature_model, train_fgt_loader)
    train_ret_feats, train_ret_labels = embed_loader_to_tensor(feature_model, train_retain_loader)
    test_fgt_feats,  test_fgt_labels  = embed_loader_to_tensor(feature_model, test_fgt_loader)
    test_ret_feats,  test_ret_labels  = embed_loader_to_tensor(feature_model, test_retain_loader)

    # wrap as embedding loaders
    all_train_emb_loader = make_emb_loader(all_train_feats, all_train_labels)
    all_test_emb_loader  = make_emb_loader(all_test_feats,  all_test_labels)
    train_fgt_emb_loader = make_emb_loader(train_fgt_feats, train_fgt_labels)
    train_ret_emb_loader = make_emb_loader(train_ret_feats, train_ret_labels)
    test_fgt_emb_loader  = make_emb_loader(test_fgt_feats,  test_fgt_labels)
    test_ret_emb_loader  = make_emb_loader(test_ret_feats,  test_ret_labels)

    # === Build synthetic (from pool) ===
    synth = build_synth_from_pool(
        pool=pool,
        num_classes=num_classes,
        forget_class=forget_class,
        retain_top_k=int(args.retain_per_class),
        per_retain_for_forget=int(args.forget_per_class),
        take_per_class=take_per_class,
        rng=rng_cpu,
        loader_batch_size=256,
    )
    print("Synthetic selection summary:", synth["summary"])

    # eval loaders (synthetic subsets)
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

    # Freeze backbone; train only FC
    for p in model.parameters(): p.requires_grad = False
    fc = _get_final_linear(model, num_classes)
    for p in fc.parameters(): p.requires_grad = True
    model.eval()

    # Train set from synthetic data (retain + forget)
    train_feats = torch.cat([synth["retain_feats"], synth["forget_feats"]], dim=0)
    train_labels = torch.cat([synth["retain_labels"], synth["forget_labels"]], dim=0)
    train_loader_syn = DataLoader(list(zip(train_feats, train_labels)),
                                  batch_size=256, shuffle=True, drop_last=False)

    # Baseline @ epoch 0
    fc.eval()
    @torch.no_grad()
    def eval_epoch(fc):
        acc_syn_ret   = eval_accuracy_on_emb_loader(fc, syn_retain_eval_loader, device)
        acc_syn_fgt   = eval_accuracy_on_emb_loader(fc, syn_forget_eval_loader, device)
        acc_syn_total = (acc_syn_ret * n_syn_ret + acc_syn_fgt * n_syn_fgt) / max(n_syn_ret + n_syn_fgt, 1)
        acc_all_train    = eval_accuracy_on_emb_loader(fc, all_train_emb_loader,    device)
        acc_all_test     = eval_accuracy_on_emb_loader(fc, all_test_emb_loader,     device)
        acc_train_fgt    = eval_accuracy_on_emb_loader(fc, train_fgt_emb_loader,    device)
        acc_train_retain = eval_accuracy_on_emb_loader(fc, train_ret_emb_loader,    device)
        acc_test_fgt     = eval_accuracy_on_emb_loader(fc, test_fgt_emb_loader,     device)
        acc_test_retain  = eval_accuracy_on_emb_loader(fc, test_ret_emb_loader,     device)
        return {
            "syn_total": acc_syn_total, "syn_retain": acc_syn_ret, "syn_forget": acc_syn_fgt,
            "all_train": acc_all_train, "all_test": acc_all_test,
            "train_fgt": acc_train_fgt, "train_retain": acc_train_retain,
            "test_fgt": acc_test_fgt,   "test_retain": acc_test_retain,
        }

    m0 = eval_epoch(fc)
    print(
        f"[Epoch 00] syn_total={m0['syn_total']:.2f}% syn_ret={m0['syn_retain']:.2f}% syn_fgt={m0['syn_forget']:.2f}% | "
        f"all_train={m0['all_train']:.2f}% all_test={m0['all_test']:.2f}% | "
        f"train_fgt={m0['train_fgt']:.2f}% train_ret={m0['train_retain']:.2f}% | "
        f"test_fgt={m0['test_fgt']:.2f}% test_ret={m0['test_retain']:.2f}%"
    )

    A_r_tu = float(m0["test_retain"])
    A_f_tu = float(m0["test_fgt"])
    A_r_tr = float(m0["test_retain"])
    A_f_tr = float(m0["test_fgt"])
    rs0 = revival_score(A_r_tr, A_f_tr, A_r_tu, A_f_tu, directional=args.rs_directional)
    print(f"[Epoch 00] RS={rs0:.4f}")

    # Early-stop on RS with retain floor
    retain_floor = args.retain_floor_frac * A_r_tu
    print(f"[Constraint] test_retain must be >= {retain_floor:.2f} "
          f"({args.retain_floor_frac:.0%} of baseline {A_r_tu:.2f}).")

    best = {"key": (-float("inf"), -float("inf"), -float("inf"), -float("inf"), -float("inf")), "row": None}
    def _safe_get(row, k):
        v = row.get(k, float("-inf"))
        return v if (isinstance(v, (int, float)) and v == v) else float("-inf")
    def _key_from_row(row):
        # priority: RS, then test_fgt, test_retain, train_fgt, train_retain
        return (float(_safe_get(row, "RS")),
                float(_safe_get(row, "test_fgt")),
                float(_safe_get(row, "test_retain")),
                float(_safe_get(row, "train_fgt")),
                float(_safe_get(row, "train_retain")))
    row0 = {
        "epoch": 0, "syn_train_loss": None, "syn_train_acc": None,
        **m0, "RS": float(rs0)
    }
    best["row"] = row0.copy(); best["key"] = _key_from_row(row0)

    # record epoch-0 metrics and plot
    hist["train_fgt"].append(float(m0["train_fgt"]))
    hist["test_fgt"].append(float(m0["test_fgt"]))
    hist["train_retain"].append(float(m0["train_retain"]))
    hist["test_retain"].append(float(m0["test_retain"]))
    hist["RS"].append(float(rs0))
    accs_dict = _hist_to_accs_dict(hist)
    epoch_for_plot = len(accs_dict["train_forget"])
    plot_unlearn_remain_acc_figure(
        epoch=epoch_for_plot,
        accs_dict=accs_dict,
        experiment_path=Path(run_plots_dir),
        plot_type="plot",
    )

    # append epoch-0 row to per-epoch CSV
    pd.DataFrame([{
        "epoch": 0, "syn_train_loss": None, "syn_train_acc": None,
        "syn_total": m0["syn_total"], "syn_retain": m0["syn_retain"], "syn_forget": m0["syn_forget"],
        "all_train": m0["all_train"], "all_test": m0["all_test"],
        "train_fgt": m0["train_fgt"], "train_retain": m0["train_retain"],
        "test_fgt": m0["test_fgt"], "test_retain": m0["test_retain"],
        "RS": rs0
    }]).to_csv(per_epoch_csv, mode="a", header=False, index=False)







    # Optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fc.parameters(), lr=1e-2, weight_decay=1e-4)

    best_rs = -1.0; best_rs_epoch = 0
    no_improve = 0; patience = int(args.rs_patience)
    saved_fc = None

    # Train epochs
    for epoch in range(1, epochs + 1):
        fc.train()
        run_loss, seen, right = 0.0, 0, 0
        for feats, targets in train_loader_syn:
            feats = feats.to(device); targets = targets.to(device)
            logits = fc(feats)
            loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            run_loss += loss.item() * feats.size(0)
            right += (logits.argmax(dim=1) == targets).sum().item()
            seen  += targets.numel()
        train_loss = run_loss / max(seen, 1); train_acc = 100.0 * right / max(seen, 1)

        # eval
        fc.eval()
        m = eval_epoch(fc)
        A_r_tr, A_f_tr = float(m["test_retain"]), float(m["test_fgt"])
        rs = revival_score(A_r_tr, A_f_tr, A_r_tu, A_f_tu, directional=args.rs_directional)

        print(
            f"[Epoch {epoch:02d}] syn_train_loss={train_loss:.4f} syn_train_acc={train_acc:.2f}% | "
            f"syn_total={m['syn_total']:.2f}% syn_ret={m['syn_retain']:.2f}% syn_fgt={m['syn_forget']:.2f}% | "
            f"all_train={m['all_train']:.2f}% all_test={m['all_test']:.2f}% | "
            f"train_fgt={m['train_fgt']:.2f}% train_ret={m['train_retain']:.2f}% | "
            f"test_fgt={m['test_fgt']:.2f}% test_ret={m['test_retain']:.2f}% | RS={rs:.4f}"
        )

        row = {"epoch": epoch, "syn_train_loss": float(train_loss), "syn_train_acc": float(train_acc), **m, "RS": float(rs)}
        key = _key_from_row(row)
        if (m["test_retain"] >= retain_floor) and (key > best["key"]):
            best["key"] = key; best["row"] = row.copy()

        # record epoch-k metrics and plot
        hist["train_fgt"].append(float(m["train_fgt"]))
        hist["test_fgt"].append(float(m["test_fgt"]))
        hist["train_retain"].append(float(m["train_retain"]))
        hist["test_retain"].append(float(m["test_retain"]))
        hist["RS"].append(float(rs))

        accs_dict = _hist_to_accs_dict(hist)
        epoch_for_plot = len(accs_dict["train_forget"])
        plot_unlearn_remain_acc_figure(
            epoch=epoch_for_plot,
            accs_dict=accs_dict,
            experiment_path=Path(run_plots_dir),
            plot_type="plot",
        )





        # append epoch-k row to per-epoch CSV
        pd.DataFrame([{
            "epoch": epoch, "syn_train_loss": float(train_loss), "syn_train_acc": float(train_acc),
            "syn_total": m["syn_total"], "syn_retain": m["syn_retain"], "syn_forget": m["syn_forget"],
            "all_train": m["all_train"], "all_test": m["all_test"],
            "train_fgt": m["train_fgt"], "train_retain": m["train_retain"],
            "test_fgt": m["test_fgt"], "test_retain": m["test_retain"],
            "RS": rs
        }]).to_csv(per_epoch_csv, mode="a", header=False, index=False)



        # RS early stop w/ retain floor guard
        if (m["test_retain"] >= retain_floor) and (rs > best_rs):
            best_rs = rs; best_rs_epoch = epoch; no_improve = 0
            saved_fc = copy.deepcopy(fc.state_dict())
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[EarlyStop] RS plateaued for {patience} epochs. Best RS={best_rs:.4f} @ epoch {best_rs_epoch}.")
                if saved_fc is not None: fc.load_state_dict(saved_fc)
                break

    # log one line per forget class
    retain_per_class = int(args.retain_per_class)
    total_retain     = int(synth["summary"]["retain_total"])
    total_forget     = int(synth["summary"]["forget_total"])
    append_best_for_class(
        forget_class,
        best["row"],
        retain_per_class=retain_per_class,
        total_retain=total_retain,
        total_forget=total_forget,
        pool_take_per_class=int(synth["summary"]["take_per_class"]),  # <-- PASS IT
    )
