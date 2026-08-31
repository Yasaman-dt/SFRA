import os, math, argparse, torch
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
from trainer import *  # load_model
from project_paths import EXPS_DIR

device = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'tiny_imagenet': 200, 'imagenet': 1000}

def _get_final_linear(model, num_classes):
    for _, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear) and module.out_features == num_classes:
            return module
    raise RuntimeError("Could not find final Linear layer with out_features == num_classes.")

@torch.no_grad()
def _sample_predicted_as_class(W, b, emb_dim, target_class, want, batch=4096, device="cuda", dtype=torch.float16):
    feats_buf, probs_buf = [], []
    got = 0
    while got < want:
        n = min(batch, want - got)
        feats = torch.randn(n, emb_dim, device=device, dtype=torch.float32)  # fp32 for logits stability
        logits = feats @ W.T + b
        probs  = F.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)
        mask   = (preds == target_class)
        if mask.any():
            sel = mask.nonzero(as_tuple=True)[0]
            feats_buf.append(feats[sel].to(dtype).cpu())
            probs_buf.append(probs[sel, target_class].to(dtype).cpu())
            got += sel.numel()
    feats_cat = torch.cat(feats_buf, dim=0)[:want]
    probs_cat = torch.cat(probs_buf, dim=0)[:want]
    return feats_cat, probs_cat

def checkpoint_for(method, dataset_name, model_name, forget_class, lr, base_dir):
    if method == 'original':
        return f"{base_dir}/test_pretrained_model/{dataset_name}_{model_name}_original_model.pth"
    elif method == 'retrained':
        return f"{base_dir}/test_pretrained_model/{dataset_name}_{model_name}_retrain_forgetcls{forget_class}_model.pth"
    else:
        return f"{base_dir}/{dataset_name}_{model_name}_forgetcls{forget_class}/{method}/lr{lr}/ckpt_best_by_aus.pth"

def main():
    p = argparse.ArgumentParser("Build synthetic pool")
    p.add_argument('--dataset', required=True)
    p.add_argument('--model_name', default='resnet18')
    p.add_argument('--method', default='original')
    p.add_argument('--forget_class', type=int, default=0)   # only for ckpt path
    p.add_argument('--lr', type=float, default=5e-5)        # only for ckpt path
    p.add_argument('--ckpt_dir', default=str(EXPS_DIR))
    p.add_argument('--per_class', type=int, required=True,
                   help='Generate exactly this many samples per class')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    num_classes = NUM_CLASSES[args.dataset.lower()]
    per_class = int(args.per_class)
    total_target = per_class * num_classes

    ckpt = checkpoint_for(args.method, args.dataset, args.model_name, args.forget_class, args.lr, args.ckpt_dir)
    model = load_model(ckpt, args.model_name, args.dataset, num_classes).to(device).eval()

    fc = _get_final_linear(model, num_classes)
    emb_dim = fc.in_features
    W = fc.weight.detach().to(device)
    b = (fc.bias.detach().to(device) if fc.bias is not None else torch.zeros(num_classes, device=device))

    pool = {
        'meta': {
            'dataset': args.dataset,
            'model_name': args.model_name,
            'method': args.method,
            'forget_class': int(args.forget_class),
            'lr': float(args.lr),
            'num_classes': num_classes,
            'emb_dim': emb_dim,
            'per_class': per_class,
            'total_target': total_target,
            'dtype': 'float16',
            'seed': args.seed,
        },
        'classes': {}   # c -> {'feats': (N, emb), 'probs': (N,)}
    }

    for c in range(num_classes):
        feats, probs = _sample_predicted_as_class(W, b, emb_dim, c, per_class,
                                                  batch=65536, device=device, dtype=torch.float16)
        pool['classes'][int(c)] = {'feats': feats, 'probs': probs}
        print(f"[pool] class {c}: {feats.shape[0]} feats (target {per_class}), emb_dim={emb_dim}")

    out_path = (
        f"synth_pools/{args.dataset}_{args.model_name}"
        f"_{args.method}_fg{int(args.forget_class)}_lr{args.lr}"
        f"_pool_{per_class}perclass_{num_classes}cls_emb{emb_dim}_seed{args.seed}.pt"
    )
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    torch.save(pool, out_path)
    print(f"[pool] saved → {out_path}")

if __name__ == '__main__':
    main()
