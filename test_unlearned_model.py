import argparse
from pathlib import Path
import torch

from utils import get_transforms, get_dataset, get_dataloader, get_unlearn_loader, gather_and_write_metrics_csv
from trainer import load_model
from project_paths import EXPS_DIR

import torch.nn.functional as F

@torch.inference_mode()
def _collect_stats(model, loader, device):
    """Return per-example tensors: max_confidence, entropy, m_entropy, correctness."""
    confs, ents, ments, corrs = [], [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        probs = F.softmax(logits, dim=1)

        max_conf, preds = probs.max(dim=1)
        correctness = (preds == y).float()

        # Shannon entropy over predicted distribution
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)

        # "modified entropy" = negative log prob of the true label
        true_p = probs.gather(1, y.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
        m_entropy = -true_p.log()

        confs.append(max_conf.detach())
        ents.append(entropy.detach())
        ments.append(m_entropy.detach())
        corrs.append(correctness.detach())

    return (
        torch.cat(confs),     # [N]
        torch.cat(ents),      # [N]
        torch.cat(ments),     # [N]
        torch.cat(corrs),     # [N]
    )

def _mia_summary_for_forget(model, train_forget_loader, test_forget_loader, device):
    """
    Summarize MIA using the forget splits:
      - Report member (train_forget) means for CSV columns (correctness/confidence/entropy/m_entropy)
      - Compute a simple threshold attack accuracy using max-confidence.
    """
    m_conf, m_ent, m_ment, m_corr = _collect_stats(model, train_forget_loader, device)
    nm_conf, nm_ent, nm_ment, nm_corr = _collect_stats(model, test_forget_loader, device)

    # member means (what we’ll store in CSV individual columns)
    member_means = {
        "correctness": float(m_corr.mean().item()),
        "confidence":  float(m_conf.mean().item()),
        "entropy":     float(m_ent.mean().item()),
        "m_entropy":   float(m_ment.mean().item()),
    }

    # Simple threshold attack on confidence: label 1=member (train_forget), 0=non-member (test_forget)
    all_conf = torch.cat([m_conf, nm_conf])
    labels   = torch.cat([torch.ones_like(m_conf), torch.zeros_like(nm_conf)])

    # threshold = midpoint of group means
    thr = 0.5 * (m_conf.mean() + nm_conf.mean())
    preds = (all_conf >= thr).float()
    attack_acc = float((preds == labels).float().mean().item())

    member_means["prob"] = attack_acc  # goes to 'mia_prob' in CSV via utils

    return member_means

def _build_tags(forget_spec):
    """
    Accepts int or list of ints. Returns:
      retrain_tag:  'forgetcls{ID}' if single, else 'forget{K}'
      dir_tag:      same rule, used for directory names
    """
    if isinstance(forget_spec, int):
        ids = [forget_spec]
    else:
        ids = sorted(set(int(x) for x in forget_spec))
    if len(ids) == 1:
        return f"forgetcls{ids[0]}", f"forgetcls{ids[0]}"
    else:
        k = len(ids)
        return f"forget{k}", f"forget{k}"
    
def checkpoint_for(method, dataset_name, model_name, forget_spec, lr, base_dir):
    retrain_tag, dir_tag = _build_tags(forget_spec)
    if method == 'original':
        return f"{base_dir}/test_pretrained_model/{dataset_name}_{model_name}_original_model.pth"
    elif method == 'retrained':
        # single: ..._retrain_forgetcls{ID}_model.pth
        # multi : ..._retrain_forget{K}_model.pth
        return f"{base_dir}/test_pretrained_model/{dataset_name}_{model_name}_retrain_{retrain_tag}_model.pth"
    else:
        if lr is None:
            raise ValueError("--unlearn_rate is required (or pass --ckpt_path).")
        # single: ..._forgetcls{ID}/{method}/lr{lr}/ckpt_best_by_aus.pth
        # multi : ..._forget{K}/{method}/lr{lr}/ckpt_best_by_aus.pth
        return f"{base_dir}/{dataset_name}_{model_name}_{dir_tag}/{method}/lr{lr}/ckpt_best_by_aus.pth"


def main():
    p = argparse.ArgumentParser("Eval-only for an unlearned checkpoint")
    p.add_argument('--dataset_name', required=True, choices=['cifar10','cifar100','tiny_imagenet','vggface'])
    p.add_argument('--model_name',   required=True, choices=['resnet18','vgg16','vit-s-16','swin-t','vit-b-16'])
    p.add_argument('--method',       required=True)          
    p.add_argument('--forget_id', type=int, required=False, help="Single class ID (one-vs-all). Use --forget_set for multi.")
    p.add_argument('--forget_set', type=int, nargs='+', default=None, help="Union of classes to forget, e.g. --forget_set 1 3 7")    
            
    
    p.add_argument('--unlearn_rate', type=float, default=None, help="Required for non-original/non-retrained unless --ckpt_path is provided.")
    p.add_argument('--exps_dir',     type=str, default=str(EXPS_DIR))
    p.add_argument('--batch_size',   type=int, default=256)
    p.add_argument('--num_workers',  type=int, default=8)
    p.add_argument('--ckpt_path',    type=str, default=None, help="(optional) direct path to .pth")
    p.add_argument(
        '--out_csv',
        type=str,
        default=None,
        help="Optional exact output CSV path. Defaults to results/<method>/...",
    )
    
    args = p.parse_args()

    if args.forget_set is None and args.forget_id is None:
        p.error("Provide either --forget_id or --forget_set.")

    if args.method not in ['original', 'retrained'] and args.ckpt_path is None and args.unlearn_rate is None:
        p.error("--unlearn_rate is required for this method unless you pass --ckpt_path.")
        

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- model + data ----
    transform_train, transform_test = get_transforms(args.dataset_name, args.model_name, wo_dataaug=False)
    trainset, testset = get_dataset(args.dataset_name, transform_train, transform_test)
    train_loader, test_loader = get_dataloader(trainset, testset, batch_size=args.batch_size, num_workers=args.num_workers)

    num_classes = max(train_loader.dataset.targets) + 1  # your trainer does the same
    
    if args.forget_set:
        forget_indices = sorted(set(int(c) for c in args.forget_set))  # e.g., [1,3,7]
    else:
        forget_indices = [args.forget_id]  # one-vs-all fallback
        
    (train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, _, 
    train_forget_index, train_remain_index, test_forget_index, test_remain_index) = get_unlearn_loader(
        trainset, testset, forget_indices, args.batch_size, float("inf"), args.num_workers
    )

    # ---- load unlearned checkpoint ----
    if args.ckpt_path:
        ckpt = args.ckpt_path
    else:
        base = str(Path(args.exps_dir).expanduser())
        ckpt = checkpoint_for(args.method, args.dataset_name, args.model_name, forget_indices, args.unlearn_rate, base)

    print(f"[INFO] Using checkpoint: {ckpt}")
    model = load_model(ckpt, args.model_name, args.dataset_name, num_classes).to(device).eval()


    forget_count = len(set(forget_indices))   # e.g., 3 for [1,3,7]

    # (optional) zero-pad for nicer sorting: f"{forget_count:02d}"
    out_csv = Path(args.out_csv).expanduser() if args.out_csv else (
        Path("results") / args.method / (
            f"{args.dataset_name}_{args.model_name}_unlearned_"
            f"{args.method}_forget{forget_count}_model_metrics_lr{args.unlearn_rate}.csv"
        )
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    mia_dict = _mia_summary_for_forget(model, train_forget_loader, test_forget_loader, device)

    # ---- write metrics CSV (overall + per-class + MIA placeholders) ----
    gather_and_write_metrics_csv(
        csv_path=str(out_csv),
        model=model,
        method=args.method,
        forget_class=forget_indices,
        train_retain_loader=train_remain_loader,
        train_forget_loader=train_forget_loader,
        test_retain_loader=test_remain_loader,
        test_forget_loader=test_forget_loader,
        train_full_loader=train_loader,
        test_full_loader=test_loader,
        mia_result=mia_dict,   
    )

    print(f"[OK] Wrote {out_csv}")

if __name__ == "__main__":
    main()
