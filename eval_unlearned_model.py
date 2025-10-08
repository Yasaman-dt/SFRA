import argparse
from pathlib import Path
import torch

from utils import get_transforms, get_dataset, get_dataloader, get_unlearn_loader, gather_and_write_metrics_csv
from trainer import load_model

def checkpoint_for(method, dataset_name, model_name, forget_class, lr, base_dir):
    # Matches your revival.py behavior
    if method == 'original':
        return f"{base_dir}/test_pretrained_model/{dataset_name}_{model_name}_original_model.pth"
    elif method == 'retrained':
        return f"{base_dir}/test_pretrained_model/{dataset_name}_{model_name}_retrain_forgetcls{forget_class}_model.pth"
    else:
        return f"{base_dir}/{dataset_name}_{model_name}_forgetcls{forget_class}/{method}/lr{lr}/ckpt_best_by_aus.pth"

def main():
    p = argparse.ArgumentParser("Eval-only for an unlearned checkpoint")
    p.add_argument('--dataset_name', required=True, choices=['cifar10','cifar100','tiny_imagenet','vggface'])
    p.add_argument('--model_name',   required=True, choices=['resnet18','vgg16','vit-s-16','swin-t','vit-b-16'])
    p.add_argument('--method',       required=True)           # e.g., neggrad_plus
    p.add_argument('--forget_id',    type=int, required=True) # single class ID (one-vs-all)
    p.add_argument('--unlearn_rate', type=float, required=True)
    p.add_argument('--exps_dir',     type=str, default="~/classification/exps")
    p.add_argument('--batch_size',   type=int, default=256)
    p.add_argument('--num_workers',  type=int, default=8)
    p.add_argument('--ckpt_path',    type=str, default=None, help="(optional) direct path to .pth")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- model + data ----
    transform_train, transform_test = get_transforms(args.dataset_name, args.model_name, wo_dataaug=False)
    trainset, testset = get_dataset(args.dataset_name, transform_train, transform_test)
    train_loader, test_loader = get_dataloader(trainset, testset, batch_size=args.batch_size, num_workers=args.num_workers)

    num_classes = max(train_loader.dataset.targets) + 1  # your trainer does the same
    forget_class_index = [args.forget_id]  # one-vs-all split
    train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, _, \
    train_forget_index, train_remain_index, test_forget_index, test_remain_index = \
        get_unlearn_loader(trainset, testset, forget_class_index, args.batch_size, float("inf"), args.num_workers)  # one-vs-all

    # ---- load unlearned checkpoint ----
    if args.ckpt_path:
        ckpt = args.ckpt_path
    else:
        base = str(Path(args.exps_dir).expanduser())
        ckpt = checkpoint_for(args.method, args.dataset_name, args.model_name, args.forget_id, args.unlearn_rate, base)

    model = load_model(ckpt, args.model_name, num_classes).to(device).eval()

    # ---- CSV path (matches your main.py convention) ----
    out_csv = Path("results") / args.method / \
        f"{args.dataset_name}_{args.model_name}_unlearned_{args.method}_forget1_model_metrics_lr{args.unlearn_rate}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ---- write metrics CSV (overall + per-class + MIA placeholders) ----
    gather_and_write_metrics_csv(
        csv_path=str(out_csv),
        model=model,
        method=args.method,
        forget_class=args.forget_id,
        train_retain_loader=train_remain_loader,
        train_forget_loader=train_forget_loader,
        test_retain_loader=test_remain_loader,
        test_forget_loader=test_forget_loader,
        train_full_loader=train_loader,
        test_full_loader=test_loader,
        mia_result=None,  # you can plug your MIA dict here later if needed
    )

    print(f"[OK] Wrote {out_csv}")

if __name__ == "__main__":
    main()
