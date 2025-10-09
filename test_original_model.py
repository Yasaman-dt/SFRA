import os
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import pandas as pd
import copy

from utils import (
    get_transforms, get_dataset, get_dataloader, get_unlearn_loader, _top1_and_per_class
)
from trainer import *  # assumes load_model is here


def accuracy(model: nn.Module, loader, device: str):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return 100.0 * correct / max(1, total)


def build_loaders(
    dataset_name: str = "cifar10",
    model_name: str = "resnet18",
    batch_size: int = 128,
    num_workers: int = 4,
    forget_id: int | None = 1,        # one-vs-all if set
    forget_class: int = 1,            # unused when forget_id is set
    wo_dataaug: bool = False,
    cfg_path: str | None = None,
):
    # 1) Transforms & datasets
    transform_train, transform_test = get_transforms(dataset_name, model_name, wo_dataaug=wo_dataaug)
    trainset, testset = get_dataset(dataset_name, transform_train, transform_test)
    train_loader, test_loader = get_dataloader(trainset, testset, batch_size, num_workers)

    # 2) Which class to forget (one-vs-all)
    forget_class_index = [forget_id]

    # 3) Retain/forget splits & loaders
    num_forget = float("inf")
    (
        train_forget_loader,
        train_retain_loader,
        test_forget_loader,
        test_retain_loader,
        repair_class_loader,
        train_forget_index,
        train_retain_index,
        test_forget_index,
        test_retain_index,
    ) = get_unlearn_loader(
        trainset, testset, forget_class_index,
        batch_size, num_forget, num_workers
    )

    return {
        "train": train_loader,
        "test": test_loader,
        "train_forget": train_forget_loader,
        "train_retain": train_retain_loader,
        "test_forget": test_forget_loader,
        "test_retain": test_retain_loader,
        "repair_class": repair_class_loader,
        "indices": {
            "train_forget": train_forget_index,
            "train_retain": train_retain_index,
            "test_forget": test_forget_index,
            "test_retain": test_retain_index,
        },
    }


def resolve_checkpoint(dataset_name: str, model_name: str, checkpoint_folder: Path, fallback_dir: Path) -> Path:
    ckpt1 = checkpoint_folder / f"{dataset_name}_{model_name}_original_model.pth"
    ckpt2 = fallback_dir / f"{dataset_name}_{model_name}_original_model.pth"
    if ckpt1.exists():
        return ckpt1
    if ckpt2.exists():
        return ckpt2
    raise FileNotFoundError(
        f"Checkpoint not found at {ckpt1} or {ckpt2}. "
        f"Set paths correctly for dataset={dataset_name} model={model_name}."
    )


def run_for_dataset(
    dataset_name: str,
    num_classes: int,
    model_name: str,
    checkpoint_folder: Path,
    fallback_dir: Path,
    out_csv_dir: Path,
    batch_size: int,
    num_workers: int,
    wo_dataaug: bool,
    device: str,
    forget_classes: list[int] | None,
):
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    original_dir = out_csv_dir / "original"
    original_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_csv_dir / f"original/{dataset_name}_{model_name}_original_model_metrics.csv"

    # ---- Determinism ----
    seed = 0
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(seed)

    # ---- Load model once ----
    ckpt_path = resolve_checkpoint(dataset_name, model_name, checkpoint_folder, fallback_dir)
    model = load_model(str(ckpt_path), model_name, dataset_name, num_classes)
    model.to(device).eval()

    # ---- Build base datasets/loaders ONCE ----
    transform_train, transform_test = get_transforms(dataset_name, model_name, wo_dataaug=wo_dataaug)
    trainset, testset = get_dataset(dataset_name, transform_train, transform_test)

    # train loader used ONLY for unlearning splits (can keep aug)
    train_loader_base, test_loader = get_dataloader(
        trainset, testset, batch_size, num_workers
    )

    # deterministic train-eval loader (NO AUG, no shuffle)
    trainset_eval = copy.copy(trainset)
    trainset_eval.transform = transform_test
    train_eval_loader = DataLoader(
        trainset_eval, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, generator=g
    )

    # ---- Loop over forget classes ----
    classes_to_run = forget_classes if forget_classes is not None else list(range(num_classes))
    for forget_class in classes_to_run:
        # Only the splits depend on forget_class
        (
            train_forget_loader, train_retain_loader,
            test_forget_loader,  test_retain_loader,
            repair_class_loader, train_forget_index, train_retain_index,
            test_forget_index,  test_retain_index
        ) = get_unlearn_loader(
            trainset, testset, [forget_class],
            batch_size, float("inf"), num_workers
        )

        # ---- Metrics (full datasets) ----
        train_accuracy, train_per_class = _top1_and_per_class(model, train_eval_loader, num_classes, device)
        test_accuracy,  test_per_class  = _top1_and_per_class(model, test_loader,     num_classes, device)

        # ---- Metrics (splits) ----
        train_retain_acc = accuracy(model, train_retain_loader, device)
        train_fgt_acc    = accuracy(model, train_forget_loader, device)
        test_retain_acc  = accuracy(model, test_retain_loader, device)
        test_fgt_acc     = accuracy(model, test_forget_loader, device)

        AUS = 1.0 / (1.0 + (test_fgt_acc / 100.0))

        row = {
            "forget_class": forget_class, "method": "original",
            "dataset": dataset_name, "model": model_name,
            "train_acc": train_accuracy, "test_acc": test_accuracy,
            "train_retain_acc": train_retain_acc, "train_forget_acc": train_fgt_acc,
            "test_retain_acc": test_retain_acc, "test_forget_acc": test_fgt_acc,
            "AUS": AUS,
        }
        for i in range(num_classes):
            row[f"train_class{i}_acc"] = train_per_class[i]
            row[f"test_class{i}_acc"]  = test_per_class[i]

        pd.DataFrame([row]).to_csv(
            results_csv, mode="a", header=not results_csv.exists(),
            index=False, float_format="%.3f",
        )

        print(
            f"[{dataset_name} | {model_name} | forget={forget_class}] "
            f"Train {train_accuracy:.2f} | Test {test_accuracy:.2f} | "
            f"Retain Train {train_retain_acc:.2f} | Retain Test {test_retain_acc:.2f} | "
            f"Forget Train {train_fgt_acc:.2f} | Forget Test {test_fgt_acc:.2f} | AUS {AUS:.3f}"
        )
        
        
DEFAULT_BATCH_SIZES = {
    "resnet18": 128,
    "swin-t": 64,
    "vgg16": 128,
    "vit-s-16": 128,
    "vit-b-16": 256,
}

DEFAULT_NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "tiny_imagenet": 200,
}

        
def main():
    parser = argparse.ArgumentParser("Evaluate original model + one-vs-all splits, CSV per dataset/model")

    # ---- What to run ----
    parser.add_argument("--datasets", nargs="+", default=["cifar10"],
                        help="Datasets to run. Example: cifar10 cifar100 tiny_imagenet")
    parser.add_argument("--num-classes", type=int, nargs="+", default=None,
                        help="Number of classes per dataset (same order as --datasets). "
                             "If omitted, will use built-in defaults for known datasets.")
    parser.add_argument("--model-name", type=str, default="resnet18",
                        choices=["resnet18", "vit-s-16", "vit-b-16", "swin-t", "vgg16"])

    # ---- Paths ----
    parser.add_argument("--checkpoint-folder", type=Path,
                        default=Path("/export/livia/home/vision/Zdehghani/classification/exps/test_pretrained_model"))
    parser.add_argument("--fallback-dir", type=Path,
                        default=Path("/projets/Zdehghani/Machine_Unlearning_Classification"))
    parser.add_argument("--out-csv-dir", type=Path, default=Path("./results"))

    # ---- Dataloader / runtime ----
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size. If not set, uses a per-model default.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--wo-dataaug", action="store_true", help="Disable data augmentation for train transforms")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # ---- Forgetting config ----
    parser.add_argument("--forget-classes", type=int, nargs="*", default=None,
                        help="Specific classes to forget; default runs all classes for each dataset")


    args = parser.parse_args()

    # ---- Batch size (with per-model defaults) ----
    if args.batch_size is None:
        batch_size = DEFAULT_BATCH_SIZES.get(args.model_name, 128)
    else:
        batch_size = args.batch_size
    print(f"Using batch size {batch_size} for model {args.model_name}")

    # ---- Num classes (auto-fill for known datasets if not provided) ----
    if args.num_classes is None:
        # use lowercase keys to be robust to casing (cifar10 vs CIFAR10)
        num_classes_lookup = {k.lower(): v for k, v in DEFAULT_NUM_CLASSES.items()}
        num_classes_list = []
        for ds in args.datasets:
            nc = num_classes_lookup.get(ds.lower())
            if nc is None:
                raise ValueError(f"Unknown dataset '{ds}'. Please pass --num-classes explicitly.")
            num_classes_list.append(nc)
    else:
        if len(args.num_classes) != len(args.datasets):
            raise ValueError("--num-classes must match length of --datasets")
        num_classes_list = args.num_classes

    # ---- Device ----
    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cpu" if args.device == "auto" else args.device)
    )

    # ---- Run ----
    for ds, ncls in zip(args.datasets, num_classes_list):
        run_for_dataset(
            dataset_name=ds,
            num_classes=ncls,
            model_name=args.model_name,
            checkpoint_folder=args.checkpoint_folder,
            fallback_dir=args.fallback_dir,
            out_csv_dir=args.out_csv_dir,
            batch_size=batch_size,               # ✅ use resolved batch_size
            num_workers=args.num_workers,
            wo_dataaug=args.wo_dataaug,
            device=device,
            forget_classes=args.forget_classes,
        )



if __name__ == "__main__":
    main()