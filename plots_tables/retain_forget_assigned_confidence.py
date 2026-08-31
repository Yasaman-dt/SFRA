"""Compare confidence assigned to retain classes for retain vs. forget samples.

Given one single-class unlearned checkpoint, this script evaluates all real test
samples and produces two confidence tables over retain classes:

  Strategy 1 (correct retain only):
    For each retain class c, compare
      - mean p_c(x) over real retain samples with y=c and pred=c;
      - mean p_c(x) over real forget samples with pred=c.

  Strategy 2 (all retain assigned to c):
    For each retain class c, compare
      - mean p_c(x) over any real retain sample with pred=c, regardless of its
        ground-truth retain class;
      - mean p_c(x) over real forget samples with pred=c.

Here p_c(x) is the softmax probability assigned by the unlearned model to class
c.  The script also reports counts for each average.

Example:
    CUDA_VISIBLE_DEVICES=0 python plots_tables/retain_forget_assigned_confidence.py \
        --method bad_teacher \
        --dataset cifar10 \
        --model_name resnet18 \
        --lr 0.001 \
        --forget_class 7

    CUDA_VISIBLE_DEVICES=0 python plots_tables/retain_forget_assigned_confidence.py \
        --method bad_teacher \
        --dataset cifar10 \
        --model_name resnet18 \
        --lr 0.001 \
        --forget_classes 0 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainer import load_model  # noqa: E402
from utils import get_dataset, get_transforms  # noqa: E402
from project_paths import DATA_DIR, EXPS_DIR  # noqa: E402


NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "tiny_imagenet": 200,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how confidently an unlearned model assigns retain classes "
            "to real retain and real forget test samples."
        )
    )
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(NUM_CLASSES),
    )
    parser.add_argument("--model", "--model_name", dest="model_name", default="resnet18")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument(
        "--forget_class",
        "--class_id",
        dest="forget_class",
        type=int,
        default=None,
        help="Single forgotten class to analyze.",
    )
    parser.add_argument(
        "--forget_classes",
        dest="forget_classes",
        type=int,
        nargs="+",
        default=None,
        help="List of forgotten classes to analyze. Loads one checkpoint per class.",
    )
    parser.add_argument(
        "--base_dir",
        "--ckpt_dir",
        dest="base_dir",
        default=str(EXPS_DIR),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional explicit checkpoint path. Overrides the standard path convention.",
    )
    parser.add_argument(
        "--data_dir",
        default=str(DATA_DIR),
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "results_single_class/analysis/confidence_tables/<method>"
        ),
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Number of decimals in LaTeX tables.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]+", "_", str(value)).strip("_")


def checkpoint_for(
    method: str,
    dataset: str,
    model_name: str,
    forget_class: int,
    lr: float,
    base_dir: str,
) -> str:
    if method == "original":
        return os.path.join(
            base_dir,
            "test_pretrained_model",
            f"{dataset}_{model_name}_original_model.pth",
        )
    if method == "retrained":
        return os.path.join(
            base_dir,
            "test_pretrained_model",
            f"{dataset}_{model_name}_retrain_forgetcls{forget_class}_model.pth",
        )
    return os.path.join(
        base_dir,
        f"{dataset}_{model_name}_forgetcls{forget_class}",
        method,
        f"lr{lr}",
        "ckpt_best_by_aus.pth",
    )


def resolve_forget_classes(args: argparse.Namespace, num_classes: int) -> List[int]:
    classes = []
    if args.forget_class is not None:
        classes.append(args.forget_class)
    if args.forget_classes is not None:
        classes.extend(args.forget_classes)

    # Preserve order while removing duplicates.
    classes = list(dict.fromkeys(classes))
    if not classes:
        raise ValueError("Provide either --forget_class or --forget_classes.")

    invalid = [c for c in classes if not 0 <= c < num_classes]
    if invalid:
        raise ValueError(
            f"Forget classes must be in [0, {num_classes - 1}]. "
            f"Invalid values: {invalid}"
        )
    if args.checkpoint is not None and len(classes) != 1:
        raise ValueError(
            "--checkpoint can only be used with one forget class. "
            "For a list, omit --checkpoint so the script can resolve one "
            "checkpoint per class."
        )
    return classes


@torch.inference_mode()
def evaluate_test_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels, preds, confs = [], [], []
    model.eval()
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probabilities = F.softmax(logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)
        labels.append(target.long().cpu())
        preds.append(prediction.long().cpu())
        confs.append(confidence.float().cpu())
    return torch.cat(labels), torch.cat(preds), torch.cat(confs)


def mean_or_nan(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(values.float().mean().item())


def build_tables(
    labels: torch.Tensor,
    preds: torch.Tensor,
    pred_confs: torch.Tensor,
    num_classes: int,
    forget_class: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    retain_classes = [c for c in range(num_classes) if c != forget_class]
    rows_correct = []
    rows_assigned = []

    is_forget = labels.eq(forget_class)
    is_retain = ~is_forget

    for class_id in retain_classes:
        pred_is_class = preds.eq(class_id)

        correct_retain_mask = labels.eq(class_id) & pred_is_class
        assigned_retain_mask = is_retain & pred_is_class
        assigned_forget_mask = is_forget & pred_is_class

        correct_retain_conf = pred_confs[correct_retain_mask]
        assigned_retain_conf = pred_confs[assigned_retain_mask]
        assigned_forget_conf = pred_confs[assigned_forget_mask]

        rows_correct.append(
            {
                "retain_class": class_id,
                "retain_correct_count": int(correct_retain_mask.sum().item()),
                "retain_correct_confidence": mean_or_nan(correct_retain_conf),
                "forget_assigned_count": int(assigned_forget_mask.sum().item()),
                "forget_assigned_confidence": mean_or_nan(assigned_forget_conf),
                "confidence_gap": (
                    mean_or_nan(correct_retain_conf)
                    - mean_or_nan(assigned_forget_conf)
                ),
            }
        )

        rows_assigned.append(
            {
                "retain_class": class_id,
                "retain_assigned_count": int(assigned_retain_mask.sum().item()),
                "retain_assigned_confidence": mean_or_nan(assigned_retain_conf),
                "forget_assigned_count": int(assigned_forget_mask.sum().item()),
                "forget_assigned_confidence": mean_or_nan(assigned_forget_conf),
                "confidence_gap": (
                    mean_or_nan(assigned_retain_conf)
                    - mean_or_nan(assigned_forget_conf)
                ),
            }
        )

    return pd.DataFrame(rows_correct), pd.DataFrame(rows_assigned)


def fmt_float(value: float, precision: int) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{precision}f}"


def dataframe_to_latex(
    df: pd.DataFrame,
    title: str,
    label: str,
    confidence_column: str,
    count_column: str,
    precision: int,
) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{title}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{c|cc|cc|c}")
    lines.append(r"\toprule")
    lines.append(
        r"Retain Class & Retain Count & Retain Conf. & "
        r"Forget Count & Forget Conf. & Gap \\"
    )
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        cells = [
            str(int(row["retain_class"])),
            str(int(row[count_column])),
            fmt_float(row[confidence_column], precision),
            str(int(row["forget_assigned_count"])),
            fmt_float(row["forget_assigned_confidence"], precision),
            fmt_float(row["confidence_gap"], precision),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_outputs(
    correct_df: pd.DataFrame,
    assigned_df: pd.DataFrame,
    out_dir: Path,
    stem: str,
    dataset: str,
    model_name: str,
    method: str,
    forget_class: int,
    precision: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    correct_csv = out_dir / f"{stem}_correct_retain_vs_forget_assigned.csv"
    assigned_csv = out_dir / f"{stem}_all_retain_assigned_vs_forget_assigned.csv"
    correct_tex = correct_csv.with_suffix(".tex")
    assigned_tex = assigned_csv.with_suffix(".tex")

    correct_df.to_csv(correct_csv, index=False)
    assigned_df.to_csv(assigned_csv, index=False)

    common = f"{dataset}/{model_name}, {method}, forgotten class {forget_class}"
    correct_caption = (
        "Confidence comparison for correctly classified retain samples and "
        f"forget samples assigned to each retain class ({common})."
    )
    assigned_caption = (
        "Confidence comparison for all retain samples assigned to each retain "
        f"class and forget samples assigned to each retain class ({common})."
    )

    correct_tex.write_text(
        dataframe_to_latex(
            correct_df,
            correct_caption,
            f"tab:{slugify(stem)}_correct_retain_confidence",
            confidence_column="retain_correct_confidence",
            count_column="retain_correct_count",
            precision=precision,
        ),
        encoding="utf-8",
    )
    assigned_tex.write_text(
        dataframe_to_latex(
            assigned_df,
            assigned_caption,
            f"tab:{slugify(stem)}_assigned_retain_confidence",
            confidence_column="retain_assigned_confidence",
            count_column="retain_assigned_count",
            precision=precision,
        ),
        encoding="utf-8",
    )

    print(f"[OK] wrote {correct_csv}")
    print(f"[OK] wrote {correct_tex}")
    print(f"[OK] wrote {assigned_csv}")
    print(f"[OK] wrote {assigned_tex}")


def main() -> None:
    args = parse_args()
    num_classes = NUM_CLASSES[args.dataset]
    forget_classes = resolve_forget_classes(args, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, transform_test = get_transforms(
        args.dataset,
        args.model_name,
        wo_dataaug=False,
    )
    _, test_dataset = get_dataset(
        args.dataset,
        transform_test,
        transform_test,
        path=Path(args.data_dir).expanduser(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    out_dir = (
        args.out_dir
        or Path("results_single_class") / "analysis" / "confidence_tables" / args.method
    )

    for forget_class in forget_classes:
        checkpoint = args.checkpoint or checkpoint_for(
            args.method,
            args.dataset,
            args.model_name,
            forget_class,
            args.lr,
            args.base_dir,
        )
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"Checkpoint not found for forget_class={forget_class}:\n"
                f"  {checkpoint}\n"
                "Pass --checkpoint if this checkpoint uses a different path."
            )

        print(f"[model] loading forget_class={forget_class}: {checkpoint}")
        model = load_model(
            checkpoint,
            args.model_name,
            args.dataset,
            num_classes,
        ).to(device).eval()

        labels, preds, pred_confs = evaluate_test_predictions(
            model,
            test_loader,
            device,
        )
        print(
            f"[eval] forget_class={forget_class}: evaluated {len(labels)} test samples"
        )

        correct_df, assigned_df = build_tables(
            labels,
            preds,
            pred_confs,
            num_classes,
            forget_class,
        )

        stem = (
            f"{args.dataset}_{args.model_name}_{args.method}_lr{args.lr:g}"
            f"_fg{forget_class}"
        )
        write_outputs(
            correct_df,
            assigned_df,
            out_dir,
            stem,
            args.dataset,
            args.model_name,
            args.method,
            forget_class,
            args.precision,
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
