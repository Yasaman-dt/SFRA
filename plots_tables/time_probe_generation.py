"""Benchmark Gaussian probe-generation cost for the source-free audit.

This script times the inner loop used by the audit: sample Gaussian features,
apply the linear classifier head, and keep samples whose predicted class matches
the target class. It does not run the image encoder, so it measures the
feature-space rejection-sampling cost discussed in the paper.

By default, it measures a few target classes per dataset and extrapolates to the
full number of classes. Running all 200 TinyImageNet classes with N=500K can be
very slow; use --benchmark_classes 0 only when you want the exact wall-clock.

Example:
  CUDA_VISIBLE_DEVICES=0 python plots_tables/time_probe_generation.py \
    --model_name resnet18 \
    --datasets cifar10 cifar100 tiny_imagenet \
    --accepted_per_class 500000 \
    --benchmark_classes 5 \
    --batch_size 65536 \
    --device cuda
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "tiny_imagenet": 200,
}

PAPER_SINGLE_CLASS_SETTINGS = {
    "cifar10": {"accepted_per_class": 500_000, "selected_per_class": 500},
    "cifar100": {"accepted_per_class": 100_000, "selected_per_class": 100},
    "tiny_imagenet": {"accepted_per_class": 50_000, "selected_per_class": 25},
}

DEFAULT_FEATURE_DIMS = {
    "resnet18": 512,
    "vgg16": 512,
    "vit-b-16": 768,
    "swin-t": 768,
}

DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "tiny_imagenet": "TinyImageNet",
}

MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "vgg16": "VGG-16",
    "vit-b-16": "ViT-B/16",
    "swin-t": "Swin-T",
}

DEFAULT_METHOD_LRS = {
    "random_label": 1e-7,
    "finetune": 2e-2,
    "gradient_ascent": 5e-5,
    "neggrad_plus": 5e-1,
    "boundary_shrink": 1e-8,
    "l2ul_adv": 1e-5,
    "scrub": 1e-3,
    "bad_teacher": 1e-3,
    "salun": 1e-3,
    "delete": 1e-3,
    "retrained": 0.0,
    "original": 0.0,
}

DEFAULT_METHOD_LRS_BY_SETTING = {
    ("resnet18", "cifar10"): {
        "random_label": 1e-7,
        "finetune": 2e-2,
        "gradient_ascent": 5e-5,
        "neggrad_plus": 5e-1,
        "l2ul_adv": 1e-5,
        "scrub": 1e-3,
        "bad_teacher": 1e-3,
        "salun": 1e-3,
        "delete": 1e-3,
    },
    ("resnet18", "cifar100"): {
        "random_label": 1e-7,
        "finetune": 2e-2,
        "gradient_ascent": 1e-2,
        "neggrad_plus": 5e-1,
        "l2ul_adv": 1e-5,
        "scrub": 1e-4,
        "bad_teacher": 1e-3,
        "salun": 1e-1,
        "delete": 1e-3,
    },
    ("resnet18", "tiny_imagenet"): {
        "random_label": 1e-4,
        "finetune": 2e-2,
        "gradient_ascent": 5e-5,
        "neggrad_plus": 5e-1,
        "boundary_shrink": 1e-4,
        "l2ul_adv": 1e-5,
        "scrub": 1e-3,
        "bad_teacher": 2e-3,
        "salun": 1e-3,
        "delete": 1e-3,
    },
    ("swin-t", "cifar10"): {
        "random_label": 1e-6,
        "finetune": 2e-2,
        "gradient_ascent": 1e-6,
        "neggrad_plus": 1e-1,
        "boundary_shrink": 1e-7,
        "l2ul_adv": 1e-6,
        "scrub": 1e-4,
        "bad_teacher": 1e-2,
        "salun": 1e-4,
        "delete": 1e-3,
    },
    ("swin-t", "cifar100"): {
        "random_label": 1e-3,
        "finetune": 2e-2,
        "gradient_ascent": 1e-3,
        "neggrad_plus": 1e-1,
        "boundary_shrink": 1e-3,
        "l2ul_adv": 1e-5,
        "scrub": 1e-4,
        "bad_teacher": 1e-3,
        "salun": 1e-2,
        "delete": 1e-3,
    },
    ("swin-t", "tiny_imagenet"): {
        "random_label": 1e-2,
        "finetune": 5e-2,
        "gradient_ascent": 1e-2,
        "neggrad_plus": 1e-1,
        "l2ul_adv": 1e-3,
        "scrub": 1e-3,
        "bad_teacher": 1e-2,
        "salun": 1e-3,
        "delete": 1e-3,
    },
    ("vit-b-16", "cifar10"): {
        "random_label": 1e-3,
        "finetune": 2e-2,
        "gradient_ascent": 1e-5,
        "neggrad_plus": 5e-1,
        "l2ul_adv": 1e-5,
        "scrub": 8e-4,
        "bad_teacher": 1e-3,
        "salun": 1e-3,
        "delete": 1e-3,
    },
    ("vit-b-16", "cifar100"): {
        "random_label": 1e-2,
        "finetune": 2e-2,
        "gradient_ascent": 1e-4,
        "neggrad_plus": 5e-1,
        "boundary_shrink": 1e-3,
        "l2ul_adv": 1e-5,
        "scrub": 5e-4,
        "bad_teacher": 1e-2,
        "salun": 1e-3,
        "delete": 1e-3,
    },
    ("vit-b-16", "tiny_imagenet"): {
        "random_label": 1e-2,
        "finetune": 5e-2,
        "gradient_ascent": 1e-3,
        "neggrad_plus": 1e-1,
        "boundary_shrink": 1e-3,
        "l2ul_adv": 1e-5,
        "scrub": 1e-3,
        "bad_teacher": 1e-2,
        "salun": 1e-3,
        "delete": 1e-3,
    },
}

DEFAULT_METHODS = [
    "random_label",
    "finetune",
    "gradient_ascent",
    "neggrad_plus",
    "boundary_shrink",
    "l2ul_adv",
    "scrub",
    "bad_teacher",
    "salun",
    "delete",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time feature-space Gaussian rejection sampling."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cifar10", "cifar100", "tiny_imagenet"],
        choices=sorted(NUM_CLASSES),
    )
    parser.add_argument("--model_name", "--model", dest="model_name", default="resnet18")
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=None,
        help="Classifier-input feature dimension. Defaults from --model_name.",
    )
    parser.add_argument(
        "--accepted_per_class",
        "--N",
        type=int,
        default=500000,
        help="Number of accepted Gaussian probes per target class.",
    )
    parser.add_argument(
        "--paper_single_class_settings",
        action="store_true",
        help=(
            "Use the paper's single-class settings per dataset: CIFAR-10 "
            "N=500K,M=500; CIFAR-100 N=100K,M=50; TinyImageNet N=50K,M=25. "
            "These values override --accepted_per_class, --retain_top_k, and "
            "--per_retain_for_forget inside each dataset loop."
        ),
    )
    parser.add_argument(
        "--benchmark_classes",
        type=int,
        default=5,
        help=(
            "Number of target classes to time per dataset. Use 0 to time all "
            "classes exactly."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--warmup_batches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--real_checkpoints",
        action="store_true",
        help=(
            "Load real single-class unlearning checkpoints and time their "
            "actual classifier heads instead of a synthetic balanced head."
        ),
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument(
        "--forget_class",
        type=int,
        default=None,
        help="Forgotten class/checkpoint to use in --real_checkpoints mode.",
    )
    parser.add_argument(
        "--target_classes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional target output classes to benchmark. By default, evenly "
            "spaced retain classes are selected."
        ),
    )
    parser.add_argument(
        "--full_probe_construction",
        action="store_true",
        help=(
            "Time the original two-pass probe construction for each retain "
            "class: one accepted pool sorted high-confidence for retain probes "
            "and another accepted pool sorted low-confidence for forget probes. "
            "This is slower but matches source_free_relearning_singleclass.py "
            "more closely than timing acceptance only."
        ),
    )
    parser.add_argument(
        "--retain_top_k",
        type=int,
        default=500,
        help="Top-confidence retain probes kept per retain class in full mode.",
    )
    parser.add_argument(
        "--per_retain_for_forget",
        type=int,
        default=500,
        help="Low-confidence forget probes kept per retain class in full mode.",
    )
    parser.add_argument(
        "--base_dir",
        "--ckpt_dir",
        dest="base_dir",
        default="/export/livia/home/vision/Zdehghani/classification/exps",
    )
    parser.add_argument(
        "--method_lr",
        action="append",
        default=[],
        metavar="METHOD=LR",
        help="Override checkpoint LR; may be repeated, e.g. --method_lr salun=0.01.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tables/probe_generation_timing"),
        help="Output prefix. Writes .csv and .tex.",
    )
    return parser.parse_args()


def method_lrs(overrides: list[str]) -> dict[str, float]:
    values = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Expected METHOD=LR, got {item!r}")
        method, value = item.split("=", 1)
        values[method] = float(value)
    return values


def lookup_method_lr(
    method: str,
    dataset: str,
    model_name: str,
    overrides: dict[str, float],
) -> float | None:
    if method in overrides:
        return overrides[method]
    setting_lrs = DEFAULT_METHOD_LRS_BY_SETTING.get((model_name, dataset), {})
    if method in setting_lrs:
        return setting_lrs[method]
    return DEFAULT_METHOD_LRS.get(method)


def should_skip_method_for_model(method: str, model_name: str) -> bool:
    return method == "boundary_shrink" and model_name in {"swin-t", "vit-b-16"}


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
        f"lr{lr:g}",
        "ckpt_best_by_aus.pth",
    )


def load_checkpoint_model(
    checkpoint: str,
    model_name: str,
    dataset: str,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    from models import get_model, load_model

    if device.type == "cuda":
        return load_model(checkpoint, model_name, dataset, num_classes).to(device).eval()

    checkpoint_object = torch.load(checkpoint, map_location="cpu")
    if not isinstance(checkpoint_object, dict):
        return checkpoint_object.to(device).eval()
    model = get_model(model_name, dataset, num_classes, use_pretrained=False)
    state_dict = {
        key.replace("module.", ""): value for key, value in checkpoint_object.items()
    }
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def final_linear(model: nn.Module, num_classes: int) -> nn.Linear:
    for _, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear) and module.out_features == num_classes:
            return module
    raise RuntimeError(f"Could not find final Linear layer with out_features={num_classes}.")


def head_tensors_from_checkpoint(
    checkpoint: str,
    model_name: str,
    dataset: str,
    num_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    checkpoint_object = torch.load(checkpoint, map_location="cpu")

    if isinstance(checkpoint_object, nn.Module):
        head = final_linear(checkpoint_object, num_classes)
        weight = head.weight.detach().float().to(device)
        if head.bias is None:
            bias = torch.zeros(num_classes, device=device)
        else:
            bias = head.bias.detach().float().to(device)
        feature_dim = int(weight.shape[1])
        del checkpoint_object
    elif isinstance(checkpoint_object, dict):
        state_dict = {
            key.replace("module.", ""): value
            for key, value in checkpoint_object.items()
            if torch.is_tensor(value)
        }
        weight_items = [
            (key, value)
            for key, value in state_dict.items()
            if value.ndim == 2 and int(value.shape[0]) == num_classes
        ]
        if not weight_items:
            raise RuntimeError(
                f"Could not find classifier weight with out_features={num_classes} "
                f"in {checkpoint}."
            )
        # The final classifier weight is normally the last matching 2-D tensor
        # in insertion order, e.g. fc.weight/head.weight/classifier.weight.
        weight_key, weight_cpu = weight_items[-1]
        prefix = weight_key.rsplit(".", 1)[0]
        bias_cpu = state_dict.get(f"{prefix}.bias")
        weight = weight_cpu.detach().float().to(device)
        if bias_cpu is None:
            bias = torch.zeros(num_classes, device=device)
        else:
            bias = bias_cpu.detach().float().to(device)
        feature_dim = int(weight.shape[1])
        del checkpoint_object, state_dict
    else:
        raise TypeError(f"Unsupported checkpoint object type: {type(checkpoint_object)}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return weight, bias, feature_dim


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_balanced_head(num_classes: int, feature_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a synthetic linear head with roughly balanced class regions."""
    weight = torch.randn(num_classes, feature_dim, device=device)
    weight = torch.nn.functional.normalize(weight, dim=1)
    bias = torch.zeros(num_classes, device=device)
    return weight, bias


@torch.inference_mode()
def warmup(weight: torch.Tensor, bias: torch.Tensor, batch_size: int, batches: int) -> None:
    if batches <= 0:
        return
    device = weight.device
    feature_dim = weight.shape[1]
    for _ in range(batches):
        features = torch.randn(batch_size, feature_dim, device=device)
        _ = (features @ weight.T + bias).argmax(dim=1)
    sync_if_needed(device)


@torch.inference_mode()
def time_one_class(
    weight: torch.Tensor,
    bias: torch.Tensor,
    target_class: int,
    accepted_per_class: int,
    batch_size: int,
) -> dict[str, float | int]:
    device = weight.device
    feature_dim = weight.shape[1]
    accepted = 0
    draws = 0
    sync_if_needed(device)
    start = time.perf_counter()
    while accepted < accepted_per_class:
        features = torch.randn(batch_size, feature_dim, device=device)
        logits = features @ weight.T + bias
        preds = logits.argmax(dim=1)
        accepted += int(preds.eq(target_class).sum().item())
        draws += batch_size
    sync_if_needed(device)
    elapsed = time.perf_counter() - start
    return {
        "target_class": target_class,
        "accepted": accepted,
        "draws": draws,
        "acceptance_rate": accepted / draws if draws else float("nan"),
        "seconds": elapsed,
        "draws_per_second": draws / elapsed if elapsed > 0 else float("nan"),
        "accepted_per_second": accepted / elapsed if elapsed > 0 else float("nan"),
    }


@torch.inference_mode()
def sample_pool_like_original(
    weight: torch.Tensor,
    bias: torch.Tensor,
    target_class: int,
    accepted_per_class: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Match _sample_predicted_as_class from source_free_relearning_singleclass.py."""
    device = weight.device
    feature_dim = weight.shape[1]
    feat_chunks = []
    prob_chunks = []
    accepted = 0
    draws = 0

    while accepted < accepted_per_class:
        features = torch.randn(batch_size, feature_dim, device=device)
        logits = features @ weight.T + bias
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        preds = probabilities.argmax(dim=1)
        mask = preds.eq(target_class)
        if mask.any():
            feat_chunks.append(features[mask])
            prob_chunks.append(probabilities[mask, target_class])
            accepted += int(mask.sum().item())
        draws += batch_size

    feats = torch.cat(feat_chunks, dim=0)
    probs = torch.cat(prob_chunks, dim=0)
    if feats.shape[0] > accepted_per_class:
        feats = feats[:accepted_per_class]
        probs = probs[:accepted_per_class]
    return feats, probs, draws


@torch.inference_mode()
def time_full_probe_construction_one_class(
    weight: torch.Tensor,
    bias: torch.Tensor,
    target_class: int,
    accepted_per_class: int,
    retain_top_k: int,
    per_retain_for_forget: int,
    batch_size: int,
) -> dict[str, float | int]:
    """Time the original retain + forget synthetic probe construction for one retain class.

    The original code samples a fresh accepted pool for retain probes, sorts by
    descending confidence, keeps retain_top_k; then samples another fresh pool
    for forget probes, sorts by ascending confidence, and keeps
    per_retain_for_forget. This function mirrors that two-pass behavior.
    """
    device = weight.device
    sync_if_needed(device)
    start = time.perf_counter()

    retain_feats, retain_probs, retain_draws = sample_pool_like_original(
        weight, bias, target_class, accepted_per_class, batch_size
    )
    _, retain_sorted = torch.sort(retain_probs, descending=True)
    _retain_selected = retain_feats.index_select(
        0, retain_sorted[: min(retain_top_k, retain_sorted.numel())]
    )

    forget_feats, forget_probs, forget_draws = sample_pool_like_original(
        weight, bias, target_class, accepted_per_class, batch_size
    )
    _, forget_sorted = torch.sort(forget_probs, descending=False)
    _forget_selected = forget_feats.index_select(
        0, forget_sorted[: min(per_retain_for_forget, forget_sorted.numel())]
    )

    sync_if_needed(device)
    elapsed = time.perf_counter() - start
    accepted_total = int(retain_feats.shape[0] + forget_feats.shape[0])
    draws_total = int(retain_draws + forget_draws)
    return {
        "target_class": target_class,
        "accepted": accepted_total,
        "draws": draws_total,
        "acceptance_rate": accepted_total / draws_total if draws_total else float("nan"),
        "seconds": elapsed,
        "draws_per_second": draws_total / elapsed if elapsed > 0 else float("nan"),
        "accepted_per_second": accepted_total / elapsed if elapsed > 0 else float("nan"),
    }


def choose_targets(
    num_classes: int,
    benchmark_classes: int,
    exclude_class: int | None = None,
    explicit_targets: list[int] | None = None,
) -> list[int]:
    candidates = [c for c in range(num_classes) if c != exclude_class]
    if explicit_targets is not None:
        invalid = [c for c in explicit_targets if c not in candidates]
        if invalid:
            raise ValueError(f"Invalid target classes for this setting: {invalid}")
        return list(dict.fromkeys(explicit_targets))
    if benchmark_classes <= 0 or benchmark_classes >= len(candidates):
        return candidates
    if benchmark_classes == 1:
        return [candidates[0]]
    indices = np.linspace(0, len(candidates) - 1, benchmark_classes, dtype=int).tolist()
    return [candidates[i] for i in sorted(set(indices))]


def fmt_seconds(seconds: float) -> str:
    if math.isnan(seconds):
        return "--"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def dataset_probe_settings(args: argparse.Namespace, dataset: str) -> tuple[int, int, int]:
    if not args.paper_single_class_settings:
        return args.accepted_per_class, args.retain_top_k, args.per_retain_for_forget
    setting = PAPER_SINGLE_CLASS_SETTINGS[dataset]
    selected = int(setting["selected_per_class"])
    return int(setting["accepted_per_class"]), selected, selected


def write_latex(
    rows: list[dict],
    path: Path,
    model_name: str,
    accepted_per_class: int,
    real_checkpoints: bool,
) -> None:
    model_label = MODEL_LABELS.get(model_name, model_name)
    full_mode = bool(rows and rows[0].get("full_probe_construction", False))
    timed_operation = (
        "full retain/forget probe construction"
        if full_mode
        else "accepted-pool rejection sampling"
    )
    unique_nm = {
        (
            int(row.get("accepted_per_class", accepted_per_class)),
            row.get("per_retain_for_forget", ""),
        )
        for row in rows
    }
    if len(unique_nm) == 1:
        only_n, _ = next(iter(unique_nm))
        n_text = rf"Each target class keeps \(N={only_n:,}\) accepted probes. "
    else:
        n_text = r"The dataset-specific \(N\) and \(M\) settings are reported in the table. "
    group_columns = ["Dataset"]
    if real_checkpoints:
        group_columns.append("Method")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (
            r"\caption{Probe-generation runtime for feature-space Gaussian "
            rf"{timed_operation} using {'real single-class unlearning checkpoints' if real_checkpoints else f'a {model_label} classifier-head shape'}. "
            + n_text
            +
            r"The measured time averages the benchmarked target classes; the full "
            r"time is extrapolated from the per-class average.}"
        ),
        r"\label{tab:probe_generation_timing}",
        rf"\begin{{tabular}}{{{'l' * len(group_columns)}rrrrrr}}",
        r"\toprule",
        " & ".join(group_columns)
        + r" & \(N\) & \(M\) & Classes & Benchmarked & Time / class & Extrapolated full time \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [DATASET_LABELS.get(row["dataset"], row["dataset"])]
        if real_checkpoints:
            cells.append(str(row["method"]).replace("_", " ").title())
        m_value = row.get("per_retain_for_forget", "")
        if m_value == "":
            m_value = row.get("retain_top_k", "")
        cells.extend([
            f"{int(row['accepted_per_class']):,}",
            str(m_value) if m_value != "" else "--",
            str(row["num_classes"]),
            str(row["benchmarked_classes"]),
            fmt_seconds(row["mean_seconds_per_class"]),
            fmt_seconds(row["extrapolated_full_seconds"]),
        ])
        lines.append(
            " & ".join(cells) + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    feature_dim = args.feature_dim or DEFAULT_FEATURE_DIMS.get(args.model_name)
    if feature_dim is None and not args.real_checkpoints:
        raise ValueError("Unknown model feature dimension; pass --feature_dim.")
    if args.real_checkpoints and args.forget_class is None:
        raise ValueError("--real_checkpoints requires --forget_class.")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    rows = []
    detail_rows = []
    lr_overrides = method_lrs(args.method_lr)
    for dataset in args.datasets:
        num_classes = NUM_CLASSES[dataset]
        if args.real_checkpoints and not (0 <= args.forget_class < num_classes):
            print(
                f"[skip] {dataset}: forget_class={args.forget_class} is outside "
                f"0..{num_classes - 1}"
            )
            continue

        method_items = args.methods if args.real_checkpoints else ["synthetic_head"]
        accepted_per_class, retain_top_k, per_retain_for_forget = dataset_probe_settings(
            args, dataset
        )
        for method in method_items:
            if should_skip_method_for_model(method, args.model_name):
                print(f"[skip] {args.model_name}/{dataset}/{method}: not included for timing table")
                continue
            if args.real_checkpoints:
                lr = lookup_method_lr(method, dataset, args.model_name, lr_overrides)
                if lr is None:
                    raise ValueError(
                        f"Missing LR for {args.model_name}/{dataset}/{method}; "
                        f"pass --method_lr {method}=LR."
                    )
                checkpoint = checkpoint_for(
                    method,
                    dataset,
                    args.model_name,
                    args.forget_class,
                    lr,
                    args.base_dir,
                )
                if not os.path.isfile(checkpoint):
                    print(f"[skip] checkpoint not found: {checkpoint}")
                    continue
                print(f"[model] loading {checkpoint}")
                weight, bias, actual_feature_dim = head_tensors_from_checkpoint(
                    checkpoint, args.model_name, dataset, num_classes, device
                )
            else:
                actual_feature_dim = int(feature_dim)
                weight, bias = make_balanced_head(num_classes, actual_feature_dim, device)
                checkpoint = ""

            targets = choose_targets(
                num_classes,
                args.benchmark_classes,
                exclude_class=args.forget_class if args.real_checkpoints else None,
                explicit_targets=args.target_classes,
            )
            full_target_count = num_classes - (1 if args.real_checkpoints else 0)
            warmup(weight, bias, args.batch_size, args.warmup_batches)

            print(
                f"[timing] {dataset}/{method}: C={num_classes}, dim={actual_feature_dim}, "
                f"N={accepted_per_class}, M={per_retain_for_forget}, "
                f"benchmark_targets={targets}"
            )
            target_results = []
            for target in targets:
                if args.full_probe_construction:
                    result = time_full_probe_construction_one_class(
                        weight,
                        bias,
                        target,
                        accepted_per_class,
                        retain_top_k,
                        per_retain_for_forget,
                        args.batch_size,
                    )
                else:
                    result = time_one_class(
                        weight,
                        bias,
                        target,
                        accepted_per_class,
                        args.batch_size,
                    )
                result.update({
                    "dataset": dataset,
                    "method": method,
                    "model_name": args.model_name,
                    "forget_class": args.forget_class if args.real_checkpoints else "",
                    "checkpoint": checkpoint,
                    "num_classes": num_classes,
                    "feature_dim": actual_feature_dim,
                    "accepted_per_class_requested": accepted_per_class,
                    "full_probe_construction": args.full_probe_construction,
                    "retain_top_k": retain_top_k if args.full_probe_construction else "",
                    "per_retain_for_forget": per_retain_for_forget if args.full_probe_construction else "",
                    "paper_single_class_settings": args.paper_single_class_settings,
                    "batch_size": args.batch_size,
                    "device": str(device),
                })
                target_results.append(result)
                detail_rows.append(result)
                print(
                    f"  class {target}: {result['seconds']:.2f}s, "
                    f"acceptance={100 * result['acceptance_rate']:.2f}%"
                )

            if not target_results:
                continue
            mean_seconds = float(np.mean([r["seconds"] for r in target_results]))
            std_seconds = float(np.std([r["seconds"] for r in target_results], ddof=1)) if len(target_results) > 1 else 0.0
            mean_acceptance = float(np.mean([r["acceptance_rate"] for r in target_results]))
            rows.append({
                "dataset": dataset,
                "method": method,
                "model_name": args.model_name,
                "forget_class": args.forget_class if args.real_checkpoints else "",
                "num_classes": num_classes,
                "full_target_classes": full_target_count,
                "feature_dim": actual_feature_dim,
                "accepted_per_class": accepted_per_class,
                "full_probe_construction": args.full_probe_construction,
                "retain_top_k": retain_top_k if args.full_probe_construction else "",
                "per_retain_for_forget": per_retain_for_forget if args.full_probe_construction else "",
                "paper_single_class_settings": args.paper_single_class_settings,
                "benchmarked_classes": len(targets),
                "mean_seconds_per_class": mean_seconds,
                "std_seconds_per_class": std_seconds,
                "mean_acceptance_rate": mean_acceptance,
                "extrapolated_full_seconds": mean_seconds * full_target_count,
                "device": str(device),
                "batch_size": args.batch_size,
                "checkpoint": checkpoint,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    detail_path = Path(str(args.out) + "_per_class.csv")
    tex_path = args.out.with_suffix(".tex")

    if not rows:
        raise RuntimeError("No timing rows were produced. Check checkpoint paths and arguments.")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)
    write_latex(rows, tex_path, args.model_name, args.accepted_per_class, args.real_checkpoints)

    print(f"[saved] {csv_path.resolve()}")
    print(f"[saved] {detail_path.resolve()}")
    print(f"[saved] {tex_path.resolve()}")


if __name__ == "__main__":
    main()
