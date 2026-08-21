"""Joint t-SNE of real pre-FC features and selected Gaussian probes.

The script loads one class-unlearned checkpoint, extracts test-set features
immediately before the final classifier, reproduces the source-free Gaussian
selection procedure, and creates a two-panel appendix figure:

  (a) joint t-SNE of real features and selected synthetic probes;
  (b) joint t-SNE of the corresponding classifier-head outputs (logits).

Example:
    python plots_tables/tsne_real_gaussian_probes.py \
        --method bad_teacher --dataset cifar10 --model_name resnet18 \
        --lr 1e-3 --forget_class 7
"""

import argparse
import copy
import inspect
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import get_model  # noqa: E402
from trainer import load_model  # noqa: E402
from utils import get_dataset, get_transforms  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})


NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "tiny_imagenet": 200,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot real pre-FC features together with selected Gaussian probes."
    )
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(NUM_CLASSES),
    )
    parser.add_argument("--model", "--model_name", dest="model_name", default="resnet18")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--forget_class", "--class_id", dest="forget_class", type=int, required=True)
    parser.add_argument(
        "--ckpt_dir",
        "--base_dir",
        dest="ckpt_dir",
        default="/export/livia/home/vision/Zdehghani/classification/exps",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional explicit checkpoint path; overrides the standard path convention.",
    )
    parser.add_argument(
        "--data_dir",
        default=str(Path("~/data").expanduser()),
    )
    parser.add_argument(
        "--generated_per_class",
        "--tpr",
        dest="generated_per_class",
        type=int,
        default=5000,
        help="Accepted Gaussian candidates per retain class before selection.",
    )
    parser.add_argument(
        "--retain_per_class",
        type=int,
        default=25,
        help="Highest-confidence Gaussian probes displayed per retain class.",
    )
    parser.add_argument(
        "--forget_per_class",
        type=int,
        default=25,
        help="Lowest-confidence boundary probes selected from each retain class.",
    )
    parser.add_argument(
        "--real_per_class",
        type=int,
        default=200,
        help="Maximum real test features displayed per class; 0 uses all.",
    )
    parser.add_argument(
        "--real_only",
        action="store_true",
        help=(
            "Plot only matched real retain/forget samples. Skip synthetic-probe "
            "generation and omit synthetic entries from the legend."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_batch_size", type=int, default=65536)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iterations", type=int, default=1500)
    parser.add_argument(
        "--tsne_preprocess",
        choices=["l2", "zscore", "none"],
        default="l2",
        help=(
            "Feature preprocessing before the joint t-SNE. L2 normalization is "
            "the default because the paper's alignment argument is directional."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG/PDF path. A descriptive PNG name is used by default.",
    )
    return parser.parse_args()


def checkpoint_for(method, dataset, model_name, forget_class, lr, base_dir):
    if method == "original":
        return os.path.join(
            base_dir, "test_pretrained_model", f"{dataset}_{model_name}_original_model.pth"
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


def load_checkpoint_model(checkpoint, model_name, dataset, num_classes, device):
    """Use the project loader on GPU, with a CPU fallback for diagnostics."""
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


def get_final_linear(model, num_classes):
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear) and module.out_features == num_classes:
            return name, module
    raise RuntimeError(
        f"Could not find the final Linear layer with out_features={num_classes}."
    )


def set_module(root, dotted_name, replacement):
    parent = root
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def make_feature_extractor(model, num_classes, device):
    feature_model = copy.deepcopy(model)
    final_name, _ = get_final_linear(feature_model, num_classes)
    set_module(feature_model, final_name, nn.Identity())
    return feature_model.eval().to(device)


@torch.inference_mode()
def extract_features(feature_model, loader, device):
    features, labels = [], []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        output = feature_model(images)
        if output.ndim > 2:
            output = torch.flatten(output, 1)
        features.append(output.detach().float().cpu())
        labels.append(target.long().cpu())
    return torch.cat(features), torch.cat(labels)


@torch.inference_mode()
def sample_candidates_for_class(
    weight,
    bias,
    embedding_dim,
    target_class,
    wanted,
    batch_size,
    generator,
):
    """Rejection-sample N(0,I) candidates predicted as target_class."""
    device = weight.device
    feature_chunks, confidence_chunks = [], []
    accepted = 0
    draws = 0
    max_draws = max(5_000_000, wanted * weight.shape[0] * 200)

    while accepted < wanted:
        current_batch = min(batch_size, max_draws - draws)
        if current_batch <= 0:
            raise RuntimeError(
                f"Could only sample {accepted}/{wanted} candidates predicted as class "
                f"{target_class}. Increase --sample_batch_size or reduce "
                "--generated_per_class."
            )
        samples = torch.randn(
            current_batch,
            embedding_dim,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        probabilities = F.softmax(samples @ weight.T + bias, dim=1)
        mask = probabilities.argmax(dim=1).eq(target_class)
        if mask.any():
            feature_chunks.append(samples[mask].cpu())
            confidence_chunks.append(probabilities[mask, target_class].cpu())
            accepted += int(mask.sum())
        draws += current_batch

    features = torch.cat(feature_chunks)[:wanted]
    confidences = torch.cat(confidence_chunks)[:wanted]
    return features, confidences


def build_selected_probes(
    model,
    num_classes,
    forget_class,
    generated_per_class,
    retain_per_class,
    forget_per_class,
    sample_batch_size,
    device,
    seed,
):
    _, classifier = get_final_linear(model, num_classes)
    weight = classifier.weight.detach().float().to(device)
    if classifier.bias is None:
        bias = torch.zeros(num_classes, device=device)
    else:
        bias = classifier.bias.detach().float().to(device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    retain_features, retain_labels = [], []
    boundary_features, boundary_origins = [], []
    retain_classes = [c for c in range(num_classes) if c != forget_class]

    for class_id in retain_classes:
        candidates, confidence = sample_candidates_for_class(
            weight=weight,
            bias=bias,
            embedding_dim=classifier.in_features,
            target_class=class_id,
            wanted=generated_per_class,
            batch_size=sample_batch_size,
            generator=generator,
        )
        order = confidence.argsort(descending=True)
        top = order[: min(retain_per_class, len(order))]
        bottom = order[-min(forget_per_class, len(order)) :]

        retain_features.append(candidates[top])
        retain_labels.append(torch.full((len(top),), class_id, dtype=torch.long))
        boundary_features.append(candidates[bottom])
        boundary_origins.append(torch.full((len(bottom),), class_id, dtype=torch.long))

        print(
            f"[sampling] class {class_id}: candidates={len(candidates)}, "
            f"retain={len(top)}, boundary={len(bottom)}"
        )

    return {
        "retain_features": torch.cat(retain_features),
        "retain_labels": torch.cat(retain_labels),
        "boundary_features": torch.cat(boundary_features),
        "boundary_origins": torch.cat(boundary_origins),
    }


def stratified_subsample(features, labels, per_class, seed):
    if per_class <= 0:
        return features, labels
    generator = torch.Generator().manual_seed(seed)
    indices = []
    for class_id in labels.unique(sorted=True):
        class_indices = torch.where(labels == class_id)[0]
        if len(class_indices) > per_class:
            permutation = torch.randperm(len(class_indices), generator=generator)[:per_class]
            class_indices = class_indices[permutation]
        indices.append(class_indices)
    selected = torch.cat(indices)
    return features[selected], labels[selected]


@torch.inference_mode()
def classifier_logits(model, num_classes, features, device):
    """Apply the final classifier head to pre-FC features."""
    _, classifier = get_final_linear(model, num_classes)
    classifier = classifier.to(device).eval()
    features = features.float().to(device)
    logits = classifier(features)
    return logits.detach().float().cpu()


def preprocess_for_tsne(features, mode):
    features = features.float()
    if mode == "l2":
        return F.normalize(features, dim=1)
    if mode == "zscore":
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True).clamp_min(1e-6)
        return (features - mean) / std
    return features


def make_tsne(features, perplexity, iterations, seed):
    kwargs = {
        "n_components": 2,
        "perplexity": min(perplexity, max(5.0, (len(features) - 1) / 3)),
        "init": "pca",
        "learning_rate": "auto",
        "random_state": seed,
        "metric": "euclidean",
    }
    if "max_iter" in inspect.signature(TSNE).parameters:
        kwargs["max_iter"] = iterations
    else:
        kwargs["n_iter"] = iterations
    return TSNE(**kwargs).fit_transform(features.numpy())


def class_colors(num_classes):
    cmap_name = "tab10" if num_classes <= 10 else "turbo"
    cmap = plt.get_cmap(cmap_name)
    denominator = max(num_classes - 1, 1)
    return {class_id: cmap(class_id / denominator) for class_id in range(num_classes)}


def plot_figure(
    projected_real,
    real_labels,
    real_pred_labels,
    projected_retain,
    retain_labels,
    projected_boundary,
    boundary_origins,
    projected_real_logits,
    projected_retain_logits,
    projected_boundary_logits,
    num_classes,
    forget_class,
    dataset,
    model_name,
    method,
    tsne_preprocess,
    output_path,
):
    colors = class_colors(num_classes)
    has_synthetic = len(retain_labels) > 0 or len(boundary_origins) > 0
    real_retain_size = 34
    real_forget_size = 42
    synthetic_size = 34
    real_retain_alpha = 0.55
    real_forget_alpha = 0.9
    fig_tsne, ax_tsne = plt.subplots(figsize=(7, 6))
    fig_logits, ax_logits = plt.subplots(figsize=(7, 6))

    for class_id in range(num_classes):
        real_mask = real_labels.numpy() == class_id
        if not np.any(real_mask):
            continue
        if class_id == forget_class:
            predicted_forget = real_pred_labels.numpy()
            for pred_class in range(num_classes):
                pred_mask = real_mask & (predicted_forget == pred_class)
                if not np.any(pred_mask):
                    continue
                ax_tsne.scatter(
                    projected_real[pred_mask, 0],
                    projected_real[pred_mask, 1],
                    s=real_forget_size,
                    marker="^",
                    color=colors[pred_class],
                    alpha=real_forget_alpha,
                    edgecolors="black",
                    linewidths=0.7,
                    zorder=3,
                )
        else:
            ax_tsne.scatter(
                projected_real[real_mask, 0],
                projected_real[real_mask, 1],
                s=real_retain_size,
                marker="o",
                color=colors[class_id],
                alpha=real_retain_alpha,
                edgecolors="white",
                linewidths=0.25,
                zorder=1,
            )

        retain_mask = retain_labels.numpy() == class_id
        if np.any(retain_mask):
            ax_tsne.scatter(
                projected_retain[retain_mask, 0],
                projected_retain[retain_mask, 1],
                s=synthetic_size,
                marker="s",
                color=colors[class_id],
                alpha=0.9,
                edgecolors="white",
                linewidths=0.25,
                zorder=4,
            )

    for class_id in range(num_classes):
        boundary_mask = boundary_origins.numpy() == class_id
        if np.any(boundary_mask):
            ax_tsne.scatter(
                projected_boundary[boundary_mask, 0],
                projected_boundary[boundary_mask, 1],
                s=synthetic_size,
                marker="x",
                color=colors[class_id],
                alpha=0.95,
                linewidths=1.0,
                zorder=5,
            )
    ax_tsne.set_xticks([])
    ax_tsne.set_yticks([])
    ax_tsne.grid(False)

    source_legend = [
        Line2D([0], [0], marker="o", linestyle="", color="gray", label="Real Retain Set"),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="",
            markerfacecolor="gray",
            markeredgecolor="black",
            color="black",
            label="Real Forget Set",
        ),
    ]
    if has_synthetic:
        source_legend.extend([
            Line2D(
                [0], [0], marker="s", linestyle="", color="gray",
                label="Synthetic Retain Set",
            ),
            Line2D(
                [0], [0], marker="x", linestyle="", color="gray",
                label="Synthetic Forget Set",
            ),
        ])
    for class_id in range(num_classes):
        real_mask = real_labels.numpy() == class_id
        if not np.any(real_mask):
            continue
        if class_id == forget_class:
            predicted_forget = real_pred_labels.numpy()
            for pred_class in range(num_classes):
                pred_mask = real_mask & (predicted_forget == pred_class)
                if not np.any(pred_mask):
                    continue
                ax_logits.scatter(
                    projected_real_logits[pred_mask, 0],
                    projected_real_logits[pred_mask, 1],
                    s=real_forget_size,
                    marker="^",
                    color=colors[pred_class],
                    alpha=real_forget_alpha,
                    edgecolors="black",
                    linewidths=0.7,
                    zorder=3,
                )
        else:
            ax_logits.scatter(
                projected_real_logits[real_mask, 0],
                projected_real_logits[real_mask, 1],
                s=real_retain_size,
                marker="o",
                color=colors[class_id],
                alpha=real_retain_alpha,
                edgecolors="white",
                linewidths=0.25,
                zorder=1,
            )

        retain_mask = retain_labels.numpy() == class_id
        if np.any(retain_mask):
            ax_logits.scatter(
                projected_retain_logits[retain_mask, 0],
                projected_retain_logits[retain_mask, 1],
                s=synthetic_size,
                marker="s",
                color=colors[class_id],
                alpha=0.9,
                edgecolors="white",
                linewidths=0.25,
                zorder=4,
            )

    for class_id in range(num_classes):
        boundary_mask = boundary_origins.numpy() == class_id
        if np.any(boundary_mask):
            ax_logits.scatter(
                projected_boundary_logits[boundary_mask, 0],
                projected_boundary_logits[boundary_mask, 1],
                s=synthetic_size,
                marker="x",
                color=colors[class_id],
                alpha=0.95,
                linewidths=1.0,
                zorder=5,
            )

    ax_logits.set_xticks([])
    ax_logits.set_yticks([])
    ax_logits.grid(False)
    ax_logits.legend(handles=source_legend, loc="best", fontsize=9, frameon=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".png"
    panel_specs = [
        ("a_feature_tsne", fig_tsne),
        ("b_logit_tsne", fig_logits),
    ]
    for tag, panel_fig in panel_specs:
        panel_fig.tight_layout()
        separate_path = output_path.with_name(f"{output_path.stem}_{tag}{suffix}")
        panel_fig.savefig(separate_path, dpi=600, bbox_inches="tight")
        if suffix.lower() != ".pdf":
            panel_fig.savefig(
                separate_path.with_suffix(".pdf"),
                bbox_inches="tight",
            )
        plt.close(panel_fig)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = NUM_CLASSES[args.dataset]
    if not 0 <= args.forget_class < num_classes:
        raise ValueError(f"--forget_class must be in [0, {num_classes - 1}].")
    if (
        not args.real_only
        and args.generated_per_class < max(args.retain_per_class, args.forget_per_class)
    ):
        raise ValueError(
            "--generated_per_class must be at least as large as both selection sizes."
        )

    checkpoint = args.checkpoint or checkpoint_for(
        args.method,
        args.dataset,
        args.model_name,
        args.forget_class,
        args.lr,
        args.ckpt_dir,
    )
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found:\n  {checkpoint}\n"
            "Pass --checkpoint explicitly if this run uses a different naming convention."
        )
    print(f"[model] loading {checkpoint}")
    model = load_checkpoint_model(
        checkpoint, args.model_name, args.dataset, num_classes, device
    )

    _, transform_test = get_transforms(
        args.dataset, args.model_name, wo_dataaug=False
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

    feature_model = make_feature_extractor(model, num_classes, device)
    real_features_all, real_labels_all = extract_features(feature_model, test_loader, device)
    real_features_plot, real_labels_plot = stratified_subsample(
        real_features_all, real_labels_all, args.real_per_class, args.seed
    )
    print(
        f"[real] extracted {len(real_features_all)} pre-FC test features; "
        f"plotting {len(real_features_plot)}"
    )

    if args.real_only:
        empty_features = torch.empty((0, real_features_plot.shape[1]), dtype=torch.float32)
        empty_labels = torch.empty(0, dtype=torch.long)
        probes = {
            "retain_features": empty_features,
            "retain_labels": empty_labels,
            "boundary_features": empty_features.clone(),
            "boundary_origins": empty_labels.clone(),
        }
        print("[sampling] real-only mode: synthetic probe generation skipped")
    else:
        probes = build_selected_probes(
            model=model,
            num_classes=num_classes,
            forget_class=args.forget_class,
            generated_per_class=args.generated_per_class,
            retain_per_class=args.retain_per_class,
            forget_per_class=args.forget_per_class,
            sample_batch_size=args.sample_batch_size,
            device=device,
            seed=args.seed,
        )

    all_features = torch.cat(
        [
            real_features_plot.float(),
            probes["retain_features"].float(),
            probes["boundary_features"].float(),
        ]
    )
    print(f"[tsne] jointly projecting {len(all_features)} points")
    tsne_features = preprocess_for_tsne(all_features, args.tsne_preprocess)
    projected = make_tsne(
        tsne_features,
        perplexity=args.perplexity,
        iterations=args.tsne_iterations,
        seed=args.seed,
    )
    n_real = len(real_features_plot)
    n_retain = len(probes["retain_features"])
    projected_real = projected[:n_real]
    projected_retain = projected[n_real : n_real + n_retain]
    projected_boundary = projected[n_real + n_retain :]

    all_logits = classifier_logits(model, num_classes, all_features, device)
    real_pred_labels_plot = all_logits[:n_real].argmax(dim=1).cpu()
    print(f"[tsne] jointly projecting {len(all_logits)} classifier-head outputs")
    logits_for_tsne = preprocess_for_tsne(all_logits, "zscore")
    projected_logits = make_tsne(
        logits_for_tsne,
        perplexity=args.perplexity,
        iterations=args.tsne_iterations,
        seed=args.seed,
    )
    projected_real_logits = projected_logits[:n_real]
    projected_retain_logits = projected_logits[n_real : n_real + n_retain]
    projected_boundary_logits = projected_logits[n_real + n_retain :]

    default_stem = (
        f"{args.dataset}_{args.model_name}_{args.method}_fg{args.forget_class}_real_only_tsne.png"
        if args.real_only
        else (
            f"{args.dataset}_{args.model_name}_{args.method}_lr{args.lr:g}_"
            f"fg{args.forget_class}_real_gaussian_tsne.png"
        )
    )
    output_path = Path(args.out) if args.out else Path(
        "results",
        args.method,
        default_stem,
    )
    plot_figure(
        projected_real=projected_real,
        real_labels=real_labels_plot,
        real_pred_labels=real_pred_labels_plot,
        projected_retain=projected_retain,
        retain_labels=probes["retain_labels"],
        projected_boundary=projected_boundary,
        boundary_origins=probes["boundary_origins"],
        projected_real_logits=projected_real_logits,
        projected_retain_logits=projected_retain_logits,
        projected_boundary_logits=projected_boundary_logits,
        num_classes=num_classes,
        forget_class=args.forget_class,
        dataset=args.dataset,
        model_name=args.model_name,
        method=args.method,
        tsne_preprocess=args.tsne_preprocess,
        output_path=output_path,
    )
    suffix = output_path.suffix or ".png"
    for tag in ["a_feature_tsne", "b_logit_tsne"]:
        separate_path = output_path.with_name(f"{output_path.stem}_{tag}{suffix}")
        print(f"[saved] {separate_path.resolve()}")
        if suffix.lower() != ".pdf":
            print(f"[saved] {separate_path.with_suffix('.pdf').resolve()}")


if __name__ == "__main__":
    main()
