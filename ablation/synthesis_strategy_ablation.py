"""Ablate synthetic-feature generation and probe-selection strategies.

The default ``one_factor`` grid changes one component at a time around the
paper baseline:

    distribution=gaussian, forget_selection=low_confidence,
    retain_selection=high_confidence.

This yields five interpretable strategies rather than a large Cartesian grid:
three sampling distributions, three forget-probe selection rules, and two
retain-probe selection rules. The distributions are variance matched:

* Gaussian: N(0, I)
* Uniform: U[-sqrt(3), sqrt(3)]
* Laplace: Laplace(0, 1/sqrt(2))

For every strategy, the script freezes the feature extractor, relearns only
the classifier head, and reports baseline/relearned forget accuracy, retain
accuracy, and the harmonic Relearning Score used in the WACV manuscript.
Runs are resumable through a per-run CSV.

Example:
    CUDA_VISIBLE_DEVICES=0 python -m ablation.synthesis_strategy_ablation \
      --method bad_teacher --dataset cifar10 --model_name resnet18 \
      --lr 0.001 --forget_class 7 --generated_per_class 500000 \
      --retain_per_class 500 --forget_per_class 500 --epochs 200
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plots_tables.tsne_real_gaussian_probes import (  # noqa: E402
    NUM_CLASSES,
    checkpoint_for,
    extract_features,
    get_final_linear,
    load_checkpoint_model,
    make_feature_extractor,
)
from utils import get_dataset, get_transforms  # noqa: E402


DISTRIBUTIONS = ("gaussian", "uniform", "laplace")
UNCERTAINTY_SCORES = ("msp", "softmax", "entropy", "energy")
FORGET_SELECTIONS = ("low_confidence", "high_confidence", "random")
RETAIN_SELECTIONS = ("high_confidence", "random")


@dataclass(frozen=True)
class Strategy:
    distribution: str
    forget_selection: str
    retain_selection: str
    uncertainty_score: str = "softmax"

    @property
    def name(self):
        base = (
            f"dist={self.distribution}__forget={self.forget_selection}"
            f"__retain={self.retain_selection}"
        )
        if self.uncertainty_score != "softmax":
            base += f"__score={self.uncertainty_score}"
        return base


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablate synthetic sampling and probe-selection strategies."
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(NUM_CLASSES))
    parser.add_argument("--model", "--model_name", dest="model_name", default="resnet18")
    parser.add_argument(
        "--lr",
        type=float,
        required=True,
        help="Unlearning learning rate used only to locate the checkpoint.",
    )
    parser.add_argument(
        "--forget_class",
        type=int,
        default=None,
        help="Single forgotten class (backward-compatible shorthand).",
    )
    parser.add_argument(
        "--forget_classes",
        type=int,
        nargs="+",
        default=None,
        help="List of independently forgotten classes, e.g. 0 1 2 3 4.",
    )
    parser.add_argument(
        "--ckpt_dir",
        default="/export/livia/home/vision/Zdehghani/classification/exps",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_dir", default=str(Path("~/data").expanduser()))
    parser.add_argument("--generated_per_class", type=int, default=500000)
    parser.add_argument("--retain_per_class", type=int, default=500)
    parser.add_argument("--forget_per_class", type=int, default=500)
    parser.add_argument("--sample_batch_size", type=int, default=4096)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--head_lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument(
        "--retain_floor_frac",
        type=float,
        default=0.90,
        help="Only checkpoints retaining this fraction of baseline retain accuracy compete.",
    )
    parser.add_argument(
        "--grid",
        choices=["one_factor", "full"],
        default="one_factor",
    )
    parser.add_argument(
        "--skip_gaussian_baseline",
        action="store_true",
        help=(
            "Do not add the Gaussian/low-confidence/high-confidence baseline "
            "automatically in the one-factor grid. Useful when that baseline "
            "has already been run and only a new distribution is requested."
        ),
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=DISTRIBUTIONS,
        default=list(DISTRIBUTIONS),
    )
    parser.add_argument(
        "--uncertainty_scores", "--uncertainty_metric",
        nargs="+",
        choices=UNCERTAINTY_SCORES,
        default=["softmax"],
        help=(
            "Score used to rank both forget and retain probes. 'msp' (alias "
            "'softmax') uses assigned-class probability, 'entropy' uses "
            "predictive entropy, and 'energy' uses -logsumexp(logits) at T=1."
        ),
    )
    parser.add_argument(
        "--forget_selections",
        nargs="+",
        choices=FORGET_SELECTIONS,
        default=list(FORGET_SELECTIONS),
    )
    parser.add_argument(
        "--retain_selections",
        nargs="+",
        choices=RETAIN_SELECTIONS,
        default=list(RETAIN_SELECTIONS),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )
    parser.add_argument(
        "--output_dir",
        default="results_synthesis_ablation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun completed strategy/seed rows.",
    )
    return parser.parse_args()


def make_strategies(args):
    baseline_score = "msp" if "msp" in args.uncertainty_scores else "softmax"
    baseline = Strategy("gaussian", "low_confidence", "high_confidence", baseline_score)
    if args.grid == "full":
        return [
            Strategy(distribution, forget_selection, retain_selection, uncertainty_score)
            for distribution in args.distributions
            for uncertainty_score in args.uncertainty_scores
            for forget_selection in args.forget_selections
            for retain_selection in args.retain_selections
        ]

    strategies = [] if args.skip_gaussian_baseline else [baseline]
    strategies.extend(
        Strategy(
            distribution,
            baseline.forget_selection,
            baseline.retain_selection,
            baseline.uncertainty_score,
        )
        for distribution in args.distributions
        if distribution != baseline.distribution
    )
    strategies.extend(
        Strategy(
            baseline.distribution,
            forget_selection,
            baseline.retain_selection,
            baseline.uncertainty_score,
        )
        for forget_selection in args.forget_selections
        if forget_selection != baseline.forget_selection
    )
    strategies.extend(
        Strategy(
            baseline.distribution,
            baseline.forget_selection,
            retain_selection,
            baseline.uncertainty_score,
        )
        for retain_selection in args.retain_selections
        if retain_selection != baseline.retain_selection
    )
    strategies.extend(
        Strategy(
            baseline.distribution,
            baseline.forget_selection,
            baseline.retain_selection,
            uncertainty_score,
        )
        for uncertainty_score in args.uncertainty_scores
        if uncertainty_score != baseline.uncertainty_score
    )
    # Preserve order while removing duplicates introduced by restricted CLI lists.
    return list(dict.fromkeys(strategies))


def sample_distribution(shape, distribution, generator, device):
    if distribution == "gaussian":
        return torch.randn(*shape, generator=generator, device=device)
    if distribution == "uniform":
        samples = torch.rand(*shape, generator=generator, device=device)
        return samples.mul_(2 * math.sqrt(3)).sub_(math.sqrt(3))
    if distribution == "laplace":
        # Inverse-CDF sampling for zero-mean, unit-variance Laplace probes.
        # A Laplace distribution has variance 2*b^2, hence b=1/sqrt(2).
        uniform = torch.rand(*shape, generator=generator, device=device)
        eps = torch.finfo(uniform.dtype).eps
        centered = uniform.clamp_(min=eps, max=1.0 - eps).sub_(0.5)
        return (
            -math.sqrt(0.5)
            * torch.sign(centered)
            * torch.log1p(-2.0 * torch.abs(centered))
        )
    raise ValueError(distribution)


@torch.inference_mode()
def sample_accepted_pool(
    weight,
    bias,
    target_class,
    wanted,
    distribution,
    batch_size,
    generator,
):
    """Return accepted features, logits, and full probabilities for one predicted class."""
    device = weight.device
    embedding_dim = weight.shape[1]
    feature_chunks, logit_chunks, probability_chunks = [], [], []
    accepted = 0
    draws = 0
    max_draws = max(10_000_000, wanted * weight.shape[0] * 400)

    while accepted < wanted:
        current_batch = min(batch_size, max_draws - draws)
        if current_batch <= 0:
            raise RuntimeError(
                f"Only sampled {accepted}/{wanted} accepted points for class "
                f"{target_class} with distribution={distribution}."
            )
        samples = sample_distribution(
            (current_batch, embedding_dim), distribution, generator, device
        )
        logits = samples @ weight.T + bias
        probabilities = F.softmax(logits, dim=1)
        mask = probabilities.argmax(dim=1).eq(target_class)
        if mask.any():
            feature_chunks.append(samples[mask].cpu())
            logit_chunks.append(logits[mask].cpu())
            probability_chunks.append(probabilities[mask].cpu())
            accepted += int(mask.sum())
        draws += current_batch

    return (
        torch.cat(feature_chunks)[:wanted],
        torch.cat(logit_chunks)[:wanted],
        torch.cat(probability_chunks)[:wanted],
        draws,
    )


def energy_score(logits):
    # Standard energy score with temperature T=1. Larger values correspond to
    # lower log-sum-exp and are commonly interpreted as more uncertain.
    return -torch.logsumexp(logits, dim=1)


def uncertainty_values(probabilities, logits, target_class, uncertainty_score):
    """Return scores where larger always means more uncertain."""
    if uncertainty_score in {"msp", "softmax"}:
        return -probabilities[:, target_class]
    if uncertainty_score == "entropy":
        return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    if uncertainty_score == "energy":
        # E=-logsumexp(logits) at T=1; larger (less negative) is more uncertain.
        return energy_score(logits)
    raise ValueError(uncertainty_score)


def forget_score(probabilities, logits, target_class, rule, uncertainty_score):
    uncertainty = uncertainty_values(
        probabilities, logits, target_class, uncertainty_score
    )
    if rule == "low_confidence":
        return uncertainty
    if rule == "high_confidence":
        return -uncertainty
    if rule == "random":
        return torch.zeros(len(probabilities))
    raise ValueError(f"rule={rule}, uncertainty_score={uncertainty_score}")


def select_indices(
    probabilities,
    logits,
    target_class,
    forget_rule,
    retain_rule,
    uncertainty_score,
    forget_k,
    retain_k,
    generator,
):
    count = len(probabilities)
    if forget_k + retain_k > count:
        raise ValueError(
            "generated_per_class must be at least retain_per_class + "
            "forget_per_class for disjoint selections."
        )

    if forget_rule == "random":
        forget_indices = torch.randperm(count, generator=generator)[:forget_k]
    else:
        scores = forget_score(
            probabilities,
            logits,
            target_class,
            forget_rule,
            uncertainty_score,
        )
        forget_indices = scores.argsort(descending=True)[:forget_k]

    available_mask = torch.ones(count, dtype=torch.bool)
    available_mask[forget_indices] = False
    available = torch.where(available_mask)[0]

    if retain_rule == "random":
        retain_indices = available[
            torch.randperm(len(available), generator=generator)[:retain_k]
        ]
    else:
        uncertainty = uncertainty_values(
            probabilities[available], logits[available], target_class, uncertainty_score
        )
        retain_indices = available[uncertainty.argsort(descending=False)[:retain_k]]
    return retain_indices, forget_indices


def build_synthetic_sets_for_strategies(
    classifier,
    num_classes,
    forget_class,
    strategies,
    generated_per_class,
    retain_per_class,
    forget_per_class,
    sample_batch_size,
    device,
    seed,
):
    """Generate each class pool once and select all strategies from that pool."""
    distributions = {strategy.distribution for strategy in strategies}
    if len(distributions) != 1:
        raise ValueError("All strategies in one pool pass must share a distribution.")
    distribution = next(iter(distributions))
    weight = classifier.weight.detach().float().to(device)
    bias = (
        classifier.bias.detach().float().to(device)
        if classifier.bias is not None
        else torch.zeros(num_classes, device=device)
    )
    sample_generator = torch.Generator(device=device).manual_seed(seed)
    selection_generator = torch.Generator().manual_seed(seed + 1729)
    selected = {
        strategy: {
            "retain_features": [],
            "retain_labels": [],
            "forget_features": [],
        }
        for strategy in strategies
    }
    total_draws = 0
    agreement_rows = []

    for retain_class in range(num_classes):
        if retain_class == forget_class:
            continue
        features, logits, probabilities, draws = sample_accepted_pool(
            weight=weight,
            bias=bias,
            target_class=retain_class,
            wanted=generated_per_class,
            distribution=distribution,
            batch_size=sample_batch_size,
            generator=sample_generator,
        )
        forget_indices_by_score = {}
        for strategy in strategies:
            retain_indices, forget_indices = select_indices(
                probabilities=probabilities,
                logits=logits,
                target_class=retain_class,
                forget_rule=strategy.forget_selection,
                retain_rule=strategy.retain_selection,
                uncertainty_score=strategy.uncertainty_score,
                forget_k=forget_per_class,
                retain_k=retain_per_class,
                generator=selection_generator,
            )
            selected[strategy]["retain_features"].append(features[retain_indices])
            selected[strategy]["retain_labels"].append(
                torch.full((retain_per_class,), retain_class, dtype=torch.long)
            )
            selected[strategy]["forget_features"].append(features[forget_indices])
            if (
                strategy.distribution == "gaussian"
                and strategy.forget_selection == "low_confidence"
                and strategy.retain_selection == "high_confidence"
            ):
                score_name = "msp" if strategy.uncertainty_score == "softmax" else strategy.uncertainty_score
                forget_indices_by_score[score_name] = forget_indices
        score_names = [name for name in ("msp", "entropy", "energy") if name in forget_indices_by_score]
        for left_index, left in enumerate(score_names):
            left_set = set(forget_indices_by_score[left].tolist())
            for right in score_names[left_index + 1:]:
                right_set = set(forget_indices_by_score[right].tolist())
                agreement_rows.append({
                    "retain_class": retain_class,
                    "metric_a": left,
                    "metric_b": right,
                    "overlap_at_m": len(left_set & right_set) / max(forget_per_class, 1),
                })
        total_draws += draws

    outputs = {}
    for strategy, parts in selected.items():
        retain_features = torch.cat(parts["retain_features"]).float()
        retain_labels = torch.cat(parts["retain_labels"])
        forget_features = torch.cat(parts["forget_features"]).float()
        forget_labels = torch.full(
            (len(forget_features),), forget_class, dtype=torch.long
        )
        outputs[strategy] = (
            retain_features,
            retain_labels,
            forget_features,
            forget_labels,
            total_draws,
        )
    return outputs, agreement_rows


@torch.inference_mode()
def accuracy(classifier, features, labels, device, batch_size):
    classifier.eval()
    correct = 0
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=False,
    )
    for batch_features, batch_labels in loader:
        logits = classifier(batch_features.to(device))
        correct += (
            logits.argmax(dim=1).cpu() == batch_labels
        ).sum().item()
    return 100.0 * correct / max(len(labels), 1)


def relearning_score(retain_un, forget_un, retain_re, forget_re):
    """Harmonic RS from the WACV manuscript; accuracies are percentages."""
    retain_preservation = 1.0 - max(0.0, (retain_un - retain_re) / 100.0)
    forget_recovery = max(0.0, (forget_re - forget_un) / 100.0)
    retain_preservation = float(np.clip(retain_preservation, 0.0, 1.0))
    forget_recovery = float(np.clip(forget_recovery, 0.0, 1.0))
    if retain_preservation + forget_recovery == 0:
        return 0.0
    return (
        2.0
        * retain_preservation
        * forget_recovery
        / (retain_preservation + forget_recovery)
    )


def train_one_strategy(
    initial_classifier,
    synthetic,
    test_retain_features,
    test_retain_labels,
    test_forget_features,
    test_forget_labels,
    args,
    device,
    seed,
):
    retain_features, retain_labels, forget_features, forget_labels, _ = synthetic
    classifier = copy.deepcopy(initial_classifier).to(device)
    classifier.train()
    train_features = torch.cat([retain_features, forget_features])
    train_labels = torch.cat([retain_labels, forget_labels])
    generator = torch.Generator().manual_seed(seed + 991)
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=generator,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    retain_un = accuracy(
        classifier,
        test_retain_features,
        test_retain_labels,
        device,
        args.eval_batch_size,
    )
    forget_un = accuracy(
        classifier,
        test_forget_features,
        test_forget_labels,
        device,
        args.eval_batch_size,
    )
    retain_floor = args.retain_floor_frac * retain_un
    best = {
        "epoch": 0,
        "retain_acc": retain_un,
        "forget_acc": forget_un,
        "RS": 0.0,
        "state": copy.deepcopy(classifier.state_dict()),
    }
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(classifier(batch_features), batch_labels)
            loss.backward()
            optimizer.step()

        if epoch % args.eval_every != 0 and epoch != args.epochs:
            continue
        retain_re = accuracy(
            classifier,
            test_retain_features,
            test_retain_labels,
            device,
            args.eval_batch_size,
        )
        forget_re = accuracy(
            classifier,
            test_forget_features,
            test_forget_labels,
            device,
            args.eval_batch_size,
        )
        score = relearning_score(retain_un, forget_un, retain_re, forget_re)
        improved = retain_re >= retain_floor and score > best["RS"] + 1e-12
        if improved:
            best = {
                "epoch": epoch,
                "retain_acc": retain_re,
                "forget_acc": forget_re,
                "RS": score,
                "state": copy.deepcopy(classifier.state_dict()),
            }
            no_improve = 0
        else:
            no_improve += args.eval_every
            if no_improve >= args.patience:
                break

    return {
        "baseline_retain_acc": retain_un,
        "baseline_forget_acc": forget_un,
        "relearned_retain_acc": best["retain_acc"],
        "relearned_forget_acc": best["forget_acc"],
        "RS": best["RS"],
        "best_epoch": best["epoch"],
        "retain_drop": retain_un - best["retain_acc"],
        "forget_gain": best["forget_acc"] - forget_un,
    }


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        with path.open(newline="") as handle:
            existing_fields = next(csv.reader(handle), [])
        missing_fields = [field for field in row if field not in existing_fields]
        if missing_fields:
            # Older resumable runs predate uncertainty_score/ranking metadata.
            # Evolve their schema before appending so columns never shift.
            prior = pd.read_csv(path)
            for field in missing_fields:
                prior[field] = ""
            prior.to_csv(path, index=False)
    with path.open("a", newline="") as handle:
        fieldnames = list(pd.read_csv(path, nrows=0).columns) if exists else list(row.keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def aggregate_results(run_csv):
    frame = pd.read_csv(run_csv)
    group_columns = [
        "strategy",
        "distribution",
        "forget_selection",
        "retain_selection",
    ]
    if "uncertainty_score" in frame.columns:
        group_columns.append("uncertainty_score")
    metrics = [
        "baseline_retain_acc",
        "baseline_forget_acc",
        "relearned_retain_acc",
        "relearned_forget_acc",
        "retain_drop",
        "forget_gain",
        "RS",
        "best_epoch",
    ]
    grouped = frame.groupby(group_columns, sort=False)[metrics].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    summary = grouped.reset_index()

    return summary


def write_latex_table(summary, path):
    def fmt(row, metric):
        mean = row[f"{metric}_mean"]
        std = row[f"{metric}_std"]
        if pd.isna(std):
            return f"{mean:.3f}"
        return f"${mean:.3f}\\pm{std:.3f}$"

    columns = "llllrrrr"
    lines = [
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        "Distribution & Forget selection & Retain selection & Score & "
        + r"$A_f^{\mathrm{re}}$ & $A_r^{\mathrm{re}}$ & Gain$_f$ & RS \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        values = [
            str(row["distribution"]).replace("_", r"\_"),
            str(row["forget_selection"]).replace("_", r"\_"),
            str(row["retain_selection"]).replace("_", r"\_"),
            str(row.get("uncertainty_score", "softmax")).replace("_", r"\_"),
            fmt(row, "relearned_forget_acc"),
            fmt(row, "relearned_retain_acc"),
            fmt(row, "forget_gain"),
            fmt(row, "RS"),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def plot_summary(summary, output_path):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
        }
    )
    labels = [
        (
            f"{row.distribution}\n{row.forget_selection}\n"
            f"{row.retain_selection}\n{getattr(row, 'uncertainty_score', 'softmax')}"
        )
        for row in summary.itertuples()
    ]
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].bar(x, summary["relearned_forget_acc_mean"], color="#D62728")
    axes[0].set_ylabel("Relearned forget accuracy (%)")
    axes[1].bar(x, summary["relearned_retain_acc_mean"], color="#1F77B4")
    axes[1].set_ylabel("Relearned retain accuracy (%)")
    axes[2].bar(x, summary["RS_mean"], color="#2CA02C")
    axes[2].set_ylabel("RS")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run_for_forget_class(args, forget_class, test_dataset, strategies, device):
    num_classes = NUM_CLASSES[args.dataset]
    checkpoint = args.checkpoint or checkpoint_for(
        args.method,
        args.dataset,
        args.model_name,
        forget_class,
        args.lr,
        args.ckpt_dir,
    )
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint for forget class {forget_class} was not found:\n"
            f"  {checkpoint}"
        )
    print(f"\n{'=' * 20} FORGET CLASS {forget_class} {'=' * 20}")
    print(f"[checkpoint] {checkpoint}")
    model = load_checkpoint_model(
        checkpoint, args.model_name, args.dataset, num_classes, device
    )
    feature_model = make_feature_extractor(model, num_classes, device)
    _, classifier = get_final_linear(model, num_classes)
    initial_classifier = copy.deepcopy(classifier).cpu()

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    real_features, real_labels = extract_features(feature_model, test_loader, device)
    forget_mask = real_labels.eq(forget_class)
    test_forget_features = real_features[forget_mask].float()
    test_forget_labels = real_labels[forget_mask]
    test_retain_features = real_features[~forget_mask].float()
    test_retain_labels = real_labels[~forget_mask]

    run_root = Path(args.output_dir) / args.method / (
        f"{args.dataset}_{args.model_name}_lr{args.lr:g}_fg{forget_class}"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    run_csv = run_root / "runs.csv"
    metadata_path = run_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                **vars(args),
                "forget_class": forget_class,
                "checkpoint": checkpoint,
                "strategies": [strategy.name for strategy in strategies],
            },
            indent=2,
        )
        + "\n"
    )

    completed = set()
    if run_csv.exists() and not args.force:
        prior = pd.read_csv(run_csv)
        completed = set(zip(prior["strategy"], prior["seed"]))

    for seed in args.seeds:
        for distribution in dict.fromkeys(
            strategy.distribution for strategy in strategies
        ):
            distribution_strategies = [
                strategy
                for strategy in strategies
                if strategy.distribution == distribution
                and (strategy.name, seed) not in completed
            ]
            if not distribution_strategies:
                continue
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            print(
                f"\n[pool] distribution={distribution} seed={seed}; "
                f"strategies={len(distribution_strategies)}"
            )
            synthetic_by_strategy, agreement_rows = build_synthetic_sets_for_strategies(
                classifier=classifier,
                num_classes=num_classes,
                forget_class=forget_class,
                strategies=distribution_strategies,
                generated_per_class=args.generated_per_class,
                retain_per_class=args.retain_per_class,
                forget_per_class=args.forget_per_class,
                sample_batch_size=args.sample_batch_size,
                device=device,
                seed=seed,
            )
            agreement_path = run_root / "ranking_agreement.csv"
            for agreement in agreement_rows:
                append_csv(agreement_path, {
                    "method": args.method,
                    "dataset": args.dataset,
                    "model": args.model_name,
                    "forget_class": forget_class,
                    "seed": seed,
                    "selected_m": args.forget_per_class,
                    **agreement,
                })
            for strategy in distribution_strategies:
                print(f"[train] {strategy.name} seed={seed}")
                synthetic = synthetic_by_strategy[strategy]
                result = train_one_strategy(
                    initial_classifier=initial_classifier,
                    synthetic=synthetic,
                    test_retain_features=test_retain_features,
                    test_retain_labels=test_retain_labels,
                    test_forget_features=test_forget_features,
                    test_forget_labels=test_forget_labels,
                    args=args,
                    device=device,
                    seed=seed,
                )
                row = {
                    "method": args.method,
                    "dataset": args.dataset,
                    "model": args.model_name,
                    "unlearn_lr": args.lr,
                    "forget_class": forget_class,
                    "seed": seed,
                    "strategy": strategy.name,
                    "distribution": strategy.distribution,
                    "forget_selection": strategy.forget_selection,
                    "retain_selection": strategy.retain_selection,
                    "uncertainty_score": strategy.uncertainty_score,
                    "generated_per_class": args.generated_per_class,
                    "retain_per_class": args.retain_per_class,
                    "forget_per_class": args.forget_per_class,
                    "total_draws": synthetic[-1],
                    **result,
                }
                append_csv(run_csv, row)
                print(
                    f"[result] Af={result['relearned_forget_acc']:.2f} "
                    f"Ar={result['relearned_retain_acc']:.2f} "
                    f"RS={result['RS']:.4f} epoch={result['best_epoch']}"
                )

    summary = aggregate_results(run_csv)
    summary.to_csv(run_root / "summary.csv", index=False)
    write_latex_table(summary, run_root / "summary_table.tex")
    plot_summary(summary, run_root / "summary_plot.png")
    print(f"\n[saved] {run_root.resolve()}")
    del model, feature_model, classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    if args.generated_per_class < args.retain_per_class + args.forget_per_class:
        raise ValueError(
            "generated_per_class must be >= retain_per_class + forget_per_class."
        )
    num_classes = NUM_CLASSES[args.dataset]
    if args.forget_classes is not None:
        forget_classes = list(dict.fromkeys(args.forget_classes))
        if args.forget_class is not None:
            forget_classes = list(
                dict.fromkeys([args.forget_class] + forget_classes)
            )
    elif args.forget_class is not None:
        forget_classes = [args.forget_class]
    else:
        raise ValueError("Provide --forget_class or --forget_classes.")

    invalid = [class_id for class_id in forget_classes if not 0 <= class_id < num_classes]
    if invalid:
        raise ValueError(
            f"Invalid forget classes {invalid}; valid range is 0..{num_classes - 1}."
        )
    if args.checkpoint is not None and len(forget_classes) > 1:
        raise ValueError(
            "--checkpoint can only be used with one forget class. For a list, "
            "the runner resolves each checkpoint from --ckpt_dir automatically."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strategies = make_strategies(args)
    _, test_transform = get_transforms(
        args.dataset, args.model_name, wo_dataaug=False
    )
    _, test_dataset = get_dataset(
        args.dataset,
        test_transform,
        test_transform,
        path=Path(args.data_dir).expanduser(),
    )

    for forget_class in forget_classes:
        run_for_forget_class(
            args=args,
            forget_class=forget_class,
            test_dataset=test_dataset,
            strategies=strategies,
            device=device,
        )


if __name__ == "__main__":
    main()
