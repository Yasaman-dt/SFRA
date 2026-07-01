"""Empirically validate the coefficient approximation used in Proposition 1.

For a specified unlearned checkpoint and forgotten class, this script generates
three matched Gaussian-probe groups:

  1. boundary: lowest-confidence accepted probes (the proposed selection);
  2. random: uniformly selected accepted probes;
  3. high-confidence: highest-confidence accepted probes.

It then produces an appendix figure with:

  (a) distributions of alpha_j(s) = 1 - p_cf(s) + p_j(s);
  (b) the exact weighted one-step margin expression versus the approximation
      obtained by replacing alpha_j(s) with one.

The script also writes probe-level and summary CSV files.
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader


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


CONTROL_ORDER = ["boundary", "random", "high-confidence"]
CONTROL_LABELS = {
    "boundary": "Low-confidence boundary",
    "random": "Random accepted",
    "high-confidence": "High-confidence",
}
CONTROL_COLORS = {
    "boundary": "#D62728",
    "random": "#7F7F7F",
    "high-confidence": "#1F77B4",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Empirically assess the coefficient and margin approximation "
            "used in the one-step relearning analysis."
        )
    )
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--dataset", required=True, choices=sorted(NUM_CLASSES)
    )
    parser.add_argument("--model", "--model_name", dest="model_name", default="resnet18")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument(
        "--forget_class", "--class_id", dest="forget_class", type=int, required=True
    )
    parser.add_argument(
        "--ckpt_dir",
        "--base_dir",
        dest="ckpt_dir",
        default="/export/livia/home/vision/Zdehghani/classification/exps",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_dir", default=str(Path("~/data").expanduser()))
    parser.add_argument(
        "--generated_per_class",
        "--tpr",
        dest="generated_per_class",
        type=int,
        default=500000,
        help="Accepted Gaussian candidates generated for each retain class.",
    )
    parser.add_argument(
        "--selected_per_class",
        "--forget_per_class",
        dest="selected_per_class",
        type=int,
        default=500,
        help="Number selected per retain class for every control group.",
    )
    parser.add_argument("--sample_batch_size", type=int, default=65536)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--bootstrap_repetitions",
        type=int,
        default=300,
        help="Bootstrap repetitions per retain class for panel (b).",
    )
    parser.add_argument(
        "--bootstrap_size",
        type=int,
        default=0,
        help="Probe count per bootstrap draw; 0 uses selected_per_class.",
    )
    parser.add_argument(
        "--max_alpha_points",
        type=int,
        default=250000,
        help="Maximum alpha values retained per control for plotting/CSV.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default=None,
        help="Output figure path. PNG and PDF are both produced.",
    )
    return parser.parse_args()


@torch.inference_mode()
def sample_accepted_candidates(
    weight,
    bias,
    target_class,
    wanted,
    batch_size,
    generator,
):
    """Sample N(0,I) features accepted as target_class and retain full softmax."""
    device = weight.device
    embedding_dim = weight.shape[1]
    feature_chunks, probability_chunks = [], []
    accepted = 0
    draws = 0
    max_draws = max(10_000_000, wanted * weight.shape[0] * 300)

    while accepted < wanted:
        current_batch = min(batch_size, max_draws - draws)
        if current_batch <= 0:
            raise RuntimeError(
                f"Only obtained {accepted}/{wanted} candidates for class "
                f"{target_class}. Reduce --generated_per_class."
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
            probability_chunks.append(probabilities[mask].cpu())
            accepted += int(mask.sum())
        draws += current_batch

    return (
        torch.cat(feature_chunks)[:wanted],
        torch.cat(probability_chunks)[:wanted],
    )


def choose_control_indices(target_confidence, selected, generator):
    count = len(target_confidence)
    if selected > count:
        raise ValueError("selected_per_class cannot exceed generated_per_class.")
    order = target_confidence.argsort(descending=True)
    high = order[:selected]
    boundary = order[-selected:]
    # Draw the random control from the middle of the ranked pool so that the
    # three controls are disjoint whenever the candidate pool is large enough.
    middle = order[selected : count - selected]
    if len(middle) >= selected:
        random_indices = middle[
            torch.randperm(len(middle), generator=generator)[:selected]
        ]
    else:
        excluded = torch.zeros(count, dtype=torch.bool)
        excluded[high] = True
        excluded[boundary] = True
        available = torch.where(~excluded)[0]
        if len(available) < selected:
            raise ValueError(
                "Disjoint controls require generated_per_class >= "
                "3 * selected_per_class."
            )
        random_indices = available[
            torch.randperm(len(available), generator=generator)[:selected]
        ]
    return {
        "boundary": boundary,
        "random": random_indices,
        "high-confidence": high,
    }


def generate_probe_controls(
    model,
    num_classes,
    forget_class,
    generated_per_class,
    selected_per_class,
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

    sample_generator = torch.Generator(device=device).manual_seed(seed)
    selection_generator = torch.Generator().manual_seed(seed + 1)
    controls = {
        name: {"features": [], "probabilities": [], "origin_classes": []}
        for name in CONTROL_ORDER
    }

    for retain_class in range(num_classes):
        if retain_class == forget_class:
            continue
        features, probabilities = sample_accepted_candidates(
            weight=weight,
            bias=bias,
            target_class=retain_class,
            wanted=generated_per_class,
            batch_size=sample_batch_size,
            generator=sample_generator,
        )
        indices = choose_control_indices(
            probabilities[:, retain_class], selected_per_class, selection_generator
        )
        for control_name, selected_indices in indices.items():
            controls[control_name]["features"].append(features[selected_indices])
            controls[control_name]["probabilities"].append(
                probabilities[selected_indices]
            )
            controls[control_name]["origin_classes"].append(
                torch.full(
                    (len(selected_indices),), retain_class, dtype=torch.long
                )
            )
        print(
            f"[sampling] retain class {retain_class}: accepted={len(features)}, "
            f"selected/control={selected_per_class}"
        )

    for control_name in CONTROL_ORDER:
        controls[control_name] = {
            key: torch.cat(value) for key, value in controls[control_name].items()
        }
    return controls


def compute_alpha(probabilities, forget_class, retain_classes):
    p_forget = probabilities[:, forget_class]
    p_retain = probabilities[:, retain_classes]
    alpha = 1.0 - p_forget[:, None] + p_retain
    return p_forget, alpha


def bootstrap_exact_vs_approx(
    features,
    probabilities,
    forget_mean,
    forget_class,
    retain_classes,
    repetitions,
    bootstrap_size,
    seed,
):
    """Return bootstrap estimates for each retain competitor j.

    exact_j  = E_s[(1-p_f(s)+p_j(s)) s]^T mu_Ef
    approx   = E_s[s]^T mu_Ef

    Learning rate eta is omitted because it scales both quantities equally.
    """
    generator = torch.Generator().manual_seed(seed)
    num_probes = len(features)
    draw_size = min(bootstrap_size, num_probes)
    forget_mean = forget_mean.float()
    records = []

    for repetition in range(repetitions):
        indices = torch.randint(
            0, num_probes, (draw_size,), generator=generator
        )
        sampled_features = features[indices].float()
        sampled_probabilities = probabilities[indices].float()
        approximate = float(sampled_features.mean(dim=0).dot(forget_mean))

        p_forget = sampled_probabilities[:, forget_class]
        for retain_class in retain_classes:
            alpha = (
                1.0
                - p_forget
                + sampled_probabilities[:, retain_class]
            )
            weighted_direction = (
                alpha[:, None] * sampled_features
            ).mean(dim=0)
            exact = float(weighted_direction.dot(forget_mean))
            records.append(
                {
                    "repetition": repetition,
                    "retain_class": retain_class,
                    "approximate": approximate,
                    "exact": exact,
                    "absolute_error": abs(exact - approximate),
                    "relative_error": abs(exact - approximate)
                    / max(abs(exact), 1e-12),
                }
            )
    return records


def summarize_control(control_name, p_forget, alpha):
    flat_alpha = alpha.flatten().numpy()
    p_forget_np = p_forget.numpy()
    return {
        "control": control_name,
        "num_probes": len(p_forget_np),
        "p_forget_mean": float(p_forget_np.mean()),
        "p_forget_std": float(p_forget_np.std()),
        "p_forget_median": float(np.median(p_forget_np)),
        "alpha_mean": float(flat_alpha.mean()),
        "alpha_std": float(flat_alpha.std()),
        "alpha_median": float(np.median(flat_alpha)),
        "alpha_q25": float(np.quantile(flat_alpha, 0.25)),
        "alpha_q75": float(np.quantile(flat_alpha, 0.75)),
        "alpha_within_005": float(np.mean(np.abs(flat_alpha - 1.0) < 0.05)),
        "alpha_within_010": float(np.mean(np.abs(flat_alpha - 1.0) < 0.10)),
    }


def subsample_array(values, maximum, seed):
    values = np.asarray(values)
    if len(values) <= maximum:
        return values
    rng = np.random.default_rng(seed)
    return values[rng.choice(len(values), size=maximum, replace=False)]


def ecdf(values):
    sorted_values = np.sort(np.asarray(values))
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    return sorted_values, cumulative


def correlation_summary(records):
    exact = np.asarray([row["exact"] for row in records])
    approximate = np.asarray([row["approximate"] for row in records])
    pearson = stats.pearsonr(approximate, exact)
    spearman = stats.spearmanr(approximate, exact)
    slope, intercept = np.polyfit(approximate, exact, 1)
    error = exact - approximate
    mae = float(np.mean(np.abs(exact - approximate)))
    rmse = float(np.sqrt(np.mean(error**2)))
    exact_std = float(np.std(exact))
    normalized_rmse = rmse / exact_std if exact_std > 0 else float("nan")
    sign_agreement = float(np.mean(np.signbit(approximate) == np.signbit(exact)))
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "regression_slope": float(slope),
        "regression_intercept": float(intercept),
        "sign_agreement": sign_agreement,
        "mae": mae,
        "rmse": rmse,
        "normalized_rmse": normalized_rmse,
    }


def full_sample_exact_vs_approx(
    features,
    probabilities,
    forget_mean,
    forget_class,
    retain_classes,
    control_name,
):
    """Evaluate the exact and approximate expressions without bootstrapping."""
    features = features.float()
    probabilities = probabilities.float()
    forget_mean = forget_mean.float()
    approximate = float(features.mean(dim=0).dot(forget_mean))
    p_forget = probabilities[:, forget_class]
    rows = []
    for retain_class in retain_classes:
        alpha = 1.0 - p_forget + probabilities[:, retain_class]
        exact = float((alpha[:, None] * features).mean(dim=0).dot(forget_mean))
        residual = exact - approximate
        rows.append(
            {
                "control": control_name,
                "retain_class": retain_class,
                "approximate": approximate,
                "exact": exact,
                "residual": residual,
                "absolute_error": abs(residual),
                "sign_agreement": int(
                    np.signbit(approximate) == np.signbit(exact)
                ),
                "robust_sufficient_condition": int(
                    approximate > abs(residual)
                ),
                "exact_condition": int(exact > 0),
            }
        )
    return rows


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(path, summary_rows):
    labels = {
        "boundary": "Boundary",
        "random": "Random",
        "high-confidence": "High-conf.",
    }
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Selection & $\mathbb{E}[\alpha_j]$ "
        r"& $\rho$ & Slope & Sign (\%) & NRMSE \\",
        r"\midrule",
    ]
    by_control = {row["control"]: row for row in summary_rows}
    for control_name in CONTROL_ORDER:
        row = by_control[control_name]
        lines.append(
            f"{labels[control_name]} & "
            f"{row['alpha_mean']:.3f} & "
            f"{row['spearman_rho']:.3f} & "
            f"{row['regression_slope']:.3f} & "
            f"{100 * row['sign_agreement']:.1f} & "
            f"{row['normalized_rmse']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def write_condition_latex_table(path, summary_rows):
    labels = {
        "boundary": "Boundary",
        "random": "Random",
        "high-confidence": "High-conf.",
    }
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Selection & Full-sample sign (\%) & Exact $>0$ (\%) "
        r"& Conservative condition (\%) \\",
        r"\midrule",
    ]
    by_control = {row["control"]: row for row in summary_rows}
    for control_name in CONTROL_ORDER:
        row = by_control[control_name]
        lines.append(
            f"{labels[control_name]} & "
            f"{100 * row['full_sample_sign_agreement']:.1f} & "
            f"{100 * row['exact_positive_rate']:.1f} & "
            f"{100 * row['robust_condition_rate']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def write_appendix_text(path, metadata, summary_rows):
    boundary = next(row for row in summary_rows if row["control"] == "boundary")
    text = f"""% Suggested appendix paragraph
\\paragraph{{Empirical assessment of the coefficient approximation.}}
We evaluate the approximation underlying Proposition~\\ref{{prop:relearning}}
for {metadata['dataset']} with a {metadata['model_name']} backbone,
{metadata['method']} unlearning, and class {metadata['forget_class']} as the
forget class. Although
$\\alpha_j(s)=1-p_{{c_f}}(s)+p_j(s)$ is not pointwise constant, its mean for
the selected boundary probes is {boundary['alpha_mean']:.3f}. More
importantly, the margin expression based on the unweighted synthetic mean
closely tracks the exact weighted expression across bootstrap samples and
retain-class competitors, with Pearson correlation
$r={boundary['pearson_r']:.3f}$, Spearman correlation
$\\rho={boundary['spearman_rho']:.3f}$, regression slope
${boundary['regression_slope']:.3f}$, {100 * boundary['sign_agreement']:.1f}\\%
sign agreement, and normalized RMSE {boundary['normalized_rmse']:.3f}.
Using the complete selected probe set, the exact margin-increase condition
holds for {100 * boundary['exact_positive_rate']:.1f}\\% of retain-class
competitors, while the conservative condition
$\\mu_{{\\mathcal S_f}}^\\top\\mu_{{\\mathcal E_f}}>|r_j|$ holds for
{100 * boundary['robust_condition_rate']:.1f}\\%.
These results support the approximation in aggregate for this evaluated
setting; the exact residual-based condition remains the formal statement.

% Suggested figure caption
\\textbf{{Empirical assessment of the coefficient approximation.}}
(a) Distribution of
$\\alpha_j(s)=1-p_{{c_f}}(s)+p_j(s)$ across selected probes and retain-class
competitors; the dashed line denotes one.
(b) Exact weighted margin expression versus the constant-coefficient
approximation over bootstrap samples. Boundary probes are compared with
disjoint random-accepted and high-confidence controls. The learning-rate
factor is omitted because it scales both axes equally.
"""
    path.write_text(text)


def plot_figure(
    alpha_by_control,
    bootstrap_by_control,
    summaries,
    correlation_summaries,
    output_path,
):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))

    # (a) Coefficient alpha.
    ax = axes[0]
    violin_values = [alpha_by_control[name] for name in CONTROL_ORDER]
    parts = ax.violinplot(
        violin_values,
        positions=np.arange(1, len(CONTROL_ORDER) + 1),
        showmeans=False,
        showmedians=True,
        showextrema=False,
        widths=0.8,
    )
    for body, control_name in zip(parts["bodies"], CONTROL_ORDER):
        body.set_facecolor(CONTROL_COLORS[control_name])
        body.set_edgecolor("black")
        body.set_alpha(0.72)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    ax.axhspan(0.95, 1.05, color="green", alpha=0.08)
    ax.set_xticks(
        np.arange(1, len(CONTROL_ORDER) + 1),
        ["Boundary", "Random", "High-confidence"],
        rotation=0,
        ha="center",
    )
    ax.set_ylabel(r"$\alpha_j(s)=1-p_{c_f}(s)+p_j(s)$")
    ax.grid(axis="y", alpha=0.2)

    # (b) Exact versus approximate margin expression.
    ax = axes[1]
    all_exact, all_approximate = [], []
    for control_name in CONTROL_ORDER:
        records = bootstrap_by_control[control_name]
        exact = np.asarray([row["exact"] for row in records])
        approximate = np.asarray([row["approximate"] for row in records])
        all_exact.append(exact)
        all_approximate.append(approximate)
        ax.scatter(
            approximate,
            exact,
            s=16,
            alpha=0.58,
            color=CONTROL_COLORS[control_name],
            edgecolors="white",
            linewidths=0.2,
            label=CONTROL_LABELS[control_name],
        )
    all_values = np.concatenate(all_exact + all_approximate)
    lower, upper = np.quantile(all_values, [0.005, 0.995])
    padding = 0.05 * max(upper - lower, 1e-8)
    ax.plot(
        [lower - padding, upper + padding],
        [lower - padding, upper + padding],
        linestyle="--",
        color="black",
        linewidth=1.2,
        label=r"$y=x$",
    )
    ax.set_xlim(lower - padding, upper + padding)
    ax.set_ylim(lower - padding, upper + padding)
    ax.set_xlabel(r"Approximate: $\mu_{\mathcal{S}_f}^{\top}\mu_{\mathcal{E}_f}$")
    ax.set_ylabel(
        r"Exact: $\mathbb{E}[\alpha_j(s)s]^{\top}\mu_{\mathcal{E}_f}$"
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.96,
        bottom=0.16,
        wspace=0.34,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


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
        raise ValueError(f"forget_class must be in [0, {num_classes - 1}].")
    if args.selected_per_class > args.generated_per_class:
        raise ValueError(
            "selected_per_class cannot exceed generated_per_class."
        )
    if args.generated_per_class < 3 * args.selected_per_class:
        raise ValueError(
            "Disjoint boundary, random, and high-confidence controls require "
            "generated_per_class >= 3 * selected_per_class."
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
        raise FileNotFoundError(checkpoint)
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
    real_features, real_labels = extract_features(
        feature_model, test_loader, device
    )
    forget_features = real_features[real_labels == args.forget_class].float()
    forget_mean = forget_features.mean(dim=0)
    print(
        f"[real] forget class {args.forget_class}: "
        f"{len(forget_features)} held-out features"
    )

    controls = generate_probe_controls(
        model=model,
        num_classes=num_classes,
        forget_class=args.forget_class,
        generated_per_class=args.generated_per_class,
        selected_per_class=args.selected_per_class,
        sample_batch_size=args.sample_batch_size,
        device=device,
        seed=args.seed,
    )
    retain_classes = [
        class_id
        for class_id in range(num_classes)
        if class_id != args.forget_class
    ]
    bootstrap_size = args.bootstrap_size or args.selected_per_class

    p_forget_by_control = {}
    alpha_by_control = {}
    bootstrap_by_control = {}
    summaries = []
    correlation_summaries = {}
    per_class_rows = []

    for control_index, control_name in enumerate(CONTROL_ORDER):
        probabilities = controls[control_name]["probabilities"].float()
        p_forget, alpha = compute_alpha(
            probabilities, args.forget_class, retain_classes
        )
        p_forget_by_control[control_name] = p_forget.numpy()
        alpha_by_control[control_name] = subsample_array(
            alpha.flatten().numpy(),
            args.max_alpha_points,
            args.seed + control_index,
        )
        summaries.append(summarize_control(control_name, p_forget, alpha))
        bootstrap_records = bootstrap_exact_vs_approx(
            features=controls[control_name]["features"],
            probabilities=probabilities,
            forget_mean=forget_mean,
            forget_class=args.forget_class,
            retain_classes=retain_classes,
            repetitions=args.bootstrap_repetitions,
            bootstrap_size=bootstrap_size,
            seed=args.seed + 100 + control_index,
        )
        for row in bootstrap_records:
            row["control"] = control_name
        bootstrap_by_control[control_name] = bootstrap_records
        correlation_summaries[control_name] = correlation_summary(
            bootstrap_records
        )
        per_class_rows.extend(
            full_sample_exact_vs_approx(
                features=controls[control_name]["features"],
                probabilities=probabilities,
                forget_mean=forget_mean,
                forget_class=args.forget_class,
                retain_classes=retain_classes,
                control_name=control_name,
            )
        )

    output_path = Path(args.out) if args.out else Path(
        "results",
        args.method,
        (
            f"{args.dataset}_{args.model_name}_{args.method}_lr{args.lr:g}_"
            f"fg{args.forget_class}_coefficient_validation.png"
        ),
    )
    output_stem = output_path.with_suffix("")

    summary_rows = []
    for row in summaries:
        combined = dict(row)
        combined.update(correlation_summaries[row["control"]])
        control_class_rows = [
            class_row
            for class_row in per_class_rows
            if class_row["control"] == row["control"]
        ]
        combined["full_sample_sign_agreement"] = float(
            np.mean(
                [class_row["sign_agreement"] for class_row in control_class_rows]
            )
        )
        combined["robust_condition_rate"] = float(
            np.mean(
                [
                    class_row["robust_sufficient_condition"]
                    for class_row in control_class_rows
                ]
            )
        )
        combined["exact_positive_rate"] = float(
            np.mean(
                [class_row["exact_condition"] for class_row in control_class_rows]
            )
        )
        summary_rows.append(combined)
    save_csv(output_stem.with_name(output_stem.name + "_summary.csv"), summary_rows)
    save_csv(
        output_stem.with_name(output_stem.name + "_bootstrap.csv"),
        [
            row
            for control_name in CONTROL_ORDER
            for row in bootstrap_by_control[control_name]
        ],
    )
    save_csv(
        output_stem.with_name(output_stem.name + "_per_retain_class.csv"),
        per_class_rows,
    )

    alpha_rows = []
    p_forget_rows = []
    for control_name in CONTROL_ORDER:
        for value in p_forget_by_control[control_name]:
            p_forget_rows.append(
                {"control": control_name, "p_forget": float(value)}
            )
        for value in alpha_by_control[control_name]:
            alpha_rows.append(
                {"control": control_name, "alpha": float(value)}
            )
    save_csv(
        output_stem.with_name(output_stem.name + "_p_forget.csv"),
        p_forget_rows,
    )
    save_csv(
        output_stem.with_name(output_stem.name + "_alpha.csv"), alpha_rows
    )

    metadata = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "method": args.method,
        "learning_rate": args.lr,
        "forget_class": args.forget_class,
        "generated_per_class": args.generated_per_class,
        "selected_per_class": args.selected_per_class,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_size": bootstrap_size,
        "seed": args.seed,
        "checkpoint": checkpoint,
    }
    output_stem.with_name(output_stem.name + "_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    write_latex_table(
        output_stem.with_name(output_stem.name + "_table.tex"),
        summary_rows,
    )
    write_condition_latex_table(
        output_stem.with_name(output_stem.name + "_condition_table.tex"),
        summary_rows,
    )
    write_appendix_text(
        output_stem.with_name(output_stem.name + "_appendix_text.tex"),
        metadata,
        summary_rows,
    )

    plot_figure(
        alpha_by_control=alpha_by_control,
        bootstrap_by_control=bootstrap_by_control,
        summaries=summaries,
        correlation_summaries=correlation_summaries,
        output_path=output_path,
    )

    print("\n[summary]")
    for row in summary_rows:
        print(
            f"{row['control']:>15}: "
            f"mean p_f={row['p_forget_mean']:.6f}, "
            f"mean alpha={row['alpha_mean']:.6f}, "
            f"within 1±.05={100*row['alpha_within_005']:.2f}%, "
            f"Spearman={row['spearman_rho']:.4f}, "
            f"slope={row['regression_slope']:.4f}, "
            f"sign={100*row['sign_agreement']:.2f}%, "
            f"NRMSE={row['normalized_rmse']:.4f}"
        )
    print(f"[saved] {output_path.resolve()}")
    if output_path.suffix.lower() != ".pdf":
        print(f"[saved] {output_path.with_suffix('.pdf').resolve()}")


if __name__ == "__main__":
    main()
