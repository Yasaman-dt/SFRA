"""Post-hoc synthetic--real representation alignment analysis.

This module does not train or alter a model. It reproduces the single-class
pipeline's classifier-input feature extraction, Gaussian rejection sampling,
and actual low-confidence forget-probe selection. Real forget examples are used
only after probe construction, for measurement.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "tiny_imagenet": 200}
DEFAULT_LRS = {
    "bad_teacher": 1e-3,
    "boundary_shrink": 1e-8,
    "delete": 1e-3,
    "finetune": 2e-2,
    "gradient_ascent": 5e-5,
    "l2ul_adv": 1e-5,
    "neggrad_plus": 5e-1,
    "random_label": 1e-7,
    "salun": 1e-3,
    "scrub": 1e-3,
    "retrained": 0.0,
}
METHOD_LATEX_LABELS = {
    "retrained": r"Retrained",
    "random_label": r"Random Label \cite{hayase2020selective}",
    "finetune": r"Finetune \cite{golatkar2020eternal}",
    "gradient_ascent": r"Negative Gradient \cite{golatkar2020eternal}",
    "neggrad_plus": r"Negative Gradient+ \cite{kurmanji2023towards}",
    "boundary_shrink": r"Boundary Shrink \cite{chen2023boundary}",
    "l2ul_adv": r"Learn to Unlearn \cite{cha2024learning}",
    "scrub": r"SCRUB \cite{kurmanji2023towards}",
    "bad_teacher": r"Bad Teacher \cite{chundawat2023can}",
    "salun": r"SalUn \cite{fan2023salun}",
    "delete": r"DELETE \cite{zhou2025decoupled}",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure alignment of synthetic probes with real forget features."
    )
    parser.add_argument("--dataset", default="cifar10", choices=sorted(NUM_CLASSES))
    parser.add_argument("--model", "--backbone", dest="backbone", default="resnet18")
    parser.add_argument(
        "--methods", nargs="+",
        default=["bad_teacher", "delete", "scrub", "retrained"],
    )
    parser.add_argument("--forget_classes", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--num_candidates", "--N", type=int, default=500000)
    parser.add_argument("--num_selected", "--M", type=int, default=500)
    parser.add_argument("--sample_batch_size", type=int, default=4096)
    parser.add_argument("--real_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--base_dir",
        default="/export/livia/home/vision/Zdehghani/classification/exps",
    )
    parser.add_argument("--results_root", default="results_single_class")
    parser.add_argument(
        "--output_dir",
        default="results_single_class/analysis/alignment_analysis",
    )
    parser.add_argument(
        "--append", action="store_true",
        help=(
            "Merge new method/class/seed rows into an existing "
            "alignment_per_class_seed.csv instead of overwriting it."
        ),
    )
    parser.add_argument(
        "--method_lr", action="append", default=[], metavar="METHOD=LR",
        help="Override a checkpoint LR; may be repeated.",
    )
    parser.add_argument(
        "--skip_retain_pass", action="store_true",
        help=(
            "Faster diagnostic mode. By default a discarded retain-pool pass is "
            "performed first to mirror the existing single-class generator."
        ),
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def method_lrs(overrides: list[str]) -> dict[str, float]:
    values = dict(DEFAULT_LRS)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Expected METHOD=LR, got {item!r}")
        method, value = item.split("=", 1)
        values[method] = float(value)
    return values


def checkpoint_for(
    method: str, dataset: str, backbone: str, forget_class: int,
    lr: float, base_dir: Path,
) -> Path:
    if method == "retrained":
        return base_dir / "test_pretrained_model" / (
            f"{dataset}_{backbone}_retrain_forgetcls{forget_class}_model.pth"
        )
    if method == "original":
        return base_dir / "test_pretrained_model" / (
            f"{dataset}_{backbone}_original_model.pth"
        )
    return base_dir / f"{dataset}_{backbone}_forgetcls{forget_class}" / method / (
        f"lr{lr:g}"
    ) / "ckpt_best_by_aus.pth"


def load_checkpoint(
    path: Path, backbone: str, dataset: str, num_classes: int,
    device: torch.device,
) -> nn.Module:
    """Device-aware equivalent of the project's model loader for analysis only."""
    from models import get_model

    checkpoint = torch.load(str(path), map_location=device)
    if isinstance(checkpoint, dict):
        model = get_model(backbone, dataset, num_classes, use_pretrained=False)
        state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
        model.load_state_dict(state_dict)
    else:
        model = checkpoint
    return model.to(device)


def final_linear(model: nn.Module, num_classes: int) -> nn.Linear:
    for _, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear) and module.out_features == num_classes:
            return module
    raise RuntimeError("Could not locate the classifier head.")


def feature_extractor(model: nn.Module, num_classes: int, device: torch.device) -> nn.Module:
    extractor = copy.deepcopy(model).eval().to(device)
    for name, module in reversed(list(extractor.named_modules())):
        if isinstance(module, nn.Linear) and module.out_features == num_classes:
            parts = name.split(".")
            parent = extractor
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], nn.Identity())
            return extractor
    raise RuntimeError("Could not replace the classifier head with Identity.")


@torch.inference_mode()
def real_forget_mean(
    model: nn.Module, testset, forget_class: int, num_classes: int,
    device: torch.device, batch_size: int, num_workers: int,
) -> tuple[torch.Tensor, int, int]:
    targets = torch.as_tensor(testset.targets)
    indices = torch.where(targets.eq(forget_class))[0].tolist()
    if not indices:
        raise RuntimeError(f"No test examples found for forget class {forget_class}.")
    loader = DataLoader(
        Subset(testset, indices), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    extractor = feature_extractor(model, num_classes, device)
    feature_sum = None
    count = 0
    observed_labels: set[int] = set()
    for images, labels in loader:
        observed_labels.update(int(v) for v in labels.unique().tolist())
        features = extractor(images.to(device, non_blocking=True)).detach().double().cpu()
        feature_sum = features.sum(0) if feature_sum is None else feature_sum + features.sum(0)
        count += len(features)
    if observed_labels != {forget_class}:
        raise AssertionError(f"Real forget loader contains labels {observed_labels}.")
    mean = feature_sum / count
    return mean, count, int(mean.numel())


@torch.inference_mode()
def accepted_gaussian_pool(
    weight: torch.Tensor, bias: torch.Tensor, target_class: int,
    wanted: int, batch_size: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Exact N(0,I) and predicted-as-class acceptance used by relearning."""
    feature_chunks, confidence_chunks = [], []
    accepted = draws = 0
    while accepted < wanted:
        features = torch.randn(batch_size, weight.shape[1], device=device)
        probabilities = F.softmax(features @ weight.T + bias, dim=1)
        mask = probabilities.argmax(1).eq(target_class)
        if mask.any():
            feature_chunks.append(features[mask].cpu())
            confidence_chunks.append(probabilities[mask, target_class].cpu())
            accepted += int(mask.sum())
        draws += batch_size
    return (
        torch.cat(feature_chunks)[:wanted],
        torch.cat(confidence_chunks)[:wanted],
        draws,
    )


@torch.inference_mode()
def synthetic_means(
    model: nn.Module, num_classes: int, forget_class: int, n: int, m: int,
    batch_size: int, seed: int, device: torch.device, emulate_retain_pass: bool,
) -> tuple[torch.Tensor, int, int]:
    if not 0 < m <= n:
        raise ValueError(f"Require 0 < M <= N; received M={m}, N={n}.")
    head = final_linear(model, num_classes)
    weight = head.weight.detach().to(device)
    bias = (
        head.bias.detach().to(device) if head.bias is not None
        else torch.zeros(num_classes, device=device)
    )
    synthetic_sum = torch.zeros(weight.shape[1], dtype=torch.float64)
    total_draws = 0
    retain_classes = [value for value in range(num_classes) if value != forget_class]

    # The original pipeline generates its independent high-confidence retain pools
    # before generating the forget pools. Discarding this pass preserves its random
    # number consumption without using any real data.
    if emulate_retain_pass:
        for retain_class in retain_classes:
            _, _, draws = accepted_gaussian_pool(
                weight, bias, retain_class, n, batch_size, device
            )
            total_draws += draws

    for retain_class in retain_classes:
        features, confidence, draws = accepted_gaussian_pool(
            weight, bias, retain_class, n, batch_size, device
        )
        total_draws += draws
        # This is exactly the current relearning rule: lowest confidence among
        # embeddings accepted as the retain class.
        selected = confidence.argsort(descending=False)[:m]
        synthetic_sum += features.index_select(0, selected).double().sum(0)
    count = len(retain_classes) * m
    return synthetic_sum / count, count, total_draws


def standardized_revival_path(
    root: Path, method: str, dataset: str, backbone: str,
) -> Path:
    return root / method / (
        f"z_standardized_revival_selected_{method}_{dataset}_{backbone}.csv"
    )


def load_relearning_metrics(
    root: Path, methods: list[str], dataset: str, backbone: str,
) -> pd.DataFrame:
    frames = []
    for method in methods:
        path = standardized_revival_path(root, method, dataset, backbone)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing standardized revival results: {path}. Run the existing "
                "single-class table standardization first."
            )
        frame = pd.read_csv(path)
        frame["method"] = method
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["forget_class"] = pd.to_numeric(data["forget_class"], errors="raise").astype(int)
    for column in ["RS2", "test_forget_acc", "test_retain_acc"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.drop_duplicates(["method", "forget_class"], keep="last")


def safe_correlations(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    valid = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(valid)
    if n < 3 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return {"pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_rho": np.nan, "spearman_p": np.nan, "n": n}
    pearson = pearsonr(valid["x"], valid["y"])
    spearman = spearmanr(valid["x"], valid["y"])
    return {"pearson_r": pearson.statistic, "pearson_p": pearson.pvalue,
            "spearman_rho": spearman.statistic, "spearman_p": spearman.pvalue,
            "n": n}


def aggregate_results(data: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "alignment_inner_product", "alignment_cosine",
        "delta_alignment_inner_product", "delta_alignment_cosine", "RS", "delta_RS",
    ]
    grouped = data.groupby("method", sort=False)
    rows = []
    for method, frame in grouped:
        row = {"method": method, "num_forget_classes": frame["forget_class"].nunique(),
               "num_seeds": frame["seed"].nunique(), "num_rows": len(frame)}
        # Overall variation, plus the two requested sources of variation.
        for metric in metrics:
            values = pd.to_numeric(frame[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std_all"] = values.std(ddof=1)
            per_class = frame.groupby("forget_class")[metric].mean()
            per_seed = frame.groupby("seed")[metric].mean()
            row[f"{metric}_std_across_classes"] = per_class.std(ddof=1)
            row[f"{metric}_std_across_seeds"] = per_seed.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_results(data: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("alignment_cosine", "RS"),
        ("alignment_inner_product", "RS"),
        ("delta_alignment_cosine", "delta_RS"),
        ("delta_alignment_inner_product", "delta_RS"),
    ]
    rows = []
    non_retrained = data[data["method"] != "retrained"]
    for x_name, y_name in pairs:
        rows.append({"x": x_name, "y": y_name,
                     **safe_correlations(non_retrained[x_name], non_retrained[y_name])})
    return pd.DataFrame(rows)


def scatter_plot(data: pd.DataFrame, x_name: str, y_name: str, path: Path) -> None:
    frame = data[data["method"] != "retrained"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for method, method_frame in frame.groupby("method"):
        ax.scatter(method_frame[x_name], method_frame[y_name], label=method.replace("_", " ").title())
    ax.set_xlabel(x_name.replace("_", " ").title())
    ax.set_ylabel(y_name.replace("_", " ").title())
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def latex_table(aggregate: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\color{red}",
        r"\caption{Synthetic--real alignment and relearning score on CIFAR-10 with ResNet-18, averaged across forget classes.}",
        r"\label{tab:synthetic_real_alignment}",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"Unlearning Method & RS & $\Delta$RS & $A_{\mathrm{IP}}$ & $A_{\mathrm{cos}}$ \\",
        r"\midrule",
    ]
    for _, row in aggregate.iterrows():
        def cell(metric: str, signed: bool = False) -> str:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std_across_classes"]
            if pd.isna(mean):
                return "--"
            mean_text = f"{mean:+.2f}" if signed else f"{mean:.2f}"
            if pd.isna(std):
                return rf"${mean_text}$"
            return rf"${mean_text}{{\scriptstyle\,\pm {std:.2f}}}$"
        method = str(row["method"])
        method_label = METHOD_LATEX_LABELS.get(
            method, method.replace("_", " ").title()
        )
        lines.append(
            f"{method_label} & "
            f"{cell('RS')} & "
            f"{cell('delta_RS', signed=True)} & "
            f"{cell('alignment_inner_product', signed=True)} & "
            f"{cell('alignment_cosine', signed=True)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    from utils import get_dataset, get_transforms

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    num_classes = NUM_CLASSES[args.dataset]
    forget_classes = args.forget_classes or list(range(num_classes))
    invalid = [value for value in forget_classes if not 0 <= value < num_classes]
    if invalid:
        raise ValueError(f"Invalid forget classes: {invalid}")
    methods = list(dict.fromkeys(args.methods))
    if "retrained" not in methods:
        methods.append("retrained")
    lrs = method_lrs(args.method_lr)
    missing_lrs = [method for method in methods if method not in lrs]
    if missing_lrs:
        raise ValueError(f"Provide --method_lr METHOD=LR for {missing_lrs}")

    metrics = load_relearning_metrics(Path(args.results_root), methods, args.dataset, args.backbone)
    _, transform_test = get_transforms(args.dataset, args.backbone, wo_dataaug=False)
    _, testset = get_dataset(args.dataset, transform_test, transform_test)
    rows = []
    base_dir = Path(args.base_dir)

    for method in methods:
        for seed in args.seeds:
            # The existing single-class program seeds once, then processes forget
            # classes in order. Preserve that continuous RNG progression here.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            for forget_class in forget_classes:
                checkpoint = checkpoint_for(
                    method, args.dataset, args.backbone, forget_class, lrs[method], base_dir
                )
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
                metric_rows = metrics[
                    (metrics["method"] == method)
                    & (metrics["forget_class"] == forget_class)
                ]
                if metric_rows.empty:
                    raise RuntimeError(
                        f"No relearning result for {method}, class {forget_class}"
                    )
                metric_row = metric_rows.iloc[-1]
                model = load_checkpoint(
                    checkpoint, args.backbone, args.dataset, num_classes, device
                ).eval()
                real_mean, real_count, feature_dim = real_forget_mean(
                    model, testset, forget_class, num_classes, device,
                    args.real_batch_size, args.num_workers,
                )
                synthetic_mean, synthetic_count, total_draws = synthetic_means(
                    model, num_classes, forget_class, args.num_candidates,
                    args.num_selected, args.sample_batch_size, seed, device,
                    emulate_retain_pass=not args.skip_retain_pass,
                )
                real_norm = float(torch.linalg.vector_norm(real_mean))
                if synthetic_mean.numel() != feature_dim:
                    raise AssertionError("Real and synthetic feature dimensions differ.")
                synthetic_norm = float(torch.linalg.vector_norm(synthetic_mean))
                inner = float(torch.dot(synthetic_mean, real_mean))
                cosine = inner / max(real_norm * synthetic_norm, 1e-12)
                rows.append({
                    "dataset": args.dataset, "backbone": args.backbone,
                    "method": method, "forget_class": forget_class, "seed": seed,
                    "num_candidates_N": args.num_candidates,
                    "num_selected_M": args.num_selected,
                    "num_real_forget_embeddings": real_count,
                    "num_synthetic_forget_embeddings": synthetic_count,
                    "real_forget_mean_norm": real_norm,
                    "synthetic_forget_mean_norm": synthetic_norm,
                    "alignment_inner_product": inner,
                    "alignment_cosine": cosine,
                    "forget_accuracy_relearned": metric_row["test_forget_acc"],
                    "retain_accuracy_relearned": metric_row["test_retain_acc"],
                    "RS": metric_row["RS2"],
                    "feature_dim": feature_dim, "gaussian_draws": total_draws,
                    "checkpoint": str(checkpoint),
                })
                print(
                    f"[alignment] method={method} class={forget_class} seed={seed} "
                    f"N={args.num_candidates} M={args.num_selected} "
                    f"real={real_count} synthetic={synthetic_count} dim={feature_dim} "
                    f"checkpoint={checkpoint}"
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    per_class = pd.DataFrame(rows)
    reference = per_class[per_class["method"] == "retrained"][
        ["forget_class", "seed", "alignment_inner_product", "alignment_cosine", "RS"]
    ].rename(columns={
        "alignment_inner_product": "retrained_alignment_inner_product",
        "alignment_cosine": "retrained_alignment_cosine",
        "RS": "retrained_RS",
    })
    per_class = per_class.merge(reference, on=["forget_class", "seed"], how="left")
    per_class["delta_alignment_inner_product"] = (
        per_class["alignment_inner_product"]
        - per_class["retrained_alignment_inner_product"]
    )
    per_class["delta_alignment_cosine"] = (
        per_class["alignment_cosine"] - per_class["retrained_alignment_cosine"]
    )
    per_class["delta_RS"] = per_class["RS"] - per_class["retrained_RS"]
    retrained_mask = per_class["method"].eq("retrained")
    per_class.loc[
        retrained_mask,
        ["delta_alignment_inner_product", "delta_alignment_cosine", "delta_RS"],
    ] = 0.0
    required_columns = [
        "dataset", "backbone", "method", "forget_class", "seed",
        "num_candidates_N", "num_selected_M", "num_real_forget_embeddings",
        "num_synthetic_forget_embeddings", "real_forget_mean_norm",
        "synthetic_forget_mean_norm", "alignment_inner_product",
        "alignment_cosine", "retrained_alignment_inner_product",
        "retrained_alignment_cosine", "delta_alignment_inner_product",
        "delta_alignment_cosine", "forget_accuracy_relearned",
        "retain_accuracy_relearned", "RS", "retrained_RS", "delta_RS",
    ]
    diagnostic_columns = [
        column for column in per_class.columns if column not in required_columns
    ]
    per_class = per_class[required_columns + diagnostic_columns]
    per_class_path = output_dir / "alignment_per_class_seed.csv"
    if args.append and per_class_path.is_file():
        existing = pd.read_csv(per_class_path)
        key = ["dataset", "backbone", "method", "forget_class", "seed"]
        replacement_keys = per_class[key].drop_duplicates()
        existing = existing.merge(
            replacement_keys.assign(_replace=True), on=key, how="left"
        )
        existing = existing[existing["_replace"].isna()].drop(columns="_replace")
        per_class = pd.concat([existing, per_class], ignore_index=True, sort=False)
        per_class = per_class.sort_values(
            ["method", "forget_class", "seed"]
        ).reset_index(drop=True)
    per_class.to_csv(per_class_path, index=False)

    aggregate = aggregate_results(per_class)
    aggregate.to_csv(output_dir / "alignment_aggregated.csv", index=False)
    correlations = correlation_results(per_class)
    correlations.to_csv(output_dir / "alignment_correlations.csv", index=False)
    latex_table(aggregate, output_dir / "alignment_summary_table.tex")
    scatter_plot(per_class, "alignment_cosine", "RS", output_dir / "alignment_vs_rs.png")
    scatter_plot(
        per_class, "delta_alignment_cosine", "delta_RS",
        output_dir / "delta_alignment_vs_delta_rs.png",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n"
    )
    print(f"[saved] {output_dir.resolve()}")


if __name__ == "__main__":
    main()
