"""Generate original-training and unlearning hyperparameter tables from configs."""

from __future__ import annotations

import argparse
import re
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import EXPS_DIR  # noqa: E402


DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "tiny_imagenet": "TinyImageNet",
}
MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "swin-t": "Swin-T",
    "vit-b-16": "ViT-B/16",
}
METHODS = [
    "random_label", "finetune", "gradient_ascent", "neggrad_plus",
    "boundary_shrink", "l2ul_adv", "scrub", "bad_teacher", "salun",
    "delete",
]
METHOD_LABELS = {
    "random_label": r"Random Label",
    "finetune": r"Finetune",
    "gradient_ascent": r"Negative Gradient",
    "neggrad_plus": r"Negative Gradient+",
    "boundary_shrink": r"Boundary Shrink",
    "l2ul_adv": r"Learn to Unlearn",
    "scrub": r"SCRUB",
    "bad_teacher": r"Bad Teacher",
    "salun": r"SalUn",
    "delete": r"DELETE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exps-dir",
        default=str(EXPS_DIR),
    )
    parser.add_argument("--output-dir", default="hyperparameters")
    parser.add_argument("--unlearning-model", default="resnet18")
    parser.add_argument(
        "--lr-source-dir", default="results/hyperparameters",
        help="Directory containing the existing architecture hyperparameter tables.",
    )
    return parser.parse_args()


def lr_values_from_tables(source_dir: Path) -> dict[tuple[str, str, str], set[str]]:
    """Read the LR cells from the existing architecture tables."""
    reverse_method = {value: key for key, value in METHOD_LABELS.items()}
    selected: dict[tuple[str, str, str], set[str]] = {}
    for model in MODEL_LABELS:
        path = source_dir / f"training_and_unlearning_hyperparameters_{model}.tex"
        if not path.is_file():
            continue
        current_dataset = None
        for line in path.read_text().splitlines():
            if " & " not in line or not line.rstrip().endswith(r"\\"):
                continue
            cells = [part.strip() for part in line.rsplit(r"\\", 1)[0].split(" & ")]
            if len(cells) not in {6, 7}:
                continue
            if len(cells) == 7:
                _, dataset_name, method_label, _, _, _, lr_cell = cells
            else:
                _, dataset_name, method_label, _, _, lr_cell = cells
            dataset_match = re.search(
                r"\{(CIFAR-10|CIFAR-100|TinyImageNet)\}", dataset_name
            )
            if dataset_match:
                dataset_name = dataset_match.group(1)
                current_dataset = dataset_name
            elif not dataset_name:
                dataset_name = current_dataset
            dataset = next(
                (key for key, value in DATASET_LABELS.items() if value == dataset_name),
                None,
            )
            method = reverse_method.get(method_label)
            if dataset is None or method is None:
                continue
            selected[(dataset, model, method)] = {
                value.strip() for value in lr_cell.split(",")
            }
    return selected


def experiment_identity(path: Path, root: Path):
    relative = path.relative_to(root)
    if len(relative.parts) < 4:
        return None
    experiment, method, lr_dir = relative.parts[:3]
    dataset = next(
        (name for name in DATASET_LABELS if experiment.startswith(name + "_")),
        None,
    )
    if dataset is None or method not in METHODS or not lr_dir.startswith("lr"):
        return None
    match = re.match(rf"^{re.escape(dataset)}_(.+)_forget(cls\d+|\d+)$", experiment)
    if not match:
        return None
    model, forget_tag = match.groups()
    forget_count = 1 if forget_tag.startswith("cls") else int(forget_tag)
    return dataset, model, method, forget_count


def compact(values) -> str:
    unique = sorted({str(value) for value in values if value is not None})
    return ", ".join(unique) if unique else "--"


def numeric_compact(values) -> str:
    cleaned = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        cleaned.append(f"{number:g}")
    return compact(cleaned)


def best_epoch_from_log(config_path: Path):
    """Return the epoch with the highest logged AUS for this run."""
    log_path = config_path.parent / "train_log.log"
    if not log_path.is_file():
        return None
    best = None
    pattern = re.compile(r"\[epoch\s+(\d+)\].*?AUS=([0-9.eE+-]+)")
    for line in log_path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        epoch, aus = int(match.group(1)), float(match.group(2))
        if best is None or aus > best[0]:
            best = (aus, epoch)
    return best[1] if best is not None else None


def mean_std_epoch(values) -> str:
    values = [float(v) for v in values if v is not None]
    if not values:
        return "--"
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.1f} $\\pm$ {std:.1f}"


def collect(root: Path, selected_lrs=None) -> pd.DataFrame:
    rows = []
    for config_path in root.glob("*_forget*/*/lr*/config.yaml"):
        if not (config_path.parent / "ckpt_best_by_aus.pth").is_file():
            continue
        identity = experiment_identity(config_path, root)
        if identity is None:
            continue
        dataset, model, method, forget_count = identity
        if selected_lrs is not None:
            allowed = selected_lrs.get((dataset, model, method))
            if not allowed:
                continue
            actual_lr = config_path.parent.name[2:]
            if actual_lr not in allowed:
                continue
        with config_path.open() as stream:
            config = yaml.safe_load(stream) or {}
        rows.append({
            "dataset": dataset,
            "model": model,
            "method": method,
            "forget_count": forget_count,
            "batch_size": config.get("batch_size"),
            "pretrain_epochs": config.get("pretrain_epoch"),
            "pretrain_lr": config.get("pretrain_lr"),
            "unlearn_epochs": config.get("unlearn_epoch"),
            "unlearn_lr": config.get("unlearn_rate", config_path.parent.name[2:]),
            "best_epoch": best_epoch_from_log(config_path),
            "config": str(config_path),
        })
    if not rows:
        raise RuntimeError(f"No completed checkpoint configurations found in {root}")
    return pd.DataFrame(rows)


def original_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    allowed_models = set(MODEL_LABELS)
    frame = data[data["model"].isin(allowed_models)]
    for (dataset, model), group in frame.groupby(["dataset", "model"]):
        # These values should be invariant; list all observed values if not.
        values = {
            "dataset": dataset,
            "model": model,
            "num_forget_classes": "--",
            "epochs": compact(group["pretrain_epochs"]),
            "best_epoch": "--",
            "batch_size": compact(group["batch_size"]),
            "learning_rate": numeric_compact(group["pretrain_lr"]),
        }
        rows.append({**values, "method": "Original"})
        rows.append({**values, "method": "Retrained"})
    return pd.DataFrame(rows).sort_values(["dataset", "model"])


def unlearning_summary(data: pd.DataFrame, model: str) -> pd.DataFrame:
    frame = data[data["model"].eq(model)].copy()
    if model in {"swin-t", "vit-b-16"}:
        frame = frame[frame["method"] != "boundary_shrink"]
    rank = {name: index for index, name in enumerate(METHODS)}
    rows = []
    for (dataset, method), group in frame.groupby(["dataset", "method"]):
        rows.append({
            "dataset": dataset,
            "model": model,
            "method": method,
            "num_forget_classes": compact(group["forget_count"]),
            "epochs": compact(group["unlearn_epochs"]),
            "best_epoch": mean_std_epoch(group["best_epoch"]),
            "batch_size": compact(group["batch_size"]),
            "learning_rate": numeric_compact(group["unlearn_lr"]),
            "_method_rank": rank.get(method, len(rank)),
        })
    return (
        pd.DataFrame(rows)
        .sort_values(["dataset", "_method_rank"])
        .drop(columns="_method_rank")
    )


def latex_rows(frame: pd.DataFrame):
    model_total = len(frame)
    dataset_counts = frame.groupby("dataset", sort=False).size().to_dict()
    previous_dataset = None
    model_written = False
    dataset_written = False
    for index, (_, row) in enumerate(frame.iterrows()):
        dataset = row["dataset"]
        first_dataset = dataset != previous_dataset
        if first_dataset:
            dataset_written = False
        model_cell = (
            rf"\multirow{{{model_total}}}{{*}}{{{MODEL_LABELS.get(row['model'], row['model'])}}}"
            if not model_written else ""
        )
        dataset_cell = (
            rf"\multirow{{{dataset_counts[dataset]}}}{{*}}{{{DATASET_LABELS.get(dataset, dataset)}}}"
            if not dataset_written else ""
        )
        cells = [
            model_cell,
            dataset_cell,
            METHOD_LABELS.get(row["method"], row["method"]),
            str(row["best_epoch"]),
            str(row["batch_size"]),
            str(row["learning_rate"]),
        ]
        line = " & ".join(cells) + r" \\"
        last_dataset_row = index == len(frame) - 1 or frame.iloc[index + 1]["dataset"] != dataset
        if last_dataset_row and index != len(frame) - 1:
            line += "\n" + r"\cline{2-6}"
        yield line
        model_written = True
        dataset_written = True
        previous_dataset = dataset


def write_table(frame: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l|l|l|ccc}", r"\toprule",
        r"Model & Dataset & Method & \multicolumn{3}{c}{Architecture} \\",
        r"\cmidrule(lr){4-6}",
        r" & & & Best Epoch & Batch & LR \\",
        r"\midrule",
        *latex_rows(frame),
        r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_longtable(
    frame: pd.DataFrame, path: Path, caption: str, label: str,
) -> None:
    header = r"Model & Dataset & Method & \multicolumn{3}{c}{Architecture} \\"
    lines = [
        r"\begingroup", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{longtable}{l|l|l|ccc}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}} \\",
        r"\toprule", header, r"\cmidrule(lr){4-6}",
        r" & & & Best Epoch & Batch & LR \\", r"\midrule", r"\endfirsthead",
        r"\multicolumn{6}{c}{\tablename\ \thetable{} -- continued from previous page} \\",
        r"\toprule", header, r"\cmidrule(lr){4-6}",
        r" & & & Best Epoch & Batch & LR \\", r"\midrule", r"\endhead",
        r"\midrule", r"\multicolumn{6}{r}{Continued on next page} \\",
        r"\endfoot", r"\bottomrule", r"\endlastfoot",
        *latex_rows(frame),
        r"\end{longtable}", r"\endgroup",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_wide_architecture_table(data: pd.DataFrame, path: Path) -> None:
    """Write one table with the backbone as rows and datasets as column groups."""
    frames = {m: pd.concat([original_summary(data[data.model.eq(m)]),
                            unlearning_summary(data, m)], ignore_index=True)
              for m in MODEL_LABELS}
    methods = [
        "Original", "Retrained",
        *[method for method in METHODS if method != "boundary_shrink"],
    ]
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l|l|ccc|ccc|ccc}",
        r"\caption{Training and unlearning hyperparameters across architectures.}",
        r"\label{tab:training_and_unlearning_hyperparameters_wide}",
        r"\toprule",
        r"\multicolumn{1}{c|}{Backbone} & \multicolumn{1}{c|}{Method} & \multicolumn{3}{c|}{CIFAR-10} & \multicolumn{3}{c|}{CIFAR-100} & \multicolumn{3}{c}{TinyImageNet} \\",
        r" & & Best Epoch & Batch & LR & Best Epoch & Batch & LR & Best Epoch & Batch & LR \\",
        r"\midrule",
    ]
    model_names = list(MODEL_LABELS)
    for model_index, model in enumerate(model_names):
        for method_index, method in enumerate(methods):
            backbone = (
                rf"\multirow[c]{{{len(methods)}}}{{*}}{{{MODEL_LABELS[model]}}}"
                if method_index == 0 else ""
            )
            cells = [backbone, METHOD_LABELS.get(method, method)]
            for dataset in DATASET_LABELS:
                row = frames[model][(frames[model].dataset == dataset) & (frames[model].method == method)]
                if row.empty:
                    cells += ["--", "--", "--"]
                else:
                    r = row.iloc[0]
                    cells += [str(r.best_epoch), str(r.batch_size), str(r.learning_rate)]
            lines.append(" & ".join(cells) + r" \\")
        if model_index < len(model_names) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.lr_source_dir)
    selected_lrs = lr_values_from_tables(source_dir)
    if not selected_lrs:
        raise RuntimeError(
            "No learning rates were found in the source architecture tables. "
            "Restore/populate those tables before regenerating."
        )
    data = collect(Path(args.exps_dir), selected_lrs=selected_lrs)
    original = original_summary(data)
    unlearning = unlearning_summary(data, args.unlearning_model)
    combined = pd.concat([original, unlearning], ignore_index=True)
    method_rank = {name: index for index, name in enumerate(METHODS, start=2)}
    method_rank.update({"Original": 0, "Retrained": 1})
    combined["_method_rank"] = combined["method"].map(method_rank).fillna(999)
    combined = combined.sort_values(["model", "dataset", "_method_rank"]).drop(
        columns="_method_rank"
    ).reset_index(drop=True)
    original.to_csv(output / "original_model_hyperparameters.csv", index=False)
    unlearning.to_csv(output / "unlearning_hyperparameters_resnet18.csv", index=False)
    combined.to_csv(
        output / "training_and_unlearning_hyperparameters.csv", index=False
    )
    write_table(
        original,
        output / "original_model_hyperparameters.tex",
        "Original-model training hyperparameters. Values are read from the configurations associated with completed unlearning checkpoints.",
        "tab:original_model_hyperparameters",
    )
    write_table(
        unlearning,
        output / "unlearning_hyperparameters_resnet18.tex",
        "Unlearning hyperparameters for ResNet-18. Multiple learning rates indicate completed checkpoint sweeps for the same setting.",
        "tab:unlearning_hyperparameters_resnet18",
    )
    for dataset, dataset_frame in unlearning.groupby("dataset", sort=False):
        dataset_label = DATASET_LABELS.get(dataset, dataset)
        write_table(
            dataset_frame,
            output / f"unlearning_hyperparameters_resnet18_{dataset}.tex",
            f"Unlearning hyperparameters for ResNet-18 on {dataset_label}. "
            "Multiple learning rates indicate completed checkpoint sweeps for the same setting.",
            f"tab:unlearning_hyperparameters_resnet18_{dataset}",
        )
    write_longtable(
        combined,
        output / "training_and_unlearning_hyperparameters.tex",
        "Training and unlearning hyperparameters. Multiple learning rates indicate completed checkpoint sweeps for the same setting.",
        "tab:training_and_unlearning_hyperparameters",
    )
    for model in MODEL_LABELS:
        model_original = original[original["model"].eq(model)]
        model_unlearning = unlearning_summary(data, model)
        model_frame = pd.concat(
            [model_original, model_unlearning], ignore_index=True
        )
        model_frame["_method_rank"] = model_frame["method"].map(method_rank).fillna(999)
        model_frame = model_frame.sort_values(
            ["dataset", "_method_rank"]
        ).drop(columns="_method_rank").reset_index(drop=True)
        if model_frame.empty:
            continue
        model_label = MODEL_LABELS[model]
        write_longtable(
            model_frame,
            output / f"training_and_unlearning_hyperparameters_{model}.tex",
            f"Training and unlearning hyperparameters for {model_label}. "
            "Multiple learning rates indicate completed checkpoint sweeps for the same setting.",
            f"tab:training_and_unlearning_hyperparameters_{model.replace('-', '_')}",
        )
    write_wide_architecture_table(
        data, output / "training_and_unlearning_hyperparameters_wide.tex"
    )
    print(f"[saved] {output.resolve()}")


if __name__ == "__main__":
    main()
