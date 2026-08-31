"""Create the Appendix N RS--DeltaRS recoverability diagnostic map."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

from method_colors import get_method_color


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results_single_class/plots/rs_recoverability_outputs"
RS_GUIDE = 0.5
DATASETS = {
    "CIFAR-10": "cifar10",
    "CIFAR-100": "cifar100",
    "TinyImageNet": "tiny_imagenet",
}
ARCHITECTURES = ("resnet18", "vit-b-16", "swin-t")
METHODS = (
    "Finetune",
    "Negative Gradient",
    "Negative Gradient+",
    "Random Label",
    "Boundary Shrink",
    "Learn to Unlearn",
    "SCRUB",
    "Bad Teacher",
    "SalUn",
    "DELETE",
)
METHOD_KEYS = {
    "Retrained": "retrained",
    "Finetune": "finetune",
    "Negative Gradient": "gradient_ascent",
    "Negative Gradient+": "neggrad_plus",
    "Random Label": "random_label",
    "Boundary Shrink": "boundary_shrink",
    "Learn to Unlearn": "l2ul_adv",
    "SCRUB": "scrub",
    "Bad Teacher": "bad_teacher",
    "SalUn": "salun",
    "DELETE": "delete",
}
METHOD_COLORS = {
    method: get_method_color(key) for method, key in METHOD_KEYS.items()
}
METHOD_MARKERS = {
    "Retrained": "P",
    "Finetune": "v",
    "Negative Gradient": "^",
    "Negative Gradient+": "X",
    "Random Label": "D",
    "Boundary Shrink": "s",
    "Learn to Unlearn": "h",
    "SCRUB": "o",
    "Bad Teacher": "o",
    "SalUn": "*",
    "DELETE": "s",
}


def _plain_method_name(cell: str) -> str | None:
    """Extract the displayed method name from a LaTeX multirow cell."""
    match = re.search(r"\\multirow\{\d+\}\{\*\}\{(.+)\}", cell.strip())
    if match is None:
        return None
    value = re.sub(r"\s*\\cite\{[^}]+\}", "", match.group(1)).strip()
    return value if value in METHOD_KEYS else None


def _numeric_cell(cell: str) -> float:
    if cell.strip() in {"-", "--", ""}:
        return np.nan
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cell)
    return float(match.group(0)) if match else np.nan


def parse_appendix_table(path: Path, dataset: str) -> pd.DataFrame:
    """Parse per-class SFRA RS and DeltaRS rows from a generated table."""
    if not path.is_file():
        raise FileNotFoundError(f"Appendix table not found: {path}")
    records: dict[tuple[str, int], dict[str, object]] = {}
    classes: list[int] = []
    current_method: str | None = None
    current_metric: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("& & &"):
            classes = [int(value) for value in re.findall(r"(?<![.\d])\d+(?![.\d])", line)]
            continue
        if " & " not in line or not line.endswith(r"\\"):
            continue
        cells = re.split(r"\s*&\s*", line[:-2].strip())
        if len(cells) < 3:
            continue

        method = _plain_method_name(cells[0])
        if method is not None:
            current_method = method

        metric_cell = cells[1]
        if r"\Delta" in metric_cell and "RS" in metric_cell:
            current_metric = "SFRA Delta RS"
        elif "RS" in metric_cell:
            current_metric = "SFRA RS"

        if (
            current_method is None
            or current_metric not in {"SFRA RS", "SFRA Delta RS"}
            or "SFRA (ours)" not in cells[2]
            or not classes
        ):
            continue

        values = cells[3:3 + len(classes)]
        if len(values) != len(classes):
            raise ValueError(
                f"Expected {len(classes)} values in {path}, got {len(values)}: {line}"
            )
        for forget_class, value in zip(classes, values):
            key = (current_method, forget_class)
            record = records.setdefault(
                key,
                {
                    "Dataset": dataset,
                    "Method": current_method,
                    "Forget Class": forget_class,
                    "SFRA RS": np.nan,
                    "SFRA Delta RS": np.nan,
                },
            )
            record[current_metric] = _numeric_cell(value)

    if not records:
        raise ValueError(f"No SFRA RS/DeltaRS rows could be parsed from {path}")
    return pd.DataFrame(records.values())


def validate_extracted_data(
    frame: pd.DataFrame,
    dataset: str,
    require_all_methods: bool = False,
    require_linear_probe: bool = False,
) -> None:
    """Validate the fields used by this diagnostic map."""
    del require_linear_probe
    required = {"Dataset", "Method", "Forget Class", "SFRA RS", "SFRA Delta RS"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{dataset}: missing parsed columns {sorted(missing)}")
    if frame.duplicated(["Method", "Forget Class"]).any():
        raise ValueError(f"{dataset}: duplicate method/forget-class rows were parsed")
    if require_all_methods:
        absent = set(METHODS) - set(frame["Method"].astype(str))
        if absent:
            raise ValueError(f"{dataset}: missing methods {sorted(absent)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=ARCHITECTURES,
        default=list(ARCHITECTURES),
        help="Backbones to plot (default: all available backbones).",
    )
    parser.add_argument(
        "--rs-guide",
        type=float,
        default=RS_GUIDE,
        help="Visual guide only; this is not treated as a decision threshold.",
    )
    parser.add_argument(
        "--annotate-regions",
        action="store_true",
        help="Add compact descriptive labels to the four visual regions.",
    )
    return parser.parse_args()


def table_paths(architecture: str) -> dict[str, Path]:
    return {
        dataset: REPO_ROOT
        / "results_single_class/tables"
        / f"{dataset_key}_{architecture}_single_class_variant_by_forget_table.tex"
        for dataset, dataset_key in DATASETS.items()
    }


def load_data(architecture: str) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    tables = table_paths(architecture)
    for dataset, path in tables.items():
        frame = parse_appendix_table(path, dataset)
        expected_classes = set(frame["Forget Class"].astype(int))
        required = ["SFRA RS"]
        excluded = []
        for method, group in frame.groupby("Method", observed=True):
            method_name = str(method)
            complete_classes = set(group["Forget Class"].astype(int)) == expected_classes
            complete_values = not group[required].isna().any(axis=None)
            complete_delta = method_name == "Retrained" or not group[
                "SFRA Delta RS"
            ].isna().any()
            if not (complete_classes and complete_values and complete_delta):
                excluded.append(method_name)
        if excluded:
            print(
                f"[info] {architecture}/{dataset}: excluded incomplete methods "
                f"{excluded}"
            )
            frame = frame[~frame["Method"].astype(str).isin(excluded)].copy()
        validate_extracted_data(
            frame,
            dataset,
            require_all_methods=False,
            require_linear_probe=False,
        )
        missing_methods = [
            method
            for method in METHODS
            if method not in set(frame["Method"].dropna().astype(str))
        ]
        if missing_methods:
            print(
                f"[info] {architecture}/{dataset}: omitted unavailable methods "
                f"{missing_methods}"
            )
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    # DeltaRS is defined relative to Retrained, so the Retrained control has
    # no displayed DeltaRS entry and is intentionally absent from this map.
    data = data.dropna(subset=["SFRA RS", "SFRA Delta RS"]).copy()
    data["Architecture"] = architecture
    return data, list(tables)


def quadrant_counts(data: pd.DataFrame, guide: float) -> pd.DataFrame:
    rows = []
    for dataset, subset in data.groupby("Dataset", sort=False):
        high_rs = subset["SFRA RS"].ge(guide)
        positive_delta = subset["SFRA Delta RS"].gt(0.0)
        rows.append(
            {
                "Dataset": dataset,
                "Total": len(subset),
                "High RS, positive DeltaRS": int((high_rs & positive_delta).sum()),
                "High RS, nonpositive DeltaRS": int((high_rs & ~positive_delta).sum()),
                "Low RS, positive DeltaRS": int((~high_rs & positive_delta).sum()),
                "Low RS, nonpositive DeltaRS": int((~high_rs & ~positive_delta).sum()),
            }
        )
    return pd.DataFrame(rows)


def add_region_labels(axis: plt.Axes, guide: float, y_min: float, y_max: float) -> None:
    positive_mid = 0.52 * y_max
    negative_mid = 0.52 * y_min
    label_style = {
        "fontsize": 7.4,
        "color": "0.32",
        "ha": "center",
        "va": "center",
        "bbox": {
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.72,
        },
        "zorder": 2,
    }
    axis.text(guide / 2, positive_mid, "Low absolute\npositive excess", **label_style)
    axis.text(
        (guide + 1.0) / 2,
        positive_mid,
        "High absolute &\npositive excess",
        **label_style,
    )
    axis.text(guide / 2, negative_mid, "Low absolute &\nnonpositive excess", **label_style)
    axis.text(
        (guide + 1.0) / 2,
        negative_mid,
        "High absolute\nlimited excess",
        **label_style,
    )


def make_figure(
    data: pd.DataFrame,
    datasets: list[str],
    output_dir: Path,
    dpi: int,
    guide: float,
    annotate_regions: bool,
    architecture: str,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    y_values = data["SFRA Delta RS"]
    y_padding = max(0.04, 0.05 * float(y_values.max() - y_values.min()))
    y_min = min(-0.05, float(y_values.min()) - y_padding)
    y_max = max(0.05, float(y_values.max()) + y_padding)

    fig, axes = plt.subplots(
        1, len(datasets), figsize=(12.6, 3.65), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    plotted_methods = [method for method in METHODS if method != "Retrained"]

    for panel_index, (axis, dataset) in enumerate(zip(axes, datasets)):
        subset = data[data["Dataset"].eq(dataset)]
        for method in plotted_methods:
            method_data = subset[subset["Method"].astype(str).eq(method)]
            if method_data.empty:
                continue
            is_salun = method == "SalUn"
            axis.scatter(
                method_data["SFRA RS"],
                method_data["SFRA Delta RS"],
                s=72 if is_salun else 39,
                marker=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                edgecolors=(
                    "#4b286d"
                    if is_salun
                    else "black" if method == "Negative Gradient+" else "white"
                ),
                linewidths=0.75 if is_salun else 0.45,
                alpha=1.0 if is_salun else 0.86,
                label=method,
                zorder=4 if is_salun else 3,
            )

        axis.axhline(0.0, color="0.28", linestyle="--", linewidth=1.05, zorder=1)
        axis.axvline(guide, color="0.45", linestyle=":", linewidth=1.0, zorder=1)
        axis.set_title(dataset, fontweight="bold")
        axis.set_xlabel("SFRA RS")
        if panel_index == 0:
            axis.set_ylabel(r"SFRA $\Delta$RS")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(y_min, y_max)
        axis.grid(True, color="0.88", linewidth=0.55, alpha=0.72, zorder=0)
        axis.tick_params(direction="out", length=3.2)
        axis.text(
            guide + 0.012,
            y_min + 0.025 * (y_max - y_min),
            "visual guide",
            rotation=90,
            ha="left",
            va="bottom",
            fontsize=7.2,
            color="0.42",
        )
        if annotate_regions:
            add_region_labels(axis, guide, y_min, y_max)

    if architecture == "resnet18":
        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=5,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            facecolor="white",
            edgecolor="0.78",
            columnspacing=1.25,
            handletextpad=0.42,
            markerscale=0.9,
        )
        legend.get_frame().set_linewidth(0.5)
    # Reserve identical space above every architecture so the three dataset
    # panels have exactly the same geometry.  ResNet-18 uses this area for the
    # shared legend; the other figures intentionally leave it blank.
    layout_rect = (0.0, 0.0, 1.0, 0.82)
    fig.tight_layout(rect=layout_rect, w_pad=1.15)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"rs_recoverability_map_{architecture}"
    if architecture == "resnet18":
        # Keep the full canvas because the shared legend occupies the top area.
        fig.savefig(stem.with_suffix(".png"), dpi=dpi)
    else:
        # Remove only the unused top strip.  This explicit crop does not
        # recompute or resize the axes, so panel geometry remains identical to
        # ResNet-18 even though the legend-free canvas is shorter.
        width, _ = fig.get_size_inches()
        crop = Bbox.from_bounds(0.0, 0.0, width, 3.00)
        fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches=crop)
    plt.close(fig)
    print(f"[saved] {stem.with_suffix('.png')}")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.rs_guide < 1.0:
        raise ValueError("--rs-guide must lie strictly between 0 and 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_data = []
    all_counts = []
    for architecture in args.architectures:
        data, datasets = load_data(architecture)
        counts = quadrant_counts(data, args.rs_guide)
        counts.insert(0, "Architecture", architecture)
        all_data.append(data)
        all_counts.append(counts)

        print(f"\nExtracted diagnostic-map counts for {architecture}:")
        for dataset in datasets:
            subset = data[data["Dataset"].eq(dataset)]
            print(
                f"  {dataset}: {len(subset)} points, "
                f"{subset['Method'].nunique()} methods, "
                f"{subset['Forget Class'].nunique()} forget classes"
            )

        make_figure(
            data,
            datasets,
            args.output_dir,
            args.dpi,
            args.rs_guide,
            args.annotate_regions,
            architecture,
        )

    combined_data = pd.concat(all_data, ignore_index=True)
    combined_counts = pd.concat(all_counts, ignore_index=True)
    combined_data.to_csv(
        args.output_dir / "rs_recoverability_map_data_all_architectures.csv",
        index=False,
    )
    combined_counts.to_csv(
        args.output_dir / "rs_recoverability_quadrant_counts_all_architectures.csv",
        index=False,
    )
    print("\nQuadrant counts (RS=0.5 is a visual guide, not a threshold):")
    print(combined_counts.to_string(index=False))


if __name__ == "__main__":
    main()
