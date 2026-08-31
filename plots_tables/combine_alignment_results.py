"""Combine separately-run alignment analyses into publication deliverables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.synthetic_real_alignment import (  # noqa: E402
    aggregate_results,
    correlation_results,
    latex_table,
    safe_correlations,
    scatter_plot,
)
from plots_tables.alignment_per_class_table import build_table as build_per_class_table  # noqa: E402


METHOD_ORDER = [
    "retrained",
    "finetune",
    "gradient_ascent",
    "neggrad_plus",
    "random_label",
    "boundary_shrink",
    "boundary_expand",
    "l2ul_adv",
    "l2ul_imp",
    "fisher",
    "wood_fisher",
    "scrub",
    "bad_teacher",
    "salun",
    "delete",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="results_single_class/analysis/alignment_analysis"
    )
    parser.add_argument(
        "--input_dirs", nargs="+",
        default=["bad_teacher", "delete", "scrub", "retrained"],
    )
    parser.add_argument(
        "--output_dir",
        default="results_single_class/analysis/alignment_analysis/combined",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    frames = []
    for name in args.input_dirs:
        path = root / name / "alignment_per_class_seed.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        # Method-specific alignment runs also include retrained control rows.
        # When a dedicated retrained input directory is provided, keep that copy
        # as the canonical control and drop duplicate retrained rows from the
        # other method folders. This avoids harmless RNG/probe differences in
        # auxiliary control rows from blocking the combined table.
        if name != "retrained" and "retrained" in args.input_dirs:
            frame = frame[frame["method"] != "retrained"]
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    key = ["dataset", "backbone", "method", "forget_class", "seed"]
    metric_columns = [
        "alignment_inner_product", "alignment_cosine", "RS",
        "retrained_alignment_inner_product", "retrained_alignment_cosine",
        "retrained_RS", "delta_alignment_inner_product",
        "delta_alignment_cosine", "delta_RS",
    ]
    # Repeated retrained controls must be numerically identical before deduplication.
    conflicts = data.groupby(key)[metric_columns].nunique(dropna=False).max(axis=1)
    if (conflicts > 1).any():
        raise ValueError(
            "Repeated method/control rows disagree: "
            f"{conflicts[conflicts > 1].index.tolist()}"
        )
    data = data.drop_duplicates(key, keep="first")
    rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    data["_rank"] = data["method"].map(rank).fillna(len(rank))
    data = data.sort_values(["_rank", "forget_class", "seed"]).drop(columns="_rank")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "alignment_per_class_seed_combined.csv", index=False)
    aggregate = aggregate_results(data)
    aggregate["_rank"] = aggregate["method"].map(rank).fillna(len(rank))
    aggregate = aggregate.sort_values("_rank").drop(columns="_rank")
    aggregate.to_csv(output / "alignment_aggregated_combined.csv", index=False)
    correlations = correlation_results(data)
    correlations.to_csv(output / "alignment_correlations_combined.csv", index=False)
    correlation_pairs = [
        ("alignment_cosine", "RS"),
        ("alignment_inner_product", "RS"),
        ("delta_alignment_cosine", "delta_RS"),
        ("delta_alignment_inner_product", "delta_RS"),
    ]
    within_method_rows = []
    for method, frame in data[data["method"] != "retrained"].groupby("method"):
        for x_name, y_name in correlation_pairs:
            within_method_rows.append({
                "method": method, "x": x_name, "y": y_name,
                **safe_correlations(frame[x_name], frame[y_name]),
            })
    pd.DataFrame(within_method_rows).to_csv(
        output / "alignment_correlations_within_method.csv", index=False
    )
    latex_table(aggregate, output / "alignment_summary_table_combined.tex")
    (output / "alignment_per_class_table_combined.tex").write_text(
        build_per_class_table(data, list(range(10)), precision=2)
    )
    scatter_plot(data, "alignment_cosine", "RS", output / "alignment_vs_rs_combined.png")
    scatter_plot(
        data, "delta_alignment_cosine", "delta_RS",
        output / "delta_alignment_vs_delta_rs_combined.png",
    )
    print(f"[saved] {output.resolve()}")


if __name__ == "__main__":
    main()
