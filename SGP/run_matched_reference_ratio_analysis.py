from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_LEVELS = [10, 15, 20, 25, 30, 35, 40]
PBL_BINS = [400, 800, 1200, 1800, np.inf]
PBL_LABELS = ["400 <= zi < 800 m", "800 <= zi < 1200 m", "1200 <= zi < 1800 m", "zi >= 1800 m"]
PBL_COLORS = {
    "400 <= zi < 800 m": "#2c7598",
    "800 <= zi < 1200 m": "#7a8f2a",
    "1200 <= zi < 1800 m": "#b85f2b",
    "zi >= 1800 m": "#222222",
}
Z_OVER_ZI_BINS = np.r_[-0.001, np.arange(0.05, 1.0001, 0.05)]
Z_OVER_ZI_LABELS = list(range(len(Z_OVER_ZI_BINS) - 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot matched resampled chord-length ratios. The numerator is the "
            "masked-at-z chord distribution, and the denominator is the matched "
            "reference-level chord distribution generated using the same z mask."
        )
    )
    parser.add_argument("--levels", nargs="+", type=int, default=DEFAULT_LEVELS)
    parser.add_argument("--input-template", default="SGP/curated_sgp_c1_resample_level_{level}.parquet")
    parser.add_argument("--output-dir", default="SGP/hypothesis_outputs")
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--min-count", type=int, default=10)
    return parser.parse_args()


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def _quantile_ci(values: pd.Series, quantile: float, confidence: float = 0.95) -> tuple[float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return np.nan, np.nan, np.nan
    clean.sort()
    estimate = float(np.quantile(clean, quantile))
    alpha = 1.0 - confidence
    lower_rank = int(max(0, binom.ppf(alpha / 2.0, clean.size, quantile) - 1))
    upper_rank = int(min(clean.size - 1, binom.ppf(1.0 - alpha / 2.0, clean.size, quantile) - 1))
    return estimate, float(clean[lower_rank]), float(clean[upper_rank])


def _analysis_subset(dataframe: pd.DataFrame) -> pd.DataFrame:
    subset = dataframe.copy()
    subset = subset[(subset["invalid_adjacent"] == False) & (subset["inside_pbl"] == True)]
    subset = subset.dropna(subset=["height_m", "pbl_height_m", "chord_length_m"])
    subset = subset[subset["pbl_height_m"] > 0].copy()
    subset["height_m"] = pd.to_numeric(subset["height_m"], errors="coerce").round()
    subset["z_over_zi"] = subset["height_m"] / subset["pbl_height_m"]
    subset["pbl_bin"] = pd.cut(subset["pbl_height_m"], bins=PBL_BINS, labels=PBL_LABELS, right=False)
    subset["z_over_zi_bin"] = pd.cut(
        subset["z_over_zi"],
        bins=Z_OVER_ZI_BINS,
        labels=Z_OVER_ZI_LABELS,
        right=True,
        include_lowest=True,
    )
    return subset.dropna(subset=["height_m", "pbl_bin", "z_over_zi_bin"])


def _summarize(
    dataframe: pd.DataFrame,
    prefix: str,
    quantile: float,
    min_count: int,
    group_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for group_values, group in dataframe.groupby(group_columns, observed=True):
        if len(group) < min_count:
            continue
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        q, lower, upper = _quantile_ci(group["chord_length_m"], quantile)
        row = dict(zip(group_columns, group_values))
        row["pbl_bin"] = str(row["pbl_bin"])
        if "height_m" in row:
            row["height_m"] = float(row["height_m"])
        if "z_over_zi_bin" in row:
            z_bin = int(row.pop("z_over_zi_bin"))
            row["z_over_zi_min"] = float(Z_OVER_ZI_BINS[z_bin])
            row["z_over_zi_max"] = float(Z_OVER_ZI_BINS[z_bin + 1])
            row["z_over_zi_mid"] = float((Z_OVER_ZI_BINS[z_bin] + Z_OVER_ZI_BINS[z_bin + 1]) / 2)
        row.update(
            {
                f"{prefix}_count": int(len(group)),
                f"{prefix}_q95": q,
                f"{prefix}_q95_ci_lower": lower,
                f"{prefix}_q95_ci_upper": upper,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _ratio_from_summaries(numerator: pd.DataFrame, denominator: pd.DataFrame, merge_columns: list[str]) -> pd.DataFrame:
    ratio = numerator.merge(denominator, on=merge_columns, how="inner")
    ratio["q95_ratio"] = ratio["matched_z_q95"] / ratio["matched_reference_q95"]
    ratio["q95_ratio_ci_lower"] = ratio["matched_z_q95_ci_lower"] / ratio["matched_reference_q95_ci_upper"]
    ratio["q95_ratio_ci_upper"] = ratio["matched_z_q95_ci_upper"] / ratio["matched_reference_q95_ci_lower"]
    ratio["pbl_bin"] = pd.Categorical(ratio["pbl_bin"], categories=PBL_LABELS, ordered=True)
    return ratio


def _require_matched_methods(dataframe: pd.DataFrame) -> None:
    methods = set(dataframe["method"].dropna().unique())
    required = {"regular", "resampled"}
    if not required.issubset(methods):
        missing = ", ".join(sorted(required - methods))
        raise ValueError(
            f"curated level dataset is missing method(s): {missing}. "
            "Regenerate level-specific event tables with --methods regular resampled, "
            "then rebuild the curated level dataset."
        )


def summarize_ratio(dataframe: pd.DataFrame, quantile: float, min_count: int) -> pd.DataFrame:
    _require_matched_methods(dataframe)
    subset = _analysis_subset(dataframe)
    group_columns = ["pbl_bin", "height_m"]
    numerator = _summarize(subset[subset["method"] == "regular"], "matched_z", quantile, min_count, group_columns)
    denominator = _summarize(
        subset[subset["method"] == "resampled"], "matched_reference", quantile, min_count, group_columns
    )
    ratio = _ratio_from_summaries(numerator, denominator, group_columns)
    return ratio.sort_values(["pbl_bin", "height_m"]).reset_index(drop=True)


def summarize_normalized_ratio(dataframe: pd.DataFrame, quantile: float, min_count: int) -> pd.DataFrame:
    _require_matched_methods(dataframe)
    subset = _analysis_subset(dataframe)
    group_columns = ["pbl_bin", "z_over_zi_bin"]
    merge_columns = ["pbl_bin", "z_over_zi_min", "z_over_zi_max", "z_over_zi_mid"]
    numerator = _summarize(subset[subset["method"] == "regular"], "matched_z", quantile, min_count, group_columns)
    denominator = _summarize(
        subset[subset["method"] == "resampled"], "matched_reference", quantile, min_count, group_columns
    )
    ratio = _ratio_from_summaries(numerator, denominator, merge_columns)
    return ratio.sort_values(["pbl_bin", "z_over_zi_mid"]).reset_index(drop=True)


def plot_physical_height(summary: pd.DataFrame, level: int, output_path: Path) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(12.5, 10), dpi=120)
    max_x = 1.0
    for label in PBL_LABELS:
        sub = summary[summary["pbl_bin"].astype(str) == label].sort_values("height_m")
        if sub.empty:
            continue
        color = PBL_COLORS[label]
        ax.errorbar(
            sub["q95_ratio"],
            sub["height_m"],
            xerr=[
                sub["q95_ratio"] - sub["q95_ratio_ci_lower"],
                sub["q95_ratio_ci_upper"] - sub["q95_ratio"],
            ],
            fmt="o-",
            color=color,
            ecolor=color,
            elinewidth=1.3,
            capsize=3,
            markersize=5.5,
            linewidth=2.6,
            label=f"{label} (matched-z N={int(sub['matched_z_count'].sum()):,})",
        )
        max_x = max(max_x, float(sub["q95_ratio_ci_upper"].quantile(0.98)))

    ax.axvline(1.0, color="0.25", linestyle="--", linewidth=1.4)
    ax.set_xlim(0, max(1.6, min(max_x * 1.08, 4.0)))
    ax.set_ylim(0, max(2050, summary["height_m"].max() + 50))
    ax.set_xlabel(
        f"Q95 chord length ratio: matched height z / level-{level} matched reference",
        fontsize=15,
    )
    ax.set_ylabel("Comparison height (m)", fontsize=15)
    ax.set_title(
        f"Height-specific Q95 chord length relative to level-{level} matched reference",
        fontsize=20,
        pad=12,
    )
    ax.text(
        0.02,
        0.98,
        "Numerator is the resampled/masked chord distribution at height z.\n"
        f"Denominator is the level-{level} reference distribution using the same z mask.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        color="0.28",
    )
    ax.grid(True, color="0.9", linewidth=1)
    ax.legend(loc="lower right", fontsize=12, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_normalized_height(summary: pd.DataFrame, level: int, output_path: Path) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(12.5, 10), dpi=120)
    max_x = 1.0
    for label in PBL_LABELS:
        sub = summary[summary["pbl_bin"].astype(str) == label].sort_values("z_over_zi_mid")
        if sub.empty:
            continue
        color = PBL_COLORS[label]
        ax.errorbar(
            sub["q95_ratio"],
            sub["z_over_zi_mid"],
            xerr=[
                sub["q95_ratio"] - sub["q95_ratio_ci_lower"],
                sub["q95_ratio_ci_upper"] - sub["q95_ratio"],
            ],
            fmt="o-",
            color=color,
            ecolor=color,
            elinewidth=1.3,
            capsize=3,
            markersize=5.5,
            linewidth=2.6,
            label=f"{label} (matched-z N={int(sub['matched_z_count'].sum()):,})",
        )
        max_x = max(max_x, float(sub["q95_ratio_ci_upper"].quantile(0.98)))

    ax.axvline(1.0, color="0.25", linestyle="--", linewidth=1.4)
    ax.set_xlim(0, max(1.6, min(max_x * 1.08, 4.0)))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(
        f"Q95 chord length ratio: matched height z / level-{level} matched reference",
        fontsize=15,
    )
    ax.set_ylabel("Normalized comparison height, z / zi", fontsize=15)
    ax.set_title(
        f"Normalized-height Q95 chord length relative to level-{level} matched reference",
        fontsize=20,
        pad=12,
    )
    ax.text(
        0.02,
        0.98,
        "Numerator is the resampled/masked chord distribution binned by z/zi.\n"
        f"Denominator is the level-{level} reference distribution using the same z/zi mask bin.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        color="0.28",
    )
    ax.grid(True, color="0.9", linewidth=1)
    ax.legend(loc="lower right", fontsize=12, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for level in args.levels:
        input_path = Path(args.input_template.format(level=level))
        dataframe = pd.read_parquet(input_path)
        summary = summarize_ratio(dataframe, quantile=args.quantile, min_count=args.min_count)
        summary_path = output_dir / f"matched_z_to_reference_level_{level}_q95_ratio_by_height_pbl_bins_summary.csv"
        figure_path = output_dir / f"matched_z_to_reference_level_{level}_q95_ratio_by_height_pbl_bins.png"
        summary.to_csv(summary_path, index=False)
        if summary.empty:
            print(f"Skipped empty summary for level {level}: {summary_path}")
            continue
        plot_physical_height(summary, level, figure_path)
        print(f"Wrote {summary_path} and {figure_path}")

        normalized_summary = summarize_normalized_ratio(dataframe, quantile=args.quantile, min_count=args.min_count)
        normalized_summary_path = (
            output_dir / f"matched_z_to_reference_level_{level}_q95_ratio_by_normalized_height_pbl_bins_summary.csv"
        )
        normalized_figure_path = (
            output_dir / f"matched_z_to_reference_level_{level}_q95_ratio_by_normalized_height_pbl_bins.png"
        )
        normalized_summary.to_csv(normalized_summary_path, index=False)
        if normalized_summary.empty:
            print(f"Skipped empty normalized summary for level {level}: {normalized_summary_path}")
            continue
        plot_normalized_height(normalized_summary, level, normalized_figure_path)
        print(f"Wrote {normalized_summary_path} and {normalized_figure_path}")


if __name__ == "__main__":
    main()
