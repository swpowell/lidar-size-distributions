from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SGP.clouds import is_cumuliform_proxy, is_strict_clear, read_all_metar_files
from SGP.run_matched_reference_ratio_analysis import (
    PBL_COLORS,
    PBL_LABELS,
    _load_pyplot,
    summarize_normalized_ratio,
)


SUMMARY_COLUMNS = [
    "pbl_bin",
    "z_over_zi_min",
    "z_over_zi_max",
    "z_over_zi_mid",
    "matched_z_count",
    "matched_z_q95",
    "matched_z_q95_ci_lower",
    "matched_z_q95_ci_upper",
    "matched_reference_count",
    "matched_reference_q95",
    "matched_reference_q95_ci_lower",
    "matched_reference_q95_ci_upper",
    "q95_ratio",
    "q95_ratio_ci_lower",
    "q95_ratio_ci_upper",
    "condition",
    "reference_level",
]


CONDITION_LABELS = {
    "cloudy_cumuliform": "Cumuliform cloudy",
    "clear": "Strict clear",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run H3 matched-reference Q95 ratio plots separately for cumuliform cloudy and strict clear cases."
    )
    parser.add_argument("--levels", nargs="+", type=int, default=[10])
    parser.add_argument("--input-template", default="SGP/curated_sgp_c1_resample_level_{level}.parquet")
    parser.add_argument("--asos-dir", default="SGP/ASOS")
    parser.add_argument("--output-dir", default="SGP/hypothesis_outputs")
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--tolerance-minutes", type=int, default=30)
    parser.add_argument("--expected-station", default="KPNC")
    return parser.parse_args()


def _asos_files(asos_dir: str | Path) -> list[Path]:
    directory = Path(asos_dir)
    return sorted(path for path in directory.iterdir() if path.is_file() and not path.name.startswith("."))


def build_strict_metar_catalog(asos_dir: str | Path, expected_station: str = "KPNC") -> pd.DataFrame:
    catalog = read_all_metar_files(_asos_files(asos_dir), expected_station=expected_station)
    if catalog.empty:
        return pd.DataFrame(columns=["metar_time", "metar", "strict_condition"])

    catalog = catalog.dropna(subset=["datetime"]).copy()
    catalog["metar_time"] = pd.to_datetime(catalog["datetime"], utc=True, errors="coerce").astype(
        "datetime64[ns, UTC]"
    )
    catalog = catalog.dropna(subset=["metar_time"]).sort_values("metar_time").reset_index(drop=True)
    catalog["strict_condition"] = np.select(
        [
            catalog["metar"].apply(is_cumuliform_proxy),
            catalog["metar"].apply(is_strict_clear),
        ],
        ["cloudy_cumuliform", "clear"],
        default=pd.NA,
    )
    return catalog[["metar_time", "metar", "strict_condition"]]


def add_nearest_strict_metar_condition(
    dataframe: pd.DataFrame,
    metar_catalog: pd.DataFrame,
    *,
    tolerance_minutes: int = 30,
) -> pd.DataFrame:
    if dataframe.empty:
        result = dataframe.copy()
        result["strict_condition"] = pd.Series(dtype="object")
        result["metar_time"] = pd.NaT
        return result

    result = dataframe.copy()
    result["event_time"] = pd.to_datetime(result["event_time"], utc=True, errors="coerce").astype(
        "datetime64[ns, UTC]"
    )
    result = result.sort_values("event_time").reset_index(drop=True)
    catalog = metar_catalog.dropna(subset=["metar_time"]).sort_values("metar_time").reset_index(drop=True)
    tolerance = pd.Timedelta(minutes=tolerance_minutes)

    return pd.merge_asof(
        result,
        catalog[["metar_time", "strict_condition"]],
        left_on="event_time",
        right_on="metar_time",
        tolerance=tolerance,
        direction="nearest",
    )


def plot_condition_normalized_height(
    summary: pd.DataFrame,
    *,
    level: int,
    condition: str,
    output_path: Path,
) -> None:
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

    condition_label = CONDITION_LABELS[condition]
    ax.axvline(1.0, color="0.25", linestyle="--", linewidth=1.4)
    ax.set_xlim(0, max(1.6, min(max_x * 1.08, 4.0)))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(
        f"Q95 chord length ratio: matched height z / level-{level} matched reference",
        fontsize=15,
    )
    ax.set_ylabel("Normalized comparison height, z / zi", fontsize=15)
    ax.set_title(
        f"H3 {condition_label}: normalized-height Q95 chord length relative to level-{level} reference",
        fontsize=18,
        pad=12,
    )
    ax.text(
        0.02,
        0.98,
        "Nearest raw METAR within 30 minutes is classified with strict H3 criteria.\n"
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


def run_level(
    *,
    level: int,
    input_template: str,
    output_dir: Path,
    metar_catalog: pd.DataFrame,
    quantile: float,
    min_count: int,
    tolerance_minutes: int,
) -> None:
    input_path = Path(input_template.format(level=level))
    dataframe = pd.read_parquet(input_path)
    dataframe = dataframe[
        (dataframe["is_day"] == True)
        & (dataframe["invalid_adjacent"] == False)
        & (dataframe["inside_pbl"] == True)
    ].copy()
    dataframe = add_nearest_strict_metar_condition(
        dataframe,
        metar_catalog,
        tolerance_minutes=tolerance_minutes,
    )

    for condition in CONDITION_LABELS:
        condition_subset = dataframe[dataframe["strict_condition"] == condition].copy()
        output_stem = (
            f"h3_matched_z_to_reference_level_{level}_q95_ratio_by_normalized_height_pbl_bins_{condition}"
        )
        summary_path = output_dir / f"{output_stem}_summary.csv"
        figure_path = output_dir / f"{output_stem}.png"

        methods = set(condition_subset["method"].dropna().unique())
        if condition_subset.empty or not {"regular", "resampled"}.issubset(methods):
            summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
            summary.to_csv(summary_path, index=False)
            print(f"Skipped empty H3 {condition} summary for level {level}: {summary_path}", flush=True)
            continue

        summary = summarize_normalized_ratio(condition_subset, quantile=quantile, min_count=min_count)
        summary["condition"] = condition
        summary["reference_level"] = level
        summary = summary[SUMMARY_COLUMNS]

        summary.to_csv(summary_path, index=False)
        if summary.empty:
            print(f"Skipped empty H3 {condition} summary for level {level}: {summary_path}", flush=True)
            continue
        plot_condition_normalized_height(summary, level=level, condition=condition, output_path=figure_path)
        print(f"Wrote {summary_path} and {figure_path}", flush=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metar_catalog = build_strict_metar_catalog(args.asos_dir, expected_station=args.expected_station)
    if metar_catalog.empty:
        raise SystemExit(f"No METAR records found in {args.asos_dir}")

    for level in args.levels:
        run_level(
            level=level,
            input_template=args.input_template,
            output_dir=output_dir,
            metar_catalog=metar_catalog,
            quantile=args.quantile,
            min_count=args.min_count,
            tolerance_minutes=args.tolerance_minutes,
        )


if __name__ == "__main__":
    main()
