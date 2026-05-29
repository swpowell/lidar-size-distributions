from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SGP.config import DaylightConfig
from SGP.curation import _load_event_directory
from SGP.labels import add_cloud_state, add_daylight_flag


DEFAULT_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build curated datasets for level-specific resampled SGP products.")
    parser.add_argument("--root-dir", default="/thumper/users/scott.powell/code-data/research-code/lidar/SGP")
    parser.add_argument("--levels", nargs="+", type=int, default=DEFAULT_LEVELS)
    parser.add_argument("--level-index", type=int, help="Process only levels[level-index], useful for Slurm arrays.")
    parser.add_argument("--input-dir-template", default="C1_resample_level_{level}")
    parser.add_argument("--output-dir", default="SGP")
    parser.add_argument("--output-template", default="curated_sgp_c1_resample_level_{level}.parquet")
    parser.add_argument("--input-format", choices=["auto", "csv", "parquet"], default="parquet")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["regular", "resampled"],
        default=["regular", "resampled"],
        help=(
            "Methods to load from each C1_resample_level_<level> directory. "
            "For matched-reference ratios, regular is the masked-at-z numerator "
            "and resampled is the matched reference-level denominator."
        ),
    )
    parser.add_argument("--cloudy-csv", default="SGP/bkn_sct_800_1500_no_lower_no_precip_no_fog.csv")
    parser.add_argument("--clear-csv", default="SGP/clear_skies.csv")
    parser.add_argument("--skip-daylight", action="store_true")
    parser.add_argument("--skip-cloud-state", action="store_true")
    return parser.parse_args()


def build_curated_resampled_level(
    *,
    root_dir: Path,
    level: int,
    input_dir_template: str,
    input_format: str,
    methods: tuple[str, ...],
    cloudy_csv: str | None,
    clear_csv: str | None,
    add_daylight: bool,
) -> pd.DataFrame:
    input_dir = root_dir / input_dir_template.format(level=level)
    site = f"C1_resample_level_{level}"
    frames = [
        _load_event_directory(input_dir, method, site, input_format=input_format)
        for method in methods
    ]
    dataframe = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    if dataframe.empty:
        return dataframe

    dataframe = dataframe.dropna(subset=["event_time", "height_m", "Chord Length"]).copy()
    dataframe["chord_length_m"] = pd.to_numeric(dataframe["Chord Length"], errors="coerce")
    dataframe["wind_speed_mps"] = pd.to_numeric(dataframe["Wind Speed"], errors="coerce")
    dataframe["chord_time_s"] = pd.to_numeric(dataframe["Chord Time"], errors="coerce")
    dataframe["resample_level"] = level
    dataframe = dataframe.sort_values("event_time").reset_index(drop=True)

    if add_daylight:
        dataframe = add_daylight_flag(dataframe, config=DaylightConfig())
    if cloudy_csv or clear_csv:
        dataframe = add_cloud_state(dataframe, cloudy_csv=cloudy_csv, clear_csv=clear_csv)
    return dataframe


def main() -> None:
    args = parse_args()
    levels = args.levels
    if args.level_index is not None:
        if args.level_index < 0 or args.level_index >= len(levels):
            raise SystemExit(f"--level-index {args.level_index} is outside levels length {len(levels)}")
        levels = [levels[args.level_index]]

    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cloudy_csv = None if args.skip_cloud_state else args.cloudy_csv
    clear_csv = None if args.skip_cloud_state else args.clear_csv

    for level in levels:
        curated = build_curated_resampled_level(
            root_dir=root_dir,
            level=level,
            input_dir_template=args.input_dir_template,
            input_format=args.input_format,
            methods=tuple(args.methods),
            cloudy_csv=cloudy_csv,
            clear_csv=clear_csv,
            add_daylight=not args.skip_daylight,
        )
        output_path = output_dir / args.output_template.format(level=level)
        curated.to_parquet(output_path, index=False)
        print(f"Wrote curated level {level} dataset with {len(curated)} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
