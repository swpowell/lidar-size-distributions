from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SGP.config import GenerationPaths, SGP_C1_DEFAULTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate canonical SGP event tables.")
    parser.add_argument("--lidar-dir", default=str(SGP_C1_DEFAULTS.lidar_dir))
    parser.add_argument("--radar-dir", default=str(SGP_C1_DEFAULTS.radar_dir))
    parser.add_argument("--pblh-dir", default=str(SGP_C1_DEFAULTS.pblh_dir))
    parser.add_argument("--wind-dir", default=str(SGP_C1_DEFAULTS.wind_dir))
    parser.add_argument("--running-mean-dir", default=str(SGP_C1_DEFAULTS.running_mean_dir))
    parser.add_argument("--windspeed-dir", default=str(SGP_C1_DEFAULTS.windspeed_dir))
    parser.add_argument("--regular-output-dir", default=str(SGP_C1_DEFAULTS.regular_output_dir))
    parser.add_argument("--resampled-output-dir", default=str(SGP_C1_DEFAULTS.resampled_output_dir))
    parser.add_argument("--compute-running-means", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resample-level", type=int, default=26)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from SGP.generation import generate_daily_event_tables

    paths = GenerationPaths(
        lidar_dir=Path(args.lidar_dir),
        radar_dir=Path(args.radar_dir) if args.radar_dir else None,
        pblh_dir=Path(args.pblh_dir) if args.pblh_dir else None,
        wind_dir=Path(args.wind_dir),
        running_mean_dir=Path(args.running_mean_dir),
        windspeed_dir=Path(args.windspeed_dir),
        regular_output_dir=Path(args.regular_output_dir),
        resampled_output_dir=Path(args.resampled_output_dir),
    )
    processed = generate_daily_event_tables(
        paths,
        compute_running_means=args.compute_running_means,
        jobs=args.jobs,
        resample_level=args.resample_level,
        threshold=args.threshold,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(f"Processed {len(processed)} dates.")


if __name__ == "__main__":
    main()
