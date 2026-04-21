from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a curated SGP event dataset.")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--locations", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", default=["regular", "resampled"])
    parser.add_argument("--cloudy-csv")
    parser.add_argument("--clear-csv")
    parser.add_argument("--skip-daylight", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from SGP.curation import build_curated_dataset

    curated = build_curated_dataset(
        args.root_dir,
        locations=args.locations,
        include_methods=tuple(args.methods),
        cloudy_csv=args.cloudy_csv,
        clear_csv=args.clear_csv,
        add_daylight=not args.skip_daylight,
    )
    curated.to_csv(args.output, index=False)
    print(f"Wrote curated dataset with {len(curated)} rows to {args.output}")


if __name__ == "__main__":
    main()
