from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H3: cloudy versus clear eddy size analysis.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--min-count", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from SGP.hypotheses import analyze_h3, plot_h3

    dataframe = pd.read_csv(args.input, parse_dates=["event_time"])
    summary = analyze_h3(dataframe, quantile=args.quantile, min_count=args.min_count)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "h3_summary.csv"
    figure_path = output_dir / "h3_figure.png"
    summary.to_csv(summary_path, index=False)
    if not summary.empty:
        try:
            plot_h3(summary, figure_path)
        except RuntimeError as exc:
            print(f"Skipped H3 figure: {exc}")
    print(f"Wrote H3 summary to {summary_path}")


if __name__ == "__main__":
    main()
