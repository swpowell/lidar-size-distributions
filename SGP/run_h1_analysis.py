from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H1: eddy size versus PBL depth.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--pbl-bin-width-m", type=int, default=200)
    parser.add_argument("--analysis-height-min-m", type=float, default=450)
    parser.add_argument("--analysis-height-max-m", type=float, default=650)
    parser.add_argument("--min-count", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from SGP.hypotheses import analyze_h1, plot_h1

    dataframe = pd.read_csv(args.input, parse_dates=["event_time"])
    summary = analyze_h1(
        dataframe,
        quantile=args.quantile,
        pbl_bin_width_m=args.pbl_bin_width_m,
        analysis_height_min_m=args.analysis_height_min_m,
        analysis_height_max_m=args.analysis_height_max_m,
        min_count=args.min_count,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "h1_summary.csv"
    figure_path = output_dir / "h1_figure.png"
    summary.to_csv(summary_path, index=False)
    if not summary.empty:
        try:
            plot_h1(summary, figure_path)
        except RuntimeError as exc:
            print(f"Skipped H1 figure: {exc}")
    print(f"Wrote H1 summary to {summary_path}")


if __name__ == "__main__":
    main()
