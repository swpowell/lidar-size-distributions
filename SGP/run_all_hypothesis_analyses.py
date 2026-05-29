from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all SGP hypothesis analyses after one curated CSV read.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--min-count", type=int, default=10)
    return parser.parse_args()


def _write_summary_and_plot(summary: pd.DataFrame, summary_path: Path, figure_path: Path, plotter) -> None:
    summary.to_csv(summary_path, index=False)
    if summary.empty:
        print(f"Skipped figure for empty summary: {figure_path}")
        return
    try:
        plotter(summary, figure_path)
    except RuntimeError as exc:
            print(f"Skipped figure {figure_path}: {exc}")


def _read_analysis_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["event_time"])


def main() -> None:
    args = parse_args()
    from SGP.hypotheses import analyze_h1, analyze_h2, analyze_h3, plot_h1, plot_h2, plot_h3

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = _read_analysis_dataset(args.input)
    dataframe["event_time"] = pd.to_datetime(dataframe["event_time"], utc=True, errors="coerce")

    h1 = analyze_h1(
        dataframe,
        quantile=args.quantile,
        min_count=args.min_count,
    )
    _write_summary_and_plot(h1, output_dir / "h1_summary.csv", output_dir / "h1_figure.png", plot_h1)

    h2 = analyze_h2(dataframe, quantile=args.quantile, min_count=args.min_count)
    _write_summary_and_plot(h2, output_dir / "h2_summary.csv", output_dir / "h2_figure.png", plot_h2)

    h3 = analyze_h3(dataframe, quantile=args.quantile, min_count=args.min_count)
    _write_summary_and_plot(h3, output_dir / "h3_summary.csv", output_dir / "h3_figure.png", plot_h3)

    print(f"Wrote hypothesis outputs to {output_dir}")


if __name__ == "__main__":
    main()
