from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom


PBL_DEPTH_BINS_M = [400, 800, 1200, 1800, np.inf]
PBL_DEPTH_LABELS = [
    "400-800 m",
    "800-1200 m",
    "1200-1800 m",
    ">1800 m",
]
PBL_DEPTH_COLORS = {
    "400-800 m": "#2c7598",
    "800-1200 m": "#7a8f2a",
    "1200-1800 m": "#b85f2b",
    ">1800 m": "#222222",
}


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("matplotlib is required to write hypothesis figures.") from exc
    return plt


def bootstrap_quantile_ci(
    series: pd.Series,
    quantile: float,
    *,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if clean.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = np.array(
        [np.quantile(rng.choice(clean, size=clean.size, replace=True), quantile) for _ in range(n_boot)]
    )
    alpha = (1 - confidence) / 2
    return np.quantile(samples, alpha), np.quantile(samples, 1 - alpha)


def quantile_rank_ci(
    series: pd.Series,
    quantile: float,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return np.nan, np.nan
    clean.sort()
    alpha = 1.0 - confidence
    lower_rank = int(max(0, binom.ppf(alpha / 2.0, clean.size, quantile) - 1))
    upper_rank = int(min(clean.size - 1, binom.ppf(1.0 - alpha / 2.0, clean.size, quantile) - 1))
    return float(clean[lower_rank]), float(clean[upper_rank])


def filter_primary_analysis(
    dataframe: pd.DataFrame,
    *,
    method: str = "resampled",
    require_inside_pbl: bool = True,
    daytime_only: bool = False,
    cloud_state: str | None = None,
    min_chord_length: float | None = None,
) -> pd.DataFrame:
    result = dataframe.copy()
    if method:
        result = result[result["method"] == method]
    result = result[result["invalid_adjacent"] == False]
    if require_inside_pbl:
        result = result[result["inside_pbl"] == True]
    if daytime_only and "is_day" in result.columns:
        result = result[result["is_day"] == True]
    if cloud_state:
        result = result[result["cloud_state"] == cloud_state]
    if min_chord_length is not None:
        result = result[result["chord_length_m"] >= min_chord_length]
    return result.copy()


def _summarize_groups(grouped, quantile: float, min_count: int) -> pd.DataFrame:
    rows = []
    for group_value, group in grouped:
        count = len(group)
        if count < min_count:
            continue
        lower, upper = quantile_rank_ci(group["chord_length_m"], quantile)
        rows.append(
            {
                "group": group_value,
                "count": count,
                "quantile": group["chord_length_m"].quantile(quantile),
                "median": group["chord_length_m"].median(),
                "mean": group["chord_length_m"].mean(),
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )
    return pd.DataFrame(rows)


def analyze_h1(
    dataframe: pd.DataFrame,
    *,
    quantile: float = 0.95,
    min_count: int = 10,
) -> pd.DataFrame:
    subset = filter_primary_analysis(dataframe)
    subset = subset.dropna(subset=["pbl_height_m"])
    subset["height_m"] = pd.to_numeric(subset["height_m"], errors="coerce").round()
    subset["pbl_depth_bin"] = pd.cut(
        subset["pbl_height_m"],
        bins=PBL_DEPTH_BINS_M,
        labels=PBL_DEPTH_LABELS,
        right=False,
    )
    subset = subset.dropna(subset=["height_m", "pbl_depth_bin"])
    summary = _summarize_groups(subset.groupby(["pbl_depth_bin", "height_m"], observed=True), quantile, min_count)
    if not summary.empty:
        pbl_depth_height = pd.DataFrame(summary["group"].tolist(), columns=["pbl_depth_bin", "height_m"])
        summary = pd.concat([pbl_depth_height, summary.drop(columns=["group"])], axis=1)
        summary["pbl_depth_bin"] = pd.Categorical(
            summary["pbl_depth_bin"],
            categories=PBL_DEPTH_LABELS,
            ordered=True,
        )
        summary = summary.sort_values(["pbl_depth_bin", "height_m"]).reset_index(drop=True)
    return summary


def analyze_h2(
    dataframe: pd.DataFrame,
    *,
    quantile: float = 0.95,
    min_count: int = 10,
) -> pd.DataFrame:
    subset = filter_primary_analysis(dataframe)
    summary = _summarize_groups(subset.groupby("height_m"), quantile, min_count)
    if not summary.empty:
        summary = summary.rename(columns={"group": "height_m"}).sort_values("height_m")
    return summary


def analyze_h3(
    dataframe: pd.DataFrame,
    *,
    quantile: float = 0.95,
    min_count: int = 10,
) -> pd.DataFrame:
    subset = filter_primary_analysis(dataframe, daytime_only=True)
    subset = subset[subset["cloud_state"].isin(["cloudy", "clear"])].copy()
    summary = _summarize_groups(subset.groupby(["cloud_state", "height_m"]), quantile, min_count)
    if not summary.empty:
        cloud_state_height = pd.DataFrame(summary["group"].tolist(), columns=["cloud_state", "height_m"])
        summary = pd.concat([cloud_state_height, summary.drop(columns=["group"])], axis=1).sort_values(
            ["cloud_state", "height_m"]
        )
    return summary


def plot_h1(summary: pd.DataFrame, output_path: str | Path) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(9, 7), dpi=120)
    max_x = 0.0
    for label in PBL_DEPTH_LABELS:
        subset = summary[summary["pbl_depth_bin"].astype(str) == label].sort_values("height_m")
        if subset.empty:
            continue
        color = PBL_DEPTH_COLORS[label]
        ax.plot(
            subset["quantile"],
            subset["height_m"],
            marker="o",
            markersize=4,
            linewidth=2.2,
            color=color,
            label=f"{label} (N={int(subset['count'].sum()):,})",
        )
        ax.fill_betweenx(
            subset["height_m"],
            subset["ci_lower"],
            subset["ci_upper"],
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        max_x = max(max_x, float(subset["ci_upper"].max()))
    ax.set_xlim(0, max_x * 1.08 if max_x > 0 else 1)
    ax.set_ylim(0, max(2050, float(summary["height_m"].max()) + 50))
    ax.set_xlabel("95th-percentile eddy chord length (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("H1: Level-15 resampled eddy size by PBL depth")
    ax.grid(True, color="0.9", linewidth=1)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_h2(summary: pd.DataFrame, output_path: str | Path) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary["quantile"], summary["height_m"], color="black")
    ax.fill_betweenx(summary["height_m"], summary["ci_lower"], summary["ci_upper"], color="0.8")
    ax.set_xlabel("Upper-tail chord length (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("H2: Eddy size versus height inside the PBL")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_h3(summary: pd.DataFrame, output_path: str | Path) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(7, 5))
    for cloud_state, subset in summary.groupby("cloud_state"):
        ax.plot(subset["quantile"], subset["height_m"], label=cloud_state.capitalize())
    ax.set_xlabel("Upper-tail chord length (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("H3: Cloudy versus clear eddy size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
