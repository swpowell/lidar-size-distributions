from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


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
        lower, upper = bootstrap_quantile_ci(group["chord_length_m"], quantile)
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
    pbl_bin_width_m: int = 200,
    analysis_height_min_m: float = 450,
    analysis_height_max_m: float = 650,
    min_count: int = 10,
) -> pd.DataFrame:
    subset = filter_primary_analysis(dataframe)
    subset = subset[(subset["height_m"] >= analysis_height_min_m) & (subset["height_m"] <= analysis_height_max_m)]
    subset = subset.dropna(subset=["pbl_height_m"])
    subset["pbl_bin_m"] = pbl_bin_width_m * np.floor(subset["pbl_height_m"] / pbl_bin_width_m)
    summary = _summarize_groups(subset.groupby("pbl_bin_m"), quantile, min_count)
    if not summary.empty:
        summary = summary.rename(columns={"group": "pbl_bin_m"}).sort_values("pbl_bin_m")
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
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary["pbl_bin_m"], summary["quantile"], color="black")
    ax.fill_between(summary["pbl_bin_m"], summary["ci_lower"], summary["ci_upper"], color="0.8")
    ax.set_xlabel("PBL height bin (m)")
    ax.set_ylabel("Upper-tail chord length (m)")
    ax.set_title("H1: Eddy size versus PBL depth")
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
