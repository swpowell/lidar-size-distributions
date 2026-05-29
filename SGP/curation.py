from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DaylightConfig
from .labels import add_cloud_state, add_daylight_flag


EXPECTED_HEIGHTS_M = np.array(
    [
        45.0,
        75.0,
        105.0,
        135.0,
        165.0,
        195.0,
        225.0,
        255.0,
        285.0,
        315.0,
        345.0,
        375.0,
        405.0,
        435.0,
        465.0,
        495.0,
        525.0,
        555.0,
        585.0,
        615.0,
        645.0,
        675.0,
        705.0,
        735.0,
        765.0,
        795.0,
        825.0,
        855.0,
        885.0,
        915.0,
        945.0,
        975.0,
        1005.0,
        1035.0,
        1065.0,
        1095.0,
        1125.0,
        1155.0,
        1185.0,
        1215.0,
        1245.0,
        1275.0,
        1305.0,
        1335.0,
        1365.0,
        1395.0,
        1425.0,
        1455.0,
        1485.0,
        1515.0,
        1545.0,
        1575.0,
        1605.0,
        1635.0,
        1665.0,
        1695.0,
        1725.0,
        1755.0,
        1785.0,
        1815.0,
        1845.0,
        1875.0,
        1905.0,
        1935.0,
        1965.0,
        1995.0,
        15.0,
    ]
)


def _normalize_height_m(series: pd.Series, method: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if method == "resampled" or values.max(skipna=True) < 10:
        return values * 1000
    return values


def _parse_event_time(dataframe: pd.DataFrame) -> pd.Series:
    if "Center Datetime" in dataframe.columns:
        return pd.to_datetime(dataframe["Center Datetime"], utc=True, errors="coerce")
    if "__index_level_0__" in dataframe.columns:
        return pd.to_datetime(dataframe["__index_level_0__"], utc=True, errors="coerce")
    if "Unnamed: 0" in dataframe.columns:
        return pd.to_datetime(dataframe["Unnamed: 0"], utc=True, errors="coerce")
    if dataframe.index.name:
        return pd.to_datetime(dataframe.index, utc=True, errors="coerce")
    return pd.to_datetime(dataframe.index, utc=True, errors="coerce")


def _read_event_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_event_directory(directory: Path, method: str, location: str, input_format: str = "auto") -> pd.DataFrame:
    if input_format == "auto":
        files = sorted(directory.glob("*.parquet"))
        if not files:
            files = sorted(directory.glob("*.csv"))
    else:
        files = sorted(directory.glob(f"*.{input_format}"))
    if method == "resampled":
        method_files = [path for path in files if path.name.startswith("updrafts_resample_")]
        files = method_files if method_files else files
    elif method == "regular":
        files = [path for path in files if not path.name.startswith("updrafts_resample_")]
    if not files:
        return pd.DataFrame()

    frames = [_read_event_table(path) for path in files]
    dataframe = pd.concat(frames, ignore_index=True)
    dataframe["site"] = location
    dataframe["method"] = method
    dataframe["event_time"] = _parse_event_time(dataframe)
    dataframe["height_m"] = _normalize_height_m(dataframe["Height"], method)
    if "PBL Height" in dataframe.columns:
        dataframe["pbl_height_m"] = pd.to_numeric(dataframe["PBL Height"], errors="coerce")
    else:
        dataframe["pbl_height_m"] = np.nan
    dataframe["inside_pbl"] = dataframe["height_m"] <= dataframe["pbl_height_m"]
    dataframe["invalid_adjacent"] = dataframe["Invalid Adjacent"].astype(str).str.lower().eq("true")
    keep_mask = np.isclose(
        dataframe["height_m"].values[:, None],
        EXPECTED_HEIGHTS_M,
        rtol=1e-5,
        atol=1e-8,
    ).any(axis=1)
    dataframe = dataframe[keep_mask].copy()
    return dataframe


def build_curated_dataset(
    root_dir: str | Path,
    *,
    locations: list[str],
    include_methods: tuple[str, ...] = ("regular", "resampled"),
    input_format: str = "auto",
    cloudy_csv: str | None = None,
    clear_csv: str | None = None,
    add_daylight: bool = True,
    daylight_config: DaylightConfig = DaylightConfig(),
) -> pd.DataFrame:
    root_dir = Path(root_dir)
    frames = []
    for location in locations:
        if "regular" in include_methods:
            frames.append(_load_event_directory(root_dir / location, "regular", location, input_format=input_format))
        if "resampled" in include_methods:
            frames.append(
                _load_event_directory(root_dir / f"{location}_resample", "resampled", location, input_format=input_format)
            )

    dataframe = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    if dataframe.empty:
        return dataframe

    dataframe = dataframe.dropna(subset=["event_time", "height_m", "Chord Length"]).copy()
    dataframe["chord_length_m"] = pd.to_numeric(dataframe["Chord Length"], errors="coerce")
    dataframe["wind_speed_mps"] = pd.to_numeric(dataframe["Wind Speed"], errors="coerce")
    dataframe["chord_time_s"] = pd.to_numeric(dataframe["Chord Time"], errors="coerce")
    dataframe = dataframe.sort_values("event_time").reset_index(drop=True)

    if add_daylight:
        dataframe = add_daylight_flag(dataframe, config=daylight_config)
    if cloudy_csv or clear_csv:
        dataframe = add_cloud_state(dataframe, cloudy_csv=cloudy_csv, clear_csv=clear_csv)

    return dataframe
