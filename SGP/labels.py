from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import DaylightConfig


def _solar_daylight_mask(event_time: pd.Series, config: DaylightConfig) -> pd.Series:
    """Return an approximate daylight flag without optional astronomy packages."""
    utc_time = pd.to_datetime(event_time, utc=True, errors="coerce")
    valid = utc_time.notna()
    result = pd.Series(False, index=event_time.index, dtype=bool)
    if not valid.any():
        return result

    selected = utc_time[valid]
    day_of_year = selected.dt.dayofyear.to_numpy(dtype=float)
    hour = (
        selected.dt.hour.to_numpy(dtype=float)
        + selected.dt.minute.to_numpy(dtype=float) / 60
        + selected.dt.second.to_numpy(dtype=float) / 3600
    )
    gamma = 2 * np.pi / 365 * (day_of_year - 1 + (hour - 12) / 24)
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )

    true_solar_minutes = hour * 60 + equation_of_time + 4 * config.longitude
    hour_angle = np.deg2rad(true_solar_minutes / 4 - 180)
    latitude = np.deg2rad(config.latitude)
    cos_zenith = (
        np.sin(latitude) * np.sin(declination)
        + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle)
    )
    zenith_deg = np.rad2deg(np.arccos(np.clip(cos_zenith, -1, 1)))
    result.loc[valid] = zenith_deg < 90.833
    return result


def add_daylight_flag(dataframe: pd.DataFrame, config: DaylightConfig = DaylightConfig()) -> pd.DataFrame:
    if dataframe.empty:
        result = dataframe.copy()
        result["is_day"] = pd.Series(dtype=bool)
        return result

    try:
        from astral import LocationInfo
        from astral.sun import sun
    except ImportError as exc:  # pragma: no cover - environment specific
        result = dataframe.copy()
        event_time = pd.to_datetime(result["event_time"], utc=True, errors="coerce")
        result["local_time"] = event_time.dt.tz_convert(ZoneInfo(config.timezone))
        result["sunrise"] = pd.NaT
        result["sunset"] = pd.NaT
        result["is_day"] = _solar_daylight_mask(result["event_time"], config=config)
        return result

    result = dataframe.copy()
    local_tz = ZoneInfo(config.timezone)
    event_time = pd.to_datetime(result["event_time"], utc=True, errors="coerce")
    local_time = event_time.dt.tz_convert(local_tz)

    location = LocationInfo(
        name=config.name,
        region=config.region,
        timezone=config.timezone,
        latitude=config.latitude,
        longitude=config.longitude,
    )
    unique_dates = np.unique(event_time.dt.date.dropna())
    sun_times = {date: sun(location.observer, date=date, tzinfo=local_tz) for date in unique_dates}

    result["local_time"] = local_time
    result["sunrise"] = pd.Series(event_time.dt.date, index=result.index).map(
        lambda date: sun_times.get(date, {}).get("sunrise")
    )
    result["sunset"] = pd.Series(event_time.dt.date, index=result.index).map(
        lambda date: sun_times.get(date, {}).get("sunset")
    )
    result["is_day"] = (result["sunrise"] < result["local_time"]) & (result["sunset"] > result["local_time"])
    return result


def _normalize_timestamp_column(series: pd.Series) -> pd.Series:
    time_str = series.astype(str).str.replace(
        r"([+-]\d{2})[.:]?(\d{2})?$",
        lambda match: f"{match.group(1)}:{match.group(2) if match.group(2) else '00'}",
        regex=True,
    )
    return pd.to_datetime(time_str, utc=True, errors="coerce").astype("datetime64[ns, UTC]")


def add_cloud_state(
    dataframe: pd.DataFrame,
    cloudy_csv: str | None = None,
    clear_csv: str | None = None,
    tolerance_minutes: int = 30,
) -> pd.DataFrame:
    result = dataframe.copy()
    result["cloud_state"] = pd.Series(index=result.index, dtype="object")
    if result.empty:
        return result

    result["event_time"] = pd.to_datetime(result["event_time"], utc=True, errors="coerce").astype(
        "datetime64[ns, UTC]"
    )
    result = result.sort_values("event_time")
    tolerance = pd.Timedelta(minutes=tolerance_minutes)

    if cloudy_csv:
        cloudy = pd.read_csv(cloudy_csv)
        cloudy["dt"] = _normalize_timestamp_column(cloudy["datetime"])
        cloudy = cloudy.dropna(subset=["dt"]).sort_values("dt")
        matched = pd.merge_asof(
            result,
            cloudy[["dt"]].rename(columns={"dt": "cloud_time"}),
            left_on="event_time",
            right_on="cloud_time",
            tolerance=tolerance,
            direction="nearest",
        )
        result.loc[matched["cloud_time"].notna(), "cloud_state"] = "cloudy"

    if clear_csv:
        clear = pd.read_csv(clear_csv)
        clear["dt"] = _normalize_timestamp_column(clear["datetime"])
        clear = clear.dropna(subset=["dt"]).sort_values("dt")
        matched = pd.merge_asof(
            result,
            clear[["dt"]].rename(columns={"dt": "clear_time"}),
            left_on="event_time",
            right_on="clear_time",
            tolerance=tolerance,
            direction="nearest",
        )
        clear_mask = matched["clear_time"].notna() & result["cloud_state"].isna()
        result.loc[clear_mask, "cloud_state"] = "clear"

    return result
