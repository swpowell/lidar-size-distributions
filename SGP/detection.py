from __future__ import annotations

import os
import time as timer
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d, label


INTENSITY_THRESHOLD = 1.008


def _joblib_prefer() -> str | None:
    return os.environ.get("LIDAR_JOBLIB_PREFER") or None


def get_date_str(date) -> str:
    return date.strftime("%Y%m%d")


def simple_smoother(input_2d: xr.DataArray, sigma: float = 1) -> np.ndarray:
    return gaussian_filter1d(input_2d, sigma=sigma, axis=0)


def compute_velocity_running_mean_file(
    name: str | Path,
    previous_file: str | Path | None,
    next_file: str | Path | None,
    output_dir: str | Path,
    intensity_threshold: float = INTENSITY_THRESHOLD,
) -> None:
    ds = xr.open_dataset(name)

    if previous_file:
        ds_prev = xr.open_dataset(previous_file)
        format_ts = np.datetime_as_string(ds_prev["time"].values[0], unit="h")
        ds_prev = ds_prev.sel(time=slice(format_ts + ":49:59", None))
    else:
        ds_prev = None

    if next_file:
        ds_next = xr.open_dataset(next_file)
        format_ts = np.datetime_as_string(ds_next["time"].values[0], unit="h")
        ds_next = ds_next.sel(time=slice(None, format_ts + ":10"))
    else:
        ds_next = None

    radial_sources = [item["radial_velocity"] for item in (ds_prev, ds, ds_next) if item is not None]
    intensity_sources = [item["intensity"] for item in (ds_prev, ds, ds_next) if item is not None]

    combined_data = xr.concat(radial_sources, dim="time")
    combined_intensity = xr.concat(intensity_sources, dim="time")

    time_values = combined_data["time"].values
    radial_velocity_values = combined_data.values
    combined_intensity_values = combined_intensity.values
    running_mean_10min_values = []

    for i in range(len(time_values)):
        start_time = time_values[i] - np.timedelta64(5, "m")
        end_time = time_values[i] + np.timedelta64(5, "m")
        mask = (time_values >= start_time) & (time_values <= end_time)
        intensity_values = combined_intensity_values[mask]
        good_fraction = (intensity_values >= intensity_threshold).sum(axis=0) / intensity_values.shape[0]
        mean_val = np.mean(radial_velocity_values[mask], axis=0)
        mean_val[good_fraction < 1] = np.nan
        running_mean_10min_values.append(mean_val)

    running_mean_10min = xr.DataArray(
        data=np.array(running_mean_10min_values),
        dims=combined_data.dims,
        coords=combined_data.coords,
        name="radial_vel_mean_10min",
    )

    if previous_file is None:
        format_ts = np.datetime_as_string(ds["time"].values[0], unit="h")
        nanout = running_mean_10min.sel(time=slice(None, format_ts + ":05:00"))
        running_mean_10min.loc[dict(time=nanout["time"])] = np.nan

    if next_file is None:
        format_ts = np.datetime_as_string(ds["time"].values[0], unit="h")
        nanout = running_mean_10min.sel(time=slice(format_ts + ":55:00", None))
        running_mean_10min.loc[dict(time=nanout["time"])] = np.nan

    ds["radial_vel_mean_10min"] = running_mean_10min.sel(time=ds["time"])
    ds["wprime"] = ds["radial_velocity"] - ds["radial_vel_mean_10min"]

    output_path = Path(output_dir) / Path(name).name.replace(".cdf", "_with_running_means.cdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)


def lidar_is_raining(lidar: xr.Dataset, radar: xr.Dataset) -> xr.DataArray:
    max_dbz = radar.reflectivity.T[radar.height <= 1500].max(axis=0)
    radar_flag = max_dbz >= 10
    matched = radar_flag.sel(time=lidar.time, method="nearest")
    return xr.DataArray(
        matched.values,
        coords={"time": lidar.time},
        dims="time",
        name="is_raining",
    )


def match_pbl_height(lidar: xr.Dataset, heights_selected: xr.DataArray) -> xr.DataArray:
    lidar_time = lidar.time
    heights_time = heights_selected.time
    nearest_indices = np.searchsorted(heights_time, lidar_time, side="left")
    nearest_indices = np.clip(nearest_indices, 0, len(heights_time) - 1)

    return xr.DataArray(
        heights_selected[nearest_indices].values,
        coords={"time": lidar_time},
        attrs={"long_name": "Boundary layer height nearest in time", "units": "meters"},
        dims="time",
    )


def derive_best_pbl_height(pblh: xr.Dataset) -> xr.DataArray:
    stacked = np.stack([pblh.bl_index_1, pblh.bl_index_2, pblh.bl_index_3])
    valid_mask = ~np.isnan(stacked)
    stacked_filled = np.where(valid_mask, stacked, -np.inf)
    max_indices = np.argmax(stacked_filled, axis=0)
    max_indices[~valid_mask.any(axis=0)] = -1
    heights = np.stack([pblh.bl_height_1, pblh.bl_height_2, pblh.bl_height_3])
    heights_selected = np.where(max_indices != -1, heights[max_indices, np.arange(stacked.shape[1])], np.nan)

    return xr.DataArray(
        heights_selected,
        coords={"time": pblh.bl_height_1.time},
        attrs={"long_name": "Best boundary layer height", "units": "meters"},
        dims="time",
    )


def _process_lidar_time_lidar(
    lidar_time: np.datetime64,
    lidar_ranges: np.ndarray,
    wind_times: np.ndarray,
    z: np.ndarray,
    wind_speed: np.ndarray,
    offset_hours: int = 3,
) -> np.ndarray:
    time_diff = np.abs(wind_times - lidar_time)
    time_idx = np.argmin(time_diff)
    if time_diff[time_idx] > np.timedelta64(offset_hours, "h"):
        return np.full(len(lidar_ranges), np.nan)

    nearest_height_indices = np.abs(z[:, None] - lidar_ranges[None, :]).argmin(axis=0)
    return wind_speed[time_idx][nearest_height_indices]


def get_wind_from_lidar(lidar: xr.Dataset, wind: xr.Dataset, jobs: int = 24) -> xr.DataArray:
    wind_speed = wind.wind_speed.values
    wind_times = wind.time.values
    z = wind.height.values
    lidar_times = lidar.time
    lidar_ranges = lidar.range
    lidar_range_values = lidar_ranges.values
    lidar_time_values = lidar_times.values

    results = Parallel(n_jobs=jobs, prefer=_joblib_prefer())(
        delayed(_process_lidar_time_lidar)(lidar_time, lidar_range_values, wind_times, z, wind_speed)
        for lidar_time in lidar_time_values
    )
    lidar_wind_speed = np.vstack(results)

    return xr.DataArray(
        lidar_wind_speed,
        coords={"time": lidar_times, "range": lidar_ranges},
        attrs={"long_name": "Wind speed", "units": "m/s"},
        dims=["time", "range"],
    )


def build_resampled_columns(
    smoothed_velocity: xr.DataArray,
    intensity: xr.DataArray,
    resample_level: int,
    max_column: int = 67,
    intensity_threshold: float = INTENSITY_THRESHOLD,
) -> dict[str, xr.DataArray]:
    resampled = {}
    intensity500 = intensity[:, resample_level]
    mask_resampled_intensity = np.minimum(intensity500, intensity[:, :max_column]) >= intensity_threshold
    mask_nan_smoothed = ~np.isnan(smoothed_velocity[:, :max_column])
    final_mask = mask_resampled_intensity & mask_nan_smoothed[:, resample_level] & mask_nan_smoothed
    wprime500 = smoothed_velocity[:, resample_level].where(final_mask, np.nan)
    smoothed_velocity[:, :max_column] = smoothed_velocity[:, :max_column].where(final_mask, np.nan)

    for ct, height in enumerate(smoothed_velocity.range[:max_column].values / 1000):
        resampled[str(height)] = wprime500[:, ct]

    return resampled


def detect_chords(
    smoothed_velocity: xr.DataArray,
    height_km: xr.DataArray,
    time_hours: np.ndarray,
    wind_speed: np.ndarray,
    threshold: float = 0.5,
    lidar_pbl_height: xr.DataArray | None = None,
    jobs: int = 24,
) -> pd.DataFrame:
    time_sec = time_hours * 3600
    smoothed_values = smoothed_velocity.values
    height_values = height_km.values
    pbl_values = lidar_pbl_height.values if lidar_pbl_height is not None else None
    updraft_mask = smoothed_values > threshold
    mask = ~np.isnan(smoothed_values)
    valid_height_mask = mask.any(axis=0)
    if not valid_height_mask.any():
        return pd.DataFrame()
    maxz = np.nanmax(height_values[valid_height_mask])
    max_pbl_height = np.nan
    if pbl_values is not None:
        max_pbl_height = np.nanmax(pbl_values)

    def process_height(idh: int, height: float) -> list[dict]:
        records = []
        if height > maxz:
            return records
        if np.isfinite(max_pbl_height) and height > (max_pbl_height / 1000):
            return records

        upeddies, _ = label(updraft_mask[:, idh])
        for eddy in np.unique(upeddies[upeddies >= 1])[1:]:
            t = time_sec[upeddies == eddy]
            dt = t.max() - t.min()
            if dt <= 0:
                continue

            idt = np.where(upeddies == eddy)[0]
            if idt.min() == 0 or idt.max() == smoothed_velocity.shape[0] - 1:
                continue

            center_time = int(np.median(t))
            center_time_index = np.argmin(np.abs(time_sec - center_time))
            if pbl_values is not None and height * 1000 > pbl_values[center_time_index]:
                continue

            cond = np.isnan(smoothed_values[idt.min() - 1, idh]) or np.isnan(smoothed_values[idt.max() + 1, idh])

            record = {
                "Updraft ID": eddy,
                "Center Time": center_time,
                "Height": float(1000 * height),
                "Chord Time": dt,
                "Wind Speed": round(float(wind_speed[center_time_index, idh]), 2),
                "Chord Length": round(float(dt * wind_speed[center_time_index, idh]), 2),
                "Invalid Adjacent": bool(cond),
            }
            if pbl_values is not None:
                record["PBL Height"] = float(pbl_values[center_time_index])
            records.append(record)
        return records

    results = Parallel(n_jobs=jobs, prefer=_joblib_prefer())(
        delayed(process_height)(idh, height) for idh, height in enumerate(height_values)
    )
    return pd.DataFrame([item for sublist in results for item in sublist])


def _detect_resampled_chords_impl(
    resampled_level: xr.DataArray | np.ndarray,
    time_hours: np.ndarray,
    ws500: np.ndarray,
    key: str,
    threshold: float = 0.5,
    lidar_pbl_height: xr.DataArray | None = None,
    min_pbl_height_m: float = 500,
    collect_timing: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, float]]:
    total_start = timer.perf_counter()
    stage_start = timer.perf_counter()
    wprime = resampled_level.values if hasattr(resampled_level, "values") else resampled_level
    time_sec = time_hours * 3600
    updraft_mask = wprime > threshold
    upeddies, _ = label(updraft_mask)
    eddy_ids = np.unique(upeddies[upeddies >= 1])[1:]
    updraft_dicts = []
    pbl_values = lidar_pbl_height.values if lidar_pbl_height is not None else None
    setup_label_s = timer.perf_counter() - stage_start

    if pbl_values is not None:
        max_pbl_height = float(np.nanmax(pbl_values))
        if np.isfinite(max_pbl_height) and max_pbl_height < min_pbl_height_m:
            result = pd.DataFrame(updraft_dicts)
            if collect_timing:
                return result, {
                    "height": float(key),
                    "setup_label_s": setup_label_s,
                    "eddy_loop_s": 0.0,
                    "dataframe_s": 0.0,
                    "total_s": timer.perf_counter() - total_start,
                    "candidate_eddies": float(len(eddy_ids)),
                    "rows": 0.0,
                }
            return result
        if not np.isfinite(max_pbl_height):
            result = pd.DataFrame(updraft_dicts)
            if collect_timing:
                return result, {
                    "height": float(key),
                    "setup_label_s": setup_label_s,
                    "eddy_loop_s": 0.0,
                    "dataframe_s": 0.0,
                    "total_s": timer.perf_counter() - total_start,
                    "candidate_eddies": float(len(eddy_ids)),
                    "rows": 0.0,
                }
            return result

    stage_start = timer.perf_counter()
    for eddy in eddy_ids:
        t = time_sec[upeddies == eddy]
        dt = t.max() - t.min()
        if dt <= 0:
            continue

        idt = np.where(upeddies == eddy)[0]
        if idt.min() == 0 or idt.max() == wprime.shape[0] - 1:
            continue

        center_time = int(np.median(t))
        center_time_index = np.argmin(np.abs(time_sec - center_time))
        if pbl_values is not None and float(key) * 1000 > pbl_values[center_time_index]:
            continue

        cond = np.isnan(wprime[idt.min() - 1]) or np.isnan(wprime[idt.max() + 1])
        record = {
            "Updraft ID": eddy,
            "Center Time": center_time,
            "Chord Time": dt,
            "Height": float(key),
            "Wind Speed": round(float(ws500[center_time_index]), 2),
            "Chord Length": round(float(dt * ws500[center_time_index]), 2),
            "Invalid Adjacent": bool(cond),
        }
        if pbl_values is not None:
            record["PBL Height"] = float(pbl_values[center_time_index])
        updraft_dicts.append(record)
    eddy_loop_s = timer.perf_counter() - stage_start

    stage_start = timer.perf_counter()
    result = pd.DataFrame(updraft_dicts)
    dataframe_s = timer.perf_counter() - stage_start
    if collect_timing:
        return result, {
            "height": float(key),
            "setup_label_s": setup_label_s,
            "eddy_loop_s": eddy_loop_s,
            "dataframe_s": dataframe_s,
            "total_s": timer.perf_counter() - total_start,
            "candidate_eddies": float(len(eddy_ids)),
            "rows": float(len(result)),
        }
    return result


def detect_resampled_chords(
    resampled_level: xr.DataArray | np.ndarray,
    time_hours: np.ndarray,
    ws500: np.ndarray,
    key: str,
    threshold: float = 0.5,
    lidar_pbl_height: xr.DataArray | None = None,
    min_pbl_height_m: float = 500,
) -> pd.DataFrame:
    return _detect_resampled_chords_impl(
        resampled_level,
        time_hours,
        ws500,
        key,
        threshold=threshold,
        lidar_pbl_height=lidar_pbl_height,
        min_pbl_height_m=min_pbl_height_m,
    )


def detect_resampled_chords_with_timing(
    resampled_level: xr.DataArray | np.ndarray,
    time_hours: np.ndarray,
    ws500: np.ndarray,
    key: str,
    threshold: float = 0.5,
    lidar_pbl_height: xr.DataArray | None = None,
    min_pbl_height_m: float = 500,
) -> tuple[pd.DataFrame, dict[str, float]]:
    return _detect_resampled_chords_impl(
        resampled_level,
        time_hours,
        ws500,
        key,
        threshold=threshold,
        lidar_pbl_height=lidar_pbl_height,
        min_pbl_height_m=min_pbl_height_m,
        collect_timing=True,
    )


def smooth_and_qc_velocity(
    lidar: xr.Dataset,
    *,
    radar: xr.Dataset | None = None,
    max_height_km: float = 2,
    intensity_threshold: float = INTENSITY_THRESHOLD,
    sigma: float = 1,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    velocity = lidar["wprime"]
    intensity = lidar["intensity"]
    height_km = lidar["range"] / 1000

    smoothed = deepcopy(velocity)
    smoothed[:] = simple_smoother(velocity, sigma=sigma)

    if radar is not None:
        rain_flag = lidar_is_raining(lidar, radar)
        smoothed[rain_flag == True, :] = np.nan

    smoothed = smoothed.where(intensity >= intensity_threshold, np.nan)
    smoothed = smoothed[:, height_km <= max_height_km]
    intensity = intensity[:, height_km <= max_height_km]
    height_km = height_km[height_km <= max_height_km]
    return smoothed, intensity, height_km
