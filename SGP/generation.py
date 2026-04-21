from __future__ import annotations

import datetime as dte
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from .config import GenerationPaths
from .detection import (
    build_resampled_columns,
    compute_velocity_running_mean_file,
    derive_best_pbl_height,
    detect_chords,
    detect_resampled_chords,
    get_date_str,
    get_wind_from_lidar,
    match_pbl_height,
    smooth_and_qc_velocity,
)


DATE_RE = re.compile(r"\.(\d{8})\.\d{6}\.")


def _extract_date(path: str | Path) -> str:
    match = DATE_RE.search(str(path))
    if not match:
        raise ValueError(f"Could not extract date from {path}")
    return match.group(1)


def _process_running_mean_file(index: int, names: list[str], output_dir: Path) -> None:
    previous_file = names[index - 1] if index > 0 else None
    next_file = names[index + 1] if index < len(names) - 1 else None
    compute_velocity_running_mean_file(names[index], previous_file, next_file, output_dir)


def build_running_mean_files(lidar_files: list[str], output_dir: Path, jobs: int = 1) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    Parallel(n_jobs=jobs)(
        delayed(_process_running_mean_file)(index, lidar_files, output_dir) for index in range(len(lidar_files))
    )


def _load_or_build_windspeed(
    lidar: xr.Dataset,
    windfiles: list[str],
    windspeed_path: Path,
    jobs: int = 24,
) -> np.ndarray:
    windspeed_path.parent.mkdir(parents=True, exist_ok=True)
    if windspeed_path.exists():
        windspeed = xr.open_dataset(windspeed_path)
        return windspeed.__xarray_dataarray_variable__.values

    wind = xr.open_mfdataset(windfiles)
    windspeed = get_wind_from_lidar(lidar, wind, jobs=jobs)
    encoding = {"range": {"_FillValue": -9999.0}}
    windspeed.to_netcdf(windspeed_path, encoding=encoding)
    return windspeed.values


def process_day(
    *,
    dayfiles: list[str],
    windfiles: list[str],
    date_str: str,
    resample_level: int,
    regular_output_dir: Path,
    resampled_output_dir: Path,
    windspeed_dir: Path,
    pblhfile: str | None = None,
    kazrfile: str | None = None,
    max_height_km: float = 2,
    threshold: float = 0.5,
    jobs: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lidar = xr.open_mfdataset(dayfiles)
    time = lidar.variables["time"][:]
    time_hours = ((time - time[0]).values / 1e9).astype(float) / 3600

    radar = xr.open_dataset(kazrfile) if kazrfile else None
    smoothed_v, intensity, height_km = smooth_and_qc_velocity(
        lidar,
        radar=radar,
        max_height_km=max_height_km,
    )

    resampled = build_resampled_columns(smoothed_v, intensity, resample_level=resample_level)

    lidar_pbl_height = None
    if pblhfile:
        pblh = xr.open_dataset(pblhfile)
        lidar_pbl_height = match_pbl_height(lidar, derive_best_pbl_height(pblh))

    windspeed_path = windspeed_dir / f"windspeed_{date_str}.cdf"
    ws = _load_or_build_windspeed(lidar, windfiles, windspeed_path, jobs=jobs)

    regular = detect_chords(
        smoothed_v,
        height_km,
        time_hours,
        ws,
        threshold=threshold,
        lidar_pbl_height=lidar_pbl_height,
        jobs=jobs,
    )
    resampled_frames = Parallel(n_jobs=jobs)(
        delayed(detect_resampled_chords)(
            resampled[key],
            time_hours,
            ws[:, resample_level],
            key,
            threshold=threshold,
            lidar_pbl_height=lidar_pbl_height,
        )
        for key in resampled
    )
    resampled_df = pd.concat(resampled_frames, ignore_index=True) if resampled_frames else pd.DataFrame()

    base_date = dte.datetime.strptime(date_str, "%Y%m%d")
    for frame in (regular, resampled_df):
        if frame.empty:
            continue
        frame["Center Datetime"] = frame["Center Time"].apply(lambda x: base_date + dte.timedelta(seconds=float(x)))
        frame.set_index("Center Datetime", inplace=True)

    regular_output_dir.mkdir(parents=True, exist_ok=True)
    resampled_output_dir.mkdir(parents=True, exist_ok=True)
    regular.to_csv(regular_output_dir / f"updrafts_{date_str}.csv")
    resampled_df.to_csv(resampled_output_dir / f"updrafts_resample_{date_str}.csv")

    return regular, resampled_df


def generate_daily_event_tables(
    paths: GenerationPaths,
    *,
    compute_running_means: bool = False,
    jobs: int = 1,
    resample_level: int = 26,
    threshold: float = 0.5,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[str]:
    lidar_files = sorted(str(path) for path in paths.lidar_dir.glob("*.cdf"))
    if compute_running_means:
        build_running_mean_files(lidar_files, paths.running_mean_dir, jobs=jobs)

    running_mean_files = sorted(str(path) for path in paths.running_mean_dir.glob("*.cdf"))
    dates = sorted({_extract_date(path) for path in lidar_files})
    if start_date:
        dates = [date for date in dates if date >= start_date]
    if end_date:
        dates = [date for date in dates if date <= end_date]

    radar_files = sorted(str(path) for path in (paths.radar_dir.glob("*.nc") if paths.radar_dir else []))
    pblh_files = sorted(
        str(path) for path in ((list(paths.pblh_dir.glob("*.cdf")) + list(paths.pblh_dir.glob("*.nc"))) if paths.pblh_dir else [])
    )
    wind_files = sorted(str(path) for path in paths.wind_dir.glob("*.nc"))

    processed = []
    for date_str in dates:
        print(f"Now processing {date_str}")
        date = dte.datetime.strptime(date_str, "%Y%m%d")
        day_before = get_date_str(date - dte.timedelta(days=1))
        day_after = get_date_str(date + dte.timedelta(days=1))

        dayfiles = [path for path in running_mean_files if date_str in path]
        if not dayfiles:
            print(f"Skipping {date_str}: no running-mean lidar files found.")
            continue

        kazrfile = next((path for path in radar_files if date_str in path), None)
        pblhfile = next((path for path in pblh_files if date_str in path), None)
        nearby_wind = [path for path in wind_files if date_str in path or day_before in path or day_after in path]
        if not nearby_wind:
            print(f"Skipping {date_str}: no wind files found.")
            continue

        try:
            process_day(
                dayfiles=dayfiles,
                windfiles=nearby_wind,
                date_str=date_str,
                resample_level=resample_level,
                regular_output_dir=paths.regular_output_dir,
                resampled_output_dir=paths.resampled_output_dir,
                windspeed_dir=paths.windspeed_dir,
                pblhfile=pblhfile,
                kazrfile=kazrfile,
                threshold=threshold,
                jobs=jobs,
            )
            processed.append(date_str)
        except Exception as exc:  # pragma: no cover - operational logging
            print(f"Something went wrong in processing {date_str}: {exc}")

    return processed
