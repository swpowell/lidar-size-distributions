from __future__ import annotations

import datetime as dte
import os
import re
import time as timer
import traceback
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
    detect_resampled_chords_with_timing,
    get_date_str,
    get_wind_from_lidar,
    match_pbl_height,
    smooth_and_qc_velocity,
)


DATE_RE = re.compile(r"\.(\d{8})\.\d{6}\.")


def _joblib_prefer() -> str | None:
    return os.environ.get("LIDAR_JOBLIB_PREFER") or None


def _extract_date(path: str | Path) -> str:
    match = DATE_RE.search(str(path))
    if not match:
        raise ValueError(f"Could not extract date from {path}")
    return match.group(1)


def _write_event_table(dataframe: pd.DataFrame, output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = dataframe.reset_index() if dataframe.index.name else dataframe
    if output_format == "parquet":
        table.to_parquet(output_path, index=False)
    elif output_format == "csv":
        table.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def _log_resampled_profile(
    date_str: str,
    results: list[tuple[pd.DataFrame, dict[str, float]]],
    wall_time_s: float,
    slowest_count: int = 5,
) -> None:
    if not results:
        print(f"[{date_str}] resampled_profile: no columns processed", flush=True)
        return

    timings = [timing for _, timing in results]
    setup_label_s = sum(item["setup_label_s"] for item in timings)
    eddy_loop_s = sum(item["eddy_loop_s"] for item in timings)
    dataframe_s = sum(item["dataframe_s"] for item in timings)
    worker_total_s = sum(item["total_s"] for item in timings)
    rows = sum(item["rows"] for item in timings)
    candidates = sum(item["candidate_eddies"] for item in timings)
    print(
        f"[{date_str}] resampled_profile: wall={wall_time_s:.2f}s "
        f"worker_total={worker_total_s:.2f}s setup_label={setup_label_s:.2f}s "
        f"eddy_loop={eddy_loop_s:.2f}s dataframe={dataframe_s:.2f}s "
        f"candidate_eddies={int(candidates)} rows={int(rows)}",
        flush=True,
    )

    slowest = sorted(timings, key=lambda item: item["total_s"], reverse=True)[:slowest_count]
    slowest_text = ", ".join(
        f"{item['height']:.3f}km:{item['total_s']:.2f}s/{int(item['candidate_eddies'])}cand/{int(item['rows'])}rows"
        for item in slowest
    )
    print(f"[{date_str}] resampled_slowest: {slowest_text}", flush=True)


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
    methods: tuple[str, ...] = ("regular", "resampled"),
    output_format: str = "csv",
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

    regular = pd.DataFrame()
    if "regular" in methods:
        regular = detect_chords(
            smoothed_v,
            height_km,
            time_hours,
            ws,
            threshold=threshold,
            lidar_pbl_height=lidar_pbl_height,
            jobs=jobs,
        )

    resampled_df = pd.DataFrame()
    if "resampled" in methods:
        dispatch_start = timer.perf_counter()
        resampled_results = [
            detect_resampled_chords_with_timing(
                resampled[key].values,
                time_hours,
                ws[:, resample_level],
                key,
                threshold=threshold,
                lidar_pbl_height=lidar_pbl_height,
            )
            for key in resampled
        ]
        resampled_frames = [frame for frame, _ in resampled_results]
        resampled_df = pd.concat(resampled_frames, ignore_index=True) if resampled_frames else pd.DataFrame()
        _log_resampled_profile(date_str, resampled_results, timer.perf_counter() - dispatch_start)

    base_date = dte.datetime.strptime(date_str, "%Y%m%d")
    for frame in (regular, resampled_df):
        if frame.empty:
            continue
        frame["Center Datetime"] = frame["Center Time"].apply(lambda x: base_date + dte.timedelta(seconds=float(x)))
        frame.set_index("Center Datetime", inplace=True)

    suffix = "parquet" if output_format == "parquet" else "csv"
    if "regular" in methods:
        _write_event_table(regular, regular_output_dir / f"updrafts_{date_str}.{suffix}", output_format)
    if "resampled" in methods:
        _write_event_table(
            resampled_df,
            resampled_output_dir / f"updrafts_resample_{date_str}.{suffix}",
            output_format,
        )

    return regular, resampled_df


def _process_date_task(task: dict[str, object]) -> str | None:
    date_str = str(task["date_str"])
    print(f"Now processing {date_str}", flush=True)
    try:
        process_day(
            dayfiles=task["dayfiles"],  # type: ignore[arg-type]
            windfiles=task["windfiles"],  # type: ignore[arg-type]
            date_str=date_str,
            resample_level=int(task["resample_level"]),
            regular_output_dir=task["regular_output_dir"],  # type: ignore[arg-type]
            resampled_output_dir=task["resampled_output_dir"],  # type: ignore[arg-type]
            windspeed_dir=task["windspeed_dir"],  # type: ignore[arg-type]
            pblhfile=task["pblhfile"],  # type: ignore[arg-type]
            kazrfile=task["kazrfile"],  # type: ignore[arg-type]
            threshold=float(task["threshold"]),
            jobs=int(task["per_date_jobs"]),
            methods=task["methods"],  # type: ignore[arg-type]
            output_format=str(task["output_format"]),
        )
        return date_str
    except Exception as exc:  # pragma: no cover - operational logging
        print(f"Something went wrong in processing {date_str}: {exc}", flush=True)
        traceback.print_exc()
        return None


def generate_daily_event_tables(
    paths: GenerationPaths,
    *,
    compute_running_means: bool = False,
    jobs: int = 1,
    resample_level: int = 26,
    threshold: float = 0.5,
    start_date: str | None = None,
    end_date: str | None = None,
    methods: tuple[str, ...] = ("regular", "resampled"),
    output_format: str = "csv",
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

    per_date_jobs = 1
    date_tasks = []
    for date_str in dates:
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

        date_tasks.append(
            {
                "date_str": date_str,
                "dayfiles": dayfiles,
                "windfiles": nearby_wind,
                "resample_level": resample_level,
                "regular_output_dir": paths.regular_output_dir,
                "resampled_output_dir": paths.resampled_output_dir,
                "windspeed_dir": paths.windspeed_dir,
                "pblhfile": pblhfile,
                "kazrfile": kazrfile,
                "threshold": threshold,
                "per_date_jobs": per_date_jobs,
                "methods": methods,
                "output_format": output_format,
            }
        )

    if not date_tasks:
        return []

    print(
        f"Processing {len(date_tasks)} dates with {jobs} date workers "
        f"and {per_date_jobs} worker per date.",
        flush=True,
    )
    if jobs == 1:
        results = [_process_date_task(task) for task in date_tasks]
    else:
        results = Parallel(n_jobs=jobs, prefer=_joblib_prefer())(
            delayed(_process_date_task)(task) for task in date_tasks
        )

    return [date_str for date_str in results if date_str is not None]
