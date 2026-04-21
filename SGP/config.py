from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenerationPaths:
    lidar_dir: Path
    radar_dir: Path | None
    pblh_dir: Path | None
    wind_dir: Path
    running_mean_dir: Path
    windspeed_dir: Path
    regular_output_dir: Path
    resampled_output_dir: Path


@dataclass(frozen=True)
class DaylightConfig:
    latitude: float = 36.605
    longitude: float = -97.485
    timezone: str = "US/Central"
    name: str = "SGP"
    region: str = "USA"


SGP_C1_DEFAULTS = GenerationPaths(
    lidar_dir=Path("/thumper/metdata/doppler-lidar/SGP/C1/vert/"),
    radar_dir=Path("/thumper/metdata/radar/KAZR_SGP/"),
    pblh_dir=Path("/thumper/metdata/PBLH/SGP/"),
    wind_dir=Path("/thumper/metdata/doppler-lidar/SGP/C1/wind/"),
    running_mean_dir=Path("/thumper/metdata/doppler-lidar/SGP_withrunningmeans/C1/"),
    windspeed_dir=Path("/thumper/users/scott.powell/code-data/research-code/lidar/SGP/C1_windspeed_lidar/"),
    regular_output_dir=Path("/thumper/users/scott.powell/code-data/research-code/lidar/SGP/C1/"),
    resampled_output_dir=Path("/thumper/users/scott.powell/code-data/research-code/lidar/SGP/C1_resample/"),
)
