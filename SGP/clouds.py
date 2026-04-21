from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


PRECIP_CODES = {
    "RA",
    "DZ",
    "SN",
    "SG",
    "PL",
    "GR",
    "GS",
    "UP",
    "IC",
    "TS",
    "SH",
    "FZRA",
    "FZDZ",
    "FZPL",
    "RASN",
    "SNRA",
}
FOG_CODES = {"FG", "FZFG"}
CLOUD_RE = re.compile(r"\b(?P<type>FEW|SCT|BKN|OVC|SKC|CLR)(?P<base>\d{3})?\b")
METAR_TG_RE = re.compile(r"\b(?P<ddhhmm>\d{6})Z\b")
DT_PATTERNS = [
    re.compile(r"(?P<dt>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?)"),
    re.compile(r"(?P<dt>\d{4}/\d{2}/\d{2}[ T]\d{2}:\d{2}(?::\d{2})?)"),
    re.compile(r"\b(?P<raw>\d{10,12})\b"),
]


def parse_clouds(metar: str) -> list[tuple[str, int | None]]:
    layers = []
    for match in CLOUD_RE.finditer(metar):
        cloud_type = match.group("type")
        base = match.group("base")
        layers.append((cloud_type, int(base) if base is not None else None))
    return layers


def has_precip(metar: str) -> bool:
    tokens = re.findall(r"\+?-?[A-Z]{2,6}", metar)
    return any(token.replace("+", "").replace("-", "") in PRECIP_CODES for token in tokens)


def has_fog(metar: str) -> bool:
    return any(code in metar.split() for code in FOG_CODES)


def meets_bkn_sct_criteria(metar: str, base_min_hundreds_ft: int = 8, base_max_hundreds_ft: int = 15) -> bool:
    layers = parse_clouds(metar)
    if not layers:
        return False

    for _, base in layers:
        if base is not None and base < base_min_hundreds_ft:
            return False

    for cloud_type, base in layers:
        if cloud_type in {"SCT", "BKN"} and base is not None and base_min_hundreds_ft <= base <= base_max_hundreds_ft:
            return True
    return False


def is_clear(metar: str) -> bool:
    layers = parse_clouds(metar)
    if not layers:
        return True
    layer_types = {cloud_type for cloud_type, _ in layers}
    return "CLR" in layer_types or "SKC" in layer_types


def extract_datetime(line: str, metar: str) -> pd.Timestamp:
    for pattern in DT_PATTERNS[:2]:
        match = pattern.search(line)
        if match:
            try:
                return pd.to_datetime(match.group("dt"), utc=True)
            except Exception:
                pass

    date_token = None
    match_iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", line)
    match_slash = re.search(r"\b(\d{4}/\d{2}/\d{2})\b", line)
    if match_iso:
        date_token = match_iso.group(1)
    elif match_slash:
        date_token = match_slash.group(1).replace("/", "-")

    time_group = METAR_TG_RE.search(metar)
    if time_group and date_token:
        ddhhmm = time_group.group("ddhhmm")
        day = int(ddhhmm[0:2])
        hour = int(ddhhmm[2:4])
        minute = int(ddhhmm[4:6])
        try:
            base = pd.to_datetime(date_token, utc=True)
            return base.replace(day=day, hour=hour, minute=minute, second=0)
        except Exception:
            return pd.NaT

    raw_match = DT_PATTERNS[2].search(line)
    if raw_match:
        raw = raw_match.group("raw")
        try:
            if len(raw) == 12:
                return pd.to_datetime(raw, format="%Y%m%d%H%M", utc=True)
            if len(raw) == 10:
                return pd.to_datetime(raw, format="%Y%m%d%H", utc=True)
            return pd.to_datetime(raw, utc=True)
        except Exception:
            pass

    return pd.NaT


def extract_metar(line: str, expected_station: str = "KPNC") -> Optional[str]:
    if expected_station and expected_station in line:
        return line[line.index(expected_station) :].strip()

    station_match = re.search(r"\bK[A-Z0-9]{3}\b", line)
    if station_match:
        return line[station_match.start() :].strip()

    if METAR_TG_RE.search(line):
        return line.strip()
    return None


def line_to_record(line: str, expected_station: str = "KPNC") -> Optional[dict]:
    metar = extract_metar(line, expected_station=expected_station)
    if not metar:
        return None
    return {"datetime": extract_datetime(line, metar), "metar": metar}


def read_all_metar_files(paths: list[Path], expected_station: str = "KPNC") -> pd.DataFrame:
    rows = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    record = line_to_record(stripped, expected_station=expected_station)
                    if record:
                        rows.append(record)
        except Exception as exc:  # pragma: no cover - operational logging
            print(f"Error reading {path}: {exc}", file=sys.stderr)

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return dataframe
    return dataframe.drop_duplicates(subset=["datetime", "metar"]).sort_values(by=["datetime", "metar"], kind="stable")


def filter_cloud_catalog(
    dataframe: pd.DataFrame,
    *,
    cloudy_base_min_hundreds_ft: int = 8,
    cloudy_base_max_hundreds_ft: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cloudy_mask = dataframe["metar"].apply(
        lambda metar: (not has_precip(metar))
        and (not has_fog(metar))
        and meets_bkn_sct_criteria(
            metar,
            base_min_hundreds_ft=cloudy_base_min_hundreds_ft,
            base_max_hundreds_ft=cloudy_base_max_hundreds_ft,
        )
    )
    clear_mask = dataframe["metar"].apply(is_clear)

    cloudy = dataframe.loc[cloudy_mask].copy()
    cloudy["condition"] = "cloudy"
    clear = dataframe.loc[clear_mask].copy()
    clear["condition"] = "clear"
    return cloudy, clear


def write_condition_catalogs(cloudy: pd.DataFrame, clear: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cloudy_path = output_dir / "bkn_sct_800_1500_no_lower_no_precip_no_fog.csv"
    clear_path = output_dir / "clear_skies.csv"
    cloudy.to_csv(cloudy_path, index=False, quoting=csv.QUOTE_MINIMAL)
    clear.to_csv(clear_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return cloudy_path, clear_path
