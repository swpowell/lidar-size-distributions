# Create a ready-to-run Python script that parses the user's Ponca City METAR files,
# filters the two requested conditions, and writes two CSVs.
#
# You can download this file, edit the GLOB pattern to match your local filenames,
# and run:  python pnc_metar_filter.py
# from textwrap import dedent
from pathlib import Path

# script_path = Path("//data/pnc_metar_filter.py")
# code = dedent(r'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PNC METAR Filter
----------------
Reads multiple Ponca City (KPNC) METAR text files, parses timestamps and METAR strings,
and outputs two CSVs:

1) bkn_sct_800_1500_no_lower_no_precip_no_fog.csv
   - No precipitation codes present
   - No fog reported (FG/FZFG)
   - Broken or scattered clouds (BKN/SCT) between 800 and 1500 ft inclusive
   - No clouds beneath 800 ft

2) clear_skies.csv
   - Skies clear via CLR or SKC (or no cloud groups)

USAGE
-----
- Place this script in the same folder as your files, or edit FILE_GLOB below.
- Run:  python pnc_metar_filter.py

NOTES
-----
- The script tries to robustly detect datetimes from several common formats.
- If only the METAR time group (ddhhmmZ) is found but not year/month/day,
  the script will leave the datetime as NaT and still include the line in output.
- If your files include an obvious date field/format not caught by the regexes,
  update the `extract_datetime` function with an additional pattern.
"""
import re
import sys
import csv
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
from datetime import datetime

# ========= CONFIG =========
# Update this to match your local filenames if needed:
FILE_GLOB = "asos-pnc-split*"

# If your files are not in the current directory, put the absolute/relative path here:
DATA_DIR = Path("/thumper/users/scott.powell/code/research-code/lidar/SGP/ASOS/")

# Station expected (optional; set to None to skip enforcing)
EXPECTED_STATION = "KPNC"
# ==========================

# Common precipitation/weather codes to exclude when "no precipitation" is required.
# This list is intentionally broad and includes TS (+ thunder) and SH (showers).
PRECIP_CODES = {
    "RA","DZ","SN","SG","PL","GR","GS","UP","IC","TS","SH","FZRA","FZDZ","FZPL","RASN","SNRA"
}

# Fog codes (any occurrence means fog is present)
FOG_CODES = {"FG","FZFG"}

# Cloud group regex (e.g., SCT030, BKN08, OVC100, FEW015, SKC, CLR)
CLOUD_RE = re.compile(r"\b(?P<type>FEW|SCT|BKN|OVC|SKC|CLR)(?P<base>\d{3})?\b")

# METAR time group like 121853Z (day 12, 18:53 Z)
METAR_TG_RE = re.compile(r"\b(?P<ddhhmm>\d{6})Z\b")

# Flexible datetime patterns that might appear in files
DT_PATTERNS = [
    # 2003-04-12 18:53[:ss]
    re.compile(r"(?P<dt>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?)"),
    # 2003/04/12 18:53[:ss]
    re.compile(r"(?P<dt>\d{4}/\d{2}/\d{2}[ T]\d{2}:\d{2}(?::\d{2})?)"),
    # yyyymmddhhmm (10-12 digits — will try best effort)
    re.compile(r"\b(?P<raw>\d{10,12})\b"),
]

def parse_clouds(metar: str) -> List[Tuple[str, Optional[int]]]:
    """
    Return list of (type, base_hundreds_ft as int or None) cloud layers found in METAR.
    Example: "SCT030 BKN080" -> [("SCT", 30), ("BKN", 80)]
    CLR/SKC -> [("CLR", None)] or [("SKC", None)]
    """
    layers = []
    for m in CLOUD_RE.finditer(metar):
        typ = m.group("type")
        base = m.group("base")
        base_int = int(base) if base is not None else None
        layers.append((typ, base_int))
    return layers

def has_precip(metar: str) -> bool:
    tokens = re.findall(r"\+?-?[A-Z]{2,6}", metar)
    for t in tokens:
        t_clean = t.replace("+","").replace("-","")
        if t_clean in PRECIP_CODES:
            return True
    return False

def has_fog(metar: str) -> bool:
    return any(code in metar.split() for code in FOG_CODES)

def meets_bkn_sct_criteria(metar: str) -> bool:
    """
    - Has SCT or BKN layer with 800–1500 ft (8–15 in hundreds of ft)
    - No clouds below 800 ft (i.e., no layer with base < 8)
    """
    layers = parse_clouds(metar)
    if not layers:
        return False

    # Any lower-than-800 cloud?
    for typ, base in layers:
        if base is not None and base < 8:
            return False

    # Any SCT/BKN layer in [8, 15]?
    for typ, base in layers:
        if typ in {"SCT","BKN"} and base is not None and 8 <= base <= 15:
            return True
    return False

def is_clear(metar: str) -> bool:
    """
    Accept CLR or SKC as clear skies.
    Some AWOS stations use CLR to mean 'no clouds below 12k ft'.
    """
    layers = parse_clouds(metar)
    if not layers:
        # If truly no cloud codes found, consider it clear as well.
        return True
    types = {t for t,_ in layers}
    return ("CLR" in types) or ("SKC" in types)

def extract_datetime(line: str, metar: str) -> Optional[pd.Timestamp]:
    """
    Try to extract an absolute datetime from the line content.
    If we only find the METAR time group (ddhhmmZ) and also find a YYYY-MM-DD or YYYY/MM/DD date,
    we combine them. Otherwise, return NaT.
    """
    # Try direct datetime patterns
    for pat in DT_PATTERNS[:2]:  # ISO-like first
        m = pat.search(line)
        if m:
            try:
                return pd.to_datetime(m.group("dt"))
            except Exception:
                pass

    # Find a date token if present
    date_token = None
    mdate_iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", line)
    mdate_slash = re.search(r"\b(\d{4}/\d{2}/\d{2})\b", line)
    if mdate_iso:
        date_token = mdate_iso.group(1)
    elif mdate_slash:
        date_token = mdate_slash.group(1).replace("/", "-")

    # Find METAR time group
    tg = METAR_TG_RE.search(metar)
    if tg and date_token:
        ddhhmm = tg.group("ddhhmm")
        day = int(ddhhmm[0:2])
        hour = int(ddhhmm[2:4])
        minute = int(ddhhmm[4:6])
        # Combine with date_token's year-month; replace day/hour/minute
        try:
            base = pd.to_datetime(date_token)
            # If day mismatches across month boundary, we still force it.
            # This is a best-effort since some files might carry day in the ddhhmmZ only.
            out = base.replace(day=day, hour=hour, minute=minute, second=0)
            return out
        except Exception:
            return pd.NaT

    # Lastly, attempt yyyymmddhhmm raw
    mraw = DT_PATTERNS[2].search(line)
    if mraw:
        raw = mraw.group("raw")
        try:
            # Try multiple lengths; pad seconds if needed
            if len(raw) == 12:
                return pd.to_datetime(raw, format="%Y%m%d%H%M")
            elif len(raw) == 10:
                return pd.to_datetime(raw, format="%Y%m%d%H")
            else:
                # Fallback
                return pd.to_datetime(raw)
        except Exception:
            pass

    return pd.NaT

def extract_metar(line: str) -> Optional[str]:
    """
    Try to pull the METAR string from the line.
    Strategy:
    - Find token 'KPNC' (or EXPECTED_STATION), take substring from there to end.
    - If not found, look for ddhhmmZ and then backtrack to previous station-like code.
    """
    station = EXPECTED_STATION
    if station and station in line:
        # Start from station appearance closest to a METAR-like token
        start = line.index(station)
        candidate = line[start:].strip()
        # If the candidate contains commas, take last comma-separated field with station
        # else return the substring
        return candidate

    # Fallback: try to find any station-like token (Kxxx)
    mstation = re.search(r"\bK[A-Z0-9]{3}\b", line)
    if mstation:
        start = mstation.start()
        return line[start:].strip()

    # As a last resort, if it looks like a METAR (contains ddhhmmZ), return entire line
    if METAR_TG_RE.search(line):
        return line.strip()

    return None

def line_to_record(line: str) -> Optional[dict]:
    metar = extract_metar(line)
    if not metar:
        return None
    dt = extract_datetime(line, metar)
    return {
        "datetime": dt,
        "metar": metar
    }

def read_all_files(paths: List[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = line_to_record(line)
                    if rec:
                        rows.append(rec)
        except Exception as e:
            print(f"Error reading {p}: {e}", file=sys.stderr)
    df = pd.DataFrame(rows)
    # Drop duplicates & sort
    if not df.empty:
        df = df.drop_duplicates(subset=["datetime","metar"])
        # It's okay if datetime has NaT; sort will put NaT last
        df = df.sort_values(by=["datetime","metar"], kind="stable")
    return df

def main():
    paths = sorted(DATA_DIR.glob(FILE_GLOB))
    if not paths:
        print(f"No files found under {DATA_DIR} matching {FILE_GLOB}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {len(paths)} files...")
    df = read_all_files(paths)
    if df.empty:
        print("No parsable lines found.", file=sys.stderr)
        sys.exit(2)

    # Filter A: no precip, no fog, SCT/BKN 800–1500 and no clouds below 800
    cond_a_mask = df["metar"].apply(lambda m: (not has_precip(m)) and (not has_fog(m)) and meets_bkn_sct_criteria(m))

    # Filter B: clear skies
    clear_mask = df["metar"].apply(is_clear)

    out_a = df.loc[cond_a_mask].copy()
    out_a["condition"] = "bkn/sct 800–1500, no lower clouds, no precip, no fog"

    out_b = df.loc[clear_mask].copy()
    out_b["condition"] = "clear"

    # Write CSVs
    out_a_path = Path("bkn_sct_800_1500_no_lower_no_precip_no_fog.csv")
    # out_b_path = Path("clear_skies.csv")

    out_a.to_csv(out_a_path, index=False, quoting=csv.QUOTE_MINIMAL)
    # out_b.to_csv(out_b_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Wrote: {out_a_path.resolve()}  ({len(out_a)} rows)")
    # print(f"Wrote: {out_b_path.resolve()}  ({len(out_b)} rows)")

if __name__ == "__main__":
    main()
