"""Parse electricity (EL) meter XLSX files.

Files in Jettesvej2Brabrand/: EL 19397045.xlsx, EL 19591177.xlsx, EL 19591181.xlsx

Key column: E17 (forbrugt fra net) [kWh] — grid consumption.
Some files also have D06 (leveret til net) [kWh] — feed-in.

Returns dict {meter_id: DataFrame} with columns: power_kW, feed_in_kW (optional).
"""
import os
import re
import glob
import pandas as pd
from .varme_meter import _parse_excel_date

EL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "Jettesvej2Brabrand"
)

_CONSUMPTION_RE = re.compile(r"E17|forbrugt", re.IGNORECASE)
_FEED_IN_RE = re.compile(r"D06|leveret", re.IGNORECASE)
_TIME_RE = re.compile(r"tidspunkt|periode", re.IGNORECASE)


def load_single(path: str, tz: str = "Europe/Copenhagen") -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")

    time_col = next((c for c in df.columns if _TIME_RE.search(str(c))), None)
    if time_col is None:
        raise ValueError(f"No time column in {path}")

    df["_time"] = _parse_excel_date(df[time_col])
    df = df.dropna(subset=["_time"]).set_index("_time").sort_index()
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize(
            tz, ambiguous="NaT", nonexistent="shift_forward"
        )
        df = df[df.index.notna()]

    out = pd.DataFrame(index=df.index)

    cons_col = next((c for c in df.columns if _CONSUMPTION_RE.search(str(c))), None)
    if cons_col:
        energy_kwh = pd.to_numeric(df[cons_col], errors="coerce")
        dt_hours = df.index.to_series().diff().dt.total_seconds() / 3600
        out["power_kW"] = energy_kwh.diff() / dt_hours
        out["power_kW"] = out["power_kW"].clip(lower=0)

    feed_col = next((c for c in df.columns if _FEED_IN_RE.search(str(c))), None)
    if feed_col:
        energy_kwh = pd.to_numeric(df[feed_col], errors="coerce")
        dt_hours = df.index.to_series().diff().dt.total_seconds() / 3600
        out["feed_in_kW"] = energy_kwh.diff() / dt_hours
        out["feed_in_kW"] = out["feed_in_kW"].clip(lower=0)

    return out


def load_all(el_dir: str = EL_DIR, tz: str = "Europe/Copenhagen") -> dict:
    result = {}
    for path in glob.glob(os.path.join(el_dir, "EL *.xlsx")):
        meter_id = re.search(r"EL (\d+)", os.path.basename(path)).group(1)
        try:
            result[meter_id] = load_single(path, tz)
        except Exception as e:
            print(f"Warning: could not load {path}: {e}")
    return result
