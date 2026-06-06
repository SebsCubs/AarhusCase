"""Parse the Skoven BMS controller XLSX export.

Source: Jettesvej2Brabrand/PredictiveOptimalControlAarhus/Jettesvej 10ASkoven-144523-30.3.2026.csv.xlsx
Columns: Tidsstempel, Udetemperatur[°C], Rumtemp.(Kr.1)[°C], Rumtemp.ref.(Kr.1)[°C],
         Fremløbstemp.(Kr.1)[°C], Fremløbstemp.ref.(Kr.1)[°C],
         Returtemp.(Kr.1)[°C], Returtemp.ref.(Kr.1)[°C]
"""
import os
import re
import pandas as pd
from .danish_decode import fix_columns, replace_nulls

BMS_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..",
    "Jettesvej2Brabrand",
    "PredictiveOptimalControlAarhus",
    "Jettesvej 10ASkoven-144523-30.3.2026.csv.xlsx",
)

COLUMN_MAP = {
    "Tidsstempel": "time",
    "Udetemperatur[°C]": "T_oa",
    "Udetemperatur[Â°C]": "T_oa",
    "Rumtemp.(Kr.1)[°C]": "T_zone_bms",
    "Rumtemp.(Kr.1)[Â°C]": "T_zone_bms",
    "Rumtemp.ref.(Kr.1)[°C]": "T_zone_set",
    "Rumtemp.ref.(Kr.1)[Â°C]": "T_zone_set",
    "Fremløbstemp.(Kr.1)[°C]": "T_sup_w",
    "Freml\u00f8bstemp.(Kr.1)[°C]": "T_sup_w",
    "Freml\u00f8bstemp.(Kr.1)[Â°C]": "T_sup_w",
    "Fremløbstemp.ref.(Kr.1)[°C]": "T_sup_w_set",
    "Freml\u00f8bstemp.ref.(Kr.1)[°C]": "T_sup_w_set",
    "Freml\u00f8bstemp.ref.(Kr.1)[Â°C]": "T_sup_w_set",
    "Returtemp.(Kr.1)[°C]": "T_ret_w",
    "Returtemp.(Kr.1)[Â°C]": "T_ret_w",
    "Returtemp.ref.(Kr.1)[°C]": "T_ret_w_set",
    "Returtemp.ref.(Kr.1)[Â°C]": "T_ret_w_set",
}


def load(path: str = BMS_PATH, tz: str = "Europe/Copenhagen") -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df = fix_columns(df)
    df = replace_nulls(df)

    # Rename columns — whitespace-insensitive match (real headers have a space
    # before "(Kr.1)" that COLUMN_MAP keys omit).
    def _norm(s: str) -> str:
        return re.sub(r"\s+", "", str(s))

    norm_map = {_norm(k): v for k, v in COLUMN_MAP.items()}
    df = df.rename(columns={c: norm_map.get(_norm(c), c) for c in df.columns})

    # Parse time and set index
    df["time"] = pd.to_datetime(df["time"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.set_index("time")
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize(
            tz, ambiguous="NaT", nonexistent="shift_forward"
        )
    else:
        df.index = df.index.tz_convert(tz)
    df = df[df.index.notna()]

    # Coerce numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_index()
    return df
