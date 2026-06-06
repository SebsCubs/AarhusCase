"""Parse district-heating (Varme) meter XLSX files.

Files in Jettesvej2Brabrand/: Varme 5339339.xlsx, 5343901.xlsx, 5344947.xlsx,
5344951.xlsx, 5344963.xlsx, 5347207.xlsx, 5347227.xlsx

Columns vary in order but always include some combination of:
  Periode, Tilbageløbstemperatur, Fremløbstemperatur, Energi [MWh], Volumen [m³]

Computes hourly power [kW] as diff(Energi[MWh]) * 1000 / Δh.
Returns dict {meter_id: DataFrame} with columns:
  T_sup, T_ret, energy_MWh, volume_m3, power_kW
"""
import os
import re
import glob
import pandas as pd

VARME_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "Jettesvej2Brabrand"
)

_COL_PATTERNS = {
    "T_sup": re.compile(r"fremløb", re.IGNORECASE),
    "T_ret": re.compile(r"tilbageløb", re.IGNORECASE),
    "energy_MWh": re.compile(r"energi", re.IGNORECASE),
    "volume_m3": re.compile(r"volumen", re.IGNORECASE),
    "time": re.compile(r"periode", re.IGNORECASE),
}


def _detect_columns(df: pd.DataFrame) -> dict:
    mapping = {}
    for target, pat in _COL_PATTERNS.items():
        for col in df.columns:
            if pat.search(str(col)):
                mapping[target] = col
                break
    return mapping


def _parse_excel_date(series: pd.Series) -> pd.Series:
    """Handle both datetime strings and Excel serial floats."""
    if pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series):
        return pd.to_datetime("1899-12-30") + pd.to_timedelta(series, unit="D")
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def load_single(path: str, tz: str = "Europe/Copenhagen") -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    col_map = _detect_columns(df)
    if "time" not in col_map:
        raise ValueError(f"No time column found in {path}")

    df["_time"] = _parse_excel_date(df[col_map["time"]])
    df = df.dropna(subset=["_time"]).set_index("_time").sort_index()
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize(
            tz, ambiguous="NaT", nonexistent="shift_forward"
        )
        df = df[df.index.notna()]

    out = pd.DataFrame(index=df.index)
    for target, src_col in col_map.items():
        if target == "time":
            continue
        out[target] = pd.to_numeric(df[src_col], errors="coerce")

    # Compute power [kW] from energy diff
    if "energy_MWh" in out.columns:
        dt_hours = out.index.to_series().diff().dt.total_seconds() / 3600
        out["power_kW"] = out["energy_MWh"].diff() * 1000 / dt_hours
        out["power_kW"] = out["power_kW"].clip(lower=0)

    return out


def load_all(varme_dir: str = VARME_DIR, tz: str = "Europe/Copenhagen") -> dict:
    result = {}
    for path in glob.glob(os.path.join(varme_dir, "Varme *.xlsx")):
        meter_id = re.search(r"Varme (\d+)", os.path.basename(path)).group(1)
        try:
            result[meter_id] = load_single(path, tz)
        except Exception as e:
            print(f"Warning: could not load {path}: {e}")
    return result
