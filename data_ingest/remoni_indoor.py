"""Parse ReMoni indoor sensor CSVs for Skoven.

Two formats:
  1. metric_1/2/3.csv — long format: one row per sensor report, tagged by 'name'.
     Columns: time, metric, co2, name, temperature, humidity, light_level
  2. metric_sensors_8_9_10_11.csv — wide format: 'metric' column contains literal
     payload like "CO2: 450 ppm, Temperatur: 20.5 °C, Humidity: 39 %, Lysniveau: 243 lux"
     along with separate numeric columns.

Output: wide DataFrame with columns {t_{sensor}, co2_{sensor}, rh_{sensor}} per sensor.
"""
import os
import re
import pandas as pd

SKOVEN_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "Jettesvej2Brabrand", "Skoven"
)

LONG_FORMAT_FILES = [
    "Jettevej 10 - Skoven-metric_1.csv",
    "Jettevej 10 - Skoven-metric_2.csv",
    "Jettevej 10 - Skoven-metric_3.csv",
]

WIDE_FORMAT_FILE = "metric_sensors_8_9_10_11.csv"

_METRIC_RE = re.compile(
    r"CO2:\s*([\d.]+)\s*ppm.*?Temperatur:\s*([\d.]+)\s*°C.*?Humidity:\s*([\d.]+)\s*%",
    re.IGNORECASE,
)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", name.lower()).strip("_")


def _load_long(path: str, tz: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"], low_memory=False)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize(
            tz, ambiguous="NaT", nonexistent="shift_forward"
        )
    else:
        df["time"] = df["time"].dt.tz_convert(tz)
    df = df.dropna(subset=["time"])

    results = []
    for sensor_name, grp in df.groupby("name"):
        slug = _slug(sensor_name)
        grp = grp.set_index("time").sort_index()
        sub = pd.DataFrame(index=grp.index)
        sub[f"t_{slug}"] = pd.to_numeric(grp["temperature"], errors="coerce")
        sub[f"co2_{slug}"] = pd.to_numeric(grp["co2"], errors="coerce")
        sub[f"rh_{slug}"] = pd.to_numeric(grp.get("humidity", pd.Series(dtype=float)), errors="coerce")
        results.append(sub)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, axis=1).sort_index()


def _load_wide(path: str, tz: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"], low_memory=False)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize(
            tz, ambiguous="NaT", nonexistent="shift_forward"
        )
    else:
        df["time"] = df["time"].dt.tz_convert(tz)
    df = df.dropna(subset=["time"])
    df = df.set_index("time").sort_index()

    # Each "metric_viben{N}" column holds a sparse text payload like
    # "Sensor: 11  Skoven   CO2: 754 ppm, Temperatur: 22.2 °C, Humidity: 44 %, ...".
    # Parse each sensor column into its own t_/co2_/rh_ columns.
    metric_cols = [c for c in df.columns if str(c).startswith("metric_")]
    parts = []
    for col in metric_cols:
        slug = _slug(col.replace("metric_", ""))  # e.g. "viben8"
        payload = df[col].dropna()
        if payload.empty:
            continue
        rows = {"t_" + slug: [], "co2_" + slug: [], "rh_" + slug: []}
        idx = []
        for ts, val in payload.items():
            m = _METRIC_RE.search(str(val))
            if not m:
                continue
            idx.append(ts)
            rows["co2_" + slug].append(float(m.group(1)))
            rows["t_" + slug].append(float(m.group(2)))
            rows["rh_" + slug].append(float(m.group(3)))
        if idx:
            parts.append(pd.DataFrame(rows, index=pd.DatetimeIndex(idx)))

    if parts:
        out = pd.concat(parts, axis=0).sort_index()
        return out

    # Fallback: already-numeric columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return df[numeric_cols]


def load(tz: str = "Europe/Copenhagen") -> pd.DataFrame:
    frames = []
    for fname in LONG_FORMAT_FILES:
        p = os.path.join(SKOVEN_DIR, fname)
        if os.path.exists(p):
            frames.append(_load_long(p, tz))

    wide_path = os.path.join(SKOVEN_DIR, WIDE_FORMAT_FILE)
    if os.path.exists(wide_path):
        frames.append(_load_wide(wide_path, tz))

    if not frames:
        return pd.DataFrame()

    # Each per-sensor/per-file frame may carry duplicate timestamps; collapse them
    # (mean) so the index is unique before an axis=1 concat (which requires it).
    deduped = []
    for f in frames:
        if f.empty:
            continue
        f = f.sort_index()
        if f.index.has_duplicates:
            f = f.groupby(level=0).mean()
        deduped.append(f)

    if not deduped:
        return pd.DataFrame()

    combined = pd.concat(deduped, axis=1).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined
