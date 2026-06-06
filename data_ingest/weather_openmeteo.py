"""Fetch historical hourly outdoor temperature for Aarhus from open-meteo.

The Skoven BMS export is only 6-hourly, so it is a poor boundary signal for
thermal calibration. open-meteo's ERA5 archive provides hourly 2 m air
temperature for the building's coordinates (56.16 N, 10.20 E) over the whole
data period, free and keyless.

Results are cached to a CSV under generated_files/data/skoven/ so the network is
only hit once per (start, end) range.

Usage:
    from data_ingest.weather_openmeteo import fetch_outdoor_temperature
    s = fetch_outdoor_temperature("2025-01-01", "2025-05-31")  # tz-aware Series
"""
import json
import os
import urllib.parse
import urllib.request

import pandas as pd

TZ = "Europe/Copenhagen"
LAT = 56.16
LON = 10.20

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

CACHE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "aarhus_model",
    "generated_files",
    "data",
    "skoven",
)


def _cache_path(start: str, end: str) -> str:
    return os.path.join(CACHE_DIR, f"openmeteo_T_oa_{start}_{end}.csv")


def fetch_outdoor_temperature(
    start: str,
    end: str,
    tz: str = TZ,
    use_cache: bool = True,
) -> pd.Series:
    """Return hourly outdoor air temperature [°C] as a tz-aware Series.

    Args:
        start, end: ISO date strings (inclusive day range).
        tz: target timezone.
        use_cache: read/write a CSV cache to avoid repeat network calls.
    """
    start = str(pd.Timestamp(start).date())
    end = str(pd.Timestamp(end).date())
    cache = _cache_path(start, end)

    if use_cache and os.path.exists(cache):
        s = pd.read_csv(cache, index_col=0, parse_dates=True)["T_oa"]
        s.index = pd.to_datetime(s.index, utc=True).tz_convert(tz)
        return s

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m",
        "timezone": tz,
    }
    url = _ARCHIVE_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)

    times = pd.to_datetime(data["hourly"]["time"])
    temps = data["hourly"]["temperature_2m"]
    idx = times.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    s = pd.Series(temps, index=idx, name="T_oa")
    s = s[s.index.notna()].astype(float)

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        s.tz_convert("UTC").to_frame("T_oa").to_csv(cache)

    return s
