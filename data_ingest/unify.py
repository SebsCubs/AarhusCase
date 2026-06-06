"""Fuse all Skoven data sources into a single tz-aware 15-min DataFrame.

Entry point: build_skoven_frame(start, end, stride="15min")

Output columns (after resampling):
  From BMS:       T_oa, T_zone_bms, T_zone_set, T_sup_w, T_sup_w_set, T_ret_w, T_ret_w_set
  From ReMoni:    t_{sensor}, co2_{sensor}, rh_{sensor}  (one triple per Viben sensor)
  From Varme:     varme_{meter_id}_power_kW, varme_{meter_id}_T_sup, varme_{meter_id}_T_ret
  From EL:        el_{meter_id}_power_kW
  Synthesized:    ghi_clearsky_Wm2

Usage:
    from data_ingest.unify import build_skoven_frame
    df = build_skoven_frame("2024-12-01", "2025-03-01")
"""
import argparse
import os
import pandas as pd
from dateutil.tz import gettz

from . import skoven_bms, remoni_indoor, varme_meter, el_meter, outdoor_synth, weather_openmeteo

TZ = "Europe/Copenhagen"


def _resample_bms(df: pd.DataFrame, stride: str) -> pd.DataFrame:
    # BMS is 6-hourly; time-interpolate within its data range so the resampled
    # 15-min signals (supply/return water, setpoints) are continuous. Leading/
    # trailing NaN (outside the BMS period) are left untouched.
    return df.resample(stride).mean().interpolate(method="time", limit_area="inside")


def _resample_remoni(df: pd.DataFrame, stride: str) -> pd.DataFrame:
    return df.resample(stride).mean()


def _resample_meter(df: pd.DataFrame, stride: str) -> pd.DataFrame:
    """Meters are hourly; resample by dividing power proportionally."""
    numeric = df.select_dtypes(include="number")
    resampled = numeric.resample(stride).interpolate(method="time").resample(stride).mean()
    return resampled


def build_skoven_frame(
    start: str = "2024-12-01",
    end: str = "2025-03-01",
    stride: str = "15min",
    fill_limit_minutes: int = 60,
    varme_meter_ids: list = None,
    el_meter_ids: list = None,
) -> pd.DataFrame:
    """Build unified Skoven data frame.

    Args:
        start: Start date string (ISO format).
        end: End date string (ISO format).
        stride: Resample stride (default '15min').
        fill_limit_minutes: Max gap to forward-fill (minutes).
        varme_meter_ids: If given, only load these Varme meters.
        el_meter_ids: If given, only load these EL meters.

    Returns:
        tz-aware DataFrame at the requested stride.
    """
    tz = gettz(TZ)
    t_start = pd.Timestamp(start, tz=TZ)
    t_end = pd.Timestamp(end, tz=TZ)
    fill_limit = int(fill_limit_minutes * 60 / pd.Timedelta(stride).total_seconds())

    frames = {}

    # --- BMS ---
    try:
        bms = skoven_bms.load()
        frames["bms"] = _resample_bms(bms, stride)
        print("BMS loaded OK")
    except Exception as e:
        print(f"Warning: BMS load failed: {e}")

    # --- ReMoni ---
    try:
        remoni = remoni_indoor.load()
        frames["remoni"] = _resample_remoni(remoni, stride)
        print(f"ReMoni loaded OK ({len(remoni.columns)} columns)")
    except Exception as e:
        print(f"Warning: ReMoni load failed: {e}")

    # --- Varme meters ---
    try:
        varme_all = varme_meter.load_all()
        for mid, mdf in varme_all.items():
            if varme_meter_ids and mid not in varme_meter_ids:
                continue
            resampled = _resample_meter(mdf, stride)
            resampled.columns = [f"varme_{mid}_{c}" for c in resampled.columns]
            frames[f"varme_{mid}"] = resampled
        print(f"Varme meters loaded: {list(varme_all.keys())}")
    except Exception as e:
        print(f"Warning: Varme load failed: {e}")

    # --- EL meters ---
    try:
        el_all = el_meter.load_all()
        for mid, mdf in el_all.items():
            if el_meter_ids and mid not in el_meter_ids:
                continue
            resampled = _resample_meter(mdf, stride)
            resampled.columns = [f"el_{mid}_{c}" for c in resampled.columns]
            frames[f"el_{mid}"] = resampled
        print(f"EL meters loaded: {list(el_all.keys())}")
    except Exception as e:
        print(f"Warning: EL load failed: {e}")

    # Combine
    if not frames:
        raise RuntimeError("No data sources loaded successfully.")

    combined = pd.concat(list(frames.values()), axis=1)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    # Trim to requested window
    combined = combined.loc[t_start:t_end]

    # --- External hourly outdoor temperature (open-meteo) ---
    # The BMS Udetemperatur is only 6-hourly; replace T_oa with hourly ERA5 data
    # resampled to the working stride. Falls back to BMS T_oa on any failure.
    try:
        t_oa = weather_openmeteo.fetch_outdoor_temperature(start, end)
        t_oa = t_oa[~t_oa.index.duplicated(keep="first")].sort_index()
        union_idx = t_oa.index.union(combined.index)
        t_oa = t_oa.reindex(union_idx).interpolate(method="time").reindex(combined.index)
        combined["T_oa"] = t_oa.values
        print("Outdoor temperature: open-meteo hourly (overrides 6-hourly BMS)")
    except Exception as e:
        print(f"Warning: open-meteo fetch failed ({e}); keeping BMS T_oa")

    # Synthesize solar irradiance aligned to combined index
    try:
        ghi = outdoor_synth.synthesize_ghi(combined.index)
        combined["ghi_clearsky_Wm2"] = ghi.values
    except Exception as e:
        print(f"Warning: solar synthesis failed: {e}")

    # Fill small gaps, mask large ones
    combined = combined.ffill(limit=fill_limit)

    nan_pct = combined.isna().mean().mean() * 100
    print(f"Unified frame: {len(combined)} rows × {len(combined.columns)} cols, {nan_pct:.1f}% NaN")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--building", default="skoven")
    parser.add_argument("--start", default="2024-12-01")
    parser.add_argument("--end", default="2025-03-01")
    parser.add_argument("--stride", default="15min")
    args = parser.parse_args()

    if args.building.lower() == "skoven":
        df = build_skoven_frame(start=args.start, end=args.end, stride=args.stride)
        print(df.head())
        print(df.dtypes)
    else:
        print(f"Building '{args.building}' not yet implemented. Only 'skoven' supported.")
