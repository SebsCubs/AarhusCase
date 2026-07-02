"""Export the unified Skoven frame into the per-signal CSVs the T4B model reads.

The filenames here MUST match the paths referenced in
aarhus_model/skoven_model.py (_csv(...) calls) and model_output_points.

Outputs (all in aarhus_model/generated_files/data/skoven/, columns time,value):
    outdoor_temperature.csv          outdoor air temp [°C]  (open-meteo hourly)
    global_irradiation.csv           clear-sky GHI [W/m²]
    outdoor_co2.csv                  constant 400 ppm background
    ecl310_TSupHea_y_processed.csv   supply water temp [°C]  (BMS Fremløb)
    ecl310_TRetHea_y_processed.csv   return water temp [°C]  (BMS Returtemp)
    ecl310_TSupSet_curve.csv         heating-curve supply setpoint [°C]
    zone_TZonSet_u_processed.csv     zone temp setpoint [°C] (BMS Rumtemp.ref)
    {zone}_indoor_temperature.csv    per-zone indoor temp [°C] (ReMoni mean)
    {zone}_heat_input.csv            per-zone heat boundary [W] (Stage 1)
    varme_meter_power_kW.csv         district-heat power [kW]  (reward/target)

Run:
    python -m data_ingest.export_t4b_csvs --building skoven --start 2025-01-01 --end 2025-05-31
"""
import argparse
import os
import sys

import pandas as pd
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from data_ingest.unify import build_skoven_frame
from aarhus_model.heating_curve import precompute_schedule

OUT_DIR = os.path.join(
    SCRIPT_DIR, "..", "aarhus_model", "generated_files", "data", "skoven"
)
CONFIG_DIR = os.path.join(SCRIPT_DIR, "..", "use_case", "building_configs")


def _load_config(building: str) -> dict:
    with open(os.path.join(CONFIG_DIR, f"{building}.yaml")) as f:
        return yaml.safe_load(f)


def _write(out_dir: str, stem: str, series: pd.Series) -> bool:
    """Write a single time,value CSV. Returns True if written."""
    s = series.dropna()
    if s.empty:
        print(f"  Skipping {stem}: no data")
        return False
    # Write in UTC so the time column has a single uniform offset. Local-time
    # CSVs spanning the DST change carry mixed offsets (+01:00/+02:00), which the
    # T4B spreadsheet loader (pd.to_datetime without utc=True) rejects.
    s = s.copy()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_convert("UTC")
    df = s.rename("value").to_frame()
    df.index.name = "time"
    df.to_csv(os.path.join(out_dir, f"{stem}.csv"))
    return True


def _zone_temp(df: pd.DataFrame, tokens: list) -> pd.Series:
    """Mean of all ReMoni temperature columns matching any token."""
    cols = [
        c for c in df.columns
        if c.startswith("t_") and any(tok in c for tok in tokens)
    ]
    if not cols:
        return pd.Series(dtype=float, index=df.index)
    return df[cols].mean(axis=1)


def export(building: str = "skoven", start: str = None, end: str = None,
           out_dir: str = OUT_DIR) -> None:
    cfg = _load_config(building)
    start = start or cfg.get("export_start", "2025-01-01")
    end = end or cfg.get("export_end", "2025-05-31")
    zones = cfg["zones"]
    varme_id = cfg.get("varme_meter_id")
    el_id = cfg.get("el_meter_id")
    hc = cfg.get("heating_curve", {})

    os.makedirs(out_dir, exist_ok=True)
    df = build_skoven_frame(
        start=start, end=end,
        varme_meter_ids=[varme_id] if varme_id else None,
        el_meter_ids=[el_id] if el_id else None,
    )

    written = []

    # --- Boundary signals ---
    if _write(out_dir, "outdoor_temperature", df["T_oa"]):
        written.append("outdoor_temperature")
    if "ghi_clearsky_Wm2" in df and _write(out_dir, "global_irradiation", df["ghi_clearsky_Wm2"]):
        written.append("global_irradiation")
    co2 = pd.Series(400.0, index=df.index)
    if _write(out_dir, "outdoor_co2", co2):
        written.append("outdoor_co2")

    # --- Water-loop signals (BMS; 6-hourly, interpolated by build_skoven_frame) ---
    if "T_sup_w" in df and _write(out_dir, "ecl310_TSupHea_y_processed", df["T_sup_w"]):
        written.append("ecl310_TSupHea_y_processed")
    if "T_ret_w" in df and _write(out_dir, "ecl310_TRetHea_y_processed", df["T_ret_w"]):
        written.append("ecl310_TRetHea_y_processed")
    if "T_zone_set" in df and _write(out_dir, "zone_TZonSet_u_processed", df["T_zone_set"]):
        written.append("zone_TZonSet_u_processed")
    # Measured ECL310 supply-temperature setpoint (BMS Fremloebstemp.ref) — the
    # real outdoor-reset target the controller tracked. Used as the setpoint for
    # the CLOSED-LOOP hydronic model (the synthetic heating curve below over-
    # predicts it). Falls back to the synthetic curve when this signal is absent.
    if "T_sup_w_set" in df and _write(out_dir, "ecl310_TSupSet_measured", df["T_sup_w_set"]):
        written.append("ecl310_TSupSet_measured")

    # --- Heating-curve supply setpoint (Phase D: drives ecl310 setpoint schedule) ---
    if {"T_oa", "T_zone_set"}.issubset(df.columns):
        try:
            curve = precompute_schedule(
                df, s=hc.get("s", 1.5), b=hc.get("b", 35.0),
                delta=hc.get("delta", 0.0),
                T_min=hc.get("T_min", 20.0), T_max=hc.get("T_max", 80.0),
            )
            if _write(out_dir, "ecl310_TSupSet_curve", curve):
                written.append("ecl310_TSupSet_curve")
        except Exception as e:
            print(f"  heating curve skipped: {e}")

    # --- Per-zone indoor temperature (ReMoni) ---
    for zone_id, zcfg in zones.items():
        tokens = zcfg.get("sensor_slugs", [])
        temp = _zone_temp(df, tokens)
        if _write(out_dir, f"{zone_id}_indoor_temperature", temp):
            written.append(f"{zone_id}_indoor_temperature")

    # --- District-heat power + per-zone heat boundary (Stage 1) ---
    power_col = f"varme_{varme_id}_power_kW" if varme_id else None
    if power_col and power_col in df.columns:
        power_kW = df[power_col]
        if _write(out_dir, "varme_meter_power_kW", power_kW):
            written.append("varme_meter_power_kW")
        # Split whole-building heat across zones (equal split — provisional)
        n = len(zones)
        per_zone_W = (power_kW * 1000.0 / n)
        for zone_id in zones:
            if _write(out_dir, f"{zone_id}_heat_input", per_zone_W):
                written.append(f"{zone_id}_heat_input")
    else:
        print(f"  No varme power column ({power_col}) — heat-input CSVs skipped")

    print(f"\nExported {len(written)} signal CSVs to {os.path.normpath(out_dir)}")
    for w in written:
        print(f"  {w}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--building", default="skoven")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    export(building=args.building, start=args.start, end=args.end)
