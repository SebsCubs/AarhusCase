"""Heuristic matcher: correlate Varme/EL meters with buildings.

Strategy: for each Varme meter, compute Pearson correlation of its power_kW
against the heat-demand proxy signal (T_zone_set - T_oa) from each building's
BMS export. The meter with the highest correlation to a building's proxy is
assigned to that building.

Outputs meter_matches.csv with correlation scores for human sign-off.
Manual overrides can be set in building_configs/*.yaml via the 'varme_meter_id'
and 'el_meter_id' keys.
"""
import os
import pandas as pd
import numpy as np
from typing import Optional


def _resample_align(
    series_a: pd.Series, series_b: pd.Series, stride: str = "1h"
) -> tuple[pd.Series, pd.Series]:
    a = series_a.resample(stride).mean().dropna()
    b = series_b.resample(stride).mean().dropna()
    idx = a.index.intersection(b.index)
    return a.loc[idx], b.loc[idx]


def match_varme_to_buildings(
    varme_meters: dict,
    bms_frames: dict,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Correlate each Varme meter's power against BMS heat-demand proxies.

    Args:
        varme_meters: dict {meter_id: DataFrame} from varme_meter.load_all().
        bms_frames: dict {building_name: DataFrame} with 'T_zone_set' and 'T_oa' columns.
        output_path: If given, write meter_matches.csv here.

    Returns:
        DataFrame with columns [meter_id, building, pearson_r, confidence].
    """
    records = []
    for meter_id, mdf in varme_meters.items():
        if "power_kW" not in mdf.columns:
            continue
        power = mdf["power_kW"].dropna()

        for building, bdf in bms_frames.items():
            if "T_zone_set" not in bdf.columns or "T_oa" not in bdf.columns:
                continue
            proxy = (bdf["T_zone_set"] - bdf["T_oa"]).dropna()
            a, b = _resample_align(power, proxy)
            if len(a) < 24:
                r = float("nan")
            else:
                r = float(np.corrcoef(a.values, b.values)[0, 1])
            records.append({"meter_id": meter_id, "building": building, "pearson_r": r})

    df = pd.DataFrame(records)
    if not df.empty:
        df["confidence"] = df["pearson_r"].abs()
        df = df.sort_values(["meter_id", "confidence"], ascending=[True, False])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"meter_matches.csv written to {output_path}")
        print("*** PLEASE REVIEW meter_matches.csv and confirm/override assignments in building_configs/*.yaml ***")

    return df


def best_match(match_df: pd.DataFrame, meter_id: str) -> str:
    """Return the building with the highest correlation for a given meter."""
    sub = match_df[match_df["meter_id"] == meter_id]
    if sub.empty:
        return "unknown"
    return sub.iloc[0]["building"]
