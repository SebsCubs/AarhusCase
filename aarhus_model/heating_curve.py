"""ECL310 outdoor-reset heating curve helper.

The Danfoss ECL310 implements an outdoor-compensated supply-water temperature
setpoint:

    T_sup_set = clip(b + s * (T_room_ref - T_oa) + delta, T_min, T_max)

where:
    s     — slope (K per K of (T_room_ref - T_oa))
    b     — base supply temperature at 0 ΔT [°C]
    delta — parallel shift / offset [°C]

Typical defaults for Nordic residential: s=1.5, b=35, delta=0.
Estimated in Stage 2 (hydronic_param_est.py).

Usage:
    from aarhus_model.heating_curve import compute_supply_setpoint, precompute_schedule

    T_sup = compute_supply_setpoint(T_oa=-5, T_room_ref=21, s=1.5, b=35, delta=0)
    schedule_df = precompute_schedule(bms_df, s=1.5, b=35, delta=0)
"""
import numpy as np
import pandas as pd

T_SUP_MIN = 20.0  # [°C] — minimum supply water temp
T_SUP_MAX = 80.0  # [°C] — maximum supply water temp


def compute_supply_setpoint(
    T_oa: float,
    T_room_ref: float,
    s: float = 1.5,
    b: float = 35.0,
    delta: float = 0.0,
    T_min: float = T_SUP_MIN,
    T_max: float = T_SUP_MAX,
) -> float:
    """Scalar ECL310 heating curve.

    Args:
        T_oa: Outdoor air temperature [°C].
        T_room_ref: Room temperature reference/setpoint [°C].
        s: Curve slope.
        b: Base offset [°C].
        delta: Parallel shift [°C].
        T_min: Minimum allowed supply temperature [°C].
        T_max: Maximum allowed supply temperature [°C].

    Returns:
        Supply water temperature setpoint [°C].
    """
    T_set = b + s * (T_room_ref - T_oa) + delta
    return float(np.clip(T_set, T_min, T_max))


def precompute_schedule(
    bms_df: pd.DataFrame,
    s: float = 1.5,
    b: float = 35.0,
    delta: float = 0.0,
    T_min: float = T_SUP_MIN,
    T_max: float = T_SUP_MAX,
    T_oa_col: str = "T_oa",
    T_room_ref_col: str = "T_zone_set",
) -> pd.Series:
    """Vectorised version: compute setpoint trajectory from BMS DataFrame.

    Returns a Series of supply-water setpoints aligned to bms_df.index.
    Suitable for writing to a ScheduleSystem CSV.
    """
    T_oa = bms_df[T_oa_col]
    T_ref = bms_df[T_room_ref_col]
    T_set = b + s * (T_ref - T_oa) + delta
    return T_set.clip(T_min, T_max).rename("T_sup_set_curve")
