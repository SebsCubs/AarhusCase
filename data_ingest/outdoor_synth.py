"""Synthesize clear-sky global horizontal irradiance for Aarhus using PVlib.

Aarhus coordinates: 56.16°N, 10.20°E, altitude ~20 m.
Returns a tz-aware Series of GHI [W/m²] at the requested time index.
"""
import pandas as pd

AARHUS_LAT = 56.16
AARHUS_LON = 10.20
AARHUS_ALT = 20.0


def synthesize_ghi(index: pd.DatetimeIndex, tz: str = "Europe/Copenhagen") -> pd.Series:
    """Compute clear-sky GHI for Aarhus at the given DatetimeIndex.

    Args:
        index: tz-aware DatetimeIndex at the desired resolution.
        tz: Timezone string (default Europe/Copenhagen).

    Returns:
        pd.Series of GHI [W/m²] aligned to index.
    """
    import pvlib

    loc = pvlib.location.Location(
        latitude=AARHUS_LAT,
        longitude=AARHUS_LON,
        tz=tz,
        altitude=AARHUS_ALT,
    )

    if index.tzinfo is None:
        index = index.tz_localize(tz)
    else:
        index = index.tz_convert(tz)

    cs = loc.get_clearsky(index, model="ineichen")
    return cs["ghi"].rename("ghi_clearsky_Wm2")
