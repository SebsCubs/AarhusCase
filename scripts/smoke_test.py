"""Smoke test: generate synthetic boundary CSVs, build the Skoven model in
simulation mode, and run a short Simulator.simulate() call.

Run with:
    uv run python scripts/smoke_test.py
"""
import os
import sys
import datetime
import pandas as pd
import numpy as np
from dateutil.tz import gettz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)

from aarhus_model.skoven_model import get_model, DATA_DIR, ZONES

# ---------------------------------------------------------------------------
# 1. Synthesize boundary-condition CSVs
# ---------------------------------------------------------------------------
TZ = gettz("Europe/Copenhagen")
START = datetime.datetime(2026, 1, 1, 0, 0, tzinfo=TZ)
END = datetime.datetime(2026, 1, 2, 0, 0, tzinfo=TZ)  # 1-day window
STEP = "10min"

os.makedirs(DATA_DIR, exist_ok=True)
times = pd.date_range(START, END, freq=STEP)

def _write_csv(name, values):
    path = os.path.join(DATA_DIR, name)
    # Do NOT clobber real exported boundary CSVs — only write synthetic data if
    # the file is absent (e.g. a fresh checkout before data_ingest has run).
    if os.path.exists(path):
        print(f"  (keeping existing {name})")
        return path
    # twin4build's loader calls pd.read_csv (header=0) so a header row is required.
    df = pd.DataFrame({"time": times, "value": values})
    df.to_csv(path, index=False)
    return path

# Outdoor: cold winter day, sinusoidal -5..+2 °C
t_hours = np.array([(t - times[0]).total_seconds() / 3600 for t in times])
T_out = -1.5 + 3.5 * np.sin((t_hours - 6) * np.pi / 12)
_write_csv("outdoor_temperature.csv", T_out)

# Irradiance: daylight only, 0..200 W/m²
ghi = np.clip(200 * np.sin((t_hours - 6) * np.pi / 12), 0, None)
_write_csv("global_irradiation.csv", ghi)

# CO2: constant 400 ppm outdoor background
_write_csv("outdoor_co2.csv", np.full_like(t_hours, 400.0))

print(f"Wrote {len(times)} rows of synthetic boundary data to {DATA_DIR}")

# ---------------------------------------------------------------------------
# 2. Build model in SIMULATION mode (no CSV sensors needed for hydronic)
# ---------------------------------------------------------------------------
import twin4build as tb

print("Building model (simulation mode)...")
try:
    model = get_model(id="skoven_smoke", calibration_mode=False)
    print(f"Model built: {len(model.components)} components")
    print("Component IDs:")
    for cid in sorted(model.components.keys())[:20]:
        print(f"  {cid}")
    if len(model.components) > 20:
        print(f"  ... and {len(model.components) - 20} more")
except Exception as e:
    print(f"MODEL BUILD FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Try a short simulation
# ---------------------------------------------------------------------------
print("\nRunning 6-hour simulation at 600 s steps...")
try:
    simulator = tb.Simulator(model)
    sim_end = START + datetime.timedelta(hours=6)
    simulator.simulate(start_time=START, end_time=sim_end, step_size=600)
    print("Simulation completed successfully.")

    # Sample outputs — dev-branch components expose history() on each port
    def _h(component_id, port):
        h = model.components[component_id].output[port].history()
        return h.detach().cpu().numpy().squeeze()

    for zid in ZONES:
        arr = _h(f"{zid}_indoor_temp_sensor", "measuredValue")
        print(f"  {zid}: T_indoor min={arr.min():.2f}, max={arr.max():.2f}, last={arr[-1]:.2f} °C")
    for zid in ZONES:
        arr = _h(f"{zid}_radiator", "Power")
        print(f"  {zid}_radiator: Power min={arr.min():.0f}, max={arr.max():.0f}, mean={arr.mean():.0f} W")
except Exception as e:
    print(f"SIMULATION FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
