"""Stage 1: Envelope parameter estimation for Skoven.

Estimates zone thermal capacitances/resistances and solar-split factors against
measured ReMoni indoor temperatures, with open-meteo outdoor temperature and a
measured per-zone heat-input boundary ({zone}_heat_input.csv) as forcing.

Parameter layout:
  - private (per zone): C_air, C_wall, C_boundary, R_out, R_in, R_boundary,
    f_wall, f_air.
  - shared (one value for the whole building): C_int, R_int — these describe the
    inter-room coupling, so a single building-wide value is estimated.

Dev-branch Estimator API:
  - tb.Estimator(simulator)
  - parameters = list of (component, attr, x0, lb, ub, "private"|"shared")
  - measurements = list of (sensor_component, standard_deviation)
  - method = ("scipy", "SLSQP", "ad")

Run:
    python aarhus_model/only_envelope_param_est.py --building skoven
"""
import argparse
import datetime
import os
import shutil
import sys

import yaml
from dateutil.tz import gettz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import twin4build as tb
from aarhus_model.skoven_model import (
    get_model, make_envelope_fcn, ZONES, ENVELOPE_RESULT_PICKLE,
    zone_air_capacitance,
)

TZ = "Europe/Copenhagen"
CONFIG_DIR = os.path.join(SCRIPT_DIR, "..", "use_case", "building_configs")
STEP_SIZE = 600  # seconds

# Per-zone INDEPENDENT ("private") parameters: (attr, x0, lb, ub).
# C_air x0 is overridden per-zone from the zone air volume (see run_estimation).
# f_wall / f_air are the solar-radiation split factors (exterior wall vs. air),
# estimated from a 0.3 prior over a wide 0..100 range.
ENVELOPE_PARAMS_PRIVATE = [
    ("C_air", 1e6, 1e4, 1e8),
    ("C_wall", 5e6, 1e4, 1e9),
    ("C_boundary", 1e6, 1e3, 1e8),
    ("R_out", 0.01, 1e-4, 0.5),
    ("R_in", 0.005, 1e-4, 0.5),
    ("R_boundary", 0.01, 1e-4, 0.5),
    ("f_wall", 0.3, 0.0, 100.0),
    ("f_air", 0.3, 0.0, 100.0),
]

# SHARED across all four zones: the inter-room coupling terms. C_int and R_int
# describe the capacitance/resistance between rooms, so a single building-wide
# value is estimated rather than one per zone.
ENVELOPE_PARAMS_SHARED = [
    ("C_int", 2e5, 1e3, 1e7),
    ("R_int", 0.02, 1e-4, 0.5),
]


def _window(building: str):
    with open(os.path.join(CONFIG_DIR, f"{building}.yaml")) as f:
        cfg = yaml.safe_load(f)
    tz = gettz(TZ)
    start = datetime.datetime.fromisoformat(cfg["envelope_start"]).replace(tzinfo=tz)
    end = datetime.datetime.fromisoformat(cfg["envelope_end"]).replace(tzinfo=tz)
    return start, end


def run_estimation(building: str = "skoven", step: int = STEP_SIZE):
    start, end = _window(building)
    maxiter = int(os.environ.get("AARHUS_MAXITER", 60))
    print(f"Stage 1 — Envelope estimation: {start} -> {end} (maxiter={maxiter})")

    model = get_model(
        id="skoven_envelope_only",
        fcn_=make_envelope_fcn(calibration_mode=True),
        calibration_mode=True,
    )

    # Parameters: per-zone private + one building-wide shared inter-room coupling.
    parameters = []
    for zone_id in ZONES:
        zone = model.components[zone_id]
        for attr, x0, lb, ub in ENVELOPE_PARAMS_PRIVATE:
            # Seed C_air from the per-zone air volume (floor area * 3 m) so the
            # estimator starts at the physical prior instead of a uniform guess.
            if attr == "C_air":
                x0 = zone_air_capacitance(zone_id)
            parameters.append((zone, attr, x0, lb, ub, "private"))

    # C_int / R_int shared: a single value tied across all four zones.
    zones_list = [model.components[z] for z in ZONES]
    for attr, x0, lb, ub in ENVELOPE_PARAMS_SHARED:
        parameters.append((zones_list, attr, x0, lb, ub, "shared"))

    # Measurements: per-zone indoor temperature sensors (std 0.5 °C)
    measurements = []
    for zone_id in ZONES:
        sensor = model.components.get(f"{zone_id}_indoor_temp_sensor")
        if sensor is not None:
            measurements.append((sensor, 0.5))
        else:
            print(
                f"WARNING: no indoor temperature sensor found for zone "
                f"'{zone_id}' (expected component "
                f"'{zone_id}_indoor_temp_sensor'); skipping its measurement.",
                file=sys.stderr,
            )

    simulator = tb.Simulator(model)
    estimator = tb.Estimator(simulator)
    estimator.estimate(
        parameters=parameters,
        measurements=measurements,
        start_time=start,
        end_time=end,
        step_size=step,
        method=("scipy", "SLSQP", "ad"),
        options={"maxiter": maxiter},
    )

    src = estimator.result_savedir_pickle
    os.makedirs(os.path.dirname(ENVELOPE_RESULT_PICKLE), exist_ok=True)
    shutil.copy(src, ENVELOPE_RESULT_PICKLE)
    model.load_estimation_result(ENVELOPE_RESULT_PICKLE)
    print(f"Stage 1 complete. Result: {ENVELOPE_RESULT_PICKLE}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--building", default="skoven")
    args = parser.parse_args()
    run_estimation(building=args.building)
