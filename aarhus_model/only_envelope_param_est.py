"""Stage 1: Envelope parameter estimation for Skoven.

Estimates zone thermal capacitances/resistances (C_air, C_wall, C_int,
C_boundary, R_out, R_in, R_int, R_boundary) against measured ReMoni indoor
temperatures, with open-meteo outdoor temperature and a measured per-zone
heat-input boundary ({zone}_heat_input.csv) as forcing.

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
)

TZ = "Europe/Copenhagen"
CONFIG_DIR = os.path.join(SCRIPT_DIR, "..", "use_case", "building_configs")
STEP_SIZE = 600  # seconds

# Per-zone envelope parameters to estimate: (attr, x0, lb, ub)
ENVELOPE_PARAMS = [
    ("C_air", 1e6, 1e4, 1e8),
    ("C_wall", 5e6, 1e4, 1e9),
    ("C_int", 2e5, 1e3, 1e7),
    ("C_boundary", 1e6, 1e3, 1e8),
    ("R_out", 0.01, 1e-4, 0.5),
    ("R_in", 0.005, 1e-4, 0.5),
    ("R_int", 0.02, 1e-4, 0.5),
    ("R_boundary", 0.01, 1e-4, 0.5),
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

    # Parameters: per-zone, independent (private)
    parameters = []
    for zone_id in ZONES:
        zone = model.components[zone_id]
        for attr, x0, lb, ub in ENVELOPE_PARAMS:
            parameters.append((zone, attr, x0, lb, ub, "private"))

    # Measurements: per-zone indoor temperature sensors (std 0.5 °C)
    measurements = []
    for zone_id in ZONES:
        sensor = model.components.get(f"{zone_id}_indoor_temp_sensor")
        if sensor is not None:
            measurements.append((sensor, 0.5))

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
