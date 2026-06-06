"""Stage 2: Hydronic system parameter estimation for Skoven.

The building has one control lever (inlet water temperature via a mixing valve +
PID) and 4 fixed-flow radiators. With the inlet temperature replayed as a
measured boundary, only the per-room radiator parameters are identifiable:
  - Per-room radiator: thermalMassHeatCapacity
  - Per-room fixed-opening valve: waterFlowRateMax (the design flow)

Radiator nominal ratings (Q_flow_nominal_sh, T_a/T_b/TAir_nominal_sh) are plain
floats in SpaceHeaterTorchSystem (no autograd gradient) and the mixing-valve /
ecl310_pid parameters are not identifiable from boundary-replayed supply temp, so
all are held at their priors. Estimation targets zone temps + return-water temp.

Loads the Stage 1 envelope pickle first.

Run:
    python aarhus_model/hydronic_param_est.py --building skoven
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
    get_model, ZONES, ENVELOPE_RESULT_PICKLE, HYDRONIC_RESULT_PICKLE,
)

TZ = "Europe/Copenhagen"
CONFIG_DIR = os.path.join(SCRIPT_DIR, "..", "use_case", "building_configs")
STEP_SIZE = 600


def _window(building: str):
    with open(os.path.join(CONFIG_DIR, f"{building}.yaml")) as f:
        cfg = yaml.safe_load(f)
    tz = gettz(TZ)
    start = datetime.datetime.fromisoformat(cfg["hydronic_start"]).replace(tzinfo=tz)
    end = datetime.datetime.fromisoformat(cfg["hydronic_end"]).replace(tzinfo=tz)
    return start, end


def run_estimation(building: str = "skoven", step: int = STEP_SIZE):
    start, end = _window(building)
    maxiter = int(os.environ.get("AARHUS_MAXITER", 60))
    print(f"Stage 2 — Hydronic estimation: {start} -> {end} (maxiter={maxiter})")

    model = get_model(id="skoven_hydronic_est", calibration_mode=True)
    if os.path.exists(ENVELOPE_RESULT_PICKLE):
        model.load_estimation_result(ENVELOPE_RESULT_PICKLE)
        print("Loaded Stage 1 envelope parameters.")
    else:
        print(f"Warning: Stage 1 pickle not found ({ENVELOPE_RESULT_PICKLE}); using defaults.")

    # Only the radiator thermal mass and the fixed design flow per room are
    # identifiable here: the supply (inlet) water temperature is a measured
    # boundary, so the mixing-valve / ecl310_pid parameters cannot be identified
    # from this data and are left at their defaults.
    parameters = []
    for zone_id in ZONES:
        radiator = model.components[f"{zone_id}_radiator"]
        rad_valve = model.components[f"{zone_id}_radiator_valve"]
        parameters += [
            (radiator, "thermalMassHeatCapacity", 5000.0, 100.0, 50000.0),
            (rad_valve, "waterFlowRateMax", 0.05, 0.001, 1.0),
        ]

    # Measurements: zone temps + return-water temp. Supply-water is a boundary
    # input (not predicted) and the varme power sensor is a leaf (no simulated
    # connection), so neither is a valid AD target here.
    measurements = []
    for zone_id in ZONES:
        sensor = model.components.get(f"{zone_id}_indoor_temp_sensor")
        if sensor is not None:
            measurements.append((sensor, 0.5))
    ret = model.components.get("ecl310_TRetHea_y")
    if ret is not None and ret.filename is not None:
        measurements.append((ret, 1.0))

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

    os.makedirs(os.path.dirname(HYDRONIC_RESULT_PICKLE), exist_ok=True)
    shutil.copy(estimator.result_savedir_pickle, HYDRONIC_RESULT_PICKLE)
    model.load_estimation_result(HYDRONIC_RESULT_PICKLE)
    print(f"Stage 2 complete. Result: {HYDRONIC_RESULT_PICKLE}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--building", default="skoven")
    args = parser.parse_args()
    run_estimation(building=args.building)
