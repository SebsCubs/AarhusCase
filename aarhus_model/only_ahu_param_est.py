"""Stage 3: AHU parameter estimation for Skoven.

Estimates: fan polynomial coefficients (c1..c4), heat-recovery effectiveness
(eps_75_h, eps_100_h, eps_75_c, eps_100_c), and AHU heating coil UA.

NOTE: Skoven AHU sensor data is sparse (AHU documentation refers to Sundhedshus).
If no AHU sensor CSV is available this script uses literature defaults and exits.
Stage 3 is optional — the RL env operates without it by using AHU_DEFAULTS.

Run:
    python aarhus_model/only_ahu_param_est.py --building skoven
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
    get_model, ENVELOPE_RESULT_PICKLE, HYDRONIC_RESULT_PICKLE, AHU_RESULT_PICKLE,
)

TZ = "Europe/Copenhagen"
CONFIG_DIR = os.path.join(SCRIPT_DIR, "..", "use_case", "building_configs")
STEP_SIZE = 600

AHU_SENSOR_IDS = ["vent_supply_air_temp_sensor", "vent_return_air_temp_sensor", "vent_power_sensor"]


def _window(building: str):
    with open(os.path.join(CONFIG_DIR, f"{building}.yaml")) as f:
        cfg = yaml.safe_load(f)
    tz = gettz(TZ)
    start = datetime.datetime.fromisoformat(cfg["hydronic_start"]).replace(tzinfo=tz)
    end = datetime.datetime.fromisoformat(cfg["hydronic_end"]).replace(tzinfo=tz)
    return start, end


def _ahu_sensors_available(model) -> bool:
    """AHU sensors can only be estimation targets if they carry observed data
    (a filename). Skoven has no AHU instrumentation, so this returns False and
    the stage falls back to AHU_DEFAULTS."""
    for sid in AHU_SENSOR_IDS:
        comp = model.components.get(sid)
        if comp is None:
            return False
        fname = getattr(comp, "filename", None)
        if not fname or not os.path.exists(fname):
            return False
    return True


def run_estimation(building: str = "skoven", step: int = STEP_SIZE):
    start, end = _window(building)
    print(f"Stage 3 — AHU estimation: {start} -> {end}")

    model = get_model(id="skoven_ahu_est", calibration_mode=True)
    for pickle_path in [ENVELOPE_RESULT_PICKLE, HYDRONIC_RESULT_PICKLE]:
        if os.path.exists(pickle_path):
            model.load_estimation_result(pickle_path)

    if not _ahu_sensors_available(model):
        print(
            "AHU sensor CSVs not found — Stage 3 skipped. "
            "Using literature defaults from AHU_DEFAULTS in skoven_model.py."
        )
        return model

    parameters = [
        (model.components["supply_fan"], "nominalPowerRate", 800.0, 100.0, 5000.0),
        (model.components["supply_fan"], "f_total", 0.8, 0.5, 1.0),
        (model.components["heat_recovery"], "eps_75_h", 0.75, 0.3, 0.95),
        (model.components["heat_recovery"], "eps_100_h", 0.70, 0.3, 0.95),
        (model.components["heat_recovery"], "eps_75_c", 0.65, 0.3, 0.95),
        (model.components["heat_recovery"], "eps_100_c", 0.60, 0.3, 0.95),
    ]
    measurements = [
        (model.components["vent_supply_air_temp_sensor"], 0.5),
        (model.components["vent_return_air_temp_sensor"], 0.5),
        (model.components["vent_power_sensor"], 10.0),
    ]

    simulator = tb.Simulator(model)
    estimator = tb.Estimator(simulator)
    estimator.estimate(
        parameters=parameters,
        measurements=measurements,
        start_time=start,
        end_time=end,
        step_size=step,
        method=("scipy", "SLSQP", "ad"),
        options={"maxiter": int(os.environ.get("AARHUS_MAXITER", 60))},
    )
    os.makedirs(os.path.dirname(AHU_RESULT_PICKLE), exist_ok=True)
    shutil.copy(estimator.result_savedir_pickle, AHU_RESULT_PICKLE)
    model.load_estimation_result(AHU_RESULT_PICKLE)
    print(f"Stage 3 complete. Result: {AHU_RESULT_PICKLE}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--building", default="skoven")
    args = parser.parse_args()
    run_estimation(building=args.building)
