"""Stage 2: Closed-loop hydronic parameter estimation for Skoven.

The hydronic system is the closed-loop ECL310 mixing-shunt substation (the only
hydronic model). A PID modulates the primary mixing valve to track the measured
supply-temperature setpoint; the secondary supply temperature is a PRODUCED output
and the return co-varies with the radiator heat extraction via the recirculation
feedback. The 4 radiators sit on fixed-opening valves (constant flow).

What is estimated vs. sized
---------------------------
- **Sized from the energy balance** (NOT estimated): each radiator's nominal
  rating (-> UA -> steady-state heat output / supply-return dT) and the fixed
  radiator-loop flow. The radiator UA is non-AD-calibratable in Twin4Build
  (SpaceHeaterTorchSystem re-solves UA from the float nominal rating inside
  initialize() and freezes it), and free-fitting it would break consistency with
  the Stage-1 envelope (calibrated with per-room heat input = district-heat
  power / 4). compute_winter_sizing() pins the rating to the measured operating
  point so total radiator output matches the envelope's heat demand and each
  radiator delivers ~Q_meter/4 by construction. See skoven_model.py.
- **Estimated by AD** (the genuinely identifiable, in-path levers):
    * ecl310_pid: kp, Ti        — controller (supply tracking)
    * per-radiator thermalMassHeatCapacity — radiator thermal inertia (dynamics)

Targets (all are PRODUCED outputs in the closed loop, so all are valid AD targets):
  4 zone temperatures + supply-water temperature + return-water temperature.

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
    get_model, make_fcn, ZONES, ENVELOPE_RESULT_PICKLE, HYDRONIC_RESULT_PICKLE,
    compute_winter_sizing, apply_energy_balance_sizing,
)

TZ = "Europe/Copenhagen"
CONFIG_DIR = os.path.join(SCRIPT_DIR, "..", "use_case", "building_configs")
STEP_SIZE = 600

# Measurement standard deviations [°C] used to weight the objective. Water
# signals get a slightly looser std than the ReMoni zone temps.
ZONE_STD = 0.5
WATER_STD = 1.0


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
    print(f"Stage 2 — Closed-loop hydronic estimation: {start} -> {end} (maxiter={maxiter})")

    if not os.path.exists(ENVELOPE_RESULT_PICKLE):
        print(f"Warning: Stage 1 pickle not found ({ENVELOPE_RESULT_PICKLE}); "
              f"sizing/estimation will use envelope defaults.")

    # Energy-balance sizing: derive the radiator rating + fixed flow from the
    # envelope's heat demand and the measured operating point (consistent with the
    # Stage-1 per-room heat split). Done up front so the figures are reproducible.
    sizing = compute_winter_sizing(start, end, step=step)
    print("Energy-balance sizing (from envelope demand + measured operating point):")
    print(f"  Q_demand_total = {sizing['Q_demand_total']:.1f} W "
          f"(~{sizing['Q_demand_total']/1000:.2f} kW)  ->  Q_per_rad = {sizing['Q_per_rad']:.1f} W")
    print(f"  operating point: T_sup={sizing['T_sup']:.1f} C  T_ret={sizing['T_ret']:.1f} C  "
          f"T_air={sizing['T_air']:.1f} C  (dT={sizing['T_sup']-sizing['T_ret']:.1f} C)")
    print(f"  fixed flow per radiator = {sizing['m_per_rad']:.4f} kg/s "
          f"(total {4*sizing['m_per_rad']:.4f} kg/s)")

    model = get_model(id="skoven_hydronic_est",
                      fcn_=make_fcn(calibration_mode=True),
                      calibration_mode=True)
    if os.path.exists(ENVELOPE_RESULT_PICKLE):
        model.load_estimation_result(ENVELOPE_RESULT_PICKLE)
        print("Loaded Stage 1 envelope parameters.")
    apply_energy_balance_sizing(model, sizing)

    # AD parameters: PID gains (controller) + per-radiator thermal mass (dynamics).
    # (component, attr, x0, lb, ub). Q_flow_nominal_sh / UA are SIZED above, not
    # estimated (non-AD), and the radiator flow is fixed by the sizing.
    pid = model.components["ecl310_pid"]
    parameters = [
        (pid, "kp", 0.05, 1e-3, 5.0),
        (pid, "Ti", 1800.0, 60.0, 7200.0),
    ]
    for zone_id in ZONES:
        radiator = model.components[f"{zone_id}_radiator"]
        parameters.append((radiator, "thermalMassHeatCapacity", 5000.0, 100.0, 50000.0))

    # Measurements: 4 zone temps + supply + return water (all produced outputs).
    measurements = []
    for zone_id in ZONES:
        sensor = model.components.get(f"{zone_id}_indoor_temp_sensor")
        if sensor is not None:
            measurements.append((sensor, ZONE_STD))
    for water_id in ("ecl310_TSupHea_y", "ecl310_TRetHea_y"):
        sensor = model.components.get(water_id)
        if sensor is not None:
            measurements.append((sensor, WATER_STD))

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
    print(f"  estimated kp={float(pid.kp.get()):.4f}  Ti={float(pid.Ti.get()):.1f} s")
    for zone_id in ZONES:
        rad = model.components[f"{zone_id}_radiator"]
        print(f"  {zone_id}: thermalMassHeatCapacity={float(rad.thermalMassHeatCapacity.get()):.0f} J/K")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--building", default="skoven")
    args = parser.parse_args()
    run_estimation(building=args.building)
