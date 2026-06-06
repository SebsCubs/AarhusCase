"""Manual calibration analysis for the Skoven 4-room model.

Builds the full calibration-mode model, loads whatever estimation pickles exist,
simulates a window, and reports per-signal RMSE (room temps, return water) plus
radiator power. Optionally writes overlay plots to scripts/plots/.

Usage:
    uv run python scripts/analyze_calibration.py                       # hydronic window
    uv run python scripts/analyze_calibration.py --start 2025-01-15 --end 2025-01-22
    uv run python scripts/analyze_calibration.py --plots
"""
import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd
from dateutil.tz import gettz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import twin4build as tb
from aarhus_model.skoven_model import (
    get_model, make_fcn, ZONES, DATA_DIR,
    ENVELOPE_RESULT_PICKLE, HYDRONIC_RESULT_PICKLE, AHU_RESULT_PICKLE,
)

TZ = "Europe/Copenhagen"


def _sim_series(model, start, n, comp, port):
    arr = model.components[comp].output[port].history(i_s=0, i_c=0).detach().cpu().numpy().reshape(-1)
    idx = pd.date_range(start, periods=len(arr), freq="600s").tz_convert("UTC")
    return pd.Series(arr, index=idx)


def _obs_series(csv):
    s = pd.read_csv(os.path.join(DATA_DIR, csv), index_col=0, parse_dates=True)["value"]
    s.index = pd.to_datetime(s.index, utc=True)
    return s


def _rmse(sim, obs):
    j = pd.concat([sim.rename("s"), obs.rename("o")], axis=1).dropna()
    if j.empty:
        return float("nan"), float("nan"), float("nan"), 0
    return float(np.sqrt(((j.s - j.o) ** 2).mean())), float(j.s.mean()), float(j.o.mean()), len(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-03-31")
    ap.add_argument("--end", default="2025-04-07")
    ap.add_argument("--step", type=int, default=600)
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    tz = gettz(TZ)
    start = datetime.datetime.fromisoformat(args.start).replace(tzinfo=tz)
    end = datetime.datetime.fromisoformat(args.end).replace(tzinfo=tz)

    model = get_model(id="skoven_calib_eval", fcn_=make_fcn(calibration_mode=True),
                      calibration_mode=True)
    for label, p in [("envelope", ENVELOPE_RESULT_PICKLE), ("hydronic", HYDRONIC_RESULT_PICKLE),
                     ("ahu", AHU_RESULT_PICKLE)]:
        if os.path.exists(p):
            model.load_estimation_result(p)
            print(f"loaded {label} pickle")
        else:
            print(f"(no {label} pickle — using priors)")

    tb.Simulator(model).simulate(start_time=start, end_time=end, step_size=args.step)

    print(f"\n=== Calibration fit {args.start} -> {args.end} ===")
    print(f"{'signal':28s} {'RMSE':>7} {'sim_mean':>9} {'obs_mean':>9} {'n':>6}")
    targets = [(f"{z}_indoor_temp_sensor", "measuredValue", f"{z}_indoor_temperature.csv") for z in ZONES]
    targets.append(("ecl310_TRetHea_y", "measuredValue", "ecl310_TRetHea_y_processed.csv"))
    series = {}
    for comp, port, csv in targets:
        sim = _sim_series(model, start, 0, comp, port)
        obs = _obs_series(csv)
        r, sm, om, n = _rmse(sim, obs)
        series[comp] = (sim, obs)
        print(f"{comp:28s} {r:7.2f} {sm:9.1f} {om:9.1f} {n:6d}")

    tot = sum(_sim_series(model, start, 0, f"{z}_radiator", "Power") for z in ZONES)
    print(f"\nradiator total power: mean={tot.mean():.0f} W  max={tot.max():.0f} W  "
          f"nonzero={bool(np.any(tot > 1))}")

    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = os.path.join(SCRIPT_DIR, "plots")
        os.makedirs(out, exist_ok=True)
        for comp, (sim, obs) in series.items():
            plt.figure(figsize=(12, 5))
            plt.plot(sim.index, sim.values, label="sim")
            plt.plot(obs.reindex(sim.index).index, obs.reindex(sim.index).values, label="measured", alpha=.7)
            plt.title(comp); plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(out, f"calib_{comp}.png")); plt.close()
        print(f"plots written to {out}")


if __name__ == "__main__":
    main()
