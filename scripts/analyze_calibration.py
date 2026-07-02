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
    get_model, make_fcn, make_envelope_fcn, ZONES, DATA_DIR,
    compute_winter_sizing, apply_energy_balance_sizing,
    ENVELOPE_RESULT_PICKLE, HYDRONIC_RESULT_PICKLE, AHU_RESULT_PICKLE,
)

TZ = "Europe/Copenhagen"
# Winter window used to derive the energy-balance radiator sizing (matches Stage 2).
HYDRONIC_WINDOW = ("2026-01-08", "2026-01-15")

# The hydronic loop is driven by BMS water signals (supply/return temperature,
# setpoint curve) that only exist in the spring overlap window. Outside that
# range those sensors are all-NaN and the full model can't be simulated, so the
# envelope window must be analysed with the envelope-only (heat-boundary) model.
_WATER_CSV = "ecl310_TSupHea_y_processed.csv"


def _data_start(csv):
    """First timestamp (UTC) present in a driving CSV."""
    idx = pd.read_csv(os.path.join(DATA_DIR, csv), index_col=0, parse_dates=True).index
    return pd.to_datetime(idx, utc=True).min()


def _water_data_start():
    """First timestamp (UTC) for which BMS water signals exist."""
    return _data_start(_WATER_CSV)


def _sim_series(model, start, comp, port, step):
    arr = model.components[comp].output[port].history(i_s=0, i_c=0).detach().cpu().numpy().reshape(-1)
    idx = pd.date_range(start, periods=len(arr), freq=f"{step}s").tz_convert("UTC")
    return pd.Series(arr, index=idx)


def _obs_series(csv):
    s = pd.read_csv(os.path.join(DATA_DIR, csv), index_col=0, parse_dates=True)["value"]
    s.index = pd.to_datetime(s.index, utc=True)
    return s


def _rmse(sim, obs):
    j = pd.concat([sim.rename("s"), obs.rename("o")], axis=1, sort=False).dropna()
    if j.empty:
        return float("nan"), float("nan"), float("nan"), 0
    return float(np.sqrt(((j.s - j.o) ** 2).mean())), float(j.s.mean()), float(j.o.mean()), len(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-03-31")
    ap.add_argument("--end", default="2025-04-07")
    ap.add_argument("--step", type=int, default=600)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument(
        "--stage", choices=["auto", "envelope", "hydronic"], default="auto",
        help="Which model topology to simulate. 'auto' picks envelope-only when "
             "the window predates the BMS water signals, else the full model.",
    )
    ap.add_argument(
        "--warmup-days", type=float, default=0.0,
        help="Simulate this many days before --start to charge the thermal mass "
             "(states init at 20 degC), then discard the spin-up before computing "
             "RMSE/plots. Clamped so the spin-up never precedes the driving data.",
    )
    args = ap.parse_args()

    tz = gettz(TZ)
    start = datetime.datetime.fromisoformat(args.start).replace(tzinfo=tz)
    end = datetime.datetime.fromisoformat(args.end).replace(tzinfo=tz)

    stage = args.stage
    if stage == "auto":
        water_start = _water_data_start()
        stage = "envelope" if pd.Timestamp(end).tz_convert("UTC") <= water_start else "hydronic"
        print(f"auto-selected stage '{stage}' (BMS water data starts {water_start:%Y-%m-%d})")

    # Spin-up: simulate from sim_start (<= start) to charge thermal mass, then
    # evaluate only on [start, end]. Clamp the spin-up to the earliest driving
    # data for this stage so the extra steps don't hit out-of-range NaNs.
    eval_start_utc = pd.Timestamp(start).tz_convert("UTC")
    eval_end_utc = pd.Timestamp(end).tz_convert("UTC")
    sim_start = start - datetime.timedelta(days=args.warmup_days)
    if args.warmup_days > 0:
        rep_csv = _WATER_CSV if stage == "hydronic" else "outdoor_temperature.csv"
        data_start = _data_start(rep_csv)
        if pd.Timestamp(sim_start).tz_convert("UTC") < data_start:
            sim_start = data_start.tz_convert(tz).to_pydatetime()
            print(f"warmup clamped to {rep_csv} start {data_start:%Y-%m-%d %H:%M}Z")
        print(f"spin-up {sim_start:%Y-%m-%d %H:%M} -> eval from {args.start} "
              f"({(start - sim_start).total_seconds() / 86400:.2f} d discarded)")

    # The envelope-only model injects measured heat (available year-round); the
    # full model drives the hydronic loop from the spring-only water sensors.
    if stage == "envelope":
        fcn_ = make_envelope_fcn(calibration_mode=True)
        pickles = [("envelope", ENVELOPE_RESULT_PICKLE)]
    else:
        fcn_ = make_fcn(calibration_mode=True)
        pickles = [("envelope", ENVELOPE_RESULT_PICKLE), ("hydronic", HYDRONIC_RESULT_PICKLE),
                   ("ahu", AHU_RESULT_PICKLE)]

    model = get_model(id="skoven_calib_eval", fcn_=fcn_, calibration_mode=True)
    # The closed-loop hydronic radiators are SIZED from the envelope energy balance
    # (Stage 2); apply the same sizing before loading the estimated pickle so the
    # evaluated model matches calibration. (Envelope-only stage has no radiators.)
    if stage != "envelope":
        wtz = gettz(TZ)
        wstart = datetime.datetime.fromisoformat(HYDRONIC_WINDOW[0]).replace(tzinfo=wtz)
        wend = datetime.datetime.fromisoformat(HYDRONIC_WINDOW[1]).replace(tzinfo=wtz)
        apply_energy_balance_sizing(model, compute_winter_sizing(wstart, wend, step=args.step))
    for label, p in pickles:
        if os.path.exists(p):
            model.load_estimation_result(p)
            print(f"loaded {label} pickle")
        else:
            print(f"(no {label} pickle — using priors)")

    tb.Simulator(model).simulate(start_time=sim_start, end_time=end, step_size=args.step)

    print(f"\n=== Calibration fit {args.start} -> {args.end} (stage: {stage}) ===")
    print(f"{'signal':28s} {'RMSE':>7} {'sim_mean':>9} {'obs_mean':>9} {'n':>6}")
    targets = [(f"{z}_indoor_temp_sensor", "measuredValue", f"{z}_indoor_temperature.csv") for z in ZONES]
    if stage == "hydronic":
        targets.append(("ecl310_TRetHea_y", "measuredValue", "ecl310_TRetHea_y_processed.csv"))
    series = {}
    for comp, port, csv in targets:
        # Sim index starts at sim_start; trim the spin-up to the eval window.
        sim = _sim_series(model, sim_start, comp, port, args.step).loc[eval_start_utc:eval_end_utc]
        obs = _obs_series(csv)
        r, sm, om, n = _rmse(sim, obs)
        series[comp] = (sim, obs)
        print(f"{comp:28s} {r:7.2f} {sm:9.1f} {om:9.1f} {n:6d}")

    if stage == "hydronic":
        tot = sum(_sim_series(model, sim_start, f"{z}_radiator", "Power", args.step)
                  for z in ZONES).loc[eval_start_utc:eval_end_utc]
        print(f"\nradiator total power: mean={tot.mean():.0f} W  max={tot.max():.0f} W  "
              f"nonzero={bool(np.any(tot > 1))}")

    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = os.path.join(SCRIPT_DIR, "plots")
        os.makedirs(out, exist_ok=True)
        for comp, (sim, obs) in series.items():
            # Measured data is on its own (coarser/offset) sampling grid, so plot
            # it on its native timestamps clipped to the sim window — reindexing
            # onto the sim grid would drop the unaligned majority to NaN and the
            # line would all but vanish.
            obs_win = obs.loc[(obs.index >= sim.index.min()) & (obs.index <= sim.index.max())]
            plt.figure(figsize=(12, 5))
            plt.plot(sim.index, sim.values, label="sim")
            plt.plot(obs_win.index, obs_win.values, label="measured", alpha=.7)
            plt.title(comp); plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(out, f"calib_{comp}.png")); plt.close()
        print(f"plots written to {out}")


if __name__ == "__main__":
    main()
