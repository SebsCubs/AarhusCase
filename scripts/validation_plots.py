"""Model-validation figures for the Skoven real-data calibration.

Produces the figures and metrics used in the "Model validation" section of the
Aarhus/Skoven paper, mirroring the structure of the BOPTEST surrogate paper:

  * per-room indoor-temperature overlays (simulated vs. measured)        -> like west_indoor_temp_sensor_overlay
  * return-water temperature overlay (the harder hydronic signal)        -> like vent_supply_airflow_sensor_overlay
  * a multi-panel scatter of simulated vs. measured for every signal     -> like multi_panel_scatter
  * a CV-RMSE / NMBE table printed to stdout (LaTeX-ready)               -> like tab:sim_model_metrics

Unlike the surrogate paper (where the "truth" is a high-fidelity BOPTEST
emulator), here the reference is REAL measured building data (ReMoni zone
sensors + BMS district-heating water temperatures), so the two calibration
stages are evaluated on the windows where their driving data actually exist:

  * envelope (zone temperatures)        -> cold January window (2025-01-15..22)
  * closed-loop hydronic (zones+supply+return) -> winter window (2026-01-08..15)
  * energy consistency (Sigma radiator Power vs Q_meter, radiator vs Q_meter/4)
        -> spring meter overlap (2025-04), the only window where the district-heat
           meter coincides with the BMS water + ReMoni data.

Usage:
    uv run python scripts/validation_plots.py                 # all figures + metrics
    uv run python scripts/validation_plots.py --no-plots      # metrics only
    uv run python scripts/validation_plots.py --envelope-only # Stage-1 only
    uv run python scripts/validation_plots.py --no-consistency
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
    get_model, make_fcn, make_envelope_fcn, ZONES, DATA_DIR, AHU_DEFAULTS,
    compute_winter_sizing, apply_energy_balance_sizing,
    ENVELOPE_RESULT_PICKLE, HYDRONIC_RESULT_PICKLE, AHU_RESULT_PICKLE,
)

TZ = "Europe/Copenhagen"

ENVELOPE_WINDOW = ("2025-01-15", "2025-01-22")
HYDRONIC_WINDOW = ("2026-01-08", "2026-01-15")  # real winter heating regime (dT ~14 C)
# The ONLY window where the district-heat meter (ends 2025-05-31) coincides with
# the BMS water + ReMoni data AND still carries a heating signal is the narrow
# late-March overlap (the meter reads ~0 kW from April onward — space heating is a
# winter signal). Used for the out-of-window energy consistency check.
CONSISTENCY_WINDOW = ("2025-03-31", "2025-04-05")  # starts inside the BMS data
                                                   # range (begins 2025-03-30 05:00 UTC)
# Meter reset/rollover spikes (diff(Energi) artifacts) clip at ~10 kW for this
# ~1 kW building; drop anything above this physical ceiling before aggregating.
METER_MAX_W = 5000.0
STEP = 600

# Pretty labels for the figures / table.
ROOM_LABEL = {
    "room_a": "Room A", "room_b": "Room B",
    "room_c": "Room C", "room_d": "Room D",
}


def _sim_series(model, start, comp, port, step):
    arr = model.components[comp].output[port].history(i_s=0, i_c=0).detach().cpu().numpy().reshape(-1)
    idx = pd.date_range(start, periods=len(arr), freq=f"{step}s").tz_convert("UTC")
    return pd.Series(arr, index=idx)


def _obs_series(csv):
    s = pd.read_csv(os.path.join(DATA_DIR, csv), index_col=0, parse_dates=True)["value"]
    s.index = pd.to_datetime(s.index, utc=True)
    return s


def _align(sim, obs):
    """Inner-join simulated and measured onto the overlap timestamps."""
    j = pd.concat([sim.rename("s"), obs.rename("o")], axis=1, sort=False).dropna()
    return j


def _metrics(j):
    """CV-RMSE [%] and NMBE [%] from an aligned (s, o) frame."""
    if j.empty:
        return float("nan"), float("nan"), 0
    err = j.s - j.o
    rmse = float(np.sqrt((err ** 2).mean()))
    obs_mean = float(j.o.mean())
    cvrmse = 100.0 * rmse / obs_mean if obs_mean else float("nan")
    nmbe = 100.0 * float(err.mean()) / obs_mean if obs_mean else float("nan")
    return cvrmse, nmbe, len(j)


def _simulate(stage, start_str, end_str):
    tz = gettz(TZ)
    start = datetime.datetime.fromisoformat(start_str).replace(tzinfo=tz)
    end = datetime.datetime.fromisoformat(end_str).replace(tzinfo=tz)

    if stage == "envelope":
        fcn_ = make_envelope_fcn(calibration_mode=True)
        pickles = [ENVELOPE_RESULT_PICKLE]
    else:
        fcn_ = make_fcn(calibration_mode=True)
        pickles = [ENVELOPE_RESULT_PICKLE, HYDRONIC_RESULT_PICKLE, AHU_RESULT_PICKLE]

    model = get_model(id=f"skoven_valid_{stage}", fcn_=fcn_, calibration_mode=True)
    for p in pickles:
        if os.path.exists(p):
            model.load_estimation_result(p)
    tb.Simulator(model).simulate(start_time=start, end_time=end, step_size=STEP)
    return model, start


def collect_envelope():
    """Stage 1 only — per-room zone temperatures (cold January window)."""
    out = {}
    model, start = _simulate("envelope", *ENVELOPE_WINDOW)
    for z in ZONES:
        sim = _sim_series(model, start, f"{z}_indoor_temp_sensor", "measuredValue", STEP)
        obs = _obs_series(f"{z}_indoor_temperature.csv")
        out[z] = (f"{ROOM_LABEL[z]} Temp.", _align(sim, obs))
    return out


def _ts(window_str):
    tz = gettz(TZ)
    return (datetime.datetime.fromisoformat(window_str[0]).replace(tzinfo=tz),
            datetime.datetime.fromisoformat(window_str[1]).replace(tzinfo=tz))


def _build_closed_model(id_):
    """Build the calibrated closed-loop model: envelope params + energy-balance
    sizing (computed on the winter window, as in Stage 2) + the estimated hydronic
    pickle (PID gains + radiator thermal mass). The sizing is applied at runtime
    (it is not stored in the pickle), so the validated model matches calibration.
    """
    model = get_model(id=id_, fcn_=make_fcn(calibration_mode=True),
                      calibration_mode=True)
    if os.path.exists(ENVELOPE_RESULT_PICKLE):
        model.load_estimation_result(ENVELOPE_RESULT_PICKLE)
    wstart, wend = _ts(HYDRONIC_WINDOW)
    sizing = compute_winter_sizing(wstart, wend, step=STEP)
    apply_energy_balance_sizing(model, sizing)
    if os.path.exists(HYDRONIC_RESULT_PICKLE):
        model.load_estimation_result(HYDRONIC_RESULT_PICKLE)
    return model, sizing


def collect_closed():
    """Validate the CLOSED-LOOP model on the winter window: the supply temperature
    is a produced/controlled OUTPUT (not a replayed boundary), so supply, return,
    and all four zone temperatures are scored together."""
    start, end = _ts(HYDRONIC_WINDOW)
    model, _ = _build_closed_model("skoven_valid_closed")
    tb.Simulator(model).simulate(start_time=start, end_time=end, step_size=STEP)

    # Discard the closed-loop cold-start transient (PID/recirculation ringing in
    # the first few hours) before scoring.
    warmup = 24  # steps (= 4 h at 600 s)

    out = {}
    for z in ZONES:
        sim = _sim_series(model, start, f"{z}_indoor_temp_sensor", "measuredValue", STEP)[warmup:]
        out[z] = (f"{ROOM_LABEL[z]} Temp.", _align(sim, _obs_series(f"{z}_indoor_temperature.csv")))
    sup = _sim_series(model, start, "ecl310_TSupHea_y", "measuredValue", STEP)[warmup:]
    out["ecl310_TSupHea_y"] = ("Supply Water Temp.", _align(sup, _obs_series("ecl310_TSupHea_y_processed.csv")))
    ret = _sim_series(model, start, "ecl310_TRetHea_y", "measuredValue", STEP)[warmup:]
    out["ecl310_TRetHea_y"] = ("Return Water Temp.", _align(ret, _obs_series("ecl310_TRetHea_y_processed.csv")))

    # Air-loop (AHU/MVHR) diagnostics — no measured AHU data, so these are reported
    # rather than scored. Ventilation airflow per room, supply/return air temp, and
    # the AHU energy (coil + fan) that offsets the ventilation heat loss.
    per_room_flow, total_flow = {}, 0.0
    for z in ZONES:
        f = _sim_series(model, start, f"{z}_supply_damper", "airFlowRate", STEP)[warmup:].mean()
        per_room_flow[z] = float(f)
        total_flow += float(f)
    ahu_info = {
        "per_room_flow": per_room_flow,
        "total_flow": total_flow,
        "ach": AHU_DEFAULTS["ventilation_ach"],
        "supply_air_T": float(_sim_series(model, start, "vent_supply_air_temp_sensor", "measuredValue", STEP)[warmup:].mean()),
        "return_air_T": float(_sim_series(model, start, "vent_return_air_temp_sensor", "measuredValue", STEP)[warmup:].mean()),
        "coil_W": float(_sim_series(model, start, "supply_heating_coil", "heatingPower", STEP)[warmup:].mean()),
        "fan_W": float(_sim_series(model, start, "vent_power_sensor", "measuredValue", STEP)[warmup:].mean()),
    }
    return out, ahu_info


def print_ahu(info):
    print("\n% --- Air loop (MVHR) diagnostics — reported, not scored (no AHU data) ---")
    print(f"  ventilation target = {info['ach']:.1f} ACH;  total supply airflow = "
          f"{info['total_flow']:.3f} kg/s")
    for z, f in info["per_room_flow"].items():
        print(f"    {ROOM_LABEL.get(z, z):8s} airflow = {f:.3f} kg/s")
    print(f"  AHU supply air T = {info['supply_air_T']:.2f} C  (return air T = "
          f"{info['return_air_T']:.2f} C; ~equal ⇒ ventilation ~thermally neutral)")
    print(f"  AHU energy: coil heating = {info['coil_W']:.0f} W  +  fan = {info['fan_W']:.0f} W")


def collect_consistency():
    """Out-of-window energy-consistency check against the district-heat meter.

    The meter (ends 2025-05-31) only overlaps the BMS water + ReMoni data in spring
    2025, so we simulate the calibrated closed loop there and compare the simulated
    radiator heat output to the metered district-heat power: total Sigma(Power) vs
    Q_meter, and per-radiator mean Power vs Q_meter/4 (the split the Stage-1
    envelope assumed). Returns ({"total_power": (label, aligned_frame)}, info-dict).
    """
    start, end = _ts(CONSISTENCY_WINDOW)
    model, _ = _build_closed_model("skoven_valid_consistency")
    tb.Simulator(model).simulate(start_time=start, end_time=end, step_size=STEP)
    warmup = 24

    # Simulated total radiator heat output as a time series (sum of 4 radiators).
    total = None
    per_rad_mean = {}
    for z in ZONES:
        sim = _sim_series(model, start, f"{z}_radiator", "Power", STEP)[warmup:]
        per_rad_mean[z] = float(sim.mean())
        total = sim if total is None else total.add(sim, fill_value=0.0)

    # Metered district-heat power [kW] -> W, aligned to the simulated series.
    # Drop meter reset/rollover spikes above the physical ceiling before scoring.
    meter_W = _obs_series("varme_meter_power_kW.csv") * 1000.0
    meter_W = meter_W[meter_W <= METER_MAX_W]
    frame = _align(total, meter_W)

    n = len(ZONES)
    q_meter_W = float(frame.o.mean()) if not frame.empty else float("nan")
    total_sim_W = float(frame.s.mean()) if not frame.empty else float("nan")
    info = {
        "per_rad_mean": per_rad_mean,
        "q_meter_W": q_meter_W,
        "q_meter_quarter_W": q_meter_W / n,
        "total_sim_W": total_sim_W,
        "ratio": (total_sim_W / q_meter_W) if q_meter_W else float("nan"),
    }
    return {"total_heat_power": ("Total Heat Power", frame)}, info


def print_consistency(info):
    print("\n% --- Envelope <-> hydronic energy consistency (spring meter overlap) ---")
    print(f"  Q_meter (mean)      = {info['q_meter_W']:8.1f} W   (Q_meter/4 = {info['q_meter_quarter_W']:7.1f} W)")
    print(f"  Sigma radiator Power = {info['total_sim_W']:8.1f} W   (closure ratio = {info['ratio']:.2f})")
    for z, p in info["per_rad_mean"].items():
        print(f"    {ROOM_LABEL.get(z, z):8s} mean Power = {p:8.1f} W  vs Q_meter/4 = {info['q_meter_quarter_W']:7.1f} W")


def print_table(data):
    print("\n% --- LaTeX-ready metrics (CV-RMSE / NMBE) ---")
    print(f"{'Signal':22s} {'CV-RMSE(%)':>11} {'NMBE(%)':>9} {'n':>6}")
    for key, (label, j) in data.items():
        cv, nm, n = _metrics(j)
        print(f"{label:22s} {cv:11.2f} {nm:9.2f} {n:6d}")


def make_plots(data, prefix="skoven"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = os.path.join(SCRIPT_DIR, "plots")
    os.makedirs(out, exist_ok=True)

    # Overlays: rooms (ReMoni), water signals (BMS), and heat power (meter).
    for key, (label, j) in data.items():
        if "power" in key:
            src, ylabel = "Measured (meter)", "Power (W)"
        elif key.startswith("ecl310"):
            src, ylabel = "Measured (BMS)", "Temperature (°C)"
        else:
            src, ylabel = "Measured (ReMoni)", "Temperature (°C)"
        plt.figure(figsize=(12, 5))
        plt.plot(j.index, j.s, label="T4B (simulated)", linewidth=2)
        plt.plot(j.index, j.o, label=src, alpha=.8)
        plt.ylabel(ylabel)
        plt.title(f"{label} — simulated vs. measured")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(out, f"{prefix}_{key}_overlay.png"), dpi=140)
        plt.close()

    # Multi-panel scatter: simulated vs. measured with 45-degree identity line.
    keys = list(data.keys())
    ncol = 3
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, key in zip(axes, keys):
        label, j = data[key]
        ax.scatter(j.o, j.s, s=6, alpha=.4)
        lo = float(min(j.o.min(), j.s.min()))
        hi = float(max(j.o.max(), j.s.max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        cv, nm, _ = _metrics(j)
        ax.set_title(f"{label}\nCV-RMSE={cv:.1f}%  NMBE={nm:.1f}%", fontsize=9)
        ax.set_xlabel("Measured"); ax.set_ylabel("Simulated (T4B)")
        ax.grid(True, alpha=.3)
    for ax in axes[len(keys):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{prefix}_multi_panel_scatter.png"), dpi=140)
    plt.close(fig)

    print(f"\nplots written to {out}")


def make_ahu_plot(info, prefix="skoven_closed"):
    """Air-loop (MVHR) diagnostic plot: per-room 3-ACH ventilation airflow plus a
    summary box (supply/return air temp, coil + fan energy). Reported, not scored —
    there is no AHU sensor data — but it shows the air loop is now ACTIVE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = os.path.join(SCRIPT_DIR, "plots")
    os.makedirs(out, exist_ok=True)

    rooms = list(info["per_room_flow"].keys())
    flows = [info["per_room_flow"][r] for r in rooms]
    labels = [ROOM_LABEL.get(r, r) for r in rooms]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, flows, color="#3b7dd8", alpha=.85)
    for b, f in zip(bars, flows):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{f:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Supply airflow (kg/s)")
    ax.set_title(f"AHU (MVHR) ACTIVE — per-room ventilation at {info['ach']:.0f} ACH")
    ax.grid(True, axis="y", alpha=.3)
    txt = (f"total = {info['total_flow']:.3f} kg/s\n"
           f"supply air = {info['supply_air_T']:.1f} °C\n"
           f"return air = {info['return_air_T']:.1f} °C  (≈ neutral)\n"
           f"coil = {info['coil_W']:.0f} W\nfan = {info['fan_W']:.0f} W")
    ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", fc="#fff6e0", ec="#caa84a"))
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{prefix}_ahu_diagnostics.png"), dpi=140)
    plt.close(fig)
    print(f"AHU diagnostic plot written to {out}/{prefix}_ahu_diagnostics.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--envelope-only", action="store_true",
                    help="Validate only the envelope stage (per-room zone "
                         "temperatures on the cold January window).")
    ap.add_argument("--closed-loop", action="store_true",
                    help="(Default) Validate the closed-loop substation model "
                         "(supply/return are produced outputs). Kept for "
                         "backwards compatibility; it is now the only model.")
    ap.add_argument("--no-consistency", action="store_true",
                    help="Skip the spring-overlap energy-consistency check.")
    ap.add_argument("--prefix", default="skoven_closed",
                    help="Filename prefix for the plots written to scripts/plots/ "
                         "(e.g. --prefix skoven_ahu_active for an identifiable set).")
    args = ap.parse_args()

    if args.envelope_only:
        data = collect_envelope()
        print_table(data)
        if not args.no_plots:
            make_plots(data, prefix=f"{args.prefix}_envelope")
        return

    # Default: closed-loop winter validation + envelope + spring consistency.
    data = collect_envelope()
    closed, ahu_info = collect_closed()
    data.update(closed)
    if not args.no_consistency:
        cons_data, cons_info = collect_consistency()
        data.update(cons_data)
    print_table(data)
    print_ahu(ahu_info)
    if not args.no_consistency:
        print_consistency(cons_info)
    if not args.no_plots:
        make_plots(data, prefix=args.prefix)
        make_ahu_plot(ahu_info, prefix=args.prefix)


if __name__ == "__main__":
    main()
