"""Skoven model evaluation: baseline simulation + RL policy rollout + plots.

Ported from T4BGymUseCase/use_case/model_eval.py, adapted for:
  - Skoven's 4-room ring (room_a..d), single building-wide heating setpoint
    (no per-room control, no cooling setpoints — residential heating only).
  - The Twin4Build dev-branch API: `.history()` on a Scalar output has no
    `i_v` kwarg (only Vector does); `GymSimulator` (the simulator instance
    behind a gym env) sets `dateTimeSteps` (camelCase), while the base
    `tb.Simulator.simulate()` sets `date_time_steps` (snake_case). Both are
    handled transparently by `_dt_index` / `_hist` below.
  - Train/eval windows read from `use_case/rl_config.py` (skoven.yaml), not
    hardcoded here, so this file can't drift out of sync with the training
    script's windows the way the old per-file constants did.
"""
import sys
import os
from gymnasium.core import Wrapper
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import twin4build as tb
from tqdm import tqdm
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(MAIN_DIR)

from aarhus_model.skoven_model import load_model_and_params, ZONES
from use_case.rl_config import (
    POLICY_CONFIG_PATH, PLOTS_DIR, COMFORT_SETPOINT_C, COMFORT_MIN_C, COMFORT_MAX_C,
    load_rl_windows, dst_safe_chunks,
)

device = "cpu"

TRAIN_START, TRAIN_END, EVAL_START, EVAL_END = load_rl_windows()


def aggregate_kpis(kpi_list: list, weights: list) -> dict:
    """Combine per-segment KPI dicts (see dst_safe_chunks) into one: energies
    and degree-hours sum across segments, percentages are duration-weighted
    averages."""
    if len(kpi_list) == 1:
        return kpi_list[0]
    total_w = sum(weights)
    out = {}
    for key in kpi_list[0]:
        if key.endswith("_kWh") or key.endswith("_hours"):
            out[key] = sum(k[key] for k in kpi_list)
        else:
            out[key] = sum(k[key] * w for k, w in zip(kpi_list, weights)) / total_w
    return out


def get_baseline(model, step_size: int = 600, plots_dir: str = None):
    """Forward-simulate the model with default (uncontrolled) schedules across
    the eval window — i.e. no rule-based curve, just the sim-mode component
    defaults (see baseline_eval.py for the outdoor-reset curve baseline).
    Diagnostic-only entry point (run via `python model_eval.py`); DST-safe via
    dst_safe_chunks (see rl_config.py for why that's necessary)."""
    kpi_list, weights = [], []
    for c_start, c_end in dst_safe_chunks(EVAL_START, EVAL_END):
        simulator = tb.Simulator(model)
        simulator.simulate(start_time=c_start, end_time=c_end, step_size=step_size)
        kpis = plot_results(simulator, save_plots=True, plots_dir=plots_dir)
        kpi_list.append(kpis)
        weights.append((c_end - c_start).total_seconds())
    return aggregate_kpis(kpi_list, weights)


def test_model(env, model, eval_start=None, eval_end=None, plots_dir: str = None,
                save_plots: bool = True):
    """Roll a trained PPO policy on `env` over [eval_start, eval_end).

    Defaults to the full EVAL window. The caller is responsible for DST
    safety of the window (see test_model_chunked, the safe default entry
    point — this single-segment version is what it calls per segment, and is
    also fine to call directly for any window known not to cross a DST
    transition, e.g. the short smoke-test window).
    """
    if eval_start is None:
        eval_start = EVAL_START
    if eval_end is None:
        eval_end = EVAL_END
    step_size = env.unwrapped.step_size
    episode_length = int((eval_end - eval_start).total_seconds() / step_size)

    if isinstance(env, Wrapper):
        env.unwrapped.random_start = False
        env.unwrapped.global_start_time = eval_start
        env.unwrapped.episode_length = episode_length
    else:
        env.random_start = False
        env.global_start_time = eval_start
        env.episode_length = episode_length

    obs, _ = env.reset()
    done = False
    rewards = []
    print(f"Simulating {eval_start} -> {eval_end} ({episode_length} steps)...")

    pbar = tqdm(total=episode_length, desc="Simulation Progress")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
        pbar.update(1)
    pbar.close()

    kpis = plot_results(env.unwrapped.simulator, rewards, save_plots=save_plots, plots_dir=plots_dir)
    return rewards, kpis


def test_model_chunked(env, model, eval_start=None, eval_end=None, plots_dir: str = None):
    """DST-safe RL rollout over the full eval window: splits at any DST
    transition inside the window (see rl_config.dst_safe_chunks) and runs
    each segment as its own episode on the SAME env (reused, not rebuilt),
    aggregating KPIs afterwards. This is what `skoven_RL_control.evaluate()`
    and `baseline_eval.compare()` should call instead of `test_model` for any
    window that might span a transition (a multi-month eval window almost
    always does)."""
    eval_start = EVAL_START if eval_start is None else eval_start
    eval_end = EVAL_END if eval_end is None else eval_end
    plots_dir = PLOTS_DIR if plots_dir is None else plots_dir

    chunks = dst_safe_chunks(eval_start, eval_end)
    if len(chunks) > 1:
        print(f"Eval window spans {len(chunks) - 1} DST transition(s) — "
              f"running as {len(chunks)} segments and aggregating KPIs.")

    kpi_list, weights, all_rewards = [], [], []
    for i, (c_start, c_end) in enumerate(chunks):
        seg_dir = plots_dir if len(chunks) == 1 else f"{plots_dir}_seg{i}"
        rewards, kpis = test_model(env, model, eval_start=c_start, eval_end=c_end, plots_dir=seg_dir)
        kpi_list.append(kpis)
        weights.append((c_end - c_start).total_seconds())
        all_rewards.extend(rewards)

    return all_rewards, aggregate_kpis(kpi_list, weights)


def _dt_index(simulator) -> pd.DatetimeIndex:
    """Simulation timestamps, handling both simulator flavors:
      - GymSimulator (env.unwrapped.simulator): sets `dateTimeSteps`, a flat
        1-D list (t4b_gym_env.py's `initialize_simulation`).
      - base tb.Simulator (after `.simulate()`): sets `date_time_steps`, a 2-D
        array (n_periods, n_steps) — we use period 0, the only period here.
    """
    dts = getattr(simulator, "dateTimeSteps", None)
    if dts is not None:
        return pd.DatetimeIndex(dts)
    dts2d = simulator.date_time_steps
    flat = dts2d[0] if getattr(dts2d, "ndim", 1) == 2 else dts2d
    return pd.DatetimeIndex(flat)


def _hist(simulator, component_id, port, which="output"):
    """Return a 1-D numpy array for a component port's simulated history.

    Dev-branch Vector outputs expose `.history(i_s=, i_c=, i_v=)`; Scalar
    outputs expose `.history(i_s=, i_c=)` with no `i_v` (raises TypeError if
    passed). Try the Vector signature first, fall back to the Scalar one —
    every port used here (sensors, radiator/coil Power, schedule values) is a
    Scalar in practice, so this mostly takes the fallback branch.
    """
    comp = simulator.model.components[component_id]
    store = comp.output if which == "output" else comp.input
    h_obj = store[port]
    try:
        h = h_obj.history(i_s=0, i_c=0, i_v=0)
    except TypeError:
        h = h_obj.history(i_s=0, i_c=0)
    return h.detach().cpu().numpy().reshape(-1)


def _series(simulator, component_id, output_key, tz="Europe/Copenhagen", which="output"):
    data = _hist(simulator, component_id, output_key, which=which)
    idx = _dt_index(simulator)
    s = pd.Series(data=data, index=idx[: len(data)])
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    s.index = s.index.tz_convert(tz)
    return s


def _step_hours(sim_times: pd.DatetimeIndex) -> float:
    """Uniform step size in hours, from the first two timestamps."""
    if len(sim_times) < 2:
        return 600.0 / 3600.0
    return (sim_times[1] - sim_times[0]).total_seconds() / 3600.0


def plot_results(simulator: tb.Simulator, rewards=None, plotting_stepSize=600,
                  save_plots=False, plots_dir: str = None):
    if plots_dir is None:
        plots_dir = PLOTS_DIR if save_plots else "plots"
    os.makedirs(plots_dir, exist_ok=True)
    sim_times = _dt_index(simulator)
    dt_hours = _step_hours(sim_times)

    # --- Per-room temperature vs the fixed comfort setpoint ---
    zone_temp_raw = {}
    for zone_id in ZONES:
        T = _hist(simulator, f"{zone_id}_indoor_temp_sensor", "measuredValue")
        zone_temp_raw[zone_id] = T
        temp = _series(simulator, f"{zone_id}_indoor_temp_sensor", "measuredValue")
        temp = temp.resample(pd.Timedelta(seconds=plotting_stepSize)).mean()

        plt.figure(figsize=(12, 6))
        plt.plot(temp.index, temp.values, label="Indoor Temperature", linewidth=2)
        plt.axhline(COMFORT_MIN_C, linestyle="--", color="tab:blue", label="Comfort band")
        plt.axhline(COMFORT_MAX_C, linestyle="--", color="tab:red")
        plt.title(f"{zone_id} — Temperature")
        plt.xlabel("Time")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(plots_dir, f"{zone_id}_temperature_setpoint.png"))
        plt.close()

    # --- Comfort KPIs: two-sided degree-hours outside [MIN,MAX], % time in band,
    #     and RMSE from the COMFORT_SETPOINT_C (21 °C) target. The RMSE-to-target
    #     is what the deviation-from-target reward term optimises, so it makes the
    #     RL-vs-baseline comparison meaningful on the tight-comfort objective (the
    #     band KPIs alone read 100%/0 for anything inside the wide window). ---
    print("\n--- Comfort KPIs ---")
    total_degree_hours = 0.0
    total_in_band_frac = []
    zone_rmse_target = []
    for zone_id, T in zone_temp_raw.items():
        viol_degC = np.maximum(0.0, COMFORT_MIN_C - T) + np.maximum(0.0, T - COMFORT_MAX_C)
        degree_hours = float(np.sum(viol_degC) * dt_hours)
        in_band_pct = float(np.mean((T >= COMFORT_MIN_C) & (T <= COMFORT_MAX_C)) * 100.0)
        rmse_target = float(np.sqrt(np.mean((T - COMFORT_SETPOINT_C) ** 2)))
        total_degree_hours += degree_hours
        total_in_band_frac.append(in_band_pct)
        zone_rmse_target.append(rmse_target)
        print(f"  {zone_id}: {degree_hours:8.2f} °C·h outside "
              f"[{COMFORT_MIN_C:.0f},{COMFORT_MAX_C:.0f}]°C, "
              f"{in_band_pct:5.1f}% in band, "
              f"RMSE→{COMFORT_SETPOINT_C:.0f}°C = {rmse_target:4.2f} °C")
    mean_rmse_target = float(np.mean(zone_rmse_target))
    print(f"  TOTAL: {total_degree_hours:8.2f} °C·h, "
          f"{np.mean(total_in_band_frac):5.1f}% mean time in band, "
          f"mean RMSE→{COMFORT_SETPOINT_C:.0f}°C = {mean_rmse_target:4.2f} °C")

    # --- Hydronic energy — sum of per-radiator Power ---
    total_radiator_power = None
    for zone_id in ZONES:
        P = _hist(simulator, f"{zone_id}_radiator", "Power")
        total_radiator_power = P if total_radiator_power is None else total_radiator_power + P
    heating_energy_kWh = float(np.sum(total_radiator_power) * dt_hours / 1000.0)
    print(f"\n--- Energy KPIs ---")
    print(f"Mean total radiator Power: {np.mean(total_radiator_power):.1f} W")
    print(f"Total heating energy: {heating_energy_kWh:.2f} kWh "
          f"over {len(sim_times) * dt_hours:.1f} h")

    plt.figure(figsize=(12, 6))
    plt.plot(sim_times[: len(total_radiator_power)], total_radiator_power,
              label="Total radiator power (W)", linewidth=2)
    plt.title("Hydronic heat output (sum of radiators ≈ varme meter)")
    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(plots_dir, "total_radiator_power.png"))
    plt.close()

    # --- AHU fan + coil power ---
    ahu_energy_kWh = 0.0
    try:
        fan_power = _hist(simulator, "vent_power_sensor", "measuredValue")
        ahu_energy_kWh += float(np.sum(fan_power) * dt_hours / 1000.0)
        print(f"Mean fan power: {np.mean(fan_power):.1f} W")
        plt.figure(figsize=(12, 6))
        plt.plot(sim_times[: len(fan_power)], fan_power, label="Fan Power", linewidth=2)
        plt.title("AHU Fan Power Consumption")
        plt.xlabel("Time")
        plt.ylabel("Power (W)")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(plots_dir, "ahu_fan_power.png"))
        plt.close()
    except KeyError:
        print("Info: AHU fan power sensor not present in this model variant.")

    try:
        coil_power = _hist(simulator, "supply_heating_coil", "Power")
        ahu_energy_kWh += float(np.sum(coil_power) * dt_hours / 1000.0)
        print(f"Mean coil power: {np.mean(coil_power):.1f} W")
    except KeyError:
        pass
    print(f"Total AHU energy (fan+coil): {ahu_energy_kWh:.2f} kWh")

    # --- Supply water temperature ---
    try:
        supply_T = _hist(simulator, "ecl310_TSupHea_y", "measuredValue")
        print(f"\nSupply water temp: mean {np.mean(supply_T):.1f} °C, "
              f"peak {np.max(supply_T):.1f} °C")
    except KeyError:
        pass

    # --- Action plots from policy schema ---
    if os.path.exists(POLICY_CONFIG_PATH):
        with open(POLICY_CONFIG_PATH, "r") as f:
            policy_config = json.load(f)

        component_ids, signal_keys = [], []
        for component_id, actions in policy_config.get("actions", {}).items():
            for _action_name, action_config in actions.items():
                component_ids.append(component_id)
                signal_keys.append(action_config["signal_key"])

        for component_id, signal_key in zip(component_ids, signal_keys):
            comp = simulator.model.components.get(component_id)
            if comp is None:
                continue
            # Action signals are either normal inputs (e.g. damper position) or
            # output-overrides for source components with no input port (e.g.
            # a ScheduleSystem's scheduleValue) — check both, matching the
            # distinction t4b_gym_env.py's `_populate_from_json` makes.
            if signal_key in comp.input:
                which = "input"
            elif signal_key in comp.output:
                which = "output"
            else:
                continue
            try:
                action = _hist(simulator, component_id, signal_key, which=which)
            except (KeyError, AttributeError):
                continue
            plt.figure(figsize=(12, 6))
            plt.plot(sim_times[: len(action)], action,
                      label=f"{component_id} — {signal_key}", linewidth=2)
            plt.title(f"Action: {component_id} — {signal_key}")
            plt.xlabel("Time")
            plt.ylabel("Action Value")
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
            plt.tight_layout()
            if save_plots:
                plt.savefig(os.path.join(plots_dir, f"action_{component_id}_{signal_key}.png"))
            plt.close()

    if rewards is not None:
        rewards_arr = np.array(rewards).squeeze()
        plt.figure(figsize=(12, 6))
        plt.plot(rewards_arr, label="Reward", linewidth=1.5)
        plt.title("Per-step reward")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(plots_dir, "rewards.png"))
        plt.close()
        print(f"\nMean reward: {rewards_arr.mean():.4f}")

    return {
        "heating_energy_kWh": heating_energy_kWh,
        "ahu_energy_kWh": ahu_energy_kWh,
        "comfort_degree_hours": total_degree_hours,
        "comfort_pct_in_band": float(np.mean(total_in_band_frac)),
        "comfort_rmse_target": mean_rmse_target,
    }


if __name__ == "__main__":
    model = load_model_and_params()
    get_baseline(model)
