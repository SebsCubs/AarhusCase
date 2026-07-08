"""Rule-based outdoor-reset baseline for the Skoven sim-mode twin, and an
RL-vs-baseline comparison.

The real Skoven ECL310 controller implements an outdoor-compensated supply-
water curve (aarhus_model/heating_curve.py; curve params from skoven.yaml's
`heating_curve:` section — the same curve used to build the calibration-mode
`ecl310_TSupSet_curve.csv`). This baseline drives that curve LIVE, every
step, against the same action the RL agent controls
(`ecl310_TSupSet_schedule.scheduleValue`), with dampers/AHU left at their
calibrated 3-ACH baseline (see policy_input_output.json's disabled-actions
note) — i.e. the incumbent control strategy the RL agent is trying to beat.

Usage:
    python use_case/baseline_eval.py            # baseline only
    python use_case/baseline_eval.py --compare   # RL (best_model.zip) vs. baseline
"""
import argparse
import os

import numpy as np
from stable_baselines3 import PPO

from aarhus_model.heating_curve import compute_supply_setpoint
from aarhus_model.skoven_model import ZONES
from use_case.model_eval import plot_results, aggregate_kpis, test_model_chunked
from use_case.rl_config import (
    LOG_DIR, PLOTS_DIR, COMFORT_SETPOINT_C, load_building_config, load_rl_windows,
    dst_safe_chunks,
)
from use_case.skoven_RL_control import build_env

BASELINE_PLOTS_DIR = PLOTS_DIR + "_baseline"


def _read_output(component, port, default=0.0) -> float:
    out = component.output[port]
    v = out.get() if hasattr(out, "get") else out
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _outdoor_temp_C(env) -> float:
    """Current outdoor air temperature, read directly off the model (same
    pattern SkovenGymEnv.get_reward uses for live state reads)."""
    model = env.unwrapped.simulator.model
    return _read_output(model.components["outdoor_environment"], "outdoorTemperature")


def _mean_room_temp_C(env) -> float:
    """Building-wide mean indoor temperature (the ECL310 room-compensation
    reference), read live off the four zone sensors."""
    model = env.unwrapped.simulator.model
    temps = [_read_output(model.components[f"{z}_indoor_temp_sensor"], "measuredValue",
                          COMFORT_SETPOINT_C) for z in ZONES]
    return sum(temps) / len(temps)


def _run_baseline_segment(env, action_norm, seg_start, seg_end, plots_dir, save_plots):
    """Run the constant-setpoint incumbent over one DST-safe segment on the
    (already-built, reused) env. The supervisory cascade in
    SupervisorySetpointGymSimulator turns the fixed setpoints into supply-water
    (mean setpoint → heating curve → ECL310 PID), supply-air (with economizer
    floor), and the per-room damper trim — so the baseline uses the SAME plant
    path as the RL policy, just with a constant action."""
    episode_length = int((seg_end - seg_start).total_seconds() / env.unwrapped.step_size)
    env.unwrapped.random_start = False
    env.unwrapped.global_start_time = seg_start
    env.unwrapped.episode_length = episode_length

    obs, _ = env.reset()
    rewards = []
    print(f"Simulating rule-based baseline {seg_start} -> {seg_end} ({episode_length} steps)...")
    for _ in range(episode_length):
        obs, reward, terminated, truncated, _ = env.step(action_norm)
        rewards.append(reward)
        if terminated or truncated:
            break

    kpis = plot_results(env.unwrapped.simulator, rewards, save_plots=save_plots, plots_dir=plots_dir)
    return rewards, kpis


def run_baseline(save_plots: bool = True, plots_dir: str = None,
                  eval_start=None, eval_end=None):
    """Incumbent baseline: hold every indoor-temp setpoint AND the supply-air
    setpoint at COMFORT_SETPOINT_C (21 °C) and run the SAME supervisory cascade
    the RL policy uses, for a like-for-like compare (mean setpoint → outdoor-
    reset curve → ECL310 PID; economizer floor; per-room damper trim, which for
    equal setpoints sits at its 3-ACH baseline).

    eval_start/eval_end default to the shared config window (skoven.yaml) but
    can be overridden for a quick smoke-sized run. DST-safe: splits the
    window at any transition (rl_config.dst_safe_chunks) and aggregates KPIs
    across segments, since a single continuous Twin4Build simulation cannot
    cross a DST transition (see rl_config.py's dst_transition_instants
    docstring for why).
    """
    if plots_dir is None:
        plots_dir = BASELINE_PLOTS_DIR

    if eval_start is None or eval_end is None:
        _, _, eval_start, eval_end = load_rl_windows()

    env = build_env(eval_start, eval_end, eval_mode=True, monitor_filename="baseline_monitor.csv")
    # Conventional incumbent: outdoor-reset hydronic + constant 3-ACH ventilation
    # (no per-room air trim — that is the RL controller's innovation).
    env.unwrapped.simulator.trim_enabled = False

    # Constant incumbent action: every setpoint (4 indoor + supply-air) = 21 °C,
    # mapped per-dimension into the NormalizedActionWrapper's [-1, 1] input using
    # the physical action-space bounds.
    low = env.unwrapped.action_space.low
    high = env.unwrapped.action_space.high
    target = np.full(low.shape, COMFORT_SETPOINT_C, dtype=np.float32)
    action_norm = np.clip(
        2.0 * (target - low) / (high - low) - 1.0, -1.0, 1.0
    ).astype(np.float32)

    chunks = dst_safe_chunks(eval_start, eval_end)
    if len(chunks) > 1:
        print(f"Eval window spans {len(chunks) - 1} DST transition(s) — "
              f"running as {len(chunks)} segments and aggregating KPIs.")

    kpi_list, weights = [], []
    for i, (seg_start, seg_end) in enumerate(chunks):
        seg_dir = plots_dir if len(chunks) == 1 else f"{plots_dir}_seg{i}"
        _, kpis = _run_baseline_segment(env, action_norm, seg_start, seg_end,
                                        seg_dir if save_plots else None, save_plots)
        kpi_list.append(kpis)
        weights.append((seg_end - seg_start).total_seconds())

    agg_kpis = aggregate_kpis(kpi_list, weights)
    print("Baseline KPIs:", agg_kpis)
    return agg_kpis


def compare():
    """Run RL (best_model.zip, deterministic) and the rule-based baseline on
    the identical eval window and print an energy/comfort savings table."""
    print("=== Rule-based baseline (outdoor-reset curve) ===")
    baseline_kpis = run_baseline(save_plots=True)

    print("\n=== RL policy ===")
    _, _, eval_start, eval_end = load_rl_windows()
    env = build_env(eval_start, eval_end, eval_mode=True, monitor_filename="rl_eval_monitor.csv")
    model_path = os.path.join(LOG_DIR, "best_model.zip")
    ppo_model = PPO.load(model_path, env=env, device="cpu")
    print(f"Evaluating model ({ppo_model.num_timesteps} training steps)")
    _, rl_kpis = test_model_chunked(env, ppo_model, plots_dir=PLOTS_DIR)

    print("\n=== RL vs. baseline ===")
    # mode: "pct" -> "% saved" (lower better, energy/viol); "pp" -> percentage
    # points (higher better, % in band); "degC" -> absolute °C delta (lower
    # better, RMSE-to-target — a fraction of a % is meaningless for a temperature).
    rows = [
        ("heating_energy_kWh", "Heating energy (kWh)", "pct"),
        ("ahu_energy_kWh", "AHU energy (kWh)", "pct"),
        ("comfort_degree_hours", "Comfort viol. (degC*h)", "pct"),
        ("comfort_pct_in_band", "Time in comfort band (%)", "pp"),
        ("comfort_rmse_target", "Temp RMSE vs 21degC (degC)", "degC"),
    ]
    print(f"{'metric':28s}{'baseline':>14s}{'RL':>14s}{'change':>16s}")
    for key, label, mode in rows:
        b, r = baseline_kpis[key], rl_kpis[key]
        if mode == "pct":
            change = 100.0 * (b - r) / b if b != 0 else float("nan")
            change_str = f"{change:+.1f}% saved"
        elif mode == "pp":
            change_str = f"{r - b:+.1f} pp"
        else:  # degC — absolute delta, negative = closer to target
            change_str = f"{r - b:+.2f} degC"
        print(f"{label:28s}{b:14.2f}{r:14.2f}{change_str:>16s}")

    heating_ok = rl_kpis["heating_energy_kWh"] <= baseline_kpis["heating_energy_kWh"]
    comfort_ok = rl_kpis["comfort_pct_in_band"] >= baseline_kpis["comfort_pct_in_band"] - 1.0
    if heating_ok and comfort_ok:
        print("\nAcceptance: RL uses <= baseline energy at no worse comfort. PASS")
    else:
        print("\nAcceptance: RL does not clearly dominate the baseline — "
              "report as a documented energy/comfort trade-off.")

    return baseline_kpis, rl_kpis


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true",
                         help="Also evaluate best_model.zip and print the RL-vs-baseline table")
    args = parser.parse_args()

    if args.compare:
        compare()
    else:
        run_baseline()
