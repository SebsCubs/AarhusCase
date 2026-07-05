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
from use_case.model_eval import plot_results, aggregate_kpis, test_model_chunked
from use_case.rl_config import (
    LOG_DIR, PLOTS_DIR, COMFORT_SETPOINT_C, load_building_config, load_rl_windows,
    dst_safe_chunks,
)
from use_case.skoven_RL_control import build_env

BASELINE_PLOTS_DIR = PLOTS_DIR + "_baseline"


def _outdoor_temp_C(env) -> float:
    """Current outdoor air temperature, read directly off the model (same
    pattern SkovenGymEnv.get_reward uses for live state reads)."""
    model = env.unwrapped.simulator.model
    out = model.components["outdoor_environment"].output["outdoorTemperature"]
    v = out.get() if hasattr(out, "get") else out
    return float(v)


def _run_curve_segment(env, hc, low, high, seg_start, seg_end, plots_dir, save_plots):
    """Drive the outdoor-reset curve over one DST-safe segment on the
    (already-built, reused) env."""
    episode_length = int((seg_end - seg_start).total_seconds() / env.unwrapped.step_size)
    env.unwrapped.random_start = False
    env.unwrapped.global_start_time = seg_start
    env.unwrapped.episode_length = episode_length

    obs, _ = env.reset()
    rewards = []
    print(f"Simulating rule-based baseline {seg_start} -> {seg_end} ({episode_length} steps)...")
    for _ in range(episode_length):
        T_oa = _outdoor_temp_C(env)
        T_set = compute_supply_setpoint(
            T_oa=T_oa, T_room_ref=COMFORT_SETPOINT_C,
            s=hc.get("s", 1.5), b=hc.get("b", 35.0), delta=hc.get("delta", 0.0),
            T_min=hc.get("T_min", 20.0), T_max=hc.get("T_max", 80.0),
        )
        T_set = float(np.clip(T_set, low, high))
        action_norm = np.array([2.0 * (T_set - low) / (high - low) - 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, _ = env.step(action_norm)
        rewards.append(reward)
        if terminated or truncated:
            break

    kpis = plot_results(env.unwrapped.simulator, rewards, save_plots=save_plots, plots_dir=plots_dir)
    return rewards, kpis


def run_baseline(save_plots: bool = True, plots_dir: str = None,
                  eval_start=None, eval_end=None):
    """Drive the rule-based outdoor-reset curve over the eval window and
    report the same KPIs/plots as the RL rollout, for a like-for-like compare.

    eval_start/eval_end default to the shared config window (skoven.yaml) but
    can be overridden for a quick smoke-sized run. DST-safe: splits the
    window at any transition (rl_config.dst_safe_chunks) and aggregates KPIs
    across segments, since a single continuous Twin4Build simulation cannot
    cross a DST transition (see rl_config.py's dst_transition_instants
    docstring for why).
    """
    if plots_dir is None:
        plots_dir = BASELINE_PLOTS_DIR

    hc = load_building_config().get("heating_curve", {})
    if eval_start is None or eval_end is None:
        _, _, eval_start, eval_end = load_rl_windows()

    env = build_env(eval_start, eval_end, eval_mode=True, monitor_filename="baseline_monitor.csv")

    # Action space bounds (physical °C) for normalizing the curve's setpoint
    # into the NormalizedActionWrapper's [-1, 1] input.
    low = float(env.unwrapped.action_space.low[0])
    high = float(env.unwrapped.action_space.high[0])

    chunks = dst_safe_chunks(eval_start, eval_end)
    if len(chunks) > 1:
        print(f"Eval window spans {len(chunks) - 1} DST transition(s) — "
              f"running as {len(chunks)} segments and aggregating KPIs.")

    kpi_list, weights = [], []
    for i, (seg_start, seg_end) in enumerate(chunks):
        seg_dir = plots_dir if len(chunks) == 1 else f"{plots_dir}_seg{i}"
        _, kpis = _run_curve_segment(env, hc, low, high, seg_start, seg_end,
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
    rows = [
        ("heating_energy_kWh", "Heating energy (kWh)", True),
        ("ahu_energy_kWh", "AHU energy (kWh)", True),
        ("comfort_degree_hours", "Comfort viol. (degC*h)", True),
        ("comfort_pct_in_band", "Time in comfort band (%)", False),
    ]
    print(f"{'metric':28s}{'baseline':>14s}{'RL':>14s}{'change':>12s}")
    for key, label, lower_is_better in rows:
        b, r = baseline_kpis[key], rl_kpis[key]
        if lower_is_better:
            change = 100.0 * (b - r) / b if b != 0 else float("nan")
            change_str = f"{change:+.1f}% saved"
        else:
            change = r - b
            change_str = f"{change:+.1f} pp"
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
