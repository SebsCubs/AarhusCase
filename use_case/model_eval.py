"""Skoven model evaluation: baseline simulation + RL policy rollout + plots.

Ported from T4BGymUseCase/use_case/model_eval.py.

Key differences from the reference:
  - Skoven zone names: core, floor0, floor1 (3-zone fallback)
  - Residential: no cooling setpoints, no AHU cooling coil
  - Eval window: 2026-01-01 → 2026-04-15 (Skoven config)
  - Heat metric: sum of per-radiator Power (varme proxy), fan Power for AHU
"""
import sys
import os
import datetime
from dateutil.tz import gettz
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

POLICY_CONFIG_PATH = os.path.join(SCRIPT_DIR, "policy_input_output.json")
device = "cpu"

EVAL_START = datetime.datetime(
    year=2026, month=1, day=1, hour=0, minute=0, second=0,
    tzinfo=gettz("Europe/Copenhagen"),
)
EVAL_END = datetime.datetime(
    year=2026, month=4, day=15, hour=0, minute=0, second=0,
    tzinfo=gettz("Europe/Copenhagen"),
)


def get_baseline(model, step_size: int = 600):
    """Forward-simulate the model with default schedules across the eval window."""
    simulator = tb.Simulator(model)
    simulator.simulate(
        start_time=EVAL_START, end_time=EVAL_END, step_size=step_size
    )
    plot_results(simulator, save_plots=True)


def test_model(env, model, episode_days: int = 15):
    """Roll a trained PPO policy on the env starting at EVAL_START."""
    stepSize = 600  # seconds
    episode_length = int(3600 * 24 * episode_days / stepSize)
    warmup_period = 0

    if isinstance(env, Wrapper):
        env.unwrapped.random_start = False
        env.unwrapped.global_start_time = EVAL_START
        env.unwrapped.episode_length = episode_length
        env.unwrapped.warmup_period = warmup_period
    else:
        env.random_start = False
        env.global_start_time = EVAL_START
        env.episode_length = episode_length
        env.warmup_period = warmup_period

    obs, _ = env.reset()
    done = False
    observations = [obs]
    rewards = []
    print("Simulating...")

    pbar = tqdm(total=episode_length, desc="Simulation Progress")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs)
        rewards.append(reward)
        done = terminated or truncated
        pbar.update(1)
    pbar.close()

    plot_results(env.unwrapped.simulator, rewards, save_plots=True)
    return observations, rewards


def _hist(simulator, component_id, port, which="output"):
    """Return a 1-D numpy array for a component port's simulated history.

    Dev-branch components expose output[port].history() -> (n_t, n_s, n_c, n_v);
    we take simulation 0 and the first component/vector slot.
    """
    comp = simulator.model.components[component_id]
    store = comp.output if which == "output" else comp.input
    h = store[port].history(i_s=0, i_c=0, i_v=0)
    return h.detach().cpu().numpy().reshape(-1)


def _series(simulator, component_id, output_key, tz="Europe/Copenhagen", which="output"):
    data = _hist(simulator, component_id, output_key, which=which)
    idx = pd.DatetimeIndex(simulator.date_time_steps)
    s = pd.Series(data=data, index=idx[: len(data)])
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    s.index = s.index.tz_convert(tz)
    return s


def plot_results(simulator: tb.Simulator, rewards=None, plotting_stepSize=600, save_plots=False):
    os.makedirs("plots", exist_ok=True)
    sim_times = pd.DatetimeIndex(simulator.date_time_steps)

    # Per-room temperature vs the fixed comfort setpoint (no per-room control).
    COMFORT_SETPOINT = 21.0
    for zone_id in ZONES:
        temp = _series(simulator, f"{zone_id}_indoor_temp_sensor", "measuredValue")
        temp = temp.resample(pd.Timedelta(seconds=plotting_stepSize)).mean()

        plt.figure(figsize=(12, 6))
        plt.plot(temp.index, temp.values, label="Indoor Temperature", linewidth=2)
        plt.axhline(COMFORT_SETPOINT, linestyle="--", color="grey", label="Comfort setpoint")
        plt.title(f"{zone_id} — Temperature")
        plt.xlabel("Time")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"plots/{zone_id}_temperature_setpoint.png")

    # Comfort violation penalty vs the fixed comfort setpoint (heating only).
    temp_violation_penalty = 0.0
    for zone_id in ZONES:
        T = _hist(simulator, f"{zone_id}_indoor_temp_sensor", "measuredValue")
        viol = np.sum(np.maximum(0, COMFORT_SETPOINT - T))
        print(f"{zone_id} heating violation (°C·step sum): {viol:.1f}")
        temp_violation_penalty += viol
    print(f"Total heating violation: {temp_violation_penalty:.1f}")

    # Hydronic energy — sum of per-radiator Power
    total_radiator_power = None
    for zone_id in ZONES:
        P = _hist(simulator, f"{zone_id}_radiator", "Power")
        total_radiator_power = P if total_radiator_power is None else total_radiator_power + P
    print(f"Mean total radiator Power: {np.mean(total_radiator_power):.1f} W")

    plt.figure(figsize=(12, 6))
    plt.plot(sim_times, total_radiator_power, label="Total radiator power (W)", linewidth=2)
    plt.title("Hydronic heat output (sum of radiators ≈ varme meter)")
    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_plots:
        plt.savefig("plots/total_radiator_power.png")

    # AHU fan power
    try:
        fan_power = _hist(simulator, "vent_power_sensor", "measuredValue")
        print(f"Mean fan power: {np.mean(fan_power):.1f} W")
        plt.figure(figsize=(12, 6))
        plt.plot(sim_times, fan_power, label="Fan Power", linewidth=2)
        plt.title("AHU Fan Power Consumption")
        plt.xlabel("Time")
        plt.ylabel("Power (W)")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_plots:
            plt.savefig("plots/ahu_fan_power.png")
    except KeyError:
        print("Info: AHU fan power sensor not present in this model variant.")

    # Action plots from policy schema
    if os.path.exists(POLICY_CONFIG_PATH):
        with open(POLICY_CONFIG_PATH, "r") as f:
            policy_config = json.load(f)

        component_ids, signal_keys = [], []
        for component_id, actions in policy_config.get("actions", {}).items():
            for _action_name, action_config in actions.items():
                component_ids.append(component_id)
                signal_keys.append(action_config["signal_key"])

        for component_id, signal_key in zip(component_ids, signal_keys):
            try:
                action = _hist(simulator, component_id, signal_key, which="input")
            except (KeyError, AttributeError):
                continue
            plt.figure(figsize=(12, 6))
            plt.plot(sim_times, action, label=f"{component_id} — {signal_key}", linewidth=2)
            plt.title(f"Action: {component_id} — {signal_key}")
            plt.xlabel("Time")
            plt.ylabel("Action Value")
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
            plt.tight_layout()
            if save_plots:
                plt.savefig(f"plots/action_{component_id}_{signal_key}.png")

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
            plt.savefig("plots/rewards.png")
        print(f"Mean reward: {rewards_arr.mean():.4f}")


if __name__ == "__main__":
    model = load_model_and_params()
    get_baseline(model)
