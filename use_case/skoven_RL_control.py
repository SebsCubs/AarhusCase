"""Skoven RL training script.

Trains a PPO agent on the Skoven T4B model to optimise district-heating
consumption and thermal comfort, using measured reward signals.

Reward (direct, per-step cost — NOT a Δ):
    reward = -(W_COMFORT * temp_violation + energy_penalty) / REWARD_SCALE
where energy_penalty = (heat_kW + 0.5*ahu_kW) but ONLY while every room is in
band (comfort floor / lexicographic-style): the agent is credited for saving
energy only once comfort is satisfied, so it can't buy energy savings by
starving heat. Direct (non-Δ) so the episodic return — and hence EvalCallback's
best_model selection — is a monotone measure of policy quality.

Usage:
    python use_case/skoven_RL_control.py           # train
    python use_case/skoven_RL_control.py --eval    # evaluate best model
"""
import argparse
import datetime
import os
import sys
import numpy as np
from dateutil.tz import gettz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from t4b_gym.t4b_gym_env import (
    T4BGymEnv, GymSimulator, NormalizedObservationWrapper, NormalizedActionWrapper,
)
from aarhus_model.skoven_model import load_model_and_params, ZONES, AHU_DEFAULTS
from use_case.model_eval import test_model_chunked
from use_case.rl_config import (
    TZ, POLICY_CONFIG_PATH, LOG_DIR, CHECKPOINT_DIR, STEP_SIZE, EPISODE_STEPS,
    COMFORT_SETPOINT_C, COMFORT_DEADBAND_C, load_rl_windows, dst_excluding_periods,
)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TRAIN_START, TRAIN_END, EVAL_START, EVAL_END = load_rl_windows()
# Random training episodes must not straddle a DST transition (dateutil's
# tzinfo objects are per-zone singletons, so Twin4Build's own elapsed-time
# arithmetic silently disagrees with pandas' across one — see rl_config.py).
TRAIN_EXCLUDING_PERIODS = dst_excluding_periods(TRAIN_START, TRAIN_END, EPISODE_STEPS, STEP_SIZE)


class EconomizerGymSimulator(GymSimulator):
    """Plant-level AHU economizer (free cooling).

    Each timestep, right after the AHU supply-air setpoint schedule steps — and
    before the coil / heat-recovery read it (the schedule is a source, so it
    steps first in the execution order) — override the setpoint with a free-
    cooling value whenever the building runs above setpoint+deadband AND the
    outdoor air is cooler. The AHU then supplies cool outdoor air (heat-recovery
    bypass, since the same setpoint drives the HR target) down to a floor,
    shedding the solar/internal gains the radiators can't remove.

    This is a property of the PLANT, so it applies to both the rule-based
    baseline and the RL rollouts; it is NOT an RL action (the agent still only
    controls the hydronic supply setpoint). Room feedback uses the previous
    step's zone temps (1-step delay), standard for a supervisory controller.
    """

    SETPOINT_COMPONENT = "supply_air_temp_setpoint_sensor"
    ECON_TARGET = float(AHU_DEFAULTS["supply_air_setpoint_C"])
    ECON_DEADBAND = float(AHU_DEFAULTS["economizer_deadband_C"])
    ECON_GAIN = float(AHU_DEFAULTS["economizer_gain"])
    ECON_MIN = float(AHU_DEFAULTS["cooling_min_supply_air_C"])
    ECON_ENABLED = bool(AHU_DEFAULTS["economizer_enabled"])

    def _economizer_setpoint(self) -> float:
        m = self.model
        temps = []
        for z in ZONES:
            comp = m.components.get(f"{z}_indoor_temp_sensor")
            if comp is None:
                continue
            v = comp.output["measuredValue"].get()
            if v is not None and np.isfinite(float(v)):
                temps.append(float(v))
        if not temps:
            return self.ECON_TARGET
        T_room = sum(temps) / len(temps)
        T_oa = float(m.components["outdoor_environment"].output["outdoorTemperature"].get())
        overshoot = T_room - self.ECON_TARGET
        if np.isfinite(T_oa) and overshoot > self.ECON_DEADBAND and T_oa < T_room:
            # Proportional free cooling: drop the supply-air setpoint below 21 in
            # proportion to the mean-room overshoot (past the deadband), floored.
            # Only free while the floor stays above outdoor temp (heat-recovery
            # bypass); the coil trims the rest.
            setpoint = self.ECON_TARGET - self.ECON_GAIN * (overshoot - self.ECON_DEADBAND)
            return float(min(max(setpoint, self.ECON_MIN), self.ECON_TARGET))
        return self.ECON_TARGET

    def _do_component_timestep(self, component, second_time, date_time, step_size, step_index):
        super()._do_component_timestep(component, second_time, date_time, step_size, step_index)
        if (self.ECON_ENABLED and component.id == self.SETPOINT_COMPONENT
                and "scheduleValue" in component.output):
            val = self._economizer_setpoint()
            out = component.output["scheduleValue"]
            out._tensor[:] = val
            if getattr(out, "_log_history", False):
                out._history[step_index] = val


class SkovenGymEnv(T4BGymEnv):
    """Custom reward for the Skoven hydronic building model."""

    simulator_class = EconomizerGymSimulator

    # Reward weights. temp_violation already grows sharply (heating_viol +
    # exp(heating_viol)-1, per zone), so W_COMFORT just sets how hard comfort
    # dominates energy; LAMBDA_ENERGY weights the (gated) heat+AHU term in kW;
    # REWARD_SCALE keeps the per-step reward O(1) for a well-conditioned value
    # function.
    W_COMFORT = 10.0
    LAMBDA_ENERGY = 1.0
    REWARD_SCALE = 10.0

    @staticmethod
    def _val(component, port, default=0.0):
        """Read a component output port as a python float (handles tps.Scalar)."""
        try:
            out = component.output[port]
        except (KeyError, TypeError):
            return default
        getter = getattr(out, "get", None)
        v = getter() if callable(getter) else out
        try:
            return float(v)
        except (TypeError, ValueError):
            try:
                return float(v.detach().cpu().reshape(-1)[-1])
            except Exception:
                return default

    # Comfort band (shared with model_eval.py/baseline_eval.py KPIs via
    # rl_config so the reward and the reported KPIs can't drift apart). A
    # small deadband avoids penalizing sub-tolerance sensor/solver noise.
    COMFORT_SETPOINT = COMFORT_SETPOINT_C
    COMFORT_DEADBAND = COMFORT_DEADBAND_C

    def get_reward(self, observations, action):
        model = self.simulator.model

        # --- Thermal comfort violations (per room vs the comfort setpoint) ---
        temp_violation_total = 0.0
        for zone_id in ZONES:
            T_zone = self._val(model.components[f"{zone_id}_indoor_temp_sensor"], "measuredValue")
            heating_viol = max(0.0, (self.COMFORT_SETPOINT - self.COMFORT_DEADBAND) - T_zone)
            # Sharp, asymmetric comfort penalty that is exactly 0 when the room is
            # comfortable. The earlier `exp(1 + heating_viol)` added a constant e¹
            # per zone even at zero violation (a 4·e¹ = 10.873 floor), which made
            # comfort_ok structurally impossible and swamped the energy terms so the
            # agent just maxed heating. `exp(x) - 1` keeps the sharp growth on a real
            # dip but vanishes at heating_viol = 0, restoring the energy signal.
            temp_violation_total += heating_viol + (np.exp(heating_viol) - 1.0)

        # --- District heating power [kW] ---
        # In calibration mode a measured varme sensor exists; in simulation mode
        # heat is the sum of per-radiator Power (no in-graph sum component).
        if "varme_meter_power_sensor" in model.components:
            heat_kW = self._val(model.components["varme_meter_power_sensor"], "measuredValue")
        else:
            heat_kW = sum(
                self._val(model.components[f"{z}_radiator"], "Power") for z in ZONES
            ) / 1000.0

        # --- AHU power [W] (ventilation; undocumented but present) ---
        ahu_W = 0.0
        if "vent_power_sensor" in model.components:
            ahu_W += self._val(model.components["vent_power_sensor"], "measuredValue")
        if "supply_heating_coil" in model.components:
            ahu_W += self._val(model.components["supply_heating_coil"], "Power")

        # --- Comfort-floor reward (direct per-step cost, NOT a Δ) ---
        # Energy is penalised ONLY when every room is in band, so the agent
        # cannot lower the energy term by starving heat (that opens a comfort
        # violation, whose penalty dominates and whose presence gates the
        # energy credit off entirely). Once comfort is held, the only remaining
        # gradient is to trim heat+AHU toward the comfort boundary.
        comfort_penalty = self.W_COMFORT * temp_violation_total
        energy_kW = heat_kW + 0.5 * ahu_W / 1000.0
        comfort_ok = temp_violation_total <= 0.0
        energy_penalty = self.LAMBDA_ENERGY * energy_kW if comfort_ok else 0.0

        reward = -(comfort_penalty + energy_penalty) / self.REWARD_SCALE

        if np.isnan(reward):
            raise ValueError("Reward is NaN — check model outputs")

        # Stashed for the Monitor's info_keywords / TensorboardCallback (see
        # SkovenGymEnv.step below) — reported un-normalized so training curves
        # read in physical units (kW, °C·h) rather than the scaled reward.
        self._reward_terms = {
            "heat_kW": float(heat_kW),
            "ahu_kW": float(ahu_W / 1000.0),
            "temp_violation": float(temp_violation_total),
            "comfort_ok": float(comfort_ok),
        }
        return reward

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info.update(getattr(self, "_reward_terms", {}))
        return obs, reward, terminated, truncated, info


REWARD_INFO_KEYS = ("heat_kW", "ahu_kW", "temp_violation", "comfort_ok")


def build_env(start, end, eval_mode=False, monitor_filename="monitor.csv", excluding_periods=None):
    model = load_model_and_params()
    env = SkovenGymEnv(
        model=model,
        io_config_file=POLICY_CONFIG_PATH,
        start_time=start,
        end_time=end,
        episode_length=EPISODE_STEPS,
        random_start=not eval_mode,
        excluding_periods=excluding_periods,
        forecast_horizon=50,
        step_size=STEP_SIZE,
        warmup_period=0,
    )
    env = NormalizedObservationWrapper(env)
    env = NormalizedActionWrapper(env)
    env = Monitor(
        env=env,
        filename=os.path.join(LOG_DIR, monitor_filename),
        info_keywords=REWARD_INFO_KEYS,
    )
    return env


class RewardTermsCallback(BaseCallback):
    """Logs the energy/comfort reward-term breakdown to TensorBoard every
    rollout, so training curves show heat_kW / ahu_kW / temp_violation
    separately instead of only the scaled scalar reward."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for key in REWARD_INFO_KEYS:
            values = [info[key] for info in infos if key in info]
            if values:
                self.logger.record_mean(f"rollout/{key}", float(np.mean(values)))
        return True


def linear_schedule(initial_value: float, final_value: float = 0.0):
    """Linear anneal from initial_value (progress_remaining=1) to final_value
    (progress_remaining=0). Lets PPO take large early steps and settle late, so
    the policy std actually contracts instead of staying pinned at its init."""
    def schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule


def train(reload: bool = False, total_timesteps: int = 500_000,
          checkpoint_freq: int = 25_000, eval_freq: int = 5_000):
    env = build_env(TRAIN_START, TRAIN_END, monitor_filename="monitor.csv",
                     excluding_periods=TRAIN_EXCLUDING_PERIODS)
    # Held-out eval env on the disjoint EVAL window (not the training window),
    # so best_model.zip is selected on out-of-sample data.
    eval_env = build_env(EVAL_START, EVAL_END, eval_mode=True, monitor_filename="eval_monitor.csv")

    ppo_model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        gamma=0.99,
        # lr 3e-4 (PPO default) annealed to 0 — the previous 1e-5 was ~30x too
        # low, so the policy std stayed pinned near its init for the whole 500k
        # (under-converged, still wandering between the heat-starve and
        # over-heat extremes). Anneal lets it settle late.
        learning_rate=linear_schedule(3e-4),
        n_steps=720,          # one full 5-day episode per rollout → cleaner GAE
        batch_size=120,       # divides 720 evenly (6 minibatches)
        n_epochs=10,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,         # no entropy bonus — we WANT std to contract
        max_grad_norm=0.5,
        tensorboard_log=LOG_DIR,
        device="cpu",
    )

    callback = CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=LOG_DIR,
            log_path=LOG_DIR,
            eval_freq=eval_freq,
            # eval env is deterministic + fixed-start, so repeated episodes are
            # identical — one is enough. best_model is now selected on the
            # direct (non-Δ) episodic return, a real measure of policy quality.
            n_eval_episodes=1,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=CHECKPOINT_DIR,
            name_prefix="skoven_ppo",
        ),
        RewardTermsCallback(),
    ])

    if reload:
        model_path = os.path.join(LOG_DIR, "best_model.zip")
        ppo_model = PPO.load(model_path, env=env)
        ppo_model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False)
    else:
        ppo_model.learn(total_timesteps=total_timesteps, callback=callback)

    ppo_model.save(os.path.join(LOG_DIR, "ppo_skoven"))
    print(f"Model saved to {LOG_DIR}/ppo_skoven.zip")


def evaluate():
    env = build_env(EVAL_START, EVAL_END, eval_mode=True)
    model_path = os.path.join(LOG_DIR, "best_model.zip")
    ppo_model = PPO.load(model_path, env=env, device="cpu")
    print(f"Evaluating model ({ppo_model.num_timesteps} training steps)")
    _, kpis = test_model_chunked(env, ppo_model)
    print("Eval KPIs:", kpis)


def smoke(n_steps: int = 10):
    """Minimal validation: reset, take n random steps, and a tiny learn() call.
    Full training/eval is run in a separate session. Uses a window inside the
    exported boundary-CSV coverage (2025-01..2025-05) with a deterministic start."""
    smoke_start = datetime.datetime(2025, 2, 1, tzinfo=gettz(TZ))
    smoke_end = datetime.datetime(2025, 5, 1, tzinfo=gettz(TZ))
    env = build_env(smoke_start, smoke_end, eval_mode=True)
    obs, _ = env.reset()
    print(f"reset OK — observation shape {np.asarray(obs).shape}, "
          f"action space {env.action_space.shape}")
    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"  step {i:2d}: reward={float(reward):+.5f} "
              f"terminated={terminated} truncated={truncated}")
        if terminated or truncated:
            obs, _ = env.reset()
    ppo_model = PPO("MlpPolicy", env, n_steps=16, batch_size=8, device="cpu", verbose=0)
    ppo_model.learn(total_timesteps=32)
    print("learn(32) OK — gym + PPO wiring validated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Evaluate the saved best model")
    parser.add_argument("--reload", action="store_true", help="Resume training from saved model")
    parser.add_argument("--smoke", action="store_true", help="10-step gym/PPO smoke test")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    args = parser.parse_args()

    if args.smoke:
        smoke()
    elif args.eval:
        evaluate()
    else:
        train(
            reload=args.reload,
            total_timesteps=args.total_timesteps,
            checkpoint_freq=args.checkpoint_freq,
            eval_freq=args.eval_freq,
        )
