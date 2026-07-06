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
from aarhus_model.skoven_model import (
    load_model_and_params, ZONES, AHU_DEFAULTS,
    zone_ventilation_flow, damper_position_for_flow,
)
from aarhus_model.heating_curve import compute_supply_setpoint
from use_case.model_eval import test_model_chunked
from use_case.rl_config import (
    TZ, POLICY_CONFIG_PATH, LOG_DIR, CHECKPOINT_DIR, STEP_SIZE, EPISODE_STEPS,
    COMFORT_MIN_C, COMFORT_MAX_C, load_rl_windows, dst_excluding_periods,
    load_building_config,
)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TRAIN_START, TRAIN_END, EVAL_START, EVAL_END = load_rl_windows()
# Random training episodes must not straddle a DST transition (dateutil's
# tzinfo objects are per-zone singletons, so Twin4Build's own elapsed-time
# arithmetic silently disagrees with pandas' across one — see rl_config.py).
TRAIN_EXCLUDING_PERIODS = dst_excluding_periods(TRAIN_START, TRAIN_END, EPISODE_STEPS, STEP_SIZE)


class SupervisorySetpointGymSimulator(GymSimulator):
    """Supervisory-setpoint plant controller.

    The RL agent emits SETPOINTS, not actuator commands: 4 per-room indoor-temp
    setpoints (`{zone}_indoor_temp_setpoint`) + 1 global supply-air-temp setpoint
    (`supply_air_temp_setpoint_sensor`). This simulator turns those setpoints into
    the low-level signals each step, leaving the ECL310 supply-water PID + mixing
    valve untouched:

      1. HYDRONIC (shared radiators): the shared loop is driven by the MEAN of the
         4 indoor setpoints through the existing outdoor-reset heating curve —
         T_water = compute_supply_setpoint(T_oa, mean(Tset)). Written to
         `ecl310_TSupSet_schedule`, which the PID then tracks (realistic: one
         substation drives all radiators together).
      2. SUPPLY AIR: the RL supply-air setpoint drives the coil/HR, with the
         economizer kept as a SAFETY FLOOR (it can force the setpoint cooler when
         the building overshoots, but never overrides the RL when it doesn't).
      3. PER-ROOM AIR TRIM: each room's VAV damper opens above its 3-ACH baseline
         to push the room toward its own setpoint, but only when the supply air is
         on the helpful side (warm air to a cold room / cool air to a hot room).

    All three are applied via the schedule-output-override pattern (write the
    source schedule's output after it steps; consumers read it the same step). RL
    setpoints are read from `control_inputs` (current action); T_oa and zone temps
    are read from the previous step (1-step delay, standard for supervisory control).
    """

    ECL_SETPOINT_COMPONENT = "ecl310_TSupSet_schedule"
    AIR_SETPOINT_COMPONENT = "supply_air_temp_setpoint_sensor"

    # Economizer safety-floor params (reused from the AHU defaults).
    ECON_TARGET = float(AHU_DEFAULTS["supply_air_setpoint_C"])
    ECON_DEADBAND = float(AHU_DEFAULTS["economizer_deadband_C"])
    ECON_GAIN = float(AHU_DEFAULTS["economizer_gain"])
    ECON_MIN = float(AHU_DEFAULTS["cooling_min_supply_air_C"])
    ECON_ENABLED = bool(AHU_DEFAULTS["economizer_enabled"])

    # Per-room air-trim rule: once a room is more than DAMPER_DEADBAND off its
    # setpoint (and the supply air is on the helpful side), its damper opens
    # `K_DAMPER` per °C of *excess* error above its 3-ACH baseline, up to fully
    # open (=2x design flow). The deadband keeps the dampers at the 3-ACH baseline
    # for the small perpetual offset a room has from its setpoint (so the trim is a
    # targeted correction, not a constant boost — and the fixed-21 baseline stays a
    # ~3-ACH incumbent, not a ventilation-heavy one).
    K_DAMPER = 0.4
    DAMPER_DEADBAND = 0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-room air trim is the RL controller's innovation; the rule-based
        # baseline disables it (dampers stay at their 3-ACH position) so it is a
        # conventional incumbent — outdoor-reset hydronic + constant ventilation.
        self.trim_enabled = True
        cfg = load_building_config()
        hc = cfg.get("heating_curve", {})
        self._hc_s = float(hc.get("s", 1.2))
        self._hc_b = float(hc.get("b", 23.6))
        self._hc_delta = float(hc.get("delta", 0.0))
        self._hc_tmin = float(hc.get("T_min", 20.0))
        self._hc_tmax = float(hc.get("T_max", 80.0))
        # Baseline damper position (the u that reproduces 3-ACH on the oversized
        # damper) per room, and the map from damper-schedule id -> zone.
        ach = float(AHU_DEFAULTS["ventilation_ach"])
        oversize = float(AHU_DEFAULTS["vav_oversize_factor"])
        a = float(AHU_DEFAULTS["damper_a"])
        self._base_pos = {}
        self._damper_pos_components = {}
        for z in ZONES:
            design = zone_ventilation_flow(z, ach)
            self._base_pos[z] = damper_position_for_flow(design, oversize * design, a)
            self._damper_pos_components[f"{z}_damper_position"] = z

    # --- helpers -----------------------------------------------------------
    def _rl_value(self, component_id, signal, default):
        """Current RL action value for a signal (control_inputs holds scalars once
        an action has been applied; a dict/None means not yet set → default)."""
        v = self.control_inputs.get(component_id, {}).get(signal, default)
        if isinstance(v, dict) or v is None:
            return float(default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    def _outdoor_temp(self) -> float:
        v = self.model.components["outdoor_environment"].output["outdoorTemperature"].get()
        return float(v) if v is not None and np.isfinite(float(v)) else 0.0

    def _prev_zone_temp(self, zone) -> float:
        comp = self.model.components.get(f"{zone}_indoor_temp_sensor")
        v = comp.output["measuredValue"].get() if comp is not None else None
        return float(v) if v is not None and np.isfinite(float(v)) else 21.0

    def _indoor_setpoint(self, zone) -> float:
        return self._rl_value(f"{zone}_indoor_temp_setpoint", "scheduleValue", 21.0)

    def _economizer_setpoint(self) -> float:
        """Free-cooling supply-air setpoint (= ECON_TARGET unless the building
        overshoots and outdoor is cooler, in which case a lower, floored value)."""
        temps = [self._prev_zone_temp(z) for z in ZONES]
        T_room = sum(temps) / len(temps)
        T_oa = self._outdoor_temp()
        overshoot = T_room - self.ECON_TARGET
        if np.isfinite(T_oa) and overshoot > self.ECON_DEADBAND and T_oa < T_room:
            setpoint = self.ECON_TARGET - self.ECON_GAIN * (overshoot - self.ECON_DEADBAND)
            return float(min(max(setpoint, self.ECON_MIN), self.ECON_TARGET))
        return self.ECON_TARGET

    def _supply_air_setpoint(self) -> float:
        """RL supply-air setpoint, with the economizer as a cooling-only floor."""
        rl_val = self._rl_value(self.AIR_SETPOINT_COMPONENT, "scheduleValue", self.ECON_TARGET)
        if not self.ECON_ENABLED:
            return rl_val
        econ = self._economizer_setpoint()
        # Only clamp when the economizer is actively demanding cooling (overshoot);
        # otherwise the RL fully owns the setpoint (including calling for heat).
        return min(rl_val, econ) if econ < self.ECON_TARGET else rl_val

    def _supply_water_setpoint(self) -> float:
        T_ref = sum(self._indoor_setpoint(z) for z in ZONES) / len(ZONES)
        return compute_supply_setpoint(
            T_oa=self._outdoor_temp(), T_room_ref=T_ref,
            s=self._hc_s, b=self._hc_b, delta=self._hc_delta,
            T_min=self._hc_tmin, T_max=self._hc_tmax,
        )

    def _damper_position(self, zone) -> float:
        if not self.trim_enabled:
            return self._base_pos[zone]
        T_set = self._indoor_setpoint(zone)
        T_room = self._prev_zone_temp(zone)
        T_air = self._supply_air_setpoint()
        base = self._base_pos[zone]
        err = T_set - T_room                       # >0: room wants to be warmer
        excess = max(0.0, abs(err) - self.DAMPER_DEADBAND)
        helpful = (err > 0 and T_air > T_room) or (err < 0 and T_air < T_room)
        if excess > 0.0 and helpful:               # meaningful deviation + air helps
            pos = base + self.K_DAMPER * excess
        else:                                       # in deadband or air can't help
            pos = base
        return float(min(max(pos, base), 1.0))

    @staticmethod
    def _write_output(component, signal, value, step_index):
        out = component.output[signal]
        fv = float(value)
        out._tensor[:] = fv
        if getattr(out, "_log_history", False):
            out._history[step_index] = fv

    def _do_component_timestep(self, component, second_time, date_time, step_size, step_index):
        super()._do_component_timestep(component, second_time, date_time, step_size, step_index)
        cid = component.id
        if cid == self.ECL_SETPOINT_COMPONENT and "scheduleValue" in component.output:
            self._write_output(component, "scheduleValue", self._supply_water_setpoint(), step_index)
        elif cid == self.AIR_SETPOINT_COMPONENT and "scheduleValue" in component.output:
            self._write_output(component, "scheduleValue", self._supply_air_setpoint(), step_index)
        elif cid in self._damper_pos_components and "scheduleValue" in component.output:
            zone = self._damper_pos_components[cid]
            self._write_output(component, "scheduleValue", self._damper_position(zone), step_index)


class SkovenGymEnv(T4BGymEnv):
    """Custom reward for the Skoven hydronic building model."""

    simulator_class = SupervisorySetpointGymSimulator

    # Reward weights. temp_violation already grows sharply (viol + exp(viol)-1,
    # per zone, two-sided about the [19,26] band), so W_COMFORT just sets how hard
    # comfort dominates energy; LAMBDA_ENERGY weights the (gated) heat+AHU term in
    # kW; REWARD_SCALE keeps the per-step reward O(1) for a well-conditioned value
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

    # Comfort RANGE [min, max] (shared with model_eval.py/baseline_eval.py KPIs
    # via rl_config so the reward and the reported KPIs can't drift apart).
    # Excursions below the min and above the max are both penalised.
    COMFORT_MIN = COMFORT_MIN_C
    COMFORT_MAX = COMFORT_MAX_C

    def get_reward(self, observations, action):
        model = self.simulator.model

        # --- Thermal comfort violations (per room vs the comfort RANGE) ---
        # Two-sided: penalise below COMFORT_MIN AND above COMFORT_MAX. A room can
        # only violate one bound at a time, so `viol` is whichever excursion is
        # non-zero. `exp(viol) - 1` grows sharply on a real excursion but is
        # exactly 0 inside the band, so comfort_ok is reachable and the energy
        # signal survives (the fix that made comfort learnable in v3).
        temp_violation_total = 0.0
        for zone_id in ZONES:
            T_zone = self._val(model.components[f"{zone_id}_indoor_temp_sensor"], "measuredValue")
            viol = max(0.0, self.COMFORT_MIN - T_zone) + max(0.0, T_zone - self.COMFORT_MAX)
            temp_violation_total += viol + (np.exp(viol) - 1.0)

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
