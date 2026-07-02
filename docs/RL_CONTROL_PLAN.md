# Skoven RL Control — Stage 2 Implementation Plan

## Context

Stage 1 (modeling/calibration) of the Aarhus Skoven digital twin is complete: a
4-room ring envelope + closed-loop ECL310 hydronic substation + activated 3-ACH
MVHR air loop, calibrated against real BMS/ReMoni data (`CALIBRATION_RUNBOOK.md`,
zone CV-RMSE ~2.4–5.8 %). The twin runs in two modes via a `calibration_mode`
flag in `aarhus_model/skoven_model.py`: calibration (boundary signals replayed
from CSV) and **simulation** (`calibration_mode=False` — the RL-controllable
actuators drive the loop, supply/return water are produced outputs).

Stage 2 is the control layer: wrap the sim-mode twin in a Gym environment and
train an RL agent that minimizes district-heating (+ ventilation) energy while
holding indoor comfort within setpoint bounds, with **checkpoints,
observability, and a validated evaluation** against a rule-based baseline.

Much of the scaffolding already exists but is incomplete or broken. This plan
finishes it into a runnable, observable, validated training + evaluation module,
modeled on the reference `T4BGymUseCase` (BOPTEST multizone building).

## Current state (what exists vs. what's missing)

**Exists and works:**
- `t4b_gym/t4b_gym_env.py` — full Gym env already rebuilt for the Twin4Build
  **dev-branch** API (`GymSimulator(tb.Simulator)` uses `_assign_component_inputs`
  + `component.do_step(...)` + post-step output overrides; reconstructs
  `dateTimeSteps`). Passes the smoke test. Provides `T4BGymEnv`,
  `NormalizedObservationWrapper`, `NormalizedActionWrapper`.
- `use_case/policy_input_output.json` — action/observation/forecast config
  (6 actions: `ecl310_TSupSet_schedule`, 4 `{room}_supply_damper`,
  `supply_air_temp_setpoint_sensor`; obs: outdoor T, supply/return water, vent
  power, 4 zone temps; time embeddings; outdoor-T + irradiation forecasts).
- `aarhus_model/skoven_model.py::load_model_and_params(calibration_mode=False)` —
  builds the sim-mode twin and applies the envelope + hydronic pickles. RL-ready.
- `use_case/skoven_RL_control.py` — `SkovenGymEnv` (custom reward), `train()`,
  `evaluate()`, `smoke()`. Smoke passes.

**Broken / missing (the work):**
1. `skoven_RL_control.py` imports `from use_case.model_eval import test_model`,
   but **`use_case/model_eval.py` does not exist** in AarhusCase → `--eval`
   crashes. (The reference has one, but it uses the *old* `savedOutput` /
   `dateTimeSteps` / `savedInput` API that is gone on the dev branch.)
2. **No checkpointing** — only `EvalCallback` saves `best_model.zip`. User wants
   periodic checkpoints.
3. **Thin observability** — only TensorBoard scalars + `monitor.csv`. No
   breakdown of reward terms (energy vs. comfort), no per-episode energy/comfort
   KPIs.
4. **Data-window bug** — sim mode's only external data need is weather
   (`outdoor_temperature.csv`, `global_irradiation.csv`), which covers
   **2024-12-31 → 2026-02-27** only. But `TRAIN_START=2024-12-01` (before) and
   `EVAL_END=2026-04-15` (after) fall **outside** coverage → out-of-range reads.
5. **No baseline / no validation** — nothing quantifies RL savings vs. the real
   building's control, so "validate the RL results" is unaddressed.
6. Minor: `SkovenGymEnv.get_reward(self, action, observation)` arg order is
   swapped vs. the base call `get_reward(observations, action)` (harmless today —
   the reward reads model state directly — but should be aligned).

## Design decisions (defaults chosen; confirm before heavy training)

- **Action scope:** start with the **supply-water-temp setpoint only**
  (`ecl310_TSupSet_schedule`) — the one calibration-identified hydronic lever —
  but keep it **config-driven** via `policy_input_output.json` so expanding to the
  AHU setpoint / dampers is a config edit, not a code change. Dampers/AHU stay at
  the 3-ACH baseline for the first validated result (runbook: they are activated
  by construction, not identifiable from zone temps).
- **Baseline:** rule-based **outdoor-reset heating curve** (replay the real BMS
  `ecl310_TSupSet_measured` / heating-curve supply setpoint) + fixed 3-ACH
  ventilation = the incumbent ECL310 strategy. RL savings are reported against it.
- **Data windows:** **re-export weather across a wide range** (open-meteo, e.g.
  2023-01 → 2026-06) and use a **disjoint train/eval holdout** split.

These are reversible; re-confirm if a different action scope/baseline is wanted
before committing to a long training run.

## Implementation

### 1. Fix the weather data coverage (`data_ingest`)
- Re-run the open-meteo export so `outdoor_temperature.csv` +
  `global_irradiation.csv` span the full intended train/eval range. Reuse the
  existing exporter: `python -m data_ingest.export_t4b_csvs --building skoven
  --start 2023-01-01 --end 2026-06-01` (weather comes from
  `data_ingest/weather_openmeteo.py`). Verify UTC CSVs (the T4B spreadsheet
  loader needs UTC — mixed DST offsets fail otherwise).
- Confirm sim mode needs **only** these two boundary CSVs (all other sim-mode
  sensor `filename`s are `None` — supply/return water etc. are produced). The
  weather span defines the usable RL window.

### 2. Finalize reward + action scope (`use_case/skoven_RL_control.py`)
- Keep the existing `SkovenGymEnv.get_reward` structure (heating-only comfort
  penalty vs. `COMFORT_SETPOINT=21`, district-heat `heat_kW`, AHU `ahu_W`,
  telescoping `-(objective - previous_objective)` convention matching the
  reference). It already correctly falls back to summing per-room
  `{z}_radiator` "Power" in sim mode (verified: `varme_meter_power_sensor` is
  calibration-only; `vent_power_sensor` + `supply_heating_coil` exist in sim mode).
- **Reduce the active action set** to `ecl310_TSupSet_schedule` by trimming the
  `actions` block of `policy_input_output.json` (keep the others documented but
  commented/removed) — the env builds its action space from this file
  (`create_action_space`).
- Fix the `get_reward` signature to `(self, observations, action)` to match the
  base-class call site (`t4b_gym_env.py:514`); stash the reward-term breakdown on
  `self._reward_terms` for observability (see §4).
- Add a **comfort deadband** option (heating setpoint 21 with a small tolerance)
  so the agent isn't penalized for sub-0.1 °C noise — keep default behavior but
  parameterize.

### 3. Training module: checkpoints + eval callback (`use_case/skoven_RL_control.py`)
- Add `CheckpointCallback` (from `stable_baselines3.common.callbacks`) writing
  `logs/checkpoints/skoven_ppo_<steps>.zip` every N steps, alongside the existing
  `EvalCallback` (best-model + `evaluations.npz`). Combine via `CallbackList`.
- Build a **separate eval env** on the held-out window (not the training env) and
  pass it to `EvalCallback` so `best_model` is selected on out-of-sample data.
  Use deterministic reset for eval (`random_start=False`).
- Set `TRAIN_START/END` and `EVAL_START/END` to **disjoint** ranges inside the
  re-exported weather coverage (e.g. train 2024-12 → 2025-11, eval 2025-12 →
  2026-04). Keep `STEP_SIZE=600`, 5-day episodes.
- Keep PPO/`MlpPolicy` hyperparameters from the reference
  (`multizone_simple_air_RL_control.py`: `lr=1e-5, n_steps=200, batch_size=50,
  n_epochs=10, clip_range=0.2, max_grad_norm=0.5, gamma=0.99`). Preserve
  `--reload` to resume with `reset_num_timesteps=False`.

### 4. Observability (`use_case/skoven_RL_control.py`)
- Override `SkovenGymEnv.step` to attach the reward-term breakdown to the `info`
  dict (`info["heat_kW"]`, `info["ahu_kW"]`, `info["temp_violation"]`,
  `info["comfort_ok"]`). `Monitor` propagates `info_keywords` into `monitor.csv`.
- Add a small `TensorboardCallback(BaseCallback)` that reads the latest `info`
  from `self.locals` and logs the reward terms as scalars
  (`rollout/heat_kW`, `rollout/temp_violation`, …) so training curves show
  energy vs. comfort separately — not just total reward.
- Keep `tensorboard_log=LOG_DIR`; document `tensorboard --logdir use_case/logs`.

### 5. Evaluation module — rewrite for the dev-branch API (`use_case/model_eval.py`, NEW)
Port the reference `T4BGymUseCase/use_case/model_eval.py` to AarhusCase, but
**replace the removed API**:
- `component.savedOutput[port]` / `savedInput` → `component.output[port].history(i_s=0, i_c=0)`
  (dev-branch; Scalars have no `i_v`). Timestamps come from the gym simulator's
  reconstructed `env.unwrapped.simulator.dateTimeSteps` (`t4b_gym_env.py:82`).
- `test_model(env, model)`: set `random_start=False`, fixed `global_start_time`,
  full-window `episode_length`; roll the deterministic policy
  (`model.predict(obs, deterministic=True)`) to episode end; collect zone temps,
  supply/return water, radiator `Power`, AHU coil/fan power, and the applied
  actions (read component ids/signal_keys from `policy_input_output.json`).
- Produce, in `use_case/plots/`: per-room temperature-vs-comfort-band overlays,
  total heat-power timeseries, AHU power, and action traces (reuse the reference's
  plotting layout, adapted to the 4 ring rooms `room_a..d`).
- Compute + print **KPIs**: total heating energy (kWh = Σ Power·Δt), AHU energy,
  comfort violation **degree-hours** and **% timesteps within band**, mean/peak
  supply temp.

### 6. Baseline + validation report (`use_case/baseline_eval.py`, NEW)
- `run_baseline()`: build the sim-mode model but drive `ecl310_TSupSet_schedule`
  from the **rule-based outdoor-reset curve** (reuse `aarhus_model/heating_curve.py`
  and/or replay `ecl310_TSupSet_measured.csv`) over the **eval window**, dampers at
  the 3-ACH baseline. Run through the same `GymSimulator` (no policy) so KPIs are
  computed identically.
- `compare()`: run RL (`best_model.zip`, deterministic) and the baseline on the
  identical eval window; emit a comparison table — **energy (kWh), % energy
  saving, comfort degree-hours, % time in band** for RL vs. baseline — and overlay
  plots (RL vs. baseline zone temps + supply setpoint + cumulative energy).
- Acceptance target: RL achieves **≤ baseline energy at no worse comfort**, or a
  documented energy/comfort trade-off (mirrors the runbook's acceptance style).

### Critical files
- Modify: `use_case/skoven_RL_control.py`, `use_case/policy_input_output.json`.
- New: `use_case/model_eval.py`, `use_case/baseline_eval.py`.
- Re-export (data): `aarhus_model/generated_files/data/skoven/outdoor_temperature.csv`,
  `.../global_irradiation.csv` via `data_ingest.export_t4b_csvs`.
- Reuse (no change): `t4b_gym/t4b_gym_env.py`, `aarhus_model/skoven_model.py`
  (`load_model_and_params`, `ZONES`, `heating_curve.py`).

## Verification

Run everything from `/home/sebscubs/repos/AarhusCase/AarhusCase` with `uv run`.
1. **Data**: after re-export, confirm both weather CSVs' first/last timestamps
   cover the new train + eval windows (UTC).
2. **Smoke** (fast, no training): `uv run python use_case/skoven_RL_control.py --smoke`
   — reset + 10 random steps + `learn(32)` must pass with the trimmed action space.
3. **Short training run**: temporarily set `total_timesteps` low (e.g. 20k) and
   run `train()`; confirm `logs/checkpoints/*.zip`, `best_model.zip`,
   `evaluations.npz`, `monitor.csv` appear and TensorBoard shows the reward-term
   scalars (energy vs. comfort) trending.
4. **Eval**: `uv run python use_case/skoven_RL_control.py --eval` — must load
   `best_model.zip`, roll the held-out window, print KPIs, and write plots to
   `use_case/plots/` (validates the new `model_eval.py` against the dev-branch API).
5. **Baseline comparison**: `uv run python use_case/baseline_eval.py` — prints the
   RL-vs-rule-based savings table and writes overlay plots; sanity-check that
   baseline zone temps sit near 21 °C and RL energy ≤ baseline at comparable comfort.
6. Full 500k-step training is deferred to a dedicated session (long-running);
   the above validates the wiring end-to-end first.
