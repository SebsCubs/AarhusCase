# RL Control Plan V2 — Supervisory-setpoint control with a two-sided comfort band

> Supersedes `docs/RL_CONTROL_PLAN.md`. The v1 plan (direct supply-water setpoint,
> then per-room VAV dampers) could not beat the rule-based baseline: a single fixed
> 21 °C target made the coldest room (room_a) the binding constraint, unreachable by
> a shared hydronic loop, and per-room ventilation proved too weak (see the RL
> Stage-2 memory / prior runs). V2 changes the control interface to supervisory
> *setpoints* over the existing PID and relaxes comfort to a range. **Status:
> implemented** — see the file list below; validated by smoke + cascade-sanity +
> baseline before the first full training run.

## Context

The Skoven RL controller currently commands a **low-level** signal: it overrides the
hydronic supply-water setpoint (`ecl310_TSupSet_schedule`) directly, plus (in the
latest VAV experiments) per-room damper positions. Every variant has been unable to
beat the rule-based baseline: with a single **fixed 21 °C comfort target**, the
binding constraint is always the coldest room (room_a), which a shared supply-water
loop can't hold without overheating others, and per-room ventilation proved too weak.

The user wants to change both the **control strategy** and the **reward**:

1. **Supervisory control.** The RL should emit *setpoints*, not actuator commands,
   and let the existing low-level PID run the hydronic mixing valve:
   - **4 per-room indoor-temperature setpoints.** The shared radiator loop (one
     ECL310 substation — realistically it can only drive all radiators together) is
     driven by the **mean** of the 4 setpoints, fed through the existing outdoor-reset
     **heating curve** → supply-water setpoint → the untouched ECL310 PID/valve.
   - **1 global supply-air-temperature setpoint** for the AHU coil/heat-recovery.
   - **Per-room air-loop trim.** Because the air loop *can* be independent per room,
     the per-room VAV dampers are used to nudge each room toward its own setpoint —
     driven by a **rule** from each room's (setpoint − measured) error, not as direct
     RL actions.
2. **Comfort as a range, not a point.** Replace the fixed 21 °C ± 0.2 target with a
   band **[19 °C, 26 °C]**. Penalize excursions below 19 **and** above 26 (the reward
   becomes two-sided; overheating is now penalized). Keep the exponential violation
   shape. Reshape the RL training around this.

Intended outcome: the agent minimizes heating + ventilation energy while keeping all
rooms inside a realistic comfort band, using a control interface that mirrors a real
BMS (zone setpoints + AHU setpoint over an existing PID), which is both more realistic
and gives the agent a cleaner energy-optimization problem (the wide band largely
dissolves the room_a bottleneck).

## Decisions (confirmed with user)

- RL action vector (5-D): `[Tset_a, Tset_b, Tset_c, Tset_d, T_supply_air]`.
- Hydronic reference = **mean** of the 4 indoor setpoints → existing heating curve.
- Indoor→supply-water via the existing `heating_curve.compute_supply_setpoint`
  (no new PID; the ECL310 supply-water PID/valve is untouched).
- Per-room dampers = **rule-driven** from setpoint error (RL emits only setpoints).
- Economizer **kept as a safety floor** on the supply-air setpoint.
- VAV dampers are no longer RL actions; the direct supply-water action is removed.

## Current architecture (reference)

- `aarhus_model/skoven_model.py:320` `hydronic_fcn` — sim-mode branch (`:382`) builds
  `ecl310_TSupSet_schedule` (ScheduleSystem, supply-water setpoint) → `ecl310_pid`
  (`:396`) → `ecl310_kr1_valve` → mixing shunt → radiators. This chain stays; only the
  *source* of the setpoint changes.
- `aarhus_model/heating_curve.py` — `compute_supply_setpoint(T_oa, T_room_ref, s, b,
  delta, T_min, T_max)`. Reuse verbatim.
- `aarhus_model/skoven_model.py:526` `supply_air_temp_setpoint_sensor` (ScheduleSystem)
  → coil `outletAirTemperatureSetpoint` + heat-recovery `primaryTemperatureOutSetpoint`.
- Per-room dampers: `{zone}_supply_damper` (oversized 2× from the VAV work) fed by
  `{zone}_damper_position` ScheduleSystems (baseline = 3-ACH position). Keep both.
- `use_case/skoven_RL_control.py` — `EconomizerGymSimulator` (`:55`) overrides the
  supply-air setpoint via the `_do_component_timestep` output-override pattern;
  `SkovenGymEnv.get_reward` (`:151`) the reward.
- Gym override mechanics (`t4b_gym/t4b_gym_env.py:93`): source-schedule outputs are
  overridden after `do_step` (step 4) and propagate to consumers the same step via the
  consumer's input assignment (step 1). RL action values live in `self.control_inputs`.
- `use_case/model_eval.py:225` comfort KPI (one-sided, below 20.8). `use_case/
  baseline_eval.py` drives the curve live and steps the env. `use_case/rl_config.py:30`
  comfort constants.

## Implementation

### 1. Comfort constants — `use_case/rl_config.py`
Add `COMFORT_MIN_C = 19.0`, `COMFORT_MAX_C = 26.0` (keep `COMFORT_SETPOINT_C = 21.0`
as the baseline target / plot reference). Optionally source from `skoven.yaml` a new
`comfort: {min: 19, max: 26}` block (mirroring how `heating_curve:` is loaded).

### 2. Model — `aarhus_model/skoven_model.py` (`hydronic_fcn`, sim-mode branch)
- Add 4 source schedules `{zone}_indoor_temp_setpoint` (ScheduleSystem, default 21 °C)
  — the RL indoor-setpoint levers. Connect each to a tiny read-back `SensorSystem`
  (`{zone}_indoor_temp_setpoint_readback`) so the component is in the connected graph
  (avoids any unconnected-source stepping issue) and is available as an observation.
- Leave `ecl310_TSupSet_schedule` in place (now written by the simulator, not the RL).
- No change to `ecl310_pid`/valve/shunt, the coil/HR, or the oversized dampers +
  `{zone}_damper_position` schedules (all reused). `ahu_fcn` unchanged.

### 3. New GymSimulator — `use_case/skoven_RL_control.py`
Replace `EconomizerGymSimulator` with `SupervisorySetpointGymSimulator(GymSimulator)`.
Load `heating_curve` params (`s,b,delta,T_min,T_max`) and comfort band via
`load_building_config()` at init (as `baseline_eval` does). Each step, read RL setpoints
from `self.control_inputs`, and previous-step `T_oa`/zone temps from component outputs
(1-step delay, standard for a supervisory loop — same as the current economizer).

Compute three derived families of **output overrides** in `_do_component_timestep`
(after `super()`, mirroring the economizer's tensor-write at `:104-125`):

- **Supply-water setpoint** — when `component.id == "ecl310_TSupSet_schedule"`:
  `T_ref = mean(Tset_a..d)`;
  `T_water = compute_supply_setpoint(T_oa, T_ref, s, b, delta, T_min, T_max)`;
  write to its `scheduleValue`. (PID then tracks it, valve unchanged.)
- **Supply-air setpoint (RL + economizer floor)** — when `component.id ==
  "supply_air_temp_setpoint_sensor"`: take the RL value and apply the existing
  economizer as a *floor*: `final = min(RL_supply_air, econ_setpoint)` when
  `mean_room > 21 + deadband and T_oa < mean_room`, else `RL_supply_air`; write it.
  (Reuse the current `_economizer_setpoint` logic for `econ_setpoint`.)
- **Per-room damper trim** — when `component.id == "{zone}_damper_position"`: apply the
  rule (below); write to its `scheduleValue`. The connection carries it to the damper.

**Damper trim rule** (per room; oversized damper, positions in `[min_pos, 1.0]`,
`base_pos` = the 3-ACH baseline position already computed in the model):
```
err  = Tset_room - T_room                 # >0: room wants warmer
need = sign(err)
help = sign(T_supply_air_final - T_room)  # +1 warm air, -1 cool air
if need != 0 and need == help:            # air can move room toward setpoint
    pos = clip(base_pos + K_DAMPER * abs(err), base_pos, 1.0)   # open to trim
else:                                      # air can't help (would push wrong way)
    pos = MIN_POS                          # throttle to min ventilation
```
`K_DAMPER` and `MIN_POS` are tunable constants on the simulator (start `K_DAMPER≈0.4`
per °C, `MIN_POS≈0.1`). This makes the air loop a per-room heater/cooler slaved to the
RL setpoints and the shared supply-air temperature.

### 4. Reward — `use_case/skoven_RL_control.py` `get_reward`
Reshape the per-zone violation to two-sided, keep the exponential and the direct
comfort-floor structure (`:154-195`):
```
viol = max(0, COMFORT_MIN - T_zone) + max(0, T_zone - COMFORT_MAX)   # 0 inside band
temp_violation_total += viol + (exp(viol) - 1.0)
# unchanged below:
comfort_penalty = W_COMFORT * temp_violation_total
energy_kW       = heat_kW + 0.5 * ahu_W/1000
comfort_ok      = temp_violation_total <= 0            # every room within [19,26]
energy_penalty  = LAMBDA_ENERGY * energy_kW if comfort_ok else 0.0
reward          = -(comfort_penalty + energy_penalty) / REWARD_SCALE
```
Keep `W_COMFORT=10, LAMBDA_ENERGY=1, REWARD_SCALE=10` (note: retune if the wide band
makes comfort too cheap). `comfort_ok` in `_reward_terms` now means "in [19,26]".

### 5. Action/observation config — `use_case/policy_input_output.json`
- `actions` (5): `{zone}_indoor_temp_setpoint.scheduleValue` ×4 (range e.g. **[18, 26]**)
  + `supply_air_temp_setpoint_sensor.scheduleValue` (range e.g. **[14, 30]**). Remove
  `ecl310_TSupSet_schedule` and the 4 `*_supply_damper.damperPosition` actions.
- `observations`: keep zone temps (4), outdoor temp, supply/return water, AHU fan/temp
  sensors; keep the 4 damper airflows; optionally add the 4 indoor-setpoint read-backs.
  Update the `_actions_note`.

### 6. Baseline — `use_case/baseline_eval.py`
The baseline now runs through the **same** supervisory cascade for an apples-to-apples
comparison: emit a **constant** action `[21,21,21,21, 21]` (normalized) each step
(4 indoor setpoints = 21, supply-air = 21). The simulator does mean→curve→PID, the
damper rule (err≈0 → baseline 3-ACH), and the economizer floor. This simplifies
`_run_curve_segment` (no more live `compute_supply_setpoint`/`room_comp_k` here — the
simulator owns the curve). Note the baseline numbers will be recomputed.

### 7. Comfort KPI — `use_case/model_eval.py:225-238`
Make two-sided: `viol = max(0,19-T)+max(0,T-26)`; degree-hours = Σ viol·dt;
`in_band = mean((T>=19)&(T<=26))·100`. Report per-room and total. Used by both
`model_eval` and `baseline_eval` so the reward and KPIs stay consistent.

## Parameters (defaults, adjustable)
- Comfort band: [19, 26] °C. Indoor-setpoint action range: [18, 26] °C. Supply-air
  action range: [14, 30] °C. `K_DAMPER≈0.4`, `MIN_POS≈0.1`. Heating-curve params
  unchanged (`skoven.yaml`: s=1.2, b=23.6). PPO hyperparameters unchanged (lr-anneal
  3e-4, n_steps 720, batch 120; 5-D action → budget ~1M steps, extend if not plateaued).

## Verification
1. **Smoke:** `uv run python use_case/skoven_RL_control.py --smoke` → expect obs/action
   shapes updated (action `(5,)`), reward finite and varying.
2. **Cascade sanity (scratch script in `$CLAUDE_JOB_DIR/tmp`):** step the env with a
   constant action; assert `ecl310_TSupSet_schedule` output == `compute_supply_setpoint`
   of the mean setpoint + live `T_oa`; assert a cold room's damper opens above
   `base_pos` and a satisfied room's sits at baseline; assert supply-air floor engages
   only when the building overshoots.
3. **Baseline:** `uv run python use_case/baseline_eval.py` (no `--compare`) → all rooms
   hold ~21 °C, ~100% within [19,26]; record heating/AHU kWh as the new reference.
4. **Train:** launch ~1M-step PPO (background + watcher), then plot training stats and
   `baseline_eval.py --compare`. Success = comfort ~100% in [19,26] at **lower** heating
   (+AHU) energy than the baseline — i.e. RL finally on the winning side of the frontier.

## Risks / notes
- Unconnected source schedules may be skipped by the step order → mitigated by the
  read-back sensor connection (step 2).
- 1-step delay on `T_oa`/zone temps in the simulator is intentional and matches the
  existing economizer; verify it doesn't destabilize the PID at 600 s steps.
- Wide band can make energy dominate so much the agent rides the 19 °C floor; that is
  acceptable (19 °C is in-band) but watch that it doesn't dip below on cold nights —
  retune `W_COMFORT` if violations appear.
- Archive the current VAV model/run before retraining (as done for prior versions).
