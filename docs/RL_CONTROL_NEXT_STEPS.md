# RL Control — Next Steps (after v2 supervisory-setpoint run)

_Status date: 2026-07-07. Follows `docs/RL_CONTROL_PLAN_V2.md`. Work is on branch
`rl-supervisory-setpoint-control`._

## Where we are

The v2 supervisory-setpoint controller (RL emits 4 per-room indoor-temp setpoints
+ 1 supply-air setpoint; comfort is the range **[19, 26] °C**) trained cleanly —
the healthiest run of the project (policy `std` 1.0 → 0.22, smooth monotone
reward, `comfort_ok ≈ 1`, fully converged). **But it does not beat the rule-based
baseline.**

Full-winter compare (`baseline_eval.py --compare`, eval window 2025-12-01 →
2026-03-01, `best_model` @245k):

| metric | baseline | RL | change |
|---|---|---|---|
| Time in band [19,26] | 100 % | 100 % | 0 |
| Heating | 2778 kWh | 2794 kWh | ~same |
| AHU (fan+coil) | 802 kWh | **1666 kWh** | **doubled** |
| Total energy | **3580 kWh** | **4460 kWh** | **+25 % (worse)** |
| Mean reward / step | −0.147 | −0.168 | worse |

The RL is worse on total energy **and** on its own reward metric.

## Root causes (diagnosed)

1. **The reward discounts AHU energy by 0.5×.**
   `energy_kW = heat_kW + 0.5 * ahu_W/1000` in `SkovenGymEnv.get_reward`
   (`use_case/skoven_RL_control.py`). The agent games this: it shifts load off the
   radiators (charged 1×) onto the AHU coil/fan (charged 0.5×) via the per-room
   damper trim. In mild weather the heat→ventilation trade lowers the *discounted*
   reward; in cold January the radiators can't back off (rooms need the heat to
   stay ≥ 19 °C), so the extra ventilation is pure added real energy. Net: real
   energy rises while the reward falls. (Deterministic Dec-only probe: heat ~0.6
   kW, dampers near baseline — it does run lean in mild weather; the cold months
   erase the saving and the doubled AHU remains.)

2. **`best_model` overfits to a mild window.**
   The `EvalCallback` eval env is a **5-day episode starting at `EVAL_START`**
   (Dec 1–6, mild), not the full winter (`train()` in
   `use_case/skoven_RL_control.py`, `EPISODE_STEPS` = 5 days). Its best eval
   reward (−73) reflects only that mild stretch, so model selection favours a
   policy tuned to mild weather that generalises worse over Dec–Feb.

## Next steps

### 1. Remove the AHU reward discount (primary fix)
- In `SkovenGymEnv.get_reward` (`use_case/skoven_RL_control.py`), change
  `energy_kW = heat_kW + 0.5 * ahu_W/1000` → **`heat_kW + ahu_W/1000`** (full
  1× weight, i.e. true kWh). Consider making the AHU weight a named class
  constant (e.g. `AHU_ENERGY_WEIGHT = 1.0`) so it is explicit and tunable.
- Rationale: the agent should see the coil/fan energy at its real cost so it
  cannot "hide" heat in the ventilation system.

### 2. Select `best_model` on the full winter, not a 5-day slice
- Make the held-out eval representative of the whole eval window. Options, in
  order of preference:
  - Increase the eval env's episode length to span the full window (or a large
    representative chunk), reusing the DST-safe chunking already in
    `use_case/model_eval.py` (`test_model_chunked`) / `rl_config.dst_safe_chunks`;
    or
  - Set `EvalCallback(n_eval_episodes=N)` with the eval env using
    `random_start=True` so N episodes sample across Dec–Feb, and select on the
    average.
- Rationale: stop over-fitting model selection to mild December.

### 3. Retrain and re-evaluate
- Clean previous run artifacts (`use_case/logs/*`, `use_case/plots_rl_training/*`),
  keeping the baseline.
- Re-run the baseline (`baseline_eval.py`) — unchanged incumbent: 100 % in
  [19,26], heating 2778 kWh, AHU 802 kWh (total 3580 kWh) is the target to beat.
- Launch a fresh ~1M-step PPO run (same supervisory config).
- Plot training stats + `baseline_eval.py --compare`.
- **Success criterion:** 100 % time in [19,26] at **total energy < 3580 kWh**
  (a genuine win on the whole winter, not just mild December).

## Optional / if step 1–2 still fall short

- **Damper-trim cost/authority tuning** (`SupervisorySetpointGymSimulator`):
  `K_DAMPER` (0.4) and `DAMPER_DEADBAND` (0.5) set how aggressively the air loop
  boosts; if ventilation is still over-used, raise the deadband / lower the gain.
- **Setpoint action range**: indoor `[18,26]`, supply-air `[14,30]` in
  `use_case/policy_input_output.json` — widen the lower indoor bound (e.g. 17) if
  we want the agent to push rooms nearer the 19 °C floor for more heating savings.
- **Report total energy explicitly** in `baseline_eval.py --compare` (heating +
  AHU as one KPI) so the headline number matches the objective.
- **Per-room hydronic** (deferred, larger change): the shared supply-water loop
  still can't target one room; per-room radiator valves remain the structural
  lever if per-room comfort (not just band membership) becomes the goal.
