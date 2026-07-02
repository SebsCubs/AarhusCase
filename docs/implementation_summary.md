# AarhusCase — Implementation Summary

Status snapshot for the `AarhusCase/` package as of the end of this session.
The goal is a Twin4Build (T4B) digital twin of the **Skoven** residential
building (Jettesvej 10A) wrapped in a Gymnasium env so a PPO agent can be
trained against measured heat-meter / electricity / comfort signals. Three
sister buildings (Bækken, Engen, Stranden) are derived by per-building config
overrides.

The full design is in `PLAN.md`. This file captures **what's actually built,
what's been verified to run, and what's left**.

---

## 1. Repo layout

```
AarhusCase/
├── pyproject.toml              # uv manifest; pins ../Twin4Build as editable
├── PLAN.md                     # design doc (single source of truth for scope)
├── implementation_summary.md   # this file
├── aarhus_model/
│   ├── skoven_model.py         # envelope_fcn + hydronic_fcn + ahu_fcn + fcn
│   ├── heating_curve.py        # ECL310 outdoor-reset curve helper
│   ├── only_envelope_param_est.py
│   ├── hydronic_param_est.py
│   ├── only_ahu_param_est.py
│   └── generated_files/
│       ├── data/skoven/        # per-signal CSVs consumed by SensorSystem
│       └── models/             # estimator pickle outputs (per stage)
├── data_ingest/
│   ├── danish_decode.py        # mojibake + '--' → NaN
│   ├── skoven_bms.py           # parse Jettesvej 10ASkoven-*.xlsx
│   ├── remoni_indoor.py        # long-format + regex 'metric' payloads
│   ├── varme_meter.py          # ΔEnergi → kW
│   ├── el_meter.py             # ΔEnergi → kW
│   ├── meter_matcher.py        # heuristic Varme/EL → building assignment
│   ├── outdoor_synth.py        # PVlib clear-sky GHI for Aarhus
│   ├── unify.py                # build_skoven_frame() entry point
│   └── export_t4b_csvs.py      # split unified df → per-signal CSVs
├── t4b_gym/
│   └── t4b_gym_env.py          # copied from reference (System import path fixed)
├── use_case/
│   ├── policy_input_output.json
│   ├── skoven_RL_control.py    # PPO entrypoint + custom reward
│   ├── model_eval.py           # baseline sim + rollout + plots (Skoven 2026 window)
│   └── building_configs/
│       ├── skoven.yaml
│       ├── baekken.yaml        # bms_source: skoven_proxy
│       ├── engen.yaml
│       └── stranden.yaml
└── scripts/
    └── smoke_test.py           # end-to-end build+simulate sanity check
```

---

## 2. Dependencies & environment

- **Tool:** `uv` (`~/.local/bin/uv`).
- **Python:** 3.12, venv at `AarhusCase/.venv/` created by `uv sync`.
- **Local Twin4Build:** declared via `[tool.uv.sources]` in `pyproject.toml`
  as `{ path = "../Twin4Build", editable = true }`. Edits to the cloned T4B
  source are picked up immediately — no reinstall step.
- **Top-level deps:** `twin4build`, `gymnasium>=0.29`, `stable-baselines3>=2.3`,
  `pandas>=2.2`, `numpy>=1.26,<2`, `openpyxl>=3.1`, `matplotlib>=3.8`,
  `pvlib>=0.11`, `pyyaml>=6.0`, `scipy>=1.13`, `python-dateutil`, `tzdata`.
- **Run any script:** `uv run python <script.py>` (auto-activates venv).

---

## 3a. Full real-data calibration status (June 2026 update)

The complete pipeline now runs end-to-end on real Skoven data. Key changes this
session and current results:

**Data ingest (data_ingest/)**
- Fixed a path bug (loaders went up 3 dirs; package is nested at `AarhusCase/AarhusCase`).
- BMS column mapping made whitespace-tolerant; DST-safe `tz_localize`
  (`ambiguous="NaT"`, `nonexistent="shift_forward"`) across all loaders; ReMoni
  duplicate-timestamp dedup; BMS 6-hourly water signals time-interpolated.
- Added `weather_openmeteo.py` — hourly ERA5 outdoor temperature for Aarhus
  (open-meteo archive, cached), used as the `T_oa` boundary instead of the
  6-hourly BMS Udetemperatur. **Decision: external hourly weather.**
- `export_t4b_csvs.py` rewritten to emit exactly the filenames the model reads
  (per-zone temps via the YAML sensor mapping, a Stage-1 `{zone}_heat_input.csv`
  boundary, the heating-curve trajectory), written in **UTC** (the T4B loader
  rejects the mixed +01:00/+02:00 offsets a DST-spanning local-time file carries).
- Provisional mappings (auto + placeholders): 3 ReMoni sensors (Viben 6/4/5) →
  zones core/floor0/floor1 1:1; Skoven heat meter = `5344947` (clearest
  space-heating signature: high in Jan, ~0 in spring, negative outdoor corr).

**Estimation (aarhus_model/) — migrated to dev-branch Estimator API**
- `Estimator(simulator)`, `parameters` as list-of-tuples (pruned to real
  `tps.Parameter` attrs — dropped `Q_occ_gain` and the SpaceHeater float
  nominals), `measurements` as `(sensor, std)` tuples, `method=("scipy","SLSQP","ad")`.
- Measurement sensors given observed-data `filename`s in calibration mode
  (the Estimator requires both a connection and a file); Stage 1 uses an
  envelope-only model with a measured heat-input boundary (`make_envelope_fcn`).
- Env-tunable `AARHUS_MAXITER`; deterministic result pickles at
  `models/skoven_{envelope,hydronic,ahu}_estimation/result.pickle`.

**Calibration results (held-out simulation vs measured)**
- Stage 1 envelope (2025-01-15..22): zone-T RMSE **0.50 / 0.74 / 0.79 °C** (PASS, <1 °C).
- Stage 2 hydronic (2025-03-31..04-07): zone-T RMSE **0.62–1.40 °C**;
  return-water RMSE **6.9 °C** (POOR — single-radiator proxy + low-heat spring
  window; the only period with BMS water signals overlapping meters). Radiators
  now deliver heat (closed loop works).
- Stage 3 AHU: skipped by design (no Skoven AHU instrumentation → `AHU_DEFAULTS`).
- Calibrated RL-ready model (sim mode): total radiator power ~3.5 kW mean, zones
  held ~20–22.6 °C near setpoint.

**Gym/RL (t4b_gym/, use_case/)** — `GymSimulator` ported to the dev-branch
stepping API (`_assign_component_inputs` + `do_step(second_time, date_time,
step_size, step_index)`, history-indexed I/O, list args to `model.initialize`).
Added output-override support so ScheduleSystem setpoints (zone heating
setpoints, the main RL lever) are controllable despite having no input ports.
`policy_input_output.json` reconciled to real sim-mode ports (per-zone dampers,
`ecl310_TSupHea_y.scheduleValue`, dropped calibration-only varme sensor; nested
forecasts; time-embedding `signal_key`s). `model_eval.py` updated from the old
`savedOutput`/`Simulator()` API to `output[port].history()`.
`uv run python use_case/skoven_RL_control.py --smoke` passes: reset (obs 272,
action 8) → 10 steps → `PPO.learn(32)`. Full training is deferred to a later session.

**Known limitations (data, not code):** space heating is a winter phenomenon but
BMS water signals only begin 2025-03-30, so Stage 2 is confined to a low-heat
spring overlap (weak return-water fit). Meter→building and Viben→room mappings
are heuristic. Default-parameter (uncalibrated) radiators stay near 0 W; heat
appears only after calibration.

## 3. What runs today

`scripts/smoke_test.py` builds the model in simulation mode and runs a
6-hour forward simulation at 600 s steps. End-to-end output:

```
Model built: 37 components
Simulation completed successfully.
  core:   T_indoor min=14.35  max=19.98  last=14.35 °C
  floor0: T_indoor min=14.25  max=19.98  last=14.25 °C
  floor1: T_indoor min=14.25  max=19.98  last=14.25 °C
  core_radiator:   Power 0 W (TRV PID kp=0.001 → valve stays closed)
  floor0_radiator: Power 0 W
  floor1_radiator: Power 0 W
```

The model contains all three subsystems wired together:
- **Envelope** (3 zones, BuildingSpaceThermalTorchSystem with adjacency)
- **Hydronic** (ECL310 PID + mixing valve + per-zone TRV + radiator)
- **AHU** (heat recovery + fan + heating coil + per-zone dampers +
  air-side supply/return junctions)

Run it yourself:

```bash
cd AarhusCase
uv run python scripts/smoke_test.py
```

---

## 4. T4B dev-branch API gotchas (learned the hard way)

These surfaced during runtime debugging — the reference `T4BGymUseCase/`
targets an older T4B API. New patterns in use here:

1. **`tb.Model(id=...)`** — no `saveSimulationResult` kwarg anywhere.
2. **`model.load(...)`** uses snake_case kwargs:
   `draw_semantic_model`, `draw_simulation_model`, `validate_model`,
   `force_config_overwrite`.
3. **Component kwargs:** `saveSimulationResult=True` is gone. All TorchSystem
   constructors accept only their physical parameters + `id`.
4. **Vector input ports:** scalar-output → vector-input connections require
   `input_port_index=<slot:int>`. Applies to:
   - `BuildingSpaceThermalTorchSystem.adjacentZoneTemperature`
   - `SupplyFlowJunctionSystem.airFlowRateOut`
   - `ReturnFlowJunctionSystem.airFlowRateIn`, `airTemperatureIn`
5. **Component registration:** a component is only added to
   `model.components[]` after its **first `add_connection`** call.
   Implication: components that are referenced later (e.g. heating-setpoint
   schedules) must be created in the same `fcn` that wires them.
6. **`tb.Simulator(model)`** — model is a positional arg at construction.
   `simulate(start_time=, end_time=, step_size=, show_progress_bar=)` — all
   snake_case.
7. **`OutdoorEnvironmentSystem`** requires all three weather CSVs
   (`filename_outdoorTemperature`, `filename_globalIrradiation`,
   `filename_outdoorCo2Concentration`). No silent defaults.
8. **CSV format:** `pd.read_csv` is called with `header=0`. CSVs must have
   a header row (`time,value`).
9. **Output access:** `component.output[port].history()` returns a torch
   tensor of shape `(n_t, n_s, n_c)`. The old `component.savedOutput[port]`
   attribute is gone.
10. **System base path:** `from twin4build.systems.saref4syst.system import System`
    (not `twin4build.saref4syst.system`).

---

## 5. Architectural decisions taken

- **3-zone fallback topology** (`core`, `floor0`, `floor1`) — placeholder until
  Viben sensor → room mapping is confirmed for Skoven. All adjacency, areas,
  and nominal flow values are educated guesses.
- **No water-side junction component.** Each TRV connects directly to its
  radiator in parallel (T4B's `SupplyFlowJunctionSystem` is air-side; port
  names like `airFlowRateIn` rejected by the validator for water signals).
  Total hydronic heat is aggregated at the gym/eval layer by summing
  per-radiator `Power`. The Stage 2 calibration target
  `ecl310_TRetHea_y` is currently wired to a **single representative radiator's
  outletWaterTemperature**, not a proper mass-weighted mix — TODO below.
- **Calibration vs simulation mode** is a single flag (`calibration_mode`)
  threaded through `envelope_fcn` / `hydronic_fcn` / `ahu_fcn` / `make_fcn` /
  `get_model` / `load_model_and_params`. The flag swaps the supply-water
  source between a CSV-backed `SensorSystem` (open-loop replay of BMS
  Fremløb) and a `ScheduleSystem` (RL-controllable setpoint). Default for
  `load_model_and_params` is **simulation** (RL-ready).
- **No cooling.** Residential building — `model_eval.py` only computes a
  heating violation penalty; no cooling setpoint or cooling coil.

---

## 6. Open issues / TODO

### High priority (block real calibration)

1. **Viben → room mapping.** Resolve from the user / floor plan; replace the
   3-zone fallback with the real topology (likely per-Viben one zone). Update
   `ZONES` in `skoven_model.py` and the `zones` block in `skoven.yaml`.
2. **Meter-matcher confidence.** Run `data_ingest/meter_matcher.py` against
   the seven `Varme N.xlsx` files and confirm/override which one belongs to
   Skoven (and the others to Bækken/Engen/Stranden). Write the result into
   each building's `*.yaml` under `varme_meter_id`.
3. **Mass-weighted return-water mix.** Either write a tiny custom system
   that does `sum(m_i · T_out_i) / sum(m_i)` over the radiators, or compute
   it in the gym layer and feed it back via a virtual SensorSystem.
   Required for a meaningful Stage 2 target on `ecl310_TRetHea_y`.

### Medium priority (correctness, not crash)

4. **Tune PID defaults.** Current `tb.PIDControllerSystem(kp=0.001, Ti=10)`
   leaves the TRVs closed even with a 7 °C tracking error. Bump `kp`
   per zone to physical-radiator scale (e.g. `kp≈0.3 K⁻¹`, `Ti≈600 s`) or
   make it an estimated parameter in Stage 2.
5. **Heating-curve precomputation.** `ecl310_TSupSet_schedule` is currently a
   constant 50 °C. Wire it to the precomputed trajectory written by
   `data_ingest/export_t4b_csvs.py` using `tb.ScheduleSystem(filename=...)`.
6. **`load_model_and_params` defaults to simulation mode but the estimator
   scripts** (`only_envelope_param_est.py`, `hydronic_param_est.py`,
   `only_ahu_param_est.py`) still need to be reviewed against the new
   `Estimator` API on the dev branch — same kind of kwargs cleanup that
   `Model` / `Simulator` needed.

### Low priority (polish)

7. Verify `t4b_gym_env.py` is API-compatible with the dev branch beyond the
   `System` import — it still references the older `tb.Measurement` /
   `savedOutput` patterns transitively. Audit by running
   `uv run python use_case/skoven_RL_control.py --smoke` once written.
8. The PPO entrypoint (`skoven_RL_control.py`) hasn't been smoke-tested. After
   #4 is fixed, run `env.reset()` + 100 random steps to verify the
   observation/action wiring against `policy_input_output.json`.
9. **CUDA warning** at import time is benign (driver too old for the bundled
   torch CUDA build). Forward simulation runs on CPU at ~150 steps/s, fine
   for development. For PPO training, pin torch to a CPU build to silence
   the warning, or upgrade the host driver.

---

## 7. End-to-end run targets (for verification)

Commands the user should be able to run, in order, once the TODOs above are
addressed:

```bash
# 1. Data ingest (writes per-signal CSVs into aarhus_model/generated_files/data/skoven/)
uv run python -m data_ingest.unify --building skoven --start 2024-12-01 --end 2025-03-01

# 2. Stage 1: envelope calibration (per-zone C/R + occupancy gain)
uv run python aarhus_model/only_envelope_param_est.py --building skoven

# 3. Stage 2: hydronic calibration (ECL310 + radiator UA + TRV PID)
uv run python aarhus_model/hydronic_param_est.py --building skoven

# 4. Stage 3: AHU calibration (fan curve + heat-recovery effectiveness)
uv run python aarhus_model/only_ahu_param_est.py --building skoven

# 5. PPO training (500 k steps; saves to use_case/logs/)
uv run python use_case/skoven_RL_control.py --train

# 6. Eval against measured 2026-Q1 window
uv run python use_case/model_eval.py --building skoven --window 2026-01-01:2026-04-15
```

Acceptance criteria (from PLAN.md §"Verification"):
- Stage 1: per-zone indoor-T RMSE < 1.0 °C, CO₂ RMSE < 100 ppm on a held-out 5-day window
- Stage 2: T_sup_w RMSE < 2 °C, T_ret_w RMSE < 2 °C, hourly heat-meter MAPE < 15 %
- Stage 3: supply-air-T RMSE < 1.5 °C (or literature defaults flagged)
- Baseline replay: heat energy within 10 % of measured Varme over the eval window
