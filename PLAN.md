# Plan — Aarhus / Skoven Twin4Build Digital Twin + RL Gym

## Context

The existing `T4BGymUseCase/` contains a working pipeline that calibrates a Twin4Build (T4B) 5-room office model, wraps it as a Gymnasium environment, and trains a PPO agent that outperforms a rule-based HVAC controller. The goal is to evolve this pattern from a synthetic 5-room office to a real residential building in the Jettesvej 10 complex in Aarhus, for which roughly 17 months of BMS, ReMoni sensor, district-heating, and electricity data have been collected (2024-11-08 → 2026-04-27). The four buildings in the complex (Bækken, Engen, Skoven, Stranden) are architecturally identical, so one well-calibrated model parameterized per building can represent all four — but the implementation must start by nailing one building end-to-end. **Skoven** has been chosen as the primary target because it has both 2-year ReMoni data AND its own BMS controller export with controllable setpoints (`Rumtemp.ref.`, `Fremløbstemp.ref.`).

The intended outcome is a self-contained `AarhusCase/` Python package that ingests Skoven's heterogeneous raw data, fits a three-stage T4B model (envelope → hydronic → AHU), exposes it through a `T4BGymEnv`, and lets an RL agent train against measured reward signals (heat-meter kWh + electricity kWh + comfort violations). The reference `T4BGymUseCase/` stays untouched.

## User decisions taken

- Primary building: **Skoven**; Bækken/Engen/Stranden are derived later via parameter deltas.
- Sundhedshus: **ignored** (out of scope, not used as boundary).
- HVAC scope: **full** — district heating → ECL310 hydronic loop → radiators → zones → AHU heat recovery → ventilation/CO₂.
- Code location: new `/home/sebscubs/repos/AarhusCase/AarhusCase/` package, parallel to `T4BGymUseCase/`.

## Twin4Build components available (verified in `/home/sebscubs/repos/AarhusCase/Twin4Build/twin4build/systems/`)

The Twin4Build dev branch ships PyTorch-backed, differentiable component models that already cover the full hydronic + AHU + envelope topology. Earlier I assumed several custom wrappers would be needed; they are not. The components below are instantiated directly — no subclassing required — and `tb.Estimator` works against their parameters via autograd. Note the class names are the new `*TorchSystem` family; the reference T4BGymUseCase was written against the older `*FMUSystem` API.

| Subsystem | T4B class | Module path |
| :-- | :-- | :-- |
| Zone (envelope + air + heat input) | `BuildingSpaceThermalTorchSystem` | `twin4build.systems.building_space.building_space_thermal_torch_system` |
| Zone with CO₂ | `BuildingSpaceTorchSystem` | `twin4build.systems.building_space.building_space_torch_system` |
| Radiator | `SpaceHeaterTorchSystem` | `twin4build.systems.space_heater.space_heater_torch_system` |
| Hydronic control valve (ECL310 mixing + thermostatic valves) | `ValveTorchSystem` | `twin4build.systems.valve.valve_torch_system` |
| Return-water manifold | `ReturnFlowJunctionSystem` | `twin4build.systems.junction.return_flow_junction_system` |
| Supply-water/air manifold | `SupplyFlowJunctionSystem` | `twin4build.systems.junction.supply_flow_junction_system` |
| AHU heat recovery | `AirToAirHeatRecoverySystem` | `twin4build.systems.air_to_air_heat_recovery.air_to_air_heat_recovery_system` |
| AHU heating coil | `CoilTorchSystem` | `twin4build.systems.coil.coil_torch_system` |
| AHU fan | `FanTorchSystem` | `twin4build.systems.fan.fan_torch_system` |
| AHU damper | `DamperTorchSystem` | `twin4build.systems.damper.damper_torch_system` |
| PI/PID closed-loop control | `PIDControllerSystem` | `twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system` |
| Schedule (occupancy, setpoints) | `ScheduleSystem`, `PiecewiseLinearScheduleSystem` | `twin4build.systems.schedule` |
| Outdoor weather | `OutdoorEnvironmentSystem` | `twin4build.systems.outdoor_environment` |
| Sensor passthrough | `SensorSystem` | `twin4build.systems.sensor` |

Key port mappings that drive the topology:
- `SpaceHeaterTorchSystem` consumes `supplyWaterTemperature`, `waterFlowRate`, `indoorTemperature`; produces `outletWaterTemperature` and `Power [W]`. Its `Power` connects directly into `BuildingSpaceThermalTorchSystem.heatGain` — that single port is the radiator-to-zone coupling, no air-side reheat coil needed for the hydronic loop.
- `ValveTorchSystem` consumes `valvePosition ∈ [0,1]` and emits `waterFlowRate`. It is the same class used for the ECL310 main mixing valve AND for each zone's thermostatic radiator valve — just different parameters (`waterFlowRateMax`, `valveAuthority`).
- `BuildingSpaceThermalTorchSystem` has an `adjacentZoneTemperature` vector input for multi-zone coupling, replacing the manual `indoorTemperature_adj` adjacency wiring of the older `BuildingSpaceNoSH1AdjBoundaryFMUSystem`.

**The only piece T4B does NOT ship is an outdoor-reset / heating-curve controller for the ECL310.** This is implemented as a small Python helper (`aarhus_model/heating_curve.py`) that computes `T_sup_set = clip(b + s·(T_room_ref − T_oa) + Δ, T_min, T_max)`. Its output is wired into a `PIDControllerSystem` whose actual signal is the measured supply-water temperature and whose output is the ECL310 mixing-valve position. Slope `s`, offset `b`, and parallel shift `Δ` are estimated in Stage 2.

## Approach

### 1. Repo layout (new files under `/home/sebscubs/repos/AarhusCase/AarhusCase/`)

```
AarhusCase/
├── pyproject.toml
├── AGENTS.md
├── PLAN.md                              # this file
├── data_ingest/                         # NEW — no equivalent in reference
│   ├── skoven_bms.py                    # parses Jettesvej 10ASkoven-…csv.xlsx
│   ├── remoni_indoor.py                 # parses Skoven/Jettevej 10 - Skoven-metric_{1,2,3}.csv + sensors_8_9_10_11.csv
│   ├── varme_meter.py                   # parses Varme N.xlsx, computes power from ΔEnergi
│   ├── el_meter.py                      # parses EL N.xlsx
│   ├── meter_matcher.py                 # heuristic Varme/EL → building correlator with manual override
│   ├── outdoor_synth.py                 # PVlib clear-sky irradiance for Aarhus 56.16°N 10.20°E
│   ├── danish_decode.py                 # Latin-1 / UTF-8 fix-up + `--`→NaN
│   ├── unify.py                         # fuses all sources into a tz-aware 15-min DataFrame
│   └── export_t4b_csvs.py               # splits into per-signal CSVs T4B SensorSystem can consume
├── aarhus_model/
│   ├── skoven_model.py                  # envelope_fcn / hydronic_fcn / ahu_fcn / fcn / get_model / load_model_and_params / model_output_points
│   ├── heating_curve.py                 # ECL310 outdoor-reset helper (T_sup_set = f(T_oa, T_room_ref, s, b, Δ))
│   ├── only_envelope_param_est.py       # stage 1
│   ├── hydronic_param_est.py            # stage 2
│   ├── only_ahu_param_est.py            # stage 3
│   └── generated_files/
│       ├── data/skoven/                 # per-signal CSVs (hvac_reaZon*_TZon_y_processed.csv style)
│       └── models/skoven_{envelope,hydronic,ahu}_estimation/...   # pickled LS results
├── t4b_gym/
│   └── t4b_gym_env.py                   # reused verbatim from T4BGymUseCase/t4b_gym/t4b_gym_env.py
└── use_case/
    ├── policy_input_output.json         # NEW — Skoven IO schema
    ├── skoven_RL_control.py             # PPO entrypoint + custom reward
    ├── model_eval.py                    # copied from reference, light edits
    ├── building_configs/
    │   ├── skoven.yaml                  # zones, sensor IDs, meter IDs, areas
    │   ├── baekken.yaml                 # bms_source: skoven_proxy
    │   ├── engen.yaml
    │   └── stranden.yaml
    └── logs/
```

### 2. Data ingest

Entry point `data_ingest/unify.py::build_skoven_frame(start, end, stride="15min")`:

1. **BMS** (`Jettesvej2Brabrand/PredictiveOptimalControlAarhus/Jettesvej 10ASkoven-144523-30.3.2026.csv.xlsx`): `openpyxl`, map `Tidsstempel→index`, treat `--`/blanks as NaN, run `danish_decode.fix()` on headers (Â°C → °C), localize Europe/Copenhagen. Columns: `Udetemperatur, Rumtemp.(Kr.1), Rumtemp.ref.(Kr.1), Fremløbstemp.(Kr.1), Fremløbstemp.ref.(Kr.1), Returtemp.(Kr.1), Returtemp.ref.(Kr.1)`.
2. **ReMoni** (`Jettesvej2Brabrand/Skoven/*.csv`): `metric_1/2/3.csv` are long-format (one row per timestamp tagged by `name` like `Viben 6 - ERS2 CO2 - E8A6`) → group by `name` then pivot. `metric_sensors_8_9_10_11.csv` is wide; regex-parse the literal `metric` text payload `"CO2: 942 ppm, Temperatur: 22.6 °C, Humidity: 42 %, …"`.
3. **Varme/EL meters**: read each xlsx, compute `power_kW = diff(Energi[MWh]) * 1000 / Δh`. Column order varies across files — detect defensively.
4. **`meter_matcher.py`**: correlate each Varme power series against `(T_zone_set − T_zone_bms)` and `(T_sup_w − T_oa)` for each candidate building; emit `meter_matches.csv` with confidence scores for human sign-off (open question §6.2). Allow override via `building_configs/*.yaml`.
5. Resample all sources to 15 min: ReMoni `.mean()`, BMS `.mean()`, meter power forward-filled then ÷4. Gaps <60 min forward-filled; larger gaps masked NaN and excluded from estimation windows.
6. `export_t4b_csvs.py` writes per-signal files under `aarhus_model/generated_files/data/skoven/` using the reference's naming convention (`hvac_reaZon{zone}_TZon_y_processed.csv`, `weaSta_reaWeaTDryBul_y_processed.csv`, etc.) so the T4B SensorSystem ingest works unchanged.

### 3. Zone topology

**Starting topology**: one zone per Viben sensor cluster. Skoven has at least sensors 6–11 visible; group co-located sensors into one logical zone if they share a room. Adjacency: a central `core`/corridor zone connected to each perimeter zone via `indoorTemperature_adj`. All zones share a single `Kr.1` supply-water boundary (the BMS only exposes one hydronic loop — the main real-world departure from the reference, which has per-zone reheat coils on a common air manifold).

**Fallback** if sensor → room mapping is unknown: collapse to 3 zones (`floor0`, `floor1`, `core`) keyed by floor from the floor-plan PDF, averaging sensors per bin. Use `BuildingSpaceNoSH1AdjBoundaryOutdoorFMUSystem` for perimeters, `BuildingSpaceNoSH1AdjBoundaryFMUSystem` for `core` — identical classes to `5_rooms_model.py:141-145`.

### 4. T4B model construction (`aarhus_model/skoven_model.py`)

Three sub-fcns + master `fcn()`, same shape as `5_rooms_model.py:136-488` and `rooms_and_ahu_model.py:95-250`, but built against the new `*TorchSystem` classes. `get_model()` and `load_model_and_params()` mirror reference signatures (see `5_rooms_model.py:490-500`).

**`envelope_fcn(self)`** — per zone instantiate `BuildingSpaceThermalTorchSystem` (`BuildingSpaceTorchSystem` if CO₂ dynamics are wanted); one `OutdoorEnvironmentSystem` fed by BMS Udetemperatur + synthesized PVlib irradiance for 56.16°N/10.20°E; `ScheduleSystem` for occupancy (initially constant, refined post-calibration); per-zone `SensorSystem` for indoor temp + CO₂ backed by ReMoni CSVs. Adjacency is wired via each zone's `adjacentZoneTemperature` vector input rather than the per-pair `indoorTemperature_adj` connections used in the older reference.

**`hydronic_fcn(self)`** — built entirely from stock T4B components:
- One `ValveTorchSystem(id="ecl310_kr1_valve")` representing the ECL310 main mixing valve. Estimated params: `waterFlowRateMax`, `valveAuthority`.
- `heating_curve.py::compute_supply_setpoint(T_oa, T_room_ref, s, b, Δ)` precomputes the supply-water setpoint trajectory and exposes it as a `ScheduleSystem(id="ecl310_TSupSet_schedule")`. Estimated params: `s`, `b`, `Δ`.
- One `PIDControllerSystem(id="ecl310_pid")` whose `setpointValue` is the schedule output and `actualValue` is the measured supply-water temperature, driving `ecl310_kr1_valve.valvePosition`.
- Per zone: `SpaceHeaterTorchSystem(id=f"{z}_radiator")` consuming `supplyWaterTemperature` from the supply manifold, `waterFlowRate` from a per-zone `ValveTorchSystem(id=f"{z}_trv")` (thermostatic radiator valve), and `indoorTemperature` from the zone. Its `Power` output connects to `BuildingSpaceThermalTorchSystem.heatGain` of the same zone. Estimated params per radiator: `Q_flow_nominal_sh`, `T_a_nominal_sh`, `T_b_nominal_sh`, `TAir_nominal_sh`, `thermalMassHeatCapacity`, `nelements` (fixed at 3).
- Per zone: a second `PIDControllerSystem(id=f"{z}_thermostat")` setting `{z}_trv.valvePosition` from `(zone_T_set − zone_T)`. Estimated params: `kp`, `Ti`.
- `ReturnFlowJunctionSystem(id="hyd_return_manifold")` aggregating radiator `outletWaterTemperature` + `waterFlowRate` outputs → outputs flow-weighted return T into a `SensorSystem(id="ecl310_TRetHea_y")`.
- `SensorSystem(id="varme_meter_power_sensor")` reads the matched `Varme {id}` time series (reward signal).
- `SensorSystem(id="ecl310_TSupHea_y")` reads BMS Fremløbstemp for the PID's `actualValue` during open-loop calibration.

**`ahu_fcn(self)`** — direct port of the AHU section in `rooms_and_ahu_model.py` rebuilt with new classes: `FanTorchSystem`, `CoilTorchSystem` (supply heating coil), `AirToAirHeatRecoverySystem` (heat-recovery wheel; estimated params `eps_75_h`, `eps_100_h`, `eps_75_c`, `eps_100_c`, `primaryAirFlowRateMax`, `secondaryAirFlowRateMax`), per-zone supply/return `DamperTorchSystem`, reuse `SupplyFlowJunctionSystem`/`ReturnFlowJunctionSystem`. If Skoven AHU sensor data is sparse (likely — most AHU docs are about Sundhedshus which is out of scope), keep the stage in the topology but use literature defaults for Stage 3 (see §5) and flag in plots.

**Master `fcn(self)`**: `envelope_fcn(self); hydronic_fcn(self); ahu_fcn(self)`.

### 5. Parameter estimation — three pickle stages

Same `tb.Estimator(...).estimate(...)` workflow as `5_rooms_model.py:680-689`. The new `*TorchSystem` components are fully differentiable, so the LS estimator runs over autograd-computed Jacobians. Pickles land under `generated_files/models/skoven_{stage}_estimation/model_parameters/estimation_results/LS_result/`. `load_model_and_params()` chains the three `model.load_estimation_result(...)` calls (template: `rooms_and_ahu_model.py:1040-1066`).

- **Stage 1 — `only_envelope_param_est.py`** (mirrors `only_rooms_param_est.py`). Target params per zone (on `BuildingSpaceThermalTorchSystem`): `C_air`, `C_wall`, `C_int`, `C_boundary`, `R_out`, `R_in`, `R_int`, `R_boundary`, `f_wall`, `f_air`, `Q_occ_gain`, infiltration / `airVolume` if exposed by the chosen zone class. Target devices: `{z}_indoor_temp_sensor` (std=0.1, scale=20), `{z}_co2_sensor` (std=10, scale=400) when using `BuildingSpaceTorchSystem`. Boundary forcing: outdoor T from BMS, a constant `heatGain` injected per zone derived from BMS supply-water T (so the envelope sees plausible radiator power before stage 2), occupancy schedule. Window: 2024-12-01 → 2025-01-15 (high-ΔT winter).
- **Stage 2 — `hydronic_param_est.py`**. Target params: heating-curve `s`, `b`, `Δ`; ECL310 `ValveTorchSystem` `waterFlowRateMax`, `valveAuthority`; ECL310 PID `kp`, `Ti`; per-zone `SpaceHeaterTorchSystem` `Q_flow_nominal_sh`, `T_a_nominal_sh`, `T_b_nominal_sh`, `TAir_nominal_sh`, `thermalMassHeatCapacity`; per-zone TRV `waterFlowRateMax`, `valveAuthority`; per-zone thermostat PID `kp`, `Ti`. Target devices: `ecl310_TSupHea_y` (std=0.5, scale=70), `ecl310_TRetHea_y` (std=0.5, scale=50), `varme_meter_power_sensor` (std=0.5, scale=20), zone temps. Load Stage 1 pickle first. Window: 2025-01-15 → 2025-02-15.
- **Stage 3 — `only_ahu_param_est.py`** (mirrors `only_ahu_model.py`). Target params: `FanTorchSystem` `c1..c4`, `nominalPowerRate`, `nominalAirFlowRate`, `f_total`; `AirToAirHeatRecoverySystem` `eps_75_h`, `eps_100_h`, `eps_75_c`, `eps_100_c`; `CoilTorchSystem` if instrumented; `DamperTorchSystem` `a`, `nominalAirFlowRate`. If sensor data insufficient, use literature defaults and proceed.

### 6. RL environment (`use_case/policy_input_output.json` + `skoven_RL_control.py`)

Top-level JSON keys identical to reference (`actions`, `observations`, `time_embeddings`, `forecasts`) — `min/max/signal_key/description` per leaf as in `T4BGymUseCase/use_case/policy_input_output.json:1-50`.

- **Actions** (~`2 + N_zones`): `ecl310_kr1.heatingsetpointValue` (30–70 °C), per-zone `{z}_temperature_heating_setpoint.scheduleValue` (18–23 °C), AHU `main_supply_damper.damperPosition` (0–1), `main_mixing_damper.damperPosition` (0–1), `supply_air_temp_setpoint_sensor.measuredValue` (16–24 °C).
- **Observations** (~`3·N_zones + 5`): per-zone indoor temp (0–40), CO₂ (300–2000), radiator valve position (0–1); `ecl310_TSupHea_y` (20–80), `ecl310_TRetHea_y` (20–60), `varme_meter_power_sensor` (0–50 kW), `outdoor_environment.outdoorTemperature` (−20–35); AHU subset if available.
- **Forecasts**: `outdoor_temperature`, `global_irradiation` (PVlib-synthesized), per-zone heating setpoints and occupancy.
- **Time embeddings**: `time_of_day, day_of_week, month_of_year` (same as reference).
- **Reward** (`skoven_RL_control.py`, mirrors `multizone_simple_air_RL_control.py:50-129`):
  `reward_t = -((Σ_z temp_violations_z · 10000) + heat_meter_kW + 0.5·ahu_kW)/1000; return -(reward_t − previous)`.
  600 s step, 5-day episodes, random start in 2024-12-01…2025-12-01 (train) / 2026-01-01…2026-04-15 (eval). PPO hyperparams identical to reference (`multizone_simple_air_RL_control.py:160-173`): `learning_rate=1e-5, batch_size=50, n_steps=200, n_epochs=10, gamma=0.99, total_timesteps=500_000`.

### 7. Multi-building generalization

`use_case/building_configs/{baekken,engen,stranden}.yaml` declares zones, sensor IDs, BMS path, Varme/EL meter IDs, area/volume scalars. `skoven_model.get_model(building="baekken")` reads YAML and parameterizes `ZONES`. Re-run the three estimation stages per building → `generated_files/models/{building}_..._estimation/`. Reward stays building-agnostic. **Bækken** lacks its own BMS export — its YAML sets `bms_source: "skoven_proxy"`, reuses Skoven's ECL310 heating-curve as the prior, and re-estimates only zone-level radiator/envelope params from ReMoni data.

### 8. Dependencies (`pyproject.toml`)

Use `uv`. `twin4build` is installed manually by the user — keep it out of the manifest.

```toml
[project]
name = "aarhuscase"
requires-python = ">=3.10,<3.13"
dependencies = [
  "gymnasium>=0.29", "stable-baselines3>=2.3",
  "pandas>=2.2", "numpy>=1.26,<2.0",
  "openpyxl>=3.1", "matplotlib>=3.8",
  "python-dateutil>=2.9", "tzdata>=2024.1",
  "pvlib>=0.11", "pyyaml>=6.0", "scipy>=1.13",
]
```

## Critical files

To **create**:
- `/home/sebscubs/repos/AarhusCase/AarhusCase/pyproject.toml`
- `/home/sebscubs/repos/AarhusCase/AarhusCase/data_ingest/{unify,skoven_bms,remoni_indoor,varme_meter,el_meter,meter_matcher,outdoor_synth,danish_decode,export_t4b_csvs}.py`
- `/home/sebscubs/repos/AarhusCase/AarhusCase/aarhus_model/skoven_model.py`
- `/home/sebscubs/repos/AarhusCase/AarhusCase/aarhus_model/{only_envelope_param_est,hydronic_param_est,only_ahu_param_est}.py`
- `/home/sebscubs/repos/AarhusCase/AarhusCase/aarhus_model/heating_curve.py`
- `/home/sebscubs/repos/AarhusCase/AarhusCase/use_case/{policy_input_output.json,skoven_RL_control.py,model_eval.py}`
- `/home/sebscubs/repos/AarhusCase/AarhusCase/use_case/building_configs/{skoven,baekken,engen,stranden}.yaml`
- `/home/sebscubs/repos/AarhusCase/AarhusCase/t4b_gym/t4b_gym_env.py` (verbatim copy of reference)

To **read / reuse**:
- `/home/sebscubs/repos/AarhusCase/T4BGymUseCase/boptest_model/5_rooms_model.py` — fcn pattern (esp. lines 136-488, 490-500, 680-689).
- `/home/sebscubs/repos/AarhusCase/T4BGymUseCase/boptest_model/rooms_and_ahu_model.py` — AHU + `load_model_and_params()` template (lines 95-250, 1040-1066).
- `/home/sebscubs/repos/AarhusCase/T4BGymUseCase/boptest_model/{only_rooms_param_est,vav_controllers_param_est,only_ahu_model}.py` — three-stage estimation templates.
- `/home/sebscubs/repos/AarhusCase/T4BGymUseCase/t4b_gym/t4b_gym_env.py` — reused verbatim.
- `/home/sebscubs/repos/AarhusCase/T4BGymUseCase/use_case/{multizone_simple_air_RL_control.py,policy_input_output.json,model_eval.py}` — PPO + reward + IO schema templates.
- Skoven sources: `Jettesvej2Brabrand/PredictiveOptimalControlAarhus/Jettesvej 10ASkoven-144523-30.3.2026.csv.xlsx`, `Jettesvej2Brabrand/Skoven/Jettevej 10 - Skoven-metric_{1,2,3}.csv`, `metric_sensors_8_9_10_11.csv`.
- Meters: `Jettesvej2Brabrand/Varme *.xlsx` (7 files), `Jettesvej2Brabrand/EL *.xlsx` (3 files).
- Building docs: `Jettesvej2Brabrand/PredictiveOptimalControlAarhus/{06100550-1.101 - Stueplan og 1. salsplan_RevC.pdf, ECL310 Jettesvej 10A-E.docx, AHU Sundhedshus.docx, PredictiveControlRequirements.docx}`.

## Verification / acceptance

- **Data ingest**: `unify.build_skoven_frame()` returns a tz-aware DataFrame with the expected columns and <5 % NaN over Dec 2024 – Mar 2025; `meter_matches.csv` produced with confidence scores.
- **Stage 1 calibration**: per-zone indoor-T RMSE < 1.0 °C and CO₂ RMSE < 100 ppm on a held-out 5-day window.
- **Stage 2 calibration**: T_sup_w RMSE < 2 °C, T_ret_w RMSE < 2 °C, hourly heat-meter kW MAPE < 15 %.
- **Stage 3 calibration**: supply-air-T RMSE < 1.5 °C if AHU data is sufficient; else literature-default fallback documented.
- **Forward simulation**: 7-day run at 60 s step finishes with no NaN/FMU crash; outputs match data shape.
- **Gym env**: `env.reset()` returns observation matching `env.observation_space.shape`; 1000 random-action steps complete; `model.learn(total_timesteps=10_000)` finishes without crash.
- **Baseline replay**: feeding historical BMS setpoints as actions yields total heat energy within 10 % of measured Varme energy over the eval window (2026-01 → 2026-04).

End-to-end commands the user will be able to run:

```bash
uv run python -m data_ingest.unify --building skoven --start 2024-12-01 --end 2025-03-01
uv run python aarhus_model/only_envelope_param_est.py --building skoven
uv run python aarhus_model/hydronic_param_est.py --building skoven
uv run python aarhus_model/only_ahu_param_est.py --building skoven
uv run python use_case/skoven_RL_control.py --train
uv run python use_case/model_eval.py --building skoven --window 2026-01-01:2026-04-15
```

## Open questions for user (non-blocking, can be answered as implementation progresses)

1. **Viben-to-room mapping for Skoven** (sensors 6, 7, 8, 9, 10, 11 plus any in `metric_1/2/3.csv`): which apartment/floor/orientation?
2. **Varme/EL → building mapping**: 7 Varme + 3 EL meters across 4 buildings. Confirm the `meter_matches.csv` heuristic output once produced.
3. **Calendar split**: proposed train 2024-12-01 → 2025-12-01, eval 2026-01-01 → 2026-04-15 — accept?
4. **Radiator UA per zone**: available from `ECL310 Jettesvej 10A-E.docx`, or estimate-only?
5. **Skoven AHU instrumentation**: is the AHU only documented in the (out-of-scope) `AHU Sundhedshus.docx`, or does Skoven have its own AHU sensors in the BMS export that we should expose?
