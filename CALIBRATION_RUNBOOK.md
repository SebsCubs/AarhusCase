# Calibration Runbook — Skoven CLOSED-LOOP model

How to (re)run parameter estimation and validation for the closed-loop Skoven
digital twin. Run everything from `/home/sebscubs/repos/AarhusCase/AarhusCase`
(so `uv run` uses the right project venv).

## What gets calibrated, and how

The closed-loop twin = a 4-zone RC **envelope** + a **closed-loop ECL310
substation** (the secondary supply-water temperature is produced by an emulated
PID + mixing valve, with the building return recirculated — see `hydronic_fcn` in
`aarhus_model/skoven_model.py`; it is now the only hydronic model).

Each subsystem mixes AD-estimated and physically-sized parameters. The split is
dictated by what the differentiable estimator can actually reach:

| Subsystem | AD-estimated | Sized physically (not AD) |
| :-- | :-- | :-- |
| Envelope (per zone) | `C_air, C_wall, C_int, C_boundary, R_out, R_in, R_int, R_boundary` (Stage 1) | — |
| Closed-loop substation | ECL310 PID `kp, Ti`; per-radiator `thermalMassHeatCapacity` (Stage 2) | radiator `Q_flow_nominal_sh` (→ UA) + fixed radiator-loop flow, from the energy balance |

Why the radiator UA + flow are *sized*, not AD-estimated:
- The radiator overall heat-transfer coefficient `UA` is the lever that sets the
  supply→return ΔT, but `SpaceHeaterTorchSystem` re-derives `UA` from the
  (non-differentiable) nominal rating inside `initialize()` and freezes it, so it
  is not reachable by the gradient estimator.
- `ValveTorchSystem(waterFlowRateMax=…)` is ignored by the dev-branch build (it
  resets to ~1 kg/s); flows must be written **post-build** onto the parameter
  tensor. The ECL310 primary valve is left high-authority on purpose (the PID only
  opens it a few percent).
- To keep the hydronic stage **consistent with the envelope** (which was calibrated
  with per-room heat input = total district-heat power / 4), the radiator rating +
  fixed flow are derived from the envelope's own heat demand and the measured
  operating point by `compute_winter_sizing()` / `apply_energy_balance_sizing()`,
  so total radiator output ≈ Q_meter and each radiator ≈ Q_meter/4 by construction.

The PID gains + radiator thermal mass ARE AD-estimated, against all six produced
signals (4 zones + supply + return), because in the closed loop they sit in the
signal path.

So the runbook is: **export data → Stage 1 envelope AD estimation → Stage 2
closed-loop AD estimation (sized radiators) → validate + energy-consistency check**.

## Current state (2026-06-28)
- **Model**: 4-room ring envelope + closed-loop ECL310 substation (the only
  hydronic model). Builds & simulates the cyclic loop (`make_fcn()`).
- **Data**: 17 per-signal CSVs exported for 2025-01-01 → 2026-02-28, including the
  measured supply set-point `ecl310_TSupSet_measured.csv` the closed loop tracks.
- **Stage 1 (envelope)**: pickle present at
  `aarhus_model/generated_files/models/skoven_envelope_estimation/result.pickle`.

---

## Step 1 — (re)export data (only if loaders/config/windows changed)
```bash
uv run python -m data_ingest.export_t4b_csvs --building skoven --start 2025-01-01 --end 2026-02-28
```
- Spans the Stage-1 envelope window (Jan 2025) and the closed-loop winter
  validation window (Jan 2026). Writes UTC CSVs incl. `ecl310_TSupSet_measured.csv`
  (BMS Fremløbstemp.ref — the real outdoor-reset set-point the loop tracks).

## Step 2 — Stage 1: envelope AD estimation
```bash
AARHUS_MAXITER=40 uv run python aarhus_model/only_envelope_param_est.py --building skoven
```
- Window: 2025-01-15 → 2025-01-22 (cold; ReMoni + measured heat boundary present;
  from `skoven.yaml: envelope_start/end`).
- Estimates the 8 RC parameters per zone (32 total) against the four
  `{zone}_indoor_temp_sensor` measurements, `method=("scipy","SLSQP","ad")`.
- Runtime ~3–5 min. Output pickle: `models/skoven_envelope_estimation/result.pickle`.
- `AARHUS_MAXITER` trades fit vs time (40 is a good default; raise for a tighter fit).

## Step 3 — Stage 2: closed-loop AD estimation (sized radiators)
```bash
AARHUS_MAXITER=60 uv run python aarhus_model/hydronic_param_est.py --building skoven
```
- Window: 2026-01-08 → 2026-01-15 (real winter heating regime, dT ≈ 14 °C; from
  `skoven.yaml: hydronic_start/end`).
- First derives the radiator sizing with `compute_winter_sizing()` (envelope heat
  demand + measured operating point → `Q_flow_nominal_sh`, fixed flow) and applies
  it; then AD-estimates ECL310 `kp, Ti` + per-radiator `thermalMassHeatCapacity`
  against the four zone temps + the produced supply + return water,
  `method=("scipy","SLSQP","ad")`.
- Output pickle: `models/skoven_hydronic_estimation/result.pickle`. The script
  prints the applied sizing and the estimated gains/thermal masses.
- Radiator UA + flow are sized (non-AD) — see the table above for why.

## Step 3b — Air loop (MVHR) — activation + sizing (no run, no AD fit)
The AHU (air-to-air heat recovery + per-room VAV dampers + idealized heating coil +
passive fan) lives in `ahu_fcn`. It is **activated by construction**, not estimated —
there is no AHU sensor data, and the idealized coil (`outletAirTemperature = setpoint`)
means the heat-recovery/coil parameters affect only the AHU ENERGY, never the zone
temperatures (the supply-air set-point is the sole zone lever). So nothing here is fit:
- **3-ACH ventilation (Danish reg.):** each `{zone}_supply_damper` `nominalAirFlowRate`
  is sized to the room's 3-ACH design mass flow (`zone_ventilation_flow`, from
  `floor_area·ROOM_HEIGHT_M·ρ_air`). In calibration mode a `vav_fixed_position` schedule
  holds every damper fully open (the dampers were previously inert — the position input
  was unconnected → 0 flow). In sim mode the per-room position is an RL action.
- **Thermally neutral supply (preserves Stage 1/2):** in calibration mode the coil
  set-point TRACKS the building mean room temperature (the AHU return-air temperature),
  so supply air ≈ room temp and the 3-ACH ventilation adds ~no net zone load (at 3 ACH
  the ventilation conductance is ~250 W/°C/room — a fixed set-point would be a large
  heating/cooling source). In sim mode the set-point is the RL schedule
  (`AHU_DEFAULTS["supply_air_setpoint_C"]`, default 21 °C).
- **Heat recovery + coil + fan stay at `AHU_DEFAULTS`** (eps≈0.7, idealized coil, fan
  polynomial). NB: the heat-recovery `primaryTemperatureOutSetpoint` is now wired to the
  supply-air target (it previously defaulted to 0 °C and clamped recovery to ~0).
- Tunables: `AHU_DEFAULTS["ventilation_ach"]` (3.0) and `["supply_air_setpoint_C"]`.

## Step 4 — Validate the closed loop + energy consistency (metrics + plots)
```bash
uv run python scripts/validation_plots.py
```
- Builds the calibrated closed-loop model (envelope pickle + energy-balance sizing
  + Stage-2 pickle), simulates the winter window (2026-01-08 → 01-15), discards the
  controller start-up transient, and prints a CV-RMSE/NMBE table for the four zone
  temps + the **produced** supply + return water.
- Also runs the **energy-consistency check** on the spring meter overlap
  (late-March): simulated Σ radiator `Power` vs metered `Q_meter`, and per-radiator
  mean `Power` vs `Q_meter/4` (the split the envelope assumed), reported as a
  closure ratio.
- Prints the **air-loop (MVHR) diagnostics**: per-room ventilation airflow (≈3 ACH),
  AHU supply/return air temps (≈equal ⇒ neutral), and AHU energy (coil + fan). These
  are reported, not scored (no AHU sensor data). The zone CV-RMSE here is with the air
  loop ACTIVE — compare to the frozen Stage-2 numbers to see the (small) drift.
- Writes overlays + a multi-panel scatter to `scripts/plots/skoven_closed_*`.
- `--no-plots` for metrics only; `--no-consistency` to skip the spring check;
  `--envelope-only` for Stage 1 alone.

Latest result (closed-loop AD fit + energy-balance sizing, **air loop ACTIVE at
3 ACH**, winter window, warmup-trimmed; regenerate with Step 4):

| Signal | CV-RMSE (%) | NMBE (%) |
| :-- | --: | --: |
| Room A Temp. | 2.39 | −0.45 |
| Room B Temp. | 2.45 | +1.42 |
| Room C Temp. | 2.87 | −2.55 |
| Room D Temp. | 5.80 | −4.53 |
| Supply Water (produced) | 26.9 | −19.5 |
| Return Water | 43.5 | −42.7 |

The previous (air-loop-inert) numbers were 3.15 / 2.26 / 3.57 / 4.60 %. Turning the
3-ACH ventilation on improves rooms A/B/C and worsens room D (≈+1.2 pp): a single
central coil supplies all four rooms one common air temperature (≈ the building mean),
so at 3 ACH the cross-zone mixing physically homogenises the rooms toward the mean and
pulls the outlier room (D) up. Supply/return water and the energy-consistency closure
are unchanged. This is the real effect of the air loop, not a regression.

Sizing (printed by Stage 2): Q ≈ 0.79 kW total (≈197 W/radiator), fixed flow
≈0.0033 kg/s/radiator (≈0.013 kg/s total), operating point T_sup 51 / T_ret 37 /
T_air 21 °C. Zones are held to ≈0.05 °C of the measured mean by construction; the
return runs cold (NMBE −43 %) — the zone↔return tension (see Known limitations).

## Acceptance criteria
- **Stage 1 (envelope)**: per-zone indoor-T RMSE < 1.0 °C on the held-out window.
- **Closed loop**: zone-T CV-RMSE within a few %, and the **produced** supply-water
  temperature tracks the measured signal as a model output (CV-RMSE ≈ 20 %), i.e.
  the controller is not given the measured supply, only the set-point.

## Known limitations
- **Zone↔return tension (cannot match both exactly).** With the envelope fixed and
  the radiator UA + flow non-AD, the zone-temperature and return-water targets pull
  the operating point in opposite directions. `compute_winter_sizing` sizes for the
  ZONES (the control-relevant signal), holding them to ≈0.05 °C of the measured
  mean; the return then runs cold (NMBE ≈ −43 %). Sizing for the return instead
  would warm the zones by ≈0.6 °C and match the return — a real, documented
  trade-off, not a tuning miss. Deeper fixes: make the radiator `UA` AD-calibratable
  in Twin4Build, or pin the loop flow from the Varme meter `Volumen [m³]`.
- **Energy consistency can't be cleanly verified (data gap).** Space heating is a
  winter signal, but the district-heat meter (ends 2025-05-31) only overlaps the BMS
  water (begins 2025-03-30) in a shoulder window where the meter is near its noise
  floor (~0.1 kW; ~0 from April). The Step-4 check (late-March overlap) gives a
  closure ratio of ~5 (model ≈0.62 kW vs meter ≈0.12 kW) — i.e. the winter-sized
  model over-delivers in mild weather, and the meter is too low there to anchor the
  check. There is NO window with both a metered heating signal and BMS water, so
  "radiator output ≈ Q_meter/4" can only be enforced by construction (the zone-match
  sizing ties radiator output to the envelope demand, itself calibrated against
  Q_meter/4 in Jan 2025), not validated directly. A future re-export pinning the
  loop flow from the meter `Volumen [m³]` in a winter meter window would close this.
- **PID gains hit their bounds** (kp→min, Ti→max): against the near-constant measured
  set-point the fit prefers minimal controller action, leaving the produced supply
  ~10 °C below set-point (NMBE −19 %). Acceptable for a baseline; revisit if the
  supply tracking matters for the RL reward.
- **Provisional mappings** to confirm against the floor plan: rooms ↔ ReMoni
  sensors viben8/9/10/11; Skoven heat meter = `5344947`.
- Space heating is a winter phenomenon; BMS water signals begin 2025-03-30, so the
  closed loop is validated on the Jan-2026 winter overlap (the meters end
  2025-05-31, so that window scores supply/return/zones, not metered energy).
