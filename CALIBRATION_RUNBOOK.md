# Calibration Runbook — Skoven 4-room model

Plan to run the calibration and analyze the results manually. Run everything from
`/home/sebscubs/repos/AarhusCase/AarhusCase` (so `uv run` uses the right project).

## Current state (2026-06-06)
- **Model**: 4-room ring envelope + simplified hydronic (one inlet-temperature
  control via mixing valve + PID; 4 fixed-opening radiator valves; no per-room
  control) + AHU (kept). Verified to build & simulate in all modes.
- **Data**: 16 per-signal CSVs exported for the 4-room layout (window 2025-01-01…05-31).
- **Stage 1 (envelope)**: DONE — pickle present. RMSE 0.28 / 0.33 / 0.27 / 0.86 °C.
- **Stage 2 (hydronic)**: NOT done — interrupted, no pickle yet.
- **Stage 3 (AHU)**: gracefully skips (no AHU sensor data → `AHU_DEFAULTS`).

## Step 1 — (re)export data (only if loaders/config changed)
```bash
uv run python -m data_ingest.export_t4b_csvs --building skoven --start 2025-01-01 --end 2025-05-31
```

## Step 2 — Stage 1 envelope (already done; re-run only if needed)
```bash
AARHUS_MAXITER=40 uv run python aarhus_model/only_envelope_param_est.py --building skoven
```
- Window: 2025-01-15 → 2025-01-22 (cold, ReMoni + meters present; from `skoven.yaml`).
- Runtime: ~3–5 min. Output: `models/skoven_envelope_estimation/result.pickle`.

## Step 3 — Stage 2 hydronic  ← next to run
```bash
AARHUS_MAXITER=12 uv run python aarhus_model/hydronic_param_est.py --building skoven
```
- Window: 2025-03-31 → 2025-04-07 (only BMS-water ∩ meter overlap; from `skoven.yaml`).
- Estimates per-room radiator `thermalMassHeatCapacity` + fixed-flow `waterFlowRateMax` (8 params).
- **Runtime is long** (~15–25 min): full closed-loop model, AD over ~1000 steps.
  Run it detached and let it finish; lower `AARHUS_MAXITER` (e.g. 8) to go faster,
  raise it for a tighter fit. Output: `models/skoven_hydronic_estimation/result.pickle`.

## Step 4 — Stage 3 AHU (will skip)
```bash
uv run python aarhus_model/only_ahu_param_est.py --building skoven
```
- No Skoven AHU instrumentation → prints "skipped", uses `AHU_DEFAULTS`. No pickle.

## Step 5 — RL wiring smoke (validation only; full training is separate)
```bash
uv run python use_case/skoven_RL_control.py --smoke
```
- Expect: `reset OK` (obs/action shapes) → 10 stepped rewards → `learn(32) OK`.

## Manual analysis of calibration results
A helper computes per-signal RMSE + radiator power for any window and (optionally)
writes overlay plots to `scripts/plots/`:
```bash
# Stage-2 hydronic window (zone temps + return water + radiator power)
uv run python scripts/analyze_calibration.py --start 2025-03-31 --end 2025-04-07 --plots

# Stage-1 envelope window (zone temps)
uv run python scripts/analyze_calibration.py --start 2025-01-15 --end 2025-01-22
```
It builds the full calibration model, loads whatever pickles exist, simulates, and
prints an `RMSE / sim_mean / obs_mean / n` table per measured signal. To check the
RL-ready (calibrated, simulation-mode) model holds comfort:
```bash
uv run python -c "import datetime,twin4build as tb; from dateutil.tz import gettz; \
from aarhus_model.skoven_model import load_model_and_params,ZONES; tz=gettz('Europe/Copenhagen'); \
m=load_model_and_params(calibration_mode=False); \
tb.Simulator(m).simulate(start_time=datetime.datetime(2025,2,1,tzinfo=tz),end_time=datetime.datetime(2025,2,2,tzinfo=tz),step_size=600); \
print('rooms:',[round(float(m.components[f'{z}_radiator'].output['Power'].get()),0) for z in ZONES])"
```

## Acceptance criteria (per PLAN.md)
- Stage 1: per-room indoor-T RMSE < 1.0 °C on a held-out slice. **(met: 0.27–0.86 °C)**
- Stage 2: per-room indoor-T + return-water RMSE; radiators deliver non-zero heat.
- Baseline replay: total heat within ~10 % of measured Varme over the window.

## Things to validate manually / known limitations
- **Return-water fit is the weak point** — the return sensor is a single-radiator
  proxy (not mass-weighted) and Stage 2 runs in a low-heat spring window (the only
  period where BMS water signals overlap the meters). Judge zone-temp fit first.
- **Provisional mappings** to confirm against the floor plan: rooms ↔ ReMoni
  sensors viben8/9/10/11; Skoven heat meter = `5344947`.
- Mixing-valve / `ecl310_pid` parameters are NOT estimated (not identifiable when
  the inlet temperature is replayed as a measured boundary) — left at priors.
- If Stage 2 zone-T RMSE is poor, check radiator `Q_flow_nominal_sh` priors in
  `RADIATOR_DEFAULTS` (plain floats, not estimated under AD) before widening bounds.
