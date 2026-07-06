"""Skoven building Twin4Build model.

Three-stage fcn() pattern (mirrors T4BGymUseCase/boptest_model/5_rooms_model.py):
  - envelope_fcn : zones + outdoor environment + occupancy + ReMoni sensors
  - hydronic_fcn : closed-loop ECL310 mixing-shunt substation → radiators
                   (PID modulates the primary mixing valve; the secondary supply
                   temperature is a PRODUCED output, return co-varies via the
                   recirculation feedback — the only hydronic model)
  - ahu_fcn      : AHU heat recovery + fan + heating coil + per-zone dampers
  - fcn          : master function calling all three

Zone topology (fallback — 3 zones):
  core    — central zone, adjacent to floor0 and floor1
  floor0  — ground-floor zone, exposed to outdoor
  floor1  — first-floor zone, exposed to outdoor

CALIBRATION vs SIMULATION mode (calibration_mode flag):
  - calibration_mode=True  : boundary inputs read from CSV (BMS Fremløb drives
    radiators open-loop). Used for Stage 1/2/3 parameter estimation.
  - calibration_mode=False : boundary inputs come from RL-controlled actuators
    (ECL310 supply-water setpoint drives radiators in closed loop). Used for
    forward simulation and RL training/eval.

Reference: T4BGymUseCase/boptest_model/rooms_and_ahu_model.py
"""
import math
import os
from typing import Callable, Optional

import twin4build as tb
import twin4build.utils.constants as t4b_constants
from twin4build.systems.building_space.building_space_thermal_torch_system import (
    BuildingSpaceThermalTorchSystem,
)
from twin4build.systems.space_heater.space_heater_torch_system import (
    SpaceHeaterTorchSystem,
)
from twin4build.systems.valve.valve_torch_system import ValveTorchSystem
from twin4build.systems.air_to_air_heat_recovery.air_to_air_heat_recovery_system import (
    AirToAirHeatRecoverySystem,
)
from twin4build.systems.coil.coil_torch_system import CoilTorchSystem
from twin4build.systems.fan.fan_torch_system import FanTorchSystem
from twin4build.systems.damper.damper_torch_system import DamperTorchSystem

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "generated_files", "data", "skoven")
MODELS_DIR = os.path.join(SCRIPT_DIR, "generated_files", "models")

# Deterministic locations the estimation scripts copy their result pickle to,
# so load_model_and_params() can find them regardless of the estimator's own
# datestamped output path.
ENVELOPE_RESULT_PICKLE = os.path.join(MODELS_DIR, "skoven_envelope_estimation", "result.pickle")
HYDRONIC_RESULT_PICKLE = os.path.join(MODELS_DIR, "skoven_hydronic_estimation", "result.pickle")
AHU_RESULT_PICKLE = os.path.join(MODELS_DIR, "skoven_ahu_estimation", "result.pickle")

# Radiator energy-balance sizing (compute_winter_sizing) is NOT stored in the
# estimation pickles — Q_flow_nominal_sh / T_*_nominal_sh / radiator flow are
# plain floats, not tps.Parameters. It must therefore be re-applied at every
# model build (load_model_and_params), or the radiators fall back to their
# oversized construction defaults (~12x too much heat, zones run away to ~35 C).
# The sizing is deterministic given the winter window, so it is computed once
# (a 9-iteration bisection) and cached to JSON keyed on that window.
HYDRONIC_SIZING_WINDOW = ("2026-01-08", "2026-01-15")  # winter heating regime (dT ~14 C)
SIZING_CACHE_PATH = os.path.join(MODELS_DIR, "skoven_hydronic_estimation", "radiator_sizing.json")

TZ = "Europe/Copenhagen"

# 1-floor building, 4 rooms in a ring. Each room is adjacent to two others and
# exposed to the outdoor. Rooms map 1:1 to ReMoni sensors viben8/9/10/11.
#
# Geometry from the Skoven floor plan ("Børnehus 01", floor level GK ~59.96 m).
# The real building has 22 named rooms (B1.S.01–B1.S.22) totalling 384.0 m² net
# floor area. We keep the 4 main thermal zones and aggregate the rooms into them
# by quadrant, each anchored by one of the four large activity rooms; the
# NW/NE/SE/SW layout matches the adjacency ring below. floor_area_m2 is the
# summed net area of a zone's constituent rooms; zone air volume = area *
# ROOM_HEIGHT_M, and C_air = AIR_VOL_HEAT_CAP * volume (see envelope_fcn).
# The room→zone allocation (esp. the central kitchen/garderoberum) is a modelling
# choice and can be re-balanced without changing the topology:
#   room_a (NW): B1.S.16 Grupperum + .17 Toiletrum + .18 Vindfang + .19 Krybberum
#   room_b (NE): B1.S.07 Grupperum + .08 Toiletrum + .21 Familiestue + .09 Garderobe
#   room_c (SE): B1.S.01 Værksted/flex + .02 + .03 + .04 + .05 + .06 + .20 + .22
#   room_d (SW): B1.S.13 Grupperum + .10 Pæd.køkken + .11 Garderobe + .12 + .14 + .15
ROOM_HEIGHT_M = 3.0            # assumed clear room height [m] (no section drawing)
AIR_VOL_HEAT_CAP = 1.2 * 1005  # volumetric heat capacity of air ≈ 1206 J/(m³·K)

ZONES = {
    "room_a": {
        "adjacent_to": ["room_b", "room_d"],
        "outdoor_exposed": True,
        "nominal_flow_kgs": 0.05,
        "floor_area_m2": 83.3,
    },
    "room_b": {
        "adjacent_to": ["room_c", "room_a"],
        "outdoor_exposed": True,
        "nominal_flow_kgs": 0.05,
        "floor_area_m2": 101.9,
    },
    "room_c": {
        "adjacent_to": ["room_d", "room_b"],
        "outdoor_exposed": True,
        "nominal_flow_kgs": 0.05,
        "floor_area_m2": 63.0,
    },
    "room_d": {
        "adjacent_to": ["room_a", "room_c"],
        "outdoor_exposed": True,
        "nominal_flow_kgs": 0.05,
        "floor_area_m2": 135.8,
    },
}


RHO_AIR = 1.2  # air density [kg/m³]


def zone_air_capacitance(zone_id: str) -> float:
    """Volumetric air heat capacity C_air [J/K] for a zone, from its floor area
    and the assumed room height: C_air = rho*cp * (floor_area * height)."""
    return AIR_VOL_HEAT_CAP * ZONES[zone_id]["floor_area_m2"] * ROOM_HEIGHT_M


def zone_ventilation_flow(zone_id: str, ach: float) -> float:
    """Design ventilation mass flow [kg/s] for a zone at `ach` air-changes/hour:
    ach * (floor_area * ROOM_HEIGHT_M) * RHO_AIR / 3600. Used to size the room's
    VAV damper nominalAirFlowRate (the fully-open flow)."""
    volume_m3 = ZONES[zone_id]["floor_area_m2"] * ROOM_HEIGHT_M
    return ach * volume_m3 * RHO_AIR / 3600.0


def damper_position_for_flow(target_flow: float, nominal_flow: float, a: float) -> float:
    """Invert the DamperTorchSystem exponential characteristic
    m = a·exp(b·u) − a  (b = log((nominal+a)/a), so m=nominal at u=1, m=0 at u=0)
    to find the damper position u∈[0,1] that delivers `target_flow`. Used to
    position an OVERSIZED damper at its 3-ACH design flow for the baseline."""
    b = math.log((nominal_flow + a) / a)
    return math.log((target_flow + a) / a) / b

# Per-radiator nominal rating PRIORS. These set the (non-AD-calibratable) UA via
# the SpaceHeaterTorchSystem nominal-output solve. They are only the build-time
# priors: Stage 2 overrides them per-room via compute_winter_sizing() +
# apply_energy_balance_sizing(), which pin the rating to the measured operating
# point (T_a/T_b/TAir/Q_flow_nominal) so the radiator output stays consistent with
# the Stage-1 envelope (which used per-room heat input = district-heat power / 4).
# See hydronic_param_est.py for why UA is SIZED rather than estimated.
RADIATOR_DEFAULTS = {
    "Q_flow_nominal_sh": 800.0,
    "T_a_nominal_sh": 55.0,
    "T_b_nominal_sh": 45.0,
    "TAir_nominal_sh": 21.0,
    "thermalMassHeatCapacity": 5000.0,
    "nelements": 3,
}

# Closed-loop ECL310 shunt parameters. The substation is modelled as a mixing
# shunt: a (roughly constant) hot district-heating primary is blended with
# recirculated building return water to produce the secondary supply temperature,
# the mixing ratio being set by the ECL310 PID tracking the outdoor-reset curve.
# These replace the open-loop "replay the measured supply temperature" boundary.
# Sizing is mass-consistent and matches the real building's low-flow / high-dT
# operation (measured supply~51, return~37, dT~14 -> loop flow ~0.05-0.08 kg/s).
# The primary valve is scaled to the circulation flow (primary_max ~ recirc), so
# the ECL310 PID has authority to regulate the secondary supply instead of the
# oversized valve pinning it near the primary temperature. recirc_kgs is kept
# consistent with the four radiator valves (4 x 0.02 = 0.08 kg/s).
SHUNT_DEFAULTS = {
    "T_primary_C": 70.0,        # district-heating primary supply temp [°C] (boundary)
    "primary_max_kgs": 0.08,    # ECL310 primary valve design flow [kg/s]
    "recirc_kgs": 0.08,         # recirculation (bypass) flow [kg/s] = 4 * rad flow
    "pid_kp": 0.05,             # ECL310 PID proportional gain
    "pid_Ti": 1800.0,           # ECL310 PID integral time [s]
}

# Ventilation: Danish-regulation 3 air-changes/hour (MVHR). Each VAV damper is
# sized so that fully open it delivers its room's 3-ACH design mass flow; the AHU
# heats the heat-recovered outdoor air to the supply set-point. fan_nominal_flow
# and the heat-recovery flow maxima are sized to the WHOLE-building 3-ACH flow
# (≈1.15 kg/s for 384 m² × 3 m) so the fan/HR operate in-range; only the AHU
# ENERGY depends on these (the zone effect is set by the supply-air set-point).
AHU_DEFAULTS = {
    "ventilation_ach": 3.0,        # air-changes/hour (Danish reg.) per room
    "supply_air_setpoint_C": 21.0, # sim-mode (RL) supply-air set-point default;
                                   # calibration mode tracks the room temp (neutral)
    # Economizer / free cooling (sim mode only, applied live by the gym's
    # EconomizerGymSimulator). When the building runs above setpoint+deadband and
    # the outdoor air is cooler, the AHU supplies cool outdoor air (heat-recovery
    # bypass) down to a floor, shedding solar/internal gains that the radiators
    # can't remove. Without it the well-insulated rooms free-float to ~26 C in
    # spring (irradiation 6x winter, radiators already at minimum).
    "economizer_enabled": True,
    "economizer_deadband_C": 0.5,      # engage cooling above setpoint + this
    "economizer_gain": 3.0,            # °C supply drop per °C mean-room overshoot
                                       # (proportional, so a small overshoot gives
                                       # gentle cooling instead of full-floor blast)
    "cooling_min_supply_air_C": 18.0,  # supply-air floor (draft/condensation +
                                       # limits overcooling of the coolest room)
    "fan_nominal_flow_kgs": 1.5,
    "fan_nominal_power_W": 800.0,
    "fan_c1": 0.027828,
    "fan_c2": 0.026583,
    "fan_c3": -0.087069,
    "fan_c4": 1.030920,
    "fan_f_total": 0.8,
    "hr_eps_75_h": 0.75,
    "hr_eps_100_h": 0.70,
    "hr_eps_75_c": 0.65,
    "hr_eps_100_c": 0.60,
    "hr_primary_max_kgs": 1.5,
    "hr_secondary_max_kgs": 1.5,
    "damper_a": 0.5,
    # Per-room VAV under RL control (sim mode): the dampers are OVERSIZED by this
    # factor so that fully open (damperPosition=1) delivers vav_oversize_factor ×
    # the room's 3-ACH design flow — giving the RL agent real authority to BOOST
    # warm supply air (21 °C) into an under-heated room (room_a) above the
    # baseline, not just throttle below it. The baseline / non-RL fallback keeps
    # exactly 3-ACH via per-room `{zone}_damper_position` schedules positioned at
    # the (oversize-corrected) u that reproduces the design flow, so baseline AHU
    # energy and ventilation are unchanged; only the RL agent's boost costs extra
    # fan/coil energy (which the reward penalises when the room is comfortable).
    "vav_oversize_factor": 2.0,
}


def _csv(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------
def envelope_fcn(self, *, calibration_mode: bool = True,
                 inject_heat_boundary: bool = False) -> None:
    """Building zones + outdoor environment + occupancy + indoor sensors.

    inject_heat_boundary: Stage-1 (envelope-only) calibration has no radiator to
    supply heat, so a measured per-zone heat-input CSV ({zone}_heat_input.csv) is
    connected to zone.heatGain. Must be False in the full model, where the
    radiator's Power already drives heatGain (a port can take only one source).
    """

    # OutdoorEnvironmentSystem is a leaf node fed by CSV — no upstream connections.
    outdoor = tb.OutdoorEnvironmentSystem(
        id="outdoor_environment",
        filename_outdoorTemperature=_csv("outdoor_temperature.csv"),
        filename_globalIrradiation=_csv("global_irradiation.csv"),
        filename_outdoorCo2Concentration=_csv("outdoor_co2.csv"),
    )

    global_occupancy = tb.ScheduleSystem(
        id="global_occupancy_schedule",
        weekDayRulesetDict={
            "ruleset_default_value": 0,
            "ruleset_start_minute": [0, 0, 0],
            "ruleset_end_minute": [0, 0, 0],
            "ruleset_start_hour": [0, 8, 17],
            "ruleset_end_hour": [8, 17, 24],
            "ruleset_value": [0, 5, 0],
        },
        
    )

    for zone_id, _cfg in ZONES.items():
        # C_air scales with the zone's air volume (floor area * ROOM_HEIGHT_M);
        # the other capacitances/resistances stay as structural priors and are
        # refined by Stage-1 calibration.
        zone = BuildingSpaceThermalTorchSystem(
            id=zone_id,
            C_air=zone_air_capacitance(zone_id),
            C_wall=5e6,
            C_int=2e5,
            C_boundary=1e6,
            R_out=0.01,
            R_in=0.005,
            R_int=0.02,
            R_boundary=0.01,
            f_wall=0.3,
            f_air=0.1,
            Q_occ_gain=80.0,
            
        )

        self.add_connection(outdoor, zone, "outdoorTemperature", "outdoorTemperature")
        self.add_connection(outdoor, zone, "globalIrradiation", "globalIrradiation")
        self.add_connection(global_occupancy, zone, "scheduleValue", "numberOfPeople")

        # Stage-1 heat boundary: measured per-zone heat input drives heatGain.
        if inject_heat_boundary:
            heat_src = tb.SensorSystem(
                id=f"{zone_id}_heat_input",
                filename=_csv(f"{zone_id}_heat_input.csv"),
            )
            self.add_connection(heat_src, zone, "measuredValue", "heatGain")

        # Indoor temperature sensor. In calibration mode it carries the observed
        # ReMoni CSV (filename) so the Estimator can compare simulated (the
        # incoming connection) vs measured. In simulation mode it is a virtual
        # passthrough the gym reads.
        temp_sensor = tb.SensorSystem(
            id=f"{zone_id}_indoor_temp_sensor",
            filename=_csv(f"{zone_id}_indoor_temperature.csv") if calibration_mode else None,
        )
        self.add_connection(zone, temp_sensor, "indoorTemperature", "measuredValue")

    # Adjacency: scalar output → Vector input requires input_port_index.
    for zone_id, cfg in ZONES.items():
        zone = self.components[zone_id]
        for slot, adj_id in enumerate(cfg["adjacent_to"]):
            adj_zone = self.components[adj_id]
            self.add_connection(
                adj_zone, zone, "indoorTemperature", "adjacentZoneTemperature",
                input_port_index=slot,
            )


# ---------------------------------------------------------------------------
# Hydronic — closed-loop mixing-shunt substation (the ONLY hydronic model)
# ---------------------------------------------------------------------------
def hydronic_fcn(self, *, calibration_mode: bool = True) -> None:
    """Closed-loop ECL310 substation: the secondary supply temperature is an
    OUTPUT of the model, not a replayed boundary.

    Structural fix for the open-loop over-determination. A mixing shunt blends a
    (constant) hot district-heating primary with recirculated building return
    water; the blend ratio is the ECL310 mixing-valve position, driven by a PID
    that tracks the outdoor-reset heating curve. The recirculation feedback
    (return -> supply) couples supply and return so they co-vary with the radiator
    heat extraction, instead of the supply being pinned to a measured constant.

        supply mix : T_sup = (m_p*T_primary + m_r*T_ret) / (m_p + m_r)
        m_p        : primary flow = ecl310_valve(position = PID output)
        m_r        : recirculation (bypass) flow [constant]
        PID        : setpoint = heating curve, feedback = T_sup (the mix output)

    The supply/return cycle mirrors the AHU's existing closed heat-recovery loop,
    which already simulates in this model, so no new solver machinery is needed.

      calibration_mode=True : the heating-curve setpoint is replayed from the BMS
        curve CSV and the produced supply/return are scored against the measured
        BMS Fremloeb/Retur (both are model OUTPUTS now).
      calibration_mode=False: the curve setpoint is an RL-controllable schedule,
        so the agent's set-point action propagates valve -> mix -> supply ->
        radiators -> return -> energy through real loop physics.
    """
    sd = SHUNT_DEFAULTS

    # ----- District-heating primary boundary (hot side of the shunt) ---------
    primary_temp = tb.ScheduleSystem(
        id="dh_primary_temp",
        weekDayRulesetDict={
            "ruleset_default_value": sd["T_primary_C"],
            "ruleset_start_minute": [0], "ruleset_end_minute": [0],
            "ruleset_start_hour": [0], "ruleset_end_hour": [24],
            "ruleset_value": [sd["T_primary_C"]],
        },
    )
    recirc_flow = tb.ScheduleSystem(
        id="ecl310_recirc_flow",
        weekDayRulesetDict={
            "ruleset_default_value": sd["recirc_kgs"],
            "ruleset_start_minute": [0], "ruleset_end_minute": [0],
            "ruleset_start_hour": [0], "ruleset_end_hour": [24],
            "ruleset_value": [sd["recirc_kgs"]],
        },
    )

    # ----- ECL310 supply-temperature setpoint (target tracked by the PID) ----
    # Prefer the MEASURED BMS supply setpoint (Fremloebstemp.ref); the synthetic
    # outdoor-reset curve over-predicts it (~64 vs ~51 degC) and would saturate
    # the valve. Fall back to the synthetic curve only if the measured file is
    # missing.
    if calibration_mode:
        _measured_set = _csv("ecl310_TSupSet_measured.csv")
        ecl310_setpoint = tb.SensorSystem(
            id="ecl310_TSupSet_schedule",
            filename=_measured_set if os.path.exists(_measured_set)
            else _csv("ecl310_TSupSet_curve.csv"),
        )
        setpoint_port = "measuredValue"
    else:
        ecl310_setpoint = tb.ScheduleSystem(
            id="ecl310_TSupSet_schedule",
            weekDayRulesetDict={
                "ruleset_default_value": 50.0,
                "ruleset_start_minute": [0], "ruleset_end_minute": [0],
                "ruleset_start_hour": [0], "ruleset_end_hour": [24],
                "ruleset_value": [50.0],
            },
        )
        setpoint_port = "scheduleValue"

    # ----- Per-room indoor-temperature setpoints (sim / RL only) -------------
    # Supervisory RL levers: the agent emits one indoor-temp setpoint per room.
    # The shared radiator loop is driven by the MEAN of these via the outdoor-
    # reset heating curve — computed live by SupervisorySetpointGymSimulator,
    # which overrides ecl310_TSupSet_schedule (the PID/valve below are untouched).
    # The per-room values also steer each room's VAV damper (per-room air trim).
    # Each schedule is connected to a read-back sensor so the source stays in the
    # connected graph and is available as an observation.
    if not calibration_mode:
        for zone_id in ZONES:
            indoor_setpoint = tb.ScheduleSystem(
                id=f"{zone_id}_indoor_temp_setpoint",
                weekDayRulesetDict={
                    "ruleset_default_value": 21.0,
                    "ruleset_start_minute": [0], "ruleset_end_minute": [0],
                    "ruleset_start_hour": [0], "ruleset_end_hour": [24],
                    "ruleset_value": [21.0],
                },
            )
            indoor_setpoint_readback = tb.SensorSystem(
                id=f"{zone_id}_indoor_temp_setpoint_readback"
            )
            self.add_connection(
                indoor_setpoint, indoor_setpoint_readback,
                "scheduleValue", "measuredValue",
            )

    # ----- ECL310 PID + primary mixing valve ---------------------------------
    # Reverse-acting: open the primary valve when the supply is BELOW the curve
    # setpoint (too cold). Direct action would close the valve when cold.
    ecl310_pid = tb.PIDControllerSystem(
        id="ecl310_pid", kp=sd["pid_kp"], Ti=sd["pid_Ti"], Td=0.0, isReverse=True,
    )
    ecl310_valve = ValveTorchSystem(
        id="ecl310_kr1_valve",
        waterFlowRateMax=sd["primary_max_kgs"],
        valveAuthority=1.0,
    )
    self.add_connection(ecl310_pid, ecl310_valve, "inputSignal", "valvePosition")

    # ----- Supply mixing shunt (flow-weighted blend of primary + recirc) -----
    # ReturnFlowJunctionSystem is reused as a generic flow-weighted temperature
    # mixer: slot 0 = hot primary, slot 1 = recirculated return.
    supply_mix = tb.ReturnFlowJunctionSystem(id="ecl310_supply_mix")
    self.add_connection(primary_temp, supply_mix, "scheduleValue", "airTemperatureIn", input_port_index=0)
    self.add_connection(ecl310_valve, supply_mix, "waterFlowRate", "airFlowRateIn", input_port_index=0)
    # slot 1 (recirc temperature) is wired after the return junction exists below.
    self.add_connection(recirc_flow, supply_mix, "scheduleValue", "airFlowRateIn", input_port_index=1)

    # Produced secondary supply temperature: PID feedback + scored sensor.
    self.add_connection(ecl310_setpoint, ecl310_pid, setpoint_port, "setpointValue")
    self.add_connection(supply_mix, ecl310_pid, "airTemperatureOut", "actualValue")

    ecl310_sup_sensor = tb.SensorSystem(
        id="ecl310_TSupHea_y",
        filename=_csv("ecl310_TSupHea_y_processed.csv") if calibration_mode else None,
    )
    self.add_connection(supply_mix, ecl310_sup_sensor, "airTemperatureOut", "measuredValue")

    # ----- Fixed-opening radiator valves + radiators -------------------------
    radiator_fixed_position = tb.ScheduleSystem(
        id="radiator_fixed_position",
        weekDayRulesetDict={
            "ruleset_default_value": 1.0,
            "ruleset_start_minute": [0], "ruleset_end_minute": [0],
            "ruleset_start_hour": [0], "ruleset_end_hour": [24],
            "ruleset_value": [1.0],
        },
    )
    return_junction = tb.ReturnFlowJunctionSystem(id="ecl310_return_junction")
    for slot, zone_id in enumerate(ZONES):
        zone = self.components[zone_id]
        # Construction max sets the (frozen) normalisation ceiling used by
        # _set_radiator_flows(...).set(normalized=False); 0.1 kg/s gives headroom
        # above the energy-balance-sized fixed flow (~0.005-0.02 kg/s).
        rad_valve = ValveTorchSystem(
            id=f"{zone_id}_radiator_valve", waterFlowRateMax=0.1, valveAuthority=1.0,
        )
        self.add_connection(radiator_fixed_position, rad_valve, "scheduleValue", "valvePosition")

        radiator = SpaceHeaterTorchSystem(id=f"{zone_id}_radiator", **RADIATOR_DEFAULTS)
        # Supply temperature now comes from the mixing shunt (the closed loop).
        self.add_connection(supply_mix, radiator, "airTemperatureOut", "supplyWaterTemperature")
        self.add_connection(rad_valve, radiator, "waterFlowRate", "waterFlowRate")
        self.add_connection(zone, radiator, "indoorTemperature", "indoorTemperature")
        self.add_connection(radiator, zone, "Power", "heatGain")

        # Radiator returns feed the return manifold (flow-weighted mix).
        self.add_connection(radiator, return_junction, "outletWaterTemperature",
                            "airTemperatureIn", input_port_index=slot)
        self.add_connection(rad_valve, return_junction, "waterFlowRate",
                            "airFlowRateIn", input_port_index=slot)

    # Close the loop: mixed return feeds the recirc branch of the supply shunt.
    self.add_connection(return_junction, supply_mix, "airTemperatureOut",
                        "airTemperatureIn", input_port_index=1)

    # Return-water sensor (scored against measured BMS Retur).
    ecl310_ret_sensor = tb.SensorSystem(
        id="ecl310_TRetHea_y",
        filename=_csv("ecl310_TRetHea_y_processed.csv") if calibration_mode else None,
    )
    self.add_connection(return_junction, ecl310_ret_sensor, "airTemperatureOut", "measuredValue")

    if calibration_mode:
        tb.SensorSystem(
            id="varme_meter_power_sensor",
            filename=_csv("varme_meter_power_kW.csv"),
        )


# ---------------------------------------------------------------------------
# AHU (air-side — Junction components OK here)
# ---------------------------------------------------------------------------
def ahu_fcn(self, *, calibration_mode: bool = True) -> None:
    """AHU: heat recovery + fan + heating coil + per-zone dampers."""
    outdoor = self.components["outdoor_environment"]

    heat_recovery = AirToAirHeatRecoverySystem(
        id="heat_recovery",
        eps_75_h=AHU_DEFAULTS["hr_eps_75_h"],
        eps_100_h=AHU_DEFAULTS["hr_eps_100_h"],
        eps_75_c=AHU_DEFAULTS["hr_eps_75_c"],
        eps_100_c=AHU_DEFAULTS["hr_eps_100_c"],
        primaryAirFlowRateMax=AHU_DEFAULTS["hr_primary_max_kgs"],
        secondaryAirFlowRateMax=AHU_DEFAULTS["hr_secondary_max_kgs"],
        
    )
    self.add_connection(outdoor, heat_recovery, "outdoorTemperature", "primaryTemperatureIn")

    supply_fan = FanTorchSystem(
        id="supply_fan",
        nominalAirFlowRate=AHU_DEFAULTS["fan_nominal_flow_kgs"],
        nominalPowerRate=AHU_DEFAULTS["fan_nominal_power_W"],
        c1=AHU_DEFAULTS["fan_c1"],
        c2=AHU_DEFAULTS["fan_c2"],
        c3=AHU_DEFAULTS["fan_c3"],
        c4=AHU_DEFAULTS["fan_c4"],
        f_total=AHU_DEFAULTS["fan_f_total"],
        
    )

    fan_power_sensor = tb.SensorSystem(id="vent_power_sensor")
    self.add_connection(supply_fan, fan_power_sensor, "Power", "measuredValue")

    # CoilTorchSystem takes only **kwargs — heats inlet air to setpoint
    supply_heating_coil = CoilTorchSystem(
        id="supply_heating_coil",
        
    )

    # Coil supply-air set-point.
    #  - sim mode (RL): an RL-controllable schedule (the agent's AHU set-point lever).
    #  - calibration mode: the set-point TRACKS the building mean room temperature
    #    (the AHU return-air temperature, wired after the return junction is built
    #    below), so ventilation is thermally ~neutral and the envelope+hydronic
    #    calibration is preserved. At 3 ACH the ventilation conductance is large
    #    (~250 W/°C per room), so a fixed set-point would be a big heating/cooling
    #    source whenever a room deviates from it.
    if not calibration_mode:
        supply_air_setpoint = tb.ScheduleSystem(
            id="supply_air_temp_setpoint_sensor",
            weekDayRulesetDict={
                "ruleset_default_value": AHU_DEFAULTS["supply_air_setpoint_C"],
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [24],
                "ruleset_value": [AHU_DEFAULTS["supply_air_setpoint_C"]],
            },
        )
        self.add_connection(
            supply_air_setpoint, supply_heating_coil, "scheduleValue", "outletAirTemperatureSetpoint"
        )
        # The heat recovery must share the supply-air target, otherwise its
        # primaryTemperatureOutSetpoint defaults to 0 °C and it clamps its
        # recovered-air output to 0 °C (no recovery → the coil over-heats).
        self.add_connection(
            supply_air_setpoint, heat_recovery, "scheduleValue", "primaryTemperatureOutSetpoint"
        )

    supply_air_temp_sensor = tb.SensorSystem(
        id="vent_supply_air_temp_sensor"
    )
    self.add_connection(
        supply_heating_coil, supply_air_temp_sensor, "outletAirTemperature", "measuredValue"
    )

    # Air-side junctions — correct usage here
    supply_junction = tb.SupplyFlowJunctionSystem(
        id="ahu_supply_junction"
    )
    return_junction = tb.ReturnFlowJunctionSystem(
        id="ahu_return_junction"
    )

    # Heat-recovery → coil inlet via fan
    self.add_connection(
        heat_recovery, supply_fan, "primaryTemperatureOut", "inletAirTemperature"
    )
    self.add_connection(
        supply_fan, supply_heating_coil, "outletAirTemperature", "inletAirTemperature"
    )

    # Per-room VAV dampers under (optional) RL control. Each damper is OVERSIZED
    # by vav_oversize_factor so fully open delivers that multiple of the room's
    # 3-ACH design flow — headroom for the agent to BOOST warm supply air into an
    # under-heated room above baseline. Each damper gets its OWN position schedule
    # (`{zone}_damper_position`) positioned at the u that reproduces the 3-ACH
    # design flow on the oversized characteristic, so the rule-based baseline /
    # non-RL fallback delivers exactly 3-ACH (unchanged AHU energy & ventilation).
    # The schedules are wired UNCONDITIONALLY so the air loop always has a sane
    # default (an unconnected damper input defaults to 0 flow, silently starving
    # ventilation). The gym's per-step input override (`_do_component_timestep`
    # step 2 in t4b_gym_env.py) runs AFTER this connection is assigned, so an RL
    # action on `{zone}_supply_damper.damperPosition` still wins whenever the
    # policy config lists it; these schedules are the fallback otherwise.
    ach = AHU_DEFAULTS["ventilation_ach"]
    oversize = AHU_DEFAULTS["vav_oversize_factor"]
    damper_a = AHU_DEFAULTS["damper_a"]
    for slot, (zone_id, cfg) in enumerate(ZONES.items()):
        design_flow = zone_ventilation_flow(zone_id, ach)
        nominal_flow = oversize * design_flow
        zone_damper = DamperTorchSystem(
            id=f"{zone_id}_supply_damper",
            nominalAirFlowRate=nominal_flow,
            a=damper_a,
        )
        zone = self.components[zone_id]

        base_pos = damper_position_for_flow(design_flow, nominal_flow, damper_a)
        zone_damper_position = tb.ScheduleSystem(
            id=f"{zone_id}_damper_position",
            weekDayRulesetDict={
                "ruleset_default_value": base_pos,
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [24],
                "ruleset_value": [base_pos],
            },
        )
        self.add_connection(
            zone_damper_position, zone_damper, "scheduleValue", "damperPosition"
        )

        # Damper aggregates total fan flow demand (Vector input)
        self.add_connection(
            zone_damper, supply_junction, "airFlowRate", "airFlowRateOut",
            input_port_index=slot,
        )

        self.add_connection(zone_damper, zone, "airFlowRate", "supplyAirFlowRate")
        self.add_connection(zone_damper, zone, "airFlowRate", "exhaustAirFlowRate")
        self.add_connection(
            supply_heating_coil, zone, "outletAirTemperature", "supplyAirTemperature"
        )

        # Return air aggregation (Vector inputs)
        self.add_connection(
            zone, return_junction, "indoorTemperature", "airTemperatureIn",
            input_port_index=slot,
        )
        self.add_connection(
            zone_damper, return_junction, "airFlowRate", "airFlowRateIn",
            input_port_index=slot,
        )

    # Total supply flow drives fan/coil/HR primary side
    self.add_connection(supply_junction, supply_fan, "airFlowRateIn", "airFlowRate")
    self.add_connection(supply_junction, supply_heating_coil, "airFlowRateIn", "airFlowRate")
    self.add_connection(
        supply_junction, heat_recovery, "airFlowRateIn", "primaryAirFlowRate"
    )

    ahu_return_air_temp_sensor = tb.SensorSystem(
        id="vent_return_air_temp_sensor"
    )
    self.add_connection(
        return_junction, ahu_return_air_temp_sensor, "airTemperatureOut", "measuredValue"
    )
    # Calibration mode: coil set-point tracks the building mean room temperature
    # (the return-air temperature) so the supply air ≈ room temp → ventilation is
    # thermally neutral and the envelope+hydronic calibration is preserved.
    if calibration_mode:
        self.add_connection(
            ahu_return_air_temp_sensor, supply_heating_coil,
            "measuredValue", "outletAirTemperatureSetpoint",
        )
        # Heat recovery shares the same supply-air target (else it clamps its
        # recovered-air output to its default 0 °C set-point → no recovery).
        self.add_connection(
            ahu_return_air_temp_sensor, heat_recovery,
            "measuredValue", "primaryTemperatureOutSetpoint",
        )
    self.add_connection(
        return_junction, heat_recovery, "airTemperatureOut", "secondaryTemperatureIn"
    )
    self.add_connection(
        return_junction, heat_recovery, "airFlowRateOut", "secondaryAirFlowRate"
    )


# ---------------------------------------------------------------------------
# Master fcn / model factories
# ---------------------------------------------------------------------------
def make_fcn(calibration_mode: bool = True) -> Callable:
    """Master fcn factory.

    The hydronic system is the closed-loop mixing-shunt substation: the secondary
    supply temperature is produced by the ECL310 PID + mixing valve and the
    recirculation feedback, so supply and return co-vary with the radiator load
    instead of the supply being pinned to a replayed boundary. This is the single
    hydronic model (the open-loop replay path was removed).
    """
    def _fcn(self) -> None:
        # inject_heat_boundary=False: radiator Power drives heatGain in the full model.
        envelope_fcn(self, calibration_mode=calibration_mode, inject_heat_boundary=False)
        hydronic_fcn(self, calibration_mode=calibration_mode)
        ahu_fcn(self, calibration_mode=calibration_mode)
    return _fcn


def make_envelope_fcn(calibration_mode: bool = True) -> Callable:
    """Stage-1 envelope-only model with the measured heat-input boundary."""
    def _fcn(self) -> None:
        envelope_fcn(self, calibration_mode=calibration_mode, inject_heat_boundary=True)
    return _fcn


def _set_radiator_flows(model, rad_flow: float) -> None:
    """Set the four fixed radiator-valve design flows [kg/s] post-build.

    tps.Parameter stores its data NORMALISED to [0, 1] against the min/max frozen
    at construction, so the physical value must be written through .set(...,
    normalized=False) (writing .data directly sets the normalised slot — e.g.
    .data.fill_(0.02) yields get()==0.02*0.02==0.0004 kg/s, which silently starves
    the radiators). The ECL310 primary mixing valve is deliberately LEFT at its
    high default max so the PID keeps fine authority over the supply blend.
    """
    for zone_id in ZONES:
        model.components[f"{zone_id}_radiator_valve"].waterFlowRateMax.set(
            float(rad_flow), normalized=False
        )


def _read_window_mean(csv_name: str, start, end) -> float:
    """Mean of a generated data CSV's 'value' column over the [start, end] window.

    CSVs are written in UTC; start/end are converted to UTC for filtering.
    """
    import pandas as pd

    s = pd.read_csv(os.path.join(DATA_DIR, csv_name), index_col=0)["value"]
    s.index = pd.to_datetime(s.index, utc=True)
    lo = pd.Timestamp(start).tz_convert("UTC")
    hi = pd.Timestamp(end).tz_convert("UTC")
    return float(s[(s.index >= lo) & (s.index <= hi)].mean())


def _apply_sizing(model, *, Q_per_rad, T_sup, T_ret, T_air, m_per_rad) -> None:
    """Pin each radiator's nominal rating to the operating point and set the fixed
    radiator-loop flow. Must run BEFORE first initialize() (SpaceHeaterTorchSystem
    solves and freezes UA from the nominal rating on first init)."""
    for zone_id in ZONES:
        rad = model.components[f"{zone_id}_radiator"]
        rad.Q_flow_nominal_sh = float(Q_per_rad)
        rad.T_a_nominal_sh = float(T_sup)
        rad.T_b_nominal_sh = float(T_ret)
        rad.TAir_nominal_sh = float(T_air)
    _set_radiator_flows(model, float(m_per_rad))


def _sim_mean_zone_and_power(Q_nominal_total, m_per_rad, *, start, end, step,
                             warmup_steps, T_sup, T_ret, T_air, envelope_pickle):
    """Build + size + simulate the closed loop for a candidate nominal rating and
    fixed flow; return (mean zone temperature across rooms, mean total radiator
    Power). If m_per_rad is None the flow is derived from the energy balance
    m = (Q_nominal/4) / (cp * dT_measured).

    A fresh model is built each call because UA is frozen on first initialize(),
    so the nominal rating cannot be changed in place.
    """
    cp = float(t4b_constants.CP_WATER)
    n = len(ZONES)
    Q_per_rad = Q_nominal_total / n
    if m_per_rad is None:
        m_per_rad = Q_per_rad / (cp * max(T_sup - T_ret, 1e-3))

    model = get_model(id="skoven_sizing_probe",
                      fcn_=make_fcn(calibration_mode=True), calibration_mode=True)
    if os.path.exists(envelope_pickle):
        model.load_estimation_result(envelope_pickle)
    _apply_sizing(model, Q_per_rad=Q_per_rad, T_sup=T_sup, T_ret=T_ret,
                  T_air=T_air, m_per_rad=m_per_rad)
    tb.Simulator(model).simulate(start_time=start, end_time=end, step_size=step)

    zone_sum = 0.0
    power_sum = None
    for z in ZONES:
        zt = model.components[f"{z}_indoor_temp_sensor"].output["measuredValue"]
        zt = zt.history(i_s=0, i_c=0).detach().cpu().numpy().reshape(-1)[warmup_steps:]
        zone_sum += float(zt.mean())
        p = model.components[f"{z}_radiator"].output["Power"]
        p = p.history(i_s=0, i_c=0).detach().cpu().numpy().reshape(-1)[warmup_steps:]
        power_sum = p if power_sum is None else power_sum + p
    return zone_sum / n, float(power_sum.mean()), m_per_rad


def _bisect_rating_for_zones(m_per_rad, *, target, n_iter, Q_bracket, label, **kw):
    """Bisection on the total nominal rating to drive the simulated mean zone temp
    to `target`. Simulated zone temp increases monotonically with delivered heat."""
    lo, hi = Q_bracket
    Q = 0.5 * (lo + hi)
    zmean = power = float("nan")
    used_flow = m_per_rad
    for i in range(n_iter):
        Q = 0.5 * (lo + hi)
        zmean, power, used_flow = _sim_mean_zone_and_power(Q, m_per_rad, **kw)
        print(f"  [{label}] iter {i+1}/{n_iter}: Q_nom={Q:7.1f} W, flow/rad="
              f"{used_flow:.4f} kg/s -> zoneT={zmean:5.2f} C (target {target:5.2f}) "
              f" Sigma Power={power:7.1f} W")
        if zmean > target:   # over-heating -> reduce rating
            hi = Q
        else:
            lo = Q
    return Q, power, used_flow


def compute_winter_sizing(
    start,
    end,
    *,
    envelope_pickle: Optional[str] = None,
    step: int = 600,
    warmup_steps: int = 24,
    n_iter: int = 9,
    Q_bracket=(400.0, 9000.0),
) -> dict:
    """Energy-balance sizing for the closed-loop radiators + fixed flow.

    The radiator steady-state heat output (UA) is NOT AD-calibratable in Twin4Build
    (SpaceHeaterTorchSystem re-solves UA from the float nominal rating inside
    initialize() and freezes it). To keep the hydronic stage consistent with the
    Stage-1 envelope — calibrated with per-room heat input = district-heat power / 4
    — we SIZE the radiators from the envelope's own heat demand:

      - Pin each radiator's nominal temps to the measured OPERATING point
        (T_a=supply, T_b=return, TAir=indoor); the fixed flow is the design flow of
        the rating, m = (Q/4) / (cp * (T_sup - T_ret)).
      - Bisect the nominal rating until the SIMULATED mean zone temperature equals
        the measured mean — i.e. the radiators hold the building exactly. This is
        the control-relevant target and the physically meaningful heat demand (a
        free-running radiator at an arbitrary rating would over/under-heat).

    Note on the supply/return ↔ zone tension: with the envelope fixed and the
    radiator UA/flow non-AD, the zone-temperature and return-water targets cannot
    both be matched exactly (matching zones leaves a return offset and vice versa).
    We size for the zones (the RL-relevant signal) and report the realized Σ Power
    vs the metered energy as the consistency diagnostic (validation_plots spring
    check). `Q_realized_total` is that realized heat output at the zone match.

    Returns a sizing dict consumed by apply_energy_balance_sizing().
    """
    n = len(ZONES)
    if envelope_pickle is None:
        envelope_pickle = ENVELOPE_RESULT_PICKLE

    T_sup = _read_window_mean("ecl310_TSupHea_y_processed.csv", start, end)
    T_ret = _read_window_mean("ecl310_TRetHea_y_processed.csv", start, end)
    T_air = sum(_read_window_mean(f"{z}_indoor_temperature.csv", start, end)
                for z in ZONES) / n

    kw = dict(start=start, end=end, step=step, warmup_steps=warmup_steps,
              target=T_air, n_iter=n_iter, Q_bracket=Q_bracket,
              T_sup=T_sup, T_ret=T_ret, T_air=T_air, envelope_pickle=envelope_pickle)

    # Bisection on the nominal rating (flow coupled via the energy balance) to hold
    # the measured mean zone temperature.
    Q_final, P_final, m_per_rad = _bisect_rating_for_zones(None, label="zone-match", **kw)

    return {
        "Q_demand_total": Q_final,
        "Q_realized_total": P_final,
        "T_sup": T_sup,
        "T_ret": T_ret,
        "T_air": T_air,
        "Q_per_rad": Q_final / n,
        "m_per_rad": m_per_rad,
    }


def apply_energy_balance_sizing(model, sizing: dict) -> None:
    """Apply an energy-balance sizing dict (from compute_winter_sizing) to a built
    model: pin each radiator's nominal rating to the operating point and set the
    fixed radiator-loop flow. Must be called BEFORE the model is initialized.
    """
    _apply_sizing(model, Q_per_rad=sizing["Q_per_rad"], T_sup=sizing["T_sup"],
                  T_ret=sizing["T_ret"], T_air=sizing["T_air"],
                  m_per_rad=sizing["m_per_rad"])


def get_or_compute_sizing(window=None, step: int = 600, cache_path: str = None,
                          force: bool = False) -> dict:
    """Return the radiator energy-balance sizing dict, from the JSON cache if it
    matches the requested window, otherwise compute it (compute_winter_sizing)
    and cache it. Keyed on the window so a changed window transparently
    recomputes. Pass force=True to always recompute."""
    import datetime
    import json
    from dateutil.tz import gettz

    if window is None:
        window = HYDRONIC_SIZING_WINDOW
    if cache_path is None:
        cache_path = SIZING_CACHE_PATH

    if not force and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("window") == list(window) and cached.get("step") == step:
                return cached["sizing"]
        except (json.JSONDecodeError, KeyError, OSError):
            pass  # cache unreadable/stale — fall through to recompute

    tz = gettz(TZ)
    start = datetime.datetime.fromisoformat(window[0]).replace(tzinfo=tz)
    end = datetime.datetime.fromisoformat(window[1]).replace(tzinfo=tz)
    sizing = compute_winter_sizing(start, end, step=step)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"window": list(window), "step": step, "sizing": sizing}, f, indent=2)
    return sizing


def fcn(self) -> None:
    """Default master fcn — calibration mode."""
    make_fcn(calibration_mode=True)(self)


def get_model(
    id: str = "skoven_model",
    fcn_: Optional[Callable] = None,
    calibration_mode: bool = True,
) -> tb.Model:
    if fcn_ is None:
        fcn_ = make_fcn(calibration_mode=calibration_mode)
    model = tb.Model(id=id)
    # force_config_overwrite=False: the parameters defined IN the fcn (constructors,
    # e.g. the 3-ACH damper sizing, RADIATOR_DEFAULTS, AHU_DEFAULTS) take priority;
    # the per-component cached JSONs only fill params left None. With True the STALE
    # cached files would override the fcn values (it silently reset the new 3-ACH
    # damper sizing back to a stale 0.05 kg/s). See SimulationModel._load_parameters.
    model.load(
        fcn=fcn_,
        draw_semantic_model=False,
        draw_simulation_model=False,
        validate_model=True,
        force_config_overwrite=False,
    )
    return model


def load_model_and_params(
    envelope_pickle: Optional[str] = None,
    hydronic_pickle: Optional[str] = None,
    ahu_pickle: Optional[str] = None,
    calibration_mode: bool = False,
    apply_sizing: bool = True,
) -> tb.Model:
    """Load Skoven model and apply calibrated parameter pickles. Defaults to
    simulation mode (calibration_mode=False) so it's RL-ready.

    The radiator energy-balance sizing (compute_winter_sizing) is applied
    between the envelope and hydronic pickles — the same order as the
    validation build (scripts/validation_plots.py). Without it the radiators
    keep their oversized construction defaults and deliver ~12x too much heat,
    so all zones run away to ~35 C (the sizing lives in Q_flow_nominal_sh /
    radiator flow, which are plain floats and are NOT restored by the pickles).
    Set apply_sizing=False only for diagnostics that want the raw defaults.
    """
    model = get_model(calibration_mode=calibration_mode)

    if envelope_pickle is None:
        envelope_pickle = ENVELOPE_RESULT_PICKLE
    if hydronic_pickle is None:
        hydronic_pickle = HYDRONIC_RESULT_PICKLE
    if ahu_pickle is None:
        ahu_pickle = AHU_RESULT_PICKLE

    # Envelope must be loaded before sizing (the bisection simulates the
    # calibrated envelope); the hydronic pickle (PID gains + radiator thermal
    # mass) is loaded after. Sizing must precede first initialize() — the
    # SpaceHeaterTorchSystem freezes UA from the nominal rating on first init.
    if os.path.exists(envelope_pickle):
        model.load_estimation_result(envelope_pickle)
    else:
        print(f"Info: pickle not found yet — {envelope_pickle}")

    if apply_sizing:
        try:
            sizing = get_or_compute_sizing()
            apply_energy_balance_sizing(model, sizing)
        except Exception as e:  # noqa: BLE001 — sizing needs winter-window CSVs
            print(f"Warning: radiator sizing not applied ({e}); radiators will "
                  f"use oversized construction defaults.")

    for pickle_path in [hydronic_pickle, ahu_pickle]:
        if os.path.exists(pickle_path):
            model.load_estimation_result(pickle_path)
        else:
            print(f"Info: pickle not found yet — {pickle_path}")

    return model


# Plot/eval targets — used by model_eval.py to overlay simulated vs measured.
model_output_points = [
    {
        "component_id": "outdoor_environment",
        "output_value": "outdoorTemperature",
        "csv_path": _csv("outdoor_temperature.csv"),
    },
    {
        "component_id": "ecl310_TSupHea_y",
        "output_value": "measuredValue",
        "csv_path": _csv("ecl310_TSupHea_y_processed.csv"),
    },
    {
        "component_id": "ecl310_TRetHea_y",
        "output_value": "measuredValue",
        "csv_path": _csv("ecl310_TRetHea_y_processed.csv"),
    },
    {
        "component_id": "varme_meter_power_sensor",
        "output_value": "measuredValue",
        "csv_path": _csv("varme_meter_power_kW.csv"),
    },
    *[
        {
            "component_id": f"{z}_indoor_temp_sensor",
            "output_value": "measuredValue",
            "csv_path": _csv(f"{z}_indoor_temperature.csv"),
        }
        for z in ZONES
    ],
]
