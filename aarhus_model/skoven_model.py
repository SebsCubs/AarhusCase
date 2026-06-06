"""Skoven building Twin4Build model.

Three-stage fcn() pattern (mirrors T4BGymUseCase/boptest_model/5_rooms_model.py):
  - envelope_fcn : zones + outdoor environment + occupancy + ReMoni sensors
  - hydronic_fcn : ECL310 heating curve → mixing valve → radiators
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
import os
from typing import Callable, Optional

import twin4build as tb
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

TZ = "Europe/Copenhagen"

# 1-floor building, 4 rooms in a ring. Each room is adjacent to two others and
# exposed to the outdoor. Rooms map 1:1 to ReMoni sensors viben8/9/10/11.
ZONES = {
    "room_a": {
        "adjacent_to": ["room_b", "room_d"],
        "outdoor_exposed": True,
        "nominal_flow_kgs": 0.05,
    },
    "room_b": {
        "adjacent_to": ["room_c", "room_a"],
        "outdoor_exposed": True,
        "nominal_flow_kgs": 0.05,
    },
    "room_c": {
        "adjacent_to": ["room_d", "room_b"],
        "outdoor_exposed": True,
        "nominal_flow_kgs": 0.05,
    },
    "room_d": {
        "adjacent_to": ["room_a", "room_c"],
        "outdoor_exposed": True,
        "nominal_flow_kgs": 0.05,
    },
}

RADIATOR_DEFAULTS = {
    "Q_flow_nominal_sh": 1500.0,
    "T_a_nominal_sh": 55.0,
    "T_b_nominal_sh": 45.0,
    "TAir_nominal_sh": 21.0,
    "thermalMassHeatCapacity": 5000.0,
    "nelements": 3,
}

AHU_DEFAULTS = {
    "fan_nominal_flow_kgs": 0.5,
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
    "hr_primary_max_kgs": 0.5,
    "hr_secondary_max_kgs": 0.5,
    "damper_a": 0.5,
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
        zone = BuildingSpaceThermalTorchSystem(
            id=zone_id,
            C_air=1e6,
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
# Hydronic
# ---------------------------------------------------------------------------
def hydronic_fcn(self, *, calibration_mode: bool = True) -> None:
    """Building inlet-temperature control + 4 fixed-flow radiators.

    The real system has ONE control lever: a mixing valve (driven by a PID
    tracking an outdoor-reset heating curve) sets the building inlet water
    temperature. Each of the 4 rooms has a radiator on a FIXED-opening valve
    (constant flow), all fed in parallel from the same supply.

      calibration_mode=True:
         BMS Fremløb CSV (leaf SensorSystem) is the supply-water boundary;
         radiators run with the measured inlet temperature.

      calibration_mode=False:
         the supply-water setpoint is an RL-controllable schedule (the mixing
         valve command / heating-curve target).
    """
    # ----- Supply-water source (mode-dependent) ------------------------------
    if calibration_mode:
        supply_water_source = tb.SensorSystem(
            id="ecl310_TSupHea_y",
            filename=_csv("ecl310_TSupHea_y_processed.csv"),
            
        )
        supply_port = "measuredValue"
    else:
        supply_water_source = tb.ScheduleSystem(
            id="ecl310_TSupHea_y",
            weekDayRulesetDict={
                "ruleset_default_value": 50.0,
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [24],
                "ruleset_value": [50.0],
            },
            
        )
        supply_port = "scheduleValue"

    # ECL310 mixing valve + PID — present in both modes for parameter
    # identifiability (Stage 2 estimates kp/Ti even in open-loop replay).
    ecl310_valve = ValveTorchSystem(
        id="ecl310_kr1_valve",
        waterFlowRateMax=0.5,
        valveAuthority=0.5,
        
    )
    ecl310_pid = tb.PIDControllerSystem(
        id="ecl310_pid",
        kp=0.5,
        Ti=300.0,
        Td=0.0,
        isReverse=False,
        
    )

    # PID setpoint: ECL310 outdoor-reset heating-curve trajectory.
    # Calibration mode replays the precomputed curve CSV (from BMS T_oa/T_set);
    # simulation mode uses a constant schedule the RL agent can override.
    if calibration_mode:
        ecl310_setpoint = tb.SensorSystem(
            id="ecl310_TSupSet_schedule",
            filename=_csv("ecl310_TSupSet_curve.csv"),
        )
        setpoint_port = "measuredValue"
    else:
        ecl310_setpoint = tb.ScheduleSystem(
            id="ecl310_TSupSet_schedule",
            weekDayRulesetDict={
                "ruleset_default_value": 50.0,
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [24],
                "ruleset_value": [50.0],
            },
        )
        setpoint_port = "scheduleValue"
    self.add_connection(ecl310_setpoint, ecl310_pid, setpoint_port, "setpointValue")
    self.add_connection(
        supply_water_source, ecl310_pid, supply_port, "actualValue"
    )
    self.add_connection(ecl310_pid, ecl310_valve, "inputSignal", "valvePosition")

    # ----- Fixed-opening radiator valves + radiators ------------------------
    # Each of the 4 rooms has one radiator fed by a FIXED-opening valve. There is
    # no per-room control: the valve position is constant and the design flow is
    # captured by the estimable waterFlowRateMax. The single control lever is the
    # building inlet temperature (mixing valve + ecl310_pid above). All radiators
    # draw from the same supply line in parallel.
    radiator_fixed_position = tb.ScheduleSystem(
        id="radiator_fixed_position",
        weekDayRulesetDict={
            "ruleset_default_value": 1.0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [0],
            "ruleset_end_hour": [24],
            "ruleset_value": [1.0],
        },
    )
    for zone_id in ZONES:
        zone = self.components[zone_id]
        rad_valve = ValveTorchSystem(
            id=f"{zone_id}_radiator_valve",
            waterFlowRateMax=0.05,
            valveAuthority=1.0,
        )
        self.add_connection(
            radiator_fixed_position, rad_valve, "scheduleValue", "valvePosition"
        )

        radiator = SpaceHeaterTorchSystem(
            id=f"{zone_id}_radiator",
            **RADIATOR_DEFAULTS,
        )
        self.add_connection(
            supply_water_source, radiator, supply_port, "supplyWaterTemperature"
        )
        self.add_connection(rad_valve, radiator, "waterFlowRate", "waterFlowRate")
        self.add_connection(zone, radiator, "indoorTemperature", "indoorTemperature")
        self.add_connection(radiator, zone, "Power", "heatGain")

    # ----- Return-water sensor (virtual — for Stage 2 RMSE target) ----------
    # NOTE: With no return-side mixing component, this sensor is connected to
    # one representative radiator's outletWaterTemperature as a proxy. A proper
    # mass-weighted mixing requires a custom WaterReturnJunctionSystem (TODO).
    ecl310_ret_sensor = tb.SensorSystem(
        id="ecl310_TRetHea_y",
        filename=_csv("ecl310_TRetHea_y_processed.csv") if calibration_mode else None,
    )
    representative_radiator = self.components[f"{next(iter(ZONES))}_radiator"]
    self.add_connection(
        representative_radiator, ecl310_ret_sensor, "outletWaterTemperature", "measuredValue"
    )

    # Varme heat-meter power sensor — leaf during calibration (target).
    # In RL mode, total heat is computed in the gym layer by summing per-radiator
    # Power outputs (no in-graph sum component exists).
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

    supply_air_setpoint = tb.ScheduleSystem(
        id="supply_air_temp_setpoint_sensor",
        weekDayRulesetDict={
            "ruleset_default_value": 18.0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [0],
            "ruleset_end_hour": [24],
            "ruleset_value": [18.0],
        },
        
    )
    self.add_connection(
        supply_air_setpoint, supply_heating_coil, "scheduleValue", "outletAirTemperatureSetpoint"
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

    for slot, (zone_id, cfg) in enumerate(ZONES.items()):
        zone_damper = DamperTorchSystem(
            id=f"{zone_id}_supply_damper",
            nominalAirFlowRate=cfg["nominal_flow_kgs"],
            a=AHU_DEFAULTS["damper_a"],
        )
        zone = self.components[zone_id]

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
    model.load(
        fcn=fcn_,
        draw_semantic_model=False,
        draw_simulation_model=False,
        validate_model=True,
        force_config_overwrite=True,
    )
    return model


def load_model_and_params(
    envelope_pickle: Optional[str] = None,
    hydronic_pickle: Optional[str] = None,
    ahu_pickle: Optional[str] = None,
    calibration_mode: bool = False,
) -> tb.Model:
    """Load Skoven model and apply calibrated parameter pickles. Defaults to
    simulation mode (calibration_mode=False) so it's RL-ready."""
    model = get_model(calibration_mode=calibration_mode)

    if envelope_pickle is None:
        envelope_pickle = ENVELOPE_RESULT_PICKLE
    if hydronic_pickle is None:
        hydronic_pickle = HYDRONIC_RESULT_PICKLE
    if ahu_pickle is None:
        ahu_pickle = AHU_RESULT_PICKLE

    for pickle_path in [envelope_pickle, hydronic_pickle, ahu_pickle]:
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
