"""Shared RL config: paths + train/eval windows, single source of truth.

Read from `use_case/building_configs/skoven.yaml` so `skoven_RL_control.py`,
`model_eval.py`, and `baseline_eval.py` can't drift out of sync the way the
old hardcoded per-file datetime constants did (train/eval windows exceeded
the exported weather CSV coverage in one file but not the other).
"""
import datetime
import os

import yaml
from dateutil.tz import gettz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

TZ = "Europe/Copenhagen"
POLICY_CONFIG_PATH = os.path.join(SCRIPT_DIR, "policy_input_output.json")
BUILDING_CONFIG_PATH = os.path.join(SCRIPT_DIR, "building_configs", "skoven.yaml")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
CHECKPOINT_DIR = os.path.join(LOG_DIR, "checkpoints")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

STEP_SIZE = 600        # seconds
EPISODE_STEPS = int(3600 * 24 * 5 / STEP_SIZE)   # 5-day episodes

# Comfort band: no per-room control, so comfort is measured against a fixed
# building-wide heating setpoint. Shared between the reward (skoven_RL_control)
# and the eval/baseline KPI computation so they can't drift apart.
COMFORT_SETPOINT_C = 21.0
COMFORT_DEADBAND_C = 0.2


def load_building_config() -> dict:
    with open(BUILDING_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_rl_windows():
    """Returns (train_start, train_end, eval_start, eval_end) as tz-aware
    datetimes in Europe/Copenhagen, parsed from skoven.yaml."""
    cfg = load_building_config()
    tz = gettz(TZ)

    def parse(s: str) -> datetime.datetime:
        return datetime.datetime.combine(
            datetime.date.fromisoformat(s), datetime.time.min, tzinfo=tz
        )

    return (
        parse(cfg["train_start"]), parse(cfg["train_end"]),
        parse(cfg["eval_start"]), parse(cfg["eval_end"]),
    )


# ---------------------------------------------------------------------------
# DST-transition avoidance.
#
# dateutil's gettz() returns a SINGLETON tzinfo object per zone name, so any
# two Europe/Copenhagen datetimes we construct end up sharing the identical
# tzinfo instance. CPython's aware-datetime subtraction has a fast path for
# "same tzinfo object" that subtracts naive wall-clock fields WITHOUT
# re-resolving each operand's utcoffset — which is wrong whenever a DST
# transition falls between the two instants (confirmed empirically: a
# 720-step/600s window starting before an Oct DST fall-back reports 432000s
# elapsed via direct subtraction, but is actually 435600s / 726 real steps).
# Twin4Build's `Simulator.get_simulation_timesteps` and the CSV loader's
# `pandas.date_range` disagree in exactly this way, crashing with a shape
# mismatch. There is no safe way to fix this from the caller's side other
# than never handing Twin4Build a simulation window that crosses a
# transition — hence the helpers below.
def _last_sunday(year: int, month: int) -> datetime.date:
    next_month = (
        datetime.date(year + 1, 1, 1) if month == 12
        else datetime.date(year, month + 1, 1)
    )
    last_day = next_month - datetime.timedelta(days=1)
    offset = (last_day.weekday() - 6) % 7  # Monday=0 .. Sunday=6
    return last_day - datetime.timedelta(days=offset)


def dst_transition_instants(start: datetime.datetime, end: datetime.datetime) -> list:
    """EU DST transitions (last Sunday of March / October, 01:00 UTC) that
    fall within [start, end]."""
    instants = []
    for year in range(start.year, end.year + 2):
        for month in (3, 10):
            d = _last_sunday(year, month)
            instant = datetime.datetime(d.year, d.month, d.day, 1, 0, tzinfo=datetime.timezone.utc)
            if start <= instant <= end:
                instants.append(instant)
    return sorted(instants)


def dst_safe_chunks(start: datetime.datetime, end: datetime.datetime) -> list:
    """Split [start, end) into sub-windows that never cross a DST transition
    — used to run a long eval/baseline simulation as several back-to-back
    segments instead of one continuous (and DST-unsafe) run."""
    bounds = [start] + dst_transition_instants(start, end) + [end]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def dst_excluding_periods(start: datetime.datetime, end: datetime.datetime,
                           episode_length: int, step_size: int) -> list:
    """`T4BGymEnv(excluding_periods=...)` value that blocks any randomly
    sampled `episode_length`-step start time whose episode window would
    straddle a DST transition."""
    episode_span = datetime.timedelta(seconds=episode_length * step_size)
    return [
        (instant - episode_span, instant + datetime.timedelta(days=1))
        for instant in dst_transition_instants(start, end)
    ]
