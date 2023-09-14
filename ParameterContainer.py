"""Provides convenient data strcutures to manage relevant data used to manage simulation runs.
"""
from dataclasses import dataclass
from typing import Any


@dataclass(eq=True)
class InitParameter:

    """Struct to provide initial (i.e., sampled) parameters required at the start of a simulation."""

    run_id: int = None
    sampled_v_ego: float = None
    sampled_v_agent: float = None
    sampled_clearance: float = None
    sampled_a_brake: float = None
    sampled_ttc_target: float = None

    sampled_fog: float = None
    sampled_rain: float = None
    is_day: bool = True
    real_friction: float = None
    real_brake: float = None
    dist_first_detection: float = None  # classifier distance given weather
    dist_x_activation: float = (
        None  # min distance before braking due to rel velocity and ttc
    )
    dist_x_braking: float = (
        None  # actual distance as combination of classifier AND ttc trigger braking
    )


@dataclass(eq=True)
class StepParameter:

    """Struct to manage relevant parameters during a simulation (i.e., making them avail. for evaluation in each sim. step)."""

    cur_v_ego: float = None
    cur_v_agent: float = None
    is_braking: bool = False
    cur_a_brake: float = None
    cur_x_clearance: float = None
    cur_collision_energy: float = None
