"""Rule parameters for Ink Arena v1.

Values marked (tune) in the design doc live here so the tuning pass can change
them in one place. Distances are in cells, times in ticks (10 ticks = 1 s).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Rules:
    # Match
    ticks: int = 600  # 60 s
    team_size: int = 2

    # Agents
    radius: float = 0.4
    max_hp: int = 3
    max_ink: float = 100.0

    # Movement (tune)
    walk_speed: float = 0.25
    swim_speed: float = 0.5
    enemy_ink_factor: float = 0.5
    turn_rate: float = math.radians(30)

    # Ink (tune)
    shot_cost: float = 4.0
    fire_cooldown: int = 3
    swim_refill: float = 5.0
    idle_refill: float = 0.5
    idle_refill_delay: int = 5

    # Shots (tune)
    shot_speed: float = 1.5
    shot_range: float = 3.0
    strip_halfwidth: float = 1.0
    splat_radius: float = 1.5
    shot_damage: int = 1

    # HP regen (tune)
    regen_delay: int = 30
    regen_interval: int = 20

    # Death and respawn (tune)
    burst_radius: float = 2.0
    respawn_delay: int = 30
    respawn_invisible: int = 20

    # Visibility (tune)
    view_half_angle: float = math.radians(60)
    view_range: float = 6.0
    submerged_view_range: float = 2.0

    # Sandbox (used from phase 2)
    max_crashes: int = 3


DEFAULT_RULES = Rules()
