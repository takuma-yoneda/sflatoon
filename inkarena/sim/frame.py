"""Team-canonical frame transform.

Team A sees the world as-is. Team B's observations and actions are rotated
180 degrees about the map centre so both teams "start at the bottom":
``x' = W - x``, ``y' = H - y``, ``theta' = theta + pi``. Grids are flipped on both
axes, and the ink sign and score order are swapped.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import wrap_angle

TEAM_A, TEAM_B = 0, 1


def pos_to_team(x: float, y: float, team: int, width: int, height: int) -> tuple[float, float]:
    if team == TEAM_A:
        return float(x), float(y)
    return float(width - x), float(height - y)


# The transform is its own inverse.
pos_from_team = pos_to_team


def angle_to_team(theta: float, team: int) -> float:
    return float(theta) if team == TEAM_A else wrap_angle(theta + math.pi)


angle_from_team = angle_to_team


def vec_to_team(dx: float, dy: float, team: int) -> tuple[float, float]:
    """Directions (moves, velocities) rotate without translation."""
    if team == TEAM_A:
        return float(dx), float(dy)
    return -float(dx), -float(dy)


vec_from_team = vec_to_team


def grid_to_team(grid: np.ndarray, team: int) -> np.ndarray:
    return grid if team == TEAM_A else grid[::-1, ::-1]


def ink_to_team(ink: np.ndarray, team: int) -> np.ndarray:
    """Ink grid with +1 = this team's ink, -1 = the other team's."""
    if team == TEAM_A:
        return ink.copy()
    return -ink[::-1, ::-1]
