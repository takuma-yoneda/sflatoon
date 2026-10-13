"""Ink Arena simulator core (numpy)."""

from .config import DEFAULT_RULES, Rules
from .engine import ACTION_DIM, Game, TickEvents
from .maps import Map, MapError, list_maps, load_map, parse_map, validate_map
from .match import MatchResult, run_match

__all__ = [
    "ACTION_DIM",
    "DEFAULT_RULES",
    "Game",
    "Map",
    "MapError",
    "MatchResult",
    "Rules",
    "TickEvents",
    "list_maps",
    "load_map",
    "parse_map",
    "run_match",
    "validate_map",
]
