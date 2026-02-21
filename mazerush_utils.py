"""Utility functions for the Mazerush environment."""

import enum


# ---------------------------------------------------------------------------
# Constants / enums
# ---------------------------------------------------------------------------

class CellType(enum.IntEnum):
    EMPTY = 0
    WALL = 1


class PlayerStatus(enum.IntEnum):
    NEUTRAL = 0
    HAS_LASER = 1
    SHOOTING = 2
