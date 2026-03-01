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


class Player:
    __slots__ = (
        "x", "y", "status", "move_cooldown_remaining",
        "shoot_ticks_remaining", "alive",
    )

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.status: PlayerStatus = PlayerStatus.NEUTRAL
        self.move_cooldown_remaining: int = 0
        self.shoot_ticks_remaining: int = 0
        self.alive: bool = True
