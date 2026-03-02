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

class ObsCell(enum.IntEnum):
  EMPTY = 0
  WALL = 1
  LASER_ITEM = 2
  PLAYER_SELF_NEUTRAL = 3
  PLAYER_SELF_HAS_LASER = 4
  PLAYER_SELF_SHOOTING = 5
  PLAYER_OTHER_NEUTRAL = 6
  PLAYER_OTHER_HAS_LASER = 7
  PLAYER_OTHER_SHOOTING = 8
  LASER_SELF = 9
  LASER_OTHER = 10


NUM_OBS_CELL_TYPES = len(ObsCell)

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
