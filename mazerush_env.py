"""Mazerush: a multi-player grid-world laser game for RL training."""

import random as _random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mazerush_utils import CellType, PlayerStatus
from renderer import MazerushRenderer

# ---------------------------------------------------------------------------
# Constants / enums
# ---------------------------------------------------------------------------


# Actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_SHOOT = 4
ACTION_NOTHING = 5
NUM_ACTIONS = 6

DIRECTION_DELTAS = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def _generate_grid(width: int, height: int, rng: _random.Random) -> np.ndarray:
    """Generate a grid with semi-regular wall patterns using recursive partitioning.

    Creates corridor-style subdivisions that produce interesting navigation
    spaces while always leaving connectivity between regions.
    """
    grid = np.zeros((width, height), dtype=np.int8)

    # Place border walls
    grid[0, :] = CellType.WALL
    grid[width - 1, :] = CellType.WALL
    grid[:, 0] = CellType.WALL
    grid[:, height - 1] = CellType.WALL

    def _partition(x1: int, y1: int, x2: int, y2: int, depth: int = 0):
        """Recursively partition the space with walls that have openings."""
        min_room = 5  # minimum room dimension
        w = x2 - x1
        h = y2 - y1

        if w < min_room * 2 or h < min_room * 2:
            return

        # Alternate between vertical and horizontal splits, with some randomness
        if w > h:
            split_vertical = True
        elif h > w:
            split_vertical = False
        else:
            split_vertical = rng.random() < 0.5

        if split_vertical:
            if w < min_room * 2:
                return
            # Choose split position (avoid edges)
            sx = rng.randint(x1 + min_room, x2 - min_room)
            # Draw vertical wall
            for y in range(y1 + 1, y2):
                grid[sx, y] = CellType.WALL
            # Make 2-3 openings in the wall
            num_openings = rng.randint(2, 3)
            possible_ys = list(range(y1 + 1, y2 - 1))
            rng.shuffle(possible_ys)
            for y in possible_ys[:num_openings]:
                grid[sx, y] = CellType.EMPTY
            # Recurse
            _partition(x1, y1, sx, y2, depth + 1)
            _partition(sx, y1, x2, y2, depth + 1)
        else:
            if h < min_room * 2:
                return
            sy = rng.randint(y1 + min_room, y2 - min_room)
            for x in range(x1 + 1, x2):
                grid[x, sy] = CellType.WALL
            num_openings = rng.randint(2, 3)
            possible_xs = list(range(x1 + 1, x2 - 1))
            rng.shuffle(possible_xs)
            for x in possible_xs[:num_openings]:
                grid[x, sy] = CellType.EMPTY
            _partition(x1, y1, x2, sy, depth + 1)
            _partition(x1, sy, x2, y2, depth + 1)

    _partition(0, 0, width - 1, height - 1)
    return grid


# ---------------------------------------------------------------------------
# Laser beam computation
# ---------------------------------------------------------------------------

def _compute_beam_cells(
    grid: np.ndarray,
    origin_x: int,
    origin_y: int,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    """Compute all cells a laser beam covers from *origin* in 4 directions.

    The beam extends until it hits a wall (exclusive of the wall cell).
    The origin cell itself is included.
    """
    cells: list[tuple[int, int]] = [(origin_x, origin_y)]
    for dx, dy in DIRECTION_DELTAS.values():
        x, y = origin_x + dx, origin_y + dy
        while 0 <= x < width and 0 <= y < height and grid[x, y] != CellType.WALL:
            cells.append((x, y))
            x += dx
            y += dy
    return cells


# ---------------------------------------------------------------------------
# Player dataclass
# ---------------------------------------------------------------------------

class _Player:
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


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MazerushEnv(gym.Env):
    """Multi-player Mazerush environment.

    ``step`` accepts a list of actions (one per player) and returns lists of
    observations, rewards, dones, truncateds, and infos (one per player).
    """

    metadata = {"render_modes": ["human", "human_full"]}

    def __init__(
        self,
        num_players: int = 2,
        width: int = 30,
        height: int = 20,
        seed: int = 42,
        fps: int = 30,
        move_cooldown: int = 5,
        laser_duration: int = 15,
        max_laser_items: int = 3,
        laser_spawn_prob: float = 0.02,
        laser_min_distance: int = 5,
        laser_spawn_retries: int = 10,
        max_episode_ticks: int = 3000,
        fov_size: int = 9,
        step_penalty: float = -0.01,
        render_mode: str | None = None,
    ):
        super().__init__()
        assert 2 <= num_players <= 4, "num_players must be 2-4"
        self.num_players = num_players
        self.width = width
        self.height = height
        self.seed_value = seed
        self.fps = fps
        self.move_cooldown = move_cooldown
        self.laser_duration = laser_duration
        self.max_laser_items = max_laser_items
        self.laser_spawn_prob = laser_spawn_prob
        self.laser_min_distance = laser_min_distance
        self.laser_spawn_retries = laser_spawn_retries
        self.max_episode_ticks = max_episode_ticks
        self.step_penalty = step_penalty
        self.render_mode = render_mode

        # Generate the static grid (walls never change).
        self._grid_rng = _random.Random(seed)
        self.grid = _generate_grid(width, height, self._grid_rng)

        # Precompute empty cells for spawning
        self._empty_cells: list[tuple[int, int]] = [
            (x, y)
            for x in range(width)
            for y in range(height)
            if self.grid[x, y] == CellType.EMPTY
        ]

        # Cache a padded grid for fast FOV slicing
        # Padding is equal to fov_radius on all sides, filled with WALL (1.0)
        self._padded_wall_grid = np.pad(
            (self.grid == CellType.WALL).astype(np.float32),
            pad_width=fov_size // 2,
            mode='constant',
            constant_values=1.0
        )

        # Action / observation spaces ---
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Observation vector per player components:
        #   own: x, y, cooldown, status  (4 components)
        #   per other player: x, y, cooldown, status, visible (5 components)
        #   per laser item slot: x, y, visible (3 components)
        #   map window: 9x9 surrounding binary matrix (81 components)
        self.fov_size = fov_size
        self.fov_radius = self.fov_size // 2

        self._player_states = [self.width, self.height, max(self.move_cooldown, self.laser_duration) + 1, 3]
        self._other_player_states = [self.fov_size, self.fov_size, max(self.move_cooldown, self.laser_duration) + 1, 3, 2]
        self._item_states = [self.fov_size, self.fov_size, 2]

        own_dim = sum(self._player_states) + 3
        other_dim = sum(self._other_player_states) + 3
        item_dim = sum(self._item_states) + 2
        map_dim = self.fov_size * self.fov_size

        self._total_dims = own_dim + other_dim * (self.num_players - 1) + item_dim * self.max_laser_items + map_dim
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self._total_dims,), dtype=np.float32)

        # Runtime state (populated in reset)
        self.players: list[_Player] = []
        self.laser_items: list[tuple[int, int]] = []
        self.occupied_cells: set[tuple[int, int]] = set()
        # Active beams: list of (player_idx, list_of_cells, ticks_remaining)
        self.active_beams: list[tuple[int, list[tuple[int, int]], int]] = []
        # Beam grid: for each cell, set of player indices whose beam covers it
        self.beam_grid: dict[tuple[int, int], set[int]] = {}
        self.tick: int = 0
        self._episode_rng = _random.Random(seed)

        # Renderer (lazy init)
        self._renderer = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None,
    ) -> tuple[list[np.ndarray], dict]:
        if seed is not None:
            self._episode_rng = _random.Random(seed)
        else:
            # Advance internal RNG so that next episode is truely random
            self._episode_rng = _random.Random(_random.randint(0, 2**31))

        self.tick = 0
        self.laser_items = []
        self.active_beams = []
        self.beam_grid = {}

        # Spawn players at distinct empty cells far apart
        self.players = []
        self.occupied_cells.clear()
        spawn_candidates = list(self._empty_cells)
        self._episode_rng.shuffle(spawn_candidates)

        for _ in range(self.num_players):
            for cx, cy in spawn_candidates:
                if (cx, cy) not in self.occupied_cells:
                    # Try to keep distance from existing players
                    if all(
                        abs(cx - p.x) + abs(cy - p.y) >= min(self.width, self.height) // 3
                        for p in self.players
                    ):
                        self.players.append(_Player(cx, cy))
                        self.occupied_cells.add((cx, cy))
                        break
            else:
                # Fallback: just pick any free cell
                for cx, cy in spawn_candidates:
                    if (cx, cy) not in self.occupied_cells:
                        self.players.append(_Player(cx, cy))
                        self.occupied_cells.add((cx, cy))
                        break

        obs_n = [self._get_obs(i) for i in range(self.num_players)]
        return obs_n, {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, action_n: list[int],
    ) -> tuple[list[np.ndarray], list[float], list[bool], list[bool], list[dict]]:
        assert len(action_n) == self.num_players
        self.tick += 1

        reward_n = [self.step_penalty] * self.num_players
        done_n = [False] * self.num_players
        truncated_n = [False] * self.num_players
        info_n: list[dict[str, Any]] = [{} for _ in range(self.num_players)]

        # --- 1 & 2. Handle cooldowns, beam expiration, and new shoots ---
        for p in self.players:
            if p.move_cooldown_remaining > 0:
                p.move_cooldown_remaining -= 1

        new_active_beams = []
        beams_changed = False

        for pidx, cells, ticks in self.active_beams:
            if ticks > 1:
                new_active_beams.append((pidx, cells, ticks - 1))
            else:
                beams_changed = True
                p = self.players[pidx]
                if p.status == PlayerStatus.SHOOTING:
                    p.status = PlayerStatus.NEUTRAL
                    p.move_cooldown_remaining = self.move_cooldown

        for i, (act, p) in enumerate(zip(action_n, self.players)):
            if act == ACTION_SHOOT and p.status == PlayerStatus.HAS_LASER and p.alive:
                beams_changed = True
                beam_cells = _compute_beam_cells(self.grid, p.x, p.y, self.width, self.height)
                p.status = PlayerStatus.SHOOTING
                p.shoot_ticks_remaining = self.laser_duration
                p.move_cooldown_remaining = self.laser_duration
                new_active_beams.append((i, beam_cells, self.laser_duration))

        self.active_beams = new_active_beams
        if beams_changed:
            self._rebuild_beam_grid()

        # --- 3. Check beam kills ---
        self._resolve_kills(reward_n, done_n, info_n)

        # --- 4. Process movement (only if episode not over) ---
        if not any(done_n):
            for i, (act, p) in enumerate(zip(action_n, self.players)):
                if act in DIRECTION_DELTAS and p.status != PlayerStatus.SHOOTING and p.alive:
                    if p.move_cooldown_remaining <= 0:
                        dx, dy = DIRECTION_DELTAS[act]
                        nx, ny = p.x + dx, p.y + dy
                        if (
                            0 <= nx < self.width
                            and 0 <= ny < self.height
                            and self.grid[nx, ny] != CellType.WALL
                            and (nx, ny) not in self.occupied_cells
                        ):
                            self.occupied_cells.remove((p.x, p.y))
                            self.occupied_cells.add((nx, ny))
                            p.x = nx
                            p.y = ny
                            p.move_cooldown_remaining = self.move_cooldown

                            # Check laser item pickup
                            if (nx, ny) in self.laser_items:
                                self.laser_items.remove((nx, ny))
                                if p.status == PlayerStatus.NEUTRAL:
                                    p.status = PlayerStatus.HAS_LASER
                                reward_n[i] += 0.1
                                # Opponents get negative reward
                                for j in range(self.num_players):
                                    if j != i:
                                        reward_n[j] -= 0.1

            # --- 5. Check beam kills again after movement ---
            self._resolve_kills(reward_n, done_n, info_n)

        # --- 6. Spawn laser items ---
        if not any(done_n):
            self._try_spawn_laser_item()

        # --- 7. Truncation ---
        if not any(done_n) and self.tick >= self.max_episode_ticks:
            truncated_n = [True] * self.num_players

        # --- 8. Render ---
        if self.render_mode in ["human", "human_full"]:
            self.render()

        obs_n = [self._get_obs(i) for i in range(self.num_players)]
        return obs_n, reward_n, done_n, truncated_n, info_n

    # ------------------------------------------------------------------
    # Kill resolution
    # ------------------------------------------------------------------

    def _resolve_kills(self, reward_n: list[float], done_n: list[bool], info_n: list[dict[str, Any]]):
        if any(done_n):
            return

        eliminated: set[int] = set()
        for i, p in enumerate(self.players):
            if not p.alive:
                continue
            cell = (p.x, p.y)
            if cell in self.beam_grid:
                hitters = self.beam_grid[cell] - {i}
                if hitters:
                    eliminated.add(i)

        if eliminated:
            winners: set[int] = set()
            for i in eliminated:
                cell = (self.players[i].x, self.players[i].y)
                winners |= self.beam_grid[cell] - {i}
                self.players[i].alive = False

            for i in range(self.num_players):
                if i in eliminated and i in winners:
                    reward_n[i] += 1.0
                    done_n[i] = True
                    info_n[i]["result"] = "draw"
                elif i in eliminated:
                    reward_n[i] += -10.0
                    done_n[i] = True
                    info_n[i]["result"] = "lose"
                elif i in winners:
                    reward_n[i] += 10.0
                    done_n[i] = True
                    info_n[i]["result"] = "win"
                else:
                    done_n[i] = True  # episode ends for everyone

    # ------------------------------------------------------------------
    # Beam grid helper
    # ------------------------------------------------------------------

    def _rebuild_beam_grid(self):
        self.beam_grid = {}
        for pidx, cells, _ in self.active_beams:
            for c in cells:
                if c not in self.beam_grid:
                    self.beam_grid[c] = {pidx}
                else:
                    self.beam_grid[c].add(pidx)

    # ------------------------------------------------------------------
    # Laser item spawning
    # ------------------------------------------------------------------

    def _try_spawn_laser_item(self):
        if len(self.laser_items) >= self.max_laser_items:
            return
        if self._episode_rng.random() > self.laser_spawn_prob:
            return

        occupied = {(p.x, p.y) for p in self.players if p.alive}
        occupied |= set(self.laser_items)
        # Also exclude wall cells and beam cells
        beam_cells = set(self.beam_grid.keys())

        for _ in range(self.laser_spawn_retries):
            cell = self._episode_rng.choice(self._empty_cells)
            if cell in occupied or cell in beam_cells:
                continue
            # Check Manhattan distance from all alive players
            if all(
                abs(cell[0] - p.x) + abs(cell[1] - p.y) >= self.laser_min_distance
                for p in self.players if p.alive
            ):
                self.laser_items.append(cell)
                return

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs(self, player_idx: int) -> np.ndarray:
        obs = np.zeros(self._total_dims, dtype=np.float32)
        idx = 0

        def _write_one_hots(vals: list[int], dims: list[int]):
            nonlocal idx
            for val, dim in zip(vals, dims):
                obs[idx + val] = 1.0
                idx += dim

        # Own player first
        own_p = self.players[player_idx]
        _write_one_hots(
            [own_p.x, own_p.y, own_p.move_cooldown_remaining, int(own_p.status)],
            self._player_states
        )
        obs[idx] = own_p.x / self.width
        obs[idx+1] = own_p.y / self.height
        obs[idx+2] = own_p.move_cooldown_remaining / self.move_cooldown
        idx += 3

        # Other players
        for i, p in enumerate(self.players):
            if i != player_idx:
                in_range = (
                    abs(p.x - own_p.x) <= self.fov_radius and
                    abs(p.y - own_p.y) <= self.fov_radius and
                    p.alive
                )
                rel_x = p.x - own_p.x + self.fov_radius
                rel_y = p.y - own_p.y + self.fov_radius
                if in_range:
                    _write_one_hots(
                        [rel_x, rel_y, p.move_cooldown_remaining, int(p.status), 1],
                        self._other_player_states
                    )
                else:
                    _write_one_hots([0, 0, 0, 0, 0], self._other_player_states)
                obs[idx] = rel_x / self.fov_size
                obs[idx+1] = rel_y / self.fov_size
                obs[idx+2] = p.move_cooldown_remaining / self.move_cooldown
                idx += 3

        # Laser items
        for slot in range(self.max_laser_items):
            if slot < len(self.laser_items):
                ix, iy = self.laser_items[slot]
                in_range = (
                    abs(ix - own_p.x) <= self.fov_radius and
                    abs(iy - own_p.y) <= self.fov_radius
                )
                rel_x = ix - own_p.x + self.fov_radius
                rel_y = iy - own_p.y + self.fov_radius
                if in_range:
                    _write_one_hots([rel_x, rel_y, 1], self._item_states)  # visible
                else:
                    _write_one_hots([0, 0, 0], self._item_states)
                obs[idx] = rel_x / self.fov_size
                obs[idx+1] = rel_y / self.fov_size
                idx += 2
            else:
                _write_one_hots([0, 0, 0], self._item_states)
                idx += 2

        # Map memory (9x9) using fast NumPy slicing
        slice_vals = self._padded_wall_grid[
            own_p.x : own_p.x + self.fov_size,
            own_p.y : own_p.y + self.fov_size
        ]
        obs[idx : idx + self.fov_size * self.fov_size] = slice_vals.flatten()
        return obs

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode not in ["human", "human_full"]:
            return
        if self._renderer is None:
            self._renderer = MazerushRenderer(self)
        self._renderer.render(self)

    def get_key_events(self) -> list:
        """Return pygame events; used by run loop to feed HumanAgent."""
        if self._renderer is not None:
            return self._renderer.get_events()
        return []

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
