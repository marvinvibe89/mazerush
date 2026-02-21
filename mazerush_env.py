"""Mazerush: a multi-player grid-world laser game for RL training."""

import enum
import random as _random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


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

    metadata = {"render_modes": ["human"]}

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

        # Action / observation spaces ---
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Observation vector per player components:
        #   own: x, y, cooldown, status  (4 components)
        #   per other player: same 4 components
        #   per laser item slot: x, y, exists  (3 components)
        player_states = [self.width, self.height, max(self.move_cooldown, self.laser_duration) + 1, 3]
        item_states = [self.width, self.height, 2]
        self._num_states = player_states * self.num_players + item_states * self.max_laser_items

        self.observation_space = spaces.MultiDiscrete(self._num_states)

        # Runtime state (populated in reset)
        self.players: list[_Player] = []
        self.laser_items: list[tuple[int, int]] = []
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
        occupied: set[tuple[int, int]] = set()
        spawn_candidates = list(self._empty_cells)
        self._episode_rng.shuffle(spawn_candidates)

        for _ in range(self.num_players):
            for cx, cy in spawn_candidates:
                if (cx, cy) not in occupied:
                    # Try to keep distance from existing players
                    if all(
                        abs(cx - p.x) + abs(cy - p.y) >= min(self.width, self.height) // 3
                        for p in self.players
                    ):
                        self.players.append(_Player(cx, cy))
                        occupied.add((cx, cy))
                        break
            else:
                # Fallback: just pick any free cell
                for cx, cy in spawn_candidates:
                    if (cx, cy) not in occupied:
                        self.players.append(_Player(cx, cy))
                        occupied.add((cx, cy))
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

        reward_n = [0.0] * self.num_players
        done_n = [False] * self.num_players
        truncated_n = [False] * self.num_players
        info_n: list[dict[str, Any]] = [{} for _ in range(self.num_players)]

        # --- 1. Decrement cooldowns / beam timers ---
        for p in self.players:
            if p.move_cooldown_remaining > 0:
                p.move_cooldown_remaining -= 1

        # Tick down active beams
        surviving_beams: list[tuple[int, list[tuple[int, int]], int]] = []
        for pidx, cells, ticks in self.active_beams:
            if ticks - 1 > 0:
                surviving_beams.append((pidx, cells, ticks - 1))
            else:
                # Beam expired – player re-enters NEUTRAL
                p = self.players[pidx]
                if p.status == PlayerStatus.SHOOTING:
                    p.status = PlayerStatus.NEUTRAL
                    p.move_cooldown_remaining = self.move_cooldown
        self.active_beams = surviving_beams
        self._rebuild_beam_grid()

        # --- 2. Resolve simultaneous shoots first ---
        shooters: list[int] = []
        for i, (act, p) in enumerate(zip(action_n, self.players)):
            if act == ACTION_SHOOT and p.status == PlayerStatus.HAS_LASER and p.alive:
                shooters.append(i)

        # Compute raw beams for all shooters
        new_beams: list[tuple[int, list[tuple[int, int]]]] = []
        for pidx in shooters:
            p = self.players[pidx]
            beam_cells = _compute_beam_cells(self.grid, p.x, p.y, self.width, self.height)
            new_beams.append((pidx, beam_cells))

        # Handle neutralization: if two beams share a cell, truncate both
        # at the intersection point (the shared cell is kept but neither
        # beam extends past it).
        if len(new_beams) >= 2:
            new_beams = self._neutralize_beams(new_beams)

        # Also neutralize new beams against *existing* beams
        if new_beams and self.active_beams:
            all_beams = [(pidx, cells) for pidx, cells, _ in self.active_beams]
            combined = all_beams + new_beams
            combined = self._neutralize_beams(combined)
            # Separate back: first len(all_beams) are existing, rest are new
            for idx, (pidx, cells) in enumerate(combined[:len(all_beams)]):
                orig_pidx, _, ticks = self.active_beams[idx]
                self.active_beams[idx] = (orig_pidx, cells, ticks)
            new_beams = combined[len(all_beams):]

        for pidx, beam_cells in new_beams:
            p = self.players[pidx]
            p.status = PlayerStatus.SHOOTING
            p.shoot_ticks_remaining = self.laser_duration
            p.move_cooldown_remaining = self.laser_duration
            self.active_beams.append((pidx, beam_cells, self.laser_duration))

        self._rebuild_beam_grid()

        # --- 3. Check beam kills ---
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
            # All eliminated players lose; the players who hit them win
            winners: set[int] = set()
            for i in eliminated:
                cell = (self.players[i].x, self.players[i].y)
                winners |= self.beam_grid[cell] - {i}
                self.players[i].alive = False

            for i in range(self.num_players):
                if i in eliminated:
                    reward_n[i] += -10.0
                    done_n[i] = True
                    info_n[i]["result"] = "lose"
                elif i in winners:
                    reward_n[i] += 10.0
                    done_n[i] = True
                    info_n[i]["result"] = "win"
                else:
                    done_n[i] = True  # episode ends for everyone

            # If anyone is eliminated, episode is over for all
            done_n = [True] * self.num_players

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
                            and not any(
                                op.x == nx and op.y == ny and op.alive
                                for j, op in enumerate(self.players) if j != i
                            )
                        ):
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
            for i, p in enumerate(self.players):
                if not p.alive:
                    continue
                cell = (p.x, p.y)
                if cell in self.beam_grid:
                    hitters = self.beam_grid[cell] - {i}
                    if hitters:
                        eliminated.add(i)

            if eliminated and not any(done_n):
                winners = set()
                for i in eliminated:
                    cell = (self.players[i].x, self.players[i].y)
                    winners |= self.beam_grid[cell] - {i}
                    self.players[i].alive = False
                for i in range(self.num_players):
                    if i in eliminated:
                        reward_n[i] += -10.0
                        info_n[i]["result"] = "lose"
                    elif i in winners:
                        reward_n[i] += 10.0
                        info_n[i]["result"] = "win"
                    done_n[i] = True
                done_n = [True] * self.num_players

        # --- 6. Spawn laser items ---
        if not any(done_n):
            self._try_spawn_laser_item()

        # --- 7. Truncation ---
        if not any(done_n) and self.tick >= self.max_episode_ticks:
            truncated_n = [True] * self.num_players

        # --- 8. Render ---
        if self.render_mode == "human":
            self.render()

        obs_n = [self._get_obs(i) for i in range(self.num_players)]
        return obs_n, reward_n, done_n, truncated_n, info_n

    # ------------------------------------------------------------------
    # Beam neutralization
    # ------------------------------------------------------------------

    def _neutralize_beams(
        self,
        beams: list[tuple[int, list[tuple[int, int]]]],
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        """Truncate beams at mutual intersection points.

        For any cell that appears in beams from different players, both beams
        are truncated so that the shared cell is the furthest extent in that
        direction. The intersection cell is kept in both beams (for visual
        merging / neutralization display).
        """
        if len(beams) < 2:
            return beams

        # Build cell → set of beam indices
        cell_owners: dict[tuple[int, int], set[int]] = {}
        for bidx, (pidx, cells) in enumerate(beams):
            for c in cells:
                cell_owners.setdefault(c, set()).add(bidx)

        # Find intersection cells (owned by beams of different players)
        conflict_cells: set[tuple[int, int]] = set()
        for cell, owners in cell_owners.items():
            if len(owners) >= 2:
                player_ids = {beams[b][0] for b in owners}
                if len(player_ids) >= 2:
                    conflict_cells.add(cell)

        if not conflict_cells:
            return beams

        # Truncate each beam's directional arms at the first conflict cell
        result: list[tuple[int, list[tuple[int, int]]]] = []
        for pidx, cells in beams:
            if not cells:
                result.append((pidx, cells))
                continue

            origin = cells[0]  # origin is always the shooter position
            # Group cells by direction from origin
            kept: set[tuple[int, int]] = {origin}
            for dx, dy in DIRECTION_DELTAS.values():
                x, y = origin[0] + dx, origin[1] + dy
                while (x, y) in set(cells):
                    kept.add((x, y))
                    if (x, y) in conflict_cells:
                        break  # keep this cell but stop extending
                    x += dx
                    y += dy
            result.append((pidx, [c for c in cells if c in kept]))
        return result

    # ------------------------------------------------------------------
    # Beam grid helper
    # ------------------------------------------------------------------

    def _rebuild_beam_grid(self):
        self.beam_grid = {}
        for pidx, cells, _ in self.active_beams:
            for c in cells:
                self.beam_grid.setdefault(c, set()).add(pidx)

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
        obs = np.zeros(len(self._num_states), dtype=np.int32)
        offset = 0

        def _write_player(p: _Player):
            nonlocal offset
            obs[offset] = p.x
            obs[offset + 1] = p.y
            obs[offset + 2] = p.move_cooldown_remaining
            obs[offset + 3] = int(p.status)
            offset += 4

        # Own player first
        _write_player(self.players[player_idx])
        # Other players
        for i, p in enumerate(self.players):
            if i != player_idx:
                _write_player(p)

        # Laser items
        for slot in range(self.max_laser_items):
            if slot < len(self.laser_items):
                ix, iy = self.laser_items[slot]
                obs[offset] = ix
                obs[offset + 1] = iy
                obs[offset + 2] = 1  # exists
            else:
                obs[offset] = 0
                obs[offset + 1] = 0
                obs[offset + 2] = 0
            offset += 3

        return obs

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode != "human":
            return
        if self._renderer is None:
            from renderer import MazerushRenderer
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
