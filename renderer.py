"""Pygame renderer for Mazerush."""

import pygame
import numpy as np

from mazerush_env import CellType, PlayerStatus


# Player colors (up to 4 players)
PLAYER_COLORS = [
    (66, 135, 245),   # blue
    (245, 100, 66),   # red-orange
    (80, 200, 80),    # green
    (200, 80, 200),   # purple
]

LASER_BEAM_COLORS = [
    (100, 180, 255),  # light blue
    (255, 150, 100),  # light orange
    (120, 255, 120),  # light green
    (255, 120, 255),  # light purple
]

LASER_ITEM_COLOR = (255, 220, 50)     # golden yellow
WALL_COLOR = (60, 60, 70)
EMPTY_COLOR = (30, 30, 40)
GRID_LINE_COLOR = (45, 45, 55)
NEUTRALIZED_COLOR = (255, 255, 255)   # white flash for neutralized beams

# Status indicator colors
STATUS_HAS_LASER_OUTLINE = (255, 255, 100)
STATUS_SHOOTING_OUTLINE = (255, 60, 60)


class MazerushRenderer:
    """Handles all Pygame drawing and event collection for Mazerush."""

    def __init__(self, env, window_width: int = 900, window_height: int = 640):
        pygame.init()
        self._env = env
        self._cell_w = window_width // env.width
        self._cell_h = window_height // env.height
        # Recompute actual window size to snap to cell grid
        self._win_w = self._cell_w * env.width
        self._win_h = self._cell_h * env.height
        self._screen = pygame.display.set_mode((self._win_w, self._win_h))
        pygame.display.set_caption("Mazerush")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("monospace", max(self._cell_h // 2, 10), bold=True)
        self._events: list[pygame.event.Event] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, env):
        """Draw the current environment state."""
        self._screen.fill(EMPTY_COLOR)

        cw, ch = self._cell_w, self._cell_h

        # 1. Draw grid cells (walls)
        for x in range(env.width):
            for y in range(env.height):
                rect = pygame.Rect(x * cw, y * ch, cw, ch)
                if env.grid[x, y] == CellType.WALL:
                    pygame.draw.rect(self._screen, WALL_COLOR, rect)
                else:
                    pygame.draw.rect(self._screen, GRID_LINE_COLOR, rect, 1)

        # 2. Draw laser beams
        for pidx, cells, _ in env.active_beams:
            for cx, cy in cells:
                rect = pygame.Rect(cx * cw, cy * ch, cw, ch)
                # Check for neutralization (multiple owners)
                owners = env.beam_grid.get((cx, cy), set())
                if len(owners) >= 2:
                    # Blend colors of all owners
                    colors = [LASER_BEAM_COLORS[o % len(LASER_BEAM_COLORS)] for o in owners]
                    avg = tuple(sum(c[i] for c in colors) // len(colors) for i in range(3))
                    beam_surf = pygame.Surface((cw, ch), pygame.SRCALPHA)
                    beam_surf.fill((*avg, 140))
                    self._screen.blit(beam_surf, rect)
                else:
                    beam_surf = pygame.Surface((cw, ch), pygame.SRCALPHA)
                    color = LASER_BEAM_COLORS[pidx % len(LASER_BEAM_COLORS)]
                    beam_surf.fill((*color, 100))
                    self._screen.blit(beam_surf, rect)

        # 3. Draw laser items
        for ix, iy in env.laser_items:
            center = (ix * cw + cw // 2, iy * ch + ch // 2)
            radius = min(cw, ch) // 3
            pygame.draw.circle(self._screen, LASER_ITEM_COLOR, center, radius)
            # Glow effect
            glow_surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*LASER_ITEM_COLOR, 50),
                               (radius * 2, radius * 2), radius * 2)
            self._screen.blit(glow_surf,
                              (center[0] - radius * 2, center[1] - radius * 2))

        # 4. Draw players
        for i, p in enumerate(env.players):
            if not p.alive:
                continue
            center = (p.x * cw + cw // 2, p.y * ch + ch // 2)
            radius = min(cw, ch) // 3
            color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
            pygame.draw.circle(self._screen, color, center, radius)

            # Status outline
            if p.status == PlayerStatus.HAS_LASER:
                pygame.draw.circle(self._screen, STATUS_HAS_LASER_OUTLINE,
                                   center, radius + 2, 2)
            elif p.status == PlayerStatus.SHOOTING:
                pygame.draw.circle(self._screen, STATUS_SHOOTING_OUTLINE,
                                   center, radius + 2, 3)

            # Player number label
            label = self._font.render(str(i + 1), True, (255, 255, 255))
            lrect = label.get_rect(center=center)
            self._screen.blit(label, lrect)

        # 5. HUD: tick counter
        hud = self._font.render(f"Tick: {env.tick}", True, (200, 200, 200))
        self._screen.blit(hud, (5, 5))

        pygame.display.flip()
        self._clock.tick(env.fps)

    def get_events(self) -> list[pygame.event.Event]:
        """Pump and return all pygame events."""
        events = list(pygame.event.get())
        return events

    def close(self):
        pygame.quit()
