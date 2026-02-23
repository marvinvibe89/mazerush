"""Pygame renderer for Mazerush."""

import pygame
import numpy as np

from mazerush_utils import CellType, PlayerStatus


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
        self.render_mode = env.render_mode
        
        if self.render_mode == "human":
            self._view_w = env.fov_size
            self._view_h = env.fov_size
        else:
            self._view_w = env.width
            self._view_h = env.height

        self._cell_w = window_width // self._view_w
        self._cell_h = window_height // self._view_h
        # Recompute actual window size to snap to cell grid
        self._win_w = self._cell_w * self._view_w
        self._win_h = self._cell_h * self._view_h
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
        
        if self.render_mode == "human":
            p0 = env.players[0]
            offset_x = p0.x - env.fov_radius
            offset_y = p0.y - env.fov_radius
            draw_range_x = range(offset_x, offset_x + env.fov_size)
            draw_range_y = range(offset_y, offset_y + env.fov_size)
        else:
            offset_x, offset_y = 0, 0
            draw_range_x = range(env.width)
            draw_range_y = range(env.height)

        # 1. Draw grid cells (walls)
        for x in draw_range_x:
            for y in draw_range_y:
                screen_x = (x - offset_x) * cw
                screen_y = (y - offset_y) * ch
                rect = pygame.Rect(screen_x, screen_y, cw, ch)
                
                if 0 <= x < env.width and 0 <= y < env.height:
                    if env.grid[x, y] == CellType.WALL:
                        pygame.draw.rect(self._screen, WALL_COLOR, rect)
                    else:
                        pygame.draw.rect(self._screen, GRID_LINE_COLOR, rect, 1)
                else:
                    # Out of bounds area for render_human
                    pygame.draw.rect(self._screen, WALL_COLOR, rect)

        # 1.5 Draw FOV highlights for render_human_full
        if self.render_mode == "human_full":
            for i, p in enumerate(env.players):
                if not p.alive: continue
                # Draw a soft highlight over the 9x9 FOV in the player's color
                fov_rect = pygame.Rect(
                    (p.x - env.fov_radius) * cw,
                    (p.y - env.fov_radius) * ch,
                    env.fov_size * cw,
                    env.fov_size * ch
                )
                highlight_surf = pygame.Surface(fov_rect.size, pygame.SRCALPHA)
                color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
                highlight_surf.fill((*color, 10)) # subtle player color
                self._screen.blit(highlight_surf, fov_rect)

        # 2. Draw laser beams
        for pidx, cells, _ in env.active_beams:
            for cx, cy in cells:
                if cx not in draw_range_x or cy not in draw_range_y:
                    continue
                
                rect = pygame.Rect((cx - offset_x) * cw, (cy - offset_y) * ch, cw, ch)
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
            if ix not in draw_range_x or iy not in draw_range_y:
                continue
                
            center = ((ix - offset_x) * cw + cw // 2, (iy - offset_y) * ch + ch // 2)
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
            if p.x not in draw_range_x or p.y not in draw_range_y:
                continue
                
            center = ((p.x - offset_x) * cw + cw // 2, (p.y - offset_y) * ch + ch // 2)
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
        if self.render_mode == "human_full":
            hud_text = f"Tick: {env.tick}"
        elif self.render_mode == "human":
            hud_text = f"Tick: {env.tick} | Pos: (x={env.players[0].x}, y={env.players[0].y})"
        hud = self._font.render(hud_text, True, (200, 200, 200))
        self._screen.blit(hud, (5, 5))

        pygame.display.flip()
        self._clock.tick(env.fps)

    def get_events(self) -> list[pygame.event.Event]:
        """Pump and return all pygame events."""
        events = list(pygame.event.get())
        return events

    def close(self):
        pygame.quit()
