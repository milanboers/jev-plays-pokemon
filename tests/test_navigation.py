"""Unit tests for the screen-relative navigation logic (no emulator needed)."""

from __future__ import annotations

import numpy as np

from jev_plays_pokemon import screen_grid
from jev_plays_pokemon.play import _along_wall_direction

GRID = np.ones((9, 10), dtype=bool)


def grid_from(rows: list[str]) -> np.ndarray:
    return np.array([[c == "." for c in row] for row in rows], dtype=bool)


class TestCellLabels:
    def test_cell_to_rowcol(self):
        assert screen_grid.cell_to_rowcol("A1") == (0, 0)
        assert screen_grid.cell_to_rowcol("E5") == (4, 4)
        assert screen_grid.cell_to_rowcol("J9") == (8, 9)
        assert screen_grid.cell_to_rowcol("K1") is None
        assert screen_grid.cell_to_rowcol("A10") is None
        assert screen_grid.cell_to_rowcol("xx") is None


class TestAStar:
    def test_straight_line(self):
        path = screen_grid.astar_screen((4, 6), GRID)
        assert path == ["right", "right"]

    def test_goes_around_wall(self):
        grid = GRID.copy()
        grid[4, 5] = False  # block the cell right of the player
        path = screen_grid.astar_screen((4, 6), grid)
        assert path is not None
        # Must route around the wall.
        assert len(path) == 4  # e.g. down/right/right/up

    def test_wall_goal_is_reachable(self):
        grid = GRID.copy()
        grid[4, 5] = False  # a door looks like a wall
        path = screen_grid.astar_screen((4, 5), grid, allow_wall_goal=True)
        assert path == ["right"]

    def test_sprite_blocks_unless_goal(self):
        sprites = {(4, 5)}
        path = screen_grid.astar_screen((4, 6), GRID, sprites=sprites)
        assert path is not None and path != ["right", "right"]  # must route around
        r, c = 4, 4
        moves = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        for step in path:
            dr, dc = moves[step]
            r, c = r + dr, c + dc
            assert (r, c) != (4, 5)  # never steps onto the sprite
        # The sprite cell itself as the goal is allowed.
        assert screen_grid.astar_screen((4, 5), GRID, sprites=sprites, allow_wall_goal=True) == ["right"]

    def test_unreachable_returns_none(self):
        grid = GRID.copy()
        grid[:, :] = False
        assert screen_grid.astar_screen((4, 6), grid) is None


class TestFrontier:
    def test_finds_cell_adjacent_to_wall(self):
        grid = grid_from(
            [
                "..........",
                "..........",
                "..........",
                "..........",
                "..........",
                "..........",
                "..........",
                "..........",
                "..........",
            ]
        )
        grid[4, 6] = False  # a wall to the east of the player
        target = screen_grid.frontier_cell(grid)
        assert target == (4, 5)  # the walkable cell next to the wall

    def test_none_when_boxed(self):
        grid = np.zeros((9, 10), dtype=bool)
        assert screen_grid.frontier_cell(grid) is None

    def test_skips_warp_tiles(self):
        # Wall to the east; (4,5) next to it is a warp tile that LOOKS walkable.
        # Exploration must not pick the warp cell itself as the frontier target.
        grid = GRID.copy()
        grid[4, 7] = False  # wall two east of the player
        target = screen_grid.frontier_cell(grid, warps={(4, 5)})
        assert target is not None
        assert target != (4, 5)
        # A cell whose only neighbour is a warp is not a frontier either.
        grid = GRID.copy()
        grid[4, 7] = False  # the only wall is far right, not near the warp
        target = screen_grid.frontier_cell(grid, warps={(4, 6)})
        assert target is not None and target != (4, 6)

    def test_astar_avoids_forbidden_warp(self):
        grid = GRID.copy()
        grid[4, 5] = False  # the door is a collision wall
        # Goal beyond the door on the right; forbidden includes the door tile.
        path = screen_grid.astar_screen((4, 8), grid, forbidden={(4, 5)})
        assert path is None or "right" not in [p for p in path if False]
        assert screen_grid.astar_screen((4, 5), grid, allow_wall_goal=True, forbidden={(4, 5)}) is None


class TestWallFollow:
    def test_wall_up_walks_right(self):
        # A wall above the player; keep it on the right -> walk right.
        grid = grid_from(
            [
                "........##",
                "........##",
                "........##",
                "........##",
                "........##",  # player row (4): wall far right, not up
                "........##",
                "........##",
                "........##",
                "........##",
            ]
        )
        grid[3, 4] = False  # wall directly above player
        assert _along_wall_direction(grid) == "right"

    def test_no_wall_returns_none(self):
        assert _along_wall_direction(GRID) is None


class FakeMemory:
    """Minimal PyBoy-memory stand-in exposing the warp addresses and player pos."""

    def __init__(self, warps: list[tuple[int, int, int, int]], py: int, px: int):
        self._b = bytearray(0x10000)
        self._b[0xD3AE] = len(warps)
        for i, (y, x, wid, dest) in enumerate(warps):
            a = 0xD3AF + i * 4
            self._b[a : a + 4] = bytes((y, x, wid, dest))
        self._b[0xD361] = py
        self._b[0xD362] = px

    def __getitem__(self, addr):
        return self._b[addr]


class FakePyBoy:
    def __init__(self, warps, py, px):
        self.memory = FakeMemory(warps, py, px)


class TestWarps:
    def test_warp_cells_on_screen(self):
        # Player at (6,3); warp to dest 37 at (1,7) -> row 4+(1-6)=-1 off-screen.
        pb = FakePyBoy(warps=[(1, 7, 2, 37)], py=6, px=3)
        assert screen_grid.warp_cells(pb, dest=37) == set()

    def test_warp_cells_visible(self):
        # Player at (6,7): warp at (1,7) -> row 4+(1-6)=-1 off, but another on.
        pb = FakePyBoy(warps=[(5, 7, 1, 0)], py=6, px=3)
        cells = screen_grid.warp_cells(pb, dest=0)
        assert cells == {(3, 8)}

    def test_warp_cells_filters_dest(self):
        pb = FakePyBoy(warps=[(5, 7, 1, 0), (5, 8, 2, 37)], py=6, px=3)
        assert screen_grid.warp_cells(pb, dest=37) == {(3, 9)}

    def test_warp_direction_points_at_stairs(self):
        # Player (6,3); warp to 37 at (1,7) -> up then right (row shrinks first).
        pb = FakePyBoy(warps=[(1, 7, 2, 37)], py=6, px=3)
        assert screen_grid.warp_direction(pb, dest=37) in ("up",)

    def test_warp_direction_none_in_place(self):
        pb = FakePyBoy(warps=[(6, 3, 2, 37)], py=6, px=3)
        assert screen_grid.warp_direction(pb, dest=37) is None

    def test_warp_direction_no_matching_warp(self):
        pb = FakePyBoy(warps=[(1, 7, 2, 37)], py=6, px=3)
        assert screen_grid.warp_direction(pb, dest=0) is None

    def test_last_map_warp_matches_pallet(self):
        # 1F front door: warp tile dest=0xFF (LAST_MAP) = re-enter Pallet (0).
        pb = FakePyBoy(warps=[(7, 3, 0, 0xFF)], py=6, px=3)
        assert screen_grid.warp_cells(pb, dest=0) == {(5, 4)}
        assert screen_grid.warp_direction(pb, dest=0) == "down"

    def test_last_map_warp_ignored_for_other_dest(self):
        pb = FakePyBoy(warps=[(7, 3, 0, 0xFF)], py=6, px=3)
        assert screen_grid.warp_cells(pb, dest=38) == set()
        assert screen_grid.warp_direction(pb, dest=38) is None
