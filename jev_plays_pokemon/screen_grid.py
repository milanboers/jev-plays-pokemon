"""Screen-relative navigation.

The player is ALWAYS at grid cell (4, 4) of the on-screen 9x10 walkability
grid (the Game Boy screen scrolls around the player). Navigation therefore
does NOT depend on the unreliable absolute map coordinates (wXCoord/wYCoord):
we read the live on-screen collision buffer each step, pathfind toward a
screen-relative target, and let the game itself confirm walkability.

This is the approach proven in ClaudePlaysPokemonStarter: allow a wall tile as
the path target (doors are walls that warp when stepped onto), treat NPC
sprites as obstacles unless they are the target, and re-verify after every
single step.
"""

from __future__ import annotations

import heapq

import numpy as np

PLAYER = (4, 4)  # (row, col) in the 9x10 grid - the player is always here
_ROWS, _COLS = 9, 10
# Active warp list for the current map (RAM, pokered wNumberOfWarps / wWarpEntries).
# Each entry is 4 bytes: Y, X, warp id, destination map id - in the same
# block-coordinate space as wYCoord/wXCoord (0xD361/0xD362).
_W_NUMBER_OF_WARPS = 0xD3AE
_W_WARP_ENTRIES = 0xD3AF
_DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
# The player's own 2x2 sprite occupies these four screen cells.
_PLAYER_AREA = {(4, 4), (4, 5), (5, 4), (5, 5)}

# Door/exit tile ids per tileset, from pokered data/tilesets/door_tile_ids.asm.
# Tileset ids follow pokered's Tileset enum (0=Overworld, 1=RedsHouse1, ...).
DOOR_TILES: dict[int, frozenset[int]] = {
    0x00: frozenset({0x1B, 0x58}),  # Overworld
    0x01: frozenset({0x1A, 0x1C}),  # RedsHouse1
    0x04: frozenset({0x1A, 0x1C}),  # RedsHouse2
    0x02: frozenset({0x5E}),  # Mart / Pokecenter
    0x03: frozenset({0x3A}),  # Forest
    0x08: frozenset({0x54}),  # House
    0x09: frozenset({0x3B}),  # ForestGate
    0x0A: frozenset({0x3B}),  # Museum
    0x0C: frozenset({0x3B}),  # Gate
    0x0D: frozenset({0x1E}),  # Ship
    0x12: frozenset({0x1C, 0x38, 0x1A}),  # Lobby
    0x13: frozenset({0x1A, 0x1C, 0x53}),  # Mansion
    0x14: frozenset({0x34}),  # Lab
    0x16: frozenset({0x43, 0x58, 0x1B}),  # Facility
    0x17: frozenset({0x3B, 0x1B}),  # Plateau
}


def door_cells(pb, tileset: int) -> set[tuple[int, int]]:
    """Screen cells that contain a door/exit tile on the current screen.

    Doors are specific background tiles that warp when stepped onto; scanning
    for them lets the agent pathfind directly to an exit instead of probing.
    """
    ids = DOOR_TILES.get(tileset)
    if not ids:
        return set()
    area = np.asarray(pb.game_wrapper.game_area())  # 18x20 tile ids
    cells: set[tuple[int, int]] = set()
    for row in range(_ROWS):
        for col in range(_COLS):
            block = area[2 * row : 2 * row + 2, 2 * col : 2 * col + 2]
            if any(int(t) in ids for t in block.flat):
                cells.add((row, col))
    return cells


def warp_entries(pb) -> list[tuple[int, int, int, int]]:
    """Active warps: [(y, x, warp_id, dest_map)] for the current map."""
    count = pb.memory[_W_NUMBER_OF_WARPS]
    out: list[tuple[int, int, int, int]] = []
    for i in range(count):
        addr = _W_WARP_ENTRIES + i * 4
        y, x, warp_id, dest = (pb.memory[addr + j] for j in range(4))
        out.append((y, x, warp_id, dest))
    return out


def warp_exits(pb) -> list[tuple[int, int, int]]:
    """(x, y, dest_map) for the current map's live warp list (stateless)."""
    return [(x, y, dest) for y, x, _warp_id, dest in warp_entries(pb)]


def _warp_matches(dest: int | None, m: int) -> bool:
    """Does warp destination ``m`` lead to ``dest``?

    ``0xFF`` is LAST_MAP: \"warp back to whichever map we entered from\", which
    for the tutorial rooms is Pallet Town (map 0).
    """
    if dest is None:
        return True
    if m == dest:
        return True
    return m == 0xFF and dest == 0


def warp_cells(pb, dest: int | None = None) -> set[tuple[int, int]]:
    """On-screen cells whose map tile is a warp to ``dest`` (or any if None)."""
    py = pb.memory[0xD361]  # player Y (block coords, same space as warps)
    px = pb.memory[0xD362]
    cells: set[tuple[int, int]] = set()
    for y, x, _warp_id, m in warp_entries(pb):
        if not _warp_matches(dest, m):
            continue
        row, col = 4 + (y - py), 4 + (x - px)
        if 0 <= row < _ROWS and 0 <= col < _COLS:
            cells.add((row, col))
    return cells


def walkable_grid(pb) -> np.ndarray:
    """9x10 walkable grid (1 = passable) with the player at (4,4)."""
    collision = np.asarray(pb.game_wrapper.game_area_collision())  # 18x20
    return collision[::2, ::2].astype(bool)


def sprite_cells(pb) -> set[tuple[int, int]]:
    """Screen cells occupied by non-player sprites (NPCs, items, enemies)."""
    cells: set[tuple[int, int]] = set()
    for i in range(40):
        sprite = pb.get_sprite(i)
        if not sprite.on_screen:
            continue
        col = (sprite.x + 8) // 16
        row = (sprite.y + 16) // 16
        if 0 <= col < _COLS and 0 <= row < _ROWS and (row, col) not in _PLAYER_AREA:
            cells.add((row, col))
    return cells


def astar_screen(
    goal: tuple[int, int],
    walkable: np.ndarray,
    sprites: set[tuple[int, int]] | None = None,
    allow_wall_goal: bool = True,
    forbidden: set[tuple[int, int]] | None = None,
) -> list[str] | None:
    """A* from PLAYER to ``goal`` on the 9x10 grid.

    Walls and sprites are impassable EXCEPT the goal itself (the player can
    step onto a door/NPC tile). ``forbidden`` cells are never passable (e.g.
    warp tiles while exploring - stepping onto one exits the room). Returns
    the list of directions or ``None``.
    """
    sprites = sprites or set()
    forbidden = forbidden or set()

    def block(row: int, col: int) -> bool:
        return not (0 <= row < _ROWS and 0 <= col < _COLS)

    def passable(row: int, col: int) -> bool:
        if (row, col) in forbidden:
            return False
        if (row, col) == goal and allow_wall_goal:
            return True
        if block(row, col) or not walkable[row, col]:
            return False
        return (row, col) not in sprites

    start = PLAYER
    if start == goal:
        return []
    heap: list[tuple[int, tuple[int, int]]] = [(0, start)]
    g = {start: 0}
    prev: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}

    def h(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while heap:
        _, cur = heapq.heappop(heap)
        if cur == goal:
            path: list[str] = []
            while cur != start:
                cur, direction = prev[cur]
                path.append(direction)
            return path[::-1]
        for direction, (dr, dc) in _DIRS.items():
            nxt = (cur[0] + dr, cur[1] + dc)
            if not passable(nxt[0], nxt[1]):
                continue
            ng = g[cur] + 1
            if ng < g.get(nxt, 1 << 30):
                g[nxt] = ng
                prev[nxt] = (cur, direction)
                heapq.heappush(heap, (ng + h(nxt, goal), nxt))
    return None


def cell_to_rowcol(cell: str) -> tuple[int, int] | None:
    """Convert an A1..J9 screen-cell label to (row, col), or None."""
    if len(cell) < 2:
        return None
    col = ord(cell[0].upper()) - ord("A")
    if not cell[1:].isdigit():
        return None
    row = int(cell[1:]) - 1
    if 0 <= row < _ROWS and 0 <= col < _COLS:
        return (row, col)
    return None


def warp_direction(pb, dest: int) -> str | None:
    """Best walk direction to shrink map-space distance to a warp to ``dest``.

    This is the map-space analog of "head toward the exit": compute the warp's
    offset from the player and pick the axis to close first. None when no warp
    leads to ``dest`` or the player is already standing on it.
    """
    py = pb.memory[0xD361]
    px = pb.memory[0xD362]
    best: tuple[str, int] | None = None
    for y, x, _warp_id, m in warp_entries(pb):
        if not _warp_matches(dest, m):
            continue
        dy, dx = y - py, x - px
        if dy == dx == 0:
            continue
        for direction, (dr, dc) in _DIRS.items():
            d = abs(dy - dr) + abs(dx - dc)
            if best is None or d < best[1]:
                best = (direction, d)
    return best[0] if best else None


def frontier_cell(
    walkable: np.ndarray,
    sprites: set[tuple[int, int]] | None = None,
    warps: set[tuple[int, int]] | None = None,
    visited: set[tuple[int, int]] | None = None,
) -> tuple[int, int] | None:
    """Nearest walkable cell that borders a wall or sprite (an unexplored area).

    ``warps`` are the live warp tiles (from RAM): door/stairs tiles LOOK
    walkable in the collision map but stepping onto them triggers a map
    transition, so exploration must never target them - otherwise the agent
    walks straight back out through the door it is supposed to be exploring
    around. The warp tiles themselves are also excluded as borders (a cell
    whose only neighbour is a door is not worth walking to).

    ``visited`` are cells the player recently stood on (from the walker's
    movement history): exploration skips them so it keeps advancing into
    genuinely new space instead of ping-ponging between two wall-adjacent
    cells right next to the player.
    """
    sprites = sprites or set()
    warps = warps or set()
    visited = visited or set()
    best: tuple[tuple[int, int], int] | None = None
    for row in range(_ROWS):
        for col in range(_COLS):
            if (row, col) == PLAYER or not walkable[row, col] or (row, col) in sprites:
                continue
            if (row, col) in warps:
                continue  # stepping onto a warp exits the room; never explore onto it
            if (row, col) in visited:
                continue  # just stood there; don't go straight back
            for dr, dc in _DIRS.values():
                nr, nc = row + dr, col + dc
                if not (0 <= nr < _ROWS and 0 <= nc < _COLS):
                    continue
                if (nr, nc) in warps:
                    continue  # a door is not a wall worth exploring up to
                if not walkable[nr, nc] or (nr, nc) in sprites:
                    dist = abs(row - PLAYER[0]) + abs(col - PLAYER[1])
                    if best is None or dist < best[1]:
                        best = ((row, col), dist)
                    break
    return best[0] if best else None
