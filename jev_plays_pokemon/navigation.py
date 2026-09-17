"""A* pathfinding over the merged collision/room grid.

The model picks high-level goals ("talk to Mom", "leave the room"); this
module computes the actual button presses needed to walk there. ``walkable``
is a callback that answers "can the player step onto map cell (x, y)?" using
the live on-screen collision grid where available and the accumulated room
map elsewhere.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable

DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
REVERSE = {"up": "down", "down": "up", "left": "right", "right": "left"}

# The on-screen collision grid is 9 rows x 10 cols with the player locked to
# block E5 (col 4, row 4).
PLAYER_COL = 4
PLAYER_ROW = 4


def astar(
    start: tuple[int, int],
    goal: tuple[int, int],
    walkable: Callable[[int, int], bool],
    max_steps: int = 200,
) -> list[str] | None:
    """Return the list of direction strings from ``start`` to ``goal``.

    The goal cell itself is always treated as passable (it is an NPC, item,
    door or exit the player wants to reach/step onto). Returns ``None`` if
    no path is found within ``max_steps`` moves.
    """
    if start == goal:
        return []

    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    heap: list[tuple[int, tuple[int, int]]] = [(0, start)]
    cost = {start: 0}
    came_from: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
    visited = 0
    while heap and visited < max_steps:
        _, current = heapq.heappop(heap)
        visited += 1
        if current == goal:
            path: list[str] = []
            while current != start:
                current, direction = came_from[current]
                path.append(direction)
            return path[::-1]
        for direction, (dx, dy) in DIRECTIONS.items():
            nxt = (current[0] + dx, current[1] + dy)
            if nxt != goal and not walkable(nxt[0], nxt[1]):
                continue
            new_cost = cost[current] + 1
            if new_cost < cost.get(nxt, 1 << 30):
                cost[nxt] = new_cost
                came_from[nxt] = (current, direction)
                heapq.heappush(heap, (new_cost + heuristic(nxt, goal), nxt))
    return None


def make_walkable(
    collision: dict, player_x: int, player_y: int, room_cells: dict
) -> Callable[[int, int], bool]:
    """Build a ``walkable(x, y)`` callback for map coordinates.

    Cells visible on the current screen use the live collision grid; anything
    off-screen falls back to the room map (unknown cells are treated as
    blocked so we only route through confirmed walkable floor).
    """
    grid = collision.get("walkable") or [[True] * 10 for _ in range(9)]

    def walkable(mx: int, my: int) -> bool:
        col = mx - player_x + PLAYER_COL
        row = my - player_y + PLAYER_ROW
        if 0 <= col < 10 and 0 <= row < 9:
            return bool(grid[row][col])
        return room_cells.get((mx, my)) == "walkable"

    return walkable


def manhattan_away(start: tuple[int, int], target: tuple[int, int]) -> str | None:
    """The single best first move to reduce distance to ``target``.

    Returns one of up/down/left/right, or None when already on the target.
    """
    dx = target[0] - start[0]
    dy = target[1] - start[1]
    if dx == 0 and dy == 0:
        return None
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"
