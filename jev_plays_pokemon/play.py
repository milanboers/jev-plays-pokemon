"""Mechanical execution of the high-level goal Jev chose.

Jev decides WHAT to do (a goal id); this module figures out the button
presses to actually do it: A* pathfinding toward NPCs and exits, dialog
advancing, and boundary probing. It hands control back to Jev whenever the
situation meaningfully changes (dialog opened/cleared, new object, map
transition).
"""

from __future__ import annotations

from typing import Any

from jev_plays_pokemon.agent import ACTION_FRAMES
from jev_plays_pokemon.navigation import astar, make_walkable
from jev_plays_pokemon.stop import stop_requested
from jev_plays_pokemon.vendor.collision import build_collision_grid

_BUTTON = {
    "press_a": "a",
    "press_b": "b",
    "press_start": "start",
    "press_select": "select",
    "walk_up": "up",
    "walk_down": "down",
    "walk_left": "left",
    "walk_right": "right",
}


def press(emu, action: str) -> None:
    if stop_requested():
        return
    frames, settle = ACTION_FRAMES[action]
    button = _BUTTON.get(action)
    if frames and button is not None:
        emu.press(button, frames)
    emu.tick(settle)


def player_map(builder) -> tuple[int, int]:
    pos = (builder.reader.read_player() or {}).get("position") or {}
    return pos.get("x"), pos.get("y")


def _walk_step(emu, builder, direction: str) -> bool:
    """Press a direction; return True if the player moved."""
    if builder.dialog_active:
        return False  # a text box locks all movement input
    before = player_map(builder)
    press(emu, f"walk_{direction}")
    after = player_map(builder)
    moved = before != after
    import logging

    logging.getLogger("jev_plays_pokemon").debug(
        "step %s: %s -> %s moved=%s", direction, before, after, moved
    )
    builder.record_move(f"walk_{direction}", moved)
    if moved:
        builder.move_history.append(after)
    else:
        _log_bump(builder, before, direction)
    return moved


def _log_bump(builder, pos: tuple[int, int], direction: str, locked: bool = False) -> None:
    """Debug: why didn't the player move? Dump the collision neighbours."""
    import logging

    from jev_plays_pokemon.vendor.collision import build_collision_grid

    log = logging.getLogger("jev_plays_pokemon")
    grid = build_collision_grid(builder.reader.emu).get("walkable")
    px, py = pos
    cells = builder.room_map.cells.get(builder.reader.read_map_info()["map_id"], {})
    open_dirs = []
    for d, (dx, dy) in (("up", (0, -1)), ("down", (0, 1)), ("left", (-1, 0)), ("right", (1, 0))):
        col, row = px + dx - (px) + 4, py + dy - (py) + 4
        on_screen = 0 <= col < 10 and 0 <= row < 9
        collision_ok = bool(grid[row][col]) if on_screen else "?"
        room = cells.get((px + dx, py + dy))
        open_dirs.append(f"{d}:collision={collision_ok} room={room}")
    log.debug(
        "bump at %s pressing %s%s | %s",
        pos,
        direction,
        " (input locked)" if locked else "",
        " ".join(open_dirs),
    )


def _walkable_fn(builder, state: dict[str, Any]):
    px, py = player_map(builder)
    map_id = state["location"]["map_id"]
    collision = build_collision_grid(builder.reader.emu)
    cells = builder.room_map.cells.get(map_id, {}) if map_id is not None else {}
    return make_walkable(collision, px, py, cells)


def _path_to(emu, builder, state, target: tuple[int, int]) -> list[str] | None:
    px, py = player_map(builder)
    return astar((px, py), target, _walkable_fn(builder, state))


def _path_to_adjacent(emu, builder, state, target: tuple[int, int]) -> list[str] | None:
    """Shortest path to any of the four cells adjacent to ``target``."""
    px, py = player_map(builder)
    best = None
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cell = (target[0] + dx, target[1] + dy)
        if cell == (px, py):
            return []
        path = astar((px, py), cell, _walkable_fn(builder, state))
        if path is not None and (best is None or len(path) < len(best)):
            best = path
    return best


def _room_walkable(builder, state: dict[str, Any]):
    """Walkable callback over the physically-verified room map.

    Only tiles we have stood on / observed as walkable are passable (plus
    known door/stairs warp tiles). This is the ground truth A*/BFS routes on;
    unexplored floor is NOT walkable here, so paths stay on confirmed ground.
    """
    map_id = state["location"]["map_id"]
    cells = builder.room_map.cells.get(map_id, {}) if map_id is not None else {}
    warp_tiles = {(w[0], w[1]) for w in builder.live_exits()}

    def walkable(mx: int, my: int) -> bool:
        if (mx, my) in warp_tiles:
            return True
        return cells.get((mx, my)) == "walkable"

    return walkable


def _astar_to(emu, builder, state, target: tuple[int, int]) -> list[str] | None:
    px, py = player_map(builder)
    return astar((px, py), target, _room_walkable(builder, state))


def _optimistic_walkable(builder, state: dict[str, Any]):
    """Walkable callback for routing.

    Collision "blocked" is NOT fully trusted: the game's collision data marks
    doors, stairs, map edges and (at some screen positions) entire walkable
    blocks as walls. Warp/exit tiles are always passable (you step onto them to
    exit); everything else is optimistic-passable so the walker is never
    paralyzed by a single bad collision reading.
    """
    map_id = state["location"]["map_id"]
    cells = builder.room_map.cells.get(map_id, {}) if map_id is not None else {}
    # Door/stairs tiles look like solid walls in collision data but you can
    # stand on them (stepping onto them triggers the exit) - always passable.
    # Stateless: from the live RAM warp table each frame.
    warp_tiles = {(w[0], w[1]) for w in builder.live_exits()}

    def walkable(mx: int, my: int) -> bool:
        if (mx, my) in warp_tiles:
            return True
        if cells.get((mx, my)) == "blocked":
            return False  # the live collision data reports it blocked
        return True  # optimistic: assume passable until proven otherwise

    return walkable


def _force_wiggle(emu, builder) -> bool:
    """Force-try all four directions in a random order.

    The game itself is the ground truth for walkability: a transient block
    (NPC/sprite) will yield when it moves; a real wall just bumps. Shuffling
    the order means a repeated stuck sequence does not keep bumping the same
    wall first every time - there is always a (stochastic) way out.
    Returns True if the player moved at all.
    """
    import random

    player_map(builder)
    for direction in random.sample(("up", "down", "left", "right"), 4):
        if _walk_step(emu, builder, direction):
            return True
    return False


def _greedy_toward(emu, builder, state, target: tuple[int, int] | None = None) -> bool:
    """Take one step toward ``target`` (or any step when ``target`` is None).

    Prefers a direction that does NOT immediately reverse the previous step
    (that caused endless two-tile oscillation), then the best progress, then a
    fresh (unvisited) tile. Tries candidates in order and takes the first that
    actually moves. Returns True if the player moved.
    """
    px, py = player_map(builder)
    if target is None:
        tx = ty = None
    else:
        tx, ty = target
    walkable = _optimistic_walkable(builder, state)
    last_dir = builder._last_dir
    reverse = {"up": "down", "down": "up", "left": "right", "right": "left"}.get(last_dir)
    moves = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }
    scored = []
    for direction, (dx, dy) in moves.items():
        nx, ny = px + dx, py + dy
        if target is None:
            progress = 0
        else:
            progress = abs(px - tx) + abs(py - ty) - (abs(nx - tx) + abs(ny - ty))
        if walkable(nx, ny):
            scored.append((progress, direction, (nx, ny) not in builder.move_history, direction == reverse))
    if not scored:
        return False
    # (not-reverse, progress, fresh) - never reverse while any other move helps.
    scored.sort(key=lambda s: (not s[3], s[0], s[2]), reverse=True)
    # Try candidates in order and take the first that actually moves - a tile
    # can look passable but be blocked by an NPC or a map quirk.
    for _, direction, _, _ in scored:
        if _walk_step(emu, builder, direction):
            return True
    return False


def execute_goal(emu, builder, goal_id: str, state: dict[str, Any], max_steps: int = 22) -> str:
    if stop_requested():
        return "interrupted"
    # A text box / script lock makes the game ignore ALL movement input. If
    # one is open, don't pathfind - mash A through it first.
    if builder.dialog_active and goal_id != "advance_text":
        return _advance_text(emu, builder, 6)
    if goal_id == "advance_text":
        return _advance_text(emu, builder, max_steps)
    if goal_id == "wait":
        press(emu, "wait")
        return "waited"
    if goal_id.startswith("talk_to_"):
        index = int(goal_id.rsplit("_", 1)[1])
        return _talk_to(emu, builder, state, index, max_steps)
    if goal_id == "reach_exit":
        return _reach_exit(emu, builder, state, max_steps, dest=None)
    if goal_id.startswith("exit_to_"):
        dest = int(goal_id.removeprefix("exit_to_"))
        return _reach_exit(emu, builder, state, max_steps, dest=dest)
    if goal_id == "explore":
        return _explore(emu, builder, state, max_steps)
    return "no-op"


def _is_nickname_prompt(text: str, hints) -> bool:
    """Is the open box the "give a nickname?" YES/NO prompt?

    The model cannot reliably tell this two-option menu from a plain dialog,
    and mashing A would confirm YES and open the name-entry keyboard. Detect
    it by text and decline with B (keeps the default species name).
    """
    lowered = text.lower()
    if "nickname" in lowered or "give a nickname" in lowered:
        return True
    return any("nickname" in h.lower() for h in hints)


def _advance_text(emu, builder, max_steps: int) -> str:
    from jev_plays_pokemon.screen_text import read_screen_text
    from jev_plays_pokemon.stop import stop_requested

    for _ in range(max_steps):
        if stop_requested():
            return "interrupted"
        if not builder.dialog_active:
            return "dialog cleared"
        pb = emu._pyboy
        # A "give a nickname?" YES/NO prompt must be declined deterministically
        # (press B = keep the default species name). The model cannot reliably
        # tell this two-option menu from a plain dialog, and mashing A would
        # confirm YES and open the name-entry keyboard, stranding the agent.
        text_now = " ".join(line.strip() for line in read_screen_text(pb) if line.strip())
        if _is_nickname_prompt(text_now, builder.hints):
            press(emu, "press_b")
            return "nickname prompt declined"
        # Pause inputs and sample until the box is fully typed. Mashing A every
        # frame races the render, so the box never settles and text decodes
        # garbled. Wait for the ▼ arrow: the engine only draws it (0xEE) in
        # WaitForTextScrollButtonPress - i.e. the page printed fully and is
        # waiting for input. It blinks, so poll a couple of frames. Capture the
        # complete page as a hint, THEN press A to advance.
        captured = False
        for _sample in range(12):
            pb.tick(2)
            if not builder.dialog_active:
                break
            if builder.box_ready:
                text = " ".join(line.strip() for line in read_screen_text(pb) if line.strip())
                builder.capture_text_box([text], True, stable=True)
                captured = True
                break
        if not captured and builder.dialog_active:
            # Box never showed ▼ (e.g. a one-frame script lock); still advance.
            pass
        press(emu, "press_a")
    return "text mashing done"


def _move_toward(emu, builder, target: tuple[int, int], forbidden=None) -> bool:
    """Screen-relative step toward a target cell (player is always (4,4)).

    Pathfinds one step on the live on-screen collision grid (walls and NPCs
    passable only as the goal), and falls back to an empirical probe when the
    plan fails - the game itself is the ground truth. ``forbidden`` cells
    (e.g. warp tiles when exploring) are never stepped onto.
    """
    from jev_plays_pokemon import screen_grid

    pb = builder.reader.emu._pyboy
    walkable = screen_grid.walkable_grid(pb)
    sprites = screen_grid.sprite_cells(pb)
    path = screen_grid.astar_screen(target, walkable, sprites, allow_wall_goal=True, forbidden=forbidden)
    if path and _walk_step(emu, builder, path[0]):
        return True
    return _force_wiggle(emu, builder)


def _move_adjacent_to(emu, builder, target: tuple[int, int]) -> bool:
    """Screen-relative step toward a cell ADJACENT to ``target``.

    Used when the target sits on a BLOCKED tile (Poke Ball on the table, an
    NPC behind a counter): you can never step onto that tile, so A* *to* it is
    wrong - when adjacent it returns the impossible straight step onto the
    wall. Route to the nearest REACHABLE walkable neighbour instead; the talk
    loop's adjacency check then faces the target and presses A.
    ``target`` is a screen cell (row, col); the player is always at (4,4).
    """
    from jev_plays_pokemon import screen_grid

    pb = builder.reader.emu._pyboy
    walkable = screen_grid.walkable_grid(pb)
    sprites = screen_grid.sprite_cells(pb)
    tr, tc = target
    neighbours: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = tr + dr, tc + dc
        if not (0 <= nr < 9 and 0 <= nc < 10):
            continue
        if (nr, nc) == screen_grid.PLAYER:
            return True  # already adjacent
        if walkable[nr, nc] and (nr, nc) not in sprites:
            neighbours.append((nr, nc))
    neighbours.sort(key=lambda c: abs(c[0] - 4) + abs(c[1] - 4))
    for cell in neighbours:
        path = screen_grid.astar_screen(cell, walkable, sprites, allow_wall_goal=True)
        if path and _walk_step(emu, builder, path[0]):
            return True
    return _force_wiggle(emu, builder)


def _step_into_wall(emu, builder, walkable) -> bool:
    """Step into an adjacent wall tile (empirically probing for a door).

    Only steps when a wall is adjacent; returns False otherwise so the caller
    can walk ALONG the boundary instead of wiggling randomly.
    """
    for direction, (dr, dc) in (("up", (-1, 0)), ("down", (1, 0)), ("left", (0, -1)), ("right", (0, 1))):
        r, c = 4 + dr, 4 + dc
        if 0 <= r < 9 and 0 <= c < 10 and not walkable[r, c]:
            if _walk_step(emu, builder, direction):
                return True
    return False


def _step_onto(emu, builder, target: tuple[int, int]) -> bool:
    """Take a step directly toward ``target`` (used to step onto a door)."""
    for direction, (dr, dc) in (("up", (-1, 0)), ("down", (1, 0)), ("left", (0, -1)), ("right", (0, 1))):
        if (4 + dr, 4 + dc) == target:
            return _walk_step(emu, builder, direction)
    return False


def _warp_collision_step(emu, builder, walkable) -> bool:
    """Trigger a warp while the player is standing ON a warp tile.

    The engine fires the transition in CheckWarpsCollision: the player must
    PHYSICALLY step into a wall while standing on a warp block - just standing
    there does nothing. So when a warp to the destination occupies the player's
    own cell, press into the adjacent boundary wall.
    """
    return _step_into_wall(emu, builder, walkable)


def _along_wall_direction(walkable) -> str | None:
    """Right-hand wall-following move along the room boundary.

    For each wall side adjacent to the player, the direction that keeps that
    wall on the player's right is preferred - this walks the perimeter
    deterministically instead of flipping between sides.
    """
    dirs = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    keep_right = {"up": "right", "right": "down", "down": "left", "left": "up"}
    for wall, (wr, wc) in dirs.items():
        if not (0 <= 4 + wr < 9 and 0 <= 4 + wc < 10) or walkable[4 + wr][4 + wc]:
            continue  # no wall on this side
        candidate = keep_right[wall]
        dr, dc = dirs[candidate]
        if 0 <= 4 + dr < 9 and 0 <= 4 + dc < 10 and walkable[4 + dr][4 + dc]:
            return candidate
    return None


def _talk_to(emu, builder, state, index: int, max_steps: int) -> str:
    from jev_plays_pokemon import screen_grid

    turn = state["turn"]
    result = "steps exhausted"
    for _ in range(max_steps):
        if stop_requested():
            return "interrupted"
        state = builder.snapshot(turn)
        objects = builder.objects
        obj = objects[index] if index < len(objects) else None
        target = screen_grid.cell_to_rowcol(obj["screen_cell"]) if obj and obj.get("screen_cell") else None
        if target is None:
            result = "target gone"
            break
        if abs(target[0] - 4) + abs(target[1] - 4) <= 1:
            # Adjacent: turn to face the target, then talk.
            for direction, (dr, dc) in (
                ("up", (-1, 0)),
                ("down", (1, 0)),
                ("left", (0, -1)),
                ("right", (0, 1)),
            ):
                if (4 + dr, 4 + dc) == target:
                    press(emu, f"walk_{direction}")
                    break
            press(emu, "press_a")
            builder.last_talked = obj.get("name") or f"NPC {index}"
            result = "talked (A pressed)"
            break
        # Walk toward the target. A target on a WALKABLE tile (a normal NPC) is
        # reached with plain A* - pathing onto its tile is fine because we
        # stop at the adjacent cell and the top-of-loop adjacency check faces
        # and talks. A target on a BLOCKED tile (Poke Ball on the table, an
        # NPC behind a counter) can never be stepped onto, so route to the
        # nearest walkable neighbour instead.
        pb = builder.reader.emu._pyboy
        target_blocked = not bool(screen_grid.walkable_grid(pb)[target[0], target[1]])
        if (target_blocked and _move_adjacent_to(emu, builder, target)) or (
            not target_blocked and _move_toward(emu, builder, target)
        ):
            continue
        if builder.dialog_active:
            result = "dialog opened while walking"
            break
        result = "blocked on the way"
        break
    if result == "steps exhausted" or result == "blocked on the way":
        picture = objects[index].get("picture") if index < len(objects) else None
        builder.talk_cooldown[(state.get("location", {}).get("map_id"), picture)] = turn + 6
    return result


def _reach_exit(emu, builder, state, max_steps: int, dest: int | None) -> str:
    """Leave the room by probing the on-screen boundary for a door.

    Doors are walls that warp when stepped onto, so we walk to the room's
    boundary and empirically step into the walls until the map transitions.
    """
    from jev_plays_pokemon import screen_grid

    start_map = state["location"]["map_id"]
    island_bailout = 0
    for _ in range(max_steps):
        if stop_requested():
            return "interrupted"
        state = builder.snapshot(state["turn"])
        if state["location"]["map_id"] != start_map:
            return "exited the room"
        pb = builder.reader.emu._pyboy
        tileset = state.get("tileset") or 0
        walkable = screen_grid.walkable_grid(pb)
        sprites = screen_grid.sprite_cells(pb)
        doors = screen_grid.door_cells(pb, tileset)
        warps_on_screen = screen_grid.warp_cells(pb, dest=dest)
        steer = screen_grid.warp_direction(pb, dest)

        # 1) The game's active warp list is authoritative: if a warp to the
        #    destination map is on-screen, pathfind straight onto it (the last
        #    step steps onto the warp tile). Step-triggered warps (stairs,
        #    holes) fire the instant the player lands on the tile. Boundary
        #    warps (house front doors) fire only when the player, already on
        #    the carpet mat, physically steps INTO the boundary wall
        #    (CheckWarpsCollision: collision while standing on a warp).
        if warps_on_screen:
            island_bailout = 0
            warp = min(warps_on_screen, key=lambda c: abs(c[0] - 4) + abs(c[1] - 4))
            if warp != screen_grid.PLAYER:
                if _move_toward(emu, builder, warp):
                    continue
                if _step_onto(emu, builder, warp):
                    continue
            elif _warp_collision_step(emu, builder, walkable):
                # Already standing on the warp: step INTO the boundary wall.
                continue

        # 2) A door/exit tile is visible: pathfind straight onto it (step 1
        #    covered it via warps when the map has a warp list, but some maps
        #    (overworld) also expose plain door tiles).
        if doors:
            island_bailout = 0
            door = min(doors, key=lambda c: abs(c[0] - 4) + abs(c[1] - 4))
            if door != screen_grid.PLAYER:
                if _move_toward(emu, builder, door):
                    continue
            if _step_onto(emu, builder, door):
                continue

        # 3) Warp not on-screen: pathfind toward it in map space. The naive
        #    cardinal "steer" step alone can't cope when furniture or an NPC
        #    blocks the direct line (e.g. Blue's House: door at (2,7)/(3,7) but
        #    Daisy/Pokedex occupy (2,3)/(3,3)) - so route around with map-space
        #    A* over the verified room map (warp tiles are passable), taking
        #    the first step of that path. Fall back to cardinal steering when
        #    no room-map path exists yet (unexplored floor).
        if steer:
            island_bailout = 0
            px, py = player_map(builder)
            warps = builder.live_exits()
            path = None
            if warps:
                warp = min(
                    (w for w in warps if screen_grid._warp_matches(dest, w[2])),
                    key=lambda w: abs(w[0] - px) + abs(w[1] - py),
                    default=None,
                )
                if warp is not None:
                    path = astar((px, py), (warp[0], warp[1]), _room_walkable(builder, state))
            if path:
                if _walk_step(emu, builder, path[0]):
                    continue
            elif _walk_step(emu, builder, steer):
                continue

        # 4) Probe the wall beside us (might be the exit if warps are stale).
        if _step_into_wall(emu, builder, walkable):
            continue
        # 5) Walk along the boundary, keeping a wall adjacent.
        direction = _along_wall_direction(walkable)
        if direction and _walk_step(emu, builder, direction):
            continue
        # 6) Not on the boundary yet: head to the nearest wall-adjacent cell.
        target = screen_grid.frontier_cell(walkable, sprites)
        if target is not None and target != screen_grid.PLAYER:
            if _move_toward(emu, builder, target):
                continue
        # 7) Empirical probe / island bailout (don't orbit furniture forever).
        if _force_wiggle(emu, builder):
            island_bailout = 0
            continue
        island_bailout += 1
        if island_bailout >= 4:
            return "no exit found"
    return "exit attempt done"


def _recent_screen_cells(builder) -> set[tuple[int, int]]:
    """Absolute map positions recently stood on -> on-screen cells.

    The player is always grid cell (4,4); an absolute position (mx,my) lands
    on screen (row, col) = (4 + my - py, 4 + mx - px). Used so exploration
    avoids cells the player just stood on instead of ping-ponging.
    """

    px, py = player_map(builder)
    cells: set[tuple[int, int]] = set()
    for mx, my in builder.move_history:
        row, col = 4 + (my - py), 4 + (mx - px)
        if 0 <= row < 9 and 0 <= col < 10:
            cells.add((row, col))
    return cells


def _explore(emu, builder, state, max_steps: int) -> str:
    """Explore the current screen toward the nearest wall/sprite frontier."""
    from jev_plays_pokemon import screen_grid

    map_id = state["location"]["map_id"]
    for _ in range(max_steps):
        if stop_requested():
            return "interrupted"
        state = builder.snapshot(state["turn"])
        if state["location"]["map_id"] != map_id:
            return "room changed"
        pb = builder.reader.emu._pyboy
        walkable = screen_grid.walkable_grid(pb)
        sprites = screen_grid.sprite_cells(pb)
        # Stepping onto a warp exits the room, so exploration never targets the
        # live warp tiles (stateless, read from RAM each frame) and never steps
        # across one en route. Recently-visited cells are also avoided so the
        # walker keeps pushing into new space instead of bouncing between the
        # two wall-adjacent cells nearest to it.
        warps = screen_grid.warp_cells(pb)
        visited = _recent_screen_cells(builder)
        target = screen_grid.frontier_cell(walkable, sprites, warps=warps, visited=visited)
        if target is not None and target != screen_grid.PLAYER:
            if _move_toward(emu, builder, target, forbidden=warps):
                continue
        if _force_wiggle(emu, builder):
            continue
        return "stuck in a dead pocket"
    return "explored a stretch"
