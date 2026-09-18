"""Assemble a single text-based state snapshot for the Jev model.

Combines RAM-derived structured state (via the vendored Red/Blue reader),
the ground-truth walkability map, the decoded on-screen text, and in-code
memory (recent hints, recent actions) into one JSON-serialisable dict.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from jev_plays_pokemon.objects import _relative, read_objects, render_objects
from jev_plays_pokemon.screen_text import read_screen_text
from jev_plays_pokemon.vendor.collision import build_collision_grid, render_ascii_map
from jev_plays_pokemon.vendor.memory.red import MAP_NAMES, RedBlueMemoryReader
from jev_plays_pokemon.vendor.state.builder import build_game_state

# Exits are NEVER hardcoded. The authoritative exit list lives in RAM
# (wNumberOfWarps @ 0xD3AE + wWarpEntries @ 0xD3AF) and is read live every
# snapshot via StateBuilder.live_exits(). Fully stateless: each frame derives
# the current room's exits directly from the game state, so the agent always
# sees reality (no stale seeds, no tutorial-specific coords).
# LAST_MAP (0xFF) is a warp whose destination is wherever we entered from -
# for the tutorial rooms that is Pallet Town (map 0).

INSTRUCTIONS = """\
You are RED in Pokemon Red on a Game Boy. Your long-term goal: earn all 8 gym badges \
and become the Champion. NPC dialog tells you where to go and what to do next - follow it. \
Control the game one button press at a time. Pick the ONE action that best advances your goal.

State:
- screen_text: dialog/menu text on screen.
- recent_hints: what people have recently told you (kept across turns even after the text box closes). Ordered OLDEST first, NEWEST LAST - the rightmost entry is what was just said. Use these to decide where to go or what to do next.
- recent_actions: the last high-level actions you took, ordered OLDEST first, NEWEST LAST - the final entry is what you just did. Don't repeat the most recent action unless it is clearly right.
- collision_map: walkable grid around you (@=you at E5, .=walkable, #=blocked).
- walkable_directions: which of up/down/left/right you can step to right now; authoritative, live from the game's collision data each frame.
- room_map: explored part of the room (@=you, .=walkable, #=blocked, space=unexplored). explored_fraction shows how much of the room you have seen (0-1). When it stops growing, you have seen everything reachable - stop exploring and pick a goal that moves you forward (talk or leave).
- objects: people/objects on the current map with their screen cell (A1..J9, you are E5) and direction/distance from you. Walk next to someone and press A to talk; talk to NPCs to progress the story.
- unexplored_hint: first step toward the nearest unexplored area.
- probe_exits=true: the room is fully explored.
- goals: the things you could do right now, listed by id. Pick the ONE goal that best advances your goal; code will then walk you there.
- known_exits: exits of this room you know about, with their tile coords and direction from you (doors/stairs look like solid walls # - step onto them to exit).

Priority: (1) dialog/text box open -> advance_text, unless it is a menu or question. \
(2) in battle -> handle the battle. \
(3) party hurt -> reach a Pokemon Center. \
(4) follow the story: talk to the NPCs listed under goals, or pick reach_exit to leave the room. \
(5) when idle, explore toward unexplored space; if stuck, press B then try another way.

Your goal probabilities are sampled, so every option has a real chance. Do not rely on one favorite - give meaningful probability to the genuinely good alternatives. \
If you just did something and nothing changed (same place, same text, same result), do not repeat it - pick the next most useful goal.

NPC dialog is your guide: people tell you where to go and what to do next. Read every dialog box; \
remember the direction or task it mentions (e.g. \"Go see Prof. Oak\", \"Deliver this\") and act on it. \
Progress in this game is shown by gym badges - find each town's Gym Leader by following the directions NPCs give.

Known exits (doors/stairs) are always offered in known_exits and goals - use them to move between rooms and towns when the story calls for it. \
Do not oscillate (going UP then DOWN then UP). Stick to a multi-step direction toward your goal until you hit an obstacle or complete it. \
Front doors and stairs look like solid walls (#) on the collision map. To exit the room, walk INTO the wall that is the exit - the game transitions when you step onto a door/stairs tile.

Buttons: press_a=confirm/interact/advance text/select highlighted; press_b=cancel/back/run; press_start=main menu; \
walk_*=one D-pad press (moves you in the overworld, moves the cursor in menus); wait=do nothing ~half a second for animations.

Battle types (use super-effective): Water>Fire/Ground/Rock; Fire>Grass/Bug/Ice; Grass>Water/Ground/Rock; \
Electric>Water/Flying; Ground>Fire/Electric/Rock/Poison; Ice>Grass/Ground/Flying/Dragon; Fighting>Normal/Rock/Ice; \
Psychic>Fighting/Poison. Special is both special attack and defense.

Progress is shown by gym badges (8 total). NPCs give directions to the next town/gym - read the dialog. \
Use the collision map and known_exits to travel between towns."""


def _walkable_directions(collision: dict[str, Any]) -> dict[str, bool]:
    """Walkability of the four cells adjacent to the player (always cell E5).

    The collision grid is 9x10 (rows x cols) with the player locked to
    row 4, col 4. Purely from the live RAM collision data: a bump/sprite/NPC
    is transient, so it is never remembered as a permanent wall.
    """
    walkable = collision.get("walkable")
    if not walkable:
        return {"up": True, "down": True, "left": True, "right": True}
    r, c = 4, 4
    return {
        "up": bool(walkable[r - 1][c]),
        "down": bool(walkable[r + 1][c]),
        "left": bool(walkable[r][c - 1]),
        "right": bool(walkable[r][c + 1]),
    }


def compute_goals(
    builder: StateBuilder,
    dialog_active: bool,
    objects: list[dict[str, Any]],
    fully_explored: bool,
    state: dict[str, Any],
    turn: int,
) -> list[dict[str, str]]:
    """Candidate high-level goals for Jev to choose between.

    Each goal is ``{"id", "desc"}``; the id drives mechanical execution in
    code (pathfinding, dialog advancing) while Jev makes the choice.
    """
    goals: list[dict[str, str]] = []

    def offered(goal_id: str) -> bool:
        """Is ``goal_id`` currently cooled down (recently run, avoid re-running)?"""
        return builder.goal_cooldown.get(goal_id, 0) <= turn

    battle = state.get("battle") or {}
    if battle.get("in_battle"):
        enemy = (battle.get("enemy") or {}).get("species", "enemy")
        goals.append(
            {
                "id": "battle",
                "desc": f"Handle the battle against {enemy}: attack with a super-effective move, "
                "weaken and throw a Poke Ball to catch, switch, or run",
            }
        )
        return goals

    if dialog_active:
        # A text box is a deterministic state: advancing it is the only goal.
        return [
            {
                "id": "advance_text",
                "desc": "Advance the dialog / text box (press A), unless it is a menu or question",
            }
        ]

    map_id = (state.get("map") or {}).get("map_id")
    for i, obj in enumerate(objects):
        if builder.talk_cooldown.get((map_id, obj.get("picture")), 0) > turn:
            continue
        if not (obj.get("on_screen") and obj.get("map")):
            continue
        goals.append(
            {
                "id": f"talk_to_{i}",
                "desc": f"Walk next to {obj['name']} at {obj['screen_cell']} ({obj['relative']}) "
                "and talk to them (press A when next to them)",
            }
        )

    map_id = (state.get("map") or {}).get("map_id")
    # Exits come from the live RAM warp table each frame. All exits are always
    # offered - never filtered by recency. Re-entering a room you just left is
    # sometimes the right move (the story or a scripted event may demand it), so
    # that's the MODEL's call; we only annotate which exits are recent via the
    # goal description. Suppression based on map_history would risk removing the
    # only exit and permanently stranding the agent.
    warps = builder.live_exits() if map_id is not None else None
    # When there are no known warps, the boundary-probe exit is only offered
    # once the room is fully explored - a fallback for "nothing left to
    # explore", so the model doesn't probe walls forever instead of exploring.
    if not dialog_active:
        if warps:
            # Drop bogus self-loop exits (map -> itself) and the warp tile the
            # player is standing on (don't re-trigger the door we just used).
            filtered = [w for w in warps if w[2] != map_id]
            player_pos = (state.get("player") or {}).get("position") or {}
            filtered = [w for w in filtered if (w[0], w[1]) != (player_pos.get("x"), player_pos.get("y"))]
            ordered = sorted(filtered, key=lambda w: w[2] in builder.map_history)
            for wx, wy, dest, direction in ordered:
                dest_name = MAP_NAMES.get(dest, f"map {dest}")
                note = " (a room you just left)" if dest in builder.map_history else ""
                goals.append(
                    {
                        "id": f"exit_to_{dest}",
                        "desc": f"Leave this room to {dest_name}{note}: walk to tile ({wx},{wy}) and step {direction}",
                    }
                )
        elif fully_explored:
            goals.append(
                {
                    "id": "reach_exit",
                    "desc": "Find and use the exit of this room - doors/stairs look like solid walls (#); walk INTO them to exit",
                }
            )

    if not fully_explored and offered("explore"):
        hint = state.get("unexplored_hint")
        desc = "Explore the room toward unexplored space"
        if hint:
            desc += f" (head {hint})"
        goals.append({"id": "explore", "desc": desc})

    # Keep the goal list non-empty: if nothing else is offered and explore is not
    # deliberately cooled down, always keep it as a fallback.
    if not fully_explored and offered("explore") and all(g["id"] == "wait" for g in goals):
        goals.append({"id": "explore", "desc": "Explore the room toward unexplored space"})

    # wait is a do-nothing: only offer it when there is genuinely nothing else
    # to pick, so the model can't retreat into it repeatedly. The cooldown set
    # in the main loop keeps it out of the list after each use, forcing the
    # model to vary.
    if not goals:
        goals.append({"id": "wait", "desc": "Wait a moment (let animations/transitions finish)"})
    return goals


def render_goals(goals: list[dict[str, str]]) -> str:
    if not goals:
        return "(no goals)"
    return " | ".join(f"{g['id']}: {g['desc']}" for g in goals)


class GameMemory:
    """Small in-code memory: recent actions and a short-term goal."""

    def __init__(self) -> None:
        self.recent: deque[tuple[str, str]] = deque(maxlen=12)

    def record(self, action: str, outcome: str) -> None:
        self.recent.append((action, outcome))

    def summary(self) -> list[str]:
        return [f"{a} -> {o}" for a, o in self.recent]


class RoomMap:
    """Persistent memory of explored rooms.

    For each map id, a grid of cells classified as ``walkable`` (the player
    stood there or could step onto it), ``blocked`` (bumped into it or the
    collision data says so), ``seen`` (ambiguous), or unexplored.
    """

    WALKABLE = "walkable"
    BLOCKED = "blocked"
    SEEN = "seen"

    def __init__(self) -> None:
        self.cells: dict[int, dict[tuple[int, int], str]] = {}

    def _room(self, map_id: int) -> dict[tuple[int, int], str]:
        return self.cells.setdefault(map_id, {})

    def update(self, map_id: int, x: int, y: int, walkable: dict[str, bool]) -> None:
        """Record the current cell and its neighbours from the live collision data."""
        room = self._room(map_id)
        room[(x, y)] = self.WALKABLE
        for direction, ok in walkable.items():
            cell = self._adjacent(x, y, direction)
            room[cell] = self.WALKABLE if ok else self.BLOCKED

    @staticmethod
    def _adjacent(x: int, y: int, direction: str) -> tuple[int, int]:
        dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[direction]
        return (x + dx, y + dy)

    def hint_toward_unexplored(self, map_id: int, x: int, y: int) -> str | None:
        """First direction from (x, y) toward the nearest unexplored cell.

        BFS over known-walkable cells; returns the direction of the first
        step toward the closest walkable cell adjacent to unexplored space.
        """
        from collections import deque

        room = self._room(map_id)
        if (x, y) not in room:
            return None
        queue = deque([(x, y)])
        visited = {(x, y)}
        first_dir: dict[tuple[int, int], str] = {}
        while queue:
            cx, cy = queue.popleft()
            for d in ("up", "down", "left", "right"):
                nx, ny = self._adjacent(cx, cy, d)
                if (nx, ny) not in room:
                    return first_dir.get((cx, cy), d)
                if (nx, ny) in visited or room.get((nx, ny)) != self.WALKABLE:
                    continue
                visited.add((nx, ny))
                first_dir[(nx, ny)] = first_dir.get((cx, cy), d)
                queue.append((nx, ny))
        return None

    @staticmethod
    def _reconstruct(prev: dict, start: tuple[int, int], node: tuple[int, int], final_dir: str) -> list[str]:
        path: list[str] = []
        while node != start:
            node, direction = prev[node]
            path.append(direction)
        path.reverse()
        path.append(final_dir)
        return path

    def path_to_unexplored(
        self, map_id: int, x: int, y: int
    ) -> tuple[tuple[int, int] | None, list[str] | None]:
        """Full path (list of directions) to the nearest unexplored area.

        Returns ``((target_x, target_y), [directions...])`` or
        ``(None, None)`` when everything reachable is explored.
        """
        from collections import deque

        room = self._room(map_id)
        if (x, y) not in room:
            return None, None
        queue = deque([(x, y)])
        visited = {(x, y)}
        prev: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
        while queue:
            cx, cy = queue.popleft()
            for d in ("up", "down", "left", "right"):
                nx, ny = self._adjacent(cx, cy, d)
                if (nx, ny) not in room:
                    return (nx, ny), self._reconstruct(prev, (x, y), (cx, cy), d)
                if (nx, ny) in visited or room.get((nx, ny)) != self.WALKABLE:
                    continue
                visited.add((nx, ny))
                prev[(nx, ny)] = ((cx, cy), d)
                queue.append((nx, ny))
        return None, None

    def path_to_boundary(
        self, map_id: int, x: int, y: int
    ) -> tuple[tuple[int, int] | None, list[str] | None]:
        """Full path to the nearest walkable cell that touches a blocked cell."""
        from collections import deque

        room = self._room(map_id)
        if (x, y) not in room:
            return None, None
        queue = deque([(x, y)])
        visited = {(x, y)}
        prev: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
        while queue:
            cx, cy = queue.popleft()
            for d in ("up", "down", "left", "right"):
                nx, ny = self._adjacent(cx, cy, d)
                if room.get((nx, ny)) == self.BLOCKED:
                    return (cx, cy), self._reconstruct(prev, (x, y), (cx, cy), d)
                if (nx, ny) in visited or room.get((nx, ny)) != self.WALKABLE:
                    continue
                visited.add((nx, ny))
                prev[(nx, ny)] = ((cx, cy), d)
                queue.append((nx, ny))
        return None, None

    def is_fully_explored(self, map_id: int) -> bool:
        """True when every known walkable cell borders only known cells."""
        return self.explored_fraction(map_id) >= 1.0

    def explored_fraction(self, map_id: int) -> float:
        """Fraction of walkable cells whose 4 neighbours are all known (0..1).

        Reaches 1.0 exactly when :func:`is_fully_explored` is True, but is a
        gradient before that - so the model sees progress ("lab 88% explored")
        and can spot when exploration has plateaued, instead of a binary
        "done / not done" that may never flip (unreachable cells keep a room
        from ever reaching 100%).
        """
        room = self._room(map_id)
        walkable = [cell for cell, kind in room.items() if kind == self.WALKABLE]
        if not walkable:
            return 0.0
        closed = sum(
            1
            for (x, y) in walkable
            if all(self._adjacent(x, y, d) in room for d in ("up", "down", "left", "right"))
        )
        return closed / len(walkable)

    def hint_toward_boundary(self, map_id: int, x: int, y: int) -> str | None:
        """First step toward the nearest walkable cell that touches a wall.

        Used when a room is fully explored: the exit is behind one of the
        boundary walls, so walk to the nearest one and probe it.
        """
        from collections import deque

        room = self._room(map_id)
        if (x, y) not in room:
            return None
        queue = deque([(x, y)])
        visited = {(x, y)}
        first_dir: dict[tuple[int, int], str] = {}
        while queue:
            cx, cy = queue.popleft()
            for d in ("up", "down", "left", "right"):
                nx, ny = self._adjacent(cx, cy, d)
                if room.get((nx, ny)) == self.BLOCKED:
                    return first_dir.get((cx, cy), d)
                if (nx, ny) in visited or room.get((nx, ny)) != self.WALKABLE:
                    continue
                visited.add((nx, ny))
                first_dir[(nx, ny)] = first_dir.get((cx, cy), d)
                queue.append((nx, ny))
        return None

    def render(self, map_id: int, player: tuple[int, int] | None = None) -> str:
        """Render the explored portion of a room as an ASCII grid.

        @ = player, . = walkable, # = blocked, ? = seen but ambiguous,
        ' ' = unexplored. Cropped to a window around the player so the
        text state stays bounded even in large maps.
        """
        room = self._room(map_id)
        if not room:
            return "(nothing explored yet)"
        xs = [c[0] for c in room]
        ys = [c[1] for c in room]
        min_x, max_x = max(0, min(xs) - 2), max(xs) + 2
        min_y, max_y = max(0, min(ys) - 2), max(ys) + 2
        # Bound the render size with a window centred on the player.
        if player is not None:
            px, py = player
            min_x, max_x = max(min_x, px - 7), min(max_x, px + 7)
            min_y, max_y = max(min_y, py - 6), min(max_y, py + 6)
        glyph = {self.WALKABLE: ".", self.BLOCKED: "#", self.SEEN: "?"}
        lines = []
        for yy in range(max_y, min_y - 1, -1):
            row = ""
            for xx in range(min_x, max_x + 1):
                if player and (xx, yy) == player:
                    row += "@"
                    continue
                row += glyph.get(room.get((xx, yy)), " ")
            lines.append(f"{yy:>2} " + row)
        return "\n".join(lines)


def capture_hint(
    hints: deque,
    dialog_active: bool,
    screen_lines: list[str],
    continuous: bool = False,
    speaker: str | None = None,
) -> None:
    """Remember what an NPC told us so the model can act on it later.

    Dialogs are advanced quickly, so ``screen_text`` only shows the current
    page - often mid-typing (e.g. just "MO"). Each snapshot accumulates the
    visible partial text into the current conversation hint instead of
    dropping it, so a hint survives even when the agent advances the box
    before it finished typing. Consecutive pages of the same text box
    (``continuous``) merge into one hint. Selection menus (▶ cursor) are not
    story hints. The hint is prefixed with who spoke (if known) so the agent
    knows it already talked to that NPC.
    """
    if not dialog_active:
        return
    raw = " ".join(line.strip() for line in screen_lines if line.strip())
    if "▶" in raw:
        return  # a selectable menu, not a story hint
    text = raw.replace("▼", "").strip()
    if len(text) < 2:
        return  # nothing decodable yet
    if continuous and hints:
        # Same box: merge the partial/full page (no speaker repeat). The box
        # scrolls text while typing, so the new snapshot often re-shows the
        # tail we already captured - only append the genuinely new suffix.
        if hints[-1].endswith(text):
            return
        prev = hints[-1]
        overlap = 0
        for k in range(min(len(text), 40), 0, -1):
            if prev.endswith(text[:k]):
                overlap = k
                break
        hints[-1] = (prev + " " + text[overlap:]).strip()[:220]
    else:
        label = f"{speaker or 'an NPC'}: {text}"
        if hints and hints[-1].endswith(label):
            return
        # Deduplicate across the whole window: talking to the same NPC twice
        # often repeats the same line, which would otherwise crowd out distinct
        # information from the (small) hint window. Only skip exact repeats.
        if label in hints:
            return
        hints.append(label[:220])


class StateBuilder:
    def __init__(self, emu) -> None:
        self.reader = RedBlueMemoryReader(emu)
        self.memory = GameMemory()
        self.room_map = RoomMap()
        # Exits are NOT seeded: the authoritative list is read live from RAM
        # (wWarpEntries) every snapshot via live_exits(). This dict only
        # accumulates *confirmed* exits we physically walked through, as a
        # fallback when a map is re-visited (RAM is authoritative while we are
        # inside it, so live_exits() takes precedence).
        self.known_warps: dict[int, list[tuple[int, int, int, str]]] = {}
        self._last_map: int | None = None
        self._last_pos: tuple[int, int] | None = None
        self._last_dir: str = "?"
        self._pending_warp: tuple[int, tuple[int, int], int, str] | None = None
        # Snapshots after a map change during which position/collision reads
        # are unstable - do not trust the map.
        self._transition_grace = 0
        # Parsed nearby objects / goals from the last snapshot.
        self.objects: list[dict[str, Any]] = []
        self.goals: list[dict[str, str]] = []
        # object (map_id, picture) -> turn until which we skip re-offering "talk to X".
        # Keyed by map+sprite (not the reordering-prone slot index). Short and
        # EXPIRING: it is only a mild "you just did this" nudge.
        self.talk_cooldown: dict[tuple[int | None, int | None], int] = {}
        # goal id -> turn until which we stop offering that goal again, so a
        # recently-run futile action (explore that got stuck, a failed exit)
        # is not retried instantly. Short and expiring.
        self.goal_cooldown: dict[str, int] = {}
        self._last_objects_map: int | None = None
        self._objects_grace = 0  # snapshots to wait after a map change
        # Recently visited maps, used to avoid pacing between rooms.
        self.map_history: deque[int] = deque(maxlen=4)
        # Recently visited positions, used to avoid pacing loops when walking.
        self.move_history: deque[tuple[int, int]] = deque(maxlen=6)
        # What NPCs have told us (persistent across turns, unlike screen_text).
        # Merged per conversation; we keep the last 14.
        self.hints: deque[str] = deque(maxlen=8)
        self._dialog_was_active = False
        self.last_talked: str | None = None  # NPC we just talked to (hint speaker)
        # Last fully-decoded on-screen text; a dialog page is "stable" (fully
        # typed) when it matches across two consecutive snapshots. None when
        # no text box is open.
        self._prev_screen_text: str | None = None
        # Longest text observed for the open dialog page. The box types out
        # while we are idle, so the decoded text GREWS until the page is
        # complete; we keep the longest rendering and commit it as a hint once
        # the box closes (or a plainly-new page starts). Mid-typing fragments
        # are never committed, only the fullest version of each page.
        self._page_text: str | None = None
        self._page_done: bool = False

    def live_exits(self) -> list[tuple[int, int, int, str]]:
        """Live exits of the current map, read from RAM every frame.

        Fully stateless: (x, y, dest, direction) derived from wWarpEntries.
        A LAST_MAP warp (0xFF, "back where you came from") maps to map 0 for
        the tutorial rooms. Direction is the cardinal step that moves the
        player onto the tile.
        """
        from jev_plays_pokemon import screen_grid

        if self._transition_grace > 0:
            return []  # the RAM warp table still holds the PREVIOUS map's warps
        pb = self.reader.emu._pyboy
        py = pb.memory[0xD361]
        px = pb.memory[0xD362]
        out: list[tuple[int, int, int, str]] = []
        for y, x, _warp_id, dest in screen_grid.warp_entries(pb):
            if dest == 0xFF:
                dest = 0  # LAST_MAP -> Pallet (entered the house from there)
            dy, dx = y - py, x - px
            direction = "up" if dy < 0 else "down" if dy > 0 else "left" if dx < 0 else "right"
            out.append((x, y, dest, direction))
        return out

    def _learn_warp(self, map_id: int, pos: dict) -> None:
        """Record an exit only when a map change is *confirmed*.

        The map-id read can flicker (0 -> 37 -> 0), which would otherwise make us
        learn a phantom exit and then chase it forever. We defer the record until
        the next snapshot confirms we actually stayed in the new map.
        """
        if self._last_map is not None and self._last_pos is not None and map_id != self._last_map:
            self._pending_warp = (self._last_map, self._last_pos, map_id, self._last_dir)
        elif self._pending_warp is not None and map_id == self._pending_warp[2]:
            old_map, old_pos, new_map, direction = self._pending_warp
            if old_map != new_map:  # never a self-loop (flicker artifact)
                warps = self.known_warps.setdefault(old_map, [])
                if (old_pos[0], old_pos[1], new_map, direction) not in warps:
                    warps.append((old_pos[0], old_pos[1], new_map, direction))
                self.map_history.append(old_map)
            self._pending_warp = None
        self._last_map = map_id
        self._last_pos = (pos.get("x"), pos.get("y"))

    def capture_text_box(self, screen_lines: list[str], dialog_active: bool, stable: bool = False) -> None:
        """Record a fully-typed dialog page as a hint.

        ``stable`` is set by the caller (the text-advance loop) once the engine
        shows the ▼ arrow, which the game only draws when a page is complete and
        awaiting input (WaitForTextScrollButtonPress). Mid-scroll frames are
        never committed - only complete pages. Each page of the same
        conversation is appended with the speaker tag.
        """
        current_text = " ".join(line.strip() for line in screen_lines if line.strip())
        if stable and dialog_active and current_text:
            capture_hint(self.hints, True, [current_text], False, speaker=self.last_talked)
        if not dialog_active:
            self.last_talked = None  # conversation over; drop the speaker tag
        self._dialog_was_active = dialog_active

    @property
    def box_ready(self) -> bool:
        """True when the dialog box is awaiting input (▼ arrow or ■ end square).

        The engine draws ▼ (0xEE) in WaitForTextScrollButtonPress while more
        pages follow, and the final page is closed with the ■ end square
        (0xEF) in some Gen 1 textboxes. Either means the current page has fully
        printed and is waiting for A/B - the safe moment to capture the text.
        Both glyphs blink, so a caller should poll a couple of frames. ▶ (0xED)
        is a MENU cursor and is deliberately NOT treated as page-complete.
        """
        pb = self.reader.emu._pyboy
        try:
            w = np.asarray(pb.tilemap_window[:, :])
            region = w[12:19, :20]
            return bool(np.isin(region, (0xEE, 0xEF)).any())
        except Exception:
            return False

    @property
    def dialog_active(self) -> bool:
        """True when a text box is open (dialog or menu).

        The text-advance arrow (▶ 0xED / ▼ 0xEE) is rendered in the bottom
        text-box rows of the window layer; its exact row depends on the box
        height, so scan the whole box region. While text is still typing the
        arrow is absent but the box is up, so any decoded on-screen text also
        counts as an active text box.
        """
        pb = self.reader.emu._pyboy
        try:
            w = np.asarray(pb.tilemap_window[:, :])
            region = w[12:19, :20]
            if np.isin(region, (0xED, 0xEE)).any():
                return True
        except Exception:
            pass
        try:
            return bool(read_screen_text(pb))
        except Exception:
            return False

    def record_move(self, action: str, moved: bool) -> None:
        """Track the last walked direction (used to avoid pacing reversals).

        Deliberately does NOT learn walls from bumps: a failed step is often a
        sprite/NPC/cutscene block or a frame glitch, and remembering it as a
        hard "blocked" direction makes movement impossible based on a transient
        state (the same anti-pattern as the removed talked_this_visit). The live
        RAM collision grid is authoritative every frame instead.
        """
        if action not in ("walk_up", "walk_down", "walk_left", "walk_right"):
            return
        if moved:
            self._last_dir = action.removeprefix("walk_")

    def snapshot(self, turn: int) -> dict[str, Any]:
        state = build_game_state(self.reader, frame_count=getattr(self.reader.emu, "frame_count", None))

        # On-screen text + collision map (needs the raw PyBoy handle).
        pb = self.reader.emu._pyboy
        screen_lines = read_screen_text(pb)
        dialog_active = self.dialog_active
        # Only remember STABLE pages as hints. Text types out one character at
        # a time and the ▼ arrow blinks, so neither a length threshold nor the
        # arrow reliably says "finished typing". A page is complete when two
        # CONSECUTIVE snapshots decode the same text - the game fixes the box
        # in place while waiting for input, so stability means the full page is
        # on screen. Mid-typing fragments are therefore never captured.
        self.capture_text_box(screen_lines, dialog_active)
        collision = build_collision_grid(self.reader.emu)
        ascii_map = render_ascii_map(collision)

        pos = (state.get("player") or {}).get("position") or {}
        map_id = (state.get("map") or {}).get("map_id")

        walkable = _walkable_directions(collision)

        if map_id is not None and "x" in pos and "y" in pos:
            room_walkable = walkable
            self.room_map.update(map_id, pos["x"], pos["y"], room_walkable)
        room_map_text = (
            self.room_map.render(map_id, (pos["x"], pos["y"])) if map_id is not None else "(unknown map)"
        )
        unexplored_hint = (
            self.room_map.hint_toward_unexplored(map_id, pos["x"], pos["y"])
            if map_id is not None and "x" in pos and "y" in pos
            else None
        )
        fully_explored = self.room_map.is_fully_explored(map_id) if map_id is not None else False
        explored_fraction = self.room_map.explored_fraction(map_id) if map_id is not None else 0.0
        if fully_explored:
            boundary_hint = (
                self.room_map.hint_toward_boundary(map_id, pos["x"], pos["y"])
                if map_id is not None and "x" in pos and "y" in pos
                else None
            )
            if boundary_hint:
                unexplored_hint = boundary_hint

        # Nearby people/objects from the game's own WRAM object memory. The
        # object memory is per-map, but right after a transition it still
        # holds the previous map's sprites for a few frames, so wait a couple
        # of snapshots before trusting it.
        in_battle = bool((state.get("battle") or {}).get("in_battle"))
        objects = []
        if map_id is not None:
            if map_id != self._last_objects_map:
                self._last_objects_map = map_id
                self._objects_grace = 2
                self._transition_grace = 3
            elif self._objects_grace > 0:
                self._objects_grace -= 1
                self._transition_grace = max(0, self._transition_grace - 1)
            elif not in_battle and "x" in pos and "y" in pos:
                self._transition_grace = max(0, self._transition_grace - 1)
                objects = read_objects(pb, pos["x"], pos["y"])
        objects_text = render_objects(objects)
        self.objects = objects

        # Learn any exit we just walked through.
        if map_id is not None and "x" in pos and "y" in pos:
            self._learn_warp(map_id, pos)

        goals = compute_goals(self, self.dialog_active, objects, fully_explored, state, turn)
        goals_text = render_goals(goals)
        self.goals = goals

        # Where the current room's exits are, relative to the player. Stateless:
        # derived live from the RAM warp table every frame.
        known_exits = ""
        if map_id is not None and "x" in pos and "y" in pos:
            exits = []
            for wx, wy, dest, direction in self.live_exits():
                if dest == map_id or (wx, wy) == (pos["x"], pos["y"]):
                    continue  # self-loop or the warp we're standing on
                dest_name = MAP_NAMES.get(dest, f"map {dest}")
                exits.append(
                    f"{dest_name} at tile ({wx},{wy}) ({_relative(wx - pos['x'], wy - pos['y'])}, step {direction})"
                )
            if exits:
                known_exits = "KNOWN EXITS: " + "; ".join(exits)

        return {
            "instructions": INSTRUCTIONS,
            "screen_text": "\n".join(screen_lines) if screen_lines else "(no text on screen)",
            "recent_hints": " | ".join(self.hints) if self.hints else "(none yet)",
            "collision_map": ascii_map,
            "tileset": collision.get("tileset"),
            "walkable_directions": walkable,
            "room_map": room_map_text,
            "objects": objects_text,
            "goals": goals_text,
            "known_exits": known_exits,
            "unexplored_hint": unexplored_hint,
            "explored_fraction": explored_fraction,
            "probe_exits": fully_explored,
            "location": {
                "map": (state.get("map") or {}).get("map_name"),
                "map_id": (state.get("map") or {}).get("map_id"),
                "x": (state.get("player") or {}).get("position", {}).get("x"),
                "y": (state.get("player") or {}).get("position", {}).get("y"),
                "facing": (state.get("player") or {}).get("facing"),
                "play_time": (state.get("player") or {}).get("play_time"),
            },
            "player": {
                "name": (state.get("player") or {}).get("name"),
                "money": (state.get("player") or {}).get("money"),
                "badges": (state.get("player") or {}).get("badges"),
            },
            "party": [
                {
                    "nickname": m.get("nickname"),
                    "species": m.get("species"),
                    "level": m.get("level"),
                    "hp": m.get("hp"),
                    "max_hp": m.get("max_hp"),
                    "status": m.get("status"),
                    "types": m.get("types"),
                    "moves": [mv.get("name") for mv in m.get("moves", [])],
                }
                for m in (state.get("party") or [])
            ],
            "bag": [{"item": b.get("item"), "quantity": b.get("quantity")} for b in (state.get("bag") or [])],
            "battle": state.get("battle"),
            "flags": {
                "has_pokedex": (state.get("flags") or {}).get("has_pokedex"),
                "has_oaks_parcel": (state.get("flags") or {}).get("has_oaks_parcel"),
                "pokedex_owned": (state.get("flags") or {}).get("pokedex_owned"),
                "pokedex_seen": (state.get("flags") or {}).get("pokedex_seen"),
                "badge_count": (state.get("flags") or {}).get("badge_count"),
            },
            "turn": turn,
            "recent_actions": self.memory.summary(),
            "dialog_active": self.dialog_active,
        }


def render_state_line(state: dict[str, Any]) -> str:
    """A single compact line summarising the state for stdout."""
    loc = state["location"]
    open_dirs = "".join(d[0] for d, ok in (state.get("walkable_directions") or {}).items() if ok)
    text = state.get("screen_text", "")
    text = " ".join(text.split())
    text = text[:44] + ("…" if len(text) > 44 else "")
    objects = state.get("objects", "")
    objects = objects.replace("\n", " | ")
    objects = objects[:60] + ("…" if len(objects) > 60 else "")
    exits = state.get("known_exits", "")
    if exits:
        exits = exits[:70] + ("…" if len(exits) > 70 else "")
    tail = f" | {objects}" if objects else ""
    tail += f" | {exits}" if exits else ""
    frac = state.get("explored_fraction")
    if frac is not None:
        tail += f" | explored {frac * 100:.0f}%"
    hints = state.get("recent_hints", "")
    if hints and hints != "(none yet)":
        latest = hints.split(" | ")[-1][:44]
        tail += f" | hint:{latest}"
    goals = state.get("goals", "")
    if goals:
        ids = ", ".join(g.split(":")[0] for g in goals.split(" | ") if ":" in g)
        tail += f" | goals:[{ids}]"
    return (
        f"turn {state['turn']} | {loc['map']} ({loc['x']},{loc['y']}) facing {loc['facing']} | "
        f"open:{open_dirs or '-'} | {tail} | text:{text!r}"
    )


def render_text_state(state: dict[str, Any]) -> str:
    """Render the state dict as compact text for console logging / debugging."""
    lines = []
    lines.append(
        f"[turn {state['turn']}] {state['location']['map']} "
        f"({state['location']['x']},{state['location']['y']}) facing {state['location']['facing']}  "
        f"${state['player'].get('money')}  badges={len(state['player'].get('badges') or [])}"
    )
    open_dirs = [d for d, ok in (state.get("walkable_directions") or {}).items() if ok]
    lines.append(f"  open: {', '.join(open_dirs) if open_dirs else 'none'}")
    hint = state.get("unexplored_hint")
    if hint:
        lines.append(f"  unexplored hint: {hint}")
    if state.get("probe_exits"):
        lines.append("  (room fully explored - probe the walls for the exit)")
    goals = state.get("goals")
    if goals:
        for g in goals.split(" | "):
            lines.append(f"  goal: {g}")
    room_map = state.get("room_map")
    if room_map and room_map != "(nothing explored yet)":
        for rm_line in room_map.splitlines():
            lines.append(f"  map|{rm_line}")
    if state["screen_text"].strip() and state["screen_text"] != "(no text on screen)":
        for ln in state["screen_text"].splitlines():
            lines.append(f"  text: {ln}")
    party = state["party"]
    if party:
        lines.append(
            "  party: "
            + " | ".join(
                f"{m['nickname']} {m['species']} L{m['level']} {m['hp']}/{m['max_hp']}HP" for m in party
            )
        )
    battle = state["battle"]
    if battle and battle.get("in_battle"):
        e = battle.get("enemy") or {}
        lines.append(
            f"  BATTLE vs {e.get('species')} Lv{e.get('level')} "
            f"{e.get('hp')}/{e.get('max_hp')}HP ({e.get('status')})"
        )
    return "\n".join(lines)
