"""Unit tests for the pure logic (no emulator / no API required)."""

from __future__ import annotations

from jev_plays_pokemon.agent import Decision, pick_action
from jev_plays_pokemon.navigation import astar, make_walkable
from jev_plays_pokemon.objects import SPRITE_NAMES, _relative, read_objects, render_objects
from jev_plays_pokemon.state import (
    RoomMap,
    _decide_objective,
    _walkable_directions,
    capture_hint,
    compute_goals,
    render_state_line,
)


def make_decision(action: str, conf: float, margin: float, nouls: dict[str, float] | None = None) -> Decision:
    nouls = nouls or {}
    if action not in nouls:
        nouls[action] = conf
    return Decision(action=action, confidence=conf, margin=margin, nouls=nouls, menu_open=0.0)


class TestWalkableDirections:
    def test_base_from_collision(self):
        collision = {"walkable": [[True] * 10 for _ in range(9)]}
        w = _walkable_directions(collision, set(), 38, 3, 6)
        assert w == {"up": True, "down": True, "left": True, "right": True}

    def test_learned_wall_blocks(self):
        collision = {"walkable": [[True] * 10 for _ in range(9)]}
        walls = {(38, 3, 6, "down")}
        w = _walkable_directions(collision, walls, 38, 3, 6)
        assert w["down"] is False
        assert w["up"] is True

    def test_wall_scoped_to_position(self):
        collision = {"walkable": [[True] * 10 for _ in range(9)]}
        walls = {(38, 3, 6, "down")}
        w = _walkable_directions(collision, walls, 38, 4, 6)
        assert w["down"] is True


class TestRoomMap:
    def test_update_records_walkable_and_blocked(self):
        rm = RoomMap()
        rm.update(38, 3, 6, {"up": True, "down": False, "left": True, "right": True})
        assert rm.cells[38][(3, 6)] == RoomMap.WALKABLE
        assert rm.cells[38][(3, 7)] == RoomMap.BLOCKED
        assert rm.cells[38][(3, 5)] == RoomMap.WALKABLE

    def test_render_marks_player(self):
        rm = RoomMap()
        rm.update(38, 3, 6, {"up": True, "down": False, "left": True, "right": True})
        text = rm.render(38, (3, 6))
        assert "@" in text

    def test_hint_toward_unexplored(self):
        rm = RoomMap()
        room = rm.cells.setdefault(38, {})
        # A sealed 3-wide corridor; the cell to the right of x=5 is unknown.
        for x in range(3, 6):
            room[(x, 6)] = RoomMap.WALKABLE
            room[(x, 5)] = RoomMap.BLOCKED
            room[(x, 7)] = RoomMap.BLOCKED
        for y in (5, 6, 7):
            room[(2, y)] = RoomMap.BLOCKED
        hint = rm.hint_toward_unexplored(38, 3, 6)
        assert hint == "right"

    def test_fully_explored_after_sealing(self):
        rm = RoomMap()
        # Player at (3,6) with all four neighbours known.
        rm.update(38, 3, 6, {"up": False, "down": False, "left": False, "right": False})
        # The neighbours are marked blocked; but the cell itself is walkable.
        assert rm.is_fully_explored(38) is True

    def test_boundary_hint_points_to_wall(self):
        rm = RoomMap()
        room = rm.cells.setdefault(38, {})
        # Sealed corridor: walls above/below/left of every cell.
        for x in range(3, 6):
            room[(x, 6)] = RoomMap.WALKABLE
            room[(x, 5)] = RoomMap.BLOCKED
            room[(x, 7)] = RoomMap.BLOCKED
        room[(2, 6)] = RoomMap.BLOCKED
        # The nearest wall from (3,6) is straight up.
        assert rm.hint_toward_boundary(38, 3, 6) == "up"

    def test_boundary_hint_walks_to_distant_wall(self):
        rm = RoomMap()
        room = rm.cells.setdefault(38, {})
        # Fully sealed 4x3 room; the player starts in the interior.
        walkable = {
            (3, 6),
            (4, 6),
            (5, 6),
            (6, 6),
            (3, 5),
            (4, 5),
            (5, 5),
            (6, 5),
            (3, 7),
            (4, 7),
            (5, 7),
            (6, 7),
        }
        for cell in walkable:
            room[cell] = RoomMap.WALKABLE
        for x in range(2, 8):
            for y in range(4, 9):
                if (x, y) not in walkable:
                    room[(x, y)] = RoomMap.BLOCKED
        # Nearest wall from the interior cell (4,6) is straight up at (4,4).
        assert rm.hint_toward_boundary(38, 4, 6) == "up"


class TestPickAction:
    OPEN = {"up": True, "down": True, "left": True, "right": True}

    def test_confident_action_used(self):
        d = make_decision("walk_up", conf=0.7, margin=0.3, nouls={"walk_up": 0.7})
        assert pick_action(d, dialog_active=False, walkable=self.OPEN) == "walk_up"

    def test_low_confidence_falls_back_to_dialog(self):
        d = make_decision("walk_up", conf=0.3, margin=0.02, nouls={"walk_up": 0.3})
        assert pick_action(d, dialog_active=True, walkable=self.OPEN) == "press_a"

    def test_low_confidence_walks_hint(self):
        d = make_decision("walk_up", conf=0.3, margin=0.02, nouls={"walk_up": 0.3})
        assert pick_action(d, dialog_active=False, walkable=self.OPEN, explore_hint="left") == "walk_left"

    def test_blocks_wall(self):
        blocked = {"up": False, "down": True, "left": True, "right": True}
        d = make_decision("walk_up", conf=0.9, margin=0.5, nouls={"walk_up": 0.9})
        assert pick_action(d, dialog_active=False, walkable=blocked) == "wait"

    def test_all_blocked_allows_walk(self):
        blocked = {"up": False, "down": False, "left": False, "right": False}
        d = make_decision("walk_up", conf=0.9, margin=0.5, nouls={"walk_up": 0.9})
        assert pick_action(d, dialog_active=False, walkable=blocked) == "walk_up"

    def test_probe_mode_allows_wall(self):
        blocked = {"up": False, "down": True, "left": True, "right": True}
        d = make_decision("walk_up", conf=0.9, margin=0.5, nouls={"walk_up": 0.9})
        assert pick_action(d, dialog_active=False, walkable=blocked, probe_exits=True) == "walk_up"

    def test_anti_reversal(self):
        nouls = {"walk_up": 0.7, "walk_left": 0.4, "walk_right": 0.3}
        d = make_decision("walk_up", conf=0.7, margin=0.3, nouls=nouls)
        assert pick_action(d, dialog_active=False, walkable=self.OPEN, last_action="walk_down") == "walk_left"


class TestNavigation:
    def test_astar_straight_line(self):
        def walkable(x, y):
            return 0 <= x <= 4 and 0 <= y <= 4

        path = astar((0, 0), (3, 0), walkable)
        assert path == ["right", "right", "right"]

    def test_astar_around_wall(self):
        def walkable(x, y):
            if (x, y) == (2, 0):
                return False  # wall
            return 0 <= x <= 4 and 0 <= y <= 2

        path = astar((0, 0), (4, 0), walkable)
        assert path is not None
        # Any shortest route that avoids (2,0) is valid.
        assert len(path) == 6
        x, y = 0, 0
        moves = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        for step in path:
            dx, dy = moves[step]
            x, y = x + dx, y + dy
            assert (x, y) != (2, 0), "walked through the wall"
        assert (x, y) == (4, 0)

    def test_astar_goal_through_wall(self):
        def walkable(x, y):
            return x == 0 and y == 0  # only the start is walkable

        # The goal cell (a wall tile like a door) may be stepped onto.
        path = astar((0, 0), (1, 0), walkable)
        assert path == ["right"]

    def test_astar_no_path(self):
        def walkable(x, y):
            return False

        assert astar((0, 0), (3, 0), walkable) is None

    def test_make_walkable_uses_collision_on_screen(self):
        grid = [[True] * 10 for _ in range(9)]
        grid[4][5] = False  # block the cell right of the player
        collision = {"walkable": grid}
        fn = make_walkable(collision, player_x=10, player_y=10, room_cells={})
        assert fn(11, 10) is False  # on screen, blocked by collision
        assert fn(10, 11) is True  # on screen, walkable
        assert fn(50, 50) is False  # off screen, unknown -> blocked


class TestGoals:
    def _builder(self, warps=None):
        class B:
            def __init__(self):
                self.known_warps = warps or {}
                self.talk_cooldown = {}
                self.talked_this_visit = set()
                self.goal_cooldown = {}
                self.map_history = []

            def live_exits(self):
                """Emit all fake warps (the real builder reads the live RAM table)."""
                return [w for warps in self.known_warps.values() for w in warps]

        return B()

    def test_battle_goal_only_in_battle(self):
        b = self._builder()
        state = {"battle": {"in_battle": True, "enemy": {"species": "Rattata"}}}
        goals = compute_goals(b, dialog_active=False, objects=[], fully_explored=False, state=state, turn=0)
        assert [g["id"] for g in goals] == ["battle"]

    def test_dialog_prunes_to_advance_text(self):
        b = self._builder()
        goals = compute_goals(b, dialog_active=True, objects=[], fully_explored=False, state={}, turn=0)
        assert [g["id"] for g in goals] == ["advance_text"]

    def test_talk_goal_from_objects(self):
        b = self._builder()
        state = {"map": {"map_id": 37}}
        objs = [
            {"name": "Mom", "on_screen": True, "map": (5, 4), "screen_cell": "C8", "relative": "2 tiles WEST"}
        ]
        goals = compute_goals(b, dialog_active=False, objects=objs, fully_explored=False, state=state, turn=0)
        ids = [g["id"] for g in goals]
        assert "talk_to_0" in ids
        assert "Mom" in next(g["desc"] for g in goals if g["id"] == "talk_to_0")

    def test_talk_goal_respects_cooldown(self):
        b = self._builder()
        b.talk_cooldown[(37, 0x33)] = 5  # Mom, keyed by (map_id, sprite)
        state = {"map": {"map_id": 37}}
        objs = [
            {
                "name": "Mom",
                "picture": 0x33,
                "on_screen": True,
                "map": (5, 4),
                "screen_cell": "C8",
                "relative": "2 tiles WEST",
            }
        ]
        goals = compute_goals(b, dialog_active=False, objects=objs, fully_explored=False, state=state, turn=3)
        assert all(g["id"] != "talk_to_0" for g in goals)
        # After the cooldown expires the NPC is offered again (multi-stage dialog).
        goals = compute_goals(b, dialog_active=False, objects=objs, fully_explored=False, state=state, turn=6)
        assert any(g["id"] == "talk_to_0" for g in goals)

    def test_talk_suppressed_for_whole_visit(self):
        b = self._builder()
        b.talked_this_visit = {(37, 0x33)}  # Mom already talked to this visit
        state = {"map": {"map_id": 37}}
        objs = [
            {
                "name": "Mom",
                "picture": 0x33,
                "on_screen": True,
                "map": (5, 4),
                "screen_cell": "C8",
                "relative": "2 tiles WEST",
            }
        ]
        goals = compute_goals(b, dialog_active=False, objects=objs, fully_explored=False, state=state, turn=3)
        assert all(g["id"] != "talk_to_0" for g in goals)

    def test_exit_goal_needs_known_warp_or_full_exploration(self):
        b = self._builder()
        state = {"map": {"map_id": 40}, "tileset": 1}
        goals = compute_goals(b, dialog_active=False, objects=[], fully_explored=False, state=state, turn=0)
        assert all(g["id"] != "reach_exit" for g in goals)
        # Once fully explored, the exit goal appears.
        goals = compute_goals(b, dialog_active=False, objects=[], fully_explored=True, state=state, turn=0)
        assert any(g["id"] == "reach_exit" for g in goals)

    def test_known_warp_triggers_exit_goal(self):
        b = self._builder({40: [(3, 8, 0, "down")]})
        state = {"map": {"map_id": 40}, "party": [{"species": "X"}], "tileset": 1}
        # A known warp is always offered, even before the room is explored.
        goals = compute_goals(b, dialog_active=False, objects=[], fully_explored=False, state=state, turn=0)
        assert any(g["id"] == "exit_to_0" for g in goals)

    def test_no_warp_probe_only_when_explored(self):
        b = self._builder()  # no warps
        state = {"map": {"map_id": 40}, "party": [{"species": "X"}], "tileset": 1}
        goals = compute_goals(b, dialog_active=False, objects=[], fully_explored=False, state=state, turn=0)
        assert all(g["id"] != "reach_exit" for g in goals)
        # Fully explored + no known warp: boundary-probe exit appears.
        goals = compute_goals(b, dialog_active=False, objects=[], fully_explored=True, state=state, turn=0)
        assert any(g["id"] == "reach_exit" for g in goals)

    def test_self_loop_warp_is_dropped(self):
        b = self._builder({40: [(3, 8, 40, "down")]})  # map 40 -> map 40 (bogus)
        state = {"map": {"map_id": 40}, "party": [{"species": "X"}], "tileset": 1}
        goals = compute_goals(b, dialog_active=False, objects=[], fully_explored=True, state=state, turn=0)
        assert all(not g["id"].startswith("exit_to_") for g in goals)

    def test_explore_suppressed_after_recent_run(self):
        b = self._builder()
        b.goal_cooldown["explore"] = 10
        state = {"map": {"map_id": 40}, "tileset": 1}
        goals = compute_goals(b, dialog_active=False, objects=[], fully_explored=False, state=state, turn=3)
        assert all(g["id"] != "explore" for g in goals)
        # Expired cooldown -> explore is offered again.
        goals = compute_goals(b, dialog_active=False, objects=[], fully_explored=False, state=state, turn=11)
        assert any(g["id"] == "explore" for g in goals)

    def test_explore_fallback_when_only_wait(self):
        # Even when nothing else is offered, explore remains as a fallback so
        # the goal list is never just "wait" (no cooldown here).
        b = self._builder()
        state = {"map": {"map_id": 40}, "tileset": 1}
        goals = compute_goals(b, dialog_active=False, objects=[], fully_explored=False, state=state, turn=3)
        assert any(g["id"] == "explore" for g in goals)
        assert any(g["id"] == "wait" for g in goals)


class TestObjects:
    def test_sprite_names(self):
        assert SPRITE_NAMES[0x01] == "Red"
        assert SPRITE_NAMES[0x33] == "Mom"
        assert SPRITE_NAMES[0x03] == "Oak"

    def test_relative_directions(self):
        assert _relative(0, 0) == "on your tile"
        assert _relative(2, 0) == "2 tiles EAST"
        assert _relative(-1, 0) == "1 tile WEST"
        assert _relative(0, 3) == "3 tiles SOUTH"
        assert _relative(0, -2) == "2 tiles NORTH"

    def test_render_empty(self):
        assert "no other" in render_objects([])

    def test_read_objects_parses_wram(self):
        class FakePB:
            def __init__(self, mem):
                self.memory = mem

        mem = bytearray(0xD400)
        # Player slot 0: picture=0x01, status=1, img=0x0e, Y=60, X=64, facing right.
        mem[0xC100 + 0x09] = 0x0C
        mem[0xC100 + 0x0E] = 0x0E
        mem[0xC100 + 0x04] = 60
        mem[0xC100 + 0x06] = 64
        mem[0xC100 + 0x01] = 1
        mem[0xC100 + 0x00] = 0x01
        # Mom slot 1: picture=0x33, status=1, img=0x18 (on screen), Y=108, X=32, facing left.
        base = 0xC100 + 0x10
        mem[base + 0x00] = 0x33
        mem[base + 0x01] = 1
        mem[base + 0x02] = 0x18
        mem[base + 0x04] = 108
        mem[base + 0x06] = 32
        mem[base + 0x09] = 0x08
        # Player at map (7,1); Mom at screen block C8 -> map (5,4).
        objs = read_objects(FakePB(mem), 7, 1)
        assert len(objs) == 1
        mom = objs[0]
        assert mom["name"] == "Mom"
        assert mom["on_screen"] is True
        assert mom["screen_cell"] == "C8"
        assert mom["map"] == (5, 4)
        assert mom["relative"] == "2 tiles WEST, 3 tiles SOUTH"
        assert mom["facing"] == "left"

    def test_off_screen_object_has_no_cell(self):
        class FakePB:
            def __init__(self, mem):
                self.memory = mem

        mem = bytearray(0xD400)
        base = 0xC100 + 0x10
        mem[base + 0x00] = 0x0D  # Girl
        mem[base + 0x01] = 1
        mem[base + 0x02] = 0xFF  # off screen
        objs = read_objects(FakePB(mem), 7, 1)
        assert objs[0]["on_screen"] is False
        assert objs[0]["screen_cell"] is None
        assert "off screen" in objs[0]["relative"]


class TestHintMemory:
    def _hints(self):
        from collections import deque

        return deque(maxlen=8)

    def test_merges_consecutive_pages_into_one_conversation(self):
        hints = self._hints()
        capture_hint(
            hints,
            dialog_active=True,
            screen_lines=["Prof. Oak is looking for you."],
            continuous=False,
            speaker="Mom",
        )
        capture_hint(
            hints,
            dialog_active=True,
            screen_lines=["Go see him right away."],
            continuous=True,
            speaker="Mom",
        )
        capture_hint(
            hints,
            dialog_active=True,
            screen_lines=["Go see him right away."],
            continuous=True,
            speaker="Mom",
        )
        assert len(hints) == 1
        assert hints[0].startswith("Mom: ")
        assert "Prof. Oak is looking for you." in hints[0]
        assert "Go see him right away." in hints[0]

    def test_new_dialog_starts_a_fresh_hint(self):
        hints = self._hints()
        capture_hint(hints, dialog_active=True, screen_lines=["First conversation."], continuous=False)
        capture_hint(hints, dialog_active=False, screen_lines=[], continuous=True)  # box closed
        capture_hint(hints, dialog_active=True, screen_lines=["Second conversation."], continuous=False)
        assert list(hints) == ["an NPC: First conversation.", "an NPC: Second conversation."]

    def test_keeps_last_eight_conversations(self):
        hints = self._hints()
        for i in range(9):
            capture_hint(
                hints, dialog_active=True, screen_lines=[f"Conversation number {i}."], continuous=False
            )
        assert len(hints) == 8
        assert list(hints) == [f"an NPC: Conversation number {i}." for i in range(1, 9)]

    def test_skips_menus_and_short_text(self):
        hints = self._hints()
        capture_hint(hints, dialog_active=True, screen_lines=["▶ POKEMON", "ITEM"])
        capture_hint(hints, dialog_active=True, screen_lines=["O"])
        assert len(hints) == 0

    def test_ignores_stale_same_page_and_inactive(self):
        hints = self._hints()
        capture_hint(hints, dialog_active=True, screen_lines=["Prof. Oak is looking for you. ▼"])
        capture_hint(
            hints, dialog_active=True, screen_lines=["Prof. Oak is looking for you. ▼"], continuous=True
        )
        capture_hint(hints, dialog_active=False, screen_lines=["Something new."], continuous=True)
        assert list(hints) == ["an NPC: Prof. Oak is looking for you."]


class TestMicroPath:
    def test_micro_action_from_goal_decision(self):
        """The menu/battle path must not crash on a GoalDecision."""
        from jev_plays_pokemon.agent import GoalDecision
        from jev_plays_pokemon.main import _micro_action

        gd = GoalDecision(
            goal="explore",
            confidence=0.5,
            nouls={"press_a": 0.7, "walk_up": 0.6, "walk_down": 0.2, "menu_open": 0.8},
            menu_open=0.8,
        )
        state = {
            "walkable_directions": {"up": True, "down": True, "left": True, "right": True},
            "unexplored_hint": "up",
            "probe_exits": False,
        }
        action = _micro_action(gd, dialog_active=False, state=state, last_action=None)
        assert action in (
            "press_a",
            "press_b",
            "press_start",
            "walk_up",
            "walk_down",
            "walk_left",
            "walk_right",
            "wait",
        )

    def test_goal_decision_summary(self):
        from jev_plays_pokemon.agent import GoalDecision

        gd = GoalDecision(
            goal="explore", confidence=0.5, nouls={"press_a": 0.7, "menu_open": 0.1}, menu_open=0.1
        )
        assert "press_a=0.70" in gd.summary()


class TestRenderStateLine:
    def test_single_line_compact(self):
        state = {
            "turn": 7,
            "location": {"map": "Red's House 1F", "x": 3, "y": 6, "facing": "up"},
            "walkable_directions": {"up": False, "down": True, "left": True, "right": True},
            "objective": "Get your starter Pokémon",
            "screen_text": "The wild RATTATA appeared!",
        }
        line = render_state_line(state)
        assert line.startswith("turn 7 | Red's House 1F (3,6) facing up")
        assert "open:dlr" in line
        assert "\n" not in line

    def test_truncates_long_fields(self):
        state = {
            "turn": 1,
            "location": {"map": "X", "x": 0, "y": 0, "facing": "up"},
            "walkable_directions": {},
            "objective": "a" * 200,
            "screen_text": "b" * 200,
        }
        line = render_state_line(state)
        assert len(line) < 220


class TestObjective:
    def test_no_party_wants_first_pokemon(self):
        state = {"party": [], "flags": {}, "player": {}}
        objective = _decide_objective(state)
        assert "Pokémon" in objective
        # High-level only: no scenario spelled out.
        assert "Oak" not in objective and "lab" not in objective.lower()

    def test_badges_objective_is_high_level(self):
        state = {
            "party": [{"species": "Charmander"}],
            "flags": {"has_oaks_parcel": True, "has_pokedex": False, "badge_count": 2},
            "player": {"badge_count": 2},
        }
        objective = _decide_objective(state)
        # Even with parcel/pokedex flags set, the goal stays high-level.
        assert "badges" in objective.lower()
        assert "Parcel" not in objective and "Oak" not in objective
        for banned in ("Brock", "Misty", "Surge", "Pewter", "Cerulean"):
            assert banned not in objective

    def test_all_badges_wants_champion(self):
        state = {
            "party": [{"species": "Charizard"}],
            "flags": {"has_oaks_parcel": False, "has_pokedex": True, "badge_count": 8},
            "player": {"badge_count": 8},
        }
        assert "Champion" in _decide_objective(state)


class TestSampleGoal:
    def test_confident_goal_wins_but_not_certainly(self):
        import random

        from jev_plays_pokemon.agent import _sample_goal

        random.seed(0)
        probs = {"explore": 0.98, "talk_to_0": 0.01, "wait": 0.01}
        counts: dict[str, int] = {}
        for _ in range(4000):
            goal, _conf = _sample_goal(probs)
            counts[goal] = counts.get(goal, 0) + 1
        # Flattened distribution: the 0.98 goal wins a big majority, but the
        # alternatives keep a real (not tiny) share - so we never hard-lock.
        assert counts["explore"] > 0.5 * 4000
        assert counts["talk_to_0"] + counts["wait"] > 0.05 * 4000

    def test_even_p1_is_not_100_percent(self):
        import random

        from jev_plays_pokemon.agent import _sample_goal

        random.seed(4)
        probs = {"a": 1.0, "b": 0.0}
        counts: dict[str, int] = {"a": 0, "b": 0}
        for _ in range(4000):
            goal, _conf = _sample_goal(probs)
            counts[goal] += 1
        # The floor guarantees b keeps real mass even when the model says 0.
        assert counts["b"] > 0.05 * 4000
        assert counts["a"] > counts["b"]

    def test_near_tie_is_broken_stochastically(self):
        import random

        from jev_plays_pokemon.agent import _sample_goal

        random.seed(1)
        probs = {"explore": 0.51, "talk_to_0": 0.49}
        counts: dict[str, int] = {}
        for _ in range(2000):
            goal, _conf = _sample_goal(probs)
            counts[goal] = counts.get(goal, 0) + 1
        # Both get a meaningful share; argmax would give 100% explore.
        assert counts["explore"] > 0
        assert counts["talk_to_0"] > 0
        assert abs(counts["explore"] - counts["talk_to_0"]) < 0.2 * 2000

    def test_confidence_returns_sampled_probability(self):
        import random

        from jev_plays_pokemon.agent import _sample_goal

        random.seed(2)
        probs = {"a": 0.9, "b": 0.1}
        goal, conf = _sample_goal(probs, temperature=1.0)
        assert goal in ("a", "b")
        assert 0.0 <= conf <= 1.0

    def test_temperature_1_keeps_ratio(self):
        import random

        from jev_plays_pokemon.agent import _sample_goal

        random.seed(3)
        probs = {"a": 0.25, "b": 0.75}
        counts: dict[str, int] = {"a": 0, "b": 0}
        for _ in range(4000):
            goal, _conf = _sample_goal(probs, temperature=1.0)
            counts[goal] += 1
        assert counts["b"] > counts["a"]  # b still preferred
