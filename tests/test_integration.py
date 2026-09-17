"""Integration tests against the real emulator + Pokemon Red ROM.

These boot the Game Boy headless, play through the intro, and verify the
state pipeline the agent relies on: intro navigation, on-screen text
decoding, the structured state snapshot, and physical movement.
"""

from __future__ import annotations

import pytest

from jev_plays_pokemon.agent import ACTION_FRAMES
from jev_plays_pokemon.screen_text import read_screen_text
from jev_plays_pokemon.state import StateBuilder

pytestmark = pytest.mark.integration


def _clear_dialog(pb):
    for _ in range(20):
        if int(pb.tilemap_window[18, 16]) != 238 and not read_screen_text(pb):
            break
        pb.button("a")
        pb.tick(40)


class TestIntro:
    def test_intro_lands_in_bedroom(self, booted):
        reader = StateBuilder(booted).reader
        info = reader.read_map_info()
        assert info["map_id"] == 38, f"expected Red's House 2F, got {info}"
        pos = reader.read_player()["position"]
        assert (pos["x"], pos["y"]) == (3, 6)

    def test_player_has_no_party_and_default_money(self, booted):
        reader = StateBuilder(booted).reader
        assert reader.read_party() == []
        assert reader.read_player()["money"] == 3000


class TestScreenText:
    def test_menu_text_decodes(self, booted):
        pb = booted._pyboy
        _clear_dialog(pb)
        pb.button("start")
        pb.tick(30)
        text = "\n".join(read_screen_text(pb))
        # NOTE: "POKéMON" uses the Poke Ball ligature (é) in the game's encoding.
        for label in ("POK", "ITEM", "SAVE", "EXIT"):
            assert label in text, f"menu label missing from {text!r}"
        pb.button("start")
        pb.tick(30)

    def test_overworld_has_no_text(self, booted):
        pb = booted._pyboy
        _clear_dialog(pb)
        pb.tick(30)
        assert read_screen_text(pb) == []


class TestStateSnapshot:
    def test_snapshot_shape(self, booted):
        builder = StateBuilder(booted)
        state = builder.snapshot(1)
        for key in (
            "instructions",
            "objective",
            "screen_text",
            "recent_hints",
            "collision_map",
            "walkable_directions",
            "room_map",
            "objects",
            "location",
            "player",
            "party",
            "bag",
            "battle",
            "flags",
            "turn",
            "recent_actions",
            "dialog_active",
        ):
            assert key in state, f"missing key {key}"
        assert set(state["walkable_directions"]) == {"up", "down", "left", "right"}
        assert state["turn"] == 1
        assert state["location"]["map"] == "Red's House 2F"

    def test_dialog_active_detects_text_box(self, emu):
        pb = emu._pyboy
        pb.set_emulation_speed(0)
        # Reach the title menu, then start a new game (Oak starts talking).
        pb.tick(120)
        for _ in range(8000):
            if int(pb.tilemap_window[2, 2]) == 141:
                break
            pb.button("start")
            pb.tick(10)
        pb.button("start")
        pb.tick(100)
        builder = StateBuilder(emu)
        for _ in range(600):
            if builder.dialog_active:
                break
            pb.tick(10)
        assert builder.dialog_active is True, "expected a dialog with advance arrow"
        # Clear it and confirm the flag drops.
        _clear_dialog(pb)
        assert builder.dialog_active is False


class TestMovement:
    def test_hold_press_moves_player(self, booted):
        reader = StateBuilder(booted).reader
        _clear_dialog(booted._pyboy)
        before = reader.read_player()["position"]
        direction = "down" if before["y"] < 7 else "up"
        frames, settle = ACTION_FRAMES[f"walk_{direction}"]
        booted.press(direction, frames)
        booted.tick(settle)
        after = reader.read_player()["position"]
        assert (after["x"], after["y"]) != (before["x"], before["y"])
