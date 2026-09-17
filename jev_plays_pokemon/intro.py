"""Intro navigation for Pokemon Red (USA, sha1 ea9bcae6...).

Navigates title screen -> NEW GAME -> Oak's intro dialog -> player name
screen (select the RED preset) -> rival name screen (select the BLUE
preset) -> first playable frame in Red's bedroom.

Layout facts (empirically verified against PyBoy 2.7.0 + this ROM):
* Title menu: "NEW GAME" first char at tilemap_window[2,2] == 141.
* Dialog text renders on the WINDOW tilemap; the advance arrow sits at
  tilemap_window[18,16] == 238.
* Preset name screens put the cursor on "NEW NAME" (row 2); the first
  preset (RED / BLUE) is one row below and one "down" press selects it.
"""

from __future__ import annotations

import numpy as np

from jev_plays_pokemon.screen_text import _DECODING

_TITLE_NEW_GAME = 141
_DIALOG_ARROW = 238


def _dec_row(tiles) -> str:
    return "".join(_DECODING.get(int(t) & 0x1FF, " ") for t in tiles)


def _is_name_screen(pb) -> bool:
    w = np.asarray(pb.tilemap_window[:, :])
    row0 = _dec_row(w[0, :20])
    row2 = _dec_row(w[2, :20])
    return "NAME" in row0 or "NEW NAME" in row2


def _skip_dialogue(pb, max_wait: int = 4000) -> bool:
    """Press A to advance one dialog box; True if a dialog was present."""
    tries = 0
    while pb.tilemap_window[18, 16] != _DIALOG_ARROW and tries < max_wait:
        pb.tick(10)
        tries += 1
    if tries >= max_wait:
        return False
    pb.button("a")
    pb.tick(60)
    return True


def start_game(pb) -> None:
    """Navigate the full intro; returns at the bedroom (dialog on screen)."""
    pb.tick(120)

    # Title screen -> NEW GAME
    for _ in range(8000):
        if pb.tilemap_window[2, 2] == _TITLE_NEW_GAME:
            break
        if pb.tilemap_window[2, 2] == 130:  # Continue
            raise RuntimeError("Title screen shows CONTINUE (existing save).")
        pb.button("start")
        pb.tick(10)
    else:
        raise RuntimeError("Could not find the title menu.")

    pb.button("start")
    pb.tick(100)  # Transition to Oak talking

    # First name screen is the PLAYER (presets RED / ASH / JACK)
    for _ in range(60):
        if _is_name_screen(pb):
            break
        if not _skip_dialogue(pb):
            raise RuntimeError("Stalled before the player name screen.")
    else:
        raise RuntimeError("Could not reach the player name screen.")

    pb.button("down")  # NEW NAME -> RED
    pb.tick(10)
    pb.button("a")
    pb.tick(60)

    # Second name screen is the RIVAL (presets BLUE / GARY / JOHN)
    for _ in range(60):
        if _is_name_screen(pb):
            break
        if not _skip_dialogue(pb):
            raise RuntimeError("Stalled before the rival name screen.")
    else:
        raise RuntimeError("Could not reach the rival name screen.")

    pb.button("down")  # NEW NAME -> BLUE
    pb.tick(10)
    pb.button("a")
    pb.tick(60)

    # Remaining dialog + shrink transition into the bedroom
    for _ in range(60):
        if not _skip_dialogue(pb):
            break
    pb.tick(240)
