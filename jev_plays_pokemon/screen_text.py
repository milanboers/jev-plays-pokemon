"""Decode the on-screen dialog / menu text into plain text for the model.

Pokemon Gen 1 renders dialog boxes and menus as tiles on the Game Boy's
tilemaps; the tile identifiers ARE the pokered text bytes (0x80..0xFF).
PyBoy's ``game_wrapper.game_area()`` returns the currently active layer
(window during dialog, background otherwise) as an 18x20 tile grid, so a
single pass decodes dialogs, name screens, menus and battle commands.
"""

from __future__ import annotations

import numpy as np

_DECODING = {
    0x4F: "\n",
    0x51: "\n",
    0x55: "\n",  # line breaks
    0x49: " ",
    0x4B: "",
    0x4C: " ",
    0x4E: " ",
    0x50: "",
    0x57: "",
    0x58: " ",
    0x7F: " ",
    383: " ",  # spaces (incl. blank menu tile)
    0x9A: "(",
    0x9B: ")",
    0x9C: ":",
    0x9D: ";",
    0x9E: "[",
    0x9F: "]",
    0xBA: "é",
    0xBB: "'d",
    0xBC: "'l",
    0xBD: "'s",
    0xBE: "'t",
    0xBF: "'v",
    0xE0: "'",
    0xE1: "PK",
    0xE2: "MN",
    0xE3: "-",
    0xE4: "'r",
    0xE5: "'m",
    0xE6: "?",
    0xE7: "!",
    0xE8: ".",
    0xEC: "▷",
    0xED: "▶",
    0xEE: "▼",
    0xEF: "♂",
    0xF0: "¥",
    0xF1: "×",
    0xF2: ".",
    0xF3: "/",
    0xF4: ",",
    0xF5: "♀",
}
_DECODING.update({0x80 + i: chr(ord("A") + i) for i in range(26)})
_DECODING.update({0xA0 + i: chr(ord("a") + i) for i in range(26)})
_DECODING.update({0xF6 + i: str(i) for i in range(10)})


def _decode_row(tiles) -> str:
    return "".join(_DECODING.get(int(t) & 0x1FF, " ") for t in tiles)


def read_screen_text(pb) -> list[str]:
    """Return the current on-screen text as a list of non-empty lines.

    Decodes every row of the active game area, keeps only rows that
    contain real text characters, and joins consecutive rows per line
    (dialog boxes span two 8px rows per 16px text line).
    """
    ga = np.asarray(pb.game_wrapper.game_area())
    lines: list[str] = []
    for r in range(ga.shape[0]):
        row = _decode_row(ga[r]).rstrip()
        if not row.strip():
            continue
        if any(_DECODING.get(int(t) & 0x1FF) for t in ga[r]):
            lines.append(row)
    return [ln.strip() for ln in lines if ln.strip()]
