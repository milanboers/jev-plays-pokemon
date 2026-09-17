"""Overworld object awareness from the game's own WRAM.

Pokemon Red keeps an "object memory" for the current map in WRAM:

* ``0xC100..0xC1FF`` — 16 slots x $10 bytes: picture ID (identity), movement
  status, sprite image index (``$ff`` = off screen), screen X/Y in pixels,
  facing direction, animation state.
* ``0xC200..0xC2FF`` — a second 16-slot table (movement/delay data).

The player is always slot 0. Picture IDs map to NPC/item names via the
game's sprite constants (e.g. ``0x33`` = Mom). Combined with the player's
map coordinates this yields a clean spatial list: who is nearby, in which
direction, and how far — which the hardware OAM (PyBoy ``get_sprite``)
cannot provide (tile ids there are just animation frames).
"""

from __future__ import annotations

from typing import Any

OOM_BASE1 = 0xC100  # sprite state data 1 (identity, screen pos, facing)
OOM_BASE2 = 0xC200  # sprite state data 2 (movement data)

# 16px on-screen block grid the model already knows from the collision map.
_COL_LABELS = "ABCDEFGHIJ"
_PLAYER_BLOCK = (4, 4)  # the player always stands on block E5

SPRITE_NAMES: dict[int, str] = {
    0x00: "None",
    0x01: "Red",
    0x02: "Blue",
    0x03: "Oak",
    0x04: "Youngster",
    0x05: "Monster",
    0x06: "Cooltrainer F",
    0x07: "Cooltrainer M",
    0x08: "Little Girl",
    0x09: "Bird",
    0x0A: "Middle Aged Man",
    0x0B: "Gambler",
    0x0C: "Super Nerd",
    0x0D: "Girl",
    0x0E: "Hiker",
    0x0F: "Beauty",
    0x10: "Gentleman",
    0x11: "Daisy",
    0x12: "Biker",
    0x13: "Sailor",
    0x14: "Cook",
    0x15: "Bike Shop Clerk",
    0x16: "Mr Fuji",
    0x17: "Giovanni",
    0x18: "Rocket",
    0x19: "Channeler",
    0x1A: "Waiter",
    0x1B: "Silph Worker F",
    0x1C: "Middle Aged Woman",
    0x1D: "Brunette Girl",
    0x1E: "Lance",
    0x1F: "Unused Scientist",
    0x20: "Scientist",
    0x21: "Rocker",
    0x22: "Swimmer",
    0x23: "Safari Zone Worker",
    0x24: "Gym Guide",
    0x25: "Gramps",
    0x26: "Clerk",
    0x27: "Fishing Guru",
    0x28: "Granny",
    0x29: "Nurse",
    0x2A: "Link Receptionist",
    0x2B: "Silph President",
    0x2C: "Silph Worker M",
    0x2D: "Warden",
    0x2E: "Captain",
    0x2F: "Fisher",
    0x30: "Koga",
    0x31: "Guard",
    0x32: "Unused Guard",
    0x33: "Mom",
    0x34: "Balding Guy",
    0x35: "Little Boy",
    0x36: "Unused Gameboy Kid",
    0x37: "Gameboy Kid",
    0x38: "Fairy",
    0x39: "Agatha",
    0x3A: "Bruno",
    0x3B: "Lorelei",
    0x3C: "Seel",
    0x3D: "Poke Ball",
    0x3E: "Fossil",
    0x3F: "Boulder",
    0x40: "Paper",
    0x41: "Pokedex",
    0x42: "Clipboard",
    0x43: "Snorlax",
    0x45: "Old Amber",
    0x48: "Gambler Asleep",
}

_FACING = {0x00: "down", 0x04: "up", 0x08: "left", 0x0C: "right"}


def _block_of(x: int, y: int) -> tuple[int, int] | None:
    """Screen pixels -> 16px block (col, row); None when off the grid."""
    col, row = (x + 8) // 16, (y + 16) // 16
    if 0 <= col < 10 and 0 <= row < 9:
        return col, row
    return None


def _cell_label(block: tuple[int, int]) -> str:
    return f"{_COL_LABELS[block[0]]}{block[1] + 1}"


def _relative(dx: int, dy: int) -> str:
    """``dx``/``dy`` (tiles east/north, + = east/north) -> human text."""
    parts = []
    if dx != 0:
        parts.append(f"{abs(dx)} tile{'s' if abs(dx) != 1 else ''} {'EAST' if dx > 0 else 'WEST'}")
    if dy != 0:
        parts.append(f"{abs(dy)} tile{'s' if abs(dy) != 1 else ''} {'SOUTH' if dy > 0 else 'NORTH'}")
    if not parts:
        return "on your tile"
    return ", ".join(parts)


def read_objects(pb, player_x: int, player_y: int) -> list[dict[str, Any]]:
    """Return active map objects (excluding the player, slot 0).

    Each object: ``name``, ``on_screen``, ``screen_cell`` (A1..J9 when on
    screen), ``map`` (absolute tile coords), ``relative`` (distance text from
    the player), and ``facing``.
    """
    mem = pb.memory
    objects: list[dict[str, Any]] = []
    for i in range(16):
        base = OOM_BASE1 + i * 0x10
        picture = mem[base]
        status = mem[base + 1]
        image_index = mem[base + 2]
        y_px = mem[base + 4]
        x_px = mem[base + 6]
        facing = mem[base + 9]
        if i == 0 or picture == 0 or status == 0:
            continue

        block = _block_of(x_px, y_px)
        on_screen = image_index != 0xFF
        obj = {
            "name": SPRITE_NAMES.get(picture, f"Sprite {picture:#x}"),
            "picture": picture,
            "on_screen": bool(on_screen),
            "facing": _FACING.get(facing, "?"),
        }
        if on_screen and block is not None:
            obj["screen_cell"] = _cell_label(block)
            map_x = player_x + (block[0] - _PLAYER_BLOCK[0])
            map_y = player_y + (block[1] - _PLAYER_BLOCK[1])
            obj["map"] = (map_x, map_y)
            obj["relative"] = _relative(map_x - player_x, map_y - player_y)
        else:
            obj["screen_cell"] = None
            obj["map"] = None
            obj["relative"] = "nearby (off screen)"
        objects.append(obj)
    return objects


def render_objects(objects: list[dict[str, Any]]) -> str:
    """Compact ``NEARBY OBJECTS / CHARACTERS`` block for the state."""
    if not objects:
        return "(no other people or objects nearby)"
    lines = ["NEARBY OBJECTS / CHARACTERS:"]
    for o in objects:
        if o["on_screen"] and o["screen_cell"]:
            lines.append(f"- {o['name']}: at {o['screen_cell']} ({o['relative']}), facing {o['facing']}")
        else:
            lines.append(f"- {o['name']}: {o['relative']}, facing {o['facing']}")
    return "\n".join(lines)
