# Vendored from pokemon-agent (https://github.com/NousResearch/pokemon-agent), MIT License, Copyright (c) Nous Research.
# See LICENSE for details.
"""Gen 1 collision-map extraction.

Reads the on-screen background tilemap (``wTileMap`` at 0xC3A0, a 20x18 grid
of 8x8 hardware tile ids) and the current tileset id (``wCurMapTileset`` at
0xD367), then classifies each of the 10x9 walkable *blocks* as walkable or
blocked using the authoritative per-tileset collision lists from the pokered
disassembly (``data/tilesets/collision_tile_ids.asm``).

A Gen 1 overworld "block" is 16x16 px = a 2x2 group of 8x8 tiles. The screen
shows 10 blocks across and 9 down. The player is locked to block (col 4,
row 4) — grid cell "E5". Walkability of a block is decided by its top-left
8x8 tile id, which is what the engine itself checks.

Output grid orientation: ``grid[row][col]`` with row 0 at the top, col 0 at
the left; ``True`` means walkable.
"""

from __future__ import annotations

from typing import Dict, List

ADDR_TILEMAP = 0xC3A0       # wTileMap, 20x18 bytes
ADDR_TILESET = 0xD367       # wCurMapTileset
TILEMAP_W, TILEMAP_H = 20, 18

BLOCK_COLS = 10             # on-screen walkable blocks across
BLOCK_ROWS = 9             # on-screen walkable blocks down
PLAYER_COL = 4             # the block the player is locked to (cell E5)
PLAYER_ROW = 4

# Per-tileset walkable tile-id sets, transcribed from pokered
# data/tilesets/collision_tile_ids.asm. Key = wCurMapTileset value.
TILESET_WALKABLE: Dict[int, frozenset] = {
    0: frozenset({0x00, 0x10, 0x1B, 0x20, 0x21, 0x23, 0x2C, 0x2D, 0x2E, 0x30, 0x31, 0x33, 0x39, 0x3C, 0x3E, 0x52, 0x54, 0x58, 0x5B}),  # Overworld
    1: frozenset({0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x14, 0x1A, 0x1C}),  # RedsHouse1
    2: frozenset({0x11, 0x1A, 0x1C, 0x3C, 0x5E}),  # Mart
    3: frozenset({0x1E, 0x20, 0x2E, 0x30, 0x34, 0x37, 0x39, 0x3A, 0x40, 0x51, 0x52, 0x5A, 0x5C, 0x5E, 0x5F}),  # Forest
    4: frozenset({0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x14, 0x1A, 0x1C}),  # RedsHouse2
    5: frozenset({0x03, 0x11, 0x16, 0x19, 0x2B, 0x3C, 0x3D, 0x3F, 0x4A, 0x4C, 0x4D}),  # Dojo
    6: frozenset({0x11, 0x1A, 0x1C, 0x3C, 0x5E}),  # Pokecenter
    7: frozenset({0x03, 0x11, 0x16, 0x19, 0x2B, 0x3C, 0x3D, 0x3F, 0x4A, 0x4C, 0x4D}),  # Gym
    8: frozenset({0x01, 0x12, 0x14, 0x28, 0x32, 0x37, 0x44, 0x54, 0x5C}),  # House
    9: frozenset({0x01, 0x12, 0x14, 0x1A, 0x1C, 0x37, 0x38, 0x3B, 0x3C, 0x5E}),  # ForestGate
    10: frozenset({0x01, 0x12, 0x14, 0x1A, 0x1C, 0x37, 0x38, 0x3B, 0x3C, 0x5E}),  # Museum
    11: frozenset({0x0B, 0x0C, 0x13, 0x15, 0x18}),  # Underground
    12: frozenset({0x01, 0x12, 0x14, 0x1A, 0x1C, 0x37, 0x38, 0x3B, 0x3C, 0x5E}),  # Gate
    13: frozenset({0x04, 0x0D, 0x17, 0x1D, 0x1E, 0x23, 0x34, 0x37, 0x39, 0x4A}),  # Ship
    14: frozenset({0x0A, 0x1A, 0x32, 0x3B}),  # ShipPort
    15: frozenset({0x01, 0x10, 0x13, 0x1B, 0x22, 0x42, 0x52}),  # Cemetery
    16: frozenset({0x04, 0x0F, 0x15, 0x1F, 0x3B, 0x45, 0x47, 0x55, 0x56}),  # Interior
    17: frozenset({0x05, 0x15, 0x18, 0x1A, 0x20, 0x21, 0x22, 0x2A, 0x2D, 0x30}),  # Cavern
    18: frozenset({0x14, 0x17, 0x1A, 0x1C, 0x20, 0x38, 0x45}),  # Lobby
    19: frozenset({0x01, 0x05, 0x11, 0x12, 0x14, 0x1A, 0x1C, 0x2C, 0x53}),  # Mansion
    20: frozenset({0x0C, 0x16, 0x1E, 0x26, 0x34, 0x37}),  # Lab
    21: frozenset({0x0F, 0x1A, 0x1F, 0x26, 0x28, 0x29, 0x2C, 0x2D, 0x2E, 0x2F, 0x41}),  # Club
    22: frozenset({0x01, 0x10, 0x11, 0x13, 0x1B, 0x20, 0x21, 0x22, 0x30, 0x31, 0x32, 0x42, 0x43, 0x48, 0x52, 0x55, 0x58, 0x5E}),  # Facility
    23: frozenset({0x1B, 0x23, 0x2C, 0x2D, 0x3B, 0x45}),  # Plateau
}

COL_LABELS = "ABCDEFGHIJ"


def cell_label(col: int, row: int) -> str:
    return f"{COL_LABELS[col]}{row + 1}"


def read_block_tile_ids(emu) -> List[List[int]]:
    """Return the 9x10 grid of representative tile ids per block.

    The player's standing tile in ``wTileMap`` is screen tile (col 8, row 9),
    so the 16px block grid is sampled with a +1 row offset: block (bc, br)
    maps to tilemap tile (bc*2, br*2 + 1). This puts the player at block
    (col 4, row 4) = cell E5 and makes the "tile above" / collision checks
    line up with the engine's own movement rules.
    """
    tm = emu.read_range(ADDR_TILEMAP, TILEMAP_W * TILEMAP_H)
    grid: List[List[int]] = []
    for br in range(BLOCK_ROWS):
        row: List[int] = []
        for bc in range(BLOCK_COLS):
            tcol, trow = bc * 2, br * 2 + 1
            row.append(tm[trow * TILEMAP_W + tcol])
        grid.append(row)
    return grid


def build_collision_grid(emu) -> Dict:
    """Build a walkability grid for the current on-screen blocks.

    Returns a dict with:
        tileset: int
        walkable: 9x10 list of bool (True = can step there)
        tile_ids: 9x10 list of int (raw representative tile ids)
        player_cell: "E5"
    The player's own cell is always reported walkable.
    """
    tileset = emu.read_u8(ADDR_TILESET)
    walk_set = TILESET_WALKABLE.get(tileset, frozenset())
    tile_ids = read_block_tile_ids(emu)

    walkable: List[List[bool]] = []
    for br in range(BLOCK_ROWS):
        row: List[bool] = []
        for bc in range(BLOCK_COLS):
            tid = tile_ids[br][bc]
            row.append(tid in walk_set)
        walkable.append(row)
    # The player block is always passable (you're standing on it).
    walkable[PLAYER_ROW][PLAYER_COL] = True

    return {
        "tileset": tileset,
        "walkable": walkable,
        "tile_ids": tile_ids,
        "player_cell": cell_label(PLAYER_COL, PLAYER_ROW),
    }


def render_ascii_map(collision: Dict, legend: bool = True) -> str:
    """Render the collision grid as a labelled ASCII map.

    Legend:
        @ = player (E5)   . = walkable   # = blocked
    Column headers A..J, row numbers 1..9.
    """
    walkable = collision["walkable"]
    lines: List[str] = []
    header = "   " + " ".join(COL_LABELS)
    lines.append(header)
    for r in range(BLOCK_ROWS):
        cells = []
        for c in range(BLOCK_COLS):
            if r == PLAYER_ROW and c == PLAYER_COL:
                cells.append("@")
            else:
                cells.append("." if walkable[r][c] else "#")
        lines.append(f"{r + 1:>2} " + " ".join(cells))
    if legend:
        lines.append("")
        lines.append("@ you (E5)   . walkable   # blocked")
        lines.append("up=row-1 down=row+1 left=col-1 right=col+1")
    return "\n".join(lines)
