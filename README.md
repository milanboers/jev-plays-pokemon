# jev-plays-pokemon — TypeSafe Jev plays Pokémon Red

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python: 3.14](https://img.shields.io/badge/python-3.14-blue)](pyproject.toml)
[![Status: experimental](https://img.shields.io/badge/status-experimental-yellow)](README.md#known-limitations)

A small autonomous Pokémon Red agent. It uses [TypeSafe's System One model, Jev](https://docs.typesafe.ai/introduction): Jev reads the game state as **text**, answers typed questions each turn, and deterministic code turns those answers into button presses on a [PyBoy](https://github.com/Baekalfen/PyBoy) Game Boy emulator.

**Jev has no vision.** It can't see the screen or write text. Instead, every turn we build a compact text snapshot of the game: the on-screen dialog, a walkability map read from RAM, your party/bag/battle state, the room layout learned so far, and a short-term objective. Jev answers *parallel yes/no questions* ("is pressing A the best action right now?", "is walking UP best?") and the code picks the strongest, safest button press.

**Jev has no conversation history.** Unlike an LLM in a chatbot, Jev doesn't get a growing transcript of past turns — each turn is a fresh, self-contained request. Whatever Jev needs to know about the past has to be *in* that request, so the harness itself does the remembering: it keeps a short-term memory (recent dialog pages, recent high-level actions, the explored fraction of the current room) and re-injects it into every snapshot. Jev decides only from what it sees in the current turn; continuity comes from the harness, not the model.

## What it can do

- Plays through the intro automatically (title → NEW GAME → names RED → names the rival), fast-forwarded.
- Reads Pokémon Red's RAM directly: location, facing, money, badges, party (species/level/HP/status/types/moves), bag, battle state, and story flags.
- Decodes the on-screen dialog box / menu text from the emulator's tilemaps (e.g. `FIGHT PKMN BAG RUN`, `POKéMON ITEM RED SAVE`).
- Reads the game's **object memory** (WRAM `0xC100`/`0xC200`) to list nearby people/objects: their **identity** (Mom, Oak, Girl, Poke Ball…), **screen cell**, and **direction/distance from you** — so it can walk up to an NPC and talk instead of wandering blind.
- Builds a walkability map from the game's collision data (`.` = walkable, `#` = blocked) and a growing room layout.
- Picks a **high-level goal each turn** (`talk to Mom`, `explore`, `leave the room`, `advance text`); code executes it with A* pathfinding. In menus/battles Jev picks individual buttons, with deterministic safeguards:
  - flattened goal sampling (a confident pick fires ~60–70% of the time, so it never hard-locks on one goal),
  - confidence/margin gating on menu/battle buttons (falls back to advancing dialog or waiting when unsure),
  - live collision grid only (a bump or NPC never becomes a permanent wall),
  - anti-pacing (never immediately reverses the last step),
  - stuck detection (press B → wait → re-route),
  - exit probing when a room is fully explored (exits/stairs are "walls" the game lets you walk through).
- **Today it reliably plays the opening sequence end-to-end**: walks out of the house, talks to people, follows Oak to his lab, picks a starter (and declines the nickname), and steps back out into Pallet Town.
- Shows a live SDL2 window so you can watch it play.

## Project layout

```
jev_plays_pokemon/
├── intro.py            # title screen -> name selection -> bedroom
├── screen_text.py      # decode dialog/menu text from PyBoy tilemaps
├── objects.py          # overworld object memory (NPCs/items: identity + position)
├── navigation.py       # A* pathfinding over the collision/room grid
├── play.py             # executes the chosen goal (walk/talk/probe/advance)
├── state.py            # text-state assembly, room map, goals, objectives, memory
├── agent.py            # the Jev questions (goal Choice + action Nouls)
├── main.py             # the play loop (CLI entry point)
└── vendor/             # MIT code from NousResearch/pokemon-agent (not on PyPI):
    ├── emulator.py     #   PyBoy wrapper (patched: configurable window)
    ├── collision.py    #   RAM walkability map + ASCII renderer
    ├── memory/red.py   #   Red/Blue RAM reader (species, items, moves, …)
    └── state/builder.py#   structured-state builder
tests/                  # unit + emulator integration tests (pytest)
roms/                   # put Pokemon Red.gb here (gitignored)
screenshots/            # one PNG per turn (gitignored)
logs/                   # one timestamped log file per run (gitignored)
```

## Setup

Requires Python ≥ 3.14 (see `.python-version`) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                     # installs all deps incl. dev (typesafe-sdk, pyboy, pillow, pytest, ruff)

# 1. ROM: your own legally-obtained copy of Pokemon Red (USA).
#    The expected SHA-1 is ea9bcae617fdf159b045185467ae58b2e4a48b9a.
unzip ~/Downloads/"Pokemon - Red Version (UE)[!].zip" "Pokemon Red.gb" -d roms/

# 2. API key
cp .env.example .env       # then add your TYPESAFE_API_KEY (console.typesafe.ai/settings/keys)
```

## Run

```bash
# Visible window (scale 3) — watch it play
uv run jev-plays-pokemon      # or: uv run python -m jev_plays_pokemon.main

# Options
uv run jev-plays-pokemon --rom path/to/Pokemon.gb    # custom ROM path
uv run jev-plays-pokemon --model jev-preview          # other Jev model
uv run jev-plays-pokemon --headless                   # no window (console logs only)
uv run jev-plays-pokemon --max-turns 200              # stop after N decisions
uv run jev-plays-pokemon --turbo                      # no realtime limit - the game flies (demo mode)
uv run jev-plays-pokemon --no-screenshots             # don't save PNGs
```

`Ctrl+C` stops cleanly within a fraction of a second — even mid-goal or mid-API-call — and saves nothing to the save file. `--turbo` removes the 60 fps realtime cap, so emulation runs as fast as the CPU allows; the Jev API call per decision (~0.5 s) becomes the pacing factor.

Every turn is logged to stdout as two compact lines: one for the input (the state Jev saw) and one for the output (Jev's decision + the action taken + whether it moved you):

```
turn 12 | Red's House 1F (6,5) facing up | open:dlr | Deliver Oak's Parcel... | text:'OAK: ...'
  -> Jev: walk_down (p=0.62, margin=0.30, menu=0.04) | walk_down | moved -> (6, 6) [walk_down=0.62, ...]
```

The same session is also written verbosely to `logs/run_<timestamp>.log`
(full room map, screen text, per-button frame counts, stuck alerts —
everything), so you can reconstruct a whole run later. Pass
`--verbose` to also see that detail on stdout. Screenshots land in
`screenshots/` (one PNG per turn).

## Logging

```bash
# console: one line per input + one line per output (above)
# log file: same plus full DEBUG detail
tail -f logs/run_*.log      # follow a run
```

## How it works

**Jev decides *what*, code does *how*.** Each loop iteration builds the text
state, works out which high-level goals are possible, and Jev picks one via a
`Choice`:

- `advance_text` — a dialog/text box is open (the only goal offered then)
- `talk_to_Mom / talk_to_*` — an NPC is on screen (with its cell + direction)
- `explore` — head toward unexplored space
- `reach_exit` — leave the room once it is fully explored or its exit is known
- `wait`

Code then **executes the goal mechanically**: BFS/A* pathfinding over the
**physically-verified room map** (tiles the agent has actually stood on) routes
it to NPCs, exits and unexplored frontier; a greedy fallback breaks through
unmapped ground when no planned path exists yet. Dialog text is advanced with
A, and room boundaries are probed for exits. Control returns to Jev when the
goal completes or the situation changes (new object, dialog, map transition).
In menus and battles Jev instead picks a single button via the 8 atomic Nouls.
Deterministic safeguards remain: wall memory, anti-pacing, stuck detection.

**State** (what Jev reads every turn): a short instruction block (how to read
the state, priority order, Gen 1 battle type chart, the long-term goal of
earning all 8 gym badges), the decoded screen text, the walkability map,
`walkable_directions`, the room map with `explored_fraction` (how much of the
room has been seen, so Jev knows when to stop exploring), **nearby objects**
(who is on the current map and where, e.g. `Mom: at C8 (2 tiles WEST, 3 tiles
SOUTH)`), the available **goals**, player/party/bag/battle/flags, and a
**short-term memory**: up to 8 deduplicated dialog pages people told you
(`recent_hints`, kept after the text box closes) and the last 12 high-level
actions you took with their outcomes (`recent_actions`, e.g. `talked to Mom ->
talked (A pressed)`). Both are ordered oldest → newest, so the model knows
which entry is "just now". The screen text itself is only ever the current frame —
the hints are how dialog survives across turns. There is no hardcoded
per-turn objective: Jev infers the next step from dialog hints, recent
actions, and the explored fraction.

**Exits are walls.** Front doors and stairs look like solid walls (`#`) on the
collision map — the game transitions when you step onto them. The agent knows
this (it's in the instructions) and walks into them. Exits are **never
hardcoded**: every frame the live RAM warp table (`wWarpEntries`) is read to
derive the current room's exits, exposed as `KNOWN EXITS` (e.g. `Pallet Town at
tile (3,7), 2 tiles WEST, 1 tile SOUTH, step down`) and offered as per-exit
goals (`exit_to_0`). Two warp mechanics are handled: **step-triggered** (stairs,
holes — transition the instant you land on the tile) and **boundary-triggered**
(house front doors — you must physically step INTO the boundary wall while
standing on the carpet mat).

## Known limitations

- **Text-only, not vision.** Jev cannot see the screen, so everything is
  reconstructed from RAM + tilemap reads. Anything the game expresses purely
  visually (animations, sparkles, sprite poses, the exact menu cursor row) is
  only approximated. This is the fundamental constraint of the approach.
- **Dialog text decode is lossy.** Text types one character at a time and the
  box scrolls, so mid-typing frames and scrolled lines can be captured
  partially or garbled (e.g. `are POKé . They` instead of the full sentence).
  The engine's ▼ "awaiting input" arrow gates hint capture, which usually but
  not always lands on a clean page. The model still gets the current `screen_text`
  every turn even when the *remembered* hint is imperfect.
- **Talk goals are never hard-removed.** Talking to an NPC once does not hide
  the option — the agent decides whether to re-talk based on `recent_actions`
  (what it just did), the anti-loop guidance, and the goal distribution. A
  short expiring cooldown only nudges away from instantly repeating a failed
  attempt.
- **Non-deterministic goal choice (by design).** Goal probabilities are sampled
  after a mild flattening (temperature 2.0, floor 0.1). So a confident pick
  fires only ~60–70% of the time, and the agent always keeps some chance of
  doing something else. This prevents hard loops but means behavior varies run
  to run, and sometimes the agent picks a clearly sub-optimal goal.
- **Items appear as "talk to" targets.** The object memory lists items (Poke
  Ball, Pokedex, …) alongside NPCs and they get a `talk_to_*` goal. That is
  deliberate for Oak's Lab (you talk to the Poke Ball to pick a starter), but
  pressing A on other items may do nothing.
- **Battles are the current weak spot.** The opening sequence works end-to-end
  (house → Pallet → Oak's Lab → starter → back outside), but the first rival
  battle (and wild battles generally) are not reliable yet: the battle menu
  micro-decisions, move selection, and run/fight flow are still being tuned.
  Expect it to stall or flounder once a fight starts.
- **Tutorial-area tuned.** Navigation, exits, the intro, and the starter pick
  are tuned and tested for the opening sequence: Red's house → Pallet Town →
  Oak's Lab → starter choice. Later towns, gym leaders, and most of the story
  are mechanically supported (badges objective, known-exits travel) but far
  less battle-tested.
- **ROM-specific addresses.** All RAM offsets come from the USA `pokered`
  decompilation. A different region/version (e.g. Blue, or a European ROM) will
  read garbage at these addresses. The expected ROM SHA-1 is pinned in Setup.
- **No save/reload.** Each run starts a fresh NEW GAME through the intro; there
  is no in-run save-state persistence, so a long session is one continuous play.
- **`--turbo` races.** At unlimited speed the game flies; fast-moving frames
  (dialog typing, transitions) can be mis-sampled, and input settle timing can
  lag. It's a demo/exploration mode, not the reliable mode.

## Tests

```bash
uv run pytest tests/          # unit + integration (needs the ROM; skips otherwise)
uv run ruff check jev_plays_pokemon tests
```

Integration tests boot the real emulator headless: intro navigation lands in the bedroom, dialog/menu text decodes, the state snapshot is well-formed, and holding a direction actually moves the player.

## Cost & latency

Jev is fast (tens–hundreds of ms per decision) and cheap: input costs
$0.042/MTok and output is effectively free. The SDK retries transient
5xx/downtime responses with exponential backoff (default: 4 retries, 1→8 s
backoff, 45 s budget; tune via `TYPESAFE_MAX_RETRIES`,
`TYPESAFE_BACKOFF_INITIAL/MAX`, `TYPESAFE_TIMEOUT`), so a flaky API stalls a
moment instead of killing the run. Ctrl+C still interrupts mid-retry.

**Approximate tokens per decision** (rough; the shared prompt grows over time):

| What | Approx. tokens |
|---|---|
| Shared instructions (strategy, battle chart, action semantics) | ~850 |
| Questions (8 atomic Nouls + `menu_open`) | ~350 |
| Per-turn state (text, maps, objects, party, flags, goals) | ~400–900 |
| **Total input** | **≈ 1,600–2,100 tokens** |

**Cost estimates** (at $0.042/MTok):

| Rate | Decisions/hour | Cost/hour | Cost/10k decisions |
|---|---|---|---|
| ~1 decision/s | 3,600 | ≈ $0.27 | ≈ $0.76 |
| ~1 decision per 4–6 s (goal executes multiple steps) | 600–900 | ≈ $0.05 | ≈ $0.76 |

A "turn" executes a whole goal (walk a path, talk, clear a dialog), so it
covers several button presses — the per-progress cost is lower than the
per-call numbers suggest. A full evening (say 10k decisions) stays well under
a dollar.

To see the real numbers for your session, run with the SDK's usage log:

```bash
TYPESAFE_LOG_LEVEL=info uv run jev-plays-pokemon --max-turns 20
```

## Tuning knobs

- `state.py:compute_goals` — which goals Jev can pick from each turn.
- `play.py:execute_goal` / `navigation.py:astar` — goal execution + pathfinding.
- `agent.py:pick_action` — menu/battle micro-decision thresholds.
- `agent.py:_sample_goal` / `TYPESAFE_TEMPERATURE`, `TYPESAFE_GOAL_FLOOR` — how much the model's goal distribution is flattened before sampling.
- `state.py:INSTRUCTIONS` — the prompt Jev reads every turn (strategy, battle chart, action semantics, long-term goal).
- `state.py:StateBuilder` — the in-code memory: `recent_hints`, `recent_actions`, explored-fraction per room. No hardcoded per-turn objective; Jev infers the next step from dialog, actions, and exploration progress.
- `agent.py:ACTION_FRAMES` — press-hold and settle frame counts per button.

## Credits

- [TypeSafe AI / Jev](https://typesafe.ai) — System One model + Python SDK.
- [PyBoy](https://github.com/Baekalfen/PyBoy) — Game Boy emulator (MIT).
- [NousResearch/pokemon-agent](https://github.com/NousResearch/pokemon-agent) — vendored MIT code for the Red/Blue RAM reader, collision map, and state builder.
- [pret/pokered](https://github.com/pret/pokered) — the decompilation all RAM addresses come from.
- [davidhershey/ClaudePlaysPokemonStarter](https://github.com/davidhershey/ClaudePlaysPokemonStarter) — the screen-relative navigation idea (allow a wall tile as the path target; re-verify after every step). No code vendored; the approach is reimplemented from scratch in `screen_grid.py`.