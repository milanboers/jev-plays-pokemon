"""Jev plays Pokemon Red — main loop."""

from __future__ import annotations

import argparse
import logging
import signal
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
ROM_DEFAULT = ROOT / "roms" / "Pokemon Red.gb"
SHOT_DIR = ROOT / "screenshots"
LOG_DIR = ROOT / "logs"

from jev_plays_pokemon import intro  # noqa: E402
from jev_plays_pokemon.agent import Decision, JevAgent, pick_action  # noqa: E402
from jev_plays_pokemon.play import execute_goal  # noqa: E402
from jev_plays_pokemon.state import StateBuilder, render_state_line, render_text_state  # noqa: E402
from jev_plays_pokemon.vendor.emulator import PyBoyEmulator  # noqa: E402

log = logging.getLogger("jev_plays_pokemon")


def setup_logging(verbose: bool) -> Path:
    """Console + file handlers; returns the log file path.

    Console shows INFO (every turn: state, Jev's answer, action, outcome)
    plus DEBUG only when ``verbose``. The file always keeps DEBUG.
    """
    LOG_DIR.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOG_DIR / f"run_{run_id}.log"

    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    log.addHandler(console)

    file_h = logging.FileHandler(log_path, encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(formatter)
    log.addHandler(file_h)

    return log_path


def _make_emulator(headless: bool) -> PyBoyEmulator:
    if headless:
        return PyBoyEmulator(window="null")
    try:
        emu = PyBoyEmulator(window="SDL2", scale=3)
        emu.load(str(ROM_DEFAULT))
        return emu
    except Exception as exc:  # e.g. no display server
        log.warning("SDL2 window failed (%s); falling back to headless.", exc)
        return PyBoyEmulator(window="null")


def _save_screenshot(pb, turn: int) -> None:
    SHOT_DIR.mkdir(exist_ok=True)
    img = pb.screen.image
    if img is not None:
        img.save(SHOT_DIR / f"turn_{turn:05d}.png")


def _unstuck(emu, builder, log) -> bool:
    """Aggressive, purely-physical recovery from a stall.

    The game ignores ALL movement input while a text window is open, so a
    frozen player is usually a missed dialog/menu, not a real wall. We
    interleave text-clearing (A/B/START) with movement attempts and stop the
    moment the player actually moves. Returns True on any movement.
    """
    from jev_plays_pokemon.play import player_map

    def moved_from(start: tuple[int, int]) -> bool:
        return player_map(builder) != start

    def wiggle() -> bool:
        start = player_map(builder)
        for direction in ("up", "down", "left", "right"):
            _execute_single(emu, builder, f"walk_{direction}")
            if moved_from(start):
                return True
        return False

    # Interleave clearing presses with movement; a text box can be several
    # pages, so alternate A/B several times.
    for _ in range(6):
        emu.press("a", 8)
        emu.tick(20)
        if wiggle():
            return True
        emu.press("b", 8)
        emu.tick(25)
        if wiggle():
            return True

    # Try closing the START menu explicitly, then retry.
    emu.press("start", 8)
    emu.tick(30)
    emu.press("b", 8)
    emu.tick(30)
    emu.press("b", 8)
    emu.tick(30)
    if wiggle():
        return True

    # Last resort: hold each direction longer (catches slow animations).
    start = player_map(builder)
    for direction in ("up", "down", "left", "right"):
        emu.press(direction, 30)
        emu.tick(30)
        emu.button_release(direction)
        if moved_from(start):
            return True
    return False


def _changed_signature(state: dict, pb) -> tuple:
    """Movement/stuck signature.

    Uses the on-screen walkable grid (which changes as the screen scrolls) and
    the map id rather than the absolute coordinates, which are unreliable in
    this game (they can freeze while the player is actually moving).
    """
    from jev_plays_pokemon import screen_grid

    loc = state["location"]
    try:
        grid = screen_grid.walkable_grid(pb)
        grid_sig = bytes(grid)
    except Exception:
        grid_sig = b""
    return (loc["map_id"], grid_sig, state["screen_text"])


def run(
    rom: str,
    model: str | None,
    headless: bool,
    max_turns: int | None,
    save_shots: bool,
    verbose: bool = False,
    turbo: bool = False,
) -> None:
    log_path = setup_logging(verbose)
    log.info("Log file: %s", log_path)

    # Ctrl+C must stop the loop even mid-goal: set a flag the executor checks.
    from jev_plays_pokemon import play as play_mod
    from jev_plays_pokemon.stop import request_stop, stop_requested

    def _on_sigint(signum, frame):
        # Signal handlers must be lock-free (no logging/I/O): just set a flag.
        request_stop()

    signal.signal(signal.SIGINT, _on_sigint)
    log.debug("SIGINT handler installed: %s", signal.getsignal(signal.SIGINT))

    emu = _make_emulator(headless)
    emu.load(rom)
    pb = emu._pyboy

    log.info("ROM: %s", emu.rom_path)
    log.info("Cartridge: %s", pb.cartridge_title)
    log.info("Running the intro (fast-forward)...")
    pb.set_emulation_speed(0)
    intro.start_game(pb)
    # Normal runs are capped to real time; --turbo removes the limit so the
    # game flies (only the per-decision Jev API call remains).
    pb.set_emulation_speed(0 if turbo else 1)
    log.info("Intro done - RED is in his bedroom. Jev is taking over (%s).", "turbo" if turbo else "realtime")

    builder = StateBuilder(emu)
    agent = JevAgent(model=model)
    last_signature = None
    stuck = 0
    hard_stuck = 0
    menu_lock = 0
    turn = 0
    last_action: str | None = None

    try:
        with agent:
            while True:
                if play_mod.stop_requested():
                    log.info("Interrupted by user.")
                    break
                turn += 1
                state = builder.snapshot(turn)
                signature = _changed_signature(state, pb)

                if last_signature is not None:
                    if signature != last_signature:
                        stuck = 0
                    else:
                        stuck += 1
                last_signature = signature

                dialog_active = builder.dialog_active
                in_battle = bool((state.get("battle") or {}).get("in_battle"))
                if dialog_active:
                    stuck = 0  # position legitimately doesn't move in text/menus

                # If we have made no progress for a few turns, something is
                # blocking (a missed dialog/menu, or a boxed-in corner). Run
                # the physical stuck-engine: clear text boxes/menus and try to
                # walk in every direction until we actually move. Skip it while
                # a text box/menu is actively open - the position legitimately
                # doesn't change there, and mashing would cancel the menu.
                if stuck >= 3 and not in_battle and not dialog_active:
                    log.warning("no progress for %d turns - running stuck engine", stuck)
                    if _unstuck(emu, builder, log):
                        log.info("stuck engine: player moved / dialog cleared")
                        hard_stuck = 0
                    else:
                        hard_stuck += 1
                        log.warning(
                            "stuck engine still cannot move (attempt %d) - "
                            "the spot may be a text-window lock or a tight pocket",
                            hard_stuck,
                        )
                    stuck = 0
                    dialog_active = builder.dialog_active

                # INFO: one line per "input" (the state Jev saw).
                log.info("%s", render_state_line(state))
                # DEBUG: the full state detail for file / --verbose.
                for line in render_text_state(state).splitlines():
                    log.debug("    %s", line)

                # Jev picks a high-level goal (overworld) or a single button
                # (menus / battles need fine-grained control).
                decision = agent.decide_goal(state, builder.goals)
                menu = (decision.menu_open or 0.0) >= 0.5
                # Once a menu is seen, stay in fine-grained mode for a few
                # turns even if the flag flickers - otherwise the agent drops
                # out mid-selection (e.g. the starter choice) and wanders off.
                if menu:
                    menu_lock = 3
                elif menu_lock > 0:
                    menu_lock -= 1
                menu = menu or menu_lock > 0

                if menu or in_battle:
                    micro = _micro_action(decision, dialog_active, state, last_action)
                    nouls = sorted(decision.nouls.items(), key=lambda kv: -kv[1])
                    top = ", ".join(f"{k}={v:.2f}" for k, v in nouls[:4])
                    log.info("  -> Jev(micro): %s [%s]", micro, top)
                    _execute_single(emu, builder, micro)
                    last_action = micro
                else:
                    goal = decision.goal
                    desc = next((g["desc"] for g in builder.goals if g["id"] == goal), "")
                    log.info("  -> Jev(goal): %s (p=%.2f) — %s", goal, decision.confidence, desc)
                    reason = execute_goal(emu, builder, goal, state)
                    log.info("  -> goal %s done: %s", goal, reason)
                    builder.memory.record(goal, reason)
                    # Disincentivise re-running the same goal immediately: a
                    # futile explore ("stuck") or a failed exit should not be
                    # retried every single turn, forcing the model to vary.
                    cooldown = 4 if goal == "explore" else 3
                    if reason in (
                        "stuck in a dead pocket",
                        "no exit found",
                        "blocked on the way",
                        "target gone",
                    ):
                        cooldown = 5
                    builder.goal_cooldown[goal] = state["turn"] + cooldown
                    last_action = None
                    # A map transition leaves position/collision reads unstable;
                    # let the game settle before the next decision.
                    if reason in ("room changed", "exited the room") and not stop_requested():
                        emu.tick(90)
                        dialog_active = builder.dialog_active

                if save_shots:
                    _save_screenshot(pb, turn)
                if max_turns and turn >= max_turns:
                    log.info("Reached max_turns=%s.", max_turns)
                    break
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        emu.close()


def _micro_action(decision, dialog_active: bool, state: dict, last_action: str | None) -> str:
    """Single-button decision from the action nouls (menus/battles)."""
    ranked = sorted(decision.nouls.items(), key=lambda kv: -kv[1])
    best, best_val = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    micro = Decision(
        action=best,
        confidence=best_val,
        margin=best_val - runner_up,
        nouls=decision.nouls,
        menu_open=decision.menu_open,
    )
    return pick_action(
        micro,
        dialog_active,
        state["walkable_directions"],
        last_action,
        state.get("unexplored_hint"),
        state.get("probe_exits", False),
    )


def _execute_single(emu, builder, action: str) -> None:
    from jev_plays_pokemon.play import player_map, press

    before = player_map(builder)
    press(emu, action)
    after = player_map(builder)
    moved = before != after
    builder.record_move(action, moved)


def main() -> None:
    parser = argparse.ArgumentParser(description="Jev plays Pokemon Red")
    parser.add_argument("--rom", default=str(ROM_DEFAULT), help="Path to the Pokemon Red .gb ROM")
    parser.add_argument("--model", default=None, help="TypeSafe model id (default: jev-latest)")
    parser.add_argument("--headless", action="store_true", help="No SDL2 window (console logs only)")
    parser.add_argument("--max-turns", type=int, default=None, help="Stop after N turns")
    parser.add_argument("--no-screenshots", action="store_true", help="Do not save per-turn screenshots")
    parser.add_argument(
        "--verbose", action="store_true", help="Debug-level detail on stdout (frame counts, dialog state)"
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Remove the realtime emulation limit so the game plays as fast as possible",
    )
    args = parser.parse_args()

    run(
        args.rom, args.model, args.headless, args.max_turns, not args.no_screenshots, args.verbose, args.turbo
    )


if __name__ == "__main__":
    main()
