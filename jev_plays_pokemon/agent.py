"""The Jev decision layer: typed questions in, one button press out.

Uses one atomic Noul per candidate action (asked in parallel in a single
call) so Jev gives a calibrated yes/no gut check for each. Code then picks
the strongest answer with a margin requirement and sanity-checks it against
the deterministic walkability map.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, RetryPolicy, TypeSafeClient

load_dotenv()

# (press_frames, settle_frames) per action, tuned for Gen 1.
ACTION_FRAMES: dict[str, tuple[int, int]] = {
    "press_a": (8, 20),
    "press_b": (8, 20),
    "press_start": (8, 20),
    "walk_up": (16, 8),
    "walk_down": (16, 8),
    "walk_left": (16, 8),
    "walk_right": (16, 8),
    "wait": (0, 30),
}

ACTION_NAMES = {
    "press_a": "press A (confirm, talk to the NPC/object you face, advance text, select highlighted item)",
    "press_b": "press B (cancel, back out of a menu, run from a wild battle)",
    "press_start": "press START (open/close the main menu)",
    "walk_up": "walk UP one tile (or move the menu cursor up)",
    "walk_down": "walk DOWN one tile (or move the menu cursor down)",
    "walk_left": "walk LEFT one tile (or move the menu cursor left)",
    "walk_right": "walk RIGHT one tile (or move the menu cursor right)",
    "wait": "press nothing for about half a second (let animations/transitions finish)",
}

DIRECTION_ACTIONS = {"walk_up", "walk_down", "walk_left", "walk_right"}
_REVERSE = {
    "walk_up": "walk_down",
    "walk_down": "walk_up",
    "walk_left": "walk_right",
    "walk_right": "walk_left",
}


def build_questions() -> dict[str, Any]:
    """One short atomic Noul per action; the action semantics live in the
    shared state ``instructions`` so the questions stay token-cheap."""
    return {
        action: Noul(instructions=f"Is the best single action right now to {ACTION_NAMES[action]}?")
        for action in ACTION_FRAMES
    } | {
        "menu_open": Noul(
            instructions=(
                "Is a selectable menu open on screen (start menu, POKEMON/BAG/ITEM list, "
                "battle FIGHT/PKMN/BAG/RUN, or a yes/no question)? A plain text dialog is NOT a menu."
            )
        ),
    }


@dataclass
class Decision:
    action: str
    confidence: float  # yes-probability of the winning action
    margin: float  # winning yes-probability minus the runner-up
    nouls: dict[str, float]
    menu_open: float | None

    def summary(self) -> str:
        ranked = sorted(self.nouls.items(), key=lambda kv: -kv[1])
        return ", ".join(f"{k}={v:.2f}" for k, v in ranked[:5])


@dataclass
class GoalDecision:
    goal: str
    confidence: float
    nouls: dict[str, float]
    menu_open: float | None

    def summary(self) -> str:
        ranked = sorted(self.nouls.items(), key=lambda kv: -kv[1])
        return ", ".join(f"{k}={v:.2f}" for k, v in ranked[:5])


def _sample_goal(
    probabilities: dict[str, float],
    temperature: float = 2.0,
    floor: float = 0.05,
) -> tuple[str, float]:
    """Pick a goal label from Jev's calibrated categorical distribution.

    ``probabilities`` is the full {goal -> prob} distribution the model emits.
    We do NOT sample it verbatim - a confidently-wrong pick (e.g. "talk to
    people" at p=0.99) would then be repeated almost every turn. Instead we
    apply a monotonic transform that always reserves real mass for the
    alternatives:

    * ``floor``: every option keeps at least this much probability mass, so
      even a p=1.0 goal is never picked 100% of the time.
    * ``temperature`` (>1): softmax flattening (log(p)/T) pulls the peak down
      and lifts the tail - with the defaults, p=0.99 lands around ~80%.

    Returns the chosen label and its (temperature-adjusted) probability.
    """
    import random

    labels = list(probabilities)
    if not labels:
        return "wait", 0.0
    probs = [max(0.0, probabilities[g]) for g in labels]
    total = sum(probs)
    if total <= 0:
        return "wait", 0.0
    probs = [p / total for p in probs]
    # Floor: guarantee every option keeps some mass (p=1 is never 100%).
    floor = max(0.0, floor)
    probs = [p + floor for p in probs]
    total = sum(probs)
    probs = [p / total for p in probs]
    # Softmax flattening: logits = log(p)/T. T>1 pulls the peak down and
    # lifts the tail, keeping a real chance of doing something else.
    if temperature != 1.0 and temperature > 0:
        import math

        logits = [math.log(p + 1e-12) / temperature for p in probs]
        m = max(logits)
        exp = [math.exp(x - m) for x in logits]
        exp_total = sum(exp)
        probs = [e / exp_total for e in exp]
    label = random.choices(labels, weights=probs, k=1)[0]
    return label, probs[labels.index(label)]


def retry_policy() -> RetryPolicy:
    """Robust retry-with-backoff for the (occasionally flaky) TypeSafe API.

    Overridable via env: ``TYPESAFE_MAX_RETRIES``, ``TYPESAFE_BACKOFF_INITIAL``,
    ``TYPESAFE_BACKOFF_MAX``, ``TYPESAFE_TIMEOUT``.
    """

    def _int(name: str, default: int) -> int:
        raw = os.getenv(name)
        return int(raw) if raw else default

    def _float(name: str, default: float) -> float:
        raw = os.getenv(name)
        return float(raw) if raw else default

    return RetryPolicy(
        max_retries=_int("TYPESAFE_MAX_RETRIES", 4),
        backoff_initial=_float("TYPESAFE_BACKOFF_INITIAL", 1.0),
        backoff_max=_float("TYPESAFE_BACKOFF_MAX", 8.0),
        backoff_jitter=0.2,
        timeout=_float("TYPESAFE_TIMEOUT", 45.0),
    )


class JevAgent:
    def __init__(self, model: str | None = None) -> None:
        if not os.getenv("TYPESAFE_API_KEY"):
            raise RuntimeError("TYPESAFE_API_KEY is not set. Create a .env file with your key.")
        kwargs = {"model": model} if model else {}
        self.client = TypeSafeClient(retry=retry_policy(), **kwargs)
        self.action_questions = build_questions()

    @staticmethod
    def _interruptible(fn, *args, **kwargs):
        """Run a blocking client call on a worker thread.

        The HTTP stack swallows SIGINT while it is running, so the main thread
        waits here in interruptible Python and checks the stop flag instead.
        On Ctrl+C the worker is abandoned (daemon) and the call is aborted.
        """
        import threading
        import time

        from jev_plays_pokemon.stop import stop_requested

        box: dict = {}

        def worker() -> None:
            box["value"] = fn(*args, **kwargs)
            box["done"] = True

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while not box.get("done"):
            if stop_requested():
                raise KeyboardInterrupt
            time.sleep(0.05)
        return box["value"]

    def decide_goal(self, state: dict[str, Any], goals: list[dict[str, str]]) -> GoalDecision:
        """Ask Jev which high-level goal to pursue (overworld decision)."""
        criteria = {g["id"]: g["desc"] for g in goals}
        questions = {
            "next_goal": Choice(
                instructions={
                    "question": "Which goal should RED pursue next?",
                    "focus": "Pick the ONE goal that best advances the objective given the screen text, "
                    "nearby objects, walkability and room map. Code will execute the walking.",
                },
                criteria=criteria,
            )
        }
        questions.update(self.action_questions)
        response = self._interruptible(self.client.system_one, state=state, questions=questions)
        choice = response.choices["next_goal"]
        nouls = {key: answer.noul for key, answer in response.nouls.items()}
        # Sample Jev's goal distribution after flattening it: the model's raw
        # probabilities are well-calibrated but confidently-wrong (e.g. "talk
        # to people" at 0.99) would otherwise repeat every turn. The transform
        # pulls the peak down and keeps every option alive, so there is always
        # a real chance of doing something else. Overridable via env:
        #   TYPESAFE_TEMPERATURE (default 2.0; >1 flattens, <1 sharpens)
        #   TYPESAFE_GOAL_FLOOR    (default 0.05; min mass kept per option)
        temperature = float(os.getenv("TYPESAFE_TEMPERATURE", "2.0"))
        floor = float(os.getenv("TYPESAFE_GOAL_FLOOR", "0.05"))
        goal, conf = _sample_goal(choice.probabilities, temperature, floor)
        return GoalDecision(
            goal=goal,
            confidence=conf,
            nouls=nouls,
            menu_open=nouls.get("menu_open"),
        )

    def decide_action(self, state: dict[str, Any]) -> Decision:
        """Ask Jev which single button to press (menu / battle micro-decision)."""
        response = self._interruptible(self.client.system_one, state=state, questions=self.action_questions)
        nouls = {key: answer.noul for key, answer in response.nouls.items()}
        ranked = sorted(nouls.items(), key=lambda kv: -kv[1])
        best_action, best_val = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        return Decision(
            action=best_action,
            confidence=best_val,
            margin=best_val - runner_up,
            nouls=nouls,
            menu_open=nouls.get("menu_open"),
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def pick_action(
    decision: Decision,
    dialog_active: bool,
    walkable: dict[str, bool],
    last_action: str | None = None,
    explore_hint: str | None = None,
    probe_exits: bool = False,
) -> str:
    """Turn the model's gut-checks into a concrete, safe button press.

    Only acts when the winner is clearly ahead (margin); otherwise falls
    back to advancing a dialog, walking toward unexplored space, or waiting.
    Never walks into a wall unless probing for an exit in a fully explored
    room, and avoids immediately reversing the previous walk.
    """
    action = decision.action
    if action not in ACTION_FRAMES:
        action = "wait"

    # Menu override: while a choice menu is open, confirm the highlighted
    # option unless the model is very confident about something else. This is
    # what reliably selects the starter (and most story choices).
    menu_open = (decision.menu_open or 0.0) >= 0.5
    if menu_open and decision.confidence < 0.7:
        action = "press_a"

    if decision.margin >= 0.08 and decision.confidence >= 0.45:
        pass  # trust the model
    elif dialog_active or menu_open:
        # A text box or an open menu: confirming the highlighted option is
        # the safe default (this is how the starter choice is selected).
        action = "press_a"
    elif explore_hint and walkable.get(explore_hint, False):
        action = f"walk_{explore_hint}"
    else:
        action = "wait"

    # In a menu, waiting does nothing - confirm the highlighted option.
    if menu_open and action == "wait":
        action = "press_a"

    # Deterministic wall check for overworld movement. When every direction
    # is "blocked" the player is likely facing a door/stairs the map can't
    # see, so let the model's choice through. In a fully explored room we
    # also allow walking into a wall to probe for the exit.
    if action in DIRECTION_ACTIONS:
        direction = action.removeprefix("walk_")
        blocked = not walkable.get(direction, True)
        all_blocked = not any(walkable.values())
        if blocked and not all_blocked and not probe_exits:
            action = "press_a" if dialog_active else "wait"

    # Anti-pacing: never immediately reverse the previous walk when there is
    # another open direction to try.
    if action in DIRECTION_ACTIONS and _REVERSE[action] == last_action:
        alternatives = [
            f"walk_{d}" for d, ok in walkable.items() if ok and f"walk_{d}" not in (action, _REVERSE[action])
        ]
        if alternatives:
            best = max(alternatives, key=lambda a: decision.nouls.get(a, 0.0))
            if decision.nouls.get(best, 0.0) >= 0.3:
                action = best

    return action
