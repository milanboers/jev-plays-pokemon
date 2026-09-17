"""Cooperative stop flag shared across modules.

A SIGINT during a blocking Jev API call is swallowed by the HTTP stack, so a
plain ``KeyboardInterrupt`` isn't reliable. The main thread instead checks this
flag (set by the signal handler) while it waits in interruptible Python code.
"""

_interrupted = False


def request_stop() -> None:
    global _interrupted
    _interrupted = True


def stop_requested() -> bool:
    return _interrupted
