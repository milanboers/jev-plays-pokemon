"""Shared fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ROM = ROOT / "roms" / "Pokemon Red.gb"


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires the Pokemon Red ROM and PyBoy")


def pytest_collection_modifyitems(config, items):
    # Only run integration tests when the ROM is present.
    if ROM.exists():
        return
    skip = pytest.mark.skip(reason=f"ROM not found at {ROM}")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def rom_path() -> str:
    return str(ROM)


@pytest.fixture
def emu(rom_path):
    from jev_plays_pokemon.vendor.emulator import PyBoyEmulator

    emulator = PyBoyEmulator(window="null")
    emulator.load(rom_path)
    yield emulator
    emulator.close()


@pytest.fixture
def booted(emu):
    """Emulator with the intro already played (Red is in his bedroom)."""
    from jev_plays_pokemon import intro

    pb = emu._pyboy
    pb.set_emulation_speed(0)
    intro.start_game(pb)
    pb.tick(60)
    return emu
