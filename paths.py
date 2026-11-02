"""
Where things live.

  APP_DIR   the program and its models (read-only is fine). Installed: the install folder. Development: this folder.
  DATA_DIR  everything the app writes: settings, glossary.txt, sessions, island.log.
            Installed: %LOCALAPPDATA%\\MeetingIsland   (always writable, survives updates and uninstalls of the program)
            Development: this folder, so nothing changes while developing.
Set MEETING_ISLAND_DATA to override DATA_DIR (used by the automated tests).
"""
import os
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))
APP_DIR = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent

if os.environ.get("MEETING_ISLAND_DATA"):
    DATA_DIR = Path(os.environ["MEETING_ISLAND_DATA"])
elif FROZEN:
    DATA_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "MeetingIsland"
else:
    DATA_DIR = APP_DIR

try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except Exception:                                  # a read-only profile should never stop the app from starting
    DATA_DIR = Path(os.environ.get("TEMP", ".")) / "MeetingIsland"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
