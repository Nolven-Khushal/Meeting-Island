"""Tests use synthetic data only and a throw-away data folder; no models and no private files are needed."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MEETING_ISLAND_DATA", tempfile.mkdtemp(prefix="mi-tests-"))


@pytest.fixture()
def clean_data():
    """an empty settings/learning folder for every test that touches it"""
    import corrections
    if corrections.DIR.is_dir():
        shutil.rmtree(corrections.DIR, ignore_errors=True)
    corrections.LEARNED.unlink(missing_ok=True)
    yield corrections
