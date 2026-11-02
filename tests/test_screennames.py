"""Reading who is talking from the screen: the speaking ring, and matching names to voices (screennames.py)."""
import numpy as np

import screennames as sn


def _frame(ring=None, color=(100, 215, 170)):          # BGR: a lime ring like the one video-call apps draw
    f = np.full((1080, 1920, 3), 25, np.uint8)
    f[200:480, 150:700] = (70, 60, 50)                  # a tile without a ring
    if ring:
        x1, y1, x2, y2 = ring
        for t in range(4):
            f[y1 + t, x1:x2] = color
            f[y2 - t, x1:x2] = color
            f[y1:y2, x1 + t] = color
            f[y1:y2, x2 - t] = color
    return f


def test_the_ringed_tile_is_found():
    rect = sn.find_ring(_frame((900, 300, 1500, 640)))
    assert rect is not None and all(abs(a - b) <= 8 for a, b in zip(rect, (900, 300, 1500, 640)))


def test_no_ring_means_no_answer():
    assert sn.find_ring(_frame()) is None


def test_names_are_matched_to_voices_only_when_consistent():
    segs = [dict(start=0, end=10, voice=1), dict(start=10, end=20, voice=2)]
    seen = [(t / 2, "Asha Verma") for t in range(0, 20)] + [(10 + t / 2, "Ravi Menon") for t in range(0, 20)]
    assert sn.assign(segs, seen) == {1: "Asha Verma", 2: "Ravi Menon"}


def test_one_stray_reading_does_not_name_a_voice():
    segs = [dict(start=0, end=10, voice=1)]
    assert sn.assign(segs, [(1.0, "Somebody"), (2.0, "Somebody")]) == {}      # fewer than 3 readings


def test_close_spellings_of_one_name_are_merged():
    ents = [(i, "Matt Caballero") for i in range(6)] + [(7, "Matt Cabaliero")]
    assert {n for _, n in sn.canonical(ents)} == {"Matt Caballero"}
