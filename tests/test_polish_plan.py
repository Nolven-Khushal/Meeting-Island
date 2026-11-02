"""Where the final pass cuts the recording (polish.plan_pieces): at pauses, never inside speech if a pause exists."""
import numpy as np

import polish

SR = 16000


def _speech(secs, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(secs * SR)) / SR
    return (0.12 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)) + 0.02 * rng.normal(size=len(t))).astype(np.float32)


def _pcm(parts):
    return (np.clip(np.concatenate(parts), -1, 1) * 32767).astype(np.int16)


def test_two_stretches_separated_by_a_long_pause_become_two_pieces():
    audio = _pcm([_speech(4), np.zeros(int(2.0 * SR), np.float32), _speech(4, 2)])
    pieces = polish.plan_pieces(audio, [])
    assert len(pieces) == 2 and not pieces[1]["cont"]


def test_a_long_monologue_is_split_for_the_speech_models_and_flagged():
    audio = _pcm([_speech(40)])                       # 40 s without any pause
    pieces = polish.plan_pieces(audio, [])
    assert len(pieces) >= 3
    assert all((p["b"] - p["a"]) / 50 <= polish.MAX_PIECE + 0.5 for p in pieces)
    assert all(p["cont"] for p in pieces[1:]) and any(p["hard_start"] for p in pieces[1:])


def test_a_short_pause_inside_a_sentence_does_not_split_it():
    audio = _pcm([_speech(3), np.zeros(int(0.5 * SR), np.float32), _speech(3, 2)])
    assert len(polish.plan_pieces(audio, [])) == 1


def test_silence_alone_has_nothing_to_transcribe():
    assert polish.plan_pieces(np.zeros(SR * 5, np.int16), []) == []
