"""End-of-meeting speaker clustering (speakers.refine): sentences of the same voice end up together."""
import time

import numpy as np

from speakers import refine


def _voices(n_per, k, noise=0.25, seed=3):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, 192))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    embs, truth = [], []
    for i in range(n_per * k):
        c = i % k
        v = centres[c] + noise * rng.normal(size=192) / np.sqrt(192) * 3
        embs.append((v / np.linalg.norm(v)).astype(np.float32))
        truth.append(c)
    return embs, truth


def test_three_voices_are_found_and_sentences_grouped_correctly():
    embs, truth = _voices(20, 3)
    labels = refine(embs)
    assert len(set(labels)) == 3
    assert len({(t, l) for t, l in zip(truth, labels)}) == 3        # one cluster per true voice, none mixed


def test_sentences_too_short_for_a_voice_stay_unlabelled():
    embs, _ = _voices(10, 2)
    embs[4] = None
    labels = refine(embs)
    assert labels[4] is None and all(l is not None for i, l in enumerate(labels) if i != 4)


def test_a_long_meeting_is_clustered_quickly():
    embs, _ = _voices(375, 4)                                       # 1500 sentences: a 2-hour meeting
    t = time.time()
    labels = refine(embs)
    assert time.time() - t < 15
    assert len(set(labels)) == 4
