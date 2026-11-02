"""
Who spoke? Voice-based speaker labels for the transcript, fully offline.

Each finished sentence gets a voice "fingerprint" (NVIDIA TitaNet-small, 192 numbers, 40 MB, ~0.014x real time) that is
compared with the voices heard so far in this session: close enough -> same person, otherwise a new speaker. When the
session ends, all sentences are re-clustered together, which fixes early mistakes (labels can only be wrong where
the voice was heard for less than ~1 s).

Measured on 177 real clips of 16 different people the model had never seen (bench/spk/):
  online labelling accuracy 97-100% for 2-5 speakers; with the end-of-session re-clustering 99-100%.
Limits: two people inside ONE uninterrupted chunk (no pause between them) get one label (splitting inside a chunk was
tested and is not reliable enough); sentences shorter than ~0.8 s have no usable voice and inherit the previous speaker;
people with very similar voices can merge; heavy background music hurts.
"""
import threading
from pathlib import Path

import numpy as np

SR = 16000
from paths import APP_DIR
MODEL = APP_DIR / "models" / "speaker" / "nemo_en_titanet_small.onnx"
MIN_SECS = 0.8        # shorter than this: no reliable voice, the caller keeps the previous speaker
TAU = 0.45            # cosine similarity needed to call two voices "the same person" (tuned on 8 speakers, tested on 8 others)


class SpeakerTracker:
    def __init__(self, tau=TAU, max_speakers=8, threads=2):
        import sherpa_onnx
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(MODEL), num_threads=threads, debug=False, provider="cpu")
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.tau, self.max, self.lock = tau, max_speakers, threading.Lock()
        self.reset()

    def reset(self):
        self.cents, self.counts = [], []

    def embed(self, audio):
        a = np.ascontiguousarray(audio[: 20 * SR], dtype=np.float32)
        with self.lock:
            s = self.ex.create_stream()
            s.accept_waveform(sample_rate=SR, waveform=a)
            s.input_finished()
            v = np.array(self.ex.compute(s), dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def assign(self, audio):
        """-> (speaker number starting at 1, embedding) or (None, None) if the audio is too short to tell."""
        if len(audio) < MIN_SECS * SR:
            return None, None
        e = self.embed(audio)
        sims = [float(c @ e) / (np.linalg.norm(c) + 1e-9) for c in self.cents]
        j = int(np.argmax(sims)) if sims else -1
        if j >= 0 and sims[j] >= self.tau:
            if len(audio) >= 2 * SR:                      # only reasonably long sentences refine a speaker's profile
                self.cents[j] = (self.cents[j] * self.counts[j] + e) / (self.counts[j] + 1)
                self.counts[j] += 1
            return j + 1, e
        if len(self.cents) < self.max:
            self.cents.append(e.copy())
            self.counts.append(1)
            return len(self.cents), e
        return (j + 1 if j >= 0 else 1), e


def refine(embs, tau=0.40, min_size=3):
    """
    Re-cluster every sentence of a finished session together (average-linkage agglomerative clustering on voice
    similarity). embs: list of embedding arrays or None. Returns a list of speaker numbers (1.. in order of first
    appearance) with None kept as None.
    """
    idx = [i for i, e in enumerate(embs) if e is not None]
    if len(idx) < 2:
        return [1 if e is not None else None for e in embs]
    E = np.stack([np.array(embs[i], dtype=np.float32) for i in idx])
    n0 = len(E)
    M = E @ E.T                                           # average-linkage similarity: mean pairwise cosine between clusters
    np.fill_diagonal(M, -2)
    alive = np.ones(n0, bool)
    members = [[i] for i in idx]
    sums_all = E.copy()
    cnt_all = np.ones(n0, np.float32)
    while alive.sum() > 1:                                # a merge updates one row of M (average linkage is linear): O(N^2) overall
        a, b = divmod(int(np.argmax(M)), n0)
        if M[a, b] < tau:
            break
        a, b = min(a, b), max(a, b)
        row = (cnt_all[a] * M[a] + cnt_all[b] * M[b]) / (cnt_all[a] + cnt_all[b])
        M[a, :] = row
        M[:, a] = row
        M[b, :] = -2
        M[:, b] = -2
        alive[b] = False
        M[a, ~alive] = -2
        M[~alive, a] = -2
        M[a, a] = -2
        members[a] += members[b]
        sums_all[a] += sums_all[b]
        cnt_all[a] += cnt_all[b]
    keep = [k for k in range(n0) if alive[k]]
    groups = [members[k] for k in keep]
    sums = [sums_all[k] for k in keep]
    cnt = [float(cnt_all[k]) for k in keep]
    # stray clusters of 1-2 sentences are usually one person's noisier moments: fold them into the closest real speaker
    big = [k for k, g in enumerate(groups) if len(g) >= min_size]
    if big and len(big) < len(groups):
        cents = {k: sums[k] / cnt[k] for k in big}
        cents = {k: v / (np.linalg.norm(v) + 1e-9) for k, v in cents.items()}
        for k, g in enumerate(groups):
            if k in cents:
                continue
            for i in g:
                e = np.array(embs[i], dtype=np.float32)
                best = max(big, key=lambda b: float(cents[b] @ e))
                if float(cents[best] @ e) >= 0.28:
                    groups[best].append(i)
                    groups[k] = [x for x in groups[k] if x != i]
        groups = [g for g in groups if g]
    labels = {}
    for k, g in enumerate(groups):
        for i in g:
            labels[i] = k
    order = {}
    out = []
    for i, e in enumerate(embs):
        if e is None:
            out.append(None)
            continue
        k = labels[i]
        if k not in order:
            order[k] = len(order) + 1
        out.append(order[k])
    return out
