"""Speaker-change detection with TitaNet window embeddings: compare the average voice left vs right of every candidate point."""
import json, sys, random, time
import numpy as np
sys.path.insert(0, '.')
import sherpa_onnx, assistant
SR = 16000
meta = json.load(open('bench/spk/meta.json')); spks = sorted({m['spk'] for m in meta}); test = spks[8:]; tune = spks[:8]
aud = {i: assistant.decode_audio(m['file']).astype(np.float32) for i, m in enumerate(meta)}
ex = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model='models/speaker/nemo_en_titanet_small.onnx', num_threads=2, debug=False, provider='cpu'))
def emb(a):
    s = ex.create_stream(); s.accept_waveform(sample_rate=SR, waveform=np.ascontiguousarray(a)); s.input_finished(); v = np.array(ex.compute(s), np.float32); return v / np.linalg.norm(v)
def detect(z, win, hop=0.5, block=2.0):
    """-> (min_score, boundary_seconds). z: mono 16 kHz. Windows are embedded once, then blocks are averaged."""
    starts = np.arange(0, len(z) / SR - win + 1e-6, hop)
    if len(starts) < 6: return 1.0, None
    W = np.stack([emb(z[int(t * SR): int((t + win) * SR)]) for t in starts]); best = (1.0, None); k = int(block / hop)
    for c in range(k, len(starts) - k + 1):
        L, R = W[c - k: c].mean(0), W[c: c + k].mean(0); sc = float(L @ R / np.linalg.norm(L) / np.linalg.norm(R))
        if sc < best[0]: best = (sc, starts[c] )
    return best
rng = random.Random(5)
def by_spk(s): return [i for i, m in enumerate(meta) if m['spk'] == s]
def trial(pool, kind, gap):
    a, b = rng.sample(pool, 2) if kind == 'diff' else (rng.choice(pool),) * 2
    ia = rng.choice(by_spk(a)); ib = rng.choice([i for i in by_spk(b) if i != ia]); x, y = aud[ia][: 6 * SR], aud[ib][: 6 * SR]
    return np.concatenate([x, np.zeros(int(gap * SR), np.float32), y]), len(x) / SR + gap / 2
for win in (1.5, 2.0):
    data = {}
    for pool_name, pool in (('tune', tune), ('TEST', test)):
        for kind in ('diff', 'same'):
            sc, bd = [], []
            for _ in range(30):
                z, tb = trial(pool, kind, 0.3); s, b = detect(z, win); sc.append(s); bd.append(abs(b - tb) if b is not None else 9)
            data[(pool_name, kind)] = (np.array(sc), np.array(bd))
    for thr in (0.25, 0.30, 0.35, 0.40, 0.45):
        r = []
        for pool_name in ('tune', 'TEST'):
            d_sc, d_bd = data[(pool_name, 'diff')]; s_sc, _ = data[(pool_name, 'same')]
            r.append(f"{pool_name}: detect {100*np.mean(d_sc < thr):.0f}% (boundary within 0.5s: {100*np.mean((d_sc < thr) & (d_bd < 0.5)):.0f}%), false alarm {100*np.mean(s_sc < thr):.0f}%")
        print(f"window {win}s threshold {thr:.2f}:  " + '   '.join(r))
t = time.perf_counter(); z, _ = trial(test, 'diff', 0.3); detect(z, 1.5); print(f"cost: {(time.perf_counter()-t)/(len(z)/SR):.3f}x real time")
