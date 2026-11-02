"""Online speaker assignment + end-of-session refinement, on conversations built from real clips of known speakers."""
import json, sys, random, collections
import numpy as np
sys.path.insert(0, '.')
import sherpa_onnx, assistant
SR = 16000
meta = json.load(open('bench/spk/meta.json'))
ex = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model='models/speaker/nemo_en_titanet_small.onnx', num_threads=2, debug=False, provider='cpu'))
def emb(a):
    s = ex.create_stream(); s.accept_waveform(sample_rate=SR, waveform=a); s.input_finished(); v = np.array(ex.compute(s), np.float32); return v / np.linalg.norm(v)
E = np.stack([emb(assistant.decode_audio(m['file']).astype(np.float32)) for m in meta]); labels = [m['spk'] for m in meta]; dur = [m['dur'] for m in meta]
np.save('bench/spk/titanet_all.npy', E); json.dump(labels, open('bench/spk/labels.json', 'w'))
spks = sorted(set(labels)); print(len(meta), 'clips,', len(spks), 'speakers')
def online(order, tau, maxk=8):
    cents, counts, pred = [], [], []
    for i in order:
        e = E[i]; sims = [float(c @ e / np.linalg.norm(c)) for c in cents]
        j = int(np.argmax(sims)) if sims else -1
        if j >= 0 and sims[j] >= tau: cents[j] = cents[j] * counts[j] + e; counts[j] += 1; cents[j] = cents[j] / counts[j]
        elif len(cents) < maxk: cents.append(e.copy()); counts.append(1); j = len(cents) - 1
        pred.append(j)
    return pred
def agglo(idx, tau):
    cl = [[i] for i in idx]
    while len(cl) > 1:
        best, bp = -1, None
        for a in range(len(cl)):
            for b in range(a + 1, len(cl)):
                s = np.mean([E[i] @ E[j] for i in cl[a] for j in cl[b]])
                if s > best: best, bp = s, (a, b)
        if best < tau: break
        cl[bp[0]] += cl[bp[1]]; del cl[bp[1]]
    pred = {}
    for k, c in enumerate(cl):
        for i in c: pred[i] = k
    return [pred[i] for i in idx]
def acc(order, pred):
    by = collections.defaultdict(collections.Counter)
    for i, p in zip(order, pred): by[p][labels[i]] += 1
    ok = sum(max(c.values()) for c in by.values()); return ok / len(order), len(by)
rng = random.Random(7); tune, test = spks[:8], spks[8:]
def conversations(pool, n, k, per=14):
    out = []
    for _ in range(n):
        sp = rng.sample(pool, k); idx = [i for i, l in enumerate(labels) if l in sp]; rng.shuffle(idx); out.append(idx[:per] if len(idx) >= per else idx)
    return out
print('tuning threshold on speakers 1-8, evaluating on 8 UNSEEN speakers')
for tau in (0.30, 0.38, 0.45, 0.52, 0.60):
    r = [acc(o, online(o, tau)) for k in (2, 3, 4) for o in conversations(tune, 25, k)]
    print(f"  online tau={tau:.2f}: tune acc {100*np.mean([a for a,_ in r]):.1f}%")
for k in (2, 3, 4, 5):
    for tau in (0.38, 0.45):
        r = [acc(o, online(o, tau)) for o in conversations(test, 40, k)]
        print(f"TEST {k} speakers  online tau={tau}: accuracy {100*np.mean([a for a,_ in r]):.1f}%  speakers found {np.mean([n for _,n in r]):.1f} (true {k})")
for k in (2, 3, 4):
    for tau in (0.30, 0.38, 0.45):
        r = []
        for o in conversations(test, 25, k): p = agglo(o, tau); r.append(acc(o, p))
        print(f"TEST {k} speakers  end-of-session clustering tau={tau}: accuracy {100*np.mean([a for a,_ in r]):.1f}%  speakers found {np.mean([n for _,n in r]):.1f}")
