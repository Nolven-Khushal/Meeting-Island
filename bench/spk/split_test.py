"""Can we find a speaker change INSIDE one chunk (and not invent one when it's the same person)?"""
import json, sys, random, time
import numpy as np
sys.path.insert(0, '.')
import sherpa_onnx, assistant
SR = 16000
meta = json.load(open('bench/spk/meta.json')); spks = sorted({m['spk'] for m in meta}); test = spks[8:]
aud = {i: assistant.decode_audio(m['file']).astype(np.float32) for i, m in enumerate(meta)}
def make(thr, minoff=0.3):
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model='models/speaker/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx'), num_threads=2),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model='models/speaker/nemo_en_titanet_small.onnx', num_threads=2),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=thr), min_duration_on=0.4, min_duration_off=minoff)
    return sherpa_onnx.OfflineSpeakerDiarization(cfg)
rng = random.Random(3)
def by_spk(s): return [i for i, m in enumerate(meta) if m['spk'] == s]
def trial(kind, gap):
    a, b = rng.sample(test, 2) if kind == 'diff' else (rng.choice(test),) * 2
    ia = rng.choice(by_spk(a)); ib = rng.choice([i for i in by_spk(b) if i != ia])
    x, y = aud[ia][: 6 * SR], aud[ib][: 6 * SR]
    z = np.concatenate([x, np.zeros(int(gap * SR), np.float32), y]); return z, len(x) / SR + gap / 2
for thr in (0.45, 0.6, 0.75):
    sd = make(thr); res = {}
    for kind in ('diff', 'same'):
        for gap in (0.15, 0.4):
            ok = bound = 0; n = 40; t = time.perf_counter(); tot = 0
            for _ in range(n):
                z, true_b = trial(kind, gap); tot += len(z) / SR
                segs = sd.process(z).sort_by_start_time(); ids = sorted({s.speaker for s in segs})
                if kind == 'diff':
                    if len(ids) >= 2:
                        ok += 1
                        sw = [segs[k].start for k in range(1, len(segs)) if segs[k].speaker != segs[k - 1].speaker]
                        bound += any(abs(t_ - true_b) < 0.7 for t_ in sw)
                else:
                    ok += (len(ids) == 1)
            res[(kind, gap)] = (100 * ok / n, 100 * bound / n if kind == 'diff' else None, (time.perf_counter() - t) / tot)
    print(f"cluster threshold {thr}:  " + '  '.join(f"{k[0]}/gap{k[1]}: {'found 2 speakers' if k[0]=='diff' else 'stayed 1 speaker'} {v[0]:.0f}%" + (f" (boundary within 0.7s {v[1]:.0f}%)" if v[1] is not None else '') for k, v in res.items()) + f"  | speed {np.mean([v[2] for v in res.values()]):.3f}x real time")
