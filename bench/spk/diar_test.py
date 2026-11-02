"""
Who spoke, in hard conversations: live method (per-pause chunks + TitaNet + online/refine clustering) versus
offline diarization (pyannote segmentation 3.0 + TitaNet + clustering, sherpa-onnx) on the whole recording.

Conversations are built from real clips of 8 speakers the clustering was NOT tuned on, with short gaps between turns
(0.1-0.8 s), so the live method's 0.5 s pause rule sometimes puts two speakers into one chunk.
Score = share of clips (turns) labelled with the right person (each predicted speaker mapped to its majority true speaker).
"""
import collections, json, random, sys, time
import numpy as np
sys.path.insert(0, '.')
import sherpa_onnx, assistant
from speakers import SpeakerTracker, refine
SR = 16000
meta = json.load(open('bench/spk/meta.json'))
spks = sorted({m['spk'] for m in meta}); tune, test = spks[:8], spks[8:]
tracker = SpeakerTracker(threads=2)


def build(rng, pool, k, turns=16, gaps=(0.12, 0.3, 0.8)):
    sp = rng.sample(pool, k)
    clips = {s: [m for m in meta if m['spk'] == s and m['dur'] <= 7] for s in sp}
    audio, truth, t, last = [], [], 0.0, None
    for _ in range(turns):
        s = rng.choice([x for x in sp if x != last]); last = s
        a = assistant.decode_audio(rng.choice(clips[s])['file']).astype(np.float32)
        gap = rng.choice(gaps)
        audio += [a, np.zeros(int(gap * SR), np.float32)]
        truth.append((s, t, t + len(a) / SR)); t += len(a) / SR + gap
    return np.concatenate(audio), truth, sp


def score(pred_per_clip, truth):
    by = collections.defaultdict(collections.Counter)
    for p, (s, _, _) in zip(pred_per_clip, truth):
        by[p][s] += 1
    ok = sum(max(c.values()) for c in by.values())
    return ok / len(truth), len(by)


def live_method(audio, truth):
    """chunks at 0.5 s pauses, like the live lane; one voice per chunk; online assignment then end-of-session refine."""
    fr = audio[: len(audio) // 320 * 320].reshape(-1, 320)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    voiced = rms > max(0.006, 0.18 * np.percentile(rms, 95))
    chunks, start, last_v = [], None, None
    for i, v in enumerate(voiced):
        if v:
            if start is None:
                start = i
            last_v = i
        elif start is not None and i - last_v >= 25:        # 0.5 s of quiet ends the chunk
            chunks.append((start, last_v + 1)); start = None
    if start is not None:
        chunks.append((start, last_v + 1))
    tracker.reset()
    embs, online = [], []
    for a, b in chunks:
        seg = audio[a * 320:b * 320]
        n, e = tracker.assign(seg)
        online.append(n); embs.append(None if e is None else e)
    ref = refine(embs)
    lab = [r if r is not None else (online[i] or 1) for i, r in enumerate(ref)]
    per_clip = []
    for _, s, e in truth:
        mid = (s + e) / 2 / 0.02
        k = next((j for j, (a, b) in enumerate(chunks) if a <= mid < b), None)
        if k is None:
            k = int(np.argmin([abs((a + b) / 2 - mid) for a, b in chunks]))
        per_clip.append(lab[k])
    return per_clip


def make_sd(threshold, min_on=0.3, min_off=0.3, seg_model='int8'):
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model='models/speaker/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx'), num_threads=2),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model='models/speaker/nemo_en_titanet_small.onnx', num_threads=2),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=threshold),
        min_duration_on=min_on, min_duration_off=min_off)
    return sherpa_onnx.OfflineSpeakerDiarization(cfg)


def diar_method(sd, audio, truth):
    res = sd.process(audio).sort_by_start_time()
    segs = [(r.start, r.end, r.speaker) for r in res]
    per_clip = []
    for _, s, e in truth:
        ov = collections.Counter()
        for a, b, sp in segs:
            o = min(e, b) - max(s, a)
            if o > 0:
                ov[sp] += o
        per_clip.append(ov.most_common(1)[0][0] if ov else -1)
    return per_clip


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    thresholds = [float(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['0.5', '0.65', '0.8'])]
    for pool_name, pool in (('TUNE speakers', tune), ('TEST (unseen) speakers', test)):
        print(f'=== {pool_name}')
        for k in (2, 3, 4):
            rng = random.Random(100 + k)
            convs = [build(rng, pool, k) for _ in range(N)]
            lv = [score(live_method(a, t), t) for a, t, _ in convs]
            print(f'  {k} speakers  LIVE chunks+TitaNet+refine : acc {100*np.mean([x for x,_ in lv]):5.1f}%  speakers found {np.mean([n for _,n in lv]):.1f}')
            for th in thresholds:
                sd = make_sd(th)
                t0 = time.time()
                dv = [score(diar_method(sd, a, t), t) for a, t, _ in convs]
                dt = time.time() - t0; tot = sum(len(a) for a, _, _ in convs) / SR
                print(f'  {k} speakers  OFFLINE diarization th={th}: acc {100*np.mean([x for x,_ in dv]):5.1f}%  speakers found {np.mean([n for _,n in dv]):.1f}   (speed {dt/tot:.2f}x real time)')


def diar_refine_method(sd, audio, truth, min_secs=0.8, tau=0.40):
    """sherpa segmentation (turn boundaries, may over-split speakers) + OUR TitaNet embeddings per turn merged with refine()."""
    res = sd.process(audio).sort_by_start_time()
    segs = []
    for r in res:                                             # glue consecutive turns of the same sherpa speaker
        if segs and segs[-1][2] == r.speaker and r.start - segs[-1][1] < 0.3:
            segs[-1][1] = r.end
        else:
            segs.append([r.start, r.end, r.speaker])
    embs = []
    for s, e, _ in segs:
        a = audio[int(s * SR):int(e * SR)]
        embs.append(tracker.embed(a) if len(a) >= min_secs * SR else None)
    lab = refine(embs, tau=tau)
    last = None
    for i in range(len(lab)):                                 # too-short turns follow the previous speaker
        if lab[i] is None:
            lab[i] = last
        else:
            last = lab[i]
    for i in range(len(lab) - 1, -1, -1):
        if lab[i] is None and i + 1 < len(lab):
            lab[i] = lab[i + 1]
    per_clip = []
    for _, s, e in truth:
        ov = collections.Counter()
        for (a, b, _sp), l in zip(segs, lab):
            o = min(e, b) - max(s, a)
            if o > 0:
                ov[l] += o
        per_clip.append(ov.most_common(1)[0][0] if ov else -1)
    return per_clip


def run_refined(N=8):
    sd = make_sd(0.5)
    for pool_name, pool in (('TUNE speakers', tune), ('TEST (unseen) speakers', test)):
        print(f'=== {pool_name}  (diarization turns + TitaNet refine)', flush=True)
        for k in (2, 3, 4):
            rng = random.Random(100 + k)
            convs = [build(rng, pool, k) for _ in range(N)]
            lv = [score(live_method(a, t), t) for a, t, _ in convs]
            dv = [score(diar_refine_method(sd, a, t), t) for a, t, _ in convs]
            print(f'  {k} speakers  LIVE: acc {100*np.mean([x for x,_ in lv]):5.1f}% found {np.mean([n for _,n in lv]):.1f}'
                  f'   | OFFLINE turns+refine: acc {100*np.mean([x for x,_ in dv]):5.1f}% found {np.mean([n for _,n in dv]):.1f}', flush=True)
