"""
The second look. After a meeting ends the recording is listened to again, slowly, with the whole conversation in view.

Live transcription has about one second to decide; it must cut the audio wherever the speaker happens to pause, guess who is
talking from one sentence, and never see what comes next. Here nothing is hurried, so:

  1. The whole recording is scanned for speech and for the natural pauses inside it (even 0.12 s between words).
  2. Offline speaker diarization (pyannote segmentation + TitaNet voices, clustered over the full meeting) finds WHO speaks
     WHEN, including a speaker change in the middle of continuous speech.
  3. The speech is cut only at pauses and speaker changes, in pieces the speech models like (<= 13 s). Where there is no
     pause at all the piece is read with an overlap and stitched, exactly like live. Each piece is transcribed with the same
     Swift + IndicConformer pipeline as live, with the glossary.
  4. Pieces that were only split because they were long are joined back into one line.

Built for 2-hour meetings on a modest laptop: the audio stays on disk (memory-mapped, converted a piece at a time), the loudness
scan works in blocks, and every finished piece is checkpointed so a closed app resumes instead of starting over.
Everything is offline. A live transcript always exists first; this replaces it only when it has finished.
"""
import json
import time
import traceback
from pathlib import Path

import numpy as np

SR = 16000
FRAME = 320                       # 20 ms
MAX_PIECE = 13.0                  # seconds a speech model hears at once (Swift is padded to 15 s)
OVERLAP = 0.9                     # seconds re-read after a cut inside continuous speech (same as live)
MAX_LINE = 28.0                   # a line in the transcript never joins more than this much speech
SEG_MODEL = "models/speaker/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx"
EMB_MODEL = "models/speaker/nemo_en_titanet_small.onnx"
CLUSTER_THRESHOLD = 0.5           # tuned in bench/spk/diar_test.py
SPAN_MERGE = int(__import__('os').environ.get('POLISH_MERGE', 60))      # frames of quiet (20 ms each) that still belong to one stretch of speech
BLOCK = 60 * 50                   # frames per block of the loudness scan (60 s)


def load_wav(path):
    """The lane recording as a read-only int16 memory map (a 2-hour lane is 230 MB on disk, nothing is loaded up front).
    Sizes come from the file, not the header: the header is only finished when a recording is closed properly."""
    path = Path(path)
    n = max(0, (path.stat().st_size - 44) // 2)
    return np.memmap(path, dtype="<i2", mode="r", offset=44, shape=(n,)) if n else np.zeros(0, np.int16)


def f32(audio, a, b):
    """samples a..b as float32 in -1..1"""
    return np.asarray(audio[max(0, a): max(0, b)], dtype=np.float32) / 32768


def voiced_mask(audio):
    """-> (voiced?, rms) per 20 ms frame. Loudness threshold adapts to the local level (same rule as live, per 30 s).
    Works in 60 s blocks so memory stays small however long the recording is."""
    n = len(audio) // FRAME
    rms = np.empty(n, np.float32)
    for i in range(0, n, BLOCK):
        j = min(n, i + BLOCK)
        x = f32(audio, i * FRAME, j * FRAME).reshape(j - i, FRAME)
        rms[i:j] = np.sqrt((x * x).mean(axis=1))
    thr = np.empty(n, np.float32)
    step = 750
    for i in range(0, n, step):
        seg = rms[max(0, i - step): min(n, i + 2 * step)]
        thr[i: i + step] = max(0.006, 0.18 * float(np.percentile(seg, 95)))
    return rms > thr, rms


def runs_of(mask):
    """(start, end) of every run of True."""
    m = np.asarray(mask, bool)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))


def speech_spans(voiced, merge=None, minlen=12):
    """Stretches of speech: gaps shorter than `merge` frames (0.3 s) stay inside, blips shorter than `minlen` are dropped."""
    merge = merge or SPAN_MERGE
    m = voiced.copy()
    for a, b in runs_of(~voiced):
        if a > 0 and b < len(m) and b - a < merge:
            m[a:b] = True
    return [(a, b) for a, b in runs_of(m) if b - a >= minlen]


def diarize(audio, threshold=CLUSTER_THRESHOLD, threads=2, progress=None):
    """-> [(start_s, end_s, speaker_number)] over the whole recording (numbers in order of first appearance).
    progress(done, total) -> truthy to stop.
    Two stages, because each is good at something different (bench/spk/diar_test.py, fast conversations, turns labelled right):
      * sherpa-onnx offline diarization (pyannote segmentation) finds WHERE the speaker changes, also inside continuous speech;
        its own clustering over-splits one person into several speakers (live 82% -> 91%, but 5 'speakers' for 2 people)
      * so each of its turns gets a TitaNet voice print and OUR average-linkage clustering (the one tuned for 97-100% on
        sentences) merges the turns into people: 96% with the right number of speakers."""
    import sherpa_onnx
    from paths import APP_DIR
    from speakers import refine
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(APP_DIR / SEG_MODEL)),
            num_threads=threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(APP_DIR / EMB_MODEL), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=threshold),
        min_duration_on=0.3, min_duration_off=0.3)
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    samples = f32(audio, 0, len(audio))                       # the one place the whole recording is in memory (4 bytes/sample)
    cb = (lambda done, total, _arg=None: 1 if (progress and progress(done, total)) else 0) if progress else None
    res = (sd.process(samples, callback=cb) if cb else sd.process(samples)).sort_by_start_time()
    segs = []
    for r in res:                                             # glue consecutive turns of the same sherpa speaker
        if segs and segs[-1][2] == r.speaker and r.start - segs[-1][1] < 0.3:
            segs[-1][1] = float(r.end)
        else:
            segs.append([float(r.start), float(r.end), r.speaker])
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(APP_DIR / EMB_MODEL), num_threads=threads, debug=False, provider="cpu"))
    embs = []
    for s0, e0, _ in segs:
        a = samples[int(s0 * SR): int(e0 * SR)]
        if len(a) < 0.8 * SR:                                 # too short for a reliable voice print
            embs.append(None)
            continue
        st = ex.create_stream()
        st.accept_waveform(sample_rate=SR, waveform=np.ascontiguousarray(a[: 20 * SR], dtype=np.float32))
        st.input_finished()
        v = np.array(ex.compute(st), dtype=np.float32)
        embs.append(v / (np.linalg.norm(v) + 1e-9))
    del samples
    lab = refine(embs)
    last = None
    for i in range(len(lab)):                                 # turns too short for a voice print follow the previous speaker
        if lab[i] is None:
            lab[i] = last
        else:
            last = lab[i]
    for i in range(len(lab) - 1, -1, -1):
        if lab[i] is None and i + 1 < len(lab):
            lab[i] = lab[i + 1]
    return [(s0, e0, int(l)) for (s0, e0, _), l in zip(segs, lab) if l is not None]


def frame_labels(n, turns):
    """Speaker of every 20 ms frame from the diarization turns (0 = nobody known; later turns win where they overlap)."""
    lab = np.zeros(n, np.int16)
    for s, e, k in turns:
        lab[int(s * 50): int(e * 50)] = k
    return lab


def split_long(a, b, rms, voiced, max_frames):
    """A stretch longer than the speech model likes -> [(cut_frame, hard)] cut at real pauses near the preferred length.
    hard=True means there was no pause at all (continuous speech): the cut falls at the quietest moment, maybe inside a word,
    so the pieces are read with an overlap and stitched (see Polisher)."""
    cuts = []
    while b - a > max_frames:
        lo, hi = a + int(0.45 * max_frames), a + max_frames
        best = None
        for x, y in runs_of(~voiced[lo:hi]):                  # candidate pauses: unvoiced runs; longer and later is better
            if y - x < 3:
                continue                                      # under 60 ms is just a dip inside a word
            score = (y - x) + 0.15 * (x / max(1, hi - lo))
            if best is None or score > best[0]:
                best = (score, lo + (x + y) // 2)
        if best is None:
            sm = np.convolve(rms[lo:hi], np.ones(5) / 5, "same")
            cuts.append((lo + int(np.argmin(sm[2:-2])) + 2, True))
        else:
            cuts.append((best[1], False))
        a = cuts[-1][0]
    return cuts


def speaker_runs(lab, rms, voiced, a, b, min_run=50):
    """Inside one stretch of speech [a, b): -> [(start, end, voice)] where the speaker changes.
    A change is only cut where there is a real pause (even 40 ms) within +-0.3 s of where the diarization puts it; speech with no
    pause there is kept whole and goes to the speaker who talks longer (cutting inside a word breaks both halves).
    A run shorter than `min_run` frames (1 s) is absorbed by its neighbour: a lone "haan" is not worth a piece of its own."""
    l = lab[a:b].astype(np.int16).copy()
    known = np.nonzero(l)[0]
    if len(known) == 0:
        return [(a, b, 0)]
    if len(known) < len(l):                               # frames nobody was assigned to take the nearest known speaker
        pos = np.searchsorted(known, np.arange(len(l)))
        left = known[np.clip(pos - 1, 0, len(known) - 1)]
        right = known[np.clip(pos, 0, len(known) - 1)]
        near = np.where(np.abs(np.arange(len(l)) - left) <= np.abs(right - np.arange(len(l))), left, right)
        l = np.where(l == 0, l[near], l)
    runs, start = [], 0
    for i in range(1, len(l) + 1):
        if i == len(l) or l[i] != l[start]:
            runs.append([start, i, int(l[start])]); start = i
    merged = []
    for r in runs:
        if merged and (r[1] - r[0] < min_run or merged[-1][2] == r[2]):
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    if len(merged) > 1 and merged[0][1] - merged[0][0] < min_run:
        merged[1][0] = 0
        merged.pop(0)
    out = [merged[0]]
    for r in merged[1:]:                                  # keep a speaker change only where a pause lets us cut cleanly
        lo, hi = max(1, r[0] - 15), min(len(l) - 1, r[0] + 15)
        pauses = [(x, y) for x, y in runs_of(~voiced[a + lo: a + hi]) if y - x >= 2] if hi > lo else []
        if pauses:
            x, y = min(pauses, key=lambda p: abs((p[0] + p[1]) / 2 + lo - r[0]))
            cut = lo + (x + y) // 2
            out[-1][1] = r[0] = cut
            out.append(r)
        elif r[1] - r[0] > out[-1][1] - out[-1][0]:       # no pause: the longer talker keeps the whole stretch
            out[-1] = [out[-1][0], r[1], r[2]]
        else:
            out[-1][1] = r[1]
    return [(a + x, a + y, v) for x, y, v in out if y - x >= 6]


def plan_pieces(audio, turns, max_piece=MAX_PIECE):
    """-> list of dict(a, b, voice, cont, hard_start, hard_end): where to cut the recording. a/b are 20 ms frame numbers;
    `cont` marks a piece that continues the previous one (cut only because the speech was long, not because of a pause or a
    new speaker); hard_* say that the cut fell inside continuous speech."""
    voiced, rms = voiced_mask(audio)
    lab = frame_labels(len(voiced), turns) if turns else np.zeros(len(voiced), np.int16)
    pieces = []
    for a, b in speech_spans(voiced):
        for ra, rb, voice in speaker_runs(lab, rms, voiced, a, b):
            cuts = split_long(ra, rb, rms, voiced, int(max_piece * 50))
            edges = [ra] + [c for c, _ in cuts] + [rb]
            hard = [False] + [h for _, h in cuts] + [False]
            for k in range(len(edges) - 1):
                pieces.append(dict(a=edges[k], b=edges[k + 1], voice=voice, cont=k > 0, hard_start=hard[k], hard_end=hard[k + 1]))
    return pieces


def _bk():
    import backend
    return backend


class Polisher:
    """Re-transcribes a finished recording. `asr` is the loaded LiveASR (its speech models are reused, nothing is loaded twice).
    cache: a folder where progress is checkpointed (a job that was interrupted continues from there)."""

    def __init__(self, asr, wait_ok=lambda: True, progress=lambda f, msg="": None, cache=None):
        self.asr, self.wait_ok, self.progress = asr, wait_ok, progress
        self.cache = Path(cache) if cache else None

    # ------------------------------------------------------------------ checkpoints
    def _load_turns(self, lab):
        f = self.cache / f"turns_{lab}.json" if self.cache else None
        if f and f.exists():
            try:
                return [tuple(t) for t in json.loads(f.read_text(encoding="utf-8"))]
            except Exception:
                pass
        return None

    def _save_turns(self, lab, turns):
        if self.cache:
            (self.cache / f"turns_{lab}.json").write_text(json.dumps(turns), encoding="utf-8")

    def _load_done(self, lab):
        done = {}
        f = self.cache / f"pieces_{lab}.jsonl" if self.cache else None
        if f and f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line)
                    done[d["a"]] = d["r"]
                except Exception:
                    pass                                      # a half-written last line after a crash
        return done

    # ------------------------------------------------------------------ run
    def run(self, lanes, mic_is_room=False, use_diarization=True):
        """lanes: {"System": wav path, "Mic": wav path}. -> list of transcript segments (same shape as live ones)."""
        t0 = time.time()
        loaded = {lab: load_wav(p) for lab, p in lanes.items() if Path(p).exists()}
        total = sum(len(a) for a in loaded.values()) / SR or 1.0
        done = 0.0
        out = []
        for lab, audio in loaded.items():
            secs = len(audio) / SR
            turns = []
            if use_diarization and (lab == "System" or mic_is_room):
                turns = self._load_turns(lab)
                if turns is None:
                    turns = []
                    self.progress(done / total * 0.15, "Working out who is speaking…")

                    def dprog(d, n, base=done, secs=secs):
                        self.progress((base + secs * 0.15 * d / max(1, n)) / total * 0.9, "Working out who is speaking…")
                        return 0 if self.wait_ok() else 1

                    try:
                        turns = diarize(audio, progress=dprog)
                        self._save_turns(lab, turns)
                    except Exception:
                        traceback.print_exc()
                if not self.wait_ok():
                    raise InterruptedError("polish cancelled")
            pieces = plan_pieces(audio, turns)
            base = done + secs * 0.15
            segs = self._transcribe(lab, audio, pieces, bool(turns), lambda frac, base=base, secs=secs: self.progress(
                (base + frac * secs * 0.85) / total * 0.9, "Listening again…"))
            out += segs
            done += secs
        out.sort(key=lambda s: s["start"])
        print(f"[polish] {len(out)} lines from {total:.0f} s of audio in {time.time() - t0:.0f} s", flush=True)
        return out

    def _transcribe(self, lab, audio, pieces, has_voices, report):
        asr = self.asr
        last_lang = None
        lines = []
        total_frames = sum(p["b"] - p["a"] for p in pieces) or 1
        seen = 0
        done = self._load_done(lab)
        ck = open(self.cache / f"pieces_{lab}.jsonl", "a", encoding="utf-8") if self.cache else None
        try:
            for i, p in enumerate(pieces):
                if not self.wait_ok():
                    raise InterruptedError("polish cancelled")
                a, b = p["a"] * FRAME, p["b"] * FRAME
                if p["a"] in done:
                    r = done[p["a"]]                          # transcribed before the app was closed
                else:
                    pad = int(0.2 * SR)                       # a little silence around the speech: models hear word edges better
                    lead = int(OVERLAP * SR) if p["hard_start"] else pad    # a cut inside continuous speech: re-read the last second
                    seg = f32(audio, a - lead, b + pad)
                    r = asr.hinglish_final(seg, last_lang, forced=p["hard_end"])   # (a half-heard last word is dropped, the next piece re-reads it)
                    if ck:
                        ck.write(json.dumps(dict(a=p["a"], r=r), ensure_ascii=False) + "\n")
                        ck.flush()
                seen += p["b"] - p["a"]
                report(seen / total_frames)
                if r is None:
                    continue
                last_lang = r["lang"]
                voice = p["voice"] if has_voices else None
                line = dict(start=round(a / SR, 2), end=round(b / SR, 2), text=r["text"], lang=r["lang"], conf=r["conf"], speaker=lab,
                            words=[tuple(w) for w in (r["words"] or [])], native=r["native"], voice=voice, emb=None, forced=False,
                            id=f"P-{lab}-{i}", utt=i)
                prev = lines[-1] if lines else None
                if (prev is not None and p["cont"] and prev["voice"] == voice and line["end"] - prev["start"] <= MAX_LINE):
                    if p["hard_start"]:
                        lines[-1] = _bk().stitch(prev, line)  # same fuzzy overlap join live uses
                    else:                                     # cut at a real pause: just continue the line
                        prev["end"] = line["end"]
                        prev["text"] = (prev["text"] + " " + line["text"]).strip()
                        pn, ln = prev.get("native"), line.get("native")
                        prev["native"] = (pn + " " + ln).strip() if (pn and ln) else None
                        prev["words"] = (prev["words"] + line["words"]) if (prev["words"] and line["words"]) else []
                        prev["conf"] = min(prev["conf"], line["conf"])
                else:
                    lines.append(line)
        finally:
            if ck:
                ck.close()
        return lines
