"""
Live transcript versus the after-meeting second pass, on CONTINUOUS speech (where live has to cut mid-sentence).

Streams are built from the benchmark clips with known transcripts, joined with short gaps (shorter than the live 0.5 s pause
rule), so live must make forced cuts, while the second pass can cut at the real micro-pauses.
  A  6 pairs of consecutive lecture clips, 1.0x      (single fast technical speaker per pair)
  B  the same style of clips at 1.3x, tighter gaps   (fast speaker)
  C  conversation clips, 0.2 s gaps, 1.0x            (people answering each other quickly)

  python bench/polish_test.py [A B C]  [--no-live]  [--variants full,nodiar]
"""
import json, sys, time, queue, wave, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
import assistant
import backend as bk
from run import norm, edits
import polish

SR = 16000
META = json.load(open(ROOT / "bench/meta.json", encoding="utf-8"))
META13 = {m["id"]: m for m in json.load(open(ROOT / "bench/meta_x1.3.json", encoding="utf-8"))}
RES = ROOT / "bench/results"
RES.mkdir(exist_ok=True)


def pairs(src):
    g = collections.OrderedDict()
    for m in META:
        if m["src"] == src:
            g.setdefault(m["id"].rsplit("_", 1)[0], []).append(m)
    return list(g.values())


def build(name):
    mu, iv = pairs("mucs"), [m for m in META if m["src"] == "ivoices"]
    audio, refs, ids = [], [], []
    if name in ("A", "B"):
        sel = mu[:7] if name == "A" else mu[7:14]
        inner, outer = (0.2, 0.8) if name == "A" else (0.12, 0.6)
        for pr in sel:
            for k, m in enumerate(pr):
                f = str(ROOT / "bench/data_x1.3" / (Path(m["file"]).name + ".wav")) if name == "B" else m["file"]
                audio.append(assistant.decode_audio(f).astype(np.float32))
                refs.append(m["ref"])
                audio.append(np.zeros(int((inner if k == 0 else outer) * SR), np.float32))
    else:
        for m in iv[:16]:
            audio.append(assistant.decode_audio(m["file"]).astype(np.float32))
            refs.append(m["ref"])
            audio.append(np.zeros(int(0.2 * SR), np.float32))
    z = np.concatenate(audio)
    path = ROOT / f"scratch/polish_stream_{name}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(z, -1, 1) * 32767).astype(np.int16).tobytes())
    return path, " ".join(refs), len(z) / SR


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    return edits(r, h) / max(1, len(r)), len(r)


def run_live(b, q, wav, dur):
    b.cfg["sim"] = str(wav)
    assert b.start_session()
    t0 = time.time()
    while time.time() - t0 < dur + 3:
        time.sleep(0.5)
        while not q.empty():
            q.get()
    b.end_session()
    while True:
        k, *rest = q.get()
        if k == "ended":
            sess = rest[0]
            break
    segs = sorted(sess.segs, key=lambda s: s["start"])
    sess.discard()
    return segs


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    streams = args or ["A", "B", "C"]
    no_live = "--no-live" in sys.argv
    variants = ["full", "nodiar"]
    for a in sys.argv:
        if a.startswith("--variants"):
            variants = a.split("=", 1)[1].split(",")
    q = queue.Queue()
    b = bk.Backend(lambda *a: q.put(a), dict(langs="en,hi", model="small", hi_model="turbo", hotwords=None, threads=4, hi_threads=6,
                                             window=14.0, mic=False, sim=None, engine="hinglish", speakers=True, record="off"))
    b.load()
    while b.asr is None or not b.hindi_ready() or b.asr.speakers is None:
        time.sleep(0.3)
    totals = collections.defaultdict(lambda: [0, 0])
    for name in streams:
        wav, ref, dur = build(name)
        print(f"\n=== stream {name}: {dur:.0f} s, {len(norm(ref))} reference words", flush=True)
        cache = RES / f"polish_live_{name}.json"
        if cache.exists() and no_live:
            live = json.loads(cache.read_text(encoding="utf-8"))
        else:
            live = run_live(b, q, wav, dur)
            cache.write_text(json.dumps(live, ensure_ascii=False), encoding="utf-8")
        w, n = wer(ref, " ".join(s["text"] for s in live))
        print(f"  LIVE            WER {100*w:5.1f}%   lines {len(live)}", flush=True)
        totals["live"][0] += w * n; totals["live"][1] += n
        for v in variants:
            t = time.time()
            P = polish.Polisher(b.asr)
            segs = P.run({"System": wav}, use_diarization=(v != "nodiar"))
            w, n = wer(ref, " ".join(s["text"] for s in segs))
            print(f"  POLISH {v:8s} WER {100*w:5.1f}%   lines {len(segs)}   ({(time.time()-t)/dur:.2f}x real time)", flush=True)
            totals[v][0] += w * n; totals[v][1] += n
    print("\n=== all streams")
    for k, (e, n) in totals.items():
        print(f"  {k:8s} WER {100*e/max(1,n):5.1f}%")
    b.shutdown()


if __name__ == "__main__":
    main()
