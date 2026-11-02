"""
Hindi / Hinglish accuracy + speed benchmark on 80 real clips (40 MUCS-2021 lecture, 40 IndicVoices conversation).

Every system's output and every reference are put into the same form before scoring: Roman script (Devanagari is
transliterated) and then a spelling-tolerant "skeleton" (aa->a, w->v, sh->s, ai->e ...), so "hai"/"he" or
"kaam"/"kam" are not counted as errors, but a wrong word is. We report word error rate (WER) and character error
rate (CER) on that skeleton, plus real-time factor (processing time / audio time, lower = faster).

  python bench/run.py system [system ...]       systems: current turbo_hi swift apex indic hybrid
"""
import gc
import os
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")
import assistant
from engine import Engine
from translit import to_roman

SR = 16000
META = json.load(open(Path(__file__).with_name(os.environ.get("BENCH_META", "meta.json")), encoding="utf-8"))


def skeleton(w):
    w = w.lower()
    for a, b in (("chh", "C"), ("ch", "C"), ("ph", "f"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"), ("bh", "b"),
                 ("jh", "j"), ("sh", "s"), ("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u"), ("ai", "e"),
                 ("ei", "e"), ("au", "o"), ("ou", "o"), ("w", "v"), ("z", "j"), ("q", "k"), ("x", "ks"), ("c", "k"), ("y", "i")):
        w = w.replace(a, b)
    w = w.replace("h", "")
    w = re.sub(r"(.)\1+", r"\1", w)
    w = re.sub(r"([aeiou])n$", r"\1", w)          # nasal endings: mein / me, hain / hai
    if len(w) > 2 and w.endswith("a"):
        w = w[:-1]
    return w


def norm(text):
    t = to_roman(text or "").lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return [k for k in (skeleton(w) for w in t.split()) if k]


def edits(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[-1]


_SUB = int(os.environ.get("BENCH_SUBSET", "1"))
if _SUB > 1:
    META = META[::_SUB]


def load_audio():
    import os
    out = []
    for m in META:
        a = assistant.decode_audio(m["file"]).astype(np.float32)
        mode = os.environ.get("BENCH_PRE", "")
        if "norm" in mode:                    # loudness-normalise: RMS of the louder (speech) part -> 0.10, soft-limited
            act = np.abs(a) > 0.02
            rms = float(np.sqrt((a[act] ** 2).mean())) if act.sum() > 800 else float(np.sqrt((a ** 2).mean()) + 1e-6)
            a = np.tanh(a * (0.10 / max(rms, 1e-4)) * 1.2) / 1.2
        if "hp" in mode:                      # remove rumble below ~70 Hz (one-pole high-pass)
            y = np.empty_like(a); prev_x = prev_y = 0.0; al = 0.973
            for i in range(len(a)):
                prev_y = al * (prev_y + a[i] - prev_x); prev_x = a[i]; y[i] = prev_y
            a = y.astype(np.float32)
        out.append((m, a))
    return out


# ----------------------------------------------------------------------------- systems
def sys_current():
    """What the app does today: detect en/hi with small, English -> small, Hindi -> turbo (unpadded)."""
    small, turbo = Engine("small", threads=4), Engine("turbo", threads=5)

    def run(a):
        lang, _ = small.detect_language(a[-6 * SR:], ["en", "hi"])
        r = (small if lang == "en" else turbo).transcribe(a, lang)
        return r["text"] if r else ""
    return run, lambda: None


def sys_turbo_hi():
    turbo = Engine("turbo", threads=5)
    return (lambda a: (turbo.transcribe(a, "hi") or {}).get("text", "")), None


def _oriserve(path, threads):
    e = Engine(path, threads=threads)

    def run(a):
        r = e.transcribe(a, "hi")
        return r["text"] if r else ""
    return run, None


def sys_swift():
    return _oriserve("models/hinglish-swift-ct2", 4)


def sys_swift_engine_pad():
    """Our own engine in padded mode - what the app will use."""
    e = Engine("models/hinglish-swift-ct2", threads=4, pad=True, roman=True)
    return (lambda a: (e.transcribe(a, "hi") or {}).get("text", "")), None


def sys_apex():
    return _oriserve("models/hinglish-apex-ct2", 5)


def _indic():
    import onnx_asr
    return onnx_asr.load_model("OpenVoiceOS/ai4bharat-indicconformer-hi-onnx", quantization="int8")


def sys_indic():
    m = _indic()
    return (lambda a: m.recognize(a, sample_rate=SR)), None


def sys_hybrid():
    """Detect en/hi with small; English -> small, Hindi -> IndicConformer."""
    small, m = Engine("small", threads=4), _indic()

    def run(a):
        lang, _ = small.detect_language(a[-6 * SR:], ["en", "hi"])
        if lang == "en":
            r = small.transcribe(a, "en")
            return r["text"] if r else ""
        return m.recognize(a, sample_rate=SR)
    return run, None


def _padded(path, lang="hi", threads=5):
    """Standard faster-whisper decoding (input padded to 30 s) - to check whether the unpadded speed trick costs accuracy."""
    from faster_whisper import WhisperModel
    m = WhisperModel(path, device="cpu", compute_type="int8", cpu_threads=threads)

    def run(a):
        segs, _ = m.transcribe(a, language=lang, beam_size=1, temperature=0.0, vad_filter=False,
                               condition_on_previous_text=False, without_timestamps=True)
        return " ".join(s.text.strip() for s in segs)
    return run, None


def _gloss_prompt():
    from glossary import Glossary
    g = Glossary(json.load(open(Path(__file__).with_name("glossary_train.json"), encoding="utf-8"))["terms"])
    return g.prompt("Haan toh dekhiye, hum log meeting mein project ke baare mein baat kar rahe hain.", limit=60)


def _swift_padn(n, beam=1, path="models/hinglish-swift-ct2", lang="hi", roman=True, threads=4, prompt=None):
    e = Engine(path, threads=threads, pad=n, roman=roman)
    return (lambda a: (e.transcribe(a, lang, prompt, beam=beam) or {}).get("text", "")), None


_PROMPT = "Hinglish conversation: Haan bhai, kya scene hai? Code check kar, compile kar, meeting scheduled hai, call me back bro, alright."
SYSTEMS = dict(turbo_en_prompt=lambda: _swift_padn(True, path="turbo", lang="en", roman=True, threads=6, prompt=_PROMPT),
               turbo_hi_prompt=lambda: _swift_padn(True, path="turbo", lang="hi", roman=True, threads=6, prompt=_PROMPT),
               turbo_hi_plain=lambda: _swift_padn(True, path="turbo", lang="hi", roman=False, threads=6),
               current=sys_current, turbo_hi=sys_turbo_hi, swift=sys_swift, apex=sys_apex, indic=sys_indic, hybrid=sys_hybrid,
               turbo_pad=lambda: _padded("turbo"), apex_pad=lambda: _padded("models/hinglish-apex-ct2"),
               swift_pad=lambda: _padded("models/hinglish-swift-ct2", threads=4), swift_engine_pad=sys_swift_engine_pad,
               swift_gloss=lambda: _swift_padn(15, prompt=_gloss_prompt()),
               swift_beam3=lambda: _swift_padn(15, beam=3), swift_beam5=lambda: _swift_padn(15, beam=5),
               hindi_small=lambda: _swift_padn(True, path="models/hindi-small-ct2", roman=False),
               hindi_medium=lambda: _swift_padn(True, path="models/hindi-medium-ct2", roman=False, threads=5),
               swift_style=lambda: _swift_padn(15, prompt="Haan toh dekhiye, hum log meeting mein project ke baare mein baat kar rahe hain, aur client ko delivery bhejni hai."),
               **{f"swift_pad{n}": (lambda n=n: _swift_padn(n)) for n in (10, 15, 20)})


def main():
    clips = load_audio()
    total_audio = sum(len(a) for _, a in clips) / SR
    print(f"{len(clips)} clips, {total_audio / 60:.1f} min of audio", flush=True)
    results = json.load(open(Path(__file__).with_name("results.json"), encoding="utf-8")) if Path(__file__).with_name("results.json").exists() else {}
    for name in sys.argv[1:]:
        t = time.time()
        run, _ = SYSTEMS[name]()
        print(f"[{name}] loaded in {time.time() - t:.0f}s", flush=True)
        run(clips[0][1][: 3 * SR])                                   # warm-up
        hyps, t_proc = [], 0.0
        for m, a in clips:
            t0 = time.perf_counter()
            try:
                h = run(a)
            except Exception as e:
                h = ""
                print("  error", m["id"], e, flush=True)
            t_proc += time.perf_counter() - t0
            hyps.append(h)
        row = dict(rtf=t_proc / total_audio, hyps=hyps)
        for src in ("mucs", "ivoices", "all"):
            idx = [i for i, (m, _) in enumerate(clips) if src == "all" or m["src"] == src]
            we = sum(edits(norm(clips[i][0]["ref"]), norm(hyps[i])) for i in idx)
            wn = sum(len(norm(clips[i][0]["ref"])) for i in idx)
            ce = sum(edits("".join(norm(clips[i][0]["ref"])), "".join(norm(hyps[i]))) for i in idx)
            cn = sum(len("".join(norm(clips[i][0]["ref"]))) for i in idx)
            row[src] = dict(wer=round(100 * we / wn, 1), cer=round(100 * ce / cn, 1))
        results[name + os.environ.get("BENCH_SUFFIX", "")] = row
        print(f"[{name}{os.environ.get('BENCH_SUFFIX', '')}] RTF {row['rtf']:.2f} | lecture WER {row['mucs']['wer']} CER {row['mucs']['cer']} | "
              f"conversation WER {row['ivoices']['wer']} CER {row['ivoices']['cer']} | ALL WER {row['all']['wer']} CER {row['all']['cer']}", flush=True)
        tag = os.environ.get("BENCH_META", "meta.json").replace("meta", "").replace(".json", "") or "_x1.0"
        outd = Path(__file__).with_name("results"); outd.mkdir(exist_ok=True)
        key = name + os.environ.get("BENCH_SUFFIX", "")
        json.dump(row, open(outd / f"{key}{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        del run
        gc.collect()


if __name__ == "__main__":
    main()
