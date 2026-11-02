"""
Session logic for Meeting Island (no UI code in here).

  Backend.load()          load the speech models in the background        -> post("models", ready)
  Backend.start_session() open the audio and begin transcribing           -> post("seg" | "draft" | "clear", ...)
  Backend.pause()/resume()   hold / continue the SAME session (paused time is not counted in timestamps)
  Backend.end_session()   stop, finish the last sentences                 -> post("ended", Session)

While a session runs its transcript is autosaved to sessions/_unsaved/<id>/ so a crash or an accidental close never
loses text; it is only removed once you save or discard it.
"""
import json
import os
import re
import shutil
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

import assistant
import live
import recorder
import screennames

UNSAVED = assistant.SESSIONS / "_unsaved"
SR = assistant.SR
LANG_NAMES = {"en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
              "mr": "Marathi", "bn": "Bengali", "gu": "Gujarati", "pa": "Punjabi", "ur": "Urdu"}


class Session:
    def __init__(self, sid=None, segs=None):
        self.id = sid or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir = UNSAVED / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "open").write_text("1")
        self.started = datetime.strptime(self.id, "%Y-%m-%d_%H-%M-%S")
        self.segs = segs or []
        self.duration = 0.0
        self.lock = threading.Lock()

    def add(self, segs):
        with self.lock:
            for g in segs:                                    # a merged sentence replaces the piece it grew from
                self.segs = [x for x in self.segs if not (g.get("id") and x.get("id") == g["id"])]
            self.segs = sorted(self.segs + segs, key=lambda s: s["start"])
            self.save(force=not segs)

    def save(self, force=False):
        """Autosave at most every 4 s (the whole transcript is rewritten each time; for a 2-hour meeting that is megabytes)."""
        now = time.monotonic()
        if not force and now - getattr(self, "_saved_at", 0.0) < 4.0:
            self._dirty = True
            return
        self._saved_at, self._dirty = now, False
        tmp = self.dir / "transcript.json.tmp"
        tmp.write_text(json.dumps(self.segs, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.dir / "transcript.json")

    def refine_speakers(self):
        """End of session: re-cluster all sentences by voice together (fixes early mistakes) and renumber speakers."""
        from speakers import refine
        idx = [i for i, g in enumerate(self.segs) if g.get("emb")]
        if len(idx) >= 2:
            labels = refine([self.segs[i]["emb"] for i in idx])
            for i, lab in zip(idx, labels):
                self.segs[i]["voice"] = lab
            # sentences too short to have a voice keep following the speaker before them
            prev = {}
            for g in self.segs:
                lane = g.get("speaker")
                if g.get("emb"):
                    prev[lane] = g.get("voice")
                elif lane in prev:
                    g["voice"] = prev[lane]
            self.add([])
            return len({g["voice"] for g in self.segs if g.get("voice")})
        return len({g["voice"] for g in self.segs if g.get("voice")})

    def voices(self):
        return sorted({g["voice"] for g in self.segs if g.get("voice")})

    def recording(self):
        """-> (audio lane files, screen video or None) kept for this session; ([], None) when it was not recorded."""
        wavs = [p for p in (self.dir / "audio_system.wav", self.dir / "audio_mic.wav") if p.exists() and p.stat().st_size > 1000]
        video = self.dir / "video_only.mp4"
        return wavs, (video if video.exists() and video.stat().st_size > 2048 else None)

    def default_name(self):
        return "Meeting " + self.started.strftime("%d %b, %I:%M %p").replace(" 0", " ").lstrip("0")

    def langs_used(self):
        seen = []
        for s in self.segs:
            n = LANG_NAMES.get(s.get("lang"), s.get("lang"))
            if n and n not in seen:
                seen.append(n)
        return seen

    def discard(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def find_unsaved(exclude=None):
    out = []
    if UNSAVED.exists():
        for d in sorted(UNSAVED.iterdir()):
            if d.is_dir() and (d / "open").exists() and (d / "transcript.json").exists() and (not exclude or d.name != exclude):
                out.append(d.name)
    return out


def load_unsaved(sid):
    s = Session(sid)
    s.segs = json.loads((s.dir / "transcript.json").read_text(encoding="utf-8"))
    s.duration = max([x["end"] for x in s.segs] or [0])
    return s


def safe_filename(name):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name or "Meeting"


def speaker_label(seg, segs, names=None):
    """Who said this line? "You" for the microphone; for other people "Speaker 2" (or the name typed in the Save window).
    Labels appear only when there is something to tell apart (two or more voices, or you plus the others)."""
    voices = {g.get("voice") for g in segs if g.get("voice")}
    has_you = any(g.get("speaker") == "Mic" and not g.get("voice") for g in segs)   # the microphone, when it is just you
    v = seg.get("voice")
    if seg.get("speaker") == "Mic" and not v:
        return "You"
    if v and len(voices) >= 2:
        return (names or {}).get(v) or f"Speaker {v}"
    if has_you:
        return (names or {}).get(v) or "Others"
    return ""


def render_text(segs, name, started, duration, roman=True, names=None, note=None):
    import translit

    def disp(t):
        return translit.to_roman(t) if roman and translit.has_devanagari(t) else t

    langs = []
    for s in segs:
        n = LANG_NAMES.get(s.get("lang"), s.get("lang"))
        if n and n not in langs:
            langs.append(n)
    voices = sorted({g["voice"] for g in segs if g.get("voice")})
    head = [f"{name}", f"Date: {started.strftime('%d %b %Y, %I:%M %p')}", f"Duration: {assistant.fmt_ts(duration)}",
            "Languages: " + (", ".join(langs) + (" (Hindi shown as Hinglish)" if roman and "Hindi" in langs else "") if langs else "-")]
    if len(voices) >= 2:
        head.append("Speakers: " + ", ".join((names or {}).get(v) or f"Speaker {v}" for v in voices))
    if note:
        head.append(note)
    head += ["", ""]
    lines = []
    for s in segs:
        words = s.get("words") or []
        if not roman and s.get("native"):
            text = s["native"].strip()
        else:
            text = "".join(disp(w) for w, _ in words).strip() if words else disp(s["text"]).strip()
        who = speaker_label(s, segs, names)
        lines.append(f"[{assistant.fmt_ts(s['start'])}] " + (f"{who}: " if who else "") + text)
    return chr(10).join(head + lines) + chr(10)


def unique_path(folder, name, ext=".txt"):
    p = Path(folder) / (safe_filename(name) + ext)
    n = 2
    while p.exists():
        p = Path(folder) / f"{safe_filename(name)} ({n}){ext}"
        n += 1
    return p


def _norm_w(w):
    return re.sub(r"[^\w]", "", w.lower())


def _similar(a, b):
    """Do two words sound alike? (skeleton spelling, so kaam/kam and paithan/paaithan match)"""
    import fusion
    x, y = fusion._skel(re.sub(r"[^\w]", "", a)), fusion._skel(re.sub(r"[^\w]", "", b))
    if not x or not y:
        return False
    return fusion._lev(x, y) / max(len(x), len(y)) <= 0.34


def overlap_len(pw, nw, longest=9):
    """How many words at the end of `pw` are the same words as the start of `nw`? Fuzzy: the two chunks hear the shared
    second or so of audio twice and may spell it a little differently."""
    for n in range(min(longest, len(pw), len(nw)), 0, -1):
        hits = sum(_similar(x, y) for x, y in zip(pw[-n:], nw[:n]))
        if (n <= 2 and hits == n) or (n >= 3 and hits >= n - max(1, n // 4)):
            return n
    return 0


def stitch(prev, seg):
    """
    A sentence the speaker never paused in is cut into pieces (every ~14 s) and read with a short overlap.
    Join the new piece onto the previous one so it stays ONE line: drop the words both pieces contain, concatenate the rest.
    """
    pw, nw = prev["text"].split(), seg["text"].split()
    k = overlap_len(pw, nw)
    out = dict(prev)
    out["text"] = " ".join(pw + nw[k:])
    out["end"] = seg["end"]
    out["forced"] = seg.get("forced", False)
    out["lang"] = seg.get("lang") or prev.get("lang")
    pwl, nwl = prev.get("words") or [], seg.get("words") or []
    out["words"] = (pwl + nwl[k:]) if (pwl and nwl) else []
    if prev.get("native") and seg.get("native"):
        a, b = prev["native"].split(), seg["native"].split()
        out["native"] = " ".join(a + b[overlap_len(a, b):])      # (Devanagari: compared by transliteration)
    else:
        out["native"] = None
    out["conf"] = min(prev.get("conf", 1), seg.get("conf", 1))
    return out


def trim_head(prev, seg):
    """Drop the words at the start of `seg` that `prev` already ended with (the re-read overlap)."""
    pw, nw = prev["text"].split(), seg["text"].split()
    k = overlap_len(pw, nw)
    if not k:
        return seg
    out = dict(seg)
    out["text"] = " ".join(nw[k:])
    wl = seg.get("words") or []
    out["words"] = wl[k:] if len(wl) == len(nw) else []
    if seg.get("native") and prev.get("native"):
        a, b = prev["native"].split(), seg["native"].split()
        out["native"] = " ".join(b[overlap_len(a, b):])
    return out


class Router:
    """What LiveASR writes finals into: whichever session is current. Also joins forced-cut pieces into one line."""

    MAX_LINE = 28.0     # seconds: past this, start a new line even if the speaker still has not paused

    def __init__(self, backend):
        self.b = backend

    def add(self, segs):
        s = self.b.session
        if s is None:
            return
        out = []
        for g in segs:
            prev = next((x for x in reversed(s.segs) if x["speaker"] == g["speaker"]), None)
            if prev and prev.get("forced") and g["start"] - prev["end"] < 1.5 and prev.get("voice") == g.get("voice"):
                if g["end"] - prev["start"] < self.MAX_LINE:
                    g = stitch(prev, g)
                else:                                          # line is long enough: start a new one, minus the shared words
                    g = trim_head(prev, g)
            out.append(g)
        s.add(out)
        for g in out:
            self.b.post("seg", g)


class Backend:
    def __init__(self, post, cfg):
        self.post, self.cfg = post, cfg
        self.asr = None
        self.session = None
        self.lanes, self.streams, self.pa = [], [], None
        self.T0 = 0.0
        self.paused_at = None
        self.sim_stop = threading.Event()
        self.state = "idle"
        self.rec_mode, self.screen, self.rec_note = "off", None, None

    # ------------------------------------------------------------------ models
    def load(self):
        def run():
            try:
                langs = assistant.parse_langs(self.cfg["langs"]) or ["en", "hi"]
                self.asr = live.LiveASR(Router(self), self._on_event, langs, self.cfg["model"], self.cfg["hi_model"],
                                        self.cfg["hotwords"], self.cfg["threads"], self.cfg["hi_threads"],
                                        engine=self.cfg.get("engine", "hinglish"), speakers=self.cfg.get("speakers", True))
                self.asr.start()
                self.post("models", True)
            except Exception as e:
                traceback.print_exc()
                self.post("error", f"{type(e).__name__}: {e}"[:90])
        threading.Thread(target=run, daemon=True).start()

    def hindi_ready(self):
        return self.asr is None or self.asr.hindi_ready() or "," not in self.cfg["langs"]

    def _on_event(self, kind, lane, utt, payload):
        if self.state in ("recording", "paused", "ending"):
            self.post(kind, lane, utt, payload)

    # ------------------------------------------------------------------ session control
    def start_session(self):
        if self.asr is None or self.state not in ("idle",):
            return False
        self.session = Session()
        if self.asr is not None:
            self.asr.mic_is_room = self.cfg.get("source", "pc") == "mic"   # in a room, the one microphone hears everybody
            if self.asr.speakers is not None:
                self.asr.speakers.reset()                   # every session starts with no known voices
        for lane_attr in ():
            pass
        try:
            from glossary import Glossary
            self.asr.glossary = Glossary()            # re-read glossary.txt: edits take effect on the next Start
        except Exception:
            traceback.print_exc()
        self.T0 = time.monotonic() + 0.4
        self.paused_at = None
        self.sim_stop.clear()
        rec_mode = self.cfg.get("record", "screen")           # screen = video + audio, audio = audio only, off
        self.rec_note = None
        if rec_mode != "off" and recorder.free_gb(self.session.dir) < 2.0:
            rec_mode = "off"                                  # never let a recording fill the disk the meeting needs
            self.rec_note = "Low disk space: this session is not being recorded"
        self.rec_mode, self.screen = rec_mode, None
        factory = lambda label, wav, jobq, T0_, rate, ch: self.asr.make_lane(label, wav, T0_, rate, ch, self.cfg["window"])
        try:
            if self.cfg.get("sim"):
                self._start_sim(factory)
            else:
                src = self.cfg.get("source", "pc")            # pc = PC audio, mic = room microphone, both = PC audio + "You"
                self.pa, self.lanes, self.streams = assistant.open_streams(
                    None, self.T0, None, self.session.dir, src in ("mic", "both"), save_audio=rec_mode != "off",
                    lane_factory=factory, use_loopback=src in ("pc", "both"))
        except Exception as e:
            traceback.print_exc()
            self.session.discard()
            self.session = None
            self.post("error", f"Audio: {type(e).__name__}: {e}"[:90])
            return False
        self.state = "recording"
        if rec_mode == "screen":
            self.watcher = None
            if self.cfg.get("screen_names", True):            # who has the speaking ring around their tile, and what is the name on it
                try:
                    w = screennames.Watcher(self.session.dir / "names.jsonl")
                    self.watcher = w if w.ok else None
                except Exception:
                    traceback.print_exc()
            self.screen = recorder.ScreenRecorder(self.session.dir / "video_only.mp4",
                                                  lambda: (self.elapsed(), self.state == "paused"), watcher=self.watcher)
            self.screen.start()
        return True

    def elapsed(self):
        if self.state == "idle" or not self.T0:
            return 0.0
        return max(0.0, (self.paused_at or time.monotonic()) - self.T0)

    def pause(self):
        if self.state != "recording":
            return
        self.paused_at = time.monotonic()
        for ln in self.lanes:
            ln.paused = True
        self.state = "paused"

    def resume(self):
        if self.state != "paused":
            return
        gap = time.monotonic() - self.paused_at
        self.T0 += gap            # the timeline carries on from where it stopped: paused time is not counted
        for ln in self.lanes:
            ln.T0 += gap
        self.paused_at = None
        for ln in self.lanes:
            ln.paused = False
        self.state = "recording"

    def levels(self):
        if not self.lanes:
            return 0.0, 0.0
        return max(l.level for l in self.lanes), max(l.last_signal for l in self.lanes)

    def end_session(self):
        if self.state not in ("recording", "paused"):
            return
        self.state = "ending"

        def run():
            try:
                self.sim_stop.set()
                if self.screen is not None:
                    self.screen.stop()
                if getattr(self, "watcher", None) is not None:
                    self.watcher.stop()
                for s in self.streams:
                    try:
                        s.stop_stream(); s.close()
                    except Exception:
                        pass
                if self.pa:
                    self.pa.terminate()
                for ln in self.lanes:
                    ln.paused = False
                    ln.closing.set()
                for ln in self.lanes:
                    ln.join(timeout=20)
                if self.asr:
                    self.asr.wait_idle(90)
                sess = self.session
                with sess.lock:
                    sess.save(force=True)                     # flush what the throttled autosave was still holding
                sess.duration = max(self.elapsed(), max([s["end"] for s in sess.segs] or [0]))
                try:
                    sess.refine_speakers()
                except Exception:
                    traceback.print_exc()
                self.lanes, self.streams, self.pa = [], [], None
                self.session = None
                self.state = "idle"
                self.post("ended", sess)
            except Exception as e:
                traceback.print_exc()
                self.state = "idle"
                self.post("error", f"{type(e).__name__}: {e}"[:90])
        threading.Thread(target=run, daemon=True).start()

    def shutdown(self):
        try:
            if self.asr:
                self.asr.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------ simulated audio (testing)
    def _start_sim(self, factory):
        audio = assistant.decode_audio(self.cfg["sim"])
        x48 = np.interp(np.arange(len(audio) * 3) / 3, np.arange(len(audio)), audio)
        pcm = np.repeat((np.clip(x48, -1, 1) * 32767).astype(np.int16)[:, None], 2, axis=1).reshape(-1)
        wav = self.session.dir / "audio_system.wav" if self.rec_mode != "off" else None
        lane = factory("System", wav, None, self.T0, 48000, 2)
        lane.start()
        self.lanes = [lane]

        def feed():
            pos, total, pause_t = 0, len(pcm) // 2, None
            T0 = self.T0
            while pos < total and not self.sim_stop.is_set():
                if self.state == "paused":
                    time.sleep(0.05)
                    continue
                T0 = self.T0                         # tracks the shifts applied on resume
                time.sleep(max(0, T0 + pos / 48000 - time.monotonic()))
                if self.state == "paused":
                    continue
                lane.push(pcm[pos * 2:(pos + 2048) * 2].tobytes())
                pos += 2048
        threading.Thread(target=feed, daemon=True).start()
