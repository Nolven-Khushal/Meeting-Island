"""
After "Save": the background finishing of a recorded session.

  1. The live transcript is already saved (so nothing waits on this).
  2. polish.Polisher listens to the recording again and produces the final transcript (progress is reported to the island).
  3. The final text replaces the live text in the saved file (unless somebody edited that file in the meantime: then it is
     saved next to it as "<name> (final).txt"), speaker names typed in the Save window are carried over.
  4. The recording (screen + audio) is written next to the transcript as "<name>.mp4" if "keep the recording" was ticked.
  5. The working files are removed.

A job survives closing the app: it is stored in sessions/_polish/<id>/job.json and resumed the next time the app starts.
While a new meeting is being recorded the job waits (the live transcript needs all the CPU).
"""
import ctypes
import json
import queue
import shutil
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import assistant
import backend as bk
import corrections
import recorder
import screennames
from polish import Polisher

POLISH_DIR = assistant.SESSIONS / "_polish"


def _priority(low):
    """The final pass runs at low priority so other programs stay snappy; live meetings get the CPU back."""
    try:
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x4000 if low else 0x8000)     # BELOW_NORMAL / ABOVE_NORMAL
    except Exception:
        pass


def _power():
    """-> (on battery?, battery percent)"""
    class SPS(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("pct", ctypes.c_ubyte), ("sys", ctypes.c_ubyte),
                    ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
    try:
        s = SPS()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            return s.ac == 0, (s.pct if s.pct <= 100 else 100)
    except Exception:
        pass
    return False, 100


def map_names(final_segs, live_segs, live_names):
    """Names were typed against the LIVE speaker numbers; the final pass may number people differently. Each final speaker takes
    the name of the live speaker it overlaps in time the most."""
    out = {}
    if not live_names:
        return out
    overlap = {}
    for g in final_segs:
        v = g.get("voice")
        if not v:
            continue
        for l in live_segs:
            lv = l.get("voice")
            if not lv or lv not in live_names:
                continue
            o = min(g["end"], l["end"]) - max(g["start"], l["start"])
            if o > 0:
                overlap.setdefault(v, {}).setdefault(lv, 0.0)
                overlap[v][lv] += o
    for v, d in overlap.items():
        best = max(d, key=d.get)
        out[v] = live_names[best]
    return out


class Finisher(threading.Thread):
    def __init__(self, backend, post):
        super().__init__(daemon=True)
        self.backend, self.post = backend, post
        self.q = queue.Queue()
        self.cancel = threading.Event()
        self.active = None                      # the job being worked on (dict) - for the island's progress line
        self._last_note, self._frac = "", 0.0

    # ------------------------------------------------------------------ jobs
    def submit(self, sess, txt_path, names, keep, roman):
        """Take over the session folder (recording + live transcript) and queue it."""
        POLISH_DIR.mkdir(parents=True, exist_ok=True)
        dest = POLISH_DIR / sess.id
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        try:
            sess.dir.rename(dest)
        except OSError:
            shutil.copytree(sess.dir, dest)
            shutil.rmtree(sess.dir, ignore_errors=True)
        try:
            (dest / "open").unlink()
        except OSError:
            pass
        job = dict(id=sess.id, name=Path(txt_path).stem, txt=str(txt_path), folder=str(Path(txt_path).parent),
                   txt_mtime=Path(txt_path).stat().st_mtime, names={str(k): v for k, v in (names or {}).items()},
                   keep=bool(keep), roman=bool(roman), started=sess.started.isoformat(), duration=sess.duration,
                   mic_is_room=bool(self.backend.cfg.get("source") == "mic"), status="queued")
        self._save(job)
        self.q.put(job["id"])
        return job

    def resume_pending(self):
        if not POLISH_DIR.exists():
            return
        for d in sorted(POLISH_DIR.iterdir()):
            if (d / "job.json").exists():
                try:
                    if json.loads((d / "job.json").read_text(encoding="utf-8")).get("status") != "done":
                        self.q.put(d.name)
                except Exception:
                    pass

    def pending(self):
        return self.q.qsize() + (1 if self.active else 0)

    def _save(self, job):
        tmp = POLISH_DIR / job["id"] / "job.json.tmp"
        tmp.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        tmp.replace(POLISH_DIR / job["id"] / "job.json")

    # ------------------------------------------------------------------ worker
    def run(self):
        while True:
            jid = self.q.get()
            if jid is None:
                return
            d = POLISH_DIR / jid
            try:
                job = json.loads((d / "job.json").read_text(encoding="utf-8"))
            except Exception:
                continue
            try:
                self.active = job
                self._process(job, d)
            except Exception:
                traceback.print_exc()
                self.post("polish", dict(id=jid, state="failed", frac=0, msg="Final pass skipped - the live transcript was kept",
                                         name=job.get("name"), path=job.get("txt")))
            finally:
                self.active = None
                _priority(False)

    def _wait_models(self):
        while not self.cancel.is_set():
            a = self.backend.asr
            if a is not None and a.hindi_ready():
                return a
            time.sleep(1)
        raise InterruptedError

    def _note(self, msg):
        if msg != self._last_note and self.active:
            self._last_note = msg
            self.post("polish", dict(id=self.active["id"], name=self.active["name"], path=self.active["txt"], state="working",
                                     frac=self._frac, msg=msg))

    def _idle(self):
        """The final pass is polite: it waits while a meeting is being recorded (the live transcript needs all the CPU) and
        while the laptop is running low on battery (about 20-40 minutes of work for a 2-hour meeting); the rest of the time
        it runs at low priority."""
        while not self.cancel.is_set():
            if self.backend.state != "idle":
                _priority(False)
                self._note("Waiting for the meeting to end…")
            else:
                battery, pct = _power()
                if battery and pct < 50:
                    _priority(False)
                    self._note("Waiting for the charger…")
                else:
                    _priority(True)
                    self._note("Listening again…")
                    return True
            time.sleep(2)
        return False

    def _process(self, job, d):
        asr = self._wait_models()
        info = dict(id=job["id"], name=job["name"], path=job["txt"], state="working", frac=0.0, msg="Listening again…")
        self.post("polish", dict(info))
        wavs, video = [p for p in (d / "audio_system.wav", d / "audio_mic.wav") if p.exists() and p.stat().st_size > 1000], None
        v = d / "video_only.mp4"
        if v.exists() and v.stat().st_size > 2048:
            video = v
        live = json.loads((d / "transcript.json").read_text(encoding="utf-8")) if (d / "transcript.json").exists() else []
        last = [0.0]

        def progress(frac, msg=""):
            if frac - last[0] >= 0.005:
                last[0] = frac
                self._frac = min(0.97, frac * 0.93)
                self._last_note = msg or self._last_note
                self.post("polish", dict(info, frac=self._frac, msg=msg or info["msg"]))

        final = None
        if wavs:
            job["status"] = "polishing"
            self._save(job)
            lanes = {("System" if "system" in p.name else "Mic"): p for p in wavs}
            final = Polisher(asr, wait_ok=self._idle, progress=progress, cache=d).run(lanes, mic_is_room=job.get("mic_is_room", False))
            live_words = sum(len((g.get("text") or "").split()) for g in live)
            final_words = sum(len((g.get("text") or "").split()) for g in final or [])
            if final and live_words >= 20 and final_words < 0.6 * live_words:
                print(f"[finisher] final pass found far fewer words ({final_words} vs {live_words}); keeping the live text", flush=True)
                final = None
        txt = Path(job["txt"])
        if final:
            names = map_names(final, live, {int(k): v for k, v in job["names"].items()})
            try:                                              # names read from the speaking tile on the screen; typed names win
                seen = screennames.load(d / "names.jsonl")
                auto = screennames.assign(final, seen)
                print(f"[finisher] screen names: {len(seen)} readings -> {auto}", flush=True)
                names = {**auto, **names}
                job["auto_named"] = len(auto)
            except Exception:
                traceback.print_exc()
            started = datetime.fromisoformat(job["started"])
            body = bk.render_text(final, job["name"], started, job["duration"], job["roman"], names,
                                  note="Version: final (checked against the recording)")
            target = txt
            try:
                if not txt.exists() or abs(txt.stat().st_mtime - job["txt_mtime"]) > 1:
                    target = bk.unique_path(txt.parent, job["name"] + " (final)")     # somebody changed the file: do not overwrite it
            except OSError:
                target = bk.unique_path(txt.parent, job["name"] + " (final)")
            target.write_text(body, encoding="utf-8-sig")
            corrections.snapshot(target, body)                # edits to the final version are what the app learns from
            job["final_txt"] = str(target)
        self.post("polish", dict(info, frac=0.95, msg="Saving the recording…"))
        out = None
        if job["keep"] and wavs:
            ext = ".mp4" if video else ".m4a"
            out = bk.unique_path(job["folder"], job["name"], ext)
            try:
                recorder.build_recording(video, wavs, out)
            except Exception:
                traceback.print_exc()
                out = None
        job["status"] = "done"
        job["recording"] = str(out) if out else None
        self._save(job)
        shutil.rmtree(d, ignore_errors=True)                  # the working files are not needed any more
        self.post("polish", dict(info, state="done", frac=1.0, msg="Final transcript ready", path=job.get("final_txt", job["txt"]),
                                 recording=job["recording"], improved=bool(final)))

    def shutdown(self):
        self.cancel.set()
        self.q.put(None)
