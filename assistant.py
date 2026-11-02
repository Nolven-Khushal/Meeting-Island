#!/usr/bin/env python
"""
Offline meeting assistant (demo). No cloud, no API keys.

  record   capture system audio (+ optional mic) and the screen, transcribe live
  file     transcribe an existing audio/video file
  polish   re-transcribe a finished session with a bigger, slower model
  review   open the transcript + video viewer for a session

Everything stays in ./sessions/<timestamp>/ on this machine.
"""
import argparse
import functools
import json
import math
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import wave
import webbrowser
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
from paths import DATA_DIR
SESSIONS = DATA_DIR / "sessions"
SR = 16000          # sample rate Whisper wants
MAX_WINDOW = 12.0   # live: longest chunk (s) before we force a cut
POLISH_WINDOW = 24.0


# --------------------------------------------------------------------------- helpers
def fmt_ts(sec, srt=False):
    sec = max(0.0, float(sec))
    h, m, s = int(sec // 3600), int(sec % 3600 // 60), sec % 60
    if srt:
        return f"{h:02d}:{m:02d}:{int(s):02d},{int((s % 1) * 1000):03d}"
    return f"{h:02d}:{m:02d}:{int(s):02d}"


def atomic_write(path, text):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for _ in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # the browser is reading it right now
            time.sleep(0.05)


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def parse_langs(s):
    if not s or s.strip().lower() == "auto":
        return None
    return [x.strip().lower() for x in s.split(",") if x.strip()]


class Status:
    def __init__(self, path):
        self.path, self.d, self.lock = Path(path), {}, threading.Lock()

    def update(self, **kw):
        with self.lock:
            self.d.update(kw)
            atomic_write(self.path, json.dumps(self.d))


class Store:
    """Thread-safe list of transcript segments, mirrored to transcript.json."""

    def __init__(self, session, quiet=False):
        self.path = session / "transcript.json"
        self.segs, self.lock, self.quiet = [], threading.Lock(), quiet

    def _flush(self):
        atomic_write(self.path, json.dumps(self.segs, ensure_ascii=False))

    def _dedupe(self, s):
        """A forced cut mid-phrase makes the next chunk re-hear the last word(s): drop the repeat."""
        prev = next((p for p in reversed(self.segs) if p["speaker"] == s["speaker"]), None)
        if not prev or s["start"] - prev["end"] > 2.0:
            return s
        norm = lambda w: re.sub(r"\W+", "", w.lower())
        pt, nt = prev["text"].split(), s["text"].split()
        for k in range(min(4, len(pt), len(nt) - 1), 1, -1):
            if [norm(w) for w in pt[-k:]] == [norm(w) for w in nt[:k]]:
                s["text"] = " ".join(nt[k:])
                words = s.get("words") or []
                s["words"] = words[k:] if len(words) == len(nt) else []
                break
        return s

    def add(self, segs):
        if not segs:
            return
        with self.lock:
            kept = []
            for s in segs:
                self.segs.append(s := self._dedupe(s))
                kept.append(s)
            segs = [s for s in kept if s["text"].strip()]
            self.segs = sorted([s for s in self.segs if s["text"].strip()], key=lambda s: s["start"])
            self._flush()
        if not self.quiet:
            for s in segs:
                flag = "  (?)" if s["conf"] < 0.6 else ""
                print(f"[{fmt_ts(s['start'])}] ({s['lang']}) {s['text']}{flag}", flush=True)

    def set(self, segs):
        with self.lock:
            self.segs = sorted(segs, key=lambda s: s["start"])
            self._flush()


# --------------------------------------------------------------------------- transcription
class Transcriber:
    def __init__(self, model_name, langs, hotwords=None, beam=1, threads=6):
        from faster_whisper import WhisperModel
        t = time.time()
        print(f"Loading Whisper '{model_name}' (CPU, int8)...", flush=True)
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=threads)
        print(f"  ready in {time.time() - t:.1f}s", flush=True)
        self.name, self.langs, self.hotwords, self.beam, self.last = model_name, langs, hotwords, beam, None

    def _pick_lang(self, audio):
        if self.langs is None:
            return None
        if len(self.langs) == 1:
            return self.langs[0]
        try:
            _, _, probs = self.model.detect_language(audio)
        except Exception:
            return self.last or self.langs[0]
        p = {l: pr for l, pr in probs if l in self.langs}
        if not p:
            return self.last or self.langs[0]
        best = max(p, key=p.get)
        if p[best] / (sum(p.values()) or 1e-9) < 0.6 and self.last:
            return self.last  # unsure on a short utterance: keep the previous language
        self.last = best
        return best

    def run(self, audio, t0, speaker, words=True):
        if len(audio) < 0.3 * SR:
            return []
        lang = self._pick_lang(audio)
        segs, info = self.model.transcribe(
            audio, language=lang, beam_size=self.beam, temperature=0.0,
            vad_filter=True, vad_parameters=dict(min_silence_duration_ms=400),
            condition_on_previous_text=False, hotwords=self.hotwords, no_speech_threshold=0.6,
            word_timestamps=words)
        out = []
        for s in segs:
            text = s.text.strip()
            if not text:
                continue
            if s.no_speech_prob > 0.7 and s.avg_logprob < -1.0:
                continue  # almost certainly a hallucination on non-speech
            if s.compression_ratio > 2.4:
                continue  # repetition loop
            words = [[w.word, round(float(w.probability), 2)] for w in (s.words or [])]
            conf = sum(p for _, p in words) / len(words) if words else math.exp(s.avg_logprob)
            out.append(dict(start=round(t0 + s.start, 2), end=round(t0 + s.end, 2), text=text,
                            lang=info.language, conf=round(conf, 2), speaker=speaker, words=words))
        return out


def worker(jobq, tr, store):
    while True:
        job = jobq.get()
        if job is None:
            return
        label, off, audio = job
        try:
            # falling behind real time? drop the (costly) word-level confidence until we catch up
            store.add(tr.run(audio, off, label, words=jobq.qsize() < 2))
        except Exception as e:  # keep recording even if one chunk fails
            print(f"  [transcribe error @ {fmt_ts(off)}: {e}]", flush=True)


# --------------------------------------------------------------------------- audio capture
class Lane(threading.Thread):
    """One audio source -> 16 kHz mono -> wav file + speech chunks for the transcriber."""
    FRAME = 320  # 20 ms

    def __init__(self, label, wav_path, jobq, T0, rate, channels):
        super().__init__(daemon=True)
        import av
        self.label, self.jobq, self.T0, self.ch, self.rate = label, jobq, T0, channels, rate
        self.raw, self.closing = queue.Queue(), threading.Event()
        self.rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
        self.wav = None
        if wav_path:
            self.wav = wave.open(str(wav_path), "wb")
            self.wav.setnchannels(1); self.wav.setsampwidth(2); self.wav.setframerate(SR)
        self.level, self.last_signal, self.paused = 0.0, 0.0, False  # for the overlay UI
        self.total = 0
        self.pend, self.pend_start, self._chk = np.zeros(0, np.float32), 0, 0

    def push(self, data):  # called from the PortAudio callback thread
        if self.paused:
            return
        self.raw.put((data, time.monotonic() - self.T0))

    def _to16k(self, data):
        import av
        a = np.frombuffer(data, dtype=np.int16)
        if self.ch > 1:
            a = a[: len(a) // self.ch * self.ch].reshape(-1, self.ch).mean(axis=1).astype(np.int16)
        fr = av.AudioFrame.from_ndarray(a[None, :], format="s16", layout="mono")
        fr.sample_rate = self.rate
        parts = [f.to_ndarray().reshape(-1) for f in self.rs.resample(fr)]
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def _pad_to(self, t_sec, min_gap=0.05):
        # WASAPI loopback delivers nothing while the speakers are silent; fill the gap so
        # transcript timestamps stay aligned with the screen recording.
        n = int(t_sec * SR) - self.total
        if n > min_gap * SR:
            self._append(np.zeros(n, np.float32))

    def _append(self, x):
        self.total += len(x)
        if len(x):
            self.level = float(np.sqrt(np.mean(x * x)))
            if self.level > 0.002:
                self.last_signal = time.monotonic()
        if self.wav:
            self.wav.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
        self.pend = np.concatenate([self.pend, x])
        self._maybe_cut()

    def _emit(self, k):
        self.jobq.put((self.label, self.pend_start / SR, self.pend[:k].copy()))
        self._drop(k)

    def _drop(self, k):
        k = max(0, k)
        self.pend, self.pend_start = self.pend[k:], self.pend_start + k
        self._chk = len(self.pend)

    def _maybe_cut(self, final=False):
        n = len(self.pend)
        if n == 0 or (not final and n - self._chk < SR // 4):
            return
        self._chk = n
        fr = self.pend[: n // self.FRAME * self.FRAME].reshape(-1, self.FRAME)
        if len(fr) == 0:
            return
        rms = np.sqrt((fr ** 2).mean(axis=1))
        voiced = rms > max(0.006, 0.18 * np.percentile(rms, 95))
        speech_s, dur = voiced.sum() * 0.02, n / SR
        if final:
            if speech_s >= 0.3:
                self._emit(n)
            return
        if speech_s < 0.25:
            if dur > 3:
                self._drop(n - int(0.4 * SR))  # nothing said: discard, keep a short lead-in
            return
        if dur >= 1.5 and len(voiced) >= 40 and not voiced[-40:].any():
            self._emit(n)  # 0.8 s pause after speech -> natural sentence boundary
        elif dur >= MAX_WINDOW:
            k = 200  # no pause: cut at the quietest moment in the last 4 s
            sm = np.convolve(rms[-k:], np.ones(5) / 5, "same")
            idx = len(rms) - k + int(np.argmin(sm[2:-2])) + 2
            self._emit(idx * self.FRAME)

    def run(self):
        while True:
            try:
                data, t = self.raw.get(timeout=0.2)
            except queue.Empty:
                if self.closing.is_set():
                    break
                if not self.paused:       # a paused lane stands still: paused time is not part of the timeline
                    self._pad_to(time.monotonic() - self.T0 - 0.35)
                continue
            if t < 0:
                continue
            x = self._to16k(data)
            if len(x):
                self._pad_to(t - len(x) / SR, min_gap=0.25)
                self._append(x)
        self._maybe_cut(final=True)
        if self.wav:
            self.wav.close()


def open_streams(lanes_cfg, T0, jobq, session, use_mic, save_audio=True, lane_factory=None, use_loopback=True):
    import pyaudiowpatch as pyaudio
    pa = pyaudio.PyAudio()
    wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    wavs = (lambda n: session / n) if save_audio else (lambda n: None)
    sources = []
    if use_loopback:                                   # what the PC is playing (calls, videos)
        try:
            spk = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        except Exception:
            raise RuntimeError("No speakers or headphones found on this PC")
        if not spk.get("isLoopbackDevice"):
            for lb in pa.get_loopback_device_info_generator():
                if spk["name"] in lb["name"]:
                    spk = lb
                    break
            else:
                raise RuntimeError("Could not listen to the PC audio (no loopback for the speakers)")
        sources.append(("System", wavs("audio_system.wav"), spk))
    if use_mic:                                        # the microphone
        try:
            mic = pa.get_device_info_by_index(wasapi["defaultInputDevice"])
        except Exception:
            raise RuntimeError("No microphone found on this PC")
        sources.append(("Mic", wavs("audio_mic.wav"), mic))
    if not sources:
        raise RuntimeError("Nothing to listen to")
    lanes, streams = [], []
    for label, wav, dev in sources:
        rate, ch = int(dev["defaultSampleRate"]), int(dev["maxInputChannels"])
        lane = (lane_factory or Lane)(label, wav, jobq, T0, rate, ch)
        lane.start()
        stream = pa.open(format=pyaudio.paInt16, channels=ch, rate=rate, input=True,
                         input_device_index=dev["index"], frames_per_buffer=2048,
                         stream_callback=lambda d, n, ti, st, lane=lane: (lane.push(d), pyaudio.paContinue))
        print(f"  {label}: {dev['name']}  ({rate} Hz, {ch} ch)", flush=True)
        lanes.append(lane); streams.append(stream)
    return pa, lanes, streams


# --------------------------------------------------------------------------- screen capture
class ScreenRecorder(threading.Thread):
    def __init__(self, path, T0, fps=5):
        super().__init__(daemon=True)
        self.path, self.T0, self.fps, self.stop_ev, self.error = path, T0, fps, threading.Event(), None

    def run(self):
        try:
            import mss
            with mss.mss() as sct:
                mon = sct.monitors[1]  # primary monitor
                first = sct.grab(mon)
                cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra",
                       "-s", f"{first.width}x{first.height}", "-r", str(self.fps), "-i", "-",
                       "-vf", "scale=min(1280\\,iw):-2", "-c:v", "libx264", "-preset", "ultrafast",
                       "-crf", "30", "-pix_fmt", "yuv420p", str(self.path)]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                i, last = 0, first.bgra
                while not self.stop_ev.is_set():
                    delay = self.T0 + i / self.fps - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    last = sct.grab(mon).bgra
                    proc.stdin.write(last); i += 1
                    while self.T0 + (i + 1) / self.fps < time.monotonic():  # fell behind: repeat frame
                        proc.stdin.write(last); i += 1
                proc.stdin.close()
                proc.wait(timeout=60)
        except Exception as e:
            self.error = e
            print(f"  [screen recording failed: {e}]", flush=True)


def mix_wavs(paths, out):
    arrs = []
    for p in paths:
        with wave.open(str(p)) as w:
            arrs.append(np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.int32))
    n = max(len(a) for a in arrs)
    mix = np.zeros(n, np.int32)
    for a in arrs:
        mix[: len(a)] += a
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(np.clip(mix, -32768, 32767).astype(np.int16).tobytes())


def finalize_media(session, has_video):
    wavs = [p for p in (session / "audio_system.wav", session / "audio_mic.wav") if p.exists()]
    if len(wavs) > 1:
        mix_wavs(wavs, session / "audio_mix.wav")
        audio = session / "audio_mix.wav"
    else:
        audio = wavs[0]
    video = session / "video_only.mp4"
    if has_video and video.exists() and video.stat().st_size > 0:
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio),
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
                        "-shortest", str(session / "meeting.mp4")], check=True)
        video.unlink()
        return "meeting.mp4"
    return audio.name


# --------------------------------------------------------------------------- polish / file pass
def decode_audio(path):
    """Any audio/video file -> 16 kHz mono float32 (own decoder: faster-whisper's breaks on PyAV 19)."""
    import av
    rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
    parts = []
    with av.open(path) as c:
        stream = next(s for s in c.streams if s.type == "audio")
        for frame in c.decode(stream):
            parts += [f.to_ndarray().reshape(-1) for f in rs.resample(frame)]
    parts += [f.to_ndarray().reshape(-1) for f in rs.resample(None)]
    return np.concatenate(parts)


def read_wav_i16(path):
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def make_windows(a16):
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    chunks = get_speech_timestamps(a16.astype(np.float32) / 32768.0, VadOptions(
        min_silence_duration_ms=600, speech_pad_ms=200, max_speech_duration_s=20, min_speech_duration_ms=250))
    wins, cur = [], None
    for c in chunks:
        if cur and c["end"] - cur[0] <= POLISH_WINDOW * SR and c["start"] - cur[1] <= 1.2 * SR:
            cur[1] = c["end"]
        else:
            if cur:
                wins.append(tuple(cur))
            cur = [c["start"], c["end"]]
    if cur:
        wins.append(tuple(cur))
    pad = int(0.15 * SR)
    return [(max(0, a - pad), min(len(a16), b + pad)) for a, b in wins]


def run_pass(session, model, langs, hotwords, threads, store, status, live_segs):
    """Re-transcribe every lane of a session with `model`; swaps results in progressively."""
    tr = Transcriber(model, langs, hotwords, beam=5, threads=threads)
    lanes = [(lbl, session / f"audio_{lbl.lower()}.wav") for lbl in ("System", "Mic")]
    lanes = [(lbl, p, read_wav_i16(p)) for lbl, p in lanes if p.exists()]
    plan = [(lbl, a16, make_windows(a16)) for lbl, _, a16 in lanes]
    total, done, new, covered = sum(len(w) for _, _, w in plan), 0, [], {}
    status.update(state="polishing", polish=dict(done=0, total=total, model=model))
    t0 = time.time()
    for lbl, a16, wins in plan:
        for a, b in wins:
            new += tr.run(a16[a:b].astype(np.float32) / 32768.0, a / SR, lbl)
            covered[lbl] = b / SR
            done += 1
            store.set(new + [s for s in live_segs if s["start"] >= covered.get(s["speaker"], 0)])
            status.update(polish=dict(done=done, total=total, model=model))
            print(f"  polish {done}/{total}  ({time.time() - t0:.0f}s)", flush=True)
        covered[lbl] = 1e9
    store.set(new)
    return new


def export_text(session, segs):
    names = {s["speaker"] for s in segs}
    lines, srt = [], []
    for i, s in enumerate(segs, 1):
        who = f"{s['speaker']}: " if len(names) > 1 else ""
        lines.append(f"[{fmt_ts(s['start'])}] {who}{s['text']}")
        srt.append(f"{i}\n{fmt_ts(s['start'], True)} --> {fmt_ts(s['end'], True)}\n{who}{s['text']}\n")
    (session / "transcript.txt").write_text("\n".join(lines), encoding="utf-8")
    (session / "transcript.srt").write_text("\n".join(srt), encoding="utf-8")


# --------------------------------------------------------------------------- web viewer
def make_handler(directory, stop_event):
    class H(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def end_headers(self):
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def do_POST(self):
            if self.path == "/stop" and self.headers.get("X-Assistant") == "1" and stop_event is not None:
                stop_event.set()
                self.send_response(204); self.end_headers()
            else:
                self.send_error(403)

        def do_GET(self):
            rng = self.headers.get("Range")
            path = self.translate_path(self.path.split("?")[0])
            m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
            if not (m and os.path.isfile(path)):
                return super().do_GET()
            size = os.path.getsize(path)
            start = int(m.group(1)) if m.group(1) else 0
            end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
            if start > end:
                return self.send_error(416)
            try:
                self.send_response(206)
                self.send_header("Content-Type", self.guess_type(path))
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(end - start + 1))
                self.end_headers()
                with open(path, "rb") as f:
                    f.seek(start)
                    left = end - start + 1
                    while left > 0:
                        chunk = f.read(min(1 << 16, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk); left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    return functools.partial(H, directory=str(directory))


def serve(session, stop_event=None, port=8765, open_browser=True):
    shutil.copy(HERE / "viewer.html", session / "viewer.html")
    for p in range(port, port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), make_handler(session, stop_event))
            break
        except OSError:
            continue
    else:
        raise RuntimeError("no free port")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{p}/viewer.html"
    print(f"Viewer: {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    return url


# --------------------------------------------------------------------------- commands
def new_session():
    SESSIONS.mkdir(exist_ok=True)
    d = SESSIONS / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    d.mkdir()
    return d


def cmd_record(a):
    session = new_session()
    status, store = Status(session / "status.json"), Store(session)
    langs, jobq = parse_langs(a.langs), queue.Queue()
    status.update(state="loading", model=a.model, langs=langs, title=a.title or session.name)
    tr = Transcriber(a.model, langs, a.hotwords, beam=1, threads=a.threads)
    stop_ev = threading.Event()
    serve(session, stop_ev, a.port, not a.no_browser)

    T0 = time.monotonic() + 1.5
    print("Starting capture...", flush=True)
    pa, lanes, streams = open_streams(None, T0, jobq, session, a.mic)
    screen = None
    if not a.no_video:
        screen = ScreenRecorder(session / "video_only.mp4", T0, a.fps)
        screen.start()
    wt = threading.Thread(target=worker, args=(jobq, tr, store), daemon=True)
    wt.start()
    wall0 = time.time() + 1.5
    status.update(state="recording", started=wall0, backlog=0)
    print("\n● RECORDING — play the meeting / video now. Press Ctrl+C here (or the Stop button in the viewer) to finish.\n", flush=True)
    try:
        while not stop_ev.wait(1.0):
            status.update(duration=round(max(0, time.time() - wall0), 1), backlog=jobq.qsize())
    except KeyboardInterrupt:
        pass

    print("\nStopping...", flush=True)
    status.update(state="finishing", backlog=jobq.qsize())
    for s in streams:
        s.stop_stream(); s.close()
    pa.terminate()
    for ln in lanes:
        ln.closing.set()
    for ln in lanes:
        ln.join()
    if jobq.qsize():
        print(f"Finishing {jobq.qsize()} queued chunk(s)...", flush=True)
    jobq.put(None)
    wt.join()
    if screen:
        screen.stop_ev.set(); screen.join(timeout=90)
    media = finalize_media(session, screen is not None and screen.error is None)
    live = list(store.segs)
    export_text(session, live)
    status.update(state="done", media=media, duration=round(time.time() - wall0, 1), backlog=0)
    print(f"Saved: {session}", flush=True)

    if a.polish:
        del tr
        shutil.copy(session / "transcript.json", session / "transcript_live.json")
        final = run_pass(session, a.polish, langs, a.hotwords, a.threads, store, status, live)
        export_text(session, final)
        status.update(state="done", polish=dict(done=1, total=1, model=a.polish, finished=True))
    if a.once:
        return
    print(f"\nReview any time: {HERE / 'review.bat'}   |   Ctrl+C to close the viewer now", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def cmd_file(a):
    src = Path(a.path).resolve()
    if not src.exists():
        sys.exit(f"Not found: {src}")
    session = new_session()
    status, store = Status(session / "status.json"), Store(session)
    print("Decoding audio...", flush=True)
    audio = decode_audio(str(src))
    with wave.open(str(session / "audio_system.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    media = "meeting" + src.suffix.lower()
    shutil.copy(src, session / media)
    status.update(state="polishing", media=media, model=a.model, title=a.title or src.stem, duration=round(len(audio) / SR, 1))
    serve(session, None, a.port, not a.no_browser)
    t = time.time()
    segs = run_pass(session, a.model, parse_langs(a.langs), a.hotwords, a.threads, store, status, [])
    export_text(session, segs)
    status.update(state="done", polish=dict(done=1, total=1, model=a.model, finished=True))
    print(f"\nDone in {time.time() - t:.0f}s for {len(audio) / SR:.0f}s of audio (x{(time.time() - t) / max(1, len(audio) / SR):.2f} real time).")
    if a.once:
        return
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def latest_session():
    ds = sorted(p for p in SESSIONS.glob("*") if p.is_dir()) if SESSIONS.exists() else []
    if not ds:
        sys.exit("No sessions yet.")
    return ds[-1]


def cmd_polish(a):
    session = Path(a.session) if a.session else latest_session()
    status, store = Status(session / "status.json"), Store(session)
    live = json.loads((session / "transcript.json").read_text(encoding="utf-8"))
    store.segs = live
    shutil.copy(session / "transcript.json", session / "transcript_live.json")
    serve(session, None, a.port, not a.no_browser)
    final = run_pass(session, a.model, parse_langs(a.langs), a.hotwords, a.threads, store, status, live)
    export_text(session, final)
    status.update(state="done", polish=dict(done=1, total=1, model=a.model, finished=True))
    print("Done.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def cmd_review(a):
    session = Path(a.session) if a.session else latest_session()
    serve(session, None, a.port, not a.no_browser)
    print("Ctrl+C to close.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, model):
        p.add_argument("--model", default=model, help="tiny|base|small|medium|turbo (bigger = slower, more accurate)")
        p.add_argument("--langs", default="en,hi", help="languages to expect, e.g. en,hi,ta  or 'auto' (default en,hi)")
        p.add_argument("--hotwords", default=None, help="names/terms to bias towards, e.g. \"Priya, Acme\"")
        p.add_argument("--threads", type=int, default=6)
        p.add_argument("--port", type=int, default=8765)
        p.add_argument("--no-browser", action="store_true")
        p.add_argument("--title", default=None)
        p.add_argument("--once", action="store_true", help="exit when done instead of keeping the viewer open")

    r = sub.add_parser("record"); common(r, "small")
    r.add_argument("--mic", action="store_true", help="also record + transcribe the microphone")
    r.add_argument("--no-video", action="store_true")
    r.add_argument("--fps", type=int, default=5)
    r.add_argument("--polish", nargs="?", const="turbo", default=None, metavar="MODEL",
                   help="after recording, re-transcribe with a bigger model (default turbo)")
    r.set_defaults(fn=cmd_record)

    f = sub.add_parser("file"); common(f, "small"); f.add_argument("path"); f.set_defaults(fn=cmd_file)
    p = sub.add_parser("polish"); common(p, "turbo"); p.add_argument("session", nargs="?"); p.set_defaults(fn=cmd_polish)
    v = sub.add_parser("review"); v.add_argument("session", nargs="?"); v.add_argument("--port", type=int, default=8765)
    v.add_argument("--no-browser", action="store_true"); v.set_defaults(fn=cmd_review)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
