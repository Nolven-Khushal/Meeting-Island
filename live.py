"""
Streaming transcription core for the island.

  audio -> StreamLane (cuts utterances at pauses, emits a growing DRAFT every ~0.6 s, then a FINAL)
        -> Scheduler  (finals first; only the newest draft per lane is kept - never a backlog of stale drafts)
        -> LiveASR    (one worker: detect language, pick engine: `small` for English, `turbo` for Hindi / other)

Events go out through `on_event(kind, lane_label, utt_id, payload)`:
  'draft'  payload = dict(text, words, lang, conf)   tentative text for the sentence being spoken
  'clear'  payload = None                            that lane's draft should disappear (it was noise)
Finals are written into the Store (which also notifies the UI).
"""
import collections
import re
import threading
import time
import traceback

import numpy as np

import assistant
from engine import Engine

SR = assistant.SR


class Job:
    __slots__ = ("lane", "utt", "off", "audio", "forced")

    def __init__(self, lane, utt, off, audio, forced=False):
        self.lane, self.utt, self.off, self.audio, self.forced = lane, utt, off, audio, forced


class Scheduler:
    def __init__(self):
        self.cv = threading.Condition()
        self.finals, self.drafts, self.closed = collections.deque(), {}, False

    def put_final(self, job):
        with self.cv:
            self.finals.append(job)
            self.drafts.pop(job.lane, None)   # the final supersedes any pending draft
            self.cv.notify()

    def put_draft(self, job):
        with self.cv:
            self.drafts[job.lane] = job       # latest wins
            self.cv.notify()

    def drop_draft(self, lane):
        with self.cv:
            self.drafts.pop(lane, None)

    def get(self):
        with self.cv:
            while not (self.finals or self.drafts or self.closed):
                self.cv.wait()
            if self.finals:
                return "final", self.finals.popleft()
            if self.drafts:
                lane = next(iter(self.drafts))
                return "draft", self.drafts.pop(lane)
            return None

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()

    def pending_finals(self):
        with self.cv:
            return len(self.finals)


class StreamLane(assistant.Lane):
    """assistant.Lane that emits drafts while you speak and a final at each pause."""

    def __init__(self, label, wav_path, sched, T0, rate, channels, max_window=6.0, pause=0.5):
        super().__init__(label, wav_path, None, T0, rate, channels)
        self.sched, self.max_window, self.pause = sched, max_window, pause
        self.utt, self.last_draft_n = 0, 0
        self.last_lang, self.det_secs = None, None     # used by LiveASR
        self.last_voice = None                         # last speaker number heard on this lane (see speakers.py)
        self.draft_every = 0.6                         # seconds of new audio between live drafts

    OVERLAP = 0.9    # seconds re-read after a forced cut, so the word that was cut in half is heard whole next time

    def _emit(self, k, forced=False):  # a finished utterance
        self.sched.put_final(Job(self, self.utt, self.pend_start / SR, self.pend[:k].copy(), forced))
        self.utt += 1
        self.last_draft_n, self.det_secs = 0, None
        ov = int(self.OVERLAP * SR)
        self._drop(k - ov if forced and k > 3 * ov else k)

    def _maybe_cut(self, final=False):
        n = len(self.pend)
        if n == 0 or (not final and n - self._chk < SR // 5):
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
            if dur > 2.5:
                if self.last_draft_n:                  # we showed a draft for what turned out to be noise
                    self.sched.drop_draft(self)
                    self.last_draft_n = 0
                    self.on_noise(self)
                self._drop(n - int(0.4 * SR))
            return
        pf = int(self.pause / 0.02)
        if dur >= 1.0 and len(voiced) >= pf and not voiced[-pf:].any():
            last = int(np.nonzero(voiced)[0][-1])      # pause after speech -> sentence boundary
            self._emit(min(n, (last + 1) * self.FRAME + int(0.25 * SR)))   # trim the silence: it makes Whisper repeat itself
        elif dur >= self.max_window:
            k = 150                                    # no pause: cut at the quietest moment of the last 3 s
            sm = np.convolve(rms[-k:], np.ones(5) / 5, "same")
            idx = len(rms) - k + int(np.argmin(sm[2:-2])) + 2
            self._emit(idx * self.FRAME, forced=True)
        elif dur >= 0.8 and n - self.last_draft_n >= int(self.draft_every * SR):
            self.last_draft_n = n
            self.sched.put_draft(Job(self, self.utt, self.pend_start / SR, self.pend.copy()))

    def on_noise(self, lane):  # replaced by LiveASR
        pass


class LiveASR(threading.Thread):
    """
    engine="hinglish" (default, chosen by benchmark - see bench/):
        drafts : Whisper-Hindi2Hinglish-Swift (Oriserve, Roman-script Hinglish, ~0.05x real time)
        finals : Swift + AI4Bharat IndicConformer, fused (fusion.py): IndicConformer gets the Hindi right,
                 Swift supplies proper English spellings. Benchmark (80 real clips): 30% WER / 19% CER,
                 versus 70% / 54% for the old small+turbo setup.
    engine="classic": the earlier small (English) + turbo (Hindi) Whisper pair.
    """

    def __init__(self, store, on_event, langs, model="small", hi_model="turbo", hotwords=None, threads=4, hi_threads=5,
                 engine="hinglish", speakers=True):
        super().__init__(daemon=True)
        self.store, self.on_event = store, on_event
        self.langs = langs or ["en", "hi"]
        self.hotwords = hotwords
        self.sched = Scheduler()
        self.engine = engine
        self.prompt = f"{hotwords}." if hotwords else None
        self.glossary = None          # set by the backend at each session start (glossary.txt is re-read every time)
        self.speakers = None          # SpeakerTracker, loaded in the background (None = off / still loading)
        self.mic_is_room = False      # microphone-only mode: the mic hears several people, so it gets speaker labels too
        if speakers:
            threading.Thread(target=self._load_speakers, daemon=True).start()
        self.turbo = self.indic = None
        if engine == "hinglish":
            # padded with silence to 15 s: as accurate as Whisper's standard 30 s window (no padding at all costs ~10 WER
            # points on this model) at half the cost. Drafts use 10 s for speed. See bench/results.json.
            self.swift = self.small = Engine("models/hinglish-swift-ct2", threads=threads, pad=15, roman=True)
            threading.Thread(target=self._load_indic, args=(hi_threads,), daemon=True).start()
        else:
            self.small = Engine(model, threads=threads)
            if hi_model and any(l != "en" for l in self.langs):
                threading.Thread(target=self._load_hi, args=(hi_model, hi_threads), daemon=True).start()

    def _load_hi(self, name, threads):
        try:
            self.turbo = Engine(name, threads=threads)
            print(f"[live] {name} ready for Hindi/other languages", flush=True)
        except Exception:
            traceback.print_exc()

    def _load_speakers(self):
        try:
            from speakers import SpeakerTracker
            self.speakers = SpeakerTracker()
            print("[live] speaker labels ready", flush=True)
        except Exception:
            traceback.print_exc()

    def voice_for(self, lane, audio):
        """Which person is speaking? Only the system-audio lane (other people); the microphone lane is always 'You'."""
        if self.speakers is None or (lane.label != "System" and not self.mic_is_room):
            return None, None
        n, e = self.speakers.assign(audio)
        if n is None:                                   # too short to recognise a voice: keep the previous speaker
            return lane.last_voice, None
        lane.last_voice = n
        return n, e

    def _load_indic(self, threads):
        try:
            from engine import IndicEngine
            self.indic = IndicEngine(threads=threads)
            print("[live] IndicConformer ready for Hindi", flush=True)
        except Exception:
            traceback.print_exc()

    STYLE_PROMPT = "Haan toh dekhiye, hum log meeting mein project ke baare mein baat kar rahe hain, aur client ko delivery bhejni hai."

    def hinglish_prompt(self, hindi_context=True):
        """A Hinglish sentence nudges the model to write Hinglish (only while the talk has been in Hindi: on plain English
        it can pull words towards Hindi); your glossary terms make them more likely to be spelled right."""
        terms = self.glossary.prompt(limit=40) if self.glossary else None
        return " ".join(x for x in (self.STYLE_PROMPT if hindi_context else None, self.prompt, terms) if x) or None

    def fix_terms(self, text):
        return self.glossary.apply(text) if self.glossary else text

    def hindi_ready(self):
        return (self.indic is not None) if self.engine == "hinglish" else (self.turbo is not None)

    def make_lane(self, label, wav_path, T0, rate, channels, max_window=6.0):
        lane = StreamLane(label, wav_path, self.sched, T0, rate, channels, max_window=max_window)
        lane.on_noise = lambda ln: self.on_event("clear", ln.label, ln.utt, None)
        if self.engine == "hinglish":
            lane.draft_every = 0.45                     # Swift is cheap: refresh drafts more often
        return lane

    # -------------------------------------------------------------------------
    def _pick_lang(self, lane, audio, secs, final):
        if len(self.langs) == 1:
            return self.langs[0]
        if secs < 1.6:
            return lane.last_lang or self.langs[0]
        if final or lane.det_secs is None or secs - lane.det_secs >= 3.0:
            lang, p = self.small.detect_language(audio[-6 * SR:], self.langs)
            lane.det_secs = secs
            if p < 0.55 and lane.last_lang:
                return lane.last_lang
            lane.last_lang = lang
        return lane.last_lang or self.langs[0]

    busy = False

    def run(self):
        while True:
            item = self.sched.get()
            if item is None:
                return
            kind, job = item
            self.busy = True
            try:
                self._process(kind, job)
            except Exception:
                traceback.print_exc()
            finally:
                self.busy = False

    def wait_idle(self, timeout=60):
        """Block until every queued final has been transcribed (used when a session ends)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if not self.busy and self.sched.pending_finals() == 0:
                return True
            time.sleep(0.05)
        return False

    def _process(self, kind, job):
        if self.engine == "hinglish":
            return self._process_hinglish(kind, job)
        lane, audio = job.lane, job.audio
        secs = len(audio) / SR
        final = kind == "final"
        lang = self._pick_lang(lane, audio, secs, final)
        eng = self.small if (lang == "en" or self.turbo is None) else self.turbo
        if not final and eng is self.turbo and secs > 4.0:
            audio = audio[-4 * SR:]            # keep Hindi drafts quick: only re-decode the most recent 4 s
        res = eng.transcribe(audio, lang, self.prompt, beam=1)   # (logits for confidence need greedy decoding)
        junk = res is None or res["conf"] < 0.2 or (res["no_speech"] > 0.6 and res["conf"] < 0.5)
        if final:
            if junk:
                self.on_event("clear", lane.label, job.utt, None)
                return
            self.store.add([dict(start=round(job.off, 2), end=round(job.off + secs, 2), text=res["text"], lang=lang,
                                 conf=res["conf"], speaker=lane.label, words=res["words"], utt=job.utt)])
        elif not junk:
            self.on_event("draft", lane.label, job.utt, dict(text=res["text"], words=res["words"], lang=lang, conf=res["conf"]))

    def hinglish_final(self, audio, last_lang, forced=False):
        """The finished text for one stretch of speech (live finals and the after-meeting pass use this same routine).
        -> dict(text, lang, words, native, conf) or None when nothing intelligible was said."""
        import fusion
        sw = self.swift.transcribe(audio, "hi", self.hinglish_prompt(last_lang != "en"), beam=1, pad=15)
        sw_text = self.fix_terms(sw["text"]) if sw else ""
        sw_junk = sw is None or sw["conf"] < 0.2 or (sw["no_speech"] > 0.6 and sw["conf"] < 0.5)
        words = [re.sub(r"[^\w'\-]", "", w) for w in sw_text.split()]
        eng = fusion.english_ratio(sw_text)
        native = None
        if not sw_junk and eng >= 0.75:               # clearly English: Swift alone, no need to wait for IndicConformer
            text, lang, wl = sw_text, "en", sw["words"]
        else:
            ind = self.indic.transcribe(audio) if self.indic is not None else ""
            if sw_junk and not ind:
                return None
            if sw_junk:                               # only IndicConformer heard speech
                text, lang, wl, native = fusion.fuse(ind, ""), "hi", [], ind
            elif not ind or eng >= 0.75:              # an English sentence: Swift's text as is
                text, lang, wl = sw_text, "en" if eng >= 0.75 else "hi", sw["words"]
            elif len(ind.split()) < 0.6 * len(words):  # IndicConformer dropped much of it (English-heavy)
                text, lang, wl = sw_text, "hi", sw["words"]
            else:                                     # Hindi / Hinglish: IndicConformer + Swift's English spellings
                text, lang, wl, native = fusion.fuse(ind, sw_text), "hi", [], ind
        text = self.fix_terms(text)
        if forced:                                    # the last word was cut in half: drop it, the next chunk re-reads it whole
            parts = text.split()
            if len(parts) > 3:
                text = " ".join(parts[:-1])
                wl = wl[:-1] if wl else wl
                if native:
                    native = " ".join(native.split()[:-1])
        return dict(text=text, lang=lang, words=wl, native=native, conf=sw["conf"] if sw else 0.8)

    def _process_hinglish(self, kind, job):
        import fusion
        lane, audio = job.lane, job.audio
        secs = len(audio) / SR
        if kind != "final":                           # a live draft: Swift only, quick
            sw = self.swift.transcribe(audio, "hi", self.hinglish_prompt(lane.last_lang != "en"), beam=1, pad=10)
            sw_text = self.fix_terms(sw["text"]) if sw else ""
            sw_junk = sw is None or sw["conf"] < 0.2 or (sw["no_speech"] > 0.6 and sw["conf"] < 0.5)
            if not sw_junk:
                self.on_event("draft", lane.label, job.utt, dict(text=sw_text, words=sw["words"],
                                                               lang="en" if fusion.english_ratio(sw_text) >= 0.75 else "hi",
                                                               conf=sw["conf"]))
            return
        r = self.hinglish_final(audio, lane.last_lang, job.forced)
        if r is None:
            self.on_event("clear", lane.label, job.utt, None)
            return
        lane.last_lang = r["lang"]                    # remembered: decides whether the next sentence gets the Hinglish style hint
        voice, emb = self.voice_for(lane, audio)
        self.store.add([dict(start=round(job.off, 2), end=round(job.off + secs, 2), text=r["text"], lang=r["lang"],
                             conf=r["conf"], speaker=lane.label, words=r["words"], utt=job.utt, native=r["native"],
                             forced=job.forced, id=f"{lane.label}-{job.utt}", voice=voice,
                             emb=[round(float(x), 3) for x in emb] if emb is not None else None)])

    def shutdown(self):
        self.sched.close()
