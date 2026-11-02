"""
A scripted film of the REAL app, recorded from inside it.

The island window is excluded from normal screen capture on purpose (so it never films itself), so this renders the app's own
frames (widget.grab()) over a generated wallpaper instead: no real desktop, window, path or person ever appears. A virtual
orange cursor, captions and chapter cards are drawn on top, the frames go to an H.264 encoder, and the script moves through the
app exactly like a person would (click Start, pause, drag, dock, End, Save ...). Chapters are plain Python generators.

Run through  tools/showcase/run_tour.py  (sets the environment, launches the app with --tour).
"""
import os
import queue
import re
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import av
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF

HERE = Path(__file__).resolve().parent
OUTW, OUTH, FPS = 1280, 720, 25
ORANGE = QColor(255, 138, 28)
FAMILIES = ["Segoe UI Variable Display", "Segoe UI", "Nirmala UI"]
SKIP_LONG = bool(os.environ.get("TOUR_QUICK"))          # quick check of the script: shorter waits, same steps


def font(px, weight=500):
    f = QFont()
    f.setFamilies(FAMILIES)
    f.setPixelSize(px)
    f.setWeight(QFont.Weight(weight))
    return f


class Film:
    def __init__(self, isl, out):
        self.isl, self.out = isl, Path(out)
        (self.out / "shots").mkdir(parents=True, exist_ok=True)
        self.scr = isl.scr.size()
        self.wall = QImage(str(HERE / "wallpaper.png")).scaled(self.scr, Qt.AspectRatioMode.IgnoreAspectRatio,
                                                               Qt.TransformationMode.SmoothTransformation)
        self.logo = QImage(str(HERE / "logo.png"))
        self.cur = QPointF(self.scr.width() * 0.62, self.scr.height() * 0.6)
        self.cur_t = QPointF(self.cur)
        self.cur_on, self.ripple = False, None
        self.caption, self.cap_since = None, 0.0
        self.card_data = None
        self.t0 = self.last = time.perf_counter()
        self.last_pts = -1
        self.q = queue.Queue(maxsize=12)
        self.container = av.open(str(self.out / "video_only.mp4"), "w")
        self.stream = self.container.add_stream("libx264", rate=FPS)
        self.stream.width, self.stream.height, self.stream.pix_fmt = OUTW, OUTH, "yuv420p"
        self.stream.options = {"crf": "27", "preset": "veryfast", "threads": "2"}
        self.enc = threading.Thread(target=self._encode, daemon=True)
        self.enc.start()
        self.wait = ("t", 0.0)
        self.gen = None
        self.dropped = 0
        self.snap_name = None

    # ------------------------------------------------------------------ encoder thread
    def _encode(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            pts, arr = item
            fr = av.VideoFrame.from_ndarray(arr, format="rgba").reformat(format="yuv420p")
            fr.pts = pts
            for pkt in self.stream.encode(fr):
                self.container.mux(pkt)
        for pkt in self.stream.encode(None):
            self.container.mux(pkt)
        self.container.close()

    # ------------------------------------------------------------------ what is on screen
    def say(self, text):
        self.caption, self.cap_since = text, time.perf_counter()

    def show(self, title, body="", label="", image=None, boxes=None, mono=False, name=None):
        self.card_data = dict(title=title, body=body, label=label, image=image, boxes=boxes or [], mono=mono, since=time.perf_counter())
        self.snap_name = name                                       # taken a moment later, when the card has faded in

    def hide(self):
        self.card_data = None

    def centre(self, name):
        for n, r in reversed(self.isl.hits):
            if n == name:
                return QPointF(self.isl.mapToGlobal(r.center().toPoint()))
        return None

    def go(self, p):
        if p is not None:
            self.cur_t = QPointF(p)
            self.cur_on = True

    def click(self):
        self.ripple = time.perf_counter()

    def compose(self, overlays=True):
        img = QImage(self.scr, QImage.Format.Format_ARGB32_Premultiplied)
        p = QPainter(img)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        if self.card_data:
            self._card(p)
            p.end()
            return img
        p.drawImage(0, 0, self.wall)
        for w in (self.isl, self.isl.dialog):
            if w is not None and w.isVisible():
                p.drawPixmap(w.mapToGlobal(QPoint(0, 0)), w.grab())
        if overlays:
            self._caption(p)
            self._cursor(p)
        p.end()
        return img

    def _caption(self, p):
        if not self.caption:
            return
        a = min(1.0, (time.perf_counter() - self.cap_since) / 0.35)
        f = font(31, 600)
        p.setFont(f)
        fm = QFontMetricsF(f)
        W = min(self.scr.width() - 200, 1500)
        r = fm.boundingRect(QRectF(0, 0, W - 80, 400), int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap), self.caption)
        box = QRectF((self.scr.width() - r.width() - 80) / 2, self.scr.height() - r.height() - 120, r.width() + 80, r.height() + 36)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(8, 10, 18, int(215 * a)))
        p.drawRoundedRect(box, 18, 18)
        p.setPen(QColor(255, 255, 255, int(255 * a)))
        p.drawText(box.adjusted(40, 18, -40, -18), int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap), self.caption)

    def _cursor(self, p):
        if not self.cur_on:
            return
        c = self.cur
        if self.ripple is not None:
            t = time.perf_counter() - self.ripple
            if t < 0.55:
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(QColor(255, 138, 28, int(220 * (1 - t / 0.55))), 4))
                rr = 10 + 60 * t / 0.55
                p.drawEllipse(c, rr, rr)
            else:
                self.ripple = None
        poly = QPolygonF([QPointF(0, 0), QPointF(0, 30), QPointF(8, 23), QPointF(14, 36), QPointF(19, 34), QPointF(13, 21), QPointF(23, 21)])
        poly.translate(c)
        p.setPen(QPen(QColor(255, 255, 255), 3))
        p.setBrush(ORANGE)
        p.drawPolygon(poly)

    def _card(self, p):
        d = self.card_data
        W, H = self.scr.width(), self.scr.height()
        g = QLinearGradient(0, 0, W, H)
        g.setColorAt(0, QColor(10, 12, 28)); g.setColorAt(1, QColor(52, 28, 84))
        p.fillRect(0, 0, W, H, g)
        a = min(1.0, (time.perf_counter() - d["since"]) / 0.5)
        p.setOpacity(a)
        x = 170
        y = 190 if not d["image"] else 70
        if d["label"]:
            p.setFont(font(24, 700)); p.setPen(ORANGE)
            p.drawText(QPointF(x, y), d["label"])
            y += 30
        p.setFont(font(58 if not d["image"] else 44, 700)); p.setPen(QColor(255, 255, 255))
        fm = QFontMetricsF(p.font())
        r = fm.boundingRect(QRectF(0, 0, W - 2 * x, 600), int(Qt.TextFlag.TextWordWrap), d["title"])
        p.drawText(QRectF(x, y, W - 2 * x, r.height() + 10), int(Qt.TextFlag.TextWordWrap), d["title"])
        y += r.height() + 34
        if d["image"]:
            im = d["image"].scaledToWidth(860, Qt.TransformationMode.SmoothTransformation)
            p.drawImage(x, y, im)
            p.setPen(QPen(QColor(170, 215, 100), 4))
            p.setBrush(Qt.BrushStyle.NoBrush)
            k = 860 / d["image"].width()
            for (bx1, by1, bx2, by2) in d["boxes"]:
                p.drawRect(QRectF(x + bx1 * k, y + by1 * k, (bx2 - bx1) * k, (by2 - by1) * k))
            x2 = x + 860 + 50
            p.setFont(font(27, 500)); p.setPen(QColor(215, 220, 238))
            p.drawText(QRectF(x2, y, W - x2 - 80, 700), int(Qt.TextFlag.TextWordWrap), d["body"])
        else:
            if d["mono"]:
                f = QFont("Consolas"); f.setPixelSize(24)
            else:
                f = font(30, 400)
            p.setFont(f); p.setPen(QColor(215, 220, 238))
            p.drawText(QRectF(x, y, W - 2 * x, H - y - 80), int(Qt.TextFlag.TextWordWrap), d["body"])
        if d["label"] == "" and d["title"].startswith("Meeting Island"):
            p.drawImage(QRectF(W - 420, 200, 240, 240), self.logo)
        p.setOpacity(1.0)

    def snap(self, name, cards_ok=True):
        img = self.compose(overlays=False)
        img.scaled(1440, 810, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation).save(
            str(self.out / "shots" / f"{name}.png"))
        print("SNAP", name, flush=True)

    # ------------------------------------------------------------------ the clock
    def tick(self):
        now = time.perf_counter()
        dt = now - self.last
        self.last = now
        k = 1 - pow(2.718281828, -dt * 7.5)
        self.cur += (self.cur_t - self.cur) * k
        pts = int((now - self.t0) * FPS)
        if pts != self.last_pts:
            self.last_pts = pts
            img = self.compose().scaled(OUTW, OUTH, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            img = img.convertToFormat(QImage.Format.Format_RGBA8888)
            arr = np.frombuffer(img.constBits(), np.uint8).reshape(OUTH, img.bytesPerLine() // 4, 4)[:, :OUTW].copy()
            try:
                self.q.put_nowait((pts, arr))
            except queue.Full:
                self.dropped += 1
        if self.snap_name and self.card_data and now - self.card_data["since"] > 0.9:
            self.snap(self.snap_name)
            self.snap_name = None
        self._advance(now)

    def _advance(self, now):
        if self.gen is None:
            return
        w = self.wait
        if w[0] == "t" and now < w[1]:
            return
        if w[0] == "c":
            try:
                done = bool(w[1]())
            except Exception:
                done = False
            if not done and now < w[2]:
                return
        try:
            step = next(self.gen)
        except StopIteration:
            self.finish()
            return
        except Exception:
            traceback.print_exc()
            self.finish()
            return
        if isinstance(step, tuple):
            self.wait = ("c", step[1], now + (5 if SKIP_LONG else step[2]))
        else:
            self.wait = ("t", now + (min(step, 0.6) if SKIP_LONG else step))

    def finish(self):
        self.gen = None
        print(f"FILM DONE {time.perf_counter() - self.t0:.0f}s, dropped {self.dropped}", flush=True)
        self.q.put(None)
        self.enc.join(60)
        QTimer.singleShot(300, self.isl.quit_app)


# ---------------------------------------------------------------------------------------------------- the story
def pause_until(fn, secs=120):
    return ("c", fn, secs)


def script(f):
    isl, out = f.isl, f.out
    sw, sh = f.scr.width(), f.scr.height()
    wav_secs = 100

    def chapter(n):
        print("CHAPTER", n, flush=True)

    yield pause_until(lambda: isl.mode == "ready" and isl.backend.hindi_ready() and isl.backend.asr.speakers is not None, 180)
    isl.expanded = False
    yield 1.2

    # ---- opening and the brief -----------------------------------------------------------------------------------------
    chapter("opening")
    f.show("Meeting Island", "A live meeting transcript that sits under your camera.\n\nOffline. English, Hindi and Hinglish. No cloud, no API.",
           name="01-title")
    yield 9
    f.show("Writing up a meeting takes longer than the meeting.",
           "The brief: a tool that listens to a meeting and writes it down while it happens, in English, Hindi and the Hinglish people "
           "really speak.\n\nIt has to work offline, with no API, on an ordinary 8 GB laptop without a graphics card, and an executive "
           "assistant has to be able to install it by double-clicking one file.", label="THE BRIEF")
    yield 12
    f.show("The first version was wrong and slow.",
           "Whisper out of the box: 14 seconds behind the speaker, and about 70% of the words wrong on real Hinglish clips.\n\n"
           "So the project became measurement: 80 real clips with human transcripts, every idea scored, most ideas thrown away.",
           label="WHY IT IS HARD")
    yield 11
    f.show("What was measured, and what was dropped.",
           "80 real clips, 40 spoken tutorials and 40 spontaneous conversations, all scored the same spelling-tolerant way.\n\n"
           "First version: 70% of words wrong\nHinglish Whisper, padded: 37%\nHindi model alone: 39%\n"
           "Both fused: 30%  (24% in conversation)\n\n"
           "Dropped after measuring: loudness normalising, beam search, a generic glossary, and a large model that was twice as slow "
           "for half a point.", label="THE BENCHMARK")
    yield 14
    f.hide()

    # ---- one click -----------------------------------------------------------------------------------------------------
    chapter("start")
    f.say("It lives at the top of the screen, under the camera. Nothing is listening yet.")
    yield 3.5
    f.go(QPointF(isl.shape_rect().center()))
    yield 1.2
    f.click()
    isl.expanded = True
    yield 1.8
    f.snap("02-ready")
    f.say("Press Start session. Everything the PC plays, or the room microphone, is transcribed live.")
    isl.record = "audio"
    isl.backend.cfg["record"] = "audio"
    yield 2.5
    f.go(f.centre("start"))
    yield 1.4
    f.click()
    yield 0.25
    isl.on_button("start")
    t_start = time.perf_counter()
    f.say("Grey text is a draft that appears as the person speaks; it settles into a sentence at the pause.")
    yield 14
    f.snap("03-live")
    f.say("Voices are told apart as the meeting goes: each speaker gets a colour.")
    yield 14
    f.snap("04-live-speakers")

    # ---- pause --------------------------------------------------------------------------------------------------------
    chapter("pause")
    f.say("Pause and resume hold the same session. The paused minutes never enter the timestamps.")
    f.go(f.centre("toggle"))
    yield 1.6
    f.click()
    isl.on_button("toggle")
    yield 1.8
    f.snap("05-paused")
    yield 1.5
    f.go(f.centre("toggle"))
    yield 1.0
    f.click()
    isl.on_button("toggle")
    yield 1.5

    # ---- drag, circle, dock, home --------------------------------------------------------------------------------------
    chapter("drag")
    f.say("Grab it and it turns into a circle while it moves.")
    c = QPointF(isl.shape_rect().center().x(), isl.shape_rect().top() + 24)
    f.go(c)
    yield 1.0
    f.click()
    x0, y0 = c.x(), c.y()
    isl.press = dict(x=x0, y=y0, moved=True)
    isl.dragging = True
    x1, y1 = sw * 0.55, sh * 0.5
    for i in range(1, 25):
        isl.pointer = QPointF(x0 + (x1 - x0) * i / 24, y0 + (y1 - y0) * i / 24)
        f.go(isl.pointer)
        yield 0.05
    yield 0.9
    f.snap("06-circle")
    isl.press = None
    isl.end_drag(x1, y1)
    f.say("Let go anywhere: it opens as a free window.")
    yield 2.4
    f.say("Drag it to an edge and it docks as a slim vertical bar.")
    c = QPointF(isl.shape_rect().center().x(), isl.shape_rect().top() + 24)
    f.go(c)
    yield 0.9
    x0, y0 = c.x(), c.y()
    x1, y1 = sw - 14, sh * 0.55
    isl.press = dict(x=x0, y=y0, moved=True)
    isl.dragging = True
    for i in range(1, 25):
        isl.pointer = QPointF(x0 + (x1 - x0) * i / 24, y0 + (y1 - y0) * i / 24)
        f.go(isl.pointer)
        yield 0.05
    isl.press = None
    isl.end_drag(x1, y1)
    yield 2.2
    f.snap("07-docked")
    f.say("Drop it near the camera and it snaps home.")
    isl.expanded = False
    yield 1.2
    c = QPointF(isl.shape_rect().center().x(), isl.shape_rect().center().y())
    f.go(c)
    yield 0.9
    x0, y0 = c.x(), c.y()
    x1, y1 = sw / 2 + 30, 40
    isl.press = dict(x=x0, y=y0, moved=True)
    isl.dragging = True
    for i in range(1, 25):
        isl.pointer = QPointF(x0 + (x1 - x0) * i / 24, y0 + (y1 - y0) * i / 24)
        f.go(isl.pointer)
        yield 0.05
    isl.press = None
    isl.end_drag(x1, y1)
    yield 2.4
    isl.expanded = True
    yield 1.4

    # ---- glass ----------------------------------------------------------------------------------------------------------
    chapter("glass")
    f.say("Two glass looks, both blurring what is behind them. This is Frost.")
    isl.set_style("frost")
    yield 2.6
    f.snap("08-frost")
    f.say("And Liquid, with a small elastic swell when the pointer comes near.")
    isl.set_style("liquid")
    yield 1.2
    isl.hovering = True
    yield 1.6
    f.snap("09-liquid")
    isl.hovering = False
    yield 1.0

    # ---- let the meeting finish ------------------------------------------------------------------------------------------
    f.say("The meeting plays on. Text keeps arriving about a second after each pause.")
    yield pause_until(lambda: time.perf_counter() - t_start > wav_secs, 140)
    yield 3
    f.snap("10-live-end")

    # ---- end, save ------------------------------------------------------------------------------------------------------
    chapter("save")
    f.say("End session. A window asks where to save it and what to call it, and lets you name the voices.")
    f.go(f.centre("end"))
    yield 1.6
    f.click()
    isl.on_button("end")
    yield pause_until(lambda: isl.dialog is not None, 90)
    yield 1.4
    d = isl.dialog
    tmp = Path(tempfile.mkdtemp(prefix="mi-demo-"))
    d.folder.setText(r"Documents\Meeting Transcripts")                  # what the window shows; the real folder is set just before saving
    d.name.setText("Weekly review")
    d.update()
    yield 1.2
    names = ["Asha", "Ravi", "Meera"]
    for v, nm in zip(sorted(d.names_edit), names):
        for i in range(1, len(nm) + 1):
            d.names_edit[v].setText(nm[:i])
            yield 0.09
        yield 0.3
    f.snap("11-save-window")
    yield 2.4
    p = d.mapToGlobal(QPoint(0, 0))
    for name_ in ("save",):
        btn = next((b for b in d.findChildren(type(d.bd_btn)) if b.objectName() == "save"), None)
        if btn is not None:
            f.go(QPointF(btn.mapToGlobal(btn.rect().center())))
    yield 1.3
    f.click()
    d.folder.setText(str(tmp))
    d.save()
    isl.cfg["save_dir"] = str(tmp)
    yield 1.0

    # ---- the final pass ---------------------------------------------------------------------------------------------------
    chapter("final")
    f.say("The live transcript is saved at once. Then a final pass listens to the recording again, with the whole meeting in view.")
    yield 4
    f.snap("12-final-pass")
    yield pause_until(lambda: isl.polish is None and isl.finisher.pending() == 0, 300)
    yield 1.5
    f.say("Final transcript ready: speakers named, sentences cut only at pauses.")
    isl.expanded = True
    yield 3.5
    f.snap("13-final-ready")
    f.say(None)

    txt = next(iter(sorted(tmp.glob("*.txt"))), None)
    body = ""
    if txt is not None:
        lines = txt.read_text(encoding="utf-8-sig").splitlines()
        body = "\n".join(l[:90] for l in lines[:13] if l.strip())
    f.show("The saved transcript", body, label="RESULT", mono=True, name="14-transcript")
    yield 15
    f.show("Live versus final.",
           "The final pass listens to the recording again with the whole meeting in view.\n\n"
           "Words wrong on three fast-speech streams: 37.8% live, 34.0% final.\n"
           "Right speaker on a line, in fast back-and-forth: 80% live, 91% final.\n\n"
           "The first final pass was worse than live. It cut the audio at every small pause and at every speaker change, even in the "
           "middle of a word. Cutting only at real pauses fixed it.", label="THE SECOND LOOK")
    yield 15
    f.hide()

    # ---- it learns from corrections ---------------------------------------------------------------------------------------
    chapter("learn")
    fixed = None
    if txt is not None:
        text = txt.read_text(encoding="utf-8-sig")
        m = re.search(r"\] [^:\n]+: (?:\S+ ){4}(\w{5,})", text)
        if m:
            fixed = m.group(1)
            txt.write_text(text.replace(fixed, "Priya", 1), encoding="utf-8")
    f.show("It learns from what you fix.",
           f"Open a saved transcript, correct a misheard word, save the file. That is all.\n\n"
           f"Here \"{fixed}\" is corrected to \"Priya\" in the .txt.\n\nThe next time the app starts it compares the file with what it wrote, "
           "keeps the small fixes as rules, and applies them to every later meeting, live and in the final version. Fixes to very common words "
           "only count after you make them twice, so one stray edit cannot do damage.", label="GETS BETTER WITH USE")
    yield 14
    f.hide()
    isl.learn_from_edits()
    isl.expanded = True
    f.say("Next start: it noticed the edit and learned the rule. You can review or delete every rule.")
    yield 4
    f.snap("15-learned")
    f.say(None)

    # ---- who is speaking, from the screen ---------------------------------------------------------------------------------
    chapter("names")
    boxes, found = [], "(no text reader on this PC)"
    try:
        sys.path.insert(0, str(HERE.parent.parent))
        import screennames as sn
        im = QImage(str(HERE / "callgrid.png")).convertToFormat(QImage.Format.Format_RGB888)
        arr = np.frombuffer(im.constBits(), np.uint8).reshape(im.height(), im.bytesPerLine())[:, :im.width() * 3].reshape(im.height(), im.width(), 3)
        bgr = np.ascontiguousarray(arr[..., ::-1])
        rect = sn.find_ring(bgr)
        if rect:
            boxes = [rect]
            nm = sn.NameReader().read(bgr, rect)
            found = f"Ring found. Name read from the tile: \"{nm}\"" if nm else "Ring found."
    except Exception:
        traceback.print_exc()
    f.show("Who is talking, from the screen.",
           "Video-call apps ring the tile of whoever is speaking and write their name on it.\n\n"
           "While a meeting is recorded, the app finds that ring on the full-resolution screen and reads the name offline. "
           "Later it matches names to voices, but only when a name was seen consistently.\n\n" + found +
           "\n\nInvented names, not a real call.", label="NAMES",
           image=QImage(str(HERE / "callgrid.png")), boxes=boxes, name="16-names")
    yield 17
    f.hide()

    # ---- shipping and honesty --------------------------------------------------------------------------------------------
    chapter("shipping")
    f.show("One file to install, a small patch to update.",
           "A single installer carries the program and the speech models: no Python, no downloads, no admin rights.\n\n"
           "Updates are a 16 MB patch that replaces only the files that changed, checks every one, and puts the old version back "
           "by itself if anything fails.\n\nIt also records the screen and audio, and a long meeting is built to stay light: "
           "a 2-hour recording is processed at low priority and waits for a charger.", label="FOR A NON-TECHNICAL USER")
    yield 13
    f.show("The honest numbers.",
           "Hinglish words wrong: 70% at the start, about 30% now (80 real clips, spelling-tolerant scoring).\n\n"
           "Final pass versus live: about 10% fewer word errors, and the right speaker on 91% of lines versus 80% in fast conversation.\n\n"
           "Text appears about one second after a pause in English and two in Hindi.\n\n"
           "The first version of the final pass was worse than live: it cut words in half at speaker changes. A real TV clip showed it, "
           "and the fix was to cut only at real pauses.", label="WHERE IT STANDS", name="17-numbers")
    yield 17
    f.show("What comes next.",
           "An Android version: the Hindi model was checked to run in the same engine that has an Android build.\n\n"
           "Training the Hinglish model further on a free GPU, with a gate that only ships a model that beats the current one on the "
           "project's own clips.\n\nRemembered voices: name a person once and they are recognised in every meeting.",
           label="NEXT")
    yield 12
    f.show("Meeting Island", "Khushal Vashistha\n\nDemo audio: IndicVoices (AI4Bharat), CC BY 4.0. All names in this film are invented.\n"
           "github.com/Nolven-Khushal/Meeting-Island", name="18-closing")
    yield 16


def run(isl, out):
    film = Film(isl, out)
    film.gen = script(film)
    film.timer = QTimer(isl)
    film.timer.timeout.connect(film.tick)
    film.timer.start(int(1000 / FPS))
    isl._film = film
