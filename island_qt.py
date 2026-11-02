#!/usr/bin/env python
"""
Meeting Island - glass edition. Live English / Hindi / Hinglish transcript that floats under your camera. Offline.

  Start session -> listening begins.  Pause / Resume holds the same session.  End session -> a window asks for a
  name and a folder, then saves  "[hh:mm:ss] text"  lines as a .txt file.  Nothing is saved until you say so
  (an autosave copy protects you from crashes; it is offered back as "Recover" next time).

  Drag it anywhere (it becomes a circle while held). Drop on the left / right edge -> slim vertical pill.
  Right-click: glass style (Frost / Liquid), Hindi as Hinglish or Devanagari, open folder, quit.
"""
import argparse
import ctypes
import json
import math
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", default="small")
ap.add_argument("--hi-model", default="turbo")
ap.add_argument("--langs", default="en,hi")
ap.add_argument("--hotwords", default=None)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--hi-threads", type=int, default=6)
ap.add_argument("--engine", default="hinglish", choices=["hinglish", "classic"],
                help="hinglish = Swift + IndicConformer fusion (default, most accurate); classic = small + turbo Whisper")
ap.add_argument("--mic", action="store_true")
ap.add_argument("--window", type=float, default=14.0, help="longest sentence chunk (s) before a forced cut")
ap.add_argument("--style", default=None, choices=["frost", "liquid"])
ap.add_argument("--script", default=None, choices=["roman", "native"])
ap.add_argument("--autostart", action="store_true", help="begin listening immediately instead of waiting for Start")
ap.add_argument("--no-speakers", action="store_true", help="do not tell speakers apart (no Speaker 1 / 2 labels)")
ap.add_argument("--collapsed", action="store_true")
ap.add_argument("--reset-pos", action="store_true")
ap.add_argument("--console", action="store_true")
ap.add_argument("--sim", default=None, help="(testing) feed this audio file instead of the speakers")
ap.add_argument("--selftest", default=None, help="(testing) run the scripted UI test, saving frames in this folder")
ap.add_argument("--tour", default=None, help="(showcase) record the scripted film of the app into this folder (tools/showcase)")
ARGS = ap.parse_args()
if not ARGS.console:
    from paths import DATA_DIR
    _f = open(DATA_DIR / "island.log", "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = _f

import numpy as np
from PySide6.QtCore import (QPointF, QRectF, Qt, QTimer, QUrl, QStandardPaths, QPropertyAnimation, QEasingCurve, QPoint, QRect,
                            QThread, Signal)
from PySide6.QtGui import (QBrush, QColor, QDesktopServices, QFont, QFontMetricsF, QGuiApplication, QImage, QLinearGradient,
                           QPainter, QPainterPath, QPen, QPixmap, QRadialGradient, QTextBlockFormat, QTextCharFormat,
                           QTextCursor, QTextDocument, QTextOption, QAction)
from PySide6.QtWidgets import (QApplication, QCheckBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMenu, QPushButton,
                               QVBoxLayout, QWidget, QMessageBox)

import assistant
import backend as bk
import corrections
import finisher
import recorder
import translit

M = 26                       # transparent margin around the shape (room for the soft shadow)
PANEL, PILL, VBAR, CIRC = (660, 344), (396, 46), (52, 164), 64
BODY_W, BODY_H = PANEL[0] - 48, PANEL[1] - 64 - 18
from paths import DATA_DIR
SETTINGS = DATA_DIR / "island_settings.json"
SPRING_K, SPRING_C = 400.0, 30.0          # framer-motion spring used by the dynamic-island component
GLOSSARY_TEMPLATE = chr(10).join([
    "# Names and terms your meetings use - one per line. Applies from the next Start session.",
    "# A plain word is spelled exactly like this whenever something sounds like it.",
    "# wrong=right   always replaces wrong with right.",
    "# Example lines (delete the # to use them):",
    "# Priya", "# Rahul", "# Kubernetes", "# kuber=Kubernetes", ""])


VOICE_COLORS = [(124, 196, 255), (255, 184, 107), (139, 226, 139), (255, 143, 171), (195, 166, 255), (255, 226, 122)]   # one per speaker

STYLES = {
    "frost": dict(tint=(16, 18, 44, 78), sat=1.65, bright=1.0, blur_f=3, blur_r=4, blur_n=3, lens=False, spec=0.16, warp=0,
                  liquid=False, rim=(255, 255, 255, 215), rim2=(255, 255, 255, 110), bar=(206, 255, 222)),
    # "liquid": a light 3 px blur, the backdrop warped by smooth noise (like SVG feDisplacementMap), a 25% white veil, and
    # thin inset highlights on the top-left / bottom-right edges; hover makes it swell with an elastic overshoot.
    "liquid": dict(tint=(10, 12, 30, 40), sat=1.25, bright=1.0, blur_f=3, blur_r=1, blur_n=1, lens=False, spec=0.0, warp=18,
                   liquid=True, rim=(255, 255, 255, 255), rim2=(255, 255, 255, 190), bar=(236, 255, 244)),
}


# ----------------------------------------------------------------------------- helpers
class Spring:
    """Damped spring. k = stiffness, c = damping. k=200, c=23 is a gentle iOS-like settle with ~3% overshoot."""

    def __init__(self, v=0.0, k=200.0, c=23.0):
        self.x, self.v, self.t, self.k, self.c = float(v), 0.0, float(v), k, c

    def step(self, dt):
        n = max(1, int(dt / 0.004))
        h = dt / n
        for _ in range(n):
            self.v += (self.k * (self.t - self.x) - self.c * self.v) * h
            self.x += self.v * h

    def settled(self, eps=0.05):
        return abs(self.t - self.x) < eps and abs(self.v) < eps * 4

    def snap(self, v):
        self.x = self.t = float(v)
        self.v = 0.0


class Layer:
    """
    One piece of content (panel / pill / dock / circle). Copies the dynamic-island React component's transitions:
      enter : opacity 0->1, scale .9->1, y +5px->0, spring(stiffness 400, damping 30)
      exit  : opacity 1->0, blur 0->10px, scale 1->.95, y 0->+20px, ~0.28 s ease-out
    """
    EXIT_SECONDS = 0.28

    def __init__(self, on=False):
        self.s = Spring(1.0 if on else 0.0, 400, 30)
        self.x, self.t, self.exiting = (1.0 if on else 0.0), (1.0 if on else 0.0), False

    def step(self, dt):
        if self.t > 0.5:
            self.exiting = False
            self.s.t = 1.0
            self.s.step(dt)
            self.x = self.s.x
        else:
            self.exiting = self.x > 0.0
            self.x = max(0.0, self.x - dt / self.EXIT_SECONDS)
            self.s.x, self.s.v = self.x, 0.0

    def settled(self, eps=0.05):
        return self.s.settled(eps) if self.t > 0.5 else self.x <= 0.0


def shape_path(r, rad):
    """Rounded rectangle with iOS-style continuous ("squircle") corners; a plain circle/stadium when fully rounded."""
    path = QPainterPath()
    rad = max(0.0, min(rad, r.width() / 2, r.height() / 2))
    if rad < 0.5:
        path.addRect(r)
        return path
    if rad >= min(r.width(), r.height()) / 2 - 0.5:          # pill / circle: exact stadium
        path.addRoundedRect(r, rad, rad)
        return path
    ext = min(rad * 1.16, r.width() / 2, r.height() / 2)    # continuous corners start a little earlier along the edge
    n, steps, pts = 3.4, 14, []
    # clockwise: top-right (top edge -> right edge), bottom-right, bottom-left, top-left; rev = walk the arc backwards
    corners = ((r.right() - ext, r.top() + ext, 1, -1, True), (r.right() - ext, r.bottom() - ext, 1, 1, False),
               (r.left() + ext, r.bottom() - ext, -1, 1, True), (r.left() + ext, r.top() + ext, -1, -1, False))
    for cx, cy, sx, sy, rev in corners:
        for k in (range(steps, -1, -1) if rev else range(steps + 1)):
            t = (math.pi / 2) * k / steps
            pts.append(QPointF(cx + sx * ext * (math.cos(t) ** (2 / n)), cy + sy * ext * (math.sin(t) ** (2 / n))))
    path.moveTo(pts[0])
    for pt in pts[1:]:
        path.lineTo(pt)
    path.closeSubpath()
    return path


_WARP = {}


def _smooth_noise(h, w, cell_y, cell_x, seed):
    """Smooth random field in [-1, 1] (bilinear value noise with smoothstep), like feTurbulence fractalNoise."""
    gh, gw = int(h / cell_y) + 3, int(w / cell_x) + 3
    g = np.random.default_rng(seed).uniform(-1, 1, (gh, gw)).astype(np.float32)
    ys, xs = np.linspace(0, gh - 3, h, dtype=np.float32), np.linspace(0, gw - 3, w, dtype=np.float32)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    wy, wx = ys - y0, xs - x0
    wy, wx = wy * wy * (3 - 2 * wy), wx * wx * (3 - 2 * wx)
    wy, wx = wy[:, None], wx[None, :]
    top = g[y0][:, x0] * (1 - wx) + g[y0][:, x0 + 1] * wx
    bot = g[y0 + 1][:, x0] * (1 - wx) + g[y0 + 1][:, x0 + 1] * wx
    return top * (1 - wy) + bot * wy


def warp_image(img, amp):
    """Displace pixels by two smooth noise fields (SVG feDisplacementMap look). img: (h, w, 3) float32."""
    h, w = img.shape[:2]
    key = (h, w)
    if key not in _WARP:                          # low frequency in x, a bit higher in y: baseFrequency="0.001 0.005"
        cy, cx = max(18, h / 3.0), max(60, w / 2.0)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        _WARP[key] = (yy, xx, _smooth_noise(h, w, cy, cx, 17), _smooth_noise(h, w, cy, cx, 29))
        if len(_WARP) > 6:
            _WARP.pop(next(iter(_WARP)))
    yy, xx, nx, ny = _WARP[key]
    ix = np.clip(xx + nx * amp, 0, w - 1).astype(np.int32)
    iy = np.clip(yy + ny * amp, 0, h - 1).astype(np.int32)
    return img[iy, ix]


def box_blur(a, r):
    k = 2 * r + 1
    for ax in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[ax] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=ax, dtype=np.float32)
        n = a.shape[ax]
        hi = [slice(None)] * a.ndim
        lo = [slice(None)] * a.ndim
        hi[ax], lo[ax] = slice(k, k + n), slice(0, n)
        a = (c[tuple(hi)] - c[tuple(lo)]) / k
    return a


def qcol(t, a=None):
    return QColor(t[0], t[1], t[2], t[3] if a is None and len(t) > 3 else (a if a is not None else 255))


class Backdrop(threading.Thread):
    """Grabs the pixels behind a window (our own windows are excluded from capture) and returns a blurred copy."""

    PAD = 64    # physical px captured around the shape, so fast moves stay covered between captures
    dpr = 1.0

    def __init__(self, get_cap, get_style, publish, get_interval=lambda: 0.05):
        super().__init__(daemon=True)
        self.get_cap, self.get_style, self.publish, self.get_interval = get_cap, get_style, publish, get_interval
        self.stop_ev = threading.Event()

    def run(self):
        import mss
        fake = os.environ.get("ISLAND_FAKE_BACKDROP")        # testing: a still image stands in for the desktop
        if fake:
            from PIL import Image
            fa = np.array(Image.open(fake).convert("RGB"))[:, :, ::-1].copy()
        rng = np.random.default_rng(1)
        with mss.MSS() as sct:
            vx, vy = sct.monitors[0]["left"], sct.monitors[0]["top"]
            vw, vh = sct.monitors[0]["width"], sct.monitors[0]["height"]
            if fake:
                vx, vy, vh, vw = 0, 0, fa.shape[0], fa.shape[1]
            while not self.stop_ev.is_set():
                t0 = time.perf_counter()
                try:
                    x, y, w, h = self.get_cap()
                    if w < 24 or h < 24:
                        time.sleep(0.05)
                        continue
                    P = self.PAD
                    x0, y0 = max(vx, x - P), max(vy, y - P)
                    x1, y1 = min(vx + vw, x + w + P), min(vy + vh, y + h + P)
                    if x1 - x0 < 24 or y1 - y0 < 24:
                        time.sleep(0.05)
                        continue
                    st = self.get_style()
                    f = st["blur_f"]
                    if fake:
                        raw = np.ascontiguousarray(fa[y0:y1, x0:x1])
                        qimg = QImage(raw.data, raw.shape[1], raw.shape[0], raw.shape[1] * 3, QImage.Format.Format_BGR888)
                    else:
                        shot = sct.grab({"left": x0, "top": y0, "width": x1 - x0, "height": y1 - y0})
                        qimg = QImage(shot.bgra, shot.width, shot.height, shot.width * 4, QImage.Format.Format_ARGB32)
                    ww, hh = max(2, qimg.width() // f), max(2, qimg.height() // f)
                    # Qt's C++ area-averaging scaler: ~10x faster than numpy and it releases the GIL
                    sm = qimg.scaled(ww, hh, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    sm = sm.convertToFormat(QImage.Format.Format_BGR888)
                    buf = np.frombuffer(sm.constBits(), np.uint8).reshape(hh, sm.bytesPerLine())[:, : ww * 3]
                    small = buf.reshape(hh, ww, 3).astype(np.float32)
                    if st.get("warp"):                                # liquid glass: refract the backdrop through smooth noise
                        small = warp_image(small, st["warp"] * self.dpr / f)
                    for _ in range(st.get("blur_n", 3)):              # box passes ~ a Gaussian (3 = frosted, 1 = light 3 px blur)
                        small = box_blur(small, st["blur_r"])
                    lum = small @ np.array([0.114, 0.587, 0.299], np.float32)
                    small = lum[..., None] + (small - lum[..., None]) * st["sat"]
                    small = small * st["bright"] + rng.normal(0, 1.1, small.shape[:2])[..., None]   # fine grain: no banding
                    small = np.ascontiguousarray(np.clip(small, 0, 255).astype(np.uint8))
                    img = QImage(small.data, small.shape[1], small.shape[0], small.shape[1] * 3, QImage.Format.Format_BGR888).copy()
                    self.publish(img, None, float(small.mean()), (x0, y0, qimg.width(), qimg.height()), None)
                except Exception:
                    traceback.print_exc()
                    time.sleep(0.3)
                time.sleep(max(0.004, self.get_interval() - (time.perf_counter() - t0)))


class VSync(QThread):
    """Ticks once per display refresh: DwmFlush() blocks until the compositor's next frame."""
    tick = Signal()

    def __init__(self, owner):
        super().__init__()
        self.pending = False

    def run(self):
        dwm = ctypes.windll.dwmapi
        last = time.perf_counter()
        while not self.isInterruptionRequested():
            ok = dwm.DwmFlush() == 0
            now = time.perf_counter()
            if not ok or now - last < 0.004:      # composition paused / returned instantly: fall back to ~60 Hz
                time.sleep(max(0.0, 0.0166 - (now - last)))
            last = time.perf_counter()
            if not self.pending:                  # never queue frames up behind a busy UI thread
                self.pending = True
                self.tick.emit()


class GlassWidget(QWidget):
    """A frameless translucent window that paints real frosted glass over whatever is behind it."""

    def __init__(self, focusable=False):
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        if not focusable:
            flags |= Qt.WindowType.WindowDoesNotAcceptFocus
        super().__init__(None, flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        if not focusable:
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.style_name = ARGS.style or "liquid"
        self.bd = self.bdmid = None
        self.cap_rect = (0, 0, 0, 0)
        self.backdrop = None

    def start_backdrop(self):
        if not self.capture_ok:                       # no live backdrop on old Windows: paint_glass uses a static frosted gradient
            return
        self.backdrop = Backdrop(lambda: self.cap_rect, lambda: STYLES[self.style_name], self._publish,
                                 lambda: 0.033 if self.moving else 0.066)
        self.backdrop.dpr = self.devicePixelRatioF()
        self.backdrop.start()

    bd_ver = 0
    bd_lum = 110.0
    moving = False
    bd_at = bdmid_at = None

    def _publish(self, img, mid, lum=110.0, at=None, mid_at=None):
        self.bd, self.bdmid, self.bd_at, self.bdmid_at = img, mid, at, mid_at
        self.bd_lum = 0.75 * self.bd_lum + 0.25 * lum        # smoothed so the tint never flickers
        self.bd_ver += 1

    def origin(self):
        """Top-left of this window in logical screen px (Island overrides with its sub-pixel value)."""
        return float(self.x()), float(self.y())

    def place(self, at, fallback):
        """Where a captured image belongs inside this window: drawn at its real screen position, never stretched."""
        if at is None:
            return fallback
        d = self.devicePixelRatioF()
        ox, oy = self.origin()
        return QRectF(at[0] / d - ox, at[1] / d - oy, at[2] / d, at[3] / d)

    capture_ok = True

    def exclude_from_capture(self):
        """Hide this window from screen capture (so the glass never blurs a picture of itself). Needs Windows 10 2004+."""
        try:
            self.capture_ok = bool(ctypes.windll.user32.SetWindowDisplayAffinity(int(self.winId()), 0x11))   # WDA_EXCLUDEFROMCAPTURE
        except Exception:
            traceback.print_exc()
            self.capture_ok = False
        if not self.capture_ok:
            print("[glass] this Windows cannot hide a window from capture: using the built-in frosted look instead of live blur", flush=True)
        return self.capture_ok

    def paint_glass(self, p, r, rad, path=None):
        st = STYLES[self.style_name]
        if path is None:
            path = shape_path(r, rad)
        p.setPen(Qt.PenStyle.NoPen)
        if st["liquid"]:                                                    # box-shadow: 0 6px 6px rgba(0,0,0,.2), 0 0 20px rgba(0,0,0,.1)
            for i in range(1, 6):
                p.setBrush(QColor(0, 0, 10, int(30 * (1 - i / 6.0) ** 2)))
                q = r.adjusted(-i * 3.4, -i * 3.4, i * 3.4, i * 3.4)
                p.drawRoundedRect(q, rad + i * 3.4, rad + i * 3.4)
            for i in range(1, 4):
                p.setBrush(QColor(0, 0, 10, int(40 * (1 - i / 4.0) ** 2)))
                q = r.adjusted(-i * 1.6, -i * 1.6 + 6, i * 1.6, i * 1.6 + 6)
                p.drawRoundedRect(q, rad + i * 1.6, rad + i * 1.6)
        else:
            for i in range(1, 6):                                          # soft drop shadow
                p.setBrush(QColor(4, 6, 24, int(24 * (1 - i / 6.0) ** 2)))
                q = r.adjusted(-i * 3.0, -i * 2.4 + 6, i * 3.0, i * 3.6 + 6)
                p.drawRoundedRect(q, rad + i * 3.0, rad + i * 3.0)
        p.save()
        p.setClipPath(path)
        t = st["tint"]
        if self.bd is not None:
            p.fillRect(r, QColor(max(0, t[0] + 26), max(0, t[1] + 28), max(0, t[2] + 30)))   # visible only if a fast move outruns the capture
            p.drawImage(self.place(self.bd_at, r), self.bd)
        else:                                                               # no live backdrop yet / unsupported Windows: frosted gradient
            g0 = QLinearGradient(r.topLeft(), r.bottomLeft())
            g0.setColorAt(0, QColor(92, 100, 140, 235))
            g0.setColorAt(1, QColor(52, 58, 92, 235))
            p.fillRect(r, g0)
        # brighter backdrop -> deeper tint, so white text always stays readable
        lum = max(0.0, (self.bd_lum - 85) / 150)
        if st["liquid"]:
            p.fillRect(r, QColor(t[0], t[1], t[2], min(130, t[3] + int(105 * lum))))
            p.fillRect(r, QColor(255, 255, 255, 56))                        # the 25% white veil
        else:
            p.fillRect(r, QColor(t[0], t[1], t[2], min(165, t[3] + int(85 * lum))))
            g = QLinearGradient(r.topLeft(), r.bottomRight())               # light falling from the top-left
            g.setColorAt(0.0, QColor(255, 255, 255, int(255 * st["spec"])))
            g.setColorAt(0.5, QColor(255, 255, 255, 6))
            g.setColorAt(1.0, QColor(255, 255, 255, int(255 * st["spec"] * 0.35)))
            p.fillRect(r, g)
        p.restore()
        if st["liquid"]:
            # inset 2px 2px 1px rgba(255,255,255,.5)  and  inset -1px -1px 1px 1px rgba(255,255,255,.5)
            p.save()
            p.setClipPath(path)
            for dx in (2.0, -2.0):
                ring = QPainterPath()
                ring.setFillRule(Qt.FillRule.OddEvenFill)
                ring.addRect(r.adjusted(-6, -6, 6, 6))
                ring.addPath(path.translated(dx, dx))
                p.fillPath(ring, QColor(255, 255, 255, 128))
            p.restore()
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(255, 255, 255, 36), 1.0))
            p.drawPath(path)
            return
        rg = QLinearGradient(r.topLeft(), r.bottomRight())                   # frost: rim light
        rg.setColorAt(0.0, qcol(st["rim"]))
        rg.setColorAt(0.32, QColor(255, 255, 255, 16))
        rg.setColorAt(0.68, QColor(255, 255, 255, 10))
        rg.setColorAt(1.0, qcol(st["rim2"]))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QBrush(rg), 1.4))
        p.drawPath(shape_path(r.adjusted(0.7, 0.7, -0.7, -0.7), rad - 0.7))


# ----------------------------------------------------------------------------- save window
class SaveDialog(GlassWidget):
    def __init__(self, island, session, on_done):
        super().__init__(focusable=True)
        self.island, self.session, self.on_done = island, session, on_done
        self.style_name = island.style_name
        self.voices = session.voices()
        self.names_edit = {}
        self.rec_wavs, self.rec_video = session.recording()
        self.has_rec = bool(self.rec_wavs)
        self.keep, self.names_out = True, {}
        self.W = 470
        self.H = 336 + (46 + 46 * min(len(self.voices), 6) if len(self.voices) >= 2 else 0) + (58 if self.has_rec else 0)
        self.resize(self.W + 2 * M, self.H + 2 * M)
        self.setStyleSheet("""
            QLabel { color: rgba(255,255,255,235); background: transparent; }
            QLabel#h { font-size: 19px; font-weight: 700; }
            QLabel#sub, QLabel#cap { color: rgba(255,255,255,150); }
            QLabel#cap { font-size: 11px; letter-spacing: 1px; }
            QLineEdit { background: rgba(255,255,255,0.13); border: 1px solid rgba(255,255,255,0.24); border-radius: 11px;
                        padding: 9px 12px; color: white; font-size: 14px; selection-background-color: rgba(130,170,255,0.65); }
            QLineEdit:focus { border: 1px solid rgba(255,255,255,0.65); background: rgba(255,255,255,0.18); }
            QPushButton { background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.22); border-radius: 11px;
                          padding: 9px 16px; color: white; font-size: 13px; font-weight: 600; }
            QPushButton:hover { background: rgba(255,255,255,0.26); }
            QPushButton#save { background: rgba(255,255,255,0.93); color: #10131c; border: none; padding: 9px 26px; }
            QPushButton#save:hover { background: white; }
            QPushButton#discard { background: transparent; border: none; color: rgba(255,130,120,230); padding: 9px 6px; }
            QPushButton#discard:hover { color: #ff6b5f; }
            QPushButton#later { background: transparent; border: none; color: rgba(255,255,255,170); padding: 9px 10px; }
            QPushButton#later:hover { color: white; }
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(M + 26, M + 22, M + 26, M + 20)
        lay.setSpacing(6)
        h = QLabel("Save session"); h.setObjectName("h")
        names = ", ".join(session.langs_used()) or "no speech"
        sub = QLabel(f"{assistant.fmt_ts(session.duration)[3:] if session.duration < 3600 else assistant.fmt_ts(session.duration)}"
                     f"  ·  {len(session.segs)} lines  ·  {names}")
        sub.setObjectName("sub")
        lay.addWidget(h)
        lay.addWidget(sub)
        lay.addSpacing(10)
        c1 = QLabel("NAME"); c1.setObjectName("cap")
        self.name = QLineEdit(session.default_name())
        lay.addWidget(c1)
        lay.addWidget(self.name)
        lay.addSpacing(6)
        c2 = QLabel("SAVE TO"); c2.setObjectName("cap")
        lay.addWidget(c2)
        row = QHBoxLayout()
        self.folder = QLineEdit(island.cfg.get("save_dir") or str(Path(QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation)) / "Meeting Transcripts"))
        self.folder.setReadOnly(True)
        self.folder.setCursorPosition(0)
        b = QPushButton("Browse…")
        b.clicked.connect(self.browse)
        row.addWidget(self.folder, 1)
        row.addWidget(b)
        lay.addLayout(row)
        self.fmt = QLabel("Saved as .txt with a [hh:mm:ss] timestamp on every line")
        self.fmt.setObjectName("sub")
        lay.addSpacing(2)
        lay.addWidget(self.fmt)
        self.keep_cb = None
        if self.has_rec:                                       # the meeting was recorded: offer to keep it, and re-check the text
            mb = (session.duration * 8000 + (self.rec_video.stat().st_size if self.rec_video else 0)) / 1e6   # AAC audio ~8 KB/s
            lay.addSpacing(8)
            self.keep_cb = QCheckBox(f"Keep the recording next to the transcript   (about {max(1, int(mb))} MB)")
            self.keep_cb.setChecked(True)
            self.keep_cb.setStyleSheet("QCheckBox { color: rgba(255,255,255,235); font-size: 13px; spacing: 9px; }"
                                       "QCheckBox::indicator { width: 17px; height: 17px; border-radius: 5px;"
                                       " border: 1px solid rgba(255,255,255,0.5); background: rgba(255,255,255,0.10); }"
                                       "QCheckBox::indicator:checked { background: rgba(255,255,255,0.93); border: none; }")
            lay.addWidget(self.keep_cb)
            note = QLabel("The transcript is saved now. A few minutes later it is replaced by a final version "
                          "that listens to the recording again, more carefully.")
            note.setObjectName("sub"); note.setWordWrap(True)
            lay.addWidget(note)
        if len(self.voices) >= 2:                              # who is who: name the voices that were told apart
            lay.addSpacing(8)
            c3 = QLabel("WHO WAS SPEAKING   (optional - leave blank to keep Speaker 1, 2 ...)"); c3.setObjectName("cap")
            lay.addWidget(c3)
            for v in self.voices[:6]:
                first = next((g for g in session.segs if g.get("voice") == v), None)
                snippet = (first["text"] if first else "")[:34]
                col = VOICE_COLORS[(v - 1) % len(VOICE_COLORS)]
                r = QHBoxLayout()
                dot = QLabel("●"); dot.setStyleSheet(f"color: rgb({col[0]},{col[1]},{col[2]}); font-size: 16px;")
                e = QLineEdit(); e.setPlaceholderText(f"Speaker {v}")
                self.names_edit[v] = e
                hint = QLabel("“" + snippet + "…”"); hint.setObjectName("sub"); hint.setMaximumWidth(190)
                r.addWidget(dot); r.addWidget(e, 1); r.addWidget(hint)
                lay.addLayout(r)
        lay.addStretch(1)
        row2 = QHBoxLayout()
        self.bd_btn = QPushButton("Discard"); self.bd_btn.setObjectName("discard")
        self.bd_btn.clicked.connect(self.discard)
        later = QPushButton("Keep for later"); later.setObjectName("later")
        later.clicked.connect(lambda: self.finish("later"))
        save = QPushButton("Save"); save.setObjectName("save"); save.setDefault(True)
        save.clicked.connect(self.save)
        row2.addWidget(self.bd_btn)
        row2.addStretch(1)
        row2.addWidget(later)
        row2.addWidget(save)
        lay.addLayout(row2)
        self._armed = False
        self.setWindowOpacity(0.0)

    def browse(self):
        d = QFileDialog.getExistingDirectory(self, "Choose a folder", self.folder.text())
        if d:
            self.folder.setText(d)

    def discard(self):
        if not self._armed:
            self._armed = True
            self.bd_btn.setText("Click again to discard")
            QTimer.singleShot(3000, self._disarm)
            return
        self.finish("discard")

    def _disarm(self):
        self._armed = False
        self.bd_btn.setText("Discard")

    def save(self):
        folder = Path(self.folder.text())
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = bk.unique_path(folder, self.name.text().strip() or self.session.default_name())
            names = {v: e.text().strip() for v, e in self.names_edit.items() if e.text().strip()}
            self.names_out = names
            self.keep = bool(self.keep_cb.isChecked()) if self.keep_cb is not None else False
            body = bk.render_text(self.session.segs, path.stem, self.session.started, self.session.duration,
                                  self.island.roman, names)
            path.write_text(body, encoding="utf-8-sig")
            corrections.snapshot(path, body)
        except Exception as e:
            self.fmt.setText(f"Could not save: {e}")
            return
        self.finish("saved", path)

    def finish(self, action, path=None):
        if getattr(self, "_done", False):
            return
        self._done = True
        self.stop()
        self.close()
        self.on_done(action, path, self.folder.text())

    def stop(self):
        if self.backdrop:
            self.backdrop.stop_ev.set()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.finish("later")
        else:
            super().keyPressEvent(e)

    def popup(self):
        scr = QGuiApplication.primaryScreen().geometry()
        ir = self.island.shape_rect()
        x = min(max(8, ir.center().x() - self.W / 2), scr.width() - self.W - 8)
        y = ir.bottom() + 14
        if y + self.H > scr.height() - 40:
            y = max(8, ir.top() - self.H - 14)
        self.final_pos = QPoint(int(x - M), int(y - M))
        self.move(self.final_pos + QPoint(0, 14))
        self.show()
        self.exclude_from_capture()
        self.update_cap()
        self.start_backdrop()
        a = QPropertyAnimation(self, b"windowOpacity", self)
        a.setDuration(260); a.setStartValue(0.0); a.setEndValue(1.0); a.setEasingCurve(QEasingCurve.Type.OutCubic)
        b = QPropertyAnimation(self, b"pos", self)
        b.setDuration(420); b.setStartValue(self.pos()); b.setEndValue(self.final_pos); b.setEasingCurve(QEasingCurve.Type.OutBack)
        a.start(); b.start()
        self._anims = (a, b)
        self.activateWindow()
        self.name.setFocus()
        self.name.selectAll()
        self._t = QTimer(self)
        self._t.timeout.connect(lambda: (self.update_cap(), self.update()))
        self._t.start(40)

    def update_cap(self):
        dpr = self.devicePixelRatioF()
        g = self.geometry()
        self.cap_rect = (int((g.x() + M) * dpr), int((g.y() + M) * dpr), int(self.W * dpr), int(self.H * dpr))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.paint_glass(p, QRectF(M, M, self.W, self.H), 30)


def app_version():
    """'v1.1' - from VERSION.txt next to the program (written by the installer/updater)."""
    from paths import APP_DIR
    for name in ("VERSION.txt", "version.txt"):
        try:
            return "v" + (APP_DIR / name).read_text(encoding="ascii").splitlines()[0].strip()
        except Exception:
            pass
    return ""


# ----------------------------------------------------------------------------- the island
class Island(GlassWidget):
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.cfg = self.load_cfg()
        self.style_name = ARGS.style or self.cfg.get("style", "liquid")
        self.roman = (ARGS.script or self.cfg.get("script", "roman")) == "roman"
        self.source = "both" if ARGS.mic else self.cfg.get("source", "pc")        # pc | mic | both
        self.record = self.cfg.get("record", "screen")                            # screen | audio | off
        self.polish = None                                                        # progress of the final-transcript pass
        scr = QGuiApplication.primaryScreen()
        self.scr = scr.geometry()
        self.dpr = scr.devicePixelRatio()
        self.dock = None if ARGS.reset_pos else self.cfg.get("dock")
        self.ax = self.scr.width() / 2 if ARGS.reset_pos else self.cfg.get("ax", self.scr.width() / 2)
        self.ay = 8 if ARGS.reset_pos else self.cfg.get("ay", 8)
        self.expanded = not ARGS.collapsed
        self.dragging, self.press, self.pressed_btn, self.hover = False, None, None, None
        self.pointer = QPointF(0, 0)
        # springs: rect + radius, one per layer opacity, scroll, bars, hover
        r = self.target_rect()
        # the dynamic-island React component's motion: spring(stiffness 400, damping 30) for size, position and corner radius
        self.sx, self.sy = Spring(r[0], SPRING_K, SPRING_C), Spring(r[1], SPRING_K, SPRING_C)
        self.sw, self.sh = Spring(r[2], SPRING_K, SPRING_C), Spring(r[3], SPRING_K, SPRING_C)
        self.srad = Spring(self.shape_radius(), SPRING_K, SPRING_C)
        self.op = {k: Layer(k == self.shape()) for k in ("panel", "pill", "vbar", "circle")}
        # liquid-glass hover: the glass swells a few px with an elastic overshoot (cubic-bezier(.175,.885,.32,2.2) feel)
        self.hg, self.hovering = Spring(0.0, 150, 9.5), False
        self.win_x = self.win_y = None
        self.resize(PANEL[0] + 2 * M + 4, PANEL[1] + 2 * M + 4)       # fixed canvas: the window only moves, never resizes
        self.scroll = Spring(0, 120, 21)
        self.bars = [Spring(0, 520, 30) for _ in range(5)]
        self.hov = {}
        # state
        self.mode = "loading"
        self.status_msg = "Loading speech model…"
        self.segs, self.drafts, self.final_utt, self.cur_lang = [], {}, {}, None
        self.latest = ""
        self.notice = None            # ("Saved ...", path, until)
        self.unsaved = bk.find_unsaved()
        self.doc_pm, self.doc_h, self.text_dirty, self.doc_ver = None, 0.0, True, 0
        self.hits = []
        self.copied_until = 0.0
        self.quit_after = False
        self.dialog = None
        self.import_queue()
        self.f_ui = self.mkfont(13.5, 600)
        self.f_small = self.mkfont(12, 600)
        self.f_hint = self.mkfont(13.5, 400)
        self.f_who = self.mkfont(14, 700)
        self.f_body = self.mkfont(16.5, 500)
        self.f_ts = QFont("Consolas"); self.f_ts.setPixelSize(11)
        self.t_last = time.perf_counter()
        self.frame_stats, self.paint_stats, self._seen_bd = [], [], -1
        self.backend = bk.Backend(lambda *a: self.q.put(a), dict(
            langs=ARGS.langs, model=ARGS.model, hi_model=ARGS.hi_model, hotwords=ARGS.hotwords, threads=ARGS.threads,
            hi_threads=ARGS.hi_threads, window=ARGS.window, mic=ARGS.mic, sim=ARGS.sim, engine=ARGS.engine,
            speakers=not ARGS.no_speakers))
        self.finisher = finisher.Finisher(self.backend, lambda *a: self.q.put(a))
        self.finisher.start()
        self.finisher.resume_pending()                 # a final transcript that was still being prepared when the app closed
        self.apply_geometry()
        self.show()
        self.exclude_from_capture()
        self.start_backdrop()
        self.backend.load()
        QTimer.singleShot(2500, self.learn_from_edits)     # what was fixed in saved transcripts since last time
        self.vsync = VSync(self)                   # one frame per screen refresh (DwmFlush), not a free-running timer
        self.vsync.tick.connect(self._on_vsync, Qt.ConnectionType.QueuedConnection)
        self.vsync.start()
        self.poll_t = QTimer(self)
        self.poll_t.timeout.connect(self.poll)
        self.poll_t.start(30)

    # ------------------------------------------------------------------ small utilities
    def import_queue(self):
        import queue
        self.q = queue.Queue()

    def mkfont(self, px, weight):
        f = QFont()
        f.setFamilies(["Segoe UI Variable Text", "Segoe UI", "Nirmala UI"])
        f.setPixelSize(int(px))
        f.setWeight(QFont.Weight(weight))
        return f

    def load_cfg(self):
        try:
            return json.loads(SETTINGS.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_cfg(self):
        if ARGS.selftest or ARGS.tour:          # test runs must never overwrite the user's folder / position
            return
        try:
            self.cfg.update(dock=self.dock, ax=self.ax, ay=self.ay, style=self.style_name, script="roman" if self.roman else "native",
                            source=self.source, record=self.record)
            SETTINGS.write_text(json.dumps(self.cfg), encoding="utf-8")
        except Exception:
            pass

    def disp(self, t):
        return translit.to_roman(t) if self.roman and translit.has_devanagari(t) else t

    # ------------------------------------------------------------------ shapes & targets
    def shape(self):
        if self.dragging:
            return "circle"
        if self.expanded:
            return "panel"
        return "vbar" if self.dock else "pill"

    def shape_radius(self):
        s = self.shape()
        return 30 if s == "panel" else {"pill": PILL[1] / 2, "vbar": VBAR[0] / 2, "circle": CIRC / 2}[s]

    def target_rect(self):
        sw, sh = self.scr.width(), self.scr.height()
        cl = lambda v, a, b: max(a, min(b, v))
        if self.dragging:
            return [self.pointer.x() - CIRC / 2, self.pointer.y() - CIRC / 2, CIRC, CIRC]
        if self.expanded:
            w, h = PANEL[0], self.panel_h()
            if self.dock == "left":
                x, y = 10, self.ay - 80
            elif self.dock == "right":
                x, y = sw - w - 10, self.ay - 80
            else:
                x, y = self.ax - w / 2, self.ay
            return [cl(x, 8, sw - w - 8), cl(y, 6, sh - h - 56), w, h]
        if self.dock:
            w, h = VBAR
            return [8 if self.dock == "left" else sw - w - 8, cl(self.ay - h / 2, 8, sh - h - 56), w, h]
        w, h = PILL
        return [cl(self.ax - w / 2, 8, sw - w - 8), cl(self.ay, 4, sh - h - 56), w, h]

    def panel_h(self):
        """Compact while idle, full height once there is a transcript to show."""
        mode, segs = getattr(self, "mode", "loading"), getattr(self, "segs", [])
        if mode in ("ready", "loading", "error") and not segs:
            notice, unsaved = getattr(self, "notice", None), getattr(self, "unsaved", [])
            extra = (notice and time.time() < notice[2]) or (unsaved and mode == "ready") or bool(self.polish)
            return 252 if extra else 200
        return PANEL[1]

    def shape_rect(self):
        return QRect(int(self.sx.x), int(self.sy.x), int(self.sw.x), int(self.sh.x))

    # ------------------------------------------------------------------ frame loop (springs + window geometry)
    def frame(self):
        now = time.perf_counter()
        dt = min(0.05, now - self.t_last)
        self.t_last = now
        r = self.target_rect()
        tight = self.dragging
        for s, v in ((self.sx, r[0]), (self.sy, r[1]), (self.sw, r[2]), (self.sh, r[3])):
            s.t = v
        self.sx.k = self.sy.k = 900 if tight else SPRING_K
        self.sx.c = self.sy.c = 62 if tight else SPRING_C
        self.hg.t = 1.0 if (self.hovering and not self.dragging and self.shape() != "circle") else 0.0
        self.hg.step(dt)
        self.srad.t = self.shape_radius()
        sh = self.shape()
        for k, s in self.op.items():
            s.t = 1.0 if k == sh else 0.0
        for s in (self.sx, self.sy, self.sw, self.sh, self.srad, self.scroll, *self.op.values(), *self.hov.values()):
            s.step(dt)
        lv, _ = self.backend.levels()
        active = self.mode == "recording"
        for i, b in enumerate(self.bars):
            lvl = min(1.0, math.sqrt(lv * 7)) if active else 0.0
            b.t = lvl * (0.55 + 0.45 * math.sin(now * 9 + i * 1.4)) if lvl > 0.02 else 0.0
            b.step(dt)
        self.apply_geometry()
        if self.text_dirty and self.op["panel"].x > 0.02:
            self.rebuild_doc()
        springs = (self.sx, self.sy, self.sw, self.sh, self.srad, self.scroll, self.hg, *self.op.values(), *self.hov.values())
        moving = self.moving = self.dragging or not all(s.settled(0.3) for s in springs)
        busy = self.mode in ("recording", "loading", "ending", "saving") or any(abs(b.x) > 0.02 for b in self.bars)
        if moving or busy or self.bd_ver != self._seen_bd:
            self._seen_bd = self.bd_ver
            self.update()
        self.frame_stats.append(dt)
        self.paint_stats.append(getattr(self, "paint_ms", 0.0))

    def _on_vsync(self):
        try:
            self.frame()
        finally:
            self.vsync.pending = False

    def origin(self):
        return float(self.win_x or 0), float(self.win_y or 0)

    def apply_geometry(self):
        x, y, w, h = self.sx.x, self.sy.x, max(CIRC * 0.6, self.sw.x), max(CIRC * 0.6, self.sh.x)
        wx, wy = math.floor(x - M), math.floor(y - M)
        if (wx, wy) != (self.win_x, self.win_y):
            self.move(wx, wy)
            self.win_x, self.win_y = wx, wy
        self.rr = QRectF(x - wx, y - wy, w, h)                             # the glass shape, sub-pixel exact
        d = self.devicePixelRatioF()
        hgs = getattr(self, "hg", None)
        gx = 6.0 * max(0.0, hgs.x if hgs else 0.0)
        tx, ty, tw, th = self.target_rect()                                 # where the glass is heading: capture that too
        ux0, uy0 = min(x - gx, tx), min(y - gx, ty)
        ux1, uy1 = max(x + w + gx, tx + tw), max(y + h + gx, ty + th)
        self.cap_rect = (int(round(ux0 * d)), int(round(uy0 * d)), int(round((ux1 - ux0) * d)), int(round((uy1 - uy0) * d)))

    # ------------------------------------------------------------------ painting
    def paintEvent(self, e):
        t0 = time.perf_counter()
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing |
                         QPainter.RenderHint.SmoothPixmapTransform)
        g = max(-0.15, self.hg.x)                                  # hover swell (overshoots, so it feels elastic)
        r = self.rr.adjusted(-5.0 * g, -5.0 * g, 5.0 * g, 5.0 * g)
        rad = min(self.srad.x + 8.0 * g, r.height() / 2, r.width() / 2)
        path = shape_path(r, rad)
        self.paint_glass(p, r, rad, path)
        p.setClipPath(path)
        self.hits = []
        native = {"panel": PANEL, "pill": PILL, "vbar": VBAR, "circle": (CIRC, CIRC)}
        for k in ("panel", "pill", "vbar", "circle"):
            lay = self.op[k]
            o = max(0.0, min(1.0, lay.x))
            if o < 0.02:
                continue
            nw, nh = native[k][0], (self.panel_h() if k == "panel" else native[k][1])
            fit = max(0.6, min(1.0, r.width() / nw, r.height() / nh))   # content grows/shrinks with the glass
            if lay.exiting:                                        # exit: fade, blur 10 px, shrink to .95, drop 20 px
                sc, dy, blur = 0.95 + 0.05 * lay.x, 20.0 * (1 - lay.x), 10.0 * (1 - lay.x)
            else:                                                  # enter: scale .9 -> 1, rise 5 px, fade in
                sc, dy, blur = 0.9 + 0.1 * min(1.2, lay.x), 5.0 * (1 - lay.x), 0.0
            sc *= 1.0 - 0.04 * max(0.0, g)                         # hover:scale-95 in the liquid-glass button: content squishes in
            live = (not lay.exiting) and o > 0.5 and fit > 0.97
            c = r.center()
            if blur > 0.8:                                         # exiting content is rendered to an image and blurred
                self._draw_blurred(p, k, r, c, sc, dy, fit, blur, o)
                continue
            p.save()
            p.setOpacity(o)
            p.translate(c.x(), c.y() + dy)
            p.scale(sc, sc)
            p.translate(-c.x(), -c.y())
            if fit < 0.999:
                anchor = r.center() if k in ("circle", "vbar") else r.topLeft()
                p.translate(anchor)
                p.scale(fit, fit)
                p.translate(-anchor)
            getattr(self, "draw_" + k)(p, r, live)
            p.restore()
        p.setOpacity(1.0)
        self.paint_ms = (time.perf_counter() - t0) * 1000

    def _draw_blurred(self, p, k, r, c, sc, dy, fit, blur, o):
        """Draw one layer into an offscreen image, blur it (down/up-scale) and composite it: the exit blur(10px) of the React island."""
        dpr = self.devicePixelRatioF()
        img = QImage(int(self.width() * dpr), int(self.height() * dpr), QImage.Format.Format_ARGB32_Premultiplied)
        img.setDevicePixelRatio(dpr)
        img.fill(0)
        ip = QPainter(img)
        ip.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing |
                          QPainter.RenderHint.SmoothPixmapTransform)
        ip.translate(c.x(), c.y() + dy)
        ip.scale(sc, sc)
        ip.translate(-c.x(), -c.y())
        if fit < 0.999:
            anchor = r.center() if k in ("circle", "vbar") else r.topLeft()
            ip.translate(anchor)
            ip.scale(fit, fit)
            ip.translate(-anchor)
        saved, self.hits = self.hits, []                          # an exiting layer must not register clickable buttons
        getattr(self, "draw_" + k)(ip, r, False)
        self.hits = saved
        ip.end()
        f = 1.0 + blur / 3.0
        small = img.scaled(max(1, int(img.width() / f)), max(1, int(img.height() / f)), Qt.AspectRatioMode.IgnoreAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
        p.save()
        p.setOpacity(o)
        p.drawImage(QRectF(0, 0, self.width(), self.height()), small)
        p.restore()

    def now_s(self):
        return time.time()

    def dot(self, p, cx, cy, big=False):
        col = {"loading": QColor(255, 179, 64), "ready": QColor(150, 155, 170), "ending": QColor(255, 179, 64),
               "saving": QColor(255, 179, 64), "error": QColor(255, 69, 58)}.get(self.mode)
        if self.mode in ("recording", "paused"):
            col = QColor(255, 69, 58) if self.mode == "recording" else QColor(150, 155, 170)
            if self.mode == "recording":
                ph = (self.now_s() * 0.9) % 1.0
                ring = QColor(255, 69, 58, int(110 * (1 - ph)))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(ring)
                rr = 5 + 9 * ph
                p.drawEllipse(QPointF(cx, cy), rr, rr)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col)
        p.drawEllipse(QPointF(cx, cy), 5.2, 5.2)

    def status_parts(self):
        """-> [(text, muted?)]  Plain text: the language is shown here as information, never as a button."""
        m = self.mode
        if m == "loading":
            return [(self.status_msg, False)]
        if m == "ready":
            if self.polish and self.polish.get("state") == "working":
                return [("Ready", False), (f"  ·  final transcript {int(self.polish.get('frac', 0) * 100)}%", True)]
            return [("Ready", False)]
        if m == "ending":
            return [("Finishing the last sentences…", False)]
        if m == "saving":
            return [("Saving session…", False)]
        if m == "error":
            return [(self.status_msg, False)]
        el = assistant.fmt_ts(self.backend.elapsed())[3:]
        _, sig = self.backend.levels()
        since = time.monotonic() - max(sig, self.backend.T0)
        if m == "recording" and since > 10:
            if sig == 0:
                return [("No sound heard — is the microphone muted?" if self.source == "mic"
                         else "No sound heard — is the volume muted?", False)]
        parts = [("Listening" if m == "recording" else "Paused", False), ("  ·  " + el, True)]
        if self.cur_lang:
            parts.append(("  ·  " + self.cur_lang.upper(), True))
        if not self.backend.hindi_ready():
            parts.append(("  ·  loading Hindi model…", True))
        return parts

    def draw_status(self, p, x, cy, font, maxw=10_000):
        fm = QFontMetricsF(font)
        p.setFont(font)
        for txt, muted in self.status_parts():
            col = QColor(255, 255, 255, 150) if muted else QColor(255, 255, 255, 245)
            if txt.startswith("No sound"):
                col = QColor(255, 191, 90)
            p.setPen(QColor(0, 0, 20, 70)); p.drawText(QPointF(x + 0.5, cy + 5.3), txt)
            p.setPen(col)
            p.drawText(QPointF(x, cy + 4.8), txt)
            x += fm.horizontalAdvance(txt)
        return x

    def draw_bars(self, p, x, cy, horizontal=False):
        c = STYLES[self.style_name]["bar"]
        p.setPen(Qt.PenStyle.NoPen)
        on = self.mode == "recording"
        for i, b in enumerate(self.bars):
            amp = 3 + 15 * max(0.0, b.x)
            col = QColor(c[0], c[1], c[2], 255 if (on and b.x > 0.04) else 70)
            p.setBrush(col)
            if horizontal:
                y = cy + i * 15
                p.drawRoundedRect(QRectF(x - amp / 2, y - 1.5, amp, 3), 1.5, 1.5)
            else:
                xx = x + i * 7
                p.drawRoundedRect(QRectF(xx - 1.5, cy - amp / 2, 3, amp), 1.5, 1.5)

    # --- buttons
    def hv(self, name):
        s = self.hov.get(name)
        if s is None:
            s = self.hov[name] = Spring(0, 420, 34)
        s.t = 1.0 if self.hover == name else 0.0
        return max(0.0, min(1.0, s.x))

    def icon_btn(self, p, name, kind, cx, cy, active=True):
        rect = QRectF(cx - 15, cy - 15, 30, 30)
        h = self.hv(name)
        p.setPen(QPen(QColor(255, 255, 255, 46 + int(40 * h)), 1))
        p.setBrush(QColor(255, 255, 255, 34 + int(44 * h)))
        p.drawEllipse(rect.adjusted(-h, -h, h, h))
        self.paint_icon(p, kind, cx, cy, QColor(255, 255, 255, 240))
        if active:
            self.hits.append((name, rect))

    def paint_icon(self, p, kind, cx, cy, col, scale=1.0):
        pen = QPen(col, 1.7 * scale)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap); pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.save()
        p.translate(cx, cy)
        p.scale(scale, scale)
        if kind == "pause":
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(col)
            p.drawRoundedRect(QRectF(-5.5, -6, 3.6, 12), 1.5, 1.5); p.drawRoundedRect(QRectF(1.9, -6, 3.6, 12), 1.5, 1.5)
        elif kind == "play":
            path = QPainterPath(); path.moveTo(-3.4, -6.4); path.lineTo(-3.4, 6.4); path.lineTo(6.2, 0); path.closeSubpath()
            p.setPen(QPen(col, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)); p.setBrush(col)
            p.drawPath(path)
        elif kind == "stop":
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(col)
            p.drawRoundedRect(QRectF(-5, -5, 10, 10), 2.4, 2.4)
        elif kind == "copy":
            p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(QRectF(-2, -2, 8, 8), 2, 2)
            p.drawArc(QRectF(-6, -6, 8, 8), 90 * 16, 90 * 16); p.drawLine(QPointF(-4, -6), QPointF(0.5, -6)); p.drawLine(QPointF(-6, -4), QPointF(-6, 0.5))
        elif kind == "min":
            p.setPen(pen); p.drawPolyline([QPointF(-5, 2.5), QPointF(0, -2.5), QPointF(5, 2.5)])
        elif kind == "close":
            p.setPen(pen); p.drawLine(QPointF(-4.5, -4.5), QPointF(4.5, 4.5)); p.drawLine(QPointF(4.5, -4.5), QPointF(-4.5, 4.5))
        elif kind == "folder":
            p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
            path = QPainterPath(); path.moveTo(-6, -3); path.lineTo(-6, 4); path.quadTo(-6, 5.5, -4.5, 5.5); path.lineTo(4.5, 5.5)
            path.quadTo(6, 5.5, 6, 4); path.lineTo(6, -2); path.quadTo(6, -3.5, 4.5, -3.5); path.lineTo(0.5, -3.5); path.lineTo(-1, -5.5)
            path.lineTo(-4.5, -5.5); path.quadTo(-6, -5.5, -6, -4); path.closeSubpath(); p.drawPath(path)
        p.restore()

    def pill_btn(self, p, name, label, kind, x_right, cy, tone):
        fm = QFontMetricsF(self.f_ui)
        w = fm.horizontalAdvance(label) + 52
        rect = QRectF(x_right - w, cy - 16, w, 32)
        h = self.hv(name)
        if tone == "primary":
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(255, 255, 255, 228 + int(27 * h))); tc = QColor(14, 17, 30)
        else:
            g = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            g.setColorAt(0, QColor(255, 94, 84, 235)); g.setColorAt(1, QColor(236, 60, 52, 235))
            p.setPen(QPen(QColor(255, 255, 255, 60), 1)); p.setBrush(QBrush(g)); tc = QColor(255, 255, 255)
        rr = rect.adjusted(-h * 1.5, -h * 1.5, h * 1.5, h * 1.5)
        p.drawRoundedRect(rr, 16, 16)
        self.paint_icon(p, kind, rect.left() + 20, cy, tc, 0.8)
        p.setFont(self.f_ui); p.setPen(tc)
        p.drawText(QPointF(rect.left() + 34, cy + 4.8), label)
        self.hits.append((name, rect))
        return rect.left() - 8

    # --- layers
    def draw_panel(self, p, r, live):
        pr = QRectF(r.left(), r.top(), PANEL[0], PANEL[1])
        if live:
            self.hits = []
        hy = pr.top() + 28
        self.dot(p, pr.left() + 24, hy)
        x = self.draw_status(p, pr.left() + 42, hy, self.f_ui)
        if self.mode in ("recording", "paused"):
            self.draw_bars(p, x + 16, hy)
        # buttons, right to left
        rx = pr.right() - 20
        m = self.mode
        if m in ("recording", "paused"):
            rx = self.pill_btn(p, "end", "End session", "stop", rx, hy, "danger")
            rx -= 4
        elif m == "ready":
            rx = self.pill_btn(p, "start", "Start session", "play", rx, hy, "primary")
            rx -= 4
        if m in ("ready", "loading", "error"):
            self.icon_btn(p, "close", "close", rx - 11, hy); rx -= 40
        self.icon_btn(p, "min", "min", rx - 11, hy, True); rx -= 38
        if m in ("recording", "paused"):
            self.icon_btn(p, "copy", "copy", rx - 11, hy); rx -= 38
            self.icon_btn(p, "toggle", "play" if m == "paused" else "pause", rx - 11, hy)
        p.setPen(QPen(QColor(255, 255, 255, 34), 1))
        p.drawLine(QPointF(pr.left() + 22, pr.top() + 56), QPointF(pr.right() - 22, pr.top() + 56))
        body = QRectF(pr.left() + 24, pr.top() + 64, BODY_W, BODY_H)
        if m in ("recording", "paused", "ending", "saving") or self.segs:
            self.draw_transcript(p, body)
        else:
            self.draw_hint(p, body)

    def draw_hint(self, p, body):
        p.setFont(self.f_body)
        lines = []
        if self.mode == "loading":
            lines = ["Getting the speech models ready.", "This takes a few seconds the first time."]
        elif self.mode == "error":
            lines = ["Something went wrong — see island.log."]
        else:
            lines = ["Press Start session to begin.", "Whatever your PC is playing is transcribed live, in English, Hindi and Hinglish.",
                     "Nothing is saved until you press End session and choose a name and folder."]
        y = body.top() + 6
        for i, t in enumerate(lines):
            p.setPen(QColor(255, 255, 255, 240 if i == 0 else 160))
            p.setFont(self.f_body if i == 0 else self.f_hint)
            p.drawText(QRectF(body.left(), y, body.width(), 40), int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap), t)
            y += 34 if i == 0 else 26
        if self.polish and self.polish.get("state") == "working" and self.mode == "ready":
            y += 8
            p.setPen(QColor(150, 200, 255, 240)); p.setFont(self.f_ui)
            p.drawText(QPointF(body.left(), y + 14), f"Preparing the final transcript  ·  {int(self.polish.get('frac', 0) * 100)}%")
            bar = QRectF(body.left(), y + 26, 300, 5)
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(255, 255, 255, 40)); p.drawRoundedRect(bar, 2.5, 2.5)
            p.setBrush(QColor(150, 200, 255, 230))
            p.drawRoundedRect(QRectF(bar.left(), bar.top(), max(5.0, bar.width() * float(self.polish.get("frac", 0))), 5), 2.5, 2.5)
        elif self.notice and time.time() < self.notice[2]:
            y += 8
            p.setPen(QColor(150, 255, 190, 240)); p.setFont(self.f_ui)
            kind = self.notice[0]
            if kind == "learned":
                n = int(self.notice[1])
                label = f"Learned from your edits  ·  {n} new correction{'s' if n != 1 else ''}"
            else:
                label = ("Final transcript ready  ·  " if kind == "final" else "Saved  ·  ") + Path(self.notice[1]).name
            p.drawText(QPointF(body.left(), y + 14), label)
            r = QRectF(body.left(), y + 26, 120, 30)
            self.link_btn(p, "openfolder", "Review" if kind == "learned" else "Open folder", r)
        elif self.unsaved and self.mode == "ready":
            y += 8
            p.setPen(QColor(255, 210, 120, 240)); p.setFont(self.f_ui)
            n = len(self.unsaved)
            p.drawText(QPointF(body.left(), y + 14), f"{n} unsaved session{'s' if n > 1 else ''} from earlier")
            self.link_btn(p, "recover", "Recover", QRectF(body.left(), y + 26, 100, 30))

    def link_btn(self, p, name, label, rect):
        h = self.hv(name)
        p.setPen(QPen(QColor(255, 255, 255, 70), 1)); p.setBrush(QColor(255, 255, 255, 30 + int(40 * h)))
        p.drawRoundedRect(rect, 15, 15)
        p.setPen(QColor(255, 255, 255, 240)); p.setFont(self.f_small)
        p.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), label)
        self.hits.append((name, rect))

    def draw_transcript(self, p, body):
        if self.doc_pm is None:
            return
        sc = round(self.scroll.x * 2) / 2                      # half-pixel steps: reuse the cached layer when nothing moved
        key = (self.doc_ver, sc)
        if key != getattr(self, "_body_key", None):
            dpr = self.devicePixelRatioF()
            tmp = QPixmap(int(BODY_W * dpr), int(BODY_H * dpr))
            tmp.setDevicePixelRatio(dpr)
            tmp.fill(Qt.GlobalColor.transparent)
            tp = QPainter(tmp)
            tp.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            sy, dy = max(0.0, sc), max(0.0, -sc)
            hh = min(BODY_H - dy, self.doc_h - sy)
            tp.drawPixmap(QRectF(0, dy, BODY_W, hh), self.doc_pm, QRectF(0, sy * dpr, BODY_W * dpr, hh * dpr))
            tp.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
            g = QLinearGradient(0, 0, 0, BODY_H)
            g.setColorAt(0, QColor(0, 0, 0, 0)); g.setColorAt(min(1.0, 30 / BODY_H), QColor(0, 0, 0, 255)); g.setColorAt(1, QColor(0, 0, 0, 255))
            tp.fillRect(QRectF(0, 0, BODY_W, BODY_H), g)
            tp.end()
            self._body_key, self._body_pm = key, tmp
        p.drawPixmap(body.topLeft(), self._body_pm)

    def draw_pill(self, p, r, live):
        pr = QRectF(r.left(), r.top(), PILL[0], PILL[1])
        if live:
            self.hits = []
        cy = pr.top() + PILL[1] / 2
        self.dot(p, pr.left() + 23, cy)
        if self.mode == "ready":
            fm = QFontMetricsF(self.f_ui)
            p.setFont(self.f_ui); p.setPen(QColor(255, 255, 255, 235))
            p.drawText(QPointF(pr.left() + 42, cy + 4.8), "Meeting Island")
            if self.polish and self.polish.get("state") == "working":
                fr = float(self.polish.get("frac", 0))
                p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(150, 200, 255, 210))
                p.drawRoundedRect(QRectF(pr.left() + 42, pr.bottom() - 9, (PILL[0] - 120) * max(0.02, fr), 2.5), 1.25, 1.25)
            self.start_dot(p, pr.right() - 30, cy)
            return
        text = self.latest or "".join(t for t, _ in self.status_parts())
        fm = QFontMetricsF(self.f_ui)
        maxw = PILL[0] - 42 - 74
        p.setFont(self.f_ui)
        el = fm.elidedText(text, Qt.TextElideMode.ElideLeft, maxw)
        p.setPen(QColor(0, 0, 20, 70)); p.drawText(QPointF(pr.left() + 42.5, cy + 5.3), el)
        p.setPen(QColor(255, 255, 255, 240 if self.latest else 150)); p.drawText(QPointF(pr.left() + 42, cy + 4.8), el)
        if self.mode in ("recording", "paused"):
            self.draw_bars(p, pr.right() - 52, cy)

    def start_dot(self, p, cx, cy):
        rect = QRectF(cx - 15, cy - 15, 30, 30)
        h = self.hv("start")
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(255, 255, 255, 225 + int(30 * h)))
        p.drawEllipse(rect.adjusted(-h, -h, h, h))
        self.paint_icon(p, "play", cx + 0.5, cy, QColor(14, 17, 30), 0.85)
        self.hits.append(("start", rect))

    def draw_vbar(self, p, r, live):
        pr = QRectF(r.left(), r.top(), VBAR[0], VBAR[1])
        if live:
            self.hits = []
        cx = pr.left() + VBAR[0] / 2
        self.dot(p, cx, pr.top() + 22)
        if self.mode == "ready":
            self.start_dot(p, cx, pr.top() + 70)
        else:
            self.draw_bars(p, cx, pr.top() + 56, horizontal=True)

    def draw_circle(self, p, r, live):
        c = r.center()
        if self.mode == "ready":
            self.paint_icon(p, "play", c.x() + 1, c.y(), QColor(255, 255, 255, 240), 1.2)
        else:
            self.draw_bars(p, c.x() - 14, c.y())

    # ------------------------------------------------------------------ transcript document
    def rebuild_doc(self):
        self.text_dirty = False
        doc = QTextDocument()
        doc.setDocumentMargin(0)
        doc.setDefaultFont(self.f_body)
        doc.setTextWidth(BODY_W)
        cur = QTextCursor(doc)
        first = True
        segs = self.segs[-60:]
        IND = 52

        def block():
            nonlocal first
            bf = QTextBlockFormat()
            bf.setLeftMargin(IND); bf.setTextIndent(-IND); bf.setBottomMargin(9); bf.setLineHeight(128, 1)
            bf.setTabPositions([QTextOption.Tab(IND, QTextOption.TabType.LeftTab)])
            if first:
                cur.setBlockFormat(bf); first = False
            else:
                cur.insertBlock(bf)

        def fmt(color, italic=False, shaky=False, font=None):
            f = QTextCharFormat()
            f.setForeground(QColor(*color))
            if font:
                f.setFont(font)
            else:
                f.setFont(self.f_body)
            f.setFontItalic(italic)
            if shaky:
                f.setUnderlineStyle(QTextCharFormat.UnderlineStyle.WaveUnderline)
                f.setUnderlineColor(QColor(255, 191, 90))
            return f

        for i, s in enumerate(segs):
            block()
            newest = i == len(segs) - 1 and not self.drafts
            col = (255, 255, 255, 255) if newest else (255, 255, 255, 196)
            cur.insertText(assistant.fmt_ts(s["start"])[3:] + "\t", fmt((255, 255, 255, 105), font=self.f_ts))
            who = bk.speaker_label(s, self.segs)
            if who:                                         # coloured speaker label, only when there is someone to tell apart
                v = s.get("voice") or 1
                colr = (255, 255, 255) if (s["speaker"] == "Mic" and not s.get("voice")) else VOICE_COLORS[(v - 1) % len(VOICE_COLORS)]
                lf = QTextCharFormat(); lf.setForeground(QColor(colr[0], colr[1], colr[2], 235)); lf.setFont(self.f_who)
                cur.insertText(who + "  ", lf)
            words = s.get("words") or []
            if not self.roman and s.get("native"):
                cur.insertText(s["native"].strip(), fmt(col))
            elif words:
                for wi, (w, pr) in enumerate(words):
                    w = self.disp(w)
                    cur.insertText(w.lstrip() if wi == 0 else w, fmt(col, shaky=pr < 0.4))
            else:
                cur.insertText(self.disp(s["text"]).lstrip(), fmt(col))
            if s["speaker"] != "System":
                cur.insertText(f"   ‹{s['speaker']}›", fmt((255, 255, 255, 105), font=self.f_ts))
        for lane, d in self.drafts.items():
            block()
            cur.insertText("\t" + self.disp(d["text"]).strip() + " ▍", fmt((255, 255, 255, 125), italic=True))
        self.doc_h = max(1.0, doc.size().height())
        dpr = self.devicePixelRatioF()
        pm = QPixmap(int(BODY_W * dpr), int(math.ceil(self.doc_h) * dpr) + 2)
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        pp = QPainter(pm)
        pp.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        doc.drawContents(pp)
        pp.end()
        self.doc_pm = pm
        self.doc_ver += 1
        self.scroll.t = self.doc_h - BODY_H if self.doc_h > BODY_H else -(BODY_H - self.doc_h)

    # ------------------------------------------------------------------ events from the backend
    def poll(self):
        try:
            for _ in range(80):
                kind, *rest = self.q.get_nowait()
                getattr(self, "on_" + kind, lambda *a: None)(*rest)
        except Exception as e:
            if type(e).__name__ != "Empty":
                traceback.print_exc()

    def on_models(self, ok):
        if self.mode == "loading":
            self.mode = "ready"
            if ARGS.autostart:
                self.start()

    def learn_from_edits(self):
        try:
            new = corrections.learn_pending()
        except Exception:
            traceback.print_exc()
            return
        if new:
            print("[learn] new rules:", new, flush=True)
            if self.mode in ("ready", "loading"):
                self.notice = ("learned", str(len(new)), time.time() + 3600)
                self.text_dirty = True

    def on_polish(self, info):
        st = info.get("state")
        if st == "working":
            self.polish = info
        else:
            self.polish = None
            if st == "done":
                self.notice = ("final", info.get("path") or "", time.time() + 3600)
            self.text_dirty = True

    def on_error(self, msg):
        self.mode, self.status_msg = "error", msg

    def on_seg(self, s):
        sid = s.get("id")
        if sid:                                     # a forced-cut piece grew into a longer sentence: replace, don't add a line
            self.segs = [x for x in self.segs if x.get("id") != sid]
        self.segs.append(s)
        self.segs.sort(key=lambda x: x["start"])
        self.drafts.pop(s["speaker"], None)
        self.final_utt[s["speaker"]] = max(self.final_utt.get(s["speaker"], -1), s.get("utt", -1))
        self.latest = self.seg_disp(s)
        self.cur_lang = s.get("lang") or self.cur_lang
        self.text_dirty = True

    def seg_disp(self, s):
        if not self.roman and s.get("native"):
            return s["native"].strip()
        return self.disp("".join(w for w, _ in s["words"]) if s.get("words") else s["text"]).strip()

    def on_draft(self, lane, utt, d):
        if utt <= self.final_utt.get(lane, -1):
            return
        self.drafts[lane] = d
        self.latest = self.disp(d["text"]).strip()
        self.cur_lang = d.get("lang") or self.cur_lang
        self.text_dirty = True

    def on_clear(self, lane, utt, _):
        if self.drafts.pop(lane, None) is not None:
            self.text_dirty = True

    def on_ended(self, sess):
        self.drafts.clear()
        self.segs = list(sess.segs)                 # speaker labels were re-checked across the whole session
        self.text_dirty = True
        self.mode = "saving"
        self.text_dirty = True
        self.dialog = SaveDialog(self, sess, lambda action, path, folder, s=sess: self.after_save(s, action, path, folder))
        self.dialog.popup()

    def after_save(self, sess, action, path, folder):
        dlg, self.dialog = self.dialog, None
        self.cfg["save_dir"] = folder
        self.mode = "ready"
        self.segs, self.drafts, self.latest, self.cur_lang, self.final_utt = [], {}, "", None, {}
        self.text_dirty = True
        if action == "saved":
            recorded = bool(sess.recording()[0])
            if recorded:
                try:                                    # the recording is kept and the transcript re-checked in the background
                    self.finisher.submit(sess, path, dlg.names_out if dlg else {}, dlg.keep if dlg else False, self.roman)
                    self.polish = dict(state="working", frac=0.0, name=Path(path).stem, msg="Starting…")
                except Exception:
                    traceback.print_exc()
                    sess.discard()
            else:
                sess.discard()
            self.notice = ("saved", str(path), time.time() + 3600)
            self.last_saved = path
        elif action == "discard":
            sess.discard()
        self.unsaved = bk.find_unsaved()
        self.save_cfg()
        if self.quit_after:
            self.quit_app()

    # ------------------------------------------------------------------ actions
    def start(self):
        if self.mode != "ready":
            return
        self.segs, self.drafts, self.latest, self.cur_lang, self.final_utt = [], {}, "", None, {}
        self.notice = None
        self.text_dirty = True
        try:
            corrections.learn_pending()                    # fixes made to a saved transcript a moment ago count for this session
        except Exception:
            traceback.print_exc()
        self.backend.cfg["source"] = self.source
        self.backend.cfg["record"] = self.record
        if self.backend.start_session():
            self.mode = "recording"
        else:
            self.mode = "error"

    def on_button(self, name):
        m = self.mode
        if name == "start":
            self.start()
        elif name == "toggle":
            if m == "recording":
                self.backend.pause(); self.mode = "paused"
            elif m == "paused":
                self.backend.resume(); self.mode = "recording"
        elif name == "end":
            self.mode = "ending"
            self.backend.end_session()
        elif name == "copy":
            QApplication.clipboard().setText(self.all_text())
        elif name == "min":
            self.expanded = False
        elif name == "close":
            self.quit_app()
        elif name == "openfolder":
            if self.notice and self.notice[0] == "learned":
                os.startfile(str(corrections.ensure_file()))
            elif self.notice:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.notice[1]).parent)))
        elif name == "recover":
            if self.unsaved:
                sess = bk.load_unsaved(self.unsaved[-1])
                self.mode = "saving"
                self.dialog = SaveDialog(self, sess, lambda a, p_, f, s=sess: self.after_save(s, a, p_, f))
                self.dialog.popup()

    def all_text(self):
        return "\n".join(f"[{assistant.fmt_ts(s['start'])}] " + (lambda w: f"{w}: " if w else "")(bk.speaker_label(s, self.segs)) + self.seg_disp(s)
                         for s in self.segs)

    def quit_app(self):
        if self.mode in ("recording", "paused"):
            self.quit_after = True
            self.mode = "ending"
            self.backend.end_session()
            return
        self.backend.shutdown()
        self.finisher.shutdown()
        for w in (self, self.dialog):
            if w is not None and w.backdrop:
                w.backdrop.stop_ev.set()
        self.save_cfg()
        self.vsync.requestInterruption()
        self.vsync.wait(200)
        QApplication.quit()

    # ------------------------------------------------------------------ mouse: buttons, click, drag
    def hit(self, pos):
        for name, rect in reversed(self.hits):
            if rect.adjusted(-3, -3, 3, 3).contains(pos):
                return name
        return None

    def drag_region(self, pos):
        if self.shape() != "panel":
            return True
        return pos.y() < self.rr.top() + 56

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        pos = e.position()
        name = self.hit(pos)
        if name:
            self.pressed_btn = name
            return
        if self.drag_region(pos):
            g = e.globalPosition()
            self.press = dict(x=g.x(), y=g.y(), moved=False)

    def mouseMoveEvent(self, e):
        pos = e.position()
        self.hovering = self.rr.contains(pos)
        if not self.press:
            h = self.hit(pos)
            if h != self.hover:
                self.hover = h
                self.setCursor(Qt.CursorShape.PointingHandCursor if h else Qt.CursorShape.ArrowCursor)
            return
        g = e.globalPosition()
        if not self.press["moved"]:
            if abs(g.x() - self.press["x"]) + abs(g.y() - self.press["y"]) < 7:
                return
            self.press["moved"] = True
            self.dragging = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self.pointer = QPointF(g.x(), g.y())

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        pos = e.position()
        if self.pressed_btn:
            n, self.pressed_btn = self.pressed_btn, None
            if self.hit(pos) == n:
                self.on_button(n)
            return
        p, self.press = self.press, None
        if not p:
            return
        self.setCursor(Qt.CursorShape.ArrowCursor)
        if p["moved"]:
            g = e.globalPosition()
            self.end_drag(g.x(), g.y())
        elif not self.expanded:
            self.expanded = True

    def end_drag(self, px, py):
        sw = self.scr.width()
        self.dragging = False
        if px <= 110:
            self.dock, self.ay = "left", py
        elif px >= sw - 110:
            self.dock, self.ay = "right", py
        elif abs(px - sw / 2) <= 170 and py <= 130:
            self.dock, self.ax, self.ay = None, sw / 2, 8
        else:
            self.dock, self.ax, self.ay = None, px, py - 22
        self.save_cfg()

    def leaveEvent(self, e):
        self.hover = None
        self.hovering = False

    def contextMenuEvent(self, e):
        m = QMenu(self)
        m.setStyleSheet("QMenu{background:#1b1e2b;color:#eef0f8;border:1px solid #333a4f;border-radius:10px;padding:6px;}"
                        "QMenu::item{padding:7px 22px 7px 14px;border-radius:7px;}QMenu::item:selected{background:#2c3350;}"
                        "QMenu::separator{height:1px;background:#333a4f;margin:5px 8px;}")
        a = m.addAction(f"Meeting Island {app_version()}"); a.setEnabled(False)
        m.addSeparator()
        for key, label in (("liquid", "Glass: Liquid"), ("frost", "Glass: Frost")):
            a = m.addAction(label); a.setCheckable(True); a.setChecked(self.style_name == key)
            a.triggered.connect(lambda _, k=key: self.set_style(k))
        m.addSeparator()
        for key, label in (("pc", "Listen to: PC audio (calls, videos)"), ("mic", "Listen to: microphone (meeting in the room)"),
                           ("both", "Listen to: PC audio + my microphone")):
            a = m.addAction(label); a.setCheckable(True); a.setChecked(self.source == key)
            a.setEnabled(self.mode == "ready")                 # changes apply from the next Start session
            a.triggered.connect(lambda _, k=key: self.set_source(k))
        m.addSeparator()
        for key, label in (("screen", "Record: screen + audio"), ("audio", "Record: audio only"), ("off", "Record: off (transcript only)")):
            a = m.addAction(label); a.setCheckable(True); a.setChecked(self.record == key)
            a.setEnabled(self.mode == "ready")
            a.triggered.connect(lambda _, k=key: self.set_record(k))
        m.addSeparator()
        for roman, label in ((True, "Hindi shown as Hinglish"), (False, "Hindi shown in Devanagari")):
            a = m.addAction(label); a.setCheckable(True); a.setChecked(self.roman == roman)
            a.triggered.connect(lambda _, v=roman: self.set_roman(v))
        m.addSeparator()
        m.addAction("Edit glossary (names & terms)…").triggered.connect(self.edit_glossary)
        m.addAction("What it learned from your edits…").triggered.connect(lambda: os.startfile(str(corrections.ensure_file())))
        m.addAction("Open transcripts folder").triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(
            self.cfg.get("save_dir") or str(Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)) / "Meeting Transcripts"))))
        m.addAction("Quit").triggered.connect(self.quit_app)
        m.exec(e.globalPos())

    def edit_glossary(self):
        gp = DATA_DIR / "glossary.txt"
        if not gp.exists():
            gp.write_text(GLOSSARY_TEMPLATE, encoding="utf-8")
        os.startfile(str(gp))

    def set_source(self, k):
        self.source = k
        self.save_cfg()

    def set_record(self, k):
        self.record = k
        self.save_cfg()

    def set_style(self, k):
        self.style_name = k
        self.save_cfg()

    def set_roman(self, v):
        self.roman = v
        self.text_dirty = True
        self.save_cfg()


# ----------------------------------------------------------------------------- scripted self-test
def run_selftest(isl, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    steps = []

    def shot(name):
        def f():
            isl.grab().save(str(out / f"{name}.png"))
            if isl.dialog is not None:
                isl.dialog.grab().save(str(out / f"{name}_dialog.png"))
            print("SHOT", name, isl.mode, isl.shape(), [int(isl.sx.x), int(isl.sy.x), int(isl.sw.x), int(isl.sh.x)],
                  "win", isl.win_x, isl.win_y, flush=True)
        return f

    def at(ms, fn):
        steps.append((ms, fn))

    def pr_(x, y):
        return QPointF(x, y)

    def drag_to(x0, y0, x1, y1):
        def f():
            isl.press = dict(x=x0, y=y0, moved=True); isl.dragging = True
            n = 14
            for i in range(1, n + 1):
                QTimer.singleShot(i * 40, lambda i=i: setattr(isl, "pointer", QPointF(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n)))
        return f

    def release(x, y):
        def f():
            isl.press = None
            isl.end_drag(x, y)
        return f

    sw = isl.scr.width()
    at(0, lambda: print("SELFTEST waiting for models", flush=True))
    at(100, shot("0_ready"))
    at(300, lambda: isl.on_button("start"))
    at(14000, shot("1_recording"))
    at(100, lambda: isl.on_button("toggle"))
    at(900, shot("2_paused"))
    at(100, lambda: isl.on_button("toggle"))
    at(4000, shot("3_resumed"))
    at(100, drag_to(sw / 2, 30, 900, 520))
    at(900, shot("4_circle"))
    at(100, release(900, 520))
    at(1200, shot("5_free_panel"))
    at(100, lambda: setattr(isl, "hovering", True))
    at(170, shot("5a_hover_in"))
    at(600, shot("5b_hover_settled"))
    at(100, lambda: setattr(isl, "hovering", False))
    at(500, lambda: None)
    at(100, lambda: isl.on_button("min"))
    at(90, shot("6a_min_mid"))
    at(1200, shot("6_free_pill"))
    at(100, lambda: setattr(isl, "expanded", True))
    at(110, shot("6b_open_mid"))
    at(1000, lambda: None)
    at(100, lambda: isl.on_button("min"))
    at(1200, lambda: None)
    at(100, drag_to(900, 540, sw - 12, 600))
    at(700, release(sw - 12, 600))
    at(1200, shot("7_docked_vbar"))
    at(100, lambda: setattr(isl, "expanded", True))
    at(1400, shot("8_docked_panel"))
    at(100, lambda: setattr(isl, "expanded", False))
    at(900, drag_to(sw - 30, 600, sw / 2 + 40, 40))
    at(700, release(sw / 2 + 40, 40))
    at(1200, shot("9_home_pill"))
    at(100, lambda: setattr(isl, "expanded", True))
    at(1200, shot("10_home_panel"))
    at(100, lambda: isl.on_button("end"))
    at(500, lambda: None if isl.dialog is not None else "wait")
    at(800, shot("11_save_dialog"))

    def do_save():
        d = isl.dialog
        if d:
            d.folder.setText(str(out / "saved"))
            d.name.setText("Selftest: weekly/review?")
            d.save()
    at(300, do_save)
    at(1200, shot("12_after_save"))
    at(500, shot("12b_polishing"))

    def wait_polish():                                    # the recording is re-listened to in the background after Save
        return None if (isl.finisher.pending() == 0 and isl.polish is None) else "wait"
    at(500, wait_polish)

    def show_outputs():
        for f in sorted((out / "saved").glob("*")):
            print("OUTPUT", f.name, f.stat().st_size, "bytes", flush=True)
            if f.suffix == ".txt":
                print(f.read_text(encoding="utf-8-sig")[:700], flush=True)
        print("NOTICE", isl.notice, flush=True)
    at(300, show_outputs)
    at(100, shot("12c_final_ready"))

    def edit_and_learn():                                  # the user fixes a word in the saved transcript, the app learns it
        f = next((p for p in (out / "saved").glob("*.txt")), None)
        if f is None:
            print("LEARN no transcript found", flush=True)
            return
        t = f.read_text(encoding="utf-8-sig")
        print("LEARN edit: 'ziro' ->", "ziro" in t, flush=True)
        f.write_text(t.replace("ziro", "zero"), encoding="utf-8")
        isl.learn_from_edits()
        import glossary
        print("LEARN file:", corrections.LEARNED.read_text(encoding="utf-8").splitlines()[-1:], flush=True)
        print("LEARN applied:", glossary.Glossary().apply("bilkul ziro se leke"), "| notice:", isl.notice, flush=True)
    at(300, edit_and_learn)

    def report():
        fs = isl.frame_stats
        if fs:
            a = np.array(fs[10:]) * 1000
            pm = np.array(isl.paint_stats[10:])
            print(f"FRAMES n={len(a)} interval avg={a.mean():.1f}ms p95={np.percentile(a, 95):.1f}ms max={a.max():.1f}ms | "
                  f"paint avg={pm.mean():.1f}ms p95={np.percentile(pm, 95):.1f}ms", flush=True)
        print("DONE", flush=True)
        isl.quit_app()
    at(600, report)

    state = {"i": 0}

    def run():
        if state["i"] >= len(steps):
            return
        ms, fn = steps[state["i"]]
        state["i"] += 1

        def go(tries=0):
            if fn() == "wait" and tries < 900:           # a step can ask to be retried (e.g. until the Save window is up)
                state["i"] -= 0
                QTimer.singleShot(300, lambda: go(tries + 1))
            else:
                run()
        QTimer.singleShot(ms, go)

    def wait_models():
        if isl.mode == "ready":
            run()
        else:
            QTimer.singleShot(300, wait_models)
    wait_models()


def single_instance():
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 47831))
    except OSError:
        return None
    s.listen(1)
    return s


def main():
    lock = single_instance()
    if lock is None:
        print("Meeting Island is already running.")
        return
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x8000)
    except Exception:
        pass
    isl = Island()
    if ARGS.selftest:
        run_selftest(isl, ARGS.selftest)
    elif ARGS.tour:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "tools" / "showcase"))
        import tour
        tour.run(isl, ARGS.tour)
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
