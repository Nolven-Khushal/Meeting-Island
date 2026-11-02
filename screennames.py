"""
Who is speaking, from the picture.

Video-call apps mark the person who is talking: Zoom draws a lime-green ring around their tile and shows their name on it.
While a meeting is recorded this module looks at the full-resolution screen about twice a second:

  find_ring(frame)     is there a ring (long straight edges in the highlight colour forming a tile)? -> the tile's rectangle
  NameReader           reads the name label at the bottom-left of that tile with Windows' built-in offline text reader
                       (Windows.Media.Ocr: nothing to install, nothing leaves the PC)
  Watcher              ties the two together on a background thread and logs  {"t": seconds, "name": "..."}  to names.jsonl

After the meeting assign(segs, entries) matches the voices that the second pass told apart with those names: a voice gets a name
only when the same name was on the ringed tile for most of the time that voice was speaking (and the name is not claimed more
strongly by another voice). Everything else stays "Speaker N" - a wrong name is worse than no name.

The analysis must use the full-resolution frame: the saved video is shrunk to 1280 px wide and the labels become unreadable.
Ring colours are listed in RING_RANGES (hue range in degrees, minimum saturation, minimum brightness); add the colour of another
app's speaking highlight there.
"""
import asyncio
import json
import queue
import re
import threading
import time
import traceback
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

RING_RANGES = [(40, 130, 0.30, 0.55)]            # Zoom: yellow-green / lime / green border around the active speaker's tile
MIN_EDGE = 90                                    # shortest edge of a tile, in screen pixels
STEP = 2                                         # the colour test looks at every 2nd pixel (a ring is 3-5 px thick)


# ---------------------------------------------------------------------------------------------------- finding the ring
def ring_mask(frame, ranges=None, step=STEP):
    """BGRA/BGR frame -> boolean mask (every `step`-th pixel) of pixels in a speaking-highlight colour."""
    px = frame[::step, ::step, :3].astype(np.float32)
    b, g, r = px[..., 0], px[..., 1], px[..., 2]
    mx = px.max(axis=2)
    mn = px.min(axis=2)
    d = np.maximum(mx - mn, 1e-3)
    sat = d / np.maximum(mx, 1.0)
    val = mx / 255.0
    hue = np.where(mx == r, 60 * (((g - b) / d) % 6), np.where(mx == g, 60 * (2 + (b - r) / d), 60 * (4 + (r - g) / d)))
    out = np.zeros(mx.shape, bool)
    for lo, hi, smin, vmin in (ranges or RING_RANGES):
        out |= (hue >= lo) & (hue <= hi) & (sat >= smin) & (val >= vmin)
    return out


def _runs(a):
    a = np.asarray(a, bool)
    if a.size == 0:
        return []
    d = np.diff(np.concatenate(([0], a.view(np.int8), [0])))
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))


def _lines(mask, axis, min_len):
    """Long straight runs of mask pixels along rows (axis=1) or columns (axis=0), grouped when adjacent:
    -> [(position, start, end)] in mask pixels."""
    counts = mask.sum(axis=axis)
    cand = np.nonzero(counts >= min_len)[0]
    found = []
    for pos in cand.tolist():
        line = mask[pos, :] if axis == 1 else mask[:, pos]
        runs = [(a, b) for a, b in _runs(line) if b - a >= min_len]
        for a, b in runs:
            found.append((pos, a, b))
    groups = []
    for pos, a, b in found:                                   # a thick edge is several neighbouring rows: keep one line
        for g in groups:
            if abs(g[0] - pos) <= 3 and abs(g[1] - a) <= 4 and abs(g[2] - b) <= 4:
                break
        else:
            groups.append([pos, a, b])
    return [tuple(g) for g in groups]


def find_ring(frame, ranges=None, step=STEP):
    """-> (x1, y1, x2, y2) in screen pixels of the tile that has a speaking ring around it, or None."""
    mask = ring_mask(frame, ranges, step)
    if mask.sum() < 2 * (MIN_EDGE // step):
        return None
    ml = MIN_EDGE // step
    H = _lines(mask, 1, ml)                                   # horizontal edges (y, x_start, x_end)
    V = _lines(mask, 0, max(10, ml // 2))                     # vertical edges   (x, y_start, y_end)
    best = None
    for i, (y1, xa1, xb1) in enumerate(H):
        for y2, xa2, xb2 in H[i + 1:]:
            if y2 < y1:
                (y1_, xa1_, xb1_), (y2_, xa2_, xb2_) = (y2, xa2, xb2), (y1, xa1, xb1)
            else:
                (y1_, xa1_, xb1_), (y2_, xa2_, xb2_) = (y1, xa1, xb1), (y2, xa2, xb2)
            if y2_ - y1_ < ml * 0.5 or abs(xa1_ - xa2_) > 5 or abs(xb1_ - xb2_) > 5:
                continue
            x1, x2 = (xa1_ + xa2_) // 2, (xb1_ + xb2_) // 2
            need = 0.6 * (y2_ - y1_)
            left = any(abs(x - x1) <= 5 and min(ye, y2_) - max(ys, y1_) >= need for x, ys, ye in V)
            right = any(abs(x - x2) <= 5 and min(ye, y2_) - max(ys, y1_) >= need for x, ys, ye in V)
            if left and right:
                area = (x2 - x1) * (y2_ - y1_)
                if best is None or area > best[0]:
                    best = (area, (x1 * step, y1_ * step, x2 * step, y2_ * step))
    return best[1] if best else None


# ---------------------------------------------------------------------------------------------------- reading the name
class NameReader:
    """Windows built-in OCR on the name label of a tile. Created once; `read` is called from one worker thread."""

    def __init__(self):
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.globalization import Language
        eng = OcrEngine.try_create_from_user_profile_languages()
        for tag in ("en-US", "en-IN", "en-GB"):
            if eng is not None:
                break
            eng = OcrEngine.try_create_from_language(Language(tag))
        if eng is None:
            raise RuntimeError("Windows has no text-recognition language installed")
        self.engine = eng

    @staticmethod
    def strip(frame, rect):
        """The part of a tile that holds the name: bottom-left."""
        x1, y1, x2, y2 = rect
        w, h = x2 - x1, y2 - y1
        top = max(y1, y2 - max(30, int(0.20 * h)))
        return frame[top:y2 - 2, x1 + 3:min(x2, x1 + max(150, int(0.75 * w))), :3]

    async def _ocr(self, bgr):
        import av
        from winrt.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat
        from winrt.windows.storage.streams import DataWriter
        h, w = bgr.shape[:2]
        scale = max(2, min(6, 90 // max(h, 1) + 1))                       # small text reads better enlarged
        fr = av.VideoFrame.from_ndarray(np.ascontiguousarray(bgr), format="bgr24").reformat(
            width=w * scale, height=h * scale, format="bgra", interpolation="BICUBIC")
        arr = fr.to_ndarray()
        wr = DataWriter()
        wr.write_bytes(arr.tobytes())
        bmp = SoftwareBitmap.create_copy_from_buffer(wr.detach_buffer(), BitmapPixelFormat.BGRA8, arr.shape[1], arr.shape[0])
        res = await self.engine.recognize_async(bmp)
        return [ln.text for ln in res.lines]

    def read(self, frame, rect):
        """-> the name written on the tile, or None"""
        crop = self.strip(frame, rect)
        if crop.shape[0] < 6 or crop.shape[1] < 20:
            return None
        lines = asyncio.run(self._ocr(crop))
        return clean_name(" ".join(lines))


def clean_name(text):
    t = re.sub(r"[^\w\s.'\-()&,]", " ", text or "", flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip(" -.,'")
    t = re.sub(r"^(?:[A-Za-z]\s)+(?=[A-Z][a-z])", "", t)             # a mic icon read as a stray letter in front of the name
    if len([c for c in t if c.isalpha()]) < 3 or len(t) > 48:
        return None
    if t.lower() in ("speaker view", "gallery view", "zoom", "participants", "chat", "share screen", "mute", "unmute"):
        return None
    return t


# ---------------------------------------------------------------------------------------------------- during the meeting
class Watcher:
    """analyse(frame, t) is called by the screen recorder with the full-resolution frame; OCR runs on its own thread."""

    def __init__(self, path, every=0.5):
        self.path, self.every = Path(path), every
        self.q = queue.Queue(maxsize=2)
        self.last_t = -9.0
        self.reader = None
        self.error = None
        self.count = 0
        self.stop_ev = threading.Event()
        try:
            self.reader = NameReader()
        except Exception as e:
            self.error = e
            print(f"[screennames] disabled: {e}", flush=True)
            return
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    @property
    def ok(self):
        return self.reader is not None

    def analyse(self, frame, t):
        if not self.ok or t - self.last_t < self.every:
            return
        self.last_t = t
        try:
            rect = find_ring(frame)
            if rect is not None and not self.q.full():
                x1, y1, x2, y2 = rect
                crop_src = frame[max(0, y2 - max(30, int((y2 - y1) * 0.2))):y2, x1:x2, :3].copy()     # copy: the frame is reused
                self.q.put_nowait((t, rect, crop_src, frame.shape))
        except Exception:
            traceback.print_exc()

    def _work(self):
        f = None
        try:
            f = open(self.path, "a", encoding="utf-8")
            while not self.stop_ev.is_set():
                try:
                    t, rect, crop, shape = self.q.get(timeout=0.5)
                except queue.Empty:
                    continue
                x1, y1, x2, y2 = rect
                try:
                    h = crop.shape[0]
                    lines = asyncio.run(self.reader._ocr(crop[:, 3:min(crop.shape[1], max(150, int(0.75 * (x2 - x1)))), :]))
                    name = clean_name(" ".join(lines))
                except Exception:
                    traceback.print_exc()
                    continue
                if name:
                    f.write(json.dumps(dict(t=round(t, 2), name=name, rect=[x1, y1, x2, y2])) + "\n")
                    f.flush()
                    self.count += 1
        finally:
            if f:
                f.close()

    def stop(self):
        self.stop_ev.set()


# ---------------------------------------------------------------------------------------------------- after the meeting
def load(path):
    out = []
    try:
        for ln in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(ln)
                out.append((float(d["t"]), d["name"]))
            except Exception:
                pass
    except Exception:
        pass
    return out


def _key(name):
    return re.sub(r"[^a-z]", "", name.lower())


def canonical(entries):
    """OCR reads the same name slightly differently now and then ('Matt Caballero' / 'Matt Cabaliero'): merge close spellings;
    the most frequent spelling wins. -> [(t, canonical name)]"""
    counts = {}
    for _, n in entries:
        counts[n] = counts.get(n, 0) + 1
    reps = []
    for n, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        for r in reps:
            if SequenceMatcher(None, _key(n), _key(r)).ratio() >= 0.84:
                break
        else:
            reps.append(n)
    out = []
    for t, n in entries:
        best = max(reps, key=lambda r: SequenceMatcher(None, _key(n), _key(r)).ratio())
        out.append((t, best if SequenceMatcher(None, _key(n), _key(best)).ratio() >= 0.84 else n))
    return out


def assign(segs, entries, min_votes=3, min_share=0.6, lag=0.4):
    """{voice number: name}. A voice is named when, among the ringed-tile readings taken while it was speaking, one name
    has at least `min_votes` readings and `min_share` of them - and no other voice owns that name more strongly."""
    ents = canonical(entries)
    if not ents or not segs:
        return {}
    votes = {}
    for g in segs:
        v = g.get("voice")
        if not v:
            continue
        for t, n in ents:
            if g["start"] - lag <= t <= g["end"] + lag:
                votes.setdefault(v, {}).setdefault(n, 0)
                votes[v][n] += 1
    chosen = {}
    for v, d in votes.items():
        total = sum(d.values())
        n, c = max(d.items(), key=lambda kv: kv[1])
        if c >= min_votes and c / total >= min_share:
            chosen[v] = (n, c, c / total)
    out = {}
    for n in {x[0] for x in chosen.values()}:                  # a name belongs to one voice: the one with the most readings
        owners = [(c, v) for v, (nn, c, _) in chosen.items() if nn == n]
        c, v = max(owners)
        runner = sorted((c2 for c2, _ in owners), reverse=True)[1:2]
        if not runner or c >= 2 * runner[0]:
            out[v] = n
    return out
