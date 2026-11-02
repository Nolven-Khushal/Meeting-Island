"""
Records the meeting for later: the screen as video, and (muxed in at the end) the audio.

  ScreenRecorder   a thread that grabs the primary screen ~5 times a second into video_only.mp4 (H.264, in-process with PyAV,
                   no ffmpeg program needed). It follows the session clock, so paused time is not recorded and the video lines up
                   with the transcript's [hh:mm:ss] stamps. The file is "fragmented": if the PC dies mid-meeting what was
                   recorded so far still plays.
  build_recording  video + the audio lanes -> one normal .mp4 (or .m4a without video)
  free_gb          disk space check
"""
import shutil
import threading
import time
import traceback
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np

SR = 16000


def free_gb(path):
    try:
        p = Path(path)
        while not p.exists() and p != p.parent:
            p = p.parent
        return shutil.disk_usage(p).free / 1e9
    except Exception:
        return 99.0


class ScreenRecorder(threading.Thread):
    def __init__(self, path, clock, fps=5, max_w=1280, watcher=None):
        """clock() -> (seconds since the session began, not counting pauses ; paused?)
        watcher: screennames.Watcher - sees every full-resolution frame (who has the speaking ring around their tile?)"""
        super().__init__(daemon=True)
        self.path, self.clock, self.fps, self.max_w, self.watcher = Path(path), clock, fps, max_w, watcher
        self.stop_ev = threading.Event()
        self.error = None
        self.frames = 0

    def _grabber(self):
        """-> (grab(), width, height). grab() returns the latest screen as BGRA, or None while nothing has changed.
        Desktop duplication (GPU) takes ~1 ms; the older GDI route (mss) takes ~250 ms per 1080p frame, so it is the fallback."""
        try:
            import dxcam
            cam = dxcam.create(output_color="BGRA", processor_backend="numpy")
            first = None
            for _ in range(60):                                         # the first call returns the whole desktop
                first = cam.grab()
                if first is not None:
                    break
                time.sleep(0.05)
            if first is None:
                raise RuntimeError("no first frame")
            self._cam = cam
            return cam.grab, first.shape[1], first.shape[0], first
        except Exception as e:
            print(f"[recorder] desktop duplication unavailable ({e}); using the slower capture", flush=True)
        import mss
        sct = mss.mss()
        mon = sct.monitors[1]
        self._cam = sct

        def grab():
            s = sct.grab(mon)
            return np.frombuffer(s.raw, np.uint8).reshape(s.height, s.width, 4)
        first = grab()
        return grab, first.shape[1], first.shape[0], first

    def run(self):
        container = None
        self._cam = None
        try:
            import av
            grab, w, h, last_img = self._grabber()
            ow = min(self.max_w, w) // 2 * 2
            oh = int(round(h * ow / w)) // 2 * 2
            container = av.open(str(self.path), "w", options={"movflags": "frag_keyframe+empty_moov+default_base_moof"})
            st = container.add_stream("libx264", rate=self.fps)
            st.width, st.height, st.pix_fmt = ow, oh, "yuv420p"
            st.time_base = Fraction(1, self.fps)
            st.options = {"preset": "ultrafast", "crf": "30", "threads": "2", "tune": "zerolatency"}
            last = -1
            while not self.stop_ev.is_set():
                t, paused = self.clock()
                idx = int(t * self.fps)
                if paused or idx <= last:
                    time.sleep(0.02)
                    continue
                if self.frames % 300 == 299 and free_gb(self.path) < 1.0:     # about every minute: never fill the disk
                    self.error = RuntimeError("disk almost full: the screen recording was stopped")
                    print("[recorder] " + str(self.error), flush=True)
                    break
                img = grab()
                if img is not None and img.shape[0] == h and img.shape[1] == w:
                    last_img = img                                      # unchanged screen: the previous picture is still right
                if self.watcher is not None:
                    self.watcher.analyse(last_img, t)                   # full resolution: the saved video is too small to read names
                fr = av.VideoFrame.from_ndarray(last_img, format="bgra").reformat(width=ow, height=oh, format="yuv420p",
                                                                                 interpolation="FAST_BILINEAR")
                fr.pts = idx
                for pkt in st.encode(fr):
                    container.mux(pkt)
                last = idx
                self.frames += 1
            for pkt in st.encode(None):
                container.mux(pkt)
        except Exception as e:
            self.error = e
            traceback.print_exc()
        finally:
            try:
                if container is not None:
                    container.close()
            except Exception:
                pass
            try:
                if self._cam is not None and hasattr(self._cam, "release"):
                    self._cam.release()
            except Exception:
                pass

    def stop(self, timeout=20):
        self.stop_ev.set()
        self.join(timeout)


def _mmap(path):
    """A lane file as a read-only int16 memory map (size from the file: the header is only final after a clean close)."""
    p = Path(path)
    n = max(0, (p.stat().st_size - 44) // 2)
    return np.memmap(p, dtype="<i2", mode="r", offset=44, shape=(n,)) if n else np.zeros(0, np.int16)


class LaneMixer:
    """Several lanes (PC audio, microphone) -> one mono 16 kHz signal, handed out in chunks (a 2-hour meeting is 230 MB per lane)."""

    def __init__(self, wavs):
        self.lanes = [_mmap(p) for p in wavs if Path(p).exists()]
        self.n = max([len(a) for a in self.lanes] or [0])

    def chunks(self, size=SR * 20):
        for i in range(0, self.n, size):
            mix = np.zeros(min(size, self.n - i), np.float32)
            for a in self.lanes:
                piece = np.asarray(a[i:i + len(mix)], dtype=np.float32) / 32768
                mix[: len(piece)] += piece
            hot = np.abs(mix) > 0.9                                      # two loud voices at once: soft limit instead of clipping
            if hot.any():
                mix[hot] = np.sign(mix[hot]) * (0.9 + 0.1 * np.tanh((np.abs(mix[hot]) - 0.9) / 0.1))
            yield i, mix


def build_recording(video, wavs, out, progress=None):
    """video: video_only.mp4 or None; wavs: audio lane files; out: .mp4 (with video) or .m4a (audio only)."""
    import av
    mixer = LaneMixer(wavs)
    out = Path(out)
    tmp = out.with_name(out.stem + ".partial" + out.suffix)
    has_video = bool(video) and Path(video).exists() and Path(video).stat().st_size > 2048
    fmt = "mp4" if has_video else "ipod"
    oc = av.open(str(tmp), "w", format=fmt, options={"movflags": "+faststart"} if has_video else {})
    try:
        ast = oc.add_stream("aac", rate=SR)
        ast.layout = "mono"
        ast.bit_rate = 64000
        ic = vst = None
        if has_video:
            ic = av.open(str(video))
            src = ic.streams.video[0]
            vst = oc.add_stream_from_template(src)
        total = max(1, mixer.n)
        step = 1024
        pts = 0
        for base, big in mixer.chunks():
            for i in range(0, len(big), step):
                chunk = big[i:i + step]
                fr = av.AudioFrame.from_ndarray(chunk[None, :].astype(np.float32), format="flt", layout="mono")
                fr.sample_rate = SR
                fr.pts = pts
                pts += len(chunk)
                for pkt in ast.encode(fr):
                    oc.mux(pkt)
            if progress:
                progress(min(1.0, (base + len(big)) / total) * (0.5 if has_video else 1.0))
        for pkt in ast.encode(None):
            oc.mux(pkt)
        if has_video:
            for pkt in ic.demux(src):
                if pkt.dts is None:
                    continue
                pkt.stream = vst
                oc.mux(pkt)
            ic.close()
            if progress:
                progress(1.0)
    finally:
        oc.close()
    tmp.replace(out)
    return out
