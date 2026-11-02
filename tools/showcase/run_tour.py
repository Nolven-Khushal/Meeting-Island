"""
Records the showcase film and screenshots of the real app in demo mode.

    python tools/showcase/run_tour.py            full film (about 8 minutes of real time)
    python tools/showcase/run_tour.py --quick    same steps with short waits, to check the script (no music, not for publishing)

Output goes to tools/showcase/out/ (ignored by git):  video_only.mp4, shots/*.png, then  meeting-island-tour.mp4  with a music bed.
Needs: the models in models/ (python tools/fetch_models.py), demo/meeting_demo.wav (included), ffmpeg (imageio-ffmpeg is enough).
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "out"


def ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg") or "ffmpeg"


def music(seconds, path):
    """a soft generated bed: a few sine tones, tremolo, echo, low-pass, fade in and out. No voice-over."""
    tones = [(220.0, 0.30), (277.18, 0.22), (329.63, 0.22), (440.0, 0.10)]
    ins, mix = [], []
    for i, (f, v) in enumerate(tones):
        ins += ["-f", "lavfi", "-i", f"sine=frequency={f}:sample_rate=44100:duration={seconds}"]
        mix.append(f"[{i}:a]volume={v},tremolo=f={0.15 + 0.07 * i}:d=0.6[t{i}]")
    chain = ";".join(mix) + ";" + "".join(f"[t{i}]" for i in range(len(tones))) + \
        f"amix=inputs={len(tones)}:normalize=0,aecho=0.8:0.7:700|1100:0.35|0.25,lowpass=f=1400,volume=0.55," \
        f"afade=t=in:st=0:d=4,afade=t=out:st={max(0, seconds - 6)}:d=6[m]"
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", *ins, "-filter_complex", chain, "-map", "[m]", str(path)], check=True)


def main():
    quick = "--quick" in sys.argv
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["ISLAND_FAKE_BACKDROP"] = str(HERE / "wallpaper.png")           # a made-up desktop: the real one is never filmed
    env["MEETING_ISLAND_DATA"] = tempfile.mkdtemp(prefix="mi-tour-data-")
    if quick:
        env["TOUR_QUICK"] = "1"
    cmd = [sys.executable, str(ROOT / "island_qt.py"), "--tour", str(OUT), "--sim", str(ROOT / "demo" / "meeting_demo.wav"),
           "--console", "--reset-pos"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)
    shutil.rmtree(env["MEETING_ISLAND_DATA"], ignore_errors=True)
    video = OUT / "video_only.mp4"
    if quick or not video.exists():
        return
    import av
    with av.open(str(video)) as c:
        secs = int(float(c.duration) / 1e6) + 1
    bed = OUT / "bed.wav"
    music(secs, bed)
    final = OUT / "meeting-island-tour.mp4"
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(video), "-i", str(bed), "-c:v", "copy", "-c:a", "aac", "-b:a", "96k",
                    "-shortest", "-movflags", "+faststart", str(final)], check=True)
    print(f"film: {final}  {final.stat().st_size / 1e6:.1f} MB, {secs} s")


if __name__ == "__main__":
    main()
