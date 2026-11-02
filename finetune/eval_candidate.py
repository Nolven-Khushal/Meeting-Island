"""
Should this fine-tuned model replace the one in the app?  Same clips, same settings, old model vs new model.

  .venv\\Scripts\\python.exe finetune\\eval_candidate.py  path\\to\\unzipped\\ct2_folder

Data: the 80 benchmark clips (lecture + spontaneous conversation, with human transcripts) and the 25 pieces of the Taarak Mehta
clip (hard TV dialogue, YouTube captions as a rough reference).  PASS needs all of:
  * overall error rate on the 80 clips at least 1.0 point lower than the current model
  * conversation clips not worse than before + 0.5, lecture clips not worse than before + 1.0
  * the TV clip not worse
  * punctuation per clip at least 80% of before (the app builds sentences from it)
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
import assistant
from engine import Engine
from run import norm, edits
import polish

cand = Path(sys.argv[1]).resolve()
meta = json.load(open(ROOT / "bench/meta.json", encoding="utf-8"))
clips = [(m, assistant.decode_audio(m["file"]).astype(np.float32)) for m in meta]
tm_audio = assistant.decode_audio(str(ROOT / "scratch/tmkoc/ep.wav")).astype(np.float32) if (ROOT / "scratch/tmkoc/ep.wav").exists() else None
tm_ref = (ROOT / "scratch/tmkoc/reference_autocaptions.txt").read_text(encoding="utf-8") if tm_audio is not None else ""


def run(path):
    eng = Engine(str(path), threads=4, pad=15, roman=True)
    hyp, t0 = [], time.time()
    for m, a in clips:
        r = eng.transcribe(a, "hi", None, beam=1, pad=15)
        hyp.append(r["text"] if r else "")
    rtf = (time.time() - t0) / (sum(len(a) for _, a in clips) / 16000)
    tv = ""
    if tm_audio is not None:
        for p in polish.plan_pieces(np.asarray(tm_audio * 32768, dtype=np.int16), []):
            seg = tm_audio[max(0, p["a"] * 320 - 3200): p["b"] * 320 + 3200]
            r = eng.transcribe(seg, "hi", None, beam=1, pad=15)
            tv += " " + (r["text"] if r else "")
    return hyp, tv, rtf


def wer(idx, hyp):
    e = n = 0
    for i in idx:
        r, h = norm(meta[i]["ref"]), norm(hyp[i])
        e += edits(r, h); n += len(r)
    return 100 * e / max(1, n)


idx = {s: [i for i, m in enumerate(meta) if m["src"] == s] for s in ("mucs", "ivoices")}
allidx = list(range(len(meta)))
res = {}
for name, path in (("current", ROOT / "models/hinglish-swift-ct2"), ("candidate", cand)):
    hyp, tv, rtf = run(path)
    res[name] = dict(lecture=wer(idx["mucs"], hyp), conv=wer(idx["ivoices"], hyp), all=wer(allidx, hyp),
                     punct=sum(len(re.findall(r"[.?!,]", h)) for h in hyp) / len(hyp),
                     tv=(100 * edits(norm(tm_ref), norm(tv)) / max(1, len(norm(tm_ref)))) if tv else None, rtf=rtf)
    print(f"{name:10s} lecture {res[name]['lecture']:5.1f}  conversation {res[name]['conv']:5.1f}  ALL {res[name]['all']:5.1f}  "
          f"TV {res[name]['tv'] if res[name]['tv'] is None else round(res[name]['tv'], 1)}  punctuation/clip {res[name]['punct']:.2f}  speed {res[name]['rtf']:.3f}x", flush=True)
c, n = res["current"], res["candidate"]
checks = {
    "overall error at least 1.0 lower": n["all"] <= c["all"] - 1.0,
    "conversation not worse (+0.5)": n["conv"] <= c["conv"] + 0.5,
    "lecture not worse (+1.0)": n["lecture"] <= c["lecture"] + 1.0,
    "TV clip not worse": (n["tv"] is None) or n["tv"] <= c["tv"],
    "punctuation kept (>= 80%)": n["punct"] >= 0.8 * c["punct"],
}
for k, v in checks.items():
    print(("  PASS  " if v else "  FAIL  ") + k)
print("VERDICT:", "SHIP IT" if all(checks.values()) else "do not ship this one")
