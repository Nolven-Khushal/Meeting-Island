"""
Fine-tune Oriserve's Whisper-Hindi2Hinglish-Swift (Whisper-base, 72 M parameters) on more Hinglish speech, then export it in
the format Meeting Island uses (CTranslate2 int8).  Runs on a free Kaggle/Colab GPU (a T4 is plenty: the model is tiny).

  Kaggle:  !pip -q install ctranslate2 soundfile datasets      (then)      !python train_swift.py --hours 150
  Smoke test on any PC (CPU, a few minutes):                             python train_swift.py --smoke

What it does
  * streams a slice of  agarwalayushi/hinglish  (2,264 h of Hindi / Hinglish / Indian English; CC BY 4.0): evenly spaced parquet
    files, so the mix of corpora is representative, with a preference for code-switched (<hi-en>) clips
  * turns every transcript into Roman Hinglish with our own transliterator (translit.py): Devanagari -> "hum log", English words
    stay as they are. That is the style the app works with.
  * augments: background noise, babble from other clips, speed 0.9-1.1x, volume - the real meetings are noisy
  * learns gently (LR 1e-5, warm-up, decay): the goal is to ADD Hinglish, not to forget the noisy-speech robustness it has
  * reports the word error rate on clips it never trained on, before and after, and writes  swift_ft_ct2.zip
Only the model that beats the current one on OUR benchmark (finetune/eval_candidate.py) is shipped.
"""
import argparse
import io
import json
import math
import queue
import random
import re
import shutil
import sys
import threading
import time
import unicodedata
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from translit import to_roman, has_devanagari  # noqa: E402

SR = 16000
REPO = "agarwalayushi/hinglish"
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Oriserve/Whisper-Hindi2Hinglish-Swift")
ap.add_argument("--hours", type=float, default=150, help="hours of speech to train on (one pass)")
ap.add_argument("--shards", type=int, default=40, help="how many evenly spaced parquet files to draw from")
ap.add_argument("--hien-share", type=float, default=0.7, help="share of code-switched (<hi-en>) clips")
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--eval-clips", type=int, default=300)
ap.add_argument("--min-sec", type=float, default=1.5)
ap.add_argument("--max-sec", type=float, default=25.0)
ap.add_argument("--noise", type=float, default=0.5, help="probability of adding noise / babble to a clip")
ap.add_argument("--out", default="swift_ft")
ap.add_argument("--smoke", action="store_true", help="tiny CPU run to check that everything works")
args = ap.parse_args()
if args.smoke:
    args.hours, args.shards, args.bs, args.eval_clips = 0.02, 1, 2, 6
dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = random.Random(7)
print(f"device {dev}; base {args.base}; {args.hours} h; out {args.out}", flush=True)


# ----------------------------------------------------------------------------------------------- data
def clean_text(t):
    hien = "<hi-en>" in (t or "")
    t = unicodedata.normalize("NFC", re.sub(r"<[^>]+>", " ", t or ""))
    t = to_roman(t) if has_devanagari(t) else t
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t, hien


def decode(example):
    import soundfile as sf
    from scipy.signal import resample_poly
    a = example["audio"]
    x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        f = Fraction(SR, sr).limit_denominator(1000)
        x = resample_poly(x, f.numerator, f.denominator).astype(np.float32)
    return x


def parquet_files():
    from huggingface_hub import list_repo_files
    files = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
    print(len(files), "parquet files in the dataset", flush=True)
    if len(files) <= 2 or args.smoke:
        return files[:1], files[:1]
    held = files[-1]                                    # never trained on: used for the before/after error rate
    pool = files[:-1]
    step = max(1, len(pool) // args.shards)
    return pool[::step][: args.shards], [held]


def stream(files):
    from datasets import Audio, load_dataset
    ds = load_dataset("parquet", data_files=[f"hf://datasets/{REPO}/{f}" for f in files], split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False)).shuffle(seed=7, buffer_size=500)
    for ex in ds:
        try:
            dur = float(ex.get("duration") or 0)
        except Exception:
            dur = 0.0
        text, hien = clean_text(ex["text"])
        if not text or len(text) < 3:
            continue
        try:
            x = decode(ex)
        except Exception:
            continue
        sec = len(x) / SR
        if not (args.min_sec <= sec <= args.max_sec) or not np.isfinite(x).all() or np.abs(x).max() < 1e-3:
            continue
        yield x, text, hien


def augment(x, babble):
    """noise / babble / speed / volume - meetings are not studio recordings"""
    from scipy.signal import resample_poly
    if rng.random() < 0.3:
        r = rng.choice([(9, 10), (10, 9), (11, 10), (10, 11)])      # speed 0.9 - 1.1x
        x = resample_poly(x, r[0], r[1]).astype(np.float32)
    if rng.random() < args.noise:
        p = float(np.sqrt(np.mean(x ** 2)) + 1e-6)
        if babble is not None and rng.random() < 0.5:                # somebody else talking in the background
            n = np.resize(babble, len(x))
            snr = rng.uniform(6, 18)
        else:
            n = np.random.randn(len(x)).astype(np.float32)
            snr = rng.uniform(8, 25)
        n = n * (p / (np.sqrt(np.mean(n ** 2)) + 1e-6)) / (10 ** (snr / 20))
        x = x + n
    x = x * rng.uniform(0.5, 1.4)
    return np.clip(x, -1, 1).astype(np.float32)


# ----------------------------------------------------------------------------------------------- model
from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402

proc = WhisperProcessor.from_pretrained(args.base)
tok, fe = proc.tokenizer, proc.feature_extractor
tok.set_prefix_tokens(language="hi", task="transcribe", predict_timestamps=False)
model = WhisperForConditionalGeneration.from_pretrained(args.base).to(dev)
model.generation_config.forced_decoder_ids = None
model.config.forced_decoder_ids = None
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def features(x):
    return fe(x, sampling_rate=SR, return_tensors="np").input_features[0]


def make_batch(items):
    feats = torch.tensor(np.stack([features(x) for x, _ in items]))
    ids = [tok(t).input_ids[:200] for _, t in items]                 # <|sot|><|hi|><|transcribe|><|notimestamps|> text <|eot|>
    L = max(len(i) for i in ids)
    dec_in = torch.full((len(ids), L - 1), PAD, dtype=torch.long)
    lab = torch.full((len(ids), L - 1), -100, dtype=torch.long)
    for k, i in enumerate(ids):
        dec_in[k, : len(i) - 1] = torch.tensor(i[:-1])
        lab[k, : len(i) - 1] = torch.tensor(i[1:])
        lab[k, :3] = -100                                            # do not train on the language/task prompt tokens
    return feats, dec_in, lab


def norm_words(t):
    return re.sub(r"[^\w\s]", " ", (t or "").lower()).split()


def edits(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[-1]


@torch.no_grad()
def evaluate(clips, label):
    model.eval()
    err = n = punct = 0
    for k in range(0, len(clips), 8):
        chunk = clips[k:k + 8]
        feats = torch.tensor(np.stack([features(x) for x, _ in chunk])).to(dev)
        out = model.generate(feats, language="hi", task="transcribe", max_new_tokens=160, num_beams=1)
        hyp = tok.batch_decode(out, skip_special_tokens=True)
        for (_, ref), h in zip(chunk, hyp):
            r, hh = norm_words(ref), norm_words(h)
            err += edits(r, hh); n += len(r); punct += len(re.findall(r"[.?!,]", h))
    wer = 100 * err / max(1, n)
    print(f"[eval] {label}: word error rate {wer:.1f}% on {len(clips)} unseen clips; punctuation marks per clip {punct / max(1, len(clips)):.2f}", flush=True)
    model.train()
    return wer


# ----------------------------------------------------------------------------------------------- held-out clips + training
train_files, held_files = parquet_files()
held = []
for x, t, _ in stream(held_files):
    held.append((x, t))
    if len(held) >= args.eval_clips:
        break
print(f"{len(held)} held-out clips", flush=True)
wer0 = evaluate(held, "BEFORE training") if held else None

q = queue.Queue(maxsize=6)
total_sec = args.hours * 3600


def producer():
    buf, babble, got, hi_n, tot = [], None, 0.0, 0, 0
    for x, t, hien in stream(train_files):
        if tot > 40 and (hi_n / tot) > args.hien_share and not hien and rng.random() < 0.8:
            continue                                                  # keep the share of code-switched clips near the target
        tot += 1; hi_n += int(hien)
        babble = x if (babble is None or rng.random() < 0.05) else babble
        buf.append((augment(x, babble), t))
        got += len(x) / SR
        if len(buf) >= args.bs:
            q.put(make_batch(buf)); buf = []
        if got >= total_sec:
            break
    q.put(None)
    print(f"[data] {got / 3600:.1f} h of speech streamed, {100 * hi_n / max(1, tot):.0f}% code-switched", flush=True)


threading.Thread(target=producer, daemon=True).start()
steps_est = max(2, int(total_sec / 8 / args.bs))                      # ~8 s per clip on average
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
warm = min(200, steps_est // 10)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.1, 1 - s / steps_est))
scaler = torch.amp.GradScaler(enabled=(dev == "cuda"))
model.train()
step, t0, run = 0, time.time(), 0.0
while True:
    b = q.get()
    if b is None:
        break
    feats, dec_in, lab = (v.to(dev) for v in b)
    with torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev == "cuda")):
        loss = model(input_features=feats, decoder_input_ids=dec_in, labels=None).logits
        loss = torch.nn.functional.cross_entropy(loss.float().reshape(-1, loss.shape[-1]), lab.reshape(-1), ignore_index=-100)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); sched.step()
    step += 1
    run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
    if step % 50 == 0 or args.smoke:
        print(f"step {step}/{steps_est}  loss {run:.3f}  lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s", flush=True)
wer1 = evaluate(held, "AFTER training") if held else None
if wer0 is not None:
    print(f"RESULT: held-out word error {wer0:.1f}% -> {wer1:.1f}%", flush=True)

# ----------------------------------------------------------------------------------------------- export
out = Path(args.out)
hf = out / "hf"
model.save_pretrained(hf)
proc.save_pretrained(hf)
fe.save_pretrained(hf)
tok.save_pretrained(hf)
if not (hf / "preprocessor_config.json").exists():                    # newer transformers versions name it differently
    from huggingface_hub import hf_hub_download
    shutil.copy(hf_hub_download(args.base, "preprocessor_config.json"), hf / "preprocessor_config.json")
import ctranslate2  # noqa: E402
ct2 = out / "ct2"
shutil.rmtree(ct2, ignore_errors=True)
ctranslate2.converters.TransformersConverter(str(hf), copy_files=["tokenizer.json", "preprocessor_config.json"]).convert(
    str(ct2), quantization="int8", force=True)
shutil.make_archive(str(out / "swift_ft_ct2"), "zip", ct2)
print("DONE ->", out / "swift_ft_ct2.zip", flush=True)
