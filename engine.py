"""
Fast Whisper decoding for live use.

faster-whisper always pads every input to a 30 s window, so even a 3 s sentence costs the full encoder
pass (~2.5 s for `small` on a laptop CPU). Here we call the CTranslate2 model directly with only the real
audio: encoder cost then scales with the length of the audio (~0.1 s per second for `small`), which makes
word-by-word live drafts possible. Decoding is timestamp-free (we know the chunk times ourselves) and
returns per-word confidence taken from the token probabilities.
"""
import math
import re
import threading
from pathlib import Path

import numpy as np

SR = 16000
MIN_SECONDS = 0.6


class Engine:
    def __init__(self, model_name, threads=6, pad=False, roman=False):
        self.pad, self.roman = pad, roman     # roman=True: the model writes Hinglish in Latin script (Oriserve models)
        import ctranslate2
        from faster_whisper import WhisperModel
        self._ct2 = ctranslate2
        from paths import APP_DIR
        local = APP_DIR / model_name
        if local.is_dir():                    # models/... folders ship with the app; never depend on the current folder
            model_name = str(local)
        self.name = model_name
        self.m = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=threads)
        self.fe = self.m.feature_extractor
        self.hf = self.m.hf_tokenizer
        self._tk = {}
        self.notimes = self.hf.token_to_id("<|notimestamps|>")
        self.prev_tok = self.hf.token_to_id("<|startofprev|>")
        self.lock = threading.Lock()  # one decode at a time per engine

    def tokenizer(self, lang):
        from faster_whisper.tokenizer import Tokenizer
        if lang not in self._tk:
            self._tk[lang] = Tokenizer(self.hf, True, task="transcribe", language=lang)
        return self._tk[lang]

    def _features(self, audio, pad=False):
        if pad:   # pad with silence: True = the standard 30 s window Whisper was trained on; a number = that many seconds
            from faster_whisper.audio import pad_or_trim
            frames = self.fe.nb_max_frames if pad is True else int(max(pad, len(audio) / SR + 0.5) * 100) // 2 * 2
            f = pad_or_trim(self.fe(audio), min(frames, self.fe.nb_max_frames))
        else:     # only the real audio: encoder cost scales with length (fast, slightly less accurate)
            f = self.fe(audio, padding=0)
            f = f[:, : (f.shape[-1] // 2) * 2]
        return self._ct2.StorageView.from_array(np.ascontiguousarray(f[None]))

    def detect_language(self, audio, allowed=None):
        """-> (lang, normalised prob among `allowed`). Cheap: the encoder only sees the real audio."""
        with self.lock:
            res = self.m.model.detect_language(self._features(audio))[0]
        probs = {tok.strip("<|>"): p for tok, p in res}
        if allowed:
            probs = {k: v for k, v in probs.items() if k in allowed}
        tot = sum(probs.values()) or 1e-9
        best = max(probs, key=probs.get)
        return best, probs[best] / tot

    def transcribe(self, audio, lang, prompt=None, beam=1, pad=None, roman=None):
        """-> dict(text, words=[[word, prob]], conf, no_speech) or None for too-short / empty audio."""
        if len(audio) < MIN_SECONDS * SR:
            return None
        secs = min(len(audio) / SR, 30.0)
        pad = self.pad if pad is None else pad
        roman = self.roman if roman is None else roman
        tk = self.tokenizer(lang)
        ids = []
        if prompt:
            ids = [self.prev_tok] + tk.encode(" " + prompt.strip())[-120:]
        ids += list(tk.sot_sequence) + [self.notimes]
        # token budget: Devanagari costs ~4-6 byte-level tokens per word, so Hindi script needs ~3x the budget of
        # English / Roman Hinglish. Too generous a budget just gives loops room to grow.
        per_s = 10 if (lang == "en" or roman) else 30
        greedy = beam <= 1          # per-token probabilities need the logits, which CTranslate2 only returns for greedy decoding
        with self.lock:
            r = self.m.model.generate(
                self._features(audio, pad), [ids], beam_size=beam,
                max_length=min(448, len(ids) + int(secs * per_s) + 16),
                return_scores=True, return_logits_vocab=greedy, return_no_speech_prob=True, repetition_penalty=1.05)[0]
        seq = list(r.sequences_ids[0])
        if seq and seq[-1] == tk.eot:
            seq = seq[:-1]
        if not seq:
            return None
        if greedy:
            L = np.asarray(r.logits[0], dtype=np.float32)          # (steps, vocab)
            L = np.exp(L - L.max(axis=1, keepdims=True))
            L /= L.sum(axis=1, keepdims=True)
            probs = [float(L[i, t]) for i, t in enumerate(seq) if i < len(L)]
        else:                                                       # beam search: only the whole sentence's score is known
            sc = float(math.exp(r.scores[0])) if r.scores else 0.5
            probs = [sc] * len(seq)
        words, cur, cur_p = [], [], []
        for tok, p in zip(seq, probs):
            if tok >= tk.eot:  # special / timestamp token
                continue
            if cur and self.hf.id_to_token(tok).startswith("Ġ"):  # byte-level BPE marker for "a space precedes"
                words.append([tk.decode(cur), round(float(np.mean(cur_p)), 2)])
                cur, cur_p = [], []
            cur.append(tok); cur_p.append(p)
        if cur:
            words.append([tk.decode(cur), round(float(np.mean(cur_p)), 2)])
        text = collapse_repeats(tk.decode([t for t in seq if t < tk.eot]).replace("�", "").strip())
        if not text:
            return None
        words = [[w.replace("�", ""), p] for w, p in words]   # a draft cut mid-character leaves a broken byte
        conf = round(float(np.mean([p for _, p in words])), 2) if words else 0.0
        if collapse_repeats(" ".join(w for w, _ in words).strip()) != " ".join(w for w, _ in words).strip():
            words = []  # a loop was removed: per-word data no longer lines up with the text
        return dict(text=text, words=words, conf=conf, no_speech=float(r.no_speech_prob or 0.0))


class IndicEngine:
    """
    AI4Bharat IndicConformer (Hindi, CTC, int8 ONNX, ~140 MB) through onnx-asr. Non-autoregressive: it labels the
    audio frame by frame instead of generating text, so it cannot loop or invent sentences, and it is fast on CPU.
    Output is Devanagari (English words included); fusion.py turns it into Hinglish.
    """

    def __init__(self, path="models/indicconformer-hi", threads=4):
        import onnx_asr
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        from paths import APP_DIR
        p = APP_DIR / path
        self.m = onnx_asr.load_model("nemo-conformer-ctc", str(p), quantization="int8", sess_options=so)
        self.lock = threading.Lock()

    def transcribe(self, audio):
        if len(audio) < 0.4 * SR:
            return ""
        with self.lock:
            return (self.m.recognize(np.ascontiguousarray(audio, dtype=np.float32), sample_rate=SR) or "").strip()


_WORD_REP = re.compile(r"(\b[\w']+\b)(?:[\s,.]+\1\b){3,}", re.IGNORECASE | re.UNICODE)
_PHRASE_REP = re.compile(r"(.{8,200}?)(?:[\s,.]*\1){2,}", re.DOTALL | re.UNICODE)
_LONG_REP = re.compile(r"(.{20,300}?)(?:[\s,.]*\1)+", re.DOTALL | re.UNICODE)  # a 20+ char phrase said twice verbatim = loop


def collapse_repeats(text):
    """Whisper's classic loops: 'aapne aapne aapne aapne' -> 'aapne', 'X Y. X Y. X Y.' -> 'X Y.'"""
    text = _LONG_REP.sub(r"\1", _PHRASE_REP.sub(r"\1", _WORD_REP.sub(r"\1", text)))
    # a loop cut off mid-way leaves a dangling copy of the sentence start: "... August. Revenue grew 12%"
    low = text.lower()
    cut = low.find(low[:12], 12) if len(low) >= 24 else -1
    return text[:cut].rstrip(" ,.;") + (text[cut - 1] if cut > 0 and text[cut - 1] in ".?!" else "") if cut > 0 else text
