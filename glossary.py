"""
Personal glossary: names, products, project terms, jargon your meetings use.

  glossary.txt   one entry per line.   Priya            -> always spelled "Priya" when something sounds like it
                                       kuber=Kubernetes -> whenever the engine writes "kuber", output "Kubernetes"
                 lines starting with # are comments.

Used in two places:
  1. as a hint to the decoder (Whisper's prompt), which makes the right word more likely in the first place;
  2. after recognition: a word that SOUNDS like a glossary term (Double Metaphone) and is spelled close to it is replaced
     by the term. This fixes "Priy", "Free you", "kuber netes" ... without ever touching ordinary words.
"""
import re
from pathlib import Path

from metaphone import doublemetaphone

from paths import DATA_DIR
PATH = DATA_DIR / "glossary.txt"
LEARNED = DATA_DIR / "learned_glossary.txt"      # rules the app learned from the user's corrections (corrections.py)


def _code(w):
    return doublemetaphone(re.sub(r"(.)\1+", r"\1", w.lower()))[0]


def _skel(w):
    w = w.lower()
    for a, b in (("ph", "f"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("sh", "s"), ("aa", "a"),
                 ("ee", "i"), ("oo", "u"), ("w", "v"), ("z", "j"), ("c", "k"), ("y", "i")):
        w = w.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", w.replace("h", ""))


def _lev(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[-1]


def _is_common(w):
    """A very common word (hai, mein, the, price...): never rewrite it into a glossary term by fuzzy sound-matching.
    Rarer words are fair game: a name you typed into your own list should beat a rare dictionary word."""
    import fusion
    from wordfreq import zipf_frequency
    lw = w.lower()
    return lw in fusion.HINGLISH_FUNC or zipf_frequency(lw, "en") >= 4.0 or zipf_frequency(lw, "hi") >= 4.6


class Glossary:
    def __init__(self, terms=None, path=None):
        self.terms, self.fixed, self.by_code, self.phrases = [], {}, {}, {}
        lines = terms if terms is not None else self._read(path or PATH) + (self._read(LEARNED) if path is None else [])
        for ln in lines:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if "=" in ln:
                wrong, right = (x.strip() for x in ln.split("=", 1))
                if wrong and right:
                    if " " in wrong:                                # a phrase: "free you=Priya"
                        self.phrases[tuple(re.sub(r"[^\w'\-]", "", t).lower() for t in wrong.split())] = right
                    else:
                        self.fixed[wrong.lower()] = right
                    ln = right
            for term in ln.split(","):
                term = term.strip()
                if term:
                    self.terms.append(term)
        seen = set()
        self.terms = [t for t in self.terms if not (t.lower() in seen or seen.add(t.lower()))]
        for t in self.terms:                                    # single words are snapped by sound; phrases only hint the decoder
            if " " not in t and len(t) >= 3:
                c = _code(t)
                if len(c) >= 2:
                    self.by_code.setdefault(c, []).append(t)

    @staticmethod
    def _read(path):
        try:
            return Path(path).read_text(encoding="utf-8").splitlines()
        except Exception:
            return []

    def __bool__(self):
        return bool(self.terms or self.fixed or self.phrases)

    def prompt(self, style=None, limit=60):
        """Decoder hint: the glossary terms (plus an optional Hinglish style sentence)."""
        terms = ", ".join(self.terms[:limit])
        return " ".join(x for x in (style, (terms + ".") if terms else "") if x) or None

    def _phrase_at(self, toks, i):
        """-> (replacement text with the original punctuation kept, number of words used) or None"""
        if not self.phrases:
            return None
        for n in (4, 3, 2):
            if i + n <= len(toks):
                key = tuple(re.sub(r"[^\w'\-]", "", t).lower() for t in toks[i:i + n])
                if key in self.phrases:
                    pre = re.match(r"^[^\w]*", toks[i]).group(0)
                    post = re.search(r"[^\w]*$", toks[i + n - 1]).group(0)
                    return pre + self.phrases[key] + post, n
        return None

    def apply(self, text, max_dist=0.34):
        if not text or not self:
            return text
        out = []
        toks = text.split(" ")
        i = 0
        while i < len(toks):
            hit = self._phrase_at(toks, i)                  # a learned phrase fix ("free you=Priya") wins over single words
            if hit:
                out.append(hit[0])
                i += hit[1]
                continue
            tok = toks[i]
            i += 1
            m = re.match(r"^([^\w]*)([\w'\-]+)([^\w]*)$", tok)
            if not m:
                out.append(tok)
                continue
            pre, w, post = m.groups()
            key = w.lower()
            if key in self.fixed:
                out.append(pre + self.fixed[key] + post)
                continue
            best, best_d = None, 9.0
            if len(w) >= 3 and w.isascii():
                cands = self.by_code.get(_code(w), [])
                exact = next((t for t in cands if t.lower() == key), None)
                if exact:                                      # same word: just fix its capitalisation (rahul -> Rahul)
                    best = exact
                elif cands and not _is_common(w):              # a fuzzy snap is only allowed on words that are NOT ordinary words
                    for term in cands:
                        a, b = _skel(w), _skel(term)
                        dist = _lev(a, b) / max(len(a), len(b), 1)
                        if dist <= max_dist and dist < best_d:
                            best, best_d = term, dist
            out.append(pre + (best or w) + post)
        return " ".join(out)
