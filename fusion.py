"""
Fuse two Hindi/Hinglish recognisers into one Roman-script (Hinglish) line:

  IndicConformer (AI4Bharat)  - best at Hindi words, but writes English words in Devanagari ("डॉक्यूमेंट")
  Whisper-Hindi2Hinglish-Swift - weaker on Hindi, but spells English words properly ("document")

The IndicConformer transcript is the backbone (romanised). Wherever one of its words sounds like an English word that
Swift heard at about the same place in the sentence, Swift's English spelling is used instead.
"""
import re
from functools import lru_cache

from translit import to_roman

_HINDI_ROMAN = None


def _skel(w):
    w = w.lower()
    for a, b in (("chh", "C"), ("ch", "C"), ("ph", "f"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"), ("bh", "b"),
                 ("jh", "j"), ("sh", "s"), ("aa", "a"), ("ee", "i"), ("oo", "u"), ("ai", "e"), ("au", "o"), ("w", "v"),
                 ("z", "j"), ("q", "k"), ("x", "ks"), ("c", "k"), ("y", "i")):
        w = w.replace(a, b)
    w = re.sub(r"(.)\1+", r"\1", w.replace("h", ""))
    return w[:-1] if len(w) > 3 and w.endswith("a") else w


def _hindi_set():
    """Skeletons of the ~5000 most common Hindi words, romanised - these are never 'English'."""
    global _HINDI_ROMAN
    if _HINDI_ROMAN is None:
        from wordfreq import top_n_list
        import translit
        s = {_skel(to_roman(w)) for w in top_n_list("hi", 5000)}
        s |= {_skel(v) for v in translit.COMMON.values()}
        _HINDI_ROMAN = s
    return _HINDI_ROMAN


@lru_cache(maxsize=20000)
def is_english(word):
    from wordfreq import zipf_frequency
    w = word.lower()
    if not re.fullmatch(r"[a-z][a-z'\-]*", w) or len(w) < 3:
        return False
    if _skel(w) in _hindi_set():
        return False
    return zipf_frequency(w, "en") >= 2.8


HINGLISH_FUNC = set("""
hai hain he hn mein main me mai ka ki ke ko se par pe bhi hi toh to tho jo tha thi the ho hoga hogi honge hum ham aap ap tum
kya kyu kyun kyon nahi nahin nhi haan han ha ji aur or ek do teen yeh ye yah woh wo vo vah voh is us iss uss isko usko inko unko
kar karo karna karte karta karti kiya raha rahe rahi rha rhe gaya gaye gayi diya liye lie liya sab koi kuch bahut bohot accha acha
achha theek thik abhi phir fir jab tab kab ab bas matlab yaar arre arey na mera meri mere tera teri apna apni apne hamara hamari
unka unki iska uska wala wali wale kaise kaisa kitna kitne yahan wahan idhar udhar dekho dekhiye chahiye sakte sakta sakti
""".split())


def english_ratio(text):
    """Share of words in a (Roman) line that are real English words - used to send English sentences down the English path."""
    from wordfreq import zipf_frequency
    ws = [re.sub(r"[^\w'\-]", "", w).lower() for w in (text or "").split()]
    ws = [w for w in ws if w and not w.isdigit()]
    eng = hin = 0
    for w in ws:
        en_word = bool(re.fullmatch(r"[a-z][a-z'\-]*", w)) and zipf_frequency(w, "en") >= 3.0
        if w in HINGLISH_FUNC:
            if not en_word:
                hin += 1           # hai, mein, ka, kya ... : clearly Hindi
            # the, is, to, main, he ... exist in both languages: no evidence either way
        elif en_word:
            eng += 1
        else:
            hin += 1               # romanised Hindi content words: kaam, rahe, pareshaan ...
    return eng / (eng + hin) if eng + hin else 0.0


_SOUNDS = None


def _english_by_sound():
    """Double-Metaphone code -> the most common English word with that sound (40k word list, len >= 4)."""
    global _SOUNDS
    if _SOUNDS is None:
        from metaphone import doublemetaphone
        from wordfreq import top_n_list, zipf_frequency
        _SOUNDS = {}
        for w in top_n_list("en", 40000):
            if len(w) < 4 or not w.isalpha() or w in HINGLISH_FUNC or zipf_frequency(w, "en") < 3.2:
                continue
            code = doublemetaphone(w)[0]
            if len(code) >= 3 and code not in _SOUNDS:      # top_n_list is frequency-ordered: first word wins
                _SOUNDS[code] = w
    return _SOUNDS


def english_by_sound(roman_word):
    """'kors' -> 'course', 'intarprided' -> 'interpreted' - only for long, non-Hindi words with an exact sound match."""
    from metaphone import doublemetaphone
    w = roman_word.lower()
    if len(w) < 4 or not w.isalpha() or _skel(w) in _hindi_set():
        return None
    code = doublemetaphone(w)[0]
    e = _english_by_sound().get(code) if len(code) >= 3 else None
    if e is None:
        return None
    a, b = _skel(w), _skel(e)                      # same consonant sound is not enough: the vowels must agree too
    return e if _lev(a, b) / max(len(a), len(b)) <= 0.34 else None


def _lev(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[-1]


def fuse(indic_text, swift_text, max_dist=0.42, max_shift=0.35, by_sound=True):
    dev = (indic_text or "").split()
    sw = [re.sub(r"[^\w'\-]", "", w) for w in (swift_text or "").split()]
    if not dev:
        return swift_text or ""
    eng = [(i, w, _skel(w)) for i, w in enumerate(sw) if w and is_english(w)]
    hindi = _hindi_set()
    out = []
    for j, wd in enumerate(dev):
        r = to_roman(wd)
        k = _skel(r)
        best = None
        if eng and len(k) >= 3 and k not in hindi:
            pj = j / max(1, len(dev) - 1)
            for i, w, ks in eng:
                if abs(i / max(1, len(sw) - 1) - pj) > max_shift:
                    continue
                dist = _lev(k, ks) / max(len(k), len(ks))
                if dist <= max_dist and (best is None or dist < best[0]):
                    best = (dist, w.lower())
        if best is None and by_sound:                # Swift didn't hear it either: match English words by sound
            e = english_by_sound(r)
            if e:
                best = (1.0, e)
        out.append(best[1] if best else r)
    return punctuate(out, swift_text)


_END = ".?!"


def punctuate(words, swift_text):
    """
    IndicConformer writes no punctuation, Swift does. Align the two word lists (by sound) and copy Swift's sentence
    endings and commas onto the matching fused words, then capitalise each sentence start. Nothing is cut: this only
    shows where the speaker's sentences really end.
    """
    sw = []
    for tok in (swift_text or "").split():
        m = re.match(r"^(.*?)([.,?!;:]*)$", tok)
        sw.append((m.group(1), m.group(2)) if m and m.group(1) else (tok, ""))
    if not words or not sw:
        return " ".join(words)
    a = [_skel(re.sub(r"[^\w]", "", w)) for w in words]
    b = [_skel(re.sub(r"[^\w]", "", w)) for w, _ in sw]
    n, m = len(a), len(b)
    GAP = 0.7
    d = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i * GAP
    for j in range(1, m + 1):
        d[0][j] = j * GAP
    sub = lambda i, j: (_lev(a[i], b[j]) / max(len(a[i]), len(b[j]), 1)) if a[i] and b[j] else 1.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + GAP, d[i][j - 1] + GAP, d[i - 1][j - 1] + sub(i - 1, j - 1))
    i, j, mark = n, m, {}
    while i and j:
        if abs(d[i][j] - (d[i - 1][j - 1] + sub(i - 1, j - 1))) < 1e-9:
            if sub(i - 1, j - 1) <= 0.6 and sw[j - 1][1]:
                mark[i - 1] = sw[j - 1][1]
            i, j = i - 1, j - 1
        elif abs(d[i][j] - (d[i - 1][j] + GAP)) < 1e-9:
            i -= 1
        else:
            j -= 1
    out, cap = [], True
    for k, w in enumerate(words):
        if cap and w[:1].isalpha():
            w = w[:1].upper() + w[1:]
        p = mark.get(k, "")
        out.append(w + p)
        cap = bool(p) and p[-1] in _END
    return " ".join(out)
