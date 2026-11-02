"""
Devanagari -> Roman ("Hinglish") transliteration, the way Hindi is normally typed in chat:
  हम लोग complex projects पर काम कर रहे होंगे  ->  hum log complex projects par kaam kar rahe honge
Latin text (English words Whisper already kept in Latin) is left alone.
Rule-based with schwa deletion plus a small dictionary of very common function words.
"""
import re

CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
}
NUKTA = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f"}
INDEP = {"अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ए": "e", "ऐ": "ai",
         "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e"}
MATRA = {"ा": "aa", "ि": "i", "ी": "i", "ु": "u", "ू": "oo", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o",
         "ौ": "au", "ॉ": "o", "ॅ": "e"}
NASAL_MARKS, NUKTA_MARK, HALANT, VISARGA = "ंँ", "़", "्", "ः"
LABIAL = set("पफबभम")
DIGITS = {c: str(i) for i, c in enumerate("०१२३४५६७८९")}

COMMON = {
    "है": "hai", "हैं": "hain", "में": "mein", "मैं": "main", "नहीं": "nahi", "नही": "nahi", "यह": "yeh", "ये": "yeh",
    "वह": "woh", "वो": "woh", "और": "aur", "कि": "ki", "की": "ki", "का": "ka", "के": "ke", "को": "ko", "से": "se",
    "पर": "par", "भी": "bhi", "ही": "hi", "तो": "toh", "जो": "jo", "था": "tha", "थी": "thi", "थे": "the",
    "हो": "ho", "हूँ": "hoon", "हूं": "hoon", "हम": "hum", "आप": "aap", "तुम": "tum", "मेरा": "mera", "मेरी": "meri",
    "मुझे": "mujhe", "लोग": "log", "क्या": "kya", "कुछ": "kuch", "बहुत": "bahut", "अच्छा": "accha", "ठीक": "theek",
    "हाँ": "haan", "हां": "haan", "लिए": "liye", "कर": "kar", "रहा": "raha", "रही": "rahi", "रहे": "rahe",
    "एक": "ek", "तब": "tab", "अब": "ab", "जब": "jab", "कब": "kab", "कैसे": "kaise", "क्यों": "kyun", "सब": "sab",
    "इस": "is", "उस": "us", "इसे": "ise", "उसे": "use", "अगर": "agar", "लेकिन": "lekin", "तक": "tak", "कोई": "koi",
    "बात": "baat", "काम": "kaam", "दिन": "din", "साथ": "saath", "बाद": "baad", "पहले": "pehle", "फिर": "phir",
    "होगा": "hoga", "होगी": "hogi", "होंगे": "honge", "गया": "gaya", "गई": "gayi", "गये": "gaye", "गए": "gaye",
    "हमें": "hamein", "आपको": "aapko", "उन्हें": "unhein", "जी": "ji", "सर": "sir",
    "चाहिए": "chahiye", "दिए": "diye", "किया": "kiya", "करो": "karo", "करेंगे": "karenge", "करना": "karna",
    # English words Whisper sometimes writes in Devanagari
    "मीटिंग": "meeting", "प्रोजेक्ट": "project", "प्रोजेक्ट्स": "projects", "ऑफिस": "office", "ऑफ़िस": "office",
    "क्लाइंट": "client", "टीम": "team", "बजट": "budget", "डिलीवरी": "delivery", "कंपनी": "company",
    "प्रोग्राम": "program", "प्रोग्रामिंग": "programming", "पाइथन": "Python", "पाइटन": "Python", "कंप्यूटर": "computer",
    "मोबाइल": "mobile", "फोन": "phone", "ईमेल": "email", "रिपोर्ट": "report", "डेटा": "data", "सॉफ्टवेयर": "software",
    "सर्वर": "server", "कोड": "code", "वीडियो": "video", "चैनल": "channel", "सब्सक्राइब": "subscribe",
    "लाइक": "like", "शेयर": "share", "ओके": "okay", "प्रॉब्लम": "problem", "इश्यू": "issue", "अपडेट": "update",
    "टारगेट": "target", "सेल्स": "sales", "मैनेजर": "manager", "टाइम": "time", "डेडलाइन": "deadline",
    "एक्शन": "action", "प्लान": "plan", "प्रोसेस": "process", "टेस्ट": "test", "रिव्यू": "review",
    "फीडबैक": "feedback", "कस्टमर": "customer", "प्रोडक्ट": "product", "मार्केटिंग": "marketing",
    "स्टार्ट": "start", "फाइनल": "final", "प्रेजेंटेशन": "presentation", "डिस्कशन": "discussion",
}

_RUN = re.compile(r"[ऀ-ॿ]+")


def _syllables(w):
    """Split one Devanagari word into syllables: dict(onset, vowel, schwa, tail)."""
    sy, onset, cluster, i, n = [], "", False, 0, len(w)
    while i < n:
        ch = w[i]
        if ch in CONS:
            c = CONS[ch]
            if i + 1 < n and w[i + 1] == NUKTA_MARK:
                c = NUKTA.get(ch, c)
                i += 1
            if ch == "ज" and i + 2 < n and w[i + 1] == HALANT and w[i + 2] == "ञ":
                c, i = "gy", i + 2  # ज्ञ -> gya
            onset += c
            nxt = w[i + 1] if i + 1 < n else ""
            if nxt == HALANT:
                cluster = True
                i += 2
                continue
            if nxt in MATRA:
                sy.append(dict(onset=onset, vowel=MATRA[nxt], schwa=False, tail="", cluster=cluster))
                i += 2
            else:
                sy.append(dict(onset=onset, vowel="a", schwa=True, tail="", cluster=cluster))
                i += 1
            onset, cluster = "", False
        elif ch in INDEP:
            sy.append(dict(onset=onset, vowel=INDEP[ch], schwa=False, tail=""))
            onset = ""
            i += 1
        elif ch in NASAL_MARKS or ch == VISARGA:
            if sy:
                nasal = "h" if ch == VISARGA else ("m" if i + 1 < n and w[i + 1] in LABIAL else "n")
                sy[-1]["tail"] += nasal
            i += 1
        elif ch in DIGITS:
            sy.append(dict(onset=DIGITS[ch], vowel="", schwa=False, tail=""))
            i += 1
        elif ch == "।":
            sy.append(dict(onset=".", vowel="", schwa=False, tail=""))
            i += 1
        else:
            i += 1
    if onset:  # trailing conjunct without a vowel (e.g. word ending in a halant)
        sy.append(dict(onset=onset, vowel="", schwa=False, tail=""))
    return sy


def _word(w):
    if w in COMMON:
        return COMMON[w]
    sy = _syllables(w)
    n = len(sy)
    if n >= 2 and sy[-1]["schwa"] and not sy[-1]["tail"]:
        sy[-1]["vowel"] = ""                     # word-final schwa is silent: kamal, kaam
    for i in range(n - 2, 0, -1):                # medial schwa deletion, right to left: karta, samajhna
        if sy[i]["schwa"] and not sy[i].get("cluster") and sy[i + 1]["vowel"] and sy[i - 1]["vowel"]:
            sy[i]["vowel"] = ""
    out = []
    for i, s in enumerate(sy):
        v = s["vowel"]
        if i == n - 1 and not s["tail"]:         # final long vowels are written short: hamara, bacchi
            v = {"aa": "a", "oo": "u"}.get(v, v)
        out.append(s["onset"] + v + s["tail"])
    return "".join(out)


def to_roman(text):
    """Transliterate every Devanagari run in `text`; everything else passes through unchanged."""
    return _RUN.sub(lambda m: _word(m.group(0)), text) if text else text


def has_devanagari(text):
    return bool(_RUN.search(text or ""))
