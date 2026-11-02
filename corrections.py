"""
The app learns from what you fix.

  1. When a transcript is saved, a copy of exactly what the app wrote is kept (corrections/<id>.json). The same happens when the
     final, re-checked version replaces the live one.
  2. You open the saved .txt and fix mistakes - a misheard name, a wrong word - and save the file as usual.
  3. The next time the app starts (and again just before every session starts) it compares each saved transcript with its copy,
     line by line. Small word fixes become rules in learned_glossary.txt:   kuber=Kubernetes   free you=Priya
  4. The glossary applies those rules to every later transcript, live and final, and uses the right spellings as hints for the
     speech models. Open learned_glossary.txt (right-click the island > "What it learned") to review or delete a rule.

Safety: a fix to a very common word ("the", "hai") is only trusted after you made the same fix twice; lines you rewrote
completely, speaker-name edits, deleted or added lines are ignored.
"""
import hashlib
import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

from paths import DATA_DIR

DIR = DATA_DIR / "corrections"
LEARNED = DATA_DIR / "learned_glossary.txt"
COUNTS = DIR / "pair_counts.json"
KEEP_DAYS = 45
HEADER = ("# Learned from the corrections you made in saved transcripts - one rule per line:  wrong=right\n"
          "# Delete a line to forget it. Applies from the next Start session. (You can also add rules here yourself.)\n")
LINE = re.compile(r"^\[(\d\d:\d\d:\d\d)\]\s?(.*)$")
LABEL = re.compile(r"^([^:\[\]]{1,32}):\s+(.*)$")


def _read(path):
    b = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("utf-8", "replace")


def parse(text):
    """-> {(timestamp, nth line with that timestamp): spoken text} - the speaker label in front of a line is dropped."""
    out, seen = {}, {}
    for raw in text.splitlines():
        m = LINE.match(raw.strip())
        if not m:
            continue
        ts, rest = m.groups()
        lm = LABEL.match(rest)
        if lm:
            rest = lm.group(2)
        k = seen[ts] = seen.get(ts, -1) + 1
        out[(ts, k)] = rest.strip()
    return out


def _bare(w):
    return re.sub(r"[^\w'\-]", "", w, flags=re.UNICODE)


def _norm(w):
    return _bare(w).lower()


def word_pairs(old, new):
    """Small word-level fixes between two versions of one line -> [(wrong, right)]."""
    a, b = old.split(), new.split()
    if not a or not b:
        return []
    na, nb = [_norm(w) for w in a], [_norm(w) for w in b]
    sm = SequenceMatcher(None, na, nb, autojunk=False)
    if sm.ratio() < 0.5:                                      # the line was rewritten, not corrected
        return []
    pairs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and 0 < i2 - i1 <= 4 and 0 < j2 - j1 <= 4:
            wrong = " ".join(_bare(w) for w in a[i1:i2]).strip()
            right = " ".join(_bare(w) for w in b[j1:j2]).strip()
            pairs.append((wrong, right))
        elif tag == "equal":                                  # same word, different capitals: rahul -> Rahul
            for x, y in zip(a[i1:i2], b[j1:j2]):
                bx, by = _bare(x), _bare(y)
                if bx != by and bx.lower() == by.lower() and len(by) >= 3:
                    pairs.append((bx, by))
    return [(w, r) for w, r in pairs if w and r and w != r and not r.isdigit() and not w.isdigit()]


def _common(w):
    try:
        from glossary import _is_common
        return _is_common(w)
    except Exception:
        return False


def _trusted(wrong, right, count):
    """One correction is enough for names, terms and phrases; an ordinary word must have been fixed the same way twice."""
    if all(_common(t) for t in wrong.split()) and all(_common(t) for t in right.split()):
        return count >= 2                                     # "the" -> "a": could be a one-off edit, wait for a repeat
    return True                                               # a misheard word turned into a name or term: clearly deliberate


# ---------------------------------------------------------------------------------------------------- snapshots
def _sid(path):
    return hashlib.sha1(str(Path(path).resolve()).lower().encode("utf-8")).hexdigest()[:16]


def snapshot(path, body):
    """Remember what the app wrote to `path` (the file the user will edit)."""
    try:
        DIR.mkdir(parents=True, exist_ok=True)
        (DIR / f"{_sid(path)}.json").write_text(
            json.dumps(dict(path=str(path), body=body, created=time.time()), ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _load_counts():
    try:
        return json.loads(COUNTS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _existing_rules():
    rules = {}
    try:
        for ln in LEARNED.read_text(encoding="utf-8").splitlines():
            if "=" in ln and not ln.strip().startswith("#"):
                w, r = ln.split("=", 1)
                rules[w.strip().lower()] = r.strip()
    except Exception:
        pass
    return rules


def _write_rules(rules):
    LEARNED.write_text(HEADER + "\n".join(f"{w}={r}" for w, r in list(rules.items())[-600:]) + "\n", encoding="utf-8")


def learn_pending():
    """Compare every saved transcript with its snapshot; learn the small fixes. -> list of new/changed rules (wrong, right)."""
    learned = []
    if not DIR.exists():
        return learned
    counts, rules = _load_counts(), _existing_rules()
    for f in sorted(DIR.glob("*.json")):
        try:
            snap = json.loads(f.read_text(encoding="utf-8"))
            p = Path(snap["path"])
            if not p.exists():
                if time.time() - snap.get("created", 0) > KEEP_DAYS * 86400:
                    f.unlink()
                continue
            cur = _read(p)
            old, new = parse(snap["body"]), parse(cur)
            changed = False
            for key, o in old.items():
                n = new.get(key)
                if n is None or n == o:
                    continue
                changed = True
                for wrong, right in word_pairs(o, n):
                    pk = f"{wrong.lower()}=>{right}"
                    counts[pk] = counts.get(pk, 0) + 1
                    if _trusted(wrong, right, counts[pk]) and rules.get(wrong.lower()) != right:
                        rules[wrong.lower()] = right
                        learned.append((wrong, right))
            if changed:                                       # later edits are compared with this version, nothing is learned twice
                snap["body"], snap["created"] = cur, time.time()
                f.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
            elif time.time() - snap.get("created", 0) > KEEP_DAYS * 86400:
                f.unlink()
        except Exception:
            continue
    if learned or (DIR.exists() and counts):
        try:
            COUNTS.write_text(json.dumps(counts, ensure_ascii=False), encoding="utf-8")
            if learned:
                _write_rules(rules)
        except Exception:
            pass
    return learned


def ensure_file():
    if not LEARNED.exists():
        LEARNED.write_text(HEADER, encoding="utf-8")
    return LEARNED
