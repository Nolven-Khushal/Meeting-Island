"""Download a fixed, reproducible sample of test clips (+ reference transcripts) from two public Hinglish test sets."""
import json, sys, time, urllib.request, urllib.parse, os
OUT = 'bench/data'; os.makedirs(OUT, exist_ok=True)
def rows(ds, offset, length=100):
    u = f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(ds)}&config=default&split=test&offset={offset}&length={length}"
    for _ in range(4):
        try:
            return json.load(urllib.request.urlopen(u, timeout=90))['rows']
        except Exception as e:
            print('retry', e); time.sleep(3)
    return []
def audio_url(v):
    return v[0]['src'] if isinstance(v, list) else v['src']
items = []
# 1) MUCS 2021 Hinglish (spoken tutorials, mixed script refs). Take English-heavy-ish + Hindi-heavy mix, every ~75th row.
for off in range(0, 3100, 155):
    for r in rows('dianavdavidson/MUCS-Hinglish', off, 2):
        x = r['row']; items.append(('mucs', x['segment_id'], audio_url(x['audio']), x['transcript']))
# 2) IndicVoices spontaneous Hinglish conversations: keep 3-15 s clips that contain English words.
picked = 0
for off in range(0, 20000, 500):
    for r in rows('dianavdavidson/indic-voices-hinglish-nospeakeroverlap-spon3.3-acronyms-fixed2', off, 100):
        x = r['row']
        if 3 <= float(x['duration']) <= 15 and int(x['count_english_words'] or 0) >= 1 and picked < 40:
            items.append(('ivoices', f"iv{off}_{picked}", audio_url(x['audio_filepath']), x['hinglish_mixed_scripts'])); picked += 1
            break
    if picked >= 40: break
meta = []
for src, sid, url, ref in items:
    fn = f"{OUT}/{src}_{sid}.audio"
    if not os.path.exists(fn):
        try:
            urllib.request.urlretrieve(url, fn)
        except Exception as e:
            print('skip', sid, e); continue
    meta.append(dict(src=src, id=sid, file=fn, ref=ref))
json.dump(meta, open('bench/meta.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('clips:', len(meta), {s: sum(1 for m in meta if m['src'] == s) for s in ('mucs', 'ivoices')})
