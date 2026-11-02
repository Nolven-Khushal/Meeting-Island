"""Real clips from several different speakers (IndicVoices Hinglish, speaker ids given) -> bench/spk/clips/"""
import json, os, time, urllib.parse, urllib.request, collections
DS = 'dianavdavidson/indic-voices-hinglish-nospeakeroverlap-spon3.3-acronyms-fixed2'
by = collections.defaultdict(list)
for off in range(0, 1200, 100):
    u = f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(DS)}&config=default&split=test&offset={off}&length=100"
    for _ in range(4):
        try:
            for r in json.load(urllib.request.urlopen(u, timeout=90))['rows']:
                x = r['row']
                if 2.0 <= float(x['duration']) <= 12:
                    by[x['speaker_id']].append((x['audio_filepath'][0]['src'], float(x['duration']), x['gender']))
            break
        except Exception as e:
            time.sleep(3)
good = {s: c for s, c in by.items() if len(c) >= 8}
print(len(by), 'speakers seen;', len(good), 'with >= 8 usable clips:', {s[-6:]: (len(c), c[0][2]) for s, c in good.items()})
os.makedirs('bench/spk/clips', exist_ok=True); meta = []
for s, c in list(good.items())[:16]:
    for k, (url, dur, g) in enumerate(c[:12]):
        fn = f'bench/spk/clips/{s[-6:]}_{k}.audio'
        if not os.path.exists(fn):
            try: urllib.request.urlretrieve(url, fn)
            except Exception as e: print('skip', e); continue
        meta.append(dict(spk=s[-6:], gender=g, file=fn, dur=dur))
json.dump(meta, open('bench/spk/meta.json', 'w'), indent=1); print(len(meta), 'clips saved')
