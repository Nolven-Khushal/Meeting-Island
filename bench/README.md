# Hindi / Hinglish speech benchmark (11 Oct 2026)

**Data:** 80 real clips with human transcripts, about 10 minutes of audio.
- 40 from MUCS-2021 Hindi-English code-switching: spoken software tutorials (`dianavdavidson/MUCS-Hinglish`, test split).
- 40 from IndicVoices: spontaneous Hinglish conversation, 3–15 s clips that contain English words
  (`dianavdavidson/indic-voices-hinglish-…-spon3.3-acronyms-fixed2`, test split).

`fetch.py` downloads the same sample again; `run.py <systems>` scores them.

**Scoring:** every reference and every output is put into Roman script, then into a spelling-tolerant
"skeleton". So *hai/he*, *kaam/kam* and *mein/me* are equal, but a different word is still an error.
WER is word error rate and CER is character error rate (lower is better). Speed is processing time ÷ audio
time on this laptop's CPU (Ryzen 5 5500U).

| System | Lecture WER/CER | Conversation WER/CER | All WER/CER | Speed |
|---|---|---|---|---|
| Old app: Whisper small + turbo (truncation bug) | 62.9 / 43.3 | 73.9 / 62.1 | 69.7 / 54.0 | 0.46 |
| Old app with the truncation bug fixed | 65.7 / 43.5 | 50.2 / 30.8 | 56.0 / 36.3 | 0.55 |
| Oriserve Whisper-Hindi2Hinglish-Apex (turbo) | 64.7 / 44.9 | 40.4 / 29.1 | 49.5 / 35.9 | 0.47 |
| Oriserve Swift, no padding | 57.9 / 39.3 | 39.9 / 25.3 | 46.7 / 31.4 | 0.05 |
| Oriserve Swift, padded to 15 s | 41.3 / 26.6 | 34.9 / 20.2 | 37.3 / 23.0 | 0.07 |
| AI4Bharat IndicConformer (Hindi, int8 ONNX) | 60.1 / 45.9 | 26.0 / 13.4 | 38.8 / 27.4 | 0.18 |
| **Shipped: Swift + IndicConformer fusion** | **41.3 / 26.8** | **23.7 / 12.5** | **30.3 / 18.7** | Swift 0.07 + IndicConformer 0.18 on Hindi lines |

## What we learned

- **Truncation bug in the old app.** It capped Whisper's output at about 9 tokens per second of audio.
  Devanagari needs 4–6 tokens per word, so long Hindi sentences were cut off.
- **Padding matters.** Feeding Whisper only the real audio, with no padding, made it fast but cost about 10 WER points
  on Swift. Padding with silence to 15 s recovers all of that, at half the cost of the standard 30 s.
- **The two models are complementary.** IndicConformer is by far the best on conversational Hindi. It does not
  generate text word by word, so it cannot loop or invent sentences. But it writes English words in Devanagari
  ("डॉक्यूमेंट") and sometimes drops English-heavy sentences entirely. Swift spells English properly.
  `fusion.py` keeps IndicConformer's words and borrows Swift's English spellings. It falls back to Swift when
  IndicConformer dropped words, and uses Swift alone for clearly English sentences.
- **Sound matching.** It uses Double Metaphone plus a spelling check and fixes loanwords neither model spelled right
  (pements → payments, resipi → recipe). Without the spelling check it hurt accuracy, so the check stays.
- **Lecture clips.** Their references are inconsistent (the same English word sometimes in Latin, sometimes in
  Devanagari), which inflates every system's lecture score.

## Live latency

From the end of a sentence to its final text, including about 0.75 s spent confirming the pause:
- English: 0.6–0.8 s.
- Hindi: 1.5–2.6 s.
- Drafts refresh about every 0.66 s while someone is talking.

## Round 2: trying to push accuracy further (11 Oct 2026)

Goal asked for: 96% accuracy. **Not reachable offline on a CPU**: published results for the best paid cloud systems on
conversational Indian speech are 15–25% word errors, and two human transcribers differ on 5–10% of words.
Everything below was measured on the same 80 clips. WER = spelling-tolerant | sound-based (English words written in
Devanagari in the references compare by sound).

| Change | Result | Kept? |
|---|---|---|
| Shipped pipeline (baseline) | 30.5 / 26.4 (conversation 23.9 / 20.7) | yes |
| Loudness normalisation | Swift 37.5 → 39.2, IndicConformer 38.8 → 39.6 (worse) | no |
| Beam search 5 for Swift | 37.3 → 36.6 (+0.7) but loses per-word confidence | no |
| Hinglish style prompt for Swift | 37.3 → 36.3 (+1.0). Hurts plain English ("grew" → "guru") | yes, only while the talk is in Hindi |
| Generic 368-term glossary prompt | 36.4 (no gain) | no |
| Hindi-specialist Whisper small (Vistaar) | 56.6 | no |
| Apex padded to 30 s (large model) | 32.8 alone at 2.2x real time; fusing it gives 30.0 vs 30.5 | no (not worth 2x real time) |
| Bigger fusion tweaks (donor model, thresholds) | all 30.0–30.6: the Hindi acoustic model is the ceiling | n/a |
| Glossary of generic terms (stress test) | 30.5 → 31.5 (hurts) | the feature ships for *your* names/terms, not as a general gain |

Remaining errors on conversation: 13% substitutions, 6% missed words, 2% extra words. Many substitutions are spelling
variants the reference itself writes inconsistently (thora/tho, zyaada/yaada).

### Fast speech (clips sped up with pitch preserved)

| Speed | Lecture | Conversation | All |
|---|---|---|---|
| 1.0x | 41.5 / 35.9 | 23.9 / 20.7 | 30.5 / 26.4 |
| 1.3x | 46.3 / 42.7 | 27.2 / 24.2 | 34.4 / 31.2 |
| 1.6x | 51.1 / 47.5 | 35.1 / 32.0 | 41.1 / 37.8 |

Errors grow about 4 points at 1.3x and 11 points at 1.6x. Fast speakers also pause less, so sentences get cut mid-way:
fixed with 14 s chunks, a 0.9 s re-read overlap, dropping the half-heard last word of a forced cut, and stitching the
pieces back into one line (backend.stitch, fuzzy overlap matching).

## Speaker labels (11 Oct 2026)

Test data: 177 real clips of 16 different people (IndicVoices Hinglish, speaker ids given), tuned on 8 speakers and
tested on 8 the models had never seen. Conversations were built by mixing clips of 2-5 speakers.

| Voice model | Same-person vs different-person similarity | Pair accuracy | Speed |
|---|---|---|---|
| NVIDIA TitaNet-small | 0.68 vs 0.18 | 98.0% | 0.014x |
| WeSpeaker ResNet34-LM | 0.84 vs 0.64 | 90.8% | 0.024x |
| 3D-Speaker CAM++ | 0.57 vs 0.50 | 84.8% | 0.014x |

Online labelling (threshold 0.45) on unseen speakers: 2 speakers 100%, 3: 98%, 4: 99%, 5: 97%.
End-of-session re-clustering (threshold 0.40 + absorbing 1-2 sentence stray clusters): 2: 100%, 3: 100%, 4: 99%, 5: 96%,
speakers found 2.2 / 3.4 / 4.4 / 5.2. Full pipeline on a 98 s meeting of 3 unseen speakers (two women): 22/22 sentences right.
Splitting a speaker change INSIDE one chunk: pyannote segmentation finds it 70-85% of the time with 8-15% false splits;
a TitaNet window detector detects 80-90% but with 10-30% false alarms and poor boundaries. Not reliable, so not used.
