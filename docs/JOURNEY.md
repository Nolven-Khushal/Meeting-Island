# How Meeting Island was built

The story in chapters: what was asked, the choices that were made (and the ones that were thrown away), the hard problems, the
mistakes, and where it stands. Numbers are real measurements from this project; no private data appears here.

## 1. The brief

Writing the minutes of a meeting takes longer than the meeting. The ask: a tool that listens to a meeting and writes it down while it
happens, in English, Hindi and the Hinglish people actually speak, and saves a timestamped text file at the end.

Constraints that shaped everything: **offline** (no API, nothing leaves the PC), an ordinary laptop (6 cores, about 6 GB of free RAM,
no GPU), and a first user who is not technical: an executive assistant who should install it by double-clicking one file.

## 2. First version: right idea, 14 seconds late

The first demo recorded what the PC was playing (Windows WASAPI loopback) and ran Whisper in chunks. It worked on English and lagged
about 14 seconds. The cause was not the model: the usual Whisper wrapper pads every chunk to 30 seconds, so a "small" model spent a
fixed 2.5 s per call and "turbo" about 10 s regardless of how short the sentence was.

*Decision:* call CTranslate2 directly with features for only the real audio, and show a live **draft** every ~0.5 s while a person
speaks, replaced by a **final** when they pause. Finals arrive about 0.6-0.8 s after a pause in English and 1.5-2.5 s in Hindi.

*Rejected:* running everything through one big model (too slow on CPU), and streaming the audio to a cloud API (breaks the brief).

## 3. The island

The user wanted the transcript to sit under the camera like a phone's dynamic island, draggable, and "very smooth, like iOS".
Tkinter could not do it. The UI became PySide6 (Qt) with:

- a **frosted / liquid glass** panel that really blurs the desktop behind it (the screen behind is grabbed, blurred and drawn under
  the window), with the window excluded from capture so it never films itself
- **spring physics** (stiffness 400, damping 30) for size, position and corner radius, and a vsync thread (`DwmFlush`) so a frame is
  drawn per screen refresh instead of by a free-running timer
- a **fixed-size window canvas**: the window only ever moves, never resizes; the shape is drawn inside it. Resizing a translucent
  window every frame was the source of most early flicker.
- drag behaviour: a circle while held, a vertical bar when dropped at an edge, snap-home near the camera

Measured: about 17-20 ms per frame, p95 under 25 ms.

## 4. A session, not a stream

*Decision:* one-click Start, Pause/Resume, End, then a Save window (name, folder, `.txt` with `[hh:mm:ss]` on every line).
Paused time is not counted in timestamps. The transcript autosaves while the session runs, so a crash or an accidental close offers
"Recover" instead of losing the meeting. Language shows as status text, not a button: the app detects it.

## 5. Hindi was bad, and the cause was partly my own bug

After the first real Hindi test the verdict was blunt: accuracy "pretty bad". Instead of tuning by feel, the project built a
**benchmark**: 80 real clips with human transcripts (40 spoken software tutorials, 40 spontaneous Hinglish conversation), scored with
a spelling-tolerant method (so *hai/he* or *kaam/kam* are not errors, a wrong word still is).

| System | Words wrong |
|---|---|
| First version (Whisper small + turbo) | 69.7% |
| Same, with the bug below fixed | 56.0% |
| Hinglish Whisper (Swift), no padding | 46.7% |
| Hinglish Whisper (Swift), padded to 15 s | 37.3% |
| IndicConformer (Hindi, CTC) alone | 38.8% |
| **Swift + IndicConformer fused** | **30.3%** (conversation 23.7%) |

What the measuring found:
- **My bug:** output was capped at about 9 tokens per second of audio. Devanagari needs 4-6 tokens per *word*, so long Hindi sentences
  were cut off.
- **Padding matters:** feeding Whisper only the real audio cost about 10 points; padding to 15 s (half the cost of 30 s) won them back.
- **The two models are complementary.** IndicConformer is far better on conversational Hindi and cannot loop or invent sentences, but
  writes English words in Devanagari. Swift spells English properly. The fusion keeps IndicConformer's words, borrows Swift's English
  spellings and punctuation, and falls back to Swift when IndicConformer dropped words.

Ideas that were measured and **rejected**: loudness normalisation (worse), beam search (+0.7 but loses per-word confidence), a generic
368-term glossary prompt (no gain), a Hindi-specialised Whisper small (56.6%), a large Hinglish Whisper (32.8% alone at 2.2x real time;
only 0.5 point better when fused). The Hindi acoustic model is the ceiling, and a request for "96% accuracy" was answered honestly:
not reachable offline on a CPU.

## 6. Fast talkers and broken sentences

A fast speaker pauses rarely, so sentences were cut mid-way. Fixes: 14 s chunks cut at the quietest moment, a 0.9 s re-read overlap,
dropping the half-heard last word of a forced cut, and fuzzy stitching of the overlap back into one line. Speeding clips up showed the
cost: about +4 points of errors at 1.3x speech, +11 at 1.6x.

## 7. Who said it

Voice prints (NVIDIA TitaNet-small via sherpa-onnx) were compared on 177 real clips of 16 people: 98.0% pair accuracy against 90.8%
(WeSpeaker) and 84.8% (CAM++). Online assignment (threshold tuned on 8 people, tested on 8 others) was 97-100% on tidy conversations,
and an end-of-meeting re-clustering fixed early mistakes.

It broke on **real turn-taking**: with 0.1-0.8 s between speakers the live method merges two people into one chunk. On hard,
fast conversations it labelled only 72.7-86.7% of turns correctly. The fix came later, in the final pass (chapter 9).

## 8. Shipping to a non-technical user

- **Installer:** PyInstaller folder build + the models + the Visual C++ runtime, wrapped by Windows' own IExpress into one
  self-extracting `Setup.exe` (about 410 MB). Per-user install (no admin), shortcuts, an "Apps & features" entry, an uninstaller that
  never deletes saved transcripts.
- **A gotcha worth recording:** IExpress silently launches nothing if the command starts with `cmd /c`; the bare file name works.
- **Updater:** each release stores a manifest (size + SHA-256 of every file). The update package carries only the files that changed
  (about 16 MB), verifies them, swaps them in with a backup, checks every file of the new version, and **puts the old version back by
  itself if anything fails**. Tested against a real earlier install, including a deliberately broken one.

## 9. The second look: recording, a final pass, and a mistake

A recording of the session (screen via GPU desktop duplication, about 1 ms per frame; audio per source) lets the app listen again after
the meeting with the whole conversation in view, and replace the live text with a final version.

**The first final pass was worse than live** (43.3 vs 44.7, 39.5 vs 43.3, 34.0 vs 36.6 words wrong on three test streams). Two causes,
found one at a time:
1. It cut the audio at every 0.3 s pause, so the speech models got tiny fragments. Packing phrases into pieces of up to 13 s, cut only at
   real pauses, turned it into a win (37.8 → 35.2 weighted).
2. It cut at every speaker change reported by the diarizer, even in the middle of a word ("Che" / "taavnitaani"). A real Hindi TV-comedy
   clip made it visible. Speaker changes are now cut only at a real pause (or not at all), and a lone "haan" no longer becomes its own
   piece. Result: **34.0 vs 37.8 for live, better on all three streams.**

Speaker labels got better for a different reason. Offline diarization (pyannote segmentation) finds the turn boundaries but over-splits
one person into several speakers (5.5 speakers for 2 people); the project's own TitaNet clustering merges them correctly. The
combination labelled **91.3% of turns correctly against 79.6% for the live method** (six hard conditions).

**A measurement trap:** on the TV clip, scoring against YouTube's captions made the fusion look 3 points worse, because the captions
spell English loanwords in Devanagari while the app writes them in English. The honest reading: on that clip final and live are about
equal (37.1% vs 37.4%), and the metric cannot say more.

## 10. A two-hour meeting must not hurt the laptop

Reasoning about a 2-hour meeting found real problems before any user did: the end-of-meeting speaker clustering was O(N³) (about a
minute at 1,500 sentences; now O(N²), 1.7 s, identical labels), autosave rewrote the whole transcript after every sentence (now at
most every 4 s), the final pass copied 2 hours of float audio into RAM (now memory-mapped, converted a piece at a time), and building
the recording loaded everything (now streamed). The final pass runs at low priority, waits while a new meeting is being recorded and
while the battery is under 50%, and checkpoints every piece so a closed app resumes instead of restarting.

## 11. It learns from corrections

Open a saved transcript, fix a word, save. At the next start the app compares the file with what it wrote and keeps the small fixes as
rules (`kuber netes=Kubernetes`). Names and terms are trusted at once; a fix to a very common word only after it was made twice.
Speaker renames and fully rewritten lines are ignored.

## 12. Names from the screen

Video-call apps ring the tile of the person speaking and write their name on it. While recording, the app finds the ring on the
full-resolution frame (a saved video is too small to read) and reads the name with Windows' built-in offline text recognition. A voice
is named only when the same name was seen consistently while it spoke. Tested on made-up call screens: the ring was found on 20 of 20
tiles (about 70 ms per frame), and 3 of 3 voices were named correctly even with 20% of readings corrupted. Not yet validated on a real
call; it assumes Zoom's ring colour.

## 13. Where it stands, and what comes next

- Roughly 75-80 of 100 words right on normal Hinglish conversation, 60-70 on technical or very fast talk; English is not formally
  benchmarked.
- A fine-tuning pipeline (`finetune/`) trains the Hinglish model further on a free GPU, with a gate that only lets a model ship if it
  beats the current one on the project's own clips. Smoke-tested end to end.
- An Android version is planned: the Hindi model was verified to run in the same engine that has an Android build.
- Not done: remembered voices (name a person once, recognised in every meeting), real-call validation of the screen names.
