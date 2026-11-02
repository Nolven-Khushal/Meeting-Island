# Features, screen by screen

Everything Meeting Island does, in the order a meeting uses it. All of it runs on the PC, offline.

## The island (always on top, under the camera)

| Part | What it does |
|---|---|
| **Ready pill** | A slim glass pill at the top centre, "Meeting Island" with a round start button. Click it to open the panel. |
| **Panel** | The full view: status line (state, elapsed time, detected language), live bars that move with the sound, the transcript, and the buttons. |
| **Status text** | "Listening · 00:35 · HI". The language is information, never a button. A hint appears if nothing is heard ("No sound heard: is the volume muted?"). |
| **Start session** | One click. Begins listening to the chosen source. |
| **Pause / Resume** | Holds the same session. Paused time never enters the timestamps. |
| **Copy** | Copies the transcript so far to the clipboard. |
| **Minimise** | Collapses the panel to the pill (and back) with a spring animation. |
| **End session** | Finishes the last sentences, then opens the Save window. |
| **Circle** | While you hold and drag it, the island becomes a small circle. |
| **Dock** | Dropped at the left or right edge it becomes a slim vertical bar; dropped near the camera it snaps home. |
| **Two glass looks** | *Liquid* (clear, with an elastic swell when the pointer approaches) and *Frost*. Both blur whatever is behind them. |
| **Recover** | If the app closed mid-meeting, the next start offers the unsaved session. |

## Right-click menu

Version line · Glass: Liquid / Frost · **Listen to**: PC audio (calls, videos), microphone (a meeting in the room), or PC audio plus
your microphone · **Record**: screen + audio, audio only, or off · Hindi shown as Hinglish (Roman letters) or in Devanagari ·
Edit glossary · What it learned from your edits · Open the transcripts folder · Quit.

## Live transcription

- English, Hindi and Hinglish in one stream; the language is detected per sentence.
- **Drafts** (grey, italic) appear about every half second while someone speaks and settle into a **final** line at the pause. Finals
  arrive about 0.6-0.8 s after a pause in English and 1.5-2.5 s in Hindi.
- Hindi can be shown as Hinglish in Roman letters (default) or in Devanagari.
- Sentences are not broken: a speaker who never pauses is cut at the quietest moment, the overlap is re-read, and the pieces are
  stitched back into one line.
- **Speakers** get a colour and "Speaker 1, 2, ..." as the meeting goes. The microphone is "You".
- A **personal glossary** (names, products, terms; `wrong=right` rules) steers the models and fixes spellings by sound, without ever
  rewriting ordinary words.

## Save window

Name, folder (default: Documents\Meeting Transcripts), a note that the file is `.txt` with `[hh:mm:ss]` on every line, optional
names for each detected voice (with a snippet of what they said), and, when the meeting was recorded, **Keep the recording next to the
transcript**. Buttons: Save, Keep for later, Discard (asks twice).

## After Save: the final transcript

- The live transcript is saved immediately.
- A background pass listens to the recording again, slowly: it finds who speaks when (offline diarization plus the project's own voice
  clustering), cuts only at pauses, and transcribes each piece with the full pipeline. The saved file is replaced by the final version
  (or saved next to it as "(final)" if you edited the original meanwhile). Speaker names you typed carry over.
- The island shows a thin progress line and "final transcript NN%". The pass waits while a new meeting is recording and while the
  battery is low, runs at low priority, and resumes after a restart.
- Names read from the screen (see below) are filled in when they were seen consistently.
- The recording is written as an `.mp4` (screen + audio) or `.m4a` (audio only).

## Recording

Screen (GPU desktop duplication, about 1 ms per frame, with a slower fallback) and audio per source, following the session clock so
the video lines up with the transcript. Stops filming (keeps the audio) if the disk is nearly full.

## Names from the screen

While recording, the full-resolution screen is checked about twice a second for the ring that video-call apps draw around the person
speaking; the name on that tile is read with Windows' built-in offline text recognition and matched to the voice that was talking.
Only used when consistent; otherwise the line stays "Speaker N".

## It learns from corrections

Fix a word in a saved transcript and save the file. At the next start (and just before every session) the app compares the file with
what it wrote and keeps the small fixes as rules in a plain-text file you can read and edit. A notice says how many it learned.

## Installing and updating

- **Setup.exe**: one file, no Python, no downloads, no admin rights; installs the program and the models, adds shortcuts and an
  "Apps & features" entry, and upgrades an older install in place keeping settings.
- **Update.exe**: a small patch with only the changed files; checks every file; restores the previous version by itself on failure.
- **Uninstall**: removes the program, optionally the settings; never the saved transcripts.

## Tools in the repository

Benchmark of speech systems on real clips (`bench/`), speaker-labelling experiments, a fine-tuning pipeline for a free GPU with a
ship/no-ship gate (`finetune/`), and the tooling that recorded the film in this showcase (`tools/showcase/`).
