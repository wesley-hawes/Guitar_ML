# Guitar_ML

**Building toward automatic guitar tablature from audio.**

Guitar_ML is a Python project that analyzes guitar recordings to identify which
notes are ringing and when new notes may have been played. The goal is to turn
a performance into a sequence of notes with timing, then map those notes to
guitar strings and frets to produce readable tabs.

**Currently, it produces a provisional note-entry table with note names, estimated
start times, and the evidence behind each row. It does not yet infer reliable
note endings or generate complete transcriptions or tabs.**

## What it does today

The current workflow reads WAV recordings and produces:

- **Pitch over time:** estimated frequencies and note names, such as A3 or C3.
- **Possible note starts:** locations where a rise in signal level suggests a
  new pick, including repeated picks of the same note.
- **Reviewable results:** plots and CSV files showing the measurements, which
  estimates passed the checks, and why others were rejected.

It runs offline through scripts or notebooks, using included guitar recordings
and separate experiments with annotated Guitar-TECHS data.

## How it works

The program examines short, overlapping sections of the recording along two
parallel paths:

1. **Estimate the pitch.** It compares the waveform with shifted copies of
   itself to find repeating patterns—a method called *autocorrelation*. The
   repetition rate gives a candidate frequency. Level, repetition strength, and
   consistency checks filter estimates before grouping them under note names.
2. **Look for a new attack.** It measures signal level using *RMS* and compares
   each section with the sound immediately before it. A strong rise suggests a
   new pick; nearby candidates are consolidated to avoid duplicate detections.
3. **Assemble a reviewable table.** Pitch groups supply notes; nearby amplitude
   rises or sharp decreases with changed pitch support their estimated starts.
   Duplicate cues for one entry merge. Additional rises within a sustained note
   propose repeats, flagged for review. Plots retain the underlying evidence.

These paths answer different questions. If you pick C twice, the pitch can stay
C throughout; a pitch tracker alone may treat it as one long note. Detecting the
second pick is essential for producing the right tab.

See the [technical design](docs/technical-design.md) for equations, window sizes,
and architecture.

## Where it is going

**The next step is to evaluate the combined table on new recordings**, checking
individual misses and extra rows, then add reliable note-ending estimates.

From there, the plan is to:

- Evaluate against annotated recordings kept separate from development.
- Map notes to strings and frets, resolving the multiple positions that can
  produce the same pitch.
- Extend the analysis toward expressive playing and, eventually, chords.
- Explore supervised ML against the DSP baseline and build an audio/tab review interface.

There is no trained ML model yet. Establishing a measurable baseline comes first.

## What still needs work

The system estimates one pitch at a time; it cannot separate chords or multiple
ringing strings. Harmonics and background sounds can produce incorrect pitches.

A soft note after a loud one can escape attack detection, while unwanted sounds
can create extra markers. In the fastest included phrase, pitch tracking finds
a transition to E3 that attack detection misses. **Correct pitches do not
automatically mean correct note timing.** A new [comparison](10_quieter_note_entries.ipynb)
uses a sharp decrease plus changed pitch to recover quieter entries; this
now feeds the combined table. Same-pitch quieter re-picks remain unresolved;
extra amplitude rises can still produce false repeated-note rows.

Bends, slides, vibrato, note endings, and varied recording conditions still need
dedicated evaluation. There is no established transcription-accuracy score yet.

## Try it

Developed and checked with Python 3.13.9. On macOS/Linux:

```bash
git clone https://github.com/wesley-hawes/Guitar_ML.git
cd Guitar_ML
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python personal_recordings_lesson.py
python note_event_pipeline.py
```

Run the scripts in that order from the repository root. They analyze the five
included phrase/repeated-note recordings; no external dataset is required.
Close plot windows to continue. Results are saved under
`outputs/pitch_tracking/personal_2026-11-03/`.

Tables are saved in the `note_events/` subdirectory as CSV and `note_events.md`.
For a walkthrough, open [notebook 08](08_personal_recordings.ipynb), then
[notebook 11](11_note_event_table.ipynb). Notebook 09 preserves the rise-only baseline. See the
[experiment guide](docs/experiments.md) for notebook setup, headless execution,
and the earlier investigations.

## Code and validation

The main implementations are [pitch estimation](pitch_analysis.py),
[attack detection](attack_analysis.py), [pitch grouping](note_events.py),
and [event integration](event_detection.py).
The stack is Python, NumPy, SoundFile, Matplotlib, and Mido, with Jupyter notebooks
for experiments.

**41 unit tests pass**, covering signal
calculations, candidate selection, grouping, and MIDI parsing. They verify code
behavior, not transcription accuracy. Run them with:

```bash
python -m unittest discover -v
```
