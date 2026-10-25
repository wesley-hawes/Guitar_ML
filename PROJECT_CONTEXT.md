# Guitar ML — project handoff

Last updated: 2026-10-24. Read this alongside README.md and the actual code.
Verify the current files and Git status before making changes; this document is
a checkpoint, not a substitute for inspecting the project.

## Purpose

The owner is a Statistics / Data Science student learning Python, software
engineering, signal processing, statistics, and eventually machine learning
through personally recorded guitar audio. Understanding the work matters more
than quickly producing a large application.

Long-term interests include pitch, timing, bends, vibrato, slides, performance
comparison, and possibly an LLM explanation layer. None of those are implemented.
Do not jump into ML or deep learning yet.

## How to collaborate

- Act as a tutor and pair programmer. Explain concepts, make a small change,
  review the result together, and check understanding before advancing.
- The user has authorized direct edits in VS Code's project and running checks.
  Keep edits small; this is not permission to build the entire roadmap at once.
- Explain math with intuition, a formula, a small example, and its Python mapping.
- Use concise, natural comments about reasoning and assumptions. Avoid comments
  that merely narrate obvious code, excessive docstrings, or elaborate architecture.
- Explain errors before fixing them and explain unfamiliar terminal commands.
  Give a few steps at a time, not a long command sequence.
- Teach Git through meaningful working milestones. Review staged files before
  committing; do not include environments or GarageBand editing projects.
- Call out statistical concerns when they arise, such as inconsistent measurement
  windows, recording gain, repeated takes, and later data leakage.
- Keep this handoff updated at meaningful milestones, including unresolved issues.

## Environment and files

- Project: `/Users/wesley/Documents/Coding-Projects/01-Active/Guitar_ML`.
- GitHub: https://github.com/wesley-hawes/Guitar_ML ; branch: `main`.
- macOS, VS Code, Python 3.13, local `.venv`; Conda `(base)` may also appear.
- Activate with `source .venv/bin/activate` from the project root.
- Install with `python -m pip install -r requirements.txt`.
- Run with `python main.py` from the project root. Close the plot to finish.
- The user sometimes runs bare `cd`, which returns to the home directory.
  Explain working directories when a file cannot be found; don't recreate files.
- `main.py`: audio loading, metadata output, RMS function, waveform display.
- `requirements.txt`: NumPy, SoundFile, Matplotlib (currently unpinned).
- `.gitignore`: excludes `.venv/`, `__pycache__/`, `.DS_Store`, and `*.band/`.
- `README.md`: current setup, behavior, and limitations.

## Current recording and behavior

`data/raw/Test_1_Cmaj.wav` contains a single C note, not a C-major chord.
The user recorded/exported it through GarageBand. The exact octave, interface,
gain, and effects settings are not confirmed; do not describe it as verified dry
audio. An earlier GarageBand screenshot showed a "Brit and Clean" amp preset.

Verified file properties:

- WAV, 16-bit PCM, stereo, sample rate 44,100 Hz.
- Loaded array: `(458356, 2)`, dtype `float64` using SoundFile defaults.
- Duration: approximately 10.39356 seconds.
- Full-recording left-channel RMS: approximately 0.12186434.
- The first five rows contain zeros in both channels.
- The displayed waveform has a large onset around 0.85 seconds and a decaying tail.
  A near-full-scale peak alone does not establish clipping.

The script assumes stereo input and uses a fixed relative path. It plots only
the left channel and does not save the figure. No windowed RMS, zoomed plot,
FFT, pitch tracking, or ML is implemented. There is no automated test suite yet.

The script has run successfully. RMS was checked with an alternating
`[0.5, -0.5, 0.5, -0.5]` array (expected 0.5), silence (expected zero), and
doubling the signal (expected twice the RMS). These were development checks,
not committed test files.

## Concepts covered

- WAV data vs a GarageBand `.band` project; renaming is not conversion.
- Samples, sample rate, stereo columns, time-step rows, and floating-point data.
- Duration = row count / sample rate; channels share the same timeline.
- Time array = sample indices / sample rate; selecting `samples[:, 0]`.
- Waveform overview vs rapid oscillations visible in a close-up.
- RMS = square root of mean squared amplitude. Ordinary signed means can cancel.
- Digital amplitude and RMS are not perceived loudness or pitch.
- The user correctly explained that adding zeros reduces RMS: the squared sum
  stays constant while the sample count increases.
- Time-window slicing was explained but has not yet been added to the code.

## Git checkpoint and next steps

The analysis milestone is committed locally as `b98e89a`, with message
"Program to read wav file and preform amplitude analysis". It includes the code,
README, requirements, example WAV, and ignore rules. Confirm push status rather
than assuming it. This handoff was added after that milestone.

Next, finish documenting/committing this handoff, then teach a selected time
window and compare its RMS with whole-recording RMS. Keep the window explicit
and explain selection bias: choosing a louder section changes the measurement.
A short waveform close-up can illustrate indexing and oscillations.

Later: meaningful RMS tests, repeat recordings under consistent conditions,
zero-crossing rate, FFT, STFT/spectrogram, spectral centroid, then pitch tracking.
Introduce modules and additional structure only when useful. For future ML,
favor interpretable features and classical baselines before neural networks.
