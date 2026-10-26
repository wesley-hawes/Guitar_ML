# Guitar ML — project handoff

Last updated: 2026-10-25. Read this alongside README.md and the actual code.
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
- Run comparisons with `python compare_recordings.py` from the project root.
  Close both plots to finish. `python main.py` still runs the earlier single-file demo.
- The user sometimes runs bare `cd`, which returns to the home directory.
  Explain working directories when a file cannot be found; don't recreate files.
- `main.py`: reusable audio loading/RMS functions and the earlier single-file demo,
  now guarded by `if __name__ == "__main__"` so imports have no plotting side effects.
- `compare_recordings.py`: onset-aligned batch amplitude comparisons.
- `test_compare_recordings.py`: seven focused unittest checks.
- `requirements.txt`: NumPy, SoundFile, Matplotlib (currently unpinned).
- `.gitignore`: excludes `.venv/`, `__pycache__/`, `.DS_Store`, and `*.band/`.
- `README.md`: current setup, behavior, and limitations.

## Current comparison milestone — 2026-10-25

The user explicitly requested a loop over WAVs with granular amplitude analysis
and comparisons anchored to each note's onset sample. This is now implemented
in `compare_recordings.py`; the single-file exercise remains in `main.py`.

- Loops through `data/raw/*_note_dry_take*.wav`; currently six files (A, C, low E,
  two takes each). Excludes the older processed/unknown-conditions `Test_1_Cmaj.wav`.
- Uses the left channel (or mono), without per-file normalization. Requires finite
  audio and a full 8 seconds after onset; unusable files are skipped with a reason.
- Estimates background RMS from the first 0.5 seconds, which must be quiet.
  Finds three consecutive 10 ms frames above 6 times that RMS (floor 1e-5),
  then selects a nearby sample threshold crossing. This is a heuristic, not a
  validated or exact physical onset. Brief clicks are rejected, but sustained
  handling noise or soft attacks can still fool it. Only the first event is used.
- `ONSET_OVERRIDES` accepts manually reviewed original-file sample indices.
- Measures consecutive 50 ms windows from the onset over the same 8-second span.
  At 44,100 Hz: 2,205 samples/window, 160 windows/file. Partial windows are
  dropped, never zero-padded. Relative time is `(sample_index - onset) / sr`.
- Plots RMS and a 0.10–0.15 second post-onset waveform close-up, grouping takes
  by note with shared amplitude scales. Alignment is by onset, not waveform phase.
- Saves `comparisons.png`, `onset_review.png`, and `measurements.csv` to
  `outputs/amplitude_comparison/` (ignored generated output). CSV includes onset
  method, background threshold, original start/exclusive-stop indices, relative
  window midpoints, RMS, and absolute peaks. No pitch/FFT/ML was added.
- Detected onset samples in the current files: A take01 89373, A take02 88489;
  C take01 68649, C take02 89384; low E take01 68712, low E take02 108551.
  Visually reviewed the onset plots; estimates include early attack transients
  and should not be interpreted as precise starts of periodic string motion.
- Seven unittest checks passed, covering silence-offset invariance, isolated
  clicks, noise-only input, gain scaling, two sample rates, partial/short windows,
  and mono/manual overrides. Real-file checks verified 960 CSV windows, original
  sample boundaries, equal comparison spans, and no data cropped by plot limits.
- The batch milestone originally had no notebook. A later guided quiet-region
  lesson is now available; see its setup/status below.

## Guided quiet-region lesson and notebook setup — 2026-10-25

- Created `01_understanding_quiet.ipynb` and matching `quiet_region_lesson.py`.
  Six short code cells cover loading, waveform inspection, selecting an editable
  quiet interval, one RMS calculation, looping over equal windows, and plotting
  a candidate threshold. The batch detector is unchanged.
- Uses C take01 with candidate quiet interval 0.20–1.00 seconds (samples
  8820:44100); this is a proposed human-reviewed region, not automatic silence
  detection or a fixed requirement for all files. There are 16 complete 50 ms
  windows. Quiet median RMS is about 0.00015758, maximum about 0.00016201.
- Candidate threshold is 1.5 times that maximum, about 0.00024301. This is an
  explicit teaching heuristic, not a validated threshold. Being above the same
  windows used to choose it is not independent validation. Start/end decisions
  and sustained-threshold logic are deliberately not implemented in the lesson.
- Executed all six code cells in sequence using the project interpreter and
  saved example outputs in the notebook; validated structure using Anaconda's
  nbformat. This was NOT a live Jupyter-kernel test. Inspected final plot.
- The user asked to install notebook support. Project `.venv` lacks ipykernel;
  VS Code lacks Microsoft's Jupyter extension. Anaconda base already has Jupyter.
  Pip and extension install attempts failed due DNS/network resolution from the
  tool session (pypi.org and marketplace.visualstudio.com). No packages or
  extensions were installed, and no kernel was registered.
- Added `requirements-notebook.txt` and README setup commands. User should run
  installation in their own terminal, install `ms-toolsai.jupyter`, and select
  `Guitar ML (.venv)` after registering it. The lesson script already works with
  existing dependencies if notebook setup is still unavailable.
- Next teaching step: open the notebook, select the project kernel once setup
  succeeds, run only the first two cells, and discuss the waveform before
  continuing. Then review the candidate quiet region and one RMS calculation.

### Additional quiet-region check — A take01

The user confirmed no playing during `A_note_dry_take01.wav` from 0 to 1.2
seconds. Added section 7 to `01_understanding_quiet.ipynb` and the matching
Python lesson. Kept the C-derived threshold fixed at approximately 0.00024301;
did not refit it to A. Used A samples 0:52920, split into 24 complete 50 ms
windows of 2,205 samples. A background RMS min/median/max were approximately
0.00016459 / 0.00016922 / 0.00026708.

One window exceeded the threshold: samples 46305:48510, 1.05–1.10 seconds.
The other 23 were below. This is a false alarm for a simple per-window activity
rule, since the user identifies this interval as no playing. Its cause is unknown;
do not claim it is a click or note. Saved measured output and graph in the
notebook and validated the cells. No automatic detector was changed.

Next small lesson: inspect the flagged waveform interval, then discuss a
persistence rule (multiple consecutive windows) and its onset-delay tradeoff.
The 24 adjacent windows are not independent recording sessions and do not
establish a general error rate. Keep the threshold fixed while examining this.

### Strong-and-sustained start experiment — notebook section 8

Added two small code cells to the notebook and matching lesson script. Uses the
median of A take01's user-confirmed 0–1.2 second quiet windows as background.
Explicit provisional settings: window RMS > 10 times background for two
consecutive 50 ms windows (100 ms total). The median baseline is about
0.00016922 and trial threshold about 0.00169218. This is separate from the
unchanged C-threshold check in section 7 and does not modify the batch script.

On A take01, the first qualifying pair begins at sample 88200 (2.000 s), with
confirmation available at sample 92610 (2.100 s). Ratios are about 39.90 and
93.74; the highest known-quiet ratio is 1.58. Verified the earliest qualifying
pair and 4,410-sample duration; inspected the saved plot. The reported start is
a 50 ms window boundary, not sample-accurate physical onset. Returning to the
start of that window avoids assigning the later confirmation time as onset.

Keep the 10× / 100 ms settings fixed for new D/G recordings before evaluating
changes. This A example is development data, not held-out validation. Its quiet
excursion is rejected by the strength condition alone, so it does not validate
the persistence condition. Soft/short notes can be missed and sustained noise
can pass. Do not reuse the high start threshold for detecting the note end.

Next teaching checkpoint: user reads section 8 and explains why detection is
confirmed after two windows but the candidate start is the first window boundary.
Then evaluate the same rule on new takes with user-confirmed quiet regions.

Learning feedback on 2026-10-25: the user finds the comparison script too much
of a black box. Slow down, explain sample rate, indices, RMS windows, and every
chosen constant before adding algorithmic complexity. The user specifically
wants variable leading/trailing silence handled and fair note comparisons.

Current limitations remain: the detector assumes a quiet first 0.5 seconds,
and no note-end detector exists. The 8-second span can include background after
a short note or omit part of a long decay. Passing numerical tests does not
validate musical boundaries or make these recordings a controlled benchmark.

Next design/learning step (not yet implemented): inspect one recording, identify
a representative quiet region and the note's start/end as original sample indices,
and distinguish natural decay into noise from a mute or file cutoff. Use reviewed
boundaries as reference cases before changing automatic start/end detection.
Quiet-region selection must not assume a fixed 0.5-second prefix; if a reliable
noise-only region is unavailable, surface that uncertainty. Compare equal-duration
windows at equal times after onset, while reporting duration separately and
avoiding treating missing/truncated audio as measured silence.

The first C take's export volume changed during setup; comparisons across notes
are exploratory and do not isolate pitch from picking strength or processing.

## Current recording and behavior

Current recording setup, confirmed by the user on 2026-10-24:

- Gain knob: 9 o'clock.
- Guitar pickup: neck pickup.

These settings describe the current recording session; they have not been
confirmed for the earlier `Test_1_Cmaj.wav` recording.

Latest replacement checked on 2026-10-24 at 19:44: `data/raw/C_note_dry_take01.wav`.

- WAV, 16-bit PCM, 44,100 Hz; shape `(458620, 2)`, about 10.400 seconds.
- Left and right channels are identical; compatible with the current left-channel analysis.
- Full-recording RMS is approximately 0.00459138; absolute peak is 0.05480957
  (about -25.22 dBFS), roughly 6 dB above the preceding export peak of -31.25 dBFS.
  No samples reach absolute amplitude 0.999. This does not rule out upstream distortion.
- About 1.5 seconds of quiet lead-in precedes the note. The 50 ms RMS curve
  shows a clear attack and decay, approaching the background level around 7 seconds.
- Initial 0.5-second RMS is approximately 0.00016060. The strongest 50 ms RMS
  window starts at 1.55 seconds.
- This replacement is usable for RMS/decay and onset exploration, although the
  exported signal level is low. A fixed -1 to 1 plot makes it look nearly flat.
- The prior export under this same filename was 13.372 seconds, had no quiet
  lead-in, and peaked at approximately -0.10 dBFS. Duration and timing changed
  as well as level; do not attribute all differences solely to normalization.
- The user's latest GarageBand screenshot shows Auto Normalize, Master Echo,
  and Master Reverb unchecked. The filename says "dry," but bypass of all
  track/master processing has not been independently confirmed.
- The latest export was checked numerically; earlier exports were also plotted.
  No listening test was performed. `main.py` still
  loads `Test_1_Cmaj.wav`; no analysis code was changed.

Additional recordings checked numerically and visually on 2026-10-24:

- `C_note_dry_take02.wav`: 13.372 seconds, peak about -28.64 dBFS,
  approximately 2.0 seconds of quiet lead-in, followed by a clear attack and decay.
- `Low_E_note_dry_take01.wav`: 14.858 seconds, peak about -27.37 dBFS,
  approximately 1.55 seconds of quiet lead-in and a sustained, fluctuating decay.
  Signal drops abruptly near 13.4 seconds; do not treat that cutoff as natural decay.
- Both are 44,100 Hz, 16-bit PCM stereo WAVs with identical left/right channels,
  finite samples, and no samples at absolute amplitude 0.999 or above. Both are
  usable for the current amplitude-analysis work. Note labels are user-provided,
  not verified by pitch estimation. No listening test was performed.
- `Low_E_note_dry_take02.wav` was subsequently checked numerically: 16.344
  seconds, 44,100 Hz, 16-bit PCM stereo, peak about -30.09 dBFS, approximately
  2.45 seconds of quiet lead-in, a decaying RMS profile, and no samples at
  absolute amplitude 0.999 or above. Usable for current analysis.
- Both open A takes were checked numerically: `A_note_dry_take01.wav` and
  `A_note_dry_take02.wav`. Each is 14.858 seconds, 44,100 Hz, 16-bit PCM stereo,
  with approximately 2.0 seconds of quiet lead-in and a decaying RMS profile.
  Peaks are about -28.10 and -29.36 dBFS respectively; all samples are finite
  and none reach absolute amplitude 0.999. Both are usable for current analysis.
- Next recordings: open D string, `D_note_dry_take01.wav` and
  `D_note_dry_take02.wav`, with unchanged settings. Keep all previous takes.
- When comparing takes, align measurement windows relative to each note's onset;
  lead-in lengths and full file durations differ.

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

The earlier `main.py` demonstration assumes stereo input and uses a fixed relative path. It plots only
the left channel and does not save the figures. It now slices 3.00–3.05
seconds (indices 132300:134505), plots those 2,205 samples in a second figure,
and reports window RMS approximately 0.10735958. The earlier 1.00–1.05 second
window (indices 44100:46305) gave RMS approximately 0.39458260.
Both plots use amplitude limits
of -1 to 1, and the overview highlights the selected interval. Window dimensions
and both figures were checked; the close-up was visually inspected.
No FFT, pitch tracking, or ML is implemented. The batch comparison tests are
described above; these earlier single-file observations remain for reference.

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
- Time-window slicing is implemented and reviewed. The user changed the window
  from 1.00–1.05 to 3.00–3.05 seconds and shared the plot and numerical results.
  The later window has smaller amplitudes, a smoother shape, and roughly similar
  cycle spacing. This is a visual observation, not verified pitch tracking.
- Equal-length windows have the same RMS denominator; the later window's lower
  RMS reflects a smaller sum of squared amplitudes. Whole-recording RMS remains
  unchanged when the selected window changes.

## Git checkpoint and next steps

The analysis milestone is committed locally as `b98e89a`, with message
"Program to read wav file and preform amplitude analysis". It includes the code,
README, requirements, example WAV, and ignore rules. Confirm push status rather
than assuming it. This handoff was added after that milestone.

The close-up milestone is already committed as `87e96f4` ("added time window
selection fro RMS and amplitude analysis"). The onset-aligned comparison code,
tests, and documentation changes are not committed. The new WAV files are also
untracked; review status/staging with the user before any commit. No push was done.

Next learning step: walk through one recording and its sample/time conversion,
then review quiet-region and start/end boundaries as described above. Do not add
a more complex detector before explaining the measurement choices to the user.
Continue collecting open D/G/B/high-E pairs under unchanged settings if desired.

Later: repeat recordings under consistent conditions,
zero-crossing rate, FFT, STFT/spectrogram, spectral centroid, then pitch tracking.
Introduce modules and additional structure only when useful. For future ML,
favor interpretable features and classical baselines before neural networks.
