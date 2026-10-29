# Guitar ML — project handoff

Last updated: 2026-10-28. Read this alongside README.md and the actual code.
Verify the current files and Git status before making changes; this document is
a checkpoint, not a substitute for inspecting the project.

## Resume here — end-of-session handoff (2026-10-28)

User explicitly asked to stop development here, remember the findings/learning,
and commit/push the current work. No acceptance/rejection rule was implemented.
Do not start further analysis until the user resumes it.

Current verdict: lesson 04's 100 ms / 20 ms sliding baseline follows the three
ringing pitches consistently. Median estimates over 0.5–3.5, 4.5–7.5, and
8.5–11.5 seconds are ~82.76, 87.91, 93.02 Hz versus nominal MIDI references
82.41, 87.31, 92.50 Hz. Low-level gaps/tails can still yield spurious pitches;
the final frame is ~888.89 Hz. This is a promising baseline, not a validated
accuracy claim. Actual tuning and estimator error have not been separated.

Learning checkpoint: user understands harmonics and that similarity is
autocorrelation, not label accuracy. Clarify again if needed: autocorrelation
compares samples WITHIN ONE analysis window against time-shifted copies of that
same window. It does not compare a window against the previous sliding window
or against MIDI. A lower score means weaker repeating-pattern agreement under
this calculation; gain reduction alone does not necessarily reduce normalized
similarity. Noise, changing vibration, and the shorter-window overlap penalty
can affect it. The 100 ms scores cannot directly inherit half-second thresholds.

Agreed next step on resumption:
1. Add a small, explicit trial acceptance/rejection rule using level and
   autocorrelation evidence, retaining raw candidates and showing rejected ones.
2. Test unchanged settings on P2 single notes to check transfer before tuning.
3. Later group accepted pitch windows into note events; repeated same-pitch
   picking will need attack detection. No extended MIDI alignment project first.

Keep explanations focused on key math, code, architecture, and concrete concerns.
Preserve the user's preference against overengineering and black-box additions.
All 15 tests passed at wrap-up. Lesson 02 now contains the user's saved 5–5.5 s
experiment; preserve that edit (the older script still defaults to 3–3.5 s).
Read the Git checkpoint below before assuming changes reached GitHub.

## Purpose

The owner is a Statistics / Data Science student learning Python, software
engineering, signal processing, statistics, and eventually machine learning
through personally recorded guitar audio. Understanding the work matters more
than quickly producing a large application.

The user clarified the primary end goal on 2026-10-27: automatically transcribe
guitar recordings into tablature, ultimately including faster picking, overlapping
notes, and chords. Pitch, attack timing, and plausible string/fret assignments
must guide the roadmap. Bends, vibrato, slides, performance comparison, and a
possible explanation layer are secondary interests. Full note-event transcription,
multi-pitch detection, and tab generation are not implemented. Preserve the tutoring pace; explain one small
step before implementing it, and avoid prematurely adding models or architecture.

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
- User reiterated on 2026-10-28: avoid overengineering. Prioritize understanding
  the data and building a useful note-analysis program. Do not turn the small
  MIDI timing mismatch into a prolonged alignment project before progressing.
- With each change, explain the key math/code and system architecture: inputs,
  outputs, module responsibilities, and why the change is useful. Raise concrete
  concerns revealed by the data or design; keep the next learning step small.

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
- `requirements.txt`: NumPy, SoundFile, Matplotlib (currently unpinned), Mido >=1.3,<2.
- `.gitignore`: excludes `.venv/`, `__pycache__/`, `.DS_Store`, and `*.band/`.
- `README.md`: current setup, behavior, and limitations.

## Latest working checkpoint — sliding-window analysis (2026-10-28)

This checkpoint supersedes earlier suggestions to focus next on MIDI timing
correction. User wants practical data understanding and a continuous analysis
prototype, with learning explanations and concrete concerns at each change.

Implemented `04_tracking_pitch.ipynb` and matching `pitch_tracking_lesson.py`:
1. Load P1 direct-input audio at 0–11.9 s (48 kHz mono), original samples 0:571200.
2. Define 100 ms windows (4,800 samples), 20 ms hops (960 samples), 60–1,000 Hz bounds.
3. Explicitly loop over all 591 complete windows, computing candidate pitch,
   original-window RMS, lag, and similarity. First midpoint 0.05 s, last 11.85 s.
4. Load MIDI labels ONLY AFTER all audio measurements; use as unchanged plot overlay.
5. Plot all candidates, RMS in dBFS, and similarity on a common time axis.
6. Save CSV/PNG in ignored `outputs/pitch_tracking/`, prefixed by dataset folder name.

Architecture: `pitch_analysis.py` is the shared single-window calculation with
no MIDI/files/plots. Input is one-dimensional samples, rate, and frequency bounds;
output is candidate Hz, lag in samples, and normalized similarity. Notebook/script
owns loading, sample-index window traversal, diagnostics, plotting, and saving.
`guitartechs_midi.py` is the separate annotation reader, used only for comparison.
Lesson 03 now imports this same pitch function rather than duplicating it; its
three default results are unchanged. Lesson 02 retains its explicit teaching
calculation and the user's edits; it was not overwritten.

The estimator returns NaN for a constant window or no positive local peak;
invalid/too-short input raises ValueError. No RMS/similarity filtering, smoothing,
MIDI-based selection/correction, label time shifts, pitch-name assignment,
note grouping, onset detector, chord decoder, training, or tabs were added.

Observed results: 590 of 591 windows have mathematical candidates (not 590
reliable notes). Candidate pitch follows E2/F2/F#2 in ringing portions. Medians
at 0.5–1.0, 4.5–5.0, 8.5–9.0 s are ~82.62/87.91/93.02 Hz. Low-level gaps/tails
produce large excursions: overall finite range 60–888.89 Hz; final frame is
888.89 Hz at RMS ~0.000115 and similarity ~0.308. All excursions stay visible.

Learning points / concerns:
- Overlapping windows are observations, not separate note events or independent
  test examples. A 20 ms hop does not imply 20 ms onset accuracy; timestamps are
  midpoints of 100 ms spans. Original start/exclusive-stop samples are in the CSV.
- Similarity falls from ~0.97 in half-second windows to ~0.88 in 100 ms ringing
  windows partly because the score sums N-k overlapping pairs but divides by
  zero-lag energy over N samples. Do not reuse thresholds across window sizes.
- RMS plot uses 20*log10(RMS), flooring only the display at 1e-12 (-240 dBFS).
  Original RMS is preserved in CSV; no audio normalization or noise gate is applied.
- Label gaps can contain ringing; the MIDI is neither perfectly timed nor perfect
  ground truth. Keep the mismatch visible rather than tuning to exact timestamps.
- 60–1,000 Hz does not cover all high notes in the full dataset. This is a
  single-note development example, not evaluated chord/full-dataset performance.

Validation: known tones with stronger second harmonics at 44.1/48 kHz passed
within 1% using 100 ms windows; gain/DC invariance, constant-input NaNs, and
invalid input checks passed (`test_pitch_analysis.py`, four test methods).
Executed all six lesson cells, validated notebook structure, inspected the plot,
and verified 591 windows, 4,800-sample widths, 960-sample hops, midpoints, and RMS.
Re-executed lesson 03 to verify unchanged results after sharing the estimator.
Saved outputs were generated by executing cells with project Python, not via a
live Jupyter kernel. No packages added in this step; Mido was already installed.

Next learning step: user reads lesson 04's explicit sliding loop and three-panel
plot. Discuss the failure regions, then try a small explainable no-reliable-pitch
rule and check another excerpt/performer before tuning. Keep progress aimed at
note events and tabs; do not start an extended timing audit or large ML system.

## Current direction — audio to tablature (2026-10-27)

### Guitar-TECHS downloads inspected — 2026-10-28

User downloaded P1/P2 chords, scales, singlenotes, techniques and P3_music into
`/Users/wesley/Downloads`. All nine distinct ZIPs match the official Zenodo MD5
checksums: https://zenodo.org/records/14963133. `P3_music (1).zip` is an identical
duplicate; nothing was deleted, moved, or extracted during inspection.
P1_singlenotes and P2_singlenotes were already extracted in Downloads.

Labels ARE present under each archive's `midi/` directory as `.mid` files.
Each has corresponding `audio/directinput/directinput_<name>.wav` and
`audio/micamp/micamp_<name>.wav`; ego/exo audio is MP3 under `video/`.
Excluding macOS metadata, each P1/P2 group contains 28 chord MIDI files,
12 scales, 1 single-note sequence, and 5 techniques. P3 has 12 music MIDI files:
104 label files total, each paired with both WAV capture types. Ignore __MACOSX
and ._ resource-fork entries when constructing manifests.

Single-note example pair:
`/Users/wesley/Downloads/P1_singlenotes/audio/directinput/directinput_allsinglenotes.wav`
and `/Users/wesley/Downloads/P1_singlenotes/midi/midi_allsinglenotes.mid`.
Both players' WAVs are 552 seconds at 48 kHz: long note sequences, not isolated
WAVs per note. P1 is mono, P2 stereo; DI is PCM24 and amp-mic PCM16.
Both MIDIs are format 1, 960 ticks/quarter, with tempo 1,000,000 microseconds per
quarter (60 BPM), a metadata track and six tracks named e/B/G/D/A/E.
Note-on/off events and string-track identities were inspected directly; no MIDI
package installed. P1 uses channel 0 across all string tracks, so channel alone
must not be used as a string label. P2 uses separate channels as well as tracks.
P1 contains 142 positive-velocity note-ons; P2 contains 137. Some P1 low-velocity
events occur outside the expected ascending string sequence, and P2 D has 22
events. These deserve review, not automatic deletion or assumptions of perfect
ground truth. Pitch range includes MIDI 86 (~1175 Hz), exceeding the lesson's
current 1,000 Hz upper bound.

Official dataset notes warn of up to 100 ms signal misalignment. Verify audio
and MIDI timing on a short excerpt before scoring attacks or training. The
official site https://guitar-techs.github.io/ states CC BY 4.0 and requests citation.
Next small step: read a short P1 direct-input/MIDI pair, display note labels as
a table and overlay timings on audio before building a training dataset. Keep
all captures of one performance together in future train/validation/test splits.
No full training, automated note-label alignment, or audio quality audit performed.

### First Guitar-TECHS label overlay — 2026-10-28

User authorized taking a short single-note excerpt and comparing labels against
waveform plots. Added `03_guitartechs_labels.ipynb` and its matching
`guitartechs_labels_lesson.py`. Five code cells read P1 direct input 0–11.9 s
(48 kHz mono, samples 0:571200), load MIDI labels, plot unchanged label spans
and a zoom around F2's start, estimate pitch in one half-second segment per label,
and plot nominal label frequencies separately from predicted frequencies.

User installed Mido successfully in `.venv` after tool-environment installation
failed. Added `mido>=1.3,<2` to requirements. `guitartechs_midi.py` uses Mido to
preserve track identity, order events, convert ticks with the tempo map, and
pair note-on/off messages (including velocity-zero note-ons). It reports invalid
pairings instead of silently dropping them. No custom binary MIDI decoder remains.

First supplied labels (all low-E string, standard-tuning frets 0/1/2):
- E2 (MIDI 40): 0–3.7354167 s; target 82.4069 Hz.
- F2 (MIDI 41): 4.0208333–7.8 s; target 87.3071 Hz.
- F#2 (MIDI 42): 8.0260417–11.790625 s; target 92.4986 Hz.
Sampling starts 0.5 s after each supplied onset, for 0.5 s. Original sample
windows: [24000:48000], [217000:241000], [409250:433250]. Same 60–1,000 Hz
autocorrelation rule returns lags 581/547/516, frequencies 82.6162/87.7514/93.0233
Hz, similarities ~0.971/0.970/0.974. Expected label pitch does not enter estimator.
These nominal-versus-measured differences are not established tuning errors;
integer lag, actual tuning, and changing pitch can contribute. No threshold tuned.

The F2 waveform attack visibly precedes the MIDI marker by a few tens of ms.
No automatic alignment correction was made. MIDI note-off is not guaranteed to
mark the final audible decay. First note has no preceding context (start at zero).
This is label-guided window selection, not automatic event detection, a
sliding-window implementation, independent validation, or model training.

All five notebook cells executed with the project interpreter; outputs saved,
notebook schema validated, both plots visually inspected (not a live kernel test).
Four MIDI tests in `test_guitartechs_midi.py` pass: tempo changes/zero-velocity note-offs,
same channel/pitch on different tracks, invalid pairing errors, and asynchronous
file rejection. Six known-tone/stronger-harmonic pitch checks at 44.1/48 kHz pass;
excerpt labels and original sample bounds also verified. Original WAV/MIDI and
the user's modified frequency notebook were left untouched.

Next teaching checkpoint: read notebook 03's first three cells and inspect
what the shaded labels mean; then review the prediction table/plot in cells 4–5.
Keep timing mismatch visible before proposing any alignment rule or training.

### Transcription roadmap

This section supersedes the earlier suggestions to keep tuning sustained-note
end thresholds. The user explicitly asked whether that work is overengineering
relative to automatic tabs for faster picking and chords. It is time to move on.
The RMS lessons remain useful foundations, not the final transcription algorithm.

- User's latest listening observation with headphones: low E take02 seems to
  fade into background around 14 seconds. Earlier estimates were about 11 and
  15 seconds; treat these as approximate subjective observations, not exact
  endpoint labels. Measured RMS approaches pre-note background near 15 seconds.
  RMS alone cannot establish how much remaining energy is guitar versus noise.
- Do not keep adjusting thresholds to resolve a 14-versus-15-second endpoint.
- A rule that waits for quiet between notes will miss new attacks during ringing.
  The current 100 ms confirmation rule is only an isolated-note learning baseline,
  not established as suitable for fast picking or polyphonic playing.
- The spectrum and first autocorrelation candidate-fundamental lesson are now
  implemented (see below). Review the similarity loop and peaks before advancing.
- First transcription milestone: recover the ordered pitches and approximate
  attack times of a short, known single-note phrase, then map them to tentative
  string/fret positions under stated tuning and position constraints.
- Same pitch can occur on multiple strings: pitch detection alone does not
  uniquely identify the player's original fingering. Distinguish plausible tabs
  from recovered actual string/fret assignments.
- Faster passages and overlapping notes/chords require later onset and multipitch
  work; multiple spectral peaks can also be harmonics of a single note. Compare
  against known recorded phrases, and keep fresh takes for validation.
- Later, consider evaluating an existing polyphonic transcription model as a
  baseline (e.g. Spotify Basic Pitch produces note events/MIDI, not guitar tabs).
  No model was installed, selected for production, or used on the recordings.

## First frequency spectrum — 2026-10-27

User authorized starting frequency detection and committing/pushing to GitHub.
Added `02_understanding_frequency.ipynb` and matching `frequency_lesson.py`,
using existing dependencies. Four visible code cells select low E take02's
3.0–3.5 s interval (samples 132300:154350), plot its location, subtract its mean,
apply a Hann window, compute NumPy rFFT magnitudes and frequency labels, then
plot a spectrum. Sample rate is read from the file, not hardcoded. Selection is
manual and editable, not a new silence assumption or automatic onset rule.

22,050 samples at 44,100 Hz give a 0.5 s segment and 2 Hz FFT bin spacing;
spacing is not a guarantee of pitch accuracy. Largest nonzero bin is 164 Hz
for low E take02; the lower peak near 82 Hz and higher harmonics illustrate an
octave-error risk. The same interval in low E take01 peaks at 82 Hz, A takes
at 110 Hz, and C takes at 262 Hz. These are dominant bins, not automatically
identified fundamentals or verified note labels. Filename labels do not enter
the peak selection. The graph shows relative magnitudes and cannot compare
recording volume. Full nonzero spectrum is searched; plot zoom is 0–1,000 Hz.

Executed the four cells with the project interpreter and saved example outputs;
validated notebook structure and visually inspected the spectrum. This was not
a live Jupyter kernel test. Exercised the actual FFT cell with known 110 Hz tones
plus DC and with a stronger 220 Hz harmonic, at both 44.1 and 48 kHz; expected
dominant bins passed. Exact silence raises an explanatory error. Noise can still
produce a largest peak: no note-presence detector is claimed. Seven existing
comparison tests passed. Audio and batch detection code were not changed.

The user subsequently understood the difference between frequency positions and
harmonic strengths, and the shifted-waveform explanation of autocorrelation.
They explicitly requested coding it; the following milestone advances this lesson.
Avoid using a 0.5 s frame as a universal fast-note setting. No multipitch model,
note-name mapping, or tablature generator added.

### Autocorrelation addition — 2026-10-27

Added sections 5–6 (two code cells) to the same notebook and matching script.
Use the same mean-centered 3.0–3.5 s segment, without the FFT Hann taper. An
explicit loop multiplies overlapping sample pairs at each lag and sums them,
dividing by the zero-lag squared sum. No FFT shortcut or new dependencies.
Longer shifts contain fewer pairs, causing a mild shorter-lag preference;
this is documented, not presented as overlap-corrected correlation or confidence.

Search range is fixed at 60–1,000 Hz, converted with ceil(sr / max_frequency)
and floor(sr / min_frequency), i.e. 45–735 samples at 44.1 kHz. Zero lag is not
a candidate. Choose the highest positive local maximum within the bounds;
compute one extra lag to check the upper boundary's neighbor. Report competing
peaks, best lag, period, candidate frequency, score, and prior FFT strongest bin.
Plot similarity versus lag plus an amplitude-preserving overlay of three periods.
No filename-based expected pitch, per-take parameter tuning, interpolation,
note-presence threshold, multiple-note tracking, or chord decoder is used.

Unchanged-settings development results on all six current takes:
- Low E take01 and take02: lag 538, 81.97 Hz (scores 0.975 / 0.979).
- A take01: lag 399, 110.53 Hz; A take02: lag 400, 110.25 Hz (scores ~0.981).
- C take01 and take02: lag 168, 262.50 Hz (scores 0.992 / 0.989).
For E take02 the competing lag 270 (163.33 Hz) scores only 0.114, illustrating
why its stronger second harmonic does not imply a fundamental of 164 Hz.
The scores are similarity measures, not accuracy probabilities. Integer-lag
rounding affects pitch precision; multiple-period/harmonic peaks remain a risk.

Executed all six notebook code cells and saved outputs, validated notebook
structure, and visually inspected the new plots. Not a live Jupyter kernel test.
Exercised the actual autocorrelation cell on 12 synthetic cases: 82.41, 110,
261.63 Hz at 44.1/48 kHz, with and without a second harmonic twice the fundamental
amplitude. All estimates were within 0.5% of the known frequency. Gain scaling
plus a DC offset preserved each chosen lag. Silence, constant, and too-short
inputs raised errors. Previous FFT synthetic checks also passed. These checks
do not establish real-world performance, especially for noise, chords, or attacks.

Next tutoring checkpoint: user reads section 5's slice/product loop and section
6's plots, understands score normalization and candidate selection. Then examine
stability across ringing intervals and move toward a known melody with shorter
sliding windows. Keep new takes for evaluation instead of tuning to these six.

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

### Low E take02 transfer check — 2026-10-26

The user requested the same rule on low E take02, expecting the note around
2.5 seconds and definitely after 2 seconds. Section 8 now has a short input
cell selecting `Low_E_note_dry_take02.wav` and candidate quiet region 0–1.5 s
(samples 0:66150), before that expected attack. Variables in this section are
now generic `test_*` names. Section 7's C-threshold/A-background check is preserved.
The matching lesson script was updated. No batch detector changes.

Settings stayed fixed: 50 ms windows, 10× median background RMS, two consecutive
qualifying windows. The search still covers the whole file; the user's expected
start was not used as a search constraint. Thirty quiet windows yielded median
RMS 0.00016757 and threshold 0.00167566. First qualifying window boundary:
sample 108045, 2.450 s; confirmation boundary: sample 112455, 2.550 s. Ratios
in those two windows were about 51.00 and 68.86. Highest selected-quiet ratio
was 2.70. The result is consistent with the user's rough expectation, but the
window boundary is not an exact physical attack time. This previously viewed
recording is a transfer check, not a wholly untouched validation example.

Executed notebook code cells, saved new section 8 results/plot, verified the
unchanged settings and earliest qualifying pair, and inspected the plot.
Next: review these two windows with the user; then test a new recording under
the same rule before choosing changes or adding note-end detection.

### Low E take02 tail inspection — 2026-10-26

The user thinks the note faded naturally (not confirmed). Numerically inspected
50 ms windows around 15 seconds against the same 0–1.5 second background median
(0.00016757). RMS is about 5.75× background at 14.00–14.05 s, 1.22× at
14.90–14.95 s, and 1.02× at 15.00–15.05 s. From 15.05 to 15.60 s it remains
approximately 0.96–0.99× background and contains nonzero recorded samples.
Last nonzero sample is 688341 (15.608639 s); the following samples are exact
zeros, through the file end at 16.343855 s. Do not count that digital-zero tail
as evidence of physical note decay or treat it as the noise floor.

This supports the user's rough ~15-second return-to-background observation,
without verifying its physical cause or defining an exact endpoint. No new
end detector or threshold has been implemented. Use this as a reviewed example
when discussing a sustained return-to-background rule; avoid tuning the rule
to hit exactly 15 seconds. The start threshold of 10× is unsuitable for the end.

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
At that earlier milestone no FFT, pitch tracking, or ML was implemented. The batch comparison tests are
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

At the 2026-10-28 wrap-up, main and cached origin/main started at `199664a`
("Autocorrelation notebook for naalyzing pitch frequency across notes"). The user
authorized committing and pushing all current project changes, including their
saved lesson 02 experiment, lessons 03/04, shared modules, tests, and handoff.
Look for the commit "Add Guitar-TECHS labels and sliding-window pitch analysis"
in the actual log; do not rely on a self-referential hash in this document.

The wrap-up fetch again failed with `Could not resolve host: github.com`.
The tool environment cannot currently reach GitHub. A local commit does not
establish upload success; check the push result and fresh remote state. If needed,
the user can run `git push origin main` from their network-enabled terminal.
Dataset ZIPs/audio stay in Downloads and generated tracking outputs are ignored.

Resume with the small rejection-rule/P2 transfer experiment described at the top,
only when the user returns. Do not proceed with that work during this wrap-up.
