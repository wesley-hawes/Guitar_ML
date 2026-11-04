# Archived development notes

This is the README preserved before the repository presentation audit. It records
historical experiments and superseded next steps; use the [current README](../../README.md)
and [experiment index](../experiments.md) for current capabilities and commands.

---

# Guitar ML

A Python learning project with the goal of automatically transcribing guitar
recordings into tablature, eventually including faster picking and chords.
Current work covers audio arrays, interpretable amplitude measurements, and a
guided frequency-spectrum and autocorrelation lesson;
pitch transcription and string/fret assignment are not implemented yet.

## Current milestone

The newest lesson is `09_detecting_attacks.ipynb` (or
`python attack_detection_lesson.py`). It proposes attacks independently of pitch:
relative RMS rise >=1.5, sufficient level/absolute rise versus the provisional
noise reference, then strongest-first 80 ms duplicate suppression. These are
initial heuristics with known misses/extras, not a finished transcription system.
20 candidates align visually with the confirmed 20 C-to-A plucks; the earlier
P2 repeated A4 now has two candidates even though pitch grouping merged it.
Soft transitions can fail: a total of 12 candidates can hide a miss and an extra.
The notebook exposes decisions, suppressed episodes, and the fastest soft-note
example. This is offline analysis; peak markers are not exact onsets. Candidates
do not yet split/assemble note events. All 32 tests pass; no new dependencies.

Run notebook 08 first to generate the measurements used in 09. Fastest now
produces the intended 12 pitches plus a short unwanted B2 group after the phrase.
Its first-to-last intended group-start span is ~52% shorter than baseline.
Next: review a few actual onset intervals and decide how independent pitch and
attack evidence should combine; don't adjust thresholds to force note counts.

The preceding lesson is `08_personal_recordings.ipynb` (or
`python personal_recordings_lesson.py`). It runs the unchanged pitch/filter/grouping
and separate energy-rise measurements on five personal WAVs. Baseline and Faster
produce the intended 12 pitch classes in order; Dynamics also has a quiet E2
estimate before the phrase. CtoA produces 20 groups (5 Cs, 5 As, repeated), split
by rejected pitch windows rather than attack detection. This is an exploratory
comparison, not a timing/accuracy benchmark. The user confirmed fifth-position
A3→A2 playing and the repeated C/A counts. The shared quiet reference (Dynamics 1.2–1.4 s) and
consistent gain remain provisional. No rules were tuned to these takes.

The notebook includes full-file plots, a repeated-C close-up, explicit caveats,
and architecture questions. Recording setup is in
`data/recording_sessions/2026-11-03_personal_takes.json`; generated CSVs/plots/audit
are under ignored `outputs/pitch_tracking/personal_2026-11-03/`.
Next: review real plucks and design an explicit attack-candidate rule; reserve
a new recording for evaluation. Current evidence groups are not final tablature.

The preceding lesson is `07_understanding_attacks.ipynb` (or `python attack_lesson.py`).
It compares current 10 ms RMS to the median of four preceding 10 ms blocks,
every 2 ms, using the pure helper in `attack_analysis.py`. Real P1/P2 repeats
show strong rises compared with ordinary ringing. A clearly labeled synthetic
example adds another excitation while the first tone still rings, demonstrating
that the cue does not require silence. Ratios can inflate near zero and real
re-picking need not raise RMS clearly; no attack threshold or event split yet.
Five focused checks run with `python -m unittest test_attack_analysis -v`;
all 26 project tests pass. No new dependencies or pitch/grouping changes.

The preceding check is `06_tracking_a_scale.ipynb` (or
`python scale_tracking_lesson.py`): first 8 seconds of the selected player's
C-scale direct-input recording, with unchanged analysis/filter settings.
Set `player = 'P1'` or `'P2'`. The current example is P2 with a 9.3-second
excerpt to include repeated A4; the following counts describe the initial 8 s.
P1 has 316/396 passing windows,
14 evidence groups, and all 109 guarded interior windows within 50 cents.
P2 has 355/396 passing windows and all 119 interiors within 50 cents.
Windows spanning changes
can be unstable; see the D3→E3 close-up and per-window decisions. This does not
establish accurate attack timing or separation of simultaneous ringing notes.
The level reference is explicitly carried from each player's single-note
recording, not measured as the scale's own noise floor. P1 starts with digital
silence and A2; P2 includes two earlier notes, explaining their group-count difference.
Keep the selected WAV/MIDI under `~/Downloads/P1_scales` or `P2_scales` (see notebook).
No new dependencies; raw/accepted candidates and all MIDI labels remain visible.

Sections 5–6 now group accepted windows into note-evidence runs using
`note_events.group_pitch_windows`. Consecutive windows with the same nearest
note are joined; rejected/missing windows split runs and gaps stay visible.
P2 produces 16 runs from all 355 accepted windows. Bars span supporting
window midpoints, not physical note start/end times. Median Hz and exact source
sample bounds are exported separately in `P2_scale_C_0-8s_note_evidence.csv`.
Repeated same-pitch plucks may merge without attack detection; no MIDI labels
enter grouping. Run `python -m unittest test_note_events -v` for its six tests.

Next attack example: set `player = 'P2'`, `excerpt_seconds = 9.3`, then run all
cells and inspect ~7.8–9.1 s. Two A4 MIDI labels become one continuous evidence
group (7.97–8.91 s); P1's corresponding pair splits because of rejected windows.
Player/duration-tagged outputs preserve both runs. No attack detector added yet.

Section 7 plots that repeated-A4 waveform, 10 ms RMS with 2 ms hops alongside
the existing 100 ms RMS, and a close-up around the second label. The finer
energy view exposes a dip then rise around 8.46 s even though pitch stays A4.
This is inspection only: no threshold or attack rule is applied. The saved plot
is `outputs/pitch_tracking/P2_scale_C_repeated_A4_waveform.png`.

The preceding check is `05_checking_more_audio.ipynb` (or
`python broader_pitch_check.py`). It compares 0–30 seconds of P1/P2 with the
existing settings fixed and adds mean/centered-RMS diagnostics without changing
acceptance. P2 has 67 additional windows dominated by their window mean; the
existing rule rejects all of them. The only passing example with that diagnostic
flag remains 3.81 s. P1 has none. Guarded note-interior comparisons show agreement
within 50 cents for all passing estimates, but P1 rejects 78 interior windows.
These are overlapping development examples, not full-recording accuracy.
Separate `0-30s` output files preserve lesson 04's results. No new dependencies.

Start `04_tracking_pitch.ipynb` for the underlying tracker and filter lesson (or run
`python pitch_tracking_lesson.py`). Select `player = "P1"` or `"P2"` in the first
cell; the current example is P2. It analyzes an 11.9-second direct-input excerpt
in 100 ms windows with 20 ms hops: 591 complete windows. The visible loop uses
audio only, measures pitch candidates/RMS/similarity, then adds MIDI references
for the plot. Every finite candidate remains visible. Sections 9–11 add a
separate trial acceptance rule; no smoothing or alignment correction is applied. Constant/no-positive-peak
windows return NaN rather than a made-up pitch.

`pitch_analysis.py` now holds the single-window autocorrelation function shared
by lessons 03 and 04. It takes samples, sample rate, and frequency bounds, and
returns candidate Hz, lag, and similarity; it has no file, plotting, or MIDI
dependencies. The notebook/script controls window selection and output.
Run its checks with `python -m unittest test_pitch_analysis -v`.

The plot and CSV are regenerated under `outputs/pitch_tracking/` (ignored by Git).
The CSV retains original sample bounds and window midpoints. Midpoints do not
establish note attack times. This example follows the three ringing pitches but
produces large pitch jumps during low-level gaps/tails. Similarity depends on
window duration and is not a confidence probability. Inspect the failure regions
alongside the trial no-reliable-pitch rule; full event detection, ML, and tabs
are later steps. Saved notebook outputs were executed with project Python and
validated, not run through a live Jupyter kernel.

The unchanged-settings P2 transfer check follows the three ringing pitches,
with medians ~81.91/86.96/91.95 Hz in the documented inspection intervals. It also
exposes high-similarity mismatches: at 7.71 s, 607.59 Hz scores ~0.901. Some P1
ringing windows have both lower RMS and lower similarity than other P2 gap
mismatches, so simple global lower thresholds cannot perfectly separate them.
The notebook records this baseline finding before applying the trial rule.
Both players' outputs are retained separately. P2's channels are identical in
the tested excerpt; the existing left-channel policy is used.

Sections 7–8 now compare window RMS with the median RMS of an editable,
candidate background interval. Only complete windows inside its sample bounds
are included. P2 ringing levels are roughly 21–52 times this reference in the
selected intervals, but a second quiet-looking tail has median ratio 1.86.
These intervals are not verified silence, and overlapping windows are not
independent observations. Inspect the waveform zoom and level-ratio plot before
interpreting the trial thresholds. The underlying pitch estimator is unchanged.

Sections 9–11 try RMS >= 3x reference, similarity >= 0.80, and two consecutive
eligible windows within 50 cents (half a semitone). The notebook explains each
check, plots raw/passing candidates, and saves *_pitch_decisions.csv with raw
values, accepted Hz, flags, reasons and settings. Rejected means no reliable
pitch under this rule, not proven silence. New notes can qualify on their second
eligible window; quiet tails, attacks, bends and fast notes may be rejected.
P1 has 546/591 passing windows; P2 has 528/591. These are not accuracy scores.
A known remaining mismatch is P2's 317.88 Hz at 3.81 s, which passes all checks.
Stable wrong pitches can pass, and overlapping windows are not independent.
Change the section 10 zoom to 3.5–4.2 s to inspect that failure. No thresholds
were retuned to remove it; review before expanding to longer recordings.

Section 12 compares P2's suspicious 3.81 s window with clear ringing at 1.01 s:
original waveforms, autocorrelation curves and selected-shift overlays. The
suspicious estimate selects a tiny local bump on a broadly declining curve,
while the ringing example has a distinct peak. This diagnostic changes no
settings or decisions; it also documents the difference between raw and
mean-subtracted RMS. Next inspect more audio with settings fixed, rather than
tuning to this example. The PNG is saved with suffix _candidate_diagnostic.png.

The preceding dataset lesson is `03_guitartechs_labels.ipynb`, with matching
`guitartechs_labels_lesson.py`. It reads the first 11.9 seconds of P1's Guitar-TECHS
direct-input single-note sequence from `~/Downloads/P1_singlenotes`, displays its
MIDI labels (E2/F2/F#2), overlays their unchanged timing on the waveform, and
compares one half-second ringing window per label with autocorrelation estimates.
This is label-guided inspection, not sliding-window transcription or training.
Keep the extracted dataset in Downloads or edit `dataset_folder` in cell 1.
No third-party audio is copied into the repository.

MIDI parsing requires `mido` (now in `requirements.txt`):
`python -m pip install mido`. The helper `guitartechs_midi.py` converts ticks to
seconds using tempo events and preserves string-track identity when channels
are shared. Focused checks run with `python -m unittest test_guitartechs_midi -v`.
The user installed Mido in the project environment. All five lesson cells were
executed there, example outputs saved, and notebook structure/plots checked;
this was not execution through a live Jupyter kernel.

Data: Pedroza et al., *Guitar-TECHS*, ICASSP 2025,
[official release](https://zenodo.org/records/14963133),
[project and CC BY 4.0 license](https://guitar-techs.github.io/).
The publisher reports possible timing misalignment up to 100 ms. The lesson
shows a small label-versus-attack discrepancy without applying a correction.

Open `02_understanding_frequency.ipynb` for the preceding step. The first four code cells
select an editable 3.0–3.5 second segment of low E take02, show its original
sample indices, apply a Hann window, and plot its FFT magnitude spectrum.
The strongest bin is 164 Hz; a lower peak near 82 Hz illustrates why a dominant
frequency is not necessarily the fundamental pitch. The same segment of low E
take01 peaks at 82 Hz. No expected note label is used to select peaks.

Sections **5–6** add two code cells for autocorrelation: explicitly multiply and
sum overlapping samples at different shifts, find positive local similarity
peaks, and overlay the waveform at the chosen shift. Both E takes select 538
samples (81.97 Hz); A takes select 399/400 samples (110.53/110.25 Hz), and C takes
select 168 samples (262.50 Hz), using the same 3.0–3.5 s segment and 60–1,000 Hz
search bounds. These are candidate fundamentals, not guaranteed pitches. The
score is normalized by the zero-lag sum, is not a probability, and favors shorter
shifts somewhat because fewer samples overlap at larger shifts. It can still
make octave errors and has no noise-versus-note decision or chord handling.

Run the matching script with `.venv/bin/python frequency_lesson.py`, or use the
notebook setup below. No new packages are needed for the script. Notebook example
outputs were generated by executing its cells with the project interpreter,
not by testing a live Jupyter kernel. Known-tone checks at 44.1 and 48 kHz cover
frequency labels, DC offset removal, and a stronger second harmonic; exact
silence is rejected. Autocorrelation additionally passed 12 known-tone cases
(E/A/C frequencies at two sample rates, with/without stronger second harmonics),
gain/DC invariance checks, and silence/constant/too-short input checks. Real-take
results are development checks, not held-out accuracy measurements.

Run all cells in order, then inspect sections 5–6. Next: check estimates across
several ringing intervals, then analyze a short known melody over time.

`compare_recordings.py` loops through the labeled dry takes, estimates each note's
onset sample, and compares the next 8 seconds using 50 ms RMS windows. It saves
comparison plots, onset-review plots, and the original sample indices behind
every measurement. It uses the existing NumPy, SoundFile, and Matplotlib dependencies.

`main.py` loads a WAV recording, prints its sample rate, array shape, datatype,
duration, and first five time steps, then calculates left-channel RMS amplitude
and displays a waveform. A second plot zooms into 3.00–3.05 seconds, with RMS
reported separately for that interval.

The included `data/raw/Test_1_Cmaj.wav` is a personally recorded single C note,
despite the filename suggesting a chord. It is stereo, 44,100 Hz, 16-bit PCM,
and approximately 10.39 seconds long. Recording gain and effects settings have
not yet been documented.

## Run locally

From the project root, create and activate a virtual environment (macOS):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies and run the script:

```bash
python -m pip install -r requirements.txt
python main.py
```

Close both plot windows to finish the script. If the environment already exists,
activate it without creating it again. Development started with Python 3.13.

## Compare recordings

For the smaller, guided learning step, start with
`01_understanding_quiet.ipynb`. It uses one C take, an editable candidate quiet
interval, and visible RMS arithmetic. It does not change the batch detector.

Notebook setup from the project folder:

```bash
.venv/bin/python -m pip install -r requirements-notebook.txt
.venv/bin/python -m ipykernel install --user --name guitar-ml --display-name "Guitar ML (.venv)"
```

Install Microsoft's **Jupyter** extension (`ms-toolsai.jupyter`) in VS Code,
open the notebook, and select **Guitar ML (.venv)** as its kernel. A kernel is
the Python process that runs notebook cells. Run the lesson from top to bottom,
one cell at a time; rerun later cells after changing the quiet-region selection.

If notebook setup is unavailable, the same lesson runs with existing dependencies:

```bash
.venv/bin/python quiet_region_lesson.py
```

Close each plot to continue to the next step. The notebook's saved example
outputs were checked by executing its code cells in order with the project Python;
they do not establish that a Jupyter kernel is installed or working.

With the environment activated, run this from the project folder:

```bash
python compare_recordings.py
```

The loop finds `data/raw/*_note_dry_take*.wav`, including future D/G/B/high-E
takes with that naming convention. It uses the left channel for stereo or the
only channel for mono. The older `Test_1_Cmaj.wav` is excluded because its
recording/processing conditions were not established.

Two figures open, and copies are saved in `outputs/amplitude_comparison/`:

- `comparisons.png`: takes grouped by note, with 8-second RMS curves and a
  waveform close-up from 0.10–0.15 seconds after onset. Each column shares its
  amplitude scale across notes. Original signal levels are preserved.
- `onset_review.png`: estimated/manual start markers on each waveform, using
  the original file's timeline. Inspect these before trusting alignment.
- `measurements.csv`: filename, note label, sample rate, onset index/time/method,
  background RMS, detection threshold, each window's original start and exclusive
  stop indices, relative midpoint time, RMS, and absolute peak.

For onset sample `n0`, a window starts at `n0 + offset_samples`. Its relative time
is `(sample_index - n0) / sample_rate`. At 44,100 Hz, each 50 ms window has 2,205
samples, giving 160 windows over 8 seconds. The final incomplete window is
dropped if you change the duration; no zero-padding or per-file normalization is used.
Recordings without the full comparison span are skipped with an explanation.

The onset detector assumes the first 0.5 seconds are quiet. It looks for three
consecutive 10 ms frames with RMS above six times the initial background RMS
(with a small absolute floor), then refines the start to a sample threshold
crossing near that transition. This is a simple energy-based estimate, not an
exact physical onset or validated benchmark. It can mistake handling noise for
a note or miss a soft attack. Only the first sustained event is analyzed.

If an onset marker is wrong, enter the reviewed **original-file sample index**
in `ONSET_OVERRIDES` at the top of the script, for example
`{"A_note_dry_take01.wav": 90000}` (illustrative, not a recommended correction).
The CSV labels that onset as manual. The plotted dotted lines show the detection
threshold; they are not confidence bounds. Manual overrides do not require a
quiet lead-in, but background statistics still describe the first half-second.

Onset alignment does not align waveform phase or establish that two notes have
the same pitch. Note names come from filenames. Differences in picking, gain,
effects, pickup, and export volume can all affect these measurements. The C
export volume was adjusted during setup, so cross-note level comparisons are
exploratory, not a controlled experiment or calibrated loudness measurement.

Run the focused alignment and measurement checks with:

```bash
python -m unittest test_compare_recordings -v
```

A notebook can later import `analyze_file` and `plot_comparisons` from
`compare_recordings` for interactive exploration; importing either script does
not load audio or open figures. Keep calculations in the script to avoid
different notebooks silently using different analysis rules.

## Measurements and limitations

- SoundFile loads the samples as floating-point values. Each row represents one
  time step; the two columns are the left and right channels.
- Duration is the number of rows divided by the sample rate.
- RMS is `sqrt(mean(signal ** 2))`. The current measurement uses the entire left
  channel, including silence, and is approximately 0.1219 for this recording.
- The selected 3.00–3.05 second interval contains 2,205 samples per channel and
  has left-channel RMS approximately 0.1074. An earlier measurement of
  1.00–1.05 seconds gave approximately 0.3946. These equal-length windows show
  the decrease in signal level as the note decays; neither represents the whole
  recording. Changing the selected interval does not change whole-recording RMS.
- RMS describes digital signal level, not perceived loudness or pitch. Silence
  and recording gain affect it, so comparisons need consistent conditions.
- The script currently uses a fixed file path and assumes stereo input. It does
  not yet detect pitch, classify techniques, or save the plot.

The WAV is included so the example can be run after cloning. GarageBand projects
and the local Python environment are excluded from Git.
