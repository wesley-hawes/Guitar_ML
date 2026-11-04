# Experiments and repository guide

[Project overview](../README.md) · [Technical design and equations](technical-design.md) ·
[Development handoff](../PROJECT_CONTEXT.md) ·
[Archived experimental history](archive/2026-11-03-development-notes.md)

## Where to start

Current integrated workflow: run 08, then [11](../11_note_event_table.ipynb)
(`python note_event_pipeline.py`). It combines rise and quieter-transition cues
into one provisional table. Ends remain unknown; within-group repeat proposals
are flagged. Experiment 09 remains the unchanged rise-only baseline.

Latest comparison: experiment 10 tests quieter pitch entries against decay and
stopping. Run 08 first, then `python compare_quieter_entries.py`. The small
helper `quieter_transitions.py` adds evidence; it does not assemble final events.

For the earlier rise-only **analysis**, run experiment 08 followed by 09.
Both use the included personal recordings. Experiment 08 calculates pitch and
energy evidence; 09 proposes attacks without changing pitch groups. Neither
produces complete transcription or tablature.

For implementation review, start with `pitch_analysis.py`, `attack_analysis.py`,
and their tests. For experimental reasoning, start with the saved notebook
plots and the failure analysis in 09 section 3.

## Repository roles

| Location | Role |
| --- | --- |
| Root: `pitch_analysis.py`, `attack_analysis.py`, `note_events.py`, `guitartechs_midi.py`, `quieter_transitions.py`, `event_detection.py` | Reusable calculations and annotation parsing; no top-level experiment execution |
| Root: `main.py`, `compare_recordings.py` | Earlier audio utilities and onset-aligned amplitude comparison; `main.py` is a waveform example, not the transcription entry point |
| Root: numbered `.ipynb`, `*_lesson.py`, `broader_pitch_check.py` | Executable experiments, selected inputs, plots, and parameter choices |
| Root: `test_*.py` | Standard-library unit tests; discover with `python -m unittest discover -v` |
| [`data/raw/`](../data/raw/) | Included personal guitar WAVs |
| [`data/recording_sessions/`](../data/recording_sessions/) | Intended phrases, recording setup, provisional references, and attack settings |
| `outputs/amplitude_comparison/`, `outputs/pitch_tracking/` | Regenerated measurements and figures, ignored by Git |
| [`docs/`](.) | Experiment navigation, published diagnostic, and archived investigation history |
| [`PROJECT_CONTEXT.md`](../PROJECT_CONTEXT.md) | Detailed working handoff; older checkpoints describe historical states |

Executable files remain at the root deliberately. Several notebooks use local
module imports and `data/` paths relative to the working directory; the amplitude
comparison script also anchors paths to its own location. Moving these into
`src/` or `notebooks/` would require coordinated path, import, and run-command
changes. A package layout can follow application consolidation. No empty
application/model directories imply features that do not exist.

The historical `*_lesson.py` names identify the notebook companions. They retain
step-by-step derivations and experiments; they are not the public application API.
Notebook parameters may differ from their companion scripts after an experiment
—for example, notebook 02 preserves a different selected time interval. Read the
first code cell before comparing outputs; scripts do not synchronize notebooks.

## Experiment index

Run scripts from the repository root with `python <script>`. For notebooks,
select the project environment and execute cells in order. Saved outputs show
prior runs; rerun after changing data or parameters.

| Experiment | Companion script | Inputs | Question / contribution |
| --- | --- | --- | --- |
| [01 — Quiet-region characterization](../01_understanding_quiet.ipynb) | [`quiet_region_lesson.py`](../quiet_region_lesson.py) | Included C, A, low-E takes | Background RMS, amplitude thresholds, sustained onset evidence |
| [02 — Spectral and periodicity analysis](../02_understanding_frequency.ipynb) | [`frequency_lesson.py`](../frequency_lesson.py) | Included low-E take; editable segment | Hann-windowed FFT, harmonics, autocorrelation, dominant peak versus fundamental |
| [03 — Annotation alignment](../03_guitartechs_labels.ipynb) | [`guitartechs_labels_lesson.py`](../guitartechs_labels_lesson.py) | Guitar-TECHS P1 single notes | Tempo-aware MIDI labels and selected ringing-window comparison; label-guided inspection |
| [04 — Sliding-window pitch](../04_tracking_pitch.ipynb) | [`pitch_tracking_lesson.py`](../pitch_tracking_lesson.py) | Guitar-TECHS P1 or P2 single notes | Audio-only estimates, separate MIDI overlays, reliability checks, suspicious-period diagnostic |
| [05 — Longer-recording checks](../05_checking_more_audio.ipynb) | [`broader_pitch_check.py`](../broader_pitch_check.py) | Both P1 and P2 single notes | Fixed-setting transfer and raw versus centered RMS/mean-offset diagnostics |
| [06 — Scales and repeated pitches](../06_tracking_a_scale.ipynb) | [`scale_tracking_lesson.py`](../scale_tracking_lesson.py) | Selected player's single notes and C scale | Pitch-evidence grouping, transitions, and a repeated A4 merged by pitch-only grouping |
| [07 — Renewed RMS](../07_understanding_attacks.ipynb) | [`attack_lesson.py`](../attack_lesson.py) | Both P1/P2 C scales plus synthetic signal | Recent-history RMS comparison; additional excitation without a quiet gap |
| [08 — Personal-recording transfer](../08_personal_recordings.ipynb) | [`personal_recordings_lesson.py`](../personal_recordings_lesson.py) | Five WAVs selected by session manifest | Original sample indices, pitch/grouping/energy CSVs, hashes, and full-file diagnostics |
| [09 — Attack candidate selection](../09_detecting_attacks.ipynb) | [`attack_detection_lesson.py`](../attack_detection_lesson.py) | Outputs of 08; optional previous P2 outputs | Explicit gates, episode consolidation, suppression, soft-transition failures |
| [10 — Quieter note entries](../10_quieter_note_entries.ipynb) | [`compare_quieter_entries.py`](../compare_quieter_entries.py) | Outputs of 08 | Signed RMS changes, pitch corroboration, decay/stop controls; supplements the unchanged attack detector |
| [11 — Integrated note-entry table](../11_note_event_table.ipynb) | [`note_event_pipeline.py`](../note_event_pipeline.py) | Outputs of 08 | Merges rise/drop entry evidence, preserves pitch-only notes, proposes flagged repeats; CSV and Markdown tables |

The separate [`compare_recordings.py`](../compare_recordings.py) finds
`data/raw/*_note_dry_take*.wav`, estimates a first onset after an assumed quiet
0.5-second lead-in, and compares eight seconds using 50 ms RMS windows. It skips
recordings without the required span and supports reviewed sample-index overrides.
This earlier first-onset method is separate from experiment 09's attack detector.

## Included data and output interpretation

The [session manifest](../data/recording_sessions/2026-11-03_personal_takes.json)
selects the baseline, dynamics, faster, fastest, and repeated C-to-A takes.
The reported setup is neck + middle pickup, a relatively dry tone, and no
metronome. The intended pentatonic phrase and confirmed repeated-note count
provide sequence-level references, not independently timed onset labels.

The current shared RMS reference comes from 1.2–1.4 seconds in the dynamics take.
It is provisional: quiet status and consistent gain/export settings have not
been independently verified. Leading digital-zero padding is not a measured
noise floor. The early E2 estimate in that take is unwanted relative to the
intended phrase; its physical cause is unresolved.

`Test_1_Cmaj.wav` is a **single C note**, despite its name. It belongs to the
original waveform example, with less well-established processing conditions.
No filenames or recording amplitudes were changed during the presentation audit.

Experiment 08 writes `{id}_windows.csv`, `{id}_note_evidence.csv`,
`{id}_energy_rise.csv`, and `audit.json` beneath
`outputs/pitch_tracking/personal_2026-11-03/`. Experiment 09 adds per-frame
reasons, selected/suppressed episodes, settings, and plots in `attack_candidates/`.
It checks current WAV hashes against the earlier audit; rerun 08 after replacing
a recording. Hash checks protect against stale source audio, not every possible
parameter or code change—rerun both experiments after those changes.

Group boundaries span supporting window midpoints, not exact note durations.
A rejected pitch frame is not proof of silence. A candidate attack is not proof
of a picked note. Counts of passing frames or proposed events are not accuracy.

## External Guitar-TECHS experiments

Source: [Guitar-TECHS project](https://guitar-techs.github.io/) and
[dataset release](https://zenodo.org/records/14963133). Consult the release for
attribution, licensing, and annotation conventions before using the data.
Third-party audio and MIDI are not included in this repository. Download and
extract the required subsets separately; the current scripts expect:

```text
~/Downloads/
├── P1_singlenotes/
│   ├── audio/directinput/directinput_allsinglenotes.wav
│   └── midi/midi_allsinglenotes.mid
├── P2_singlenotes/       # same relative files as P1_singlenotes
├── P1_scales/
│   ├── audio/directinput/directinput_C.wav
│   └── midi/midi_C.mid
└── P2_scales/            # same relative files as P1_scales
```

Edit the folder/player variables in the relevant script or first notebook cell
if your extraction layout differs. For the optional P2 comparison in 09,
run 06 with `player='P2'`, `excerpt_seconds=9.3`, then 07. Experiment 07 requires
both players' scale files. Without these generated outputs, 09 skips that
comparison while still analyzing the included recordings.

Keep supplied MIDI timing unchanged during inspection. An annotation boundary
can differ from the audible attack; windows containing a transition need separate
review from stable note interiors. The parser preserves track identity even
when tracks share MIDI channels. That identity comes from annotations—it is not
an inferred guitar-string prediction.

## Published diagnostic

[`assets/fastest-soft-transition.png`](assets/fastest-soft-transition.png) is an
unmodified copy of the figure generated by `attack_detection_lesson.py` section 3:

```text
outputs/pitch_tracking/personal_2026-11-03/attack_candidates/fastest_soft_transition.png
```

It shows 1.75–2.40 seconds of the included fastest take. Red lines are selected
candidate peak midpoints, gray dotted lines are suppressed episodes, orange
pitch points are raw candidates, and green points pass the pitch checks.
The bottom panel caps the displayed ratio at 5; exported measurements are not
clipped. The absent marker at the soft G3→E3 transition illustrates a failure
mode and motivates joint pitch/attack reasoning.

Regenerate with the two commands in the README, then deliberately refresh this
published copy if parameters or source data change. Other generated figures
remain untracked; embedded notebook outputs preserve the broader investigations.

## Audit decisions

The presentation audit separated reusable DSP functions from experimental
orchestration in documentation without changing algorithms, imports, or paths.
The previous README is preserved in the [archive](archive/2026-11-03-development-notes.md)
to retain investigations without mixing superseded plans into current claims.

Remaining engineering work includes centralizing duplicated tracking checks,
packaging entry points, dependency locking, and CI. Those changes need their own
validation and are not implied by this documentation update. An audio example
paired with reviewed onset annotations would be a more useful next portfolio
artifact than an unvalidated overall accuracy number.

## Headless execution and notebook setup

From the repository root with the project environment activated, run without
interactive plot windows on macOS/Linux:


```bash
MPLBACKEND=Agg python personal_recordings_lesson.py
MPLBACKEND=Agg python attack_detection_lesson.py
```

Outputs are generated under `outputs/pitch_tracking/personal_2026-11-03/`, with
attack decisions in its `attack_candidates/` subdirectory. Generated outputs are
ignored by Git; the README figure is a deliberately published snapshot. No
Guitar-TECHS download is needed for this route. Its optional P2 comparison is
skipped if earlier external-data outputs are absent.

For notebooks, install the kernel dependency and register the environment:

```bash
python -m pip install -r requirements-notebook.txt
python -m ipykernel install --user --name guitar-ml --display-name "Guitar ML (.venv)"
```

Open the repository folder in a notebook-capable editor, such as VS Code with
the Jupyter extension. Select **Guitar ML (.venv)** and run
[08_personal_recordings.ipynb](../08_personal_recordings.ipynb) before
[09_detecting_attacks.ipynb](../09_detecting_attacks.ipynb). The notebook requirements
install a kernel, not a standalone JupyterLab server. The
[experiment guide](../docs/experiments.md) covers earlier analyses and external data.

