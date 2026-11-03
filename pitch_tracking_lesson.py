# %% [markdown]
# # Follow pitch through a recording, without using labels to select windows
# Our question: does the candidate frequency follow E2 → F2 → F#2, and what
# happens during the attacks and fading tails? Keep every candidate visible.
# This is a continuous analysis prototype, not yet note-event or tab generation.
#
# Architecture: WAV → samples → overlapping windows → pitch/RMS/similarity rows
# → plots and CSV. `pitch_analysis.py` contains the reusable single-window
# calculation. This notebook owns the explicit loop and plots. MIDI enters only
# AFTER all audio measurements are complete, as a comparison layer.
#
# ## 1. Transfer the unchanged method to P2
# Read 0–11.9 s from the selected player's direct-input WAV. Use its actual
# sample rate (48 kHz for both P1 and P2). Only the player changes for this test:
# window/hop sizes, pitch bounds, and the estimator stay fixed. No rejection
# rule has been added yet. Testing the raw baseline first shows what it needs.
# P2 has two identical channels in this excerpt; we use the left channel,
# following the existing mono/left-channel policy. No mixing or normalization.
# Keep original sample indices so every plotted estimate can be traced to audio.
# No new dependencies; use the Guitar ML (.venv) kernel and run from this folder.

# %%
from pathlib import Path
import csv
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pitch_analysis import estimate_period

player = "P2"  # Change to "P1" to reproduce the first example.
dataset_folder = Path.home() / "Downloads" / f"{player}_singlenotes"
audio_path = dataset_folder / "audio/directinput/directinput_allsinglenotes.wav"
excerpt_start_seconds = 0.0
excerpt_stop_seconds = 11.9
info = sf.info(audio_path)
sample_rate = info.samplerate
excerpt_start_sample = round(excerpt_start_seconds * sample_rate)
excerpt_stop_sample = round(excerpt_stop_seconds * sample_rate)
if not 0 <= excerpt_start_sample < excerpt_stop_sample <= info.frames:
    raise ValueError("Choose an excerpt entirely inside the WAV.")
audio, sample_rate = sf.read(audio_path, start=excerpt_start_sample, stop=excerpt_stop_sample)
signal = audio[:, 0] if audio.ndim == 2 else audio
if not np.all(np.isfinite(signal)):
    raise ValueError("Audio contains nonfinite samples.")
print(f"{dataset_folder.name}/{audio_path.name}: {sample_rate:,} samples/second")
print(f"Source channels: {info.channels}; analyzing {'left channel' if audio.ndim == 2 else 'mono'}")
print(f"Original excerpt samples [{excerpt_start_sample}:{excerpt_stop_sample}]")

# %% [markdown]
# ## 2. Define the window and hop separately
# A 100 ms window contains 4,800 samples at 48 kHz; a 20 ms hop advances 960
# samples. Consecutive windows overlap by 80 ms (3,840 samples). These are trial
# settings, not proven optimal values. Short windows follow changes but contain
# fewer cycles, making pitch estimates harder. At 82 Hz, 100 ms is only ~8 cycles.
#
# `range(0, len(signal) - window_samples + 1, hop_samples)` visits every full
# window. The `+1` includes a window ending exactly at the excerpt boundary.
# A final incomplete window is dropped, never filled with zeros. For this excerpt
# the last full window ends exactly at the boundary, so no tail is dropped.
# Each point is placed at its window midpoint. A 20 ms hop is not a guarantee of
# 20 ms onset accuracy: each estimate still summarizes 100 ms of audio.

# %%
window_seconds = 0.100
hop_seconds = 0.020
minimum_frequency = 60.0
maximum_frequency = 1000.0
window_samples = round(window_seconds * sample_rate)
hop_samples = round(hop_seconds * sample_rate)
if window_samples <= 0 or hop_samples <= 0 or window_samples > len(signal):
    raise ValueError("Choose positive window/hop sizes and an excerpt at least one window long.")
window_starts = range(0, len(signal) - window_samples + 1, hop_samples)
print(f"Window: {window_samples} samples ({window_samples / sample_rate:.3f} s)")
print(f"Hop: {hop_samples} samples ({hop_samples / sample_rate:.3f} s)")
print(f"Complete windows: {len(window_starts)}")
print(f"Search: {minimum_frequency:g}–{maximum_frequency:g} Hz, unchanged for every window")

# %% [markdown]
# ## 3. The sliding loop — audio only
# `local_start` indexes the loaded excerpt. Adding `excerpt_start_sample` gives
# the index in the original WAV. RMS is computed on the original window; the
# pitch function subtracts its mean internally to compare repeating patterns.
#
# The function returns candidate Hz, the selected lag, and similarity. A constant
# window or no positive local peak gives NaN (no candidate). That mathematical
# exception is NOT a learned or threshold-based noise detector. Quiet noise may
# still produce a frequency and a high similarity score. No smoothing, label-based
# correction, clipping of wrong guesses, or RMS/similarity rejection is applied.
#
# Each row is an observation window, not a separate played note. Adjacent rows
# overlap and are not independent test examples. We will inspect them before
# deciding whether a simple no-reliable-pitch rule is needed.

# %%
rows = []
for local_start in window_starts:
    window = signal[local_start:local_start + window_samples]
    candidate_hz, lag, score = estimate_period(
        window, sample_rate, minimum_frequency, maximum_frequency)
    original_start = excerpt_start_sample + local_start
    original_stop = original_start + window_samples
    rows.append(dict(
        dataset=dataset_folder.name,
        source_file=audio_path.name,
        sample_rate=sample_rate,
        start_sample=original_start,
        stop_sample_exclusive=original_stop,
        midpoint_seconds=(original_start + original_stop) / (2 * sample_rate),
        rms=float(np.sqrt(np.mean(window ** 2))),
        candidate_hz=float(candidate_hz),
        lag_samples=float(lag),
        similarity=float(score),
    ))

times = np.array([row["midpoint_seconds"] for row in rows])
pitches = np.array([row["candidate_hz"] for row in rows])
rms = np.array([row["rms"] for row in rows])
scores = np.array([row["similarity"] for row in rows])
print(f"Computed {len(rows)} windows; {np.isfinite(pitches).sum()} have mathematical pitch candidates.")
print("This count is not a count of reliable notes.")
print("First row:", rows[0])
print("Last row:", rows[-1])

# %% [markdown]
# ## 4. Add MIDI references after prediction
# Only now load the MIDI annotations. The same audio loop works without them.
# Missing MIDI only removes the comparison overlay; it does not block analysis.
# We keep the original label times, including any misalignment seen in lesson 03.
# A gap between labels is not proof of acoustic silence: ringing may continue.

# %%
labels = []
midi_path = dataset_folder / "midi/midi_allsinglenotes.mid"
if midi_path.exists():
    from guitartechs_midi import read_note_labels
    all_labels, _ = read_note_labels(midi_path)
    labels = [note for note in all_labels
              if note["end_seconds"] > excerpt_start_sample / sample_rate
              and note["start_seconds"] < excerpt_stop_sample / sample_rate]
    pitch_names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    for note in labels:
        pitch = note["midi_note"]
        note["name"] = f"{pitch_names[pitch % 12]}{pitch // 12 - 1}"
        note["label_hz"] = 440 * 2 ** ((pitch - 69) / 12)
    print(f"Loaded {len(labels)} reference labels after completing all estimates.")
else:
    print("No MIDI file: showing the audio measurements without reference labels.")

# %% [markdown]
# ## 5. Read pitch, level, and similarity together
# Orange dots show ALL finite candidates; blue lines are supplied MIDI frequencies.
# The frequency axis is logarithmic so octave jumps remain visible without
# cropping high-frequency mistakes. Each dot summarizes a complete 100 ms window.
#
# RMS is displayed in dBFS: `20 * log10(RMS)`. RMS 0.1 gives -20 dBFS; RMS 0.01
# gives -40 dBFS. More negative means weaker digital signal, not lower pitch.
# For plotting only, zero/tiny RMS is floored at 1e-12 (-240 dBFS); the CSV keeps
# original RMS. This scale makes faint tails visible without normalizing the WAV.
# Similarity near 1 means repeating waveform structure, not certainty about a note.
# Shorter windows also lower this particular score: at lag k, only N-k of N
# samples overlap. A strong repeat near 580 samples in a 4,800-sample window
# can score around 0.88, versus roughly 0.98 in a half-second window. Do not
# transfer similarity thresholds between window lengths without checking them.
#
# Inspect early ringing, each fading tail, and the transitions near 4 and 8 seconds.
# Look for octave jumps or stable-looking pitches at very low RMS. Do not tune a
# threshold simply to match this one recording or assume all label gaps are silent.

# %%
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
axes[0].scatter(times, pitches, s=10, color="tab:orange", label="All audio pitch candidates")
for index, note in enumerate(labels):
    first = max(note["start_seconds"], excerpt_start_sample / sample_rate)
    last = min(note["end_seconds"], excerpt_stop_sample / sample_rate)
    axes[0].plot([first, last], [note["label_hz"]] * 2, color="tab:blue", linewidth=2,
                 label="Supplied MIDI frequency" if index == 0 else None)
    axes[0].annotate(note["name"], ((first + last) / 2, note["label_hz"]),
                     xytext=(0, 9), textcoords="offset points", ha="center")
axes[0].set(yscale="log", ylim=(minimum_frequency * 0.9, maximum_frequency * 1.1),
            ylabel="Candidate frequency (Hz)",
            title=f"{dataset_folder.name}: 100 ms windows, 20 ms hops — no candidate filtering")
axes[0].set_yticks([60, 82, 110, 220, 440, 880, 1000],
                   labels=["60", "82", "110", "220", "440", "880", "1000"])
axes[0].minorticks_off()
axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, borderaxespad=0)
axes[1].plot(times, 20 * np.log10(np.maximum(rms, 1e-12)), color="0.3")
axes[1].set(ylabel="Window RMS (dBFS)")
axes[2].plot(times, scores, color="tab:green", marker=".", markersize=2, linewidth=0.7)
axes[2].set(ylabel="Similarity (not probability)", xlabel="Window midpoint in original file (seconds)",
            ylim=(-0.02, 1.02))
for ax in axes:
    ax.set_xlim(excerpt_start_sample / sample_rate, excerpt_stop_sample / sample_rate)
    ax.grid(alpha=0.2)
    for note in labels:
        ax.axvline(note["start_seconds"], color="tab:blue", alpha=0.2, linestyle="--")
plt.show()

# %% [markdown]
# ## 6. Save measurements for inspection
# The CSV has one row per window and the exact original sample boundaries. It
# contains audio-derived measurements only, no MIDI-derived targets. It can be
# regenerated and is ignored by Git. No notes have been grouped into events yet.

# %%
output_folder = Path("outputs/pitch_tracking")
output_folder.mkdir(parents=True, exist_ok=True)
output_path = output_folder / f"{dataset_folder.name}_windows.csv"
with output_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
fig.savefig(output_folder / f"{dataset_folder.name}_pitch_tracking.png", dpi=150)
print("Saved:", output_path)

# %% [markdown]
# ## What the P2 transfer check shows (2026-10-29)
# Same 100 ms / 20 ms settings and frequency bounds, with no rejection threshold.
# Each excerpt has 591 windows. P2's stereo channels are identical over 0–11.9 s;
# using the left channel avoids changing the input policy or mixing amplitude.
#
# Median candidate Hz in the same inspection intervals (these are summaries of
# ringing portions, NOT full-recording accuracy scores or independent examples):
#
# | Original time | MIDI reference | P1 median Hz | P2 median Hz |
# |---|---|---|---|
# | 0.5–3.5 s | E2, 82.41 Hz | 82.76 | 81.91 |
# | 4.5–7.5 s | F2, 87.31 Hz | 87.91 | 86.96 |
# | 8.5–11.5 s | F#2, 92.50 Hz | 93.02 | 91.95 |
#
# P2 follows the note sequence during ringing but has more unstable gap/tail
# estimates. Example: midpoint 7.71 s gives **607.59 Hz**, similarity **0.901**,
# RMS **0.000563**. This is not evidence of a new confidently identified note.
# It is repeating structure that our method selects in a gap between labels.
# Its physical source has not been established. At 3.87 s, another mismatched
# candidate is 666.67 Hz with similarity 0.878 and RMS 0.001389.
#
# Some P1 ringing windows have RMS below 0.001389 and similarity around 0.88.
# Consequently, simply requiring larger RMS and similarity cannot perfectly
# separate all these useful and mismatched estimates across both recordings.
# High score does not mean correct pitch. Do not narrow the frequency search to
# 75–110 Hz to hide errors; other valid guitar notes lie outside this excerpt.
#
# Next: use this evidence when designing a small rejection experiment. Nearby
# pitch consistency and local level changes may help, but must also be checked
# on real note transitions and eventually faster notes. Do not tune a score-only
# cutoff to P1, and do not claim an acceptance rule has already been validated.
#
# %% [markdown]
# ## 7. Compare level with a candidate background interval
# Before choosing a filter, inspect an editable, quiet-looking part of the audio.
# These times were chosen from the level plot, NOT from MIDI gaps. They are NOT
# verified silence: muted-string vibration or handling sounds may remain.
# P1 and P2 need separate selections because their playing times differ.
#
# Reuse the existing 100 ms RMS measurements. Include a window only when its
# ENTIRE sample interval lies inside the selection; a midpoint alone could admit
# samples from the next attack. No re-windowing or change to pitch estimation.
#
# The median is the middle window RMS, less influenced by one unusual window
# than the mean. It is a provisional level reference, not a detection threshold.
# `level_ratio = window_RMS / reference_RMS`: 1 means equal level, 4 means four
# times the RMS amplitude (about +12 dB). It is NOT a measured signal-to-noise
# ratio: each window still contains a mixture of guitar and background.

# %%
candidate_intervals = {"P1": (7.78, 7.90), "P2": (7.70, 7.86)}
background_start_seconds, background_stop_seconds = candidate_intervals[player]
background_start_sample = round(background_start_seconds * sample_rate)
background_stop_sample = round(background_stop_seconds * sample_rate)
starts = np.array([row["start_sample"] for row in rows])
stops = np.array([row["stop_sample_exclusive"] for row in rows])
background_mask = ((starts >= background_start_sample)
                   & (stops <= background_stop_sample))
if not np.any(background_mask):
    raise ValueError("Select a candidate background interval containing a complete analysis window.")
reference_rms = float(np.median(rms[background_mask]))
if reference_rms <= 0:
    raise ValueError("A digital-zero reference cannot be used for a level ratio; inspect another interval.")
level_ratio = rms / reference_rms
print(f"Candidate interval: [{background_start_seconds:.2f}, {background_stop_seconds:.2f}) s")
print(f"Original samples: [{background_start_sample}:{background_stop_sample}]")
print(f"Reference RMS = median of {background_mask.sum()} overlapping windows = {reference_rms:.7f}")
print("These windows are NOT independent noise examples; this short interval needs review.")
print("\nInterval                         windows   min / median / max RMS-to-reference")
inspection_intervals = [
    ("Candidate background", background_start_seconds, background_stop_seconds),
    ("Second candidate tail", 11.60, 11.90),
    ("E2 ringing", 0.50, 1.50),
    ("E2 later ringing", 2.50, 3.50),
    ("F2 ringing", 4.50, 5.50),
    ("F2 later ringing", 6.50, 7.40),
    ("F#2 ringing", 8.50, 9.50),
]
for name, first, last in inspection_intervals:
    inside = ((starts >= round(first * sample_rate))
              & (stops <= round(last * sample_rate)))
    if np.any(inside):
        values = level_ratio[inside]
        print(f"{name:32s} {inside.sum():4d}       "
              f"{values.min():6.2f} / {np.median(values):6.2f} / {values.max():6.2f}")
print("\nNo candidate was accepted or rejected. Changing this selection only changes the comparison.")

# %% [markdown]
# ## 8. Inspect the reference before trusting it
# The zoom shows actual samples and the selected region. The lower plot shows
# all 591 RMS ratios. Its dashed line is equal-to-reference, NOT a cutoff.
# Compare the two candidate quiet intervals: different levels suggest our short
# selection may not represent background throughout the recording.
# Raising a future threshold rejects more background but also loses quiet note
# tails. Similarity will provide another check, but high similarity can still
# accompany a wrong pitch. We have not yet added persistence across windows.

# %%
background_fig, background_axes = plt.subplots(
    2, 1, figsize=(11, 6), constrained_layout=True)
zoom_start = max(excerpt_start_sample, background_start_sample - round(0.30 * sample_rate))
zoom_stop = min(excerpt_stop_sample, background_stop_sample + round(0.30 * sample_rate))
zoom_indices = np.arange(zoom_start, zoom_stop)
background_axes[0].plot(
    zoom_indices / sample_rate, signal[zoom_indices - excerpt_start_sample],
    color="0.35", linewidth=0.5)
background_axes[0].axvspan(background_start_seconds, background_stop_seconds,
                           color="tab:purple", alpha=0.25, label="Candidate background")
background_axes[0].set(ylabel="Sample amplitude", xlabel="Original-file time (s)",
                       title=f"{player}: inspect the proposed level reference")
background_axes[0].legend()
background_axes[1].plot(times, level_ratio, color="0.35")
background_axes[1].axhline(1, color="tab:purple", linestyle="--", label="Equal to reference RMS")
for first, last in [(background_start_seconds, background_stop_seconds), (11.60, 11.90)]:
    background_axes[1].axvspan(first, last, color="tab:purple", alpha=0.2)
background_axes[1].set(yscale="log", ylabel="RMS / candidate reference RMS",
                       xlabel="Window midpoint in original file (s)")
background_axes[1].legend()
for ax in background_axes:
    ax.grid(alpha=0.2)
background_fig.savefig(output_folder / f"{dataset_folder.name}_background_comparison.png", dpi=150)
plt.show()

# %% [markdown]
# Learning checkpoint: trace `background_mask` → `reference_rms` → `level_ratio`.
# This is an inspection layer after the unchanged pitch loop. The original CSV
# and raw pitch plot remain intact. No MIDI labels enter the level calculation.
# Next, review this comparison before trying one explicit level-plus-similarity
# rule. Keep raw candidates alongside rejection reasons, call failures "no
# reliable pitch" rather than "silence", and test both players. A future
# automatic background estimator must not assume the recording starts silent.

# %% [markdown]
# ## 9. Try an explicit three-check rule
# This is a trial, not a validated note detector. The candidate background from
# section 7 is still manually chosen and may contain residual guitar vibration.
#
# 1. Level: RMS >= 3 times that reference (about +9.5 dB in amplitude level).
# 2. Repetition: autocorrelation similarity >= 0.80 for our 100 ms windows.
# 3. Local agreement: this AND the previous window pass checks 1–2, and their
# pitches differ by <= 50 cents. 100 cents is one semitone; 1,200 is an octave.
# `1200 * log2(current_hz / previous_hz)` expresses a frequency ratio in cents:
# doubling frequency gives 1,200, and equal frequencies give zero. Take absolute
# value because either an upward or downward change can break agreement.
#
# Why these numbers? They are readable starting hypotheses: a modest level
# margin, substantial repetition, and agreement within half a semitone (~2.9%).
# They were not optimized against MIDI. Keep them fixed when switching players.
# Overlapping windows share 80% of their samples: agreement is supporting evidence,
# not two independent confirmations. A stable wrong harmonic or hum can pass.
#
# This checks the previous RAW candidate, not the last accepted pitch. Otherwise
# the tracker could get stuck on the old note and reject every new pitch. A new
# pitch can qualify on its second consecutive eligible window. The first window
# stays unaccepted; we do not backfill it. That adds one 20 ms hop to this check,
# beyond the 100 ms analysis window. Fast notes, bends and attacks may be lost.
# These rules express "passes our checks", never proof of correctness or silence.

# %%
minimum_level_ratio = 3.0
minimum_similarity = 0.80
maximum_step_cents = 50.0

finite_pitch = np.isfinite(pitches) & (pitches > 0)
level_ok = np.isfinite(level_ratio) & (level_ratio >= minimum_level_ratio)
repetition_ok = np.isfinite(scores) & (scores >= minimum_similarity)
eligible = finite_pitch & level_ok & repetition_ok
step_cents = np.full(len(rows), np.nan)
continuity_ok = np.zeros(len(rows), dtype=bool)
accepted = np.zeros(len(rows), dtype=bool)
rejection_reasons = []
for index in range(len(rows)):
    if index > 0 and finite_pitch[index] and finite_pitch[index - 1]:
        step_cents[index] = abs(1200 * np.log2(pitches[index] / pitches[index - 1]))
        continuity_ok[index] = (eligible[index - 1]
                                and step_cents[index] <= maximum_step_cents)
    accepted[index] = eligible[index] and continuity_ok[index]
    reasons = []
    if not finite_pitch[index]:
        reasons.append("no_finite_pitch")
    if not level_ok[index]:
        reasons.append("low_relative_level")
    if not repetition_ok[index]:
        reasons.append("weak_repetition")
    if index == 0 or not eligible[index - 1]:
        reasons.append("needs_previous_eligible_window")
    elif finite_pitch[index] and step_cents[index] > maximum_step_cents:
        reasons.append("pitch_change")
    rejection_reasons.append(";".join(reasons))

accepted_hz = np.where(accepted, pitches, np.nan)
print(f"{accepted.sum()} / {len(rows)} windows pass the trial rule; this is NOT accuracy.")
print(f"Level >= {minimum_level_ratio:g}x reference; similarity >= {minimum_similarity:.2f}; "
      f"pitch step <= {maximum_step_cents:g} cents with previous eligible window.")
print("Raw frequencies remain unchanged. Rejected windows mean no reliable pitch under this rule.")

# %% [markdown]
# ## 10. See what the rule does through a tail and the next attack
# Orange points are raw candidates; green points pass all three checks.
# Zoom starts at 7.3 s to include F2 ringing, its tail/gap, and the F#2 attack.
# Change `review_start_seconds` / `review_stop_seconds` to 3.5 / 4.2 to inspect
# the first transition too. Blue MIDI lines are comparison references ONLY.
# The printed table exposes individual decisions; it does not label noise truth.

# %%
review_start_seconds = 7.3
review_stop_seconds = 8.2
review = (times >= review_start_seconds) & (times <= review_stop_seconds)
filter_fig, filter_axes = plt.subplots(4, 1, figsize=(11, 10), constrained_layout=True)
for ax in filter_axes[:2]:
    ax.scatter(times, pitches, color="tab:orange", s=14, alpha=0.6, label="Raw candidates")
    ax.scatter(times, accepted_hz, color="tab:green", s=9, label="Pass trial rule")
    for number, note in enumerate(labels):
        ax.plot([note["start_seconds"], note["end_seconds"]], [note["label_hz"]] * 2,
                color="tab:blue", linewidth=1, label="MIDI reference" if number == 0 else None)
    ax.set(yscale="log", ylim=(minimum_frequency * 0.9, maximum_frequency * 1.1), ylabel="Hz")
    ax.set_yticks([60, 82, 110, 220, 440, 880], labels=["60", "82", "110", "220", "440", "880"])
    ax.minorticks_off()
filter_axes[0].set(xlim=(excerpt_start_seconds, excerpt_stop_seconds),
                   title=f"{player}: raw estimates and an experimental acceptance rule")
filter_axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
filter_axes[2].plot(times, level_ratio, color="0.3")
filter_axes[2].axhline(minimum_level_ratio, color="tab:red", linestyle="--")
filter_axes[2].set(yscale="log", ylabel="RMS / reference RMS")
filter_axes[3].plot(times, scores, color="0.3")
filter_axes[3].axhline(minimum_similarity, color="tab:red", linestyle="--")
filter_axes[3].set(ylabel="Similarity", ylim=(0, 1.02), xlabel="Original-file window midpoint (s)")
for ax in filter_axes[1:]:
    ax.set_xlim(review_start_seconds, review_stop_seconds)
for ax in filter_axes:
    ax.grid(alpha=0.2)
filter_fig.savefig(output_folder / f"{dataset_folder.name}_pitch_filter.png", dpi=150)
plt.show()
print("Time    Hz       Level  Score  Step(cents)  Decision")
for index in np.flatnonzero(review):
    print(f"{times[index]:5.2f}  {pitches[index]:7.2f}  {level_ratio[index]:5.1f}  "
          f"{scores[index]:5.3f}  {step_cents[index]:11.1f}  "
          f"{'PASS' if accepted[index] else rejection_reasons[index]}")

# %% [markdown]
# ## 11. Save decisions separately from the raw measurements
# Each decision keeps original sample boundaries, raw Hz, each check, rejection
# reasons and accepted Hz (NaN if rejected). Store the reference and settings too,
# so a later threshold change cannot silently change what an older CSV meant.
# Architecture: raw audio measurements → decision checks → inspection/export.
# `pitch_analysis.py` still only estimates one window. MIDI is never an input to
# these decisions; labels only help us inspect results. No note events yet.

# %%
decision_rows = []
for index, row in enumerate(rows):
    decision_rows.append(dict(
        row,
        background_start_sample=background_start_sample,
        background_stop_sample_exclusive=background_stop_sample,
        reference_rms=reference_rms,
        minimum_level_ratio=minimum_level_ratio,
        minimum_similarity=minimum_similarity,
        maximum_step_cents=maximum_step_cents,
        level_ratio=float(level_ratio[index]),
        finite_pitch=bool(finite_pitch[index]),
        level_ok=bool(level_ok[index]),
        repetition_ok=bool(repetition_ok[index]),
        step_cents=float(step_cents[index]),
        continuity_ok=bool(continuity_ok[index]),
        accepted=bool(accepted[index]),
        accepted_hz=float(accepted_hz[index]),
        rejection_reasons=rejection_reasons[index],
    ))
decision_path = output_folder / f"{dataset_folder.name}_pitch_decisions.csv"
with decision_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(decision_rows[0]))
    writer.writeheader()
    writer.writerows(decision_rows)
print("Saved decisions:", decision_path)

# %% [markdown]
# ## What this first filter changes — and still misses
# With the SAME 3x / 0.80 / 50-cent settings, P1 has 546 passing windows and
# P2 has 528 out of 591 each. These counts are NOT accuracy scores. All complete
# windows in our five selected ringing intervals (section 7 table) pass for both.
# P2's 607.59 Hz candidate at 7.71 s fails the level check despite similarity
# 0.901; 666.67 Hz at 3.87 s fails the pitch-step check.
#
# A remaining counterexample: P2 at 3.81 s gives 317.88 Hz and PASSES. It has
# enough level and similarity and agrees with the previous candidate. A brief
# wrong stable sequence can pass all three checks. Inspect 3.5–4.2 s in section
# 10 to see it. We did not retune thresholds to hide this failure.
# P1 also keeps two windows in the second candidate tail: that interval was
# never established as pure noise, and some ringing may remain.
#
# This rule compares adjacent pitches, not the whole tail against a previously
# established note. That avoids locking out new notes but permits stable wrong
# sequences. Before adding complexity, understand one accepted and one rejected
# row, then inspect a longer passage with the settings fixed. Note-event grouping
# and attack detection are separate next steps; repeated same-pitch plucks will
# not be distinguished by pitch continuity alone.

# %% [markdown]
# ## 12. Inspect one suspicious estimate without tuning around it
# P2's 3.81 s point summarizes samples from 3.76–3.86 s. Compare it with a
# clear ringing window centered at 1.01 s, using the SAME 100 ms length and
# 60–1,000 Hz search. These are deliberately chosen examples, not an accuracy
# evaluation or evidence about how common this behavior is. If player is P1,
# the same times inspect P1 instead; the P2 finding need not appear.
#
# Top: original sample amplitudes, with the SAME vertical scale in both columns.
# Middle: the exact mean-subtracted, full-window-energy-normalized correlation
# used by our estimator. Each local peak is a possible repeating shift; orange
# marks the selected one. The horizontal axis is SHIFT, not recording time.
# Bottom: first 30 ms of the centered waveform and its selected shifted copy.
# Both traces use the original amplitudes, but each column gets its own vertical
# scale so quiet patterns remain visible. This close-up does not change the
# full 100 ms calculation. No Hann window or FFT is involved in autocorrelation.
#
# Diagnostic architecture: read the existing windows → reproduce the correlation
# curve → verify it matches estimate_period → plot. No settings, estimates, or
# decisions are changed. A high shifted match means similarity at that shift;
# a slow trend can also match well. It does not prove a new audible note, its
# physical source, or the correct fundamental.

# %%
diagnostic_times = [("Clear ringing example", 1.01), ("Suspicious P2 example", 3.81)]
diagnostic_fig, diagnostic_axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
diagnostic_peak_amplitudes = []
for column, (description, requested_time) in enumerate(diagnostic_times):
    frame_index = int(np.argmin(abs(times - requested_time)))
    if abs(times[frame_index] - requested_time) > hop_seconds / 2 + 1e-9:
        raise ValueError("Diagnostic time is outside the analyzed window midpoints.")
    frame = rows[frame_index]
    local_start = frame["start_sample"] - excerpt_start_sample
    samples = signal[local_start:local_start + window_samples]
    measured_hz, selected_lag, measured_score = estimate_period(
        samples, sample_rate, minimum_frequency, maximum_frequency)
    if not np.isfinite(selected_lag):
        raise ValueError("This example has no finite repeating candidate; choose another window.")
    selected_lag = int(selected_lag)
    centered = samples - np.mean(samples)
    energy = np.sum(centered ** 2)
    first_lag = int(np.ceil(sample_rate / maximum_frequency))
    last_lag = int(np.floor(sample_rate / minimum_frequency))
    correlation = np.empty(last_lag + 2)
    correlation[0] = 1.0
    for shift in range(1, len(correlation)):
        correlation[shift] = np.sum(centered[:-shift] * centered[shift:]) / energy
    local_peaks = [shift for shift in range(first_lag, last_lag + 1)
                   if correlation[shift] > 0
                   and correlation[shift] > correlation[shift - 1]
                   and correlation[shift] >= correlation[shift + 1]]
    assert selected_lag == max(local_peaks, key=lambda shift: correlation[shift])
    assert np.isclose(correlation[selected_lag], measured_score)
    assert np.isclose(measured_hz, frame["candidate_hz"])
    assert np.isclose(measured_score, frame["similarity"])

    original_time = (frame["start_sample"] + np.arange(len(samples))) / sample_rate
    diagnostic_axes[0, column].plot(original_time, samples, linewidth=0.7)
    diagnostic_axes[0, column].set(
        title=f"{player}: {description}\nmidpoint {times[frame_index]:.2f} s",
        xlabel="Original-file time (s)", ylabel="Sample amplitude")
    diagnostic_peak_amplitudes.append(float(np.max(abs(samples))))

    shifts = np.arange(first_lag, last_lag + 1)
    ax = diagnostic_axes[1, column]
    ax.plot(shifts / sample_rate * 1000, correlation[shifts], linewidth=1)
    ax.scatter(np.array(local_peaks) / sample_rate * 1000,
               correlation[local_peaks], s=15, color="0.5", label="Candidate peaks")
    ax.scatter(selected_lag / sample_rate * 1000, measured_score,
               color="tab:orange", s=50, zorder=3, label="Selected repeat")
    ax.set(xlabel="Shift (milliseconds)", ylabel="Autocorrelation similarity",
           ylim=(-1.05, 1.05), title=f"{selected_lag} samples → {measured_hz:.2f} Hz; score {measured_score:.3f}")
    ax.legend(fontsize=8, loc="lower right")

    view_samples = min(round(0.030 * sample_rate), len(samples) - selected_lag)
    relative_ms = np.arange(view_samples) / sample_rate * 1000
    ax = diagnostic_axes[2, column]
    ax.plot(relative_ms, centered[:view_samples], label="Original", linewidth=1)
    ax.plot(relative_ms, centered[selected_lag:selected_lag + view_samples],
            label=f"Shifted by {selected_lag} samples", linewidth=1, alpha=0.8)
    ax.set(xlabel="Time from window start (ms)", ylabel="Centered amplitude",
           title="Selected-shift overlay (separate vertical scale)")
    ax.legend(fontsize=8)
    print(f"\n{player}, midpoint {times[frame_index]:.2f} s: "
          f"samples [{frame['start_sample']}:{frame['stop_sample_exclusive']}]")
    print(f"RMS {frame['rms']:.7f}; level/reference {level_ratio[frame_index]:.2f}; "
          f"passes filter: {accepted[frame_index]}")
    print("Strongest local peaks: shift(samples), Hz, similarity")
    for shift in sorted(local_peaks, key=lambda shift: correlation[shift], reverse=True)[:3]:
        print(f"  {shift:4d}   {sample_rate / shift:8.2f}   {correlation[shift]:.4f}")

shared_limit = max(diagnostic_peak_amplitudes) * 1.05
for ax in diagnostic_axes[0]:
    ax.set_ylim(-shared_limit, shared_limit)
for ax in diagnostic_axes.flat:
    ax.grid(alpha=0.2)
diagnostic_path = output_folder / f"{dataset_folder.name}_candidate_diagnostic.png"
diagnostic_fig.savefig(diagnostic_path, dpi=150)
plt.show()
print("Saved diagnostic:", diagnostic_path)
print("No thresholds or decisions changed. Inspect more audio with settings fixed next.")

# %% [markdown]
# Reading this P2 example: at 1.01 s the selected 586-sample shift has a distinct
# correlation peak (~81.91 Hz). At 3.81 s the 151-sample shift (~317.88 Hz) is a
# tiny local bump on a broadly declining curve. The code asks for the highest
# positive local maximum; it does not require that peak to stand out from nearby
# shifts. A slowly varying waveform can therefore produce a high score without
# the clear cycle structure seen on the left. The physical source remains unknown.
#
# Another observation: raw RMS includes a window's mean offset. In the suspicious
# window raw RMS is ~0.002189, its mean is ~-0.002124, and mean-subtracted RMS is
# ~0.000529. The level check uses raw RMS while autocorrelation subtracts the
# mean. These are different measurements. This is worth checking across more
# recordings before considering a preprocessing change; none is made here.
#
# Stop after understanding why this shift was selected. Do not add a special
# rule for 318 Hz or this timestamp. Next expand the excerpt to 0–30 seconds
# with the existing settings fixed, report passing and rejected regions, and
# inspect more note interiors and transitions. That broadens our evidence before
# changing the algorithm; it is still development data, not a held-out test.

# %% [markdown]
# Next: inspect this trial rule before extending it. Still missing are
# validated note-presence decisions, attack detection, grouping windows into note
# events, chord handling, and string/fret assignment. Shorter windows may favor
# octave errors; the 60–1,000 Hz search excludes some higher guitar notes.
# This first comparison does not establish accuracy on other performers or sounds.
#
# Dataset: Pedroza et al., *Guitar-TECHS*, ICASSP 2025,
# [official release](https://zenodo.org/records/14963133),
# [project and CC BY 4.0 license](https://guitar-techs.github.io/).
