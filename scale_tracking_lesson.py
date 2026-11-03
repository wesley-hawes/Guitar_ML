# %% [markdown]
# # First scale check: transitions before chords
# P2's C-scale direct-input recording, first 8 seconds, with the SAME estimator,
# 100 ms windows, 20 ms hops, 60–1,000 Hz bounds, and 3x / 0.80 / 50-cent rule.
# The filename is C; read actual MIDI pitches rather than assuming it starts on C.
# The first sections test the unchanged tracker. Sections 5–6 add simple
# note-evidence grouping, with no new pitch detector or chord recognition.
#
# Three different meanings of overlap:
# - Our analysis windows overlap each other by 80%.
# - One window can contain the end of one note and beginning of another, even
#   when the two sounds never occur at the exact same instant.
# - Strings can actually ring simultaneously. A one-frequency estimator cannot
#   represent both pitches; MIDI overlap alone does not prove acoustic overlap.
#
# Keep the extracted audio/MIDI in Downloads. If not extracted yet, unzip ONLY
# P2_scales/audio/directinput/directinput_C.wav and P2_scales/midi/midi_C.mid
# from ~/Downloads/P2_scales.zip into ~/Downloads, preserving those subfolders.
# Use the existing Guitar ML (.venv) kernel; no new dependencies.

# %%
from pathlib import Path
import csv
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pitch_analysis import estimate_period
from guitartechs_midi import read_note_labels

folder = Path.home() / 'Downloads/P2_scales'
audio_path = folder / 'audio/directinput/directinput_C.wav'
midi_path = folder / 'midi/midi_C.mid'
excerpt_seconds = 8.0
window_seconds, hop_seconds = 0.100, 0.020
minimum_frequency, maximum_frequency = 60.0, 1000.0
minimum_level_ratio, minimum_similarity, maximum_step_cents = 3.0, 0.80, 50.0
info = sf.info(audio_path)
rate = info.samplerate
stop_sample = round(excerpt_seconds * rate)
if stop_sample > info.frames:
    raise ValueError('Requested excerpt exceeds the recording.')
audio, rate = sf.read(audio_path, stop=stop_sample)
signal = audio[:, 0] if audio.ndim == 2 else audio
width, hop = round(window_seconds * rate), round(hop_seconds * rate)
print(f'{audio_path.name}: {rate:,} Hz, {info.channels} channel(s); original samples [0:{stop_sample}]')
if audio.ndim == 2:
    print('Stereo channels identical in this excerpt:', np.array_equal(audio[:, 0], audio[:, 1]))

# %% [markdown]
# ## 1. Carry over the old reference; don't pretend the opening is silence
# Use exactly the P2 single-note reference from lessons 04–05, computed from its
# original 100 ms windows fully inside 7.70–7.86 s. The scale starts with notes.
# Same player/source type does NOT guarantee identical gain/noise across files:
# this is an explicitly provisional transfer, not a measured scale noise floor.
# Inspect raw estimates even when the level check rejects them.

# %%
reference_path = Path.home() / 'Downloads/P2_singlenotes/audio/directinput/directinput_allsinglenotes.wav'
reference_info = sf.info(reference_path)
reference_rate = reference_info.samplerate
reference_start, reference_stop = round(7.70 * reference_rate), round(7.86 * reference_rate)
reference_audio, _ = sf.read(reference_path, start=reference_start, stop=reference_stop)
reference_signal = reference_audio[:, 0] if reference_audio.ndim == 2 else reference_audio
reference_width, reference_hop = round(window_seconds * reference_rate), round(hop_seconds * reference_rate)
# Align to the original file's hop grid, not an arbitrary new grid.
first_reference_frame = ((reference_start + reference_hop - 1) // reference_hop) * reference_hop
reference_values = []
for start in range(first_reference_frame, reference_stop - reference_width + 1, reference_hop):
    local_start = start - reference_start
    segment = reference_signal[local_start:local_start + reference_width]
    reference_values.append(float(np.sqrt(np.mean(segment ** 2))))
if not reference_values or np.median(reference_values) <= 0:
    raise ValueError('Reference must contain positive-RMS complete windows.')
reference_rms = float(np.median(reference_values))
print(f'Carried-over reference: {reference_rms:.7f} from {len(reference_values)} old-file windows')

# %% [markdown]
# ## 2. Same audio calculations and checks, new recording
# Each row keeps original sample indices. Frequency comes from estimate_period;
# the visible checks below reproduce the earlier acceptance rule.
# First qualifying window after a pitch jump waits for agreement with the next.
# That can help reject unstable estimates but can also delay/drop short notes.

# %%
rows = []
previous_eligible, previous_hz = False, np.nan
for start in range(0, len(signal) - width + 1, hop):
    window = signal[start:start + width]
    hz, lag, score = estimate_period(window, rate, minimum_frequency, maximum_frequency)
    rms = float(np.sqrt(np.mean(window ** 2)))
    level_ratio = rms / reference_rms
    finite_pitch = bool(np.isfinite(hz) and hz > 0)
    level_ok = bool(level_ratio >= minimum_level_ratio)
    repetition_ok = bool(np.isfinite(score) and score >= minimum_similarity)
    eligible = finite_pitch and level_ok and repetition_ok
    step_cents = abs(1200 * np.log2(hz / previous_hz)) if finite_pitch and np.isfinite(previous_hz) and previous_hz > 0 else np.nan
    continuity_ok = bool(previous_eligible and np.isfinite(step_cents) and step_cents <= maximum_step_cents)
    accepted = bool(eligible and continuity_ok)
    reasons = []
    if not finite_pitch: reasons.append('no_finite_pitch')
    if not level_ok: reasons.append('low_relative_level')
    if not repetition_ok: reasons.append('weak_repetition')
    if not previous_eligible: reasons.append('needs_previous_eligible_window')
    elif finite_pitch and step_cents > maximum_step_cents: reasons.append('pitch_change')
    rows.append(dict(source_file=audio_path.name, sample_rate=rate,
        start_sample=start, stop_sample_exclusive=start + width,
        midpoint_seconds=(start + width / 2) / rate, rms=rms, candidate_hz=hz,
        lag_samples=lag, similarity=score, level_ratio=level_ratio, level_ok=level_ok,
        repetition_ok=repetition_ok, step_cents=step_cents, continuity_ok=continuity_ok,
        accepted=accepted, accepted_hz=hz if accepted else np.nan,
        rejection_reasons=';'.join(reasons), reference_source=str(reference_path),
        reference_start_sample=reference_start, reference_stop_sample_exclusive=reference_stop,
        reference_rms=reference_rms, minimum_level_ratio=minimum_level_ratio,
        minimum_similarity=minimum_similarity, maximum_step_cents=maximum_step_cents))
    previous_eligible, previous_hz = eligible, hz
print(f"{len(rows)} complete windows; {sum(r['accepted'] for r in rows)} pass (NOT accuracy).")

# %% [markdown]
# ## 3. Add reference notes and identify windows spanning label changes
# MIDI only enters AFTER prediction and filtering. A window intersecting two
# label spans isn't necessarily two notes sounding simultaneously: it can cross
# the change between sequential notes. Labels can also lag audio and contain
# brief artifacts. Keep all labels visible; don't silently repair them.
# A 'guarded interior' window is fully inside exactly one label with 100 ms
# margins at both ends. This comparison excludes uncertain boundaries, not all
# hard cases. Counts overlap and must not be called full transcription accuracy.

# %%
all_labels, _ = read_note_labels(midi_path)
labels = [note for note in all_labels if note['start_seconds'] < excerpt_seconds and note['end_seconds'] > 0]
names = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
for note in labels:
    midi = note['midi_note']
    note['name'] = f'{names[midi % 12]}{midi // 12 - 1}'
    note['hz'] = 440 * 2 ** ((midi - 69) / 12)
    duration = note['end_seconds'] - note['start_seconds']
    print(f"{note['name']:4s} {note['start_seconds']:6.3f}–{note['end_seconds']:6.3f}s, "
          f"string {note['string_track']}, duration {duration:.3f}s" + (' — brief label; review' if duration < 0.1 else ''))
interior_rows, mixed_rows = [], []
for row in rows:
    first, last = row['start_sample'] / rate, row['stop_sample_exclusive'] / rate
    intersecting = [note for note in labels if note['start_seconds'] < last and note['end_seconds'] > first]
    row['intersecting_label_count'] = len(intersecting)
    row['intersecting_label_names'] = ';'.join(note['name'] for note in intersecting)
    row['interior_label_hz'] = np.nan
    row['interior_error_cents'] = np.nan
    if len(intersecting) >= 2:
        mixed_rows.append(row)
    if len(intersecting) == 1:
        note = intersecting[0]
        if first >= note['start_seconds'] + 0.1 and last <= note['end_seconds'] - 0.1:
            row['interior_label_hz'] = note['hz']
            if np.isfinite(row['candidate_hz']) and row['candidate_hz'] > 0:
                row['interior_error_cents'] = 1200 * np.log2(row['candidate_hz'] / note['hz'])
            interior_rows.append(row)
print(f"\nWindows intersecting multiple MIDI spans: {len(mixed_rows)}; {sum(r['accepted'] for r in mixed_rows)} pass.")
passing_interiors = [row for row in interior_rows if row['accepted']]
print(f'Guarded interiors: {len(passing_interiors)}/{len(interior_rows)} pass; '
      f"{sum(abs(r['interior_error_cents']) > 50 for r in passing_interiors)} passing estimates differ >50 cents.")

# %% [markdown]
# ## 4. Overview and a string-change close-up
# Zoom 2.75–3.35 s: D3 on the A string moves to E3 on the D string according to
# MIDI. The highlighted window is [2.94:3.04] s, midpoint 2.99 s. It intersects
# both label spans, with a label gap between them; that alone doesn't establish
# simultaneous acoustic ringing. The waveform is NOT separated into strings.
# A single-frequency model may select an old note, a new note, a harmonic or a
# compromise. Green means it passes our rule, not that both notes were resolved.

# %%
time = np.array([r['midpoint_seconds'] for r in rows])
pitches = np.array([r['candidate_hz'] for r in rows])
accepted_hz = np.array([r['accepted_hz'] for r in rows])
fig, axes = plt.subplots(5, 1, figsize=(12, 12), constrained_layout=True)
for ax in (axes[0], axes[2]):
    ax.scatter(time, pitches, s=12, color='tab:orange', label='Raw candidates')
    ax.scatter(time, accepted_hz, s=9, color='tab:green', label='Pass unchanged rule')
    for i, note in enumerate(labels):
        ax.plot([note['start_seconds'], min(note['end_seconds'], excerpt_seconds)], [note['hz']] * 2,
                color='tab:blue', linewidth=1.5, label='MIDI reference' if i == 0 else None)
        if note['end_seconds'] - note['start_seconds'] >= 0.1:
            midpoint = (note['start_seconds'] + min(note['end_seconds'], excerpt_seconds)) / 2
            if ax is axes[0] or 2.75 <= midpoint <= 3.35:
                ax.annotate(note['name'], (midpoint, note['hz']), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
    ax.set(yscale='log', ylim=(55, 1100), ylabel='Candidate Hz')
    ax.set_yticks([60, 100, 200, 400, 800], labels=['60', '100', '200', '400', '800'])
    ax.minorticks_off()
axes[0].set(title='P2 C-scale recording: 100 ms windows, 20 ms hops, unchanged filter', xlim=(0, excerpt_seconds))
axes[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
zoom_start, zoom_stop = round(2.75 * rate), round(3.35 * rate)
axes[1].plot(np.arange(zoom_start, zoom_stop) / rate, signal[zoom_start:zoom_stop], linewidth=0.5, color='0.35')
axes[1].set(ylabel='Sample amplitude', title='One window crossing a string change — not proof of simultaneous notes')
axes[3].plot(time, [r['level_ratio'] for r in rows], color='0.35')
axes[3].axhline(minimum_level_ratio, color='tab:red', linestyle='--')
axes[3].set(yscale='log', ylabel='RMS / carried reference')
axes[4].plot(time, [r['similarity'] for r in rows], color='0.35')
axes[4].axhline(minimum_similarity, color='tab:red', linestyle='--')
axes[4].set(ylim=(0, 1.02), ylabel='Similarity', xlabel='Original-file time (s)')
for ax in axes:
    ax.grid(alpha=0.2)
for ax in axes[1:]:
    ax.set_xlim(2.75, 3.35)
    ax.axvspan(2.94, 3.04, color='tab:purple', alpha=0.12)
    for note in labels:
        if 2.75 <= note['start_seconds'] <= 3.35:
            ax.axvline(note['start_seconds'], color='tab:blue', linestyle=':', alpha=0.6)
output_folder = Path('outputs/pitch_tracking')
output_folder.mkdir(parents=True, exist_ok=True)
fig.savefig(output_folder / 'P2_scale_C_0-8s_transitions.png', dpi=150)
plt.show()
print('Midpoint  Hz     Score  Level  Decision / reference spans')
for row in rows:
    if 2.85 <= row['midpoint_seconds'] <= 3.25:
        print(f"{row['midpoint_seconds']:.2f}   {row['candidate_hz']:6.1f}  {row['similarity']:.3f}  {row['level_ratio']:5.1f}  "
              f"{'PASS' if row['accepted'] else row['rejection_reasons']} / {row['intersecting_label_names']}")
path = output_folder / 'P2_scale_C_0-8s_windows.csv'
with path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print('Saved:', path)

# %% [markdown]
# ## What this first scale check tells us (2026-11-03)
# 396 windows; 355 pass. All 119 guarded interior windows pass and agree within
# 50 cents of their labels. This is conditional interior agreement, not 100%
# transcription accuracy: transition regions are deliberately excluded.
# 42 windows intersect multiple label spans; 21 pass. Passing one pitch in such
# a window does NOT mean both notes were detected. All labels remain visible,
# including the brief D#4 annotation near 6.93 s and A4 just at the excerpt end.
#
# Read the D3→E3 transition table:
# - 2.93 s midpoint: ~147.7 Hz (D3), similarity 0.824, passes.
# - 2.95 s: ~155.8 Hz between the two notes, similarity 0.427, rejected.
# - 2.99 s: ~164.4 Hz (E3), similarity 0.869; waits for an eligible predecessor.
# - 3.01 s: ~164.4 Hz, similarity 0.909, passes.
# Every point summarizes 100 ms; these midpoints are not attack timestamps. The
# 3.01 s point covers [2.96:3.06] s, so it uses samples after the MIDI start at
# ~3.019 s. An estimate plotted before a marker doesn't mean the code predicts
# the future; window placement and annotation misalignment both matter.
#
# We haven't measured simultaneous acoustic overlap or separated two strings.
# The plot supports inspecting transitions, not claiming a solved chord detector.
# Keep settings fixed. Next group successive passing windows into rough note
# events, showing uncertain boundary gaps instead of pretending exact timing.
# Repeated same-pitch plucks will later need attack detection. Do not add special
# corrections for this scale or silently delete short MIDI labels.

# %% [markdown]
# ## 5. Group accepted windows into note-evidence runs
# Read `group_pitch_windows` in note_events.py: it takes audio measurements only,
# has no file/plot/MIDI dependencies, and returns a table. This separates the new
# grouping step from the unchanged pitch estimator and acceptance rule.
#
# Map Hz to a note coordinate: `69 + 12 * log2(hz / 440)`.
# A4=440 Hz maps to 69; doubling Hz adds 12 (one octave). Round to the nearest
# integer to name the nearest equal-tempered note. This is a naming convention,
# NOT a confidence score or a lookup in the supplied MIDI annotations. It assumes
# A4=440 Hz. The typical measured frequency is retained instead of snapped to it.
#
# Walk through windows in order:
# 1. Rejected or missing window: finish the current group; never fill the gap.
# 2. Accepted, same nearest note: append to the current group.
# 3. Accepted, different note: finish the old group and start another.
# Finish the final group at excerpt end. Retain one-window groups as points;
# no duration cutoff or merging was added just to make this scale look cleaner.
#
# Timing distinction: event bars run from FIRST to LAST accepted window MIDPOINT.
# These are evidence spans, NOT estimated attack/end times. The source samples
# behind each run extend half a window beyond those midpoints and can overlap
# another run's source samples. That is analysis overlap, not proof of two notes
# sounding together. A singleton has one midpoint, not zero physical duration.

# %%
from note_events import group_pitch_windows

# Exclude reference-label fields explicitly: grouping must work without MIDI.
audio_rows = [{key: row[key] for key in (
    'start_sample', 'stop_sample_exclusive', 'sample_rate', 'candidate_hz', 'accepted')}
    for row in rows]
events = group_pitch_windows(audio_rows, hop)
assert sum(event['window_count'] for event in events) == sum(row['accepted'] for row in rows)
print(f'{len(events)} evidence runs from {sum(row["accepted"] for row in rows)} accepted windows.')
print('ID  Note   Median Hz  First midpoint  Last midpoint  Windows')
for event in events:
    print(f"{event['event_id']:2d}  {event['note_name']:4s}   {event['median_hz']:8.2f}  "
          f"{event['first_midpoint_seconds']:8.3f} s     {event['last_midpoint_seconds']:8.3f} s   "
          f"{event['window_count']:4d}")
print('Rejected windows remain gaps. These are not confirmed attacks or separate plucks.')

# %% [markdown]
# ## 6. Inspect the groups beside the reference, keeping uncertainty visible
# Upper panel: green bars summarize accepted estimates; green dots mark the
# contributing window midpoints. Gray crosses are rejected raw estimates.
# No bar bridges a rejection. A missing bar is NOT proof of acoustic silence.
# Lower panel: the original MIDI intervals, used only for visual comparison.
# Their boundaries mean something different from our window-midpoint spans.
# Repeated same-note groups can be fragmentation rather than separate plucks;
# uninterrupted repeated picking can merge. Bends/vibrato can cross note bins.

# %%
event_fig, event_axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
rejected = np.array([not row['accepted'] for row in rows])
event_axes[0].scatter(time[rejected], pitches[rejected], marker='x', s=20,
                      color='0.65', label='Rejected raw estimates')
for index, event in enumerate(events):
    first, last = event['first_midpoint_seconds'], event['last_midpoint_seconds']
    hz = event['median_hz']
    event_axes[0].plot([first, last], [hz, hz], color='tab:green', linewidth=4,
                       label='Grouped midpoint span' if index == 0 else None)
    event_axes[0].scatter([row['midpoint_seconds'] for row in rows[event['first_frame_index']:event['last_frame_index'] + 1]],
                          [hz] * event['window_count'], color='tab:green', s=8)
    event_axes[0].annotate(event['note_name'], ((first + last) / 2, hz),
                          xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8)
for index, note in enumerate(labels):
    first, last = max(0, note['start_seconds']), min(excerpt_seconds, note['end_seconds'])
    event_axes[1].plot([first, last], [note['hz']] * 2, color='tab:blue', linewidth=3,
                       label='Original MIDI intervals' if index == 0 else None)
    if last - first >= 0.1:
        event_axes[1].annotate(note['name'], ((first + last) / 2, note['hz']),
                              xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8)
for ax in event_axes:
    ax.set(yscale='log', ylim=(minimum_frequency * 0.9, maximum_frequency * 1.1),
           ylabel='Frequency (Hz)', xlim=(0, excerpt_seconds))
    ax.set_yticks([82, 110, 165, 220, 330, 440], labels=['82', '110', '165', '220', '330', '440'])
    ax.minorticks_off()
    ax.grid(alpha=0.2)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
event_axes[0].set_title('Grouped pitch evidence — gaps preserved, no attack times inferred')
event_axes[1].set_xlabel('Original-file time (s)')
event_fig.savefig(output_folder / 'P2_scale_C_0-8s_note_evidence.png', dpi=150)
plt.show()

# Keep the old per-window CSV unchanged. Export the grouping separately.
event_path = output_folder / 'P2_scale_C_0-8s_note_evidence.csv'
event_fields = ['event_id', 'note_number', 'note_name', 'median_hz', 'window_count',
                'first_frame_index', 'last_frame_index', 'first_midpoint_seconds',
                'last_midpoint_seconds', 'source_start_sample', 'source_stop_sample_exclusive', 'sample_rate']
with event_path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=event_fields)
    writer.writeheader()
    writer.writerows(events)
print('Saved:', event_path)

# %% [markdown]
# ## Grouping result and limits
# 355 accepted windows form 16 runs, F2 → G2 → A2 → B2 → C3 → D3 → E3 → F3
# → G3 → A3 → B3 → C4 → D4 → E4 → F4 → G4, without using MIDI labels to group.
# No accepted window is discarded. For example, D3 has midpoint evidence from
# 2.51–2.93 s and E3 from 3.01–3.43 s. The gap is visible; it is not called silence.
# Their underlying sample spans overlap by 20 ms because each window is 100 ms.
#
# The original MIDI still includes a brief D#4 annotation and A4 at the excerpt
# edge; the grouping didn't force a match to the annotation count. A4's final
# raw candidate is rejected by the existing filter, so no A4 group is fabricated.
# These are provisional note candidates, not a validated tab transcription.
# Next inspect one group's member rows, then try a passage with repeated pitches
# to see what attack detection must add. Keep uncertainty visible rather than
# bridging every gap or tuning durations until these 16 bars match the labels.
