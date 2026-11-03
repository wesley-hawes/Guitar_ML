"""Group pitch evidence without inferring attacks, silence, or overlapping notes."""

import numpy as np


def group_pitch_windows(rows, hop_samples):
    """Return runs of consecutive accepted windows assigned to the same note.

    Frequency maps to the nearest equal-tempered note with A4 = 440 Hz.
    Rejected/missing windows and note changes split runs. Keep singletons.
    Only audio-derived fields are read; MIDI annotations are not inputs.
    Midpoint spans describe evidence, not physical note durations. Source
    sample spans can overlap because the input analysis windows overlap.
    """
    if not isinstance(hop_samples, (int, np.integer)) or hop_samples <= 0:
        raise ValueError('hop_samples must be a positive integer.')
    events, current = [], []
    previous_start, sample_rate = None, None
    names = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')

    def finish_run():
        if not current:
            return
        first_index, first, note_number = current[0]
        last_index, last, _ = current[-1]
        first_midpoint = (first['start_sample'] + first['stop_sample_exclusive']) / (2 * sample_rate)
        last_midpoint = (last['start_sample'] + last['stop_sample_exclusive']) / (2 * sample_rate)
        events.append(dict(
            event_id=len(events) + 1,
            note_number=note_number,
            note_name=f'{names[note_number % 12]}{note_number // 12 - 1}',
            median_hz=float(np.median([row['candidate_hz'] for _, row, _ in current])),
            window_count=len(current), first_frame_index=first_index, last_frame_index=last_index,
            first_midpoint_seconds=first_midpoint, last_midpoint_seconds=last_midpoint,
            source_start_sample=first['start_sample'],
            source_stop_sample_exclusive=last['stop_sample_exclusive'], sample_rate=sample_rate,
        ))
        current.clear()

    for index, row in enumerate(rows):
        start, stop, rate = row['start_sample'], row['stop_sample_exclusive'], row['sample_rate']
        if not np.isfinite(rate) or rate <= 0 or start < 0 or stop <= start:
            raise ValueError('Windows require valid positive-rate sample bounds.')
        if sample_rate is None:
            sample_rate = rate
        if rate != sample_rate or (previous_start is not None and start <= previous_start):
            raise ValueError('Use one sample rate and strictly increasing window starts.')
        if previous_start is not None and start - previous_start != hop_samples:
            finish_run()  # A missing window must not silently bridge the evidence.
        previous_start = start
        if not row['accepted']:
            finish_run()
            continue
        hz = row['candidate_hz']
        if not np.isfinite(hz) or hz <= 0:
            raise ValueError('An accepted window must have a finite positive frequency.')
        # 69 is A4; doubling frequency adds 12 semitones. Halfway ties go up.
        note_number = int(np.floor(69 + 12 * np.log2(hz / 440.0) + 0.5))
        if current and note_number != current[0][2]:
            finish_run()
        current.append((index, row, note_number))
    finish_run()  # Preserve the final run when the excerpt ends.
    return events
