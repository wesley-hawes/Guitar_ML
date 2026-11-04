"""Propose quieter pitch transitions; no claim that each transition was picked."""
import numpy as np


def find_quieter_entries(groups, energy_rows, reference_rms, lookback_seconds=.100,
                         minimum_drop_ratio=1.5, minimum_drop_over_reference=3.):
    """Combine existing pitch groups with a sharp RMS drop using fixed settings.

    Require different adjacent note names, at most one pitch-window duration
    between evidence runs, and a drop during the 100 ms before new pitch evidence.
    Both current level and drop must clear the reference floor. Return the
    strongest qualifying drop per transition, with original sample bounds.
    This is offline corroboration, not exact onset timing or proof of a pick.
    Same-pitch quiet re-picks and legato classification remain unresolved.
    """
    if not (np.isfinite(reference_rms) and reference_rms > 0
            and np.isfinite(lookback_seconds) and lookback_seconds > 0
            and np.isfinite(minimum_drop_ratio) and minimum_drop_ratio > 1
            and np.isfinite(minimum_drop_over_reference) and minimum_drop_over_reference > 0):
        raise ValueError('Use positive reference/timing/drop settings and ratio > 1.')
    proposals = []
    for previous, current in zip(groups, groups[1:]):
        first = current['first_midpoint_seconds']
        rate = current['sample_rate']
        if rate != previous['sample_rate'] or not np.isfinite(rate) or rate <= 0:
            raise ValueError('Use one positive sample rate.')
        # Compare rounded source-sample distances, not floating time equality.
        gap = round((first - previous['last_midpoint_seconds']) * rate)
        if gap <= 0:
            raise ValueError('Pitch groups must be chronological and disjoint in midpoint time.')
        if previous['note_number'] == current['note_number'] or gap > round(lookback_seconds * rate):
            continue
        eligible = []
        for row in energy_rows:
            t = row['midpoint_seconds']
            distance = round((first - t) * rate)
            if not 0 <= distance <= round(lookback_seconds * rate):
                continue
            if row['sample_rate'] != rate:
                raise ValueError('Energy and pitch must use the same sample rate.')
            level, past = row['current_rms'], row['past_median_rms']
            if (level >= 3 * reference_rms
                    and past >= minimum_drop_ratio * max(level, reference_rms)
                    and past - level >= minimum_drop_over_reference * reference_rms):
                eligible.append(row)
        if not eligible:
            continue
        peak = max(eligible, key=lambda r: r['past_median_rms'] - r['current_rms'])
        proposals.append(dict(from_note=previous['note_name'], to_note=current['note_name'],
            pitch_evidence_seconds=first, change_midpoint_seconds=peak['midpoint_seconds'],
            start_sample=peak['start_sample'], stop_sample_exclusive=peak['stop_sample_exclusive'],
            sample_rate=peak['sample_rate'], current_rms=peak['current_rms'],
            past_median_rms=peak['past_median_rms'],
            reason='sharp_level_drop_with_changed_pitch'))
    return proposals
