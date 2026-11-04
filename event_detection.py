"""Minimal offline integration of pitch groups and attack/transition evidence."""
import numpy as np
from attack_analysis import detect_attack_candidates
from quieter_transitions import find_quieter_entries


def build_note_events(groups, energy_rows, reference_rms, association_seconds=.100,
                      attack_settings=None):
    """Return (events, evidence_audit); no expected notes or target counts.

    Use existing rise detector plus quieter-entry corroboration. Each pitch
    group retains one entry, including pitch-only entries. Assign each selected
    rise to the nearest group start within one pitch-window duration (100 ms).
    Otherwise, a rise inside an evidence run proposes a same-pitch repeat.
    Multiple rises attached to one group start are consolidated; strongest wins.
    Rise timing takes precedence over a fall because a fall can be pre-pick
    damping. Preserve both sources in the audit/reason. Unmatched rises do not
    invent pitched notes. A group may itself be wrong or fragmented.

    Start estimates are measurement midpoints, not physical onset truth. Group
    evidence spans can be shared by repeats; no note ends/durations are inferred.
    This offline MVP does not classify picked versus legato transitions.
    """
    if not np.isfinite(association_seconds) or association_seconds <= 0:
        raise ValueError('Association interval must be finite and positive.')
    previous_end, rate = -np.inf, None
    for g in groups:
        first, last = g['first_midpoint_seconds'], g['last_midpoint_seconds']
        if (not all(np.isfinite(v) for v in (first,last,g['sample_rate'],g['median_hz']))
                or first < 0 or first > last or first <= previous_end
                or g['sample_rate'] <= 0 or g['median_hz'] <= 0):
            raise ValueError('Use ordered, disjoint pitch evidence with valid rate/frequency.')
        if rate is not None and rate != g['sample_rate']:
            raise ValueError('Use one sample rate.')
        rate, previous_end = g['sample_rate'], last
    if rate is not None and any(r['sample_rate'] != rate for r in energy_rows):
        raise ValueError('Energy and pitch must use the same sample rate.')
    episodes, _ = detect_attack_candidates(energy_rows, reference_rms, **(attack_settings or {}))
    drops = find_quieter_entries(groups, energy_rows, reference_rms,
                                 lookback_seconds=association_seconds)
    starts = {i: [] for i in range(len(groups))}
    repeats = {i: [] for i in range(len(groups))}
    assignments = []
    for episode in episodes:
        if not episode['selected']:
            continue
        t = episode['peak_midpoint_seconds']
        nearest = min(range(len(groups)), key=lambda i: abs(groups[i]['first_midpoint_seconds']-t)) if groups else None
        if nearest is not None and round(abs(groups[nearest]['first_midpoint_seconds']-t)*rate) <= round(association_seconds*rate):
            starts[nearest].append(episode)
            role, owner = 'group_entry', nearest
        else:
            owner = next((i for i,g in enumerate(groups)
                          if g['first_midpoint_seconds'] < t < g['last_midpoint_seconds']), None)
            role = 'within_group_repeat' if owner is not None else 'unassigned_no_pitch_support'
            if owner is not None:
                repeats[owner].append(episode)
        assignments.append(dict(episode_id=episode['episode_id'], time_seconds=t,
                                assignment=role, group_index=owner))

    events = []
    for i,g in enumerate(groups):
        first = g['first_midpoint_seconds']
        drop = next((d for d in drops if d['pitch_evidence_seconds'] == first), None)
        rise = max(starts[i], key=lambda e:e['positive_rise']) if starts[i] else None
        t = rise['peak_midpoint_seconds'] if rise else drop['change_midpoint_seconds'] if drop else first
        reason = 'pitch+rise+drop' if rise and drop else 'pitch+rise' if rise else 'pitch+drop' if drop else 'pitch_only'
        entries = [(t, reason, rise, drop)]
        entries += [(e['peak_midpoint_seconds'], 'pitch+rise_repeat', e, None) for e in repeats[i]]
        for start, reason, rise, drop in entries:
            marker = rise or drop
            events.append(dict(group_index=i, note_name=g['note_name'], note_number=g['note_number'],
                median_hz=g['median_hz'], estimated_start_seconds=start,
                estimated_start_sample=round(start*rate), end_seconds=None,
                evidence_from_seconds=first, evidence_until_seconds=g['last_midpoint_seconds'],
                reason=reason, timing_status='pitch_window_only' if marker is None else 'candidate_peak',
                review_repeat=reason == 'pitch+rise_repeat',
                rise_episode_ids=','.join(str(e['episode_id']) for e in starts[i]) if reason != 'pitch+rise_repeat' else str(rise['episode_id']),
                drop_midpoint_seconds=drop['change_midpoint_seconds'] if drop else None,
                marker_start_sample=(rise['peak_start_sample'] if rise else drop['start_sample']) if marker else None,
                marker_stop_sample_exclusive=(rise['peak_stop_sample_exclusive'] if rise else drop['stop_sample_exclusive']) if marker else None,
                sample_rate=rate))
    events.sort(key=lambda e:(e['estimated_start_seconds'],e['group_index']))
    for i,event in enumerate(events,1):
        event['event_id']=i
    return events, dict(rise_episodes=episodes, quieter_entries=drops, rise_assignments=assignments)
