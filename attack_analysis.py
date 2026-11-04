"""Measure renewed energy and propose attacks; no pitch or event assembly."""

import numpy as np


def measure_energy_rise(samples, sample_rate, origin_sample=0,
                        window_seconds=0.010, hop_seconds=0.002, history_seconds=0.040):
    """Compare current RMS to median RMS of preceding nonoverlapping blocks.

    Default history: four 10 ms blocks, all ending before the current window.
    Advance the calculation by 2 ms. Skip the initial history and incomplete
    final window. Ratio is NaN when baseline is zero, not an invented infinity.
    Original midpoint is a plot coordinate; the measurement needs samples
    through stop_sample_exclusive. MIDI and pitch estimates are not inputs.
    """
    signal = np.asarray(samples, dtype=float)
    if signal.ndim != 1 or not np.all(np.isfinite(signal)):
        raise ValueError('Use finite, one-dimensional audio.')
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError('Sample rate must be finite and positive.')
    if not isinstance(origin_sample, (int, np.integer)) or origin_sample < 0:
        raise ValueError('Original-file sample offset must be a nonnegative integer.')
    settings = (window_seconds, hop_seconds, history_seconds)
    if not all(np.isfinite(value) and value > 0 for value in settings):
        raise ValueError('Window, hop and history must be finite positive durations.')
    width, hop, history = [round(value * sample_rate) for value in settings]
    if width < 1 or hop < 1 or history < width or history % width:
        raise ValueError('History must contain a whole number of complete RMS windows.')

    rows = []
    for start in range(history, len(signal) - width + 1, hop):
        current = signal[start:start + width]
        past_blocks = signal[start - history:start].reshape(-1, width)
        past_rms = np.sqrt(np.mean(past_blocks ** 2, axis=1))
        baseline = float(np.median(past_rms))
        current_rms = float(np.sqrt(np.mean(current ** 2)))
        change = current_rms - baseline
        original_start = origin_sample + start
        rows.append(dict(
            sample_rate=sample_rate, start_sample=original_start,
            stop_sample_exclusive=original_start + width,
            history_start_sample=original_start - history,
            history_stop_sample_exclusive=original_start,
            midpoint_seconds=(original_start + width / 2) / sample_rate,
            rms_window_samples=width, hop_samples=hop, history_samples=history,
            current_rms=current_rms, past_median_rms=baseline,
            signed_change=change, positive_rise=max(0.0, change),
            rms_ratio=current_rms / baseline if baseline > 0 else np.nan,
        ))
    return rows


def detect_attack_candidates(rows, reference_rms, minimum_level_ratio=3.0,
                             minimum_rise_ratio=1.5, minimum_rise_over_reference=3.0,
                             minimum_separation_seconds=0.080):
    """Return (episodes, frame_decisions) from measure_energy_rise rows.

    Qualify when current RMS clears the reference floor, is >=1.5x
    max(recent RMS, reference), and increases by >=3x reference RMS. Defaults
    are explicit initial heuristics, not learned thresholds or probabilities.
    Using max(recent, reference) avoids division by zero/digital padding.

    Consecutive qualifying frames form one episode represented by its largest
    positive rise. Keep the strongest episodes at least 80 ms apart; ties
    prefer the earlier peak. All episodes remain visible with selected/reason.
    This strongest-first suppression is OFFLINE: future peaks can suppress an
    earlier one. There is no minimum sustained duration or pitch requirement.
    A transient/noise click can pass; soft picks or closely spaced picks can fail.

    A marker is the peak measurement midpoint, NOT a physical onset. Preserve
    first qualifying window and peak window bounds for later timing review.
    No expected pitches, labels, or target counts enter this function.
    """
    values = (reference_rms, minimum_level_ratio, minimum_rise_over_reference,
              minimum_separation_seconds)
    if not all(np.isfinite(v) and v > 0 for v in values):
        raise ValueError('Reference, level/rise multipliers and separation must be positive.')
    if not np.isfinite(minimum_rise_ratio) or minimum_rise_ratio <= 1:
        raise ValueError('Rise ratio must exceed one.')
    decisions, episodes, active = [], [], []
    previous_start, rate = None, None

    def finish_episode():
        if not active:
            return
        peak = max(active, key=lambda r: r['positive_rise'])
        episodes.append(dict(
            episode_id=len(episodes) + 1, sample_rate=rate,
            first_qualifying_start_sample=active[0]['start_sample'],
            last_qualifying_stop_sample_exclusive=active[-1]['stop_sample_exclusive'],
            peak_start_sample=peak['start_sample'],
            peak_stop_sample_exclusive=peak['stop_sample_exclusive'],
            peak_midpoint_seconds=(peak['start_sample'] + peak['stop_sample_exclusive']) / (2 * rate),
            qualifying_frame_count=len(active), positive_rise=peak['positive_rise'],
            current_rms=peak['current_rms'], past_median_rms=peak['past_median_rms'],
            rise_ratio=peak['rise_ratio'], selected=False, suppression_reason=''))
        active.clear()

    for row in rows:
        start, stop, row_rate = row['start_sample'], row['stop_sample_exclusive'], row['sample_rate']
        current, past = row['current_rms'], row['past_median_rms']
        if (not all(np.isfinite(v) for v in (start, stop, row_rate, current, past, row['hop_samples']))
                or row_rate <= 0 or start < 0 or stop <= start or current < 0 or past < 0
                or row['hop_samples'] <= 0):
            raise ValueError('Use finite, nonnegative RMS and valid sample bounds/rate/hop.')
        if rate is None:
            rate = row_rate
        if rate != row_rate or (previous_start is not None and start <= previous_start):
            raise ValueError('Use one sample rate and strictly increasing windows.')
        if previous_start is not None and start - previous_start != row['hop_samples']:
            finish_episode()  # Never join across an unmeasured gap.
        previous_start = start
        rise = max(0.0, current - past)
        ratio = current / max(past, reference_rms)
        level_ok = current >= minimum_level_ratio * reference_rms
        ratio_ok = ratio >= minimum_rise_ratio
        rise_ok = rise >= minimum_rise_over_reference * reference_rms
        qualifies = bool(level_ok and ratio_ok and rise_ok)
        reasons = []
        if not level_ok: reasons.append('low_level')
        if not ratio_ok: reasons.append('small_relative_rise')
        if not rise_ok: reasons.append('small_absolute_rise')
        decision = dict(start_sample=start, stop_sample_exclusive=stop, sample_rate=rate,
                        current_rms=current, past_median_rms=past, positive_rise=rise,
                        rise_ratio=ratio, qualifies=qualifies, rejection_reasons=';'.join(reasons))
        decisions.append(decision)
        if qualifies:
            active.append(decision)
        else:
            finish_episode()
    finish_episode()

    selected = []
    for episode in sorted(episodes, key=lambda e: (-e['positive_rise'], e['peak_start_sample'])):
        neighbor = next((other for other in selected
                         if abs(episode['peak_midpoint_seconds'] - other['peak_midpoint_seconds'])
                         < minimum_separation_seconds), None)
        if neighbor is None:
            episode['selected'] = True
            selected.append(episode)
        else:
            episode['suppression_reason'] = f"within_separation_of_episode_{neighbor['episode_id']}"
    return episodes, decisions
