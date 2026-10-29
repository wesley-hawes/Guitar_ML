"""A single-window pitch candidate calculation, independent of files and labels."""

import numpy as np


def estimate_period(window_samples, rate, minimum_frequency=60.0, maximum_frequency=1000.0):
    """Return (candidate Hz, lag in samples, similarity), not a note event.

    Constant windows or a lack of positive local peaks return three NaNs.
    Invalid inputs raise ValueError. No noise threshold, smoothing, or MIDI
    information is used. Similarity is not a confidence probability.
    """
    samples = np.asarray(window_samples, dtype=float)
    if samples.ndim != 1 or not np.all(np.isfinite(samples)):
        raise ValueError("Pass a finite, one-dimensional audio window.")
    if not np.isfinite(rate) or not 0 < minimum_frequency < maximum_frequency < rate / 2:
        raise ValueError("Use positive frequency bounds below half the sample rate.")
    minimum_lag = int(np.ceil(rate / maximum_frequency))
    maximum_lag = int(np.floor(rate / minimum_frequency))
    if minimum_lag > maximum_lag or len(samples) <= 2 * maximum_lag:
        raise ValueError("Use a longer window or a valid narrower frequency range.")

    centered = samples - np.mean(samples)
    energy = np.sum(centered ** 2)
    if energy == 0:
        return np.nan, np.nan, np.nan

    # Same overlapping-product calculation as the earlier lessons. Larger
    # lags contain fewer pairs, so this normalization somewhat favors short lags.
    similarity = np.empty(maximum_lag + 2)
    similarity[0] = 1.0
    for lag in range(1, len(similarity)):
        similarity[lag] = np.sum(centered[:-lag] * centered[lag:]) / energy
    peaks = [lag for lag in range(minimum_lag, maximum_lag + 1)
             if similarity[lag] > 0 and similarity[lag] > similarity[lag - 1]
             and similarity[lag] >= similarity[lag + 1]]
    if not peaks:
        return np.nan, np.nan, np.nan
    best_lag = max(peaks, key=lambda lag: similarity[lag])
    return rate / best_lag, best_lag, similarity[best_lag]
