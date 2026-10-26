from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from main import load_audio, rms_amplitude


PROJECT = Path(__file__).resolve().parent
WINDOW_SECONDS = 0.05
COMPARE_SECONDS = 8.0
# Add a filename: sample_index here only after reviewing its onset plot.
ONSET_OVERRIDES = {}


def detect_onset(signal, sample_rate):
    """Estimate the first note's onset; assumes a quiet first 0.5 seconds."""
    baseline_end = round(0.5 * sample_rate)
    frame_size = round(0.01 * sample_rate)
    if len(signal) < baseline_end + 3 * frame_size:
        raise ValueError("Need a quiet 0.5-second lead-in followed by a note.")

    noise_rms = rms_amplitude(signal[:baseline_end])
    threshold = max(6 * noise_rms, 1e-5)
    frame_levels = []
    for start in range(0, len(signal) - frame_size + 1, frame_size):
        frame_levels.append(rms_amplitude(signal[start:start + frame_size]))

    # Three consecutive loud frames reject brief isolated spikes.
    first_frame = (baseline_end + frame_size - 1) // frame_size
    for frame in range(first_frame, len(frame_levels) - 2):
        if all(level > threshold for level in frame_levels[frame:frame + 3]):
            # The preceding frame can contain the attack before its RMS rises.
            start = max(baseline_end, (frame - 1) * frame_size)
            stop = (frame + 1) * frame_size
            crossings = np.flatnonzero(np.abs(signal[start:stop]) > threshold)
            return start + int(crossings[0]), float(noise_rms), float(threshold)

    raise ValueError("No sustained onset found above the background threshold.")


def measure_windows(signal, sample_rate, onset_index,
                    duration=COMPARE_SECONDS, window_seconds=WINDOW_SECONDS):
    """Measure equal windows starting at an original-file sample index."""
    if not isinstance(onset_index, (int, np.integer)) or onset_index < 0:
        raise ValueError("Onset must be a nonnegative integer sample index.")
    if duration <= 0 or window_seconds <= 0:
        raise ValueError("Window and comparison durations must be positive.")
    length = round(duration * sample_rate)
    size = round(window_seconds * sample_rate)
    if size < 1 or length < size:
        raise ValueError("Comparison must contain at least one complete window.")
    if onset_index + length > len(signal):
        raise ValueError(f"Need {duration:g} seconds of audio after the onset.")

    measurements = []
    # Drop any partial final window; padding with zeros would bias RMS downward.
    for offset in range(0, length - size + 1, size):
        start = onset_index + offset
        stop = start + size
        section = signal[start:stop]
        measurements.append({
            "start_sample": start,
            "stop_sample_exclusive": stop,
            "seconds_after_onset": (offset + size / 2) / sample_rate,
            "rms": float(rms_amplitude(section)),
            "absolute_peak": float(np.max(np.abs(section))),
        })
    return measurements


def analyze_file(path, onset_override=None):
    samples, sample_rate = load_audio(path)
    # Use the same channel throughout; mono files already have one dimension.
    signal = samples if samples.ndim == 1 else samples[:, 0]
    if not len(signal) or not np.isfinite(signal).all():
        raise ValueError("Audio must be nonempty and contain only finite samples.")
    if onset_override is None:
        onset, noise, threshold = detect_onset(signal, sample_rate)
        method = "estimated"
    else:
        onset = onset_override
        noise = float(rms_amplitude(signal[:round(0.5 * sample_rate)]))
        threshold = max(6 * noise, 1e-5)
        method = "manual"
    windows = measure_windows(signal, sample_rate, onset)
    return {
        "filename": path.name,
        "note": path.stem.split("_note_dry_take")[0],
        "sample_rate": sample_rate,
        "signal": signal,
        "onset_sample": onset,
        "onset_seconds": onset / sample_rate,
        "onset_method": method,
        "noise_rms": noise,
        "threshold": threshold,
        "windows": windows,
    }


def plot_comparisons(recordings):
    notes = sorted({r["note"] for r in recordings})
    rms_limit = 1.05 * max(w["rms"] for r in recordings for w in r["windows"])
    waveform_limit = 1.05 * max(
        np.max(np.abs(r["signal"][r["onset_sample"] + round(.10 * r["sample_rate"]):
                                   r["onset_sample"] + round(.15 * r["sample_rate"])]))
        for r in recordings
    )
    fig, axes = plt.subplots(len(notes), 2, figsize=(12, 3 * len(notes)),
                             squeeze=False, sharex="col", sharey="col")
    # One common amplitude scale per column makes comparisons across notes fair.
    for row, note in enumerate(notes):
        for r in recordings:
            if r["note"] != note:
                continue
            times = [w["seconds_after_onset"] for w in r["windows"]]
            levels = [w["rms"] for w in r["windows"]]
            label = r["filename"].split("_dry_")[1].removesuffix(".wav")
            axes[row, 0].plot(times, levels, label=label)
            # Compare the same 50 ms of each note, 100 ms after its attack.
            sr, onset = r["sample_rate"], r["onset_sample"]
            start, stop = onset + round(.10 * sr), onset + round(.15 * sr)
            times = (np.arange(start, stop) - onset) / sr
            axes[row, 1].plot(times, r["signal"][start:stop], label=label, alpha=.8)
        axes[row, 0].set(title=f"{note}: 50 ms RMS", ylabel="RMS (digital amplitude)",
                         ylim=(0, rms_limit), xlim=(0, COMPARE_SECONDS))
        axes[row, 1].set(title=f"{note}: waveform close-up", ylabel="Signed amplitude",
                         ylim=(-waveform_limit, waveform_limit))
    for ax in axes.flat:
        ax.set_xlabel("Seconds after estimated/manual onset")
        ax.legend()
        ax.grid(alpha=.25)
    fig.suptitle("Onset-aligned comparisons — original levels, no normalization")
    fig.tight_layout()

    review, axes = plt.subplots(len(recordings), 1,
                                figsize=(11, 2.1 * len(recordings)), squeeze=False)
    for r, ax in zip(recordings, axes[:, 0]):
        onset, sr = r["onset_sample"], r["sample_rate"]
        start, stop = max(0, onset - round(.05 * sr)), onset + round(.05 * sr)
        ax.plot(np.arange(start, stop) / sr, r["signal"][start:stop], linewidth=.7)
        ax.axvline(onset / sr, color="red", linestyle="--")
        ax.axhline(r["threshold"], color="gray", linestyle=":", linewidth=.6)
        ax.axhline(-r["threshold"], color="gray", linestyle=":", linewidth=.6)
        ax.set(title=f'{r["filename"]}: {r["onset_method"]} sample {onset}',
               xlabel="Seconds in original file", ylabel="Amplitude")
    review.suptitle("Review onset estimates — red line is time zero for comparisons")
    review.tight_layout()
    return fig, review


def main():
    # The old processed Test_1 recording is intentionally outside this pattern.
    paths = sorted((PROJECT / "data/raw").glob("*_note_dry_take*.wav"))
    if not paths:
        raise SystemExit("No *_note_dry_take*.wav recordings found in data/raw.")
    recordings = []
    for path in paths:
        try:
            result = analyze_file(path, ONSET_OVERRIDES.get(path.name))
        except (ValueError, RuntimeError) as error:
            print(f"Skipped {path.name}: {error}")
            continue
        recordings.append(result)
        print(f'{path.name}: {result["onset_method"]} onset sample '
              f'{result["onset_sample"]} ({result["onset_seconds"]:.6f} s), '
              f'{len(result["windows"])} equal windows')
    if not recordings:
        raise SystemExit("No valid recordings to compare. Review the messages above.")

    output = PROJECT / "outputs/amplitude_comparison"
    output.mkdir(parents=True, exist_ok=True)
    metadata = ["filename", "note", "sample_rate", "onset_sample", "onset_seconds",
                "onset_method", "noise_rms", "threshold"]
    rows = [{**{key: r[key] for key in metadata}, **w}
            for r in recordings for w in r["windows"]]
    with (output / "measurements.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparison, review = plot_comparisons(recordings)
    comparison.savefig(output / "comparisons.png", dpi=150)
    review.savefig(output / "onset_review.png", dpi=150)
    print(f"Saved plots and sample-index measurements to {output}")
    print("Review onset markers before drawing conclusions; detection is approximate.")
    plt.show()


if __name__ == "__main__":
    main()
