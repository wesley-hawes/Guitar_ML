import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


def load_audio(file_path):
    """Return audio samples as floats and the sample rate in Hz."""
    samples, sample_rate = sf.read(file_path)
    return samples, sample_rate


def rms_amplitude(signal):
    """Return RMS amplitude for a nonempty, one-channel float array."""
    squared_amplitudes = signal ** 2
    mean_square = np.mean(squared_amplitudes)
    return np.sqrt(mean_square)


def main():
    samples, sample_rate = load_audio("data/raw/Test_1_Cmaj.wav")

    print("Sample rate (Hz):", sample_rate)
    print("Array shape:", samples.shape)
    print("Array datatype:", samples.dtype)
    print("First five time steps:")
    print(samples[:5])

    # Channels share the same timeline, so duration depends on rows, not total values.
    duration = samples.shape[0] / sample_rate
    print("Duration (seconds):", duration)

    # Each row is one time step; convert its index into seconds.
    time = np.arange(samples.shape[0]) / sample_rate

    # This recording is stereo; plot the left channel to keep the waveform readable.
    left_channel = samples[:, 0]

    # Include the entire recording here, including the quiet lead-in and tail.
    rms = rms_amplitude(left_channel)
    print("RMS amplitude (full recording, left channel):", rms)

    start_seconds = 3.00
    end_seconds = 3.05

    # Convert seconds to sample indices; the slice excludes the end index.
    start_index = int(start_seconds * sample_rate)
    end_index = int(end_seconds * sample_rate)
    time_section = time[start_index:end_index]
    audio_section = left_channel[start_index:end_index]

    section_rms = rms_amplitude(audio_section)
    print("Selected interval (seconds):", start_seconds, "to", end_seconds)
    print("Samples in selected interval:", audio_section.size)
    print("RMS amplitude (selected interval, left channel):", section_rms)

    plt.figure(figsize=(10, 4))
    plt.plot(time, left_channel, linewidth=0.5)
    plt.axvspan(start_seconds, end_seconds, color="orange", alpha=0.4,
                label="Selected interval")
    plt.title("Single C note — left channel")
    plt.legend()
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (digital full scale)")
    plt.ylim(-1, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(time_section, audio_section, linewidth=1)
    plt.title(f"Single C note — {start_seconds:.2f}–{end_seconds:.2f} seconds")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (digital full scale)")
    # Keep the same amplitude scale so zooming doesn't make the signal look louder.
    plt.ylim(-1, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
