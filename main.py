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

plt.figure(figsize=(10, 4))
plt.plot(time, left_channel, linewidth=0.5)
plt.title("Single C note — left channel")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (digital full scale)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
