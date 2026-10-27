
# %% [markdown]
# # What does “quiet” look like in one guitar recording?
#
# Work through one cell at a time. This lesson measures background level in **one** C take. It does not automatically find a note's start/end or change the comparison script.
#
# Choose **Guitar ML (.venv)** as the kernel after setup. Run from the Guitar_ML project folder.
#
# ## 1. Load the recording
# The file supplies its sample rate. At 44,100 Hz, each channel has 44,100 amplitude measurements per second. We use the left channel consistently.

# %%
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

file_path = Path("data/raw/C_note_dry_take01.wav")
if not file_path.exists():
    raise FileNotFoundError("Run this lesson from the Guitar_ML project folder.")
samples, sample_rate = sf.read(file_path)
signal = samples[:, 0] if samples.ndim == 2 else samples
sample_indices = np.arange(len(signal))
time_seconds = sample_indices / sample_rate

print("Sample rate:", sample_rate, "samples per second per channel")
print("Array shape:", samples.shape)
print("Duration:", round(len(signal) / sample_rate, 3), "seconds")

# %% [markdown]
# ## 2. Look before deciding
# The waveform shows individual signed amplitudes. Locate the quiet lead-in and the strong attack. A flat-looking line at this scale can still contain background noise.

# %%
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(time_seconds, signal, linewidth=0.5)
ax.set(xlabel="Seconds in the original file", ylabel="Signed amplitude",
       title=file_path.name)
ax.grid(alpha=0.25)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Select a candidate quiet interval
# For **this take**, 0.20–1.00 seconds is before the visible attack. These are editable selections, not a universal silence duration. Only use a region where you were not playing; an image alone cannot prove it contains no faint note.
#
# Seconds × sample rate gives a sample index. Python's slice includes the start and excludes the stop. At 44,100 Hz, this selection is `signal[8820:44100]`.

# %%
quiet_start_seconds = 0.20
quiet_end_seconds = 1.00
quiet_start = round(quiet_start_seconds * sample_rate)
quiet_stop = round(quiet_end_seconds * sample_rate)
if not 0 <= quiet_start < quiet_stop <= len(signal):
    raise ValueError("Choose a nonempty quiet interval inside this recording.")

print(f"Candidate quiet slice: signal[{quiet_start}:{quiet_stop}]")
print("Samples selected:", quiet_stop - quiet_start)
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(time_seconds, signal, linewidth=0.5)
ax.axvspan(quiet_start / sample_rate, quiet_stop / sample_rate,
           color="orange", alpha=0.35, label="Candidate quiet region — review this")
ax.set(xlabel="Seconds in original file", ylabel="Signed amplitude")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Calculate ONE window by following the arithmetic
# A 50 ms window contains 2,205 samples at 44,100 Hz. It is a useful starting resolution, not a law: shorter windows show faster changes; longer windows smooth them.
#
# RMS = square root of the mean of the squared amplitudes. Squaring prevents positive and negative values from cancelling. Here are those operations separately.

# %%
window_seconds = 0.05
window_samples = round(window_seconds * sample_rate)
if quiet_stop - quiet_start < window_samples:
    raise ValueError("Choose a quiet region at least one window long.")

one_window = signal[quiet_start:quiet_start + window_samples]
squared_samples = one_window ** 2
mean_square = np.mean(squared_samples)
one_window_rms = np.sqrt(mean_square)

print("Samples in this window:", len(one_window))
print("Mean square:", mean_square)
print("RMS:", one_window_rms)

# %% [markdown]
# ## 5. Repeat exactly that calculation across the recording
# The loop advances by `window_samples` each time. Each plotted time is the window's midpoint in the original file. A partial final window is dropped, not filled with zeros.
#
# For the background estimate we retain only windows **fully inside** your selected quiet interval. A window crossing its boundary is excluded.

# %%
window_starts = np.arange(0, len(signal) - window_samples + 1, window_samples)
window_stops = window_starts + window_samples
rms_values = []

for start, stop in zip(window_starts, window_stops):
    window = signal[start:stop]
    rms_values.append(np.sqrt(np.mean(window ** 2)))

rms_values = np.array(rms_values)
window_times = (window_starts + window_samples / 2) / sample_rate
inside_quiet = (window_starts >= quiet_start) & (window_stops <= quiet_stop)
quiet_rms = rms_values[inside_quiet]
if len(quiet_rms) < 3:
    raise ValueError("Select a longer quiet region with at least three complete windows.")

print("Quiet-window RMS values:", quiet_rms)
print("Median (middle value):", np.median(quiet_rms))
print("Highest observed quiet-window RMS:", np.max(quiet_rms))

# %% [markdown]
# ## 6. Draw a candidate threshold, then question it
# For a transparent first experiment, set the threshold to **1.5 × the largest RMS in the selected quiet region**. That means 50% above the largest background window we observed. The multiplier is an adjustable guess, not a scientifically established boundary or a confidence level.
#
# It sits above the selected background by construction. That alone does **not** validate it: it might still react to noise elsewhere or miss a soft attack. A candidate region containing a click may make this threshold too high.
#
# The top plot shows the whole recording; the bottom magnifies only the selected background windows. Their vertical scales differ intentionally. Original amplitudes are unchanged.

# %%
threshold_multiplier = 1.5
candidate_threshold = threshold_multiplier * np.max(quiet_rms)
print("Candidate RMS threshold:", candidate_threshold)

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(window_times, rms_values, label="50 ms RMS")
axes[0].axvspan(quiet_start / sample_rate, quiet_stop / sample_rate,
               color="orange", alpha=0.25, label="Candidate quiet region")
axes[0].set_title("Whole recording — no automatic start/end decision yet")
axes[1].plot(window_times[inside_quiet], quiet_rms, "o-", label="Quiet-window RMS")
axes[1].set_title("Selected background windows — expanded vertical scale")
for ax in axes:
    ax.axhline(candidate_threshold, color="red", linestyle="--", label="Candidate threshold")
    ax.set(xlabel="Seconds in original file", ylabel="RMS amplitude")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=0.25)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Stop here and interpret
# - Why isn't background RMS zero?
# - If you move the quiet interval, which values must be recalculated?
# - Does the selected interval really contain only background? The program cannot know your playing intent.
#
# Next, review another known quiet interval and the soft part of the note against this candidate threshold. Then we can discuss how long a change must persist before calling it a start or an end. No automatic note-end detector or benchmark claim is made in this lesson.

# %% [markdown]
# ## 7. Check the same threshold on A take01, 0–1.2 seconds
# You confirmed that you were not playing during this interval. This gives us a **new background region** to check, separate from the C region used to choose the threshold.
#
# Keep the C-based threshold unchanged. Recalculating it from A first would defeat this check. Each A window is still 50 ms long, converted using A's own sample rate.
#
# At 44,100 Hz, the selected interval is samples `0:52920`: 24 complete windows of 2,205 samples each. A quiet window above the threshold would be a false alarm for a simple window-by-window sound rule.

# %%
c_threshold = float(candidate_threshold)
a_path = Path("data/raw/A_note_dry_take01.wav")
a_samples, a_sample_rate = sf.read(a_path)
a_signal = a_samples[:, 0] if a_samples.ndim == 2 else a_samples

a_quiet_start_seconds = 0.0
a_quiet_end_seconds = 1.2
a_start = round(a_quiet_start_seconds * a_sample_rate)
a_stop = round(a_quiet_end_seconds * a_sample_rate)
a_window_size = round(window_seconds * a_sample_rate)
if not 0 <= a_start < a_stop <= len(a_signal):
    raise ValueError("The selected A interval must be inside the recording.")

a_window_starts = np.arange(a_start, a_stop - a_window_size + 1, a_window_size)
a_quiet_rms = []
for start in a_window_starts:
    window = a_signal[start:start + a_window_size]
    a_quiet_rms.append(np.sqrt(np.mean(window ** 2)))
a_quiet_rms = np.array(a_quiet_rms)
if len(a_quiet_rms) == 0:
    raise ValueError("The selected A interval needs at least one complete window.")

a_above_threshold = a_quiet_rms > c_threshold
print(f"A quiet slice: a_signal[{a_start}:{a_stop}]")
print("Window size:", a_window_size, "samples")
print("Number of windows:", len(a_quiet_rms))
print(f"A quiet RMS min / median / max: {a_quiet_rms.min():.8f} / "
      f"{np.median(a_quiet_rms):.8f} / {a_quiet_rms.max():.8f}")
print(f"Unchanged C threshold: {c_threshold:.8f}")
print(f"Quiet A windows above threshold: {a_above_threshold.sum()} / {len(a_quiet_rms)}")
for index in np.flatnonzero(a_above_threshold):
    start = int(a_window_starts[index])
    stop = start + a_window_size
    print(f"Above threshold: samples [{start}:{stop}], "
          f"{start / a_sample_rate:.2f}–{stop / a_sample_rate:.2f} seconds")

# %%
a_window_times = (a_window_starts + a_window_size / 2) / a_sample_rate
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(a_window_times, a_quiet_rms, "o-", label="A take01: known quiet windows")
ax.axhline(c_threshold, color="red", linestyle="--", label="Unchanged threshold from C")
ax.set(xlabel="Seconds in A take01", ylabel="RMS amplitude",
       title="Does the C-based threshold reject A's background?", xlim=(0, 1.2))
ax.set_ylim(bottom=0)
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Result for these recordings
# **1 of 24 known-quiet windows exceeded the unchanged C-based threshold.** It covers 1.05–1.10 seconds, original samples `46305:48510`, with RMS about 0.00026708 versus the threshold of 0.00024301.
#
# The other 23 windows stayed below the line. This isolated excursion is a false alarm for a rule that labels every above-threshold window as a note. Its physical cause has not been identified; we cannot call it a click or guitar note from RMS alone.
#
# These adjacent windows come from one recording, not 24 independent sessions. This result does not estimate a general false-alarm rate or prove that the rule will detect quiet notes.
#
# Next, inspect the flagged waveform interval. Then consider whether requiring a rise to persist across multiple windows would reject this excursion while still detecting the note's attack. That duration rule has not been added here. Keep the threshold fixed while studying this example rather than adjusting it to force a pass.

# %% [markdown]
# ## 8. A strong rise that lasts: one start-detection experiment
# The previous section tested a C-derived absolute threshold on A's background. Leave that result intact. Here we try a **separate, explicitly provisional rule** for isolated picked notes:
#
# 1. Estimate the selected recording's typical background using the median of its reviewed quiet windows.
# 2. Divide each window's RMS by that background level. A ratio of 1 means typical background; 10 means ten times its RMS amplitude, not ten times perceived loudness.
# 3. Require a ratio above **10 for two consecutive 50 ms windows**.
#
# Ten times background is an intentionally conservative trial for clearly picked notes; 100 ms is short compared with our sustained notes but longer than one isolated window. Neither setting is established as optimal. Keep these settings fixed when testing new D/G takes; report errors before changing them. Quiet-region selection remains manual, not a fixed lead-in assumption.
#
# The median is the middle value after sorting; it is less affected by the one high background window than the maximum. This is not a formal signal-to-noise-ratio calculation: the measured windows contain signal plus noise.
#
# **Current transfer check:** low E take02, with candidate quiet interval 0–1.5 seconds. You expect the attack around 2.5 seconds and after 2 seconds. That expectation is for reviewing the result; the detector still searches the entire recording, with no forced minimum start time. The rule remains 10× background for two 50 ms windows.
#
# The small input cell below makes the file and quiet selection explicit. The earlier C/A threshold check in section 7 remains unchanged.

# %%
# Change only this file and its reviewed quiet interval for a new recording.
test_path = Path("data/raw/Low_E_note_dry_take02.wav")
test_quiet_start_seconds = 0.0
test_quiet_end_seconds = 1.5

# Keep the measurement window and decision rule fixed for this transfer check.
window_seconds = 0.05
test_samples, test_sample_rate = sf.read(test_path)
test_signal = test_samples[:, 0] if test_samples.ndim == 2 else test_samples
test_window_size = round(window_seconds * test_sample_rate)
test_quiet_start = round(test_quiet_start_seconds * test_sample_rate)
test_quiet_stop = round(test_quiet_end_seconds * test_sample_rate)
if not 0 <= test_quiet_start < test_quiet_stop <= len(test_signal):
    raise ValueError("Choose a nonempty quiet interval inside the test recording.")
print("Testing:", test_path.name)
print(f"Candidate quiet interval: samples {test_quiet_start}:{test_quiet_stop}")

# %%
strength_ratio = 10.0
required_windows = 2
all_test_starts = np.arange(0, len(test_signal) - test_window_size + 1, test_window_size)
all_test_rms = []
for start in all_test_starts:
    window = test_signal[start:start + test_window_size]
    all_test_rms.append(np.sqrt(np.mean(window ** 2)))
all_test_rms = np.array(all_test_rms)

# Use complete windows in this file's selected quiet region for its background.
in_test_quiet = ((all_test_starts >= test_quiet_start) &
                 (all_test_starts + test_window_size <= test_quiet_stop))
test_quiet_rms = all_test_rms[in_test_quiet]
if len(test_quiet_rms) < 3:
    raise ValueError("Select a quiet interval containing at least three complete windows.")
background_rms = np.median(test_quiet_rms)
if background_rms <= 0:
    raise ValueError("The selected background is zero; choose a genuine noise-only interval.")
relative_rms = all_test_rms / background_rms
strong = relative_rms > strength_ratio

first_run = None
for index in range(len(strong) - required_windows + 1):
    if np.all(strong[index:index + required_windows]):
        first_run = index
        break

print(f"Median background RMS: {background_rms:.8f}")
print(f"Trial RMS threshold: {strength_ratio * background_rms:.8f}")
print("Required duration:", required_windows * test_window_size / test_sample_rate, "seconds")
if first_run is None:
    print("No sustained start found under this trial rule.")
else:
    start_sample = int(all_test_starts[first_run])
    confirmation_sample = start_sample + required_windows * test_window_size
    print(f"Candidate start-window boundary: sample {start_sample}, "
          f"{start_sample / test_sample_rate:.3f} seconds")
    print(f"Enough evidence available at sample {confirmation_sample}, "
          f"{confirmation_sample / test_sample_rate:.3f} seconds")
    print("Ratios in the confirming windows:", relative_rms[first_run:first_run + required_windows])
print("Highest ratio in the selected quiet interval:", np.max(test_quiet_rms / background_rms))

# %% [markdown]
# ### Start estimate versus confirmation time
# We must wait for both windows to finish before confirming the rise. We then mark the **beginning of the first qualifying window**, not the later confirmation time.
#
# That is a window boundary expressed as an original-file sample index, **not a sample-accurate attack measurement**. The attack can occur partway through a window, and a soft beginning may be missed. Smaller windows or later sample-level refinement could improve timing, but first understand this simple rule.

# %%
all_test_times = (all_test_starts + test_window_size / 2) / test_sample_rate
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(all_test_times, relative_rms, label="Window RMS / median background RMS")
ax.axhline(strength_ratio, color="red", linestyle="--", label="Trial strength threshold: 10×")
ax.axvspan(test_quiet_start / test_sample_rate, test_quiet_stop / test_sample_rate,
           color="orange", alpha=0.2, label="Selected quiet interval")
if first_run is not None:
    ax.axvline(start_sample / test_sample_rate, color="green", label="Candidate start-window boundary")
    ax.axvspan(start_sample / test_sample_rate, confirmation_sample / test_sample_rate,
               color="green", alpha=0.2, label="Two windows used for confirmation")
ax.set(xlabel="Seconds in original file", ylabel="RMS relative to background",
       title=f"{test_path.name} — unchanged strong-and-sustained rule", xlim=(0, len(test_signal) / test_sample_rate))
ax.set_ylim(bottom=0)
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Pause and explain the loop
# Why do we mark the start of the first strong window rather than the end of the second?
#
# This is a transfer check with unchanged settings. We have previously inspected this recording's waveform, so it is not an untouched validation recording. The earlier A quiet excursion was rejected by the strength condition alone; it did not establish the value of persistence. Sustained handling noise could still pass both conditions, and quiet or short real notes could fail.
#
# **This is a start rule only.** Do not use the 10× line to declare the end of a fading note: it would discard quieter ringing that is still above background. End detection needs a separately explained rule. The batch comparison script remains unchanged.
