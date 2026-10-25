# Guitar ML

A Python learning project for analyzing guitar recordings, starting with audio
arrays and interpretable signal measurements before introducing machine learning.

## Current milestone

`main.py` loads a WAV recording, prints its sample rate, array shape, datatype,
duration, and first five time steps, then calculates left-channel RMS amplitude
and displays a waveform. A second plot zooms into 3.00–3.05 seconds, with RMS
reported separately for that interval.

The included `data/raw/Test_1_Cmaj.wav` is a personally recorded single C note,
despite the filename suggesting a chord. It is stereo, 44,100 Hz, 16-bit PCM,
and approximately 10.39 seconds long. Recording gain and effects settings have
not yet been documented.

## Run locally

From the project root, create and activate a virtual environment (macOS):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies and run the script:

```bash
python -m pip install -r requirements.txt
python main.py
```

Close both plot windows to finish the script. If the environment already exists,
activate it without creating it again. Development started with Python 3.13.

## Measurements and limitations

- SoundFile loads the samples as floating-point values. Each row represents one
  time step; the two columns are the left and right channels.
- Duration is the number of rows divided by the sample rate.
- RMS is `sqrt(mean(signal ** 2))`. The current measurement uses the entire left
  channel, including silence, and is approximately 0.1219 for this recording.
- The selected 3.00–3.05 second interval contains 2,205 samples per channel and
  has left-channel RMS approximately 0.1074. An earlier measurement of
  1.00–1.05 seconds gave approximately 0.3946. These equal-length windows show
  the decrease in signal level as the note decays; neither represents the whole
  recording. Changing the selected interval does not change whole-recording RMS.
- RMS describes digital signal level, not perceived loudness or pitch. Silence
  and recording gain affect it, so comparisons need consistent conditions.
- The script currently uses a fixed file path and assumes stereo input. It does
  not yet detect pitch, classify techniques, or save the plot.

The WAV is included so the example can be run after cloning. GarageBand projects
and the local Python environment are excluded from Git.
